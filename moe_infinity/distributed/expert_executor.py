# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from moe_infinity.utils import ArcherConfig

try:
    import nvtx  # pyright: ignore[reportMissingTypeStubs]
except Exception:
    nvtx = None

try:
    from moe_infinity.profiling.io_profiler import (  # pyright: ignore[reportMissingImports]
        IOProfiler,
    )
except Exception:
    IOProfiler = None


def _nvtx_ctx(name: str):
    if nvtx is None:
        return nullcontext()

    annotate = getattr(nvtx, "annotate", None)
    if annotate is None:
        return nullcontext()

    return annotate(name, color="green")


def _profiler_instance():
    if IOProfiler is None:
        return None

    instance = getattr(IOProfiler, "instance", None)
    if instance is None:
        return None

    return instance()


_route_ahead_impl = None


def _load_route_ahead_impl():
    # Lazy import: a top-level import would be circular (spec_decode.__init__
    # -> dflash -> big_modeling -> model_offload -> this module). Both targets
    # are leaf modules, so importing them at first dispatch is safe.
    global _route_ahead_impl
    if _route_ahead_impl is None:
        import moe_infinity.spec_decode._route_ahead_ctx as route_ahead_ctx
        from moe_infinity.spec_decode._prefetch_route import (
            union_experts_from_mask,
        )

        _route_ahead_impl = (route_ahead_ctx, union_experts_from_mask)
    return _route_ahead_impl


def _call_expert_dispatcher(method, *args, **kwargs):
    global _expert_dispatcher
    func = getattr(_expert_dispatcher, method)
    return func(*args, **kwargs)


def _layer_expert_nbytes(prefetcher, layer_id, expert_ids):
    """``{expert_id: stored_bytes}`` for this layer's prefetched set, or None.

    Reads the registration-time ``ExpertPrefetcher.expert_nbytes_map`` -- a real
    ``dict`` only on the offloaded native path. Mocks, resident runs, and any
    engine without the map yield ``None`` so the A5 recorder keeps byte-accurate
    absence instead of a fabricated average expert size, and never calls
    ``int()`` on a mock attribute.
    """
    if not expert_ids:
        return None
    nbytes_map = getattr(prefetcher, "expert_nbytes_map", None)
    if not isinstance(nbytes_map, dict) or not nbytes_map:
        return None
    entry = {}
    for expert_id in expert_ids:
        nbytes = nbytes_map.get((layer_id, expert_id))
        if nbytes is not None:
            entry[expert_id] = int(nbytes)
    return entry or None


class DistributedExpertExecutor:
    def __init__(self, archer_config: ArcherConfig):
        self.archer_config = archer_config
        self.expert_dispatcher = cast(Any, None)
        self.device_map_manager = cast(Any, None)
        self.prefetcher = None
        self._speculative_prefetch_overlap = bool(
            getattr(archer_config, "speculative_prefetch_overlap", False)
        )
        self._gpu_only_expert_routing = bool(
            getattr(archer_config, "gpu_only_expert_routing", False)
        )
        self._last_dispatch_used_native_routing = False
        self._gpu_route_fallback_count = 0
        self._pending_prefetch = None

    def set_expert_dispatcher(self, expert_dispatcher):
        global _expert_dispatcher
        _expert_dispatcher = expert_dispatcher
        self.expert_dispatcher = expert_dispatcher

    def set_device_map_manager(self, device_map_manager):
        self.device_map_manager = device_map_manager

    def set_prefetcher(self, prefetcher):
        self.prefetcher = prefetcher

    def trigger_speculative_prefetch(self, layer_id, router_logits):
        if self.prefetcher is not None:
            self.prefetcher.speculative_prefetch(layer_id, router_logits)

    def _maybe_route_ahead_prefetch(
        self, layer_id, router_mask, num_expert, prefetcher=None
    ) -> bool:
        """Track A3 route-ahead seam; True iff the union prefetch fired.

        Active only inside a DFlash verify forward (``_route_ahead_ctx``).
        Pins the ACTUAL routed union of this layer via
        ``fetch_experts_lock_cache`` and enqueues it through the A2
        explicit-set ``speculative_prefetch`` -- cache warming for the reads
        this dispatch is about to issue, never a routing change. Inactive
        context, no prefetcher (resident mode / ``speculative_prefetch``
        config off), or an empty union returns False so the caller falls
        through to the legacy mean/topk path byte-identically.

        A4: when the context carries a ``RouteAheadStats`` handle, the
        predicted set and this layer's mask are reported to it (read-only
        observation; ``None`` handle = zero overhead). Offloaded
        executor-backed models (DeepSeek/Qwen/Mixtral) reach this seam with
        no resident-only gate; gpt-oss never reaches it at all
        (``model_offload.py`` wires no ``expert_executor`` into
        ``SyncGptOssMLP``).
        """
        ctx, union_experts_from_mask = _load_route_ahead_impl()
        if not ctx.is_active():
            return False
        stats = ctx.current_stats()
        route_prefetcher = prefetcher
        if route_prefetcher is None:
            route_prefetcher = ctx.current_prefetcher()
        if route_prefetcher is None:
            route_prefetcher = self.prefetcher
        mask_2d = router_mask.reshape(-1, num_expert)
        union_expert_ids = union_experts_from_mask(mask_2d)
        fired = False
        if route_prefetcher is not None and union_expert_ids:
            # A0 section 2/5 (A4 guard): pin exactly ONE layer's union per
            # dispatch -- ``ReplaceCacheCandidates`` is global and clears the
            # background queues (task_scheduler.h), so folding several layers
            # into one pin would evict candidates the next layer's dispatch
            # still needs. Never batch pins across layers; never pin the
            # empty set (short-circuited above).
            route_prefetcher.fetch_experts_lock_cache(
                layer_id, union_expert_ids
            )
            route_prefetcher.speculative_prefetch(
                layer_id,
                expert_ids=union_expert_ids,
                prefetch_layer_id=layer_id,
            )
            fired = True
        if stats is not None:
            # A5 read-only observation: predicted == the pinned union when
            # the prefetch fired, else [] (coverage 0 for this layer).
            predicted_ids = union_expert_ids if fired else []
            stats.observe_layer(
                layer_id,
                predicted_ids,
                mask_2d,
                expert_nbytes=_layer_expert_nbytes(
                    route_prefetcher, layer_id, predicted_ids
                ),
            )
        return fired

    def dispatch_local(
        self,
        layer_id,
        hidden_states,
        router_mask,
        router_weights,
        router_logits=None,
        prefetcher=None,
    ):
        profiler = _profiler_instance()
        routing_nvtx_ctx = _nvtx_ctx("moe_routing")
        routing_profiler_ctx = (
            profiler.time("routing", layer=layer_id, expert=-1)
            if profiler is not None
            else nullcontext()
        )
        with routing_nvtx_ctx:
            with routing_profiler_ctx:
                num_expert = router_mask.shape[-1]
                expert_count = (
                    torch.sum(router_mask.view((-1, num_expert)), dim=0)
                    .cpu()
                    .numpy()
                    .flatten()
                )

                expert_list = (
                    np.arange(num_expert).astype(int)[expert_count > 0].tolist()
                )
                expected_wait_cnt = len(expert_list)

        self.expert_dispatcher.set_inputs(
            hidden_states, router_mask.bool(), router_weights
        )
        self.expert_dispatcher.set_expected_queue(expected_wait_cnt)

        # Route-ahead pin + enqueue must precede every enqueue_expert below
        # (A0 section 2). Inactive context: no-op, legacy flow unchanged.
        route_ahead_handled = self._maybe_route_ahead_prefetch(
            layer_id, router_mask, num_expert, prefetcher
        )

        dispatch_nvtx_ctx = _nvtx_ctx("expert_dispatch")
        dispatch_profiler_ctx = (
            profiler.time("expert_dispatch", layer=layer_id, expert=-1)
            if profiler is not None
            else nullcontext()
        )
        with dispatch_nvtx_ctx:
            with dispatch_profiler_ctx:
                total_gpus = torch.cuda.device_count()
                for expert_id in expert_list:
                    gpu_id = expert_id % total_gpus
                    self.expert_dispatcher.enqueue_expert(
                        layer_id, expert_id, gpu_id, False
                    )
        self.expert_dispatcher.notify_fetch_start()

        if prefetcher is None:
            prefetcher = self.prefetcher

        if route_ahead_handled:
            # A0 section 3: the exact-union prefetch REPLACES the legacy
            # mean(0)/topk prediction for this dispatch, so neither the
            # overlap-triggered nor the deferred pooled call may fire.
            pending_router_logits = None
        elif (
            self._speculative_prefetch_overlap
            and prefetcher is not None
            and router_logits is not None
        ):
            self.trigger_speculative_prefetch(layer_id, router_logits)
            pending_router_logits = None
        else:
            pending_router_logits = router_logits

        self._pending_prefetch = (
            prefetcher,
            layer_id,
            expert_list,
            pending_router_logits,
        )

    def wait_dispatch_local(self):
        profiler = _profiler_instance()
        wait_nvtx_ctx = _nvtx_ctx("expert_wait_barrier")
        wait_profiler_ctx = (
            profiler.time("sync_wait", expert=-1)
            if profiler is not None
            else nullcontext()
        )
        with wait_nvtx_ctx:
            with wait_profiler_ctx:
                result = self.expert_dispatcher.wait_expert()

        pending = getattr(self, "_pending_prefetch", None)
        if pending is not None:
            prefetcher, layer_id, expert_list, router_logits = pending
            self._pending_prefetch = None
            if prefetcher is not None:
                prefetcher.correct_prefetch(layer_id + 1, expert_list)
            if router_logits is not None:
                self.trigger_speculative_prefetch(layer_id, router_logits)

        return result

    def dispatch(self, hidden_states, router_mask, layer_id):
        num_expert = router_mask.shape[-1]
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )

        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )

        device_list = self.device_map_manager.get_target_device(expert_list)
        visited_ranks = set()
        rank_wait_cnt = {r: 0 for r in range(dist.get_world_size())}
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            visited_ranks.add(rank)
            rank_wait_cnt[rank] += 1

        futures = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_inputs", hidden_states.cpu(), router_mask.cpu()),
                )
                futures.append(future)
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_expected_queue", rank_wait_cnt[rank]),
                )
                futures.append(future)
            else:
                self.expert_dispatcher.set_inputs(hidden_states, router_mask)
                self.expert_dispatcher.set_expected_queue(rank_wait_cnt[rank])

        # wait for all futures
        for future in futures:
            future.wait()

        futures = []
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            if rank == dist.get_rank():
                self.expert_dispatcher.enqueue_expert(
                    layer_id, expert_id, gpu_id, False
                )
            else:
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("enqueue_expert", layer_id, expert_id, gpu_id, True),
                )
                futures.append(future)

        # wait for all futures
        for future in futures:
            future.wait()

        result_list = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                result = rpc.rpc_sync(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("wait_expert",),
                )
                result_list += result
            else:
                result = self.expert_dispatcher.wait_expert()
                result_list += result

        return result_list


# Alias for backward compatibility
ExpertExecutor = DistributedExpertExecutor
