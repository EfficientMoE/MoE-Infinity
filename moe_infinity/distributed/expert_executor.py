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


def _call_expert_dispatcher(method, *args, **kwargs):
    global _expert_dispatcher
    func = getattr(_expert_dispatcher, method)
    return func(*args, **kwargs)


class DistributedExpertExecutor:
    def __init__(self, archer_config: ArcherConfig):
        self.archer_config = archer_config
        self.expert_dispatcher = cast(Any, None)
        self.device_map_manager = cast(Any, None)
        self.prefetcher = None

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

        self._pending_prefetch = (
            prefetcher,
            layer_id,
            expert_list,
            router_logits,
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
