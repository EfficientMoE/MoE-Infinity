# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from transformers import PretrainedConfig

from moe_infinity.memory.expert_policy import ExpertPhase, PhasePolicySettings
from moe_infinity.utils import parse_moe_param


def _inert_phase_policy() -> PhasePolicySettings:
    return PhasePolicySettings(
        False, "transient_on_pressure", "cache", 0, 2, 2, 1, 1.0, 4.0, 8
    )


# Native prefetch priority bands; must mirror core/prefetch/task_scheduler.h.
ON_DEMAND_PRIORITY = 0
ROUTE_AHEAD_PRIORITY = 1
BACKGROUND_PREFETCH_PRIORITY = 2

try:
    import nvtx  # type: ignore[reportMissingTypeStubs]
except ImportError:
    nvtx = None

HAS_NVTX = nvtx is not None

try:
    from moe_infinity.profiling.io_profiler import (  # pyright: ignore[reportMissingImports]
        IOProfiler,
    )
except Exception:
    IOProfiler = None


def _hit_rate_from_visit_counts(counts: Any) -> Optional[float]:
    if counts is None or not hasattr(counts, "numel"):
        return None
    try:
        if counts.numel() == 0 or counts.dim() != 2 or counts.shape[1] < 4:
            return None
        visit = float(counts[:, 0].sum().item())
        hit = float(counts[:, 3].sum().item())
    except Exception:
        return None
    if visit <= 0.0:
        return None
    return hit / visit


class ExpertPrefetcher(object):
    cache_file_rd: Optional[Any] = None
    first_k_dense_replace: int = 0
    route_ahead_priority: int = ROUTE_AHEAD_PRIORITY
    archer_engine: Any
    expert_dispatcher: Optional[Any] = None
    expert_tensor_map: dict[tuple[int, int], int]
    expert_nbytes_map: dict[tuple[int, int], int]

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.archer_engine: Optional[Any] = None
        self.expert_dispatcher: Optional[Any] = None
        self.expert_tensor_map: Dict[Tuple[int, int], int] = {}
        self.expert_nbytes_map: Dict[Tuple[int, int], int] = {}
        self.phase_policy: PhasePolicySettings = _inert_phase_policy()
        self._last_speculative_prediction: Dict[ExpertPhase, Set[int]] = {}

    def set_archer_engine(self, archer_engine: Any):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    @property
    def num_offloaded_experts(self) -> int:
        engine = self.archer_engine
        checker = (
            getattr(engine, "is_tensor_offloaded", None)
            if engine is not None
            else None
        )
        if not callable(checker):
            return len(self.expert_tensor_map)
        count = 0
        for tensor_id in self.expert_tensor_map.values():
            try:
                if checker(int(tensor_id)):
                    count += 1
            except Exception:
                continue
        return count

    def get_hit_rate(self) -> float:
        dispatcher = self.expert_dispatcher
        if dispatcher is not None:
            getter = getattr(dispatcher, "get_cache_hit_rate", None)
            if callable(getter):
                try:
                    rate = float(getter())
                except Exception:
                    rate = 0.0
                if rate:
                    return rate
        engine = self.archer_engine
        getter = (
            getattr(engine, "get_hit_rate", None)
            if engine is not None
            else None
        )
        if callable(getter):
            try:
                counts = getter()
            except Exception:
                counts = None
            rate = _hit_rate_from_visit_counts(counts)
            if rate is not None:
                return rate
        return 0.0

    def expert_occupancy_bytes(self) -> float:
        total = 0.0
        dispatcher = self.expert_dispatcher
        if dispatcher is not None:
            getter = getattr(dispatcher, "get_cache_occupancy_bytes", None)
            if callable(getter):
                try:
                    total += float(getter())
                except Exception:
                    pass
        engine = self.archer_engine
        getter = (
            getattr(engine, "get_expert_occupancy_bytes", None)
            if engine is not None
            else None
        )
        if callable(getter):
            try:
                total += float(getter())
            except Exception:
                pass
        return total

    def wasted_prefetch_bytes(self) -> float:
        engine = self.archer_engine
        getter = (
            getattr(engine, "get_wasted_prefetch_bytes", None)
            if engine is not None
            else None
        )
        if callable(getter):
            try:
                return float(getter())
            except Exception:
                return 0.0
        return 0.0

    def prefetch_experts_list(
        self,
        layer_id: int,
        expert_list: List[int],
        priority: Optional[int] = None,
        phase: ExpertPhase = ExpertPhase.MIXED,
    ):
        if self.archer_engine is None:
            return
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        if not tensor_ids:
            return
        band = self.route_ahead_priority if priority is None else priority
        batched_issue = getattr(self.archer_engine, "prefetch_tensors", None)
        if callable(batched_issue):
            batched_issue(tensor_ids, priority=band, phase=int(phase))
            return
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def fetch_experts_lock_cache(self, layer_id: int, expert_list: List[int]):
        if self.archer_engine is None:
            return
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts(self, layer_id: int, expert_matrix):
        if self.archer_engine is None:
            return
        profiler = IOProfiler.instance() if IOProfiler is not None else None
        nvtx_cm = nullcontext()
        if HAS_NVTX and nvtx is not None:
            nvtx_cm = nvtx.annotate("prefetch_trigger", color="green")
        profiler_cm = (
            profiler.time("prefetch_trigger")
            if profiler is not None
            else nullcontext()
        )

        with profiler_cm:
            with nvtx_cm:
                expert_list = []
                for i in range(layer_id, self.num_layers):
                    for j in range(self.num_experts):
                        if expert_matrix[i, j] > 0:
                            expert_list.append(
                                (
                                    self.expert_tensor_map[(i, j)],
                                    expert_matrix[i, j],
                                )
                            )
                ordered_expert_list = sorted(
                    expert_list, key=lambda x: x[1], reverse=True
                )
                tensor_ids = [x[0] for x in ordered_expert_list]
                assert len(np.unique(tensor_ids)) == len(tensor_ids)
                self.archer_engine.replace_cache_candidates(tensor_ids)
                for tensor_id in tensor_ids:
                    gpu_id = self.archer_engine.get_node_default_device(
                        [tensor_id]
                    )
                    self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def _phase_top_k(self, phase: ExpertPhase) -> int:
        if phase is ExpertPhase.PREFILL:
            return self.phase_policy.prefill_prefetch_top_k
        return self.phase_policy.decode_prefetch_top_k

    def _phase_priority(self, phase: ExpertPhase) -> int:
        if phase is ExpertPhase.PREFILL:
            return self.phase_policy.prefill_prefetch_priority
        return self.phase_policy.decode_prefetch_priority

    def _record_prediction(
        self, phase: ExpertPhase, expert_ids: Set[int]
    ) -> None:
        if isinstance(self._last_speculative_prediction, dict):
            self._last_speculative_prediction[phase] = set(expert_ids)
        else:
            self._last_speculative_prediction = set(expert_ids)

    def speculative_prefetch(
        self,
        layer_idx: int,
        router_logits: Optional[Any] = None,
        *,
        expert_ids: Optional[List[int]] = None,
        prefetch_layer_id: Optional[int] = None,
        phase: ExpertPhase = ExpertPhase.MIXED,
    ):
        if expert_ids is not None:
            if not expert_ids:
                return
            target_layer = (
                prefetch_layer_id
                if prefetch_layer_id is not None
                else layer_idx + 1
            )
            if target_layer >= self.num_layers:
                return
            self.prefetch_experts_list(
                target_layer, list(expert_ids), phase=ExpertPhase.DECODE
            )
            self._record_prediction(ExpertPhase.DECODE, set(expert_ids))
            return

        if router_logits is None:
            raise ValueError(
                "speculative_prefetch requires router_logits (legacy mode) "
                + "or expert_ids (explicit route-ahead mode); got neither."
            )

        next_layer = layer_idx + 1
        if next_layer >= self.num_layers:
            return

        if self.phase_policy.enabled:
            if phase is ExpertPhase.MIXED:
                return
            top_k = self._phase_top_k(phase)
            if top_k <= 0:
                return
            num_experts_to_prefetch = min(top_k, self.num_experts)
            priority = self._phase_priority(phase)
        else:
            num_experts_to_prefetch = min(2, self.num_experts)
            priority = BACKGROUND_PREFETCH_PRIORITY

        if hasattr(router_logits, "topk"):
            import torch

            topk_indices: List[int] = (
                torch.topk(
                    router_logits.float().view(-1, self.num_experts).mean(0),
                    num_experts_to_prefetch,
                )
                .indices.cpu()
                .tolist()
            )
        else:
            logits_np = (
                np.array(router_logits).reshape(-1, self.num_experts).mean(0)
            )
            topk_indices = np.argsort(logits_np)[-num_experts_to_prefetch:][
                ::-1
            ].tolist()

        self.prefetch_experts_list(
            next_layer, topk_indices, priority=priority, phase=phase
        )
        self._record_prediction(phase, set(topk_indices))

    def _consume_prediction(self, phase: ExpertPhase) -> Set[int]:
        if isinstance(self._last_speculative_prediction, dict):
            predicted = self._last_speculative_prediction.get(phase, set())
            self._last_speculative_prediction[phase] = set()
            return predicted
        predicted = self._last_speculative_prediction
        self._last_speculative_prediction = set()
        return predicted

    def correct_prefetch(
        self,
        layer_idx: int,
        actual_expert_ids: List[int],
        predicted_expert_ids: Optional[Set[int]] = None,
        *,
        phase: ExpertPhase = ExpertPhase.MIXED,
    ):
        if layer_idx >= self.num_layers:
            _ = self._consume_prediction(phase)
            return

        predicted = predicted_expert_ids
        if predicted is None:
            predicted = self._consume_prediction(phase)
        else:
            _ = self._consume_prediction(phase)

        missed = [e for e in actual_expert_ids if e not in predicted]
        if missed:
            self.prefetch_experts_list(layer_idx, missed, phase=phase)
