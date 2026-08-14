# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from transformers import PretrainedConfig

from moe_infinity.utils import parse_moe_param

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


class ExpertPrefetcher(object):
    cache_file_rd: Optional[Any] = None
    first_k_dense_replace: int = 0
    archer_engine: Any
    expert_tensor_map: dict[tuple[int, int], int]

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.archer_engine: Optional[Any] = None
        self.expert_tensor_map: Dict[Tuple[int, int], int] = {}
        self._last_speculative_prediction: Set[int] = set()

    def set_archer_engine(self, archer_engine: Any):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def prefetch_experts_list(self, layer_id: int, expert_list: List[int]):
        if self.archer_engine is None:
            return
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
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
                # print("expert_tensor_map", self.expert_tensor_map)
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

    def speculative_prefetch(
        self,
        layer_idx: int,
        router_logits: Optional[Any] = None,
        *,
        expert_ids: Optional[List[int]] = None,
        prefetch_layer_id: Optional[int] = None,
    ):
        """Speculatively prefetch experts for an upcoming layer. Two modes:

        * Legacy (default): pool ``router_logits`` via ``mean(0)`` and
          prefetch the ``min(2, num_experts)`` top experts for
          ``layer_idx + 1``. Unchanged pre-A2 behavior (non-spec decode).
        * Explicit (DFlash route-ahead seam, Track A2): when ``expert_ids``
          is given, prefetch exactly that set for ``prefetch_layer_id``
          (default ``layer_idx + 1``) via ``prefetch_experts_list`` -- no
          mean/topk pooling. Empty ``expert_ids`` is a safe no-op.

        Returns ``None``. Raises ``ValueError`` if both are ``None``.
        """
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
            self.prefetch_experts_list(target_layer, list(expert_ids))
            self._last_speculative_prediction = set(expert_ids)
            return

        if router_logits is None:
            raise ValueError(
                "speculative_prefetch requires router_logits (legacy mode) "
                + "or expert_ids (explicit route-ahead mode); got neither."
            )

        next_layer = layer_idx + 1
        if next_layer >= self.num_layers:
            return

        num_experts_to_prefetch = min(2, self.num_experts)
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

        self.prefetch_experts_list(next_layer, topk_indices)
        self._last_speculative_prediction = set(topk_indices)

    def correct_prefetch(
        self,
        layer_idx: int,
        actual_expert_ids: List[int],
        predicted_expert_ids: Optional[Set[int]] = None,
    ):
        if layer_idx >= self.num_layers:
            self._last_speculative_prediction = set()
            return

        predicted = predicted_expert_ids
        if predicted is None:
            predicted = getattr(self, "_last_speculative_prediction", set())

        missed = [e for e in actual_expert_ids if e not in predicted]
        if missed:
            self.prefetch_experts_list(layer_idx, missed)

        self._last_speculative_prediction = set()
