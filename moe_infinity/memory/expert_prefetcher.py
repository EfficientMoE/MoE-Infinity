# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from transformers import PretrainedConfig

from moe_infinity.utils import parse_moe_param


class ExpertPrefetcher(object):
    cache_file_rd: Optional[Any] = None
    first_k_dense_replace: int = 0

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
        expert_list = []
        # print("expert_tensor_map", self.expert_tensor_map)
        for i in range(layer_id, self.num_layers):
            for j in range(self.num_experts):
                if expert_matrix[i, j] > 0:
                    expert_list.append(
                        (self.expert_tensor_map[(i, j)], expert_matrix[i, j])
                    )
        ordered_expert_list = sorted(
            expert_list, key=lambda x: x[1], reverse=True
        )
        tensor_ids = [x[0] for x in ordered_expert_list]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def speculative_prefetch(
        self, layer_idx: int, router_logits: Optional[Any] = None
    ):
        next_layer = layer_idx + 1
        if next_layer >= self.num_layers:
            return
        if router_logits is None:
            return

        num_experts_to_prefetch = min(8, self.num_experts)
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
        self._last_correction_misses = missed

        self._last_speculative_prediction = set()
