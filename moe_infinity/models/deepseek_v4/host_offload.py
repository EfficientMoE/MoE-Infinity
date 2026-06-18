# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import torch


class HostOffloadBundleProvider:
    def __init__(
        self,
        indexer,
        device: torch.device,
        max_resident_experts: int = 8,
        pin_memory: bool = True,
    ):
        self.indexer = indexer
        self.device = torch.device(device)
        self.max_resident_experts = max_resident_experts
        self.pin_memory = pin_memory and self.device.type == "cuda"

        self._host: Dict[Tuple[int, int], Sequence[torch.Tensor]] = {}
        self._gpu: "OrderedDict[Tuple[int, int], Sequence[torch.Tensor]]" = (
            OrderedDict()
        )
        if self.device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=self.device)
        else:
            self._copy_stream = None

    def preload_layer(self, layer_id: int) -> None:
        for expert_id in range(self.indexer.n_routed_experts):
            self._ensure_host(layer_id, expert_id)

    def _ensure_host(
        self, layer_id: int, expert_id: int
    ) -> Sequence[torch.Tensor]:
        key = (layer_id, expert_id)
        cached = self._host.get(key)
        if cached is not None:
            return cached
        bundle = self.indexer.bundle(layer_id, expert_id)
        tensors = self.indexer.load_bundle_tensors(bundle)
        if self.pin_memory:
            tensors = [t.pin_memory() for t in tensors]
        self._host[key] = tensors
        return tensors

    def resident_experts(self) -> List[Tuple[int, int]]:
        return list(self._gpu.keys())

    def is_resident(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self._gpu

    def _evict_if_needed(self) -> None:
        while len(self._gpu) > self.max_resident_experts:
            self._gpu.popitem(last=False)

    def __call__(self, layer_id: int, expert_id: int) -> Sequence[torch.Tensor]:
        key = (layer_id, expert_id)
        if key in self._gpu:
            self._gpu.move_to_end(key)
            return self._gpu[key]

        host_tensors = self._ensure_host(layer_id, expert_id)

        if self.device.type != "cuda":
            self._gpu[key] = host_tensors
            self._evict_if_needed()
            return host_tensors

        with torch.cuda.stream(self._copy_stream):
            gpu_tensors = [
                t.to(self.device, non_blocking=True) for t in host_tensors
            ]
        torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)

        self._gpu[key] = gpu_tensors
        self._evict_if_needed()
        return gpu_tensors
