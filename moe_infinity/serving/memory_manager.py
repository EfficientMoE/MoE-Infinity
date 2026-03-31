from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class MemoryBudget:
    total_gpu_memory_bytes: int
    model_memory_bytes: int
    expert_cache_ratio: float
    kv_cache_ratio: float
    activation_reserve_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.total_gpu_memory_bytes < 0:
            raise ValueError(
                f"total_gpu_memory_bytes must be >= 0, got {self.total_gpu_memory_bytes}"
            )
        if self.model_memory_bytes < 0:
            raise ValueError(
                f"model_memory_bytes must be >= 0, got {self.model_memory_bytes}"
            )
        for name, value in (
            ("expert_cache_ratio", self.expert_cache_ratio),
            ("kv_cache_ratio", self.kv_cache_ratio),
            ("activation_reserve_ratio", self.activation_reserve_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def available_bytes(self) -> int:
        activation_reserve_bytes = int(
            self.total_gpu_memory_bytes * self.activation_reserve_ratio
        )
        available = (
            self.total_gpu_memory_bytes
            - self.model_memory_bytes
            - activation_reserve_bytes
        )
        return max(0, available)

    @property
    def expert_cache_bytes(self) -> int:
        requested = int(self.available_bytes * self.expert_cache_ratio)
        return min(self.available_bytes, max(0, requested))

    @property
    def kv_cache_bytes(self) -> int:
        requested = int(self.available_bytes * self.kv_cache_ratio)
        remaining_after_expert = max(
            0, self.available_bytes - self.expert_cache_bytes
        )
        return min(remaining_after_expert, max(0, requested))


class MemoryManager:
    device: torch.device
    device_memory_ratio: float
    kv_cache_ratio: float
    activation_reserve_ratio: float
    total_gpu_memory_bytes: int
    _last_budget: Optional[MemoryBudget]

    def __init__(
        self,
        device: Optional[torch.device] = None,
        device_memory_ratio: float = 0.75,
        kv_cache_ratio: float = 0.25,
        activation_reserve_ratio: float = 0.10,
    ) -> None:
        if not 0.0 <= device_memory_ratio <= 1.0:
            raise ValueError(
                f"device_memory_ratio must be in [0, 1], got {device_memory_ratio}"
            )
        if not 0.0 <= kv_cache_ratio <= 1.0:
            raise ValueError(
                f"kv_cache_ratio must be in [0, 1], got {kv_cache_ratio}"
            )
        if not 0.0 <= activation_reserve_ratio <= 1.0:
            raise ValueError(
                f"activation_reserve_ratio must be in [0, 1], got {activation_reserve_ratio}"
            )

        self.device = self._resolve_device(device)
        self.device_memory_ratio = device_memory_ratio
        self.kv_cache_ratio = kv_cache_ratio
        self.activation_reserve_ratio = activation_reserve_ratio
        self.total_gpu_memory_bytes = self._get_total_gpu_memory_bytes(
            self.device
        )
        self._last_budget = None

    def compute_budget(self, model_memory_bytes: int) -> MemoryBudget:
        if model_memory_bytes < 0:
            raise ValueError(
                f"model_memory_bytes must be >= 0, got {model_memory_bytes}"
            )

        budget = MemoryBudget(
            total_gpu_memory_bytes=self.total_gpu_memory_bytes,
            model_memory_bytes=model_memory_bytes,
            expert_cache_ratio=self.get_expert_cache_ratio(),
            kv_cache_ratio=self.device_memory_ratio * self.kv_cache_ratio,
            activation_reserve_ratio=self.activation_reserve_ratio,
        )
        self._last_budget = budget
        return budget

    def get_max_kv_blocks(
        self,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        for name, value in (
            ("block_size", block_size),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("head_dim", head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        kv_bytes_per_block = (
            2
            * block_size
            * num_layers
            * num_heads
            * head_dim
            * bytes_per_element
        )

        budget = self._last_budget
        if budget is None:
            budget = self.compute_budget(model_memory_bytes=0)

        if kv_bytes_per_block <= 0:
            return 0
        return max(0, budget.kv_cache_bytes // kv_bytes_per_block)

    def get_expert_cache_ratio(self) -> float:
        return self.device_memory_ratio - (
            self.device_memory_ratio * self.kv_cache_ratio
        )

    def report(self) -> dict[str, Union[str, int, float]]:
        budget = self._last_budget
        if budget is None:
            budget = self.compute_budget(model_memory_bytes=0)

        return {
            "device": str(self.device),
            "total_gpu_memory_bytes": budget.total_gpu_memory_bytes,
            "model_memory_bytes": budget.model_memory_bytes,
            "available_bytes": budget.available_bytes,
            "expert_cache_ratio": budget.expert_cache_ratio,
            "kv_cache_ratio": budget.kv_cache_ratio,
            "activation_reserve_ratio": budget.activation_reserve_ratio,
            "expert_cache_bytes": budget.expert_cache_bytes,
            "kv_cache_bytes": budget.kv_cache_bytes,
        }

    @staticmethod
    def _resolve_device(device: Optional[torch.device]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    @staticmethod
    def _get_total_gpu_memory_bytes(device: torch.device) -> int:
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0

        try:
            _, total = torch.cuda.mem_get_info(device)
            return int(total)
        except Exception:
            return 0


__all__ = ["MemoryBudget", "MemoryManager"]
