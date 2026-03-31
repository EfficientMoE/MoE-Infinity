from __future__ import annotations

from typing import Protocol

import torch

from moe_infinity.engine.transfer_types import TransferRequest, TransferType


class _SchedulerLike(Protocol):
    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None: ...


class KVCacheOffloadCoordinator:
    def __init__(
        self,
        kv_tensors: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        block_pool: object,
        config: object | None,
    ) -> None:
        self._kv_tensors: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (
            kv_tensors
        )
        self._block_pool: object = block_pool
        self._config: object | None = config
        self._transfer_scheduler: _SchedulerLike | None = None
        self._cpu_cache: dict[str, torch.Tensor] = {}

    def register_with_scheduler(self, scheduler: _SchedulerLike) -> None:
        self._transfer_scheduler = scheduler
        scheduler.register_handler(
            TransferType.KV_SWAP_OUT, self.handle_swap_out
        )
        scheduler.register_handler(TransferType.KV_SWAP_IN, self.handle_swap_in)

    def handle_swap_out(self, request: TransferRequest) -> None:
        _ = request
        raise NotImplementedError

    def handle_swap_in(self, request: TransferRequest) -> None:
        _ = request
        raise NotImplementedError

    def _select_block_tensors(
        self, block_ids: list[int]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self._kv_tensors, tuple):
            k_cache, v_cache = self._kv_tensors
            return k_cache[block_ids, ...], v_cache[block_ids, ...]
        return self._kv_tensors[:, block_ids, ...]


__all__ = ["KVCacheOffloadCoordinator"]
