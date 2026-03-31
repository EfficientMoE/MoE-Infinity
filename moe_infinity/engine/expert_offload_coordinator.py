from __future__ import annotations

from typing import Callable, Optional, Protocol, cast

from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.unified_transfer_scheduler import (
    UnifiedTransferScheduler,
)


class ExpertPrefetcherLike(Protocol):
    def prefetch_experts_list(
        self, layer_id: int, expert_list: list[int]
    ) -> None: ...

    def fetch_experts_lock_cache(
        self, layer_id: int, expert_list: list[int]
    ) -> None: ...


_ImportedExpertPrefetcher: Callable[[object], ExpertPrefetcherLike] | None

try:
    from moe_infinity.memory.expert_prefetcher import (
        ExpertPrefetcher as _ExpertPrefetcherImpl,
    )
except Exception:
    _ImportedExpertPrefetcher = None
else:
    _ImportedExpertPrefetcher = cast(
        Callable[[object], ExpertPrefetcherLike],
        _ExpertPrefetcherImpl,
    )


class _StubExpertPrefetcher:
    def __init__(self, config: object | None = None) -> None:
        _ = config

    def prefetch_experts_list(
        self, layer_id: int, expert_list: list[int]
    ) -> None:
        _ = (layer_id, expert_list)

    def fetch_experts_lock_cache(
        self, layer_id: int, expert_list: list[int]
    ) -> None:
        _ = (layer_id, expert_list)


class ExpertOffloadCoordinator:
    def __init__(
        self,
        config: object | None = None,
        expert_prefetcher: ExpertPrefetcherLike | None = None,
    ) -> None:
        self._transfer_scheduler: Optional[UnifiedTransferScheduler] = None
        self._expert_prefetcher: ExpertPrefetcherLike = self._build_prefetcher(
            config=config,
            expert_prefetcher=expert_prefetcher,
        )

    def _build_prefetcher(
        self,
        config: object | None,
        expert_prefetcher: ExpertPrefetcherLike | None,
    ) -> ExpertPrefetcherLike:
        if expert_prefetcher is not None:
            return expert_prefetcher
        if _ImportedExpertPrefetcher is not None and config is not None:
            try:
                return _ImportedExpertPrefetcher(config)
            except Exception:
                return _StubExpertPrefetcher(config)
        return _StubExpertPrefetcher(config)

    def register_with_scheduler(
        self, transfer_scheduler: UnifiedTransferScheduler
    ) -> None:
        self._transfer_scheduler = transfer_scheduler
        transfer_scheduler.register_handler(
            TransferType.EXPERT_FETCH,
            self._handle_expert_fetch,
        )
        transfer_scheduler.register_handler(
            TransferType.EXPERT_EVICT,
            self._handle_expert_evict,
        )

    def prefetch_experts(
        self,
        layer_id: int,
        expert_ids: list[int],
        priority: TransferPriority = TransferPriority.HIGH,
    ) -> str:
        if self._transfer_scheduler is None:
            raise RuntimeError("transfer scheduler must be registered first")

        request = TransferRequest(
            transfer_id="",
            transfer_type=TransferType.EXPERT_FETCH,
            priority=priority,
            source_device="cpu",
            target_device="cuda:0",
            tensor_id=str(layer_id),
            block_ids=list(expert_ids),
        )
        return self._transfer_scheduler.enqueue(request)

    def _handle_expert_fetch(self, request: TransferRequest) -> None:
        layer_id = self._decode_layer_id(request.tensor_id)
        expert_ids = list(request.block_ids)
        self._expert_prefetcher.prefetch_experts_list(layer_id, expert_ids)

    def _handle_expert_evict(self, request: TransferRequest) -> None:
        layer_id = self._decode_layer_id(request.tensor_id)
        expert_ids = list(request.block_ids)
        self._expert_prefetcher.fetch_experts_lock_cache(layer_id, expert_ids)

    @staticmethod
    def _decode_layer_id(layer_tensor_id: Optional[str]) -> int:
        if layer_tensor_id is None:
            return 0
        return int(layer_tensor_id)


__all__ = ["ExpertOffloadCoordinator"]
