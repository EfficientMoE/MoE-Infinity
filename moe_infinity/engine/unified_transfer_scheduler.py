from abc import ABC, abstractmethod

from moe_infinity.engine.transfer_types import TransferRequest, TransferType


class TransferScheduler(ABC):
    @abstractmethod
    def enqueue(self, request: TransferRequest) -> str: ...

    @abstractmethod
    def cancel(self, transfer_id: str) -> bool: ...

    @abstractmethod
    def wait(self, transfer_id: str, timeout_ms: float = 5000.0) -> bool: ...

    @abstractmethod
    def get_pending_count(self) -> dict[TransferType, int]: ...

    @abstractmethod
    def set_bandwidth_budget(
        self, expert_ratio: float, kv_ratio: float
    ) -> None: ...
