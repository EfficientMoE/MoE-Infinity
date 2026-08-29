import threading
import time
from typing import Optional

import pytest

from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.unified_transfer_scheduler import (
    UnifiedTransferScheduler,
)


def _request(
    transfer_id: str,
    transfer_type: TransferType,
    priority: TransferPriority,
    block_ids: Optional[list[int]] = None,
) -> TransferRequest:
    return TransferRequest(
        transfer_id=transfer_id,
        transfer_type=transfer_type,
        priority=priority,
        source_device="cpu",
        target_device="cuda:0",
        device_id=0,
        block_ids=[] if block_ids is None else block_ids,
    )


def test_priority_ordering() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    dispatch_order: list[str] = []
    lock = threading.Lock()

    def handler(req: TransferRequest) -> None:
        with lock:
            dispatch_order.append(req.transfer_id)

    scheduler.register_handler(TransferType.KV_SWAP_OUT, handler)

    low_id = scheduler.enqueue(
        _request("low", TransferType.KV_SWAP_OUT, TransferPriority.LOW)
    )
    urgent_id = scheduler.enqueue(
        _request("urgent", TransferType.KV_SWAP_OUT, TransferPriority.URGENT)
    )

    try:
        assert scheduler.wait(urgent_id, timeout_ms=2000)
        assert scheduler.wait(low_id, timeout_ms=2000)
        assert dispatch_order[:2] == ["urgent", "low"]
    finally:
        scheduler.shutdown()


def test_expert_kv_interleave() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    progress = {"expert": 0, "kv": 0}
    lock = threading.Lock()

    def expert_handler(_req: TransferRequest) -> None:
        with lock:
            progress["expert"] += 1

    def kv_handler(_req: TransferRequest) -> None:
        with lock:
            progress["kv"] += 1

    scheduler.register_handler(TransferType.EXPERT_FETCH, expert_handler)
    scheduler.register_handler(TransferType.KV_SWAP_IN, kv_handler)

    tids = [
        scheduler.enqueue(
            _request(
                "expert-1", TransferType.EXPERT_FETCH, TransferPriority.NORMAL
            )
        ),
        scheduler.enqueue(
            _request("kv-1", TransferType.KV_SWAP_IN, TransferPriority.NORMAL)
        ),
        scheduler.enqueue(
            _request(
                "expert-2", TransferType.EXPERT_FETCH, TransferPriority.NORMAL
            )
        ),
        scheduler.enqueue(
            _request("kv-2", TransferType.KV_SWAP_IN, TransferPriority.NORMAL)
        ),
    ]

    try:
        for tid in tids:
            assert scheduler.wait(tid, timeout_ms=2000)
        assert progress["expert"] > 0
        assert progress["kv"] > 0
    finally:
        scheduler.shutdown()


def test_cancel_pending() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    release = threading.Event()

    def handler(req: TransferRequest) -> None:
        if req.transfer_id == "first":
            _ = release.wait(timeout=2.0)

    scheduler.register_handler(TransferType.KV_SWAP_OUT, handler)

    first_id = scheduler.enqueue(
        _request("first", TransferType.KV_SWAP_OUT, TransferPriority.NORMAL)
    )
    second_id = scheduler.enqueue(
        _request("second", TransferType.KV_SWAP_OUT, TransferPriority.NORMAL)
    )

    try:
        time.sleep(0.05)
        assert scheduler.cancel(second_id)
        release.set()
        assert scheduler.wait(first_id, timeout_ms=2000)
        assert scheduler.wait(second_id, timeout_ms=2000)
        result = scheduler.get_result(second_id)
        assert result is not None
        assert result.status == "CANCELLED"
    finally:
        scheduler.shutdown()


def test_wait_timeout() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)

    def slow_handler(_req: TransferRequest) -> None:
        time.sleep(0.25)

    scheduler.register_handler(TransferType.EXPERT_FETCH, slow_handler)
    transfer_id = scheduler.enqueue(
        _request("slow", TransferType.EXPERT_FETCH, TransferPriority.NORMAL)
    )

    try:
        assert not scheduler.wait(transfer_id, timeout_ms=10)
        assert scheduler.wait(transfer_id, timeout_ms=2000)
    finally:
        scheduler.shutdown()


def test_metrics_tracked() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)

    def noop(_req: TransferRequest) -> None:
        return

    scheduler.register_handler(TransferType.EXPERT_FETCH, noop)
    scheduler.register_handler(TransferType.KV_SWAP_OUT, noop)

    expert_id = scheduler.enqueue(
        _request("exp", TransferType.EXPERT_FETCH, TransferPriority.NORMAL)
    )
    kv_id = scheduler.enqueue(
        _request(
            "kv",
            TransferType.KV_SWAP_OUT,
            TransferPriority.NORMAL,
            block_ids=[1, 2, 3],
        )
    )

    try:
        assert scheduler.wait(expert_id, timeout_ms=2000)
        assert scheduler.wait(kv_id, timeout_ms=2000)
        metrics = scheduler.get_metrics()
        assert metrics["EXPERT_FETCH"]["count"] == 1
        assert metrics["KV_SWAP_OUT"]["count"] == 1
        assert metrics["KV_SWAP_OUT"]["bytes"] == 3
    finally:
        scheduler.shutdown()


def test_wait_for_device_ignores_unrelated_gpu() -> None:
    gate0, gate1 = threading.Event(), threading.Event()
    scheduler = UnifiedTransferScheduler(max_workers=2)
    scheduler.register_handler(
        TransferType.KV_SWAP_OUT,
        lambda request: {0: gate0, 1: gate1}[request.device_id].wait(),
    )

    def swap(device_id: int) -> TransferRequest:
        return TransferRequest(
            transfer_id=f"d{device_id}",
            transfer_type=TransferType.KV_SWAP_OUT,
            priority=TransferPriority.HIGH,
            source_device=f"cuda:{device_id}",
            target_device="cpu",
            device_id=device_id,
        )

    id0 = scheduler.enqueue(swap(0))
    id1 = scheduler.enqueue(swap(1))
    try:
        gate0.set()
        assert scheduler.wait(id0, timeout_ms=1000)
        assert scheduler.wait_for_device(0, timeout_ms=10)
        assert not scheduler.wait_for_device(1, timeout_ms=1)
        gate1.set()
        assert scheduler.wait(id1, timeout_ms=1000)
        assert scheduler.wait_for_device(1, timeout_ms=10)
    finally:
        gate0.set()
        gate1.set()
        scheduler.shutdown()


@pytest.mark.parametrize("bad", [-1, 2])
def test_transfer_scheduler_rejects_endpoint_for_other_device(bad: int) -> None:
    scheduler = UnifiedTransferScheduler()
    try:
        with pytest.raises(ValueError, match="device_id"):
            scheduler.enqueue(
                TransferRequest(
                    transfer_id="wrong",
                    transfer_type=TransferType.KV_SWAP_OUT,
                    priority=TransferPriority.HIGH,
                    source_device=f"cuda:{bad}",
                    target_device="cpu",
                    device_id=1,
                )
            )
    finally:
        scheduler.shutdown()


def test_enqueue_normalization_preserves_device_owner() -> None:
    scheduler = UnifiedTransferScheduler()
    seen: list[int] = []
    scheduler.register_handler(
        TransferType.EXPERT_FETCH,
        lambda request: seen.append(request.device_id),
    )
    transfer_id = scheduler.enqueue(
        TransferRequest(
            transfer_id="owned",
            transfer_type=TransferType.EXPERT_FETCH,
            priority=TransferPriority.NORMAL,
            source_device="cpu",
            target_device="cuda:3",
            device_id=3,
        )
    )
    try:
        assert scheduler.wait(transfer_id, timeout_ms=1000)
        assert seen == [3]
    finally:
        scheduler.shutdown()
