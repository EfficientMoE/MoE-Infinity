import time
from typing import Optional

from typing_extensions import override

from moe_infinity.engine.scheduler import Scheduler
from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.types import Request, SamplingParams, SequenceStatus
from moe_infinity.engine.unified_transfer_scheduler import (
    TransferScheduler,
)
from moe_infinity.memory.kv_cache_manager import KVCacheManager


class RecordingTransferScheduler(TransferScheduler):
    def __init__(self, fail_types: Optional[set[TransferType]] = None):
        self.requests: list[TransferRequest] = []
        self.cancelled: list[str] = []
        self._by_id: dict[str, TransferRequest] = {}
        self._fail_types: set[TransferType] = fail_types or set()

    @override
    def enqueue(self, request: TransferRequest) -> str:
        self.requests.append(request)
        self._by_id[request.transfer_id] = request
        return request.transfer_id

    @override
    def wait(self, transfer_id: str, timeout_ms: float = 5000.0) -> bool:
        _ = timeout_ms
        req = self._by_id[transfer_id]
        return req.transfer_type not in self._fail_types

    @override
    def cancel(self, transfer_id: str) -> bool:
        self.cancelled.append(transfer_id)
        return True

    def shutdown(self, wait: bool = True) -> None:
        _ = wait

    @override
    def get_pending_count(self) -> dict[TransferType, int]:
        return {}

    @override
    def set_bandwidth_budget(
        self, expert_ratio: float, kv_ratio: float
    ) -> None:
        _ = expert_ratio
        _ = kv_ratio


def make_request(req_id: str, num_tokens: int = 8) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )


def test_preempt_swap_resume_no_transfer_scheduler() -> None:
    mgr = KVCacheManager(num_gpu_blocks=2, num_cpu_blocks=10, block_size=4)
    sched = Scheduler(kv_cache_manager=mgr)

    r1 = make_request("r1", 8)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    _ = sched.schedule()
    assert r1.status == SequenceStatus.RUNNING

    sched.add_request(r2)
    out = sched.schedule()
    assert any(req.request_id == "r1" for req in out.preempted_seqs)
    assert r1.status == SequenceStatus.SWAPPED
    assert r2.status == SequenceStatus.RUNNING

    sched.finish_request("r2")
    out = sched.schedule()
    assert any(req.request_id == "r1" for req in out.swapped_in_seqs)
    assert r1.status == SequenceStatus.RUNNING
    assert mgr.num_free_cpu_blocks == 10


def test_preempt_with_transfer_scheduler_priorities() -> None:
    mgr = KVCacheManager(num_gpu_blocks=2, num_cpu_blocks=10, block_size=4)
    transfer_sched = RecordingTransferScheduler()
    sched = Scheduler(kv_cache_manager=mgr, transfer_scheduler=transfer_sched)

    r1 = make_request("r1", 8)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    _ = sched.schedule()
    sched.add_request(r2)
    _ = sched.schedule()

    sched.finish_request("r2")
    _ = sched.schedule()

    assert any(
        req.transfer_type == TransferType.KV_SWAP_OUT
        and req.priority == TransferPriority.HIGH
        for req in transfer_sched.requests
    )
    assert any(
        req.transfer_type == TransferType.KV_SWAP_IN
        and req.priority == TransferPriority.NORMAL
        for req in transfer_sched.requests
    )
    transfer_sched.shutdown()


def test_interrupted_swap_out_cancels_and_falls_back_to_reprefill() -> None:
    mgr = KVCacheManager(num_gpu_blocks=2, num_cpu_blocks=10, block_size=4)
    transfer_sched = RecordingTransferScheduler(
        fail_types={TransferType.KV_SWAP_OUT}
    )
    sched = Scheduler(kv_cache_manager=mgr, transfer_scheduler=transfer_sched)

    r1 = make_request("r1", 8)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    _ = sched.schedule()

    sched.add_request(r2)
    _ = sched.schedule()
    assert "swap_out_r1" in transfer_sched.cancelled

    sched.finish_request("r2")
    out = sched.schedule()
    assert any(req.request_id == "r1" for req in out.swapped_in_seqs)
    assert r1.status == SequenceStatus.WAITING
    assert all(
        req.transfer_type != TransferType.KV_SWAP_IN
        for req in transfer_sched.requests
    )

    _ = sched.schedule()
    assert r1.status == SequenceStatus.RUNNING
    transfer_sched.shutdown()


def test_no_crash_on_multiple_schedule_cycles() -> None:
    mgr = KVCacheManager(num_gpu_blocks=10, num_cpu_blocks=10, block_size=4)
    sched = Scheduler(kv_cache_manager=mgr)

    for i in range(5):
        sched.add_request(make_request(f"r{i}", 4))

    for _ in range(10):
        out = sched.schedule()
        for req in list(out.scheduled_seqs):
            sched.finish_request(req.request_id)

    assert mgr.num_free_gpu_blocks == 10
