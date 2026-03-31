import time

from moe_infinity.engine.scheduler import Scheduler
from moe_infinity.engine.types import Request, SamplingParams, SequenceStatus
from moe_infinity.memory.kv_cache_manager import KVCacheManager


def make_scheduler(
    num_gpu_blocks: int = 100, block_size: int = 4
) -> tuple[Scheduler, KVCacheManager]:
    mgr = KVCacheManager(
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=50,
        block_size=block_size,
    )
    return (
        Scheduler(
            kv_cache_manager=mgr,
            max_num_seqs=16,
            max_num_batched_tokens=1024,
        ),
        mgr,
    )


def make_request(req_id: str, num_tokens: int = 8) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )


def test_schedule_cycle() -> None:
    sched, _ = make_scheduler()
    req = make_request("r1", 8)
    sched.add_request(req)

    out = sched.schedule()

    assert req in out.scheduled_seqs
    assert req.status == SequenceStatus.RUNNING
    assert out.num_batched_tokens >= 8


def test_fifo_ordering() -> None:
    sched, _ = make_scheduler()
    r1 = make_request("r1", 8)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    sched.add_request(r2)

    out = sched.schedule()

    assert out.scheduled_seqs[0] is r1
    assert out.scheduled_seqs[1] is r2


def test_preemption_oldest_running() -> None:
    sched, _ = make_scheduler(num_gpu_blocks=4, block_size=4)
    r1 = make_request("r1", 8)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    sched.add_request(r2)

    _ = sched.schedule()

    r3 = make_request("r3", 8)
    sched.add_request(r3)
    out = sched.schedule()

    assert any(req.request_id == "r1" for req in out.preempted_seqs)
    assert r1.status == SequenceStatus.SWAPPED
    assert r3 in out.scheduled_seqs


def test_finish_frees_blocks() -> None:
    sched, mgr = make_scheduler(num_gpu_blocks=20)
    req = make_request("r1", 8)
    sched.add_request(req)
    _ = sched.schedule()

    free_before = mgr.num_free_gpu_blocks
    sched.finish_request("r1")

    assert mgr.num_free_gpu_blocks > free_before


def test_abort_request() -> None:
    sched, _ = make_scheduler()
    req = make_request("r1", 8)
    sched.add_request(req)

    sched.abort_request("r1")

    assert sched.num_waiting == 0
