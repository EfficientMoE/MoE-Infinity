import time

from moe_infinity.engine.scheduler import Scheduler
from moe_infinity.engine.types import Request, SamplingParams
from moe_infinity.memory.kv_cache_manager import KVCacheManager


def make_request(req_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )


def test_prefix_cache_hit_reduces_allocated_tokens() -> None:
    mgr = KVCacheManager(num_gpu_blocks=50, num_cpu_blocks=20, block_size=4)
    sched = Scheduler(kv_cache_manager=mgr)

    req1 = make_request("r1", [1, 2, 3, 4, 5, 6, 7, 8, 99, 100])
    sched.add_request(req1)
    out1 = sched.schedule()
    assert req1 in out1.scheduled_seqs
    assert out1.num_batched_tokens == 10

    sched.finish_request(req1.request_id)

    req2 = make_request("r2", [1, 2, 3, 4, 5, 6, 7, 8, 42, 43])
    sched.add_request(req2)
    out2 = sched.schedule()
    assert req2 in out2.scheduled_seqs
    assert out2.num_batched_tokens == 2
    assert len(mgr.get_block_table(req2.request_id)) == 3


def test_prefix_cache_miss_allocates_fresh() -> None:
    mgr = KVCacheManager(num_gpu_blocks=50, num_cpu_blocks=20, block_size=4)
    sched = Scheduler(kv_cache_manager=mgr)

    req = make_request("miss", [11, 12, 13, 14, 21, 22, 23, 24, 31])
    sched.add_request(req)
    out = sched.schedule()

    assert req in out.scheduled_seqs
    assert out.num_batched_tokens == 9
    assert len(mgr.get_block_table(req.request_id)) == 3


def test_register_after_completion() -> None:
    mgr = KVCacheManager(num_gpu_blocks=50, num_cpu_blocks=20, block_size=4)
    sched = Scheduler(kv_cache_manager=mgr)

    req1 = make_request("done-1", [1, 2, 3, 4, 5, 6, 7, 8, 77])
    sched.add_request(req1)
    out1 = sched.schedule()
    assert req1 in out1.scheduled_seqs

    sched.finish_request(req1.request_id)
    assert mgr.num_cached_gpu_blocks >= 2

    req2 = make_request("done-2", [1, 2, 3, 4, 5, 6, 7, 8, 66])
    sched.add_request(req2)
    out2 = sched.schedule()
    assert req2 in out2.scheduled_seqs
    assert out2.num_batched_tokens == 1
