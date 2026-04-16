from __future__ import annotations

import time

from moe_infinity.engine.scheduler import Scheduler
from moe_infinity.engine.types import Request, SamplingParams
from moe_infinity.memory.kv_cache_manager import KVCacheManager


def _make_scheduler(
    *,
    num_gpu_blocks: int = 64,
    block_size: int = 4,
    max_num_seqs: int = 1,
) -> Scheduler:
    mgr = KVCacheManager(
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=64,
        block_size=block_size,
    )
    return Scheduler(
        kv_cache_manager=mgr,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=1024,
    )


def _make_request(req_id: str, prompt_tokens: list[int]) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt_tokens,
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )


class _MockCPManager:
    _scores: dict[str, float]

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.allocated_calls: list[tuple[str, list[int]]] = []
        self.freed_calls: list[tuple[str, list[int]]] = []

    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float:
        _ = token_ids
        return self._scores.get(request_id, 0.0)

    def get_cp_cached_blocks(self, request_id: str) -> list[int]:
        _ = request_id
        return []

    def notify_blocks_allocated(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        self.allocated_calls.append((request_id, list(block_hashes)))

    def notify_blocks_freed(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        self.freed_calls.append((request_id, list(block_hashes)))

    def get_allocation_priority(self, request_ids: list[str]) -> list[str]:
        return sorted(
            request_ids,
            key=lambda request_id: self._scores.get(request_id, 0.0),
            reverse=True,
        )


def test_native_cp_aware_ordering() -> None:
    scheduler = _make_scheduler(max_num_seqs=1)
    manager = _MockCPManager({"req-low": 0.1, "req-high": 0.9})
    scheduler.set_cp_kv_manager(manager)

    req_low = _make_request("req-low", [1, 2, 3, 4, 5, 6, 7, 8])
    req_high = _make_request("req-high", [10, 11, 12, 13, 14, 15, 16, 17])
    scheduler.add_request(req_low)
    scheduler.add_request(req_high)

    output = scheduler.schedule()

    assert [req.request_id for req in output.scheduled_seqs] == ["req-high"]
    assert manager.allocated_calls
    assert manager.allocated_calls[0][0] == "req-high"
    assert len(manager.allocated_calls[0][1]) == 2

    scheduler.finish_request("req-high")
    assert manager.freed_calls
    assert manager.freed_calls[0][0] == "req-high"


def test_native_scheduler_without_cp() -> None:
    scheduler = _make_scheduler(max_num_seqs=1)

    req1 = _make_request("req-1", [1, 2, 3, 4, 5, 6])
    req2 = _make_request("req-2", [7, 8, 9, 10, 11, 12])
    scheduler.add_request(req1)
    scheduler.add_request(req2)

    output = scheduler.schedule()

    assert [req.request_id for req in output.scheduled_seqs] == ["req-1"]


def test_set_cp_kv_manager_native() -> None:
    scheduler = _make_scheduler(max_num_seqs=1)
    manager = _MockCPManager({})

    scheduler.set_cp_kv_manager(manager)

    assert getattr(scheduler, "_cp_kv_manager") is manager
