import importlib.machinery
import sys
import types
from collections import deque
from typing import cast

import pytest
import torch

from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)

if (
    "flash_attn" not in sys.modules
    or getattr(sys.modules["flash_attn"], "__spec__", None) is None
):
    flash_attn_stub = sys.modules.get(
        "flash_attn", types.ModuleType("flash_attn")
    )
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub

from moe_infinity.runtime.model_offload import OffloadEngine


def _make_kv_cache(
    num_blocks: int = 4,
    dtype: torch.dtype = torch.float16,
) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def _make_scheduler(num_blocks: int = 4) -> Scheduler:
    return Scheduler(
        kv_cache=_make_kv_cache(num_blocks=num_blocks),
        max_batch_size=4,
        max_tokens_per_step=64,
    )


def test_partial_swap_failure_rollback() -> None:
    from moe_infinity.engine.kv_transfer import (
        KVTransferState,
        PageableCPUBufferRecord,
    )

    kv_cache = _make_kv_cache()
    seq_id = 7
    kv_cache.allocate_sequence(seq_id=seq_id, num_tokens=4)
    kv_cache.swap_out(seq_id)
    kv_cache.free_gpu_blocks(seq_id)

    record = kv_cache._kv_records[seq_id]
    assert record.state is KVTransferState.HOST_RESIDENT

    class _FaultyCpuBuffer(PageableCPUBufferRecord):
        def require_tensor(self) -> torch.Tensor:
            raise RuntimeError("simulated swap_in failure")

    record.pageable_buffer = _FaultyCpuBuffer(tensor=torch.zeros(1), nbytes=0)

    with pytest.raises(RuntimeError, match="simulated swap_in failure"):
        kv_cache.swap_in(seq_id)

    assert kv_cache._kv_records[seq_id].state is KVTransferState.HOST_RESIDENT


def test_gpu_oom_on_swap_in_handled_gracefully(monkeypatch) -> None:
    scheduler = _make_scheduler(num_blocks=2)
    sequence = SequenceData(
        seq_id=42,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(),
    )
    sequence.set_status(SequenceStatus.SWAPPED)
    group = SequenceGroup(request_id="oom-req", sequences=[sequence])

    swapped_queue = cast(deque[SequenceGroup], getattr(scheduler, "_swapped"))
    request_map = cast(
        dict[str, SequenceGroup], getattr(scheduler, "_request_map")
    )
    sequence_map = cast(
        dict[int, SequenceData], getattr(scheduler, "_sequence_map")
    )
    running_queue = cast(deque[SequenceGroup], getattr(scheduler, "_running"))

    swapped_queue.append(group)
    request_map[group.request_id] = group
    sequence_map[sequence.seq_id] = sequence

    def _raise_oom(_seq_id: int) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(scheduler.kv_cache, "swap_in", _raise_oom)

    scheduler._recover_swapped_groups([group])

    assert sequence.status is SequenceStatus.SWAPPED
    assert group in swapped_queue
    assert group not in running_queue


def test_dtype_preserved_through_swap_cycle() -> None:
    kv_cache = _make_kv_cache(dtype=torch.float16)
    seq_id = 3
    kv_cache.allocate_sequence(seq_id=seq_id, num_tokens=8)
    block_ids = kv_cache.get_block_table(seq_id)
    assert block_ids

    original = torch.randn(
        (
            kv_cache.num_layers,
            len(block_ids),
            2,
            kv_cache.block_size,
            kv_cache.num_heads,
            kv_cache.head_dim,
        ),
        dtype=torch.float16,
    )
    kv_tensor = kv_cache.get_kv_cache_tensors()
    kv_tensor[:, block_ids, ...] = original

    kv_cache.swap_out(seq_id)
    kv_tensor[:, block_ids, ...] = torch.zeros_like(original)
    kv_cache.swap_in(seq_id)

    restored = kv_cache.get_kv_cache_tensors()[:, block_ids, ...]
    assert restored.dtype == original.dtype
    assert torch.allclose(restored, original)


def test_swap_in_with_empty_cpu_cache_noop() -> None:
    from moe_infinity.engine.kv_transfer import KVTransferState

    kv_cache = _make_kv_cache()
    seq_id = 99
    kv_cache.allocate_sequence(seq_id=seq_id, num_tokens=4)

    state_before = kv_cache.transfer_state(seq_id)
    blocks_before = kv_cache.get_block_table(seq_id)

    kv_cache.swap_in(seq_id)

    assert kv_cache.transfer_state(seq_id) is state_before
    assert kv_cache.transfer_state(seq_id) is KVTransferState.GPU_RESIDENT
    assert kv_cache.get_block_table(seq_id) == blocks_before


def test_capture_kv_with_none_past_kv_values_noop() -> None:
    engine = OffloadEngine.__new__(OffloadEngine)
    setattr(engine, "_enable_kv_cache_offload", True)
    setattr(engine, "_captured_kv", {})

    OffloadEngine._capture_kv_cache(
        engine,
        seq_id=0,
        past_key_values=None,
    )

    assert getattr(engine, "_captured_kv") == {}
