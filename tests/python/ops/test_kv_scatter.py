"""Single-launch KV-cache scatter vs the former per-token Python loop."""

from __future__ import annotations

import pytest
import torch

from moe_infinity.kernel.paged_kv_write import (
    flashinfer_kv_scatter_,
    paged_kv_scatter_,
)
from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    PagedKVStorageSpec,
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
X = 8


def _loop_native(k_store, v_store, key, value, slots, block_size):
    """The pre-#305 ``PagedAttentionBackend.write_kv`` body."""
    num_heads, head_dim = key.shape[1:]
    for i in range(key.shape[0]):
        slot = int(slots[i].item())
        block_id, off = slot // block_size, slot % block_size
        k_store[block_id, :, :, off, :] = key[i].reshape(
            num_heads, head_dim // X, X
        )
        v_store[block_id, :, :, off] = value[i]


def _loop_flashinfer(fi_store, key, value, slots, block_size):
    """The pre-#305 ``PagedAttentionBackend.write_kv_flashinfer`` body."""
    for i in range(key.shape[0]):
        slot = int(slots[i].item())
        block_id, off = slot // block_size, slot % block_size
        fi_store[block_id, 0, off] = key[i]
        fi_store[block_id, 1, off] = value[i]


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _rand(*shape, dtype, device, gen):
    return torch.randn(*shape, generator=gen, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def _slots(T, nslots, block_size, pattern, gen):
    if pattern == "perm":
        return torch.randperm(nslots, generator=gen)[:T]
    # Prefill: consecutive positions of one sequence over a shuffled block
    # table, optionally starting mid-block (prefix-cached / chunked prefill).
    start = 0 if pattern == "prefill" else block_size // 2 + 1
    table = torch.randperm(nslots // block_size, generator=gen)
    pos = torch.arange(start, start + T)
    return table[pos // block_size] * block_size + pos % block_size


def _inputs(
    T,
    H,
    D,
    dtype,
    device,
    gen,
    *,
    contiguous,
    slot_dtype,
    nslots,
    block_size=16,
    pattern="perm",
):
    if contiguous:
        key = _rand(T, H, D, dtype=dtype, device=device, gen=gen)
        value = _rand(T, H, D, dtype=dtype, device=device, gen=gen)
    else:
        # K/V as strided views of a fused QKV projection output.
        qkv = _rand(T, 3 * H, D, dtype=dtype, device=device, gen=gen)
        key, value = qkv[:, H : 2 * H], qkv[:, 2 * H :]
    slots = _slots(T, nslots, block_size, pattern, gen).to(slot_dtype)
    return key, value, slots.to(device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize(
    ("T", "H", "D", "block_size"),
    [
        (1, 4, 128, 16),
        (7, 2, 256, 16),
        (300, 4, 128, 16),
        (65, 1, 192, 32),
        (33, 8, 64, 1),
        (50, 2, 64, 24),
        # >= 1024 tokens: whole-block tiles (transposed store path).
        (1100, 2, 128, 16),
        (1030, 1, 192, 16),
        (1024, 1, 64, 32),
    ],
)
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("pattern", ["perm", "prefill", "prefill_unaligned"])
def test_native_scatter_byte_identical_to_loop(
    device, dtype, T, H, D, block_size, contiguous, slot_dtype, pattern
):
    gen = torch.Generator().manual_seed(T * 1000 + D + H)
    num_blocks = -(-T // block_size) + 3
    key, value, slots = _inputs(
        T,
        H,
        D,
        dtype,
        device,
        gen,
        contiguous=contiguous,
        slot_dtype=slot_dtype,
        nslots=num_blocks * block_size,
        block_size=block_size,
        pattern=pattern,
    )
    k_init = _rand(
        num_blocks,
        H,
        D // X,
        block_size,
        X,
        dtype=dtype,
        device=device,
        gen=gen,
    )
    v_init = _rand(
        num_blocks, H, D, block_size, dtype=dtype, device=device, gen=gen
    )
    k_ref, v_ref = k_init.clone(), v_init.clone()
    k_new, v_new = k_init.clone(), v_init.clone()

    _loop_native(k_ref, v_ref, key, value, slots, block_size)
    # The first launch of a configuration goes through the Triton JIT; later
    # ones reuse the cached compiled kernel. Check the cached launch.
    paged_kv_scatter_(k_init.clone(), v_init.clone(), key, value, slots)
    paged_kv_scatter_(k_new, v_new, key, value, slots)

    assert _bytes_equal(k_new, k_ref)
    assert _bytes_equal(v_new, v_ref)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("T", "H", "D"), [(1, 4, 128), (257, 2, 64)])
def test_flashinfer_scatter_byte_identical_to_loop(device, dtype, T, H, D):
    gen = torch.Generator().manual_seed(T + D)
    block_size = 16
    num_blocks = -(-T // block_size) + 2
    key, value, slots = _inputs(
        T,
        H,
        D,
        dtype,
        device,
        gen,
        contiguous=False,
        slot_dtype=torch.int64,
        nslots=num_blocks * block_size,
    )
    fi_init = _rand(
        num_blocks, 2, block_size, H, D, dtype=dtype, device=device, gen=gen
    )
    fi_ref, fi_new = fi_init.clone(), fi_init.clone()

    _loop_flashinfer(fi_ref, key, value, slots, block_size)
    flashinfer_kv_scatter_(fi_init.clone(), key, value, slots)
    flashinfer_kv_scatter_(fi_new, key, value, slots)

    assert _bytes_equal(fi_new, fi_ref)


@pytest.mark.parametrize("device", _DEVICES)
def test_backend_and_storage_writers_match_loop(device):
    gen = torch.Generator().manual_seed(0)
    T, H, D, block_size, num_blocks = 50, 4, 128, 16, 6
    spec = KVCacheSpec(
        num_kv_heads=H, head_dim=D, dtype=torch.bfloat16, block_size=block_size
    )
    backend = PagedAttentionBackend(
        spec, num_blocks, torch.device(device), num_layers=2
    )
    storage = PagedKVStorage(
        PagedKVStorageSpec(
            num_layers=2,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=H,
            head_dim=D,
            dtype=torch.bfloat16,
            device=torch.device(device),
        )
    )
    # fp32 inputs exercise the dtype normalisation of the eager writers.
    key, value, slots = _inputs(
        T,
        H,
        D,
        torch.float32,
        device,
        gen,
        contiguous=True,
        slot_dtype=torch.int64,
        nslots=num_blocks * block_size,
    )
    backend.write_kv(key, value, slots, layer_idx=1)
    storage.write_kv(layer_idx=1, key=key, value=value, slot_mapping=slots)

    k_ref = torch.zeros_like(backend._k_cache[1])
    v_ref = torch.zeros_like(backend._v_cache[1])
    _loop_native(
        k_ref,
        v_ref,
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        slots,
        block_size,
    )
    for k_cache, v_cache in (
        (backend._k_cache, backend._v_cache),
        (storage.key_cache, storage.value_cache),
    ):
        assert _bytes_equal(k_cache[1], k_ref)
        assert _bytes_equal(v_cache[1], v_ref)
        assert torch.count_nonzero(k_cache[0]).item() == 0
        assert torch.count_nonzero(v_cache[0]).item() == 0


@pytest.mark.parametrize("slot", [-1, 64])
def test_cpu_rejects_out_of_range_slot(slot):
    k = torch.zeros(4, 2, 8 // X, 16, X)
    v = torch.zeros(4, 2, 8, 16)
    key = torch.ones(2, 2, 8)
    slots = torch.tensor([3, slot])
    with pytest.raises(ValueError, match="slot_mapping"):
        paged_kv_scatter_(k, v, key, key, slots)
    assert torch.count_nonzero(k).item() == 0


@requires_cuda
def test_cuda_masks_out_of_range_slots(monkeypatch):
    monkeypatch.delenv("MOE_INFINITY_KV_WRITE_CHECK", raising=False)
    num_blocks, block_size = 4, 16
    k = torch.zeros(num_blocks, 2, 1, block_size, X, device="cuda")
    v = torch.zeros(num_blocks, 2, 8, block_size, device="cuda")
    key = torch.ones(3, 2, 8, device="cuda")
    slots = torch.tensor([-1, num_blocks * block_size, 5], device="cuda")
    paged_kv_scatter_(k, v, key, key, slots)
    torch.cuda.synchronize()
    # Only slot 5 (block 0, offset 5) is written; nothing out of bounds.
    assert torch.count_nonzero(k).item() == 2 * 8
    assert torch.count_nonzero(k[0, :, :, 5]).item() == 2 * 8
    assert torch.count_nonzero(v[0, :, :, 5]).item() == 2 * 8

    monkeypatch.setenv("MOE_INFINITY_KV_WRITE_CHECK", "1")
    with pytest.raises(ValueError, match="negative"):
        paged_kv_scatter_(k, v, key, key, slots)


@requires_cuda
def test_cuda_cached_launch_is_used():
    from moe_infinity.kernel import paged_kv_write

    k = torch.zeros(4, 2, 1, 16, X, device="cuda")
    v = torch.zeros(4, 2, 8, 16, device="cuda")
    key = torch.ones(3, 2, 8, device="cuda")
    slots = torch.tensor([1, 2, 40], device="cuda")
    paged_kv_scatter_(k, v, key, key, slots)
    assert paged_kv_write._FAST_LAUNCH
    assert paged_kv_write._COMPILED
    k.zero_()
    v.zero_()
    paged_kv_scatter_(k, v, key * 2, key * 3, slots)
    torch.cuda.synchronize()
    assert paged_kv_write._FAST_LAUNCH
    assert torch.equal(k[2, :, :, 8], torch.full((2, 1, X), 2.0, device="cuda"))
    assert torch.equal(v[0, :, :, 1], torch.full((2, 8), 3.0, device="cuda"))


@requires_cuda
def test_cuda_write_kv_has_no_host_sync():
    H, D, block_size, num_blocks = 4, 128, 16, 64
    spec = KVCacheSpec(
        num_kv_heads=H, head_dim=D, dtype=torch.bfloat16, block_size=block_size
    )
    backend = PagedAttentionBackend(
        spec, num_blocks, torch.device("cuda"), num_layers=1
    )
    key = torch.randn(512, H, D, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(num_blocks * block_size, device="cuda")[:512]
    backend.write_kv(key, key, slots)  # compile outside the check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        backend.write_kv(key, key, slots)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
