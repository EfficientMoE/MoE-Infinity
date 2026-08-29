from __future__ import annotations

import pytest
import torch

from moe_infinity.kernel.paged_kv_write import paged_kv_write_
from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    PagedKVStorageSpec,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _make_storage(
    *,
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    device: torch.device | None = None,
) -> PagedKVStorage:
    spec = PagedKVStorageSpec(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=device or torch.device("cpu"),
    )
    return PagedKVStorage(spec)


def _assert_slots_equal(storage, *, layer_idx, slots, key, value) -> None:
    x = 8
    block_size = storage.spec.block_size
    for i in range(slots.shape[0]):
        slot = int(slots[i].item())
        block_id = slot // block_size
        offset = slot % block_size
        expected_key = key[i].reshape(
            storage.spec.num_kv_heads, storage.spec.head_dim // x, x
        )
        torch.testing.assert_close(
            storage.key_cache[layer_idx, block_id, :, :, offset, :].cpu(),
            expected_key.cpu().to(storage.spec.dtype),
        )
        torch.testing.assert_close(
            storage.value_cache[layer_idx, block_id, :, :, offset].cpu(),
            value[i].cpu().to(storage.spec.dtype),
        )


def test_paged_kv_write_allocation_free_layout_cpu() -> None:
    storage = _make_storage(num_layers=2, num_blocks=4, block_size=4)
    slots = torch.tensor([1, 6], dtype=torch.int64)
    key = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8)
    value = key + 100

    paged_kv_write_(
        storage, layer_idx=1, key=key, value=value, slot_mapping=slots
    )

    _assert_slots_equal(storage, layer_idx=1, slots=slots, key=key, value=value)
    assert torch.count_nonzero(storage.value_cache[0]).item() == 0


def test_paged_kv_write_rejects_device_mismatch_cpu() -> None:
    storage = _make_storage()
    slots = torch.tensor([0], dtype=torch.int64)
    key = torch.zeros(1, 2, 8)
    value = torch.zeros(1, 2, 8)
    with pytest.raises(ValueError):
        paged_kv_write_(
            storage,
            layer_idx=5,
            key=key,
            value=value,
            slot_mapping=slots,
        )


@requires_cuda
def test_graph_safe_kv_write_persists_current_token_per_layer() -> None:
    storage = _make_storage(
        num_layers=2, num_blocks=4, block_size=4, device=torch.device("cuda")
    )
    slots = torch.tensor([1, 6], dtype=torch.int64, device="cuda")
    key = torch.arange(2 * 2 * 8, device="cuda", dtype=torch.float32).reshape(
        2, 2, 8
    )
    value = key + 100
    paged_kv_write_(
        storage, layer_idx=1, key=key, value=value, slot_mapping=slots
    )
    _assert_slots_equal(storage, layer_idx=1, slots=slots, key=key, value=value)
    assert torch.count_nonzero(storage.value_cache[0]).item() == 0
