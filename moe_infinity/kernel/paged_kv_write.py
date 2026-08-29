from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage

_X = 8


def _validate(
    storage: "PagedKVStorage",
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if not 0 <= layer_idx < storage.spec.num_layers:
        raise ValueError(
            f"layer_idx {layer_idx} out of range [0, {storage.spec.num_layers})"
        )
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if key.ndim != 3:
        raise ValueError(
            "key/value must have shape [num_tokens, num_kv_heads, head_dim]"
        )
    num_tokens, num_kv_heads, head_dim = key.shape
    if num_kv_heads != storage.spec.num_kv_heads:
        raise ValueError("num_kv_heads mismatch with storage spec")
    if head_dim != storage.spec.head_dim:
        raise ValueError("head_dim mismatch with storage spec")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != num_tokens:
        raise ValueError("slot_mapping must have shape [num_tokens]")
    if key.device != storage.spec.device or value.device != storage.spec.device:
        raise ValueError("key/value must be on the storage device")
    if slot_mapping.device != storage.spec.device:
        raise ValueError("slot_mapping must be on the storage device")


def paged_kv_write_(
    storage: "PagedKVStorage",
    *,
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Allocation-free in-place current-token K/V write, safe under CUDA graph
    capture: validation precedes all writes and no new tensor/list/workspace is
    allocated on the write path."""
    _validate(storage, layer_idx, key, value, slot_mapping)

    block_size = storage.spec.block_size
    x = _X
    key_cache = storage.key_cache
    value_cache = storage.value_cache

    slots = slot_mapping.to(dtype=torch.long)
    block_ids = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots - block_ids * block_size

    num_tokens = key.shape[0]
    key_view = key.view(
        num_tokens, storage.spec.num_kv_heads, storage.spec.head_dim // x, x
    )
    for i in range(num_tokens):
        block_id = block_ids[i]
        offset = offsets[i]
        key_cache[layer_idx, block_id, :, :, offset, :] = key_view[i]
        value_cache[layer_idx, block_id, :, :, offset] = value[i]


__all__ = ["paged_kv_write_"]
