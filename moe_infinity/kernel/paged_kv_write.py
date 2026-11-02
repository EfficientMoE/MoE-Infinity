"""Single-launch paged KV-cache writes.

Every writer scatters ``[num_tokens, num_kv_heads, head_dim]`` K/V into a paged
cache with one Triton launch: ``block_id`` / ``offset`` are computed on device
from ``slot_mapping`` and nothing is read back to the host.

Supported destination layouts (per-layer views, unchanged from the callers):

* native K ``[blocks, H, D // 8, block, 8]`` and native V ``[blocks, H, D, block]``
  (consumed by ``paged_attention_v1`` and the Triton decode kernels);
* FlashInfer NHD ``[blocks, 2, block, H, D]``.

Slot validation: a slot is written only if ``0 <= slot < blocks * block``; any
other slot is dropped on device, so a bad slot can never write out of bounds.
Slots are produced on the host by the block allocator and only reference
allocated blocks, so the strict ``ValueError`` check (which needs a device to
host read of every slot) runs only where it is free (CPU tensors) or when
``MOE_INFINITY_KV_WRITE_CHECK=1`` is set for debugging.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

try:
    import triton
    import triton.language as tl
    from triton import knobs
    from triton.runtime import driver as _driver

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage

_STRICT_ENV = "MOE_INFINITY_KV_WRITE_CHECK"
# Elements (BLOCK_T * BLOCK_D) per program and tensor below _FULL_BLOCK_TOKENS:
# small tiles keep many programs in flight for decode-sized batches.
_TILE_ELEMS = 512
# Elements per warp.
_WARP_ELEMS = 512
# From this many tokens on, a program writes one whole cache block.
_FULL_BLOCK_TOKENS = 1024


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
    if key.dtype != storage.spec.dtype or value.dtype != storage.spec.dtype:
        raise ValueError("key/value dtype must match the storage dtype")


def check_slots(slot_mapping: torch.Tensor, num_blocks: int, block_size: int):
    """Raise ``ValueError`` for a slot outside the allocated blocks.

    Free on CPU tensors; on CUDA it synchronises, so writers only call it when
    ``MOE_INFINITY_KV_WRITE_CHECK=1``.
    """
    if slot_mapping.numel() == 0:
        return
    if bool((slot_mapping < 0).any()):
        raise ValueError("slot_mapping contains negative slot index")
    if bool((slot_mapping >= num_blocks * block_size).any()):
        raise ValueError("slot_mapping points past allocated blocks")


def _strict(slot_mapping: torch.Tensor) -> bool:
    return not slot_mapping.is_cuda or os.environ.get(_STRICT_ENV, "") not in (
        "",
        "0",
    )


if _HAS_TRITON:

    @triton.jit
    def _scatter_tile(
        src_ptr,
        dst_ptr,
        tok0,
        head,
        slot,
        valid,
        blk0,
        off0,
        is_run,
        is_full,
        S_T: tl.constexpr,
        S_H: tl.constexpr,
        S_B: tl.constexpr,
        S_HD: tl.constexpr,
        S_DO: tl.constexpr,
        S_DI: tl.constexpr,
        S_O: tl.constexpr,
        X: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # src = tok*S_T + head*S_H + d
        # dst = blk*S_B + head*S_HD + (d // X)*S_DO + (d % X)*S_DI + off*S_O
        # Scalar bases are int64; offsets inside one tile stay int32.
        t = tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        mask = valid[:, None] & (d < HEAD_DIM)[None, :]
        x = tl.load(
            src_ptr + tok0 * S_T + head * S_H + t[:, None] * S_T + d[None, :],
            mask=mask,
            eviction_policy="evict_first",
        ).to(dst_ptr.dtype.element_ty)
        d_off = (d // X)[None, :] * S_DO + (d % X)[None, :] * S_DI
        base = dst_ptr + blk0 * S_B + head * S_HD
        if is_full:
            # Whole-block tile of a canonical native page: the page of this
            # (block, head) is one contiguous BLOCK_SIZE * HEAD_DIM run.
            # Transpose in registers so the stores are contiguous, vectorised
            # and run along tokens across lanes.
            if X == 1:
                # V [D, block]
                tl.store(
                    base + d[:, None] * BLOCK_SIZE + t[None, :], tl.trans(x)
                )
            else:
                # K [D // X, block, X] as a [t, g, i] tile
                g = tl.arange(0, BLOCK_D // X)
                i = tl.arange(0, X)
                tl.store(
                    base
                    + t[:, None, None] * X
                    + g[None, :, None] * (BLOCK_SIZE * X)
                    + i[None, None, :],
                    tl.reshape(x, (BLOCK_T, BLOCK_D // X, X)),
                )
        elif is_run:
            # One run of consecutive slots inside a single block (prefill):
            # scalar + arange offsets coalesce the stores along tokens.
            off = (off0 + t)[:, None]
            tl.store(base + off * S_O + d_off, x, mask=mask)
        else:
            slot = tl.where(valid, slot, 0)
            blk = (slot // BLOCK_SIZE)[:, None]
            off = (slot % BLOCK_SIZE).to(tl.int32)[:, None]
            tl.store(
                dst_ptr + blk * S_B + head * S_HD + off * S_O + d_off,
                x,
                mask=mask,
            )

    @triton.jit
    def _kv_scatter_kernel(
        key_ptr,
        value_ptr,
        slot_ptr,
        k_dst_ptr,
        v_dst_ptr,
        num_tokens,
        num_slots,
        S_KT: tl.constexpr,
        S_KH: tl.constexpr,
        S_VT: tl.constexpr,
        S_VH: tl.constexpr,
        K_SB: tl.constexpr,
        K_SH: tl.constexpr,
        K_SDO: tl.constexpr,
        K_SDI: tl.constexpr,
        K_SO: tl.constexpr,
        KX: tl.constexpr,
        V_SB: tl.constexpr,
        V_SH: tl.constexpr,
        V_SDO: tl.constexpr,
        V_SDI: tl.constexpr,
        V_SO: tl.constexpr,
        VX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        CANONICAL: tl.constexpr,
    ):
        head = tl.program_id(1)
        t = tl.arange(0, BLOCK_T)
        tok = tl.program_id(0) * BLOCK_T + t
        tok_mask = tok < num_tokens
        slot = tl.load(slot_ptr + tok, mask=tok_mask, other=-1).to(tl.int64)
        valid = tok_mask & (slot >= 0) & (slot < num_slots)
        tok0 = tl.program_id(0).to(tl.int64) * BLOCK_T

        slot0 = tl.load(slot_ptr + tl.program_id(0) * BLOCK_T).to(tl.int64)
        blk0 = slot0 // BLOCK_SIZE
        off0 = (slot0 - blk0 * BLOCK_SIZE).to(tl.int32)
        n_gap = tl.sum(((slot != slot0 + t) & tok_mask).to(tl.int32), axis=0)
        is_run = (
            (n_gap == 0)
            & (slot0 >= 0)
            & (off0 + BLOCK_T <= BLOCK_SIZE)
            & (slot0 + BLOCK_T <= num_slots)
        )
        if CANONICAL and BLOCK_T == BLOCK_SIZE and HEAD_DIM == BLOCK_D:
            n_tok = tl.sum(tok_mask.to(tl.int32), axis=0)
            is_full = is_run & (n_tok == BLOCK_T)
        else:
            is_full = False

        _scatter_tile(
            key_ptr, k_dst_ptr, tok0, head, slot, valid, blk0, off0, is_run,
            is_full, S_KT, S_KH, K_SB, K_SH, K_SDO, K_SDI, K_SO, KX,
            HEAD_DIM, BLOCK_D, BLOCK_T, BLOCK_SIZE,
        )  # fmt: skip
        _scatter_tile(
            value_ptr, v_dst_ptr, tok0, head, slot, valid, blk0, off0, is_run,
            is_full, S_VT, S_VH, V_SB, V_SH, V_SDO, V_SDI, V_SO, VX,
            HEAD_DIM, BLOCK_D, BLOCK_T, BLOCK_SIZE,
        )  # fmt: skip


def _x_layout_geometry(cache: torch.Tensor):
    """``[blocks, H, D // x, block, x]`` -> (s_b, s_h, s_do, s_di, s_o, x)."""
    s = cache.stride()
    return s[0], s[1], s[2], s[4], s[3], cache.shape[-1]


def _hd_block_geometry(cache: torch.Tensor):
    """``[blocks, H, D, block]`` -> (s_b, s_h, s_do, 0, s_o, 1)."""
    s = cache.stride()
    return s[0], s[1], s[2], 0, s[3], 1


def _nhd_geometry(cache: torch.Tensor):
    """``[blocks, block, H, D]`` -> (s_b, s_h, s_do, 0, s_o, 1)."""
    s = cache.stride()
    return s[0], s[2], s[3], 0, s[1], 1


def _canonical_native(k_geo, v_geo, block_size: int) -> bool:
    """Each (block, head) page is dense: K ``[D // x, block, x]``, V
    ``[D, block]`` (true for per-layer views of the contiguous caches)."""
    _, _, k_do, k_di, k_o, x = k_geo
    _, _, v_do, _, v_o, _ = v_geo
    return (
        k_di == 1
        and k_o == x
        and k_do == block_size * x
        and v_o == 1
        and v_do == block_size
    )


def _src(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


# Compiled kernels keyed by everything Triton specialises on, so steady-state
# launches skip the JIT dispatcher (~10 us of host time per call).
_COMPILED: dict = {}
_FAST_LAUNCH = True


def _get_stream(device_index: int):
    return _driver.active.get_current_stream(device_index)


def _launch_hooks_active() -> bool:
    # Launch hooks (e.g. proton) need launch metadata: take the JIT path.
    hook = knobs.runtime.launch_enter_hook
    return hook is not None and bool(getattr(hook, "calls", True))


def _align16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _launch(key, value, slots, k_dst, v_dst, k_geo, v_geo, num_blocks, bs):
    global _FAST_LAUNCH
    num_tokens, num_kv_heads, head_dim = key.shape
    if num_tokens == 0:
        return
    key = _src(key)
    value = _src(value)
    # Plain-Python helpers: triton.next_power_of_2 / cdiv are slow to call.
    block_d = 1 << (head_dim - 1).bit_length()
    # Tokens per program: at most one cache block, so a prefill tile can be a
    # single consecutive slot run (see the kernel's run / whole-block paths).
    block_cap = 1 << (bs.bit_length() - 1)
    if num_tokens >= _FULL_BLOCK_TOKENS:
        block_t = block_cap
    else:
        block_t = min(
            1 << (num_tokens - 1).bit_length(),
            max(1, _TILE_ELEMS // block_d),
            block_cap,
        )
    num_warps = max(1, min(8, block_t * block_d // _WARP_ELEMS))
    grid = (-(-num_tokens // block_t), num_kv_heads, 1)
    num_slots = num_blocks * bs
    args = (key, value, slots, k_dst, v_dst, num_tokens, num_slots)
    s_k = key.stride()
    s_v = value.stride()
    constexprs = (
        s_k[0],
        s_k[1],
        s_v[0],
        s_v[1],
        *k_geo,
        *v_geo,
        head_dim,
        block_d,
        block_t,
        bs,
        _canonical_native(k_geo, v_geo, bs),
    )
    spec = (
        constexprs,
        num_warps,
        key.dtype,
        value.dtype,
        slots.dtype,
        k_dst.dtype,
        v_dst.dtype,
        key.device.index,
        num_tokens == 1,
        num_tokens % 16 == 0,
        num_slots,
        _align16(key),
        _align16(value),
        _align16(slots),
        _align16(k_dst),
        _align16(v_dst),
    )
    kernel = _COMPILED.get(spec) if _FAST_LAUNCH else None
    if kernel is not None and not _launch_hooks_active():
        try:
            kernel.run(
                grid[0],
                grid[1],
                1,
                _get_stream(key.device.index),
                kernel.function,
                kernel.packed_metadata,
                None,
                None,
                None,
                *args,
                *constexprs,
            )
            return
        except Exception:  # Triton launcher ABI changed: use the JIT path
            _FAST_LAUNCH = False
    kernel = _kv_scatter_kernel[grid](*args, *constexprs, num_warps=num_warps)
    if _FAST_LAUNCH and len(_COMPILED) < 256:
        _COMPILED[spec] = kernel


def _use_triton(t: torch.Tensor) -> bool:
    return _HAS_TRITON and t.is_cuda


def _block_offsets(slots: torch.Tensor, block_size: int):
    slots = slots.to(dtype=torch.long)
    block_ids = torch.div(slots, block_size, rounding_mode="floor")
    return block_ids, slots - block_ids * block_size


def paged_kv_scatter_(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write K/V into one layer of the native paged cache in one launch.

    ``k_cache``: ``[blocks, H, D // x, block, x]``; ``v_cache``:
    ``[blocks, H, D, block]``; ``key`` / ``value``: ``[T, H, D]`` on the cache
    device (any dtype; cast on store); ``slot_mapping``: ``[T]`` int32/int64.
    """
    num_blocks, _, _, block_size, x = k_cache.shape
    if _strict(slot_mapping):
        check_slots(slot_mapping, num_blocks, block_size)
    if _use_triton(k_cache):
        _launch(
            key,
            value,
            slot_mapping,
            k_cache,
            v_cache,
            _x_layout_geometry(k_cache),
            _hd_block_geometry(v_cache),
            num_blocks,
            block_size,
        )
        return
    num_tokens, num_kv_heads, head_dim = key.shape
    block_ids, offsets = _block_offsets(slot_mapping, block_size)
    k_cache[block_ids, :, :, offsets, :] = key.reshape(
        num_tokens, num_kv_heads, head_dim // x, x
    ).to(k_cache.dtype)
    v_cache[block_ids, :, :, offsets] = value.to(v_cache.dtype)


def flashinfer_kv_scatter_(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write K/V into one layer of a FlashInfer NHD cache
    ``[blocks, 2, block, H, D]`` in one launch."""
    num_blocks, _, block_size = kv_cache.shape[:3]
    if _strict(slot_mapping):
        check_slots(slot_mapping, num_blocks, block_size)
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    if _use_triton(kv_cache):
        _launch(
            key,
            value,
            slot_mapping,
            k_cache,
            v_cache,
            _nhd_geometry(k_cache),
            _nhd_geometry(v_cache),
            num_blocks,
            block_size,
        )
        return
    block_ids, offsets = _block_offsets(slot_mapping, block_size)
    k_cache[block_ids, offsets] = key.to(kv_cache.dtype)
    v_cache[block_ids, offsets] = value.to(kv_cache.dtype)


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
    allocated on the write path.

    On CUDA the write is a single Triton kernel launch whose grid is fixed by
    the captured ``[num_tokens, num_kv_heads]`` dimensions. On CPU (or when
    Triton is unavailable) the equivalent vectorised indexed write is used."""
    _validate(storage, layer_idx, key, value, slot_mapping)
    paged_kv_scatter_(
        storage.key_cache[layer_idx],
        storage.value_cache[layer_idx],
        key,
        value,
        slot_mapping,
    )


__all__ = [
    "check_slots",
    "flashinfer_kv_scatter_",
    "paged_kv_scatter_",
    "paged_kv_write_",
]
