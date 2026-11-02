"""GQA-native SDPA attention for HF models that have no paged shim.

HF ``eager`` attention materialises ``[B, H, Q, K]`` scores and ``repeat_kv``
copies the whole KV cache on every call, so it is 2-8 % of the BF16 roof for
prefill (OOM from ~12K tokens) and moves 12-33x the ideal KV bytes in decode.
HF ``sdpa`` avoids that only without an explicit mask and has no attention
sinks (GPT-OSS).  This module provides one attention function, registered with
the HF attention interface, that

* decode (one query token): a split-KV Triton kernel
  (``kernel/gqa_attention.py``) that reads every KV byte once, with the
  sliding window, the key mask and the sinks fused in;
* prefill without an explicit mask: PyTorch's built-in FlashAttention-2
  (``aten._flash_attention_forward``): native GQA (K/V never expanded),
  bottom-right causal, in-kernel sliding window;
* applies attention sinks exactly from the kernel's log-sum-exp:
  ``softmax([s, sink])[:-1] @ V == (softmax(s) @ V) * sigmoid(lse - sink)``;
* prefill with an explicit mask (padding, MiniMax-M3 block mask, GLM DSA
  index mask) keeps HF's SDPA-with-mask semantics (decode without Triton
  folds the query-head group into the query axis: zero-copy GQA);
* falls back to HF's ``sdpa`` (or a math reference when sinks / a sliding
  window are involved) for anything the fast paths do not cover.

Two names are registered:

``"moe_gqa"``  attention + mask function.  The mask function never
    materialises the causal / sliding-window pattern when there is no
    padding; the window is applied in the kernel from the ``sliding_window``
    kwarg.  Used for GPT-OSS (HF marks it ``_supports_sdpa = False`` because
    of the sinks).
``"sdpa"``      routed to the same function with HF ``sdpa`` semantics (mask
    is the whole truth, ``sliding_window`` ignored).  GLM-DSA / MiniMax-M3
    only build their sparse masks when the implementation is literally
    ``"sdpa"``, so the key itself is kept and only the kernel changes.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F

MOE_GQA = "moe_gqa"  # no "sdpa" substring: HF gates those on _supports_sdpa

# Families whose HF attention passes extra kwargs (sinks) that HF sdpa ignores.
_MOE_GQA_MODEL_TYPES = {"gpt_oss"}
# Kept on eager: the model is cast to float8 before conversion (unvalidated).
_EAGER_MODEL_TYPES = {"deepseek_v3"}
# HF code that edits the SDPA padding mask as if it were additive.
_ADDITIVE_SDPA_MASK_TYPES = {"glm_moe_dsa"}

_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@functools.lru_cache(maxsize=1)
def _has_flash_op() -> bool:
    try:
        schema = str(torch.ops.aten._flash_attention_forward.default._schema)
    except (AttributeError, RuntimeError):
        return False
    return "window_size_left" in schema


@functools.lru_cache(maxsize=1)
def _triton_kernels():
    if os.environ.get("MOE_DISABLE_FUSED_KERNELS", "0") == "1":
        return None
    try:
        from moe_infinity.kernel import gqa_attention
    except Exception:  # no Triton: FA2 / SDPA decode paths below
        return None
    return gqa_attention


def _flash_eligible(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    d = q.shape[-1]
    return (
        q.is_cuda
        and q.dtype in _FLASH_DTYPES
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and k.shape[-1] == d
        and v.shape[-1] == d
        and d % 8 == 0
        and d <= 256
        and q.shape[1] % k.shape[1] == 0
        and _has_flash_op()
    )


def _last_dim_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal: bool,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 on ``[B, H, T, D]`` inputs (K/V may have fewer heads).

    Returns ``(out [B, Q, H, D], lse [B, H, Q] fp32)``.  ``causal`` is
    bottom-right aligned; ``window`` keeps keys ``(q - window, q]``.
    """
    q = _last_dim_contiguous(query.transpose(1, 2))
    k = _last_dim_contiguous(key.transpose(1, 2))
    v = _last_dim_contiguous(value.transpose(1, 2))
    out, lse, *_ = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[1],
        k.shape[1],
        0.0,
        causal and q.shape[1] > 1,
        False,
        scale=scale,
        window_size_left=None if window is None else window - 1,
        window_size_right=None if window is None else 0,
    )
    return out, lse


def _additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype == torch.bool:
        zero = torch.zeros((), dtype=dtype, device=mask.device)
        return torch.where(mask, zero, torch.finfo(dtype).min)
    return mask.to(dtype)


def _efficient_attention_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient kernel with an explicit mask; returns (out, lse).

    q/k/v are ``[B, H, T, D]`` with equal head counts; ``mask`` broadcasts
    to ``[B, H, Q, K]``.  The bias is copied into a 16-aligned buffer
    (kernel requirement) without expanding broadcast dims.
    """
    bsz, heads, qlen, _ = q.shape
    klen = k.shape[2]
    bias = _additive(mask, q.dtype)
    while bias.dim() < 4:
        bias = bias.unsqueeze(0)
    padded = -(-klen // 16) * 16
    buf = bias.new_zeros(*bias.shape[:-1], padded)[..., :klen]
    buf.copy_(bias)
    buf = buf.expand(bsz, heads, qlen, klen)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, buf, True, scale=scale
    )
    return out, lse[..., :qlen]


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scale: float,
    causal: bool,
    window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """fp32 math reference (CPU / unsupported inputs); returns [B, Q, H, Dv]."""
    n_rep = query.shape[1] // key.shape[1]
    q = query.float()
    k = _repeat_kv(key, n_rep).float()
    v = _repeat_kv(value, n_rep).float()
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    qlen, klen = s.shape[-2:]
    if attention_mask is not None:
        s = s + _additive(attention_mask, torch.float32)
    else:
        qi = torch.arange(qlen, device=s.device)[:, None] + (klen - qlen)
        ki = torch.arange(klen, device=s.device)[None, :]
        keep = torch.ones(qlen, klen, dtype=torch.bool, device=s.device)
        if causal and qlen > 1:
            keep &= ki <= qi
        if window is not None:
            keep &= ki > qi - window
        s = s.masked_fill(~keep, float("-inf"))
    if sinks is not None:
        snk = sinks.float().reshape(1, -1, 1, 1).expand(*s.shape[:-1], 1)
        p = torch.softmax(torch.cat([s, snk], dim=-1), dim=-1)[..., :-1]
    else:
        p = torch.softmax(s, dim=-1)
    out = torch.matmul(p, v).to(query.dtype)
    return out.transpose(1, 2).contiguous()


def _apply_sinks(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """``out [B, Q, H, D]`` *= ``Z / (Z + e^sink)`` with ``lse = log Z``.

    In place (one fused Triton pass on CUDA); ``out`` must be owned.
    """
    kernels = _triton_kernels() if out.is_cuda else None
    if kernels is not None and out.stride(-1) == 1:
        return kernels.apply_attention_sinks(out, lse, sinks)
    scale = torch.sigmoid(lse - sinks.float().reshape(1, -1, 1))
    return out.mul_(scale.transpose(1, 2).unsqueeze(-1))


def _masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Explicit-mask attention on CUDA fp16/bf16; returns [B, Q, H, Dv]."""
    bsz, heads, qlen, _ = q.shape
    kv_heads = k.shape[1]
    n_rep = heads // kv_heads
    fold = qlen == 1 and n_rep > 1 and mask.dim() == 4 and mask.shape[1] == 1
    if fold:
        # Decode: the group's query heads become the query axis of one
        # KV head, so every KV head is read once and the [B, 1, 1, K] mask
        # broadcasts over the group.
        q = q.reshape(bsz, kv_heads, n_rep, q.shape[-1])
    else:
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
    if sinks is None:
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale
        )
        lse = None
    else:
        out, lse = _efficient_attention_lse(q, k, v, mask, scale)
    if fold:
        out = out.reshape(bsz, heads, 1, out.shape[-1])
        if lse is not None:
            lse = lse.reshape(bsz, heads, 1)
    out = out.transpose(1, 2)
    if lse is not None:
        out = _apply_sinks(out, lse, sinks)
    return out.contiguous()


def _attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    is_causal: bool | None,
    window: int | None,
    sinks: torch.Tensor | None,
    hf_fallback,
    **kwargs,
):
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    causal = (
        is_causal
        if is_causal is not None
        else getattr(module, "is_causal", True)
    )
    qlen, klen = query.shape[2], key.shape[2]
    unsupported = (
        (dropout > 0.0 and module.training)
        or kwargs.get("output_attentions", False)
        or query.shape[1] % key.shape[1] != 0
    )
    flash = not unsupported and _flash_eligible(query, key, value)

    if qlen == 1 and not unsupported:
        decode = _triton_kernels()
        if decode is not None and decode.gqa_decode_supported(
            query, key, value, attention_mask
        ):
            # Split-KV Triton kernel: GQA group per program, window / key
            # mask / sinks fused, so the KV is streamed exactly once.
            out = decode.gqa_decode_attention(
                query,
                key,
                value,
                scaling,
                window=window,
                sinks=sinks,
                mask=attention_mask,
            )
            return out, None

    if attention_mask is None and flash:
        # FA2 is bottom-right causal; HF/SDPA only skip the mask when the two
        # alignments agree (q_len == 1 or q_len == kv_len).
        if not causal or qlen == 1 or qlen == klen:
            out, lse = flash_attention(
                query, key, value, scaling, causal, window if causal else None
            )
            if sinks is not None:
                out = _apply_sinks(out, lse, sinks)
            return out, None

    if (
        attention_mask is not None
        and not unsupported
        and query.is_cuda
        and query.dtype in _FLASH_DTYPES
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    ):
        return _masked_attention(
            query, key, value, attention_mask, scaling, sinks
        ), None

    if sinks is None and window is None and hf_fallback is not None:
        return hf_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
    return reference_attention(
        query, key, value, attention_mask, scaling, causal, window, sinks
    ), None


def moe_gqa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    sliding_window: int | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
):
    """HF attention function for ``attn_implementation="moe_gqa"``.

    ``attention_mask`` (if any) already contains the window; when it is None
    the causal + ``sliding_window`` pattern is applied in the kernel.
    """
    return _attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout,
        scaling,
        is_causal,
        sliding_window if attention_mask is None else None,
        s_aux,
        _hf_sdpa(),
        **kwargs,
    )


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    """Drop-in for HF ``sdpa`` (same semantics) without ``repeat_kv``."""
    kwargs.pop("sliding_window", None)
    return _attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout,
        scaling,
        is_causal,
        None,
        None,
        _hf_sdpa(),
        **kwargs,
    )


@functools.lru_cache(maxsize=1)
def _hf_sdpa():
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward as hf_sdpa,
    )

    return hf_sdpa


def moe_gqa_mask(*args, local_size: int | None = None, **kwargs):
    """HF ``sdpa_mask`` that may skip the mask even for sliding-window layers.

    Without padding the causal / window pattern is applied by the kernel
    (``sliding_window`` kwarg), so no ``[B, 1, Q, K]`` band is materialised.
    """
    from transformers.masking_utils import sdpa_mask

    return sdpa_mask(*args, **kwargs)


def sdpa_mask_compat(*args, **kwargs):
    """HF ``sdpa_mask``; additive float for models that ``masked_fill`` it.

    GLM-MoE-DSA (transformers 5.12) folds its index mask in with
    ``attention_mask.masked_fill(index_mask, finfo.min)``; on the boolean SDPA
    padding mask that sets the dropped keys to True (visible).  Handing it
    the eager-style additive mask restores the eager semantics.
    """
    from transformers.masking_utils import sdpa_mask

    mask = sdpa_mask(*args, **kwargs)
    config = kwargs.get("config")
    if (
        isinstance(mask, torch.Tensor)
        and mask.dtype == torch.bool
        and getattr(config, "model_type", None) in _ADDITIVE_SDPA_MASK_TYPES
    ):
        mask = _additive(mask, kwargs.get("dtype") or torch.float32)
    return mask


_installed = False


def install() -> None:
    """Register ``moe_gqa`` and route HF ``sdpa`` through this module."""
    global _installed
    if _installed:
        return
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        AttentionMaskInterface,
    )
    from transformers.modeling_utils import (
        ALL_ATTENTION_FUNCTIONS,
        AttentionInterface,
    )

    _hf_sdpa()  # bind the original before rerouting the key
    AttentionInterface.register(MOE_GQA, moe_gqa_attention_forward)
    AttentionMaskInterface.register(MOE_GQA, moe_gqa_mask)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward
    ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_compat
    _installed = True


def select_attn_implementation(model_cls, config, flash_attn_ok: bool) -> str:
    """Attention backend for a model built through the offload engine.

    ``flash_attn_ok``: the ``flash_attn`` package is importable and allowed
    for this architecture.
    """
    model_type = getattr(config, "model_type", "")
    if flash_attn_ok and (
        getattr(model_cls, "_supports_flash_attn", False)
        or getattr(model_cls, "_supports_flash_attn_2", False)
    ):
        return "flash_attention_2"
    if model_type in _EAGER_MODEL_TYPES:
        return "eager"
    if model_type in _MOE_GQA_MODEL_TYPES:
        install()
        return MOE_GQA
    if getattr(model_cls, "_supports_sdpa", False):
        install()
        return "sdpa"
    return "eager"
