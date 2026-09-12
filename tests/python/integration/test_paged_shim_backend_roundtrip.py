"""Prefill + decode through the REAL PagedAttentionBackend, shim vs stock attention.

Discriminates the paged shim <-> backend interaction (KV layout written by
``write_kv`` and read back by ``paged_attention_fwd``) from the serving
engine's metadata. One sequence: prefill 5 tokens, then two decode steps,
each compared against the stock HF attention running on a ``DynamicCache``.

Runs for the OLMoE shim and, as an in-process positive control, for the
Qwen3 shim through the same backend. ``head_dim`` is 128, like the real
OLMoE checkpoint, so on CUDA the native ``paged_attention_v1`` kernel is
exercised; a parametrisation forces the SDPA-from-store fallback instead.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache

from moe_infinity.kernel import paged_attention_ops
from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)

HEAD_DIM = 128
BLOCK_SIZE = 4
NUM_BLOCKS = 8
PREFILL = 5
DECODE_STEPS = 2


def _device_and_dtype():
    if os.environ.get("ROUNDTRIP_DEVICE", "").lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), torch.float32
    return torch.device("cuda"), torch.bfloat16


def _families():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeAttention, OlmoeRotaryEmbedding
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeRotaryEmbedding

    from moe_infinity.models.olmoe_paged_attention import OlmoePagedAttention
    from moe_infinity.models.qwen3_paged_attention import Qwen3PagedAttention

    def olmoe(kv_heads):
        cfg = OlmoeConfig(
            vocab_size=128, hidden_size=4 * HEAD_DIM, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=kv_heads, num_experts=4, num_experts_per_tok=2,
            rms_norm_eps=1e-5, clip_qkv=None, attention_bias=False,
        )
        cfg._attn_implementation = "eager"
        return cfg, OlmoeAttention, OlmoePagedAttention, OlmoeRotaryEmbedding

    def qwen3(kv_heads):
        cfg = Qwen3MoeConfig(
            vocab_size=128, hidden_size=4 * HEAD_DIM, intermediate_size=64, moe_intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=HEAD_DIM,
            num_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-5,
        )
        cfg._attn_implementation = "eager"
        return cfg, Qwen3MoeAttention, Qwen3PagedAttention, Qwen3MoeRotaryEmbedding

    return {"olmoe": olmoe, "qwen3": qwen3}


def _causal_mask(length: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros(1, 1, length, length, device=device, dtype=dtype)
    tri = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
    mask[..., tri] = torch.finfo(dtype).min
    return mask


def _meta(is_prefill: bool, context: int, query: int, device) -> AttentionMetadata:
    kv = context + query
    slots = torch.arange(context, kv, device=device, dtype=torch.int64)
    return AttentionMetadata(
        block_tables=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        max_seq_len=kv,
        num_prefill_tokens=query if is_prefill else 0,
        num_decode_tokens=0 if is_prefill else query,
        slot_mapping=slots,
        is_prefill=is_prefill,
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([query], dtype=torch.int32, device=device),
            query_offsets=torch.tensor([0, query], dtype=torch.int32, device=device),
            context_lengths=torch.tensor([context], dtype=torch.int32, device=device),
            kv_seq_lengths=torch.tensor([kv], dtype=torch.int32, device=device),
        ),
    )


def _tol(dtype):
    # fp32: SDPA-from-store vs eager differ by float rounding only (measured 1.2e-7)
    return {"rtol": 1e-4, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}


@pytest.mark.parametrize("family", ["olmoe", "qwen3"])
def test_two_layers_share_one_backend_without_aliasing(family):
    """Two shim instances (layer_idx 0 and 1) on one backend built with
    num_layers=2: each layer must read back its own KV. A backend sized for
    one layer would alias them and give coherent-but-wrong decode outputs."""
    device, dtype = _device_and_dtype()
    cfg, Stock, Shim, Rotary = _families()[family](2)
    torch.manual_seed(3)
    shims = [Shim(cfg, layer_idx=i).to(device=device, dtype=dtype).eval() for i in range(2)]
    stocks = [Stock(cfg, layer_idx=i).to(device=device, dtype=dtype).eval() for i in range(2)]
    for s, st in zip(shims, stocks):
        st.load_state_dict(s.state_dict())
    rotary = Rotary(cfg).to(device)
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(2, HEAD_DIM, dtype, BLOCK_SIZE), num_gpu_blocks=NUM_BLOCKS, num_layers=2, device=device
    )
    cache = DynamicCache()
    tol = _tol(dtype)
    total = PREFILL + DECODE_STEPS
    hidden = [torch.randn(1, total, cfg.hidden_size, device=device, dtype=dtype) for _ in range(2)]

    def run(is_prefill, start, length):
        pos = torch.arange(start, start + length, device=device).unsqueeze(0)
        for layer in range(2):
            h = hidden[layer][:, start : start + length]
            cos, sin = rotary(h, pos)
            with torch.no_grad():
                ref, _ = stocks[layer](
                    hidden_states=h, position_embeddings=(cos, sin),
                    attention_mask=_causal_mask(length, device, dtype) if is_prefill else None,
                    past_key_values=cache,
                )
                Shim.set_paged_context(backend, _meta(is_prefill, start, length, device))
                try:
                    out, _ = shims[layer](hidden_states=h, position_embeddings=(cos, sin), attention_mask=None)
                finally:
                    Shim.clear_paged_context()
            diff = (out.float() - ref.float()).abs().max().item()
            assert torch.allclose(out, ref, **tol), f"{family} layer {layer} prefill={is_prefill} start={start}: max|diff|={diff:.4f}"

    run(True, 0, PREFILL)
    for step in range(DECODE_STEPS):
        run(False, PREFILL + step, 1)


@pytest.mark.parametrize("family", ["olmoe", "qwen3"])
@pytest.mark.parametrize("kv_heads", [4, 2], ids=["mha", "gqa2"])
@pytest.mark.parametrize("force_fallback", [False, True], ids=["kernel", "sdpa-fallback"])
def test_shim_prefill_and_decode_match_stock_attention(family, kv_heads, force_fallback, monkeypatch):
    device, dtype = _device_and_dtype()
    if force_fallback:
        monkeypatch.setattr(paged_attention_ops, "HAS_PAGED_ATTN", False)
    elif device.type == "cpu":
        pytest.skip("native kernel path needs CUDA; the fallback case covers CPU")

    cfg, Stock, Shim, Rotary = _families()[family](kv_heads)
    torch.manual_seed(0)
    shim = Shim(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    stock = Stock(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    stock.load_state_dict(shim.state_dict())
    rotary = Rotary(cfg).to(device)

    backend = PagedAttentionBackend(
        spec=KVCacheSpec(kv_heads, HEAD_DIM, dtype, BLOCK_SIZE),
        num_gpu_blocks=NUM_BLOCKS,
        num_layers=1,
        device=device,
    )
    cache = DynamicCache()
    tol = _tol(dtype)

    total = PREFILL + DECODE_STEPS
    hidden_all = torch.randn(1, total, cfg.hidden_size, device=device, dtype=dtype)

    # --- prefill -----------------------------------------------------------
    pos = torch.arange(PREFILL, device=device).unsqueeze(0)
    cos, sin = rotary(hidden_all[:, :PREFILL], pos)
    with torch.no_grad():
        ref, _ = stock(
            hidden_states=hidden_all[:, :PREFILL],
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(PREFILL, device, dtype),
            past_key_values=cache,
        )
        Shim.set_paged_context(backend, _meta(True, 0, PREFILL, device))
        try:
            out, _ = shim(
                hidden_states=hidden_all[:, :PREFILL],
                position_embeddings=(cos, sin),
                attention_mask=None,
            )
        finally:
            Shim.clear_paged_context()
    torch.testing.assert_close(out, ref, **tol)

    # --- decode steps ------------------------------------------------------
    for step in range(DECODE_STEPS):
        t = PREFILL + step
        pos = torch.tensor([[t]], device=device)
        h = hidden_all[:, t : t + 1]
        cos, sin = rotary(h, pos)
        with torch.no_grad():
            ref, _ = stock(
                hidden_states=h,
                position_embeddings=(cos, sin),
                attention_mask=None,          # one query over the whole cache
                past_key_values=cache,
            )
            Shim.set_paged_context(backend, _meta(False, t, 1, device))
            try:
                out, _ = shim(hidden_states=h, position_embeddings=(cos, sin), attention_mask=None)
            finally:
                Shim.clear_paged_context()
        diff = (out.float() - ref.float()).abs().max().item()
        assert torch.allclose(out, ref, **tol), (
            f"{family} {kv_heads=} fallback={force_fallback} decode step {step} (pos {t}): "
            f"max|diff|={diff:.4f}"
        )
