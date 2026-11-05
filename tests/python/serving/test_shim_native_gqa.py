"""Paged-shim SDPA uses native GQA instead of expanding K/V (#303)."""

import pytest
import torch
import torch.nn.functional as F

from moe_infinity.runtime import attention_backend as ab
from moe_infinity.runtime.attention_backend import (
    PagedAttentionBackend,
    causal_prefill_sdpa,
)
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.runtime.kv_cache_format import allocate_layered_paged_kv_store


def _repeat_reference(q, k, v, scale):
    ratio = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(ratio, dim=1)
    v = v.repeat_interleave(ratio, dim=1)
    return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)


def _qkv(T=7, H=8, KVH=2, d=16, **kw):
    return (
        torch.randn(1, H, T, d, **kw),
        torch.randn(1, KVH, T, d, **kw),
        torch.randn(1, KVH, T, d, **kw),
    )


@pytest.fixture(autouse=True)
def _reset_gqa_probe(monkeypatch):
    monkeypatch.setattr(ab, "_SDPA_ENABLE_GQA", None)


@pytest.mark.parametrize("kvh", [1, 2, 8])
def test_causal_prefill_sdpa_matches_repeat_reference(kvh):
    torch.manual_seed(0)
    q, k, v = _qkv(KVH=kvh)
    out = causal_prefill_sdpa(q, k, v, 0.25)
    torch.testing.assert_close(out, _repeat_reference(q, k, v, 0.25))


def test_causal_prefill_sdpa_rejects_indivisible_heads():
    q, k, v = _qkv(H=6, KVH=4)
    with pytest.raises(ValueError, match="divisible"):
        causal_prefill_sdpa(q, k, v, 0.25)


def test_native_gqa_call_does_not_expand_kv(monkeypatch):
    torch.manual_seed(1)
    q, k, v = _qkv()
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q_, k_, v_, **kw):
        seen.append((k_.shape[1], kw.get("enable_gqa", False)))
        return real(q_, k_, v_, **kw)

    monkeypatch.setattr(ab, "_use_native_gqa", lambda _q: True)
    monkeypatch.setattr(ab.F, "scaled_dot_product_attention", spy)
    out = causal_prefill_sdpa(q, k, v, 0.25)
    assert seen == [(2, True)]
    assert ab._SDPA_ENABLE_GQA is True
    torch.testing.assert_close(out, _repeat_reference(q, k, v, 0.25))


def test_falls_back_when_torch_lacks_enable_gqa(monkeypatch):
    torch.manual_seed(2)
    q, k, v = _qkv()
    real = F.scaled_dot_product_attention

    def old_sdpa(
        q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        return real(
            q_,
            k_,
            v_,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    monkeypatch.setattr(ab, "_use_native_gqa", lambda _q: True)
    monkeypatch.setattr(ab.F, "scaled_dot_product_attention", old_sdpa)
    out = causal_prefill_sdpa(q, k, v, 0.25)
    assert ab._SDPA_ENABLE_GQA is False
    torch.testing.assert_close(out, _repeat_reference(q, k, v, 0.25))


def test_decode_sdpa_fallback_folds_query_heads():
    torch.manual_seed(3)
    L, H, KVH, d = 6, 8, 2, 8
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=KVH,
            head_dim=d,
            dtype=torch.float32,
            block_size=4,
            format_name="int8_sym",
        ),
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )
    store = allocate_layered_paged_kv_store(
        owner_id="owner",
        format_name="int8_sym",
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=KVH,
        head_dim=d,
        execution_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    backend.bind_store(store, owner_id="owner")
    assert backend.execution_backend == "sdpa_dequant"
    backend.write_chunk(
        layer_idx=0,
        key_chunk=torch.randn(L, KVH, d),
        value_chunk=torch.randn(L, KVH, d),
        slot_mapping=torch.arange(L),
    )
    block_table = torch.tensor([0, 1], dtype=torch.int64)
    query = torch.randn(1, H, d)
    out = backend._decode_sdpa_from_store(
        query, store, block_table[None], torch.tensor([L]), 0.3, 0
    )
    k, v = store.read_prefix(
        layer_idx=0,
        block_table=block_table,
        seq_len=L,
        execution_dtype=torch.float32,
    )
    k = k.permute(1, 0, 2).repeat_interleave(H // KVH, dim=0)[None]
    v = v.permute(1, 0, 2).repeat_interleave(H // KVH, dim=0)[None]
    ref = F.scaled_dot_product_attention(
        query[0][None, :, None, :], k, v, scale=0.3
    )[0, :, 0, :]
    assert out.shape == (1, H, d)
    torch.testing.assert_close(out[0], ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("T,H,KVH,d", [(1024, 64, 4, 128), (333, 16, 2, 256)])
def test_cuda_bf16_prefill_bitwise_matches_repeat_path(T, H, KVH, d):
    torch.manual_seed(4)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    query = torch.randn(T, H, d, **kw)
    key = torch.randn(T, KVH, d, **kw)
    value = torch.randn(T, KVH, d, **kw)
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    out = causal_prefill_sdpa(q, k, v, d**-0.5)
    assert ab._SDPA_ENABLE_GQA is True
    assert torch.equal(out, _repeat_reference(q, k, v, d**-0.5))
