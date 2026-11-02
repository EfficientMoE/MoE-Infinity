"""GQA-native SDPA attention (runtime/hf_attention.py) vs HF eager.

CPU tests exercise the math reference / fallback paths (CI); the CUDA tests
exercise the FlashAttention-2 / masked fast paths on tiny random-weight
configs of every family that used to fall back to eager (#301, #302).
"""

import contextlib
import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from moe_infinity.runtime import hf_attention  # noqa: E402
from moe_infinity.runtime.hf_attention import (  # noqa: E402
    MOE_GQA,
    reference_attention,
    select_attn_implementation,
)

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


# --------------------------------------------------------------------------
# tiny configs
# --------------------------------------------------------------------------


def _gpt_oss(window=8):
    from transformers import GptOssConfig, GptOssForCausalLM

    cfg = GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=2,
        num_experts_per_tok=2,
        sliding_window=window,
        layer_types=["sliding_attention", "full_attention"] * 2,
        max_position_embeddings=1024,
    )
    return GptOssForCausalLM, cfg


def _mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM

    cfg = MixtralConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
    )
    return MixtralForCausalLM, cfg


def _minimax_m3():
    from transformers import MiniMaxM3VLForCausalLM, MiniMaxM3VLTextConfig

    cfg = MiniMaxM3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=2,
        num_experts_per_tok=2,
        index_n_heads=2,
        index_head_dim=16,
        index_block_size=4,
        index_topk_blocks=2,
        index_local_blocks=1,
        layer_types=["full_attention", "minimax_m3_sparse"],
        max_position_embeddings=1024,
    )
    return MiniMaxM3VLForCausalLM, cfg


def _glm_dsa():
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    cfg = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=2,
        num_experts=2,
        num_experts_per_tok=2,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=24,
        head_dim=8,
        index_topk=8,
        index_head_dim=16,
        index_n_heads=2,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
    )
    return GlmMoeDsaForCausalLM, cfg


FAMILIES = {
    "gpt_oss": _gpt_oss,
    "mixtral": _mixtral,
    "minimax_m3": _minimax_m3,
    "glm_moe_dsa": _glm_dsa,
}


def _family(family, **kw):
    try:
        return FAMILIES[family](**kw)
    except (ImportError, TypeError, ValueError) as e:  # pragma: no cover
        pytest.skip(f"{family} unavailable in this transformers: {e}")


def _build(family, **kw):
    """fp32 CPU (eager, selected-backend) twins with identical weights."""
    cls, cfg = _family(family, **kw)
    impl = select_attn_implementation(cls, cfg, False)
    torch.manual_seed(0)
    eager = cls._from_config(copy.deepcopy(cfg), attn_implementation="eager")
    with torch.no_grad():
        for name, p in eager.named_parameters():
            if name.endswith("sinks"):
                p.copy_(torch.randn_like(p) * 2.0)
    fast = cls._from_config(copy.deepcopy(cfg), attn_implementation=impl)
    fast.load_state_dict(eager.state_dict())
    return eager.eval(), fast.eval(), impl


@contextlib.contextmanager
def _hf_sdpa_kernel():
    """Route the ``sdpa`` key back to HF's own implementation."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["sdpa"] = hf_attention._hf_sdpa()
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = hf_attention.sdpa_attention_forward


# MiniMax-M3's indexer scores the hidden states of padded positions, whose
# values differ between backends (eager: uniform softmax over a fully masked
# row, SDPA: zeros), so HF eager and HF sdpa already disagree under padding.
# There the reference is the same model on HF's own sdpa kernel.
_PAD_REFERENCE_IS_HF_SDPA = {"minimax_m3"}


def _reference(family, pad, eager, fast):
    if pad and family in _PAD_REFERENCE_IS_HF_SDPA:
        return copy.deepcopy(fast), _hf_sdpa_kernel
    return copy.deepcopy(eager), contextlib.nullcontext


def _run(model, ids, mask, decode_ids):
    """Prefill + teacher-forced decode with a DynamicCache; all logits."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    outs = []
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )
        outs.append(out.logits.float())
        for step in range(decode_ids.shape[1]):
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            out = model(
                input_ids=decode_ids[:, step : step + 1],
                attention_mask=mask,
                past_key_values=out.past_key_values,
                use_cache=True,
            )
            outs.append(out.logits.float())
    return outs


def _inputs(device, batch, seq, pad):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 128, (batch, seq), generator=g).to(device)
    dec = torch.randint(0, 128, (batch, 4), generator=g).to(device)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    if pad:
        mask[0, :3] = 0  # left padding on the first sequence
    return ids, mask, dec


def _errors(ref, got, pad):
    """Max |diff| per step; padded query rows are undefined in every backend."""
    errs = []
    for step, (r, g) in enumerate(zip(ref, got)):
        if pad and step == 0:
            r, g = r.clone(), g.clone()
            r[0, :3] = 0
            g[0, :3] = 0
        errs.append((r - g).abs().max().item())
    return errs


def _compare_fp32(family, device, seq, pad, **kw):
    eager, fast, impl = _build(family, **kw)
    ref_model, ctx = _reference(family, pad, eager, fast)
    ids, mask, dec = _inputs(device, 2, seq, pad)
    with ctx():
        ref = _run(ref_model.to(device), ids, mask, dec)
    got = _run(fast.to(device), ids, mask, dec)
    errs = _errors(ref, got, pad)
    assert max(errs) <= 1e-4, f"{family}/{impl}: per-step max |diff| {errs}"


def _compare_bf16(family, seq, pad, **kw):
    """bf16 fast path vs fp32 reference, bounded by bf16 reference's error."""
    eager, fast, impl = _build(family, **kw)
    ref_model, ctx = _reference(family, pad, eager, fast)
    ids, mask, dec = _inputs("cuda", 2, seq, pad)
    with ctx():
        truth = _run(copy.deepcopy(ref_model).cuda(), ids, mask, dec)
        base = _run(ref_model.cuda().bfloat16(), ids, mask, dec)
    got = _run(fast.cuda().bfloat16(), ids, mask, dec)
    e_base = _errors(truth, base, pad)
    e_got = _errors(truth, got, pad)
    assert (
        max(e_got) <= 1.25 * max(e_base) + 5e-3
    ), f"{family}/{impl}: bf16 error {e_got} vs bf16 reference {e_base}"
    assert sum(e_got) <= 1.25 * sum(e_base) + 5e-3 * len(e_got)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_select_attn_implementation():
    cls, cfg = _family("gpt_oss")
    assert select_attn_implementation(cls, cfg, False) == MOE_GQA
    assert select_attn_implementation(cls, cfg, True) == "flash_attention_2"
    cls, cfg = _family("mixtral")
    assert select_attn_implementation(cls, cfg, False) == "sdpa"
    cls, cfg = _family("glm_moe_dsa")
    # GLM-DSA has no FA2 support: never eager, never flash_attention_2.
    assert select_attn_implementation(cls, cfg, True) == "sdpa"

    class _NoSdpa:
        _supports_sdpa = False

    assert (
        select_attn_implementation(_NoSdpa, type("C", (), {})(), False)
        == "eager"
    )


def test_install_routes_sdpa_key():
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    hf_attention.install()
    assert (
        ALL_ATTENTION_FUNCTIONS["sdpa"] is hf_attention.sdpa_attention_forward
    )
    assert (
        ALL_ATTENTION_FUNCTIONS[MOE_GQA]
        is hf_attention.moe_gqa_attention_forward
    )
    assert ALL_MASK_ATTENTION_FUNCTIONS[MOE_GQA] is hf_attention.moe_gqa_mask


# --------------------------------------------------------------------------
# kernel-level math
# --------------------------------------------------------------------------


def _naive(q, k, v, causal, window, sinks):
    n = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n, 1).float()
    v = v.repeat_interleave(n, 1).float()
    s = q.float() @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    ql, kl = s.shape[-2:]
    qi = torch.arange(ql)[:, None] + kl - ql
    ki = torch.arange(kl)[None]
    keep = torch.ones(ql, kl, dtype=torch.bool)
    if causal:
        keep &= ki <= qi
    if window:
        keep &= ki > qi - window
    s = s.masked_fill(~keep.to(s.device), float("-inf"))
    if sinks is not None:
        col = sinks.float().view(1, -1, 1, 1).expand(*s.shape[:-1], 1)
        p = torch.cat([s, col], -1).softmax(-1)[..., :-1]
    else:
        p = s.softmax(-1)
    return (p @ v).transpose(1, 2)


@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("qlen", [1, 17])
def test_reference_attention(window, sinks, qlen):
    torch.manual_seed(0)
    q = torch.randn(2, 8, qlen, 16)
    k = torch.randn(2, 2, 17, 16)
    v = torch.randn(2, 2, 17, 16)
    snk = torch.randn(8) if sinks else None
    got = reference_attention(q, k, v, None, 16**-0.5, True, window, snk)
    torch.testing.assert_close(got, _naive(q, k, v, True, window, snk))


def test_sink_lse_identity():
    """softmax([s, sink])[:-1] @ V == (softmax(s) @ V) * sigmoid(lse - sink)."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 9, 8)
    k = torch.randn(1, 4, 9, 8)
    v = torch.randn(1, 4, 9, 8)
    snk = torch.randn(4)
    s = q @ k.transpose(-1, -2) * 8**-0.5
    lse = torch.logsumexp(s, -1)
    out = (s.softmax(-1) @ v).transpose(1, 2)
    got = hf_attention._apply_sinks(out, lse, snk)
    torch.testing.assert_close(got, _naive(q, k, v, False, None, snk))


@needs_cuda
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("qlen,klen", [(300, 300), (1, 300), (1, 2000)])
def test_flash_path_cuda(window, sinks, qlen, klen):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(2, 8, qlen, 64, device=dev, dtype=dt)
    k = torch.randn(2, 1, klen, 64, device=dev, dtype=dt)
    v = torch.randn(2, 1, klen, 64, device=dev, dtype=dt)
    snk = torch.randn(8, device=dev, dtype=dt) if sinks else None
    mod = torch.nn.Module()
    out, _ = hf_attention.moe_gqa_attention_forward(
        mod, q, k, v, None, scaling=64**-0.5, sliding_window=window, s_aux=snk
    )
    torch.testing.assert_close(
        out.float(),
        _naive(q, k, v, True, window, snk),
        atol=2e-2,
        rtol=2e-2,
    )


@needs_cuda
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("qlen", [1, 37])
def test_masked_path_cuda(sinks, qlen):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    klen = 301
    q = torch.randn(2, 8, qlen, 64, device=dev, dtype=dt)
    k = torch.randn(2, 2, klen, 64, device=dev, dtype=dt)
    v = torch.randn(2, 2, klen, 64, device=dev, dtype=dt)
    keep = torch.rand(2, 1, qlen, klen, device=dev) > 0.3
    keep[..., -1] = True
    snk = torch.randn(8, device=dev, dtype=dt) if sinks else None
    out, _ = hf_attention.moe_gqa_attention_forward(
        torch.nn.Module(), q, k, v, keep, scaling=64**-0.5, s_aux=snk
    )
    ref = reference_attention(
        q.float(), k.float(), v.float(), keep, 64**-0.5, True, None, snk
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------
# model parity vs eager (prefill + teacher-forced decode)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("pad", [False, True])
def test_model_parity_cpu_fp32(family, pad):
    _compare_fp32(family, "cpu", seq=20, pad=pad)


def _count_paths(monkeypatch):
    calls = {"flash": 0, "masked": 0, "decode": 0}
    decode = hf_attention._triton_kernels()
    if decode is not None:

        class _Spy:
            gqa_decode_supported = staticmethod(decode.gqa_decode_supported)
            apply_attention_sinks = staticmethod(decode.apply_attention_sinks)

            @staticmethod
            def gqa_decode_attention(*a, **k):
                calls["decode"] += 1
                return decode.gqa_decode_attention(*a, **k)

        monkeypatch.setattr(hf_attention, "_triton_kernels", lambda: _Spy)

    def spy(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)

        return wrapped

    monkeypatch.setattr(
        hf_attention,
        "flash_attention",
        spy("flash", hf_attention.flash_attention),
    )
    monkeypatch.setattr(
        hf_attention,
        "_masked_attention",
        spy("masked", hf_attention._masked_attention),
    )
    return calls


@needs_cuda
@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("pad", [False, True])
def test_model_parity_cuda_bf16(family, pad, monkeypatch):
    calls = _count_paths(monkeypatch)
    _compare_bf16(family, seq=40, pad=pad)
    # GLM-DSA always passes its index mask; everything else runs FA2 unpadded.
    if family == "glm_moe_dsa" or pad:
        assert calls["masked"] > 0
    else:
        assert calls["flash"] > 0
    assert calls["decode"] > 0


@needs_cuda
def test_gpt_oss_window_128_cuda(monkeypatch):
    """Real GPT-OSS window (128), prompt and decode beyond the window."""
    calls = _count_paths(monkeypatch)
    _compare_bf16("gpt_oss", seq=200, pad=False, window=128)
    assert calls["flash"] > 0 and calls["masked"] == 0 and calls["decode"] > 0


# --------------------------------------------------------------------------
# split-KV decode kernel
# --------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize(
    "B,H,KVH,L,D,DV",
    [
        (2, 8, 2, 300, 64, 64),
        (1, 64, 4, 5000, 128, 128),
        (2, 16, 16, 777, 256, 256),
        (3, 4, 4, 33, 24, 24),
        (2, 8, 1, 4099, 64, 32),
    ],
)
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("mask", [None, "bool", "additive", "bool_bcast"])
def test_gqa_decode_kernel(B, H, KVH, L, D, DV, window, sinks, mask):
    from moe_infinity.kernel.gqa_attention import (
        gqa_decode_attention,
        gqa_decode_supported,
    )

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(B, H, 1, D, device=dev, dtype=dt)
    # Strided cache views (as DynamicCache slices / transposes produce).
    k = torch.randn(B, L + 5, KVH, D, device=dev, dtype=dt)[:, 5:].transpose(
        1, 2
    )
    v = torch.randn(B, KVH, L, DV + 8, device=dev, dtype=dt)[..., :DV]
    snk = torch.randn(H, device=dev, dtype=dt) if sinks else None
    m = None
    if mask is not None:
        keep = torch.rand(B, 1, 1, L, device=dev) > 0.4
        keep[..., -1] = True
        if mask == "bool_bcast":
            keep = keep[:1].expand(B, 1, 1, L)
        m = keep
        if mask == "additive":
            m = torch.zeros(B, 1, 1, L, device=dev, dtype=dt).masked_fill(
                ~keep, torch.finfo(dt).min
            )
    assert gqa_decode_supported(q, k, v, m)
    out, lse = gqa_decode_attention(
        q, k, v, D**-0.5, window=window, sinks=snk, mask=m, return_lse=True
    )
    ref_mask = m
    if m is not None and window is not None:  # kernel applies both
        band = torch.arange(L, device=dev) >= L - window
        ref_mask = (
            (keep & band)
            if mask != "additive"
            else m.masked_fill(~band, torch.finfo(dt).min)
        )
    ref = reference_attention(
        q.float(), k.float(), v.float(), ref_mask, D**-0.5, True, window, snk
    )
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    assert out.shape == (B, 1, H, DV) and lse.shape == (B, H)


@needs_cuda
@pytest.mark.parametrize("D", [24, 64, 128, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_apply_attention_sinks_kernel(D, strided):
    from moe_infinity.kernel.gqa_attention import apply_attention_sinks

    torch.manual_seed(0)
    B, Q, H = 2, 37, 8
    base = torch.randn(B, H, Q, D, device="cuda", dtype=torch.bfloat16)
    out = base.transpose(1, 2) if strided else base.transpose(1, 2).contiguous()
    lse = torch.randn(B, H, Q, device="cuda")
    snk = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    want = out.float() * torch.sigmoid(
        lse - snk.float()[None, :, None]
    ).transpose(1, 2).unsqueeze(-1)
    got = apply_attention_sinks(out, lse, snk)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=1e-2)


@needs_cuda
def test_gqa_decode_kernel_fully_masked_row():
    from moe_infinity.kernel.gqa_attention import gqa_decode_attention

    q = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 100, 64, device="cuda", dtype=torch.bfloat16)
    keep = torch.zeros(1, 1, 1, 100, device="cuda", dtype=torch.bool)
    out = gqa_decode_attention(q, k, k, 0.125, mask=keep)
    assert torch.isfinite(out).all() and out.abs().max() == 0
