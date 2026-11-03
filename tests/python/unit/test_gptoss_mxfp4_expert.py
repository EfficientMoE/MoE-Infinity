"""GPT-OSS offloaded-expert MXFP4 kernels (#311).

`moe_infinity._v4_fp4.gptoss_mxfp4_expert` runs the same entry point as
MoEMLP::ForwardHelper for type-6 experts (fused W4A16 MXFP4 GEMMs with the
bias / SwiGLU-OAI epilogue, and the dense-down path for large token counts).
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4 += [-v for v in FP4]


def _ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        import moe_infinity._v4_fp4 as ext
    except Exception:
        pytest.skip("native MXFP4 extension not built")
    if not hasattr(ext, "gptoss_mxfp4_expert"):
        pytest.skip("extension built without gptoss_mxfp4_expert")
    return ext


def _dequant_ref(blocks, scales):
    lut = torch.tensor(FP4, device=blocks.device)
    vals = torch.stack(
        [lut[(blocks & 15).long()], lut[(blocks >> 4).long()]], -1
    ).reshape(blocks.shape[0], -1, 32)
    exp = (scales.int() - 127).clamp(-126, 127).float()
    return (vals * torch.exp2(exp)[..., None]).reshape(blocks.shape[0], -1)


def _swiglu_oai(gate_up, alpha=1.702, limit=7.0):
    gate = gate_up[..., ::2].clamp(max=limit)
    up = gate_up[..., 1::2].clamp(-limit, limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


def _expert(H, I, scale_lo=118, scale_hi=124, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, device="cuda", dtype=torch.uint8, generator=g
        )

    def bias(n):
        return (torch.randn(n, device="cuda", generator=g) * 0.5).to(
            torch.bfloat16
        )

    return (
        u8(2 * I, H // 2),
        u8(2 * I, H // 32, lo=scale_lo, hi=scale_hi),
        bias(2 * I),
        u8(H, I // 2),
        u8(H, I // 32, lo=scale_lo, hi=scale_hi),
        bias(H),
    )


def _reference(x, w, bf16_steps):
    """HF GptOssExperts math. bf16_steps rounds like the BF16 module (and the
    previous dequant + cuBLAS path); otherwise everything stays FP32."""
    gq, gs, gb, dq, ds, db = w
    wgu, wd = _dequant_ref(gq, gs), _dequant_ref(dq, ds)
    if bf16_steps:
        gate_up = (x @ wgu.to(torch.bfloat16).T) + gb
        act = _swiglu_oai(gate_up)
        return act @ wd.to(torch.bfloat16).T + db
    gate_up = x.float() @ wgu.T + gb.float()
    act = _swiglu_oai(gate_up).to(torch.bfloat16).float()
    return act @ wd.T + db.float()


@pytest.mark.parametrize("H,I", [(64, 96), (2880, 2880)])
@pytest.mark.parametrize(
    "M", [1, 2, 3, 8, 13, 17, 32, 33, 64, 200, 288, 320, 700]
)
def test_matches_reference(H, I, M):
    ext = _ext()
    w = _expert(H, I, seed=M)
    x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
    out = ext.gptoss_mxfp4_expert(x, *w)
    torch.cuda.synchronize()
    assert out.shape == (M, H) and out.dtype == torch.bfloat16
    ref = _reference(x, w, bf16_steps=False)
    legacy = _reference(x, w, bf16_steps=True).float()
    scale = ref.abs().max().item()
    # Kernel vs FP32 math: only the BF16 output rounding plus FP32
    # accumulation-order noise.
    assert (out.float() - ref).abs().max().item() <= 1e-2 * scale
    # Kernel vs the BF16 module / previous path: BF16 tolerance.
    assert (out.float() - legacy).abs().max().item() <= 3e-2 * scale


def test_matches_hf_gpt_oss_experts():
    ext = _ext()
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import (
            GptOssConfig,
        )
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssExperts,
        )
    except Exception:
        pytest.skip("transformers GPT-OSS classes unavailable")
    H, I, E, M = 128, 64, 2, 11
    cfg = GptOssConfig(hidden_size=H, intermediate_size=I, num_local_experts=E)
    hf = GptOssExperts(cfg).to("cuda", torch.bfloat16)
    experts = [_expert(H, I, seed=10 + e) for e in range(E)]
    with torch.no_grad():
        for e, (gq, gs, gb, dq, ds, db) in enumerate(experts):
            hf.gate_up_proj[e].copy_(_dequant_ref(gq, gs).T)
            hf.gate_up_proj_bias[e].copy_(gb)
            hf.down_proj[e].copy_(_dequant_ref(dq, ds).T)
            hf.down_proj_bias[e].copy_(db)
    x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
    for e in range(E):
        idx = torch.full((M, 1), e, device="cuda", dtype=torch.long)
        ones = torch.ones(M, 1, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            expected = hf(x, router_indices=idx, routing_weights=ones)
        out = ext.gptoss_mxfp4_expert(x, *experts[e])
        scale = expected.float().abs().max().item()
        assert (out.float() - expected.float()).abs().max() <= 3e-2 * scale


def test_scale_edge_cases():
    """E8M0 0 / 255 clamp like the legacy dequant; +-0 and max codes."""
    ext = _ext()
    H, I, M = 64, 64, 5
    gq, gs, gb, dq, ds, db = _expert(H, I, seed=3)
    gs[0, 0] = 0  # clamps to 2^-126
    gs[1, 1], gq[1, 16:32] = 255, 0  # clamps to 2^127 (zero codes: finite)
    ds[2, 0] = 140  # large but finite
    gq[:, :4] = 0x80  # +0 / -0 codes
    gq[:, 4:8] = 0x77  # +6 / +6
    dq[:, :2] = 0xFF  # -6 / -6
    w = (gq, gs, gb, dq, ds, db)
    # Seeded: with the 2^13 down scale, one BF16 ulp of act (accumulation
    # order) moves an output by ~1 % of the max, so unseeded x made this flaky.
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(M, H, device="cuda", generator=g).to(torch.bfloat16)
    out = ext.gptoss_mxfp4_expert(x, *w)
    ref = _reference(x, w, bf16_steps=False)
    assert torch.isfinite(out).all() and torch.isfinite(ref).all()
    scale = ref.abs().max().item()
    assert (out.float() - ref).abs().max().item() <= 1e-2 * scale


def test_dequant_matches_hf_layout():
    """mxfp4_dequant (fallback path) == transformers' MXFP4 conversion."""
    ext = _ext()
    try:
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    except Exception:
        pytest.skip("transformers MXFP4 integration unavailable")
    rows, k = 96, 256
    blocks = torch.randint(
        0, 256, (1, rows, k // 32, 16), device="cuda", dtype=torch.uint8
    )
    scales = torch.randint(
        100, 150, (1, rows, k // 32), device="cuda", dtype=torch.uint8
    )
    hf = convert_moe_packed_tensors(blocks, scales)[0].T  # [rows, k]
    ours = ext.mxfp4_dequant(blocks.reshape(rows, k // 2), scales[0])
    assert torch.equal(
        ours.view(torch.int16), hf.contiguous().view(torch.int16)
    )
