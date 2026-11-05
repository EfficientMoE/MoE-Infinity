"""Native FP8 expert kernels (#310) vs. de-quantising references.

M <= 8 runs the weight-only GEMV: exact weight de-quantisation and BF16
activations, so it is compared against the FP32 de-quantised reference.
M > 8 runs the CUTLASS blockwise W8A8 GEMM: it is compared against an
emulation that quantises the activations exactly like the kernel (per token,
per 128 channels, e4m3 with an fp32 amax/448 scale).
"""

import pytest
import torch
import torch.nn.functional as F

from tests.python.ops.conftest import requires_cuda, requires_fp8

E4M3_MAX = 448.0


def _store():
    try:
        import moe_infinity._store as store
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"moe_infinity._store unavailable: {exc}")
    if not hasattr(store, "fp8_moe_ffn_sm120_pro6000_into"):
        pytest.skip("_store built without the native FP8 expert kernels")
    return store


def _skip_unless_native(store, hidden, inter):
    if not store.fp8_moe_sm120_pro6000_supported(hidden, inter):
        pytest.skip("native FP8 experts are gated to sm_120 + RTX PRO 6000")


def _weight(n, k, gen):
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    s = (
        torch.rand(n // 128, k // 128, generator=gen, device="cuda") * 0.01
        + 1e-3
    )
    return w, s


def _dequant(w, s):
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def _quant_act(x):
    """Per-token, per-128-channel e4m3 quantisation as the kernel does it."""
    m, k = x.shape
    g = x.float().view(m, k // 128, 128)
    amax = g.abs().amax(-1, keepdim=True)
    nz = amax > 0
    inv = torch.where(nz, E4M3_MAX / amax, torch.ones_like(amax))
    scale = torch.where(nz, amax / E4M3_MAX, torch.ones_like(amax))
    q = (g * inv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).view(m, k)


def _reference(x, gate, up, down, w8a8):
    g, u, d = (_dequant(*t) for t in (gate, up, down))
    if not w8a8:
        h = F.silu(x.float() @ g.T) * (x.float() @ u.T)
        return h @ d.T
    xq = _quant_act(x)
    gu = torch.cat([xq @ g.T, xq @ u.T], dim=1).bfloat16().float()
    i = g.shape[0]
    h = F.silu(gu[:, :i]) * gu[:, i:]
    return _quant_act(h) @ d.T


def _rel_err(out, ref):
    """Relative RMS error (single e4m3 rounding-boundary flips in the
    emulated activations make a max-norm metric noisy)."""
    return ((out.float() - ref).norm() / ref.norm()).item()


@requires_cuda
@requires_fp8
@pytest.mark.parametrize("hidden,inter", [(256, 128), (1024, 512)])
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 9, 33, 300])
def test_fp8_native_expert_matches_reference(hidden, inter, m):
    store = _store()
    _skip_unless_native(store, hidden, inter)
    gen = torch.Generator(device="cuda").manual_seed(1000 + m)
    gate = _weight(inter, hidden, gen)
    up = _weight(inter, hidden, gen)
    down = _weight(hidden, inter, gen)
    x = torch.randn(m, hidden, generator=gen, device="cuda").bfloat16()
    out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)

    store.fp8_moe_ffn_sm120_pro6000_into(x, *gate, *up, *down, out)
    torch.cuda.synchronize()

    gemv = m <= 8
    ref = _reference(x, gate, up, down, w8a8=not gemv)
    assert torch.isfinite(out).all()
    assert _rel_err(out, ref) < (5e-3 if gemv else 1e-2)
    if not gemv:
        # W8A8 vs. the BF16-activation reference: bounded activation rounding.
        exact = _reference(x, gate, up, down, w8a8=False)
        assert _rel_err(out, exact) < 0.08


@requires_cuda
@requires_fp8
def test_fp8_native_expert_uint8_weights_and_glm_shape():
    store = _store()
    hidden, inter = 6144, 2048
    _skip_unless_native(store, hidden, inter)
    gen = torch.Generator(device="cuda").manual_seed(7)
    gate, up, down = (
        _weight(n, k, gen)
        for n, k in ((inter, hidden), (inter, hidden), (hidden, inter))
    )
    for m in (1, 16):
        x = torch.randn(m, hidden, generator=gen, device="cuda").bfloat16()
        out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
        store.fp8_moe_ffn_sm120_pro6000_into(
            x,
            gate[0].view(torch.uint8),
            gate[1],
            up[0].view(torch.uint8),
            up[1],
            down[0].view(torch.uint8),
            down[1],
            out,
        )
        torch.cuda.synchronize()
        ref = _reference(x, gate, up, down, w8a8=m > 8)
        assert _rel_err(out, ref) < 1e-2


@requires_cuda
@requires_fp8
def test_fp8_native_expert_allocates_no_weight_copy():
    store = _store()
    hidden, inter = 4096, 2048
    _skip_unless_native(store, hidden, inter)
    gen = torch.Generator(device="cuda").manual_seed(3)
    gate, up, down = (
        _weight(n, k, gen)
        for n, k in ((inter, hidden), (inter, hidden), (hidden, inter))
    )
    fp8_weight_bytes = 3 * hidden * inter
    for m in (1, 8, 64):
        x = torch.randn(m, hidden, generator=gen, device="cuda").bfloat16()
        out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
        store.fp8_moe_ffn_sm120_pro6000_into(x, *gate, *up, *down, out)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        store.fp8_moe_ffn_sm120_pro6000_into(x, *gate, *up, *down, out)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        # Only activation-sized workspace; a BF16 weight copy would be 2x this.
        assert peak < fp8_weight_bytes // 8, (m, peak)
