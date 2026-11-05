"""CUTLASS expert FFN epilogue: every activation kind vs the HF reference."""

import importlib

import pytest
import torch

from moe_infinity.runtime.expert_activation import (
    CLAMPED_SILU,
    SILU,
    SWIGLU_OAI,
    ExpertActivation,
    reference_gated_act,
)
from tests.python.ops.conftest import requires_cuda


def _store():
    try:
        store = importlib.import_module("moe_infinity._store")
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"moe_infinity._store not importable: {exc}")
    if not hasattr(store, "fused_moe_ffn_into"):
        pytest.skip("moe_infinity._store has no fused_moe_ffn_into binding")
    return store


ACTS = {
    "silu": ExpertActivation(),
    "glm5_next_clamped_silu": ExpertActivation(CLAMPED_SILU, 1.0, 10.0),
    "minimax_m3_swiglu_oai": ExpertActivation(SWIGLU_OAI, 1.702, 7.0),
}

# (tokens, hidden, intermediate): GLM-5.3-Flash (4096, 2048) runs the small-K
# tiles, MiniMax-M3 (6144, 3072) the large-K tiles (I >= 3072).
SHAPES = {
    "small": (16, 256, 512),
    "glm5_next": (33, 4096, 2048),
    "minimax_m3": (7, 6144, 3072),
}


def _run(store, x, g, u, d, act):
    gb = torch.empty(x.size(0), g.size(0), dtype=x.dtype, device=x.device)
    fb = torch.empty_like(gb)
    out = torch.empty_like(x)
    store.fused_moe_ffn_into(
        x, g, u, d, gb, fb, out, None, act.kind, act.alpha, act.limit
    )
    return out, fb


def _reference(x, g, u, d, act):
    # HF eager order: bf16 projections, activation, bf16 down projection.
    gate = x @ g.t()
    up = x @ u.t()
    fused = reference_gated_act(gate.float(), up.float(), act).to(x.dtype)
    return fused @ d.t(), fused


def _weights(T, H, I, gate_up_std):
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    # std(x @ w.T) = gate_up_std: |gate|, |up| well beyond the 7 / 10 limits.
    s = gate_up_std / H**0.5
    g = (torch.randn(I, H, device="cuda") * s).to(torch.bfloat16)
    u = (torch.randn(I, H, device="cuda") * s).to(torch.bfloat16)
    d = (torch.randn(H, I, device="cuda") / I**0.5).to(torch.bfloat16)
    return x, g, u, d


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@requires_cuda
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("act_name", list(ACTS))
def test_fused_moe_ffn_activation_matches_reference(
    seed_everything, shape, act_name
):
    store = _store()
    act = ACTS[act_name]
    T, H, I = SHAPES[shape]
    x, g, u, d = _weights(T, H, I, gate_up_std=15.0)

    out, fused = _run(store, x, g, u, d, act)
    ref_out, ref_fused = _reference(x, g, u, d, act)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    # bf16 intermediate rounding differs (epilogue reads the fp32 up
    # accumulator); everything else is the same math.
    assert _rel(fused, ref_fused) < 2e-2
    assert _rel(out, ref_out) < 2e-2

    if act.kind != SILU:
        # The clamp is active at these magnitudes: plain SiLU must be far off.
        assert _rel(out, _reference(x, g, u, d, ACTS["silu"])[0]) > 0.2


@requires_cuda
def test_fused_moe_ffn_default_is_silu(seed_everything):
    store = _store()
    x, g, u, d = _weights(8, 256, 512, gate_up_std=3.0)
    gb = torch.empty(8, 512, dtype=x.dtype, device="cuda")
    fb = torch.empty_like(gb)
    out = torch.empty_like(x)
    store.fused_moe_ffn_into(x, g, u, d, gb, fb, out, None)
    explicit, _ = _run(store, x, g, u, d, ACTS["silu"])
    torch.testing.assert_close(out, explicit, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize(
    "kind,alpha,limit",
    [(CLAMPED_SILU, 1.0, 0.0), (SWIGLU_OAI, 0.0, 7.0), (SWIGLU_OAI, 1.7, -1.0)],
)
def test_fused_moe_ffn_rejects_invalid_descriptor(kind, alpha, limit):
    store = _store()
    x, g, u, d = _weights(4, 256, 512, gate_up_std=1.0)
    with pytest.raises(RuntimeError):
        _run(store, x, g, u, d, ExpertActivation(kind, alpha, limit))
