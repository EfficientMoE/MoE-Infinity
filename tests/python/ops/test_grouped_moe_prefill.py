"""Grouped prefill MoE layer (extensions/kernel/grouped_moe_prefill.cu)."""

import importlib

import pytest

from tests.python.ops.conftest import requires_cuda

torch = importlib.import_module("torch")
F = importlib.import_module("torch.nn.functional")

try:
    _STORE = importlib.import_module("moe_infinity._store")
except Exception:  # pragma: no cover - CPU-only CI
    _STORE = None


def _requires_binding():
    if _STORE is None or not hasattr(_STORE, "grouped_moe_prefill"):
        pytest.skip("moe_infinity._store has no grouped_moe_prefill binding")


def _layer(T, E, n, k, H, I, weight_dtype=torch.float32):
    x = torch.randn(T, H, device="cuda").bfloat16()
    ids = torch.randperm(E, device="cuda")[:n]
    sel = torch.rand(T, n, device="cuda").topk(k, dim=1).indices
    mask = torch.zeros(T, E, dtype=torch.bool, device="cuda")
    mask.scatter_(1, ids[sel], True)
    w = (torch.rand(T, E, device="cuda") * mask).to(weight_dtype)
    experts = [
        (
            (torch.randn(I, H, device="cuda") * H**-0.5).bfloat16(),
            (torch.randn(I, H, device="cuda") * H**-0.5).bfloat16(),
            (torch.randn(H, I, device="cuda") * I**-0.5).bfloat16(),
        )
        for _ in range(n)
    ]
    table = torch.tensor(
        [
            [g.data_ptr(), u.data_ptr(), d.data_ptr(), int(ids[j])]
            for j, (g, u, d) in enumerate(experts)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    return x, mask, w, experts, ids, table


def _reference(x, mask, w, experts, ids):
    """fused_moe_ffn_into per expert + fp32 weighted accumulate."""
    ref = torch.zeros(x.size(0), x.size(1), device="cuda")
    xf = x.float()
    for j, (g, u, d) in enumerate(experts):
        e = int(ids[j])
        tok = mask[:, e].nonzero().squeeze(1)
        if tok.numel() == 0:
            continue
        a = F.silu(xf[tok] @ g.float().T) * (xf[tok] @ u.float().T)
        y = (a.bfloat16().float() @ d.float().T).bfloat16().float()
        ref.index_add_(0, tok, y * w[tok, e : e + 1].float())
    return ref


def _assert_close(out, ref):
    scale = ref.abs().max().item()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-2 * scale)


@requires_cuda
@pytest.mark.parametrize(
    "T,E,n,k,H,I",
    [
        (64, 16, 8, 2, 256, 128),
        (300, 64, 64, 4, 512, 256),
        (1000, 128, 100, 8, 1024, 512),
        (777, 8, 8, 2, 264, 520),
        (2000, 64, 64, 8, 512, 256),  # 250 rows / expert: 128 + 64 + 64
        (1152, 64, 64, 8, 512, 256),  # 144 rows / expert: 128 + 16
    ],
)
def test_grouped_moe_prefill_every_config(seed_everything, T, E, n, k, H, I):
    _requires_binding()
    x, mask, w, experts, ids, table = _layer(T, E, n, k, H, I)
    ref = _reference(x, mask, w, experts, ids)
    # 6 gate/up and 8 down configs, incl. the mixed-height ones (-1 = auto)
    for dual_cfg in range(-1, 6):
        for down_cfg in range(-1, 8):
            for combine in (0, 1):
                out = torch.full((T, H), 7.0, device="cuda")
                _STORE.grouped_moe_prefill(
                    x,
                    mask,
                    w,
                    table,
                    I,
                    k,
                    out,
                    True,
                    dual_cfg,
                    down_cfg,
                    combine,
                )
                _assert_close(out, ref)


@requires_cuda
@pytest.mark.parametrize("combine", [0, 1])
@pytest.mark.parametrize("weight_dtype", ["float32", "bfloat16", "float16"])
def test_grouped_moe_prefill_accumulates(
    seed_everything, weight_dtype, combine
):
    """combine=0: fp32 scatter-add epilogue; 1: bf16 rows + per-token sum."""
    _requires_binding()
    dtype = getattr(torch, weight_dtype)
    x, mask, w, experts, ids, table = _layer(500, 32, 32, 4, 512, 256, dtype)
    ref = _reference(x, mask, w, experts, ids)
    base = torch.randn(500, 512, device="cuda")
    out = base.clone()
    _STORE.grouped_moe_prefill(
        x, mask, w, table, 256, 4, out, False, combine=combine
    )
    _assert_close(out - base, ref)


@requires_cuda
@pytest.mark.parametrize("dual_cfg,down_cfg", [(-1, -1), (3, 3), (4, 6)])
def test_grouped_moe_prefill_combine_deterministic(
    seed_everything, dual_cfg, down_cfg
):
    """Combine mode sums every token's rows in a fixed order: bitwise
    reproducible, and equal to the scatter-add path to fp32 rounding."""
    _requires_binding()
    x, mask, w, experts, ids, table = _layer(1500, 64, 48, 6, 1024, 512)
    ref = _reference(x, mask, w, experts, ids)
    outs = []
    for combine in (1, 1, 0):
        out = torch.full((1500, 1024), 5.0, device="cuda")
        _STORE.grouped_moe_prefill(
            x, mask, w, table, 512, 6, out, True, dual_cfg, down_cfg, combine
        )
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    _assert_close(outs[0], ref)
    torch.testing.assert_close(outs[0], outs[2], rtol=1e-5, atol=1e-5)


@requires_cuda
def test_grouped_moe_prefill_large_layer_passes(seed_everything):
    """T x H fp32 output above the per-pass L2 budget: the down GEMM sweeps
    the output in several passes of N panels."""
    _requires_binding()
    x, mask, w, experts, ids, table = _layer(6000, 64, 64, 4, 2048, 256)
    ref = _reference(x, mask, w, experts, ids)
    out = torch.empty(6000, 2048, device="cuda")
    _STORE.grouped_moe_prefill(x, mask, w, table, 256, 4, out, True)
    _assert_close(out, ref)


@requires_cuda
def test_grouped_moe_prefill_gate_matches_python(seed_everything):
    """The C++ hardware gate and moe_infinity.utils.expert_ffn_paths agree."""
    _requires_binding()
    from moe_infinity.utils import expert_ffn_paths as paths

    dev = torch.cuda.current_device()
    assert _STORE.grouped_moe_prefill_pro6000_enabled(dev) == (
        paths.grouped_prefill_path(dev) == paths.GROUPED_PRO6000
    )
    assert _STORE.fused_moe_ffn_pro6000_enabled(dev) == (
        paths.expert_ffn_path(dev) == paths.CUTLASS_PRO6000
    )


@requires_cuda
@pytest.mark.parametrize("zero_out", [True, False])
def test_grouped_moe_prefill_per_expert_fallback(seed_everything, zero_out):
    """The path other GPUs run (one generic expert FFN per expert)."""
    _requires_binding()
    x, mask, w, experts, ids, table = _layer(300, 64, 64, 4, 512, 256)
    ref = _reference(x, mask, w, experts, ids)
    base = torch.randn(300, 512, device="cuda")
    out = base.clone()
    _STORE.grouped_moe_prefill_per_expert(x, mask, w, table, 256, out, zero_out)
    _assert_close(out if zero_out else out - base, ref)
