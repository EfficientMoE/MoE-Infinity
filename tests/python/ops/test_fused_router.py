"""Fused router (#315) vs the routers it replaces.

Expert selection must be identical to the current torch routers:
* integer-valued ("tie-heavy") inputs: logits are exact in any summation
  order, so every token -- ties included -- must select the same experts;
* random inputs: identical selection for every token whose reference gap at
  the selection boundary exceeds fp32 / bf16 accumulation rounding.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.python.ops.conftest import requires_cuda, requires_triton

pytestmark = [requires_cuda, requires_triton]


def _has_cuda_ext():
    try:
        from moe_infinity import _router  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(
    params=[
        "triton",
        pytest.param(
            "prefill",
            marks=pytest.mark.skipif(
                not _has_cuda_ext(), reason="moe_infinity._router not built"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not _has_cuda_ext(), reason="moe_infinity._router not built"
            ),
        ),
    ]
)
def route(request):
    from moe_infinity.kernel.router import fused_route

    def _route(*args, **kwargs):
        return fused_route(*args, backend=request.param, **kwargs)

    return _route


def _softmax_ref(x, w, k, renorm=True):
    logits = F.linear(x, w)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    wts, idx = torch.topk(probs, k, dim=-1)
    if renorm:
        wts = wts / wts.sum(dim=-1, keepdim=True)
    return idx, wts.to(x.dtype), probs, logits


def _grouped_ref(x, w, bias, k, n_group, topk_group, renorm, scale, fn):
    """GLM-MoE-DSA / DeepSeek-V3 noaux_tc (moe_store glm_moe_dsa wrapper)."""
    logits = F.linear(x.float(), w.float())
    scores = fn(logits)
    sel = scores + bias if bias is not None else scores
    T, E = sel.shape
    if n_group > 1:
        group_scores = (
            sel.view(T, n_group, E // n_group).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, E // n_group)
            .reshape(T, E)
        )
        sel = sel.masked_fill(~score_mask.bool(), float("-inf"))
    idx = torch.topk(sel, k=k, dim=-1, sorted=False)[1]
    wts = scores.gather(1, idx)
    if renorm:
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, wts * scale, sel, logits


def _sqrtsoftplus(x):
    return F.softplus(x).sqrt()


def _decisive(sel, k, tol):
    """Tokens whose k-th and (k+1)-th selection scores differ by > tol."""
    top = torch.topk(sel.double(), k + 1, dim=-1).values
    if top.shape[-1] <= k:
        return torch.ones(sel.shape[0], dtype=torch.bool, device=sel.device)
    kth, nxt = top[:, k - 1], top[:, k]
    nxt = torch.where(torch.isinf(nxt), kth - 1e9, nxt)
    return (kth - nxt) > tol * kth.abs().clamp_min(1.0)


def _same_set(a, b):
    return (a.long().sort(-1).values == b.long().sort(-1).values).all(-1)


def _check_dense(ids, tw, mask, dw, E):
    T = ids.shape[0]
    ref_mask = torch.zeros(T, E, dtype=torch.bool, device=ids.device)
    ref_mask.scatter_(1, ids.long(), True)
    assert torch.equal(mask, ref_mask)
    ref_dw = torch.zeros(T, E, dtype=torch.float32, device=ids.device)
    ref_dw.scatter_(1, ids.long(), tw)
    torch.testing.assert_close(dw.float(), ref_dw, rtol=1e-2, atol=1e-3)


def _int_inputs(T, H, E, density=0.02, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def sparse_int(*shape):
        v = torch.randint(-1, 2, shape, device="cuda", generator=g)
        keep = torch.rand(shape, device="cuda", generator=g) < density
        return (v * keep).to(torch.bfloat16)

    x = sparse_int(T, H)
    w = sparse_int(E, H)
    # duplicate expert rows -> exactly tied logits for many tokens
    w[1::7] = w[0::7][: w[1::7].shape[0]]
    return x, w


SOFTMAX_SHAPES = [
    # (T, H, E, k): Qwen3-235B, Qwen3.5-35B, Qwen3.5-397B, GPT-OSS-120B
    (1, 4096, 128, 8),
    (7, 2048, 256, 8),
    (64, 4096, 512, 10),
    (300, 2880, 128, 4),
    (5000, 2048, 256, 8),
]


@pytest.mark.parametrize("T,H,E,k", SOFTMAX_SHAPES)
def test_softmax_tie_heavy_identical(T, H, E, k, route):
    x, w = _int_inputs(T, H, E)
    ref_idx, ref_w, _, _ = _softmax_ref(x, w, k)
    ids, tw, mask, dw, _, _ = route(x, w, k, scoring="softmax")
    same = _same_set(ids, ref_idx)
    assert same.all(), f"{(~same).sum().item()} / {T} tokens differ"
    _check_dense(ids, tw, mask, dw, E)


@pytest.mark.parametrize("T,H,E,k", SOFTMAX_SHAPES)
def test_softmax_random_matches(T, H, E, k, route):
    torch.manual_seed(T + E)
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    w = (torch.randn(E, H, device="cuda") * 0.02).to(torch.bfloat16)
    ref_idx, ref_w, _, ref_logits = _softmax_ref(x, w, k)
    ids, tw, mask, dw, logits, _ = route(
        x, w, k, scoring="softmax", return_logits=True
    )
    # one bf16 ulp on the (bf16) logits at the boundary
    dec = _decisive(ref_logits.float(), k, 2.0**-7)
    same = _same_set(ids, ref_idx)
    assert same[
        dec
    ].all(), f"{(~same[dec]).sum().item()} decisive tokens differ"
    assert dec.float().mean() > 0.5
    _check_dense(ids, tw, mask, dw, E)
    order = ids.long().sort(-1)
    ref_order = ref_idx.sort(-1)
    rows = same & dec
    torch.testing.assert_close(
        tw.gather(1, order.indices)[rows],
        ref_w.float().gather(1, ref_order.indices)[rows],
        # a one-ulp bf16 logit difference moves exp() by up to ~3 %
        rtol=4e-2,
        atol=1e-3,
    )
    # at most one bf16 ulp apart (summation order before rounding)
    torch.testing.assert_close(
        logits.float(), ref_logits.float(), rtol=2.0**-7, atol=1e-2
    )


GROUPED_CASES = [
    # (T, H, E, k, n_group, topk_group, scoring, scale)
    (1, 6144, 256, 8, 1, 1, "sigmoid", 2.5),  # GLM-5.3
    (33, 6144, 256, 8, 8, 4, "sigmoid", 2.5),  # DeepSeek-V3 style groups
    (130, 4096, 288, 8, 1, 1, "sigmoid", 2.5),  # GLM-5.3-Flash
    (5000, 6144, 256, 8, 8, 4, "sigmoid", 1.0),
    (1, 4096, 256, 6, 1, 1, "sqrtsoftplus", 1.5),  # DeepSeek-V4-Flash
    (257, 5120, 384, 6, 1, 1, "sqrtsoftplus", 1.5),  # DeepSeek-V4.1-Flash
    (5000, 4096, 256, 6, 1, 1, "sqrtsoftplus", 1.5),
]


def _fn(scoring):
    return torch.sigmoid if scoring == "sigmoid" else _sqrtsoftplus


@pytest.mark.parametrize("T,H,E,k,ng,tg,scoring,scale", GROUPED_CASES)
def test_grouped_tie_heavy_identical(T, H, E, k, ng, tg, scoring, scale, route):
    x, w = _int_inputs(T, H, E, seed=1)
    # coarse bias -> ties survive the bias add
    g = torch.Generator(device="cuda").manual_seed(T + E)
    bias = (
        torch.randint(-2, 3, (E,), device="cuda", generator=g) * 0.25
    ).float()
    ref_idx, ref_w, _, _ = _grouped_ref(
        x, w, bias, k, ng, tg, True, scale, _fn(scoring)
    )
    ids, tw, mask, dw, _, _ = route(
        x, w, k, scoring=scoring, bias=bias, n_group=ng, topk_group=tg,
        renormalize=True, scaling=scale, eps=1e-20,
    )  # fmt: skip
    same = _same_set(ids, ref_idx)
    assert same.all(), f"{(~same).sum().item()} / {T} tokens differ"
    _check_dense(ids, tw, mask, dw, E)


@pytest.mark.parametrize("T,H,E,k,ng,tg,scoring,scale", GROUPED_CASES)
def test_grouped_random_matches(T, H, E, k, ng, tg, scoring, scale, route):
    torch.manual_seed(T + E + k)
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    w = (torch.randn(E, H, device="cuda") * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda") * 0.01
    ref_idx, ref_w, ref_sel, _ = _grouped_ref(
        x, w, bias, k, ng, tg, True, scale, _fn(scoring)
    )
    ids, tw, mask, dw, _, _ = route(
        x, w, k, scoring=scoring, bias=bias, n_group=ng, topk_group=tg,
        renormalize=True, scaling=scale, eps=1e-20,
    )  # fmt: skip
    dec = _decisive(ref_sel, k, 1e-5)
    if ng > 1:
        sel_full = _fn(scoring)(F.linear(x.float(), w.float())) + bias
        gs = sel_full.view(T, ng, E // ng).topk(2, -1)[0].sum(-1)
        dec &= _decisive(gs, tg, 1e-5)
    same = _same_set(ids, ref_idx)
    assert same[
        dec
    ].all(), f"{(~same[dec]).sum().item()} decisive tokens differ"
    assert dec.float().mean() > 0.5
    _check_dense(ids, tw, mask, dw, E)
    rows = same & dec
    torch.testing.assert_close(
        tw.gather(1, ids.long().sort(-1).indices)[rows],
        ref_w.gather(1, ref_idx.sort(-1).indices)[rows],
        rtol=1e-4,
        atol=1e-5,
    )


def test_fp32_weight_uses_fp32_logits():
    """fp32 operands: the Triton path (tf32x3), auto-selected."""
    from moe_infinity.kernel.router import fused_route as route

    torch.manual_seed(0)
    x = torch.randn(17, 4096, device="cuda")
    w = torch.randn(256, 4096, device="cuda") * 0.02
    bias = torch.randn(256, device="cuda") * 0.01
    ref_idx, ref_w, ref_sel, _ = _grouped_ref(
        x, w, bias, 6, 1, 1, True, 1.5, _sqrtsoftplus
    )
    ids, tw, *_ = route(
        x, w, 6, scoring="sqrtsoftplus", bias=bias, scaling=1.5, eps=1e-20
    )
    dec = _decisive(ref_sel, 6, 1e-5)
    assert _same_set(ids, ref_idx)[dec].all()


def test_repeated_calls_are_deterministic(route):
    """Split-K counters self-reset; the reduction order is fixed."""
    torch.manual_seed(0)
    x = torch.randn(3, 4096, device="cuda").to(torch.bfloat16)
    w = (torch.randn(256, 4096, device="cuda") * 0.02).to(torch.bfloat16)
    first = route(x, w, 8, scoring="sigmoid", return_logits=True)
    for _ in range(20):
        again = route(x, w, 8, scoring="sigmoid", return_logits=True)
        assert torch.equal(first[0], again[0])
        assert torch.equal(first[4], again[4])


def test_cuda_graph_capture(route):
    """No host sync: the router can be captured and replayed."""
    torch.manual_seed(0)
    x = torch.randn(4, 2048, device="cuda").to(torch.bfloat16)
    w = (torch.randn(256, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ref = route(x, w, 8)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = route(x, w, 8)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[2], ref[2])


@pytest.mark.skipif(
    not _has_cuda_ext(), reason="moe_infinity._router not built"
)
@pytest.mark.parametrize("T", [1, 16, 200, 3000])
def test_active_flags_match_mask(T):
    from moe_infinity.kernel.router import fused_route, take_active_flags

    torch.manual_seed(T)
    x = torch.randn(T, 2048, device="cuda").to(torch.bfloat16)
    w = (torch.randn(256, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    for _ in range(3):  # device flags are re-zeroed after every publish
        out = fused_route(x, w, 8, backend="cuda", active_flags=True)
        flags = take_active_flags(out.router_mask)
        assert flags is not None and flags.is_pinned()
        torch.cuda.synchronize()
        assert torch.equal(flags, out.router_mask.any(0).cpu())
    assert take_active_flags(out.router_mask) is None


@pytest.mark.skipif(
    not _has_cuda_ext(), reason="moe_infinity._router not built"
)
@pytest.mark.parametrize("T", [1, 64, 2000])
def test_no_host_sync_and_graph_replay(T):
    """Rule D: the per-layer routing step never synchronises the host, and a
    captured CUDA graph replays it with identical outputs."""
    from moe_infinity.kernel.router import make_router, take_active_flags

    torch.manual_seed(T)
    x = torch.randn(T, 4096, device="cuda").to(torch.bfloat16)
    w = (torch.randn(256, 4096, device="cuda") * 0.02).to(torch.bfloat16)
    bias = torch.randn(256, device="cuda") * 0.01
    route = make_router(
        w, 8, scoring="sigmoid", bias=bias, eps=1e-20, active_flags=True,
        compact=False,
    )  # fmt: skip
    route(x)  # warm-up: one-time allocations / plan selection
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            out = route(x)
            assert take_active_flags(out[2]) is out[5]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    ref_mask, ref_w = out[2].clone(), out[3].clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        route(x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            cap = route(x)
    torch.cuda.current_stream().wait_stream(s)
    cap[2].zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(cap[2], ref_mask)
    assert torch.equal(cap[3], ref_w)


@pytest.mark.skipif(
    not _has_cuda_ext(), reason="moe_infinity._router not built"
)
@pytest.mark.parametrize("E,k", [(32, 4), (8, 2)])
def test_prefill_kernel_default_path_flags(E, k):
    """Few-expert prefill takes the single Triton GEMM + top-k launch; its
    pinned flags match the mask and selection matches torch."""
    from moe_infinity.kernel.router import fused_route, take_active_flags

    x, w = _int_inputs(4096, 2048, E, density=0.05)
    out = fused_route(x, w, k, active_flags=True)
    flags = take_active_flags(out.router_mask)
    torch.cuda.synchronize()
    assert flags is not None
    assert torch.equal(flags, out.router_mask.any(0).cpu())
    ref_idx, *_ = _softmax_ref(x, w, k)
    assert _same_set(out.topk_ids, ref_idx).all()


@pytest.mark.skipif(
    not _has_cuda_ext(), reason="moe_infinity._router not built"
)
@pytest.mark.parametrize(
    "scoring,k", [("softmax", 8), ("sigmoid", 8), ("sqrtsoftplus", 6)]
)
@pytest.mark.parametrize("tie", [True, False])
def test_wide_fused_gemm_topk(scoring, k, tie):
    """E = 256 prefill in one tile wave takes the fused CUTLASS GEMM + top-k
    kernel; selection matches the torch routers."""
    from moe_infinity.kernel.router import fused_route

    T, H, E = 4500, 2048, 256
    if tie:
        x, w = _int_inputs(T, H, E, seed=7)
    else:
        torch.manual_seed(7)
        x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
        w = (torch.randn(E, H, device="cuda") * 0.02).to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(3)
    bias = (
        None
        if scoring == "softmax"
        else (
            torch.randint(-2, 3, (E,), device="cuda", generator=g) * 0.25
        ).float()
    )
    eps = 0.0 if scoring == "softmax" else 1e-20
    out = fused_route(
        x, w, k, scoring=scoring, bias=bias, eps=eps, backend="cuda",
        active_flags=True,
    )  # fmt: skip
    if scoring == "softmax":
        ref_idx, _, _, lg = _softmax_ref(x, w, k)
        sel, tol = lg.float(), 2.0**-6  # cuBLAS bf16 output: ~1 ulp noise
    else:
        fn = torch.sigmoid if scoring == "sigmoid" else _sqrtsoftplus
        ref_idx, _, sel, _ = _grouped_ref(x, w, bias, k, 1, 1, True, 1.0, fn)
        tol = 1e-5
    same = _same_set(out.topk_ids, ref_idx)
    dec = torch.ones_like(same) if tie else _decisive(sel, k, tol)
    assert same[dec].all(), f"{(~same[dec]).sum().item()} tokens differ"
    _check_dense(
        out.topk_ids, out.topk_weights, out.router_mask, out.routing_weights, E
    )
