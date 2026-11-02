"""Grouped MXFP4 MoE (GPT-OSS resident path) vs the per-expert loop."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

ALPHA, LIMIT = 1.702, 7.0


def _swiglu(gate, up):
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(-LIMIT, LIMIT)
    return (up + 1) * (gate * torch.sigmoid(gate * ALPHA))


def _weights(E, H, I, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=g, device="cuda", dtype=torch.uint8
        )

    def bf(*shape, scale=0.1):
        return (
            torch.randn(*shape, generator=g, device="cuda") * scale
        ).bfloat16()

    return dict(
        gate_up_blocks=u8(E, 2 * I, H // 2),
        gate_up_scales=u8(E, 2 * I, H // 32, lo=118, hi=124),
        gate_up_bias=bf(E, 2 * I),
        down_blocks=u8(E, H, I // 2),
        down_scales=u8(E, H, I // 32, lo=118, hi=124),
        down_bias=bf(E, H),
    )


def _route(T, H, E, k, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, generator=g, device="cuda").bfloat16()
    logits = torch.randn(T, E, generator=g, device="cuda")
    p = torch.softmax(logits, dim=-1)
    w, ids = torch.topk(p, k, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).bfloat16()
    return x, ids, w


def _loop_reference(x, ids, w, wt):
    """SyncGptOssMLP's resident loop (moe-store wrappers/gpt_oss.py)."""
    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    T = x.shape[0]
    E = wt["gate_up_blocks"].shape[0]
    mask = torch.zeros(T, E, dtype=torch.bool, device="cuda")
    mask.scatter_(1, ids, True)
    wmask = torch.zeros(T, E, dtype=x.dtype, device="cuda")
    wmask.scatter_(1, ids, w)
    out = torch.zeros_like(x)
    for e in range(E):
        tm = mask[:, e]
        if not tm.any():
            continue
        gu = fused_mxfp4_gemm(
            x[tm],
            wt["gate_up_blocks"][e],
            wt["gate_up_scales"][e],
            wt["gate_up_bias"][e],
        )
        act = _swiglu(gu[..., ::2], gu[..., 1::2])
        y = fused_mxfp4_gemm(
            act.bfloat16(),
            wt["down_blocks"][e],
            wt["down_scales"][e],
            wt["down_bias"][e],
        )
        out[tm] += wmask[tm, e].unsqueeze(-1) * y
    return out


def _fp32_reference(x, ids, w, wt):
    from moe_infinity.kernel.mxfp4_gemm import mxfp4_dequantize

    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    for e in ids.unique().tolist():
        gw = mxfp4_dequantize(
            wt["gate_up_blocks"][e], wt["gate_up_scales"][e], torch.float32
        )
        dw = mxfp4_dequantize(
            wt["down_blocks"][e], wt["down_scales"][e], torch.float32
        )
        t, j = (ids == e).nonzero(as_tuple=True)
        gu = x[t].float() @ gw.T + wt["gate_up_bias"][e].float()
        act = _swiglu(gu[:, ::2], gu[:, 1::2])
        y = act @ dw.T + wt["down_bias"][e].float()
        out.index_add_(0, t, w[t, j].float()[:, None] * y)
    return out


@pytest.mark.parametrize(
    "T,E,k,H,I",
    [
        (1, 32, 4, 64, 64),  # decode b1, gpt-oss-20b-like routing
        (5, 8, 2, 96, 160),  # K not a multiple of the K tile
        (16, 128, 4, 128, 64),  # decode b16, gpt-oss-120b-like routing
        (300, 32, 4, 64, 96),  # prefill, many rows per expert
        (1, 32, 4, 2880, 2880),  # real gpt-oss shapes
        (64, 32, 4, 2880, 2880),
    ],
)
@pytest.mark.parametrize("path", ["auto", "nosplit", "mma"])
def test_matches_resident_loop(T, E, k, H, I, path, monkeypatch):
    import moe_infinity.kernel.mxfp4_moe as mm
    from moe_infinity.kernel.mxfp4_moe import fused_moe_mxfp4

    if path == "nosplit":
        monkeypatch.setattr(mm, "SPLIT_MAX_TK", 0)
    if path in ("nosplit", "mma"):
        monkeypatch.setattr(mm, "GEMV_MAX_TK", 0)

    wt = _weights(E, H, I)
    x, ids, w = _route(T, H, E, k)
    out = fused_moe_mxfp4(x, ids, w, **wt, alpha=ALPHA, limit=LIMIT)
    ref = _loop_reference(x, ids, w, wt)
    exact = _fp32_reference(x, ids, w, wt)

    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    scale = exact.abs().max().clamp_min(1e-6)
    err_new = ((out.float() - exact).abs().max() / scale).item()
    err_ref = ((ref.float() - exact).abs().max() / scale).item()
    diff = ((out.float() - ref.float()).abs().max() / scale).item()
    # Both paths round GEMM outputs/partials to bf16; the grouped kernel only
    # changes fp32 accumulation order and the order of the k bf16 adds.
    assert err_new <= max(2 * err_ref, 2e-2), (err_new, err_ref)
    assert diff < 2e-2, diff


def test_align_groups_every_assignment_once():
    from moe_infinity.kernel.mxfp4_moe import moe_align_mxfp4

    for T, E, k, bm in [(1, 32, 4, 16), (37, 8, 2, 16), (1000, 128, 4, 32)]:
        _, ids, _ = _route(T, 8, E, k, seed=T)
        out = torch.ones(T, 24, dtype=torch.bfloat16, device="cuda")
        sorted_ids, block_expert = moe_align_mxfp4(ids, E, bm, out)
        tk = T * k
        flat = ids.reshape(-1)
        s = sorted_ids.view(-1, bm)
        assert torch.all(out == 0)
        seen = []
        for b in range(s.shape[0]):
            e = int(block_expert[b])
            rows = s[b][s[b] < tk]
            if e < 0:
                assert rows.numel() == 0
                continue
            assert torch.all(flat[rows.long()] == e)
            seen.append(rows)
        seen = torch.cat(seen).sort().values
        assert torch.equal(seen.cpu(), torch.arange(tk, dtype=torch.int32))


def test_no_host_sync_and_three_launches():
    from moe_infinity.kernel.mxfp4_moe import fused_moe_mxfp4

    E, H, I = 32, 2880, 2880
    wt = _weights(E, H, I)
    x, ids, w = _route(1, H, E, 4)
    fused_moe_mxfp4(x, ids, w, **wt)  # compile
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        fused_moe_mxfp4(x, ids, w, **wt)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fused_moe_mxfp4(x, ids, w, **wt)
        torch.cuda.synchronize()
    kernels = [
        e
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) <= 3, [e.name for e in kernels]


def _mxfp4_block(E, k, H, I):
    from types import SimpleNamespace

    from moe_store.wrappers.gpt_oss import SyncGptOssMLP

    cfg = SimpleNamespace(
        hidden_size=H,
        intermediate_size=I,
        num_local_experts=E,
        num_experts_per_tok=k,
    )
    block = SyncGptOssMLP(cfg).cuda()
    wt = _weights(E, H, I, seed=3)
    ex = block.experts
    for name, key in (
        ("gate_up_proj", "gate_up_blocks"),
        ("gate_up_proj_scales", "gate_up_scales"),
        ("gate_up_proj_bias", "gate_up_bias"),
        ("down_proj", "down_blocks"),
        ("down_proj_scales", "down_scales"),
        ("down_proj_bias", "down_bias"),
    ):
        # as _load_resident_gpt_oss: frozen resident params
        param = getattr(ex, name)
        param.requires_grad_(False)
        param.data = wt[key]
    torch.nn.init.normal_(block.router.weight, std=0.5)
    block.router.bfloat16()
    return block


@pytest.mark.parametrize("T", [1, 16, 200])
def test_sync_gpt_oss_mlp_grouped_vs_loop(T, monkeypatch):
    """Tiny GPT-OSS block: grouped op == per-expert loop, no host sync."""
    pytest.importorskip("moe_store")
    from moe_store import hooks

    from moe_infinity.runtime.engine_ops import register_engine_ops

    register_engine_ops()
    assert hooks.ops.fused_moe_mxfp4 is not None
    E, k, H, I = 32, 4, 128, 64
    block = _mxfp4_block(E, k, H, I)
    x = torch.randn(1, T, H, device="cuda").bfloat16()

    with torch.no_grad():
        out, logits = block(x)  # grouped (also compiles)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            block(x)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        monkeypatch.setattr(hooks.ops, "fused_moe_mxfp4", None)
        ref, ref_logits = block(x)

    torch.testing.assert_close(logits, ref_logits)
    scale = ref.float().abs().max()
    diff = ((out.float() - ref.float()).abs().max() / scale).item()
    assert diff < 2e-2, diff


@pytest.mark.parametrize("T", [1, 16, 300])
def test_cuda_graph_replay_matches_eager(T):
    """Rule D: the sync-free forward captures in a CUDA graph and replays."""
    from moe_infinity.kernel.mxfp4_moe import fused_moe_mxfp4

    E, k, H, I = 32, 4, 256, 128
    wt = _weights(E, H, I)
    x, ids, w = _route(T, H, E, k)
    eager = fused_moe_mxfp4(x, ids, w, **wt)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_moe_mxfp4(x, ids, w, **wt)
    graph.replay()
    torch.cuda.synchronize()
    # Equal up to the order of the k atomic bf16 adds per output element.
    scale = eager.float().abs().max()
    diff = ((out.float() - eager.float()).abs().max() / scale).item()
    assert diff < 1e-2, diff
