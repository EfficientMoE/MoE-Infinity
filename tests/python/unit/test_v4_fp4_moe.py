"""Native W4A8 block-scaled FP4 MoE (DeepSeek-V4 routed experts, sm_120a).

Checks `_v4_fp4.v4_moe_forward` / `v4_expert_forward_fp4` against a PyTorch
emulation of the official DeepSeek-V4 inference numerics (act_quant with
per-128 ue8m0 FP8 activations, FP4 e2m1 weights with per-32 ue8m0 scales,
BF16 gate/up, FP32 clamp + SiLU * up * routing weight, BF16 h, FP32 sum over
experts in ascending id order) on tiny random configs.
"""

from __future__ import annotations

import pytest
import torch


def _ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("sm_120a block-scaled MMA required")
    try:
        from moe_infinity import _v4_fp4
    except ImportError:
        pytest.skip("moe_infinity._v4_fp4 extension not built")
    if not hasattr(_v4_fp4, "v4_moe_forward"):
        pytest.skip("_v4_fp4 built without v4_moe_forward")
    return _v4_fp4


_E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    tab = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    p = packed.view(torch.uint8)
    v = torch.stack([tab[(p & 15).long()], tab[(p >> 4).long()]], -1)
    v = v.reshape(p.shape[0], -1)
    s = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return v * s.repeat_interleave(32, dim=1)


def act_quant_ref(x: torch.Tensor) -> torch.Tensor:
    """Official act_quant(block 128, ue8m0) followed by dequantisation."""
    xf = x.float().reshape(x.shape[0], -1, 128)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    s = torch.exp2(e.float())
    q = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(x.shape)


def expert_ref(x, w1, w3, w2, limit, weight=None):
    xq = act_quant_ref(x)
    gate = (xq @ w1.T).bfloat16().float()
    up = (xq @ w3.T).bfloat16().float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    h = torch.nn.functional.silu(gate) * up
    if weight is not None:
        h = weight * h
    h = h.bfloat16()
    return (act_quant_ref(h) @ w2.T).bfloat16()


def make_expert(H, I, gen):
    def w(n, k):
        packed = torch.randint(
            0, 256, (n, k // 2), generator=gen, dtype=torch.uint8
        )
        scale = torch.randint(
            118, 124, (n, k // 32), generator=gen, dtype=torch.uint8
        )
        return packed.cuda(), scale.cuda()

    w1, s1 = w(I, H)
    w3, s3 = w(I, H)
    w2, s2 = w(H, I)
    return dict(w1=w1, s1=s1, w3=w3, s3=s3, w2=w2, s2=s2)


def table_of(experts):
    rows = [
        [e[k].data_ptr() for k in ("w1", "s1", "w3", "s3", "w2", "s2")]
        for e in experts
    ]
    return torch.tensor(rows, dtype=torch.int64, device="cuda")


def moe_ref(x, ids, wts, experts, limit):
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    deq = {}
    for e in sorted(set(ids.flatten().tolist())):
        if e < 0 or e >= len(experts):
            continue
        if e not in deq:
            p = experts[e]
            deq[e] = (
                dequant_fp4(p["w1"], p["s1"]),
                dequant_fp4(p["w3"], p["s3"]),
                dequant_fp4(p["w2"], p["s2"]),
            )
        idx, top = torch.where(ids == e)
        y[idx] += expert_ref(
            x[idx], *deq[e], limit, wts[idx, top, None]
        ).float()
    return y


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.parametrize(
    "T,topk,E,H,I",
    [
        (1, 6, 16, 256, 128),  # BT=8, grouped kernels (I % 256 != 0)
        (1, 6, 16, 256, 256),  # BT=8, persistent decode kernel
        (4, 6, 16, 512, 256),
        (40, 6, 64, 256, 512),  # decode kernel, several tiles per expert
        (2, 6, 32, 4096, 2048),  # V4-Flash dims
        (3, 6, 24, 5120, 2304),  # V4.1-Flash dims
        (24, 4, 8, 256, 256),  # BT=16
        (48, 4, 8, 512, 128),  # BT=32
        (100, 4, 8, 256, 384),  # BT=64
        (700, 6, 8, 512, 256),  # BT=128, multiple row tiles per expert
    ],
)
def test_moe_matches_official_numerics(T, topk, E, H, I):
    ext = _ext()
    gen = torch.Generator().manual_seed(T * 131 + E)
    experts = [make_expert(H, I, gen) for _ in range(E)]
    x = torch.randn(T, H, generator=gen).bfloat16().cuda()
    scores = torch.rand(T, E, generator=gen)
    ids = scores.topk(topk, -1).indices.cuda()
    wts = torch.rand(T, topk, generator=gen).cuda() + 0.1
    out = ext.v4_moe_forward(x, ids, wts, table_of(experts), I, 10.0)
    ref = moe_ref(x, ids, wts, experts, 10.0)
    assert out.dtype == torch.float32 and out.shape == (T, H)
    # Differences come only from FP32 accumulation order and the fast SiLU
    # exp, which can flip one BF16 ulp of h / the output.
    assert _rel(out, ref) < 2e-2
    assert (out - ref).abs().mean().item() < 2e-3 * ref.abs().mean().item()


def test_moe_int32_ids_skips_invalid_and_empty_experts():
    ext = _ext()
    gen = torch.Generator().manual_seed(7)
    E, H, I, T = 12, 256, 128, 9
    experts = [make_expert(H, I, gen) for _ in range(E)]
    x = torch.randn(T, H, generator=gen).bfloat16().cuda()
    ids = torch.tensor(
        [
            [0, 3],
            [3, -1],
            [11, 0],
            [3, 0],
            [5, 3],
            [0, 11],
            [-1, -1],
            [3, 5],
            [0, 12],
        ],
        dtype=torch.int32,
    ).cuda()
    wts = torch.rand(T, 2, generator=gen).cuda()
    out = ext.v4_moe_forward(x, ids, wts, table_of(experts), I, 0.0)
    ref = moe_ref(x, ids.long(), wts, experts, 0.0)
    assert torch.all(out[6] == 0)
    assert _rel(out, ref) < 2e-2


def test_single_expert_vs_bf16_dequant_path():
    ext = _ext()
    gen = torch.Generator().manual_seed(3)
    H, I, M = 512, 256, 37
    p = make_expert(H, I, gen)
    x = torch.randn(M, H, generator=gen).bfloat16().cuda()
    args = (x, p["w1"], p["s1"], p["w2"], p["s2"], p["w3"], p["s3"], 10.0)
    out = ext.v4_expert_forward_fp4(*args)
    w = [
        dequant_fp4(p[k], p[s])
        for k, s in (("w1", "s1"), ("w3", "s3"), ("w2", "s2"))
    ]
    ref = expert_ref(x, *w, 10.0)
    assert out.dtype == torch.bfloat16 and out.shape == (M, H)
    assert _rel(out, ref) < 2e-2
    # vs. the BF16-activation dequant path: only FP8 activation rounding differs
    bf16 = ext.v4_expert_forward(*args)
    assert _rel(out, bf16) < 0.1


@pytest.mark.parametrize("T", [1, 3, 300])  # decode kernel, decode, grouped
def test_moe_no_host_sync_and_cuda_graph_replay(T):
    """No host synchronisation inside the op, and the decode kernel's
    self-resetting counters survive repeated CUDA-graph replays."""
    ext = _ext()
    gen = torch.Generator().manual_seed(T)
    E, H, I, k = 16, 512, 256, 6
    experts = [make_expert(H, I, gen) for _ in range(E)]
    table = table_of(experts)
    x = torch.randn(T, H, generator=gen).bfloat16().cuda()
    ids = torch.rand(T, E, generator=gen).topk(k, -1).indices.cuda()
    wts = torch.rand(T, k, generator=gen).cuda()
    ref = ext.v4_moe_forward(x, ids, wts, table, I, 10.0)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = ext.v4_moe_forward(x, ids, wts, table, I, 10.0)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ext.v4_moe_forward(x, ids, wts, table, I, 10.0)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ext.v4_moe_forward(x, ids, wts, table, I, 10.0)
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
    assert torch.equal(eager, ref)
