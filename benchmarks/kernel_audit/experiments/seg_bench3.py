"""Round 3: sequence-split h+o vs the sequential fused kernel (GDN + KDA)."""

import sys

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, __import__("os").path.dirname(__file__))
from benchutil import bench  # noqa: E402

from moe_infinity.kernel.gdn_fused import delta_rule_fwd_h_o  # noqa: E402
from moe_infinity.kernel.gdn_fused_seg import seg_fwd_h_o  # noqa: E402
from moe_infinity.kernel.gdn_prep import gdn_fwd_prep  # noqa: E402

D, scale = 128, 128**-0.5
kind, Ts, Ls = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
H = 32 if kind == "gdn" else 64
for T in map(int, Ts):
    torch.manual_seed(0)
    q = l2norm_fwd(torch.randn(1, T, H, D, device="cuda").bfloat16())[0]
    k = torch.randn(1, T, H, D, device="cuda").bfloat16()
    v = torch.randn(1, T, H, D, device="cuda").bfloat16()
    beta = torch.rand(1, T, H, device="cuda").bfloat16()
    h0 = torch.randn(1, H, D, D, device="cuda") * 0.1
    if kind == "gdn":
        g = chunk_local_cumsum(
            -torch.rand(1, T, H, device="cuda") * 0.5,
            chunk_size=64,
            scale=RCP_LN2,
        )
        w, u, kk = gdn_fwd_prep(k, v, g, beta, True)
        A = None
    else:
        g = chunk_local_cumsum(
            -5.0 * torch.sigmoid(torch.randn(1, T, H, D, device="cuda") - 4.0),
            chunk_size=64,
            scale=RCP_LN2,
        )
        w, u, _, kk, A, _ = chunk_kda_fwd_intra(
            q=q, k=l2norm_fwd(k)[0], v=v, gk=g, beta=beta, scale=scale
        )
    fs = lambda: delta_rule_fwd_h_o(q, kk, w, u, g, scale, A, h0, True)
    o0, s0 = fs()
    print(f"{kind} T={T}: sequential {bench(fs):8.1f} us")
    for L in map(int, Ls):
        f = lambda: seg_fwd_h_o(
            q, kk, w, u, g, scale, A, h0, True, seg_chunks=L
        )
        o, s = f()
        rel = ((o.float() - o0.float()).norm() / o0.float().norm()).item()
        t = bench(f)
        f()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            f()
            torch.cuda.synchronize()
        ks = [e for e in p.events() if e.device_type.name == "CUDA"]
        parts = " + ".join(
            f"{e.name.split('_kernel')[0][:8]} {e.device_time:.0f}" for e in ks
        )
        print(
            f"  L={L:3d}: {t:8.1f} us [{parts}]  rel|o-seq| {rel:.1e} max|S-seq| {(s - s0).abs().max().item():.1e}"
        )
