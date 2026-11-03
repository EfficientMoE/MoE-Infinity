import sys

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

sys.path.insert(0, __import__("os").path.dirname(__file__))
import moe_infinity.kernel.gdn_fused_seg as S  # noqa: E402
from moe_infinity.kernel.gdn_fused import delta_rule_fwd_h_o  # noqa: E402
from moe_infinity.kernel.gdn_prep import gdn_fwd_prep  # noqa: E402

H, D, T, L = 32, 128, 2000, 6
scale = D**-0.5
torch.manual_seed(0)
q = l2norm_fwd(torch.randn(1, T, H, D, device="cuda").bfloat16())[0]
k = torch.randn(1, T, H, D, device="cuda").bfloat16()
v = torch.randn(1, T, H, D, device="cuda").bfloat16()
beta = torch.rand(1, T, H, device="cuda").bfloat16()
h0 = torch.randn(1, H, D, D, device="cuda") * 0.1
g = chunk_local_cumsum(
    -torch.rand(1, T, H, device="cuda") * float(sys.argv[1]),
    chunk_size=64,
    scale=RCP_LN2,
)
w, u, kk = gdn_fwd_prep(k, v, g, beta, True)

# capture hs by calling the kernels directly
NT = -(-T // 64)
NS = -(-NT // L)
pq = torch.full((NS - 1, 1, H, D, 2 * D), float("nan"), device="cuda")
o = torch.full_like(u, float("nan"))
S.seg_h_o_kernel[((NS - 1) * H * 2 * D // 32,)](
    q,
    kk,
    w,
    u,
    g,
    u,
    o,
    u,
    pq,
    scale,
    T,
    L,
    HV=H,
    K=D,
    V=D,
    BT=64,
    BV=32,
    GATE_VEC=False,
    STATE_ONLY=True,
    USE_INITIAL_STATE=False,
    STORE_FINAL_STATE=False,
    num_warps=4,
    num_stages=2,
)
hs = torch.full((NS, 1, H, D, D), float("nan"), device="cuda")
S.seg_combine_kernel[(H * 4,)](
    pq, h0, hs, NS, H, K=D, V=D, BV=32, USE_INITIAL_STATE=True, num_warps=4
)
for s in range(1, NS):
    n = s * L * 64
    _, st = delta_rule_fwd_h_o(
        q[:, :n], kk[:, :n], w[:, :n], u[:, :n], g[:, :n], scale, None, h0, True
    )
    d = (hs[s, 0] - st[0]).abs().max().item()
    print(
        f"segment {s}: max|hs - sequential state| {d:.3e}  |state| {st.abs().max().item():.2f}  nan {hs[s].isnan().any().item()}"
    )
o1, s1 = S.seg_fwd_h_o(q, kk, w, u, g, scale, None, h0, True, seg_chunks=L)
o0, s0 = delta_rule_fwd_h_o(q, kk, w, u, g, scale, None, h0, True)
print(
    "seg vs seq: max|do|",
    (o1.float() - o0.float()).abs().max().item(),
    "equal",
    torch.equal(o1, o0),
    "max|dS|",
    (s1 - s0).abs().max().item(),
)
o2, _ = S.seg_fwd_h_o(q, kk, w, u, g, scale, None, h0 * 0, True, seg_chunks=L)
print(
    "sanity, different h0 changes o:",
    (o2.float() - o1.float()).abs().max().item(),
)
