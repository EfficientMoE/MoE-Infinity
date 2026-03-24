import pytest
import torch
import torch.nn.functional as F

import moe_infinity._engine as _engine
from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    seed_everything,
)


@requires_cuda
@pytest.mark.parametrize(
    "batch,hidden_dim,output_dim",
    [(1, 128, 512), (16, 128, 512), (32, 128, 1024)],
)
def test_fused_silu_gemm_matches_reference(
    seed_everything, batch, hidden_dim, output_dim
):
    """fused_silu_gemm output matches F.silu(hidden @ gate.T)."""
    hidden = torch.randn(
        batch, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    gate_proj = torch.randn(
        output_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    custom = _engine.fused_silu_gemm(hidden, gate_proj)
    reference = F.silu(torch.mm(hidden, gate_proj.t()))
    torch.testing.assert_close(
        custom, reference, rtol=BF16_RTOL, atol=BF16_ATOL
    )
