import pytest
import torch


@pytest.mark.gpu
def test_native_mxfp4_gate_up_dequant_is_exact():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from moe_infinity._v4_fp4 import mxfp4_dequant

    from moe_infinity.kernel.mxfp4_gemm import mxfp4_dequantize

    torch.manual_seed(137)
    blocks = torch.randint(
        0, 256, (5760, 1440), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randint(
        120, 135, (5760, 90), dtype=torch.uint8, device="cuda"
    )
    expected = mxfp4_dequantize(
        blocks, scales, dtype=torch.bfloat16, block_size=32
    )
    actual = mxfp4_dequant(blocks, scales)

    relative_error = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-12)
    ).max()
    assert actual.shape == (5760, 2880)
    assert actual.dtype == torch.bfloat16
    assert relative_error.item() == 0.0


@pytest.mark.gpu
def test_dequantized_option_a_matches_resident_expert_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from moe_infinity._v4_fp4 import mxfp4_dequant
    except Exception:
        pytest.skip("native MXFP4 dequant extension not built")

    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    torch.manual_seed(137)
    tokens, hidden, intermediate = 3, 64, 32
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate_blocks = torch.randint(
        0,
        256,
        (2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
    ).zero_()
    gate_scales = torch.randint(
        120,
        135,
        (2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    down_blocks = torch.randint(
        0,
        256,
        (hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    down_scales = torch.randint(
        120,
        135,
        (hidden, intermediate // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    gate_bias = torch.randn(
        2 * intermediate, dtype=torch.bfloat16, device="cuda"
    )
    down_bias = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")

    resident_gate_up = fused_mxfp4_gemm(
        x, gate_blocks, gate_scales, gate_bias
    )
    resident_gate, resident_up = (
        resident_gate_up[:, ::2],
        resident_gate_up[:, 1::2],
    )
    resident_activated = (resident_up.clamp(-7, 7) + 1) * (
        resident_gate.clamp(max=7)
        * torch.sigmoid(resident_gate.clamp(max=7) * 1.702)
    )
    resident = fused_mxfp4_gemm(
        resident_activated.to(torch.bfloat16),
        down_blocks,
        down_scales,
        down_bias,
    )

    gate_weight = mxfp4_dequant(gate_blocks, gate_scales)
    down_weight = mxfp4_dequant(down_blocks, down_scales)
    option_a_gate_up = x @ gate_weight.t() + gate_bias
    option_a_gate, option_a_up = (
        option_a_gate_up[:, ::2],
        option_a_gate_up[:, 1::2],
    )
    option_a_activated = (option_a_up.clamp(-7, 7) + 1) * (
        option_a_gate.clamp(max=7)
        * torch.sigmoid(option_a_gate.clamp(max=7) * 1.702)
    )
    option_a = option_a_activated @ down_weight.t() + down_bias

    torch.testing.assert_close(option_a, resident, rtol=1e-2, atol=5e-2)
