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
