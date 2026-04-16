import pytest
import torch

from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm, mxfp4_dequantize


def _random_mxfp4_weight(N, K, device="cpu"):
    packed = torch.randint(
        0, 256, (N, K // 2), dtype=torch.uint8, device=device
    )
    scales = torch.randint(
        120, 135, (N, K // 32), dtype=torch.uint8, device=device
    )
    return packed, scales


def test_dequantize_zero_scale():
    packed = torch.zeros(1, 16, dtype=torch.uint8)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    result = mxfp4_dequantize(packed, scales)
    assert result.shape == (1, 32)
    assert (result == 0).all()


def test_dequantize_known_values():
    packed = torch.tensor([[0x10]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    result = mxfp4_dequantize(packed, scales, dtype=torch.float32)
    assert result.shape == (1, 2)
    assert result[0, 0].item() == 0.0
    assert result[0, 1].item() == 0.5


def test_dequantize_negative_values():
    packed = torch.tensor([[0x98]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    result = mxfp4_dequantize(packed, scales, dtype=torch.float32)
    assert result[0, 0].item() == pytest.approx(-0.0, abs=1e-7)
    assert result[0, 1].item() == pytest.approx(-0.5, abs=1e-7)


def test_dequantize_with_scale():
    packed = torch.tensor([[0x21]], dtype=torch.uint8)
    scales = torch.tensor([[128]], dtype=torch.uint8)
    result = mxfp4_dequantize(packed, scales, dtype=torch.float32)
    assert result[0, 0].item() == pytest.approx(0.5 * 2, abs=1e-5)
    assert result[0, 1].item() == pytest.approx(1.0 * 2, abs=1e-5)


def test_dequantize_shape_2d():
    N, K = 64, 128
    packed, scales = _random_mxfp4_weight(N, K)
    result = mxfp4_dequantize(packed, scales)
    assert result.shape == (N, K)
    assert result.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_gemm_matches_reference():
    torch.manual_seed(42)
    M, N, K = 4, 64, 128
    packed, scales = _random_mxfp4_weight(N, K, device="cuda")
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    weight_bf16 = mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)
    ref = x @ weight_bf16.t() + bias

    fused = fused_mxfp4_gemm(x, packed, scales, bias)

    torch.testing.assert_close(fused, ref, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_gemm_no_bias():
    torch.manual_seed(42)
    M, N, K = 2, 32, 64
    packed, scales = _random_mxfp4_weight(N, K, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    weight_bf16 = mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)
    ref = x @ weight_bf16.t()

    fused = fused_mxfp4_gemm(x, packed, scales)

    torch.testing.assert_close(fused, ref, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_gemm_batched_input():
    torch.manual_seed(42)
    B, S, N, K = 2, 3, 64, 128
    packed, scales = _random_mxfp4_weight(N, K, device="cuda")
    x = torch.randn(B, S, K, dtype=torch.bfloat16, device="cuda")

    weight_bf16 = mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)
    ref = x.reshape(-1, K) @ weight_bf16.t()
    ref = ref.reshape(B, S, N)

    fused = fused_mxfp4_gemm(x, packed, scales)

    assert fused.shape == (B, S, N)
    torch.testing.assert_close(fused, ref, rtol=1e-2, atol=5e-2)
