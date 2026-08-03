import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_native_fp8_dequant_matches_python():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from moe_infinity._v4_fp4 import fp8_dequant_blockwise
    except Exception:
        pytest.skip("native fp8 dequant not built")
    from moe_infinity.utils.fp8 import dequant_fp8_blockwise

    torch.manual_seed(0)
    N, K = 256, 512
    w = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    s = torch.rand((N // 128, K // 128), device="cuda", dtype=torch.float32) + 0.5
    ref = dequant_fp8_blockwise(w, s, dtype=torch.bfloat16, block_size=128)
    got = fp8_dequant_blockwise(w, s).to(torch.bfloat16)
    assert torch.allclose(got.float(), ref.float(), atol=1e-2, rtol=1e-2)
