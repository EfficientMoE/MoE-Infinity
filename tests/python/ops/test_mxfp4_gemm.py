"""Numerics and dispatch of the fused MXFP4 GEMM (GPT-OSS resident path, #312)."""

import pytest

from tests.python.ops.conftest import requires_cuda, requires_triton

torch = pytest.importorskip("torch")


def _weights(N, K, lo=118, hi=136, device="cuda"):
    packed = torch.randint(
        0, 256, (N, K // 2), dtype=torch.uint8, device=device
    )
    scales = torch.randint(
        lo, hi, (N, K // 32), dtype=torch.uint8, device=device
    )
    return packed, scales


def _reference(x, packed, scales, bias):
    from moe_infinity.kernel.mxfp4_gemm import mxfp4_dequantize

    w = mxfp4_dequantize(packed, scales, dtype=torch.float32)
    ref = x.reshape(-1, x.shape[-1]).float() @ w.t()
    if bias is not None:
        ref = ref + bias.float()
    return ref.reshape(*x.shape[:-1], -1)


def _check(out, ref):
    assert out.dtype == torch.bfloat16
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # bf16 output rounding plus fp32 accumulation-order differences
    scale = ref.abs().max().item()
    torch.testing.assert_close(
        out.float(), ref, rtol=1e-2, atol=1e-2 * scale + 1e-3
    )


@requires_cuda
@requires_triton
@pytest.mark.parametrize(
    "N,K", [(5760, 2880), (2880, 2880), (96, 128), (40, 64)]
)
@pytest.mark.parametrize(
    "M", [1, 2, 3, 16, 17, 37, 64, 65, 255, 600, 1009, 4097]
)
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_mxfp4_gemm_matches_dequant_matmul(N, K, M, with_bias):
    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    torch.manual_seed(M * 7 + N)
    packed, scales = _weights(N, K)
    bias = (
        torch.randn(N, dtype=torch.bfloat16, device="cuda")
        if with_bias
        else None
    )
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    _check(
        fused_mxfp4_gemm(x, packed, scales, bias),
        _reference(x, packed, scales, bias),
    )


@requires_cuda
@requires_triton
def test_fused_mxfp4_gemm_every_e2m1_code_and_scale():
    """All 256 byte patterns x a wide scale range, incl. E2M1 subnormals."""
    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    K = 512
    codes = torch.arange(256, dtype=torch.int32, device="cuda")
    rows = torch.arange(64, dtype=torch.int32, device="cuda")[:, None]
    packed = ((codes[None, :] + 37 * rows) % 256).to(torch.uint8)
    assert packed.shape == (64, K // 2)
    scales = torch.arange(64, 64 + 64 * 16, dtype=torch.int32, device="cuda")
    scales = (scales.reshape(64, 16) % 192 + 32).to(torch.uint8)
    for M in (1, 4, 300):
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        _check(
            fused_mxfp4_gemm(x, packed, scales),
            _reference(x, packed, scales, None),
        )


@requires_cuda
@requires_triton
def test_fused_mxfp4_gemm_batched_and_unaligned_input():
    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    N, K = 128, 256
    packed, scales = _weights(N, K)
    x3 = torch.randn(2, 3, K, dtype=torch.bfloat16, device="cuda")
    out = fused_mxfp4_gemm(x3, packed, scales)
    assert out.shape == (2, 3, N)
    _check(out, _reference(x3, packed, scales, None))
    # a row view starting at an odd element offset (GEMV reads bf16 pairs)
    flat = torch.randn(K + 1, dtype=torch.bfloat16, device="cuda")
    x1 = flat[1:].view(1, K)
    _check(
        fused_mxfp4_gemm(x1, packed, scales),
        _reference(x1, packed, scales, None),
    )
    assert fused_mxfp4_gemm(x3[:0], packed, scales).shape == (0, 3, N)


@requires_cuda
@requires_triton
@pytest.mark.parametrize("M", [1, 5, 40])
def test_split_k_is_deterministic_and_reusable(M):
    """Split-K partials are reduced in a fixed order by the last CTA, and the
    per-tile arrival counters are reset for the next launch."""
    from moe_infinity.kernel import mxfp4_gemm as mg

    N, K = 2880, 2880
    packed, scales = _weights(N, K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    first = mg.fused_mxfp4_gemm(x, packed, scales)
    for _ in range(20):
        assert torch.equal(mg.fused_mxfp4_gemm(x, packed, scales), first)
    _check(first, _reference(x, packed, scales, None))
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        again = mg.fused_mxfp4_gemm(x, packed, scales)
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(again, first)


@requires_cuda
@requires_triton
def test_no_compile_or_autotune_for_unseen_m_after_warmup():
    from moe_infinity.kernel import mxfp4_gemm as mg

    N, K = 640, 2880
    packed, scales = _weights(N, K)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    mg.warmup_mxfp4_gemm(N, K, torch.device("cuda"))

    def n_compiled():
        return sum(
            len(cache)
            for kern in (mg._mxfp4_gemv_kernel, mg._mxfp4_mma_kernel)
            for cache, *_ in kern.device_caches.values()
        )

    before = n_compiled()
    for M in (1, 3, 29, 77, 131, 333, 1500, 5000):
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        mg.fused_mxfp4_gemm(x, packed, scales, bias)
    torch.cuda.synchronize()
    assert n_compiled() == before
    assert not hasattr(mg._mxfp4_mma_kernel, "configs")  # not autotuned


def test_m_bucket_and_configs_cpu():
    from moe_infinity.kernel import mxfp4_gemm as mg

    assert [mg._m_bucket(m) for m in (1, 2, 3, 17, 4096, 10**6)] == [
        1,
        2,
        4,
        32,
        4096,
        4096,
    ]
    for bucket in (1 << i for i in range(13)):
        for N, K in ((5760, 2880), (2880, 2880), (64, 128)):
            cfg = mg._config(bucket, N, K, 188)
            assert cfg["kind"] == ("gemv" if bucket == 1 else "mma")
            assert cfg["SPLIT_K"] >= 1
