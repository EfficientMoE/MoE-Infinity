"""Numerics and dispatch of the fused MXFP4 GEMM (GPT-OSS resident path, #312)."""

import pytest

from tests.python.ops.conftest import requires_cuda, requires_triton

torch = pytest.importorskip("torch")


def _gated():
    from moe_infinity.kernel import mxfp4_gemm as mg

    return torch.cuda.is_available() and mg._pro6000_sm120(torch.device("cuda"))


# Tests of the SM120 / RTX PRO 6000 kernels themselves (the public entry
# point is covered on every GPU by the parity tests below).
requires_pro6000 = pytest.mark.skipif(
    not _gated(),
    reason="needs an RTX PRO 6000 Blackwell (SM120 + PRO6000 gate)",
)


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


@requires_pro6000
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


@requires_pro6000
@requires_cuda
@requires_triton
@pytest.mark.parametrize(
    "N,K", [(5760, 2880), (2880, 2880), (40, 64), (100, 4128)]
)
@pytest.mark.parametrize("M", [2, 3, 5, 8])
def test_small_m_cuda_kernel_matches_triton_and_reference(monkeypatch, N, K, M):
    """M = 2..8 decode kernel (moe_infinity._mxfp4_gemv, full-K strip per
    CTA) agrees with the reference and with the Triton fallback."""
    from moe_infinity.kernel import mxfp4_gemm as mg

    if (
        not mg._small_m_sm120_ext()
        or torch.cuda.get_device_capability()[0] < 10
    ):
        pytest.skip("moe_infinity._mxfp4_gemv needs an sm_100a+ build")
    packed, scales = _weights(N, K, lo=100, hi=150)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    assert mg._small_m_sm120_cuda(x, packed, scales, bias, out)
    ref = _reference(x, packed, scales, bias)
    _check(out, ref)
    monkeypatch.setattr(mg, "_DISABLE_SMALL_M_SM120", True)
    _check(mg.fused_mxfp4_gemm(x, packed, scales, bias), ref)
    # outside the contract -> declines without launching
    big = torch.randn(9, K, dtype=torch.bfloat16, device="cuda")
    assert not mg._small_m_sm120_ext().mxfp4_gemv_small(
        big,
        packed,
        scales,
        bias,
        torch.empty(9, N, dtype=torch.bfloat16, device="cuda"),
        2,
    )


@requires_pro6000
@requires_cuda
@requires_triton
@pytest.mark.parametrize("dp", ["full", "minus1", "none"])
@pytest.mark.parametrize("M", [1600, 2900])
def test_stream_k_schedules_match_reference(dp, M):
    """Persistent DP + stream-K hybrid: tiles split across CTAs are reduced
    deterministically by the last arriving CTA."""
    from moe_infinity.kernel import mxfp4_gemm as mg

    N, K = 2880, 2880
    packed, scales = _weights(N, K)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    cfg = dict(
        kind="streamk",
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=4,
        num_stages=2,
        DP=dp,
        CVT=True,
        SPLIT_K=1,
    )
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    mg._launch(x, packed, scales, bias, out, cfg)
    _check(out, _reference(x, packed, scales, bias))
    again = torch.empty_like(out)
    mg._launch(x, packed, scales, bias, again, cfg)
    assert torch.equal(again, out)


@requires_pro6000
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
            for kern in (
                mg._mxfp4_gemv_sm120_kernel,
                mg._mxfp4_mma_sm120_kernel,
                mg._mxfp4_mma_streamk_sm120_kernel,
            )
            for cache, *_ in kern.device_caches.values()
        )

    before = n_compiled()
    for M in (1, 3, 29, 77, 131, 333, 1500, 5000):
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        mg.fused_mxfp4_gemm(x, packed, scales, bias)
    torch.cuda.synchronize()
    assert n_compiled() == before
    assert not hasattr(mg._mxfp4_mma_sm120_kernel, "configs")  # not autotuned


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
            cfg = mg._pro6000_config(bucket, N, K, 188)
            if bucket == 1:
                assert cfg["kind"] == "gemv"
            elif bucket >= 1024 and N < 4096:
                assert cfg["kind"] == "streamk"
            else:
                assert cfg["kind"] == "mma"
            assert cfg["SPLIT_K"] >= 1


@requires_cuda
@requires_triton
@pytest.mark.parametrize("M", [1, 5, 300])
def test_generic_path_matches_reference(M):
    """The previous kernel, which every non-RTX-PRO-6000 GPU runs."""
    from moe_infinity.kernel.mxfp4_gemm_generic import fused_mxfp4_gemm_generic

    N, K = 96, 256
    packed, scales = _weights(N, K)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    _check(
        fused_mxfp4_gemm_generic(x, packed, scales, bias),
        _reference(x, packed, scales, bias),
    )


_FAKE_GPUS = {
    "rtx_pro_6000": dict(mp=188, l2=128, cap=(12, 0)),
    "rtx_5090": dict(mp=170, l2=96, cap=(12, 0)),
    "h100": dict(mp=132, l2=50, cap=(9, 0)),
}


@pytest.mark.parametrize("name", list(_FAKE_GPUS))
def test_dispatch_is_hardware_gated_cpu(monkeypatch, name):
    """Only the RTX PRO 6000 (SM120 + RTX_PRO_6000) takes the new kernels;
    an H100 and an RTX 5090 run the previous (generic) implementation."""
    from types import SimpleNamespace

    from moe_infinity.kernel import mxfp4_gemm as mg
    from moe_infinity.utils import hw_profile

    g = _FAKE_GPUS[name]
    props = SimpleNamespace(
        major=g["cap"][0],
        minor=g["cap"][1],
        multi_processor_count=g["mp"],
        L2_cache_size=g["l2"] * 1024 * 1024,
        shared_memory_per_block_optin=99 * 1024
        if g["cap"][0] == 12
        else 227 * 1024,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.delenv("MOE_INFINITY_UNVALIDATED_HW", raising=False)
    monkeypatch.setattr(mg, "_GATED", {})
    hw_profile._gates.cache_clear()
    try:
        calls = []
        monkeypatch.setattr(
            mg,
            "fused_mxfp4_gemm_generic",
            lambda *a, **k: calls.append("generic") or "generic",
        )
        monkeypatch.setattr(
            mg, "warmup_mxfp4_gemm", lambda *a, **k: calls.append("gated")
        )
        monkeypatch.setattr(
            mg, "_launch", lambda *a, **k: calls.append("launch")
        )
        monkeypatch.setattr(mg, "_small_m_sm120_cuda", lambda *a, **k: False)
        x = torch.zeros(3, 64, dtype=torch.bfloat16)
        monkeypatch.setattr(
            mg,
            "_pro6000_sm120",
            lambda d, f=mg._pro6000_sm120: f(torch.device("cuda", 0)),
        )
        w = torch.zeros(16, 32, dtype=torch.uint8)
        s = torch.zeros(16, 2, dtype=torch.uint8)
        out = mg.fused_mxfp4_gemm(x, w, s)
        gated = name == "rtx_pro_6000"
        assert mg._GATED[0] is gated
        if gated:
            assert calls[:2] == ["gated", "launch"]
        else:
            assert calls == ["generic"] and out == "generic"
    finally:
        hw_profile._gates.cache_clear()
