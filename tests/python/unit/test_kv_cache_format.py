import pytest
import torch

from moe_infinity.runtime.kv_cache_format import (
    KVCacheBackendCapabilities,
    KVCacheFormat,
    KVCacheModelInfo,
    quantize_tokenwise_symmetric,
    resolve_kv_cache_format,
)


def test_int8_sym_calibration_round_trip_bound() -> None:
    x = torch.tensor([[[0.0, -1.0, 0.25, 2.0, 127.03125]]], dtype=torch.float32)
    q, scale = quantize_tokenwise_symmetric(x)
    restored = q.float() * scale.float().unsqueeze(-1)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert int(q.min()) >= -127
    assert torch.all(torch.isfinite(scale))
    assert torch.all(127 * scale.float() >= x.abs().amax(dim=-1))
    assert torch.all(
        (restored - x).abs() <= scale.float().unsqueeze(-1) / 2 + 1e-6
    )


def test_int8_sym_zero_and_fp16_rounding_adversaries() -> None:
    raw_scales = torch.tensor([0.0, 1.0001, 63.999, 65504.0])
    x = torch.stack((raw_scales * 127, -raw_scales * 127), dim=-1)
    q, scale = quantize_tokenwise_symmetric(x)
    restored = q.float() * scale.float().unsqueeze(-1)
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert torch.all(127 * scale.float() >= x.abs().amax(dim=-1))
    assert torch.all(
        (restored - x).abs() <= scale.float().unsqueeze(-1) / 2 + 1e-6
    )


def test_int8_sym_rejects_nonfinite_or_unrepresentable_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        quantize_tokenwise_symmetric(torch.tensor([[float("inf")]]))
    with pytest.raises(ValueError, match="FP16 scale"):
        quantize_tokenwise_symmetric(torch.tensor([[65504.0 * 128.0]]))


def test_int8_sym_page_bytes_include_scales() -> None:
    fmt = KVCacheFormat.parse("int8_sym")
    assert (
        fmt.page_size_bytes(block_size=16, num_kv_heads=8, head_dim=128)
        == 33280
    )


@pytest.mark.parametrize(
    "device,backend,capabilities,expected_backend,expected_reason",
    [
        (
            torch.device("cuda"),
            "native",
            KVCacheBackendCapabilities(False, True, True, None),
            "native_int8",
            None,
        ),
        (
            torch.device("cpu"),
            "auto",
            KVCacheBackendCapabilities(
                False, False, True, "native_int8_module_unavailable"
            ),
            "sdpa_dequant",
            "cpu_sdpa_dequant",
        ),
        (
            torch.device("cuda"),
            "auto",
            KVCacheBackendCapabilities(
                False, False, True, "native_int8_binding_missing"
            ),
            "sdpa_dequant",
            "native_int8_binding_missing",
        ),
        (
            torch.device("cuda"),
            "flashinfer",
            KVCacheBackendCapabilities(True, True, True, None),
            "native_int8",
            "flashinfer_no_int8_sym_contract",
        ),
    ],
)
def test_resolver_consumes_device_kernel_and_backend_capability(
    device, backend, capabilities, expected_backend, expected_reason
) -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(32, 8, 128, False),
        device=device,
        backend_preference=backend,
        capabilities=capabilities,
        allow_fallback=True,
    )
    assert decision.effective_format.name == "int8_sym"
    assert decision.execution_backend == expected_backend
    assert decision.reason == expected_reason


def test_mla_request_falls_back_visibly() -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(
            num_attention_heads=16, num_kv_heads=1, head_dim=512, is_mla=True
        ),
        device=torch.device("cuda"),
        backend_preference="auto",
        capabilities=KVCacheBackendCapabilities(False, True, True, None),
        allow_fallback=True,
    )
    assert decision.effective_format.name == "native"
    assert decision.reason == "mla_not_validated"


def test_unknown_format_fails_before_allocation() -> None:
    with pytest.raises(ValueError, match="unsupported KV cache format"):
        KVCacheFormat.parse("int4")
