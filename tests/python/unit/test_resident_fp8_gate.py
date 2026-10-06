import pytest

from benchmarks.adaptive_precision.resident_fp8_gate import (
    KERNEL_MATRIX,
    nll_gate_passes,
    resident_ratio_passes,
)


def test_kernel_matrix_is_the_existing_blockwise_path():
    assert KERNEL_MATRIX == {
        "format": "fp8_e4m3_block128",
        "execution": "fp8_dequant_bf16_gemm",
        "block_size": 128,
        "output_dtype": "bfloat16",
        "architecture_floor": None,
        "online_tier_changes": False,
        "uniform_per_dispatch": True,
    }


def test_resident_ratio_gate_is_1_8x():
    assert resident_ratio_passes(18, 10) is True
    assert resident_ratio_passes(17, 10) is False


def test_nll_gate_allows_one_percent_worse():
    assert nll_gate_passes(100.0, 101.0) is True
    assert nll_gate_passes(100.0, 101.0 + 0.1) is False


def test_gates_reject_nonpositive_baselines():
    with pytest.raises(ValueError):
        resident_ratio_passes(18, 0)
    with pytest.raises(ValueError):
        nll_gate_passes(0.0, 1.0)


def test_nll_gate_rejects_negative_fp8_measurement():
    with pytest.raises(ValueError, match="NLL values"):
        nll_gate_passes(1.0, -0.1)
