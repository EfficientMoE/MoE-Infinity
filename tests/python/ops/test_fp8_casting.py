import pytest

from tests.python.ops.conftest import (
    FP8_ATOL,
    FP8_RTOL,
    requires_cuda,
    requires_fp8,
    seed_everything,
)

torch = pytest.importorskip("torch")


def _cast_to_fp8_or_skip(tensor):
    try:
        return tensor.to(torch.float8_e4m3fn)
    except RuntimeError as exc:
        pytest.skip(f"FP8 cast not supported by current runtime/device: {exc}")


@requires_cuda
@requires_fp8
@pytest.mark.usefixtures(seed_everything.__name__)
@pytest.mark.parametrize("std", [0.02, 1.0, 0.001])
def test_fp8_cast_round_trip_no_nan_inf_and_within_tolerance(std):
    weight = (
        torch.randn(2048, 512, dtype=torch.bfloat16, device="cuda") * std
    ).contiguous()

    fp8_weight = _cast_to_fp8_or_skip(weight)
    recovered = fp8_weight.to(torch.bfloat16)

    assert not recovered.isnan().any(), "FP8 cast produced NaN"
    assert not recovered.isinf().any(), "FP8 cast produced Inf"
    torch.testing.assert_close(
        recovered,
        weight,
        rtol=FP8_RTOL,
        atol=FP8_ATOL,
    )


@requires_cuda
@requires_fp8
@pytest.mark.usefixtures(seed_everything.__name__)
@pytest.mark.parametrize("std", [0.02, 1.0, 0.001])
def test_fp8_recovered_weight_matmul_matches_bf16_reference(std):
    x = torch.randn(16, 512, dtype=torch.bfloat16, device="cuda").contiguous()
    weight_bf16 = (
        torch.randn(2048, 512, dtype=torch.bfloat16, device="cuda") * std
    ).contiguous()

    weight_fp8_recovered = _cast_to_fp8_or_skip(weight_bf16).to(torch.bfloat16)

    out_ref = x @ weight_bf16.t()
    out_fp8 = x @ weight_fp8_recovered.t()

    assert not out_fp8.isnan().any(), "FP8 recovered matmul produced NaN"
    assert not out_fp8.isinf().any(), "FP8 recovered matmul produced Inf"
    torch.testing.assert_close(
        out_fp8,
        out_ref,
        rtol=FP8_RTOL,
        atol=FP8_ATOL,
    )
