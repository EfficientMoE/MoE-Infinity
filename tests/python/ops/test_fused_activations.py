"""
Numerical consistency tests for fused activation kernels.

Tests compare the C++ fused activation kernels against vanilla PyTorch reference
implementations for:
- silu_and_mul: ACT_FN(gate) * up
- gelu_and_mul: ACT_FN(gate) * up
- gelu_tanh_and_mul: ACT_FN(gate) * up
- fatrelu_and_mul: ACT_FN(gate) * up

Note: These kernels are defined in activation_kernels.cu but may not be
directly exposed via Python bindings. Tests will skip if functions are unavailable.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    SHAPES,
    requires_cuda,
    seed_everything,
)


def try_import_store():
    """Try to import moe_infinity._store, return None if unavailable."""
    try:
        import moe_infinity._store as store

        return store
    except (ImportError, AttributeError):
        return None


# Attempt to import _store module
_store = try_import_store()


def reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Reference: silu(gate) * up where gate=x[..., :d], up=x[..., d:].
    C++ implementation uses act_first=True: ACT_FN(first_half) * second_half.
    """
    d = x.size(-1) // 2
    gate = x[..., :d]
    up = x[..., d:]
    return F.silu(gate) * up


def reference_gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Reference: gelu(gate) * up where gate=x[..., :d], up=x[..., d:].
    C++ implementation uses act_first=True: ACT_FN(first_half) * second_half.
    Uses default gelu approximation ('none').
    """
    d = x.size(-1) // 2
    gate = x[..., :d]
    up = x[..., d:]
    return F.gelu(gate) * up


def reference_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Reference: gelu(gate, approximate='tanh') * up where gate=x[..., :d], up=x[..., d:].
    C++ implementation uses act_first=True: ACT_FN(first_half) * second_half.
    """
    d = x.size(-1) // 2
    gate = x[..., :d]
    up = x[..., d:]
    return F.gelu(gate, approximate="tanh") * up


def reference_fatrelu_and_mul(
    x: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Reference: fatrelu(gate, threshold) * up where gate=x[..., :d], up=x[..., d:].
    C++ implementation applies threshold to gate before multiplication.
    fatrelu(x, t) = x if x > t else 0
    """
    d = x.size(-1) // 2
    gate = x[..., :d]
    up = x[..., d:]
    # F.relu doesn't support threshold parameter in the same way
    # Using torch.where for exact replication of C++ fatrelu_kernel
    return torch.where(gate > threshold, gate, torch.zeros_like(gate)) * up


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_silu_and_mul_matches_reference(seed_everything, shape):
    """Test silu_and_mul kernel output matches F.silu(gate) * up."""
    if _store is None or not hasattr(_store, "silu_and_mul"):
        pytest.skip("silu_and_mul not exposed via Python bindings")

    # Input shape: [..., 2*d] where d = shape[-1] // 2
    d = shape[-1] // 2
    input_tensor = torch.randn(
        *shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    # Allocate output tensor
    output = torch.empty(shape[:-1] + (d,), dtype=torch.bfloat16, device="cuda")

    # Call C++ kernel
    _store.silu_and_mul(output, input_tensor)

    # Reference implementation
    expected = reference_silu_and_mul(input_tensor)

    torch.testing.assert_close(output, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_and_mul_matches_reference(seed_everything, shape):
    """Test gelu_and_mul kernel output matches F.gelu(gate) * up."""
    if _store is None or not hasattr(_store, "gelu_and_mul"):
        pytest.skip("gelu_and_mul not exposed via Python bindings")

    d = shape[-1] // 2
    input_tensor = torch.randn(
        *shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    output = torch.empty(shape[:-1] + (d,), dtype=torch.bfloat16, device="cuda")

    _store.gelu_and_mul(output, input_tensor)

    expected = reference_gelu_and_mul(input_tensor)

    torch.testing.assert_close(output, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_tanh_and_mul_matches_reference(seed_everything, shape):
    """Test gelu_tanh_and_mul kernel matches F.gelu(gate, tanh) * up."""
    if _store is None or not hasattr(_store, "gelu_tanh_and_mul"):
        pytest.skip("gelu_tanh_and_mul not exposed via Python bindings")

    d = shape[-1] // 2
    input_tensor = torch.randn(
        *shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    output = torch.empty(shape[:-1] + (d,), dtype=torch.bfloat16, device="cuda")

    _store.gelu_tanh_and_mul(output, input_tensor)

    expected = reference_gelu_tanh_and_mul(input_tensor)

    torch.testing.assert_close(output, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("threshold", [0.0, 1.0, 2.0])
def test_fatrelu_and_mul_matches_reference(seed_everything, shape, threshold):
    """Test fatrelu_and_mul kernel matches fatrelu(gate, threshold) * up."""
    if _store is None or not hasattr(_store, "fatrelu_and_mul"):
        pytest.skip("fatrelu_and_mul not exposed via Python bindings")

    d = shape[-1] // 2
    input_tensor = torch.randn(
        *shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    output = torch.empty(shape[:-1] + (d,), dtype=torch.bfloat16, device="cuda")

    _store.fatrelu_and_mul(output, input_tensor, threshold)

    expected = reference_fatrelu_and_mul(input_tensor, threshold)

    torch.testing.assert_close(output, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
