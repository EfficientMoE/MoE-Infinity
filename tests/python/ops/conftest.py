import importlib.util

import pytest
import torch
import torch.nn.functional as F

# Auto-mark every test in tests/python/ops/ as ``gpu`` so that
# ``pytest -m "not gpu"`` deselects the entire directory in CPU-only CI.
pytestmark = pytest.mark.gpu

BF16_RTOL = 1e-2
BF16_ATOL = 1e-2
FP8_RTOL = 0.1
FP8_ATOL = 0.1
SHAPES = [(1, 128), (16, 512), (32, 2048)]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
requires_fp8 = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="FP8 not available"
)
requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="Triton not installed"
)


@pytest.fixture
def seed_everything():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def reference_silu_and_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.silu(x) * y


def reference_gelu_and_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.gelu(x) * y


def reference_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
