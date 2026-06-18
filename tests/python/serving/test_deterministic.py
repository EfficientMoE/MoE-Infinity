import importlib
import os
import sys

import torch
import torch.nn.functional as F

from tests.python.ops.conftest import requires_cuda

MODULE_NAME = "moe_infinity.kernel.deterministic_matmul"


def _load_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_enable_disable_deterministic_mode():
    module = _load_module()
    original_algorithms = torch.are_deterministic_algorithms_enabled()
    original_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    original_nccl = os.environ.get("NCCL_ALGO")

    module.enable_deterministic_mode()

    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["NCCL_ALGO"] == "Tree"

    module.disable_deterministic_mode()

    assert torch.are_deterministic_algorithms_enabled() == original_algorithms
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == original_cublas
    assert os.environ.get("NCCL_ALGO") == original_nccl


def test_deterministic_linear_cpu():
    module = _load_module()
    input = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 7
    bias = torch.arange(5, dtype=torch.float32) / 11

    output = module.deterministic_linear(input, weight, bias)
    expected = F.linear(input, weight, bias)

    # deterministic_linear computes per-sample GEMV for batch invariance, which
    # may differ from F.linear's batched GEMM by float32 rounding/BLAS backend;
    # bitwise equality is not part of the kernel's contract, closeness is.
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)


@requires_cuda
def test_deterministic_linear_batch_invariant():
    module = _load_module()
    input = torch.randn(8, 16, device="cuda")
    weight = torch.randn(32, 16, device="cuda")
    bias = torch.randn(32, device="cuda")

    batched = module.deterministic_linear(input, weight, bias)
    individual = torch.stack(
        [module.deterministic_linear(sample, weight, bias) for sample in input],
        dim=0,
    )

    assert torch.equal(batched, individual)


def test_env_var_auto_enable(monkeypatch):
    original_algorithms = torch.are_deterministic_algorithms_enabled()
    original_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    original_nccl = os.environ.get("NCCL_ALGO")

    monkeypatch.setenv("MOE_DETERMINISTIC", "1")
    module = _load_module()

    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["NCCL_ALGO"] == "Tree"

    module.disable_deterministic_mode()

    assert torch.are_deterministic_algorithms_enabled() == original_algorithms
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == original_cublas
    assert os.environ.get("NCCL_ALGO") == original_nccl
