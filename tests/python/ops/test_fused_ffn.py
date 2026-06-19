import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    requires_triton,
)

torch = importlib.import_module("torch")


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "moe_infinity/kernel/fused_ffn.py"


@lru_cache(maxsize=1)
def _load_fused_ffn_module():
    spec = importlib.util.spec_from_file_location(
        "standalone_fused_ffn",
        _MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_cuda
@requires_triton
@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("intermediate_size", [5504, 11008])
@pytest.mark.parametrize("M", [1, 4, 16, 64])
def test_fused_ffn_matches_reference(
    seed_everything,
    M,
    hidden_dim,
    intermediate_size,
):
    fused_ffn_module = _load_fused_ffn_module()

    x = torch.randn(
        M, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    gate_w = torch.randn(
        intermediate_size,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    up_w = torch.randn(
        intermediate_size,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    down_w = torch.randn(
        hidden_dim,
        intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()

    fused = fused_ffn_module.fused_ffn(x, gate_w, up_w, down_w)
    ref = fused_ffn_module.reference_ffn(x, gate_w, up_w, down_w)

    assert torch.allclose(fused, ref, atol=BF16_ATOL, rtol=BF16_RTOL)
