import importlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
)

torch = importlib.import_module("torch")
F = importlib.import_module("torch.nn.functional")


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYBIND_SRC = _REPO_ROOT / "core/python/py_archer_prefetch.cpp"
_SETUP_PY = _REPO_ROOT / "setup.py"

_SINGLE_EXPERT_CANDIDATES = (
    "fused_moe_ffn_into",
    "launch_fused_moe_ffn",
    "expert_fused_mlp",
)
_BATCHED_CANDIDATES = ("expert_fused_mlp_batched",)


def _read_store_binding_metadata() -> Tuple[List[str], List[str], bool]:
    pybind_text = _PYBIND_SRC.read_text(encoding="utf-8")
    setup_text = _SETUP_PY.read_text(encoding="utf-8")

    pybind_functions = sorted(
        set(re.findall(r'm\.def\("([^"]+)"', pybind_text))
    )
    pybind_classes = sorted(
        set(
            re.findall(
                r'py::class_<[^>]+>\(m,\s*"([^"]+)"\)',
                pybind_text,
            )
        )
    )

    setup_has_store_extension = (
        re.search(r'name\s*=\s*["\']moe_infinity\._store["\']', setup_text)
        is not None
    )

    return pybind_functions, pybind_classes, setup_has_store_extension


def _import_store_module():
    try:
        return importlib.import_module("moe_infinity._store"), None
    except Exception as exc:
        return None, str(exc)


def _reference_moe_mlp(x, gate_w, up_w, down_w):
    gate_out = F.silu(x @ gate_w.t())
    up_out = x @ up_w.t()
    return (gate_out * up_out) @ down_w.t()


def _reference_moe_mlp_batched(
    x,
    gate_w_all,
    up_w_all,
    down_w_all,
    expert_ids,
):
    output = torch.zeros((x.size(0), x.size(1)), dtype=x.dtype, device=x.device)
    for expert_idx in range(gate_w_all.size(0)):
        token_mask = expert_ids == expert_idx
        if not token_mask.any():
            continue
        output[token_mask] = _reference_moe_mlp(
            x[token_mask],
            gate_w_all[expert_idx],
            up_w_all[expert_idx],
            down_w_all[expert_idx],
        )
    return output


_PYBIND_FUNCTIONS, _PYBIND_CLASSES, _SETUP_HAS_STORE_EXTENSION = (
    _read_store_binding_metadata()
)
_STORE, _STORE_IMPORT_ERROR = _import_store_module()


def _skip_reason_if_direct_fused_not_callable() -> Optional[str]:
    if not _SETUP_HAS_STORE_EXTENSION:
        return "setup.py does not declare moe_infinity._store CUDA extension"

    if _STORE is None:
        return f"moe_infinity._store not importable: {_STORE_IMPORT_ERROR}"

    for name in _SINGLE_EXPERT_CANDIDATES:
        if name in _PYBIND_FUNCTIONS and hasattr(_STORE, name):
            return None

    exported_funcs = ", ".join(_PYBIND_FUNCTIONS)
    exported_classes = ", ".join(_PYBIND_CLASSES)
    return (
        "requires engine init: py_archer_prefetch.cpp exposes only "
        f"[{exported_funcs}] (+ classes [{exported_classes}]); "
        "no direct fused MoE MLP Python binding is exported from "
        "moe_infinity._store"
    )


def _resolve_single_expert_binding_name() -> str:
    if _STORE is None:
        pytest.skip(
            f"moe_infinity._store not importable: {_STORE_IMPORT_ERROR}"
        )

    for name in _SINGLE_EXPERT_CANDIDATES:
        if hasattr(_STORE, name):
            return name

    reason = _skip_reason_if_direct_fused_not_callable()
    if reason is None:
        reason = (
            "no direct fused single-expert entrypoint in moe_infinity._store"
        )
    pytest.skip(reason)


def _resolve_batched_binding_name() -> str:
    if _STORE is None:
        pytest.skip(
            f"moe_infinity._store not importable: {_STORE_IMPORT_ERROR}"
        )

    for name in _BATCHED_CANDIDATES:
        if hasattr(_STORE, name):
            return name

    pytest.skip(
        "requires engine init or unexposed batched entrypoint: "
        "moe_infinity._store has no expert_fused_mlp_batched"
    )


def _call_single_expert_fused(binding_name: str, x, gate_w, up_w, down_w):
    assert _STORE is not None
    if binding_name == "fused_moe_ffn_into":
        gate_buf = torch.empty(
            x.size(0), gate_w.size(0), dtype=x.dtype, device=x.device
        )
        fused_buf = torch.empty_like(gate_buf)
        output = torch.empty_like(x)
        _STORE.fused_moe_ffn_into(
            x,
            gate_w,
            up_w,
            down_w,
            gate_buf,
            fused_buf,
            output,
            None,
        )
        return output

    fused_fn = getattr(_STORE, binding_name)
    try:
        return fused_fn(x, gate_w, up_w, down_w)
    except TypeError:
        if binding_name == "launch_fused_moe_ffn":
            return fused_fn(x, gate_w, up_w, down_w, None)
        raise


_SKIP_REASON = _skip_reason_if_direct_fused_not_callable()


@requires_cuda
@pytest.mark.parametrize(
    "tokens,hidden_dim,intermediate_dim",
    [(8, 128, 256), (32, 256, 512)],
)
def test_fused_moe_mlp_single_expert_matches_reference(
    seed_everything,
    tokens,
    hidden_dim,
    intermediate_dim,
):
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)

    binding_name = _resolve_single_expert_binding_name()

    x = torch.randn(
        tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    gate_w = torch.randn(
        intermediate_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    up_w = torch.randn(
        intermediate_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    down_w = torch.randn(
        hidden_dim, intermediate_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    fused_out = _call_single_expert_fused(binding_name, x, gate_w, up_w, down_w)
    reference_out = _reference_moe_mlp(x, gate_w, up_w, down_w)

    torch.testing.assert_close(
        fused_out,
        reference_out,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )


@requires_cuda
def test_fused_moe_mlp_batched_matches_reference(seed_everything):
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)

    binding_name = _resolve_batched_binding_name()

    tokens, hidden_dim, intermediate_dim, num_experts = 24, 128, 256, 4

    x = torch.randn(
        tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    gate_w_all = torch.randn(
        num_experts,
        intermediate_dim,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    up_w_all = torch.randn(
        num_experts,
        intermediate_dim,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    down_w_all = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    expert_ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int64) % num_experts
    ).contiguous()

    fused_batched_fn = getattr(_STORE, binding_name)
    fused_out = fused_batched_fn(
        x, gate_w_all, up_w_all, down_w_all, expert_ids
    )
    reference_out = _reference_moe_mlp_batched(
        x,
        gate_w_all,
        up_w_all,
        down_w_all,
        expert_ids,
    )

    torch.testing.assert_close(
        fused_out,
        reference_out,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
