"""Kernel report for linear-attention layers (Gated DeltaNet, KDA).

Qwen3.5 / Qwen3-Next Gated DeltaNet and GLM-5.3-Flash KDA layers come from
``transformers``, which binds the chunked Triton kernels of
``flash-linear-attention`` (``fla``) when that package is importable and
otherwise falls back to pure-torch reference loops (a Python loop over
64-token chunks; ~20x slower at 9K tokens on sm_120, see #304).

Transformers wires the kernels in two ways, both resolved here:

* transformers <= 5.12 (``qwen3_5_moe``): instance attributes such as
  ``self.chunk_gated_delta_rule`` set in ``__init__``;
* transformers >= 5.16 (``qwen3_5_moe``, ``glm5_next``): module-level
  functions wrapped by ``use_kernel_func_from_hub_with_fallback`` whose
  closure holds the resolved implementation.

Install the kernels with ``pip install 'flash-linear-attention>=0.6.0'``
(or the ``linear_attn`` extra). ``causal-conv1d`` is optional: without it
the short convolution runs as a cuDNN depthwise conv.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import sys
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "pip install 'flash-linear-attention>=0.6.0' "
    "(or pip install 'moe-infinity[linear_attn]')"
)

# role -> (instance attributes, module-level globals), checked in order.
_ROLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "chunk": (
        ("chunk_gated_delta_rule",),
        (
            "torch_chunk_gated_delta_rule",
            "chunk_kimi_delta_attention",
        ),
    ),
    "recurrent": (
        ("recurrent_gated_delta_rule",),
        (
            "torch_recurrent_gated_delta_rule",
            "recurrent_kimi_delta_attention",
        ),
    ),
    "conv": (("causal_conv1d_fn",), ("causal_conv1d_fn",)),
}
# Roles whose torch fallback costs O(T) Python-driven launches.
_REQUIRED_ROLES = ("chunk", "recurrent")
_CLASS_MARKERS = ("GatedDeltaNet", "LinearAttention", "DeltaAttention")
_warned: set[str] = set()


def _backend_of(fn: Callable | None, depth: int = 0) -> str:
    """What implements ``fn``: ``fused`` (MoE-Infinity fused kernels, which
    fall back to fla), ``fla``, ``causal_conv1d`` or ``torch``.

    Follows ``functools.wraps`` chains and closure cells, which is where
    ``use_kernel_func_from_hub_with_fallback`` keeps the implementation.
    """
    if fn is None:
        return "torch"
    tag = getattr(fn, "_moe_backend", None)
    if tag:
        return tag
    mod = getattr(fn, "__module__", None) or ""
    for pkg in ("fla", "causal_conv1d"):
        if mod == pkg or mod.startswith(pkg + "."):
            return pkg
    if depth >= 4:
        return "torch"
    inner: list = []
    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None:
        inner.append(wrapped)
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            inner.append(cell.cell_contents)
        except ValueError:  # empty cell
            continue
    for obj in inner:
        if callable(obj) and obj is not fn:
            found = _backend_of(obj, depth + 1)
            if found != "torch":
                return found
    return "torch"


def _resolve_role(module: torch.nn.Module, role: str) -> tuple[bool, str]:
    attrs, globals_ = _ROLES[role]
    for name in attrs:
        if name in vars(module) or hasattr(type(module), name):
            return True, _backend_of(getattr(module, name))
    defining = sys.modules.get(type(module).__module__)
    for name in globals_:
        fn = getattr(defining, name, None)
        if fn is not None:
            return True, _backend_of(fn)
    return False, "torch"


def is_linear_attention_module(module: torch.nn.Module) -> bool:
    name = type(module).__name__
    if not any(marker in name for marker in _CLASS_MARKERS):
        return False
    return _resolve_role(module, "chunk")[0]


def linear_attention_kernels(model: torch.nn.Module) -> dict[str, dict]:
    """Per linear-attention class: bound backend per role and layer count.

    Returns ``{class_name: {"chunk": "fla"|"torch", "recurrent": ...,
    "conv": "fla"|"causal_conv1d"|"torch", "layers": n}}``; empty when the
    model has no linear-attention layers.
    """
    report: dict[str, dict] = {}
    for module in model.modules():
        if not is_linear_attention_module(module):
            continue
        name = type(module).__name__
        if name in report:
            report[name]["layers"] += 1
            continue
        entry: dict = {"layers": 1}
        for role in _ROLES:
            entry[role] = _resolve_role(module, role)[1]
        report[name] = entry
    return report


# transformers modeling modules with linear-attention layers
_HF_MODULES = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_next.modeling_qwen3_next",
    "transformers.models.glm5_next.modeling_glm5_next",
)
# global -> torch reference global in the same module (None: unwrap itself)
_DISPATCH_GLOBALS = {
    # transformers <= 5.12: fla functions imported as module globals
    "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule",
    # transformers >= 5.16: hub-with-fallback wrappers around the torch refs
    "torch_chunk_gated_delta_rule": None,
    "torch_recurrent_gated_delta_rule": None,
    "chunk_kimi_delta_attention": None,
    "recurrent_kimi_delta_attention": None,
}


def _first_tensor(args, kwargs) -> torch.Tensor | None:
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for key in ("query", "q", "hidden_states", "x"):
        if isinstance(kwargs.get(key), torch.Tensor):
            return kwargs[key]
    return None


def _device_dispatch(fast: Callable, ref: Callable) -> Callable:
    @functools.wraps(fast)
    def dispatch(*args, **kwargs):
        x = _first_tensor(args, kwargs)
        if x is not None and not x.is_cuda:
            return ref(*args, **kwargs)
        return fast(*args, **kwargs)

    dispatch._moe_device_safe = True  # type: ignore[attr-defined]
    return dispatch


def _device_safe_norm(fused_cls: type) -> type:
    class DeviceSafeFusedRMSNormGated(fused_cls):
        """fla gated RMSNorm with a torch path for non-CUDA inputs."""

        _moe_device_safe = True

        def forward(self, x, g=None, *args, **kwargs):
            if x.is_cuda:
                return super().forward(x, g, *args, **kwargs)
            dtype = x.dtype
            h = x.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            if self.weight is not None:
                h = h * self.weight.to(x.device, torch.float32)
            if g is not None:
                act = torch.sigmoid if self.activation == "sigmoid" else F.silu
                h = h * act(g.float())
            return h.to(dtype)

    DeviceSafeFusedRMSNormGated.__name__ = fused_cls.__name__
    DeviceSafeFusedRMSNormGated.__qualname__ = fused_cls.__qualname__
    return DeviceSafeFusedRMSNormGated


def make_linear_attention_device_safe() -> list[str]:
    """Route fla-bound linear-attention calls on non-CUDA tensors to torch.

    transformers binds the Triton kernels whenever fla is installed and CUDA
    exists, even for models that run on CPU (tiny test models, CPU
    reference runs), where Triton fails with ``cannot be accessed from
    Triton (cpu tensor?)``. This wraps the bound kernels with a device check
    (CUDA inputs keep the fla kernels). Idempotent; returns the patched
    ``module.global`` names. Call before building transformers <= 5.12
    models (they bind the kernels in ``__init__``).
    """
    patched: list[str] = []
    for mod_name in _HF_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - model family absent in this transformers
            continue
        for name, ref_name in _DISPATCH_GLOBALS.items():
            fast = getattr(mod, name, None)
            if fast is None or getattr(fast, "_moe_device_safe", False):
                continue
            if _backend_of(fast) == "torch":
                continue
            ref = getattr(mod, ref_name) if ref_name else fast
            ref = inspect.unwrap(ref)
            if _backend_of(ref) != "torch":
                continue
            setattr(mod, name, _device_dispatch(fast, ref))
            patched.append(f"{mod_name}.{name}")
        norm = getattr(mod, "FusedRMSNormGated", None)
        if isinstance(norm, type) and not getattr(
            norm, "_moe_device_safe", False
        ):
            mod.FusedRMSNormGated = _device_safe_norm(norm)
            patched.append(f"{mod_name}.FusedRMSNormGated")
    return patched


# chunk-kernel globals -> fused kind (GDN scalar gate / KDA per-channel gate)
_FUSED_GLOBALS = {
    "chunk_gated_delta_rule": "gdn",  # transformers <= 5.12
    "torch_chunk_gated_delta_rule": "gdn",  # transformers >= 5.16
    "chunk_kimi_delta_attention": "kda",
}


def install_fused_linear_attention() -> list[str]:
    """Route the fla-bound chunked GDN / KDA prefill kernels of the
    transformers linear-attention modules through the fused sm_120 path
    (``linear_attention_fused``); unsupported calls still reach fla.

    Only globals that already resolve to fla are wrapped, so without fla
    installed nothing changes. Idempotent. ``MOE_LINEAR_ATTN_FUSED=0``
    disables the fused path at call time.
    """
    from moe_infinity.kernel.linear_attention_fused import make_fused

    patched: list[str] = []
    for mod_name in _HF_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - model family absent
            continue
        for name, kind in _FUSED_GLOBALS.items():
            fn = getattr(mod, name, None)
            if fn is None or _backend_of(fn) != "fla":
                continue
            setattr(mod, name, make_fused(kind, fn))
            patched.append(f"{mod_name}.{name}")
    return patched


def _bind_instances(model: torch.nn.Module) -> int:
    """transformers <= 5.12 binds ``self.chunk_gated_delta_rule`` in
    ``__init__``: point existing layers at the (now fused) module global."""
    n = 0
    for module in model.modules():
        fn = vars(module).get("chunk_gated_delta_rule")
        if fn is None or _backend_of(fn) != "fla":
            continue
        new = getattr(
            sys.modules.get(type(module).__module__),
            "chunk_gated_delta_rule",
            None,
        )
        if new is not None and _backend_of(new) == "fused":
            module.chunk_gated_delta_rule = new
            n += 1
    return n


def enable_linear_attention_kernels(model: torch.nn.Module) -> dict[str, dict]:
    """Install the fused path on ``model``'s linear-attention layers, then
    report the bound kernels (warning once on a torch fallback)."""
    if torch.cuda.is_available():
        try:
            install_fused_linear_attention()
            _bind_instances(model)
        except Exception as exc:  # noqa: BLE001 - keep fla / torch path
            logger.warning("fused linear-attention path unavailable: %s", exc)
    return check_linear_attention_kernels(model)


def check_linear_attention_kernels(model: torch.nn.Module) -> dict[str, dict]:
    """Log the bound linear-attention kernels; warn once on torch fallback."""
    report = linear_attention_kernels(model)
    for name, entry in report.items():
        slow = [r for r in _REQUIRED_ROLES if entry[r] == "torch"]
        if slow and torch.cuda.is_available():
            if name not in _warned:
                _warned.add(name)
                logger.warning(
                    "%s (%d layers) runs the transformers torch fallback for "
                    "%s: prefill is a Python loop over 64-token chunks "
                    "(~20x slower at 9K tokens). Install the chunked Triton "
                    "kernels: %s",
                    name,
                    entry["layers"],
                    "/".join(slow),
                    INSTALL_HINT,
                )
        else:
            logger.info("%s linear-attention kernels: %s", name, entry)
    return report
