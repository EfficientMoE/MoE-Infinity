"""Run custom-kernel entry points on the device of their tensors.

Triton launches (and native kernels that call ``getCurrentCUDAStream()`` or
pass the NULL stream) use the *current* CUDA device of the calling thread. With
the model's layers spread over several GPUs that is often not the device of the
tensors, and the kernel then runs in the other GPU's context on that GPU's
default stream: peer access lets it touch the tensors, but nothing orders it
against the stream that produces its inputs or consumes its outputs.

``device_guard`` makes the device of the first CUDA tensor among the arguments
current for the duration of the call and restores the previous device
afterwards. When that device is already current (single GPU, or a correct
multi-GPU caller) it is a plain call.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable)

# torch._C._cuda_getDevice is the cheap form of torch.cuda.current_device().
# CPU-only builds of torch do not have it; there no CUDA tensor can reach
# the comparison in device_guard, so the fallback is never called.
_get_device = getattr(torch._C, "_cuda_getDevice", torch.cuda.current_device)


def _scan(items) -> int | None:
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                return item.get_device()
        elif isinstance(item, torch.device):
            if item.type == "cuda" and item.index is not None:
                return item.index
        elif isinstance(item, (list, tuple)):
            index = _scan(item)
            if index is not None:
                return index
    return None


def tensor_device_index(args: tuple, kwargs: dict) -> int | None:
    """CUDA device index of the first CUDA tensor (or ``torch.device``) in
    ``args`` / ``kwargs`` (lists and tuples are searched one level deep, and
    nested ones recursively)."""
    index = _scan(args)
    if index is None and kwargs:
        index = _scan(kwargs.values())
    return index


def device_guard(fn: F) -> F:
    """Decorator: run ``fn`` with its tensors' CUDA device current."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        index = tensor_device_index(args, kwargs)
        if index is None or index == _get_device():
            return fn(*args, **kwargs)
        with torch.cuda.device(index):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
