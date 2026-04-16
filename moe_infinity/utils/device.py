from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_num_devices() -> int:
    if not is_cuda_available():
        return 0
    return torch.cuda.device_count()


def get_default_device() -> str:
    if is_cuda_available() and get_num_devices() > 0:
        return "cuda:0"
    return "cpu"


def get_device(device_id: Optional[int] = None) -> str:
    if device_id is None:
        return get_default_device()
    if is_cuda_available() and 0 <= device_id < get_num_devices():
        return f"cuda:{device_id}"
    return "cpu"


def get_pinned_memory_device() -> str:
    return "cpu"


def to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device)


@dataclass
class DeviceConfig:
    default_device: str
    offload_device: str = "cpu"
    num_devices: int = 0
