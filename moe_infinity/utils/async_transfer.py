from __future__ import annotations

from typing import cast

import torch


def async_d2h(tensor: torch.Tensor, stream: object) -> torch.Tensor:
    """Non-blocking GPU→CPU copy. Returns CPU tensor."""
    cuda_stream = cast(torch.cuda.Stream, stream)
    with torch.cuda.stream(cuda_stream):
        return tensor.to("cpu", non_blocking=True)


def async_h2d(
    tensor: torch.Tensor,
    device: torch.device,
    stream: object,
) -> torch.Tensor:
    """Non-blocking CPU→GPU copy. Returns GPU tensor."""
    cuda_stream = cast(torch.cuda.Stream, stream)
    with torch.cuda.stream(cuda_stream):
        return tensor.to(device, non_blocking=True)


def wait_transfer(stream: object) -> None:
    """Synchronize the stream — wait for all pending copies."""
    cuda_stream = cast(torch.cuda.Stream, stream)
    cuda_stream.synchronize()
