# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Engine-hooks boundary consumed by model integration code.

``EngineHooks`` is the narrow surface model wrappers and the offload
runtime use to talk to the offloading engine (acquire/release tensors,
blocking fetches, prefetch hints). Model-side code moving to the moe-store
repository must depend only on this protocol, never on ``_store`` directly.
"""

from typing import Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class EngineHooks(Protocol):
    def begin(
        self, request_id: int, tensor: torch.Tensor, tensor_id: int
    ) -> None: ...

    def end(
        self, request_id: int, tensor: torch.Tensor, tensor_id: int
    ) -> None: ...

    def fetch_tensors(
        self, request_id: int, tensor_ids: Sequence[int]
    ) -> None: ...

    def prefetch_tensors(
        self, tensor_ids: Sequence[int], priority: int
    ) -> None: ...


class ArcherEngineHooks:
    def __init__(self, archer_engine) -> None:
        self._engine = archer_engine

    def begin(
        self, request_id: int, tensor: torch.Tensor, tensor_id: int
    ) -> None:
        self._engine.begin(request_id, tensor, tensor_id)

    def end(
        self, request_id: int, tensor: torch.Tensor, tensor_id: int
    ) -> None:
        self._engine.end(request_id, tensor, tensor_id)

    def fetch_tensors(self, request_id: int, tensor_ids: Sequence[int]) -> None:
        self._engine.fetch_tensors(request_id, tensor_ids)

    def prefetch_tensors(
        self, tensor_ids: Sequence[int], priority: int
    ) -> None:
        self._engine.prefetch_tensors(tensor_ids, priority)
