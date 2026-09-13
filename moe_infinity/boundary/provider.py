# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Tensor-provider boundary between the parameter store and the engine.

``TensorProvider`` is the seam the offloading engine consumes instead of
reading model architecture details (expert naming, layer counts, checkpoint
layout) directly. The concrete implementation currently lives in
``moe_infinity.runtime.model_offload``; it relocates to the moe-store
repository in the split, at which point the engine depends only on this
protocol.
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class GroupSpec:
    """One topology node: an expert's weight set or a dense module's tensors.

    ``tensor_ids`` preserves slot order (e.g. ``[gate, up, down]`` for an
    expert); the fused MoE kernels read weights by position.
    """

    name: str
    is_sparse: bool
    layer_id: int
    expert_id: int
    tensor_ids: Sequence[int] = field(default_factory=tuple)
    corr_id: int = 0
    nbytes: int = 0


@runtime_checkable
class TensorProvider(Protocol):
    def groups(self) -> Iterator[GroupSpec]: ...

    def tensor_id(
        self, layer_id: int, expert_id: int, slot: int = 0
    ) -> int: ...

    def expert_nbytes(self, layer_id: int, expert_id: int) -> int: ...

    def resident_names(self) -> set[str]: ...
