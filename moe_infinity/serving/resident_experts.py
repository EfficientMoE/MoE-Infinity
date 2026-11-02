"""Capture-safe decode for MoE layers whose experts are all resident.

``install_grouped_resident_experts`` routes the decode-sized forward of every
HF-style fused experts module (``gate_up_proj [E, 2I, H]``, ``down_proj
[E, H, I]``, SiLU) through the grouped decode kernel of ``moe_infinity._store``
(two launches per layer, routing read on the device). The kernel reads the
expert weights through a device pointer table, so a residency change only
rewrites table rows (``ResidentExpertTable.update``): CUDA graphs captured over
the layer stay valid and are not re-recorded.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

# The grouped kernel scans the router mask per expert; it takes up to this
# many tokens per call (decode batches). Larger calls use the module's own
# forward.
MAX_GROUPED_TOKENS = 256


def grouped_decode_op() -> Callable[..., None] | None:
    try:
        from moe_infinity import _store
    except Exception:
        return None
    op = getattr(_store, "grouped_moe_decode", None)
    return op if callable(op) else None


class ResidentExpertTable:
    """Device table of ``[gate_ptr, up_ptr, down_ptr, expert_id]`` rows."""

    def __init__(self, experts: torch.nn.Module) -> None:
        gate_up = experts.gate_up_proj
        down = experts.down_proj
        self.num_experts = int(gate_up.shape[0])
        self.intermediate = int(down.shape[2])
        self.table = torch.empty(
            (self.num_experts, 4), dtype=torch.int64, device=gate_up.device
        )
        self._weights: list[tuple[torch.Tensor, ...]] = []
        for expert in range(self.num_experts):
            gate, up = gate_up[expert].chunk(2, dim=0)
            self._weights.append((gate, up, down[expert]))
        self._write_rows(range(self.num_experts))

    def update(
        self,
        expert: int,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        """Point ``expert`` at new resident weights (in place)."""
        for tensor in (gate, up, down):
            if not tensor.is_contiguous() or tensor.dtype != torch.bfloat16:
                raise ValueError("expert weights must be contiguous bf16")
        self._weights[expert] = (gate, up, down)
        self._write_rows([expert])

    def _write_rows(self, experts) -> None:
        rows = [[t.data_ptr() for t in self._weights[e]] + [e] for e in experts]
        index = torch.tensor(list(experts), device=self.table.device)
        values = torch.tensor(rows, dtype=torch.int64, device="cpu")
        self.table.index_copy_(0, index, values.to(self.table.device))


def _eligible(experts: torch.nn.Module) -> bool:
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    act = getattr(experts, "act_fn", None)
    return (
        isinstance(gate_up, torch.Tensor)
        and isinstance(down, torch.Tensor)
        and gate_up.ndim == 3
        and down.ndim == 3
        and gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and down.dtype == torch.bfloat16
        and gate_up.is_contiguous()
        and down.is_contiguous()
        and gate_up.shape[1] == 2 * down.shape[2]
        and type(act).__name__ in ("SiLU", "SiLUActivation")
    )


def _grouped_forward(
    self: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    tokens = hidden_states.shape[0]
    if tokens > MAX_GROUPED_TOKENS or hidden_states.dtype != torch.bfloat16:
        return self._eager_forward(hidden_states, top_k_index, top_k_weights)
    table: ResidentExpertTable = self._resident_table
    num_experts = table.num_experts
    mask = torch.zeros(
        (tokens, num_experts), dtype=torch.bool, device=hidden_states.device
    ).scatter_(1, top_k_index, True)
    weights = torch.zeros(
        (tokens, num_experts),
        dtype=top_k_weights.dtype,
        device=hidden_states.device,
    ).scatter_(1, top_k_index, top_k_weights)
    out = torch.empty(
        hidden_states.shape, dtype=torch.float32, device=hidden_states.device
    )
    self._grouped_op(
        hidden_states.contiguous(),
        mask,
        weights,
        table.table,
        table.intermediate,
        int(top_k_index.shape[1]),
        out,
        zero_out=True,
    )
    return out.to(hidden_states.dtype)


def install_grouped_resident_experts(
    model: torch.nn.Module,
) -> list[ResidentExpertTable]:
    """Patch every eligible experts module of ``model``; returns the tables.

    Returns an empty list when the grouped decode kernel is not built.
    """
    op = grouped_decode_op()
    if op is None:
        return []
    tables: list[ResidentExpertTable] = []
    for module in model.modules():
        if getattr(module, "_resident_table", None) is not None:
            continue
        if not _eligible(module):
            continue
        table = ResidentExpertTable(module)
        module._resident_table = table
        module._grouped_op = op
        module._eager_forward = module.forward
        module.forward = types.MethodType(_grouped_forward, module)
        tables.append(table)
    return tables


__all__ = [
    "MAX_GROUPED_TOKENS",
    "ResidentExpertTable",
    "grouped_decode_op",
    "install_grouped_resident_experts",
]
