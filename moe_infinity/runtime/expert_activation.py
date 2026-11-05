# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Routed-expert gated activation for the native (offloaded) expert path.

The CUTLASS expert FFN (``fused_moe_ffn_into``) fuses ``act(gate, up)`` into
the up-projection epilogue. Most gate/up/down families use plain
``silu(gate) * up``; a few checkpoints need a clamped variant:

* ``clamped_silu``: ``silu(min(gate, limit)) * clamp(up, -limit, limit)``
  (``swiglu_limit`` in GLM-5.3-Flash and DeepSeek-V4 / V4.1 configs).
* ``swiglu_oai``: ``(clamp(up, -limit, limit) + 1) * g * sigmoid(alpha * g)``
  with ``g = min(gate, limit)`` (MiniMax-M3, same gate as GPT-OSS).

Kind values mirror ``MoEActivationKind`` in
``extensions/kernel/moe_activation.h``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SILU = 0
CLAMPED_SILU = 1
SWIGLU_OAI = 2

KIND_NAMES = {
    SILU: "silu",
    CLAMPED_SILU: "clamped_silu",
    SWIGLU_OAI: "swiglu_oai",
}

# Expert types whose routed experts run the gate/up/down CUTLASS path.
_GATED_EXPERT_TYPES = (4, 5)

# Families whose routed experts use the SwiGLU-OAI gate on split gate/up
# projections (GPT-OSS, expert type 6, has its own native branch).
_SWIGLU_OAI_MODEL_TYPES = frozenset({"minimax_m3_vl", "minimax_m3_vl_text"})

_SILU_ACTS = frozenset({"silu", "swish"})


@dataclass(frozen=True)
class ExpertActivation:
    kind: int = SILU
    alpha: float = 1.0
    limit: float = 0.0

    @property
    def name(self) -> str:
        return KIND_NAMES[self.kind]

    def as_args(self) -> tuple[int, float, float]:
        return self.kind, float(self.alpha), float(self.limit)


def _as_config(value):
    if isinstance(value, dict):
        from transformers import PretrainedConfig

        return PretrainedConfig.from_dict(value)
    return value


def _text_config(config):
    # Config holding the routed-expert MLP fields: Qwen3-Omni nests it under
    # thinker_config.text_config, VL wrappers under text_config.
    thinker = getattr(config, "thinker_config", None)
    if thinker is not None:
        thinker = _as_config(thinker)
        return _as_config(getattr(thinker, "text_config", None) or thinker)
    text = getattr(config, "text_config", None)
    if text is None:
        getter = getattr(config, "get_text_config", None)
        text = getter() if callable(getter) else config
    return _as_config(text)


def _get(config, text, name):
    value = getattr(text, name, None)
    if value is None:
        value = getattr(config, name, None)
    return value


def resolve_expert_activation(config, expert_type: int) -> ExpertActivation:
    """Activation descriptor of ``config``'s routed experts.

    Warns for a gate/up/down family whose ``hidden_act`` the native expert
    kernel cannot express (it runs the SiLU variant).
    """
    if expert_type not in _GATED_EXPERT_TYPES:
        return ExpertActivation()
    text = _text_config(config)
    model_types = {
        getattr(config, "model_type", None),
        getattr(text, "model_type", None),
    }
    limit = _get(config, text, "swiglu_limit")
    limit = float(limit) if limit is not None else 0.0

    if model_types & _SWIGLU_OAI_MODEL_TYPES:
        alpha = _get(config, text, "swiglu_alpha")
        alpha = float(alpha) if alpha is not None else 1.702
        if limit <= 0.0:
            limit = 7.0
        return ExpertActivation(SWIGLU_OAI, alpha, limit)

    hidden_act = _get(config, text, "hidden_act")
    if hidden_act is not None and str(hidden_act).lower() not in _SILU_ACTS:
        warnings.warn(
            f"native expert path supports SiLU-gated experts only; "
            f"{sorted(t for t in model_types if t)} declares "
            f"hidden_act={hidden_act!r}",
            stacklevel=2,
        )
    if limit > 0.0:
        return ExpertActivation(CLAMPED_SILU, 1.0, limit)
    return ExpertActivation()


def apply_expert_activation(dispatcher, activation: ExpertActivation) -> None:
    """Push ``activation`` into a native ``expert_dispatcher``."""
    setter = getattr(dispatcher, "set_expert_activation", None)
    if setter is None:
        if activation.kind != SILU:
            raise RuntimeError(
                f"routed experts need the {activation.name} activation but the "
                "native extension has no set_expert_activation; rebuild "
                "moe_infinity._store"
            )
        return
    setter(*activation.as_args())


def reference_gated_act(
    gate: torch.Tensor, up: torch.Tensor, activation: ExpertActivation
) -> torch.Tensor:
    """Eager reference of the epilogue math (HF formulation)."""
    if activation.kind == SILU:
        return F.silu(gate) * up
    gate = gate.clamp(max=activation.limit)
    up = up.clamp(min=-activation.limit, max=activation.limit)
    if activation.kind == CLAMPED_SILU:
        return F.silu(gate) * up
    return (up + 1.0) * (gate * torch.sigmoid(gate * activation.alpha))
