"""Dispatch flag for the grouped decode MoE kernels (#306).

RTX PRO 6000 only: the kernels' split-K / row-split layouts and launch table
(extensions/kernel/grouped_moe_decode_pro6000.cu) were tuned on the RTX PRO
6000 Blackwell Max-Q. Every other device keeps per-expert execution. The
native dispatcher applies the same gate itself (moe_infinity::hw::Supports);
this mirrors it for Python-side setup and tests.

The kernels implement ``silu(gate) * up`` only. ``grouped_decode_activation_kind``
classifies a config the same way as #330's ``resolve_expert_activation``
(MoEActivationKind: 0 SiLU, 1 clamped SiLU, 2 SwiGLU-OAI, -1 unknown) so the
dispatcher keeps per-expert execution for every other activation.
"""

from __future__ import annotations

import os

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports


def grouped_decode_pro6000_enabled(device=None) -> bool:
    """True if the grouped decode path should run on ``device``."""
    if os.environ.get("MOE_GROUPED_DECODE", "1") == "0":
        return False
    return supports(RTX_PRO_6000, device)


SILU = 0
CLAMPED_SILU = 1
SWIGLU_OAI = 2
UNKNOWN = -1

_SWIGLU_OAI_MODEL_TYPES = frozenset({"minimax_m3_vl", "minimax_m3_vl_text"})
_SILU_ACTS = frozenset({"silu", "swish"})


def _text_config(config):
    for name in ("thinker_config", "text_config"):
        sub = getattr(config, name, None)
        if sub is not None and not isinstance(sub, dict):
            inner = getattr(sub, "text_config", None)
            return (
                inner
                if inner is not None and not isinstance(inner, dict)
                else sub
            )
    return config


def _get(config, text, name):
    value = getattr(text, name, None)
    return getattr(config, name, None) if value is None else value


def grouped_decode_activation_kind(config) -> int:
    """Gated activation of ``config``'s routed experts (0 = plain SiLU)."""
    text = _text_config(config)
    model_types = {
        getattr(config, "model_type", None),
        getattr(text, "model_type", None),
    }
    if model_types & _SWIGLU_OAI_MODEL_TYPES:
        return SWIGLU_OAI
    limit = _get(config, text, "swiglu_limit")
    if limit is not None and float(limit) > 0.0:
        return CLAMPED_SILU
    hidden_act = _get(config, text, "hidden_act")
    if hidden_act is not None and str(hidden_act).lower() not in _SILU_ACTS:
        return UNKNOWN
    return SILU
