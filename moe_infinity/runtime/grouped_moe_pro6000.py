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

GROUPED_DECODE_MAX_TOKENS = 256  # kGroupedMoePro6000MaxTokens
GROUPED_PREFILL_DEFAULT_MIN_TOKENS = (
    2048  # moe::kGroupedPrefillDefaultMinTokens
)
GROUPED_MAX_ACT_BYTES = 1 << 30  # moe::kGroupedMaxActBytes

NONE = "none"
DECODE = "decode"
PREFILL = "prefill"


def grouped_decode_pro6000_enabled(device=None) -> bool:
    """True if the grouped decode path should run on ``device``."""
    if os.environ.get("MOE_GROUPED_DECODE", "1") == "0":
        return False
    return supports(RTX_PRO_6000, device)


def grouped_prefill_enabled(device=None) -> bool:
    """True if the grouped prefill path (#307) should run on ``device``.

    Opt-in (``MOE_GROUPED_PREFILL=1``) until validated on a GPU.
    """
    if os.environ.get("MOE_GROUPED_PREFILL", "0") != "1":
        return False
    return supports(RTX_PRO_6000, device)


def grouped_prefill_min_tokens() -> int:
    """Smallest batch (tokens) that takes the grouped prefill path."""
    raw = os.environ.get("MOE_GROUPED_PREFILL_MIN_TOKENS", "")
    try:
        value = int(raw)
    except ValueError:
        return GROUPED_PREFILL_DEFAULT_MIN_TOKENS
    return value if value > 0 else GROUPED_PREFILL_DEFAULT_MIN_TOKENS


def select_grouped_mode(
    tokens: int,
    prefill_enabled: bool,
    min_tokens: int = GROUPED_PREFILL_DEFAULT_MIN_TOKENS,
) -> str:
    """Grouped kernel for a batch of ``tokens`` (mirror of the C++ dispatcher's
    ``moe::SelectGroupedMoeMode``): decode up to 256 tokens, prefill from
    ``min_tokens`` (never inside the decode range) when enabled, else none."""
    if tokens <= 0:
        return NONE
    if tokens <= GROUPED_DECODE_MAX_TOKENS:
        return DECODE
    if not prefill_enabled:
        return NONE
    floor = max(min_tokens, GROUPED_DECODE_MAX_TOKENS + 1)
    return PREFILL if tokens >= floor else NONE


def grouped_workspace_fits(tokens: int, kcap: int, inter: int) -> bool:
    """The [tokens * kcap, inter] bf16 activation workspace stays <= 1 GiB."""
    return tokens * kcap * inter * 2 <= GROUPED_MAX_ACT_BYTES


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
