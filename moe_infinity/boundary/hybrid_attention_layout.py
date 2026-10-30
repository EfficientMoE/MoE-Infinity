# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Which decoder layers of a hybrid model own paged KV.

Qwen3.5-MoE interleaves Gated DeltaNet linear-attention layers (recurrent and
conv state, no per-token KV) with gated full-attention layers. Only the
full-attention layers write the paged KV cache, so the serving runtime keys
that cache by a dense *paged layer index*: the ordinal of a layer among the
full-attention layers. Non-hybrid models keep ``paged index == layer_idx``.

The layout is architecture knowledge and stays at this boundary, keyed on the
model types that ship a hybrid paged-attention shim, so model families without
one (and their KV sizing) are unaffected.
"""

from __future__ import annotations

from typing import Any, Optional

HYBRID_PAGED_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})

_FULL_ATTENTION = "full_attention"
_LINEAR_ATTENTION = "linear_attention"


def text_config_of(config: Any) -> Any:
    """Return the decoder text config (``text_config`` for VL wrappers)."""
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        return get_text()
    text_config = getattr(config, "text_config", None)
    return config if text_config is None else text_config


def _hybrid_layer_types(config: Any) -> Optional[tuple[str, ...]]:
    if config is None:
        return None
    text_config = text_config_of(config)
    model_types = {
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
    }
    if not model_types & HYBRID_PAGED_MODEL_TYPES:
        return None
    layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)):
        return None
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if isinstance(num_hidden_layers, int):
        layer_types = layer_types[:num_hidden_layers]
    layer_types = tuple(str(layer_type) for layer_type in layer_types)
    if _LINEAR_ATTENTION not in layer_types:
        return None
    return layer_types


def paged_kv_layer_indices(config: Any) -> Optional[tuple[int, ...]]:
    """Absolute indices of the paged (full-attention) layers of a hybrid model.

    Returns ``None`` for non-hybrid models, where every decoder layer is a
    paged-attention layer.
    """
    layer_types = _hybrid_layer_types(config)
    if layer_types is None:
        return None
    return tuple(
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types)
        if layer_type == _FULL_ATTENTION
    )


def linear_attention_layer_indices(config: Any) -> tuple[int, ...]:
    """Absolute indices of the recurrent linear-attention layers (may be empty)."""
    layer_types = _hybrid_layer_types(config)
    if layer_types is None:
        return ()
    return tuple(
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types)
        if layer_type == _LINEAR_ATTENTION
    )


def paged_kv_layer_count(config: Any, default: int) -> int:
    """Number of layer planes the paged KV cache needs for ``config``."""
    indices = paged_kv_layer_indices(config)
    if not indices:
        return default
    return len(indices)


def paged_layer_index_for(config: Any, layer_idx: int) -> int:
    """Map an absolute decoder ``layer_idx`` to its dense paged layer index."""
    indices = paged_kv_layer_indices(config)
    if indices is None:
        return layer_idx
    try:
        return indices.index(layer_idx)
    except ValueError:
        raise ValueError(
            f"layer {layer_idx} is not a full-attention layer of this hybrid "
            "model and owns no paged KV"
        ) from None


__all__ = [
    "HYBRID_PAGED_MODEL_TYPES",
    "linear_attention_layer_indices",
    "paged_kv_layer_count",
    "paged_kv_layer_indices",
    "paged_layer_index_for",
    "text_config_of",
]
