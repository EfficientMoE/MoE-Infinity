# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""M-RoPE positions for serving ``Qwen3_5MoeForConditionalGeneration``.

HuggingFace only derives the 3-D (temporal/height/width) rotary positions of
image and video tokens when ``position_ids`` is ``None``, and during decode it
offsets the text position by the per-sequence ``rope_deltas`` it computed at
prefill. The serving runner always passes explicit positions, so it computes
the prompt positions here with the model's own ``get_rope_index`` and keeps
the delta per sequence for later decode steps.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

_MROPE_MODEL_TYPES = frozenset({"qwen3_5_moe"})
_GRID_KEYS = ("image_grid_thw", "video_grid_thw")


def _rope_index_owner(model: Any) -> Optional[Any]:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) not in _MROPE_MODEL_TYPES:
        return None
    inner = getattr(model, "model", None)
    if not callable(getattr(inner, "get_rope_index", None)):
        return None
    return inner


def supports_mrope_positions(model: Any) -> bool:
    return _rope_index_owner(model) is not None


def has_vision_grid(row_inputs: Optional[dict[str, Any]]) -> bool:
    return row_inputs is not None and any(
        row_inputs.get(key) is not None for key in _GRID_KEYS
    )


def _mm_token_type_ids(
    config: Any, input_ids: torch.Tensor, row_inputs: dict[str, Any]
) -> torch.Tensor:
    token_types = row_inputs.get("mm_token_type_ids")
    if isinstance(token_types, torch.Tensor):
        token_types = token_types.reshape(1, -1).to(input_ids.device)
        if token_types.shape == input_ids.shape:
            return token_types
    # Same typing the processor emits: text 0, image 1, video 2.
    derived = torch.zeros_like(input_ids)
    derived[input_ids == int(config.image_token_id)] = 1
    derived[input_ids == int(config.video_token_id)] = 2
    return derived


def mrope_prompt_positions(
    model: Any,
    prompt_token_ids: list[int],
    row_inputs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return ``([3, prompt_len] positions, rope delta)`` for one prompt."""
    owner = _rope_index_owner(model)
    if owner is None:
        raise ValueError("model does not provide M-RoPE positions")
    input_ids = torch.tensor(
        [prompt_token_ids], dtype=torch.long, device=device
    )
    grids = {
        key: row_inputs[key].to(device)
        for key in _GRID_KEYS
        if isinstance(row_inputs.get(key), torch.Tensor)
    }
    positions, deltas = owner.get_rope_index(
        input_ids,
        mm_token_type_ids=_mm_token_type_ids(
            model.config, input_ids, row_inputs
        ),
        **grids,
    )
    return positions[:, 0], int(deltas.reshape(-1)[0])


__all__ = [
    "has_vision_grid",
    "mrope_prompt_positions",
    "supports_mrope_positions",
]
