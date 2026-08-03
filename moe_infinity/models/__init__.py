# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from .dbrx import SyncDbrxFFNBlock
from .deepseek import DeepseekMoEBlock
from .deepseek_v2_wrapper import SyncDeepseekV2MoEBlock
from .deepseek_v3_wrapper import SyncDeepseekV3MoEBlock
from .gpt_oss import SyncGptOssMLP
from .jamba import SyncJambaMoEBlock
from .mixtral import SyncMixtralSparseMoeBlock
from .model_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_deepseek,
    rotate_half,
)
from .nllb_moe import SyncNllbMoeSparseMLP
from .olmoe import SyncOlmoeMoEBlock
from .qwen import Qwen3MoEBlock
from .glm_moe_dsa import SyncGlmMoeDsaMoEBlock
from .qwen3_5_moe import SyncQwen3_5MoeSparseMoeBlock

# Qwen3PagedAttention is lazily imported to avoid a circular dependency:
# model_offload -> moe_infinity.models -> qwen3_paged_attention
#   -> moe_infinity.runtime.attention_backend -> (back to runtime)

__all__ = [
    "DeepseekMoEBlock",
    "Qwen3MoEBlock",
    "Qwen3PagedAttention",
    "SyncDbrxFFNBlock",
    "SyncDeepseekV2MoEBlock",
    "SyncDeepseekV3MoEBlock",
    "SyncGlmMoeDsaMoEBlock",
    "SyncGptOssMLP",
    "SyncJambaMoEBlock",
    "SyncMixtralSparseMoeBlock",
    "SyncNllbMoeSparseMLP",
    "SyncOlmoeMoEBlock",
    "SyncQwen3_5MoeSparseMoeBlock",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_deepseek",
    "rotate_half",
]


def __getattr__(name: str):
    if name == "Qwen3PagedAttention":
        from .qwen3_paged_attention import Qwen3PagedAttention

        return Qwen3PagedAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
