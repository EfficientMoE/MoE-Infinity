# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Transformers model-class overrides at the moe-store integration boundary.

The offload runtime owns the lifecycle of these overrides, while the model
module names and replacement classes are architecture knowledge.  Keep that
knowledge at this boundary until moe-store exposes the same lifecycle hooks.
"""

import importlib

from moe_store.wrappers import (
    DeepseekV2PagedAttention,
    DeepseekV3PagedAttention,
    Qwen3MoEBlock,
    Qwen3PagedAttention,
    SyncDbrxFFNBlock,
    SyncDeepseekV2MoEBlock,
    SyncDeepseekV3MoEBlock,
    SyncGlm5NextMoEBlock,
    SyncGlmMoeDsaMoEBlock,
    SyncGptOssMLP,
    SyncJambaMoEBlock,
    SyncMixtralSparseMoeBlock,
    SyncNllbMoeSparseMLP,
    SyncOlmoeMoEBlock,
    SyncQwen3_5MoeSparseMoeBlock,
)

try:
    from moe_store.wrappers import OlmoePagedAttention
except ImportError:
    OlmoePagedAttention = None


def _model_module(name: str):
    return importlib.import_module(
        f"transformers.models.{name}.modeling_{name}"
    )


def install_transformers_model_overrides() -> None:
    nllb = _model_module("nllb_moe")
    nllb._old_sparse_mlp = nllb.NllbMoeSparseMLP
    nllb.NllbMoeSparseMLP = SyncNllbMoeSparseMLP

    mixtral = _model_module("mixtral")
    mixtral._old_sparse_mlp = mixtral.MixtralSparseMoeBlock
    mixtral.MixtralSparseMoeBlock = SyncMixtralSparseMoeBlock

    qwen3 = _model_module("qwen3_moe")
    qwen3._old_sparse_mlp = qwen3.Qwen3MoeSparseMoeBlock
    qwen3.Qwen3MoeSparseMoeBlock = Qwen3MoEBlock
    qwen3._old_qwen3_attention = qwen3.Qwen3MoeAttention
    qwen3.Qwen3MoeAttention = Qwen3PagedAttention

    dbrx = _model_module("dbrx")
    dbrx._old_dbrx_ffn = dbrx.DbrxFFN
    dbrx.DbrxFFN = SyncDbrxFFNBlock

    olmoe = _model_module("olmoe")
    olmoe._old_olmoe_moe = olmoe.OlmoeSparseMoeBlock
    olmoe.OlmoeSparseMoeBlock = SyncOlmoeMoEBlock
    if OlmoePagedAttention is not None:
        olmoe._old_olmoe_attention = olmoe.OlmoeAttention
        olmoe.OlmoeAttention = OlmoePagedAttention

    jamba = _model_module("jamba")
    jamba._old_jamba_moe = jamba.JambaSparseMoeBlock
    jamba.JambaSparseMoeBlock = SyncJambaMoEBlock

    deepseek_v2 = _model_module("deepseek_v2")
    old_deepseek_v2_moe = getattr(
        deepseek_v2, "DeepseekV2MoE", None
    ) or getattr(deepseek_v2, "DeepseekV2Moe", None)
    deepseek_v2._old_deepseek_v2_moe = old_deepseek_v2_moe
    deepseek_v2_moe_attr = (
        "DeepseekV2MoE"
        if hasattr(deepseek_v2, "DeepseekV2MoE")
        else "DeepseekV2Moe"
    )
    setattr(deepseek_v2, deepseek_v2_moe_attr, SyncDeepseekV2MoEBlock)
    deepseek_v2._old_deepseek_v2_attention = deepseek_v2.DeepseekV2Attention
    deepseek_v2.DeepseekV2Attention = DeepseekV2PagedAttention

    deepseek_v3 = _model_module("deepseek_v3")
    deepseek_v3._old_deepseek_v3_moe = deepseek_v3.DeepseekV3MoE
    deepseek_v3.DeepseekV3MoE = SyncDeepseekV3MoEBlock
    deepseek_v3._old_deepseek_v3_attention = deepseek_v3.DeepseekV3Attention
    deepseek_v3.DeepseekV3Attention = DeepseekV3PagedAttention

    gpt_oss = _model_module("gpt_oss")
    gpt_oss._old_gpt_oss_mlp = gpt_oss.GptOssMLP
    gpt_oss.GptOssMLP = SyncGptOssMLP

    qwen3_5 = _model_module("qwen3_5_moe")
    qwen3_5._old_qwen3_5_sparse_moe = qwen3_5.Qwen3_5MoeSparseMoeBlock
    qwen3_5.Qwen3_5MoeSparseMoeBlock = SyncQwen3_5MoeSparseMoeBlock

    try:
        glm = _model_module("glm_moe_dsa")
        glm._old_glm_moe_dsa_moe = glm.GlmMoeDsaMoE
        glm.GlmMoeDsaMoE = SyncGlmMoeDsaMoEBlock
    except (ImportError, AttributeError):
        pass

    try:
        glm5_next = _model_module("glm5_next")
        glm5_next._old_glm5_next_moe = glm5_next.Glm5NextTextMoE
        glm5_next.Glm5NextTextMoE = SyncGlm5NextMoEBlock
    except (ImportError, AttributeError):
        pass


def restore_context_transformers_model_overrides() -> None:
    gpt_oss = _model_module("gpt_oss")
    gpt_oss.GptOssMLP = gpt_oss._old_gpt_oss_mlp

    qwen3_5 = _model_module("qwen3_5_moe")
    if hasattr(qwen3_5, "_old_qwen3_5_sparse_moe"):
        qwen3_5.Qwen3_5MoeSparseMoeBlock = qwen3_5._old_qwen3_5_sparse_moe

    olmoe = _model_module("olmoe")
    if hasattr(olmoe, "_old_olmoe_attention"):
        olmoe.OlmoeAttention = olmoe._old_olmoe_attention

    try:
        glm = _model_module("glm_moe_dsa")
        if hasattr(glm, "_old_glm_moe_dsa_moe"):
            glm.GlmMoeDsaMoE = glm._old_glm_moe_dsa_moe
    except (ImportError, AttributeError):
        pass

    try:
        glm5_next = _model_module("glm5_next")
        if hasattr(glm5_next, "_old_glm5_next_moe"):
            glm5_next.Glm5NextTextMoE = glm5_next._old_glm5_next_moe
    except (ImportError, AttributeError):
        pass


def restore_runtime_transformers_model_overrides() -> None:
    nllb = _model_module("nllb_moe")
    nllb.NllbMoeSparseMLP = nllb._old_sparse_mlp

    mixtral = _model_module("mixtral")
    mixtral.MixtralSparseMoeBlock = mixtral._old_sparse_mlp

    qwen3 = _model_module("qwen3_moe")
    qwen3.Qwen3MoeSparseMoeBlock = qwen3._old_sparse_mlp
    if hasattr(qwen3, "_old_qwen3_attention"):
        qwen3.Qwen3MoeAttention = qwen3._old_qwen3_attention

    dbrx = _model_module("dbrx")
    dbrx.DbrxFFN = dbrx._old_dbrx_ffn

    olmoe = _model_module("olmoe")
    olmoe.OlmoeSparseMoeBlock = olmoe._old_olmoe_moe

    jamba = _model_module("jamba")
    jamba.JambaSparseMoeBlock = jamba._old_jamba_moe

    deepseek_v2 = _model_module("deepseek_v2")
    deepseek_v2_moe_attr = (
        "DeepseekV2MoE"
        if hasattr(deepseek_v2, "DeepseekV2MoE")
        else "DeepseekV2Moe"
    )
    setattr(
        deepseek_v2,
        deepseek_v2_moe_attr,
        deepseek_v2._old_deepseek_v2_moe,
    )
    if hasattr(deepseek_v2, "_old_deepseek_v2_attention"):
        deepseek_v2.DeepseekV2Attention = deepseek_v2._old_deepseek_v2_attention

    deepseek_v3 = _model_module("deepseek_v3")
    deepseek_v3.DeepseekV3MoE = deepseek_v3._old_deepseek_v3_moe
    if hasattr(deepseek_v3, "_old_deepseek_v3_attention"):
        deepseek_v3.DeepseekV3Attention = deepseek_v3._old_deepseek_v3_attention

    gpt_oss = _model_module("gpt_oss")
    gpt_oss.GptOssMLP = gpt_oss._old_gpt_oss_mlp

    qwen3_5 = _model_module("qwen3_5_moe")
    if hasattr(qwen3_5, "_old_qwen3_5_sparse_moe"):
        qwen3_5.Qwen3_5MoeSparseMoeBlock = qwen3_5._old_qwen3_5_sparse_moe
