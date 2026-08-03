from __future__ import annotations

from benchmarks.performance_model.types import ModelParams


def _expert_dtype_from_config(config) -> str:
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return "bf16"
    if isinstance(qc, dict):
        method = qc.get("quant_method", "")
    else:
        method = getattr(qc, "quant_method", "")
    return "fp8" if "fp8" in str(method).lower() else "bf16"


def _extract_glm(config) -> ModelParams:
    return ModelParams(
        name=getattr(config, "_name_or_path", "glm_moe_dsa"),
        num_layers=config.num_hidden_layers,
        num_attn_heads=config.num_attention_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_experts=config.n_routed_experts,
        top_k=config.num_experts_per_tok,
        shared_experts=getattr(config, "n_shared_experts", 1),
        expert_intermediate_size=config.moe_intermediate_size,
        first_k_dense=getattr(config, "first_k_dense_replace", 0),
        expert_dtype=_expert_dtype_from_config(config),
        attn_dtype="bf16",
        kv_lora_rank=getattr(config, "kv_lora_rank", None),
        q_lora_rank=getattr(config, "q_lora_rank", None),
    )


def _extract_generic(config) -> ModelParams:
    num_experts = getattr(config, "num_local_experts", None) or getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 1)
    top_k = getattr(config, "num_experts_per_tok", None) or getattr(config, "top_k", 1)
    return ModelParams(
        name=getattr(config, "_name_or_path", "unknown"),
        num_layers=config.num_hidden_layers,
        num_attn_heads=config.num_attention_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_experts=num_experts,
        top_k=top_k,
        shared_experts=getattr(config, "n_shared_experts", 0),
        expert_intermediate_size=getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", config.hidden_size * 4)),
        first_k_dense=getattr(config, "first_k_dense_replace", 0),
        expert_dtype=_expert_dtype_from_config(config),
        attn_dtype="bf16",
        kv_lora_rank=getattr(config, "kv_lora_rank", None),
        q_lora_rank=getattr(config, "q_lora_rank", None),
    )


def extract_model_params(model_name_or_path: str) -> ModelParams:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    arch = (getattr(config, "architectures", None) or [""])[0].lower()

    if "glmmoedsa" in arch:
        return _extract_glm(config)
    return _extract_generic(config)
