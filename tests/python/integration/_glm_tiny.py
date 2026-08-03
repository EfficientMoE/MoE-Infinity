from __future__ import annotations

import os
import torch


def build_tiny_glm(save_dir: str) -> str:
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    cfg = AutoConfig.from_pretrained("zai-org/GLM-5.2-FP8", trust_remote_code=True)

    cfg.num_hidden_layers = 4
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.moe_intermediate_size = 128
    cfg.n_routed_experts = 8
    cfg.num_experts_per_tok = 2
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8
    cfg.first_k_dense_replace = 1
    cfg.num_nextn_predict_layers = 1

    if hasattr(cfg, "kv_lora_rank"):
        cfg.kv_lora_rank = 64
    if hasattr(cfg, "qk_rope_head_dim"):
        cfg.qk_rope_head_dim = 16
    if hasattr(cfg, "v_head_dim"):
        cfg.v_head_dim = 32
    if hasattr(cfg, "qk_nope_head_dim"):
        cfg.qk_nope_head_dim = 16
    if hasattr(cfg, "qk_head_dim"):
        cfg.qk_head_dim = 32
    if hasattr(cfg, "q_lora_rank"):
        cfg.q_lora_rank = 64
    if hasattr(cfg, "index_topk"):
        cfg.index_topk = 16
    if hasattr(cfg, "indexer_types") and cfg.indexer_types:
        cfg.indexer_types = ["full", "shared", "shared", "shared"]
    if hasattr(cfg, "layer_types") and cfg.layer_types:
        cfg.layer_types = cfg.layer_types[: cfg.num_hidden_layers]
    if hasattr(cfg, "mlp_layer_types") and cfg.mlp_layer_types:
        cfg.mlp_layer_types = cfg.mlp_layer_types[: cfg.num_hidden_layers]

    cfg.torch_dtype = "bfloat16"

    if hasattr(cfg, "quantization_config"):
        try:
            delattr(cfg, "quantization_config")
        except Exception:
            cfg.quantization_config = None

    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(save_dir, safe_serialization=True)
    cfg.save_pretrained(save_dir)
    return save_dir
