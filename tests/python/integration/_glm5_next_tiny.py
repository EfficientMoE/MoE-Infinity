from __future__ import annotations

import os

import torch


def build_tiny_glm5_next(save_dir: str) -> str:
    os.environ.setdefault(
        "HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub"
    )
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    fixture_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3_flash"
    )
    cfg = AutoConfig.from_pretrained(fixture_dir)

    text = cfg.text_config
    text.num_hidden_layers = 4
    text.hidden_size = 256
    text.intermediate_size = 512
    text.moe_intermediate_size = 128
    text.n_routed_experts = 8
    text.num_experts_per_tok = 2
    text.num_attention_heads = 8
    text.num_key_value_heads = 8
    text.first_k_dense_replace = 3
    text.num_nextn_predict_layers = 1
    text.kv_lora_rank = 64
    text.q_lora_rank = 64
    text.qk_head_dim = 32
    text.qk_nope_head_dim = 32
    text.v_head_dim = 32
    text.head_dim = 0
    text.index_topk = 16
    text.index_head_dim = 32
    text.index_n_heads = 4
    text.layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ]
    text.mlp_layer_types = ["dense", "dense", "dense", "sparse"]
    text.indexer_types = ["full", "shared", "shared", "shared"]
    if isinstance(text.linear_attn_config, dict):
        text.linear_attn_config = dict(
            text.linear_attn_config,
            num_heads=4,
            head_dim=32,
            kda_layers=[0, 1, 2],
            full_attn_layers=[3],
        )

    vis = cfg.vision_config
    vis.depth = 2
    vis.hidden_size = 64
    vis.intermediate_size = 128
    vis.num_heads = 2
    vis.out_hidden_size = text.hidden_size

    for holder in (cfg, text):
        if hasattr(holder, "quantization_config"):
            try:
                delattr(holder, "quantization_config")
            except Exception:
                holder.quantization_config = None

    cfg.torch_dtype = "bfloat16"

    torch.manual_seed(0)
    model = Glm5NextForConditionalGeneration(cfg).to(torch.bfloat16)
    model.save_pretrained(save_dir, safe_serialization=True)
    cfg.save_pretrained(save_dir)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.3-Flash")
        tokenizer.save_pretrained(save_dir)
    except Exception:
        pass

    return save_dir
