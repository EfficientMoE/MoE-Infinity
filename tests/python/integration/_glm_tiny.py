from __future__ import annotations

import math
import os

import torch


def build_tiny_glm_fp8(save_dir: str) -> str:
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
    from transformers import AutoConfig, AutoTokenizer
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
    cfg.first_k_dense_replace = 3
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

    FP8_MAX = 448.0
    BLOCK = 128

    def _quantize_weight_fp8(w: torch.Tensor):
        N, K = w.shape
        SN = math.ceil(N / BLOCK)
        SK = math.ceil(K / BLOCK)
        w_f32 = w.float()
        scale_inv = torch.zeros(SN, SK, dtype=torch.float32)
        q = torch.zeros(N, K, dtype=torch.float32)
        for i in range(SN):
            for j in range(SK):
                r0, r1 = i * BLOCK, min((i + 1) * BLOCK, N)
                c0, c1 = j * BLOCK, min((j + 1) * BLOCK, K)
                block = w_f32[r0:r1, c0:c1]
                amax = block.abs().max().item()
                s = amax / FP8_MAX if amax > 0 else 1.0
                scale_inv[i, j] = s
                q[r0:r1, c0:c1] = (block / s).clamp(-FP8_MAX, FP8_MAX)
        q_fp8 = q.to(torch.float8_e4m3fn)
        return q_fp8, scale_inv

    expert_weight_suffixes = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    for name, param in list(model.named_parameters(recurse=True)):
        if "shared_expert" in name:
            continue
        if not any(name.endswith(sfx) for sfx in expert_weight_suffixes):
            continue
        if "experts" not in name:
            continue
        q_fp8, scale_inv = _quantize_weight_fp8(param.data)
        param.data = q_fp8
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        scale_attr = parts[-1] + "_scale_inv"
        parent.register_buffer(scale_attr, scale_inv)

    cfg.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    }

    model.save_pretrained(save_dir, safe_serialization=True)
    cfg.save_pretrained(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        "zai-org/GLM-5.2-FP8", trust_remote_code=True
    )
    tokenizer.save_pretrained(save_dir)

    return save_dir


def build_tiny_glm(save_dir: str) -> str:
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
    from transformers import AutoConfig, AutoTokenizer
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
    cfg.first_k_dense_replace = 3
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

    tokenizer = AutoTokenizer.from_pretrained(
        "zai-org/GLM-5.2-FP8", trust_remote_code=True
    )
    tokenizer.save_pretrained(save_dir)

    return save_dir
