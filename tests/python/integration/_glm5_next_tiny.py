from __future__ import annotations

import math
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
    base = AutoConfig.from_pretrained(fixture_dir)

    # Rebuild the config through from_dict so every attribute derived inside
    # __init__ (fused KDA projection widths, head splits, layer maps) is
    # recomputed from the shrunken values; mutating a live config post-init
    # leaves stale derived attributes and mismatched projection splits.
    payload = base.to_dict()
    text = payload["text_config"]
    text.update(
        {
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "first_k_dense_replace": 3,
            "num_nextn_predict_layers": 1,
            "kv_lora_rank": 64,
            "q_lora_rank": 64,
            "qk_head_dim": 32,
            "qk_nope_head_dim": 32,
            "v_head_dim": 32,
            "head_dim": 0,
            "index_topk": 16,
            "index_head_dim": 32,
            "index_n_heads": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
            "indexer_types": ["full", "full", "full", "full"],
        }
    )
    text["linear_attn_config"] = dict(
        text.get("linear_attn_config") or {},
        num_heads=4,
        head_dim=32,
        kda_layers=[0, 1, 2],
        full_attn_layers=[3],
    )
    text.pop("quantization_config", None)
    payload["vision_config"].update(
        {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": 256,
        }
    )
    payload.pop("quantization_config", None)
    payload["torch_dtype"] = "bfloat16"
    cfg = type(base).from_dict(payload)

    torch.manual_seed(0)
    model = Glm5NextForConditionalGeneration(cfg).to(torch.bfloat16)
    model.save_pretrained(save_dir, safe_serialization=True)
    cfg.save_pretrained(save_dir)

    _save_tokenizer(save_dir)
    return save_dir


def build_tiny_glm5_next_fp8(save_dir: str) -> str:
    bf16_dir = build_tiny_glm5_next(os.path.join(save_dir, "_bf16"))

    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(bf16_dir)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        bf16_dir, torch_dtype=torch.bfloat16
    )

    expert_weight_suffixes = (
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )
    for name, param in list(model.named_parameters(recurse=True)):
        if "shared_expert" in name:
            continue
        if "experts" not in name:
            continue
        if not any(name.endswith(sfx) for sfx in expert_weight_suffixes):
            continue
        q_fp8, scale_inv = _quantize_weight_fp8(param.data)
        param.data = q_fp8
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        parent.register_buffer(parts[-1] + "_scale_inv", scale_inv)

    cfg.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    }

    model.save_pretrained(save_dir, safe_serialization=True)
    cfg.save_pretrained(save_dir)
    _save_tokenizer(save_dir)
    return save_dir


def _quantize_weight_fp8(w: torch.Tensor):
    FP8_MAX = 448.0
    BLOCK = 128
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
    return q.to(torch.float8_e4m3fn), scale_inv


def _save_tokenizer(save_dir: str) -> None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.3-Flash")
        tokenizer.save_pretrained(save_dir)
    except Exception:
        pass
