"""Tiny random-weight checkpoints whose routed experts exercise the clamps.

Routed-expert gate/up weights are scaled so ``|gate|`` and ``|up|`` routinely
exceed ``swiglu_limit``; the plain-SiLU native path then visibly diverges from
the HF reference (#317).
"""

from __future__ import annotations

import torch

# std of the routed-expert gate/up projections at the first sparse layer input
GATE_UP_STD = 15.0


def _scale_routed_experts(model, hidden_size: int) -> int:
    scaled = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("mlp.gate.weight"):
                # Some routers initialise to zeros: every expert score ties
                # and top-k tie-breaking then differs between implementations.
                param.normal_(0.0, 1.0 / hidden_size**0.5)
                continue
            if "experts" not in name or "shared" in name:
                continue
            if name.endswith(
                ("gate_up_proj", "gate_proj.weight", "up_proj.weight")
            ):
                param.normal_(0.0, GATE_UP_STD / hidden_size**0.5)
                scaled += 1
    if scaled == 0:
        raise AssertionError("no routed-expert gate/up weights found")
    return scaled


def build_tiny_minimax_m3(save_dir: str) -> str:
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
    )
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration,
    )

    text = {
        "vocab_size": 512,
        "hidden_size": 256,
        "intermediate_size": 128,
        "shared_intermediate_size": 128,
        "dense_intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rotary_dim": 32,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "layer_types": ["full_attention", "full_attention"],
        "mlp_layer_types": ["sparse", "sparse"],
        "max_position_embeddings": 512,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    vision = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
    }
    cfg = MiniMaxM3VLConfig(
        text_config=text,
        vision_config=vision,
        projector_hidden_size=256,
        image_token_index=500,
        video_token_index=501,
    )
    cfg.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
    cfg.torch_dtype = "bfloat16"

    torch.manual_seed(0)
    model = MiniMaxM3SparseForConditionalGeneration(cfg)
    _scale_routed_experts(model, text["hidden_size"])
    model = model.to(torch.bfloat16)
    # Write the in-memory (transformers-v5) key layout the released
    # checkpoint ships (``model.language_model.*``, packed experts);
    # save_pretrained would revert to the legacy ``block_sparse_moe`` names.
    import os

    from safetensors.torch import save_file

    os.makedirs(save_dir, exist_ok=True)
    state = {
        k: v.contiguous()
        for k, v in model.state_dict().items()
        if k != "lm_head.weight" or not cfg.tie_word_embeddings
    }
    save_file(
        state,
        os.path.join(save_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )
    cfg.save_pretrained(save_dir)
    return save_dir


def build_tiny_glm5_next_clamped(save_dir: str) -> str:
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    from tests.python.integration._glm5_next_tiny import build_tiny_glm5_next

    build_tiny_glm5_next(save_dir)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        save_dir, torch_dtype=torch.float32
    )
    torch.manual_seed(1)
    _scale_routed_experts(model, model.config.text_config.hidden_size)
    model.to(torch.bfloat16).save_pretrained(save_dir, safe_serialization=True)
    return save_dir
