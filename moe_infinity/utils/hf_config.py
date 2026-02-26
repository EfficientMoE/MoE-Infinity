import re
from typing import Optional, Tuple

import torch
from transformers import PretrainedConfig


def parse_expert_dtype(config: PretrainedConfig) -> int:
    dtype = config.torch_dtype
    if dtype == torch.bfloat16:
        dtype = 0
    elif dtype == torch.float32:
        dtype = 1
    elif dtype == torch.float16:
        dtype = 2
    elif hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        dtype = 3  # FP8; C++ backend stores as BF16, casting handled externally
    else:
        assert False, "Unknown dtype %s" % dtype

    return dtype


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int]:
    arch = config.architectures[0].lower()
    model_type = getattr(config, "model_type", "").lower()

    if "nllb" in arch:
        num_encoder_layers = config.encoder_layers // config.encoder_sparse_step
        num_decoder_layers = config.decoder_layers // config.decoder_sparse_step
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "mixtral" in arch or "arctic" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    elif "grok" in arch or "qwen3" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
    elif "deepseek" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
    elif "minimax_m2" in arch or model_type == "minimax_m2":
        # MiniMax-M2.5: 62 MoE layers, 256 experts per layer
        # MTP layers (if any) are not counted here
        num_encoder_layers = 0
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts  # 256
    elif "kimi_vl" in arch or model_type == "kimi_vl":
        # Kimi-VL: language decoder has 27 layers, 64 routed experts
        # MoE params live in config.text_config for VL models
        text_cfg = getattr(config, "text_config", config)
        num_encoder_layers = 0
        num_layers = text_cfg.num_hidden_layers  # 27
        num_experts = text_cfg.n_routed_experts  # 64
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers


def parse_expert_id(
    param_name: str, config: PretrainedConfig
) -> Tuple[Optional[int], Optional[int]]:
    arch = config.architectures[0].lower()
    model_type = getattr(config, "model_type", "").lower()
    _, _, num_encoder_layers = parse_moe_param(config)

    if "nllb" in arch:
        # example "decoder.block.1.layer.2.mlp.experts.expert_100.wi.weight"
        encoder_sparse_step = config.encoder_sparse_step
        decoder_sparse_step = config.decoder_sparse_step

        result = re.findall(
            r"(encoder|decoder)\.[a-z]+\.(\d+).*expert_(\d+)", param_name
        )

        if result:
            layer_type, layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    elif "mixtral" in arch or "arctic" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        result = re.findall(
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "grok" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.moe_block.experts.0.linear_1.weight"
        result = re.findall(
            r"layers\.(\d+)\.moe_block\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            # print(f"layer_id: {layer_id}, expert_id: {expert_id}")
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "deepseek" in arch or "qwen3" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.1.mlp.experts.0.gate_proj.weight"
        result = re.findall(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            # print(f"layer_id: {layer_id}, expert_id: {expert_id}")
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "minimax_m2" in arch or model_type == "minimax_m2":
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # TODO: Confirm exact param path by inspecting MiniMax-M2.5 state dict.
        # Expected: "model.layers.{i}.feed_forward.experts.{j}.*"
        result = re.findall(
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "kimi_vl" in arch or model_type == "kimi_vl":
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # TODO: Confirm exact param path by inspecting Kimi-VL state dict.
        # Expected: "language_model.model.layers.{i}.mlp.experts.{j}.*"
        result = re.findall(
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    if result:
        if layer_type == "decoder":
            layer_id = layer_id // decoder_sparse_step + num_encoder_layers
        elif layer_type == "encoder":
            layer_id = layer_id // encoder_sparse_step
        else:
            raise ValueError(f"Unsupported layer type {layer_type}")

        return layer_id, expert_id

    return None, None
