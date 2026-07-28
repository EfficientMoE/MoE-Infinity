import re
from typing import Optional, Tuple

import torch
from transformers import PretrainedConfig

_DEEPSEEK_V2_DEFAULTS = {
    "head_dim": None,
    "mlp_bias": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "aux_loss_alpha": 0.001,
    "seq_aux": True,
    "norm_topk_prob": False,
}


def _apply_missing_attrs(cfg, defaults):
    changed = False
    for key, default in defaults.items():
        if key not in cfg.__dict__:
            if key == "head_dim":
                val = getattr(cfg, "qk_rope_head_dim", 0)
                if val == 0 and hasattr(cfg, "num_attention_heads"):
                    val = cfg.hidden_size // cfg.num_attention_heads
            else:
                val = default
            cfg.__dict__[key] = val
            changed = True
    return changed


def ensure_config_compat(config: PretrainedConfig) -> PretrainedConfig:
    arch_name = getattr(config, "model_type", "")
    if "deepseek" not in arch_name:
        return config

    if _apply_missing_attrs(config, _DEEPSEEK_V2_DEFAULTS):
        cfg_cls = type(config)
        _orig_init = cfg_cls.__init__

        def _patched_init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            _apply_missing_attrs(self, _DEEPSEEK_V2_DEFAULTS)

        cfg_cls.__init__ = _patched_init
    return config


def resolve_config_dtype(config: object) -> Optional[torch.dtype]:
    # transformers v5 renamed config.torch_dtype -> config.dtype. Read the new
    # name first, fall back to the old one so both 4.x and 5.x configs work.
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    return dtype


def parse_expert_dtype(config: PretrainedConfig) -> int:
    dtype = resolve_config_dtype(config)
    if dtype is None:
        dtype = torch.bfloat16
    if dtype == torch.bfloat16:
        return 0
    elif dtype == torch.float32:
        return 1
    elif dtype == torch.float16:
        return 2
    else:
        assert False, "Unknown dtype %s" % dtype


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int]:
    arch = (config.architectures or [""])[0].lower()

    if "nllb" in arch:
        encoder_sparse_step = int(config.encoder_sparse_step)
        decoder_sparse_step = int(config.decoder_sparse_step)
        num_encoder_layers = config.encoder_layers // encoder_sparse_step
        num_decoder_layers = config.decoder_layers // decoder_sparse_step
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "mixtral" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    elif "qwen3" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
    elif "deepseek" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
    elif "gpt_oss" in arch or "gptoss" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    elif "glmmoedsa" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers


def parse_expert_id(
    param_name: str, config: PretrainedConfig
) -> Tuple[Optional[int], Optional[int]]:
    arch = (config.architectures or [""])[0].lower()
    _, _, num_encoder_layers = parse_moe_param(config)
    result = None
    layer_type = ""
    layer_id = 0
    expert_id = 0
    encoder_sparse_step = 1
    decoder_sparse_step = 1

    if "nllb" in arch:
        # example "decoder.block.1.layer.2.mlp.experts.expert_100.wi.weight"
        encoder_sparse_step = int(config.encoder_sparse_step)
        decoder_sparse_step = int(config.decoder_sparse_step)

        result = re.findall(
            r"(encoder|decoder)\.[a-z]+\.(\d+).*expert_(\d+)", param_name
        )

        if result:
            layer_type, layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    elif "mixtral" in arch:
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
    elif "deepseekv4" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # native key: "layers.1.ffn.experts.0.w1.weight"
        result = re.findall(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "deepseek" in arch or "qwen3" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.1.mlp.experts.0.gate_proj.weight"
        result = re.findall(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            # print(f"layer_id: {layer_id}, expert_id: {expert_id}")
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "gpt_oss" in arch or "gptoss" in arch:
        result = re.findall(
            r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)",
            param_name,
        )
        if result:
            layer_id = int(result[0][0])
            return layer_id, None
    elif "glmmoedsa" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        result = re.findall(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
            if layer_id >= config.num_hidden_layers:
                return None, None

    if result:
        if layer_type == "decoder":
            layer_id = layer_id // decoder_sparse_step + num_encoder_layers
        elif layer_type == "encoder":
            layer_id = layer_id // encoder_sparse_step
        else:
            raise ValueError(f"Unsupported layer type {layer_type}")

        return layer_id, expert_id

    return None, None
