from unittest.mock import MagicMock


def import_constants_module():
    import importlib.machinery
    import sys
    import types

    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    sys.modules["flash_attn"] = module

    from moe_infinity.common import constants

    return constants


def make_gpt_oss_config():
    cfg = MagicMock()
    cfg.architectures = ["GptOssForCausalLM"]
    cfg.model_type = "gpt_oss"
    cfg.num_hidden_layers = 24
    cfg.num_local_experts = 32
    cfg.num_experts_per_tok = 4
    cfg.hidden_size = 2880
    cfg.intermediate_size = 2880
    return cfg


def test_parse_moe_param_gpt_oss():
    from moe_infinity.utils.hf_config import parse_moe_param

    config = make_gpt_oss_config()
    num_layers, num_experts, num_encoder_layers = parse_moe_param(config)
    assert num_layers == 24
    assert num_experts == 32
    assert num_encoder_layers == 0


def test_parse_expert_id_gpt_oss_gate_up_slice():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    assert parse_expert_id(
        "model.layers.5.mlp.experts.17.gate_up_proj_blocks", config
    ) == (5, 17)


def test_parse_expert_id_gpt_oss_down_slice():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    assert parse_expert_id(
        "model.layers.11.mlp.experts.31.down_proj_bias", config
    ) == (11, 31)


def test_parse_expert_id_gpt_oss_rejects_out_of_range_expert():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    assert parse_expert_id(
        "model.layers.5.mlp.experts.32.gate_up_proj_blocks", config
    ) == (None, None)


def test_parse_expert_id_gpt_oss_router():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    assert parse_expert_id("model.layers.5.mlp.router.weight", config) == (
        None,
        None,
    )


def test_parse_expert_dtype_gpt_oss_none():
    from moe_infinity.utils.hf_config import parse_expert_dtype

    config = MagicMock()
    config.torch_dtype = None
    config.dtype = None
    result = parse_expert_dtype(config)
    assert result == 0, f"Expected 0 (bfloat16), got {result}"


def test_gpt_oss_model_mapping():
    MODEL_MAPPING_NAMES = import_constants_module().MODEL_MAPPING_NAMES
    from transformers import GptOssForCausalLM

    assert (
        "gptoss" in MODEL_MAPPING_NAMES
    ), "gptoss key missing from MODEL_MAPPING_NAMES"
    assert MODEL_MAPPING_NAMES["gptoss"] is GptOssForCausalLM


def test_gpt_oss_has_dedicated_dispatcher_expert_type():
    parse_expert_type = import_constants_module().parse_expert_type

    assert parse_expert_type(make_gpt_oss_config()) == 6


def test_gpt_oss_arch_string_matching():
    MODEL_MAPPING_NAMES = import_constants_module().MODEL_MAPPING_NAMES

    arch_str = "gptossforcausallm"
    matched = None
    for k in MODEL_MAPPING_NAMES:
        if k in arch_str:
            matched = k
            break
    assert matched == "gptoss", f"Expected 'gptoss' to match, got: {matched}"


def test_gpt_oss_flash_attn_excluded_in_big_modeling():
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "moe_infinity"
        / "entrypoints"
        / "big_modeling.py"
    )
    content = source.read_text(encoding="utf-8")

    exclusion_pattern = (
        r"if\s*\([\s\S]*arch\s*==\s*\"deepseek\"[\s\S]*"
        r"arch\s*==\s*\"deepseek_v3\"[\s\S]*"
        r"arch\s*==\s*\"nllb\"[\s\S]*"
        r"arch\s*==\s*\"gptoss\"[\s\S]*\):\s*"
        r"is_flash_attn_available\s*=\s*False"
    )
    assert re.search(
        exclusion_pattern, content
    ), "big_modeling.py must exclude gptoss from flash attention"
