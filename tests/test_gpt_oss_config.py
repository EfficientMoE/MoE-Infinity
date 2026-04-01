from unittest.mock import MagicMock


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


def test_parse_expert_id_gpt_oss_packed():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.mlp.experts.gate_up_proj_blocks", config
    )
    assert layer_id == 5
    assert expert_id is None


def test_parse_expert_id_gpt_oss_router():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.mlp.router.weight", config
    )
    assert layer_id is None
    assert expert_id is None


def test_parse_expert_id_gpt_oss_down_proj():
    from moe_infinity.utils.hf_config import parse_expert_id

    config = make_gpt_oss_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.11.mlp.experts.down_proj_blocks", config
    )
    assert layer_id == 11
    assert expert_id is None
