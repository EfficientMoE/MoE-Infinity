from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantizationConfig:
    quant_method: str
    modules_to_not_convert: list[str]


@dataclass
class ModelConfig:
    model_type: str
    quantization_config: Optional[QuantizationConfig]


def make_mxfp4_config():
    return ModelConfig(
        model_type="gpt_oss",
        quantization_config=QuantizationConfig(
            quant_method="mxfp4",
            modules_to_not_convert=[
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
        ),
    )


def make_non_quantized_config():
    return ModelConfig(model_type="mixtral", quantization_config=None)


def test_is_mxfp4_quantized_positive():
    from moe_infinity.utils.mxfp4 import is_mxfp4_quantized

    config = make_mxfp4_config()
    assert is_mxfp4_quantized(config) is True


def test_is_mxfp4_quantized_negative():
    from moe_infinity.utils.mxfp4 import is_mxfp4_quantized

    config = make_non_quantized_config()
    assert is_mxfp4_quantized(config) is False


def test_get_modules_to_not_convert():
    from moe_infinity.utils.mxfp4 import get_mxfp4_modules_to_not_convert

    config = make_mxfp4_config()
    modules = get_mxfp4_modules_to_not_convert(config)
    assert isinstance(modules, list)
    assert len(modules) > 0
    assert "model.embed_tokens" in modules
    assert "lm_head" in modules


def test_identify_mxfp4_pairs():
    from moe_infinity.utils.mxfp4 import identify_mxfp4_pairs

    weight_names = [
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
        "model.layers.0.mlp.experts.down_proj_blocks",
        "model.layers.0.mlp.experts.down_proj_scales",
        "model.layers.0.mlp.experts.down_proj_bias",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
    ]
    pairs = identify_mxfp4_pairs(weight_names)
    assert len(pairs) == 2
    for blocks_name, scales_name in pairs:
        assert blocks_name.endswith("_blocks")
        assert scales_name.endswith("_scales")
        assert blocks_name[: -len("_blocks")] == scales_name[: -len("_scales")]
