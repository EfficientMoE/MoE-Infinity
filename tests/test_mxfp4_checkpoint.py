def test_mxfp4_scales_tensors_skipped():
    from moe_infinity.utils.mxfp4 import identify_mxfp4_pairs

    weight_keys = [
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
        "model.layers.0.mlp.experts.down_proj_blocks",
        "model.layers.0.mlp.experts.down_proj_scales",
        "model.layers.0.mlp.experts.down_proj_bias",
        "model.layers.0.mlp.router.weight",
        "model.embed_tokens.weight",
    ]

    pairs = identify_mxfp4_pairs(weight_keys)
    scales_keys = {scales for _, scales in pairs}
    assert scales_keys == {
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.down_proj_scales",
    }


def test_mxfp4_blocks_and_bias_remain():
    weight_keys = [
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
    ]
    filtered = [k for k in weight_keys if not k.endswith("_scales")]
    assert "model.layers.0.mlp.experts.gate_up_proj_blocks" in filtered
    assert "model.layers.0.mlp.experts.gate_up_proj_bias" in filtered
    assert "model.layers.0.mlp.experts.gate_up_proj_scales" not in filtered


def test_model_offload_has_mxfp4_handling():
    from pathlib import Path

    source = Path("moe_infinity/runtime/model_offload.py").read_text()
    assert "identify_mxfp4_pairs" in source
    assert 'k.endswith("_scales")' in source
    assert 'k.endswith("_blocks")' in source
