from types import SimpleNamespace

import torch

from moe_infinity.runtime.model_offload import (
    _expand_gpt_oss_packed_experts,
)


def _config(layers=2, experts=128):
    return SimpleNamespace(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=layers,
        num_local_experts=experts,
    )


def _packed_layer(layer_id, experts=128):
    prefix = f"model.layers.{layer_id}.mlp.experts"
    return {
        f"{prefix}.gate_up_proj_blocks": torch.empty(
            experts, 12, 8, dtype=torch.uint8
        ),
        f"{prefix}.gate_up_proj_scales": torch.empty(
            experts, 12, 1, dtype=torch.uint8
        ),
        f"{prefix}.gate_up_proj_bias": torch.empty(
            experts, 12, dtype=torch.bfloat16
        ),
        f"{prefix}.down_proj_blocks": torch.empty(
            experts, 6, 6, dtype=torch.uint8
        ),
        f"{prefix}.down_proj_scales": torch.empty(
            experts, 6, 1, dtype=torch.uint8
        ),
        f"{prefix}.down_proj_bias": torch.empty(
            experts, 6, dtype=torch.bfloat16
        ),
    }


def test_expansion_creates_128_identities_per_layer_without_copy():
    state = {**_packed_layer(0), **_packed_layer(1)}
    originals = dict(state)

    _expand_gpt_oss_packed_experts(state, _config())

    prefixes = {
        key.rsplit(".", 1)[0] for key in state if ".mlp.experts." in key
    }
    assert len(prefixes) == 128 * 2
    assert len(state) == 128 * 2 * 6

    for layer_id in range(2):
        packed_prefix = f"model.layers.{layer_id}.mlp.experts"
        for expert_idx in range(128):
            expert_prefix = f"{packed_prefix}.{expert_idx}"
            for field in (
                "gate_up_proj_blocks",
                "gate_up_proj_scales",
                "gate_up_proj_bias",
                "down_proj_blocks",
                "down_proj_scales",
                "down_proj_bias",
            ):
                view = state[f"{expert_prefix}.{field}"]
                packed = originals[f"{packed_prefix}.{field}"]
                assert view.is_contiguous()
                assert (
                    view.untyped_storage().data_ptr()
                    == packed.untyped_storage().data_ptr()
                )
                assert view.storage_offset() == expert_idx * packed.stride(0)


def test_expansion_rejects_incomplete_layer():
    state = _packed_layer(0)
    del state["model.layers.0.mlp.experts.down_proj_scales"]

    try:
        _expand_gpt_oss_packed_experts(state, _config(layers=1))
    except ValueError as exc:
        assert "down_proj_scales" in str(exc)
    else:
        raise AssertionError("incomplete GPT-OSS packed layer was accepted")
