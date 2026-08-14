import warnings

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from moe_infinity.common.constants import (
    MODEL_MAPPING_NAMES,
    MODEL_MAPPING_TYPES,
    parse_expert_type,
)
from moe_infinity.models import SyncQwen3_5MoeSparseMoeBlock
from moe_infinity.runtime.model_offload import OffloadEngine
from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param


def _arch(name):
    cfg = Qwen3_5MoeConfig()
    cfg.architectures = [name]
    return cfg


def _tiny_text_config():
    return Qwen3_5MoeTextConfig(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        vocab_size=256,
        hidden_act="silu",
    )


def test_qwen3_5_registered():
    assert "qwen3_5" in MODEL_MAPPING_NAMES
    assert (
        MODEL_MAPPING_NAMES["qwen3_5"].__name__
        == "Qwen3_5MoeForConditionalGeneration"
    )
    assert MODEL_MAPPING_TYPES["qwen3_5"] == 5


@pytest.mark.parametrize(
    "arch, expected_type",
    [
        ("Qwen3_5MoeForConditionalGeneration", 5),
        ("Qwen3MoeForCausalLM", 5),
        ("DeepseekV3ForCausalLM", 5),
        ("MixtralForCausalLM", 4),
    ],
)
def test_arch_resolver_most_specific(arch, expected_type):
    assert parse_expert_type(_arch(arch)) == expected_type


def test_parse_moe_param_reads_nested_text_config():
    cfg = _arch("Qwen3_5MoeForConditionalGeneration")
    text = cfg.get_text_config()
    assert parse_moe_param(cfg) == (
        text.num_hidden_layers,
        text.num_experts,
        0,
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "model.language_model.layers.13.mlp.experts.7.gate_proj.weight",
            (13, 7),
        ),
        (
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            (0, 0),
        ),
        (
            "model.language_model.layers.39.mlp.experts.255.up_proj.weight",
            (39, 255),
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            (None, None),
        ),
        ("model.language_model.layers.0.mlp.gate.weight", (None, None)),
        ("mtp.layers.0.mlp.experts.0.gate_proj.weight", (None, None)),
        ("model.visual.blocks.0.mlp.linear_fc1.weight", (None, None)),
    ],
)
def test_parse_expert_id_qwen3_5(name, expected):
    cfg = _arch("Qwen3_5MoeForConditionalGeneration")
    assert parse_expert_id(name, cfg) == expected


def test_sync_block_matches_hf_block():
    warnings.filterwarnings("ignore")
    cfg = _tiny_text_config()
    torch.manual_seed(0)

    hf_block = Qwen3_5MoeSparseMoeBlock(cfg).eval()
    for p in hf_block.parameters():
        torch.nn.init.normal_(p, std=0.05)

    sync_block = SyncQwen3_5MoeSparseMoeBlock(cfg).eval()
    sd = dict(hf_block.state_dict())
    gate_up = sd.pop("experts.gate_up_proj")
    down = sd.pop("experts.down_proj")
    for e in range(cfg.num_experts):
        gate_w, up_w = gate_up[e].chunk(2, dim=0)
        sd[f"experts.{e}.gate_proj.weight"] = gate_w.contiguous()
        sd[f"experts.{e}.up_proj.weight"] = up_w.contiguous()
        sd[f"experts.{e}.down_proj.weight"] = down[e].contiguous()
    missing, unexpected = sync_block.load_state_dict(sd, strict=False)
    assert missing == [], f"missing: {missing}"

    x = torch.randn(2, 6, cfg.hidden_size)
    with torch.no_grad():
        out_hf = hf_block(x)
        out_sync = sync_block(x)

    assert isinstance(out_sync, torch.Tensor)
    assert out_sync.shape == (2, 6, cfg.hidden_size)
    assert torch.allclose(out_hf, out_sync, rtol=1e-3, atol=1e-4)


def test_sync_block_executor_stays_unset():
    cfg = _tiny_text_config()
    block = SyncQwen3_5MoeSparseMoeBlock(cfg)
    assert block.expert_executor is None


def test_qwen3_5_topology_distinguishes_shared_and_routed_experts():
    class TinyQwen35(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(
                [torch.nn.Module()]
            )
            layer = self.model.language_model.layers[0]
            layer.mlp = SyncQwen3_5MoeSparseMoeBlock(_tiny_text_config())

    model = TinyQwen35()
    engine = object.__new__(OffloadEngine)
    engine.config = _arch("Qwen3_5MoeForConditionalGeneration")
    engine.name_id_map = {
        name: tensor_id
        for tensor_id, (name, _) in enumerate(model.named_parameters())
    }

    topology = dict(engine.get_topology(model))

    routed = topology["model.language_model.layers.0.mlp.experts"]
    assert len(routed) == 8
    shared_stage = topology["model.language_model.layers.0"][0]
    shared_ids = {
        engine.name_id_map[name]
        for name, _ in model.named_parameters()
        if ".shared_expert" in name
    }
    assert shared_ids <= set(shared_stage)
