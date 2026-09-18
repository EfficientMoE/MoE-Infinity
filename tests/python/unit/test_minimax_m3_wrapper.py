"""MiniMax-M3 (minimax_m3_vl) Stage 3: registry, parsing, Sync MoE block, and
offload residency policy.

Registration is guarded on the transformers ``MiniMaxM3SparseForConditional
Generation`` class; tests that build the config or the HF block skip when it is
absent so the suite stays green on transformers builds without minimax_m3_vl.
The Sync block parity test pins the packed -> per-expert state-dict expansion
(``experts.gate_up_proj`` chunked on dim 0 into ``gate_proj``/``up_proj``) and
numerical equivalence to the HF block. The residency tests pin the
whole-namespace policy: only routed experts are offloaded; the text backbone,
shared expert, vision tower, and multimodal projector stay resident.
"""

import warnings

import pytest
import torch
from moe_store.parsing.hf_config import parse_expert_id, parse_moe_param
from moe_store.registry.constants import (
    MODEL_MAPPING_NAMES,
    MODEL_MAPPING_TYPES,
    parse_expert_type,
)
from moe_store.wrappers import SyncMiniMaxM3VLSparseMoeBlock

from moe_infinity.runtime.model_offload import OffloadEngine

_HAS_MINIMAX_M3 = "minimaxm3" in MODEL_MAPPING_NAMES

_SKIP = pytest.mark.skipif(
    not _HAS_MINIMAX_M3, reason="transformers lacks MiniMaxM3 classes"
)


def _arch():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
    )

    cfg = MiniMaxM3VLConfig()
    cfg.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
    return cfg


def _tiny_text_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )

    return MiniMaxM3VLTextConfig(
        num_local_experts=8,
        num_experts_per_tok=2,
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        vocab_size=256,
    )


@_SKIP
def test_minimax_m3_registered():
    assert "minimaxm3" in MODEL_MAPPING_NAMES
    assert (
        MODEL_MAPPING_NAMES["minimaxm3"].__name__
        == "MiniMaxM3SparseForConditionalGeneration"
    )
    assert MODEL_MAPPING_TYPES["minimaxm3"] == 5


@_SKIP
def test_arch_resolver_most_specific():
    assert parse_expert_type(_arch()) == 5


@_SKIP
def test_parse_moe_param_reads_nested_text_config():
    num_layers, num_experts, num_encoder_layers = parse_moe_param(_arch())
    assert num_layers == 60
    assert num_experts == 128
    assert num_encoder_layers == 0


@_SKIP
@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "model.language_model.layers.7.mlp.experts.0.gate_proj.weight",
            (7, 0),
        ),
        (
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            (0, 0),
        ),
        (
            "model.language_model.layers.59.mlp.experts.127.up_proj.weight",
            (59, 127),
        ),
        (
            "model.language_model.layers.0.mlp.shared_experts.gate_up_proj.weight",
            (None, None),
        ),
        ("model.language_model.layers.0.mlp.gate.weight", (None, None)),
        ("model.vision_tower.blocks.0.attn.proj.weight", (None, None)),
        ("model.multi_modal_projector.linear_1.weight", (None, None)),
    ],
)
def test_parse_expert_id_minimax_m3(name, expected):
    assert parse_expert_id(name, _arch()) == expected


@_SKIP
def test_sync_block_matches_hf_block():
    warnings.filterwarnings("ignore")
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLSparseMoeBlock,
    )

    cfg = _tiny_text_config()
    torch.manual_seed(0)

    hf_block = MiniMaxM3VLSparseMoeBlock(cfg).eval()
    for p in hf_block.parameters():
        torch.nn.init.normal_(p, std=0.05)

    sync_block = SyncMiniMaxM3VLSparseMoeBlock(cfg).eval()
    sd = dict(hf_block.state_dict())
    gate_up = sd.pop("experts.gate_up_proj")
    down = sd.pop("experts.down_proj")
    for e in range(cfg.num_local_experts):
        gate_w, up_w = gate_up[e].chunk(2, dim=0)
        sd[f"experts.{e}.gate_proj.weight"] = gate_w.contiguous()
        sd[f"experts.{e}.up_proj.weight"] = up_w.contiguous()
        sd[f"experts.{e}.down_proj.weight"] = down[e].contiguous()
    missing, unexpected = sync_block.load_state_dict(sd, strict=False)
    assert missing == [], f"missing: {missing}"
    assert unexpected == [], f"unexpected: {unexpected}"

    x = torch.randn(2, 6, cfg.hidden_size)
    with torch.no_grad():
        out_hf = hf_block(x)
        out_sync = sync_block(x)

    assert isinstance(out_sync, torch.Tensor)
    assert out_sync.shape == (2, 6, cfg.hidden_size)
    assert torch.allclose(out_hf, out_sync, rtol=1e-3, atol=1e-4)


@_SKIP
def test_sync_block_executor_stays_unset():
    block = SyncMiniMaxM3VLSparseMoeBlock(_tiny_text_config())
    assert block.expert_executor is None


@_SKIP
def test_minimax_m3_topology_distinguishes_shared_and_routed_experts():
    class TinyMiniMaxM3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(
                [torch.nn.Module()]
            )
            layer = self.model.language_model.layers[0]
            layer.mlp = SyncMiniMaxM3VLSparseMoeBlock(_tiny_text_config())

    model = TinyMiniMaxM3()
    engine = object.__new__(OffloadEngine)
    engine.config = _arch()
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
        if ".shared_experts" in name
    }
    assert shared_ids <= set(shared_stage)


def _minimax_resident(name: str) -> bool:
    engine = object.__new__(OffloadEngine)
    engine.config = _arch()
    return engine._is_shared_expert_param(name)


@_SKIP
@pytest.mark.parametrize(
    "name",
    [
        "model.vision_tower.blocks.0.attn.proj.weight",
        "model.vision_tower.embeddings.patch_embedding.weight",
        "model.multi_modal_projector.linear_1.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.shared_experts.gate_up_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "lm_head.weight",
    ],
)
def test_minimax_m3_backbone_vision_and_shared_stay_resident(name):
    assert _minimax_resident(name) is True


@_SKIP
@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.7.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.127.down_proj.weight",
        "model.language_model.layers.59.mlp.experts.63.up_proj.weight",
    ],
)
def test_minimax_m3_routed_experts_are_offloaded(name):
    assert _minimax_resident(name) is False
