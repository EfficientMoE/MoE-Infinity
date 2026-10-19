"""GLM-5.3-Flash checkpoint-key handling against the real safetensors index.

The trimmed fixture holds the real zai-org/GLM-5.3-Flash weight map for one
DSA MoE layer (3, with all 288 routed FP8 experts), one KDA layer's
attention keys (4), and every non-layer key. Pins three invariants of the
resident loader: routed-expert keys resolve to their expert id (offloaded)
while everything else stays resident; transformers-v5 WeightRenaming entries
(mHC hc_* and KDA forget-gate keys) are applied to resident keys but never
touch expert keys; and the only honored fusion is the literal q/k/v_conv1d
concat - wildcard batched-expert converters must not be applied because the
Sync MoE block keeps the raw per-expert layout.
"""

import json
import os
import re

import pytest
from transformers import PretrainedConfig

pytest.importorskip(
    "transformers.models.glm5_next",
    reason="transformers >= 5.16 with glm5_next required",
)

from moe_store.parsing.hf_config import parse_expert_id  # noqa: E402

from moe_infinity.runtime.model_offload import OffloadEngine  # noqa: E402

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3_flash"
)


@pytest.fixture(scope="module")
def config() -> PretrainedConfig:
    with open(os.path.join(FIXTURE_DIR, "config.json")) as f:
        return PretrainedConfig.from_dict(json.load(f))


@pytest.fixture(scope="module")
def weight_map() -> dict:
    path = os.path.join(FIXTURE_DIR, "model.safetensors.index.trimmed.json")
    with open(path) as f:
        return json.load(f)["weight_map"]


def _engine(config) -> OffloadEngine:
    engine = object.__new__(OffloadEngine)
    engine.config = config
    return engine


def test_every_routed_expert_key_resolves_to_its_expert_id(config, weight_map):
    expert_keys = [k for k in weight_map if ".mlp.experts." in k]
    assert len(expert_keys) == 288 * 3 * 2

    for key in expert_keys:
        _, expert_id = parse_expert_id(key, config)
        expected = int(re.search(r"\.experts\.(\d+)\.", key).group(1))
        assert expert_id == expected, key


def test_non_expert_keys_stay_resident(config, weight_map):
    non_expert = [k for k in weight_map if ".mlp.experts." not in k]
    assert non_expert

    for key in non_expert:
        _, expert_id = parse_expert_id(key, config)
        assert expert_id is None, key


def test_renamer_applies_v5_weight_renamings(config, weight_map):
    rename = _engine(config)._resident_ckpt_key_renamer()

    assert "model.language_model.layers.3.hc_attn_base" in weight_map
    assert (
        rename("model.language_model.layers.3.hc_attn_base")
        == "model.language_model.layers.3.attn_hc.base"
    )
    assert (
        rename("model.language_model.layers.3.hc_ffn_scale")
        == "model.language_model.layers.3.ffn_hc.scale"
    )

    assert "model.language_model.layers.4.self_attn.A_log" in weight_map
    assert (
        rename("model.language_model.layers.4.self_attn.A_log")
        == "model.language_model.layers.4.self_attn.forget_gate.A_log"
    )
    assert (
        rename("model.language_model.layers.4.self_attn.f_a_proj.weight")
        == "model.language_model.layers.4.self_attn.forget_gate.f_a_proj.weight"
    )


def test_renamer_never_touches_expert_keys(config, weight_map):
    rename = _engine(config)._resident_ckpt_key_renamer()
    for key in (k for k in weight_map if ".mlp.experts." in k):
        assert rename(key) == key


def test_fusions_cover_kda_conv1d_only(config, weight_map):
    fusions = _engine(config)._resident_ckpt_fusions()

    assert (
        "self_attn.conv1d.weight",
        [
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
        ],
        0,
    ) in fusions

    for key in (
        "model.language_model.layers.4.self_attn.q_conv1d.weight",
        "model.language_model.layers.4.self_attn.k_conv1d.weight",
        "model.language_model.layers.4.self_attn.v_conv1d.weight",
    ):
        assert key in weight_map

    for _target, sources, _dim in fusions:
        assert all("experts" not in s for s in sources)
