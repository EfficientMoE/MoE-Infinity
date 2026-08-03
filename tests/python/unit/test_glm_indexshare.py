import pytest
from moe_infinity.models.glm_dsa import num_owned_indexers, indexer_owner_map, owns_indexer

_GLM_LOCAL = "/mnt/raid0nvme0/public/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541"


def _real_cfg():
    try:
        from transformers import AutoConfig
        try:
            return AutoConfig.from_pretrained("zai-org/GLM-5.2-FP8", trust_remote_code=True)
        except Exception:
            return AutoConfig.from_pretrained(_GLM_LOCAL, trust_remote_code=True)
    except Exception:
        pytest.skip("GLM config unavailable offline")


def test_owned_indexers_subset_of_layers():
    cfg = _real_cfg()
    n = num_owned_indexers(cfg)
    assert 0 < n <= cfg.num_hidden_layers


def test_shared_layers_map_to_full_owner():
    cfg = _real_cfg()
    m = indexer_owner_map(cfg)
    for layer, owner in m.items():
        if owner is not None:
            assert owns_indexer(cfg, owner), f"layer {layer} owner {owner} must be a 'full' layer"
