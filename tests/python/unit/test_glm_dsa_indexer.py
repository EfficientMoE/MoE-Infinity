from types import SimpleNamespace

import pytest

from moe_infinity.models.glm_dsa import (
    get_indexer_types,
    indexer_owner_map,
    num_owned_indexers,
    owns_indexer,
)


def cfg_with_types(types):
    return SimpleNamespace(indexer_types=types, num_hidden_layers=len(types))


def cfg_derive(n=12, freq=4, first_dense=3):
    return SimpleNamespace(
        indexer_types=None,
        num_hidden_layers=n,
        index_topk_freq=freq,
        first_k_dense_replace=first_dense,
    )


def test_owns_indexer_explicit():
    c = cfg_with_types(["full", "shared", "shared", "full", "shared"])
    assert owns_indexer(c, 0) is True
    assert owns_indexer(c, 1) is False
    assert owns_indexer(c, 3) is True


def test_owns_indexer_out_of_bounds():
    c = cfg_with_types(["full", "shared"])
    assert owns_indexer(c, -1) is False
    assert owns_indexer(c, 2) is False


def test_owner_map_explicit():
    c = cfg_with_types(["full", "shared", "shared", "full", "shared"])
    m = indexer_owner_map(c)
    assert m == {0: 0, 1: 0, 2: 0, 3: 3, 4: 3}


def test_num_owned():
    c = cfg_with_types(["full", "shared", "shared", "full", "shared"])
    assert num_owned_indexers(c) == 2


def test_num_owned_all_full():
    c = cfg_with_types(["full", "full", "full"])
    assert num_owned_indexers(c) == 3


def test_num_owned_none_layers():
    c = cfg_with_types(["none", "none", "full", "shared"])
    assert num_owned_indexers(c) == 1


def test_derive_from_freq():
    c = cfg_derive(n=11, freq=4, first_dense=3)
    types = get_indexer_types(c)
    assert types[0] == "none"
    assert types[1] == "none"
    assert types[2] == "none"
    assert types[3] == "full"
    assert types[4] == "shared"
    assert types[5] == "shared"
    assert types[6] == "shared"
    assert types[7] == "full"
    assert types[8] == "shared"
    assert types[9] == "shared"
    assert types[10] == "shared"


def test_derive_length():
    c = cfg_derive(n=11, freq=4, first_dense=3)
    assert len(get_indexer_types(c)) == 11


def test_shared_before_any_full_maps_none():
    c = cfg_with_types(["shared", "full", "shared"])
    m = indexer_owner_map(c)
    assert m[0] is None
    assert m[1] == 1
    assert m[2] == 1


def test_none_layers_map_to_none():
    c = cfg_with_types(["none", "none", "full", "shared"])
    m = indexer_owner_map(c)
    assert m[0] is None
    assert m[1] is None
    assert m[2] == 2
    assert m[3] == 2


def test_get_indexer_types_passthrough():
    types = ["full", "shared", "none"]
    c = cfg_with_types(types)
    assert get_indexer_types(c) == types


def test_real_config_owned_fraction():
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(
            "/mnt/raid0nvme0/public/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541",
            trust_remote_code=True,
        )
    except Exception:
        pytest.skip("GLM config not available offline")
    n_owned = num_owned_indexers(c)
    total = c.num_hidden_layers
    assert 0 < n_owned <= total
    assert n_owned == 21


def test_real_config_owner_map_consistency():
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(
            "/mnt/raid0nvme0/public/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541",
            trust_remote_code=True,
        )
    except Exception:
        pytest.skip("GLM config not available offline")
    m = indexer_owner_map(c)
    types = get_indexer_types(c)
    for i, t in enumerate(types):
        if t == "full":
            assert m[i] == i
        elif t == "shared":
            owner = m[i]
            assert owner is not None
            assert types[owner] == "full"
            assert owner < i
        else:
            assert m[i] is None
