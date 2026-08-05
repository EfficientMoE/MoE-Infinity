"""Contract-assert tests for the hardened DFlash drafter loader (Task 4).

Uses FAKE config/model objects only (no network, no checkpoint download, no
GPU). ``DFlashSpeculator.__init__`` calls the same helpers on the
``AutoModel.from_pretrained(..., trust_remote_code=True)`` drafter.

Enforced contract (z-lab/gpt-oss-120b-DFlash):
    hidden_size == target.hidden_size
    fc.in_features == 5 * hidden_size            (== 14400)
    mask_token_id (200000) < vocab_size
    block_size == 10
    target_layer_ids == [1, 9, 17, 25, 33]
    vocab_size == target.vocab_size
"""

from types import SimpleNamespace

import pytest

from moe_infinity.spec_decode.dflash import (
    bind_shared_weights,
    read_dflash_config,
    validate_drafter,
    validate_pairing,
)


def _fake_draft_config(
    *,
    block_size=10,
    hidden_size=2880,
    vocab_size=201088,
    mask_token_id=200000,
    target_layer_ids=None,
    num_target_layers=36,
):
    return SimpleNamespace(
        block_size=block_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_target_layers=num_target_layers,
        dflash_config={
            "mask_token_id": mask_token_id,
            "target_layer_ids": list(target_layer_ids or [1, 9, 17, 25, 33]),
        },
    )


def _fake_drafter(*, fc_in_features=14400, **cfg_kwargs):
    return SimpleNamespace(
        config=_fake_draft_config(**cfg_kwargs),
        fc=SimpleNamespace(in_features=fc_in_features),
    )


def _fake_target_config(*, hidden_size=2880, vocab_size=201088, num_hidden_layers=36):
    return SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
    )


def test_validate_drafter_accepts_matching_contract():
    validate_drafter(_fake_drafter(), _fake_target_config())


def test_validate_drafter_accepts_when_config_precomputed():
    drafter = _fake_drafter()
    cfg = read_dflash_config(drafter.config)
    validate_drafter(drafter, _fake_target_config(), draft_cfg=cfg)


def test_validate_drafter_rejects_fc_in_features_mismatch():
    with pytest.raises(ValueError, match="in_features"):
        validate_drafter(_fake_drafter(fc_in_features=9999), _fake_target_config())


def test_validate_drafter_requires_fc_projection():
    drafter = SimpleNamespace(config=_fake_draft_config())
    with pytest.raises(ValueError, match="fc"):
        validate_drafter(drafter, _fake_target_config())


def test_validate_pairing_rejects_hidden_mismatch():
    cfg = read_dflash_config(_fake_draft_config(hidden_size=2880))
    with pytest.raises(ValueError, match="hidden_size"):
        validate_pairing(cfg, _fake_target_config(hidden_size=4096))


def test_validate_pairing_rejects_mask_ge_vocab():
    cfg = read_dflash_config(_fake_draft_config(mask_token_id=201088))
    with pytest.raises(ValueError, match="mask_token_id"):
        validate_pairing(cfg, _fake_target_config())


def test_validate_pairing_rejects_block_size_mismatch():
    cfg = read_dflash_config(_fake_draft_config(block_size=8))
    with pytest.raises(ValueError, match="block_size"):
        validate_pairing(cfg, _fake_target_config())


def test_validate_pairing_rejects_target_layer_ids_mismatch():
    cfg = read_dflash_config(_fake_draft_config(target_layer_ids=[1, 9, 17, 25, 34]))
    with pytest.raises(ValueError, match="target_layer_ids"):
        validate_pairing(cfg, _fake_target_config())


def test_validate_pairing_rejects_vocab_mismatch():
    cfg = read_dflash_config(_fake_draft_config(vocab_size=201088))
    with pytest.raises(ValueError, match="vocab_size"):
        validate_pairing(cfg, _fake_target_config(vocab_size=201000))


def test_bind_shared_weights_uses_target_references():
    embed = SimpleNamespace(tag="embed")
    lm_head = SimpleNamespace(tag="lm_head")
    target = SimpleNamespace(
        get_input_embeddings=lambda: embed,
        get_output_embeddings=lambda: lm_head,
    )
    drafter = SimpleNamespace()

    embed_ref, lm_head_ref = bind_shared_weights(drafter, target)

    assert drafter.embed_tokens is embed
    assert drafter.lm_head is lm_head
    assert embed_ref is embed
    assert lm_head_ref is lm_head


def test_bind_shared_weights_falls_back_to_attributes():
    embed = SimpleNamespace(tag="embed")
    lm_head = SimpleNamespace(tag="lm_head")
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=embed),
        lm_head=lm_head,
    )
    drafter = SimpleNamespace()

    bind_shared_weights(drafter, target)

    assert drafter.embed_tokens is embed
    assert drafter.lm_head is lm_head


def test_bind_shared_weights_raises_when_embeddings_unresolvable():
    target = SimpleNamespace()
    with pytest.raises(ValueError, match="embed"):
        bind_shared_weights(SimpleNamespace(), target)
