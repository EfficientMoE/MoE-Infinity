import os
from types import SimpleNamespace

import pytest

from moe_infinity.spec_decode import (
    DFlashConfig,
    read_dflash_config,
    validate_pairing,
)

DRAFT = os.environ.get("DFLASH_DRAFT", "z-lab/gpt-oss-120b-DFlash")
TARGET = os.environ.get("DFLASH_TARGET", "openai/gpt-oss-120b")


def _draft_config_stub():
    return SimpleNamespace(
        block_size=10,
        hidden_size=2880,
        vocab_size=201088,
        num_target_layers=36,
        dflash_config={
            "mask_token_id": 200000,
            "target_layer_ids": [1, 9, 17, 25, 33],
        },
    )


def test_read_dflash_config_from_stub():
    cfg = read_dflash_config(_draft_config_stub())
    assert cfg.block_size == 10
    assert cfg.mask_token_id == 200000
    assert cfg.target_layer_ids == [1, 9, 17, 25, 33]
    assert cfg.hidden_size == 2880
    assert cfg.vocab_size == 201088
    assert cfg.num_target_layers == 36


def test_read_dflash_config_missing_fields_raises():
    with pytest.raises(ValueError):
        read_dflash_config(SimpleNamespace(hidden_size=2880, vocab_size=1))


def test_validate_pairing_accepts_matching_target():
    cfg = read_dflash_config(_draft_config_stub())
    target = SimpleNamespace(hidden_size=2880, vocab_size=201088, num_hidden_layers=36)
    validate_pairing(cfg, target)


def test_validate_pairing_rejects_hidden_mismatch():
    cfg = read_dflash_config(_draft_config_stub())
    target = SimpleNamespace(hidden_size=4096, vocab_size=201088, num_hidden_layers=36)
    with pytest.raises(ValueError):
        validate_pairing(cfg, target)


def test_validate_pairing_rejects_mask_outside_vocab():
    cfg = DFlashConfig(
        block_size=10,
        mask_token_id=200000,
        target_layer_ids=[1, 9, 17, 25, 33],
        num_target_layers=36,
        hidden_size=2880,
        vocab_size=201088,
    )
    target = SimpleNamespace(hidden_size=2880, vocab_size=1000, num_hidden_layers=36)
    with pytest.raises(ValueError):
        validate_pairing(cfg, target)


def test_validate_pairing_rejects_layer_out_of_range():
    cfg = DFlashConfig(
        block_size=10,
        mask_token_id=200000,
        target_layer_ids=[1, 9, 17, 25, 40],
        num_target_layers=36,
        hidden_size=2880,
        vocab_size=201088,
    )
    target = SimpleNamespace(hidden_size=2880, vocab_size=201088, num_hidden_layers=36)
    with pytest.raises(ValueError):
        validate_pairing(cfg, target)


@pytest.mark.network
def test_checkpoint_config_matches_expected():
    transformers = pytest.importorskip("transformers")
    try:
        draft_cfg = transformers.AutoConfig.from_pretrained(DRAFT, trust_remote_code=True)
        target_cfg = transformers.AutoConfig.from_pretrained(TARGET, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"checkpoint configs unavailable: {exc}")

    cfg = read_dflash_config(draft_cfg)
    assert cfg.block_size == 10
    assert cfg.mask_token_id == 200000
    assert cfg.target_layer_ids == [1, 9, 17, 25, 33]
    assert cfg.hidden_size == 2880
    validate_pairing(cfg, target_cfg)
