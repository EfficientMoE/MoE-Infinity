"""Tiny gpt-oss target + DFlash drafter fixtures — CPU determinism contract.

Pins the Task 5 QA scenarios: (1) tiny target greedy is bit-reproducible on CPU
(the token-identity gate the Task 9 losslessness proof depends on) and (2) the
tiny drafter projects a random 5-layer feature to ``[B, block_size-1, vocab]``
through the *target* lm_head. The fixtures under test deliberately contain NO
DFlash acceptance / verify / rollback logic — that is the code under test in
Task 6.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    TINY_BLOCK_SIZE,
    TINY_HIDDEN,
    TINY_MASK_TOKEN_ID,
    TINY_TARGET_LAYER_IDS,
    TINY_VOCAB,
    build_tiny_drafter,
    build_tiny_target,
    context_feature_from_hidden_states,
    make_tiny_drafter_config,
    plain_greedy_decode,
)

PROMPT = torch.tensor([[3, 7, 11, 2, 5]])


def test_tiny_target_greedy_bit_reproducible_across_fresh_builds():
    g1 = plain_greedy_decode(
        build_tiny_target(seed=0), PROMPT, max_new_tokens=32
    )
    g2 = plain_greedy_decode(
        build_tiny_target(seed=0), PROMPT, max_new_tokens=32
    )

    assert g1.dtype == torch.long
    assert tuple(g1.shape) == (1, PROMPT.shape[1] + 32)
    assert torch.equal(g1, g2), (
        "tiny target greedy decode is not bit-reproducible across two fresh "
        f"same-seed builds:\n  run1={g1.tolist()}\n  run2={g2.tolist()}"
    )


def test_tiny_target_greedy_reproducible_same_instance():
    model = build_tiny_target(seed=0)
    a = plain_greedy_decode(model, PROMPT, max_new_tokens=24)
    b = plain_greedy_decode(model, PROMPT, max_new_tokens=24)
    assert torch.equal(a, b)


def test_tiny_drafter_shape_via_target_lm_head():
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)

    batch = 1
    ctx_len = PROMPT.shape[1]
    feat_dim = len(TINY_TARGET_LAYER_IDS) * TINY_HIDDEN
    feature = torch.randn(batch, ctx_len, feat_dim)

    anchor = 4
    block = torch.tensor(
        [[anchor] + [TINY_MASK_TOKEN_ID] * (TINY_BLOCK_SIZE - 1)]
    )
    assert tuple(block.shape) == (batch, TINY_BLOCK_SIZE)

    with torch.no_grad():
        drafter_out = drafter(block, feature)
        assert tuple(drafter_out.shape) == (batch, TINY_BLOCK_SIZE, TINY_HIDDEN)
        draft_logits = target.lm_head(drafter_out)[
            :, -(TINY_BLOCK_SIZE - 1) :, :
        ]

    assert tuple(draft_logits.shape) == (batch, TINY_BLOCK_SIZE - 1, TINY_VOCAB)
    assert torch.isfinite(draft_logits).all()


def test_tiny_drafter_deterministic_and_non_causal():
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)

    feature = torch.randn(
        1, PROMPT.shape[1], len(TINY_TARGET_LAYER_IDS) * TINY_HIDDEN
    )
    block = torch.tensor([[4] + [TINY_MASK_TOKEN_ID] * (TINY_BLOCK_SIZE - 1)])

    with torch.no_grad():
        out_a = drafter(block, feature)
        out_b = drafter(block, feature)

    assert torch.equal(out_a, out_b)
    # Non-causal attention is part of the DFlash drafter contract (RFC §1.2).
    assert getattr(drafter, "is_causal", True) is False


def test_target_exposes_five_layer_context_feature():
    target = build_tiny_target(seed=0)
    with torch.no_grad():
        out = target(PROMPT, output_hidden_states=True, use_cache=False)

    # hidden_states[0] is the embedding output; layer i output is index i+1.
    assert len(out.hidden_states) == target.config.num_hidden_layers + 1

    feat = context_feature_from_hidden_states(
        out.hidden_states, TINY_TARGET_LAYER_IDS
    )
    assert tuple(feat.shape[:2]) == (1, PROMPT.shape[1])
    assert feat.shape[-1] == len(TINY_TARGET_LAYER_IDS) * TINY_HIDDEN


def test_tiny_pair_parsed_by_real_config_layer():
    from moe_infinity.spec_decode import read_dflash_config

    target = build_tiny_target(seed=0)
    cfg = read_dflash_config(make_tiny_drafter_config(target.config))

    assert cfg.block_size == TINY_BLOCK_SIZE
    assert cfg.mask_token_id == TINY_MASK_TOKEN_ID
    assert list(cfg.target_layer_ids) == list(TINY_TARGET_LAYER_IDS)
    assert cfg.hidden_size == TINY_HIDDEN == int(target.config.hidden_size)
    assert cfg.vocab_size == TINY_VOCAB == int(target.config.vocab_size)

    assert cfg.mask_token_id < int(target.config.vocab_size)
    assert max(cfg.target_layer_ids) + 1 <= int(target.config.num_hidden_layers)
