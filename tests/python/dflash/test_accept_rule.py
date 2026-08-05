"""Hand-checked unit tests for the DFlash accept-rule + block-build pure ops.

Autonomous correctness gate for Task 3 of
``.sisyphus/plans/gpt-oss-dflash-native-integration.md``. Pure, CPU-only tensor
helpers only -- no model loading, no KV/state-machine logic.

Accept rule (RFC 1.2), for ``block_size = 10`` (1 anchor + 9 masks):

    block     = [anchor, d1, ..., d9]                       # drafter candidates
    posterior = [p0, ..., p9]                               # argmax of the verify
    accept    = cumprod(block[:, 1:] == posterior[:, :-1]).sum()

``cumprod`` stops at the first mismatch, so ``accept`` counts leading matches
(0..block_size-1). The step then emits the ``accept`` accepted drafts plus one
bonus token ``posterior[:, accept]`` (the target's correction / next anchor).
"""

from __future__ import annotations

import pytest
import torch

from moe_infinity.spec_decode._dflash_ops import (
    acceptance_length,
    build_block,
    committed_tokens,
)

BLOCK_SIZE = 10
MASK = 200000
ANCHOR = 100
DRAFTS = [10 + i for i in range(1, BLOCK_SIZE)]  # d1..d9 = 11..19


def _row(values) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.long)


def _make_case(k: int):
    """Return ``(block, posterior, expected_accept=k)`` for a hand-checked case.

    Posterior is laid out so that ``acceptance_length`` is exactly ``k``:
    positions ``0..k-1`` equal ``d_{i+1}`` (match); position ``k`` (if ``k<9``)
    is ``900+k`` (forced mismatch); positions ``k+1..8`` are arbitrary
    (post-cumprod, irrelevant); position 9 is ``999``. Hence the bonus token
    ``posterior[:, k]`` is ``900+k`` for ``k<9`` and ``999`` for ``k==9``.
    """
    assert 0 <= k <= BLOCK_SIZE - 1
    block = [ANCHOR] + DRAFTS[:]

    posterior = [0] * BLOCK_SIZE
    for i in range(k):
        posterior[i] = DRAFTS[i]
    if k < BLOCK_SIZE - 1:
        posterior[k] = 900 + k
        for i in range(k + 1, BLOCK_SIZE - 1):
            posterior[i] = 800 + i
    posterior[BLOCK_SIZE - 1] = 999

    return _row(block), _row(posterior), k


def test_build_block_shape_and_contents():
    block = build_block(_row([ANCHOR]), MASK, BLOCK_SIZE)
    assert block.shape == (1, BLOCK_SIZE)
    assert block.dtype == torch.long
    assert block[0, 0].item() == ANCHOR
    assert block[0, 1:].tolist() == [MASK] * (BLOCK_SIZE - 1)


def test_build_block_mask_count_is_block_size_minus_one():
    block = build_block(_row([ANCHOR]), MASK, BLOCK_SIZE)
    assert (block[0] == MASK).sum().item() == BLOCK_SIZE - 1


@pytest.mark.parametrize("anchor", [ANCHOR, _row([ANCHOR]), torch.tensor([[ANCHOR]])])
def test_build_block_accepts_int_and_tensor_anchor(anchor):
    block = build_block(anchor, MASK, BLOCK_SIZE)
    assert block.shape == (1, BLOCK_SIZE)
    assert block[0, 0].item() == ANCHOR
    assert block[0, 1:].tolist() == [MASK] * (BLOCK_SIZE - 1)


def test_build_block_preserves_device_of_tensor_anchor():
    anchor = _row([ANCHOR])
    assert build_block(anchor, MASK, BLOCK_SIZE).device == anchor.device


def test_acceptance_full_accept_equals_block_size_minus_one():
    block, posterior, _ = _make_case(BLOCK_SIZE - 1)
    assert acceptance_length(block, posterior) == BLOCK_SIZE - 1 == 9


def test_acceptance_first_mismatch_at_k():
    k = 4
    block, posterior, _ = _make_case(k)
    assert acceptance_length(block, posterior) == k


def test_acceptance_none_immediate_mismatch():
    block, posterior, _ = _make_case(0)
    assert acceptance_length(block, posterior) == 0


@pytest.mark.parametrize("k", list(range(BLOCK_SIZE)))
def test_acceptance_length_matches_expected_for_every_k(k):
    block, posterior, expected = _make_case(k)
    assert acceptance_length(block, posterior) == expected


def test_acceptance_length_returns_python_int():
    block, posterior, _ = _make_case(3)
    assert isinstance(acceptance_length(block, posterior), int)


def test_acceptance_explicit_handchecked_vector():
    #   block[:, 1:]      = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    #   posterior[:, :-1] = [11, 12, 13, 77, 55,  0,  0,  0,  0]
    #   matches           = [ T,  T,  T,  F,  F,  F,  F,  F,  F]  => cumprod sum = 3
    block = _row([100, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    posterior = _row([11, 12, 13, 77, 55, 0, 0, 0, 0, 999])
    assert acceptance_length(block, posterior) == 3


def test_committed_accept0_emits_single_bonus_token():
    # QA task-3-bonus-token: accept==0 => 1 emitted token == posterior[:, 0].
    block, posterior, _ = _make_case(0)
    res = committed_tokens(block, posterior, accept=0)
    assert res.emitted.shape == (1, 1)
    assert res.emitted[0, 0].item() == posterior[0, 0].item()


def test_committed_accept9_emits_ten_tokens_last_is_bonus():
    # QA task-3-bonus-token: accept==9 => 10 emitted tokens, last == posterior[:, 9].
    block, posterior, _ = _make_case(BLOCK_SIZE - 1)
    res = committed_tokens(block, posterior, accept=BLOCK_SIZE - 1)
    assert res.emitted.shape == (1, BLOCK_SIZE)
    assert res.emitted[0, -1].item() == posterior[0, BLOCK_SIZE - 1].item()


@pytest.mark.parametrize("k", list(range(BLOCK_SIZE)))
def test_committed_emitted_is_accepted_drafts_then_bonus(k):
    block, posterior, _ = _make_case(k)
    res = committed_tokens(block, posterior, accept=k)
    assert res.emitted.shape == (1, k + 1)
    if k > 0:
        assert res.emitted[0, :k].tolist() == block[0, 1 : k + 1].tolist()
    assert res.bonus.shape == (1, 1)
    assert res.bonus[0, 0].item() == posterior[0, k].item()
    assert res.emitted[0, -1].item() == posterior[0, k].item()


@pytest.mark.parametrize("k", list(range(BLOCK_SIZE)))
def test_committed_block_prefix_is_anchor_plus_accepted_drafts(k):
    # block_prefix = block[:, :accept+1] (anchor + accepted drafts): the KV-retained
    # slice (start += accept+1). The bonus is excluded -- emitted-but-not-cached.
    block, posterior, _ = _make_case(k)
    res = committed_tokens(block, posterior, accept=k)
    assert res.block_prefix.shape == (1, k + 1)
    assert res.block_prefix[0].tolist() == block[0, : k + 1].tolist()
    assert res.block_prefix[0, 0].item() == ANCHOR


def test_committed_bonus_absent_from_cached_prefix_on_partial_accept():
    block, posterior, _ = _make_case(4)
    res = committed_tokens(block, posterior, accept=4)
    assert res.bonus[0, 0].item() == posterior[0, 4].item()
    assert res.bonus[0, 0].item() not in res.block_prefix[0].tolist()


def test_build_then_accept_then_commit_full_pipeline():
    block = build_block(_row([ANCHOR]), MASK, BLOCK_SIZE)
    block[0, 1:] = _row(DRAFTS)
    posterior = _row(DRAFTS + [999])

    accept = acceptance_length(block, posterior)
    assert accept == BLOCK_SIZE - 1

    res = committed_tokens(block, posterior, accept=accept)
    assert res.emitted[0].tolist() == DRAFTS + [999]
    assert res.block_prefix[0].tolist() == [ANCHOR] + DRAFTS
    assert res.bonus[0, 0].item() == 999
