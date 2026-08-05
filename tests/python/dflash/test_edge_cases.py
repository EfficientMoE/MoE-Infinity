"""Task 7: edge-case handling in the native DFlash loop.

Pins the six Task-7 edge cases on the tiny CPU fixtures (T5), with forced
accept lengths / EOS placements via a scripted drafter head and targeted
target-argmax overrides:

(a) first step — the anchor comes from the prefill forward over the prompt.
(b) full-accept (``accept == block_size - 1 == 9``) — commit 10, continue.
(c) full-reject (``accept == 0``) — commit exactly 1 (the bonus becomes the
    next anchor); start advances by 1.
(d) EOS inside a block — emitted is truncated at the first stop id
    (inclusive), the loop stops, and neither cache extends past the last
    kept token. Covered for an EOS as an *accepted draft* (block position 3;
    cached, since it went through the verify forward) and as the *bonus*
    (emitted but not cached, so the cache trails the emitted sequence by
    one), plus EOS as the prefill anchor (stop before the first block).
(e) ``max_new_tokens`` crossing a block boundary — the step that would
    overshoot is truncated to the remaining budget and the loop stops; the
    return is exactly ``max_new_tokens`` new tokens and the cache is cropped
    to the truncated ``start``.
(f) batch > 1 — raises ``NotImplementedError`` naming the batch==1
    constraint (v1 is single-sequence).

State accounting: ``committed.emitted = [d_1 .. d_accept, bonus]`` while the
verify-forward KV covers ``[anchor, d_1 .. d_accept]`` only. Keeping ``k``
emitted tokens therefore commits ``min(k, accept) + 1`` cached tokens
(anchor + min(k, accept) drafts); the bonus is never cached. The step trace
records the *effective* accept (drafts actually committed), so the Task-6
invariant ``start == prev_start + accept + 1`` holds in every branch.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    TINY_BLOCK_SIZE,
    TINY_HIDDEN,
    TINY_VOCAB,
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
    plain_greedy_decode,
)

from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)

PROMPT = torch.tensor([[3, 7, 11, 2, 5]])
PROMPT_LEN = int(PROMPT.shape[1])
EOS_ID = 62  # absent from the tiny target's greedy continuation for PROMPT

_TARGET = None
_DRAFTER = None


def _tiny_spec():
    """Fresh speculator per test (cheap ``from_models``) over shared models."""
    global _TARGET, _DRAFTER
    if _TARGET is None:
        _TARGET = build_tiny_target(seed=0)
        _DRAFTER = build_tiny_drafter(_TARGET, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(_TARGET.config))
    spec = DFlashSpeculator.from_models(_TARGET, _DRAFTER, config=config, device="cpu")
    return spec, _TARGET


def _greedy_full(target, n_new: int) -> list[int]:
    """Absolute ids (prompt ++ greedy continuation) for indexing drafts."""
    return plain_greedy_decode(target, PROMPT, max_new_tokens=n_new)[0].tolist()


class _ScriptedHead:
    """Fake ``lm_head`` programming the drafter's per-position draft argmax.

    ``draft_fn(call_index)`` must return the ``block_size - 1`` draft ids for
    that drafter pass; the returned logits make the loop's
    ``lm_head(drafter_out)[:, -(block_size-1):].argmax`` pick exactly them.
    """

    def __init__(self, draft_fn) -> None:
        self.draft_fn = draft_fn
        self.calls = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        drafts = [int(t) for t in self.draft_fn(self.calls)]
        self.calls += 1
        assert len(drafts) == TINY_BLOCK_SIZE - 1
        length = hidden.shape[1]
        logits = torch.zeros(
            1, length, TINY_VOCAB, dtype=hidden.dtype, device=hidden.device
        )
        for i, tok in enumerate(drafts):
            logits[0, length - (TINY_BLOCK_SIZE - 1) + i, tok] = 1.0
        return logits


def _install_scripted_drafter(monkeypatch, spec, draft_fn) -> None:
    """Bypass drafter compute; feed scripted draft ids into the accept rule.

    The verify forward stays REAL (the genuine target), so acceptance is
    exercised for real against the scripted drafts.
    """
    monkeypatch.setattr(
        spec,
        "_run_drafter",
        lambda block, context_feature, start, draft_kv: torch.zeros(
            1, TINY_BLOCK_SIZE, TINY_HIDDEN
        ),
    )
    monkeypatch.setattr(spec, "lm_head", _ScriptedHead(draft_fn))


def _force_target_argmax(monkeypatch, spec, *, prefill=None, verify=None):
    """Override the target's argmax at chosen rows of the rich forward.

    ``prefill``/``verify`` map ``row -> token_id``; the row's logits are
    replaced by a one-hot so ``argmax`` yields ``token_id``. KV/hidden are
    untouched, so the override only steers the discrete accept/emission
    path (which is exactly what these tests pin).
    """
    orig = spec._forward_target

    def wrapped(input_ids, past_key_values=None, logits_to_keep=0):
        logits, hidden, kv = orig(
            input_ids, past_key_values=past_key_values, logits_to_keep=logits_to_keep
        )
        rows = prefill if int(logits_to_keep) == 1 else verify
        if rows:
            logits = logits.clone()
            for row, tok in rows.items():
                logits[0, row, :] = -1e9
                logits[0, row, int(tok)] = 1e9
        return logits, hidden, kv

    monkeypatch.setattr(spec, "_forward_target", wrapped)


def _assert_trace_invariants(spec) -> None:
    for rec in spec.step_trace:
        assert rec.start == rec.prev_start + rec.accept + 1
        assert rec.target_cache_len == rec.start


# (a) first step: anchor from the prefill forward over the prompt only
def test_first_step_anchor_from_prefill():
    spec, target = _tiny_spec()

    out = spec.generate(PROMPT, max_new_tokens=2)
    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=2)

    assert len(spec.step_trace) == 1
    rec = spec.step_trace[0]
    assert rec.prev_start == PROMPT_LEN
    assert int(out[0, PROMPT_LEN].item()) == int(plain[0, PROMPT_LEN].item())
    assert torch.equal(out, plain)


# (b) full-accept: accept == block_size - 1 == 9 -> commit 10, continue
def test_full_accept_commits_whole_block_and_continues(monkeypatch):
    spec, target = _tiny_spec()
    full = _greedy_full(target, 25)

    # Drafts == true greedy continuation => every position accepted.
    def draft_fn(step: int):
        anchor_idx = PROMPT_LEN + TINY_BLOCK_SIZE * step
        return full[anchor_idx + 1 : anchor_idx + TINY_BLOCK_SIZE]

    _install_scripted_drafter(monkeypatch, spec, draft_fn)

    max_new = 2 * TINY_BLOCK_SIZE + 1  # two exact full-accept steps
    out = spec.generate(PROMPT, max_new_tokens=max_new)

    assert len(spec.step_trace) == 2
    for i, rec in enumerate(spec.step_trace):
        assert rec.accept == TINY_BLOCK_SIZE - 1
        assert rec.start == PROMPT_LEN + TINY_BLOCK_SIZE * (i + 1)
    _assert_trace_invariants(spec)

    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=max_new)
    assert torch.equal(out, plain)


# (c) full-reject: accept == 0 -> commit 1 (bonus becomes next anchor)
def test_full_reject_commits_bonus_only(monkeypatch):
    spec, target = _tiny_spec()
    full = _greedy_full(target, 8)

    # First draft mismatches the target's argmax => accept == 0 every step;
    # the committed token is the bonus (true greedy token), the next anchor.
    def draft_fn(step: int):
        wrong = (full[PROMPT_LEN + step + 1] + 1) % TINY_VOCAB
        return [wrong] * (TINY_BLOCK_SIZE - 1)

    _install_scripted_drafter(monkeypatch, spec, draft_fn)

    max_new = 5
    out = spec.generate(PROMPT, max_new_tokens=max_new)

    assert len(spec.step_trace) == max_new - 1  # one token per step
    for i, rec in enumerate(spec.step_trace):
        assert rec.accept == 0
        assert rec.start == PROMPT_LEN + (i + 1)
    _assert_trace_invariants(spec)

    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=max_new)
    assert torch.equal(out, plain)


# (d) EOS inside a block
def test_eos_mid_block_accepted_draft_truncates_and_stops(monkeypatch):
    """QA headline: EOS at block position 3 (an accepted draft)."""
    spec, target = _tiny_spec()
    full = _greedy_full(target, 16)

    other = 40  # forced posterior at the position after EOS (never emitted)
    # d1, d2 = greedy (accepted); d3 = EOS (accepted via forced posterior);
    # d4 forced to mismatch so accept == 3.
    drafts = [full[PROMPT_LEN + 1], full[PROMPT_LEN + 2], EOS_ID, (other + 1) % TINY_VOCAB]
    drafts += [0] * (TINY_BLOCK_SIZE - 1 - len(drafts))
    _install_scripted_drafter(monkeypatch, spec, lambda step: drafts)
    _force_target_argmax(monkeypatch, spec, verify={2: EOS_ID, 3: other})

    out = spec.generate(PROMPT, max_new_tokens=40, stop_token_ids=[EOS_ID])

    # One step only (loop stopped); the accept rule saw 3 accepted drafts.
    assert len(spec.step_trace) == 1
    rec = spec.step_trace[0]
    assert rec.accept == 3

    # Emitted ends exactly at EOS: the forced bonus and further blocks are
    # never emitted despite the large max_new_tokens budget.
    new_ids = out[0, PROMPT_LEN:].tolist()
    assert new_ids == [full[PROMPT_LEN], full[PROMPT_LEN + 1], full[PROMPT_LEN + 2], EOS_ID]
    assert other not in new_ids

    # cache length == start_at_EOS: anchor + 3 drafts (the EOS draft DID go
    # through the verify forward, so its KV stays).
    assert rec.start == PROMPT_LEN + 4
    assert rec.target_cache_len == PROMPT_LEN + 4
    assert int(spec.last_target_cache.get_seq_length()) == PROMPT_LEN + 4
    _assert_trace_invariants(spec)


def test_eos_bonus_token_truncates_and_stops(monkeypatch):
    """EOS as the bonus (emitted but NOT cached): cache trails emitted by 1."""
    spec, target = _tiny_spec()
    full = _greedy_full(target, 16)

    # All drafts greedy; posterior row 2 forced to EOS => accept == 2 and
    # the bonus token itself IS the EOS.
    def draft_fn(step: int):
        return full[PROMPT_LEN + 1 : PROMPT_LEN + TINY_BLOCK_SIZE]

    _install_scripted_drafter(monkeypatch, spec, draft_fn)
    _force_target_argmax(monkeypatch, spec, verify={2: EOS_ID})

    out = spec.generate(PROMPT, max_new_tokens=40, stop_token_ids=[EOS_ID])

    assert len(spec.step_trace) == 1
    rec = spec.step_trace[0]
    assert rec.accept == 2

    new_ids = out[0, PROMPT_LEN:].tolist()
    assert new_ids == [full[PROMPT_LEN], full[PROMPT_LEN + 1], full[PROMPT_LEN + 2], EOS_ID]

    # The bonus is never forwarded, so the cache covers anchor + 2 accepted
    # drafts only: exactly one behind the emitted sequence.
    assert rec.start == PROMPT_LEN + 3
    assert rec.target_cache_len == PROMPT_LEN + 3
    assert int(spec.last_target_cache.get_seq_length()) == PROMPT_LEN + 3
    assert (PROMPT_LEN + len(new_ids)) - rec.target_cache_len == 1
    _assert_trace_invariants(spec)


def test_eos_prefill_anchor_stops_before_first_block(monkeypatch):
    """The prefill anchor itself is a stop id: emit it, run no block step."""
    spec, _ = _tiny_spec()

    _force_target_argmax(monkeypatch, spec, prefill={-1: EOS_ID}, verify={})
    out = spec.generate(PROMPT, max_new_tokens=16, stop_token_ids=[EOS_ID])

    assert out[0, PROMPT_LEN:].tolist() == [EOS_ID]
    assert tuple(out.shape) == (1, PROMPT_LEN + 1)
    assert spec.step_trace == []
    assert int(spec.last_target_cache.get_seq_length()) == PROMPT_LEN


# (e) max_new_tokens crossing a block boundary
def test_max_new_tokens_truncates_at_block_boundary(monkeypatch):
    spec, target = _tiny_spec()
    full = _greedy_full(target, 25)

    def draft_fn(step: int):
        anchor_idx = PROMPT_LEN + TINY_BLOCK_SIZE * step
        return full[anchor_idx + 1 : anchor_idx + TINY_BLOCK_SIZE]

    _install_scripted_drafter(monkeypatch, spec, draft_fn)

    max_new = TINY_BLOCK_SIZE + 5  # one full 10-token step + a 4-token step
    out = spec.generate(PROMPT, max_new_tokens=max_new)

    # Step 2 would have committed 10 but is truncated to the remaining
    # budget of 4; the loop stops (no third step).
    assert len(spec.step_trace) == 2
    first, last = spec.step_trace
    assert first.accept == TINY_BLOCK_SIZE - 1
    assert first.start == PROMPT_LEN + TINY_BLOCK_SIZE
    assert last.accept == 4  # effective accept: drafts actually committed
    assert last.start == PROMPT_LEN + max_new
    _assert_trace_invariants(spec)

    # Exactly max_new_tokens new tokens, token-identical to plain greedy;
    # the cache is cropped to the truncated start.
    assert tuple(out.shape) == (1, PROMPT_LEN + max_new)
    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=max_new)
    assert torch.equal(out, plain)
    assert int(spec.last_target_cache.get_seq_length()) == PROMPT_LEN + max_new


# (f) batch > 1 guard
def test_batch_greater_than_one_raises_not_implemented():
    spec, _ = _tiny_spec()
    batched = torch.cat([PROMPT, PROMPT], dim=0)
    assert batched.shape[0] == 2
    with pytest.raises(NotImplementedError, match="batch==1"):
        spec.generate(batched, max_new_tokens=4)
