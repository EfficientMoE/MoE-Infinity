"""Task 6: native DFlash draft->verify->rollback state machine.

Pins the two Task-6 QA scenarios on the tiny CPU fixtures (T5):

(a) A single native step commits ``accept + 1`` tokens and crops the target
    cache to ``start`` — the bonus token is emitted but NOT cached, so the
    cache length trails the emitted count by exactly one (the bonus-token
    trap: ``cache length == emitted length`` would mean the bonus was cached).
(b) The verify forward runs with FULL logits — no ``logits_to_keep=1``
    slicing — spied at both the speculator's rich-forward seam and the target
    model's own forward (prefill/anchor uses 1, verify uses none).

It also pins multi-step greedy parity against ``plain_greedy_decode`` (the
losslessness smoke signal for the state machine; the strict E2E gate is
Task 9) and routing through the MoE engine's ``_native_model_forward_rich``
seam (the production path Task 8 wires up).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    TINY_BLOCK_SIZE,
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
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _tiny_spec(device: str = "cpu"):
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    if device != "cpu":
        target = target.to(device)
        drafter = drafter.to(device)
    spec = DFlashSpeculator.from_models(target, drafter, config=config, device=device)
    return spec, target, drafter


def _assert_step_invariants(spec: DFlashSpeculator) -> None:
    for rec in spec.step_trace:
        # The cache advance is accept+1 (anchor + accepted drafts); the bonus
        # token is emitted on top and is NOT part of the cache advance.
        assert rec.start == rec.prev_start + rec.accept + 1
        assert rec.target_cache_len == rec.start


def test_single_native_step_commits_and_crops_caches():
    """QA scenario (a): one native step; emitted vs cached accounting."""
    spec, _, _ = _tiny_spec()

    # max_new_tokens=2: the prefill anchor is token 1, so exactly one
    # draft->verify->rollback step runs before the loop exits.
    out = spec.generate(PROMPT, max_new_tokens=2)

    assert len(spec.step_trace) == 1
    rec = spec.step_trace[0]

    assert rec.prev_start == PROMPT_LEN
    assert 0 <= rec.accept <= TINY_BLOCK_SIZE - 1
    # start advances by accept+1: the bonus token is NOT added to the cache.
    assert rec.start == PROMPT_LEN + rec.accept + 1
    # The target cache is cropped to start.
    assert rec.target_cache_len == rec.start
    # emitted grew by accept+1 over the prefill anchor (accepted drafts ++ bonus).
    assert rec.emitted_len == 1 + rec.accept + 1
    # Bonus-token trap guard: the absolute emitted length (prompt + emitted)
    # must be exactly one ahead of the cache length — never equal.
    assert (PROMPT_LEN + rec.emitted_len) - rec.target_cache_len == 1

    assert spec.last_target_cache is not None
    assert int(spec.last_target_cache.get_seq_length()) == rec.start
    # The tiny drafter is stateless: no draft KV cache exists on this path.
    assert spec.last_draft_cache is None

    # The public contract returns prompt ++ at most max_new_tokens new ids.
    assert tuple(out.shape) == (1, PROMPT_LEN + 2)

    print(
        "step-state prev_start={} accept={} start={} emitted_len={} "
        "target_cache_len={} draft_cache_len={}".format(
            rec.prev_start,
            rec.accept,
            rec.start,
            rec.emitted_len,
            rec.target_cache_len,
            rec.draft_cache_len,
        )
    )


def test_verify_forward_uses_full_logits():
    """QA scenario (b): prefill slices logits, verify must NOT."""
    spec, target, _ = _tiny_spec()

    seam_calls = []
    orig_forward_target = spec._forward_target

    def seam_spy(input_ids, past_key_values=None, logits_to_keep=0, **fwd_kwargs):
        seam_calls.append(int(logits_to_keep))
        return orig_forward_target(
            input_ids,
            past_key_values=past_key_values,
            logits_to_keep=logits_to_keep,
            **fwd_kwargs,
        )

    spec._forward_target = seam_spy

    model_calls = []
    orig_model_forward = target.forward

    def model_spy(*args, **kwargs):
        model_calls.append((args, dict(kwargs)))
        return orig_model_forward(*args, **kwargs)

    target.forward = model_spy

    spec.generate(PROMPT, max_new_tokens=2)

    # One prefill (anchor, sliced) + one verify (full) for a single step.
    assert seam_calls == [1, 0]

    assert len(model_calls) == 2
    _, prefill_kwargs = model_calls[0]
    assert prefill_kwargs.get("past_key_values") is None
    assert prefill_kwargs.get("logits_to_keep") == 1

    verify_args, verify_kwargs = model_calls[1]
    assert verify_kwargs.get("past_key_values") is not None
    # Full-logits verify: the kwarg is absent (helper/seam only pass it when > 0).
    assert "logits_to_keep" not in verify_kwargs
    block_ids = verify_args[0] if verify_args else verify_kwargs["input_ids"]
    assert tuple(block_ids.shape) == (1, TINY_BLOCK_SIZE)

    print(f"verify-fulllogits seam_calls={seam_calls}")


def test_native_multistep_greedy_matches_plain_greedy():
    """Core loop works: native multi-step greedy == plain greedy (CPU, fp32)."""
    spec, target, _ = _tiny_spec()

    max_new_tokens = 32
    native = spec.generate(PROMPT, max_new_tokens=max_new_tokens)
    plain = plain_greedy_decode(target, PROMPT, max_new_tokens=max_new_tokens)

    assert tuple(native.shape) == (1, PROMPT_LEN + max_new_tokens)
    assert torch.equal(native, plain), (
        "native DFlash greedy diverged from plain greedy:\n"
        f"  native={native[0].tolist()}\n  plain={plain[0].tolist()}"
    )

    # Multi-step actually ran, with consistent emitted-vs-cached accounting.
    assert len(spec.step_trace) >= 2
    _assert_step_invariants(spec)
    assert int(spec.last_target_cache.get_seq_length()) == spec.step_trace[-1].start

    accepts = [rec.accept for rec in spec.step_trace]
    print(f"multistep steps={len(accepts)} accepts={accepts}")


def test_native_step_routes_through_moe_rich_forward():
    """Production seam: with an MoE shell present, every target forward goes
    through ``_native_model_forward_rich`` (standard expert dispatch)."""
    from moe_infinity.entrypoints.big_modeling import MoE

    spec, target, _ = _tiny_spec(device=DEVICE)

    shell = MoE.__new__(MoE)
    shell.model = target
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    # The bare shell has no offload runtime; the speculator skips a None hook.
    shell._configure_hook = None

    spec = DFlashSpeculator.from_models(
        shell, spec.draft, config=spec.config, device=DEVICE
    )

    rich_calls = []
    orig_rich = shell._native_model_forward_rich

    def rich_spy(token_ids, attention_metadata=None, logits_to_keep=0):
        rich_calls.append(
            {
                "n_tokens": len(token_ids),
                "logits_to_keep": int(logits_to_keep),
                "is_prefill": (
                    True
                    if attention_metadata is None
                    else bool(getattr(attention_metadata, "is_prefill", True))
                ),
            }
        )
        return orig_rich(token_ids, attention_metadata, logits_to_keep=logits_to_keep)

    shell._native_model_forward_rich = rich_spy

    max_new_tokens = 16
    out = spec.generate(PROMPT.to(DEVICE), max_new_tokens=max_new_tokens)

    # Every target forward went through the rich helper: prefill (sliced
    # logits) then one full-logits verify per step.
    assert len(rich_calls) == 1 + len(spec.step_trace)
    assert rich_calls[0]["logits_to_keep"] == 1
    assert rich_calls[0]["is_prefill"] is True
    for call in rich_calls[1:]:
        assert call["logits_to_keep"] == 0
        assert call["is_prefill"] is False
        assert call["n_tokens"] == TINY_BLOCK_SIZE

    _assert_step_invariants(spec)
    # The engine-side cache (same object the loop cropped) ends at start.
    assert int(shell._cached_past_key_values.get_seq_length()) == spec.step_trace[-1].start

    # Parity against sequential greedy through the SAME rich helper.
    shell._cached_past_key_values = None
    reference = []
    step_ids = PROMPT[0].tolist()
    for i in range(max_new_tokens):
        meta = None if i == 0 else SimpleNamespace(is_prefill=False)
        logits, _, _ = orig_rich(step_ids, meta, logits_to_keep=1)
        nxt = int(logits[0, -1].argmax().item())
        reference.append(nxt)
        step_ids = [nxt]

    assert out[0, PROMPT_LEN:].tolist() == reference

    print(
        f"moe-rich forwards={len(rich_calls)} steps={len(spec.step_trace)} "
        f"final_cache={int(shell._cached_past_key_values.get_seq_length())}"
    )
