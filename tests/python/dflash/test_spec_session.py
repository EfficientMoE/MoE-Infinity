"""CPU tests for the PD-DFlash Task 6 Step 5 single-round ``SpecSession`` seam.

Pins three properties of ``begin_session``/``draft_round``/``verify_round`` on
the tiny CPU fixtures (T5), plus the read-only route projection:

* identity: the engine-driven session loop reproduces ``generate()``'s output
  byte-for-byte (greedy AND lossless-sampled), proving the seam externalizes
  ``_generate_single`` without changing its semantics;
* round APIs: ``draft_round`` returns a byte-free warm-up demand, ``verify_round``
  reports accepted tokens / finished, and the DRAFT<->VERIFY hand-off is
  order-enforced;
* projection: ``RouteUnionCollector`` + ``project_expert_bytes`` sum the EXACT
  ``expert_nbytes`` over the routed union, never an expert count.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
)

from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.spec_decode.dflash import (  # noqa: E402
    DraftResult,
    RouteUnionCollector,
    VerifyResult,
    project_expert_bytes,
)

PROMPT = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)
PROMPT_LEN = PROMPT.shape[1]


def _tiny_spec(seed: int = 0) -> DFlashSpeculator:
    target = build_tiny_target(seed=seed)
    drafter = build_tiny_drafter(target, seed=seed + 1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    return DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )


def _run_session(
    spec: DFlashSpeculator, *, max_new_tokens: int, temperature: float = 0.0
) -> list[int]:
    session = spec.begin_session(
        PROMPT.clone(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    guard = 0
    while not session.finished:
        draft = spec.draft_round(session)
        assert isinstance(draft, DraftResult)
        assert draft.tokens == spec.config.block_size
        result = spec.verify_round(session)
        assert isinstance(result, VerifyResult)
        guard += 1
        assert guard <= max_new_tokens + 2, "session failed to terminate"
    return session.output_ids


def test_session_loop_matches_generate_greedy() -> None:
    max_new_tokens = 16

    spec = _tiny_spec()
    baseline = spec.generate(
        PROMPT.clone(), max_new_tokens=max_new_tokens, temperature=0.0
    )
    baseline_ids = baseline[0, PROMPT_LEN:].tolist()

    session_ids = _run_session(_tiny_spec(), max_new_tokens=max_new_tokens)

    assert session_ids == baseline_ids
    assert 0 < len(session_ids) <= max_new_tokens


def test_session_loop_matches_generate_sampled() -> None:
    max_new_tokens = 12

    torch.manual_seed(1234)
    baseline_ids = (
        _tiny_spec()
        .generate(
            PROMPT.clone(), max_new_tokens=max_new_tokens, temperature=0.8
        )[0, PROMPT_LEN:]
        .tolist()
    )

    torch.manual_seed(1234)
    session_ids = _run_session(
        _tiny_spec(), max_new_tokens=max_new_tokens, temperature=0.8
    )

    assert session_ids == baseline_ids


def test_draft_round_is_byte_free_warmup_without_offload() -> None:
    spec = _tiny_spec()
    session = spec.begin_session(PROMPT.clone(), max_new_tokens=8)

    draft = spec.draft_round(session)

    assert draft.tokens == spec.config.block_size
    assert draft.expert_union == frozenset()
    assert draft.expert_bytes == 0


def test_draft_verify_handoff_is_order_enforced() -> None:
    spec = _tiny_spec()
    session = spec.begin_session(PROMPT.clone(), max_new_tokens=8)

    with pytest.raises(RuntimeError, match="without a pending draft"):
        spec.verify_round(session)

    spec.draft_round(session)
    with pytest.raises(RuntimeError, match="un-verified pending block"):
        spec.draft_round(session)

    result = spec.verify_round(session)
    assert isinstance(result, VerifyResult)
    assert result.committed_count == len(result.accepted_token_ids)


def test_finished_session_rejects_further_drafts() -> None:
    spec = _tiny_spec()
    _run_session(spec, max_new_tokens=6)
    session = spec.begin_session(PROMPT.clone(), max_new_tokens=6)
    while not session.finished:
        spec.draft_round(session)
        spec.verify_round(session)
    with pytest.raises(RuntimeError, match="finished session"):
        spec.draft_round(session)


def test_immediate_stop_anchor_finishes_without_a_round() -> None:
    spec = _tiny_spec()
    baseline_session = spec.begin_session(PROMPT.clone(), max_new_tokens=8)
    anchor = baseline_session.anchor

    stopped = spec.begin_session(
        PROMPT.clone(), max_new_tokens=8, stop_token_ids=[anchor]
    )
    assert stopped.finished is True
    assert stopped.output_ids == [anchor]


def test_route_union_collector_captures_routed_union_read_only() -> None:
    collector = RouteUnionCollector()
    collector.begin_step()
    # layer 0 routes tokens to experts {1, 3}; layer 4 routes to {2}.
    layer0_mask = torch.tensor([[0, 1, 0, 1], [0, 0, 0, 1]], dtype=torch.bool)
    layer4_mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    collector.observe_layer(0, predicted_ids=[1], router_mask=layer0_mask)
    collector.observe_layer(4, predicted_ids=[], router_mask=layer4_mask)
    collector.commit_step(kept_rows=1)

    assert collector.union == {(0, 1), (0, 3), (4, 2)}
    # the masks are untouched by observation
    assert layer0_mask.dtype == torch.bool


def test_project_expert_bytes_sums_exact_payload_not_count() -> None:
    nbytes_map = {(0, 1): 100, (0, 3): 250, (4, 2): 400, (9, 9): 999}
    union = frozenset({(0, 1), (0, 3), (4, 2)})

    total = project_expert_bytes(union, nbytes_map)

    # exact Sigma of the real per-expert payloads, not len(union) * average
    assert total == 100 + 250 + 400
    assert total != len(union)


def test_project_expert_bytes_is_zero_without_a_map() -> None:
    union = frozenset({(0, 1), (0, 3)})
    assert project_expert_bytes(union, None) == 0
    assert project_expert_bytes(union, {}) == 0
    assert project_expert_bytes(frozenset(), {(0, 1): 100}) == 0


def test_project_expert_bytes_ignores_pairs_absent_from_map() -> None:
    nbytes_map = {(0, 1): 100}
    union = frozenset({(0, 1), (7, 7)})
    assert project_expert_bytes(union, nbytes_map) == 100
