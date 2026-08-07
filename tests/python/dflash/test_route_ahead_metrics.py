"""Unit tests for the Track A5 route-ahead coverage/waste metrics.

Autonomous correctness gate for Track A5 of
``.sisyphus/plans/dflash-deferred-tracks-plan.md`` (A0 section 4 formulas).
Covers, all CPU-only with mocked dispatcher/archer engine:

(a) coverage accounting 0 / partial / 1, driven through the real executor
    route-ahead seam, cross-checked against the A1 ``prefetch_coverage``;
(b) rejected-token waste accounting vs. the kept prefix (A1
    ``rejected_expert_ids`` semantics), including clamped and vacuous edges;
(c) default-off / zero-overhead behavior: no stats handle anywhere means no
    recording and byte-identical legacy dispatch behavior;
(d) aborted-step isolation (``begin_step`` drops uncommitted records);
(e) E2E: an offloaded executor-backed MoE shell (DeepSeek/Qwen pattern --
    rich forward whose MoE blocks call ``dispatch_local``) runs the full
    DFlash generate loop with route-ahead firing on every verify step,
    hitting no resident-only block (A4), with token-identical outputs and
    identical prefetch calls whether metrics are on or off.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
)

from moe_infinity.distributed.expert_executor import (  # noqa: E402
    DistributedExpertExecutor,
)
from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher  # noqa: E402
from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.spec_decode._prefetch_route import (  # noqa: E402
    prefetch_coverage,
    rejected_expert_ids,
    union_experts_from_mask,
)
from moe_infinity.spec_decode._route_ahead_ctx import (  # noqa: E402
    current_stats,
    route_ahead_context,
)
from moe_infinity.spec_decode._route_ahead_stats import (  # noqa: E402
    RouteAheadStats,
    RouteAheadStepSummary,
)
from moe_infinity.utils import ArcherConfig  # noqa: E402

LAYER_ID = 3
# [3 tokens, 8 experts]; the union routed by ANY token is {0, 1, 2, 5, 7};
# the anchor-only kept prefix routes {0, 1}.
ROUTER_MASK = torch.tensor(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=torch.bool,
)
UNION = [0, 1, 2, 5, 7]
HIDDEN = torch.zeros(3, 4)
WEIGHTS = torch.zeros(3, 8)
LOGITS = torch.zeros(3, 8)

PROMPT = torch.tensor([[3, 7, 11, 2, 5]])


@pytest.fixture(autouse=True)
def _cpu_dispatch_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "moe_infinity.distributed.expert_executor.IOProfiler", None
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def _make_executor(*, overlap: bool = False, prefetcher=None):
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp/moe-infinity-route-ahead-metrics-test",
            "trace_capacity": 16,
            "prefetch": False,
            "speculative_prefetch": True,
            "speculative_prefetch_overlap": overlap,
        }
    )
    executor = DistributedExpertExecutor(config)
    executor.set_expert_dispatcher(MagicMock(name="ExpertDispatcher"))
    if prefetcher is not None:
        executor.set_prefetcher(prefetcher)
    return executor


def _make_real_prefetcher(num_layers: int = 8, num_experts: int = 8):
    """A2-style real ExpertPrefetcher; tensor id = layer * 100 + expert."""
    prefetcher = ExpertPrefetcher.__new__(ExpertPrefetcher)
    prefetcher.num_layers = num_layers
    prefetcher.num_experts = num_experts
    engine = MagicMock(name="ArcherEngine")
    engine.get_node_default_device.return_value = 0
    prefetcher.archer_engine = engine
    prefetcher.expert_tensor_map = {
        (layer, expert): layer * 100 + expert
        for layer in range(num_layers)
        for expert in range(num_experts)
    }
    prefetcher._last_speculative_prediction = set()
    return prefetcher, engine


def _dispatch(executor, *, router_logits=None, prefetcher=None):
    executor.dispatch_local(
        LAYER_ID,
        HIDDEN,
        ROUTER_MASK,
        WEIGHTS,
        router_logits=router_logits,
        prefetcher=prefetcher,
    )


# ---------------------------------------------------------------------------
# (a) coverage accounting: 0 / partial / 1 through the executor seam
# ---------------------------------------------------------------------------


def test_full_coverage_when_route_ahead_fires():
    stats = RouteAheadStats()
    prefetcher = MagicMock(name="ExpertPrefetcher")
    stats.begin_step()
    with route_ahead_context(prefetcher=prefetcher, stats=stats):
        _dispatch(_make_executor())
    summary = stats.commit_step(kept_rows=3)

    prefetcher.fetch_experts_lock_cache.assert_called_once_with(LAYER_ID, UNION)
    assert summary == RouteAheadStepSummary(
        layers=1, predicted=5, actual=5, covered=5, kept=5, wasted=0
    )
    assert summary.coverage == prefetch_coverage(UNION, UNION) == 1.0
    assert stats.steps == 1 and stats.layers_observed == 1
    assert stats.coverage == 1.0
    assert stats.waste_ratio == 0.0


def test_zero_coverage_when_route_ahead_never_fires():
    stats = RouteAheadStats()
    executor = _make_executor()  # no prefetcher anywhere -> nothing prefetched
    stats.begin_step()
    with route_ahead_context(stats=stats):
        _dispatch(executor)
    summary = stats.commit_step(kept_rows=2)

    assert summary.predicted == 0 and summary.actual == len(UNION)
    assert summary.covered == 0
    assert prefetch_coverage([], UNION) == 0.0
    assert stats.coverage == 0.0
    # Nothing was prefetched, so nothing can be prefetch-wasted.
    assert stats.wasted_experts == 0 and stats.waste_ratio == 0.0
    # The dispatch itself is untouched: legacy pending tuple preserved.
    assert executor._pending_prefetch is not None
    assert executor._pending_prefetch[1] == LAYER_ID


def test_partial_coverage_when_only_some_layers_prefetch():
    stats = RouteAheadStats()
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor()  # no context/executor prefetcher
    stats.begin_step()
    with route_ahead_context(stats=stats):
        # Layer 3 fires via the per-call prefetcher argument...
        _dispatch(executor, prefetcher=prefetcher)
        # ...layer 4 has no prefetcher at all -> predicted set is empty.
        executor.dispatch_local(LAYER_ID + 1, HIDDEN, ROUTER_MASK, WEIGHTS)
    summary = stats.commit_step(kept_rows=3)

    assert summary.layers == 2
    assert summary.covered == 5 and summary.actual == 10
    # A0 section 4 ratio-of-sums, cross-checked per layer with the A1 helper.
    per_layer = [
        prefetch_coverage(UNION, UNION),
        prefetch_coverage([], UNION),
    ]
    assert per_layer == [1.0, 0.0]
    assert stats.coverage == (5 + 0) / (5 + 5) == 0.5


# ---------------------------------------------------------------------------
# (b) rejected-token waste accounting vs. the kept prefix
# ---------------------------------------------------------------------------


def test_waste_scales_with_rejected_rows():
    stats = RouteAheadStats()
    prefetcher = MagicMock(name="ExpertPrefetcher")

    # Full accept: kept == full union -> zero waste.
    stats.begin_step()
    with route_ahead_context(prefetcher=prefetcher, stats=stats):
        _dispatch(_make_executor())
    full = stats.commit_step(kept_rows=3)
    assert full.wasted == 0 and stats.coverage == 1.0

    # Anchor-only keep: rows 1..2 rejected -> experts {2, 5, 7} wasted.
    stats.begin_step()
    with route_ahead_context(prefetcher=prefetcher, stats=stats):
        _dispatch(_make_executor())
    part = stats.commit_step(kept_rows=1)
    assert rejected_expert_ids(UNION, [0, 1]) == [2, 5, 7]
    assert part.kept == 2 and part.wasted == 3
    assert stats.wasted_experts == 3
    assert stats.waste_ratio == 3 / 10
    assert stats.coverage == 1.0  # waste does not dent coverage


def test_commit_clamps_kept_rows():
    stats = RouteAheadStats()
    stats.begin_step()
    stats.observe_layer(0, UNION, ROUTER_MASK)
    over = stats.commit_step(kept_rows=99)
    assert over.kept == len(UNION) and over.wasted == 0

    stats.begin_step()
    stats.observe_layer(0, UNION, ROUTER_MASK)
    zero = stats.commit_step(kept_rows=0)
    assert zero.kept == 0 and zero.wasted == len(UNION)


def test_empty_union_dispatch_is_vacuous_noop():
    stats = RouteAheadStats()
    prefetcher = MagicMock(name="ExpertPrefetcher")
    stats.begin_step()
    with route_ahead_context(prefetcher=prefetcher, stats=stats):
        _make_executor().dispatch_local(
            LAYER_ID,
            HIDDEN[:1],
            torch.zeros(1, 8, dtype=torch.bool),
            WEIGHTS[:1],
        )
    # Empty union: the pin/prefetch no-op is preserved under metrics.
    prefetcher.fetch_experts_lock_cache.assert_not_called()
    prefetcher.speculative_prefetch.assert_not_called()
    summary = stats.commit_step(kept_rows=1)
    assert summary.layers == 1
    assert summary.predicted == summary.actual == summary.wasted == 0
    assert stats.coverage == 1.0  # nothing to cover, nothing wasted


# ---------------------------------------------------------------------------
# (c) default-off / zero-overhead: no handle, no recording, legacy behavior
# ---------------------------------------------------------------------------


def test_inactive_context_records_and_changes_nothing():
    stats = RouteAheadStats()
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor(overlap=True, prefetcher=prefetcher)

    _dispatch(executor, router_logits=LOGITS)  # no route_ahead_context

    # Legacy overlap path byte-identical (A3 gate): pooled prefetch fired...
    prefetcher.fetch_experts_lock_cache.assert_not_called()
    assert prefetcher.speculative_prefetch.call_count == 1
    args, kwargs = prefetcher.speculative_prefetch.call_args
    assert args[0] == LAYER_ID and args[1] is LOGITS and not kwargs
    # ...and the stats handle was never consulted.
    assert stats.as_dict() == RouteAheadStats().as_dict()
    assert stats.commit_step(kept_rows=3).layers == 0
    assert stats.steps == 0


def test_context_without_stats_handle_records_nothing():
    with route_ahead_context(prefetcher=MagicMock()):
        assert current_stats() is None
    assert current_stats() is None


def test_speculator_metrics_default_off():
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    spec = DFlashSpeculator.from_models(target, drafter, config=config, device="cpu")

    assert spec.route_ahead_stats is None
    spec.generate(PROMPT, max_new_tokens=4, stop_token_ids=[])
    assert spec.route_ahead_stats is None  # never implicitly created
    assert len(spec.step_trace) >= 1


def test_enable_route_ahead_stats_returns_reset_recorder():
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    spec = DFlashSpeculator.from_models(target, drafter, config=config, device="cpu")

    stats = spec.enable_route_ahead_stats()
    assert spec.route_ahead_stats is stats
    assert stats.as_dict() == RouteAheadStats().as_dict()
    stats.steps = 99
    assert spec.enable_route_ahead_stats() is stats
    assert stats.steps == 0


# ---------------------------------------------------------------------------
# (d) aborted-step isolation
# ---------------------------------------------------------------------------


def test_begin_step_drops_uncommitted_records():
    stats = RouteAheadStats()
    stats.begin_step()
    stats.observe_layer(0, UNION, ROUTER_MASK)
    stats.begin_step()  # prior step aborted before commit: discarded
    assert stats.commit_step(kept_rows=1).layers == 0
    assert stats.as_dict() == RouteAheadStats().as_dict()


# ---------------------------------------------------------------------------
# (e) E2E: offloaded executor-backed shell through the full generate loop
# ---------------------------------------------------------------------------


def _e2e_masks() -> dict[int, torch.Tensor]:
    """Two synthetic MoE layers, 10 block rows x 8 experts.

    Layer 0 routes rows {i%8, (i+3)%8} (full union = all 8 experts); layer 2
    routes {(2i)%8} (full union = {0, 2, 4, 6}).
    """
    rows_a = [
        [1 if j in (i % 8, (i + 3) % 8) else 0 for j in range(8)]
        for i in range(10)
    ]
    rows_b = [[1 if j == (2 * i) % 8 else 0 for j in range(8)] for i in range(10)]
    return {
        0: torch.tensor(rows_a, dtype=torch.bool),
        2: torch.tensor(rows_b, dtype=torch.bool),
    }


class _OffloadedExecutorShell:
    """MoE-engine shell for the offloaded executor-backed path (A4).

    Mirrors ``big_modeling._native_model_forward_rich`` plus the DeepSeek/
    Qwen/Mixtral MoE-block pattern: on verify forwards each block routes its
    tokens and calls ``DistributedExpertExecutor.dispatch_local`` with its
    router mask (with fixed synthetic masks so the A5 accounting is exactly
    checkable). ``engine.expert_prefetcher`` is a real ExpertPrefetcher over
    a mocked archer engine -- i.e. the offloaded configuration, the exact
    setup a resident-only guard would have blocked.
    """

    def __init__(self, target, executor, prefetcher, layer_masks):
        self.model = target
        self.engine = SimpleNamespace(expert_prefetcher=prefetcher)
        self._executor = executor
        self._layer_masks = layer_masks
        self._cached_past_key_values = None

    def _native_model_forward_rich(
        self, token_ids, _attention_metadata=None, logits_to_keep=0
    ):
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        is_prefill = _attention_metadata is None
        kwargs: dict[str, Any] = {"use_cache": True, "output_hidden_states": True}
        if logits_to_keep:
            kwargs["logits_to_keep"] = int(logits_to_keep)
        if not is_prefill:
            kwargs["past_key_values"] = self._cached_past_key_values
        outputs = self.model(input_tensor, **kwargs)
        self._cached_past_key_values = outputs.past_key_values
        if not is_prefill:
            num_tokens = len(token_ids)
            hidden = torch.zeros(num_tokens, 4)
            for layer_id, mask in self._layer_masks.items():
                self._executor.dispatch_local(
                    layer_id, hidden, mask, mask.to(torch.float32)
                )
                self._executor.wait_dispatch_local()
        return outputs.logits, outputs.hidden_states, outputs.past_key_values


def _offloaded_spec(*, with_stats: bool):
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    prefetcher, engine = _make_real_prefetcher()
    executor = _make_executor(prefetcher=prefetcher)
    shell = _OffloadedExecutorShell(target, executor, prefetcher, _e2e_masks())
    spec = DFlashSpeculator.from_models(shell, drafter, config=config, device="cpu")
    if with_stats:
        spec.enable_route_ahead_stats()
    return spec, engine


def test_generate_offloaded_executor_route_ahead_e2e():
    spec_on, engine_on = _offloaded_spec(with_stats=True)
    spec_off, engine_off = _offloaded_spec(with_stats=False)

    out_on = spec_on.generate(PROMPT, max_new_tokens=25, stop_token_ids=[])
    out_off = spec_off.generate(PROMPT, max_new_tokens=25, stop_token_ids=[])

    # Metrics + prefetch are read-only observers: token-identical outputs...
    assert torch.equal(out_on, out_off)
    # ...and byte-identical prefetch behavior with metrics on vs. off.
    assert (
        engine_on.replace_cache_candidates.call_args_list
        == engine_off.replace_cache_candidates.call_args_list
    )
    assert (
        engine_on.enqueue_prefetch.call_args_list
        == engine_off.enqueue_prefetch.call_args_list
    )

    stats = spec_on.route_ahead_stats
    assert stats is not None and spec_off.route_ahead_stats is None
    steps = len(spec_on.step_trace)
    assert stats.steps == steps and steps >= 2
    assert stats.layers_observed == steps * 2

    # Route-ahead fired on every layer of every step: perfect coverage.
    masks = _e2e_masks()
    per_step_actual = sum(len(union_experts_from_mask(m)) for m in masks.values())
    assert per_step_actual == 12
    assert stats.actual_experts == steps * per_step_actual
    assert stats.predicted_experts == stats.actual_experts
    assert stats.covered_experts == stats.actual_experts
    assert stats.coverage == 1.0

    # Rejected-token waste, recomputed independently from the kept prefixes
    # the accept rule committed (step_trace.accept + 1 block rows).
    expected_waste = 0
    for rec in spec_on.step_trace:
        kept_rows = rec.accept + 1
        for mask in masks.values():
            full_union = union_experts_from_mask(mask)
            kept_union = union_experts_from_mask(mask[:kept_rows])
            expected_waste += len(rejected_expert_ids(full_union, kept_union))
    assert stats.wasted_experts == expected_waste
    assert 0.0 <= stats.waste_ratio <= 1.0

    # A4 guard: every pin targeted exactly ONE layer, alternating between
    # the shell's two MoE layers -- never a cross-layer batched pin.
    pin_calls = engine_on.replace_cache_candidates.call_args_list
    assert len(pin_calls) == steps * 2
    pinned_layers = []
    for call in pin_calls:
        layers = {tensor_id // 100 for tensor_id in call.args[0]}
        assert len(layers) == 1
        pinned_layers.append(next(iter(layers)))
    assert pinned_layers == [0, 2] * steps


def test_generate_offloaded_metrics_match_step_trace_accepts():
    spec, _engine = _offloaded_spec(with_stats=True)
    spec.generate(PROMPT, max_new_tokens=12, stop_token_ids=[])

    stats = spec.route_ahead_stats
    assert stats is not None
    # Every committed verify step was observed exactly once per MoE layer.
    assert stats.steps == len(spec.step_trace)
    assert stats.layers_observed == 2 * stats.steps
    # kept_experts aggregates the kept-prefix union sizes of both layers.
    masks = _e2e_masks()
    expected_kept = 0
    for rec in spec.step_trace:
        for mask in masks.values():
            expected_kept += len(union_experts_from_mask(mask[: rec.accept + 1]))
    assert stats.kept_experts == expected_kept
