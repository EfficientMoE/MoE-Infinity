# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""CPU-only tests for the BM3 exposed-fetch instrumentation (plan Task 9).

Blocker B of PR #167: ``bench_prefetch_priority._exposed_fetch_seconds()``
returned ``0.0`` for every arm because nothing timed the synchronous on-demand
expert fetch, so ``bm3_decision.exposed_fetch_improved`` could never be true.

The fix adds a read-only accumulator on ``OffloadEngine``: the pre-forward
hook's ``fetch_tensors`` call is the exposed-fetch window (compute stalls there
whenever an expert was not already prefetched), so we time exactly that call
and surface the running total via ``get_exposed_fetch_seconds()`` with a
``reset_exposed_fetch_seconds()`` to zero it per ablation arm.

These tests are pure: they construct ``OffloadEngine`` via ``__new__`` with a
mocked ``archer_engine`` and never import the native extension, load a
checkpoint, or touch CUDA. They also exercise the harness-side
``_exposed_fetch_seconds`` probe, which the docstring guarantees is CPU-safe.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from benchmarks.dflash.bench_prefetch_priority import _exposed_fetch_seconds
from moe_infinity.runtime.model_offload import OffloadEngine


def _bare_engine() -> OffloadEngine:
    engine = OffloadEngine.__new__(OffloadEngine)
    engine.request_id = 7
    engine._exposed_fetch_seconds = 0.0
    engine.archer_engine = MagicMock()
    return engine


# ---------------------------------------------------------------------------
# OffloadEngine read-only exposed-fetch accumulator
# ---------------------------------------------------------------------------


def test_offload_engine_exposes_exposed_fetch_surface():
    engine = _bare_engine()
    assert callable(getattr(engine, "get_exposed_fetch_seconds", None))
    assert callable(getattr(engine, "reset_exposed_fetch_seconds", None))
    assert callable(getattr(engine, "_fetch_tensors_timed", None))
    assert engine.get_exposed_fetch_seconds() == 0.0


def test_timed_fetch_calls_native_fetch_unchanged():
    engine = _bare_engine()
    tensors = [11, 22, 33]
    engine._fetch_tensors_timed(tensors)
    engine.archer_engine.fetch_tensors.assert_called_once_with(7, tensors)


def test_timed_fetch_accumulates_wall_time():
    engine = _bare_engine()

    def _slow_fetch(_request_id, _tensors):
        time.sleep(0.02)

    engine.archer_engine.fetch_tensors.side_effect = _slow_fetch
    engine._fetch_tensors_timed([1])
    first = engine.get_exposed_fetch_seconds()
    assert first >= 0.02
    engine._fetch_tensors_timed([2])
    assert engine.get_exposed_fetch_seconds() >= first + 0.02 - 1e-6


def test_reset_zeros_the_accumulator():
    engine = _bare_engine()
    engine.archer_engine.fetch_tensors.side_effect = (
        lambda *_a, **_k: time.sleep(0.005)
    )
    engine._fetch_tensors_timed([1])
    assert engine.get_exposed_fetch_seconds() > 0.0
    engine.reset_exposed_fetch_seconds()
    assert engine.get_exposed_fetch_seconds() == 0.0


def test_get_exposed_fetch_seconds_defaults_to_zero_without_field():
    engine = OffloadEngine.__new__(OffloadEngine)
    assert engine.get_exposed_fetch_seconds() == 0.0


# ---------------------------------------------------------------------------
# harness probe: _exposed_fetch_seconds reads the real engine getter
# ---------------------------------------------------------------------------


class _EngineWithGetter:
    def __init__(self, value):
        self._value = value

    def get_exposed_fetch_seconds(self):
        return self._value


class _EngineWithPrefetcher:
    def __init__(self, value):
        self.expert_prefetcher = _EngineWithGetter(value)


def test_probe_reads_engine_get_exposed_fetch_seconds():
    assert _exposed_fetch_seconds(_EngineWithGetter(1.5), None) == 1.5


def test_probe_reads_prefetcher_getter_when_engine_bare():
    assert _exposed_fetch_seconds(_EngineWithPrefetcher(0.75), None) == 0.75


def test_probe_returns_zero_when_nothing_instrumented():
    class _Bare:
        pass

    assert _exposed_fetch_seconds(_Bare(), None) == 0.0
