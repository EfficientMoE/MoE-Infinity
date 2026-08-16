"""Unit tests for the additive BM4 NVTX phase markers (``_nvtx.nvtx_phase``).

These push/pop ranges are the producers ``benchmarks/dflash/parse_overlap.py``
unions to attribute expert-H2D bytes to hiding compute (``dflash_draft``,
``route_ahead_router``, ``target_verify``). The marker must be:

(a) a balanced NVTX ``range_push``/``range_pop`` pair emitting the EXACT name
    (push/pop, so nsys ``nvtx_pushpop_trace`` captures it -- start/end ranges
    on the default domain are NOT captured by nsys 2025.1.3);
(b) exception-safe -- the pop fires even when the wrapped body raises, so a
    failed forward can never leak an unbalanced range into the trace; and
(c) zero-overhead / no-op safe when NVTX/CUDA is unavailable (CPU test host):
    a push that raises must be swallowed, the body must still run, and exit
    must not pop an un-pushed range.

CPU-only: the real ``torch.cuda.nvtx`` hooks are monkeypatched, so no CUDA or
native extension is required.
"""

from __future__ import annotations

import pytest

from moe_infinity.spec_decode import _nvtx


@pytest.fixture
def recorder(monkeypatch):
    """Record ordered push/pop calls against monkeypatched nvtx hooks."""
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        _nvtx, "_range_push", lambda name: events.append(("push", name))
    )
    monkeypatch.setattr(
        _nvtx, "_range_pop", lambda: events.append(("pop", None))
    )
    return events


def test_balanced_push_pop_with_exact_name(recorder):
    with _nvtx.nvtx_phase("target_verify"):
        recorder.append(("body", None))
    assert recorder == [
        ("push", "target_verify"),
        ("body", None),
        ("pop", None),
    ]


def test_pop_fires_when_body_raises(recorder):
    with pytest.raises(ValueError):
        with _nvtx.nvtx_phase("dflash_draft"):
            recorder.append(("body", None))
            raise ValueError("verify blew up")
    assert recorder == [
        ("push", "dflash_draft"),
        ("body", None),
        ("pop", None),
    ]


def test_no_pop_when_push_raises(monkeypatch):
    """A push that raises (no CUDA) must not be followed by a stray pop."""
    events: list[str] = []

    def _boom(_name):
        raise RuntimeError("no NVTX here")

    monkeypatch.setattr(_nvtx, "_range_push", _boom)
    monkeypatch.setattr(_nvtx, "_range_pop", lambda: events.append("pop"))

    ran = False
    with _nvtx.nvtx_phase("route_ahead_router"):
        ran = True
    assert ran is True
    assert events == []


def test_hooks_none_is_noop(monkeypatch):
    """When the nvtx hooks are unavailable the marker is a transparent no-op."""
    monkeypatch.setattr(_nvtx, "_range_push", None)
    monkeypatch.setattr(_nvtx, "_range_pop", None)

    ran = False
    with _nvtx.nvtx_phase("dflash_draft"):
        ran = True
    assert ran is True


def test_default_compute_range_names_are_the_parser_contract():
    """The helper's phase names match ``parse_overlap.DEFAULT_COMPUTE_RANGES``."""
    from benchmarks.dflash.parse_overlap import DEFAULT_COMPUTE_RANGES

    assert set(_nvtx.BM4_COMPUTE_PHASES) == set(DEFAULT_COMPUTE_RANGES)
    assert _nvtx.BM4_COMPUTE_PHASES == (
        "dflash_draft",
        "route_ahead_router",
        "target_verify",
    )


def test_default_names_match_nsys_empty_domain_colon_prefix():
    """nsys renders push/pop ranges as ``:name``; the default names must match.

    ``torch.cuda.nvtx.range_push('dflash_draft')`` is captured as
    ``:dflash_draft`` (empty-domain prefix) in nsys 2025.1.3, so the bare
    ``DEFAULT_COMPUTE_RANGES`` would silently miss it without empty-domain
    normalisation -- the exact latent second bug that kept BM4 at 0.0.
    """
    from benchmarks.dflash.parse_overlap import (
        DEFAULT_COMPUTE_RANGES,
        Memcpy,
        NvtxRange,
        compute_overlap,
    )

    memcpys = [Memcpy(start=0.0, end=10.0, bytes=100)]
    ranges = [
        NvtxRange(name=":" + name, start=0.0, end=10.0)
        for name in DEFAULT_COMPUTE_RANGES
    ]
    result = compute_overlap(memcpys, ranges, DEFAULT_COMPUTE_RANGES)
    assert result.overlap_fraction == 1.0


def test_native_colon_compute_range_override_still_matches():
    """``--compute-range :expert_compute`` against captured ``:expert_compute``."""
    from benchmarks.dflash.parse_overlap import (
        Memcpy,
        NvtxRange,
        compute_overlap,
    )

    memcpys = [Memcpy(start=0.0, end=10.0, bytes=64)]
    ranges = [NvtxRange(name=":expert_compute", start=0.0, end=10.0)]
    assert (
        compute_overlap(
            memcpys, ranges, compute_ranges=[":expert_compute"]
        ).overlap_fraction
        == 1.0
    )
    assert (
        compute_overlap(
            memcpys, ranges, compute_ranges=["expert_compute"]
        ).overlap_fraction
        == 1.0
    )
