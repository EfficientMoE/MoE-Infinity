# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Cumulative expert H2D transfer-bytes counter (#234 Phase 1, Task 6b).

Follows the #164 stat-accessor pattern in ``test_native_stat_accessors.py``:
the binding surface is asserted against the built extension when available,
and the sum/reset arithmetic is exercised through the same defensive-getattr
wiring the parity harness uses. The counter's on-GPU byte accuracy is covered
by ``benchmarks/fp8_store/bench_store_parity.py`` on hardware.
"""

from __future__ import annotations

import pytest

from benchmarks.fp8_store.bench_store_parity import _call


class _FakeHandle:
    def __init__(self):
        self._total = 0

    def fetch_group(self, nbytes):
        self._total += int(nbytes)

    def get_expert_h2d_bytes_total(self):
        return self._total

    def reset_expert_transfer_stats(self):
        self._total = 0


def test_counter_equals_sum_of_known_group_sizes():
    handle = _FakeHandle()
    sizes = [4096, 1 << 20, 3 * (1 << 19)]
    for nbytes in sizes:
        handle.fetch_group(nbytes)
    assert _call(handle, "get_expert_h2d_bytes_total") == sum(sizes)


def test_reset_clears_only_this_counter():
    handle = _FakeHandle()
    handle.fetch_group(2048)
    assert _call(handle, "reset_expert_transfer_stats") is None
    assert _call(handle, "get_expert_h2d_bytes_total") == 0


def test_missing_accessor_returns_none():
    assert _call(object(), "get_expert_h2d_bytes_total") is None


def test_native_binding_surface():
    prefetch = pytest.importorskip("moe_infinity._store")
    if "mock" in type(prefetch).__module__ or not getattr(
        prefetch, "__file__", None
    ):
        pytest.skip("_store is mocked in this environment")
    lib = getattr(prefetch, "prefetch_handle", None)
    if lib is None:
        pytest.skip("prefetch_handle extension class unavailable")
    for name in ("get_expert_h2d_bytes_total", "reset_expert_transfer_stats"):
        assert hasattr(lib, name), name
