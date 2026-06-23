from __future__ import annotations

import importlib
import time
from typing import Protocol, cast


class _CircuitBreakerLike(Protocol):
    def __init__(
        self,
        error_rate_threshold: float = 0.01,
        latency_threshold_ms: float = 500.0,
        window_size: int = 100,
        cooldown_seconds: float = 60.0,
    ) -> None: ...

    def record_success(self, latency_ms: float) -> None: ...

    def record_failure(self) -> None: ...

    def is_open(self) -> bool: ...

    def get_state(self) -> str: ...


CircuitBreaker = cast(
    type[_CircuitBreakerLike],
    getattr(
        importlib.import_module(
            "moe_infinity.serving.contextpilot_circuit_breaker"
        ),
        "CircuitBreaker",
    ),
)


def _open_with_error_rate(cb: _CircuitBreakerLike) -> None:
    cb.record_success(latency_ms=10.0)
    cb.record_success(latency_ms=12.0)
    cb.record_failure()


def test_opens_on_high_error_rate() -> None:
    cb = CircuitBreaker(
        error_rate_threshold=0.25,
        latency_threshold_ms=1000.0,
        window_size=4,
        cooldown_seconds=60.0,
    )

    _open_with_error_rate(cb)

    assert cb.is_open() is True
    assert cb.get_state() == "open"


def test_opens_on_high_latency() -> None:
    cb = CircuitBreaker(
        error_rate_threshold=1.0,
        latency_threshold_ms=50.0,
        window_size=100,
        cooldown_seconds=60.0,
    )

    for _ in range(10):
        cb.record_success(latency_ms=120.0)

    assert cb.is_open() is True
    assert cb.get_state() == "open"


def test_auto_reset_after_cooldown() -> None:
    cb = CircuitBreaker(
        error_rate_threshold=0.25,
        latency_threshold_ms=1000.0,
        window_size=4,
        cooldown_seconds=0.01,
    )

    _open_with_error_rate(cb)
    assert cb.get_state() == "open"

    time.sleep(0.03)

    assert cb.get_state() == "half_open"
    assert cb.is_open() is False


def test_closes_after_success_in_half_open() -> None:
    cb = CircuitBreaker(
        error_rate_threshold=0.25,
        latency_threshold_ms=1000.0,
        window_size=4,
        cooldown_seconds=0.01,
    )

    _open_with_error_rate(cb)
    time.sleep(0.03)
    assert cb.get_state() == "half_open"

    cb.record_success(latency_ms=5.0)

    assert cb.get_state() == "closed"
    assert cb.is_open() is False


def test_reopens_after_failure_in_half_open() -> None:
    cb = CircuitBreaker(
        error_rate_threshold=0.25,
        latency_threshold_ms=1000.0,
        window_size=4,
        cooldown_seconds=0.01,
    )

    _open_with_error_rate(cb)
    time.sleep(0.03)
    assert cb.get_state() == "half_open"

    cb.record_failure()

    assert cb.get_state() == "open"
    assert cb.is_open() is True
