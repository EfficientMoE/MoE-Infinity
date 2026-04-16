from __future__ import annotations

import math
import threading
import time
from collections import deque


class CircuitBreaker:
    _STATE_CLOSED: str = "closed"
    _STATE_OPEN: str = "open"
    _STATE_HALF_OPEN: str = "half_open"

    def __init__(
        self,
        error_rate_threshold: float = 0.01,
        latency_threshold_ms: float = 500.0,
        window_size: int = 100,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._error_rate_threshold: float = float(error_rate_threshold)
        self._latency_threshold_ms: float = float(latency_threshold_ms)
        self._window_size: int = max(1, int(window_size))
        self._cooldown_seconds: float = max(0.0, float(cooldown_seconds))

        self._lock: threading.Lock = threading.Lock()
        self._state: str = self._STATE_CLOSED
        self._opened_at: float | None = None

        self._request_window: deque[bool] = deque(maxlen=self._window_size)
        self._latency_window: deque[float] = deque(maxlen=10)

    def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self._refresh_state_locked()

            if self._state == self._STATE_HALF_OPEN:
                self._close_locked()
                return

            self._request_window.append(True)
            self._latency_window.append(float(latency_ms))
            self._evaluate_closed_thresholds_locked()

    def record_failure(self) -> None:
        with self._lock:
            self._refresh_state_locked()

            if self._state == self._STATE_HALF_OPEN:
                self._open_locked()
                return

            self._request_window.append(False)
            self._evaluate_closed_thresholds_locked()

    def is_open(self) -> bool:
        with self._lock:
            self._refresh_state_locked()
            return self._state == self._STATE_OPEN

    def get_state(self) -> str:
        with self._lock:
            self._refresh_state_locked()
            return self._state

    def _evaluate_closed_thresholds_locked(self) -> None:
        if self._state != self._STATE_CLOSED:
            return

        if self._request_window:
            failures = sum(1 for success in self._request_window if not success)
            error_rate = failures / len(self._request_window)
            if error_rate > self._error_rate_threshold:
                self._open_locked()
                return

        if len(self._latency_window) == self._latency_window.maxlen:
            latency_p99 = self._compute_p99(list(self._latency_window))
            if latency_p99 > self._latency_threshold_ms:
                self._open_locked()

    def _refresh_state_locked(self) -> None:
        if self._state != self._STATE_OPEN:
            return
        if self._opened_at is None:
            return

        if (time.monotonic() - self._opened_at) >= self._cooldown_seconds:
            self._state = self._STATE_HALF_OPEN

    def _open_locked(self) -> None:
        self._state = self._STATE_OPEN
        self._opened_at = time.monotonic()

    def _close_locked(self) -> None:
        self._state = self._STATE_CLOSED
        self._opened_at = None
        self._request_window.clear()
        self._latency_window.clear()

    @staticmethod
    def _compute_p99(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(0.99 * len(ordered)) - 1)
        return ordered[index]


__all__ = ["CircuitBreaker"]
