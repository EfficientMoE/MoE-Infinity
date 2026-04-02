from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from typing_extensions import override

if TYPE_CHECKING:
    from moe_infinity.serving.health import ServerHealthState


logger = logging.getLogger(__name__)


def _hard_exit() -> None:
    os._exit(1)


@dataclass
class WatchdogConfig:
    """Configuration for watchdog threads. None = disabled."""

    startup_timeout: Optional[float] = None
    decode_step_timeout: Optional[float] = None
    enable_pyspy_dump: bool = False

    def __post_init__(self) -> None:
        if self.startup_timeout is not None and self.startup_timeout <= 0:
            raise ValueError(
                f"startup_timeout must be > 0, got {self.startup_timeout}"
            )
        if (
            self.decode_step_timeout is not None
            and self.decode_step_timeout <= 0
        ):
            raise ValueError(
                f"decode_step_timeout must be > 0, got {self.decode_step_timeout}"
            )

    def is_startup_watchdog_enabled(self) -> bool:
        return self.startup_timeout is not None

    def is_decode_watchdog_enabled(self) -> bool:
        return self.decode_step_timeout is not None


class StartupWatchdog(threading.Thread):
    def __init__(
        self,
        health_state: "ServerHealthState",
        config: WatchdogConfig,
        is_ready: Callable[[], bool],
    ) -> None:
        super().__init__(daemon=True, name="startup-watchdog")
        self._health_state: ServerHealthState = health_state
        self._config: WatchdogConfig = config
        self._is_ready: Callable[[], bool] = is_ready
        self._cancelled: threading.Event = threading.Event()

    @override
    def run(self) -> None:
        timeout = self._config.startup_timeout
        if timeout is None:
            return

        deadline = time.monotonic() + timeout
        while not self._cancelled.is_set():
            if self._is_ready():
                return
            if time.monotonic() >= deadline:
                logger.error(
                    "Startup watchdog timeout after %.1fs. Server startup incomplete.",
                    timeout,
                )
                self._health_state.set_unhealthy(
                    f"Startup watchdog timeout after {timeout:.1f}s"
                )
                _hard_exit()
                return
            time.sleep(1.0)

    def cancel(self) -> None:
        self._cancelled.set()


def start_startup_watchdog(
    health_state: "ServerHealthState",
    config: WatchdogConfig,
    is_ready: Callable[[], bool],
) -> Optional[StartupWatchdog]:
    if not config.is_startup_watchdog_enabled():
        return None

    watchdog = StartupWatchdog(health_state, config, is_ready)
    watchdog.start()
    return watchdog


class DecodeWatchdog(threading.Thread):
    """Daemon thread that marks server unhealthy if decode steps stop completing.

    Uses feed() to reset the timer. Only active during decode (not during init,
    idle). Does NOT call os._exit — just marks unhealthy and lets external
    orchestrator handle.
    """

    def __init__(
        self,
        health_state: "ServerHealthState",
        config: WatchdogConfig,
    ) -> None:
        super().__init__(daemon=True, name="decode-watchdog")
        self._health_state: ServerHealthState = health_state
        self._config: WatchdogConfig = config
        self._active: threading.Event = threading.Event()
        self._stop_event: threading.Event = threading.Event()
        self._last_feed_time: float = time.monotonic()

    def feed(self) -> None:
        """Reset the decode timeout timer. Must be called each decode step.

        This is a hot path — only updates a monotonic timestamp.
        """
        self._last_feed_time = time.monotonic()

    def activate(self) -> None:
        """Enable timeout monitoring (call when decode loop has pending requests)."""
        self._last_feed_time = time.monotonic()
        self._active.set()

    def deactivate(self) -> None:
        """Disable timeout monitoring (call when engine is idle)."""
        self._active.clear()

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()

    @override
    def run(self) -> None:
        """Monitor decode progress. Marks unhealthy if timeout exceeded while active."""
        timeout = self._config.decode_step_timeout
        if timeout is None:
            return

        while not self._stop_event.is_set():
            if self._active.is_set():
                elapsed = time.monotonic() - self._last_feed_time
                if elapsed > timeout:
                    logger.error(
                        "Decode watchdog timeout: no decode step completed in %.1fs",
                        elapsed,
                    )
                    self._health_state.set_unhealthy(
                        f"Decode watchdog timeout: no step in {elapsed:.1f}s"
                    )
            time.sleep(0.1)


def start_decode_watchdog(
    health_state: "ServerHealthState",
    config: WatchdogConfig,
) -> Optional[DecodeWatchdog]:
    """Start decode watchdog if enabled in config. Returns watchdog or None."""
    if not config.is_decode_watchdog_enabled():
        return None
    watchdog = DecodeWatchdog(health_state, config)
    watchdog.start()
    return watchdog
