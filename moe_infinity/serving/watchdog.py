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
