# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false

import os
import time
from unittest.mock import patch

from moe_infinity.serving.health import ServerHealthState
from moe_infinity.serving.watchdog import (
    StartupWatchdog,
    WatchdogConfig,
    start_startup_watchdog,
)


def test_disabled_when_startup_timeout_none() -> None:
    health = ServerHealthState()
    config = WatchdogConfig(startup_timeout=None)

    watchdog = start_startup_watchdog(
        health_state=health,
        config=config,
        is_ready=lambda: False,
    )

    assert watchdog is None


def test_cancel_on_ready() -> None:
    health = ServerHealthState()
    config = WatchdogConfig(startup_timeout=1.0)
    watchdog = StartupWatchdog(health, config, is_ready=lambda: True)

    watchdog.start()
    watchdog.join(timeout=1.0)

    assert watchdog.is_alive() is False
    assert health.get_status_dict() == {"status": "starting", "reason": None}


@patch.object(os, "_exit")
def test_timeout_triggers_unhealthy(mock_exit) -> None:
    health = ServerHealthState()
    config = WatchdogConfig(startup_timeout=0.1)
    watchdog = StartupWatchdog(health, config, is_ready=lambda: False)

    watchdog.start()
    watchdog.join(timeout=2.0)

    assert health.is_healthy() is False
    assert health.get_status_dict()["status"] == "unhealthy"
    mock_exit.assert_called_once_with(1)


def test_cancel_stops_polling() -> None:
    health = ServerHealthState()
    config = WatchdogConfig(startup_timeout=5.0)
    watchdog = StartupWatchdog(health, config, is_ready=lambda: False)

    watchdog.start()
    time.sleep(0.05)
    watchdog.cancel()
    watchdog.join(timeout=2.0)

    assert watchdog.is_alive() is False
