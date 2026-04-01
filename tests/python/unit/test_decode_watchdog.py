# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false

import time

from moe_infinity.serving.health import ServerHealthState
from moe_infinity.serving.watchdog import (
    DecodeWatchdog,
    WatchdogConfig,
    start_decode_watchdog,
)


def test_disabled_when_decode_timeout_none() -> None:
    health = ServerHealthState()
    config = WatchdogConfig(decode_step_timeout=None)

    watchdog = start_decode_watchdog(health_state=health, config=config)

    assert watchdog is None


def test_feed_prevents_timeout() -> None:
    health = ServerHealthState()
    health.set_healthy()
    config = WatchdogConfig(decode_step_timeout=0.1)
    watchdog = DecodeWatchdog(health, config)

    watchdog.start()
    watchdog.activate()
    try:
        for _ in range(8):
            time.sleep(0.03)
            watchdog.feed()

        assert health.is_healthy() is True
        assert health.get_status_dict() == {"status": "healthy", "reason": None}
    finally:
        watchdog.stop()
        watchdog.join(timeout=1.0)


def test_timeout_fires_when_active_without_feed() -> None:
    health = ServerHealthState()
    health.set_healthy()
    config = WatchdogConfig(decode_step_timeout=0.1)
    watchdog = DecodeWatchdog(health, config)

    watchdog.start()
    watchdog.activate()
    try:
        time.sleep(0.5)

        assert health.is_healthy() is False
        assert health.get_status_dict()["status"] == "unhealthy"
    finally:
        watchdog.stop()
        watchdog.join(timeout=1.0)


def test_deactivated_no_trigger() -> None:
    health = ServerHealthState()
    health.set_healthy()
    config = WatchdogConfig(decode_step_timeout=0.1)
    watchdog = DecodeWatchdog(health, config)

    watchdog.start()
    watchdog.activate()
    watchdog.feed()
    watchdog.deactivate()
    try:
        time.sleep(0.4)

        assert health.is_healthy() is True
        assert health.get_status_dict() == {"status": "healthy", "reason": None}
    finally:
        watchdog.stop()
        watchdog.join(timeout=1.0)
