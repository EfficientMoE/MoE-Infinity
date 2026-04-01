# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportMissingImports=false

import pytest

from moe_infinity.serving.watchdog import WatchdogConfig


def test_defaults_disable_watchdogs():
    config = WatchdogConfig()

    assert config.startup_timeout is None
    assert config.decode_step_timeout is None
    assert config.enable_pyspy_dump is False


def test_startup_watchdog_enabled_only_when_timeout_set():
    assert WatchdogConfig().is_startup_watchdog_enabled() is False
    assert (
        WatchdogConfig(startup_timeout=1.5).is_startup_watchdog_enabled()
        is True
    )


def test_decode_watchdog_enabled_only_when_timeout_set():
    assert WatchdogConfig().is_decode_watchdog_enabled() is False
    assert (
        WatchdogConfig(decode_step_timeout=2.0).is_decode_watchdog_enabled()
        is True
    )


def test_negative_startup_timeout_raises():
    with pytest.raises(ValueError, match="startup_timeout must be > 0"):
        WatchdogConfig(startup_timeout=-1.0)


def test_negative_decode_step_timeout_raises():
    with pytest.raises(ValueError, match="decode_step_timeout must be > 0"):
        WatchdogConfig(decode_step_timeout=0)
