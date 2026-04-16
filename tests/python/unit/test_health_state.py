# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportAny=false

import threading
from typing import Optional

from moe_infinity.serving.health import HealthState, ServerHealthState


def test_initial_state_is_starting() -> None:
    state = ServerHealthState()

    assert state.get_status_dict() == {"status": "starting", "reason": None}
    assert state.is_healthy() is False


def test_set_healthy_returns_healthy_status_dict() -> None:
    state = ServerHealthState()

    state.set_healthy()

    assert state.get_status_dict() == {"status": "healthy", "reason": None}


def test_set_unhealthy_returns_reason() -> None:
    state = ServerHealthState()

    state.set_unhealthy("reason msg")

    assert state.get_status_dict() == {
        "status": "unhealthy",
        "reason": "reason msg",
    }
    assert state.is_healthy() is False


def test_set_starting_clears_reason() -> None:
    state = ServerHealthState()

    state.set_unhealthy("reason msg")
    state.set_starting()

    assert state.get_status_dict() == {"status": "starting", "reason": None}
    assert state.is_healthy() is False


def test_is_healthy_only_when_healthy() -> None:
    state = ServerHealthState()

    assert state.is_healthy() is False
    state.set_starting()
    assert state.is_healthy() is False
    state.set_unhealthy("bad")
    assert state.is_healthy() is False
    state.set_healthy()
    assert state.is_healthy() is True


def test_state_transitions_work_correctly() -> None:
    state = ServerHealthState()

    state.set_healthy()
    assert state.get_status_dict() == {"status": "healthy", "reason": None}

    state.set_unhealthy("degraded")
    assert state.get_status_dict() == {
        "status": "unhealthy",
        "reason": "degraded",
    }

    state.set_healthy()
    assert state.get_status_dict() == {"status": "healthy", "reason": None}


def test_health_state_enum_values() -> None:
    assert HealthState.STARTING.value == "starting"
    assert HealthState.HEALTHY.value == "healthy"
    assert HealthState.UNHEALTHY.value == "unhealthy"


def test_thread_safety_under_concurrent_updates() -> None:
    state = ServerHealthState()
    errors: list[BaseException] = []

    def toggle(thread_index: int) -> None:
        try:
            for iteration in range(1000):
                if iteration % 2 == 0:
                    state.set_healthy()
                else:
                    state.set_unhealthy(f"reason-{thread_index}-{iteration}")
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=toggle, args=(index,)) for index in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    status: dict[str, Optional[str]] = state.get_status_dict()
    assert status["status"] in ("healthy", "unhealthy")
    if status["status"] == "healthy":
        assert status["reason"] is None
    else:
        assert isinstance(status["reason"], str)
