import enum
import threading
from threading import Lock
from typing import Optional


class HealthState(enum.Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ServerHealthState:
    def __init__(self) -> None:
        self._lock: Lock = threading.Lock()
        self._state: HealthState = HealthState.STARTING
        self._reason: Optional[str] = None

    def set_healthy(self) -> None:
        with self._lock:
            self._state = HealthState.HEALTHY
            self._reason = None

    def set_unhealthy(self, reason: str) -> None:
        with self._lock:
            self._state = HealthState.UNHEALTHY
            self._reason = reason

    def set_starting(self) -> None:
        with self._lock:
            self._state = HealthState.STARTING
            self._reason = None

    def is_healthy(self) -> bool:
        with self._lock:
            return self._state is HealthState.HEALTHY

    def get_status_dict(self) -> dict[str, Optional[str]]:
        with self._lock:
            return {"status": self._state.value, "reason": self._reason}
