from dataclasses import dataclass
from typing import Optional


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
