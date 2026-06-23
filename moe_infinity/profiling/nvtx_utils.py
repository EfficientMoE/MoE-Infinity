from __future__ import annotations

import functools
import importlib
from types import TracebackType
from typing import Protocol, cast

from .io_profiler import IOProfiler


class _DecoratableCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _ContextManagerLike(Protocol):
    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...


class _NoOpContext:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


_NOOP = _NoOpContext()
_cached_nvtx: object | None = None


def _get_nvtx() -> object | None:
    global _cached_nvtx
    if _cached_nvtx is None:
        try:
            _cached_nvtx = importlib.import_module("nvtx")
        except Exception:
            return None
    return _cached_nvtx


class _ProfileAndAnnotate:
    _stage: str
    _nvtx_color: str
    _profiler_cm: _ContextManagerLike
    _nvtx_cm: _ContextManagerLike

    def __init__(self, stage: str, nvtx_color: str = "blue"):
        self._stage = stage
        self._nvtx_color = nvtx_color
        self._profiler_cm = _NOOP
        self._nvtx_cm = _NOOP

    def __call__(self, fn: _DecoratableCallable) -> _DecoratableCallable:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            with self:
                return fn(*args, **kwargs)

        return wrapper

    def __enter__(self) -> None:
        self._profiler_cm = IOProfiler.instance().time(self._stage)
        _ = self._profiler_cm.__enter__()

        self._nvtx_cm = _NOOP
        nvtx_mod = _get_nvtx()
        if nvtx_mod is not None:
            annotate = getattr(nvtx_mod, "annotate", None)
            if callable(annotate):
                self._nvtx_cm = cast(
                    _ContextManagerLike,
                    annotate(self._stage, color=self._nvtx_color),
                )
        _ = self._nvtx_cm.__enter__()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        suppress_nvtx = self._nvtx_cm.__exit__(exc_type, exc, tb)
        suppress_profiler = self._profiler_cm.__exit__(exc_type, exc, tb)
        return bool(suppress_nvtx or suppress_profiler)


def profile_and_annotate(
    stage: str,
    nvtx_color: str = "blue",
) -> _ProfileAndAnnotate:
    return _ProfileAndAnnotate(stage=stage, nvtx_color=nvtx_color)
