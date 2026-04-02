from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast


class _NoArgCtor(Protocol):
    def __call__(self) -> object: ...


class _WithMiddlewareCtor(Protocol):
    def __call__(self, middleware: object) -> object: ...


def _load_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "moe_infinity" / "serving" / "cp_kv_interface.py"
    spec = importlib.util.spec_from_file_location(
        "cp_kv_interface", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cp_kv_interface module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_module()
CPAwareKVManager = cast(object, getattr(_MODULE, "CPAwareKVManager"))
ContextPilotKVManager = cast(
    _WithMiddlewareCtor, getattr(_MODULE, "ContextPilotKVManager")
)
NullCPAwareKVManager = cast(
    _NoArgCtor, getattr(_MODULE, "NullCPAwareKVManager")
)


def test_null_manager_predict_returns_zero() -> None:
    manager = NullCPAwareKVManager()
    predict = cast(
        Callable[[str, list[int]], float],
        getattr(manager, "predict_prefix_reuse"),
    )

    assert predict("req-1", [1, 2, 3]) == 0.0


def test_null_manager_allocation_priority_is_identity() -> None:
    manager = NullCPAwareKVManager()
    priority = cast(
        Callable[[list[str]], list[str]],
        getattr(manager, "get_allocation_priority"),
    )
    request_ids = ["req-c", "req-a", "req-b"]

    assert priority(request_ids) == request_ids


def test_cp_manager_predict_returns_float_in_range() -> None:
    class FakeMiddleware:
        def predict_prefix_reuse(
            self, request_id: str, token_ids: list[int]
        ) -> float:
            _ = request_id
            _ = token_ids
            return 0.73

    manager = ContextPilotKVManager(FakeMiddleware())
    predict = cast(
        Callable[[str, list[int]], float],
        getattr(manager, "predict_prefix_reuse"),
    )

    score = predict("req-1", [11, 12, 13])

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_cp_manager_get_allocation_priority_returns_sorted() -> None:
    class FakeMiddleware:
        _scores: dict[str, float] = {
            "req-low": 0.1,
            "req-high": 0.9,
            "req-mid": 0.4,
        }

        def predict_prefix_reuse(
            self, request_id: str, token_ids: list[int]
        ) -> float:
            _ = token_ids
            return float(self._scores.get(request_id, 0.0))

    manager = ContextPilotKVManager(FakeMiddleware())
    priority = cast(
        Callable[[list[str]], list[str]],
        getattr(manager, "get_allocation_priority"),
    )

    ordered = priority(["req-low", "req-high", "req-mid"])

    assert ordered == ["req-high", "req-mid", "req-low"]


def test_interface_contract_completeness() -> None:
    required = {
        "predict_prefix_reuse",
        "get_cp_cached_blocks",
        "notify_blocks_allocated",
        "notify_blocks_freed",
        "get_allocation_priority",
    }

    assert inspect.isabstract(CPAwareKVManager)

    empty_methods: set[str] = set()

    cp_abs = cast(
        set[str],
        getattr(CPAwareKVManager, "__abstractmethods__", empty_methods),
    )
    null_abs = cast(
        set[str],
        getattr(NullCPAwareKVManager, "__abstractmethods__", empty_methods),
    )
    cp_impl_abs = cast(
        set[str],
        getattr(ContextPilotKVManager, "__abstractmethods__", empty_methods),
    )

    assert required.issubset(cp_abs)
    assert null_abs == set()
    assert cp_impl_abs == set()

    _ = NullCPAwareKVManager()
    _ = ContextPilotKVManager(object())
