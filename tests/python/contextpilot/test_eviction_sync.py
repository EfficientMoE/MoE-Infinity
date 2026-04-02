# pyright: reportAny=false

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_EVICTION_SYNC_MODULE = _load_module(
    "moe_infinity.serving.eviction_sync",
    ROOT / "moe_infinity" / "serving" / "eviction_sync.py",
)
EvictionSyncAdapter = _EVICTION_SYNC_MODULE.EvictionSyncAdapter


class MockCP:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def on_request_complete(self, rid: str) -> None:
        self.removed.append(rid)


def test_finished_triggers_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-finished")

    assert cp.removed == ["req-finished"]


def test_aborted_triggers_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_aborted("req-aborted")

    assert cp.removed == ["req-aborted"]


def test_kv_blocks_freed_triggers_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_freed("req-freed")

    assert cp.removed == ["req-freed"]


def test_swap_does_not_evict() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_kv_blocks_swapped("req-swapped")

    assert cp.removed == []
    assert adapter.get_counters() == {
        "evict_incoming": 0,
        "evict_removed": 0,
        "evict_not_found": 0,
    }


def test_idempotent_eviction() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-idempotent")
    adapter.on_request_aborted("req-idempotent")

    assert cp.removed == ["req-idempotent"]


def test_counters_increment() -> None:
    cp = MockCP()
    adapter = EvictionSyncAdapter(cp)

    adapter.on_request_finished("req-1")
    adapter.on_request_aborted("req-2")
    adapter.on_kv_blocks_freed("req-2")
    adapter.on_kv_blocks_swapped("req-3")

    assert adapter.get_counters() == {
        "evict_incoming": 3,
        "evict_removed": 2,
        "evict_not_found": 1,
    }


def test_no_middleware_is_noop() -> None:
    adapter = EvictionSyncAdapter(None)

    adapter.on_request_finished("req-finished")
    adapter.on_request_aborted("req-aborted")
    adapter.on_kv_blocks_freed("req-freed")
    adapter.on_kv_blocks_swapped("req-swapped")

    assert adapter.get_counters() == {
        "evict_incoming": 3,
        "evict_removed": 0,
        "evict_not_found": 0,
    }
