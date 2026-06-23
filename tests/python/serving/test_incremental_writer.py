# pyright: reportAny=false, reportImplicitOverride=false

import importlib.util
import sys
import threading
import time
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


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_INCREMENTAL_WRITER_MODULE = _load_module(
    "moe_infinity.serving.incremental_writer",
    ROOT / "moe_infinity" / "serving" / "incremental_writer.py",
)

IncrementalWriter = _INCREMENTAL_WRITER_MODULE.IncrementalWriter


def test_write_and_load_completed(tmp_path: Path) -> None:
    path = tmp_path / "incremental.jsonl"
    writer = IncrementalWriter(path)

    try:
        for index in range(3):
            writer.save(f"seq-{index}", [index], {"index": index})

        assert writer.load_completed() == {"seq-0", "seq-1", "seq-2"}
    finally:
        writer.close()


def test_partial_recovery(tmp_path: Path) -> None:
    path = tmp_path / "incremental.jsonl"
    writer = IncrementalWriter(path)

    try:
        writer.save("seq-1", [1], {"index": 1})
        writer.save("seq-2", [2], {"index": 2})
    finally:
        writer.close()

    recovered = IncrementalWriter(path)
    try:
        assert recovered.load_completed() == {"seq-1", "seq-2"}
    finally:
        recovered.close()


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    writer = IncrementalWriter(path)

    try:
        writer.close()
        path.unlink()
        assert writer.load_completed() == set()
    finally:
        if path.exists():
            path.unlink()


def test_concurrent_writes(tmp_path: Path) -> None:
    path = tmp_path / "incremental.jsonl"
    writer = IncrementalWriter(path)
    barrier = threading.Barrier(10)

    def worker(index: int) -> None:
        barrier.wait()
        writer.save(f"seq-{index}", [index], {"index": index})

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(10)
    ]

    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert writer.load_completed() == {
            f"seq-{index}" for index in range(10)
        }
    finally:
        writer.close()


def test_performance_fsync(tmp_path: Path) -> None:
    path = tmp_path / "incremental.jsonl"
    writer = IncrementalWriter(path)

    try:
        start = time.perf_counter()
        for index in range(100):
            writer.save(f"seq-{index}", [index], {"index": index})
        elapsed = time.perf_counter() - start

        assert elapsed < 5
        assert len(writer.load_completed()) == 100
    finally:
        writer.close()
