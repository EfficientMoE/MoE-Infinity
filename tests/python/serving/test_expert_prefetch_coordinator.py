# pyright: reportAny=false

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

_COORDINATOR_MODULE = _load_module(
    "moe_infinity.serving.expert_prefetch_coordinator",
    ROOT / "moe_infinity" / "serving" / "expert_prefetch_coordinator.py",
)

ExpertPrefetchCoordinator = _COORDINATOR_MODULE.ExpertPrefetchCoordinator


def test_update_and_get_priority() -> None:
    coordinator = ExpertPrefetchCoordinator(num_layers=8, num_experts=64)

    coordinator.update_sequence_activations(1, 3, [42, 42])
    coordinator.update_sequence_activations(2, 3, [42, 7])

    priority = coordinator.get_priority_experts(next_layer_id=3, max_experts=8)

    assert priority[0] == (3, 42)
    assert set(priority) == {(3, 42), (3, 7)}


def test_deduplication_across_sequences() -> None:
    coordinator = ExpertPrefetchCoordinator(num_layers=8, num_experts=64)

    coordinator.update_sequence_activations(10, 3, [42])
    coordinator.update_sequence_activations(11, 3, [42])
    coordinator.update_sequence_activations(12, 3, [42])

    priority = coordinator.get_priority_experts(next_layer_id=3, max_experts=8)

    assert priority == [(3, 42)]


def test_clear_sequence_removes_entries() -> None:
    coordinator = ExpertPrefetchCoordinator(num_layers=8, num_experts=64)

    coordinator.update_sequence_activations(1, 3, [42])
    coordinator.update_sequence_activations(2, 3, [42])

    coordinator.clear_sequence(2)

    priority = coordinator.get_priority_experts(next_layer_id=3, max_experts=8)
    stats = coordinator.get_dedup_stats()

    assert priority == [(3, 42)]
    assert stats["tracked_sequences"] == 1
    assert stats["deduplicated_loads"] == 0


def test_priority_ordering() -> None:
    coordinator = ExpertPrefetchCoordinator(num_layers=8, num_experts=64)

    coordinator.update_sequence_activations(1, 3, [10])
    coordinator.update_sequence_activations(2, 3, [10])
    coordinator.update_sequence_activations(3, 3, [11])
    coordinator.update_sequence_activations(4, 3, [12])

    priority = coordinator.get_priority_experts(next_layer_id=3, max_experts=8)

    assert priority[:3] == [(3, 10), (3, 11), (3, 12)]


def test_dedup_stats() -> None:
    coordinator = ExpertPrefetchCoordinator(num_layers=8, num_experts=64)

    coordinator.update_sequence_activations(1, 3, [42, 7])
    coordinator.update_sequence_activations(2, 3, [42])
    coordinator.update_sequence_activations(3, 3, [7])

    priority = coordinator.get_priority_experts(next_layer_id=3, max_experts=8)
    stats = coordinator.get_dedup_stats()

    assert set(priority) == {(3, 42), (3, 7)}
    assert stats["total_requested_loads"] == 4
    assert stats["unique_loads"] == 2
    assert stats["deduplicated_loads"] == 2
    assert stats["num_priority_queries"] == 1
