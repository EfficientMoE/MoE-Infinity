import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, Protocol, Union, cast

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_ = sys.modules.pop("moe_infinity", None)
_ = sys.modules.pop("moe_infinity.serving", None)
MEMORY_MANAGER_PATH = (
    Path(ROOT) / "moe_infinity" / "serving" / "memory_manager.py"
)


class MemoryBudgetProtocol(Protocol):
    total_gpu_memory_bytes: int
    model_memory_bytes: int
    expert_cache_ratio: float
    kv_cache_ratio: float
    activation_reserve_ratio: float

    @property
    def available_bytes(self) -> int: ...

    @property
    def expert_cache_bytes(self) -> int: ...

    @property
    def kv_cache_bytes(self) -> int: ...


class MemoryManagerProtocol(Protocol):
    total_gpu_memory_bytes: int
    device_memory_ratio: float
    kv_cache_ratio: float

    def __init__(
        self,
        device: Optional[torch.device] = None,
        device_memory_ratio: float = 0.75,
        kv_cache_ratio: float = 0.25,
        activation_reserve_ratio: float = 0.10,
    ) -> None: ...

    def compute_budget(
        self, model_memory_bytes: int
    ) -> MemoryBudgetProtocol: ...

    def get_max_kv_blocks(
        self,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int: ...

    def get_expert_cache_ratio(self) -> float: ...

    def report(self) -> dict[str, Union[str, int, float]]: ...


def _load_classes() -> (
    tuple[
        type[MemoryBudgetProtocol],
        type[MemoryManagerProtocol],
    ]
):
    module_name = "task7_memory_manager"
    spec = importlib.util.spec_from_file_location(
        module_name, MEMORY_MANAGER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MEMORY_MANAGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return (
        cast(type[MemoryBudgetProtocol], getattr(module, "MemoryBudget")),
        cast(type[MemoryManagerProtocol], getattr(module, "MemoryManager")),
    )


def test_budget_computation() -> None:
    _, MemoryManager = _load_classes()
    gib = 1024**3
    manager = MemoryManager(
        device=torch.device("cpu"),
        device_memory_ratio=1.0,
        kv_cache_ratio=0.5,
        activation_reserve_ratio=0.1,
    )
    manager.total_gpu_memory_bytes = 24 * gib

    budget = manager.compute_budget(model_memory_bytes=10 * gib)
    expected_available = (24 * gib) - (10 * gib) - int((24 * gib) * 0.1)
    expected_expert = int(expected_available * 0.5)
    expected_kv = int(expected_available * 0.5)

    assert budget.available_bytes == expected_available
    assert budget.expert_cache_bytes == expected_expert
    assert budget.kv_cache_bytes == expected_kv


def test_get_max_kv_blocks() -> None:
    _, MemoryManager = _load_classes()
    gib = 1024**3
    manager = MemoryManager(
        device=torch.device("cpu"),
        device_memory_ratio=1.0,
        kv_cache_ratio=0.5,
        activation_reserve_ratio=0.0,
    )
    manager.total_gpu_memory_bytes = 4 * gib
    budget = manager.compute_budget(model_memory_bytes=0)

    blocks = manager.get_max_kv_blocks(
        block_size=16,
        num_layers=2,
        num_heads=8,
        head_dim=64,
        dtype=torch.float16,
    )
    kv_bytes_per_block = 2 * 16 * 2 * 8 * 64 * 2
    expected_blocks = budget.kv_cache_bytes // kv_bytes_per_block

    assert isinstance(blocks, int)
    assert blocks > 0
    assert blocks == expected_blocks


def test_expert_cache_ratio() -> None:
    _, MemoryManager = _load_classes()
    manager = MemoryManager(
        device=torch.device("cpu"),
        device_memory_ratio=0.8,
        kv_cache_ratio=0.25,
    )

    expert_ratio = manager.get_expert_cache_ratio()
    kv_portion = manager.device_memory_ratio * manager.kv_cache_ratio
    assert abs(expert_ratio + kv_portion - manager.device_memory_ratio) < 1e-12


def test_report_serializable() -> None:
    _, MemoryManager = _load_classes()
    gib = 1024**3
    manager = MemoryManager(device=torch.device("cpu"))
    manager.total_gpu_memory_bytes = 8 * gib
    _ = manager.compute_budget(model_memory_bytes=1 * gib)

    report = manager.report()
    encoded = json.dumps(report)

    assert isinstance(report, dict)
    assert isinstance(encoded, str)
    assert "kv_cache_bytes" in report


def test_handles_no_gpu() -> None:
    _, MemoryManager = _load_classes()
    manager = MemoryManager(device=torch.device("cpu"))

    budget = manager.compute_budget(model_memory_bytes=1024)
    assert manager.total_gpu_memory_bytes == 0
    assert budget.available_bytes == 0
    assert budget.expert_cache_bytes == 0
    assert budget.kv_cache_bytes == 0


def test_absolute_targets_cannot_consume_free_reserve() -> None:
    _, MemoryManager = _load_classes()
    manager = MemoryManager(
        device=torch.device("cpu"),
        device_memory_ratio=1.0,
        kv_cache_ratio=0.5,
        activation_reserve_ratio=0.1,
    )
    manager.total_gpu_memory_bytes = 2048 * 1024**2
    budget = manager.compute_budget(
        model_memory_bytes=512 * 1024**2,
        free_memory_reserve_bytes=256 * 1024**2,
    )
    assert budget.expert_cache_bytes + budget.kv_cache_bytes <= 1076 * 1024**2


def test_static_call_keeps_existing_ratio_result() -> None:
    _, MemoryManager = _load_classes()
    manager = MemoryManager(
        device=torch.device("cpu"),
        device_memory_ratio=0.8,
        kv_cache_ratio=0.25,
    )
    manager.total_gpu_memory_bytes = 8 * 1024**3
    assert manager.compute_budget(0).expert_cache_ratio == pytest.approx(0.6)
