import importlib
import sys
from pathlib import Path
from types import ModuleType

import torch

from moe_infinity.engine.kv_cache_offload_coordinator import (
    KVCacheOffloadCoordinator,
)
from moe_infinity.engine.transfer_types import TransferType

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_import_works():
    module: ModuleType = importlib.import_module(
        "moe_infinity.engine.kv_cache_offload_coordinator"
    )
    assert hasattr(module, "KVCacheOffloadCoordinator")


def test_class_has_required_methods():
    assert hasattr(KVCacheOffloadCoordinator, "register_with_scheduler")
    assert hasattr(KVCacheOffloadCoordinator, "handle_swap_out")
    assert hasattr(KVCacheOffloadCoordinator, "handle_swap_in")


def test_instantiation_with_tensor_does_not_crash():
    kv_tensors = torch.zeros(2, 4, 8)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors, block_pool=object(), config={}
    )
    assert coordinator.__dict__["_cpu_cache"] == {}


def test_register_with_scheduler_accepts_scheduler_like_object():
    class DummyScheduler:
        def __init__(self) -> None:
            self.handlers: list[tuple[TransferType, object]] = []

        def register_handler(
            self, transfer_type: TransferType, handler: object
        ) -> None:
            self.handlers.append((transfer_type, handler))

    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 8), block_pool=object(), config=None
    )
    scheduler = DummyScheduler()
    coordinator.register_with_scheduler(scheduler)
    assert len(scheduler.handlers) == 2
