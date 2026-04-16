import importlib.machinery
import sys
import types
from types import SimpleNamespace
from typing import Optional, cast

import torch

from moe_infinity.engine.kv_cache_offload_coordinator import (
    KVCacheOffloadCoordinator,
)
from moe_infinity.engine.transfer_types import TransferType
from moe_infinity.utils.config import ArcherConfig

if (
    "flash_attn" not in sys.modules
    or getattr(sys.modules["flash_attn"], "__spec__", None) is None
):
    flash_attn_stub = sys.modules.get(
        "flash_attn", types.ModuleType("flash_attn")
    )
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub

from moe_infinity.runtime.model_offload import OffloadEngine


class _DummyScheduler:
    def __init__(self) -> None:
        self.handlers: dict[TransferType, object] = {}

    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None:
        self.handlers[transfer_type] = handler


def test_default_config_has_offload_disabled() -> None:
    config = ArcherConfig()
    assert not config.enable_kv_cache_offload


def test_coordinator_not_registered_when_disabled() -> None:
    scheduler = _DummyScheduler()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 8),
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=False),
    )

    coordinator.register_with_scheduler(scheduler)

    assert TransferType.KV_SWAP_OUT not in scheduler.handlers
    assert TransferType.KV_SWAP_IN not in scheduler.handlers


def test_coordinator_registered_when_enabled() -> None:
    scheduler = _DummyScheduler()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 8),
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    coordinator.register_with_scheduler(scheduler)

    assert TransferType.KV_SWAP_OUT in scheduler.handlers
    assert TransferType.KV_SWAP_IN in scheduler.handlers


def test_capture_kv_skips_when_disabled() -> None:
    engine = OffloadEngine.__new__(OffloadEngine)
    setattr(engine, "_enable_kv_cache_offload", False)
    setattr(engine, "_captured_kv", {})

    mock_past_kv = ((torch.randn(2, 4, 8), torch.randn(2, 4, 8)),)

    getattr(OffloadEngine, "_capture_kv_cache")(
        engine,
        seq_id=0,
        past_key_values=mock_past_kv,
    )

    assert getattr(engine, "_captured_kv") == {}


def test_capture_reload_active_when_enabled() -> None:
    engine = OffloadEngine.__new__(OffloadEngine)
    setattr(engine, "_enable_kv_cache_offload", True)
    setattr(engine, "_captured_kv", {})
    engine.model = None

    past_key_values = ((torch.randn(2, 4, 8), torch.randn(2, 4, 8)),)

    getattr(OffloadEngine, "_capture_kv_cache")(
        engine,
        seq_id=7,
        past_key_values=past_key_values,
    )
    assert 7 in getattr(engine, "_captured_kv")

    restored = cast(
        Optional[tuple[object, ...]],
        getattr(OffloadEngine, "_reload_kv_cache")(engine, seq_id=7),
    )
    assert restored is not None
    assert len(restored) == len(past_key_values)
    assert 7 not in getattr(engine, "_captured_kv")
