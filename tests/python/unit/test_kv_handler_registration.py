from types import SimpleNamespace
from typing import Union, cast

import torch

from moe_infinity.engine.kv_cache_offload_coordinator import (
    KVCacheOffloadCoordinator,
)
from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)

KVCachedData = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class _DummyScheduler:
    def __init__(self) -> None:
        self._handlers: dict[TransferType, object] = {}

    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None:
        self._handlers[transfer_type] = handler


def _request(
    transfer_id: str,
    transfer_type: TransferType,
    block_ids: list[int],
    target_device: str = "cpu",
) -> TransferRequest:
    return TransferRequest(
        transfer_id=transfer_id,
        transfer_type=transfer_type,
        priority=TransferPriority.NORMAL,
        source_device="cuda:0",
        target_device=target_device,
        tensor_id="kv",
        block_ids=block_ids,
    )


def test_handlers_registered_when_enabled() -> None:
    scheduler = _DummyScheduler()
    config = SimpleNamespace(enable_kv_cache_offload=True)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 3),
        block_pool=None,
        config=config,
    )

    coordinator.register_with_scheduler(scheduler)

    handlers = cast(dict[TransferType, object], getattr(scheduler, "_handlers"))
    assert TransferType.KV_SWAP_OUT in handlers
    assert TransferType.KV_SWAP_IN in handlers


def test_handlers_not_registered_when_disabled() -> None:
    scheduler = _DummyScheduler()
    config = SimpleNamespace(enable_kv_cache_offload=False)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 3),
        block_pool=None,
        config=config,
    )

    coordinator.register_with_scheduler(scheduler)

    handlers = cast(dict[TransferType, object], getattr(scheduler, "_handlers"))
    assert TransferType.KV_SWAP_OUT not in handlers
    assert TransferType.KV_SWAP_IN not in handlers


def test_handle_swap_out_copies_tensors() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="transfer-1",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0, 1],
    )

    coordinator.handle_swap_out(swap_out)

    cpu_cache = cast(
        dict[str, KVCachedData], getattr(coordinator, "_cpu_cache")
    )
    assert "transfer-1" in cpu_cache
    cached = cpu_cache["transfer-1"]
    assert isinstance(cached, torch.Tensor)
    torch.testing.assert_close(cached, kv_tensors[:, [0, 1], ...])


def test_handle_swap_in_restores_tensors() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    original = kv_tensors.clone()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="transfer-2",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0, 1],
    )
    coordinator.handle_swap_out(swap_out)

    kv_tensors[:, [0, 1], ...] = -1.0

    swap_in = _request(
        transfer_id="transfer-2",
        transfer_type=TransferType.KV_SWAP_IN,
        block_ids=[0, 1],
        target_device="cpu",
    )
    coordinator.handle_swap_in(swap_in)

    torch.testing.assert_close(
        kv_tensors[:, [0, 1], ...], original[:, [0, 1], ...]
    )
    cpu_cache = cast(
        dict[str, KVCachedData], getattr(coordinator, "_cpu_cache")
    )
    assert "transfer-2" not in cpu_cache


def test_set_kv_store_installs_store_identity_before_registration() -> None:
    from moe_infinity.runtime.kv_cache_format import (
        allocate_layered_paged_kv_store,
    )

    store = allocate_layered_paged_kv_store(
        owner_id="coord-owner",
        format_name="int8_sym",
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        execution_dtype=torch.float16,
        device=torch.device("cpu"),
    )
    scheduler = _DummyScheduler()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=None,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )
    coordinator.set_kv_store(store)
    coordinator.register_with_scheduler(scheduler)

    assert getattr(coordinator, "_kv_store") is store
    assert getattr(coordinator, "_kv_store").format.name == "int8_sym"
    handlers = cast(dict[TransferType, object], getattr(scheduler, "_handlers"))
    assert TransferType.KV_SWAP_OUT in handlers
    assert TransferType.KV_SWAP_IN in handlers
