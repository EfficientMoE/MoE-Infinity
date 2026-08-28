from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Union, cast

import torch

from moe_infinity.engine.transfer_types import TransferRequest, TransferType
from moe_infinity.utils.async_transfer import (
    async_d2h,
    async_h2d,
    wait_transfer,
)

KVTensors = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class _SchedulerLike(Protocol):
    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None: ...


class KVCacheOffloadCoordinator:
    def __init__(
        self,
        kv_tensors: KVTensors | None,
        block_pool: object,
        config: Mapping[str, object] | object | None,
    ) -> None:
        self._kv_tensors: KVTensors | None = kv_tensors
        self._block_pool: object = block_pool
        self._config: Mapping[str, object] | object | None = config
        self._transfer_scheduler: _SchedulerLike | None = None
        self._cpu_cache: dict[str, KVTensors] = {}
        self._kv_store: object | None = None

    def set_kv_tensors(self, kv_tensors: KVTensors) -> None:
        self._kv_tensors = kv_tensors

    def set_kv_store(self, store: object) -> None:
        self._kv_store = store
        tensors = getattr(store, "tensors", None)
        if tensors is None:
            raise ValueError(
                "KVCacheOffloadCoordinator.set_kv_store requires a store "
                "exposing packed payload/scale tensors"
            )
        if len(tensors) == 1:
            self._kv_tensors = tensors[0]
        else:
            self._kv_tensors = (tensors[0], tensors[1])

    def register_with_scheduler(self, scheduler: _SchedulerLike) -> None:
        self._transfer_scheduler = scheduler
        if not self._is_enabled():
            return
        scheduler.register_handler(
            TransferType.KV_SWAP_OUT, self.handle_swap_out
        )
        scheduler.register_handler(TransferType.KV_SWAP_IN, self.handle_swap_in)

    def handle_swap_out(self, request: TransferRequest) -> None:
        if self._kv_tensors is None:
            return

        block_ids = list(request.block_ids)
        stream = self._make_cuda_stream_if_needed()

        if isinstance(self._kv_tensors, tuple):
            k_cache, v_cache = self._kv_tensors
            k_blocks = k_cache[block_ids, ...].clone()
            v_blocks = v_cache[block_ids, ...].clone()
            if stream is not None and k_blocks.is_cuda and v_blocks.is_cuda:
                cpu_data: KVTensors = (
                    async_d2h(k_blocks, stream),
                    async_d2h(v_blocks, stream),
                )
            else:
                cpu_data = (
                    k_blocks.to("cpu", non_blocking=True),
                    v_blocks.to("cpu", non_blocking=True),
                )
        else:
            selected = self._kv_tensors[:, block_ids, ...].clone()
            if stream is not None and selected.is_cuda:
                cpu_data = async_d2h(selected, stream)
            else:
                cpu_data = selected.to("cpu", non_blocking=True)

        if stream is not None:
            wait_transfer(stream)

        self._cpu_cache[request.transfer_id] = cpu_data

    def handle_swap_in(self, request: TransferRequest) -> None:
        if self._kv_tensors is None:
            return

        cpu_data = self._cpu_cache.pop(request.transfer_id, None)
        if cpu_data is None:
            return

        block_ids = list(request.block_ids)
        target_device = torch.device(request.target_device)
        stream = (
            torch.cuda.Stream(device=target_device)
            if target_device.type == "cuda" and torch.cuda.is_available()
            else None
        )

        if isinstance(cpu_data, tuple):
            if not isinstance(self._kv_tensors, tuple):
                return
            k_blocks_cpu, v_blocks_cpu = cpu_data
            k_cache, v_cache = self._kv_tensors
            if stream is not None:
                k_blocks = async_h2d(k_blocks_cpu, target_device, stream)
                v_blocks = async_h2d(v_blocks_cpu, target_device, stream)
            else:
                k_blocks = k_blocks_cpu.to(target_device, non_blocking=True)
                v_blocks = v_blocks_cpu.to(target_device, non_blocking=True)
            if stream is not None:
                wait_transfer(stream)
            k_cache[block_ids, ...] = k_blocks
            v_cache[block_ids, ...] = v_blocks
            return

        if isinstance(self._kv_tensors, tuple):
            return

        if stream is not None:
            gpu_data = async_h2d(cpu_data, target_device, stream)
            wait_transfer(stream)
        else:
            gpu_data = cpu_data.to(target_device, non_blocking=True)
        self._kv_tensors[:, block_ids, ...] = gpu_data

    def _is_enabled(self) -> bool:
        if isinstance(self._config, Mapping):
            config_map = cast(Mapping[str, object], self._config)
            return bool(config_map.get("enable_kv_cache_offload", False))
        return bool(getattr(self._config, "enable_kv_cache_offload", False))

    def _make_cuda_stream_if_needed(self) -> object | None:
        if not torch.cuda.is_available():
            return None
        if self._kv_tensors is None:
            return None
        if isinstance(self._kv_tensors, tuple):
            return torch.cuda.Stream() if self._kv_tensors[0].is_cuda else None
        return torch.cuda.Stream() if self._kv_tensors.is_cuda else None

    def _select_block_tensors(self, block_ids: list[int]) -> KVTensors:
        if self._kv_tensors is None:
            raise RuntimeError("KV tensors are not initialized")
        if isinstance(self._kv_tensors, tuple):
            k_cache, v_cache = self._kv_tensors
            return k_cache[block_ids, ...], v_cache[block_ids, ...]
        return self._kv_tensors[:, block_ids, ...]


__all__ = ["KVCacheOffloadCoordinator"]
