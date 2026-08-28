from __future__ import annotations

import dataclasses
import sys as _sys
from collections.abc import Callable
from types import SimpleNamespace

import torch

for _stale in [
    name
    for name in list(_sys.modules)
    if name.startswith("moe_infinity.serving")
    or name.startswith("moe_infinity.runtime")
]:
    del _sys.modules[_stale]

from moe_infinity.runtime.attention_backend import (
    LayeredPagedKVPayload,
    LayeredPagedKVStore,
    PagedAttentionBackend,
)
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.serving.kv_cache import PagedKVCache, SequenceAllocationPlan
from moe_infinity.serving.model_runner import ModelRunner
from moe_infinity.serving.prefix_cache import CacheNamespace, PrefixCache
from moe_infinity.serving.prefix_contract import PrefixLease, PrefixMatch
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)

SHARED = list(range(8))

__all__ = [
    "SHARED",
    "CacheNamespace",
    "PrefixCache",
    "PrefixLease",
    "PrefixMatch",
    "RecordingLayeredPagedKVStore",
    "RecordingPrefixLeaseProvider",
    "RefRecorder",
    "SamplingParams",
    "SequenceAllocationPlan",
    "SequenceData",
    "SequenceGroup",
    "SequenceStatus",
    "make_cache",
    "make_group",
    "make_namespace",
    "make_paged_backend",
    "make_qwen_runner",
    "make_seeded_scheduler",
]


def make_namespace(**changes: object) -> CacheNamespace:
    base = CacheNamespace(
        model_id="Qwen/Qwen3-30B-A3B",
        model_revision="rev-a",
        tokenizer_id="Qwen/Qwen3-30B-A3B",
        tokenizer_revision="rev-a",
        tokenizer_config_digest="tok-digest",
        adapter_id=None,
        adapter_revision=None,
        dtype="bfloat16",
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        attention_backend="flashinfer-paged",
        attention_layout="NHD",
        position_config_digest="rope-default;window=none",
        runtime_epoch="epoch-1",
    )
    return dataclasses.replace(base, **changes)


class RefRecorder:
    def __init__(self) -> None:
        self.retained: list[list[int]] = []
        self.released: list[list[int]] = []

    def retain(self, ids: list[int]) -> None:
        self.retained.append(list(ids))

    def release(self, ids: list[int]) -> None:
        self.released.append(list(ids))


class RecordingLayeredPagedKVStore(LayeredPagedKVStore):
    def __init__(
        self,
        num_layers: int = 3,
        num_blocks: int = 8,
        block_size: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 8,
        dtype: torch.dtype = torch.float32,
        owner: object | None = None,
    ) -> None:
        owner = owner or SimpleNamespace()
        super().__init__(
            owner=owner,
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=torch.device("cpu"),
            use_flashinfer=True,
        )
        setattr(owner, "block_store", self)
        self.copies: list[tuple[int, int, tuple[int, ...]]] = []

    def import_blocks(
        self, block_ids: list[int], payload: LayeredPagedKVPayload
    ) -> None:
        super().import_blocks(block_ids, payload)
        if len(block_ids) == 1 and len(payload.source_block_ids) == 1:
            self.copies.append(
                (
                    payload.source_block_ids[0],
                    block_ids[0],
                    tuple(range(self.num_layers)),
                )
            )

    def layer_values(self, block_ids: list[int]) -> list[list[float]]:
        return self.fi_kv_cache[:, block_ids, 0, 0, 0, 0].tolist()


def make_cache(
    store: RecordingLayeredPagedKVStore | None = None, num_blocks: int = 8
) -> PagedKVCache:
    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=3,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    if store is not None:
        cache.set_block_store(store, owner=store.owner)
    return cache


def make_paged_backend(num_blocks: int) -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            block_size=4,
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
    )


def make_qwen_runner(
    layer_indices: list[int], expected_layers: int
) -> ModelRunner:
    class FakeQwen3PagedAttention:
        @classmethod
        def set_paged_context(cls, backend, metadata) -> None:
            cls.backend, cls.metadata = backend, metadata

        @classmethod
        def clear_paged_context(cls) -> None:
            cls.backend, cls.metadata = None, None

    FakeQwen3PagedAttention.__name__ = "Qwen3PagedAttention"
    modules = []
    for layer_idx in layer_indices:
        module = FakeQwen3PagedAttention()
        module.layer_idx = layer_idx
        modules.append(module)
    model = SimpleNamespace(
        modules=lambda: iter(modules),
        config=SimpleNamespace(
            num_hidden_layers=expected_layers, sliding_window=None
        ),
    )
    backend = SimpleNamespace()
    store = RecordingLayeredPagedKVStore(
        num_layers=expected_layers, owner=backend
    )
    backend.num_gpu_blocks = store.num_blocks
    backend.spec = SimpleNamespace(
        block_size=store.block_size,
        num_kv_heads=store.num_kv_heads,
        head_dim=store.head_dim,
        dtype=store.dtype,
    )
    backend.register_layers = lambda registrations: None
    backend.create_layered_store = lambda *, layer_count: store
    backend._flashinfer_enabled = lambda: True
    owner = SimpleNamespace(
        get_attention_backend=lambda: backend, expert_layer_modules=[]
    )
    return ModelRunner(model, owner, device=torch.device("cpu"))


class RecordingPrefixLeaseProvider:
    def __init__(self, cache: PrefixCache) -> None:
        self.cache = cache
        self.events: list[str] = []

    @property
    def open_leases(self) -> int:
        return self.cache.open_leases

    def acquire_prefix_lease(
        self,
        namespace: CacheNamespace,
        token_ids: list[int],
        max_prefix_tokens: int,
    ) -> PrefixLease:
        self.events.append(f"pin:{token_ids[-1]}")
        return self.cache.acquire_prefix_lease(
            namespace, token_ids, max_prefix_tokens
        )

    def evict_until(self, predicate: Callable[[], bool]) -> None:
        self.events.append("evict")
        self.cache.evict_until(predicate)


def make_group(
    request_id: str, rows: list[tuple[int, list[int]]]
) -> SequenceGroup:
    return SequenceGroup(
        request_id=request_id,
        sequences=[
            SequenceData(
                seq_id=seq_id,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(),
            )
            for seq_id, tokens in rows
        ],
    )


def make_seeded_scheduler(
    *, num_blocks: int, max_batch_size: int
) -> tuple[
    Scheduler, PagedKVCache, RecordingPrefixLeaseProvider, CacheNamespace
]:
    store = RecordingLayeredPagedKVStore(num_blocks=num_blocks)
    cache = make_cache(store=store, num_blocks=num_blocks)
    namespace = make_namespace(num_layers=3)
    prefix = PrefixCache(
        block_size=4,
        max_entries=8,
        on_retain=cache.block_allocator.retain,
        on_release=cache.block_allocator.release,
    )
    cache.allocate_sequence(999, 9)
    seed_blocks = cache.get_block_table(999)
    prefix.insert(namespace, SHARED + [99], seed_blocks, committed_tokens=8)
    cache.free_sequence(999)
    provider = RecordingPrefixLeaseProvider(prefix)
    scheduler = Scheduler(
        cache,
        max_batch_size=max_batch_size,
        max_tokens_per_step=64,
        prefix_lease_provider=provider,
        cache_namespace=namespace,
    )
    return scheduler, cache, provider, namespace
