from __future__ import annotations

import dataclasses

from moe_infinity.serving.prefix_cache import CacheNamespace, PrefixCache
from moe_infinity.serving.prefix_contract import PrefixLease, PrefixMatch

__all__ = [
    "CacheNamespace",
    "PrefixCache",
    "PrefixLease",
    "PrefixMatch",
    "RefRecorder",
    "make_namespace",
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
