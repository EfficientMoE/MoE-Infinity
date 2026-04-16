from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass


def hash_token_block(token_ids: list[int]) -> str:
    token_bytes = bytes(token_id % 256 for token_id in token_ids)
    return hashlib.sha256(token_bytes).hexdigest()[:16]


@dataclass
class _CacheEntry:
    block_id: int
    token_block: tuple[int, ...]


class PrefixCache:
    block_size: int
    max_entries: int
    _entries: OrderedDict[str, _CacheEntry]
    _hits: int
    _misses: int

    def __init__(self, block_size: int = 16, max_entries: int = 1000) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")

        self.block_size = block_size
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._hits = 0
        self._misses = 0

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        matched_block_ids: list[int] = []

        for block_tokens in self._iter_full_blocks(token_ids):
            block_hash = hash_token_block(block_tokens)
            cache_entry = self._entries.get(block_hash)
            if cache_entry is None:
                break

            if cache_entry.token_block != tuple(block_tokens):
                break

            matched_block_ids.append(cache_entry.block_id)
            self._entries.move_to_end(block_hash)

        if matched_block_ids:
            self._hits += 1
        else:
            self._misses += 1

        num_matched_tokens = len(matched_block_ids) * self.block_size
        return num_matched_tokens, matched_block_ids

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None:
        num_full_blocks = len(token_ids) // self.block_size
        if len(block_ids) < num_full_blocks:
            raise ValueError(
                f"block_ids must include one entry per full token block; need {num_full_blocks}, got {len(block_ids)}"
            )

        for block_index, block_tokens in enumerate(
            self._iter_full_blocks(token_ids)
        ):
            block_hash = hash_token_block(block_tokens)
            self._entries[block_hash] = _CacheEntry(
                block_id=block_ids[block_index],
                token_block=tuple(block_tokens),
            )
            self._entries.move_to_end(block_hash)

        overflow = len(self._entries) - self.max_entries
        if overflow > 0:
            _ = self.evict_lru(overflow)

    def evict_lru(self, n: int = 1) -> list[str]:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")

        evicted_hashes: list[str] = []
        for _ in range(min(n, len(self._entries))):
            evicted_hash, _ = self._entries.popitem(last=False)
            evicted_hashes.append(evicted_hash)
        return evicted_hashes

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def _iter_full_blocks(self, token_ids: list[int]) -> Iterator[list[int]]:
        for start in range(
            0, len(token_ids) - self.block_size + 1, self.block_size
        ):
            yield token_ids[start : start + self.block_size]


__all__ = ["PrefixCache", "hash_token_block"]
