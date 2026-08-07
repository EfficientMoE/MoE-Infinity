from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifyStepAccounting:
    committed: int
    truncate_target: int
    cached_len: int
    emitted_len: int


@dataclass
class SpecDecodeState:
    """Per-sequence cached-vs-emitted bookkeeping for serving-path DFlash.

    ``cached_len`` is the number of tokens whose KV is committed to the paged
    cache; ``emitted_len`` is the number of committed output tokens returned to
    the user. Invariant after every committed step:
    ``cached_len == prompt_len + emitted_len``.
    """

    seq_id: int
    prompt_len: int
    cached_len: int = -1
    emitted_len: int = 0

    def __post_init__(self) -> None:
        if self.prompt_len < 0:
            raise ValueError(f"prompt_len must be >= 0, got {self.prompt_len}")
        if self.cached_len < 0:
            self.cached_len = self.prompt_len

    def record_verify(self, block_len: int, committed: int) -> VerifyStepAccounting:
        """Reconcile counters after a verify step that appended ``block_len``.

        The verify forward transiently writes KV for all ``block_len`` block
        tokens; ``committed`` (1..block_len) of them are kept. Returns the target
        length to which the paged cache must be truncated to drop the rejected
        tail, and advances the cached/emitted counters.
        """
        if block_len < 1:
            raise ValueError(f"block_len must be >= 1, got {block_len}")
        if not 1 <= committed <= block_len:
            raise ValueError(
                f"committed must be in [1, {block_len}], got {committed}"
            )
        self.cached_len += committed
        self.emitted_len += committed
        return VerifyStepAccounting(
            committed=committed,
            truncate_target=self.cached_len,
            cached_len=self.cached_len,
            emitted_len=self.emitted_len,
        )

    def invariant_holds(self) -> bool:
        return self.cached_len == self.prompt_len + self.emitted_len


__all__ = ["SpecDecodeState", "VerifyStepAccounting"]
