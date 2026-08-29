"""Serving-path two-phase memory resize adapter and transaction receipt.

This module provides the safe, drain-first reallocation primitives used by the
adaptive expert/KV controller on the serving path:

- :class:`ResizeReceipt` is the immutable, single-use record produced by a
  scheduler quiescence window. It carries the affected device, the CUDA
  completion events that must have synchronized before any storage is replaced,
  an optional post-publication event guarding release of the retained old
  bundle, and the strong references to old storage/wrappers kept alive until
  CUDA proves no kernel still references them.
- :class:`ServingMemoryResizer` enforces donor-first ordering between the expert
  cache and the KV cache and reports honest partial-donor commits when an
  irreversible expert eviction is followed by a failed KV growth.

No CUDA mutation policy lives here; policy targets come from
``moe_infinity.memory.adaptive_memory``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from moe_infinity.memory.adaptive_memory import (
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
    ResizeResult,
)


@dataclass
class ResizeReceipt:
    """Single-use record of a serving KV-resize maintenance window.

    A receipt is created while admissions are paused and the request/native
    queues have been drained. It proves that a completion event was recorded on
    every affected stream; ``resize_num_blocks`` refuses to run until every
    event in ``completion_events`` reports complete.

    The receipt also owns the strong references to the *old* KV tensor, block
    allocator, and FlashInfer wrappers (populated by the cache during resize).
    Those objects are only released via :meth:`release_retained_objects` after a
    post-publication CUDA event proves no kernel still references them.

    A receipt may be consumed or cancelled exactly once; reuse raises.
    """

    device_id: int
    completion_events: tuple[Any, ...] = ()
    post_publish_event: Any | None = None
    admissions_paused: bool = False
    retained_objects: list[Any] = field(default_factory=list)
    _consumed: bool = field(default=False, repr=False)
    _cancelled: bool = field(default=False, repr=False)
    _released: bool = field(default=False, repr=False)

    def events_synchronized(self) -> bool:
        """True iff every recorded completion event reports complete."""
        return all(
            self._event_complete(event) for event in self.completion_events
        )

    def ensure_usable(self, *, device_id: int) -> None:
        """Validate the receipt targets ``device_id`` and is unconsumed.

        Raises ``RuntimeError`` on device mismatch, prior consumption, or
        completion events that have not yet synchronized.
        """
        if self._consumed or self._cancelled:
            raise RuntimeError("resize receipt already consumed")
        if self.device_id != device_id:
            raise RuntimeError(
                "resize receipt device mismatch: receipt is for device "
                f"{self.device_id}, requested device {device_id}"
            )
        if not self.events_synchronized():
            raise RuntimeError(
                "resize receipt CUDA completion events have not synchronized"
            )

    def consume(self) -> None:
        """Mark the receipt consumed. Idempotency is not allowed: reuse raises."""
        if self._consumed or self._cancelled:
            raise RuntimeError("resize receipt already consumed")
        self._consumed = True

    def cancel(self) -> None:
        """Mark the receipt cancelled without releasing retained references."""
        if self._consumed or self._cancelled:
            raise RuntimeError("resize receipt already consumed")
        self._cancelled = True

    def retain(self, *objects: Any) -> None:
        """Record strong references to old storage/wrappers for later release."""
        self.retained_objects.extend(objects)

    def release_retained_objects(self) -> None:
        """Drop strong references to the old bundle after CUDA completion.

        Refuses to release until the ``post_publish_event`` (if any) reports
        complete, so a kernel that still references the old storage cannot have
        it freed underneath it.
        """
        if self._released:
            return
        if self.post_publish_event is not None and not self._event_complete(
            self.post_publish_event
        ):
            raise RuntimeError(
                "cannot release retained objects before post-publication "
                "CUDA completion"
            )
        self.retained_objects = []
        self._released = True

    @staticmethod
    def _event_complete(event: Any) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if callable(query):
            return bool(query())
        return True


class ServingMemoryResizer:
    """Donor-first serving reallocation between the expert and KV caches.

    ``expert_pool`` is the resident expert cache (either the two-phase
    ``reserve_victims``/``commit_reserved_victims`` API, or a simpler
    ``shrink_to(target) -> bool`` fallback). ``kv`` is the paged KV cache whose
    physical block count grows via ``resize_num_blocks``. ``reserve_probe`` maps
    a device id to currently free device bytes; growth is gated on
    ``free >= configured_reserve + receiver_growth``.
    """

    def __init__(
        self,
        expert_pool: Any,
        kv: Any,
        *,
        reserve_probe: Callable[[int], int],
        free_reserve_bytes: int = 0,
    ) -> None:
        self._expert = expert_pool
        self._kv = kv
        self._reserve_probe = reserve_probe
        self._free_reserve_bytes = free_reserve_bytes

    def apply(
        self,
        device_id: int,
        targets: MemoryTargets,
        *,
        current_expert_bytes: int,
        current_kv_blocks: int,
        kv_block_bytes: int,
    ) -> ResizeResult:
        if targets.direction is ResizeDirection.EXPERT_TO_KV:
            return self._expert_to_kv(
                device_id,
                targets,
                current_expert_bytes=current_expert_bytes,
                current_kv_blocks=current_kv_blocks,
                kv_block_bytes=kv_block_bytes,
            )
        return ResizeResult(
            device_id=device_id,
            outcome=ResizeOutcome.REJECTED,
            expert_bytes=current_expert_bytes,
            kv_blocks=current_kv_blocks,
            reason=f"unsupported direction {targets.direction.value}",
            kv_supported=targets.kv_supported,
        )

    def _expert_to_kv(
        self,
        device_id: int,
        targets: MemoryTargets,
        *,
        current_expert_bytes: int,
        current_kv_blocks: int,
        kv_block_bytes: int,
    ) -> ResizeResult:
        reservation = self._reserve_expert_donor(
            device_id, targets.expert_bytes
        )
        if reservation is None:
            return ResizeResult(
                device_id=device_id,
                outcome=ResizeOutcome.REJECTED,
                expert_bytes=current_expert_bytes,
                kv_blocks=current_kv_blocks,
                reason="expert donor shrink rejected",
                kv_supported=targets.kv_supported,
            )

        committed_expert_bytes = self._commit_expert_donor(
            reservation, targets.expert_bytes
        )

        growth_bytes = max(
            0, (targets.kv_blocks - current_kv_blocks) * kv_block_bytes
        )
        free_bytes = int(self._reserve_probe(device_id))
        if free_bytes < self._free_reserve_bytes + growth_bytes:
            return ResizeResult(
                device_id=device_id,
                outcome=ResizeOutcome.PARTIAL_DONOR_COMMITTED,
                expert_bytes=committed_expert_bytes,
                kv_blocks=current_kv_blocks,
                reason="insufficient free memory for KV growth after eviction",
                kv_supported=targets.kv_supported,
            )

        try:
            self._kv.resize_num_blocks(targets.kv_blocks)
        except (torch.OutOfMemoryError, RuntimeError):
            return ResizeResult(
                device_id=device_id,
                outcome=ResizeOutcome.PARTIAL_DONOR_COMMITTED,
                expert_bytes=committed_expert_bytes,
                kv_blocks=current_kv_blocks,
                reason="kv growth failed after expert eviction",
                kv_supported=targets.kv_supported,
            )

        return ResizeResult(
            device_id=device_id,
            outcome=ResizeOutcome.COMMITTED,
            expert_bytes=committed_expert_bytes,
            kv_blocks=targets.kv_blocks,
            reason=targets.reason,
            kv_supported=targets.kv_supported,
        )

    def _reserve_expert_donor(
        self, device_id: int, target_bytes: int
    ) -> Any | None:
        """Reserve donor victims, returning ``None`` when shrink is rejected.

        Prefers the reversible two-phase ``reserve_victims`` API; falls back to
        a boolean ``shrink_to`` when that is all the pool exposes.
        """
        reserve_victims = getattr(type(self._expert), "reserve_victims", None)
        if callable(reserve_victims):
            reservation = self._expert.reserve_victims(device_id, target_bytes)
            if reservation is None or not getattr(reservation, "ready", True):
                cancel = getattr(self._expert, "cancel_reservation", None)
                if reservation is not None and callable(cancel):
                    cancel(reservation)
                return None
            return reservation

        shrink_to = getattr(self._expert, "shrink_to", None)
        if callable(shrink_to):
            return _ShrinkToDonor if shrink_to(target_bytes) else None

        return None

    def _commit_expert_donor(self, reservation: Any, target_bytes: int) -> int:
        """Commit the reserved donor eviction, returning resident bytes."""
        if reservation is _ShrinkToDonor:
            return target_bytes
        commit = getattr(self._expert, "commit_reserved_victims", None)
        if callable(commit):
            return int(commit(reservation))
        return target_bytes


class _ShrinkToDonor:
    """Sentinel marking a committed ``shrink_to`` fallback reservation."""


__all__ = ["ResizeReceipt", "ServingMemoryResizer"]
