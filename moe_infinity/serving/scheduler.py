from __future__ import annotations

import logging
from collections import deque
from math import ceil

logger = logging.getLogger(__name__)

from .batch import SchedulerOutput
from .kv_cache import PagedKVCache
from .sequence import SequenceData, SequenceGroup, SequenceStatus


class Scheduler:
    kv_cache: PagedKVCache
    max_batch_size: int
    max_tokens_per_step: int

    def __init__(
        self,
        kv_cache: PagedKVCache,
        max_batch_size: int = 32,
        max_tokens_per_step: int = 2048,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be > 0, got {max_batch_size}"
            )
        if max_tokens_per_step <= 0:
            raise ValueError(
                f"max_tokens_per_step must be > 0, got {max_tokens_per_step}"
            )

        self.kv_cache = kv_cache
        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step

        self._waiting: deque[SequenceGroup] = deque()
        self._running: deque[SequenceGroup] = deque()
        self._swapped: deque[SequenceGroup] = deque()

        self._sequence_map: dict[int, SequenceData] = {}
        self._request_map: dict[str, SequenceGroup] = {}

    def add_request(self, seq_group: SequenceGroup) -> None:
        if seq_group.request_id in self._request_map:
            raise ValueError(
                f"request_id '{seq_group.request_id}' already exists"
            )

        for sequence in seq_group.sequences:
            if sequence.seq_id in self._sequence_map:
                raise ValueError(
                    f"sequence id {sequence.seq_id} already exists"
                )
            if sequence.status is not SequenceStatus.WAITING:
                sequence.set_status(SequenceStatus.WAITING)
            self._sequence_map[sequence.seq_id] = sequence

        self._request_map[seq_group.request_id] = seq_group
        self._waiting.append(seq_group)

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        swapped_snapshot = list(self._swapped)

        scheduled_seqs = 0
        scheduled_tokens = 0

        self._recover_swapped_groups(swapped_snapshot)

        waiting_blocked = False
        while self._waiting and not waiting_blocked:
            next_group = self._waiting[0]
            next_seqs = [
                sequence
                for sequence in next_group.sequences
                if sequence.status is SequenceStatus.WAITING
            ]

            if not next_seqs:
                _ = self._waiting.popleft()
                continue

            prefill_tokens = sum(
                self._num_prefill_tokens(sequence) for sequence in next_seqs
            )

            if scheduled_seqs + len(next_seqs) > self.max_batch_size:
                break
            if scheduled_tokens + prefill_tokens > self.max_tokens_per_step:
                break

            required_blocks = sum(
                self._required_blocks(sequence) for sequence in next_seqs
            )

            if self.kv_cache.block_allocator.num_free_blocks < required_blocks:
                preempted_seq_ids = self._preempt_oldest_running_group()
                if not preempted_seq_ids:
                    waiting_blocked = True
                    continue
                output.preempted_seq_ids.extend(preempted_seq_ids)

                if (
                    self.kv_cache.block_allocator.num_free_blocks
                    < required_blocks
                ):
                    waiting_blocked = True
                    continue

            _ = self._waiting.popleft()
            self._running.append(next_group)

            for sequence in next_seqs:
                self.kv_cache.allocate_sequence(
                    sequence.seq_id,
                    num_tokens=sequence.prompt_length,
                )
                sequence.set_status(SequenceStatus.PREFILL)
                output.prefill_seq_ids.append(sequence.seq_id)

            scheduled_seqs += len(next_seqs)
            scheduled_tokens += prefill_tokens
            output.num_prefill_tokens += prefill_tokens

        for group in self._running:
            for sequence in group.sequences:
                if sequence.status is not SequenceStatus.DECODE:
                    continue
                if scheduled_seqs >= self.max_batch_size:
                    return output
                if scheduled_tokens >= self.max_tokens_per_step:
                    return output

                output.decode_seq_ids.append(sequence.seq_id)
                output.num_decode_tokens += 1
                scheduled_seqs += 1
                scheduled_tokens += 1

        return output

    def update_after_step(
        self,
        completed_seq_ids: list[int],
        new_decode_seq_ids: list[int],
    ) -> None:
        completed = set(completed_seq_ids)

        for seq_id in dict.fromkeys(new_decode_seq_ids):
            sequence = self._sequence_map.get(seq_id)
            if sequence is None or seq_id in completed:
                continue

            if sequence.status is SequenceStatus.PREFILL:
                sequence.set_status(SequenceStatus.DECODE)

            if sequence.status is SequenceStatus.DECODE:
                try:
                    self.kv_cache.append_tokens(seq_id, num_new_tokens=1)
                except KeyError:
                    pass

        for seq_id in completed:
            sequence = self._sequence_map.get(seq_id)
            if sequence is None:
                continue

            if sequence.status in (
                SequenceStatus.WAITING,
                SequenceStatus.PREFILL,
                SequenceStatus.DECODE,
                SequenceStatus.SWAPPED,
            ):
                sequence.set_status(SequenceStatus.FINISHED)
            self.kv_cache.free_sequence(seq_id)

        self._prune_finished_and_cancelled_requests()

    def abort_request(self, request_id: str) -> None:
        group = self._request_map.get(request_id)
        if group is None:
            return

        for sequence in group.sequences:
            if sequence.status not in (
                SequenceStatus.FINISHED,
                SequenceStatus.CANCELLED,
            ):
                sequence.set_status(SequenceStatus.CANCELLED)
            self.kv_cache.free_sequence(sequence.seq_id)

        self._waiting = deque(
            queued
            for queued in self._waiting
            if queued.request_id != request_id
        )
        self._running = deque(
            queued
            for queued in self._running
            if queued.request_id != request_id
        )
        self._swapped = deque(
            queued
            for queued in self._swapped
            if queued.request_id != request_id
        )

        self._drop_request_metadata(group)
        self._prune_finished_and_cancelled_requests()

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def get_running_seq_ids(self) -> list[int]:
        running_seq_ids: list[int] = []
        for group in self._running:
            for sequence in group.sequences:
                if sequence.status in (
                    SequenceStatus.PREFILL,
                    SequenceStatus.DECODE,
                ):
                    running_seq_ids.append(sequence.seq_id)
        return running_seq_ids

    def _preempt_oldest_running_group(self) -> list[int]:
        while self._running:
            group = self._running.popleft()
            preempted_seq_ids: list[int] = []

            for sequence in group.sequences:
                if sequence.status not in (
                    SequenceStatus.PREFILL,
                    SequenceStatus.DECODE,
                ):
                    continue

                try:
                    self.kv_cache.swap_out(sequence.seq_id)
                except KeyError:
                    pass
                self.kv_cache.free_gpu_blocks(sequence.seq_id)
                sequence.set_status(SequenceStatus.SWAPPED)
                preempted_seq_ids.append(sequence.seq_id)

            if preempted_seq_ids:
                self._swapped.append(group)
                return preempted_seq_ids

        return []

    def _recover_swapped_groups(
        self, swapped_groups: list[SequenceGroup]
    ) -> None:
        if self.kv_cache.block_allocator.num_free_blocks <= 0:
            return

        for group in swapped_groups:
            if group not in self._swapped:
                continue

            swapped_sequences = [
                sequence
                for sequence in group.sequences
                if sequence.status is SequenceStatus.SWAPPED
            ]
            if not swapped_sequences:
                _ = self._swapped.remove(group)
                continue

            recovered = True
            for sequence in swapped_sequences:
                try:
                    self.kv_cache.swap_in(sequence.seq_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "swap_in failed for seq_id=%s: %s",
                        sequence.seq_id,
                        exc,
                    )
                    recovered = False
                    break

            if not recovered:
                continue

            for sequence in swapped_sequences:
                sequence.set_status(SequenceStatus.DECODE)

            _ = self._swapped.remove(group)
            self._running.appendleft(group)

    def _prune_finished_and_cancelled_requests(self) -> None:
        active_requests: dict[str, SequenceGroup] = {}
        finished_groups: list[SequenceGroup] = []

        for group in list(self._request_map.values()):
            has_active = False
            for sequence in group.sequences:
                if sequence.status not in (
                    SequenceStatus.FINISHED,
                    SequenceStatus.CANCELLED,
                ):
                    has_active = True
                    break

            if has_active:
                active_requests[group.request_id] = group
            else:
                finished_groups.append(group)

        for group in finished_groups:
            self._drop_request_metadata(group)

        self._request_map = active_requests
        self._waiting = deque(
            group
            for group in self._waiting
            if group.request_id in self._request_map
        )
        self._running = deque(
            group
            for group in self._running
            if group.request_id in self._request_map
        )
        self._swapped = deque(
            group
            for group in self._swapped
            if group.request_id in self._request_map
        )

    def _drop_request_metadata(self, group: SequenceGroup) -> None:
        _ = self._request_map.pop(group.request_id, None)
        for sequence in group.sequences:
            _ = self._sequence_map.pop(sequence.seq_id, None)

    def _required_blocks(self, sequence: SequenceData) -> int:
        if sequence.prompt_length == 0:
            return 0
        return ceil(sequence.prompt_length / self.kv_cache.block_size)

    @staticmethod
    def _num_prefill_tokens(sequence: SequenceData) -> int:
        return max(0, sequence.prompt_length - sequence.num_computed_tokens)


RequestScheduler = Scheduler


__all__ = ["Scheduler", "RequestScheduler"]
