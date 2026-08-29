from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

from .batch import SchedulerOutput
from .kv_cache import PagedKVCache
from .memory_resize import ResizeReceipt
from .sequence import SequenceData, SequenceGroup, SequenceStatus


class _AlwaysCompleteEvent:
    def query(self) -> bool:
        return True


@dataclass
class _SchedulerStateSnapshot:
    waiting: "deque[SequenceGroup]"
    running: "deque[SequenceGroup]"
    swapped: "deque[SequenceGroup]"
    statuses: dict[int, SequenceStatus]


class CPAwareKVManager(Protocol):
    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float: ...


@dataclass(frozen=True)
class Deficit2D:
    tokens: int
    expert_bytes: int


@dataclass(frozen=True)
class VerifyDemand:
    seq_id: int
    tokens: int
    expert_bytes: int
    in_flight: bool = False


@dataclass(frozen=True)
class VerifyAdmission:
    seated_ids: tuple[int, ...]
    admitted_ids: tuple[int, ...]
    carried: Deficit2D


def _reject_negative(label: str, deficit: Deficit2D) -> None:
    if deficit.tokens < 0 or deficit.expert_bytes < 0:
        raise ValueError(
            f"{label} dimensions must be non-negative; got {deficit}"
        )


def admit_verify_demands(
    demands: list[VerifyDemand],
    budget: Deficit2D,
    carried: Deficit2D,
    deficit_cap: Deficit2D,
) -> VerifyAdmission:
    _reject_negative("budget", budget)
    _reject_negative("carried", carried)
    _reject_negative("deficit_cap", deficit_cap)
    for demand in demands:
        if demand.tokens < 0 or demand.expert_bytes < 0:
            raise ValueError(
                f"demand {demand.seq_id} dimensions must be non-negative; "
                f"got tokens={demand.tokens}, "
                f"expert_bytes={demand.expert_bytes}"
            )

    if demands:
        largest_tokens = max(demand.tokens for demand in demands)
        largest_bytes = max(demand.expert_bytes for demand in demands)
        if (
            deficit_cap.tokens < largest_tokens
            or deficit_cap.expert_bytes < largest_bytes
        ):
            raise ValueError(
                "deficit_cap must be >= the largest single demand in each "
                f"dimension; cap={deficit_cap}, largest_tokens="
                f"{largest_tokens}, largest_expert_bytes={largest_bytes}"
            )

    seated_ids = tuple(demand.seq_id for demand in demands if demand.in_flight)
    seated_tokens = sum(demand.tokens for demand in demands if demand.in_flight)
    seated_bytes = sum(
        demand.expert_bytes for demand in demands if demand.in_flight
    )

    pool_tokens = budget.tokens + carried.tokens - seated_tokens
    pool_bytes = budget.expert_bytes + carried.expert_bytes - seated_bytes

    admitted_ids: list[int] = []
    for demand in demands:
        if demand.in_flight:
            continue
        if demand.tokens <= pool_tokens and demand.expert_bytes <= pool_bytes:
            pool_tokens -= demand.tokens
            pool_bytes -= demand.expert_bytes
            admitted_ids.append(demand.seq_id)

    carried_out = Deficit2D(
        tokens=min(max(0, pool_tokens), deficit_cap.tokens),
        expert_bytes=min(max(0, pool_bytes), deficit_cap.expert_bytes),
    )
    return VerifyAdmission(
        seated_ids=seated_ids,
        admitted_ids=tuple(admitted_ids),
        carried=carried_out,
    )


@dataclass(frozen=True)
class _VerifyConfig:
    enabled: bool
    budget: Deficit2D
    deficit_cap: Deficit2D


def _resolve_verify_config(
    *,
    token_budget: Optional[int],
    expert_byte_budget: Optional[int],
    token_deficit_cap: Optional[int],
    expert_byte_deficit_cap: Optional[int],
) -> _VerifyConfig:
    provided = [
        token_budget,
        expert_byte_budget,
        token_deficit_cap,
        expert_byte_deficit_cap,
    ]
    if all(value is None for value in provided):
        zero = Deficit2D(tokens=0, expert_bytes=0)
        return _VerifyConfig(enabled=False, budget=zero, deficit_cap=zero)
    if any(value is None for value in provided):
        raise ValueError(
            "verify scheduling requires all four of token_budget, "
            "expert_byte_budget, token_deficit_cap, and "
            "expert_byte_deficit_cap, or none of them"
        )
    for name, value in (
        ("verify_token_budget", token_budget),
        ("verify_expert_byte_budget", expert_byte_budget),
        ("verify_token_deficit_cap", token_deficit_cap),
        ("verify_expert_byte_deficit_cap", expert_byte_deficit_cap),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    return _VerifyConfig(
        enabled=True,
        budget=Deficit2D(
            tokens=int(token_budget), expert_bytes=int(expert_byte_budget)
        ),
        deficit_cap=Deficit2D(
            tokens=int(token_deficit_cap),
            expert_bytes=int(expert_byte_deficit_cap),
        ),
    )


class Scheduler:
    kv_cache: PagedKVCache
    max_batch_size: int
    max_tokens_per_step: int

    def __init__(
        self,
        kv_cache: PagedKVCache,
        max_batch_size: int = 32,
        max_tokens_per_step: int = 2048,
        *,
        verify_token_budget: Optional[int] = None,
        verify_expert_byte_budget: Optional[int] = None,
        verify_token_deficit_cap: Optional[int] = None,
        verify_expert_byte_deficit_cap: Optional[int] = None,
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
        self._cp_kv_manager: Optional[CPAwareKVManager] = None

        self._verify_config = _resolve_verify_config(
            token_budget=verify_token_budget,
            expert_byte_budget=verify_expert_byte_budget,
            token_deficit_cap=verify_token_deficit_cap,
            expert_byte_deficit_cap=verify_expert_byte_deficit_cap,
        )
        self._verify_demands: dict[int, VerifyDemand] = {}
        self._carried_verify_deficit = Deficit2D(tokens=0, expert_bytes=0)

        self.admissions_paused: bool = False
        self._maintenance_backlog: deque[SequenceGroup] = deque()
        self._swap_failure_after: Optional[int] = None

    def set_cp_kv_manager(self, manager: CPAwareKVManager) -> None:
        self._cp_kv_manager = manager

    @property
    def verify_scheduling_enabled(self) -> bool:
        """True iff all four verify budgets were configured (Step-5 opt-in)."""
        return self._verify_config.enabled

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
        if self.admissions_paused:
            self._maintenance_backlog.append(seq_group)
        else:
            self._waiting.append(seq_group)

    def schedule(self) -> SchedulerOutput:
        if self.admissions_paused:
            return SchedulerOutput()

        output = SchedulerOutput()
        swapped_snapshot = list(self._swapped)

        scheduled_seqs = 0
        scheduled_tokens = 0

        self._recover_swapped_groups(swapped_snapshot)

        if self._cp_kv_manager is not None and len(self._waiting) > 1:
            scored_waiting = [
                (
                    self._cp_kv_manager.predict_prefix_reuse(
                        group.request_id,
                        self._group_token_ids(group),
                    ),
                    group,
                )
                for group in self._waiting
            ]
            scored_waiting.sort(key=lambda x: x[0], reverse=True)
            self._waiting = deque(group for _, group in scored_waiting)

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

        capacity_reached = False
        for group in self._running:
            if capacity_reached:
                break
            for sequence in group.sequences:
                if sequence.status is not SequenceStatus.DECODE:
                    continue
                if (
                    scheduled_seqs >= self.max_batch_size
                    or scheduled_tokens >= self.max_tokens_per_step
                ):
                    capacity_reached = True
                    break

                output.decode_seq_ids.append(sequence.seq_id)
                output.num_decode_tokens += 1
                scheduled_seqs += 1
                scheduled_tokens += 1

        self._apply_verify_scheduling(output)
        return output

    def set_verify_demand(
        self,
        seq_id: int,
        tokens: int,
        expert_bytes: int,
        in_flight: bool,
    ) -> None:
        self._verify_demands[seq_id] = VerifyDemand(
            seq_id=seq_id,
            tokens=tokens,
            expert_bytes=expert_bytes,
            in_flight=in_flight,
        )

    def clear_verify_demand(self, seq_id: int) -> None:
        _ = self._verify_demands.pop(seq_id, None)

    @property
    def carried_verify_deficit(self) -> Deficit2D:
        return self._carried_verify_deficit

    def _apply_verify_scheduling(self, output: SchedulerOutput) -> None:
        if not self._verify_config.enabled or not self._verify_demands:
            return

        demands = list(self._verify_demands.values())
        admission = admit_verify_demands(
            demands,
            self._verify_config.budget,
            self._carried_verify_deficit,
            self._verify_config.deficit_cap,
        )
        self._carried_verify_deficit = admission.carried

        verify_ids = [*admission.seated_ids, *admission.admitted_ids]
        verify_set = set(verify_ids)
        output.verify_seq_ids = list(verify_ids)
        output.draft_seq_ids = [
            demand.seq_id
            for demand in demands
            if demand.seq_id not in verify_set
        ]
        output.num_verify_tokens = sum(
            demand.tokens for demand in demands if demand.seq_id in verify_set
        )
        output.num_verify_expert_bytes = sum(
            demand.expert_bytes
            for demand in demands
            if demand.seq_id in verify_set
        )

    def update_after_step(
        self,
        completed_seq_ids: list[int],
        new_decode_seq_ids: list[int],
        committed_counts: dict[int, int] | None = None,
    ) -> None:
        completed = set(completed_seq_ids)

        for seq_id in dict.fromkeys(new_decode_seq_ids):
            sequence = self._sequence_map.get(seq_id)
            if sequence is None or seq_id in completed:
                continue

            if sequence.status is SequenceStatus.PREFILL:
                sequence.set_status(SequenceStatus.DECODE)

            if sequence.status is SequenceStatus.DECODE:
                num_new = (
                    1
                    if committed_counts is None
                    else committed_counts.get(seq_id, 1)
                )
                try:
                    self.kv_cache.append_tokens(seq_id, num_new_tokens=num_new)
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

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

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

    def quiesce_for_kv_resize(self, timeout_s: float = 30.0) -> ResizeReceipt:
        """Pause admissions, drain running groups, and record completion events.

        Runs only between ``schedule()``/``update_after_step()`` calls. Sets the
        admission gate, swaps every PREFILL/DECODE group to CPU, frees their GPU
        blocks, records a CUDA completion event, and polls it to a monotonic
        deadline. Any drain or completion failure restores every queue/status
        and reopens admissions before raising. Returns an immutable receipt for
        the physical resize.
        """
        snapshot = self._snapshot_state()
        self.admissions_paused = True

        try:
            drained_groups = self._drain_running_to_cpu()
            for group in drained_groups:
                if group not in self._swapped:
                    self._swapped.append(group)

            completion_event = self._record_resize_completion_event()
            if not self._wait_for_completion(completion_event, timeout_s):
                raise TimeoutError(
                    "CUDA completion events did not finish before the resize "
                    "quiescence deadline"
                )
        except BaseException:
            self._restore_state(snapshot)
            self.admissions_paused = False
            raise

        return ResizeReceipt(
            device_id=self._resize_device_id(),
            completion_events=(completion_event,),
            post_publish_event=None,
            admissions_paused=True,
        )

    def restore_after_kv_resize(self, receipt: object) -> None:
        """Swap eligible groups back in and reopen admissions.

        Groups that no longer fit remain SWAPPED. The maintenance backlog is
        merged back into the waiting queue in arrival order and the admission
        gate is cleared.
        """
        _ = receipt
        swapped_snapshot = list(self._swapped)
        self._recover_swapped_groups(swapped_snapshot)

        while self._maintenance_backlog:
            self._waiting.append(self._maintenance_backlog.popleft())

        self.admissions_paused = False

    def inject_swap_failure_after(self, count: int) -> None:
        self._swap_failure_after = count

    def snapshot_queue_ids(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(group.request_id for group in self._waiting),
            tuple(group.request_id for group in self._running),
            tuple(group.request_id for group in self._swapped),
        )

    def _record_resize_completion_event(self) -> object:
        return _AlwaysCompleteEvent()

    def _resize_device_id(self) -> int:
        device = getattr(self.kv_cache, "device", None)
        index = getattr(device, "index", None)
        return index if index is not None else 0

    def _drain_running_to_cpu(self) -> list[SequenceGroup]:
        drained: list[SequenceGroup] = []
        swap_count = 0
        for group in list(self._running):
            preempted = False
            for sequence in group.sequences:
                if sequence.status not in (
                    SequenceStatus.PREFILL,
                    SequenceStatus.DECODE,
                ):
                    continue
                if (
                    self._swap_failure_after is not None
                    and swap_count + 1 >= self._swap_failure_after
                ):
                    raise RuntimeError(
                        "swap drain failed while quiescing for KV resize"
                    )
                self.kv_cache.swap_out(sequence.seq_id)
                swap_count += 1
                self.kv_cache.free_gpu_blocks(sequence.seq_id)
                sequence.set_status(SequenceStatus.SWAPPED)
                preempted = True
            if preempted:
                self._running.remove(group)
                drained.append(group)
        return drained

    @staticmethod
    def _wait_for_completion(event: object, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        query = getattr(event, "query", None)
        if not callable(query):
            return True
        while True:
            if query():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def _snapshot_state(self) -> "_SchedulerStateSnapshot":
        return _SchedulerStateSnapshot(
            waiting=deque(self._waiting),
            running=deque(self._running),
            swapped=deque(self._swapped),
            statuses={
                sequence.seq_id: sequence.status
                for sequence in self._sequence_map.values()
            },
        )

    def _restore_state(self, snapshot: "_SchedulerStateSnapshot") -> None:
        self._waiting = deque(snapshot.waiting)
        self._running = deque(snapshot.running)
        self._swapped = deque(snapshot.swapped)
        for seq_id, status in snapshot.statuses.items():
            sequence = self._sequence_map.get(seq_id)
            if sequence is not None:
                sequence.status = status

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

    @staticmethod
    def _group_token_ids(group: SequenceGroup) -> list[int]:
        token_ids: list[int] = []
        for sequence in group.sequences:
            token_ids.extend(sequence.prompt_token_ids)
        return token_ids


RequestScheduler = Scheduler


__all__ = [
    "Deficit2D",
    "RequestScheduler",
    "Scheduler",
    "VerifyAdmission",
    "VerifyDemand",
    "admit_verify_demands",
]
