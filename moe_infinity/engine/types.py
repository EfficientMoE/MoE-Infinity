import time
from dataclasses import dataclass, field
from enum import Enum


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED_STOPPED = "FINISHED_STOPPED"
    FINISHED_LENGTH = "FINISHED_LENGTH"


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 256
    stop_sequences: tuple[str, ...] = ()


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.time)

    def transition_to(self, new_status: SequenceStatus) -> None:
        allowed_transitions = {
            SequenceStatus.WAITING: {
                SequenceStatus.RUNNING,
                SequenceStatus.FINISHED_STOPPED,
            },
            SequenceStatus.RUNNING: {
                SequenceStatus.SWAPPED,
                SequenceStatus.FINISHED_STOPPED,
                SequenceStatus.FINISHED_LENGTH,
            },
            SequenceStatus.SWAPPED: {
                SequenceStatus.WAITING,
                SequenceStatus.FINISHED_STOPPED,
            },
        }
        if new_status not in allowed_transitions.get(self.status, set()):
            raise ValueError(
                f"Invalid transition: {self.status} -> {new_status}"
            )
        self.status = new_status


@dataclass
class Sequence:
    seq_id: str
    request_id: str
    token_ids: list[int]
    num_computed_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)


@dataclass
class SchedulerOutput:
    scheduled_seqs: list[Sequence]
    preempted_seqs: list[Sequence]
    swapped_in_seqs: list[Sequence]
    num_batched_tokens: int
