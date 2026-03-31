# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field

from .kv_cache import PagedKVCache
from .sequence import SamplingParams, SequenceData


@dataclass
class SchedulerOutput:
    prefill_seq_ids: list[int] = field(default_factory=list)
    decode_seq_ids: list[int] = field(default_factory=list)
    preempted_seq_ids: list[int] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    def __post_init__(self) -> None:
        self.prefill_seq_ids = list(self.prefill_seq_ids)
        self.decode_seq_ids = list(self.decode_seq_ids)
        self.preempted_seq_ids = list(self.preempted_seq_ids)


@dataclass
class BatchMetadata:
    seq_ids: list[int]
    input_token_ids: list[int]
    seq_lengths: list[int]
    context_lengths: list[int]
    is_prefill: list[bool]
    block_tables: list[list[int]]
    token_offsets: list[int]
    sampling_params: list[SamplingParams]

    def __post_init__(self) -> None:
        expected = len(self.seq_ids)
        for field_name, value in (
            ("seq_lengths", self.seq_lengths),
            ("context_lengths", self.context_lengths),
            ("is_prefill", self.is_prefill),
            ("block_tables", self.block_tables),
            ("sampling_params", self.sampling_params),
        ):
            if len(value) != expected:
                raise ValueError(
                    f"{field_name} must have length {expected}, got {len(value)}"
                )

        if len(self.token_offsets) != expected + 1:
            raise ValueError(
                f"token_offsets must have length {expected + 1}, got {len(self.token_offsets)}"
            )
        if self.token_offsets[:1] != [0]:
            raise ValueError("token_offsets must start at 0")

        running_total = 0
        for idx, length in enumerate(self.seq_lengths):
            running_total += length
            if self.token_offsets[idx + 1] != running_total:
                raise ValueError(
                    "token_offsets must be a cumulative sum of seq_lengths"
                )

        if running_total != len(self.input_token_ids):
            raise ValueError(
                "seq_lengths must sum to the number of packed input tokens"
            )

    @property
    def total_tokens(self) -> int:
        return len(self.input_token_ids)


class BatchBuilder:
    @staticmethod
    def from_scheduler_output(
        scheduler_output: SchedulerOutput,
        sequences: dict[int, SequenceData],
        kv_cache: PagedKVCache,
    ) -> BatchMetadata:
        seq_ids = [
            *scheduler_output.prefill_seq_ids,
            *scheduler_output.decode_seq_ids,
        ]

        input_token_ids: list[int] = []
        seq_lengths: list[int] = []
        context_lengths: list[int] = []
        is_prefill: list[bool] = []
        block_tables: list[list[int]] = []
        sampling_params: list[SamplingParams] = []

        for seq_id in scheduler_output.prefill_seq_ids:
            sequence = sequences[seq_id]
            tokens = sequence.prompt_token_ids[sequence.num_computed_tokens :]
            input_token_ids.extend(tokens)
            seq_lengths.append(len(tokens))
            context_lengths.append(sequence.num_computed_tokens)
            is_prefill.append(True)
            block_tables.append(kv_cache.get_block_table(seq_id))
            sampling_params.append(sequence.sampling_params)

        for seq_id in scheduler_output.decode_seq_ids:
            sequence = sequences[seq_id]
            token = (
                sequence.total_token_ids[-1:]
                if sequence.total_token_ids
                else []
            )
            input_token_ids.extend(token)
            seq_lengths.append(len(token))
            context_lengths.append(sequence.num_computed_tokens)
            is_prefill.append(False)
            block_tables.append(kv_cache.get_block_table(seq_id))
            sampling_params.append(sequence.sampling_params)

        token_offsets = [0]
        for length in seq_lengths:
            token_offsets.append(token_offsets[-1] + length)

        return BatchMetadata(
            seq_ids=seq_ids,
            input_token_ids=input_token_ids,
            seq_lengths=seq_lengths,
            context_lengths=context_lengths,
            is_prefill=is_prefill,
            block_tables=block_tables,
            token_offsets=token_offsets,
            sampling_params=sampling_params,
        )


__all__ = ["BatchBuilder", "BatchMetadata", "SchedulerOutput"]
