from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional, Protocol, cast

import torch

from .batch import BatchBuilder, BatchMetadata, split_prefill_decode_batch
from .kv_cache import PagedKVCache
from .memory_manager import MemoryManager
from .model_runner import ModelRunner
from .sampler import Sampler
from .scheduler import Scheduler
from .sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)


class EvictionSyncAdapter(Protocol):
    def on_request_finished(self, request_id: str) -> None: ...

    def on_request_aborted(self, request_id: str) -> None: ...


_eviction_sync: Optional[EvictionSyncAdapter] = None


def set_eviction_sync(adapter: Optional[EvictionSyncAdapter]) -> None:
    global _eviction_sync
    _eviction_sync = adapter


@dataclass
class RequestOutput:
    request_id: str
    seq_id: int
    token_id: int
    token_text: Optional[str]
    finished: bool
    finish_reason: Optional[str] = None
    token_logprob: Optional[float] = None
    top_logprobs: Optional[dict[int, float]] = None
    usage: Optional[dict[str, int]] = None


class ContinuousBatchingEngine:
    model: object
    engine: object
    config: dict[str, object]
    tokenizer: Optional[object]
    device: torch.device
    dtype: torch.dtype
    eos_token_id: Optional[int]
    memory_manager: MemoryManager
    kv_cache: PagedKVCache
    scheduler: Scheduler
    model_runner: ModelRunner
    sampler: Sampler
    batch_builder: BatchBuilder
    _next_seq_id: int
    _num_steps: int
    _total_generated_tokens: int

    def __init__(
        self,
        model: object,
        engine: object,
        config: dict[str, object],
        tokenizer: Optional[object] = None,
    ) -> None:
        self.model = model
        self.engine = engine
        self.config = dict(config)
        self.tokenizer = tokenizer
        self.device = self._resolve_device(model)
        self.dtype = self._resolve_dtype(self.config["dtype"])
        self.eos_token_id = self._resolve_eos_token_id(model, self.config)

        self.memory_manager = MemoryManager(
            device=self.device,
            device_memory_ratio=self._get_float_config(
                "device_memory_ratio", 0.75
            ),
            kv_cache_ratio=self._get_float_config("kv_cache_ratio", 0.25),
        )
        _ = self.memory_manager.compute_budget(
            model_memory_bytes=self._resolve_model_memory_bytes(model)
        )

        block_size = self._get_int_config("block_size")
        num_layers = self._get_int_config("num_layers")
        num_kv_heads = self._get_int_config("num_kv_heads")
        head_dim = self._get_int_config("head_dim")
        num_blocks = self._resolve_num_blocks(
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        self.kv_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.scheduler = Scheduler(
            self.kv_cache,
            max_batch_size=self._get_int_config("max_batch_size", 32),
            max_tokens_per_step=self._get_int_config(
                "max_tokens_per_step", 2048
            ),
        )
        self.model_runner = ModelRunner(model, engine, device=self.device)
        self.sampler = Sampler()
        self.batch_builder = BatchBuilder()

        self._next_seq_id = 0
        self._sequences: dict[int, SequenceData] = {}
        self._sequence_to_request_id: dict[int, str] = {}
        self._request_to_seq_ids: dict[str, list[int]] = {}
        self._request_outputs: dict[str, dict[int, list[int]]] = {}
        self._callbacks: dict[str, list[Callable[[RequestOutput], None]]] = {}
        self._completed_request_ids: set[str] = set()
        self._cancelled_request_ids: set[str] = set()
        self._num_steps = 0
        self._total_generated_tokens = 0

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        on_token: Optional[Callable[[RequestOutput], None]] = None,
        n: int = 1,
    ) -> None:
        if request_id in self._request_to_seq_ids:
            raise ValueError(f"request_id '{request_id}' already exists")
        if n <= 0:
            raise ValueError("n must be greater than 0")

        sequences: list[SequenceData] = []
        seq_ids: list[int] = []
        output_map: dict[int, list[int]] = {}
        for _ in range(n):
            seq_id = self._next_seq_id
            self._next_seq_id += 1
            seq_ids.append(seq_id)
            sequence = SequenceData(
                seq_id=seq_id,
                prompt_token_ids=list(prompt_token_ids),
                sampling_params=sampling_params,
            )
            sequences.append(sequence)
            self._sequences[seq_id] = sequence
            self._sequence_to_request_id[seq_id] = request_id
            output_map[seq_id] = []

        group = SequenceGroup(request_id=request_id, sequences=sequences)

        self._request_to_seq_ids[request_id] = seq_ids
        self._request_outputs[request_id] = output_map
        if on_token is not None:
            self._callbacks.setdefault(request_id, []).append(on_token)

        self.scheduler.add_request(group)

    def step(self) -> list[RequestOutput]:
        scheduler_output = self.scheduler.schedule()
        if (
            not scheduler_output.prefill_seq_ids
            and not scheduler_output.decode_seq_ids
        ):
            return []

        batch = self.batch_builder.from_scheduler_output(
            scheduler_output,
            self._sequences,
            self.kv_cache,
        )
        if batch.total_tokens == 0:
            raise RuntimeError(
                "scheduler produced an empty batch; empty prompts are not supported"
            )

        logits = self._execute_batch(batch)
        last_token_logits = self._extract_last_token_logits(logits, batch)
        sampler_output = self.sampler.sample(
            last_token_logits,
            batch.sampling_params,
        )
        next_token_ids = sampler_output.token_ids

        outputs: list[RequestOutput] = []
        completed_seq_ids: list[int] = []
        new_decode_seq_ids: list[int] = []
        touched_request_ids: set[str] = set()

        for index, seq_id in enumerate(batch.seq_ids):
            sequence = self._sequences[seq_id]
            request_id = self._sequence_to_request_id[seq_id]
            touched_request_ids.add(request_id)
            token_id = int(next_token_ids[index].item())

            sequence.append_output_token(token_id)
            self._request_outputs[request_id][seq_id].append(token_id)
            self._total_generated_tokens += 1

            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            if finished:
                completed_seq_ids.append(seq_id)
            else:
                new_decode_seq_ids.append(seq_id)

            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    seq_id=seq_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    finish_reason=finish_reason,
                    token_logprob=(
                        sampler_output.token_logprobs[index]
                        if sampler_output.token_logprobs is not None
                        else None
                    ),
                    top_logprobs=(
                        sampler_output.top_logprobs[index]
                        if sampler_output.top_logprobs is not None
                        else None
                    ),
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )

        self.scheduler.update_after_step(
            completed_seq_ids=completed_seq_ids,
            new_decode_seq_ids=new_decode_seq_ids,
        )
        self._num_steps += 1

        finished_request_ids = {
            request_id
            for request_id in touched_request_ids
            if self._is_request_finished(request_id)
        }
        self._completed_request_ids.update(finished_request_ids)

        for output in outputs:
            for callback in self._callbacks.get(output.request_id, []):
                callback(output)

        for request_id in finished_request_ids:
            if _eviction_sync is not None:
                _eviction_sync.on_request_finished(request_id)
            _ = self._callbacks.pop(request_id, None)

        return outputs

    def run_until_done(self) -> dict[str, list[int] | list[list[int]]]:
        while self.has_pending_requests():
            outputs = self.step()
            if outputs:
                continue

            pending_request_ids = self._pending_request_ids()
            raise RuntimeError(
                f"engine made no progress with pending requests: {pending_request_ids}"
            )

        return {
            request_id: self._format_request_outputs(request_id)
            for request_id in self._request_outputs
        }

    def get_request_n_outputs(self, request_id: str) -> list[list[int]]:
        if request_id not in self._request_outputs:
            raise KeyError(f"unknown request_id '{request_id}'")

        return [
            list(self._request_outputs[request_id][seq_id])
            for seq_id in self._request_to_seq_ids.get(request_id, [])
        ]

    def abort_request(self, request_id: str) -> None:
        seq_ids = list(self._request_to_seq_ids.get(request_id, []))
        was_pending = any(
            self._sequences[seq_id].status
            in {
                SequenceStatus.WAITING,
                SequenceStatus.PREFILL,
                SequenceStatus.DECODE,
                SequenceStatus.SWAPPED,
            }
            for seq_id in seq_ids
            if seq_id in self._sequences
        )

        self.scheduler.abort_request(request_id)
        _ = self._callbacks.pop(request_id, None)

        if not was_pending:
            return

        if _eviction_sync is not None:
            _eviction_sync.on_request_aborted(request_id)

        self._cancelled_request_ids.add(request_id)
        _ = self._request_outputs.pop(request_id, None)
        _ = self._request_to_seq_ids.pop(request_id, None)

        for seq_id in seq_ids:
            _ = self._sequence_to_request_id.pop(seq_id, None)
            _ = self._sequences.pop(seq_id, None)

    def has_pending_requests(self) -> bool:
        return bool(self._pending_request_ids())

    def get_stats(self) -> dict[str, object]:
        status_counts = {status.value: 0 for status in SequenceStatus}
        for sequence in self._sequences.values():
            status_counts[sequence.status.value] += 1

        return {
            "pending_requests": len(self._pending_request_ids()),
            "completed_requests": len(self._completed_request_ids),
            "cancelled_requests": len(self._cancelled_request_ids),
            "num_steps": self._num_steps,
            "total_generated_tokens": self._total_generated_tokens,
            "kv_cache_num_blocks": self.kv_cache.num_blocks,
            "kv_cache_free_blocks": self.kv_cache.block_allocator.num_free_blocks,
            "sequence_status_counts": status_counts,
            "memory": self.memory_manager.report(),
        }

    def _resolve_num_blocks(
        self,
        *,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        explicit_num_blocks = self.config.get("num_kv_blocks")
        if explicit_num_blocks is not None:
            if isinstance(explicit_num_blocks, bool) or not isinstance(
                explicit_num_blocks, int
            ):
                raise ValueError(
                    "num_kv_blocks must be an integer when provided"
                )
            num_blocks = explicit_num_blocks
        else:
            num_blocks = self.memory_manager.get_max_kv_blocks(
                block_size=block_size,
                num_layers=num_layers,
                num_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=self.dtype,
            )

        if num_blocks > 0:
            return num_blocks
        return self._fallback_num_blocks(block_size)

    def _fallback_num_blocks(self, block_size: int) -> int:
        max_tokens_per_step = self._get_int_config("max_tokens_per_step", 1)
        return max(1, ceil(max_tokens_per_step / max(1, block_size)))

    def _execute_batch(self, batch: BatchMetadata) -> torch.Tensor:
        has_prefill = any(batch.is_prefill)
        has_decode = any(not p for p in batch.is_prefill)
        paged_classes_getter = getattr(
            self.model_runner,
            "_get_paged_attention_classes",
            None,
        )
        paged_classes: list[object] = []
        if callable(paged_classes_getter):
            maybe_paged_classes: object = paged_classes_getter()
            if isinstance(maybe_paged_classes, list):
                paged_classes = cast(list[object], maybe_paged_classes)
        uses_paged = bool(paged_classes)

        if not uses_paged or not (has_prefill and has_decode):
            return self.model_runner.execute(batch)

        split = split_prefill_decode_batch(batch)
        prefill_logits = None
        decode_logits = None
        if split.prefill_batch is not None:
            prefill_logits = self.model_runner.execute(split.prefill_batch)
        if split.decode_batch is not None:
            decode_logits = self.model_runner.execute(split.decode_batch)
        return split.recombine_outputs(prefill_logits, decode_logits)

    @staticmethod
    def _extract_last_token_logits(
        logits: torch.Tensor,
        batch: BatchMetadata,
    ) -> torch.Tensor:
        last_token_indices: list[int] = []
        for index, seq_length in enumerate(batch.seq_lengths):
            if seq_length <= 0:
                raise RuntimeError(
                    "scheduled sequence has no tokens; empty prompts are not supported"
                )
            last_token_indices.append(batch.token_offsets[index + 1] - 1)

        return logits[last_token_indices]

    def _get_finish_reason(
        self,
        sequence: SequenceData,
        token_id: int,
    ) -> Optional[str]:
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return "stop"
        if (
            len(sequence.output_token_ids)
            >= sequence.sampling_params.max_tokens
        ):
            return "length"

        if not sequence.sampling_params.stop:
            return None

        decoded_text = self._decode_tokens(sequence.output_token_ids)
        if decoded_text is None:
            return None

        if any(
            decoded_text.endswith(stop_text)
            for stop_text in sequence.sampling_params.stop
        ):
            return "stop"
        return None

    @staticmethod
    def _build_usage(sequence: SequenceData) -> dict[str, int]:
        prompt_tokens = sequence.prompt_length
        completion_tokens = len(sequence.output_token_ids)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _pending_request_ids(self) -> list[str]:
        pending_request_ids: list[str] = []
        active_statuses = {
            SequenceStatus.WAITING,
            SequenceStatus.PREFILL,
            SequenceStatus.DECODE,
            SequenceStatus.SWAPPED,
        }

        for request_id, seq_ids in self._request_to_seq_ids.items():
            if any(
                self._sequences[seq_id].status in active_statuses
                for seq_id in seq_ids
                if seq_id in self._sequences
            ):
                pending_request_ids.append(request_id)

        return pending_request_ids

    def _is_request_finished(self, request_id: str) -> bool:
        seq_ids = self._request_to_seq_ids.get(request_id, [])
        if not seq_ids:
            return False

        return all(
            self._sequences[seq_id].status is SequenceStatus.FINISHED
            for seq_id in seq_ids
            if seq_id in self._sequences
        )

    def _format_request_outputs(
        self, request_id: str
    ) -> list[int] | list[list[int]]:
        outputs = self.get_request_n_outputs(request_id)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def _decode_token(self, token_id: int) -> Optional[str]:
        return self._decode_tokens([token_id])

    def _decode_tokens(self, token_ids: list[int]) -> Optional[str]:
        if self.tokenizer is None:
            return None

        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return None

        try:
            decoded = decode(token_ids, skip_special_tokens=False)
        except TypeError:
            try:
                decoded = decode(token_ids)
            except Exception:
                return None
        except Exception:
            return None

        if isinstance(decoded, str):
            return decoded
        return None

    @staticmethod
    def _resolve_dtype(value: object) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value

        dtype_map = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(str(value).lower())
        if dtype is None:
            raise ValueError(f"unsupported dtype: {value}")
        return dtype

    @staticmethod
    def _resolve_eos_token_id(
        model: object,
        config: dict[str, object],
    ) -> Optional[int]:
        eos_token_id = config.get("eos_token_id")
        if isinstance(eos_token_id, int):
            return eos_token_id

        model_config = getattr(model, "config", None)
        model_eos_token_id = getattr(model_config, "eos_token_id", None)
        if isinstance(model_eos_token_id, int):
            return model_eos_token_id
        return None

    @staticmethod
    def _resolve_device(model: object) -> torch.device:
        model_device = getattr(model, "device", None)
        if (
            isinstance(model_device, torch.device)
            and model_device.type == "cuda"
        ):
            if not torch.cuda.is_available():
                return torch.device("cpu")
            return model_device

        parameters_fn = getattr(model, "parameters", None)
        if callable(parameters_fn):
            try:
                parameter_source = parameters_fn()
                if isinstance(parameter_source, Iterator):
                    first_param = next(parameter_source, None)
                elif isinstance(parameter_source, Iterable):
                    first_param = next(iter(parameter_source), None)
                else:
                    first_param = None
            except StopIteration:
                first_param = None
            except Exception:
                first_param = None

            if (
                isinstance(first_param, torch.Tensor)
                and first_param.device.type == "cuda"
            ):
                if not torch.cuda.is_available():
                    return torch.device("cpu")
                return first_param.device

        # model.device / first_param.device returned "cpu" — when CUDA is
        # available this means the model is managed by OffloadEngine which
        # moves tensors to GPU during forward (model_offload.py:1117-1130).
        if torch.cuda.is_available():
            last_gpu = torch.cuda.device_count() - 1
            return torch.device(f"cuda:{last_gpu}")
        return torch.device("cpu")

    @staticmethod
    def _resolve_model_memory_bytes(model: object) -> int:
        get_memory_footprint = getattr(model, "get_memory_footprint", None)
        if callable(get_memory_footprint):
            try:
                memory_bytes = get_memory_footprint()
            except Exception:
                memory_bytes = None
            if isinstance(memory_bytes, (int, float)):
                return max(0, int(memory_bytes))

        total_bytes = 0
        for attribute_name in ("parameters", "buffers"):
            iterator_factory = getattr(model, attribute_name, None)
            if not callable(iterator_factory):
                continue

            try:
                tensor_source = iterator_factory()
            except Exception:
                continue

            if isinstance(tensor_source, Iterator):
                iterator: Iterable[object] = tensor_source
            elif isinstance(tensor_source, Iterable):
                iterator = tensor_source
            else:
                continue

            try:
                for tensor in iterator:
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.numel() * tensor.element_size()
            except Exception:
                continue

        return max(0, int(total_bytes))

    def _get_float_config(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a float-compatible value")
        return float(value)

    def _get_int_config(self, key: str, default: Optional[int] = None) -> int:
        if default is None:
            if key not in self.config:
                raise KeyError(key)
            value = self.config[key]
        else:
            value = self.config.get(key, default)

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer value")
        return value


__all__ = [
    "ContinuousBatchingEngine",
    "RequestOutput",
    "set_eviction_sync",
]
