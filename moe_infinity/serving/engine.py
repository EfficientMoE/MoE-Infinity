from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional

import torch

from .batch import BatchBuilder, BatchMetadata
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


@dataclass
class RequestOutput:
    request_id: str
    token_id: int
    token_text: Optional[str]
    finished: bool
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
        self._request_outputs: dict[str, list[int]] = {}
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
    ) -> None:
        if request_id in self._request_to_seq_ids:
            raise ValueError(f"request_id '{request_id}' already exists")

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        sequence = SequenceData(
            seq_id=seq_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
        )
        group = SequenceGroup(request_id=request_id, sequences=[sequence])

        self._sequences[seq_id] = sequence
        self._sequence_to_request_id[seq_id] = request_id
        self._request_to_seq_ids[request_id] = [seq_id]
        self._request_outputs[request_id] = []
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

        logits = self.model_runner.execute(batch)
        last_token_logits = self._extract_last_token_logits(logits, batch)
        next_token_ids = self.sampler.sample(
            last_token_logits,
            batch.sampling_params,
        )

        outputs: list[RequestOutput] = []
        completed_seq_ids: list[int] = []
        new_decode_seq_ids: list[int] = []

        for index, seq_id in enumerate(batch.seq_ids):
            sequence = self._sequences[seq_id]
            request_id = self._sequence_to_request_id[seq_id]
            token_id = int(next_token_ids[index].item())

            sequence.append_output_token(token_id)
            self._request_outputs[request_id].append(token_id)
            self._total_generated_tokens += 1

            finished = self._should_finish_sequence(sequence, token_id)
            if finished:
                completed_seq_ids.append(seq_id)
                self._completed_request_ids.add(request_id)
            else:
                new_decode_seq_ids.append(seq_id)

            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )

        self.scheduler.update_after_step(
            completed_seq_ids=completed_seq_ids,
            new_decode_seq_ids=new_decode_seq_ids,
        )
        self._num_steps += 1

        for output in outputs:
            for callback in self._callbacks.get(output.request_id, []):
                callback(output)
            if output.finished:
                _ = self._callbacks.pop(output.request_id, None)

        return outputs

    def run_until_done(self) -> dict[str, list[int]]:
        while self.has_pending_requests():
            outputs = self.step()
            if outputs:
                continue

            pending_request_ids = self._pending_request_ids()
            raise RuntimeError(
                f"engine made no progress with pending requests: {pending_request_ids}"
            )

        return {
            request_id: list(output_token_ids)
            for request_id, output_token_ids in self._request_outputs.items()
        }

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

    def _should_finish_sequence(
        self,
        sequence: SequenceData,
        token_id: int,
    ) -> bool:
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return True
        if (
            len(sequence.output_token_ids)
            >= sequence.sampling_params.max_tokens
        ):
            return True

        if not sequence.sampling_params.stop:
            return False

        decoded_text = self._decode_tokens(sequence.output_token_ids)
        if decoded_text is None:
            return False

        return any(
            decoded_text.endswith(stop_text)
            for stop_text in sequence.sampling_params.stop
        )

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


__all__ = ["ContinuousBatchingEngine", "RequestOutput"]
