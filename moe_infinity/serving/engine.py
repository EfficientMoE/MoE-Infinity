from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional, Protocol, cast

import torch

from moe_infinity.memory.adaptive_memory import (
    AdaptiveMemoryConfig,
    AdaptiveMemoryController,
    MemorySignals,
    ResizeDirection,
)

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


class SpeculativeGenerator(Protocol):
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor: ...


_eviction_sync: Optional[EvictionSyncAdapter] = None

_VERIFY_ADMISSION_MAX_RETRIES = 1024


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
    speculative_draft: SpeculativeGenerator | None

    def __init__(
        self,
        model: object,
        engine: object,
        config: dict[str, object],
        tokenizer: Optional[object] = None,
        speculative_draft: SpeculativeGenerator | None = None,
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
        self._memory_budget = self.memory_manager.compute_budget(
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
            verify_token_budget=self._get_optional_int_config(
                "verify_token_budget"
            ),
            verify_expert_byte_budget=self._get_optional_int_config(
                "verify_expert_byte_budget"
            ),
            verify_token_deficit_cap=self._get_optional_int_config(
                "verify_token_deficit_cap"
            ),
            verify_expert_byte_deficit_cap=self._get_optional_int_config(
                "verify_expert_byte_deficit_cap"
            ),
        )
        self.model_runner = ModelRunner(model, engine, device=self.device)
        self.sampler = Sampler()
        self.batch_builder = BatchBuilder()
        self.speculative_draft = speculative_draft
        self._verify_scheduling_enabled = (
            self.scheduler.verify_scheduling_enabled
        )

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
        self._adaptive_tick_counter = 0
        self._adaptive_interval_steps = int(
            self.config.get("adaptive_memory_interval_steps", 64)
        )
        test_device_count = self.config.pop(
            "adaptive_memory_device_count_for_test", None
        )
        detected_devices = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        self._adaptive_device_count = int(
            test_device_count or detected_devices or 1
        )
        self._adaptive_kv_block_bytes = (
            2
            * block_size
            * num_layers
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )
        self._adaptive_targets: dict[int, tuple[int, int, bool]] = {}
        self._memory_resizers: dict[int, object] = {}
        self.memory_controller: AdaptiveMemoryController | None = None
        if bool(self.config.get("adaptive_memory_enabled", False)):
            self.memory_controller = AdaptiveMemoryController(
                self._adaptive_config_from_values()
            )
        for device_id in range(self._adaptive_device_count):
            kv_supported = device_id == int(self.device.index or 0)
            kv_blocks = self.kv_cache.num_blocks if kv_supported else 0
            expert_bytes = int(self._memory_budget.expert_cache_bytes)
            self._adaptive_targets[device_id] = (
                expert_bytes,
                kv_blocks,
                kv_supported,
            )
            if self.memory_controller is not None:
                self.memory_controller.observe(
                    MemorySignals(
                        device_id=device_id,
                        step=0,
                        expert_misses=0,
                        expert_accesses=0,
                        expert_fetch_stall_ms=0.0,
                        kv_used_blocks=0,
                        kv_total_blocks=kv_blocks,
                        kv_swap_bytes=0,
                        kv_swap_stall_ms=0.0,
                        kv_preemptions=0,
                        free_gpu_bytes=self._free_gpu_bytes(device_id),
                        kv_supported=kv_supported,
                    )
                )

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
        self._adaptive_tick_counter += 1
        if (
            self.memory_controller is not None
            and self._adaptive_tick_counter % self._adaptive_interval_steps == 0
        ):
            self._tick_adaptive_memory()
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

        if self._can_delegate_speculative(batch):
            if self._can_drive_verify_rounds():
                return self._step_speculative_session(batch)
            return self._step_speculative(batch)

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

    def _can_delegate_speculative(self, batch: BatchMetadata) -> bool:
        """Whether this fresh singleton request can use the proven sync loop.

        DFlash owns a separate ``DynamicCache`` here. The paged serving cache is
        used only for admission accounting and freed when the delegated request
        completes. Mixed batches, resumed decode rows, sampling, penalties, and
        logprob requests stay on the existing serving path unchanged.
        """
        if self.speculative_draft is None or len(batch.seq_ids) != 1:
            return False
        if batch.is_prefill != [True]:
            return False

        sequence = self._sequences[batch.seq_ids[0]]
        params = sequence.sampling_params
        return (
            not sequence.output_token_ids
            and not params.stop
            and params.max_tokens <= self.scheduler.max_tokens_per_step
            and params.temperature == 0
            and params.top_k <= 0
            and params.top_p >= 1.0
            and params.repetition_penalty == 1.0
            and params.logprobs <= 0
        )

    def _step_speculative(self, batch: BatchMetadata) -> list[RequestOutput]:
        """Complete one eligible request through DFlash's own DynamicCache.

        ``DFlashSpeculator.generate`` is the already GPU-proven greedy loop. A
        single serving ``step`` may therefore emit several accepted tokens;
        each is still recorded and streamed as an individual ``RequestOutput``.
        """
        speculator = self.speculative_draft
        if speculator is None:
            raise RuntimeError("speculative generator is not configured")

        seq_id = batch.seq_ids[0]
        sequence = self._sequences[seq_id]
        request_id = self._sequence_to_request_id[seq_id]
        prompt = torch.tensor([sequence.prompt_token_ids], dtype=torch.long)
        stop_token_ids = (
            [self.eos_token_id] if self.eos_token_id is not None else None
        )

        owner = cast(object | None, getattr(speculator, "moe", None))
        if owner is not None:
            setattr(owner, "_cached_past_key_values", None)
        try:
            generated = speculator.generate(
                prompt,
                max_new_tokens=sequence.sampling_params.max_tokens,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                top_k=sequence.sampling_params.top_k,
                top_p=sequence.sampling_params.top_p,
            )
        finally:
            if owner is not None:
                setattr(owner, "_cached_past_key_values", None)

        if generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError(
                "speculative generator must return token ids with shape [1, seq]"
            )
        prompt_len = sequence.prompt_length
        generated_ids = cast(
            list[int], generated[0, prompt_len:].to(device="cpu").tolist()
        )

        outputs: list[RequestOutput] = []
        for token_id in generated_ids:
            sequence.append_output_token(token_id)
            self._request_outputs[request_id][seq_id].append(token_id)
            self._total_generated_tokens += 1

            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    seq_id=seq_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    finish_reason=finish_reason,
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )
            if finished:
                break

        self.scheduler.update_after_step(
            completed_seq_ids=[seq_id],
            new_decode_seq_ids=[],
            committed_counts={seq_id: len(outputs)},
        )
        self._num_steps += 1
        self._completed_request_ids.add(request_id)

        for output in outputs:
            for callback in self._callbacks.get(request_id, []):
                callback(output)
        if _eviction_sync is not None:
            _eviction_sync.on_request_finished(request_id)
        _ = self._callbacks.pop(request_id, None)
        return outputs

    def _can_drive_verify_rounds(self) -> bool:
        """Opt-in gate for the Step-5 per-round DRAFT->VERIFY->DRAFT driver.

        Off by default: only when the operator configured verify budgets AND
        the speculator exposes the single-round ``SpecSession`` seam. Otherwise
        the delegated request keeps the proven whole-request ``generate()`` path
        unchanged, so default serving stays byte-for-byte identical.
        """
        speculator = self.speculative_draft
        return (
            self._verify_scheduling_enabled
            and speculator is not None
            and callable(getattr(speculator, "begin_session", None))
            and callable(getattr(speculator, "draft_round", None))
            and callable(getattr(speculator, "verify_round", None))
        )

    def _step_speculative_session(
        self, batch: BatchMetadata
    ) -> list[RequestOutput]:
        """Drive one eligible request through the scheduled single-round seam.

        Per round the engine drafts a block, registers that pending verify's
        EXACT token/byte demand with the 2-D verify scheduler
        (``set_verify_demand`` with the drafter-projected ``expert_nbytes`` sum,
        never a fabricated estimate), and runs the verify ONLY once the
        scheduler admits it (``verify_seq_ids``). The serving sequence advances
        PREFILL -> DRAFT -> (VERIFY -> DRAFT)* -> FINISHED; emitted tokens stream
        exactly as on the whole-request path. This runs only under the opt-in
        gate; ordinary serving and non-session speculators are untouched.
        """
        speculator = self.speculative_draft
        if speculator is None:
            raise RuntimeError("speculative generator is not configured")

        seq_id = batch.seq_ids[0]
        sequence = self._sequences[seq_id]
        request_id = self._sequence_to_request_id[seq_id]
        prompt = torch.tensor([sequence.prompt_token_ids], dtype=torch.long)
        stop_token_ids = (
            [self.eos_token_id] if self.eos_token_id is not None else None
        )
        params = sequence.sampling_params

        owner = cast(object | None, getattr(speculator, "moe", None))
        if owner is not None:
            setattr(owner, "_cached_past_key_values", None)

        outputs: list[RequestOutput] = []
        try:
            session = speculator.begin_session(
                prompt,
                max_new_tokens=params.max_tokens,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                top_k=params.top_k,
                top_p=params.top_p,
                collect_route_union=True,
            )
            sequence.set_status(SequenceStatus.DRAFT)

            streamed = 0
            outputs.extend(
                self._emit_session_tokens(
                    sequence, request_id, seq_id, session, streamed
                )
            )
            streamed = len(session.emitted)

            while not session.finished and not self._output_finished(outputs):
                draft = speculator.draft_round(session)
                self._admit_verify_round(seq_id, draft)
                sequence.set_status(SequenceStatus.VERIFY)
                speculator.verify_round(session)
                self.scheduler.clear_verify_demand(seq_id)
                if not session.finished:
                    sequence.set_status(SequenceStatus.DRAFT)

                outputs.extend(
                    self._emit_session_tokens(
                        sequence, request_id, seq_id, session, streamed
                    )
                )
                streamed = len(session.emitted)
                if self._output_finished(outputs):
                    break
        finally:
            if owner is not None:
                setattr(owner, "_cached_past_key_values", None)
            self.scheduler.clear_verify_demand(seq_id)

        if sequence.status in (
            SequenceStatus.DRAFT,
            SequenceStatus.VERIFY,
        ):
            sequence.set_status(SequenceStatus.FINISHED)

        self.scheduler.update_after_step(
            completed_seq_ids=[seq_id],
            new_decode_seq_ids=[],
            committed_counts={seq_id: len(outputs)},
        )
        self._num_steps += 1
        self._completed_request_ids.add(request_id)

        for output in outputs:
            for callback in self._callbacks.get(request_id, []):
                callback(output)
        if _eviction_sync is not None:
            _eviction_sync.on_request_finished(request_id)
        _ = self._callbacks.pop(request_id, None)
        return outputs

    def _admit_verify_round(self, seq_id: int, draft: object) -> None:
        """Register a pending verify's exact demand and gate on admission.

        Seats the demand as a new (non-in-flight) DRAFT round and lets
        ``Scheduler`` decide when the verify runs; unadmitted rounds carry their
        2-D deficit and are retried as the budget accrues. Raises when a
        correctly-shaped demand is never admitted (a budget misconfiguration:
        e.g. a zero budget in a dimension the demand needs).
        """
        tokens = int(getattr(draft, "tokens"))
        expert_bytes = int(getattr(draft, "expert_bytes"))
        self.scheduler.set_verify_demand(
            seq_id,
            tokens=tokens,
            expert_bytes=expert_bytes,
            in_flight=False,
        )
        for _ in range(_VERIFY_ADMISSION_MAX_RETRIES):
            output = self.scheduler.schedule()
            if seq_id in output.verify_seq_ids:
                return
        raise RuntimeError(
            "verify round for seq_id "
            f"{seq_id} was never admitted after "
            f"{_VERIFY_ADMISSION_MAX_RETRIES} scheduling passes; raise "
            "verify_token_budget / verify_expert_byte_budget to cover a single "
            f"verify (tokens={tokens}, expert_bytes={expert_bytes})"
        )

    def _emit_session_tokens(
        self,
        sequence: SequenceData,
        request_id: str,
        seq_id: int,
        session: object,
        streamed: int,
    ) -> list[RequestOutput]:
        """Stream the session's newly emitted tokens as ``RequestOutput`` rows."""
        emitted = cast(list[int], getattr(session, "emitted"))
        outputs: list[RequestOutput] = []
        for token_id in emitted[streamed:]:
            if self._output_finished(outputs):
                break
            sequence.append_output_token(token_id)
            self._request_outputs[request_id][seq_id].append(token_id)
            self._total_generated_tokens += 1

            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    seq_id=seq_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    finish_reason=finish_reason,
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )
        return outputs

    @staticmethod
    def _output_finished(outputs: list[RequestOutput]) -> bool:
        return bool(outputs) and outputs[-1].finished

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

        memory = self.memory_manager.report()
        adaptive_devices = (
            self.memory_controller.report()
            if self.memory_controller is not None
            else {}
        )
        for device_id, (
            expert_bytes,
            kv_blocks,
            _,
        ) in self._adaptive_targets.items():
            device = adaptive_devices.setdefault(device_id, {})
            device.setdefault("enabled", self.memory_controller is not None)
            device.setdefault("fallback_static", False)
            device.setdefault("fallback_reason", "")
            device.setdefault("resize_attempts", 0)
            device.setdefault("resize_failures", 0)
            device.setdefault("last_reason", "init")
            device.setdefault("hard_budget_violations", 0)
            device.setdefault("minimum_capacity_violations", 0)
            device.setdefault(
                "min_free_gpu_bytes", self._free_gpu_bytes(device_id)
            )
            adaptive_config = self._adaptive_config_from_values()
            device.setdefault(
                "configured_reserve_bytes",
                adaptive_config.free_memory_reserve_bytes,
            )
            device.setdefault("resize_count", int(device["resize_attempts"]))
            device.setdefault(
                "max_resize_count",
                self._adaptive_tick_counter // adaptive_config.cooldown_steps
                + 1,
            )
            if int(device.get("expert_target_bytes", 0)) == 0:
                device["expert_target_bytes"] = expert_bytes
            if int(device.get("kv_target_blocks", 0)) == 0:
                device["kv_target_blocks"] = kv_blocks
        memory["adaptive"] = {
            "enabled": self.memory_controller is not None,
            "devices": adaptive_devices,
            "completed": not self.has_pending_requests(),
            "failure_limit": self._adaptive_config_from_values().failure_limit,
        }
        return {
            "pending_requests": len(self._pending_request_ids()),
            "completed_requests": len(self._completed_request_ids),
            "cancelled_requests": len(self._cancelled_request_ids),
            "num_steps": self._num_steps,
            "total_generated_tokens": self._total_generated_tokens,
            "kv_cache_num_blocks": self.kv_cache.num_blocks,
            "kv_cache_free_blocks": self.kv_cache.block_allocator.num_free_blocks,
            "sequence_status_counts": status_counts,
            "memory": memory,
        }

    def _adaptive_config_from_values(self) -> AdaptiveMemoryConfig:
        defaults = AdaptiveMemoryConfig()
        return AdaptiveMemoryConfig(
            enabled=bool(self.config.get("adaptive_memory_enabled", False)),
            interval_steps=int(
                self.config.get(
                    "adaptive_memory_interval_steps", defaults.interval_steps
                )
            ),
            cooldown_steps=int(
                self.config.get(
                    "adaptive_memory_cooldown_steps", defaults.cooldown_steps
                )
            ),
            ewma_alpha=float(
                self.config.get(
                    "adaptive_memory_ewma_alpha", defaults.ewma_alpha
                )
            ),
            hysteresis_ratio=float(
                self.config.get(
                    "adaptive_memory_hysteresis_ratio",
                    defaults.hysteresis_ratio,
                )
            ),
            max_resize_step_bytes=int(
                self.config.get(
                    "adaptive_memory_max_resize_step_bytes",
                    defaults.max_resize_step_bytes,
                )
            ),
            min_expert_cache_bytes=int(
                self.config.get(
                    "adaptive_memory_min_expert_cache_bytes",
                    defaults.min_expert_cache_bytes,
                )
            ),
            min_kv_cache_blocks=int(
                self.config.get(
                    "adaptive_memory_min_kv_cache_blocks",
                    defaults.min_kv_cache_blocks,
                )
            ),
            free_memory_reserve_bytes=int(
                self.config.get(
                    "adaptive_memory_free_reserve_bytes",
                    defaults.free_memory_reserve_bytes,
                )
            ),
            failure_limit=int(
                self.config.get(
                    "adaptive_memory_failure_limit", defaults.failure_limit
                )
            ),
        )

    def _free_gpu_bytes(self, device_id: int) -> int:
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info(device_id)[0])
        return int(self.memory_manager.total_gpu_memory_bytes)

    def _tick_adaptive_memory(self) -> None:
        controller = self.memory_controller
        if controller is None:
            return
        used_blocks = (
            self.kv_cache.num_blocks
            - self.kv_cache.block_allocator.num_free_blocks
        )
        expert_snapshot = getattr(self.engine, "adaptive_memory_snapshot", None)
        snapshot = expert_snapshot() if callable(expert_snapshot) else {}
        for device_id, (
            expert_bytes,
            kv_blocks,
            kv_supported,
        ) in self._adaptive_targets.items():
            controller.observe(
                MemorySignals(
                    device_id=device_id,
                    step=self._adaptive_tick_counter,
                    expert_misses=int(snapshot.get("expert_misses", 0)),
                    expert_accesses=int(snapshot.get("expert_accesses", 0)),
                    expert_fetch_stall_ms=float(
                        snapshot.get("expert_fetch_stall_ms", 0.0)
                    ),
                    kv_used_blocks=used_blocks if kv_supported else 0,
                    kv_total_blocks=kv_blocks,
                    kv_swap_bytes=0,
                    kv_swap_stall_ms=0.0,
                    kv_preemptions=0,
                    free_gpu_bytes=self._free_gpu_bytes(device_id),
                    kv_supported=kv_supported,
                )
            )
            target = controller.propose(
                device_id=device_id,
                step=self._adaptive_tick_counter,
                total_bytes=self.memory_manager.total_gpu_memory_bytes,
                model_bytes=self._memory_budget.model_memory_bytes,
                activation_reserve_bytes=(
                    self.memory_manager.total_gpu_memory_bytes
                    - self._memory_budget.model_memory_bytes
                    - self._memory_budget.available_bytes
                ),
                kv_block_bytes=self._adaptive_kv_block_bytes,
                current_expert_bytes=expert_bytes,
                current_kv_blocks=kv_blocks,
                kv_supported=kv_supported,
            )
            if target.direction is ResizeDirection.HOLD:
                continue
            resizer = self._memory_resizers.get(device_id)
            if resizer is None:
                continue
            result = resizer.apply(
                device_id,
                target,
                current_expert_bytes=expert_bytes,
                current_kv_blocks=kv_blocks,
                kv_block_bytes=self._adaptive_kv_block_bytes,
            )
            controller.record_resize(result, step=self._adaptive_tick_counter)
            if result.committed:
                self._adaptive_targets[device_id] = (
                    result.expert_bytes,
                    result.kv_blocks,
                    result.kv_supported,
                )

    def get_config(self) -> dict[str, object]:
        config: dict[str, object] = {}
        for key, value in self.config.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                config[key] = value
            else:
                config[key] = str(value)
        return config

    def update_config(self, updates: dict[str, object]) -> dict[str, object]:
        applied: dict[str, object] = {}
        for key in ("max_batch_size", "max_tokens_per_step"):
            if key not in updates:
                continue
            value = updates[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer value")
            self.config[key] = value
            setattr(self.scheduler, key, value)
            applied[key] = value
        if "adaptive_memory_enabled" in updates:
            value = updates["adaptive_memory_enabled"]
            if not isinstance(value, bool):
                raise ValueError(
                    "adaptive_memory_enabled must be a boolean value"
                )
            current = self.memory_controller is not None
            if current and not value:
                self.restore_static_memory_targets(transactional=True)
            elif value and not current:
                self.config["adaptive_memory_enabled"] = True
                self.memory_controller = AdaptiveMemoryController(
                    self._adaptive_config_from_values()
                )
                for device_id, (
                    _,
                    kv_blocks,
                    kv_supported,
                ) in self._adaptive_targets.items():
                    self.memory_controller.observe(
                        MemorySignals(
                            device_id=device_id,
                            step=self._adaptive_tick_counter,
                            expert_misses=0,
                            expert_accesses=0,
                            expert_fetch_stall_ms=0.0,
                            kv_used_blocks=0,
                            kv_total_blocks=kv_blocks,
                            kv_swap_bytes=0,
                            kv_swap_stall_ms=0.0,
                            kv_preemptions=0,
                            free_gpu_bytes=self._free_gpu_bytes(device_id),
                            kv_supported=kv_supported,
                        )
                    )
            self.config["adaptive_memory_enabled"] = value
            applied["adaptive_memory_enabled"] = value
        return applied

    def restore_static_memory_targets(
        self, *, transactional: bool = True
    ) -> None:
        if not transactional:
            raise ValueError("static memory restoration must be transactional")
        for device_id in sorted(self._adaptive_targets):
            resizer = self._memory_resizers.get(device_id)
            restore = getattr(resizer, "restore_static_targets", None)
            if callable(restore):
                restore()
            elif self.memory_controller is not None:
                self.memory_controller.disable_to_static(
                    device_id, "static_restore_deferred"
                )
        self.memory_controller = None
        self.config["adaptive_memory_enabled"] = False

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

    def _get_optional_int_config(self, key: str) -> Optional[int]:
        if key not in self.config:
            return None
        value = self.config[key]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer value")
        return value

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
