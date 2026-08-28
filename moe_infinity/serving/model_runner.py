# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional, Protocol, runtime_checkable

import torch

from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.attention_types import (
    PagedBatchLengths,
)

from .batch import BatchMetadata


@runtime_checkable
class _ExpertTracerLike(Protocol):
    def create_entry(self) -> object: ...


@runtime_checkable
class _ExpertLayerModuleLike(Protocol):
    seq_id_list: list[object]


class ModelRunner:
    model: object
    engine: object
    device: torch.device
    seq_id_list: list[object]

    def __init__(
        self,
        model: object,
        engine: object,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.engine = engine
        self.device = self._resolve_device(device)
        self.seq_id_list = []

    def prepare_inputs(self, batch: BatchMetadata) -> dict[str, torch.Tensor]:
        num_seqs = len(batch.seq_ids)
        max_seq_len = max(batch.seq_lengths, default=0)

        input_ids = torch.zeros(
            (num_seqs, max_seq_len),
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for seq_idx in range(num_seqs):
            start = batch.token_offsets[seq_idx]
            end = batch.token_offsets[seq_idx + 1]
            seq_tokens = batch.input_token_ids[start:end]
            seq_len = batch.seq_lengths[seq_idx]

            if len(seq_tokens) != seq_len:
                raise ValueError(
                    "batch metadata is inconsistent: seq_lengths and token_offsets disagree"
                )
            if seq_len == 0:
                continue

            token_tensor = torch.tensor(
                seq_tokens,
                dtype=torch.long,
                device=self.device,
            )
            context_len = batch.context_lengths[seq_idx]
            position_tensor = torch.arange(
                context_len,
                context_len + seq_len,
                dtype=torch.long,
                device=self.device,
            )

            input_ids[seq_idx, :seq_len] = token_tensor
            position_ids[seq_idx, :seq_len] = position_tensor
            attention_mask[seq_idx, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def execute(
        self,
        batch: BatchMetadata,
        past_key_values: object = None,
    ) -> torch.Tensor:
        self._configure_expert_tracing(len(batch.seq_ids))
        self._advance_request_id()

        if batch.total_tokens == 0:
            return self._empty_logits()

        model_inputs = self.prepare_inputs(batch)

        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            _ = eval_fn()

        forward_kwargs: dict[str, object] = {
            **model_inputs,
            "use_cache": True,
        }
        if past_key_values is not None:
            forward_kwargs["past_key_values"] = past_key_values

        forward_fn = getattr(self.model, "forward", None)
        if not callable(forward_fn):
            raise ValueError("model must define callable forward()")

        paged_attention_classes = self._get_paged_attention_classes()
        backend = self._get_attention_backend()
        use_paged_context = bool(
            paged_attention_classes and backend is not None
        )

        with torch.no_grad():
            if not use_paged_context:
                outputs = forward_fn(**forward_kwargs)
            else:
                metadata = self._build_runtime_attention_metadata(batch)
                for attn_cls in paged_attention_classes:
                    attn_cls.set_paged_context(backend, metadata)
                try:
                    outputs = forward_fn(**forward_kwargs)
                finally:
                    for attn_cls in paged_attention_classes:
                        attn_cls.clear_paged_context()

        logits = self._extract_logits(outputs)
        if logits.dim() == 3:
            token_mask = model_inputs["attention_mask"].to(
                dtype=torch.bool, device=logits.device
            )
            logits = logits[token_mask]
        elif logits.dim() != 2:
            raise ValueError(
                f"model output logits must be rank-2 or rank-3, got rank-{logits.dim()}"
            )

        if logits.size(0) != batch.total_tokens:
            raise ValueError(
                f"packed logits row count must match batch.total_tokens; got {logits.size(0)} vs {batch.total_tokens}"
            )
        return logits

    def _configure_expert_tracing(self, num_sequences: int) -> None:
        tracer = getattr(self.engine, "expert_tracer", None)
        if not isinstance(tracer, _ExpertTracerLike) or num_sequences <= 0:
            self.seq_id_list = []
        else:
            self.seq_id_list = [
                tracer.create_entry() for _ in range(num_sequences)
            ]

        for module in getattr(self.engine, "expert_layer_modules", []):
            if isinstance(module, _ExpertLayerModuleLike):
                module.seq_id_list = self.seq_id_list

    def _advance_request_id(self) -> None:
        generator = getattr(self.engine, "_generate_request_id", None)
        if callable(generator):
            _ = generator()
            return

        request_id = getattr(self.engine, "request_id", None)
        if isinstance(request_id, int):
            setattr(self.engine, "request_id", request_id + 1)

    def _empty_logits(self) -> torch.Tensor:
        vocab_size = self._resolve_vocab_size()
        return torch.empty(
            (0, vocab_size), dtype=torch.float32, device=self.device
        )

    def _build_runtime_attention_metadata(
        self, batch: BatchMetadata
    ) -> RuntimeAttentionMetadata:
        block_size = self._resolve_block_size()
        max_blocks = max((len(row) for row in batch.block_tables), default=0)
        block_tables = torch.zeros(
            (len(batch.block_tables), max_blocks),
            dtype=torch.int32,
            device=self.device,
        )
        for row_idx, row in enumerate(batch.block_tables):
            if row:
                block_tables[row_idx, : len(row)] = torch.tensor(
                    row,
                    dtype=torch.int32,
                    device=self.device,
                )

        seq_lens_values = [
            context_len + seq_len
            for context_len, seq_len in zip(
                batch.context_lengths, batch.seq_lengths
            )
        ]
        seq_lens = torch.tensor(
            seq_lens_values,
            dtype=torch.int32,
            device=self.device,
        )

        slot_mapping = torch.tensor(
            self._build_slot_mapping(batch, block_size),
            dtype=torch.int64,
            device=self.device,
        )

        num_prefill_tokens = sum(
            seq_len
            for seq_len, is_prefill in zip(batch.seq_lengths, batch.is_prefill)
            if is_prefill
        )
        num_decode_tokens = batch.total_tokens - num_prefill_tokens

        lengths: PagedBatchLengths | None = None
        if batch.seq_ids and all(length > 0 for length in batch.seq_lengths):
            lengths = PagedBatchLengths(
                query_lengths=list(batch.seq_lengths),
                query_offsets=list(batch.token_offsets),
                context_lengths=list(batch.context_lengths),
                kv_seq_lengths=list(seq_lens_values),
            )

        return RuntimeAttentionMetadata(
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max(seq_lens_values, default=0),
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            is_prefill=bool(batch.is_prefill and all(batch.is_prefill)),
            lengths=lengths,
        )

    @staticmethod
    def _build_slot_mapping(batch: BatchMetadata, block_size: int) -> list[int]:
        slots: list[int] = []
        for seq_idx, seq_len in enumerate(batch.seq_lengths):
            block_table = batch.block_tables[seq_idx]
            context_len = batch.context_lengths[seq_idx]
            for token_idx in range(seq_len):
                token_pos = context_len + token_idx
                block_idx = token_pos // block_size
                token_offset = token_pos % block_size
                if block_idx >= len(block_table):
                    raise ValueError(
                        "batch metadata block table is too short for sequence length"
                    )
                slots.append(block_table[block_idx] * block_size + token_offset)
        return slots

    def _resolve_block_size(self) -> int:
        kv_cache = getattr(self.engine, "kv_cache", None)
        block_size = getattr(kv_cache, "block_size", None)
        if isinstance(block_size, int) and block_size > 0:
            return block_size

        backend = self._get_attention_backend()
        spec = getattr(backend, "spec", None)
        spec_block_size = getattr(spec, "block_size", None)
        if isinstance(spec_block_size, int) and spec_block_size > 0:
            return spec_block_size

        return 1

    def _get_attention_backend(self) -> object | None:
        getter = getattr(self.engine, "get_attention_backend", None)
        if callable(getter):
            backend = getter()
            if backend is not None:
                return backend

        for attr_name in (
            "attention_backend",
            "_attention_backend",
            "_native_attention_backend",
        ):
            backend = getattr(self.engine, attr_name, None)
            if backend is not None:
                return backend
        return None

    def _get_paged_attention_classes(self) -> list[type[Any]]:
        paged_class_names = {
            "DeepseekV2PagedAttention",
            "DeepseekV3PagedAttention",
            "Qwen3PagedAttention",
        }
        classes: list[type[Any]] = []
        seen: set[type[Any]] = set()

        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return classes

        modules = modules_fn()
        if not isinstance(modules, Iterable):
            return classes

        for module in modules:
            cls = module.__class__
            if cls in seen:
                continue
            if cls.__name__ not in paged_class_names:
                continue
            if not hasattr(cls, "set_paged_context") or not hasattr(
                cls, "clear_paged_context"
            ):
                continue
            seen.add(cls)
            classes.append(cls)

        return classes

    def _get_qwen3_paged_attention_modules(self) -> list[Any]:
        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return []
        modules = modules_fn()
        if not isinstance(modules, Iterable):
            return []
        found: list[Any] = []
        for module in modules:
            if module.__class__.__name__ == "Qwen3PagedAttention" and hasattr(
                module, "layer_idx"
            ):
                found.append(module)
        return found

    def get_attention_backend(self) -> object | None:
        return self._get_attention_backend()

    def supports_chunked_prefill(self) -> bool:
        from moe_infinity.runtime.attention_backend import (
            LayeredPagedKVStore,
            PagedAttentionBackend,
        )

        backend = self._get_attention_backend()
        qwen_modules = self._get_qwen3_paged_attention_modules()
        expected_layers = int(
            getattr(self.model.config, "num_hidden_layers", 0)
        )
        layer_indices = [int(module.layer_idx) for module in qwen_modules]
        return bool(
            qwen_modules
            and sorted(layer_indices) == list(range(expected_layers))
            and len(set(layer_indices)) == len(layer_indices)
            and isinstance(backend, PagedAttentionBackend)
            and isinstance(backend.block_store, LayeredPagedKVStore)
            and backend.supports_chunked_prefill()
        )

    def chunked_prefill_unavailable_reason(self) -> str:
        from moe_infinity.runtime.attention_backend import (
            LayeredPagedKVStore,
            PagedAttentionBackend,
        )

        qwen_modules = self._get_qwen3_paged_attention_modules()
        expected_layers = int(
            getattr(self.model.config, "num_hidden_layers", 0)
        )
        indices = [int(module.layer_idx) for module in qwen_modules]
        if not qwen_modules or sorted(indices) != list(range(expected_layers)):
            return "incomplete_qwen3_paged_layer_registry"
        backend = self._get_attention_backend()
        if not isinstance(backend, PagedAttentionBackend) or not isinstance(
            backend.block_store, LayeredPagedKVStore
        ):
            return "layered_paged_kv_store_unavailable"
        if not backend.supports_chunked_prefill():
            return "paged_backend_lacks_chunk_history"
        return "none"

    def _resolve_vocab_size(self) -> int:
        config = getattr(self.model, "config", None)
        if config is not None:
            config_vocab_size = getattr(config, "vocab_size", None)
            if isinstance(config_vocab_size, int):
                return config_vocab_size

        vocab_size = getattr(self.model, "vocab_size", None)
        if isinstance(vocab_size, int):
            return vocab_size

        raise ValueError("unable to infer vocab_size from model")

    @staticmethod
    def _extract_logits(outputs: object) -> torch.Tensor:
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, Mapping):
            as_dict = dict(outputs)
            logits = as_dict.get("logits")
        if logits is None and isinstance(outputs, tuple) and outputs:
            logits = outputs[0]

        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "model forward output does not include tensor logits"
            )
        return logits

    @staticmethod
    def _resolve_device(device: Optional[torch.device]) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device


__all__ = ["ModelRunner"]
