# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Protocol, runtime_checkable

import torch

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

        with torch.no_grad():
            outputs = forward_fn(**forward_kwargs)

        logits = self._extract_logits(outputs)
        if logits.dim() == 3:
            token_mask = model_inputs["attention_mask"].to(dtype=torch.bool)
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
