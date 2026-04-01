from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.sequence import SamplingParams

"""
Phase Transition GPU State Audit — Task 11

Discovery Results (2026-04-01):
- Investigated: serving/model_runner.py, serving/engine.py, serving/batch.py,
  serving/scheduler.py (transition in update_after_step), and runtime/*.
- Findings:
  - serving/model_runner.py allocates input tensors as locals inside execute();
    no decode-phase tensor is retained as ModelRunner instance state.
  - serving/engine.py keeps scheduling/sequence metadata and a persistent
    PagedKVCache tensor, but no per-step decode tensor buffers are persisted
    across execute() calls.
  - serving/batch.py builds Python lists/metadata only (no tensor state on
    BatchBuilder instance).
  - serving/scheduler.py transition PREFILL->DECODE appends KV-cache tokens and
    frees sequence KV blocks on completion; no decode temporary tensor attrs.
  - runtime/model_offload.py contains _captured_kv state, but it is gated by
    enable_kv_cache_offload + kv_cache_manager and is not exercised by
    ContinuousBatchingEngine.step() in current serving path.
- Verdict: NO LEAK DETECTED in serving phase transitions for decode-step tensors.
- Action Taken: Added stability tests that exercise prefill->decode->prefill and
  assert tensor-state invariants + block reuse stability without adding cleanup.
"""


@dataclass
class _MockConfig:
    vocab_size: int = 64
    eos_token_id: int = 10_000


class _MockModel:
    config: _MockConfig
    device: torch.device

    def __init__(self) -> None:
        self.config = _MockConfig()
        self.device = torch.device("cpu")

    def eval(self) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.config.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row_idx in range(batch_size):
            for token_idx in range(seq_len):
                token_id = int(input_ids[row_idx, token_idx].item())
                next_token = (token_id + 1) % self.config.vocab_size
                logits[row_idx, token_idx, next_token] = 0.0
        return SimpleNamespace(logits=logits)


class _MockExpertTracer:
    _next: int

    def __init__(self) -> None:
        self._next = 0

    def create_entry(self) -> int:
        value = self._next
        self._next += 1
        return value


class _MockOffloadEngine:
    request_id: int
    expert_tracer: _MockExpertTracer
    expert_layer_modules: list[SimpleNamespace]

    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        value = self.request_id
        self.request_id += 1
        return value


def _make_engine() -> ContinuousBatchingEngine:
    with (
        patch(
            "moe_infinity.serving.engine.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "moe_infinity.serving.model_runner.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "moe_infinity.serving.kv_cache.torch.cuda.is_available",
            return_value=False,
        ),
    ):
        return ContinuousBatchingEngine(
            model=_MockModel(),
            engine=_MockOffloadEngine(),
            config={
                "dtype": "float16",
                "device_memory_ratio": 0.75,
                "kv_cache_ratio": 0.25,
                "block_size": 4,
                "num_layers": 1,
                "num_kv_heads": 2,
                "head_dim": 8,
                "max_batch_size": 8,
                "max_tokens_per_step": 16,
                "eos_token_id": 10_000,
                "num_kv_blocks": 8,
            },
            tokenizer=None,
        )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return any(_contains_tensor(item) for item in mapping.values())
    if isinstance(value, list):
        values = cast(list[object], value)
        return any(_contains_tensor(item) for item in values)
    if isinstance(value, tuple):
        values = cast(tuple[object, ...], value)
        return any(_contains_tensor(item) for item in values)
    if isinstance(value, set):
        values = cast(set[object], value)
        return any(_contains_tensor(item) for item in values)
    return False


def _tensor_attr_names(obj: object) -> set[str]:
    attrs_obj = getattr(obj, "__dict__", None)
    if not isinstance(attrs_obj, dict):
        return set()
    attrs = cast(dict[str, object], attrs_obj)
    return {name for name, value in attrs.items() if _contains_tensor(value)}


def _snapshot_tensor_attrs(
    engine: ContinuousBatchingEngine,
) -> dict[str, set[str]]:
    return {
        "engine": _tensor_attr_names(engine),
        "model_runner": _tensor_attr_names(engine.model_runner),
        "scheduler": _tensor_attr_names(engine.scheduler),
        "batch_builder": _tensor_attr_names(engine.batch_builder),
        "kv_cache": _tensor_attr_names(engine.kv_cache),
    }


def test_phase_transition_memory_stability() -> None:
    engine = _make_engine()
    baseline_free_blocks = engine.kv_cache.block_allocator.num_free_blocks
    baseline_kv_ptr = engine.kv_cache.get_kv_cache_tensors().data_ptr()
    baseline_tensor_attrs = _snapshot_tensor_attrs(engine)

    engine.add_request(
        request_id="req-prefill-decode",
        prompt_token_ids=[7],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
    )

    first_step = engine.step()
    assert len(first_step) == 1
    assert first_step[0].finished is False
    running_seq_ids = engine.scheduler.get_running_seq_ids()
    assert len(running_seq_ids) == 1
    assert engine.has_pending_requests() is True

    second_step = engine.step()
    assert len(second_step) == 1
    assert second_step[0].finished is True
    assert (
        engine.kv_cache.block_allocator.num_free_blocks == baseline_free_blocks
    )

    engine.add_request(
        request_id="req-second-prefill",
        prompt_token_ids=[11],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )
    third_step = engine.step()
    assert len(third_step) == 1
    assert third_step[0].finished is True

    assert (
        engine.kv_cache.block_allocator.num_free_blocks == baseline_free_blocks
    )
    assert engine.kv_cache.get_kv_cache_tensors().data_ptr() == baseline_kv_ptr
    assert _snapshot_tensor_attrs(engine) == baseline_tensor_attrs


def test_cleanup_is_idempotent_if_applicable() -> None:
    engine = _make_engine()

    cleanup_methods: list[Callable[[], object]] = []
    for owner in (engine, engine.model_runner, engine.engine):
        method = getattr(owner, "_cleanup_decode_gpu_state", None)
        if callable(method):
            cleanup_methods.append(cast(Callable[[], object], method))

    if not cleanup_methods:
        assert _tensor_attr_names(engine.model_runner) == set()
        return

    for method in cleanup_methods:
        _ = method()
        _ = method()
