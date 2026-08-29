import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_ = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_ = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_ = _load_module(
    "moe_infinity.serving.memory_manager",
    ROOT / "moe_infinity" / "serving" / "memory_manager.py",
)
_ = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)
_ = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)
_ = _load_module(
    "moe_infinity.serving.sampler",
    ROOT / "moe_infinity" / "serving" / "sampler.py",
)
_ = _load_module(
    "moe_infinity.serving.engine",
    ROOT / "moe_infinity" / "serving" / "engine.py",
)

from moe_infinity.memory.adaptive_memory import ResizeOutcome, ResizeResult
from moe_infinity.serving.engine import (  # type: ignore[reportMissingImports]
    ContinuousBatchingEngine,
    RequestOutput,
)
from moe_infinity.serving.sequence import (  # type: ignore[reportMissingImports]
    SamplingParams,
)


@dataclass
class MockConfig:
    vocab_size: int = 100
    eos_token_id: int = 2


class MockModel:
    config: MockConfig
    eval_called: bool

    def __init__(
        self,
        vocab_size: int = 100,
        eos_token_id: int = 2,
    ) -> None:
        self.config = MockConfig(
            vocab_size=vocab_size, eos_token_id=eos_token_id
        )
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        _ = kwargs
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.config.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
        )

        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                token_id = int(input_ids[batch_idx, token_idx].item())
                next_token_id = (token_id + 1) % self.config.vocab_size
                logits[batch_idx, token_idx, next_token_id] = 0.0

        return types.SimpleNamespace(logits=logits)


class _MockExpertTracer:
    _next_entry: int

    def __init__(self) -> None:
        self._next_entry = 0

    def create_entry(self) -> int:
        entry = self._next_entry
        self._next_entry += 1
        return entry


class MockOffloadEngine:
    request_id: int
    expert_tracer: _MockExpertTracer
    expert_layer_modules: list[types.SimpleNamespace]

    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        request_id = self.request_id
        self.request_id += 1
        return request_id


class MockTokenizer:
    def decode(
        self,
        token_ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
    ) -> str:
        _ = skip_special_tokens
        if isinstance(token_ids, int):
            ids = [token_ids]
        else:
            ids = list(token_ids)
        return "".join(f"tok-{token_id}" for token_id in ids)


class MockSpeculator:
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        _ = (temperature, stop_token_ids, top_k, top_p)
        self.calls += 1
        last_token = int(input_ids[0, -1].item())
        continuation = torch.arange(
            last_token + 1,
            last_token + max_new_tokens + 1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(0)
        return torch.cat([input_ids, continuation], dim=1)


def _make_config() -> dict[str, object]:
    return {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 8,
        "max_tokens_per_step": 16,
        "block_size": 4,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float16",
        "eos_token_id": 2,
    }


def _make_engine(
    tokenizer: Optional[object] = None,
    *,
    adaptive: bool = False,
    interval_steps: int = 64,
    device_count: int = 1,
) -> ContinuousBatchingEngine:
    config = _make_config()
    config.update(
        {
            "adaptive_memory_enabled": adaptive,
            "adaptive_memory_interval_steps": interval_steps,
            "adaptive_memory_device_count_for_test": device_count,
        }
    )
    return ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=config,
        tokenizer=tokenizer,
    )


def test_engine_ticks_controller_only_at_safe_interval() -> None:
    engine = _make_engine(adaptive=True, interval_steps=4)
    engine.memory_controller = Mock()
    for _ in range(3):
        engine.step()
    engine.memory_controller.propose.assert_not_called()
    engine.step()
    engine.memory_controller.propose.assert_called_once()


def test_stats_expose_last_committed_split_and_failures() -> None:
    engine = _make_engine(adaptive=True, device_count=2)
    memory = engine.get_stats()["memory"]["adaptive"]
    assert set(memory["devices"]) == {0, 1}
    assert {
        "enabled",
        "fallback_static",
        "expert_target_bytes",
        "kv_target_blocks",
        "resize_attempts",
        "resize_failures",
        "last_reason",
    }.issubset(memory["devices"][0])


def test_failure_on_one_device_does_not_latch_other_device() -> None:
    engine = _make_engine(adaptive=True, device_count=2)
    assert engine.memory_controller is not None
    engine.memory_controller.record_resize(
        ResizeResult(0, ResizeOutcome.REJECTED, 512, 8, "pinned"), step=64
    )
    stats = engine.get_stats()["memory"]["adaptive"]["devices"]
    assert stats[0]["resize_failures"] == 1
    assert stats[1]["resize_failures"] == 0


def test_engine_single_request_run_until_done() -> None:
    engine = _make_engine(tokenizer=MockTokenizer())
    callback_outputs: list[RequestOutput] = []

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
        on_token=callback_outputs.append,
    )

    outputs = engine.run_until_done()

    assert outputs == {"req-1": [11, 12, 13]}
    assert [output.token_id for output in callback_outputs] == [11, 12, 13]
    assert [output.token_text for output in callback_outputs] == [
        "tok-11",
        "tok-12",
        "tok-13",
    ]
    assert callback_outputs[-1].finished is True
    assert callback_outputs[-1].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 3,
        "total_tokens": 4,
    }
    assert engine.has_pending_requests() is False


def test_engine_delegates_single_greedy_request_to_speculator() -> None:
    speculator = MockSpeculator()
    engine = ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=_make_config(),
        tokenizer=MockTokenizer(),
        speculative_draft=speculator,
    )
    callback_outputs: list[RequestOutput] = []
    engine.add_request(
        request_id="req-spec",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
        on_token=callback_outputs.append,
    )

    step_outputs = engine.step()

    assert speculator.calls == 1
    assert [output.token_id for output in step_outputs] == [11, 12, 13]
    assert [output.token_id for output in callback_outputs] == [11, 12, 13]
    assert step_outputs[-1].finished is True
    assert step_outputs[-1].finish_reason == "length"
    assert engine.get_request_n_outputs("req-spec") == [[11, 12, 13]]
    assert engine.has_pending_requests() is False


def test_engine_does_not_delegate_sampled_request_to_speculator() -> None:
    speculator = MockSpeculator()
    engine = ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=_make_config(),
        speculative_draft=speculator,
    )
    engine.add_request(
        request_id="req-sampled",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=1.0, max_tokens=1),
    )

    _ = engine.run_until_done()

    assert speculator.calls == 0


def test_engine_does_not_delegate_stop_string_request() -> None:
    speculator = MockSpeculator()
    engine = ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=_make_config(),
        speculative_draft=speculator,
    )
    engine.add_request(
        request_id="req-stop",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=1,
            stop=["tok-11"],
        ),
    )

    _ = engine.run_until_done()

    assert speculator.calls == 0


def test_engine_does_not_delegate_past_step_token_budget() -> None:
    speculator = MockSpeculator()
    config = _make_config()
    config["max_tokens_per_step"] = 2
    engine = ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=config,
        speculative_draft=speculator,
    )
    engine.add_request(
        request_id="req-large",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )

    _ = engine.run_until_done()

    assert speculator.calls == 0


def test_engine_multiple_requests() -> None:
    engine = _make_engine()

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
    )
    engine.add_request(
        request_id="req-2",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=5),
    )
    engine.add_request(
        request_id="req-3",
        prompt_token_ids=[20],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )

    outputs = engine.run_until_done()

    assert outputs == {
        "req-1": [11, 12],
        "req-2": [2],
        "req-3": [21],
    }
    assert engine.get_stats()["pending_requests"] == 0


def test_engine_add_request_after_running() -> None:
    engine = _make_engine()

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )

    first_step_outputs = engine.step()

    assert [output.token_id for output in first_step_outputs] == [11]
    assert first_step_outputs[0].finished is False
    assert engine.has_pending_requests() is True

    engine.add_request(
        request_id="req-2",
        prompt_token_ids=[40],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
    )

    outputs = engine.run_until_done()

    assert outputs == {
        "req-1": [11, 12, 13],
        "req-2": [41, 42],
    }


def test_engine_abort_request() -> None:
    engine = _make_engine()

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )
    _ = engine.step()

    engine.abort_request("req-1")

    assert engine.has_pending_requests() is False
    assert engine.get_stats()["cancelled_requests"] == 1
    assert engine.step() == []


def test_engine_has_pending_requests() -> None:
    engine = _make_engine()

    assert engine.has_pending_requests() is False

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )

    assert engine.has_pending_requests() is True

    _ = engine.run_until_done()

    assert engine.has_pending_requests() is False


def test_engine_n_creates_multiple_sequences() -> None:
    engine = _make_engine(tokenizer=MockTokenizer())

    engine.add_request(
        request_id="req-n",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
        n=3,
    )

    assert len(engine._request_to_seq_ids["req-n"]) == 3

    outputs = engine.run_until_done()

    assert outputs == {
        "req-n": [
            [11, 12],
            [11, 12],
            [11, 12],
        ]
    }
    assert engine.get_request_n_outputs("req-n") == [
        [11, 12],
        [11, 12],
        [11, 12],
    ]


def test_engine_n_finished_when_all_complete() -> None:
    engine = _make_engine()

    engine.add_request(
        request_id="req-n",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
        n=2,
    )

    original_get_finish_reason = engine._get_finish_reason

    def _get_finish_reason(sequence, token_id):
        if sequence.seq_id == 0 and len(sequence.output_token_ids) == 1:
            return "stop"
        return original_get_finish_reason(sequence, token_id)

    engine._get_finish_reason = _get_finish_reason

    first_step_outputs = engine.step()

    assert any(
        output.request_id == "req-n" and output.finished
        for output in first_step_outputs
    )
    assert engine.has_pending_requests() is True
    assert "req-n" not in engine._completed_request_ids
