from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from typing import Protocol

import torch

from moe_infinity.serving.engine import ContinuousBatchingEngine, RequestOutput
from moe_infinity.serving.sequence import SamplingParams


class _ModelConfigLike(Protocol):
    eos_token_id: int
    vocab_size: int


class _MockModelLike(Protocol):
    config: _ModelConfigLike

    def force_token(self, token_id: int) -> None: ...
    def forward(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> SimpleNamespace: ...


def test_single_request_completes(
    cb_engine: ContinuousBatchingEngine,
) -> None:
    cb_engine.add_request(
        request_id="req-single",
        prompt_token_ids=[9, 10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )

    outputs = cb_engine.run_until_done()

    assert list(outputs.keys()) == ["req-single"]
    assert len(outputs["req-single"]) == 4
    assert all(isinstance(token_id, int) for token_id in outputs["req-single"])

    stats = cb_engine.get_stats()
    assert stats["pending_requests"] == 0
    assert stats["completed_requests"] == 1
    assert stats["total_generated_tokens"] == 4


def test_multiple_requests_all_complete(
    cb_engine: ContinuousBatchingEngine,
) -> None:
    requests = {
        "req-a": (1, 2),
        "req-b": (20, 3),
        "req-c": (33, 5),
    }

    for request_id, (prompt_token, max_tokens) in requests.items():
        cb_engine.add_request(
            request_id=request_id,
            prompt_token_ids=[prompt_token],
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=max_tokens
            ),
        )

    outputs = cb_engine.run_until_done()

    assert set(outputs.keys()) == set(requests.keys())
    for request_id, (_, max_tokens) in requests.items():
        assert len(outputs[request_id]) == max_tokens

    stats = cb_engine.get_stats()
    assert stats["pending_requests"] == 0
    assert stats["completed_requests"] == 3


def test_greedy_deterministic(
    cb_engine_factory: Callable[..., ContinuousBatchingEngine],
) -> None:
    baseline_rng_state = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(baseline_rng_state.clone())
    engine_a = cb_engine_factory()
    engine_a.add_request(
        request_id="req-greedy",
        prompt_token_ids=[7],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=6),
    )
    outputs_a = engine_a.run_until_done()

    torch.random.set_rng_state(baseline_rng_state.clone())
    engine_b = cb_engine_factory()
    engine_b.add_request(
        request_id="req-greedy",
        prompt_token_ids=[7],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=6),
    )
    outputs_b = engine_b.run_until_done()

    assert outputs_a == outputs_b


def test_max_tokens_stops_generation(
    cb_engine: ContinuousBatchingEngine,
) -> None:
    cb_engine.add_request(
        request_id="req-max",
        prompt_token_ids=[4],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )

    step_outputs: list[RequestOutput] = []
    while cb_engine.has_pending_requests():
        new_outputs = cb_engine.step()
        assert new_outputs
        step_outputs.extend(new_outputs)

    assert len(step_outputs) == 3
    assert [output.finished for output in step_outputs] == [False, False, True]
    assert step_outputs[-1].usage is not None
    assert all(output.usage is None for output in step_outputs[:-1])


def test_eos_stops_generation(
    cb_engine_factory: Callable[..., ContinuousBatchingEngine],
    mock_model: _MockModelLike,
) -> None:
    eos_token_id = mock_model.config.eos_token_id
    mock_model.force_token(eos_token_id)
    engine = cb_engine_factory(model=mock_model)

    engine.add_request(
        request_id="req-eos",
        prompt_token_ids=[18],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=10),
    )

    outputs = engine.run_until_done()

    assert outputs == {"req-eos": [eos_token_id]}
    assert engine.get_stats()["total_generated_tokens"] == 1


def test_streaming_callback_fires(
    cb_engine: ContinuousBatchingEngine,
) -> None:
    callback_outputs: list[RequestOutput] = []

    def _on_token(output: RequestOutput) -> None:
        callback_outputs.append(output)

    cb_engine.add_request(
        request_id="req-stream",
        prompt_token_ids=[2],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        on_token=_on_token,
    )

    outputs = cb_engine.run_until_done()

    assert len(outputs["req-stream"]) == 4
    assert len(callback_outputs) == 4
    assert [output.request_id for output in callback_outputs] == [
        "req-stream"
    ] * 4
    assert [output.finished for output in callback_outputs] == [
        False,
        False,
        False,
        True,
    ]


def test_request_ids_in_output(
    cb_engine: ContinuousBatchingEngine,
) -> None:
    cb_engine.add_request(
        request_id="req-left",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
    )
    cb_engine.add_request(
        request_id="req-right",
        prompt_token_ids=[30],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )

    all_outputs: list[RequestOutput] = []
    while cb_engine.has_pending_requests():
        all_outputs.extend(cb_engine.step())

    assert {output.request_id for output in all_outputs} == {
        "req-left",
        "req-right",
    }

    output_counts = Counter(output.request_id for output in all_outputs)
    assert output_counts == {"req-left": 2, "req-right": 3}


def test_abort_before_start(
    cb_engine_factory: Callable[..., ContinuousBatchingEngine],
) -> None:
    engine = cb_engine_factory(config_overrides={"max_batch_size": 1})

    engine.add_request(
        request_id="req-keep",
        prompt_token_ids=[11],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
    )
    engine.add_request(
        request_id="req-abort",
        prompt_token_ids=[22],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )

    engine.abort_request("req-abort")
    outputs = engine.run_until_done()

    assert "req-abort" not in outputs
    assert len(outputs["req-keep"]) == 2
    stats = engine.get_stats()
    assert stats["cancelled_requests"] == 1
    assert stats["completed_requests"] == 1


def test_sampling_pipeline_respects_top_k(
    cb_engine_factory: Callable[..., ContinuousBatchingEngine],
    mock_model: _MockModelLike,
) -> None:
    chosen_token_id = 17

    def _fixed_forward(
        input_ids: torch.Tensor, **kwargs: object
    ) -> SimpleNamespace:
        _ = kwargs
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, mock_model.config.vocab_size),
            -1000.0,
            dtype=torch.float32,
        )
        logits[..., chosen_token_id] = 1.0
        logits[..., 13] = 0.9
        return SimpleNamespace(logits=logits)

    mock_model.forward = _fixed_forward
    engine = cb_engine_factory(model=mock_model)

    engine.add_request(
        request_id="req-top-k",
        prompt_token_ids=[5],
        sampling_params=SamplingParams(
            temperature=1.0,
            top_k=1,
            max_tokens=4,
        ),
    )

    outputs = engine.run_until_done()

    assert outputs == {"req-top-k": [chosen_token_id] * 4}
