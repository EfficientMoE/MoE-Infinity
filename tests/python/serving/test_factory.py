"""CPU unit tests for the in-process serving-engine factory.

The real ``ContinuousBatchingEngine`` needs a GPU and a loaded model, so
``create_serving_engine`` / ``generate_tokens`` are exercised here against a
recording fake monkeypatched in for the engine class. Token-identical parity
against the real engine lives in the GPU-gated
``tests/python/integration/test_factory_gpu.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.serving import factory
from moe_infinity.serving.factory import (
    build_serving_config,
    create_serving_engine,
    generate_tokens,
)
from moe_infinity.serving.sequence import SamplingParams


def _fake_moe(num_layers=4, num_kv_heads=2, head_dim=64, eos=7):
    model_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        eos_token_id=eos,
    )
    inner = SimpleNamespace(config=model_config, dtype=torch.bfloat16)
    engine_config = SimpleNamespace(device_memory_ratio=0.5)
    return SimpleNamespace(
        model=inner, engine=object(), engine_config=engine_config
    )


class _RecordingEngine:
    def __init__(
        self,
        model,
        engine,
        config,
        tokenizer=None,
        speculative_draft=None,
    ):
        self.model = model
        self.engine = engine
        self.config = config
        self.tokenizer = tokenizer
        self.speculative_draft = speculative_draft
        self.requests: list[dict[str, object]] = []
        self.shutdown_calls = 0

    def add_request(self, request_id, prompt_token_ids, sampling_params):
        self.requests.append(
            {
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": sampling_params,
            }
        )

    def run_until_done(self):
        return {
            request["request_id"]: [len(request["prompt_token_ids"])]
            for request in self.requests
        }

    def shutdown(self, timeout_ms: float = 5000.0) -> None:
        self.shutdown_calls += 1


def test_build_serving_config_derives_from_model_config():
    cfg = build_serving_config(_fake_moe())
    assert cfg["num_layers"] == 4
    assert cfg["num_kv_heads"] == 2
    assert cfg["head_dim"] == 64
    assert cfg["eos_token_id"] == 7
    assert cfg["dtype"] == "bfloat16"
    assert cfg["device_memory_ratio"] == 0.5
    assert cfg["block_size"] == 16
    assert cfg["max_batch_size"] == 1


def test_build_serving_config_overrides_win():
    cfg = build_serving_config(
        _fake_moe(), max_batch_size=8, kv_cache_ratio=0.1
    )
    assert cfg["max_batch_size"] == 8
    assert cfg["kv_cache_ratio"] == 0.1


def test_build_serving_config_text_config_fallback():
    text = SimpleNamespace(
        num_hidden_layers=3, num_key_value_heads=1, head_dim=32, eos_token_id=2
    )
    outer = SimpleNamespace(get_text_config=lambda: text, eos_token_id=2)
    inner = SimpleNamespace(config=outer, dtype=torch.float16)
    moe = SimpleNamespace(
        model=inner, engine=object(), engine_config=SimpleNamespace()
    )
    cfg = build_serving_config(moe)
    assert cfg["num_layers"] == 3
    assert cfg["dtype"] == "float16"


def test_build_serving_config_missing_field_raises():
    inner = SimpleNamespace(config=SimpleNamespace(), dtype=torch.bfloat16)
    moe = SimpleNamespace(
        model=inner, engine=object(), engine_config=SimpleNamespace()
    )
    with pytest.raises(RuntimeError, match="unable to resolve"):
        build_serving_config(moe)


def test_build_serving_config_device_memory_ratio_default():
    inner = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=16,
            eos_token_id=0,
        ),
        dtype=torch.bfloat16,
    )
    moe = SimpleNamespace(
        model=inner, engine=object(), engine_config=SimpleNamespace()
    )
    cfg = build_serving_config(moe)
    assert cfg["device_memory_ratio"] == 0.75
    assert cfg["kv_cache_ratio"] == 0.25


def test_create_serving_engine_plumbs_arguments(monkeypatch):
    monkeypatch.setattr(factory, "ContinuousBatchingEngine", _RecordingEngine)
    moe = _fake_moe()
    tokenizer = object()
    speculative_draft = object()

    engine = create_serving_engine(
        moe,
        tokenizer=tokenizer,
        speculative_draft=speculative_draft,
        max_batch_size=4,
    )

    assert isinstance(engine, _RecordingEngine)
    assert engine.model is moe.model
    assert engine.engine is moe.engine
    assert engine.tokenizer is tokenizer
    assert engine.speculative_draft is speculative_draft
    assert engine.config["num_layers"] == 4
    assert engine.config["max_batch_size"] == 4


def test_generate_tokens_orchestrates_requests(monkeypatch):
    monkeypatch.setattr(factory, "ContinuousBatchingEngine", _RecordingEngine)
    engine = create_serving_engine(_fake_moe())

    result = generate_tokens(
        engine,
        [[1, 2, 3], [4, 5]],
        max_new_tokens=5,
        temperature=0.7,
        top_k=3,
        top_p=0.9,
    )

    assert [r["request_id"] for r in engine.requests] == [
        "inproc-0",
        "inproc-1",
    ]
    assert engine.requests[0]["prompt_token_ids"] == [1, 2, 3]
    assert engine.requests[1]["prompt_token_ids"] == [4, 5]

    params = engine.requests[0]["sampling_params"]
    assert isinstance(params, SamplingParams)
    assert params.temperature == 0.7
    assert params.top_k == 3
    assert params.top_p == 0.9
    assert params.max_tokens == 5

    assert result == [[3], [2]]


def test_generate_tokens_greedy_defaults(monkeypatch):
    monkeypatch.setattr(factory, "ContinuousBatchingEngine", _RecordingEngine)
    engine = create_serving_engine(_fake_moe())

    generate_tokens(engine, [[9, 9]], max_new_tokens=1)

    params = engine.requests[0]["sampling_params"]
    assert params.temperature == 0.0
    assert params.top_k == 0
    assert params.top_p == 1.0


def test_factory_engine_shutdown_is_idempotent(monkeypatch):
    monkeypatch.setattr(factory, "ContinuousBatchingEngine", _RecordingEngine)
    engine = create_serving_engine(_fake_moe())
    generate_tokens(engine, [[1, 2]], max_new_tokens=1)

    engine.shutdown()
    engine.shutdown()
    assert engine.shutdown_calls == 2
