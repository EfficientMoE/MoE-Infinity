"""Serving refuses models it would decode without KV history (#318)."""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM

from moe_infinity.serving.batch import BatchMetadata
from moe_infinity.serving.model_runner import (
    STRICT_PAGED_ATTENTION_ENV,
    ModelRunner,
)
from moe_infinity.serving.sequence import SamplingParams

from .test_mixtral_paged_attention import (
    _backend,
    _build_model,
    _ServingEngine,
)


def _gpt_oss_model() -> torch.nn.Module:
    config = GptOssConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GptOssForCausalLM(config).eval()


def _no_backend_engine() -> SimpleNamespace:
    # GLM-5.x / GLM-5.3-Flash / MiniMax-M3 force the native engine off, so
    # the offload engine exposes no attention backend.
    return SimpleNamespace(
        get_attention_backend=lambda: None, expert_layer_modules=[]
    )


@pytest.fixture(autouse=True)
def _default_strictness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STRICT_PAGED_ATTENTION_ENV, raising=False)


def test_model_without_shim_fails_at_startup() -> None:
    runner = ModelRunner(
        _gpt_oss_model(),
        _ServingEngine(_backend("cpu")),
        device=torch.device("cpu"),
    )
    assert runner.paged_kv_history_unsupported_reason() == (
        "no *PagedAttention shim matched its attention layers"
    )
    with pytest.raises(RuntimeError) as excinfo:
        runner.require_paged_kv_history()
    message = str(excinfo.value)
    assert "GptOssForCausalLM" in message
    assert "model_type='gpt_oss'" in message
    assert f"{STRICT_PAGED_ATTENTION_ENV}=0" in message


def test_model_without_backend_fails_at_startup() -> None:
    runner = ModelRunner(
        _build_model(), _no_backend_engine(), device=torch.device("cpu")
    )
    assert runner.paged_kv_history_unsupported_reason() == (
        "no paged attention backend was built for it"
    )
    with pytest.raises(RuntimeError, match="no paged attention backend"):
        runner.require_paged_kv_history()


def test_opt_out_only_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(STRICT_PAGED_ATTENTION_ENV, "0")
    runner = ModelRunner(
        _gpt_oss_model(), _no_backend_engine(), device=torch.device("cpu")
    )
    with caplog.at_level(logging.WARNING):
        runner.require_paged_kv_history()
        runner.require_paged_kv_history()
    assert caplog.text.count("cannot keep KV history") == 1


def _decode_batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[1],
        input_token_ids=[3, 9, 27],
        seq_lengths=[3],
        context_lengths=[0],
        is_prefill=[True],
        block_tables=[[0, 1]],
        sampling_params=[SamplingParams()],
    )


def test_runtime_check_warns_by_default_and_raises_when_strict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runner = ModelRunner(
        _gpt_oss_model(),
        _ServingEngine(_backend("cpu")),
        device=torch.device("cpu"),
    )
    with caplog.at_level(logging.WARNING):
        runner.execute(_decode_batch())
    assert "cannot keep KV history" in caplog.text

    monkeypatch.setenv(STRICT_PAGED_ATTENTION_ENV, "1")
    strict = ModelRunner(
        _gpt_oss_model(),
        _ServingEngine(_backend("cpu")),
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="cannot keep KV history"):
        strict.execute(_decode_batch())


def _initialize(
    monkeypatch: pytest.MonkeyPatch, model: torch.nn.Module, engine: object
) -> None:
    import moe_infinity.entrypoints.openai.api_server_v2 as srv

    monkeypatch.setattr(
        srv.ContinuousBatchingEngine,
        "_resolve_device",
        staticmethod(lambda model: torch.device("cpu")),
    )
    monkeypatch.setattr(srv, "engine", None)
    srv.initialize_with_model(
        moe_model=SimpleNamespace(model=model, engine=engine),
        model_name="tiny",
        tok=None,
        max_seq_length=128,
        max_batch_size=4,
    )


@pytest.mark.parametrize(
    ("model_fn", "engine_fn", "match"),
    [
        (
            _gpt_oss_model,
            lambda: _ServingEngine(_backend("cpu")),
            "no \\*PagedAttention shim",
        ),
        (_build_model, _no_backend_engine, "no paged attention backend"),
    ],
    ids=["gpt-oss-no-shim", "no-backend"],
)
def test_serve_entry_point_refuses_unsupported_models(
    monkeypatch: pytest.MonkeyPatch, model_fn, engine_fn, match: str
) -> None:
    import moe_infinity.entrypoints.openai.api_server_v2 as srv

    with pytest.raises(RuntimeError, match=match):
        _initialize(monkeypatch, model_fn(), engine_fn())
    assert srv.engine is None


def test_serve_entry_point_accepts_mixtral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moe_infinity.entrypoints.openai.api_server_v2 as srv

    _initialize(monkeypatch, _build_model(), _ServingEngine(_backend("cpu")))
    assert srv.engine is not None
    assert srv.engine.model_runner.paged_kv_history_unsupported_reason() is None
    srv.engine.shutdown()
