from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest
import torch


@dataclass
class MockConfig:
    vocab_size: int = 64
    eos_token_id: int = 2


class MockHFModel:
    config: MockConfig
    device: torch.device
    eval: Mock
    _forced_token_id: Optional[int]
    _masked_token_ids: set[int]

    def __init__(
        self,
        vocab_size: int = 64,
        eos_token_id: int = 2,
    ) -> None:
        self.config = MockConfig(
            vocab_size=vocab_size, eos_token_id=eos_token_id
        )
        self.device = torch.device("cpu")
        self.eval = Mock(return_value=None)
        self._forced_token_id = None
        self._masked_token_ids = {eos_token_id}

    def force_token(self, token_id: int) -> None:
        self._forced_token_id = token_id

    def clear_forced_token(self) -> None:
        self._forced_token_id = None

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        _ = kwargs
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(
            batch_size,
            seq_len,
            self.config.vocab_size,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        for token_id in self._masked_token_ids:
            logits[..., token_id] = -1e9

        if self._forced_token_id is not None:
            logits[..., self._forced_token_id] = 1e9

        return SimpleNamespace(logits=logits)


def _make_mock_engine_obj() -> Mock:
    class _ExpertTracer:
        _next_entry: int

        def __init__(self) -> None:
            self._next_entry = 0

        def create_entry(self) -> int:
            entry = self._next_entry
            self._next_entry += 1
            return entry

    tracer = _ExpertTracer()

    request_counter = {"value": 0}

    def _generate_request_id() -> int:
        request_id = request_counter["value"]
        request_counter["value"] += 1
        return request_id

    engine_obj = Mock()
    engine_obj.expert_tracer = tracer
    engine_obj.expert_layer_modules = [SimpleNamespace(seq_id_list=[])]
    engine_obj._generate_request_id = MagicMock(
        side_effect=_generate_request_id
    )
    engine_obj.request_id = 0
    return engine_obj


@pytest.fixture
def mock_model() -> MockHFModel:
    return MockHFModel()


@pytest.fixture
def mock_engine_obj() -> Mock:
    return _make_mock_engine_obj()


@pytest.fixture
def cb_base_config() -> dict[str, object]:
    return {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 8,
        "max_tokens_per_step": 128,
        "block_size": 4,
        "num_layers": 2,
        "num_kv_heads": 4,
        "head_dim": 32,
        "dtype": "float32",
        "eos_token_id": 2,
        "model_memory_bytes": 0,
        "num_kv_blocks": 32,
    }


@pytest.fixture
def cb_engine_factory(cb_base_config: dict[str, object]):
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    def _factory(
        *,
        model: Optional[MockHFModel] = None,
        engine_obj: Optional[Mock] = None,
        tokenizer: Optional[object] = None,
        config_overrides: Optional[dict[str, object]] = None,
    ) -> ContinuousBatchingEngine:
        config = dict(cb_base_config)
        if config_overrides:
            config.update(config_overrides)
        return ContinuousBatchingEngine(
            model=model or MockHFModel(),
            engine=engine_obj or _make_mock_engine_obj(),
            config=config,
            tokenizer=tokenizer,
        )

    return _factory


@pytest.fixture
def cb_engine(
    mock_model: MockHFModel,
    mock_engine_obj: Mock,
    cb_base_config: dict[str, object],
):
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    return ContinuousBatchingEngine(mock_model, mock_engine_obj, cb_base_config)
