from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import moe_infinity.entrypoints.openai.api_server_v2 as srv
from moe_infinity.serving.batch import BatchBuilder
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.model_runner import ModelRunner
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
)


def _tiny_png_data_url() -> str:
    image = Image.new("RGB", (64, 64), color=(12, 34, 56))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _vision_payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _tiny_png_data_url()},
                    },
                ],
            }
        ],
        "max_tokens": 1,
    }


class _TextTokenizer:
    def encode(self, prompt: str) -> list[int]:
        _ = prompt
        return [1, 2, 3]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        _ = token_ids, kwargs
        return "ok"

    def apply_chat_template(self, **kwargs: object) -> str:
        _ = kwargs
        return "user: Describe this image.\nassistant:"


class _StubProcessor(_TextTokenizer):
    def __init__(self) -> None:
        self.received_image: Image.Image | None = None

    def apply_chat_template(self, conversation: Any, **kwargs: object) -> Any:
        content = conversation[0]["content"]
        self.received_image = content[1]["image"]
        assert isinstance(self.received_image, Image.Image)
        assert self.received_image.size == (64, 64)
        assert kwargs == {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 64, 64)),
            "image_grid_thw": torch.tensor([[1, 8, 8]]),
        }


@dataclass
class _MockOutput:
    seq_id: int = 0
    token_id: int = 7
    token_text: str = "ok"
    finished: bool = True
    finish_reason: str = "stop"
    token_logprob: float | None = None
    top_logprobs: dict[int, float] | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        }
    )


class _AdmissionEngine:
    def __init__(self) -> None:
        cache = PagedKVCache(
            num_blocks=16,
            block_size=4,
            num_layers=1,
            num_heads=1,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.scheduler = Scheduler(
            cache, max_batch_size=2, max_tokens_per_step=32
        )
        self.cache = cache
        self.batch = None
        self.scheduler_output = None

    def add_request(
        self,
        *,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        multimodal_inputs: dict[str, Any] | None,
        on_token: Any,
        n: int,
    ) -> None:
        assert n == 1
        sequence = SequenceData(
            seq_id=0,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            multimodal_inputs=multimodal_inputs,
        )
        self.scheduler.add_request(
            SequenceGroup(request_id=request_id, sequences=[sequence])
        )
        self.scheduler_output = self.scheduler.schedule()
        self.batch = BatchBuilder.from_scheduler_output(
            self.scheduler_output, {0: sequence}, self.cache
        )
        on_token(_MockOutput())

    def has_pending_requests(self) -> bool:
        return False

    def abort_request(self, request_id: str) -> None:
        _ = request_id

    def shutdown(self) -> None:
        pass


@pytest.fixture
def server_state() -> Any:
    original = {
        "engine": srv.engine,
        "stream_manager": srv.stream_manager,
        "tokenizer": srv.tokenizer,
        "processor": getattr(srv, "processor", None),
        "model_name_global": srv.model_name_global,
    }
    srv.stream_manager = object()
    srv.tokenizer = _TextTokenizer()
    srv.model_name_global = "test-model"
    srv._health_state.set_healthy()
    try:
        with TestClient(srv.app) as client:
            yield client
    finally:
        for name, value in original.items():
            setattr(srv, name, value)


def test_image_request_rejected_for_text_only_model(
    server_state: TestClient,
) -> None:
    srv.engine = SimpleNamespace(
        scheduler=SimpleNamespace(num_waiting=0),
        has_pending_requests=lambda: False,
        shutdown=lambda: None,
    )
    srv.processor = None

    response = server_state.post("/v1/chat/completions", json=_vision_payload())

    assert response.status_code == 400
    assert "does not support image input" in response.json()["error"]["message"]


def test_vl_request_carries_processor_inputs_and_expanded_token_count(
    server_state: TestClient,
) -> None:
    processor = _StubProcessor()
    admission_engine = _AdmissionEngine()
    srv.processor = processor
    srv.engine = admission_engine

    response = server_state.post("/v1/chat/completions", json=_vision_payload())

    assert response.status_code == 200
    assert processor.received_image is not None
    assert admission_engine.batch is not None
    assert admission_engine.batch.multimodal_inputs is not None
    assert admission_engine.batch.multimodal_inputs["pixel_values"].shape == (
        1,
        3,
        64,
        64,
    )
    assert admission_engine.batch.multimodal_inputs["image_grid_thw"].shape == (
        1,
        3,
    )
    assert admission_engine.scheduler_output.num_prefill_tokens == 5
    prepared_inputs = ModelRunner(
        model=object(), engine=object(), device=torch.device("cpu")
    ).prepare_inputs(admission_engine.batch)
    assert prepared_inputs["pixel_values"].shape == (1, 3, 64, 64)


def test_processor_resolution_falls_back_to_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = object()
    processor_loader = MagicMock(side_effect=ValueError("not multimodal"))
    tokenizer_loader = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)

    resolved_tokenizer, resolved_processor = (
        srv._resolve_processor_and_tokenizer("text-model")
    )

    assert resolved_tokenizer is tokenizer
    assert resolved_processor is None
    processor_loader.assert_called_once_with(
        "text-model", trust_remote_code=True
    )
    tokenizer_loader.assert_called_once_with(
        "text-model", trust_remote_code=True
    )


def test_processor_resolution_uses_multimodal_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoProcessor

    tokenizer = object()
    loaded_processor = SimpleNamespace(
        image_processor=object(), tokenizer=tokenizer
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        MagicMock(return_value=loaded_processor),
    )

    resolved_tokenizer, resolved_processor = (
        srv._resolve_processor_and_tokenizer("vl-model")
    )

    assert resolved_tokenizer is tokenizer
    assert resolved_processor is loaded_processor
