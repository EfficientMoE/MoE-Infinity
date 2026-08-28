# pyright: reportAny=false, reportImplicitOverride=false

import importlib.util
import sys
import types
from pathlib import Path

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

_SEQUENCE_MODULE = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_BATCH_MODULE = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_MODEL_RUNNER_MODULE = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)

from moe_infinity.runtime.attention_types import PagedBatchLengths  # noqa: E402

SamplingParams = _SEQUENCE_MODULE.SamplingParams
BatchMetadata = _BATCH_MODULE.BatchMetadata
ModelRunner = _MODEL_RUNNER_MODULE.ModelRunner


class _MockOutput:
    logits: torch.Tensor

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class MockModel:
    vocab_size: int
    rank3_logits: bool
    config: types.SimpleNamespace
    eval_called: bool

    def __init__(
        self, vocab_size: int = 100, rank3_logits: bool = True
    ) -> None:
        self.vocab_size = vocab_size
        self.rank3_logits = rank3_logits
        self.config = types.SimpleNamespace(vocab_size=vocab_size)
        self.last_kwargs: dict[str, object] = {}
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> _MockOutput:
        self.last_kwargs = {"input_ids": input_ids, **kwargs}
        if self.rank3_logits:
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
        else:
            batch_size = input_ids.shape[0]
            logits = torch.randn(batch_size, self.vocab_size)
        return _MockOutput(logits=logits)


class _MockExpertTracer:
    def __init__(self) -> None:
        self.created: list[int] = []

    def create_entry(self) -> int:
        seq_id = len(self.created)
        self.created.append(seq_id)
        return seq_id


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


class _NoForwardModel(MockModel):
    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> _MockOutput:
        _ = input_ids
        _ = kwargs
        raise AssertionError("forward should not be called")


def _make_batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[10, 11],
        input_token_ids=[11, 12, 13, 21],
        lengths=PagedBatchLengths(
            query_lengths=[3, 1],
            query_offsets=[0, 3, 4],
            context_lengths=[0, 4],
            kv_seq_lengths=[3, 5],
        ),
        is_prefill=[True, False],
        block_tables=[[0], [1]],
        sampling_params=[SamplingParams(), SamplingParams()],
    )


def test_prepare_inputs_builds_padded_batch() -> None:
    runner = ModelRunner(MockModel(), MockOffloadEngine())
    batch = _make_batch()

    model_inputs = runner.prepare_inputs(batch)

    assert model_inputs["input_ids"].tolist() == [[11, 12, 13], [21, 0, 0]]
    assert model_inputs["position_ids"].tolist() == [[0, 1, 2], [4, 0, 0]]
    assert model_inputs["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_execute_runs_forward_and_returns_packed_logits() -> None:
    model = MockModel(vocab_size=32, rank3_logits=True)
    engine = MockOffloadEngine()
    runner = ModelRunner(model, engine)
    batch = _make_batch()

    logits = runner.execute(batch, past_key_values="kv")

    assert logits.shape == (4, 32)
    assert model.eval_called
    assert model.last_kwargs["use_cache"] is True
    assert model.last_kwargs["past_key_values"] == "kv"
    assert engine.request_id == 1
    assert engine.expert_tracer.created == [0, 1]
    assert runner.seq_id_list == [0, 1]
    assert engine.expert_layer_modules[0].seq_id_list == [0, 1]


def test_execute_supports_rank2_logits_for_decode_batches() -> None:
    model = MockModel(vocab_size=16, rank3_logits=False)
    runner = ModelRunner(model, MockOffloadEngine())
    batch = BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=[30, 31],
        lengths=PagedBatchLengths(
            query_lengths=[1, 1],
            query_offsets=[0, 1, 2],
            context_lengths=[8, 3],
            kv_seq_lengths=[9, 4],
        ),
        is_prefill=[False, False],
        block_tables=[[0], [1]],
        sampling_params=[SamplingParams(), SamplingParams()],
    )

    logits = runner.execute(batch)

    assert logits.shape == (2, 16)


def test_execute_empty_batch_skips_forward() -> None:
    model = _NoForwardModel(vocab_size=7)
    runner = ModelRunner(model, MockOffloadEngine())
    batch = BatchMetadata(
        seq_ids=[99],
        input_token_ids=[],
        lengths=PagedBatchLengths(
            query_lengths=[0],
            query_offsets=[0, 0],
            context_lengths=[10],
            kv_seq_lengths=[10],
        ),
        is_prefill=[False],
        block_tables=[[0]],
        sampling_params=[SamplingParams()],
    )

    logits = runner.execute(batch)

    assert logits.shape == (0, 7)
