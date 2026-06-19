# pyright: reportAny=false, reportImplicitOverride=false, reportPrivateUsage=false

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

from tests.python.ops.conftest import requires_cuda

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
_CUDA_GRAPH_MODULE = _load_module(
    "moe_infinity.serving.cuda_graph",
    ROOT / "moe_infinity" / "serving" / "cuda_graph.py",
)

SamplingParams = _SEQUENCE_MODULE.SamplingParams
BatchMetadata = _BATCH_MODULE.BatchMetadata
ModelRunner = _MODEL_RUNNER_MODULE.ModelRunner
CudaGraphRunner = _CUDA_GRAPH_MODULE.CudaGraphRunner


class _MockOutput:
    logits: torch.Tensor

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _MockModel:
    config: types.SimpleNamespace

    def __init__(self, vocab_size: int = 8) -> None:
        self.config = types.SimpleNamespace(vocab_size=vocab_size)

    def eval(self) -> None:
        return None

    def forward(self, input_ids: torch.Tensor, **_: object) -> _MockOutput:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.config.vocab_size)
        return _MockOutput(logits=logits)


class _MockEngine:
    def _generate_request_id(self) -> int:
        return 0


class _CudaModel(_MockModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> _MockOutput:
        values = (
            input_ids.to(dtype=torch.float32)
            + position_ids.to(dtype=torch.float32)
            + attention_mask.to(dtype=torch.float32)
        )
        logits = torch.stack((values, values + 1.0, values + 2.0), dim=-1)
        return _MockOutput(logits=logits)


def _make_decode_batch(
    *,
    batch_size: int = 2,
    context_lengths: list[int] | None = None,
    input_token_ids: list[int] | None = None,
) -> BatchMetadata:
    if context_lengths is None:
        context_lengths = list(range(batch_size))
    if input_token_ids is None:
        input_token_ids = [idx + 10 for idx in range(batch_size)]
    return BatchMetadata(
        seq_ids=list(range(batch_size)),
        input_token_ids=input_token_ids,
        seq_lengths=[1] * batch_size,
        context_lengths=context_lengths,
        is_prefill=[False] * batch_size,
        block_tables=[[0] for _ in range(batch_size)],
        token_offsets=list(range(batch_size + 1)),
        sampling_params=[SamplingParams() for _ in range(batch_size)],
    )


def test_is_compatible_requires_decode_only_captured_batch() -> None:
    runner = CudaGraphRunner(
        ModelRunner(_MockModel(), _MockEngine(), device=torch.device("cpu")),
        max_batch_sizes=(2,),
    )
    runner._is_cuda_device = lambda: True
    runner._graphs[2] = types.SimpleNamespace()

    decode_batch = _make_decode_batch(batch_size=2)
    prefill_batch = BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=[11, 12],
        seq_lengths=[1, 1],
        context_lengths=[0, 0],
        is_prefill=[True, False],
        block_tables=[[0], [0]],
        token_offsets=[0, 1, 2],
        sampling_params=[SamplingParams(), SamplingParams()],
    )
    multi_token_batch = BatchMetadata(
        seq_ids=[1],
        input_token_ids=[11, 12],
        seq_lengths=[2],
        context_lengths=[0],
        is_prefill=[False],
        block_tables=[[0]],
        token_offsets=[0, 2],
        sampling_params=[SamplingParams()],
    )

    assert runner.is_compatible(decode_batch)
    assert not runner.is_compatible(prefill_batch)
    assert not runner.is_compatible(multi_token_batch)


def test_invalidate_clears_captured_state() -> None:
    runner = CudaGraphRunner(
        ModelRunner(_MockModel(), _MockEngine(), device=torch.device("cpu")),
        max_batch_sizes=(2,),
    )
    runner._graphs[2] = types.SimpleNamespace()
    runner._warmed_batch_sizes.add(2)

    runner.invalidate()

    assert runner._graphs == {}
    assert runner._warmed_batch_sizes == set()


def test_replay_empty_batch_returns_empty_logits_without_graph() -> None:
    runner = CudaGraphRunner(
        ModelRunner(
            _MockModel(vocab_size=5), _MockEngine(), device=torch.device("cpu")
        ),
        max_batch_sizes=(1,),
    )
    empty_batch = BatchMetadata(
        seq_ids=[1],
        input_token_ids=[],
        seq_lengths=[0],
        context_lengths=[7],
        is_prefill=[False],
        block_tables=[[0]],
        token_offsets=[0, 0],
        sampling_params=[SamplingParams()],
    )

    logits = runner.replay(empty_batch)

    assert logits.shape == (0, 5)


def test_env_flag_disables_graph_compatibility(monkeypatch) -> None:
    monkeypatch.setenv("MOE_DISABLE_CUDA_GRAPHS", "1")
    runner = CudaGraphRunner(
        ModelRunner(_MockModel(), _MockEngine(), device=torch.device("cuda")),
        max_batch_sizes=(2,),
    )
    runner._graphs[2] = types.SimpleNamespace()

    assert not runner.is_compatible(_make_decode_batch(batch_size=2))

    monkeypatch.delenv("MOE_DISABLE_CUDA_GRAPHS", raising=False)


@requires_cuda
def test_capture_and_replay_decode_batch_on_cuda() -> None:
    os.environ.pop("MOE_DISABLE_CUDA_GRAPHS", None)
    model_runner = ModelRunner(_CudaModel(vocab_size=3), _MockEngine())
    runner = CudaGraphRunner(model_runner, max_batch_sizes=(2,))
    capture_batch = _make_decode_batch(
        batch_size=2,
        context_lengths=[3, 7],
        input_token_ids=[20, 30],
    )
    replay_batch = _make_decode_batch(
        batch_size=2,
        context_lengths=[4, 10],
        input_token_ids=[21, 31],
    )

    runner.warmup(capture_batch, num_iters=1)
    runner.capture(capture_batch)
    logits = runner.replay(replay_batch)

    expected = torch.tensor(
        [[26.0, 27.0, 28.0], [42.0, 43.0, 44.0]], device=logits.device
    )
    assert runner.is_compatible(replay_batch)
    assert torch.allclose(logits, expected)
