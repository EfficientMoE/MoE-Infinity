from importlib import import_module
from types import ModuleType
from typing import Callable, Optional, Protocol, cast

import torch

from moe_infinity.engine.types import SamplingParams, SequenceStatus
from moe_infinity.memory.kv_cache_manager import KVCacheManager
from moe_infinity.runtime.attention_types import KVCacheSpec


class _GenerationResultLike(Protocol):
    output_token_ids: list[int]
    finish_reason: SequenceStatus


class _GenerationEngineLike(Protocol):
    eos_token_id: int

    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> _GenerationResultLike: ...


class _GenerationEngineCtor(Protocol):
    def __call__(
        self,
        kv_cache_manager: KVCacheManager,
        kv_spec: KVCacheSpec,
        num_layers: int,
        vocab_size: int,
        model_forward_fn: Optional[
            Callable[[list[int], object], torch.Tensor]
        ] = None,
        eos_token_id: int = 2,
    ) -> _GenerationEngineLike: ...


_generation_loop_module: ModuleType = import_module(
    "moe_infinity.engine.generation_loop"
)
GenerationEngine = cast(
    _GenerationEngineCtor,
    getattr(_generation_loop_module, "GenerationEngine"),
)
GenerationResult = cast(
    type[object],
    getattr(_generation_loop_module, "GenerationResult"),
)


def make_engine(
    num_gpu_blocks: int = 50,
    vocab_size: int = 100,
    block_size: int = 4,
    model_forward_fn: Optional[
        Callable[[list[int], object], torch.Tensor]
    ] = None,
    eos_token_id: int = 2,
) -> _GenerationEngineLike:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=block_size,
    )
    mgr = KVCacheManager(
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=20,
        block_size=block_size,
    )
    return GenerationEngine(
        kv_cache_manager=mgr,
        kv_spec=spec,
        num_layers=2,
        vocab_size=vocab_size,
        model_forward_fn=model_forward_fn,
        eos_token_id=eos_token_id,
    )


def test_basic_generation() -> None:
    engine = make_engine()
    sp = SamplingParams(max_tokens=5, temperature=1.0)
    result = engine.generate(
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
    )

    assert isinstance(result, GenerationResult)
    assert len(result.output_token_ids) <= 5
    assert result.finish_reason in (
        SequenceStatus.FINISHED_LENGTH,
        SequenceStatus.FINISHED_STOPPED,
    )


def test_block_lifecycle() -> None:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )
    mgr = KVCacheManager(num_gpu_blocks=50, num_cpu_blocks=20, block_size=4)
    initial_free = mgr.num_free_gpu_blocks
    engine = GenerationEngine(
        kv_cache_manager=mgr,
        kv_spec=spec,
        num_layers=2,
        vocab_size=100,
    )

    _ = engine.generate(
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=3),
    )
    assert mgr.num_free_gpu_blocks == initial_free


def test_eos_stops_generation() -> None:
    eos_token = 42

    def mock_forward(token_ids: list[int], _meta: object) -> torch.Tensor:
        logits = torch.full((len(token_ids), 100), -1e9)
        logits[:, eos_token] = 10.0
        return logits

    engine = make_engine(
        vocab_size=100,
        model_forward_fn=mock_forward,
        eos_token_id=eos_token,
    )
    sp = SamplingParams(max_tokens=10, temperature=0)
    result = engine.generate([1, 2], sp)
    assert result.finish_reason == SequenceStatus.FINISHED_STOPPED
    assert result.output_token_ids
    assert result.output_token_ids[0] == eos_token


def test_greedy_sampling() -> None:
    def mock_forward(token_ids: list[int], _meta: object) -> torch.Tensor:
        logits = torch.zeros(len(token_ids), 50)
        logits[:, 7] = 10.0
        return logits

    engine = make_engine(vocab_size=50, model_forward_fn=mock_forward)
    sp = SamplingParams(max_tokens=3, temperature=0)
    result = engine.generate([1, 2], sp)

    assert result.output_token_ids
    assert all(token_id == 7 for token_id in result.output_token_ids)
