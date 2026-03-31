# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false

import time

import torch

from moe_infinity.engine.generation_loop import GenerationEngine
from moe_infinity.engine.types import SamplingParams
from moe_infinity.memory.kv_cache_manager import KVCacheManager
from moe_infinity.runtime.attention_types import KVCacheSpec


def _mock_forward(token_ids: list[int], _meta: object) -> torch.Tensor:
    vocab_size = 128
    logits = torch.zeros((len(token_ids), vocab_size), dtype=torch.float32)
    logits[:, 9] = 10.0
    return logits


def _build_engine() -> GenerationEngine:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )
    kv_mgr = KVCacheManager(
        num_gpu_blocks=256, num_cpu_blocks=256, block_size=4
    )
    return GenerationEngine(
        kv_cache_manager=kv_mgr,
        kv_spec=spec,
        num_layers=2,
        vocab_size=128,
        model_forward_fn=_mock_forward,
        eos_token_id=127,
        max_seq_length=512,
    )


def _measure_tokens_per_second(max_tokens: int) -> tuple[int, float]:
    engine = _build_engine()
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.perf_counter()
    result = engine.generate([1, 2, 3, 4], sampling_params=params)
    elapsed = max(time.perf_counter() - start, 1e-9)
    tokens_generated = len(result.output_token_ids)
    return tokens_generated, tokens_generated / elapsed


def test_throughput_tokens_per_second_positive() -> None:
    tokens, tokens_per_sec = _measure_tokens_per_second(max_tokens=128)
    assert tokens == 128
    assert tokens_per_sec > 0.0


def test_throughput_longer_decode_generates_more_tokens() -> None:
    short_tokens, short_tps = _measure_tokens_per_second(max_tokens=32)
    long_tokens, long_tps = _measure_tokens_per_second(max_tokens=96)
    assert short_tokens == 32
    assert long_tokens == 96
    assert short_tps > 0.0
    assert long_tps > 0.0
