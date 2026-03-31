# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false

import torch

from moe_infinity.engine.generation_loop import GenerationEngine
from moe_infinity.engine.types import SamplingParams
from moe_infinity.memory.kv_cache_manager import KVCacheManager
from moe_infinity.runtime.attention_types import KVCacheSpec


def _make_engine(
    *,
    vocab_size: int = 64,
    eos_token_id: int = 63,
    model_forward_fn=None,
) -> GenerationEngine:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )
    kv_mgr = KVCacheManager(
        num_gpu_blocks=128, num_cpu_blocks=128, block_size=4
    )
    return GenerationEngine(
        kv_cache_manager=kv_mgr,
        kv_spec=spec,
        num_layers=2,
        vocab_size=vocab_size,
        model_forward_fn=model_forward_fn,
        eos_token_id=eos_token_id,
        max_seq_length=256,
    )


def test_greedy_determinism() -> None:
    target_token = 17

    def forward(_token_ids: list[int], _meta: object) -> torch.Tensor:
        logits = torch.full((1, 64), -1e9)
        logits[:, target_token] = 100.0
        return logits

    engine = _make_engine(model_forward_fn=forward)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    first = engine.generate([1, 2, 3], sampling_params=sampling_params)
    second = engine.generate([1, 2, 3], sampling_params=sampling_params)

    assert first.output_token_ids == second.output_token_ids
    assert all(token_id == target_token for token_id in first.output_token_ids)


def test_sampling_distribution() -> None:
    valid_tokens = {3, 7, 11}

    def forward(token_ids: list[int], _meta: object) -> torch.Tensor:
        logits = torch.full((len(token_ids), 32), -1e9)
        logits[:, 3] = 4.0
        logits[:, 7] = 3.0
        logits[:, 11] = 2.0
        return logits

    engine = _make_engine(
        vocab_size=32, eos_token_id=31, model_forward_fn=forward
    )
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=24)

    result = engine.generate([5, 6], sampling_params=sampling_params)

    assert result.output_token_ids
    assert all(token_id in valid_tokens for token_id in result.output_token_ids)
