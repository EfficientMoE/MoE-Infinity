# pyright: reportAny=false

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[3]
sequence_module = _load_module(
    "sequence_module_sampler_test",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
sampler_module = _load_module(
    "sampler_module",
    ROOT / "moe_infinity" / "serving" / "sampler.py",
)

SamplingParams = sequence_module.SamplingParams
Sampler = sampler_module.Sampler


def _sample_many(
    sampler: Sampler,
    logits: torch.Tensor,
    params: SamplingParams,
    trials: int = 200,
) -> torch.Tensor:
    counts = torch.zeros(logits.size(-1), dtype=torch.long)
    for _ in range(trials):
        token = sampler.sample(logits, [params]).token_ids[0].item()
        counts[token] += 1
    return counts


def test_greedy_sampling() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0]])

    sampled = sampler.sample(logits, [SamplingParams(temperature=0.0)])

    assert sampled.token_ids.tolist() == [1]


def test_top_k_restricts_to_k() -> None:
    sampler = Sampler()
    logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]])
    params = SamplingParams(temperature=1.0, top_k=2)

    sampled = [
        sampler.sample(logits, [params]).token_ids[0].item()
        for _ in range(100)
    ]

    assert set(sampled).issubset({0, 1})


def test_temperature_scaling() -> None:
    sampler = Sampler()
    logits = torch.tensor([[4.0, 1.0, 0.0]])

    low_temp_counts = _sample_many(
        sampler,
        logits,
        SamplingParams(temperature=0.25),
    )

    high_temp_counts = _sample_many(
        sampler,
        logits,
        SamplingParams(temperature=2.5),
    )

    assert low_temp_counts[0] > high_temp_counts[0]
    assert low_temp_counts[0] >= 190
    assert high_temp_counts[0] <= 170


def test_batch_different_params() -> None:
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 0.7, 0.2],
            [4.0, 3.0, 0.0],
        ]
    )
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=2),
    ]

    sampled = sampler.sample(logits, params)

    assert sampled.token_ids[0].item() == 1
    assert sampled.token_ids[1].item() in {0, 1}


def test_nucleus_sampling() -> None:
    sampler = Sampler()
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    params = SamplingParams(temperature=1.0, top_p=0.7)

    sampled = [
        sampler.sample(logits, [params]).token_ids[0].item()
        for _ in range(100)
    ]

    assert set(sampled).issubset({0, 1})


def test_sample_returns_sampler_output() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0]])

    result = sampler.sample(logits, [SamplingParams(temperature=0.0)])

    assert hasattr(result, "token_ids")
    assert result.token_ids.tolist() == [1]


def test_logprobs_returned_when_requested() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    params = SamplingParams(temperature=0.0, logprobs=2)

    result = sampler.sample(logits, [params])

    assert result.token_logprobs is not None
    assert len(result.token_logprobs) == 1
    assert result.top_logprobs is not None
    assert len(result.top_logprobs[0]) == 2


def test_logprobs_disabled_returns_none() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    params = SamplingParams(temperature=0.0, logprobs=0)

    result = sampler.sample(logits, [params])

    assert result.token_logprobs is None
    assert result.top_logprobs is None


def test_logprobs_values_are_valid() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    params = SamplingParams(temperature=0.0, logprobs=3)

    result = sampler.sample(logits, [params])

    assert result.token_logprobs is not None
    assert result.top_logprobs is not None
    for lp in result.token_logprobs:
        assert lp <= 0.0
    for top_lp_dict in result.top_logprobs:
        for value in top_lp_dict.values():
            assert value <= 0.0
