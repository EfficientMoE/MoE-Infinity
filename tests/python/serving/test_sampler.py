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
        token = sampler.sample(logits, [params])[0].item()
        counts[token] += 1
    return counts


def test_greedy_sampling() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0]])

    sampled = sampler.sample(logits, [SamplingParams(temperature=0.0)])

    assert sampled.tolist() == [1]


def test_top_k_restricts_to_k() -> None:
    sampler = Sampler()
    logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]])
    params = SamplingParams(temperature=1.0, top_k=2)

    sampled = [sampler.sample(logits, [params])[0].item() for _ in range(100)]

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

    assert sampled[0].item() == 1
    assert sampled[1].item() in {0, 1}


def test_nucleus_sampling() -> None:
    sampler = Sampler()
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    params = SamplingParams(temperature=1.0, top_p=0.7)

    sampled = [sampler.sample(logits, [params])[0].item() for _ in range(100)]

    assert set(sampled).issubset({0, 1})
