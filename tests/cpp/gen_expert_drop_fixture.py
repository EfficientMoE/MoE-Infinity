"""Emit a parity fixture for core/parallel/expert_drop_select.cc.

Inputs and expected outputs come from the Python reference
select_expert_drops. Flat whitespace format (see test_expert_drop_select.cc)
so the C++ side needs no JSON dependency.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_infinity.runtime.expert_drop import select_expert_drops  # noqa: E402


def _random_case(seed):
    gen = torch.Generator().manual_seed(seed)
    rows = int(torch.randint(1, 65, (1,), generator=gen))
    experts = int(torch.randint(8, 257, (1,), generator=gen))
    topk = int(torch.randint(2, min(17, experts + 1), (1,), generator=gen))
    weights = torch.rand(rows, experts, generator=gen)
    mask = torch.zeros(rows, experts, dtype=torch.bool)
    mask.scatter_(-1, weights.topk(topk, dim=-1).indices, True)
    weights = torch.where(mask, weights, torch.zeros(()))
    weights = weights / weights.sum(-1, keepdim=True)
    resident = torch.rand(experts, generator=gen) < 0.4
    min_k = int(torch.randint(1, topk + 1, (1,), generator=gen))
    mass_budget = float(torch.rand(1, generator=gen)) * 0.3
    flatness_floor = float(torch.rand(1, generator=gen))
    return mask, weights, resident, min_k, mass_budget, flatness_floor


def _targeted_cases():
    mask = torch.tensor([[True, True, True, False]])
    resident = torch.tensor([True, False, False, True])
    yield (
        mask,
        torch.tensor([[0.5, 0.25, 0.25, 0.0]]),
        resident,
        1,
        0.25,
        0.5,
    )
    yield (
        mask,
        torch.tensor([[0.5, float("nan"), 0.2, 0.0]]),
        resident,
        1,
        0.25,
        0.5,
    )
    yield (
        mask,
        torch.tensor([[0.5, -0.3, 0.2, 0.0]]),
        resident,
        1,
        0.25,
        0.5,
    )
    yield mask, torch.zeros(1, 4), resident, 1, 0.25, 0.5
    yield (
        mask,
        torch.tensor([[0.4, 0.4, 0.2, 0.0]]),
        resident,
        1,
        0.25,
        0.5,
    )
    yield (
        mask,
        torch.tensor([[0.5, 0.3, 0.2, 0.0]]),
        torch.ones(4, dtype=torch.bool),
        1,
        0.25,
        0.5,
    )
    yield (
        mask,
        torch.tensor([[0.0, 0.6, 0.4, 0.0]]),
        resident,
        1,
        1.0,
        0.5,
    )
    yield (
        mask,
        torch.tensor([[0.5, 0.3, 0.2, 0.0]]),
        resident,
        3,
        0.25,
        0.5,
    )


def main():
    out = []
    cases = [_random_case(seed) for seed in range(200)]
    cases.extend(_targeted_cases())
    out.append(str(len(cases)))
    for mask, weights, resident, min_k, budget, floor in cases:
        exp_mask, exp_w, counts = select_expert_drops(
            mask.clone(),
            weights.clone(),
            resident,
            min_k=min_k,
            mass_budget=budget,
            flatness_floor=floor,
        )
        rows, experts = mask.shape
        out.append(f"{rows} {experts} {min_k} {budget!r} {floor!r}")
        out.append(" ".join(f"{v!r}" for v in weights.flatten().tolist()))
        out.append(" ".join(str(int(v)) for v in mask.flatten().tolist()))
        out.append(" ".join(str(int(v)) for v in resident.tolist()))
        out.append(" ".join(f"{v!r}" for v in exp_w.flatten().tolist()))
        out.append(
            " ".join(str(int(v)) for v in exp_mask.bool().flatten().tolist())
        )
        out.append(
            f"{counts.tokens_seen} {counts.tokens_changed} "
            f"{counts.experts_dropped} {counts.flatness_bypasses}"
        )
    Path(sys.argv[1]).write_text("\n".join(out) + "\n")
    print(f"wrote {len(cases)} cases to {sys.argv[1]}")


if __name__ == "__main__":
    main()
