"""Microbenchmark: host-loop vs device-side expert-drop selection.

Compares the PR #257 `select_expert_drops` (packed D2H copy + Python row
loop) against the prototype `select_expert_drops_device` (pure tensor ops,
zero host sync) at OLMoE decode geometry. Wall time captures what decode
actually pays: the old path blocks the host; the new path only enqueues.

Usage: CUDA_VISIBLE_DEVICES=<idle-gpu> python benchmarks/expert_drop/bench_device_select.py
"""

import time

import torch

from moe_infinity.runtime.expert_drop import (
    select_expert_drops,
    select_expert_drops_device,
)

WARMUP = 50
ITERS = 200


def _make_inputs(rows, experts, topk, seed=0):
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(rows, experts, generator=gen)
    mask = torch.zeros(rows, experts, dtype=torch.bool)
    mask.scatter_(-1, weights.topk(topk, dim=-1).indices, True)
    weights = torch.where(mask, weights, torch.zeros(()))
    weights = weights / weights.sum(-1, keepdim=True)
    resident = torch.rand(experts, generator=gen) < 0.4
    return mask.cuda(), weights.cuda(), resident.cuda()


def _time_wall(fn):
    samples = []
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    for _ in range(ITERS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    torch.cuda.synchronize()
    t = torch.tensor(samples)
    return (
        float(t.mean()),
        float(t.quantile(0.5)),
        float(t.quantile(0.99)),
    )


def main():
    assert torch.cuda.is_available()
    kw = dict(min_k=1, mass_budget=0.05, flatness_floor=0.5)
    compiled = torch.compile(
        select_expert_drops_device, mode="reduce-overhead", dynamic=False
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"warmup={WARMUP} iters={ITERS}  budget={kw['mass_budget']}")
    print()
    header = (
        "| geometry | host-loop wall us (mean/p50/p99) | "
        "device eager us (mean/p50/p99) | "
        "device compiled us (mean/p50/p99) | best speedup |"
    )
    print(header)
    print("|---|---|---|---|---|")
    for rows, experts, topk in (
        (1, 64, 8),
        (4, 64, 8),
        (8, 64, 8),
        (24, 64, 8),
        (24, 256, 8),
    ):
        mask, weights, resident = _make_inputs(rows, experts, topk)
        ref = select_expert_drops(
            mask.cpu(), weights.cpu(), resident.cpu(), **kw
        )
        got = compiled(mask, weights, resident, **kw)
        assert torch.equal(got[0].cpu().bool(), ref[0].bool())
        assert torch.allclose(got[1].cpu(), ref[1], rtol=1e-5, atol=1e-7)
        old = _time_wall(
            lambda: select_expert_drops(mask, weights, resident, **kw)
        )
        new = _time_wall(
            lambda: select_expert_drops_device(mask, weights, resident, **kw)
        )
        comp = _time_wall(lambda: compiled(mask, weights, resident, **kw))
        best = min(new[0], comp[0])
        print(
            f"| rows={rows} E={experts} k={topk} "
            f"| {old[0]:.0f} / {old[1]:.0f} / {old[2]:.0f} "
            f"| {new[0]:.0f} / {new[1]:.0f} / {new[2]:.0f} "
            f"| {comp[0]:.0f} / {comp[1]:.0f} / {comp[2]:.0f} "
            f"| {old[0] / best:.1f}x |"
        )


if __name__ == "__main__":
    main()
