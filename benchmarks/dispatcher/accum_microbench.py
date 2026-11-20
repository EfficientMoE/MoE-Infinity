"""Dispatcher-level microbenchmark: slot (deterministic) vs atomic accumulation.

Builds a synthetic MoE dispatcher (random bf16 SwiGLU experts, ``--layers``
layers so that an expert is not re-enqueued before its previous retirement) and
times ``set_inputs -> dispatch_experts -> wait_expert`` per dispatch for every
combination of

* accumulation mode: ``atomic`` (``set_deterministic_accum(False)``, the old
  path) and ``slot`` (``set_deterministic_accum(True)``), and
* expert path: ``grouped`` (grouped decode kernel) and ``per-expert``.

The modes are toggled on the same dispatcher and interleaved (the order
alternates every repetition), so slow drift of the host hits all of them.

Per configuration and repetition it reports, in microseconds:

* ``wall``    - host wall time per dispatch (median / p10 / p90 over every
                dispatch), no device synchronisation inside the loop;
* ``batch``   - wall time per dispatch of a batch of ``--batch`` dispatches
                ended by one ``torch.cuda.synchronize()`` (median / p10 / p90
                over the batches). This is the number to compare;
* ``gpu``     - with ``--gpu-events``: time between a CUDA event recorded
                before ``set_inputs`` and one recorded after ``wait_expert``
                on the caller stream (includes host launch gaps).

``--launch-count`` instead counts the GPU operations (kernels, memsets,
memcpys) issued per dispatch from ``torch.profiler`` (CUPTI activity records),
for each configuration, and prints the names.

Run (repo root on ``PYTHONPATH``, GPU 0 only, as quiet a host as you can get):

    CUDA_VISIBLE_DEVICES=0 python benchmarks/dispatcher/accum_microbench.py \
        --hidden 2048 --experts 64 --top-k 8 --tokens 1 --reps 3
    CUDA_VISIBLE_DEVICES=0 python benchmarks/dispatcher/accum_microbench.py \
        --launch-count

Needs an RTX PRO 6000 for the ``grouped`` rows (skipped otherwise).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DTYPE_BF16 = 0
DEEPSEEK_EXPERT_TYPE = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=1024)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument(
        "--batches", type=int, default=20, help="batches per repetition"
    )
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--gpu-events", action="store_true")
    p.add_argument("--launch-count", action="store_true")
    p.add_argument("--launch-dispatches", type=int, default=40)
    p.add_argument(
        "--configs",
        default="atomic-grouped,slot-grouped,atomic-per-expert,slot-per-expert",
        help="comma separated mode-path pairs",
    )
    p.add_argument(
        "--no-hint",
        action="store_true",
        help="do not tell the dispatcher the top-k (R = num_experts)",
    )
    p.add_argument("--offload-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build(args):
    from moe_infinity.runtime.model_offload import _load_prefetch_lib

    lib = _load_prefetch_lib()
    offload = args.offload_dir or tempfile.mkdtemp(prefix="accum-microbench-")
    os.makedirs(offload, exist_ok=True)
    engine = lib.prefetch_handle(f"{offload}/", 0.5)
    gen = torch.Generator().manual_seed(args.seed)
    E, H, I, L = args.experts, args.hidden, args.inter, args.layers
    # One set of expert weights, offloaded once per layer under new tensor ids.
    weights = [
        tuple(
            (torch.randn(*s, generator=gen) * 0.05).bfloat16()
            for s in ((I, H), (I, H), (H, I))
        )
        for _ in range(E)
    ]
    layer_ids = []
    for layer in range(L):
        ids_per_expert = []
        for e in range(E):
            ids = [3 * (layer * E + e) + k for k in range(3)]
            for tensor_id, tensor in zip(ids, weights[e]):
                engine.offload(tensor, tensor_id)
            ids_per_expert.append(ids)
        layer_ids.append(ids_per_expert)
    ndev = torch.cuda.device_count()
    dense = list(range(3 * L * E, 3 * L * E + 2 * ndev))
    for tensor_id in dense:
        engine.offload(torch.zeros(1, dtype=torch.bfloat16), tensor_id)
    split = len(dense) // 2
    engine.set_topology(
        [(f"model.dense.{i}", [[t]]) for i, t in enumerate(dense[:split])]
        + [
            (f"model.layers.{layer}.mlp.experts", layer_ids[layer])
            for layer in range(L)
        ]
        + [
            (f"model.dense.{i + split}", [[t]])
            for i, t in enumerate(dense[split:])
        ]
    )
    dispatcher = lib.expert_dispatcher(
        E, L, DTYPE_BF16, DEEPSEEK_EXPERT_TYPE, 4
    )
    for layer in range(L):
        for e in range(E):
            dispatcher.register_expert(layer, e, layer_ids[layer][e], "")
    return engine, dispatcher


def make_inputs(args, patterns=61):
    gen = torch.Generator().manual_seed(args.seed + 1)
    T, E, H, L = args.tokens, args.experts, args.hidden, args.layers
    x = [torch.randn(T, H, generator=gen).bfloat16().cuda() for _ in range(L)]
    routes = []
    for _ in range(patterns):
        logits = torch.randn(T, E, generator=gen)
        top = logits.topk(args.top_k, dim=1).indices
        mask = torch.zeros(T, E, dtype=torch.bool)
        mask.scatter_(1, top, True)
        weights = torch.softmax(logits, dim=1) * mask
        routes.append((mask.cuda(), weights.to(torch.bfloat16).cuda()))
    return x, routes


class Pacer:
    """Keeps the host at most ``layers - 2`` dispatches ahead of the device.

    ``wait_expert`` returns when the dispatch is *enqueued*; an expert may only
    be enqueued again after its previous retirement (a fatal CAS failure
    otherwise), which happens when the device has finished that dispatch. The
    pacer waits for the end event of the dispatch ``layers - 2`` back, i.e. the
    device is at most that far behind: no device synchronisation per dispatch,
    just a bound on the host run-ahead (a decode loop is bounded the same way
    by its per-token sampling).
    """

    def __init__(self, layers):
        self.layers = layers
        self.events = [None] * layers
        self.n = 0

    def before(self):
        ev = self.events[(self.n + 2) % self.layers]
        if ev is not None and self.n >= self.layers - 2:
            ev.synchronize()

    def after(self):
        ev = torch.cuda.Event()
        ev.record()
        self.events[self.n % self.layers] = ev
        self.n += 1


def dispatch_once(dispatcher, layer, x, route, pacer=None):
    if pacer is not None:
        pacer.before()
    mask, weights = route
    dispatcher.set_inputs(x, mask, weights)
    dispatcher.dispatch_experts(layer)
    out = dispatcher.wait_expert()
    if pacer is not None:
        pacer.after()
    return out


def configure(dispatcher, config, args):
    mode, path = config.split("-", 1)
    if hasattr(dispatcher, "set_deterministic_accum"):
        dispatcher.set_deterministic_accum(mode == "slot")
    else:  # an engine without slot mode (origin/main): atomic rows only
        assert mode == "atomic", "this engine has no set_deterministic_accum"
    hint = 0 if args.no_hint else args.top_k
    dispatcher.set_grouped_decode(path == "grouped", hint)


def warm_up(dispatcher, args, x, routes):
    """Make every expert of every layer resident, then warm the caches."""
    allmask = torch.ones(
        args.tokens, args.experts, dtype=torch.bool, device="cuda"
    )
    allw = (allmask.float() / args.experts).bfloat16()
    dispatcher.set_grouped_decode(False, 0)
    for layer in range(args.layers):
        dispatch_once(dispatcher, layer, x[layer], (allmask, allw))
        torch.cuda.synchronize()
        time.sleep(0.05)
    torch.cuda.synchronize()


def pct(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def measure(dispatcher, args, x, routes, gpu_events, pacer):
    walls, batches, gpus = [], [], []
    n = 0
    for _ in range(args.batches):
        events = []
        t_batch = time.perf_counter_ns()
        for _ in range(args.batch):
            layer = n % args.layers
            route = routes[n % len(routes)]
            n += 1
            if gpu_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            t0 = time.perf_counter_ns()
            dispatch_once(dispatcher, layer, x[layer], route, pacer)
            walls.append((time.perf_counter_ns() - t0) / 1e3)
            if gpu_events:
                end.record()
                events.append((start, end))
        torch.cuda.synchronize()
        batches.append((time.perf_counter_ns() - t_batch) / 1e3 / args.batch)
        gpus.extend(s.elapsed_time(e) * 1e3 for s, e in events)
    return walls, batches, gpus


def summarize(values):
    if not values:
        return "-"
    return (
        f"{statistics.median(values):8.1f} "
        f"[{pct(values, 0.1):7.1f}, {pct(values, 0.9):7.1f}]"
    )


def count_launches(dispatcher, args, x, routes, config, pacer):
    from torch.profiler import ProfilerActivity, profile

    configure(dispatcher, config, args)
    for i in range(args.warmup // 4):
        dispatch_once(
            dispatcher,
            i % args.layers,
            x[i % args.layers],
            routes[i % len(routes)],
            pacer,
        )
    torch.cuda.synchronize()
    n = args.launch_dispatches
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(n):
            layer = i % args.layers
            dispatch_once(
                dispatcher, layer, x[layer], routes[i % len(routes)], pacer
            )
        torch.cuda.synchronize()
    names = Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            names[ev.name] += 1
    return sum(names.values()) / n, names, n


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports

    configs = args.configs.split(",")
    if not hasattr(
        __import__("moe_infinity.runtime.model_offload", fromlist=["x"])
        ._load_prefetch_lib()
        .expert_dispatcher,
        "set_deterministic_accum",
    ):
        configs = [c for c in configs if c.startswith("atomic")]
    if not supports(RTX_PRO_6000):
        configs = [c for c in configs if c.endswith("per-expert")]
        print("not an RTX PRO 6000: grouped rows skipped")
    engine, dispatcher = build(args)
    x, routes = make_inputs(args)
    warm_up(dispatcher, args, x, routes)
    pacer = Pacer(args.layers)
    print(
        f"CONFIG hidden={args.hidden} inter={args.inter} experts={args.experts}"
        f" top_k={args.top_k} tokens={args.tokens} layers={args.layers}"
        f" hint={'none' if args.no_hint else args.top_k}"
        f" load_avg={os.getloadavg()}",
        flush=True,
    )
    if args.launch_count:
        print(f"{'config':22s} GPU ops / dispatch")
        for config in configs:
            per, names, n = count_launches(
                dispatcher, args, x, routes, config, pacer
            )
            print(f"{config:22s} {per:6.2f}", flush=True)
            for name, count in names.most_common():
                print(f"    {count / n:6.2f}  {name[:110]}")
        sys.stdout.flush()
        os._exit(0)

    results = {c: {"wall": [], "batch": [], "gpu": []} for c in configs}
    for rep in range(args.reps):
        order = configs if rep % 2 == 0 else configs[::-1]
        for config in order:
            configure(dispatcher, config, args)
            for i in range(args.warmup):  # per-config warm-up
                dispatch_once(
                    dispatcher,
                    i % args.layers,
                    x[i % args.layers],
                    routes[i % len(routes)],
                    pacer,
                )
            torch.cuda.synchronize()
            walls, batches, gpus = measure(
                dispatcher, args, x, routes, args.gpu_events, pacer
            )
            r = results[config]
            r["wall"].append(walls)
            r["batch"].append(batches)
            r["gpu"].append(gpus)
            print(
                f"rep {rep} {config:20s} wall {summarize(walls)}  "
                f"batch {summarize(batches)}  load={os.getloadavg()[0]:.1f}",
                flush=True,
            )
    print(
        "\n=== per repetition: batch median us/dispatch (p10, p90 of batches)"
    )
    for config in configs:
        meds = [statistics.median(b) for b in results[config]["batch"]]
        print(
            f"{config:22s} " + "  ".join(f"{m:8.1f}" for m in meds),
            f" | median of reps {statistics.median(meds):8.1f}",
        )
    print("\n=== pooled over repetitions (median [p10, p90]) us/dispatch")
    pooled = {}
    for config in configs:
        wall = [v for rep in results[config]["wall"] for v in rep]
        batch = [v for rep in results[config]["batch"] for v in rep]
        gpu = [v for rep in results[config]["gpu"] for v in rep]
        pooled[config] = (wall, batch, gpu)
        line = f"{config:22s} wall {summarize(wall)}   batch {summarize(batch)}"
        if gpu:
            line += f"   gpu {summarize(gpu)}"
        print(line)
    for path in ("grouped", "per-expert"):
        a, s = f"atomic-{path}", f"slot-{path}"
        if a in pooled and s in pooled:
            ratios = [
                statistics.median(results[s]["batch"][r])
                / statistics.median(results[a]["batch"][r])
                for r in range(args.reps)
            ]
            print(
                f"slot/atomic ({path}) per-repetition batch-median ratio: "
                + ", ".join(f"{q:.3f}" for q in ratios)
                + f"  (mean {statistics.mean(ratios):.3f}, "
                f"min {min(ratios):.3f}, max {max(ratios):.3f})"
            )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
