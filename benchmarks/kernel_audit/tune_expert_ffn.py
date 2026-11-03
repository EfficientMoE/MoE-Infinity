"""Tune the CUTLASS expert-FFN tile table (extensions/kernel/fused_moe_mlp.cu).

For every BF16-executed model in models.py and a grid of tokens-per-expert M,
time every (tile config, split-K) of the gate/up dual GEMM and of the down GEMM
(device time from CUPTI, L2 flushed before each call) and keep the fastest.
The result is written as a C++ table that fused_moe_ffn_into consults before
its analytic cost model:

  gpu_run.sh python -m benchmarks.kernel_audit.tune_expert_ffn \\
      --out /tmp/tune.json [--models qwen3-30b-a3b ...]
  python -m benchmarks.kernel_audit.tune_expert_ffn --emit /tmp/tune*.json \\
      > extensions/kernel/fused_moe_mlp_tuned.h

Split the model list across several runs to keep each GPU lock short.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from statistics import median

import torch

from benchmarks.kernel_audit.common import l2_flush
from benchmarks.kernel_audit.models import MODELS

# Dense where tile quantisation moves the optimum (multiples of 64 / 128
# rows); runtime lookup rounds M up to the next grid point.
M_GRID = (
    (1, 4, 16, 32)
    + tuple(range(64, 1025, 64))
    + tuple(range(1152, 2049, 128))
    + tuple(range(2304, 4097, 256))
    + tuple(range(4608, 8193, 512))
    + (9216, 10240, 11264, 12288, 12800)
)
SPLITS = (1, 2, 3, 4, 6, 8)
WORKSPACE_BYTES = 32 << 20  # kMaxWorkspaceBytes in fused_moe_mlp.cu
_DUAL = re.compile(r", true, (true|false), moe_gemm")
_DOWN = re.compile(r", false, (true|false), moe_gemm")


def _kernel_us_batch(fns, pattern, iters):
    """Device time (median of `iters` calls, L2 flushed before each) of the
    kernel matching `pattern` in every fn, profiled in one session."""
    from torch.profiler import ProfilerActivity, profile

    for fn in fns:  # first-use setup (smem attribute, split-K workspace)
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for fn in fns:
            for _ in range(iters):
                l2_flush()
                fn()
        torch.cuda.synchronize()
    events = sorted(
        (
            e
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and pattern.search(e.name)
        ),
        key=lambda e: e.time_range.start,
    )
    times = [e.device_time_total for e in events]
    assert len(times) == len(fns) * iters, (len(times), len(fns), iters)
    return [median(times[i * iters : (i + 1) * iters]) for i in range(len(fns))]


def _candidates(tile, M, N, K, acc):
    """(splits) worth timing: stream-K remainder (0) and the distinct
    effective uniform slice counts whose split-K workspace fits (larger ones
    fall back to one slice at run time)."""
    tm, tn, tk = tile
    tiles = -(-M // tm) * -(-N // tn)
    k_tiles = -(-K // tk)
    yield 0  # stream-K remainder (planned at run time)
    seen = set()
    for s in SPLITS:
        k_per = -(-k_tiles // s)
        eff = -(-k_tiles // k_per)
        if eff in seen:
            continue
        if eff > 1 and tiles * eff * tm * tn * acc * 4 > WORKSPACE_BYTES:
            continue
        seen.add(eff)
        yield eff


def _models(keys):
    for key, m in MODELS.items():
        if keys and key not in keys:
            continue
        if m.expert_type == 6:  # GPT-OSS does not use this kernel
            continue
        yield key, m


def _prior_winners(paths):
    """{(kind, n, k): {cfg}} of configs that won anywhere in earlier runs."""
    won = {}
    for p in paths or ():
        for path in glob.glob(p):
            for r in json.load(open(path))["results"]:
                won.setdefault(("dual", r["I"], r["H"]), set()).add(
                    r["dual"][1]
                )
                won.setdefault(("down", r["H"], r["I"]), set()).add(
                    r["down"][1]
                )
    return won


def tune(keys, out_path, prior=None, grid=M_GRID, iters=5):
    import moe_infinity._store as store

    num_dual, num_down = store.fused_moe_ffn_num_configs()
    dual_tiles, down_tiles = store.fused_moe_ffn_tile_shapes()
    won = _prior_winners(prior)
    results = []
    seen = set()
    for key, m in _models(keys):
        H, I = m.hidden, m.inter
        if (H, I) in seen:
            continue
        seen.add((H, I))
        rnd = lambda *s, sc=0.02: (  # noqa: E731
            torch.randn(*s, device="cuda") * sc
        ).to(torch.bfloat16)
        g, u, d = rnd(I, H), rnd(I, H), rnd(H, I)
        for M in grid:
            if M * H * 2 > 2**30:
                continue
            x = rnd(M, H, sc=1.0)
            fb = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
            best = {}
            for kind, n_cfg, pattern, tiles in (
                ("dual", num_dual, _DUAL, dual_tiles),
                ("down", num_down, _DOWN, down_tiles),
            ):
                nk = (I, H) if kind == "dual" else (H, I)
                acc = 2 if kind == "dual" else 1
                cfgs = won.get((kind, *nk)) or range(n_cfg)
                keys_ = [
                    (cfg, s)
                    for cfg in sorted(cfgs)
                    for s in _candidates(tiles[cfg], M, *nk, acc)
                ]
                fns = [
                    (
                        lambda a=(
                            (cfg, s, num_down - 1, 1)
                            if kind == "dual"
                            else (num_dual - 1, 1, cfg, s)
                        ): store.fused_moe_ffn_pro6000(x, g, u, d, fb, out, *a)
                    )
                    for cfg, s in keys_
                ]
                times = _kernel_us_batch(fns, pattern, iters)
                best[kind] = min(
                    (t, cfg, s) for t, (cfg, s) in zip(times, keys_)
                )
            rec = {
                "model": key,
                "H": H,
                "I": I,
                "M": M,
                "dual": best["dual"],
                "down": best["down"],
            }
            results.append(rec)
            print(json.dumps(rec), flush=True)
    with open(out_path, "w") as f:
        json.dump(
            {"device": torch.cuda.get_device_name(), "results": results}, f
        )


def emit(paths):
    rows = []
    for p in paths:
        for path in glob.glob(p):
            rows += json.load(open(path))["results"]
    entries = set()
    for r in rows:
        H, I, M = r["H"], r["I"], r["M"]
        _, c, s = r["dual"]
        entries.add((1, I, H, M, c, s))
        _, c, s = r["down"]
        entries.add((0, H, I, M, c, s))
    print(
        "// Generated by benchmarks/kernel_audit/tune_expert_ffn.py; do not edit."
    )
    print("// Fastest (tile config, split-K) per GEMM measured on RTX PRO 6000")
    print("// Blackwell Max-Q (sm_120, 188 SMs), L2 flushed before each call.")
    print("// RTX PRO 6000 only: fused_moe_ffn_into uses it behind the")
    print("// RTX_PRO_6000 gate; other GPUs run fused_moe_ffn_into_generic.")
    print("#pragma once\n")
    print("struct TunedExpertGemm {")
    print("  bool dual;  // gate+up dual GEMM (else down GEMM)")
    print("  int n, k, m;")
    print("  int cfg, splits;")
    print("};\n")
    print("constexpr int kTunedSmsPro6000 = 188;")
    print("constexpr TunedExpertGemm kTunedExpertGemmPro6000[] = {")
    for e in sorted(entries):
        print(
            f"    {{{'true' if e[0] else 'false'}, {e[1]}, {e[2]}, {e[3]}, "
            f"{e[4]}, {e[5]}}},"
        )
    print("};")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--emit", nargs="*")
    ap.add_argument(
        "--prior",
        nargs="*",
        help="earlier tune JSONs: only time configs that won there",
    )
    ap.add_argument("--m", nargs="*", type=int, help="override the M grid")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    if args.emit:
        emit(args.emit)
    else:
        if not args.out:
            sys.exit("--out required")
        tune(
            args.models,
            args.out,
            args.prior,
            tuple(args.m) if args.m else M_GRID,
            args.iters,
        )


if __name__ == "__main__":
    main()
