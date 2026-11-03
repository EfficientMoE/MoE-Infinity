"""Run audit cases in timing mode, GPU-busy mode, or as an ncu target.

  # device time per call (cold L2 before each call)
  CUDA_VISIBLE_DEVICES=5 python -m benchmarks.kernel_audit.run_cases \
      --suite expert --mode time --out results/expert_time.jsonl

  # wall time vs summed kernel time (host-bound detection) via torch.profiler
  ... --mode busy

  # hardware counters: run under ncu (see ncu_driver.py)
  python -m benchmarks.kernel_audit.ncu_driver --suite expert --tag expert
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback

import torch

from benchmarks.kernel_audit.common import (
    l2_flush,
    other_gpu_procs,
    time_cuda,
    warm_gpu,
)
from benchmarks.kernel_audit.suites import all_cases


def _select(cases, args):
    out = []
    for c in cases:
        if args.models and c.model not in args.models:
            continue
        if args.impl and not re.search(args.impl, c.impl):
            continue
        if args.match and not re.search(args.match, c.cid):
            continue
        if args.exclude and re.search(args.exclude, c.cid):
            continue
        out.append(c)
    return out


def busy(fn, reps: int = 1) -> dict:
    """Wall time of `fn` vs. sum of its kernel times (CUPTI via torch.profiler)."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    l2_flush()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]
    ) as prof:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / reps
    kern = 0.0
    n = 0
    for ev in prof.events():
        if (
            ev.device_type == torch.autograd.DeviceType.CUDA
            and ev.name != "Memset"
        ):
            if "flush" in ev.name:
                continue
            kern += (
                ev.device_time_total
                if hasattr(ev, "device_time_total")
                else ev.cuda_time_total
            )
            n += 1
    kern_us = kern / reps
    return {
        "wall_us": wall * 1e6,
        "kernel_us": kern_us,
        "n_kernels": n / reps,
        "gpu_busy": kern_us / (wall * 1e6) if wall > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument(
        "--mode", choices=["time", "busy", "ncu", "list"], default="time"
    )
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--impl", default=None, help="regex on impl name")
    ap.add_argument("--match", default=None, help="regex on case id")
    ap.add_argument("--exclude", default=None, help="regex on case id to skip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    cases = _select(all_cases(args.suite, args.models), args)
    if args.mode == "list":
        for c in cases:
            print(c.cid, json.dumps(c.meta), c.skip or "")
        print(len(cases), "cases", file=sys.stderr)
        return

    out_f = open(args.out, "a") if args.out else None
    cudart = torch.cuda.cudart()
    if args.mode == "time":
        warm_gpu()
    for i, c in enumerate(cases):
        rec = {
            "cid": c.cid,
            "suite": c.suite,
            "model": c.model,
            "phase": c.phase,
            "size": c.size,
            "impl": c.impl,
            "flops": c.flops,
            "bytes_min": c.bytes_min,
            "per_layer": c.per_layer,
            "layers": c.layers,
            "meta": c.meta,
            "mode": args.mode,
        }
        fn = None
        if c.skip:
            rec.update(ok=False, skip=c.skip)
        else:
            try:
                fn = c.build()
                if args.mode == "time":
                    t = time_cuda(
                        fn,
                        iters=c.iters or args.iters,
                        warmup=1 if c.iters else 3,
                    )
                    rec.update(
                        t_med_us=t.median_us,
                        t_mean_us=t.mean_us,
                        t_min_us=t.min_us,
                        t_p90_us=t.p90_us,
                        n=t.n,
                    )
                    if i % 25 == 0:
                        rec["other_procs"] = other_gpu_procs()
                elif args.mode == "busy":
                    rec.update(busy(fn))
                else:
                    fn()
                    fn()
                    torch.cuda.synchronize()
                    cudart.cudaProfilerStart()
                    torch.cuda.nvtx.range_push(c.cid)
                    fn()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    cudart.cudaProfilerStop()
                rec["ok"] = True
            except torch.OutOfMemoryError as exc:
                rec.update(ok=False, skip=f"OOM: {str(exc)[:160]}")
            except Exception as exc:  # keep sweeping; record the failure
                rec["ok"] = False
                rec["error"] = repr(exc)[:400]
                traceback.print_exc(limit=2)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        fn = None
        gc.collect()
        torch.cuda.empty_cache()
        line = json.dumps(rec, default=float)
        if out_f:
            out_f.write(line + "\n")
            out_f.flush()
        if args.mode == "time" and rec.get("ok"):
            msg = f"{rec['t_med_us']:12.1f} us"
        elif args.mode == "busy" and rec.get("ok"):
            msg = f"wall {rec['wall_us']:12.1f} us  kern {rec['kernel_us']:12.1f} us  busy {rec['gpu_busy']:.2f}  nk {rec['n_kernels']:.0f}"
        else:
            msg = str(rec.get("skip") or rec.get("error") or "ok")[:90]
        print(f"[{i + 1}/{len(cases)}] {c.cid:58s} {msg}", flush=True)
    if out_f:
        out_f.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)
