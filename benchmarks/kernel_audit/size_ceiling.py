"""Size-matched memory ceilings for memory-bound gates.

The DRAM roof in ceilings.json comes from a 4 GiB stream. A kernel that moves
a few MB cannot reach it: launch latency, ramp-up and the last partial wave
are a large share of a few-microsecond run. This tool measures the best
bandwidth a plain streaming read (or copy) reaches on the SAME bytes, split
into the SAME number of launches, timed like `run_cases --mode time`
(common.time_cuda: L2 flushed before each call, CUDA events). A memory-bound
point is judged as achieved bandwidth / size-matched ceiling.

Each launch takes the best of several Triton grid / block / warp configs and
the matching torch op (`sum` for reads, `copy_` for copies).

Compare like with like: the ceiling's `mean_us` against the kernel's
`t_mean_us` from `run_cases --mode time --iters 200`. CUDA events tick at
~1.024 us here and add ~3 us per launch, so medians of short kernels snap to
tick multiples; both sides pay the same per-launch cost and the same L2-flush
write-back, which is why the ceiling is timed with the harness's own
`time_cuda`.

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m benchmarks.kernel_audit.size_ceiling \\
      --read 4.2MB,50MB,400MB          # one launch per size
  ... --read 34MB+17MB                 # two launches, timed as one sequence
  ... --copy 2MB,32MB                  # read N + write N bytes per launch
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import triton
import triton.language as tl

from benchmarks.kernel_audit.common import RESULTS_DIR, gpu_identity, time_cuda

_UNITS = {
    "": 1,
    "b": 1,
    "k": 10**3,
    "kb": 10**3,
    "m": 10**6,
    "mb": 10**6,
    "g": 10**9,
    "gb": 10**9,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
}


@triton.jit
def _read_kernel(x_ptr, out_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(pid * BLOCK, n_elem, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc ^= tl.load(x_ptr + offs, mask=offs < n_elem, other=0)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in range(pid * BLOCK, n_elem, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n_elem
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m), mask=m)


def parse_bytes(tok: str) -> int:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", tok)
    if not m or m.group(2).lower() not in _UNITS:
        raise argparse.ArgumentTypeError(
            f"bad size {tok!r} (e.g. 4.2MB, 50MiB)"
        )
    return int(float(m.group(1)) * _UNITS[m.group(2).lower()])


def _candidates(n_elem: int, sms: int):
    for block in (1024, 2048, 4096, 8192):
        tiles = triton.cdiv(n_elem, block)
        grids = {tiles} | {min(tiles, sms * k) for k in (1, 2, 4, 8, 16)}
        for grid in sorted(grids):
            for warps in (4, 8):
                yield block, grid, warps


def _best_launch(kind: str, nbytes: int, sms: int, iters: int) -> dict:
    n_elem = max(1, nbytes // 4)
    src = torch.zeros(n_elem, dtype=torch.int32, device="cuda")
    dst = torch.empty_like(src) if kind == "copy" else None
    out = torch.empty(
        sms * 16 + triton.cdiv(n_elem, 1024), dtype=torch.int32, device="cuda"
    )
    fns = {}
    for block, grid, warps in _candidates(n_elem, sms):
        if kind == "read":
            fn = lambda b=block, g=grid, w=warps: _read_kernel[(g,)](  # noqa: E731
                src, out, n_elem, BLOCK=b, num_warps=w
            )
        else:
            fn = lambda b=block, g=grid, w=warps: _copy_kernel[(g,)](  # noqa: E731
                src, dst, n_elem, BLOCK=b, num_warps=w
            )
        fns[f"triton b{block} g{grid} w{warps}"] = fn
    fns["torch.sum" if kind == "read" else "torch.copy_"] = (
        (lambda: src.sum()) if kind == "read" else (lambda: dst.copy_(src))
    )
    # Coarse pass over every config, then the full sample count on the top 4.
    coarse = sorted(
        fns, key=lambda k: time_cuda(fns[k], iters=20, warmup=2).mean_us
    )
    best = None
    for name in coarse[:4]:
        t = time_cuda(fns[name], iters=iters, warmup=2)
        if best is None or t.mean_us < best["t"].mean_us:
            best = {"cfg": name, "t": t, "fn": fns[name]}
    best["bytes"] = n_elem * 4 * (2 if kind == "copy" else 1)
    best["keep"] = (src, dst, out)
    return best


def measure(kind: str, parts: list[int], iters: int = 200) -> dict:
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    launches = [_best_launch(kind, b, sms, iters) for b in parts]
    if len(launches) == 1:
        t = launches[0]["t"]
    else:
        fns = [x["fn"] for x in launches]
        t = time_cuda(lambda: [f() for f in fns], iters=iters, warmup=2)
    total = sum(x["bytes"] for x in launches)
    return {
        "kind": kind,
        "launches": len(launches),
        "part_bytes": [x["bytes"] for x in launches],
        "bytes": total,
        "mean_us": t.mean_us,
        "median_us": t.median_us,
        "gbs": total / t.mean_us / 1e3,
        "gbs_median": total / t.median_us / 1e3,
        "cfgs": [x["cfg"] for x in launches],
    }


def _roofs() -> dict:
    p = RESULTS_DIR / "ceilings.json"
    if not p.exists():
        return {}
    c = json.loads(p.read_text())
    return {"read": c["dram_read"]["gbs"], "copy": c["dram_copy"]["gbs"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--read", default="", help="comma list; '+' joins launches")
    ap.add_argument("--copy", default="", help="comma list; '+' joins launches")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=Path, help="append JSON lines here")
    args = ap.parse_args()
    roofs = _roofs()
    print(json.dumps({"gpu": gpu_identity(), "roof_gbs": roofs}))
    for kind in ("read", "copy"):
        for item in filter(None, getattr(args, kind).split(",")):
            rec = measure(
                kind, [parse_bytes(t) for t in item.split("+")], args.iters
            )
            if kind in roofs:
                rec["pct_of_roof"] = 100 * rec["gbs"] / roofs[kind]
            line = json.dumps(rec)
            print(line)
            if args.out:
                with args.out.open("a") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
