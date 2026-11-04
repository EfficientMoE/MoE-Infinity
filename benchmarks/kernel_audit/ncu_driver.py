"""Profile audit cases with Nsight Compute and flatten the report to JSONL.

Each case is wrapped in cudaProfilerStart/Stop + an NVTX range named by the
case id, so every profiled kernel is attributed to exactly one case.

  CUDA_VISIBLE_DEVICES=5 python -m benchmarks.kernel_audit.ncu_driver \
      --suite expert --models qwen3.5-35b-a3b --tag expert_qwen35
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.kernel_audit.common import RESULTS_DIR

NCU = os.environ.get("NCU", "ncu")


def _ncu_python_dir() -> str:
    """`extras/python` (holding `ncu_report.py`) of the Nsight Compute that
    `$NCU` runs; the report format is version-specific. Override: $NCU_PY."""
    if os.environ.get("NCU_PY"):
        return os.environ["NCU_PY"]
    import glob
    import re
    import shutil

    exe = shutil.which(NCU)
    roots = []
    if exe:
        real = Path(os.path.realpath(exe))
        roots += [real.parent, real.parent.parent]
        # The CUDA toolkit's bin/ncu is a wrapper that runs the newest of
        # <cuda>/nsight-compute-* and /opt/nvidia/nsight-compute/*.
        roots += [
            Path(p)
            for p in glob.glob(str(real.parent.parent / "nsight-compute*"))
        ]
    roots += [Path(p) for p in glob.glob("/opt/nvidia/nsight-compute/*")]
    found = [
        r / "extras/python"
        for r in roots
        if (r / "extras/python/ncu_report.py").exists()
    ]
    if not found:
        raise FileNotFoundError(
            "ncu_report.py not found; set NCU_PY to <nsight-compute>/extras/python"
        )
    out = subprocess.run(
        [NCU, "--version"], capture_output=True, text=True
    ).stdout
    ver = re.search(r"Version (\d+\.\d+\.\d+)", out)
    if ver:
        for d in found:
            if ver.group(1) in str(d):
                return str(d)

    def _key(d: Path):
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", str(d))
        return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)

    return str(max(found, key=_key))


# Chosen to fit in 3 replay passes on GB202 (sm__throughput alone costs +5).
METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_sector_hit_rate.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_elapsed.avg.per_second",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "sm__maximum_warps_per_active_cycle_pct",
]


def run_ncu(args) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = RESULTS_DIR / f"ncu_{args.tag}"
    cmd = [
        NCU,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--clock-control",
        args.clock_control,
        "--cache-control",
        args.cache_control,
        "--nvtx",
        "--metrics",
        ",".join(METRICS),
        "--force-overwrite",
        "-o",
        str(rep),
        sys.executable,
        "-m",
        "benchmarks.kernel_audit.run_cases",
        "--suite",
        args.suite,
        "--mode",
        "ncu",
    ]
    if args.models:
        cmd += ["--models", *args.models]
    if args.impl:
        cmd += ["--impl", args.impl]
    if args.match:
        cmd += ["--match", args.match]
    if args.exclude:
        cmd += ["--exclude", args.exclude]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return _report_file(args.tag)


def _report_file(tag: str) -> Path:
    hits = sorted(RESULTS_DIR.glob(f"ncu_{tag}.ncu-rep*"))
    if not hits:
        raise FileNotFoundError(f"no ncu report for tag {tag}")
    return hits[0]


def flatten(rep_path: Path, out_jsonl: Path) -> int:
    sys.path.insert(0, _ncu_python_dir())
    import ncu_report  # type: ignore

    ctx = ncu_report.load_report(str(rep_path))
    n = 0
    with open(out_jsonl, "w") as f:
        for r in range(ctx.num_ranges()):
            rng = ctx.range_by_idx(r)
            for a in range(rng.num_actions()):
                act = rng.action_by_idx(a)
                case = None
                st = act.nvtx_state()
                for dom_id in st.domains():
                    ranges = st.domain_by_id(dom_id).push_pop_ranges()
                    if ranges:
                        case = ranges[-1]
                rec = {"case": case, "range": r, "idx": a, "kernel": act.name()}
                for m in METRICS:
                    mv = act.metric_by_name(m)
                    if mv is None:
                        continue
                    try:
                        rec[m] = mv.as_double()
                    except Exception:
                        rec[m] = mv.as_string()
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--impl")
    ap.add_argument("--match")
    ap.add_argument("--exclude")
    ap.add_argument("--clock-control", default="none")
    ap.add_argument("--cache-control", default="all")
    ap.add_argument("--flatten-only", action="store_true")
    args = ap.parse_args()
    rep = _report_file(args.tag) if args.flatten_only else run_ncu(args)
    out = RESULTS_DIR / f"ncu_{args.tag}.jsonl"
    n = flatten(rep, out)
    print(f"wrote {n} kernel records -> {out}")


if __name__ == "__main__":
    main()
