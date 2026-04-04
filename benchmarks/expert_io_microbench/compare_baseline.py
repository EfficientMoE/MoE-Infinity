from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_EXPORT_JSON = Path("/tmp/nsys_export_t16.json")
OVERHEAD_THRESHOLD_PCT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing .nsys-rep NVTX baseline against benchmark JSON "
            "timings and report instrumentation overhead regression."
        )
    )
    parser.add_argument(
        "--nsys-report",
        required=True,
        help="Path to .nsys-rep profile used as baseline",
    )
    parser.add_argument(
        "--benchmark-json",
        required=True,
        help="Path to benchmark JSON containing new timing results",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write comparison JSON result",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_number(text: str) -> float | None:
    clean = text.replace(",", "").strip()
    if not clean:
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def _parse_nvtx_stats_text(raw_text: str) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Time (%)") or line.startswith("-"):
            continue
        cols = re.split(r"\s{2,}", line)
        if len(cols) < 10:
            continue
        pct = _to_number(cols[0])
        total_ns = _to_number(cols[1])
        instances = _to_number(cols[2])
        avg_ns = _to_number(cols[3])
        if (
            pct is None
            or total_ns is None
            or instances is None
            or avg_ns is None
        ):
            continue
        ranges.append(
            {
                "range": "  ".join(cols[9:]).strip(),
                "time_pct": pct,
                "total_time_ns": total_ns,
                "instances": int(instances),
                "avg_ns": avg_ns,
            }
        )
    return ranges


def collect_nvtx_ranges_from_stats(
    nsys_report: Path,
) -> tuple[list[dict[str, Any]], str]:
    attempts: tuple[str, ...] = ("nvtx_sum", "nvtxsum")
    diagnostics: list[str] = []
    for report_name in attempts:
        proc = _run(
            [
                "nsys",
                "stats",
                str(nsys_report),
                "--report",
                report_name,
            ]
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = _parse_nvtx_stats_text(combined)
        diagnostics.append(
            f"{report_name}: rc={proc.returncode}, parsed_ranges={len(parsed)}"
        )
        if proc.returncode == 0 and parsed:
            return parsed, "; ".join(diagnostics)
    return [], "; ".join(diagnostics)


def try_nsys_export_json(nsys_report: Path) -> dict[str, Any]:
    out_prefix = str(DEFAULT_EXPORT_JSON.with_suffix(""))
    proc = _run(
        [
            "nsys",
            "export",
            "--type=json",
            f"--output={out_prefix}",
            str(nsys_report),
        ]
    )
    exported_path = Path(f"{out_prefix}.json")
    exists = exported_path.exists()
    return {
        "command": (
            f"nsys export --type=json --output={out_prefix} {nsys_report}"
        ),
        "returncode": proc.returncode,
        "export_json": str(exported_path),
        "export_json_exists": exists,
    }


def _find_step_total_ns(value: Any) -> float | None:
    if isinstance(value, dict):
        direct = _to_float(value.get("step_total_ns"))
        if direct is not None and direct > 0.0:
            return direct

        for nested in value.values():
            maybe = _find_step_total_ns(nested)
            if maybe is not None:
                return maybe
    elif isinstance(value, list):
        for item in value:
            maybe = _find_step_total_ns(item)
            if maybe is not None:
                return maybe
    return None


def extract_benchmark_timing_ns(payload: dict[str, Any]) -> float:
    known_paths = (
        ("step_decomposition_mean", "step_total_ns"),
        ("scenarios", "bubble", "step_decomposition_mean", "step_total_ns"),
        ("bubble", "step_decomposition_mean", "step_total_ns"),
    )
    for path in known_paths:
        cur: Any = payload
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        maybe = _to_float(cur)
        if maybe is not None and maybe > 0.0:
            return maybe

    recursive = _find_step_total_ns(payload)
    if recursive is not None:
        return recursive

    raise ValueError(
        "Cannot find step_total_ns in benchmark JSON. "
        + "Expected key like step_decomposition_mean.step_total_ns"
    )


def choose_reference_range(ranges: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        item
        for item in ranges
        if isinstance(item.get("range"), str)
        and "DeepseekMoEBlock" in str(item.get("range"))
    ]
    if preferred:
        return max(preferred, key=lambda x: float(x.get("total_time_ns", 0.0)))
    return max(ranges, key=lambda x: float(x.get("total_time_ns", 0.0)))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        _ = f.write("\n")


def main() -> int:
    args = parse_args()
    nsys_report = Path(args.nsys_report)
    benchmark_json = Path(args.benchmark_json)
    output_json = Path(args.output_json)

    if shutil.which("nsys") is None:
        skipped = {"status": "skipped", "reason": "nsys not available"}
        write_json(output_json, skipped)
        return 0

    if not nsys_report.exists():
        raise FileNotFoundError(f"nsys report not found: {nsys_report}")
    if not benchmark_json.exists():
        raise FileNotFoundError(f"benchmark json not found: {benchmark_json}")

    bench_payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    if not isinstance(bench_payload, dict):
        raise ValueError("benchmark json root must be an object")

    baseline_ranges, stats_diag = collect_nvtx_ranges_from_stats(nsys_report)
    export_meta = try_nsys_export_json(nsys_report)

    if not baseline_ranges:
        result = {
            "status": "error",
            "reason": "failed to parse NVTX ranges from nsys stats output",
            "baseline_nvtx_ranges": [],
            "new_instrumentation_overhead_pct": None,
            "overhead_regression_pct": None,
            "verdict": "unknown",
            "stats_attempts": stats_diag,
            "export_attempt": export_meta,
        }
        write_json(output_json, result)
        return 1

    benchmark_timing_ns = extract_benchmark_timing_ns(bench_payload)
    reference = choose_reference_range(baseline_ranges)
    baseline_avg_ns = float(reference["avg_ns"])

    new_overhead_pct = (
        (benchmark_timing_ns - baseline_avg_ns) / baseline_avg_ns
    ) * 100.0
    overhead_regression_pct = max(new_overhead_pct, 0.0)
    verdict = (
        "pass" if overhead_regression_pct < OVERHEAD_THRESHOLD_PCT else "fail"
    )

    result = {
        "status": "ok",
        "baseline_nvtx_ranges": baseline_ranges,
        "baseline_reference_range": reference.get("range"),
        "baseline_reference_avg_ns": baseline_avg_ns,
        "benchmark_timing_ns": benchmark_timing_ns,
        "new_instrumentation_overhead_pct": new_overhead_pct,
        "overhead_regression_pct": overhead_regression_pct,
        "threshold_pct": OVERHEAD_THRESHOLD_PCT,
        "verdict": verdict,
        "stats_attempts": stats_diag,
        "export_attempt": export_meta,
    }
    write_json(output_json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
