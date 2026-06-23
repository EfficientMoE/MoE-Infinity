from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import argparse
import json
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reduce v4flash_ab.py results JSON to a markdown report + GO/NO-GO."
    )
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    return p.parse_args()


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def build_report(payload: dict[str, object]) -> tuple[str, bool, bool]:
    workloads = cast("dict[str, dict[str, Any]]", payload.get("workloads", {}))
    deltas = cast("dict[str, dict[str, Any]]", payload.get("delta_pct", {}))
    go = bool(payload.get("go_no_go", False))

    lines: list[str] = []
    lines.append("# ContextPilot on DeepSeek-V4-Flash — A/B results")
    lines.append("")
    lines.append(
        f"- model: `{payload.get('model')}`  |  world_size: {payload.get('world_size')}  "
        f"|  resident_experts(max): {payload.get('max_resident_experts')}"
    )
    lines.append(
        f"- max_new_tokens: {payload.get('max_new_tokens')}  |  repeats: "
        f"{payload.get('repeats')}  |  temperature: {payload.get('temperature')}"
    )
    lines.append(f"- ttft_method: `{payload.get('ttft_method')}`")
    lines.append("")
    lines.append(
        "| workload | TTFT p50 off→on (s) | TTFT Δ% | prompt-tok Δ% | "
        "E2E p50 Δ% | decode tok/s off→on | decode Δ% | CP overhead (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")

    decode_warn = False
    for name, w in workloads.items():
        off = w["cp_off"]
        on = w["cp_on"]
        d = deltas.get(name, {})
        decode_dpct = float(d.get("decode_tok_s_pct", 0.0))
        if abs(decode_dpct) > 3.0:
            decode_warn = True
        lines.append(
            f"| {name} "
            f"| {_fmt(off['ttft_p50'], 3)}→{_fmt(on['ttft_p50'], 3)} "
            f"| {_fmt(d.get('ttft_pct', 0.0))} "
            f"| {_fmt(d.get('prompt_tokens_pct', 0.0))} "
            f"| {_fmt(d.get('e2e_latency_pct', 0.0))} "
            f"| {_fmt(off['decode_tok_s'], 2)}→{_fmt(on['decode_tok_s'], 2)} "
            f"| {_fmt(decode_dpct)} "
            f"| {_fmt(w.get('cp_overhead_ms_mean', 0.0), 2)} |"
        )

    lines.append("")
    lines.append(f"**Verdict: {'GO' if go else 'NO-GO'}**")
    lines.append("")
    lines.append(
        "GO criterion (plan §2): TTFT p50 improves >10% on both overlap workloads "
        "(shared_prefix_rag, batch_with_overlap) AND no_overlap_baseline regresses <2%."
    )
    if decode_warn:
        lines.append("")
        lines.append(
            "WARN: decode contamination — decode tok/s differs >3% between CP-on/off; "
            "CP should not affect decode. Investigate measurement noise."
        )

    return "\n".join(lines) + "\n", go, decode_warn


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    report, go, decode_warn = build_report(payload)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    if decode_warn:
        print("WARN: decode contamination")
    print("GO" if go else "NO-GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
