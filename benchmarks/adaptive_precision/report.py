from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REQUIRED = {
    "mode",
    "model",
    "checkpoint_fingerprint",
    "format",
    "converter_version",
    "quality_attestation_sha256",
    "hardware",
    "software",
    "workload",
    "quality",
    "memory",
    "transfer",
    "latency",
    "throughput",
    "policy",
}


def validate_run(row: dict) -> None:
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"missing benchmark fields: {sorted(missing)}")
    if (
        len(row["checkpoint_fingerprint"]) != 64
        or len(row["quality_attestation_sha256"]) != 64
    ):
        raise ValueError("invalid digest")
    if row["memory"]["peak_accounted_bytes"] > row["memory"]["budget_bytes"]:
        raise ValueError("adaptive budget exceeded")
    for key in ("tpot_ms_p50", "tpot_ms_p90", "tpot_ms_p99"):
        if not math.isfinite(float(row["latency"][key])):
            raise ValueError("nonfinite TPOT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line
    ]
    for row in rows:
        validate_run(row)
    Path(args.output).write_text(
        json.dumps(
            {"runs": len(rows), "release_gate": "candidate-only"},
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
