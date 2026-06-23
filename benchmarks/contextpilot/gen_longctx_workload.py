from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import argparse
import json
import random
from pathlib import Path
from typing import Any, cast

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_TOPICS = [
    "distributed storage replication and consistency",
    "incident response and on-call rotation policy",
    "data retention, classification, and audit logging",
    "vendor risk scoring and procurement controls",
    "model evaluation, experiment tracking, and uncertainty reporting",
    "network exposure mapping and breach notification timing",
    "capacity planning for refrigerated logistics under disruption",
    "compliance escalation and duty-executive decision logging",
]

_SENTENCES = [
    "Teams must classify each record by sensitivity tier before it enters any shared workspace.",
    "Unresolved regulatory risk during an active incident escalates to the duty executive within one business day.",
    "Severity-1 incidents require immediate paging of on-call engineering and a legal contact when confidentiality is implicated.",
    "Payment corrections retain the original invoice identifier, rationale text, and final settlement timestamp.",
    "Approved vendors accept quarterly control attestations and emergency revocation terms.",
    "Status updates are issued every thirty minutes during major incidents with affected services and next-update time.",
    "Backup retention guarantees and documented restore drills are verified on a fixed quarterly cadence.",
    "Decision owners are recorded for every exception with an explicit accept or reject outcome in the log.",
]


def _make_shared_document(target_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    parts = ["Shared Reference Corpus (v7.4). "]
    n = 0
    while sum(len(p) for p in parts) < target_chars:
        topic = rng.choice(_TOPICS)
        sent = rng.choice(_SENTENCES)
        parts.append(f"[Section {n}: {topic}] {sent} ")
        n += 1
    return "".join(parts)


def _make_unique_tail(idx: int, target_chars: int, seed: int) -> str:
    rng = random.Random(seed + idx * 1000)
    parts = [f"[Request-specific addendum {idx}] "]
    while sum(len(p) for p in parts) < target_chars:
        parts.append(rng.choice(_SENTENCES) + " ")
    return "".join(parts)


def build_workload(
    name: str,
    num_requests: int,
    shared_chars: int,
    unique_chars: int,
    seed: int,
) -> dict[str, object]:
    shared_doc = _make_shared_document(shared_chars, seed)
    requests = []
    for i in range(num_requests):
        unique = _make_unique_tail(i, unique_chars, seed)
        messages = [
            {"role": "system", "content": shared_doc},
            {"role": "system", "content": unique},
            {
                "role": "user",
                "content": (
                    f"Using the shared reference corpus and addendum {i}, "
                    "summarize the mandatory controls and the single most "
                    "important escalation rule."
                ),
            },
        ]
        total_chars = sum(len(m["content"]) for m in messages)
        requests.append(
            {
                "messages": messages,
                "expected_token_count": total_chars // 4,
                "context_overlap_with_prev": (
                    0.0
                    if i == 0
                    else round(shared_chars / max(1, total_chars), 3)
                ),
            }
        )
    overlap_ratio = round(shared_chars / max(1, shared_chars + unique_chars), 3)
    return {
        "metadata": {
            "name": name,
            "overlap_ratio": overlap_ratio,
            "request_count": num_requests,
            "shared_chars": shared_chars,
            "unique_chars": unique_chars,
            "approx_shared_tokens": shared_chars // 4,
            "description": (
                "Long-context shared-prefix workload: every request shares one "
                "large reference corpus prefix plus a small unique addendum. "
                "Designed to expose cross-request prefix-cache reuse benefit."
            ),
        },
        "requests": requests,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a long-context overlap workload fixture."
    )
    p.add_argument("--name", default="longctx_shared_prefix")
    p.add_argument("--num-requests", type=int, default=6)
    p.add_argument(
        "--shared-tokens",
        type=int,
        default=120000,
        help="approx shared-prefix tokens (chars = 4x)",
    )
    p.add_argument("--unique-tokens", type=int, default=2000)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workload = build_workload(
        name=args.name,
        num_requests=args.num_requests,
        shared_chars=args.shared_tokens * 4,
        unique_chars=args.unique_tokens * 4,
        seed=args.seed,
    )
    out = Path(args.out) if args.out else _FIXTURES_DIR / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(workload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    md = cast("dict[str, Any]", workload["metadata"])
    print(
        f"wrote {out} | {md['request_count']} reqs | "
        f"~{md['approx_shared_tokens']} shared tok | overlap {md['overlap_ratio']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
