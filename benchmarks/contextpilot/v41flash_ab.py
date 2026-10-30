"""Standalone A/B harness: ContextPilot benefit on DeepSeek-V4.1-Flash (mp1).

Measures CP-on vs CP-off across the 4 ContextPilot workload fixtures, on the
Path B V4.1-Flash offload loader (FP4 experts streamed through the V4 host
store, engram tables host-offloaded). ContextPilot Phase-B middleware
(reorder + dedup) is applied to the message list BEFORE the official DeepSeek
V4.1 chat encoding; both variants then run through the identical generate()
path. Single-process, single-GPU (mp1) — no torchrun.

Launch (repo root; see docs/deepseek-v41-flash.md for checkpoint prep):

    MOE_V41_FLASH_CKPT=/ckpt/DeepSeek-V4.1-Flash \
    MOE_V41_OFFICIAL_DIR=/path/to/official-v41 \
    CUDA_VISIBLE_DEVICES=0 python benchmarks/contextpilot/v41flash_ab.py \
        --workloads all --repeats 3 --max-new-tokens 64 \
        --max-resident-experts 16 --temperature 0 \
        --out benchmarks/contextpilot/results/v41flash_ab_<ts>.json

MOE_V41_OFFICIAL_DIR is the pinned official download root containing
``inference/`` and ``encoding/`` (revision in
moe-store wrappers/deepseek_v41/OFFICIAL_PIN.md).

TTFT method: no-image-edit prefill-time proxy (same as v4flash_ab.py).
  ttft  = time of a max_new_tokens=1 generate() (cuda-synced)
  e2e   = time of a max_new_tokens=N generate()
  decode_tok_s = (N-1) / (e2e - ttft)
Greedy decoding is forced through ``model.temperature`` (the reference
``generate()`` has no temperature argument; ``sample()`` arg-maxes at 0).
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportArgumentType=false, reportOptionalSubscript=false, reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportUnknownParameterType=false
import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import torch

CKPT_DIR = os.environ.get("MOE_V41_FLASH_CKPT", "/ckpt/DeepSeek-V4.1-Flash")
OFFICIAL_DIR = os.environ.get("MOE_V41_OFFICIAL_DIR", "/workspace/official-v41")

sys.path.insert(0, os.path.join(OFFICIAL_DIR, "inference"))
sys.path.insert(0, os.path.join(OFFICIAL_DIR, "encoding"))
import model as M  # noqa: E402
from encoding import encode_messages  # noqa: E402
from generate import generate  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)

from moe_store.wrappers.deepseek_v41 import (  # noqa: E402
    load_offloaded_v41_flash,
)

from benchmarks.contextpilot.benchmark_utils import (  # noqa: E402
    compute_percentiles,
)
from benchmarks.contextpilot.dataset_utils import (  # noqa: E402
    get_workload_names,
    load_workload,
)
from moe_infinity.serving.contextpilot_middleware import (  # noqa: E402
    ContextPilotMiddleware,
)

CONFIG_PATH = os.path.join(OFFICIAL_DIR, "inference", "config.json")


def assert_contextpilot_real() -> None:
    import contextpilot  # noqa: F401  (ImportError => abort)

    mw = ContextPilotMiddleware(
        enabled=True, dedup_enabled=True, reorder_enabled=True
    )
    assert mw.is_enabled(), "ContextPilot middleware not enabled — CP-on would be passthrough; ABORT."
    probe = [
        {"role": "system", "content": "X" * 400},
        {"role": "system", "content": "X" * 400},
        {"role": "user", "content": "summarize"},
    ]

    def _content_chars(msgs: list[dict[str, str]]) -> int:
        return sum(len(str(m.get("content", ""))) for m in msgs)

    before_chars = _content_chars(probe)
    out = mw.process_chat_request([dict(m) for m in probe])
    changed = (len(out) != len(probe)) or (_content_chars(out) != before_chars)
    assert changed, (
        "ContextPilot left a known-duplicate probe unchanged — CP-on would be "
        "a no-op (passthrough); ABORT."
    )


def encode_prompt(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    text = encode_messages(messages, thinking_mode="chat")
    return tokenizer.encode(text)


def _timed_generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int,
) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = generate(
        model,
        [prompt_ids],
        max_new_tokens=max_new_tokens,
        eos_id=eos_id,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure_one(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int,
) -> dict[str, float]:
    ttft = _timed_generate(model, prompt_ids, 1, eos_id)
    e2e = _timed_generate(model, prompt_ids, max_new_tokens, eos_id)
    decode_tokens = max(0, max_new_tokens - 1)
    decode_window = max(1e-6, e2e - ttft)
    return {
        "ttft": float(ttft),
        "e2e": float(e2e),
        "prompt_tokens": float(len(prompt_ids)),
        "decode_tok_s": float(decode_tokens / decode_window),
    }


def _load_workload_any(workload_name: str) -> list[dict[str, object]]:
    if workload_name.endswith(".json") or os.path.sep in workload_name:
        with open(workload_name, encoding="utf-8") as f:
            payload = json.load(f)
        return list(payload["requests"])
    return load_workload(workload_name)


def run_workload(
    model,
    tokenizer,
    mw: ContextPilotMiddleware,
    workload_name: str,
    repeats: int,
    max_new_tokens: int,
    eos_id: int,
) -> dict:
    requests = _load_workload_any(workload_name)

    per_request: list[dict] = []
    off_ttft: list[float] = []
    on_ttft: list[float] = []
    off_e2e: list[float] = []
    on_e2e: list[float] = []
    off_decode: list[float] = []
    on_decode: list[float] = []
    off_ptoks: list[float] = []
    on_ptoks: list[float] = []
    cp_overhead_ms: list[float] = []
    cp_savings_pct: list[float] = []

    for ri, request in enumerate(requests):
        messages = request["messages"]
        ids_off = encode_prompt(tokenizer, messages)
        messages_cp = mw.process_chat_request([dict(m) for m in messages])
        ids_on = encode_prompt(tokenizer, messages_cp)
        m = mw.get_last_request_metrics()
        req_overhead_ms = float(m["reorder_latency_ms"]) + float(
            m["dedup_latency_ms"]
        )
        req_savings = float(m["savings_pct"])

        _ = measure_one(model, ids_off, max_new_tokens, eos_id)  # warmup

        off_runs: list[dict] = []
        on_runs: list[dict] = []
        for _ in range(repeats):
            off_runs.append(measure_one(model, ids_off, max_new_tokens, eos_id))
            on_runs.append(measure_one(model, ids_on, max_new_tokens, eos_id))

        def med(runs, key):
            return float(median([r[key] for r in runs]))

        r_off = {
            k: med(off_runs, k)
            for k in ("ttft", "e2e", "decode_tok_s", "prompt_tokens")
        }
        r_on = {
            k: med(on_runs, k)
            for k in ("ttft", "e2e", "decode_tok_s", "prompt_tokens")
        }

        off_ttft.append(r_off["ttft"])
        on_ttft.append(r_on["ttft"])
        off_e2e.append(r_off["e2e"])
        on_e2e.append(r_on["e2e"])
        off_decode.append(r_off["decode_tok_s"])
        on_decode.append(r_on["decode_tok_s"])
        off_ptoks.append(r_off["prompt_tokens"])
        on_ptoks.append(r_on["prompt_tokens"])
        cp_overhead_ms.append(req_overhead_ms)
        cp_savings_pct.append(req_savings)

        per_request.append(
            {
                "index": ri,
                "cp_off": r_off,
                "cp_on": r_on,
                "cp_overhead_ms": req_overhead_ms,
                "cp_savings_pct": req_savings,
                "prompt_tokens_off": r_off["prompt_tokens"],
                "prompt_tokens_on": r_on["prompt_tokens"],
            }
        )

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    off_ttft_pct = compute_percentiles(off_ttft, pcts=(50, 90, 99))
    on_ttft_pct = compute_percentiles(on_ttft, pcts=(50, 90, 99))
    off_e2e_pct = compute_percentiles(off_e2e, pcts=(50, 90, 99))
    on_e2e_pct = compute_percentiles(on_e2e, pcts=(50, 90, 99))

    summary = {
        "cp_off": {
            "ttft_p50": off_ttft_pct["p50"],
            "ttft_p90": off_ttft_pct["p90"],
            "ttft_p99": off_ttft_pct["p99"],
            "e2e_p50": off_e2e_pct["p50"],
            "e2e_p90": off_e2e_pct["p90"],
            "e2e_p99": off_e2e_pct["p99"],
            "decode_tok_s": _mean(off_decode),
            "prompt_tokens_mean": _mean(off_ptoks),
        },
        "cp_on": {
            "ttft_p50": on_ttft_pct["p50"],
            "ttft_p90": on_ttft_pct["p90"],
            "ttft_p99": on_ttft_pct["p99"],
            "e2e_p50": on_e2e_pct["p50"],
            "e2e_p90": on_e2e_pct["p90"],
            "e2e_p99": on_e2e_pct["p99"],
            "decode_tok_s": _mean(on_decode),
            "prompt_tokens_mean": _mean(on_ptoks),
        },
        "cp_overhead_ms_mean": _mean(cp_overhead_ms),
        "cp_savings_pct_mean": _mean(cp_savings_pct),
        "num_requests": len(requests),
        "repeats": repeats,
        "per_request": per_request,
    }
    return summary


def _rel_change(before: float, after: float, lower_is_better: bool) -> float:
    if before <= 0.0:
        return 0.0
    return (
        ((before - after) / before * 100.0)
        if lower_is_better
        else ((after - before) / before * 100.0)
    )


def compute_delta(summary: dict) -> dict:
    off = summary["cp_off"]
    on = summary["cp_on"]
    return {
        "ttft_pct": _rel_change(off["ttft_p50"], on["ttft_p50"], True),
        "e2e_latency_pct": _rel_change(off["e2e_p50"], on["e2e_p50"], True),
        "prompt_tokens_pct": _rel_change(
            off["prompt_tokens_mean"], on["prompt_tokens_mean"], True
        ),
        "decode_tok_s_pct": _rel_change(
            off["decode_tok_s"], on["decode_tok_s"], False
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ContextPilot A/B on DeepSeek-V4.1-Flash (mp1)"
    )
    p.add_argument(
        "--workloads",
        default="all",
        help="'all' or comma-separated workload names",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-resident-experts", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--repeats and --max-new-tokens must be > 0")

    assert_contextpilot_real()

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    device = torch.device("cuda", 0)

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    model, store = load_offloaded_v41_flash(
        M,
        ckpt_path=CKPT_DIR,
        config_path=CONFIG_PATH,
        device=device,
        max_resident_experts=args.max_resident_experts,
        text_only=True,
        tokenizer=tokenizer,
    )
    # The reference generate() samples inside Transformer.forward with
    # model.temperature; 0 arg-maxes (greedy).
    model.temperature = float(args.temperature)
    torch.set_default_device(device)
    torch.set_grad_enabled(False)

    eos_id = tokenizer.eos_token_id

    mw = ContextPilotMiddleware(
        enabled=True, dedup_enabled=True, reorder_enabled=True
    )

    if args.workloads.strip().lower() == "all":
        names = get_workload_names()
    else:
        names = [w.strip() for w in args.workloads.split(",") if w.strip()]

    results: dict[str, dict] = {}
    delta_pct: dict[str, dict] = {}
    for name in names:
        summary = run_workload(
            model,
            tokenizer,
            mw,
            name,
            repeats=args.repeats,
            max_new_tokens=args.max_new_tokens,
            eos_id=eos_id,
        )
        results[name] = summary
        delta_pct[name] = compute_delta(summary)

    ttft_gain_overlap_workloads = [
        "shared_prefix_rag",
        "batch_with_overlap",
    ]
    overlap_ttft_gains = [
        delta_pct[w]["ttft_pct"]
        for w in ttft_gain_overlap_workloads
        if w in delta_pct
    ]
    no_overlap_ttft_regression = delta_pct.get("no_overlap_baseline", {}).get(
        "ttft_pct", 0.0
    )
    go = bool(
        overlap_ttft_gains
        and all(g > 10.0 for g in overlap_ttft_gains)
        and no_overlap_ttft_regression > -2.0
    )
    payload = {
        "mode": "real",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "world_size": 1,
        "max_resident_experts": args.max_resident_experts,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "ttft_method": "prefill_proxy_max_new_tokens_1",
        "resident_experts": len(store.resident_experts()),
        "workloads": results,
        "delta_pct": delta_pct,
        "go_no_go": go,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"RESULT written {out_path}", flush=True)
    for name in names:
        d = delta_pct[name]
        print(
            f"RESULT {name} ttft_dpct={d['ttft_pct']:.1f} "
            f"ptok_dpct={d['prompt_tokens_pct']:.1f} "
            f"e2e_dpct={d['e2e_latency_pct']:.1f} "
            f"decode_dpct={d['decode_tok_s_pct']:.1f}",
            flush=True,
        )
    print(f"RESULT GO_NO_GO {'GO' if go else 'NO-GO'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
