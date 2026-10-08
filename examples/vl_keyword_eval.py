"""MoE-Infinity VL keyword-accuracy harness (issue #221 phase 4).

Reads a paths-only JSON fixture of image/prompt/keyword rows and scores how
many model answers contain their keyword. Real mode lazily loads
``AutoProcessor`` and ``MoE`` and follows ``examples/vl_example.py``'s greedy
generate path. Fake mode (``--fake-answers``) injects answers and timings for
CPU tests and imports no torch, transformers, or moe_infinity code, so the
score gate and the machine-readable ``REQUEST_PERF`` / ``KEYWORD_SCORE`` /
``PERF_SUMMARY`` lines can be exercised without a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_KEYS = ("id", "image", "prompt", "keyword")
GATE_FRACTION = 0.8


def parse_args(argv=None):
    default_offload = os.path.join(os.path.expanduser("~"), "moe-infinity-vl")
    parser = argparse.ArgumentParser(
        description=(
            "Score VL keyword accuracy for the MiniMax-M3 BF16 eager "
            "baseline (issue #221 phase 4)."
        )
    )
    parser.add_argument(
        "--fixture",
        required=True,
        help="Path to the paths-only image/prompt/keyword JSON fixture",
    )
    parser.add_argument(
        "--fake-answers",
        default=None,
        help=(
            "JSON object of fixture-id to answer string. Enables CPU fake "
            "mode and skips all model loading."
        ),
    )
    parser.add_argument(
        "--fake-timings",
        default=None,
        help=(
            "JSON object of fixture-id to {ttft_ms, decode_tokens_per_s}. "
            "Only valid together with --fake-answers."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="MiniMaxAI/MiniMax-M3",
        help="HuggingFace VL checkpoint used in real mode",
    )
    parser.add_argument(
        "--offload_dir",
        default=default_offload,
        help="Directory for offloading expert weights",
    )
    parser.add_argument(
        "--device_memory_ratio",
        type=float,
        default=0.5,
        help="Fraction of device memory used for expert caching",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per request",
    )
    return parser.parse_args(argv)


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def load_fixture(path: str) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("fixture must be a JSON array")
    if not data:
        raise ValueError("fixture must contain at least one row")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each fixture row must be a JSON object")
        if set(item.keys()) != set(FIXTURE_KEYS):
            raise ValueError(
                "each fixture row must have exactly the keys "
                "id, image, prompt, keyword"
            )
        for key in FIXTURE_KEYS:
            value = item[key]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"fixture field {key!r} must be a non-empty string"
                )
        if item["id"] in seen:
            raise ValueError(f"duplicate fixture id {item['id']!r}")
        seen.add(item["id"])
        rows.append(item)
    return rows


def load_fake_answers(raw: str) -> dict[str, str]:
    answers = json.loads(raw)
    if not isinstance(answers, dict):
        raise ValueError("--fake-answers must be a JSON object")
    for key, value in answers.items():
        if not isinstance(value, str):
            raise ValueError(
                f"--fake-answers value for {key!r} must be a string"
            )
    return answers


def load_fake_timings(
    raw: str, rows: list[dict[str, str]]
) -> dict[str, dict[str, float]]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--fake-timings must be a JSON object")

    timings: dict[str, dict[str, float]] = {}
    for row in rows:
        entry = data.get(row["id"])
        if not isinstance(entry, dict):
            raise ValueError(
                f"--fake-timings missing entry for id {row['id']!r}"
            )
        values: dict[str, float] = {}
        for field in ("ttft_ms", "decode_tokens_per_s"):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"--fake-timings {field} for {row['id']!r} "
                    "must be a number"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"--fake-timings {field} for {row['id']!r} "
                    "must be finite and non-negative"
                )
            values[field] = number
        timings[row["id"]] = values
    return timings


def is_hit(keyword: str, answer: str) -> bool:
    return keyword.lower() in answer.lower()


def emit_request_perf(row_id: str, ttft_ms: float, decode_tps: float) -> None:
    print(
        f"REQUEST_PERF id={row_id} "
        f"ttft_ms={ttft_ms:.3f} "
        f"decode_tokens_per_s={decode_tps:.3f}"
    )


def emit_perf_summary(summary: dict[str, float | int]) -> None:
    print(
        "PERF_SUMMARY "
        f"sample_count={summary['sample_count']} "
        f"ttft_p50_ms={summary['ttft_p50_ms']:.3f} "
        f"ttft_p99_ms={summary['ttft_p99_ms']:.3f} "
        f"decode_tok_s_p50={summary['decode_tok_s_p50']:.3f} "
        f"decode_tok_s_p99={summary['decode_tok_s_p99']:.3f}"
    )


def summarize(
    ttft_samples: list[float], decode_samples: list[float]
) -> dict[str, float | int]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from benchmarks.minimax_m3.perf_summary import summarize_performance

    return summarize_performance(ttft_samples, decode_samples)


def run_fake_mode(
    rows: list[dict[str, str]],
    gate: int,
    answers_raw: str,
    timings_raw: str | None,
) -> int:
    try:
        answers = load_fake_answers(answers_raw)
    except (ValueError, json.JSONDecodeError) as error:
        return fail(f"invalid --fake-answers: {error}")

    timings = None
    if timings_raw is not None:
        try:
            timings = load_fake_timings(timings_raw, rows)
        except (ValueError, json.JSONDecodeError) as error:
            return fail(f"invalid --fake-timings: {error}")

    ttft_samples: list[float] = []
    decode_samples: list[float] = []
    score = 0
    for row in rows:
        answer = answers.get(row["id"], "")
        if is_hit(row["keyword"], answer):
            score += 1
        if timings is not None:
            entry = timings[row["id"]]
            emit_request_perf(
                row["id"], entry["ttft_ms"], entry["decode_tokens_per_s"]
            )
            ttft_samples.append(entry["ttft_ms"])
            decode_samples.append(entry["decode_tokens_per_s"])

    print(f"KEYWORD_SCORE {score}/{len(rows)}")
    if timings is not None:
        emit_perf_summary(summarize(ttft_samples, decode_samples))
    return 0 if score >= gate else 1


class _TimingStreamer:
    def __init__(self) -> None:
        self.first_token_at: float | None = None
        self._seen_prompt = False

    def put(self, value) -> None:
        _ = value
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def end(self) -> None:
        return None


def generate_one(model, processor, row, args, torch):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": row["image"]},
                {"type": "text", "text": row["prompt"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = inputs["input_ids"].to("cuda:0")
    generate_kwargs = {
        key: (value.to("cuda:0") if isinstance(value, torch.Tensor) else value)
        for key, value in inputs.items()
        if key != "input_ids"
    }
    if "pixel_values" in generate_kwargs:
        generate_kwargs["pixel_values"] = generate_kwargs["pixel_values"].to(
            torch.bfloat16
        )

    streamer = _TimingStreamer()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=processor.tokenizer.eos_token_id,
            **generate_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    answer = processor.tokenizer.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    generated = int(output_ids.shape[-1] - input_ids.shape[1])
    if streamer.first_token_at is not None:
        ttft_ms = (streamer.first_token_at - start) * 1000.0
    else:
        ttft_ms = (end - start) * 1000.0
    if generated >= 2 and streamer.first_token_at is not None:
        decode_tps = (generated - 1) / (end - streamer.first_token_at)
    else:
        decode_tps = 0.0
    return answer, ttft_ms, decode_tps


def report_memory(torch) -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        peak = int(torch.cuda.max_memory_allocated(index))
        print(f"GPU_MAX_MEMORY device={index} bytes={peak}")


def run_real_mode(
    rows: list[dict[str, str]], gate: int, args: argparse.Namespace
) -> int:
    import torch
    from transformers import AutoProcessor

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from moe_infinity import MoE

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model = MoE(
        args.checkpoint,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
        },
    )

    ttft_samples: list[float] = []
    decode_samples: list[float] = []
    score = 0
    for row in rows:
        if not Path(row["image"]).exists():
            print(f"SKIP id={row['id']} reason=missing_image")
            continue
        answer, ttft_ms, decode_tps = generate_one(
            model, processor, row, args, torch
        )
        print(f"ANSWER id={row['id']}: {answer}")
        if is_hit(row["keyword"], answer):
            score += 1
        emit_request_perf(row["id"], ttft_ms, decode_tps)
        ttft_samples.append(ttft_ms)
        decode_samples.append(decode_tps)

    report_memory(torch)
    print(f"KEYWORD_SCORE {score}/{len(rows)}")
    if ttft_samples:
        emit_perf_summary(summarize(ttft_samples, decode_samples))
    return 0 if score >= gate else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fake_timings is not None and args.fake_answers is None:
        return fail("--fake-timings requires --fake-answers")

    try:
        rows = load_fixture(args.fixture)
    except (ValueError, json.JSONDecodeError, OSError) as error:
        return fail(f"invalid fixture: {error}")

    gate = math.ceil(GATE_FRACTION * len(rows))
    if args.fake_answers is not None:
        return run_fake_mode(rows, gate, args.fake_answers, args.fake_timings)
    return run_real_mode(rows, gate, args)


if __name__ == "__main__":
    raise SystemExit(main())
