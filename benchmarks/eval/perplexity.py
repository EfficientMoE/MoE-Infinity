#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument(
        "--dataset", default="wikitext", choices=["wikitext", "c4", "ptb"]
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_dataset(dataset: str, split: str) -> tuple[str, str | None, str]:
    if dataset == "wikitext":
        return "wikitext", "wikitext-2-raw-v1", split
    if dataset == "c4":
        return "allenai/c4", "en", "validation" if split == "test" else split
    if dataset == "ptb":
        return "ptb_text_only", "penn_treebank", split
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_texts(dataset: str, split: str, max_samples: int) -> list[str]:
    dataset_name, dataset_config, dataset_split = resolve_dataset(
        dataset, split
    )
    if dataset_config is None:
        raw_dataset = load_dataset(dataset_name, split=dataset_split)
    else:
        raw_dataset = load_dataset(
            dataset_name, dataset_config, split=dataset_split
        )

    texts: list[str] = []
    for row in raw_dataset:
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def _model_forward(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    target = getattr(model, "model", model)
    outputs = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise RuntimeError("Model forward pass did not return logits")


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    dataset: list[str],
    seq_length: int,
    max_samples: int,
    batch_size: int,
) -> tuple[float, float, int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if seq_length <= 1:
        raise ValueError(f"seq_length must be > 1, got {seq_length}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = getattr(model, "model", model)
    if hasattr(target, "eval"):
        target.eval()

    total_nll = 0.0
    total_tokens = 0
    processed = 0

    with torch.inference_mode():
        for start in range(0, min(len(dataset), max_samples), batch_size):
            batch_texts = dataset[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=seq_length + 1,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            if input_ids.shape[1] < 2:
                continue

            logits = _model_forward(model, input_ids, attention_mask)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid_tokens = int((shift_labels != -100).sum().item())
            if valid_tokens > 0:
                total_nll += float(loss.item())
                total_tokens += valid_tokens
                processed += len(batch_texts)

    if total_tokens <= 0:
        raise RuntimeError(
            "No valid tokens were processed for perplexity evaluation"
        )

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    return ppl, mean_nll, processed


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    model = MoE(args.model, {"offload_path": args.offload_dir})
    texts = load_texts(args.dataset, args.split, args.max_samples)

    start_time = time.perf_counter()
    ppl, nll, n = evaluate_perplexity(
        model,
        tokenizer,
        texts,
        args.seq_length,
        args.max_samples,
        args.batch_size,
    )
    elapsed_s = time.perf_counter() - start_time

    print(f"Perplexity: {ppl:.4f} (NLL: {nll:.4f}, samples: {n})")

    if args.output_json:
        write_json(
            Path(args.output_json),
            {
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "max_samples": args.max_samples,
                "seq_length": args.seq_length,
                "batch_size": args.batch_size,
                "perplexity": ppl,
                "nll": nll,
                "samples": n,
                "elapsed_s": elapsed_s,
                "offload_dir": args.offload_dir,
            },
        )


if __name__ == "__main__":
    main()
