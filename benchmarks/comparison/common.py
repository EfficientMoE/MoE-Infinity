from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import cast

# TTFT = Time from submitting a prompt until the first generated token is produced.
# ITL = Average time between consecutive generated tokens during the decode phase.


@dataclass
class BenchmarkConfig:
    model_id: str
    model_name: str
    precision: str
    max_new_tokens: int = 128
    num_warmup: int = 5
    num_measured: int = 20
    prompt_dataset: str = "mixed"
    output_dir: str = "results/"


@dataclass
class BenchmarkResult:
    model: str
    framework: str
    precision: str
    per_token_latency_s: float | None
    ttft_s: float | None
    peak_gpu_mb: float | None
    num_iterations: int
    timestamp: str
    gpu_name: str
    notes: str = ""


MODEL_CONFIGS: dict[str, str] = {
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


PROMPT_DATASET: list[str] = [
    "What is the capital of Japan?",
    "Who wrote Pride and Prejudice?",
    "What is the chemical symbol for gold?",
    "How many days are there in a leap year?",
    "What does CPU stand for?",
    "If a train travels 120 kilometers in 2 hours, what is its average speed in kilometers per hour? Show the reasoning clearly.",
    "A rectangle has a length of 14 and a width of 9. What is its area, and how do you compute it step by step?",
    "Explain why dividing by zero is undefined in ordinary arithmetic, using a simple example.",
    "A store discounts a $50 item by 20 percent and then applies a 10 percent tax. What is the final price?",
    "Summarize the difference between supervised learning and unsupervised learning in machine learning.",
    "A jar contains 3 red, 4 blue, and 5 green marbles. If one marble is chosen at random, what is the probability it is not blue? Explain the calculation.",
    "Describe one practical strategy to reduce memory usage in a large language model during inference and explain why it helps.",
    "You are given the equation 2x + 7 = 19. Solve for x and explain each transformation.",
    "Read the passage and answer the question: A small town built a new bridge after repeated floods cut off the only road to the hospital. The bridge reduced emergency travel time by 18 minutes and stayed open during the next storm season. Why was the bridge important?",
    "Read the following paragraph and provide a concise summary: Researchers tested two scheduling policies for a shared GPU cluster. The first policy maximized average throughput but occasionally caused long tail latencies. The second policy slightly reduced throughput, yet it kept request completion times more consistent across users. The team concluded that the best policy depended on whether the workload prioritized fairness or raw throughput.",
    "A student claims that adding more training data always guarantees better generalization. Explain why this claim can be false, and mention at least two factors that affect generalization.",
    "A company wants to choose between renting one very large GPU and two smaller GPUs for model serving. Compare the trade-offs in cost, utilization, failure risk, and operational complexity.",
    "Read the passage and answer the question: During a week-long field study, a sensor network in a forest recorded temperature, humidity, and wind every ten minutes. Halfway through the study, several nodes lost power, but the remaining nodes continued transmitting enough data for the scientists to estimate daily trends. What does this suggest about the system design?",
    "Read the following article excerpt and summarize the key argument in two sentences: The report argues that benchmark numbers are only meaningful when the evaluation setup matches real deployment conditions. It warns that cherry-picked prompts, short runs, and hidden configuration changes can make one system appear faster than another without reflecting actual user experience.",
    "A long-form answer is needed: explain how caching can improve performance in both web applications and language-model inference, and compare what is being reused in each case.",
]


def get_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if output:
            return output.splitlines()[0].strip()
    except Exception:
        pass

    return "Unknown"


def save_result(result: BenchmarkResult, output_dir: str) -> pathlib.Path:
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{result.framework}_{result.model}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(result), f, indent=2)
        _ = f.write("\n")
    return file_path


def load_results(results_dir: str) -> list[BenchmarkResult]:
    path = pathlib.Path(os.fspath(results_dir))
    if not path.exists():
        return []

    results: list[BenchmarkResult] = []
    for file_path in sorted(path.glob("*.json")):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = cast(dict[str, object], json.load(f))
            results.append(
                BenchmarkResult(
                    model=cast(str, payload["model"]),
                    framework=cast(str, payload["framework"]),
                    precision=cast(str, payload["precision"]),
                    per_token_latency_s=cast(
                        float | None, payload["per_token_latency_s"]
                    ),
                    ttft_s=cast(float | None, payload["ttft_s"]),
                    peak_gpu_mb=cast(float | None, payload["peak_gpu_mb"]),
                    num_iterations=cast(int, payload["num_iterations"]),
                    timestamp=cast(str, payload["timestamp"]),
                    gpu_name=cast(str, payload["gpu_name"]),
                    notes=cast(str, payload.get("notes", "")),
                )
            )
        except Exception as exc:
            print(f"Skipping invalid result file {file_path}: {exc}")
    return results
