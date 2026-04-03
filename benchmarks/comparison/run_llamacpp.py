#!/usr/bin/env python3
"""llama.cpp benchmark script for the comparison suite."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false, reportImplicitOverride=false, reportDeprecated=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import requests
from huggingface_hub import hf_hub_download

try:
    from benchmarks.comparison.common import (
        MODEL_CONFIGS,
        PROMPT_DATASET,
        BenchmarkResult,
        get_gpu_name,
        save_result,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        MODEL_CONFIGS,
        PROMPT_DATASET,
        BenchmarkResult,
        get_gpu_name,
        save_result,
    )


GGUF_MODELS: Dict[str, Optional[Dict[str, str]]] = {
    "deepseek-v2-lite": {
        "repo": "mradermacher/DeepSeek-V2-Lite-Chat-GGUF",
        "filename": "DeepSeek-V2-Lite-Chat.Q4_K_M.gguf",
    },
    "mixtral-8x7b": {
        "repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "filename": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    },
    "qwen3-30b": {
        "repo": "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF",
        "filename": "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    },
    "gpt-oss-20b": None,
}

LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
HEALTH_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/health"
COMPLETION_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/completion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run llama.cpp benchmarks for comparison suite."
    )
    parser.add_argument(
        "--model",
        choices=[
            "deepseek-v2-lite",
            "mixtral-8x7b",
            "qwen3-30b",
            "gpt-oss-20b",
            "all",
        ],
        default="all",
        help="Model key to benchmark, or 'all' for all models.",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="Number of layers to offload to GPU for llama.cpp.",
    )
    parser.add_argument(
        "--output-dir",
        default="/results",
        help="Directory to write JSON benchmark results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens generated per measured prompt.",
    )
    parser.add_argument(
        "--llama-server",
        default="/llama-server",
        help="Path to llama-server binary.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default="/models",
        help="Directory where GGUF files are cached.",
    )
    return parser.parse_args()


def download_gguf(model_name: str, cache_dir: str) -> Optional[Path]:
    gguf_info = GGUF_MODELS.get(model_name)
    if gguf_info is None:
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=gguf_info["repo"],
        filename=gguf_info["filename"],
        local_dir=str(cache_path),
    )
    return Path(local_path)


def _empty_result(model_name: str, notes: str) -> BenchmarkResult:
    return BenchmarkResult(
        model=model_name,
        framework="llamacpp",
        precision="Q4_K_M",
        per_token_latency_s=None,
        ttft_s=None,
        peak_gpu_mb=None,
        num_iterations=len(PROMPT_DATASET),
        timestamp=datetime.now().isoformat(),
        gpu_name=get_gpu_name(),
        notes=notes,
    )


def _wait_for_server_ready(
    server_proc: subprocess.Popen[Any], timeout_s: int = 120
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_proc.poll() is not None:
            return False
        try:
            response = requests.get(HEALTH_URL, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def _query_peak_gpu_mb() -> Optional[float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not output:
            return None
        values = [
            float(line.strip()) for line in output.splitlines() if line.strip()
        ]
        if not values:
            return None
        return max(values)
    except Exception:
        return None


def _completion_request(prompt: str, n_predict: int) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": int(n_predict),
        "temperature": 0,
    }
    start = time.perf_counter()
    response = requests.post(COMPLETION_URL, json=payload, timeout=300)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("llama-server /completion returned non-object JSON")

    timings = data.get("timings")
    ttft_s: Optional[float] = None
    per_token_latency_s: Optional[float] = None

    if isinstance(timings, dict):
        prompt_ms = timings.get("prompt_ms")
        predicted_per_token_ms = timings.get("predicted_per_token_ms")
        predicted_ms = timings.get("predicted_ms")

        if isinstance(prompt_ms, (int, float)):
            ttft_s = float(prompt_ms) / 1000.0
        if isinstance(predicted_per_token_ms, (int, float)):
            per_token_latency_s = float(predicted_per_token_ms) / 1000.0
        elif isinstance(predicted_ms, (int, float)) and n_predict > 0:
            per_token_latency_s = (
                float(predicted_ms) / 1000.0 / float(n_predict)
            )

    if ttft_s is None:
        ttft_s = elapsed
    if per_token_latency_s is None:
        per_token_latency_s = elapsed / float(max(n_predict, 1))

    return {
        "ttft_s": ttft_s,
        "per_token_latency_s": per_token_latency_s,
    }


def _terminate_server(server_proc: subprocess.Popen[Any]) -> None:
    if server_proc.poll() is not None:
        return
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        try:
            server_proc.wait(timeout=5)
        except Exception as e:
            print(f"Server cleanup warning: {e}", file=sys.stderr)


def run_single_model(
    model_name: str, args: argparse.Namespace
) -> BenchmarkResult:
    gguf_info = GGUF_MODELS.get(model_name)
    if gguf_info is None:
        return _empty_result(
            model_name,
            "No GGUF model available for gpt-oss-20b",
        )

    try:
        gguf_path = download_gguf(model_name, args.model_cache_dir)
    except Exception as exc:
        return _empty_result(model_name, f"Failed to download GGUF: {exc}")

    if gguf_path is None:
        return _empty_result(
            model_name, "Failed to download GGUF: unknown error"
        )

    server_proc: Optional[subprocess.Popen[Any]] = None

    try:
        server_proc = subprocess.Popen(
            [
                args.llama_server,
                "-m",
                str(gguf_path),
                "-ngl",
                str(args.ngl),
                "-c",
                "2048",
                "--port",
                str(LLAMA_PORT),
                "--host",
                LLAMA_HOST,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not _wait_for_server_ready(server_proc, timeout_s=120):
            return _empty_result(model_name, "Server failed to start")

        for _ in range(5):
            try:
                _completion_request("Hello", 8)
            except Exception as e:
                print(
                    f"Warmup request failed (non-fatal): {e}",
                    file=sys.stderr,
                )

        ttft_values: List[float] = []
        per_token_values: List[float] = []
        for prompt in PROMPT_DATASET:
            timing = _completion_request(prompt, args.max_new_tokens)
            ttft_values.append(float(timing["ttft_s"]))
            per_token_values.append(float(timing["per_token_latency_s"]))

        return BenchmarkResult(
            model=model_name,
            framework="llamacpp",
            precision="Q4_K_M",
            per_token_latency_s=mean(per_token_values),
            ttft_s=mean(ttft_values),
            peak_gpu_mb=_query_peak_gpu_mb(),
            num_iterations=len(PROMPT_DATASET),
            timestamp=datetime.now().isoformat(),
            gpu_name=get_gpu_name(),
            notes="",
        )
    except Exception as exc:
        error_text = str(exc)
        if "out of memory" in error_text.lower() or "oom" in error_text.lower():
            return _empty_result(model_name, "OOM")
        return _empty_result(
            model_name, f"Benchmark failed: {type(exc).__name__}: {exc}"
        )
    finally:
        if server_proc is not None:
            _terminate_server(server_proc)


def _result_kind(result: BenchmarkResult) -> str:
    if not result.notes:
        return "success"
    note_lower = result.notes.lower()
    if "no gguf" in note_lower or "failed to download gguf" in note_lower:
        return "unavailable"
    if "oom" in note_lower or "server failed to start" in note_lower:
        return "oom"
    return "runtime"


def main() -> int:
    args = parse_args()

    if args.model == "all":
        model_names = list(MODEL_CONFIGS.keys())
    else:
        model_names = [args.model]

    results: List[BenchmarkResult] = []
    for model_name in model_names:
        result = run_single_model(model_name, args)
        output_path = save_result(result, args.output_dir)
        results.append(result)
        print(
            f"[{model_name}] framework={result.framework} "
            f"ttft_s={result.ttft_s} "
            f"per_token_latency_s={result.per_token_latency_s} "
            f"peak_gpu_mb={result.peak_gpu_mb} notes={result.notes} "
            f"saved={output_path}"
        )

    kinds = [_result_kind(result) for result in results]
    if kinds and all(kind == "unavailable" for kind in kinds):
        return 4
    if kinds and all(kind == "oom" for kind in kinds):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
