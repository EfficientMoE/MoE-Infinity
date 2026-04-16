# pyright: reportDeprecated=false

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys
from typing import Dict, List, Optional, cast

try:
    from benchmarks.comparison.common import BenchmarkResult, load_results
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from benchmarks.comparison.common import BenchmarkResult, load_results


FRAMEWORKS = [
    ("moe_infinity", "MoE-Infinity (FP16)"),
    ("vllm", "vLLM v0.18.1 (FP8)"),
    ("llamacpp", "llama.cpp b8640 (Q4_K_M)"),
]

MODELS = [
    ("deepseek-v2-lite", "DeepSeek-V2-Lite-Chat"),
    ("mixtral-8x7b", "Mixtral-8x7b"),
    ("qwen3-30b", "Qwen3-30B-A3B"),
    ("gpt-oss-20b", "gpt-oss-20b"),
]

MATRIX_PRESENT_KEY = "__present__"


def build_matrix(
    results: List[BenchmarkResult],
) -> Dict[str, object]:
    matrix: Dict[str, object] = {}
    present: Dict[str, bool] = {}

    for framework_key, _framework_label in FRAMEWORKS:
        matrix[framework_key] = {}
        framework_row = cast(Dict[str, Optional[float]], matrix[framework_key])
        present[framework_key] = False
        for model_key, _model_label in MODELS:
            framework_row[model_key] = None

    for result in results:
        if result.framework not in matrix:
            continue
        framework_row = cast(
            Dict[str, Optional[float]], matrix[result.framework]
        )
        if result.model not in framework_row:
            continue
        present[result.framework] = True
        framework_row[result.model] = result.per_token_latency_s

    matrix[MATRIX_PRESENT_KEY] = present
    return matrix


def format_cell(value: Optional[float]) -> str:
    if value is None:
        return "X"
    return f"{value:.3f}"


def _present_map(matrix: Dict[str, object]) -> Dict[str, bool]:
    return cast(Dict[str, bool], matrix.get(MATRIX_PRESENT_KEY, {}))


def _row_map(
    matrix: Dict[str, object], framework_key: str
) -> Dict[str, Optional[float]]:
    return cast(Dict[str, Optional[float]], matrix[framework_key])


def render_markdown(matrix: Dict[str, object]) -> str:
    present = _present_map(matrix)

    lines: List[str] = []
    header = [""] + [model_label for _model_key, model_label in MODELS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(":---:" for _ in header) + " |")

    for framework_key, framework_label in FRAMEWORKS:
        if not present.get(framework_key, False):
            row_values = ["—" for _model_key, _model_label in MODELS]
        else:
            row_map = _row_map(matrix, framework_key)
            row_values = [
                format_cell(row_map[model_key])
                for model_key, _model_label in MODELS
            ]
        lines.append("| " + " | ".join([framework_label] + row_values) + " |")

    return "\n".join(lines)


def render_csv(matrix: Dict[str, object]) -> str:
    present = _present_map(matrix)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([""] + [model_label for _model_key, model_label in MODELS])
    for framework_key, framework_label in FRAMEWORKS:
        if not present.get(framework_key, False):
            row_values = ["—" for _model_key, _model_label in MODELS]
        else:
            row_map = _row_map(matrix, framework_key)
            row_values = [
                format_cell(row_map[model_key])
                for model_key, _model_label in MODELS
            ]
        writer.writerow([framework_label] + row_values)
    return buffer.getvalue().rstrip("\n")


def render_json(matrix: Dict[str, object]) -> str:
    present = _present_map(matrix)
    payload: Dict[str, Dict[str, Optional[float]]] = {}
    for framework_key, framework_label in FRAMEWORKS:
        if not present.get(framework_key, False):
            payload[framework_label] = {
                model_label: None for _model_key, model_label in MODELS
            }
        else:
            row_map = _row_map(matrix, framework_key)
            payload[framework_label] = {
                model_label: row_map[model_key]
                for model_key, model_label in MODELS
            }
    return json.dumps(payload, indent=2)


def _write_output(
    text: str, output_path: str, results_dir: str, format_name: str
) -> None:
    if output_path == "-":
        _ = sys.stdout.write(text)
        _ = sys.stdout.write("\n")
    else:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(text + "\n", encoding="utf-8")

    if format_name == "markdown":
        table_path = pathlib.Path(results_dir) / "comparison_table.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        _ = table_path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark JSON results into tabular summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--results-dir",
        default="benchmarks/comparison/results/",
        help="Directory containing benchmark result JSON files.",
    )
    _ = parser.add_argument(
        "--output",
        default="-",
        help='Output file path, or "-" for stdout.',
    )
    _ = parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = cast(str, args.results_dir)
    output = cast(str, args.output)
    output_format = cast(str, args.format)

    results = load_results(results_dir)
    matrix = build_matrix(results)

    if output_format == "markdown":
        rendered = render_markdown(matrix)
    elif output_format == "csv":
        rendered = render_csv(matrix)
    else:
        rendered = render_json(matrix)

    _write_output(rendered, output, results_dir, output_format)


if __name__ == "__main__":
    main()
