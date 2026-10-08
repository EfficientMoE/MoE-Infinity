"""CPU subprocess tests for examples/vl_keyword_eval.py (#221 phase 4).

Fake mode must score injected answers and emit the pinned machine-readable
lines without importing torch, transformers, or moe_infinity, so these run
on CPU with no checkpoint download.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "vl_keyword_eval.py"
CPU_FIXTURE = REPO_ROOT / "tests/fixtures/vl/keyword_eval_cpu.json"
ACCEPT_FIXTURE = REPO_ROOT / "tests/fixtures/vl/minimax_m3_keyword_eval.json"


def run_eval(args, extra_env=None):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def load_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def correct_answers(rows):
    return {row["id"]: row["keyword"] for row in rows}


def answers_with_n_correct(rows, n):
    out = {}
    for index, row in enumerate(rows):
        out[row["id"]] = row["keyword"] if index < n else ""
    return out


def deterministic_timings(rows):
    out = {}
    for index, row in enumerate(rows):
        out[row["id"]] = {
            "ttft_ms": 100.0 + index,
            "decode_tokens_per_s": 10.0 + index,
        }
    return out


def test_cpu_fixture_all_correct_exits_zero():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "red", "cpu-3": "two"}),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "KEYWORD_SCORE 3/3" in result.stdout


def test_cpu_fixture_one_correct_exits_nonzero():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "wrong", "cpu-3": "wrong"}),
        ]
    )
    assert result.returncode != 0
    assert "KEYWORD_SCORE 1/3" in result.stdout


def test_keyword_match_is_case_insensitive_substring():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps(
                {
                    "cpu-1": "It is a CAT.",
                    "cpu-2": "A bright RED ball.",
                    "cpu-3": "There are TWO of them.",
                }
            ),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "KEYWORD_SCORE 3/3" in result.stdout


def test_fake_mode_does_not_import_model(tmp_path):
    sabotage = tmp_path / "sabotage"
    sabotage.mkdir()
    for module in ("torch", "transformers", "moe_infinity"):
        (sabotage / f"{module}.py").write_text(
            "raise ImportError('model import must not happen in fake mode')\n",
            encoding="utf-8",
        )
    env = {
        "PYTHONPATH": os.pathsep.join([str(sabotage), str(REPO_ROOT)]),
    }
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "red", "cpu-3": "two"}),
        ],
        extra_env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "KEYWORD_SCORE 3/3" in result.stdout
    assert "ANSWER:" not in result.stdout


def test_deterministic_fake_timings_emit_pinned_lines():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "red", "cpu-3": "two"}),
            "--fake-timings",
            json.dumps(
                {
                    "cpu-1": {"ttft_ms": 100, "decode_tokens_per_s": 10},
                    "cpu-2": {"ttft_ms": 200, "decode_tokens_per_s": 20},
                    "cpu-3": {"ttft_ms": 300, "decode_tokens_per_s": 30},
                }
            ),
        ]
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert (
        "REQUEST_PERF id=cpu-1 ttft_ms=100.000 "
        "decode_tokens_per_s=10.000" in stdout
    )
    assert (
        "REQUEST_PERF id=cpu-3 ttft_ms=300.000 "
        "decode_tokens_per_s=30.000" in stdout
    )
    assert "KEYWORD_SCORE 3/3" in stdout
    assert (
        "PERF_SUMMARY sample_count=3 ttft_p50_ms=200.000 "
        "ttft_p99_ms=298.000 decode_tok_s_p50=20.000 "
        "decode_tok_s_p99=29.800" in stdout
    )


def test_perf_summary_prints_after_keyword_score():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "red", "cpu-3": "two"}),
            "--fake-timings",
            json.dumps(
                {
                    "cpu-1": {"ttft_ms": 100, "decode_tokens_per_s": 10},
                    "cpu-2": {"ttft_ms": 200, "decode_tokens_per_s": 20},
                    "cpu-3": {"ttft_ms": 300, "decode_tokens_per_s": 30},
                }
            ),
        ]
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    score_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("KEYWORD_SCORE")
    )
    perf_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("PERF_SUMMARY")
    )
    assert perf_idx > score_idx


def test_fake_timings_requires_fake_answers():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-timings",
            json.dumps({"cpu-1": {"ttft_ms": 1, "decode_tokens_per_s": 1}}),
        ]
    )
    assert result.returncode != 0


def test_fake_timings_missing_id_is_malformed():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "red", "cpu-3": "two"}),
            "--fake-timings",
            json.dumps(
                {
                    "cpu-1": {"ttft_ms": 100, "decode_tokens_per_s": 10},
                    "cpu-2": {"ttft_ms": 200, "decode_tokens_per_s": 20},
                }
            ),
        ]
    )
    assert result.returncode != 0


def test_fake_timings_negative_value_is_malformed():
    result = run_eval(
        [
            "--fixture",
            str(CPU_FIXTURE),
            "--fake-answers",
            json.dumps({"cpu-1": "cat", "cpu-2": "red", "cpu-3": "two"}),
            "--fake-timings",
            json.dumps(
                {
                    "cpu-1": {"ttft_ms": -1, "decode_tokens_per_s": 10},
                    "cpu-2": {"ttft_ms": 200, "decode_tokens_per_s": 20},
                    "cpu-3": {"ttft_ms": 300, "decode_tokens_per_s": 30},
                }
            ),
        ]
    )
    assert result.returncode != 0


def _write_fixture(tmp_path, payload):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_malformed_fixture_not_array_rejected(tmp_path):
    path = _write_fixture(tmp_path, {"id": "x"})
    result = run_eval(
        ["--fixture", str(path), "--fake-answers", json.dumps({})]
    )
    assert result.returncode != 0


def test_malformed_fixture_empty_array_rejected(tmp_path):
    path = _write_fixture(tmp_path, [])
    result = run_eval(
        ["--fixture", str(path), "--fake-answers", json.dumps({})]
    )
    assert result.returncode != 0


def test_malformed_fixture_extra_key_rejected(tmp_path):
    path = _write_fixture(
        tmp_path,
        [
            {
                "id": "x",
                "image": "a.jpg",
                "prompt": "q",
                "keyword": "k",
                "extra": "nope",
            }
        ],
    )
    result = run_eval(
        ["--fixture", str(path), "--fake-answers", json.dumps({"x": "k"})]
    )
    assert result.returncode != 0


def test_malformed_fixture_empty_string_rejected(tmp_path):
    path = _write_fixture(
        tmp_path,
        [{"id": "", "image": "a.jpg", "prompt": "q", "keyword": "k"}],
    )
    result = run_eval(
        ["--fixture", str(path), "--fake-answers", json.dumps({"x": "k"})]
    )
    assert result.returncode != 0


def test_malformed_fixture_duplicate_id_rejected(tmp_path):
    row = {"id": "x", "image": "a.jpg", "prompt": "q", "keyword": "k"}
    path = _write_fixture(tmp_path, [row, dict(row)])
    result = run_eval(
        ["--fixture", str(path), "--fake-answers", json.dumps({"x": "k"})]
    )
    assert result.returncode != 0


def test_acceptance_fixture_is_frozen_twenty_rows():
    rows = load_rows(ACCEPT_FIXTURE)
    assert isinstance(rows, list)
    assert len(rows) == 20
    required = {"id", "image", "prompt", "keyword"}
    ids = set()
    for row in rows:
        assert set(row.keys()) == required
        for key in required:
            assert isinstance(row[key], str) and row[key]
        assert row["image"].startswith("tests/fixtures/vl/images/")
        assert row["image"].endswith(".jpg")
        ids.add(row["id"])
    assert len(ids) == 20


def test_acceptance_fixture_scores_all_twenty():
    rows = load_rows(ACCEPT_FIXTURE)
    result = run_eval(
        [
            "--fixture",
            str(ACCEPT_FIXTURE),
            "--fake-answers",
            json.dumps(correct_answers(rows)),
            "--fake-timings",
            json.dumps(deterministic_timings(rows)),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "KEYWORD_SCORE 20/20" in result.stdout
    assert "PERF_SUMMARY sample_count=20" in result.stdout


def test_acceptance_fixture_gate_is_sixteen_of_twenty():
    rows = load_rows(ACCEPT_FIXTURE)
    passing = run_eval(
        [
            "--fixture",
            str(ACCEPT_FIXTURE),
            "--fake-answers",
            json.dumps(answers_with_n_correct(rows, 16)),
        ]
    )
    assert passing.returncode == 0, passing.stderr
    assert "KEYWORD_SCORE 16/20" in passing.stdout

    failing = run_eval(
        [
            "--fixture",
            str(ACCEPT_FIXTURE),
            "--fake-answers",
            json.dumps(answers_with_n_correct(rows, 15)),
        ]
    )
    assert failing.returncode != 0
    assert "KEYWORD_SCORE 15/20" in failing.stdout
