"""Deterministic replay of the committed tail-classification latency JSONs.

Guards the #295 per-prompt decomposition: the committed reports
(``results/tailclass_qwen_{off,mass20}_report.json``) must stay exactly
reproducible from the committed raw latency JSONs
(``results/tailclass_qwen_{off,mass20}.json``).  Everything asserted here is
pure ITL arithmetic — no trace dumps, no GPU, no network — so the check is
fully deterministic and offline.

The decomposition is recomputed independently below (same semantics as
``classify_tail_tokens.py``: pooled ``np.percentile([50, 90, 99])`` bands with
``>=`` boundaries, per-prompt summaries, tail-carrier concentration
``1/sum(share^2)``, paired per-prompt mean-ITL deltas).  Keeping a second
implementation means a semantics change in the classifier shows up as a
mismatch against the committed reports rather than passing silently.

The last test pins the published #295 findings.  If a future re-capture
legitimately changes the artifacts, update the reports AND those pins in the
same commit so the finding text can never drift from the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("off", "mass20")
BANDS = ("p0_50", "p50_90", "p90_99", "p99_100")
FLOAT_TOL = 1e-12  # identical ops on identical inputs; absorbs JSON formatting only


def _decompose(latency: dict) -> dict:
    """Recompute the latency-derived report fields from itl_ms_per_prompt."""
    itl_pp = latency["itl_ms_per_prompt"]
    flat = [x for xs in itl_pp for x in xs]
    p50, p90, p99 = np.percentile(np.array(flat), [50, 90, 99])

    def band_of(x: float) -> str:
        if x >= p99:
            return "p99_100"
        if x >= p90:
            return "p90_99"
        if x >= p50:
            return "p50_90"
        return "p0_50"

    per_prompt = []
    for p, xs in enumerate(itl_pp):
        arr = np.array(xs)
        counts = {b: 0 for b in BANDS}
        for x in xs:
            counts[band_of(x)] += 1
        per_prompt.append(
            {
                "prompt": p,
                "tokens": len(xs),
                "itl_ms_mean": float(arr.mean()),
                "itl_ms_p50": float(np.percentile(arr, 50)),
                "itl_ms_max": float(arr.max()),
                "band_tokens": counts,
            }
        )

    tail_total = sum(pp["band_tokens"]["p99_100"] for pp in per_prompt)
    shares = np.array(
        [pp["band_tokens"]["p99_100"] / tail_total if tail_total else 0.0 for pp in per_prompt]
    )
    order = np.argsort(-shares, kind="stable")
    return {
        "num_tokens": len(flat),
        "percentiles": {"p50": float(p50), "p90": float(p90), "p99": float(p99)},
        "per_prompt": per_prompt,
        "tail_concentration": {
            "tail_tokens": tail_total,
            "prompts": len(itl_pp),
            "per_prompt_share": [float(s) for s in shares],
            "top3_prompts": [int(i) for i in order[:3]],
            "top3_share": float(shares[order[:3]].sum()),
            "effective_prompt_count": (
                float(1.0 / float((shares**2).sum())) if tail_total else None
            ),
        },
    }


def _load(arm: str) -> tuple[dict, dict]:
    latency = json.loads((RESULTS / f"tailclass_qwen_{arm}.json").read_text())
    report = json.loads((RESULTS / f"tailclass_qwen_{arm}_report.json").read_text())
    return latency, report


def _assert_close(actual: float, expected: float, what: str) -> None:
    assert actual == pytest.approx(expected, rel=FLOAT_TOL, abs=FLOAT_TOL), what


@pytest.mark.parametrize("arm", ARMS)
def test_report_replays_from_latency(arm: str) -> None:
    latency, report = _load(arm)
    rec = _decompose(latency)

    assert report["num_tokens"] == rec["num_tokens"] == latency["samples"]
    for key, value in rec["percentiles"].items():
        _assert_close(value, report["itl_percentiles_ms"][key], f"{arm} percentile {key}")

    # Band token counts are latency-derived; per-band fetch statistics need
    # the trace dump and are intentionally not replayed here.
    for band in BANDS:
        recomputed = sum(pp["band_tokens"][band] for pp in rec["per_prompt"])
        assert recomputed == report["bands"][band]["tokens"], f"{arm} band {band} count"

    assert len(rec["per_prompt"]) == len(report["per_prompt"])
    for mine, theirs in zip(rec["per_prompt"], report["per_prompt"]):
        for key in ("prompt", "tokens", "band_tokens"):
            assert mine[key] == theirs[key], f"{arm} prompt {mine['prompt']} {key}"
        for key in ("itl_ms_mean", "itl_ms_p50", "itl_ms_max"):
            _assert_close(mine[key], theirs[key], f"{arm} prompt {mine['prompt']} {key}")

    mine, theirs = rec["tail_concentration"], report["tail_concentration"]
    for key in ("tail_tokens", "prompts", "top3_prompts"):
        assert mine[key] == theirs[key], f"{arm} tail_concentration {key}"
    for key in ("top3_share", "effective_prompt_count"):
        _assert_close(mine[key], theirs[key], f"{arm} tail_concentration {key}")
    for i, (a, b) in enumerate(zip(mine["per_prompt_share"], theirs["per_prompt_share"])):
        _assert_close(a, b, f"{arm} tail share[{i}]")


def test_paired_compare_replays() -> None:
    _, off = _load("off")
    _, mass = _load("mass20")
    pc = mass["paired_compare"]
    assert pc["against"].endswith("tailclass_qwen_off_report.json")

    mine = np.array([pp["itl_ms_mean"] for pp in mass["per_prompt"]])
    base = np.array([pp["itl_ms_mean"] for pp in off["per_prompt"]])
    delta_pct = (mine - base) / base * 100.0

    assert len(delta_pct) == len(pc["per_prompt_mean_itl_delta_pct"])
    for i, (a, b) in enumerate(zip(delta_pct, pc["per_prompt_mean_itl_delta_pct"])):
        _assert_close(float(a), b, f"paired delta[{i}]")
    _assert_close(float(np.median(delta_pct)), pc["median_delta_pct"], "paired median")
    _assert_close(float(np.mean(delta_pct)), pc["mean_delta_pct"], "paired mean")
    assert int((delta_pct < 0).sum()) == pc["prompts_improved"]
    assert sorted(mass["tail_concentration"]["top3_prompts"]) == pc["tail_top3_this"]
    assert sorted(off["tail_concentration"]["top3_prompts"]) == pc["tail_top3_other"]
    assert pc["tail_top3_overlap"] == len(
        set(pc["tail_top3_this"]) & set(pc["tail_top3_other"])
    )


def test_pinned_295_findings() -> None:
    """Pin the published findings; update together with the artifacts."""
    _, off = _load("off")
    _, mass = _load("mass20")
    o, m = off["tail_concentration"], mass["tail_concentration"]

    assert o["top3_prompts"] == [5, 4, 1]
    assert m["top3_prompts"] == [22, 16, 10]
    assert o["top3_share"] == pytest.approx(0.8064516129032258, rel=1e-9)
    assert m["top3_share"] == pytest.approx(0.8064516129032259, rel=1e-9)
    assert o["effective_prompt_count"] == pytest.approx(3.1716, abs=1e-3)
    assert m["effective_prompt_count"] == pytest.approx(3.2799, abs=1e-3)
    assert o["tail_tokens"] == m["tail_tokens"] == 31

    pc = mass["paired_compare"]
    assert pc["tail_top3_overlap"] == 0
    assert pc["prompts_improved"] == 14
    assert pc["median_delta_pct"] == pytest.approx(-3.4205412263433645, abs=1e-6)


if __name__ == "__main__":  # pragma: no cover - manual smoke without pytest
    for _arm in ARMS:
        test_report_replays_from_latency(_arm)
    test_paired_compare_replays()
    test_pinned_295_findings()
    print("tailclass replay: all checks passed")
