#!/usr/bin/env python3
"""exp667 CostFit analyzer: verdict from frozen CSVs + in-process resimulation.

Reads:  results_real_{qwen3-30b-a3b,granite3.1-moe-3b}-costfit.csv (sweep)
        results_real_qwen3-30b-a3b_decode.csv, results_real_granite3_1_moe_3b_decode.csv (baselines)
Writes: exp667-verdict.json  (decision rules frozen in PREREG.md sha 462b9cdb)
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moei-granite3b"))

from benchmarks.cache_simulator.research.run_policies_real import load_accesses
from benchmarks.cache_simulator.policies import build_policy

REPO = Path.home() / "warp-work/moei-granite3b/benchmarks/cache_simulator/research"
MODELS = {
    "qwen30b": {
        "sweep": "results_real_qwen3-30b-a3b-costfit.csv",
        "base": "results_real_qwen3-30b-a3b_decode.csv",
        "caps": [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
        "trace": ["data/belady/stream_qwen30b.pt"],
    },
    "granite": {
        "sweep": "results_real_granite3.1-moe-3b-costfit.csv",
        "base": "results_real_granite3_1_moe_3b_decode.csv",
        "caps": [128, 256, 512, 768, 1024],
        "trace": ["benchmarks/cache_simulator/data/handcrafted8-greedy512/belady/stream_granite3b.pt"],
    },
}


def rows(path):
    with open(REPO / path) as f:
        return {(r["policy"], int(r["capacity"])): float(r["hit_rate"]) for r in csv.DictReader(f)}


def main():
    out = {"per_model": {}, "rules": {"margin": 0.005, "max_switches": 6}}
    for m, cfg in MODELS.items():
        sw, bs = rows(cfg["sweep"]), rows(cfg["base"])
        root = str(REPO.parents[2])
        trace = [str(Path(root, *t)) for t in (p.split("/") for p in cfg["trace"])]
        acc, st = load_accesses(trace, ignore_layers=0, prefill="skip")
        n_exp = max(e for _, e in acc) + 1
        gk = {"num_layers": st["num_layers"], "prior": st["top_k"] / n_exp}
        per_cap = {}
        for cap in cfg["caps"]:
            csv_rate = sw[("costfit", cap)]
            pol = build_policy("costfit", cap, **gk)
            hits = sum(1 for k in acc if pol.on_access(k))
            sim_rate, switches = hits / len(acc), pol.switches
            lcnu, s3f = bs[("lcnu", cap)], bs[("s3fifo", cap)]
            best, best_name = max((lcnu, "lcnu"), (s3f, "s3fifo"))
            per_cap[str(cap)] = {
                "costfit_csv": csv_rate,
                "resim": round(sim_rate, 6),
                "determinism_ok": abs(csv_rate - sim_rate) < 5e-7,
                "switches": switches,
                "lcnu": lcnu, "s3fifo": s3f,
                "best": best, "best_policy": best_name,
                "deficit_pp": round((best - csv_rate) * 100, 3),
                "h_pass": csv_rate >= best - 0.005,
                "switches_ok": switches <= 6,
            }
        det = all(v["determinism_ok"] for v in per_cap.values())
        h1h2 = all(v["h_pass"] for v in per_cap.values())
        # exp667 H3 accounting: at switched caps CSV must differ from EITHER
        # inner's standalone cumulative rate (exp665 final-choice bug signature)
        acc_ok = all(
            v["switches"] == 0
            or (abs(v["costfit_csv"] - v["lcnu"]) > 5e-7 and abs(v["costfit_csv"] - v["s3fifo"]) > 5e-7)
            for v in per_cap.values()
        )
        h3 = all(v["switches_ok"] for v in per_cap.values())
        out["per_model"][m] = {
            "caps": per_cap,
            "determinism_ok": det,
            "near_dominance": h1h2,
            "stability": h3,
            "accounting_ok": acc_ok and det,
            "worst_deficit_pp": max(v["deficit_pp"] for v in per_cap.values()),
            "max_switches": max(v["switches"] for v in per_cap.values()),
        }
    p30, pg = out["per_model"]["qwen30b"], out["per_model"]["granite"]
    out["H1_near_dominance"] = p30["near_dominance"] and pg["near_dominance"]
    out["H2_stability"] = p30["stability"] and pg["stability"]
    out["H3_accounting"] = p30["accounting_ok"] and pg["accounting_ok"]
    out["verdict"] = (
        "SUPPORTED" if out["H1_near_dominance"] and out["H2_stability"] and out["H3_accounting"]
        else "FAIL"
    )
    Path(Path(__file__).parent, "exp667-verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verdict", "H1_near_dominance", "H2_stability", "H3_accounting")}))
    for m in ("qwen30b", "granite"):
        pm = out["per_model"][m]
        print(m, "worst_deficit_pp", pm["worst_deficit_pp"], "max_sw", pm["max_switches"], "det", pm["determinism_ok"], "acct", pm["accounting_ok"])


if __name__ == "__main__":
    main()
