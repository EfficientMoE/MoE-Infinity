#!/usr/bin/env python3
"""exp684: inner-admission probation — admit fast, demote never-rehit first.

Mechanism from exp681+682+683: the lever is admission SPEED with
post-admission demotion. The door (wrapper filter) protected the hot set
(recovery flipped to advantage) but delayed admission (in-burst worse).
ProbationLCNU admits on first sight (no latency) and overrides victim
order: the OLDEST never-rehit resident is evicted first; re-hit promotes
to full citizen. Pure subclass of sealed LayerCycleNextUsePolicy
(_victim/_on_miss/_on_hit hooks) — sealed files untouched.

Arms on exp681's EXACT crowd streams (8 cells):
  lcnu        best standalone reference
  center      frozen composite (REPLICATION vs exp681 published per-cell)
  prob_lcnu   ProbationLCNU standalone
  center_prob composite with ProbationLCNU as its lcnu inner

Bars (frozen in LEDGER.md before launch):
  REPL  center loop/burst/rec == exp681 costfit per-cell (exact)
  H1    prob_lcnu burst & loop >= lcnu - 0.5pp in ALL cells (no-loss)
  H2    protection: prob_lcnu rec >= lcnu rec - 0.5pp ALL;
        center_prob rec >= center rec - 0.5pp ALL
  H3    prize: center_prob loop >= max(lcnu, center) in >= 4/8 cells,
        and >= lcnu - 0.5pp ALL
  acc   center double-sim identical; det byte-identical double grid
Verdict: PROB-WINS > PROB-MECH > PROB-NULL > PROB-HARM.
Exit codes: 0 settled | 2 det | 3 acc | 4 replication.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
from benchmarks.cache_simulator.policies import (  # noqa: E402
    LayerCycleNextUsePolicy, build_policy)

spec = importlib.util.spec_from_file_location(
    "st", str(Path.home() / "warp-work/exp681-moei-crowd/run_crowd.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)
rd = st.rd

D = Path(__file__).parent
CostFitPolicy = type(build_policy("costfit", 64, num_layers=4, prior=0.1))
ARMS = ["lcnu", "center", "prob_lcnu", "center_prob"]


class ProbationLCNU(LayerCycleNextUsePolicy):
    """LCNU + probation: never-rehit residents evict first (oldest first)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._prob = {}  # key -> None, insertion-ordered (oldest first)

    def _on_hit(self, key):
        self._prob.pop(key, None)  # re-hit promotes to full citizen

    def _on_miss(self, key):
        super()._on_miss(key)
        self._prob[key] = None

    def _victim(self):
        for k in [k for k in self._prob if k not in self.resident]:
            del self._prob[k]  # lazy cleanup
        for k in self._prob:  # oldest resident probationer first
            return k[0], k
        return super()._victim()


class ProbCostFit(CostFitPolicy):
    """Frozen composite with ProbationLCNU as the lcnu inner (from t=0)."""

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self._inners["lcnu"] = ProbationLCNU(
            capacity, prior=prior, num_layers=num_layers)


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    out = {}
    for cap in st.CAPS:
        cell = {}
        builders = {
            "lcnu": lambda: build_policy("lcnu", cap, **gk),
            "center": lambda: build_policy("costfit", cap, **gk),
            "prob_lcnu": lambda: ProbationLCNU(cap, **gk),
            "center_prob": lambda: ProbCostFit(cap, **gk),
        }
        for name in ARMS:
            pol = builders[name]()
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            cell[name] = {"loop": float(h.mean()),
                          "burst": float(h[burst].mean()),
                          "rec": float(h[rec].mean())}
        c2 = builders["center"]()
        h2 = np.array([c2.on_access(k) for k in keys], dtype=bool)
        cell["center"]["loop2"] = float(h2.mean())
        out[str(cap)] = cell
    return out


def run_all():
    grid, t0 = [], time.time()
    seeds = np.random.default_rng(68120261103).spawn(2)  # exp681 scheme, FRESH
    base_seed = int(seeds[0].integers(0, 2 ** 63))
    crowd_seed = int(seeds[1].integers(0, 2 ** 63))
    for d in st.BURSTS:
        keys, seg, mid = st.crowd_stream(base_seed, crowd_seed, st.N, d)
        grid.append({"d": d, "caps": evaluate(keys, seg)})
        print(f"d={d} done [{time.time() - t0:.0f}s]", flush=True)
    return grid


def replication_check(grid):
    ref = json.loads((Path.home()
                      / "warp-work/exp681-moei-crowd/exp681-crowd.json").read_text())
    refc = {}
    for cell in ref["grid"]:
        refc[cell["d"]] = cell["caps"]
    for cell in grid:
        for cap, c in cell["caps"].items():
            r = refc[cell["d"]][cap]["costfit"]
            for m in ("loop", "burst", "rec"):
                if c["center"][m] != r[m]:
                    return (f"repl mismatch d={cell['d']} cap={cap} {m}: "
                            f"{c['center'][m]} != {r[m]}")
    return None


def adjudicate(grid):
    rows, flags = [], {"h1": True, "h2": True, "h3n": 0, "h3_all": True}
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["center"]["loop"] != c["center"]["loop2"]:
                return {"bad": "acc", "d": cell["d"], "cap": cap}
            row = {"d": cell["d"], "cap": int(cap)}
            for a in ARMS:
                row[a] = {m: round(c[a][m], 6) for m in ("loop", "burst", "rec")}
            row["pl_vs_l_burst_pp"] = round((c["prob_lcnu"]["burst"]
                                             - c["lcnu"]["burst"]) * 100, 3)
            row["pl_vs_l_loop_pp"] = round((c["prob_lcnu"]["loop"]
                                            - c["lcnu"]["loop"]) * 100, 3)
            row["pl_vs_l_rec_pp"] = round((c["prob_lcnu"]["rec"]
                                           - c["lcnu"]["rec"]) * 100, 3)
            row["cp_vs_cf_rec_pp"] = round((c["center_prob"]["rec"]
                                            - c["center"]["rec"]) * 100, 3)
            row["cp_vs_best_loop_pp"] = round(
                (c["center_prob"]["loop"]
                 - max(c["lcnu"]["loop"], c["center"]["loop"])) * 100, 3)
            row["cp_vs_l_loop_pp"] = round((c["center_prob"]["loop"]
                                            - c["lcnu"]["loop"]) * 100, 3)
            if row["pl_vs_l_burst_pp"] < -0.5 or row["pl_vs_l_loop_pp"] < -0.5:
                flags["h1"] = False
            if row["pl_vs_l_rec_pp"] < -0.5 or row["cp_vs_cf_rec_pp"] < -0.5:
                flags["h2"] = False
            if row["cp_vs_l_loop_pp"] < -0.5:
                flags["h3_all"] = False
            if row["cp_vs_best_loop_pp"] >= 0.0:
                flags["h3n"] += 1
            rows.append(row)
    if flags["h3n"] >= 4 and flags["h3_all"]:
        v = "PROB-WINS"
    elif flags["h1"] and flags["h2"]:
        v = "PROB-MECH"
    elif (not flags["h1"]) or (not flags["h2"]):
        v = "PROB-HARM"
    else:
        v = "PROB-NULL"
    return {"verdict": v, "h1": flags["h1"], "h2": flags["h2"],
            "h3_wins": f"{flags['h3n']}/8", "h3_all": flags["h3_all"],
            "rows": rows}


if __name__ == "__main__":
    t0 = time.time()
    g1 = run_all()
    res = adjudicate(g1)
    if "bad" in res:
        print("ACC FAIL:", res)
        sys.exit(3)
    bad = replication_check(g1)
    if bad:
        print(bad)
        sys.exit(4)
    g2 = run_all()
    res["det"] = json.dumps(g1) == json.dumps(g2)
    res["repl"] = True
    res["grid"] = g1
    res["rules"] = {"streams": "exp681 exact (68120261103)",
                    "arms": ARMS,
                    "bars": {"REPL": "center == exp681 costfit per-cell exact",
                             "H1": "prob_lcnu burst&loop >= lcnu-0.5pp ALL",
                             "H2": "prob rec >= refs-0.5pp ALL",
                             "H3": "center_prob >= max(lcnu,center) in >=4/8 AND >= lcnu-0.5pp ALL"}}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp684-prob.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| det:", res["det"], "| repl: True",
          "| h1:", res["h1"], "| h2:", res["h2"], "| h3 wins:", res["h3_wins"],
          "| h3_all:", res["h3_all"])
    for r in res["rows"]:
        print("cell:", {k: r[k] for k in ("d", "cap", "pl_vs_l_burst_pp",
                                          "pl_vs_l_rec_pp", "cp_vs_cf_rec_pp",
                                          "cp_vs_best_loop_pp")})
    sys.exit(0 if res["det"] else 2)
