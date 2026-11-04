#!/usr/bin/env python3
"""exp682: anti-clustering candidate — doorkeeper admission filter.

exp678 (COST-H1) showed the center policy's extra misses vs lcnu CLUSTER in
time under bursty reload cost (phi=4), and that expert heterogeneity (kappa)
is NOT the failure mode. exp680 ruled out arbitration cadence. Mechanism:
one-touch cold admissions evict soon-reused keys -> the extra misses arrive
in bursts during reload pressure.

Intervention (wrapper only; sealed policy files untouched): DOORKEEPER.
First-sight missing keys are recorded but NOT admitted (evict nothing);
second sight admits via the base policy. Resident-key hits bypass the door
untouched. Dose: door capacity K in {512, 4096} (LRU door).

Testbed: exp678's frozen stationary streams + cost metric, seeds identical
-> center arm must reproduce exp678's published numbers per cell
(replication gate, exact). Grid: skew {0.9, 1.3} x cap {512, 2048} x
kappa {0, 20} x phi {0, 4} = 16 cells. Arms: costfit center, door512,
door4096, lcnu, lcnu+door4096 (generalization probe).

Bars (PREREG frozen pre-launch): H1 door TIME < center at >= half of phi=4
cells by >= 0.1% absolute; H2 door HIT >= center hit - 0.5pp everywhere;
H3 no cell door time > center + 0.1%; acc center misses==misses2; det
byte-identical double grid. Verdict: DOOR-WINS > DOOR-NULL > DOOR-HARM.
"""
import importlib.util
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "cm", str(Path.home() / "warp-work/exp678-moei-costmodel/run_costmodel.py"))
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
rd = cm.rd

D = Path(__file__).parent
KAPPAS = [0.0, 20.0]
DOOR_CAPS = [512, 4096]


def mk_door(base_mk, k_cap):
    """Wrap a policy with a doorkeeper admission filter (K LRU)."""
    pol = base_mk()
    door = OrderedDict()
    orig = pol.on_access

    def on_access(k):
        if k in pol.resident:
            return orig(k)
        if k in door:
            door.pop(k)
            return orig(k)
        door[k] = None
        if len(door) > k_cap:
            door.popitem(last=False)
        return False

    pol.on_access = on_access
    return pol


def run_all():
    seeds = np.random.default_rng(cm.SEED).spawn(len(cm.SKEWS))
    grid = []
    for i, sk in enumerate(cm.SKEWS):
        keys = cm.gen_stationary(int(seeds[i].integers(0, 2 ** 63)), sk, cm.LAM)
        rng_u = np.random.default_rng(cm.SEED + 1000 + i)
        u = cm.cost_arrays(rng_u)
        gk = {"num_layers": rd.L, "prior": 8 / rd.E}
        for cap in cm.CAPS:
            for kap in KAPPAS:
                for phi in cm.PHIS:
                    cell = {}
                    tavg, ms = cm.sim_time(
                        keys, lambda: rd.build_policy("lcnu", cap), u, kap, phi)
                    cell["lcnu"] = {"time": tavg, "hit": 1 - ms / len(keys)}
                    if hasattr(rd.build_policy("lcnu", cap), "resident"):
                        tavg, ms = cm.sim_time(
                            keys, lambda: mk_door(
                                lambda: rd.build_policy("lcnu", cap), 4096),
                            u, kap, phi)
                        cell["lcnu_door"] = {"time": tavg,
                                             "hit": 1 - ms / len(keys)}
                    tavg, ms = cm.sim_time(keys, lambda: rd._mk_cf(cap, gk),
                                           u, kap, phi)
                    t2, ms2 = cm.sim_time(keys, lambda: rd._mk_cf(cap, gk),
                                          u, kap, phi)
                    cell["costfit"] = {"time": tavg, "hit": 1 - ms / len(keys),
                                       "time2": t2, "misses": ms, "misses2": ms2}
                    for dc in DOOR_CAPS:
                        tavg, ms = cm.sim_time(
                            keys, lambda dc=dc: mk_door(
                                lambda: rd._mk_cf(cap, gk), dc), u, kap, phi)
                        cell[f"door{dc}"] = {"time": tavg,
                                             "hit": 1 - ms / len(keys)}
                    grid.append({"skew": sk, "cap": cap, "kappa": kap,
                                 "phi": phi, "cell": cell})
        print(f"skew {sk} done [{time.time() - t0:.0f}s]", flush=True)
    return grid


def replication(grid):
    """Center arm must equal exp678's published per-cell time/hit exactly."""
    ref = json.loads((Path.home() /
                      "warp-work/exp678-moei-costmodel/exp678-costmodel.json").read_text())
    refmap = {(g["skew"], g["cap"], g["kappa"], g["phi"]): g["cell"]["costfit"]
              for g in ref["grid"]}
    bad = 0
    for g in grid:
        key = (g["skew"], g["cap"], g["kappa"], g["phi"])
        if key in refmap:
            r = refmap[key]
            if (g["cell"]["costfit"]["time"] != r["time"]
                    or g["cell"]["costfit"]["hit"] != r["hit"]):
                bad += 1
    return bad == 0, len(refmap)


def adjudicate(grid):
    rows = []
    h1_cells = h1_pass = h2_bad = h3_bad = 0
    for g in grid:
        c = g["cell"]
        if c["costfit"]["misses"] != c["costfit"]["misses2"]:
            rows.append({"bad": "acc", **{k: g[k] for k in
                                          ("skew", "cap", "kappa", "phi")}})
            continue
        for arm in ("door512", "door4096"):
            dt = (c["costfit"]["time"] - c[arm]["time"]) / c["costfit"]["time"]
            dh = (c[arm]["hit"] - c["costfit"]["hit"]) * 100
            if g["phi"] > 0:
                h1_cells += 1
                if dt >= 0.001:
                    h1_pass += 1
            if dh < -0.5:
                h2_bad += 1
            if dt < -0.001:
                h3_bad += 1
            rows.append({"skew": g["skew"], "cap": g["cap"], "kappa": g["kappa"],
                         "phi": g["phi"], "arm": arm,
                         "d_time_pct": round(dt * 100, 3),
                         "d_hit_pp": round(dh, 3)})
    ld = None
    if "lcnu_door" in grid[0]["cell"]:
        dl = sum(1 for g in grid if g["cell"]["lcnu_door"]["time"]
                 < g["cell"]["lcnu"]["time"])
        ld = f"{dl}/{len(grid)}"
    verdict = ("DOOR-WINS" if (h1_pass * 2 >= h1_cells and not h2_bad and not h3_bad)
               else "DOOR-HARM" if (h2_bad or h3_bad) else "DOOR-NULL")
    return {"verdict": verdict, "h1_pass": h1_pass, "h1_cells": h1_cells,
            "h2_bad": h2_bad, "h3_bad": h3_bad, "lcnu_door_better": ld,
            "rows": rows}


t0 = time.time()

if __name__ == "__main__":
    g1 = run_all()
    res = adjudicate(g1)
    repl, nref = replication(g1)
    g2 = run_all()
    res["det"] = json.dumps(g1) == json.dumps(g2)
    res["replication"] = repl
    res["grid"] = g1
    res["rules"] = {"kappas": KAPPAS, "door_caps": DOOR_CAPS,
                    "bars": {"h1": ">=50% of phi>0 cells improve >=0.1% time",
                             "h2": "hit >= center-0.5pp", "h3": "time <= center+0.1%"},
                    "replication_cells": nref}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp682-door.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| det:", res["det"],
          "| repl:", repl, f"({nref} cells)", "| h1:", res["h1_pass"],
          "/", res["h1_cells"], "| h2bad:", res["h2_bad"],
          "| h3bad:", res["h3_bad"], "| lcnu_door:", res["lcnu_door_better"])
