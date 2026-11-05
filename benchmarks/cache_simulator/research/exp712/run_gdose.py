#!/usr/bin/env python3
"""exp712: single-knob GAMMA dose line at center (M, HL) — certified gamma band.

exp711 (MG-BREAK:G-DRIVEN) attributed exp674's SAFETY-BROKEN BO point to gamma: the G-only
corner (margin 0.005, HL 32768, gamma = exp674 bo gamma) fails the no-harm bar while every
center-gamma corner passes. This runner walks gamma from center 0.0025 to the BO value on a
6-point geometric line g_k = C * (BO/C) ** (k/5), k = 0..5 (BO gamma parsed at runtime from
sealed exp674 bo_params), at center margin and center half-life.

H0: the two endpoints are already sealed in exp711 and must replicate EXACTLY per cell
(cf_g0 == exp711 cf_000, cf_g5 == exp711 cf_0G0; standalones too).
Streams: exactly exp670/exp674/exp710/exp711's 11 (sha8-asserted vs research/exp670-synth.json).
Adjudication per frozen PREREG. No hand-transcribed values.

Modes: default = full frozen run.  SMOKE=1 = disclosed machinery smoke (full-N H0 replication
on 1 stream x 2 caps with ONLY the two sealed-bookend arms; reduced unpinned grid N=60k for all
arms).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

D = Path(__file__).parent
REPO = Path.home() / "warp-work/moei-granite3b"
N_ACC = 638_208
H_RECENCY, R_WINDOW = 2048, 4096
SKEWS = [0.8, 1.1, 1.4]
LOCALITIES = [0.0, 0.3, 0.6]
CAPS = [128, 256, 512, 1024, 2048, 4096]
CENTER = (0.005, 0.0025, 32768.0)
EXP670 = REPO / "benchmarks/cache_simulator/research/exp670-synth.json"
EXP674 = Path.home() / "warp-work/exp674-moei-hl-transfer/exp674-hl-transfer.json"
EXP711 = Path.home() / "warp-work/exp711-moei-mgfactorial/exp711-mgfactorial.json"
_BO = json.loads(EXP674.read_text())["bo_params"]
BO_GAMMA = float(_BO["gamma"])
K = 5
GAMMAS = [CENTER[1] * (BO_GAMMA / CENTER[1]) ** (k / K) for k in range(K)] + [BO_GAMMA]
ARMS = [f"g{k}" for k in range(K + 1)]
BOOKENDS = {"g0": "cf_000", "g5": "cf_0G0"}
SAFETY_PP, SWITCH_MAX, INFORMATIVE_PP = 0.5, 6, 0.1


def arm_params(a):
    return (CENTER[0], GAMMAS[int(a[1:])], CENTER[2])


def arm(a):
    return f"cf_{a}"


def gen_stream(seed, E, L, s, lam, n_acc=N_ACC):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, E + 1, dtype=np.float64)
    zipf_p = ranks ** (-s)
    zipf_p /= zipf_p.sum()
    experts = rng.choice(E, size=n_acc, p=zipf_p)
    layers = rng.integers(0, L, size=n_acc)
    keys = list(zip(layers.tolist(), experts.tolist()))
    n_replay = int(lam * n_acc)
    idx = rng.choice(np.arange(R_WINDOW, n_acc), size=n_replay, replace=False)
    w = 0.5 ** (np.arange(1, R_WINDOW + 1) / H_RECENCY)
    w /= w.sum()
    ages = rng.choice(R_WINDOW, size=n_replay, p=w) + 1
    for i, a in zip(idx.tolist(), ages.tolist()):
        keys[i] = keys[i - a]
    return keys


def stream_sha(keys):
    raw = "\x00".join(f"{l}:{e}" for l, e in keys[:50000])
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def make_costfit(cap, gk, params):
    m, g, hl = params
    pol = build_policy("costfit", cap, **gk)
    pol.MARGIN, pol.GAMMA, pol.HALF_LIFE = m, g, hl
    pol._decay = 0.5 ** (1.0 / hl)
    return pol


def evaluate(keys, E, L, caps, arms, top_k=8):
    out = {}
    gk = {"num_layers": L, "prior": top_k / E}
    n = len(keys)
    for cap in caps:
        cell = {}
        for name in ("lcnu", "s3fifo"):
            pol = build_policy(name, cap)
            hits = sum(1 for k in keys if pol.on_access(k))
            cell[name] = {"rate": round(hits / n, 6), "prop": round(pol.effective_hit_rate, 6)}
        for a in arms:
            p = arm_params(a)
            pol = make_costfit(cap, gk, p)
            assert (pol.MARGIN, pol.GAMMA, pol.HALF_LIFE) == p and pol._decay == 0.5 ** (1.0 / p[2])
            hits = sum(1 for k in keys if pol.on_access(k))
            cell[arm(a)] = {"rate": round(hits / n, 6), "prop": round(pol.effective_hit_rate, 6),
                            "switches": getattr(pol, "switches", 0)}
        out[str(cap)] = cell
    return out


def stream_cfgs():
    streams = [{"s": s, "lam": lam, "E": 128, "L": 48} for s in SKEWS for lam in LOCALITIES]
    streams.append({"s": 1.1, "lam": 0.3, "E": 32, "L": 32})
    streams.append({"s": 1.4, "lam": 0.6, "E": 32, "L": 32})
    return streams


def run_all(expected_shas, idxs=None, caps=CAPS, arms=ARMS, n_acc=N_ACC, check_sha=True, seed_perturb=0):
    streams = stream_cfgs()
    seeds = np.random.default_rng(67020261104).spawn(11)
    t0 = time.time()
    grid = []
    for i, cfg in enumerate(streams):
        if idxs is not None and i not in idxs:
            continue
        keys = gen_stream(int(seeds[i].integers(0, 2**63)) + seed_perturb, cfg["E"], cfg["L"], cfg["s"], cfg["lam"], n_acc)
        sha = stream_sha(keys)
        if check_sha:
            assert sha == expected_shas[i], f"stream {i} sha {sha} != exp670 {expected_shas[i]}"
        res = evaluate(keys, cfg["E"], cfg["L"], caps, arms)
        grid.append({"cfg": cfg, "seed": sha, "caps": res})
        print(f"stream {i} done [{time.time()-t0:.0f}s]", flush=True)  # timings only until settlement
    return grid


def h0_replication(grid, ref_grid):
    ref = {(json.dumps(st["cfg"], sort_keys=True), cap): cell for st in ref_grid for cap, cell in st["caps"].items()}
    mism, n = [], 0
    for st in grid:
        for cap, cell in st["caps"].items():
            r = ref[(json.dumps(st["cfg"], sort_keys=True), cap)]
            for a, refarm in BOOKENDS.items():
                if arm(a) not in cell:
                    continue
                n += 1
                if cell[arm(a)] != r[refarm]:
                    mism.append({"cfg": st["cfg"], "cap": cap, "arm": arm(a), "ref": refarm})
            for a in ("lcnu", "s3fifo"):
                n += 1
                if cell[a] != r[a]:
                    mism.append({"cfg": st["cfg"], "cap": cap, "arm": a, "ref": a})
    return {"cells": n, "mismatches": mism, "ok": n > 0 and not mism}


def classify(safe_list):
    """Frozen ladder (PREREG). safe_list: H1 bool per arm in gamma order g0..g5.
    Returns (verdict, certified_edge_index, nonmonotone)."""
    edge = -1
    for i, s in enumerate(safe_list):
        if s:
            edge = i
        else:
            break
    nonmono = any(safe_list[j] for j in range(edge + 2, len(safe_list))) if edge + 1 < len(safe_list) else False
    if edge == len(safe_list) - 1:
        v = "G-LINE-SAFE"
    elif edge <= 0:
        v = "G-BAND-COLLAPSED"
    else:
        v = f"G-BOUNDARY@g{edge + 1}"
    return v, edge, nonmono


def adjudicate(grid, h0, arms=ARMS):
    informative, acct = 0, []
    per = {a: {"worst": None, "safety_cells": 0, "sw_cells": 0, "max_sw": 0, "deltas": [], "over": []} for a in arms}
    for st in grid:
        for cap, cell in st["caps"].items():
            for a in arms:
                x = cell[arm(a)]
                if x["rate"] != x["prop"]:
                    acct.append({"cfg": st["cfg"], "cap": cap, "arm": arm(a)})
            lc, s3 = cell["lcnu"]["rate"], cell["s3fifo"]["rate"]
            if abs(lc - s3) * 100 < INFORMATIVE_PP:
                continue
            informative += 1
            best = max(lc, s3)
            for a in arms:
                x = cell[arm(a)]; p = per[a]
                d = (best - x["rate"]) * 100
                p["deltas"].append((x["rate"] - cell[arm("g0")]["rate"]) * 100)
                p["max_sw"] = max(p["max_sw"], x["switches"])
                if p["worst"] is None or d > p["worst"]["deficit_pp"]:
                    p["worst"] = {"cfg": st["cfg"], "cap": cap, "deficit_pp": round(d, 3), "switches": x["switches"]}
                if d > SAFETY_PP:
                    p["safety_cells"] += 1
                    p["over"].append({"cfg": st["cfg"], "cap": cap, "deficit_pp": round(d, 3)})
                if x["switches"] > SWITCH_MAX:
                    p["sw_cells"] += 1
    out_arms = {}
    for a in arms:
        p = per[a]
        out_arms[a] = {"params": list(arm_params(a)), "H1_safe": p["safety_cells"] == 0 and p["sw_cells"] == 0,
                       "worst": p["worst"], "cells_over_bar": p["safety_cells"], "over_cells": p["over"],
                       "cells_over_switch": p["sw_cells"], "max_switches": p["max_sw"],
                       "mean_delta_vs_g0_pp": round(sum(p["deltas"]) / len(p["deltas"]), 4) if p["deltas"] else None}
    h4 = not acct
    edge, nonmono = None, None
    if not (h0["ok"] and h4):
        verdict = "ACCOUNTING-FAIL"
    elif list(arms) == ARMS:
        verdict, ei, nonmono = classify([out_arms[a]["H1_safe"] for a in ARMS])
        edge = GAMMAS[ei] if ei >= 0 else None
    else:
        verdict = "PARTIAL-GRID"
    return {"verdict": verdict, "certified_gamma_upper_edge": edge, "nonmonotone": nonmono,
            "H0_replication": h0, "H4_accounting": h4, "accounting_violations": acct,
            "informative_cells": informative, "arms": out_arms}


def smoke():
    exp670 = json.loads(EXP670.read_text()); shas = [st["seed"] for st in exp670["grid"]]
    ref = json.loads(EXP711.read_text())["grid"]
    try:
        run_all(shas, idxs={0}, caps=[], arms=[], seed_perturb=1)
        print("SMOKE S1 FAIL: perturbed seed not caught"); return 1
    except AssertionError:
        print("SMOKE S1 ok: sha assertion bites on perturbed seed")
    g = run_all(shas, idxs={4}, caps=[512, 2048], arms=list(BOOKENDS))
    h0 = h0_replication(g, ref)
    assert h0["cells"] == 8, h0
    print("SMOKE S2 H0 machinery replication cells", h0["cells"], "ok", h0["ok"])
    bad = json.loads(json.dumps(g)); bad[0]["caps"]["512"][arm("g5")]["rate"] += 1e-6
    print("SMOKE S3 H0 negative control detects", not h0_replication(bad, ref)["ok"])
    r1 = run_all(shas, idxs={0, 9}, caps=[128, 256], n_acc=60_000, check_sha=False)
    r2 = run_all(shas, idxs={0, 9}, caps=[128, 256], n_acc=60_000, check_sha=False)
    det = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    adj = adjudicate(r1, {"ok": True, "cells": 0, "mismatches": []})
    wf = set(adj["arms"]) == set(ARMS) and adj["verdict"] != "PARTIAL-GRID"
    T, F = True, False
    lad = [classify([T, T, T, T, T, T]) == ("G-LINE-SAFE", 5, False),
           classify([T, T, T, F, F, F]) == ("G-BOUNDARY@g3", 2, False),
           classify([T, T, F, T, F, F]) == ("G-BOUNDARY@g2", 1, True),
           classify([T, F, F, F, F, F]) == ("G-BAND-COLLAPSED", 0, False),
           classify([F, T, T, T, T, T]) == ("G-BAND-COLLAPSED", -1, True),
           classify([T, T, T, T, F, F]) == ("G-BOUNDARY@g4", 3, False)]
    gam = (GAMMAS[0] == CENTER[1] and GAMMAS[-1] == BO_GAMMA and len(GAMMAS) == 6
           and all(GAMMAS[i] < GAMMAS[i + 1] for i in range(5)))
    print("SMOKE S4 det", det, "adjudicate well-formed", wf, "accounting", adj["H4_accounting"], "ladder", all(lad),
          "gamma line", gam)
    ok = h0["ok"] and det and wf and adj["H4_accounting"] and all(lad) and gam
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


def main():
    if os.environ.get("SMOKE") == "1":
        sys.exit(smoke())
    exp670 = json.loads(EXP670.read_text()); shas = [st["seed"] for st in exp670["grid"]]
    ref = json.loads(EXP711.read_text())["grid"]
    t0 = time.time()
    g1 = run_all(shas)
    g2 = run_all(shas)
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True), "nondeterministic"
    v = adjudicate(g1, h0_replication(g1, ref))
    out = {**v, "determinism_two_runs": True, "elapsed_s": round(time.time() - t0, 1), "grid": g1,
           "rules": {"caps": CAPS, "skews": SKEWS, "localities": LOCALITIES, "n_acc": N_ACC,
                     "center": list(CENTER), "bo_gamma": BO_GAMMA, "bo_source": "exp674 bo_params",
                     "gammas": GAMMAS, "arms": ARMS, "bookends": BOOKENDS, "bookend_source": "exp711",
                     "safety_pp": SAFETY_PP, "switch_max": SWITCH_MAX, "informative_pp": INFORMATIVE_PP}}
    (D / "exp712-gdose.json").write_text(json.dumps(out, indent=1))
    print("settled verdict=" + v["verdict"], flush=True)


if __name__ == "__main__":
    main()
