#!/usr/bin/env python3
"""exp711: full 2^3 factorial attribution of exp674's SAFETY-BROKEN joint BO point.

Factors (center -> exp669 BO value, each read at runtime from sealed exp674 bo_params):
  M = MARGIN 0.005 -> bo margin, G = GAMMA 0.0025 -> bo gamma, H = HALF_LIFE 32768 -> bo half_life.
Arms: lcnu, s3fifo + costfit at all 8 corners cf_{000,M00,0G0,MG0,00H,M0H,0GH,MGH}.
H0: three corners are already sealed in exp674 and must replicate EXACTLY per cell
(cf_000==cf_center, cf_00H==cf_hlonly, cf_MGH==cf_bo; standalones too).
Streams: exactly exp670/exp674/exp710's 11 (sha8-asserted vs research/exp670-synth.json).
Adjudication per frozen PREREG. No hand-transcribed values.

Modes: default = full frozen run.  SMOKE=1 = disclosed machinery smoke (full-N H0
replication on 1 stream x 2 caps with ONLY the three sealed-bookend corners; reduced
unpinned grid N=60k for all 10 arms).
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
_BO = json.loads(EXP674.read_text())["bo_params"]
BO = (float(_BO["margin"]), float(_BO["gamma"]), float(_BO["half_life"]))
CORNERS = ["000", "M00", "0G0", "MG0", "00H", "M0H", "0GH", "MGH"]
BOOKENDS = {"000": "cf_center", "00H": "cf_hlonly", "MGH": "cf_bo"}
SAFETY_PP, SWITCH_MAX, INFORMATIVE_PP = 0.5, 6, 0.1


def corner_params(c):
    m = BO[0] if c[0] == "M" else CENTER[0]
    g = BO[1] if c[1] == "G" else CENTER[1]
    h = BO[2] if c[2] == "H" else CENTER[2]
    return (m, g, h)


def arm(c):
    return f"cf_{c}"


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


def evaluate(keys, E, L, caps, corners, top_k=8):
    out = {}
    gk = {"num_layers": L, "prior": top_k / E}
    n = len(keys)
    for cap in caps:
        cell = {}
        for name in ("lcnu", "s3fifo"):
            pol = build_policy(name, cap)
            hits = sum(1 for k in keys if pol.on_access(k))
            cell[name] = {"rate": round(hits / n, 6), "prop": round(pol.effective_hit_rate, 6)}
        for c in corners:
            p = corner_params(c)
            pol = make_costfit(cap, gk, p)
            assert (pol.MARGIN, pol.GAMMA, pol.HALF_LIFE) == p and pol._decay == 0.5 ** (1.0 / p[2])
            hits = sum(1 for k in keys if pol.on_access(k))
            cell[arm(c)] = {"rate": round(hits / n, 6), "prop": round(pol.effective_hit_rate, 6),
                            "switches": getattr(pol, "switches", 0)}
        out[str(cap)] = cell
    return out


def stream_cfgs():
    streams = [{"s": s, "lam": lam, "E": 128, "L": 48} for s in SKEWS for lam in LOCALITIES]
    streams.append({"s": 1.1, "lam": 0.3, "E": 32, "L": 32})
    streams.append({"s": 1.4, "lam": 0.6, "E": 32, "L": 32})
    return streams


def run_all(expected_shas, idxs=None, caps=CAPS, corners=CORNERS, n_acc=N_ACC, check_sha=True, seed_perturb=0):
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
        res = evaluate(keys, cfg["E"], cfg["L"], caps, corners)
        grid.append({"cfg": cfg, "seed": sha, "caps": res})
        print(f"stream {i} done [{time.time()-t0:.0f}s]", flush=True)  # timings only until settlement
    return grid


def h0_replication(grid, ref_grid):
    ref = {(json.dumps(st["cfg"], sort_keys=True), cap): cell for st in ref_grid for cap, cell in st["caps"].items()}
    mism, n = [], 0
    for st in grid:
        for cap, cell in st["caps"].items():
            r = ref[(json.dumps(st["cfg"], sort_keys=True), cap)]
            for c, refarm in BOOKENDS.items():
                if arm(c) not in cell:
                    continue
                n += 1
                if cell[arm(c)] != r[refarm]:
                    mism.append({"cfg": st["cfg"], "cap": cap, "arm": arm(c), "ref": refarm})
            for a in ("lcnu", "s3fifo"):
                n += 1
                if cell[a] != r[a]:
                    mism.append({"cfg": st["cfg"], "cap": cap, "arm": a, "ref": a})
    return {"cells": n, "mismatches": mism, "ok": n > 0 and not mism}


def classify(safe):
    """Frozen attribution ladder (PREREG). safe: dict corner -> H1 bool."""
    if not safe["MG0"]:
        m, g = safe["M00"], safe["0G0"]
        if not m and g:
            return "MG-BREAK:M-DRIVEN"
        if m and not g:
            return "MG-BREAK:G-DRIVEN"
        if not m and not g:
            return "MG-BREAK:BOTH-KNOBS"
        return "MG-BREAK:SYNERGY"
    if safe["MGH"]:
        return "NO-BREAK"
    mh, gh = safe["M0H"], safe["0GH"]
    if not mh and gh:
        return "HL-INTERACTION:MxH"
    if mh and not gh:
        return "HL-INTERACTION:GxH"
    if not mh and not gh:
        return "HL-INTERACTION:BOTH"
    return "HL-INTERACTION:THREE-WAY"


def yates(vals):
    """Standard 2^3 effects (main + interactions) on per-corner scalar values."""
    sign = lambda c, f: 1 if f in c else -1  # noqa: E731
    eff = {}
    for name, fs in [("M", "M"), ("G", "G"), ("H", "H"), ("MG", "MG"), ("MH", "MH"), ("GH", "GH"), ("MGH", "MGH")]:
        s = 0.0
        for c in CORNERS:
            prod = 1
            for f in fs:
                prod *= sign(c, f)
            s += prod * vals[c]
        eff[name] = round(s / 4.0, 4)
    return eff


def adjudicate(grid, h0, corners=CORNERS):
    informative, acct = 0, []
    per = {c: {"worst": None, "safety_cells": 0, "sw_cells": 0, "max_sw": 0, "deltas": [], "over": []} for c in corners}
    for st in grid:
        for cap, cell in st["caps"].items():
            for c in corners:
                a = cell[arm(c)]
                if a["rate"] != a["prop"]:
                    acct.append({"cfg": st["cfg"], "cap": cap, "arm": arm(c)})
            lc, s3 = cell["lcnu"]["rate"], cell["s3fifo"]["rate"]
            if abs(lc - s3) * 100 < INFORMATIVE_PP:
                continue
            informative += 1
            best = max(lc, s3)
            for c in corners:
                a = cell[arm(c)]; p = per[c]
                d = (best - a["rate"]) * 100
                p["deltas"].append((a["rate"] - cell[arm("000")]["rate"]) * 100)
                p["max_sw"] = max(p["max_sw"], a["switches"])
                if p["worst"] is None or d > p["worst"]["deficit_pp"]:
                    p["worst"] = {"cfg": st["cfg"], "cap": cap, "deficit_pp": round(d, 3), "switches": a["switches"]}
                if d > SAFETY_PP:
                    p["safety_cells"] += 1
                    p["over"].append({"cfg": st["cfg"], "cap": cap, "deficit_pp": round(d, 3)})
                if a["switches"] > SWITCH_MAX:
                    p["sw_cells"] += 1
    arms = {}
    for c in corners:
        p = per[c]
        arms[c] = {"params": list(corner_params(c)), "H1_safe": p["safety_cells"] == 0 and p["sw_cells"] == 0,
                   "worst": p["worst"], "cells_over_bar": p["safety_cells"], "over_cells": p["over"],
                   "cells_over_switch": p["sw_cells"], "max_switches": p["max_sw"],
                   "mean_delta_vs_000_pp": round(sum(p["deltas"]) / len(p["deltas"]), 4) if p["deltas"] else None}
    h4 = not acct
    if not (h0["ok"] and h4):
        verdict = "ACCOUNTING-FAIL"
    elif set(corners) == set(CORNERS):
        verdict = classify({c: arms[c]["H1_safe"] for c in CORNERS})
    else:
        verdict = "PARTIAL-GRID"
    effects = None
    if set(corners) == set(CORNERS) and all(arms[c]["worst"] for c in CORNERS):
        effects = {"worst_deficit_pp": yates({c: arms[c]["worst"]["deficit_pp"] for c in CORNERS}),
                   "mean_delta_pp": yates({c: arms[c]["mean_delta_vs_000_pp"] for c in CORNERS})}
    return {"verdict": verdict, "H0_replication": h0, "H4_accounting": h4, "accounting_violations": acct,
            "informative_cells": informative, "arms": arms, "effects": effects}


def smoke():
    exp670 = json.loads(EXP670.read_text()); shas = [st["seed"] for st in exp670["grid"]]
    ref = json.loads(EXP674.read_text())["grid"]
    try:
        run_all(shas, idxs={0}, caps=[], corners=[], seed_perturb=1)
        print("SMOKE S1 FAIL: perturbed seed not caught"); return 1
    except AssertionError:
        print("SMOKE S1 ok: sha assertion bites on perturbed seed")
    g = run_all(shas, idxs={4}, caps=[512, 2048], corners=list(BOOKENDS))
    h0 = h0_replication(g, ref)
    assert h0["cells"] == 10, h0
    print("SMOKE S2 H0 machinery replication cells", h0["cells"], "ok", h0["ok"])
    bad = json.loads(json.dumps(g)); bad[0]["caps"]["512"][arm("MGH")]["rate"] += 1e-6
    print("SMOKE S3 H0 negative control detects", not h0_replication(bad, ref)["ok"])
    r1 = run_all(shas, idxs={0, 9}, caps=[128, 256], n_acc=60_000, check_sha=False)
    r2 = run_all(shas, idxs={0, 9}, caps=[128, 256], n_acc=60_000, check_sha=False)
    det = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    adj = adjudicate(r1, {"ok": True, "cells": 0, "mismatches": []})
    wf = set(adj["arms"]) == set(CORNERS) and adj["verdict"] != "PARTIAL-GRID"
    # classify ladder unit checks (pure logic, no data)
    allsafe = {c: True for c in CORNERS}
    lad = [classify({**allsafe, "MG0": False, "M00": False}) == "MG-BREAK:M-DRIVEN",
           classify({**allsafe, "MG0": False, "0G0": False}) == "MG-BREAK:G-DRIVEN",
           classify({**allsafe, "MG0": False}) == "MG-BREAK:SYNERGY",
           classify({**allsafe, "MGH": False}) == "HL-INTERACTION:THREE-WAY",
           classify({**allsafe, "MGH": False, "M0H": False}) == "HL-INTERACTION:MxH",
           classify(allsafe) == "NO-BREAK"]
    print("SMOKE S4 det", det, "adjudicate well-formed", wf, "accounting", adj["H4_accounting"], "ladder", all(lad))
    ok = h0["ok"] and det and wf and adj["H4_accounting"] and all(lad)
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


def main():
    if os.environ.get("SMOKE") == "1":
        sys.exit(smoke())
    exp670 = json.loads(EXP670.read_text()); shas = [st["seed"] for st in exp670["grid"]]
    ref = json.loads(EXP674.read_text())["grid"]
    t0 = time.time()
    g1 = run_all(shas)
    g2 = run_all(shas)
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True), "nondeterministic"
    v = adjudicate(g1, h0_replication(g1, ref))
    out = {**v, "determinism_two_runs": True, "elapsed_s": round(time.time() - t0, 1), "grid": g1,
           "rules": {"caps": CAPS, "skews": SKEWS, "localities": LOCALITIES, "n_acc": N_ACC,
                     "center": list(CENTER), "bo": list(BO), "bo_source": "exp674 bo_params",
                     "corners": CORNERS, "bookends": BOOKENDS,
                     "safety_pp": SAFETY_PP, "switch_max": SWITCH_MAX, "informative_pp": INFORMATIVE_PP}}
    (D / "exp711-mgfactorial.json").write_text(json.dumps(out, indent=1))
    print("settled verdict=" + v["verdict"], flush=True)


if __name__ == "__main__":
    main()
