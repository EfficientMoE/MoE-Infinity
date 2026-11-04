#!/usr/bin/env python3
"""exp681: sub-block flash-crowd probe — the drift axis arbitration cannot see.

Stationary base (skew 1.1, lam 0.3 — mid locality, no regime change). At the
block-aligned midpoint, replace d accesses with a COLD CROWD: layer uniform,
expert uniform over ranks 64..127 (bottom half by base weight at skew 1.1).
d in {256, 1024, 2048, 4096}; the barrier arbitrates only at 4096 boundaries,
so d < 4096 is structurally invisible to arbitration mid-window; d = 4096 is
the aligned control. caps {512, 2048}; 8 cells. Segments: 3 = burst,
4 = recovery (4096 accesses after burst end), 0 = base.

Bars (frozen in PREREG.md before launch): H1 burst deficit <= 1.0pp,
H2 recovery deficit <= 0.5pp, H3 switches <= 3 per 4096-window, acc
(loop == prop for costfit), det (byte-identical double run; seed spawn FRESH
inside run_all — exp680's disclosed lesson).
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "rd", str(Path.home() / "warp-work/exp676-moei-drift/run_drift.py"))
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

D = Path(__file__).parent
B = 1024
N = 638_976
WIN = 4096
BURSTS = [256, 1024, 2048, 4096]
CAPS = [512, 2048]
S, LAM = 1.1, 0.3
COLD_LO = 64  # crowd experts: uniform over [64, 128)
BASE_SEED_H = "exp681-base"


def _base(seed, n_acc):
    """Stationary blockwise stream, single regime (S, LAM)."""
    assert n_acc % B == 0
    rng = np.random.default_rng(seed)
    n_blocks = n_acc // B
    ranks = np.arange(1, rd.E + 1, dtype=np.float64)
    p = ranks ** (-S); p /= p.sum()
    keys = []
    for _ in range(n_blocks):
        ex = rng.choice(rd.E, size=B, p=p)
        lay = rng.integers(0, rd.L, size=B)
        blk = list(zip(lay.tolist(), ex.tolist()))
        n_replay = int(LAM * B)
        if n_replay:
            idx = rng.choice(np.arange(1, B), size=n_replay, replace=False)
            w = 0.5 ** (np.arange(1, rd.R_WINDOW + 1) / rd.H_RECENCY); w /= w.sum()
            ages = rng.choice(rd.R_WINDOW, size=n_replay, p=w) + 1
            ages = np.minimum(ages, idx)
            for i, a_ in zip(idx.tolist(), ages.tolist()):
                blk[i] = blk[i - a_]
        keys.extend(blk)
    return keys


def crowd_stream(base_seed, crowd_seed, n_acc, d):
    """Overlay a cold-crowd burst of d accesses at the block-aligned midpoint.

    Same base seed + crowd-prefix design across cells: base identical, crowd
    draws identical prefix, only d (and thus recovery start) varies.
    """
    keys = _base(base_seed, n_acc)
    rng = np.random.default_rng(crowd_seed)
    lay = rng.integers(0, rd.L, size=BURSTS[-1])
    ex = rng.integers(COLD_LO, rd.E, size=BURSTS[-1])
    mid = (n_acc // 2 // B) * B
    for j in range(d):
        keys[mid + j] = (int(lay[j]), int(ex[j]))
    seg = np.zeros(n_acc, dtype=np.int64)
    seg[mid:mid + d] = 3
    seg[mid + d:mid + d + WIN] = 4
    return keys, seg, mid


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    out = {}
    for cap in CAPS:
        cell = {}
        for name in ("lcnu", "s3fifo"):
            pol = rd.build_policy(name, cap)
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            cell[name] = {"loop": h.mean(), "burst": h[burst].mean(),
                          "rec": h[rec].mean()}
        pol = rd._mk_cf(cap, gk)
        h = np.array([pol.on_access(k) for k in keys], dtype=bool)
        cell["costfit"] = {"loop": h.mean(), "burst": h[burst].mean(),
                           "rec": h[rec].mean(), "prop": pol.effective_hit_rate,
                           "max_sw_win": rd.window_switches(
                               keys, lambda: rd._mk_cf(cap, gk), win=WIN)}
        out[str(cap)] = cell
    return out


def adjudicate(grid):
    rows, viol_h1, viol_h2, viol_h3 = [], 0, 0, 0
    for cell in grid:
        for cap, c in cell["caps"].items():
            best_b = max(c["lcnu"]["burst"], c["s3fifo"]["burst"])
            best_r = max(c["lcnu"]["rec"], c["s3fifo"]["rec"])
            d_b = (best_b - c["costfit"]["burst"]) * 100
            d_r = (best_r - c["costfit"]["rec"]) * 100
            acc = c["costfit"]["prop"] == c["costfit"]["loop"]
            if d_b > 1.0:
                viol_h1 += 1
            if d_r > 0.5:
                viol_h2 += 1
            if c["costfit"]["max_sw_win"] > 3:
                viol_h3 += 1
            rows.append({"d": cell["d"], "cap": int(cap),
                         "d_burst_pp": round(float(d_b), 3),
                         "d_rec_pp": round(float(d_r), 3),
                         "sw": int(c["costfit"]["max_sw_win"]),
                         "burst_cf": float(c["costfit"]["burst"]),
                         "rec_cf": float(c["costfit"]["rec"]),
                         "acc": bool(acc)})
    det = True
    return {"verdict": ("CROWD-H1" if viol_h1 else "CROWD-H2" if viol_h2
                        else "CROWD-SAFE"),
            "h1_viol": viol_h1, "h2_viol": viol_h2, "h3_viol": viol_h3,
            "acc_all": all(r["acc"] for r in rows), "rows": rows, "det": det}


def run_all():
    grid, t0 = [], time.time()
    # FRESH spawn inside run_all (exp680 lesson): identical draws each call.
    seeds = np.random.default_rng(68120261103).spawn(2)
    base_seed = int(seeds[0].integers(0, 2 ** 63))
    crowd_seed = int(seeds[1].integers(0, 2 ** 63))
    for d in BURSTS:
        keys, seg, mid = crowd_stream(base_seed, crowd_seed, N, d)
        res = evaluate(keys, seg)
        grid.append({"d": d, "caps": res})
        print(f"d={d} done [{time.time() - t0:.0f}s]", flush=True)
    return grid


if __name__ == "__main__":
    t0 = time.time()
    g1 = run_all()
    res = adjudicate(g1)
    g2 = run_all()
    res["det"] = json.dumps(g1) == json.dumps(g2)
    res["grid"] = g1
    res["rules"] = {"base": f"skew {S} lam {LAM} stationary, B={B}",
                    "crowd": f"uniform ranks [{COLD_LO},{rd.E}) at midpoint",
                    "bursts": BURSTS, "caps": CAPS,
                    "bars": {"h1_burst_pp": 1.0, "h2_rec_pp": 0.5,
                             "h3_sw": 3}}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp681-crowd.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| det:", res["det"],
          "| h1:", res["h1_viol"], "h2:", res["h2_viol"],
          "h3:", res["h3_viol"], "| acc:", res["acc_all"])
    for r in res["rows"]:
        print("cell:", r)
