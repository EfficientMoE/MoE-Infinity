#!/usr/bin/env python3
"""exp682: sub-block flash-crowd probe — transient locality crowds that
arrive and revert entirely INSIDE one arbitration window (RESEGMENT=4096).

Lineage: exp677 found peak transition harm exactly at W=RESEGMENT (0.439pp);
exp680 refuted arbitration cadence as the lever (barrier never fires at the
peak — cadence irrelevant by construction). Residual axis named by
exp676/677: sub-block intra-window shifts. A flash crowd = a two-way
locality spike (lam 0.6 -> 0.95 -> back) of duration D << 4096 with NO
re-segment boundary inside it: arbitration cannot react by construction.
If harm appears anyway it flows through the inner policy's per-access
response. If harm grows with D, duration is the exposure variable.

Streams are built with exp677's exact block machinery (Zipf skew 1.1,
recency-weighted replay for locality, E=128, L=48); stationary base regime
(s1.1, lam0.6); every PERIOD accesses a crowd run of D accesses at lam0.95.
Control cell = exp677's exact W=4096 stream (same seed scheme) and must
reproduce exp677's published peak center deficit (parsed from
exp677-wsweep.json at runtime, never transcribed).
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, str(Path.home() / p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rd = load("rd", "warp-work/exp676-moei-drift/run_drift.py")
ws = load("ws", "warp-work/exp677-moei-wsweep/run_wsweep.py")

D_DIR = Path(__file__).parent
B = 1024
N = 638_976
WIN = 4096
PERIOD = 16_384  # crowd spacing (crowd + recovery never overlap)
DURS = [256, 512, 1024, 2048]
CAPS = [2048, 4096]
S, LAM_BASE, LAM_CROWD = 1.1, 0.6, 0.95
SEED_BASE = 68220261103
E677_FILE = Path.home() / "warp-work/exp677-moei-wsweep/exp677-wsweep.json"
PEAK_D_TOL = 0.005


def gen_block(rng, b, lam, s=S):
    ranks = np.arange(1, rd.E + 1, dtype=np.float64)
    p = ranks ** (-s)
    p /= p.sum()
    ex = rng.choice(rd.E, size=b, p=p)
    lay = rng.integers(0, rd.L, size=b)
    blk = list(zip(lay.tolist(), ex.tolist()))
    n_replay = int(lam * b)
    if n_replay:
        idx = rng.choice(np.arange(1, b), size=n_replay, replace=False)
        w = 0.5 ** (np.arange(1, rd.R_WINDOW + 1) / rd.H_RECENCY)
        w /= w.sum()
        ages = rng.choice(rd.R_WINDOW, size=n_replay, p=w) + 1
        ages = np.minimum(ages, idx)
        for i, a_ in zip(idx.tolist(), ages.tolist()):
            blk[i] = blk[i - a_]
    return blk


def crowd_starts(n_acc):
    return list(range(PERIOD // 2, n_acc, PERIOD))


def gen_crowd_stream(seed, dur, n_acc):
    """Stationary base; crowd runs of `dur` accesses at lam 0.95 starting
    at PERIOD//2 then every PERIOD accesses. Crowd + recovery (B accesses)
    never overlap the next crowd by construction (PERIOD >> DURS + B)."""
    assert n_acc % B == 0 and dur <= PERIOD - 2 * B
    rng = np.random.default_rng(seed)
    keys, crowd = [], np.zeros(n_acc, dtype=bool)
    starts = crowd_starts(n_acc)
    start_set = set(starts)
    rem = 0
    for pos in range(0, n_acc, B):
        blk = []
        off = 0
        while off < B:
            if pos + off in start_set:
                rem = dur
            take = min(B - off, rem) if rem else 0
            if take:
                blk.extend(gen_block(rng, take, LAM_CROWD))
                crowd[pos + off : pos + off + take] = True
                rem -= take
                off += take
            else:
                nb = B - off
                blk.extend(gen_block(rng, nb, LAM_BASE))
                off += nb
        keys.extend(blk)
    post = np.zeros(n_acc, dtype=bool)
    for st in starts:
        end = st + dur
        post[end : min(end + B, n_acc)] = True
    assert not (crowd & post).any()
    assert crowd.sum() == len(starts) * dur
    return keys, crowd, post


def hits_of(keys, mk):
    pol = mk()
    h = np.zeros(len(keys), dtype=np.bool_)
    for i, k in enumerate(keys):
        h[i] = pol.on_access(k)
    return h, pol


def evaluate_crowd(keys, crowd, post, caps):
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    out = {}
    for cap in caps:
        cell = {}
        for name in ("lcnu", "s3fifo"):
            h, _ = hits_of(keys, lambda n_=name: rd.build_policy(n_, cap))
            cell[name] = {
                "loop": float(h.mean()),
                "crowd": float(h[crowd].mean()),
                "post": float(h[post].mean()),
            }
        h, pol = hits_of(keys, lambda: rd._mk_cf(cap, gk))
        cell["costfit"] = {
            "loop": float(h.mean()),
            "crowd": float(h[crowd].mean()),
            "post": float(h[post].mean()),
            "prop": float(pol.effective_hit_rate),
            "max_sw_win": rd.window_switches(
                keys, lambda: rd._mk_cf(cap, gk), win=WIN
            ),
        }
        out[str(cap)] = cell
    return out


def control_cell():
    """exp677's exact W=4096 stream, cap 4096, center arm: must reproduce
    exp677's published peak center deficit (parsed at runtime)."""
    seed = int(
        np.random.default_rng(67720261103).spawn(4)[2].integers(0, 2**63)
    )  # ws.WIDTHS[2] == 4096
    keys, seg, sw_b, wb = ws._gen(seed, 4096, N, B)
    trans = seg == 0
    assert trans.sum() == 4096
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    cap = 4096
    rates = {}
    for name in ("lcnu", "s3fifo"):
        h, _ = hits_of(keys, lambda n_=name: rd.build_policy(n_, cap))
        rates[name] = float(h[trans].mean())
    h, pol = hits_of(keys, lambda: rd._mk_cf(cap, gk))
    rates["costfit"] = float(h[trans].mean())
    d_ours = (max(rates["lcnu"], rates["s3fifo"]) - rates["costfit"]) * 100
    r677 = json.loads(E677_FILE.read_text())
    c677 = next(s for s in r677["grid"] if s["width"] == 4096)["caps"]["4096"]
    b677 = max(c677["lcnu"]["trans"], c677["s3fifo"]["trans"])
    d_677 = (b677 - c677["costfit"]["trans"]) * 100
    return {
        "d_center_trans_pp": d_ours,
        "exp677_file_d_pp": d_677,
        "replicates": abs(d_ours - d_677) < PEAK_D_TOL,
    }


def run_grid():
    grid, t0 = [], time.time()
    seeds = np.random.default_rng(SEED_BASE).spawn(len(DURS))
    for i, dur in enumerate(DURS):
        seed = int(seeds[i].integers(0, 2**63))
        keys, crowd, post = gen_crowd_stream(seed, dur, N)
        grid.append(
            {
                "dur": dur,
                "n_crowds": int(crowd.sum() // dur),
                "caps": evaluate_crowd(keys, crowd, post, CAPS),
            }
        )
        print(f"D={dur} done [{time.time()-t0:.0f}s]", flush=True)
    return grid


def adjudicate(grid, control):
    rows, viol = [], []
    for st in grid:
        for cap in CAPS:
            c = st["caps"][str(cap)]
            bc = max(c["lcnu"]["crowd"], c["s3fifo"]["crowd"])
            bp = max(c["lcnu"]["post"], c["s3fifo"]["post"])
            row = {
                "dur": st["dur"],
                "cap": cap,
                "d_crowd_pp": round((bc - c["costfit"]["crowd"]) * 100, 3),
                "d_post_pp": round((bp - c["costfit"]["post"]) * 100, 3),
                "sw": c["costfit"]["max_sw_win"],
                "acc": c["costfit"]["loop"] == c["costfit"]["prop"],
            }
            rows.append(row)
            if row["d_crowd_pp"] > 1.0:
                viol.append({**row, "mode": "H1"})
            if row["d_post_pp"] > 0.5:
                viol.append({**row, "mode": "H2"})
            if row["sw"] > 3:
                viol.append({**row, "mode": "H3"})
            if not row["acc"]:
                viol.append({**row, "mode": "ACC"})
    h4_val = max(
        next(
            r["d_crowd_pp"]
            for r in rows
            if r["dur"] == 2048 and r["cap"] == cap
        )
        - next(
            r["d_crowd_pp"] for r in rows if r["dur"] == 256 and r["cap"] == cap
        )
        for cap in CAPS
    )
    h4 = h4_val > 0.1
    repl = control["replicates"]
    safe = not viol
    verdict = (
        "DURATION-SENSITIVE"
        if safe and h4 and repl
        else "CROWD-SAFE"
        if safe and repl
        else "REPLICATION-FAIL"
        if safe
        else "CROWD-BROKEN"
    )
    return {
        "verdict": verdict,
        "rows": rows,
        "cells": len(rows),
        "violations": viol,
        "h4_duration_effect_pp": round(h4_val, 3),
        "h4_threshold_0p1pp": h4,
        "control": control,
        "replication_of_exp677": repl,
    }, rows


def main():
    grid = run_grid()
    control = control_cell()
    res, rows = adjudicate(grid, control)
    grid2 = run_grid()
    det = json.dumps(grid, sort_keys=True) == json.dumps(grid2, sort_keys=True)
    out = {
        **res,
        "determinism": det,
        "grid": grid,
        "rules": {
            "N": N,
            "B": B,
            "period": PERIOD,
            "durs": DURS,
            "caps": CAPS,
            "win": WIN,
            "skew": S,
            "lam_base": LAM_BASE,
            "lam_crowd": LAM_CROWD,
            "seed_base": SEED_BASE,
            "bars": {
                "H1_crowd": 1.0,
                "H2_post": 0.5,
                "H3_sw_per_4096": 3,
                "H4_duration": 0.1,
            },
            "center_params": list(rd.CENTER),
            "stream": "exp677 block machinery, stationary + crowds",
            "control": "exp677 exact W=4096 stream (spawn idx 2)",
        },
    }
    (D_DIR / "exp682-flashcrowd.json").write_text(json.dumps(out, indent=1))
    print(
        "VERDICT:",
        out["verdict"],
        "| det:",
        det,
        "| h4_val:",
        out["h4_duration_effect_pp"],
        "| repl:",
        out["replication_of_exp677"],
    )
    for r in rows:
        print("cell:", r)
    for x in res["violations"][:8]:
        print("violation:", x)


def smoke():
    """Disclosed pre-freeze smoke: reduced stream, one duration, both caps.
    Verifies crowd/post accounting, policy accounting, and double-run
    determinism on one cell. No adjudication (reduced scale)."""
    n = N // 4
    seed = int(np.random.default_rng(SEED_BASE).spawn(4)[0].integers(0, 2**63))
    keys, crowd, post = gen_crowd_stream(seed, 1024, n)
    print(
        f"smoke N={n} D=1024 crowds={int(crowd.sum())} "
        f"posts={int(post.sum())} overlap={int((crowd & post).sum())}"
    )
    assert crowd.sum() == len(crowd_starts(n)) * 1024
    cell = evaluate_crowd(keys, crowd, post, CAPS)
    for cap in CAPS:
        c = cell[str(cap)]
        acc = c["costfit"]["loop"] == c["costfit"]["prop"]
        print(
            f"cap {cap}: lcnu {c['lcnu']['crowd']:.4f} s3fifo "
            f"{c['s3fifo']['crowd']:.4f} cf {c['costfit']['crowd']:.4f} "
            f"| post cf {c['costfit']['post']:.4f} | sw "
            f"{c['costfit']['max_sw_win']} | acc {acc}"
        )
        assert acc
    keys2, crowd2, post2 = gen_crowd_stream(seed, 1024, n)
    det = keys == keys2 and (crowd == crowd2).all() and (post == post2).all()
    print("stream determinism (double gen):", det)
    assert det
    print("SMOKE OK")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
