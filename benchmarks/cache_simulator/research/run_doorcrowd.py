#!/usr/bin/env python3
"""exp683: doorkeeper under flash crowds — the home-field test.

exp681 (CROWD-H1): cold-crowd bursts cost the center up to 10.86pp in-burst
and 6.98pp in recovery; one-touch crowd admissions evict the hot set and
repeating crowd keys admit too slowly. exp682 (DOOR-HARM): the doorkeeper
hurts on stationary streams (no crowd to block, constant double-miss tax)
— the correct null-space result. exp683 puts the door where its blocked
event actually occurs: exp681's EXACT crowd streams (same seeds), arms
lcnu/s3fifo/costfit-center/door512/door4096, all 8 exp681 cells.

PRIMARY bar (mechanism): door shrinks the center's burst deficit vs best
standalone in >= half of cells (H1); recovery deficit shrinks in >= half
(H3). SAFETY (H2): door loop hit >= center loop hit - 0.5pp everywhere.
acc: center double-sim identical. det: byte-identical double grid.
Verdict ladder: DOOR-CROWD-WINS > DOOR-CROWD-NULL > DOOR-CROWD-HARM.
If HARM: admission filtering is dead for this policy family; the fix must
live inside inner admission — pre-registered consequence.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "st", str(Path.home() / "warp-work/exp681-moei-crowd/run_crowd.py")
)
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)
spec2 = importlib.util.spec_from_file_location(
    "d82", str(Path.home() / "warp-work/exp682-moei-door/run_door.py")
)
d82 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(d82)
rd = st.rd

D = Path(__file__).parent
ARMS = ["lcnu", "s3fifo", "costfit", "door512", "door4096"]


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    out = {}
    for cap in st.CAPS:
        cell = {}
        builders = {
            "lcnu": lambda: rd.build_policy("lcnu", cap),
            "s3fifo": lambda: rd.build_policy("s3fifo", cap),
            "costfit": lambda: rd._mk_cf(cap, gk),
            "door512": lambda: d82.mk_door(lambda: rd._mk_cf(cap, gk), 512),
            "door4096": lambda: d82.mk_door(lambda: rd._mk_cf(cap, gk), 4096),
        }
        for name in ARMS:
            pol = builders[name]()
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            cell[name] = {
                "loop": float(h.mean()),
                "burst": float(h[burst].mean()),
                "rec": float(h[rec].mean()),
            }
        c2 = rd._mk_cf(cap, gk)
        h2 = np.array([c2.on_access(k) for k in keys], dtype=bool)
        cell["costfit"]["loop2"] = float(h2.mean())
        out[str(cap)] = cell
    return out


def run_all():
    grid, t0 = [], time.time()
    seeds = np.random.default_rng(68120261103).spawn(2)  # exp681 scheme, FRESH
    base_seed = int(seeds[0].integers(0, 2**63))
    crowd_seed = int(seeds[1].integers(0, 2**63))
    for d in st.BURSTS:
        keys, seg, mid = st.crowd_stream(base_seed, crowd_seed, st.N, d)
        grid.append({"d": d, "caps": evaluate(keys, seg)})
        print(f"d={d} done [{time.time() - t0:.0f}s]", flush=True)
    return grid


def adjudicate(grid):
    rows = []
    h1_n = h1_p = h3_n = h3_p = h2_bad = 0
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["costfit"]["loop"] != c["costfit"]["loop2"]:
                rows.append({"bad": "acc", "d": cell["d"], "cap": cap})
                continue
            bb = max(c["lcnu"]["burst"], c["s3fifo"]["burst"])
            br = max(c["lcnu"]["rec"], c["s3fifo"]["rec"])
            dcb = (bb - c["costfit"]["burst"]) * 100
            dcr = (br - c["costfit"]["rec"]) * 100
            for arm in ("door512", "door4096"):
                db = (bb - c[arm]["burst"]) * 100
                dr = (br - c[arm]["rec"]) * 100
                dl = (c[arm]["loop"] - c["costfit"]["loop"]) * 100
                h1_n += 1
                if db < dcb:
                    h1_p += 1
                h3_n += 1
                if dr < dcr:
                    h3_p += 1
                if dl < -0.5:
                    h2_bad += 1
                rows.append(
                    {
                        "d": cell["d"],
                        "cap": int(cap),
                        "arm": arm,
                        "cf_burst_def": round(dcb, 3),
                        "door_burst_def": round(db, 3),
                        "cf_rec_def": round(dcr, 3),
                        "door_rec_def": round(dr, 3),
                        "door_minus_cf_loop_pp": round(dl, 3),
                    }
                )
    verdict = (
        "DOOR-CROWD-WINS"
        if (2 * h1_p >= h1_n and 2 * h3_p >= h3_n and not h2_bad)
        else "DOOR-CROWD-HARM"
        if h2_bad
        else "DOOR-CROWD-NULL"
    )
    return {
        "verdict": verdict,
        "h1_shrink": f"{h1_p}/{h1_n}",
        "h3_shrink": f"{h3_p}/{h3_n}",
        "h2_bad": h2_bad,
        "rows": rows,
    }


if __name__ == "__main__":
    t0 = time.time()
    g1 = run_all()
    res = adjudicate(g1)
    g2 = run_all()
    res["det"] = json.dumps(g1) == json.dumps(g2)
    res["grid"] = g1
    res["rules"] = {
        "streams": "exp681 exact (68120261103)",
        "arms": ARMS,
        "bars": {
            "H1": "door burst deficit < center's in >=50% cells",
            "H2": "door loop >= center-0.5pp",
            "H3": "door recovery deficit < center's in >=50%",
        },
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp683-doorcrowd.json").write_text(json.dumps(res, indent=1))
    print(
        "VERDICT:",
        res["verdict"],
        "| det:",
        res["det"],
        "| h1 shrink:",
        res["h1_shrink"],
        "| h3 shrink:",
        res["h3_shrink"],
        "| h2bad:",
        res["h2_bad"],
    )
    for r in res["rows"]:
        print("cell:", r)
