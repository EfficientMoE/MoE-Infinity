#!/usr/bin/env python3
"""exp694: Capacity-scaling map of the burst/recovery frontier.

exp693 closed the arbitration axis with BOTH_PASS + null capture:
burst-vs-recovery at cap512 is a capacity frontier. Registered next
(exp693 LEDGER + ADDENDUM36): map WHERE the frontier sits as capacity
grows — does ring10's dual advantage (burst +5.96pp, rec +0.73pp over
lcnu at d4096/cap512) survive, shrink, or invert at caps 1024..8192?

Protocol is exp693's exactly (streams exp681 spawn scheme; policies
x85/x93 machinery; single-pass seg metrics 3=burst 4=rec; center
loop2 acc check). Only the CAP grid is new.

Grids:
  repl_grid : st.BURSTS x st.CAPS(512,2048) x {center, ring10}
              — must equal exp681/exp685 JSONs exactly (exit 4/6).
  map_grid  : d in {1024,4096} x caps {512,1024,2048,4096,8192}
              x arms {lcnu, center, ring10, ssr_10}.

Bars (pre-registered, on map_grid):
  Per cap C at d4096 (home depth):
    ring_burst_adv = (lcnu_burst - ring_burst)*100   [pp lcnu loses to ring in burst]
    ring_rec_adv   = (ring_rec - lcnu_rec)*100       [pp ring gains in recovery]
    DUAL at C: burst_adv >= 1.0 AND rec_adv >= 0.30
    LCNU_CATCH at C: lcnu_burst >= ring_burst - 1.0pp AND ring_rec_adv < 0.30
  Verdicts:
    DUAL_STABLE   : DUAL at >= 3 of 5 caps
    LCNU_CATCHES  : LCNU_CATCH at >= 3 of 5 caps (frontier dissolves:
                     capacity alone restores burst AND erases ring rec edge)
    MIXED         : otherwise (report transition cap)
  SAFETY (advisory): ssr_10 loop >= lcnu - 0.5pp all map cells.
  det: rerun d4096 caps {512, 8192} x 4 arms — byte-identical.
Exit: 0 settled | 2 det | 3 acc | 4 repl1 | 6 repl3.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "warp-work/moei-granite3b"))

spec85 = importlib.util.spec_from_file_location(
    "x85", str(HOME / "warp-work/exp685-moei-ring/run_ring.py"))
x85 = importlib.util.module_from_spec(spec85)
sys.modules["x85"] = x85
spec85.loader.exec_module(x85)
st, rd = x85.st, x85.rd
RingProbationLCNU = x85.RingProbationLCNU

spec93 = importlib.util.spec_from_file_location(
    "x93", str(HOME / "warp-work/exp693-moei-ssr/run_ssr.py"))
x93 = importlib.util.module_from_spec(spec93)
sys.modules["x93"] = x93
spec93.loader.exec_module(x93)
SecondSightLCNU = x93.SecondSightLCNU

from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

D = Path(__file__).parent
MAP_CAPS = [512, 1024, 2048, 4096, 8192]
MAP_DEPTHS = [1024, 4096]
MAP_ARMS = ["lcnu", "center", "ring10", "ssr_10"]
DET_CELLS = [(4096, 512), (4096, 8192)]


def mk_arm(name, cap):
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "center":
        return build_policy("costfit", cap, **gk)
    if name == "ring10":
        return RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "ssr_10":
        return SecondSightLCNU(cap, annex_frac=0.10, **gk)
    raise KeyError(name)


def stream_for(d):
    seeds = np.random.default_rng(68120261103).spawn(2)
    base_seed = int(seeds[0].integers(0, 2 ** 63))
    crowd_seed = int(seeds[1].integers(0, 2 ** 63))
    return st.crowd_stream(base_seed, crowd_seed, st.N, d)


def cell_eval(keys, seg, caps, arms):
    burst, rec = (seg == 3), (seg == 4)
    out = {}
    for cap in caps:
        cell = {}
        for name in arms:
            pol = mk_arm(name, cap)
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            cell[name] = {"loop": float(h.mean()),
                          "burst": float(h[burst].mean()),
                          "rec": float(h[rec].mean())}
        out[str(cap)] = cell
    return out


def replication_check(repl_grid):
    r1 = json.loads((HOME / "warp-work/exp681-moei-crowd/exp681-crowd.json").read_text())
    r5 = json.loads((HOME / "warp-work/exp685-moei-ring/exp685-ring.json").read_text())
    ref1 = {c["d"]: c["caps"] for c in r1["grid"]}
    ref5 = {c["d"]: c["caps"] for c in r5["grid"]}
    for cell in repl_grid:
        for cap, c in cell["caps"].items():
            for ref, arm, refname, code in (
                    (ref1, "center", "costfit", 4),
                    (ref5, "ring10", "ring10", 6)):
                a = ref[cell["d"]][cap][refname]
                for m in ("loop", "burst", "rec"):
                    if c[arm][m] != a[m]:
                        return code, (f"repl d={cell['d']} cap={cap} {arm} "
                                      f"{m}: {c[arm][m]} != {a[m]}")
    return 0, None


def main():
    t0 = time.time()
    # repl grid: original caps/depths, center+ring10 only
    repl_grid = []
    for d in st.BURSTS:
        keys, seg, _mid = stream_for(d)
        repl_grid.append({"d": d, "caps": cell_eval(keys, seg, st.CAPS,
                                                    ["center", "ring10"])})
        print(f"repl d={d} done [{time.time()-t0:.0f}s]", flush=True)
    rc, msg = replication_check(repl_grid)
    if rc:
        print(msg)
        sys.exit(rc)
    print(f"REPL OK [{time.time()-t0:.0f}s]", flush=True)

    # map grid
    map_grid = []
    for d in MAP_DEPTHS:
        keys, seg, _mid = stream_for(d)
        map_grid.append({"d": d, "caps": cell_eval(keys, seg, MAP_CAPS, MAP_ARMS)})
        print(f"map d={d} done [{time.time()-t0:.0f}s]", flush=True)

    # determinism spot: rerun two home-depth cells
    det = True
    keys, seg, _mid = stream_for(4096)
    for cap in (512, 8192):
        c = cell_eval(keys, seg, [cap], MAP_ARMS)
        first = map_grid[1]["caps"][str(cap)]
        for name in MAP_ARMS:
            for m in ("loop", "burst", "rec"):
                if c[str(cap)][name][m] != first[name][m]:
                    det = False
    print(f"det={det} [{time.time()-t0:.0f}s]", flush=True)

    # adjudicate: map table at d4096 (home depth) per cap
    home = {int(k): v for k, v in map_grid[1]["caps"].items()}
    table = []
    dual = catch = 0
    safety = True
    for cap in MAP_CAPS:
        h = home[cap]
        b_adv = (h["lcnu"]["burst"] - h["ring10"]["burst"]) * 100
        r_adv = (h["ring10"]["rec"] - h["lcnu"]["rec"]) * 100
        is_dual = b_adv >= 1.0 and r_adv >= 0.30
        is_catch = (h["lcnu"]["burst"] - h["ring10"]["burst"]) * 100 <= 1.0 and r_adv < 0.30
        dual += is_dual
        catch += is_catch
        if (h["ssr_10"]["loop"] - h["lcnu"]["loop"]) * 100 < -0.5:
            safety = False
        table.append({"cap": cap, "lcnu_burst": round(h["lcnu"]["burst"], 4),
                      "ring_burst": round(h["ring10"]["burst"], 4),
                      "ring_burst_adv_pp": round(b_adv, 3),
                      "ring_rec_adv_pp": round(r_adv, 3),
                      "lcnu_rec": round(h["lcnu"]["rec"], 4),
                      "ring_rec": round(h["ring10"]["rec"], 4),
                      "dual": is_dual, "lcnu_catch": is_catch})
    if dual >= 3:
        verdict = "DUAL_STABLE"
    elif catch >= 3:
        verdict = "LCNU_CATCHES"
    else:
        verdict = "MIXED"
    out = {"exp": "exp694-capmap", "verdict": verdict, "det": det,
           "repl": "exp681+exp685 exact (center/ring10 @512,2048 all depths)",
           "table_d4096": table, "map_grid": map_grid, "repl_grid": repl_grid,
           "ssr_safety": safety,
           "elapsed_s": round(time.time() - t0, 1)}
    (D / "exp694-capmap.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", verdict, "| det:", det, "| ssr_safety:", safety)
    for r in table:
        print("cap", r["cap"], "| burst_adv", r["ring_burst_adv_pp"],
              "| rec_adv", r["ring_rec_adv_pp"], "| dual", r["dual"],
              "| catch", r["lcnu_catch"])
    sys.exit(0 if det else 2)


if __name__ == "__main__":
    main()
