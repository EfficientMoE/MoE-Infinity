#!/usr/bin/env python3
"""exp685: bounded nursery ring — admission-free probation with partition dose.

Michael's design (2026-11-03), from the exp683 decomposition: wrapper
filters pay 1/k of the burst reuse window at the door (k>=2 sighting
admission); the recovery win came from one-timers never displacing the hot
set. exp684 (concurrent) tests the UNBOUNDED probation endpoint (likely
HARM at small cap: burst reuse stride > probation capacity). exp685 adds
the missing axis: a BOUNDED FIFO nursery ring inside the resident set.

RingProbationLCNU (subclass of sealed LayerCycleNextUsePolicy):
- miss admits immediately (zero door latency) and joins the ring
  (insertion-ordered FIFO, capacity = ring_frac * cap, min 1)
- rehit promotes: leaves the ring (full citizen, LCNU scoring protects)
- eviction drains the ring first (oldest one), else base LCNU scoring
  over rehit citizens; >= (1-ring_frac) of the cache always scoring-managed

Arms on exp681's EXACT streams (8 cells):
  lcnu         best standalone (exp681 ref)
  center       frozen composite (REPLICATION vs exp681 per-cell)
  door4096     exp683's best recovery arm (REPLICATION vs exp683 per-cell)
  ring10       RingProbationLCNU, ring_frac 0.10
  ring25       RingProbationLCNU, ring_frac 0.25
  center_ring25 composite with ring25 as its lcnu inner

Dual-Pareto bars (pre-committed, per Michael's ladder):
  REPL1  center == exp681 costfit per-cell exact        (exit 4)
  REPL2  door4096 == exp683 published per-cell exact    (exit 5)
  BURST  (primary, Michael's ladder) center_ring25 burst_def <= center
         burst_def in ALL cells — match/beat exp681 center, erasing the
         door's 12.7-14.8pp admission-delay penalty
  STRETCH center_ring25 burst >= lcnu burst - 0.5pp ALL cells (recorded,
         not verdict-gating)
  RECOV  center_ring25 rec_def at d=4096/cap=512 <= 2.5pp (retain >=
         ~60% of the door's 6.98 -> -1.0pp improvement) AND
         ring25 rec >= lcnu rec - 0.5pp ALL cells
  SAFETY ring arms loop >= lcnu loop - 0.5pp ALL cells
  acc    center double-sim identical (exit 3); det double grid (exit 2)
Verdicts: BOTH_PASS > IN_BURST_FIXED_ONLY > RECOVERY_KEPT_ONLY > RING-HARM.
Exit codes: 0 settled | 2 det | 3 acc | 4 repl1 | 5 repl2.
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
spec82 = importlib.util.spec_from_file_location(
    "d82", str(Path.home() / "warp-work/exp682-moei-door/run_door.py"))
d82 = importlib.util.module_from_spec(spec82)
spec82.loader.exec_module(d82)
rd = st.rd

D = Path(__file__).parent
CostFitPolicy = type(build_policy("costfit", 64, num_layers=4, prior=0.1))
ARMS = ["lcnu", "center", "door4096", "ring10", "ring25", "center_ring25"]


class RingProbationLCNU(LayerCycleNextUsePolicy):
    """Bounded FIFO nursery ring inside the resident set.

    Miss -> admit + join ring (if ring full, the admission itself first
    drains the oldest ring member through _victim). Rehit -> leave the
    ring (promotion). _victim drains ring before LCNU scoring.
    """

    def __init__(self, capacity, ring_frac=0.25, **k):
        super().__init__(capacity, **k)
        if not 0.0 < ring_frac < 1.0:
            raise ValueError("ring_frac in (0,1)")
        self.ring_frac = ring_frac
        self._ring_cap = max(1, int(round(ring_frac * capacity)))
        self._ring = {}  # key -> None, insertion order = admission order

    def _on_hit(self, key):
        self._ring.pop(key, None)  # promotion to scoring-managed citizen

    def _on_miss(self, key):
        super()._on_miss(key)
        self._ring[key] = None

    def _victim(self):
        for k in [k for k in self._ring if k not in self.resident]:
            del self._ring[k]  # lazy cleanup
        if len(self._ring) >= self._ring_cap:
            # ring FULL: one-timers churn among themselves; the remaining
            # (1 - ring_frac) of the cache stays scoring-protected
            k = next(iter(self._ring))  # oldest ring member
            return k[0], k
        # ring has spare capacity: LCNU scoring over all residents (ring
        # members compete by score -> in-burst adaptivity preserved)
        return super()._victim()


class RingCostFit(CostFitPolicy):
    """Frozen composite with RingProbationLCNU as the lcnu inner (t=0)."""

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16,
                 ring_frac=0.25):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self._inners["lcnu"] = RingProbationLCNU(
            capacity, ring_frac=ring_frac, prior=prior,
            num_layers=num_layers)


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}

    def center(cap):
        return build_policy("costfit", cap, **gk)

    out = {}
    for cap in st.CAPS:
        cell = {}
        builders = {
            "lcnu": lambda: build_policy("lcnu", cap, **gk),
            "center": lambda: center(cap),
            "door4096": lambda: d82.mk_door(lambda: center(cap), 4096),
            "ring10": lambda: RingProbationLCNU(cap, ring_frac=0.10, **gk),
            "ring25": lambda: RingProbationLCNU(cap, ring_frac=0.25, **gk),
            "center_ring25": lambda: RingCostFit(cap, ring_frac=0.25, **gk),
        }
        for name in ARMS:
            pol = builders[name]()
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            cell[name] = {"loop": float(h.mean()),
                          "burst": float(h[burst].mean()),
                          "rec": float(h[rec].mean())}
        c2 = center(cap)
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
    r1 = json.loads((Path.home()
                     / "warp-work/exp681-moei-crowd/exp681-crowd.json").read_text())
    r3 = json.loads((Path.home() / "warp-work/exp683-moei-doorcrowd/"
                     "exp683-doorcrowd.json").read_text())
    ref1 = {c["d"]: c["caps"] for c in r1["grid"]}
    ref3 = {c["d"]: c["caps"] for c in r3["grid"]}
    for cell in grid:
        for cap, c in cell["caps"].items():
            a = ref1[cell["d"]][cap]["costfit"]
            for m in ("loop", "burst", "rec"):
                if c["center"][m] != a[m]:
                    return 4, (f"repl1 d={cell['d']} cap={cap} {m}: "
                               f"{c['center'][m]} != {a[m]}")
            b = ref3[cell["d"]][cap]["door4096"]
            for m in ("loop", "burst", "rec"):
                if c["door4096"][m] != b[m]:
                    return 5, (f"repl2 d={cell['d']} cap={cap} {m}: "
                               f"{c['door4096'][m]} != {b[m]}")
    return 0, None


def adjudicate(grid):
    rows = []
    f = {"burst": True, "burst_def": True, "recov": True, "safety": True,
         "recov_n": 0}
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["center"]["loop"] != c["center"]["loop2"]:
                return {"bad": "acc", "d": cell["d"], "cap": cap}
            bb = c["lcnu"]["burst"]
            row = {"d": cell["d"], "cap": int(cap)}
            for a in ARMS:
                row[a] = {m: round(c[a][m], 6) for m in ("loop", "burst", "rec")}
            row["cr_vs_l_burst_pp"] = round((c["center_ring25"]["burst"]
                                             - bb) * 100, 3)
            bd_cf = (bb - c["center"]["burst"]) * 100
            bd_cr = (bb - c["center_ring25"]["burst"]) * 100
            row["burst_def_cf_pp"] = round(bd_cf, 3)
            row["burst_def_cr_pp"] = round(bd_cr, 3)
            rd_l = (c["lcnu"]["rec"] - c["lcnu"]["rec"]) * 100
            row["rec_def_cr_pp"] = round((c["lcnu"]["rec"]
                                          - c["center_ring25"]["rec"]) * 100, 3)
            row["ring25_vs_l_rec_pp"] = round((c["ring25"]["rec"]
                                               - c["lcnu"]["rec"]) * 100, 3)
            for a in ("ring10", "ring25", "center_ring25"):
                if (c[a]["loop"] - c["lcnu"]["loop"]) * 100 < -0.5:
                    f["safety"] = False
            if row["cr_vs_l_burst_pp"] < -0.5:
                f["burst"] = False
            if bd_cr > bd_cf + 1e-9:
                f["burst_def"] = False
            if row["ring25_vs_l_rec_pp"] < -0.5:
                f["recov"] = False
            if cell["d"] == 4096 and int(cap) == 512 and row["rec_def_cr_pp"] <= 2.5:
                f["recov_n"] += 1
            rows.append(row)
    if f["burst_def"] and f["recov"] and f["safety"] and f["recov_n"] == 1:
        v = "BOTH_PASS"
    elif f["burst_def"] and f["safety"]:
        v = "IN_BURST_FIXED_ONLY"
    elif f["recov"] and f["safety"]:
        v = "RECOVERY_KEPT_ONLY"
    else:
        v = "RING-HARM"
    return {"verdict": v, "bars": f, "rows": rows}


if __name__ == "__main__":
    t0 = time.time()
    g1 = run_all()
    res = adjudicate(g1)
    if "bad" in res:
        print("ACC FAIL:", res)
        sys.exit(3)
    rc, msg = replication_check(g1)
    if rc:
        print(msg)
        sys.exit(rc)
    g2 = run_all()
    res["det"] = json.dumps(g1) == json.dumps(g2)
    res["repl"] = "exp681+exp683 exact"
    res["grid"] = g1
    res["rules"] = {"streams": "exp681 exact (68120261103)",
                    "arms": ARMS,
                    "bars": {"REPL1": "center == exp681 costfit exact (exit 4)",
                             "REPL2": "door4096 == exp683 exact (exit 5)",
                             "BURST": "cr burst >= lcnu-0.5 ALL & def <= cf def ALL",
                             "RECOV": "cr rec_def <= 2.5pp @d4096/cap512 & ring25 rec >= lcnu-0.5 ALL",
                             "SAFETY": "ring loops >= lcnu-0.5 ALL"}}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp685-ring.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| det:", res["det"], "| repl:",
          res["repl"], "| bars:", res["bars"])
    for r in res["rows"]:
        print("cell:", {k: r[k] for k in ("d", "cap", "cr_vs_l_burst_pp",
                                          "burst_def_cr_pp", "rec_def_cr_pp",
                                          "ring25_vs_l_rec_pp")})
    sys.exit(0 if res["det"] else 2)
