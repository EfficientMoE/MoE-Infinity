#!/usr/bin/env python3
"""exp686a: ring as a FIRST-CLASS arbitration candidate (visibility test).

exp685 verdict IN_BURST_FIXED_ONLY: the ring fixes victim selection
(standalone Pareto point) but the composite gate failed at d4096/cap512
with rec_def +6.98pp == center EXACTLY — center_ring25 is byte-equal to
center, i.e. the ring inner is INVISIBLE to 2-way arbitration at small
cap (probe 2026-11-03: center's choice timeline at cap512 is
lcnu->s3fifo @8191 then s3fifo for the WHOLE stream, burst and recovery
included; at cap2048 it returns to lcnu @135167 before the burst).

exp686a (registered branch of exp685 + data amendment): make the ring a
FIRST-CANDIDATE of the sealed CostFit3 3-way barrier (exp673 machinery:
decayed cumulative-rate scoreboard, MARGIN 0.005 + GAMMA 0.0025 *
decayed switch mass, RESEGMENT 4096, initial lcnu — ALL sealed params
inherited unedited from policies_costfit3.CostFit3Policy; only the
candidate set changes: {lcnu, s3fifo, ring}).

PRE-REGISTERED PREDICTION (frozen before the grid): at cap512 the ring
cannot clear the barrier (its decayed cumulative rate tracks lcnu's,
which already fails to displace s3fifo's base lead; needed lead
~0.0075 vs a ~0.006 one-window boost) -> H1 FAILS, verdict
IN_BURST_KEPT_VISIBLE. The genuinely uncertain cell: ring10's
whole-stream loop at cap512 (0.6805) EXCEEDS s3fifo's (0.6768) — if any
re-segment boundary puts ring10's decayed rate ahead by >= barrier,
adoption happens and H1 can pass. Either outcome decides exp687.

Secondary (frozen branch text): lazy promotion dose — promotion out of
the nursery ring deferred to the 2nd rehit (3 sightings total), so
k=2 crowd keys never displace citizens. Stride diagnostic: pure stream
statistics (crowd rehit gaps, concurrent probation load) to size
exp686b's ring; NOT a policy measurement.

Arms on exp681's EXACT streams (seeds 68120261103 spawn scheme), 8 cells:
  lcnu | s3fifo | center (REPL1 vs exp681 costfit) |
  ring10 | ring25 (REPL2 vs exp685) |
  lazy2 (LazyRingProbationLCNU, ring_frac .25, promote_after 2) |
  ringfit3 (RingFit3, ring .25) | ringfit3_10 (ring .10) |
  ringfit3_lazy2 (ring .25, lazy)

Bars (pre-committed; exit codes):
  REPL1 center == exp681 costfit per-cell EXACT            (exit 4)
  REPL2 ring10+ring25 == exp685 per-cell EXACT             (exit 5)
  H1  ringfit3 rec_def @ d=4096/cap=512 <= 2.5pp           (primary)
  H2  ringfit3 burst_def <= center burst_def ALL cells
      (Michael's ladder: keep exp685's in-burst fix)
  H3  new arms loop >= lcnu loop - 0.5pp ALL cells         (safety)
  H4  ringfit3* switches <= 6; acc credited==loop every
      composite cell; center double-sim identical          (exit 3)
  det byte-identical double grid                           (exit 2)
Verdicts: RING-FIRST-CLASS > IN_BURST_KEPT_VISIBLE > HARM.
"""

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
from benchmarks.cache_simulator.policies import build_policy  # noqa: E402
from benchmarks.cache_simulator.research.policies_costfit3 import (  # noqa: E402
    CostFit3Policy,
)

spec85 = importlib.util.spec_from_file_location(
    "r85", str(Path.home() / "warp-work/exp685-moei-ring/run_ring.py")
)
r85 = importlib.util.module_from_spec(spec85)
spec85.loader.exec_module(r85)
st, rd = r85.st, r85.rd
RingProbationLCNU = r85.RingProbationLCNU

D = Path(__file__).parent
ARMS = [
    "lcnu",
    "s3fifo",
    "center",
    "ring10",
    "ring25",
    "lazy2",
    "ringfit3",
    "ringfit3_10",
    "ringfit3_lazy2",
]
NEW_ARMS = ["lazy2", "ringfit3", "ringfit3_10", "ringfit3_lazy2"]
FIT3 = ["ringfit3", "ringfit3_10", "ringfit3_lazy2"]


class LazyRingProbationLCNU(RingProbationLCNU):
    """Promotion deferred to the promote_after-th rehit (default 2).

    exp685's ring promotes on the FIRST rehit (2 sightings -> citizen);
    exp681's crowd has k=2 repeating cold keys, so eager promotion still
    lets them displace citizens through scoring. lazy2 keeps k=2 keys
    ring-managed their whole life (drain order, never scoring-protected).
    Readmission after eviction restarts probation (rehit count resets).
    """

    def __init__(self, capacity, ring_frac=0.25, promote_after=2, **k):
        super().__init__(capacity, ring_frac=ring_frac, **k)
        self.promote_after = promote_after
        self._rehits = {}

    def _on_hit(self, key):
        if key in self._ring:
            n = self._rehits.get(key, 0) + 1
            if n >= self.promote_after:
                self._ring.pop(key, None)
                self._rehits.pop(key, None)
            else:
                self._rehits[key] = n

    def _on_miss(self, key):
        super()._on_miss(key)
        self._rehits.pop(key, None)  # fresh probation on readmission

    def _victim(self):
        for k in [k for k in self._rehits if k not in self._ring]:
            self._rehits.pop(k, None)
        return super()._victim()


class RingFit3(CostFit3Policy):
    """Sealed CostFit3 barrier with the nursery ring as 3rd candidate.

    Every parameter (HALF_LIFE 32768, MARGIN 0.005, GAMMA 0.0025,
    RESEGMENT 4096, initial lcnu) is inherited UNEDITED from the sealed
    class; only _inners differs: rothead -> ring (exp685's
    RingProbationLCNU, optionally lazy). Scoreboard, barrier, credited
    accounting: byte-identical mechanism to exp673.
    """

    def __init__(
        self, capacity, num_layers=1, prior=1.0 / 16, ring_frac=0.25, lazy=False
    ):
        self.capacity = capacity
        self.choice = "lcnu"
        self.switches = 0
        self.choice_switches = {"lcnu": 0, "s3fifo": 0, "ring": 0}
        self._since = 0
        self._decay = 0.5 ** (1.0 / self.HALF_LIFE)
        ring_cls = LazyRingProbationLCNU if lazy else RingProbationLCNU
        kw = {"num_layers": num_layers, "prior": prior, "ring_frac": ring_frac}
        if lazy:
            kw["promote_after"] = 2
        self._inners = {
            "lcnu": build_policy(
                "lcnu", capacity, num_layers=num_layers, prior=prior
            ),
            "s3fifo": build_policy("s3fifo", capacity),
            "ring": ring_cls(capacity, **kw),
        }
        self._mass = {n: [0.0, 0.0] for n in self._inners}
        self._sw_mass = 0.0
        self._credited = [0, 0]


def mk_arm(name, cap):
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "s3fifo":
        return build_policy("s3fifo", cap)
    if name == "center":
        return rd._mk_cf(cap, gk)
    if name == "ring10":
        return RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "ring25":
        return RingProbationLCNU(cap, ring_frac=0.25, **gk)
    if name == "lazy2":
        return LazyRingProbationLCNU(cap, ring_frac=0.25, promote_after=2, **gk)
    if name == "ringfit3":
        return RingFit3(cap, ring_frac=0.25, **gk)
    if name == "ringfit3_10":
        return RingFit3(cap, ring_frac=0.10, **gk)
    if name == "ringfit3_lazy2":
        return RingFit3(cap, ring_frac=0.25, lazy=True, **gk)
    raise KeyError(name)


def evaluate(keys, seg, mid, d):
    burst, rec = (seg == 3), (seg == 4)
    out = {}
    for cap in st.CAPS:
        cell = {}
        for name in ARMS:
            pol = mk_arm(name, cap)
            tl = []
            h = np.empty(len(keys), dtype=bool)
            for i, k in enumerate(keys):
                h[i] = pol.on_access(k)
                ch = getattr(pol, "choice", None)
                if ch is not None and (not tl or tl[-1][1] != ch):
                    tl.append((i, ch))
            entry = {
                "loop": float(h.mean()),
                "burst": float(h[burst].mean()),
                "rec": float(h[rec].mean()),
            }
            if hasattr(pol, "switches"):
                entry["switches"] = int(pol.switches)
                entry["tl"] = tl
                entry["choice_switches"] = {
                    str(a): int(b)
                    for a, b in getattr(pol, "choice_switches", {}).items()
                }
                eff = getattr(pol, "effective_hit_rate", None)
                entry["acc"] = bool(eff is not None and eff == entry["loop"])
            cell[name] = entry
        c2 = mk_arm("center", cap)
        h2 = np.array([c2.on_access(k) for k in keys], dtype=bool)
        cell["center"]["loop2"] = float(h2.mean())
        out[str(cap)] = cell
    return out


def choice_at(tl, i):
    cur = None
    for idx, ch in tl:
        if idx <= i:
            cur = ch
        else:
            break
    return cur


def run_all():
    grid, t0 = [], time.time()
    seeds = np.random.default_rng(68120261103).spawn(2)  # exp681 scheme, FRESH
    base_seed = int(seeds[0].integers(0, 2**63))
    crowd_seed = int(seeds[1].integers(0, 2**63))
    for d in st.BURSTS:
        keys, seg, mid = st.crowd_stream(base_seed, crowd_seed, st.N, d)
        grid.append({"d": d, "mid": mid, "caps": evaluate(keys, seg, mid, d)})
        print(f"d={d} done [{time.time() - t0:.0f}s]", flush=True)
    return grid


def replication_check(grid):
    r1 = json.loads(
        (
            Path.home() / "warp-work/exp681-moei-crowd/exp681-crowd.json"
        ).read_text()
    )
    r5 = json.loads(
        (
            Path.home() / "warp-work/exp685-moei-ring/exp685-ring.json"
        ).read_text()
    )
    ref1 = {c["d"]: c["caps"] for c in r1["grid"]}
    ref5 = {c["d"]: c["caps"] for c in r5["grid"]}
    for cell in grid:
        for cap, c in cell["caps"].items():
            a = ref1[cell["d"]][cap]["costfit"]
            for m in ("loop", "burst", "rec"):
                if c["center"][m] != a[m]:
                    return 4, (
                        f"repl1 d={cell['d']} cap={cap} {m}: "
                        f"{c['center'][m]} != {a[m]}"
                    )
            for arm in ("ring10", "ring25"):
                b = ref5[cell["d"]][cap][arm]
                for m in ("loop", "burst", "rec"):
                    if c[arm][m] != b[m]:
                        return 5, (
                            f"repl2 d={cell['d']} cap={cap} {arm} {m}: "
                            f"{c[arm][m]} != {b[m]}"
                        )
    return 0, None


def adjudicate(grid):
    rows = []
    f = {"h1": None, "h2": True, "h3": True, "h4": True, "acc": True}
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["center"]["loop2"] != c["center"]["loop"]:
                return {"bad": "acc-center", "d": cell["d"], "cap": cap}
            lc = c["lcnu"]
            row = {"d": cell["d"], "cap": int(cap)}
            for a in ARMS:
                row[a] = {
                    m: round(c[a][m], 6) for m in ("loop", "burst", "rec")
                }
            row["burst_def_cf_pp"] = round(
                (lc["burst"] - c["center"]["burst"]) * 100, 3
            )
            row["rec_def_cf_pp"] = round(
                (lc["rec"] - c["center"]["rec"]) * 100, 3
            )
            for a in NEW_ARMS:
                row[f"rec_def_{a}_pp"] = round(
                    (lc["rec"] - c[a]["rec"]) * 100, 3
                )
                row[f"burst_def_{a}_pp"] = round(
                    (lc["burst"] - c[a]["burst"]) * 100, 3
                )
                row[f"loop_vs_l_{a}_pp"] = round(
                    (c[a]["loop"] - lc["loop"]) * 100, 3
                )
            if cell["d"] == 4096 and int(cap) == 512:
                f["h1"] = bool(row["rec_def_ringfit3_pp"] <= 2.5)
                row["h1_rec_def_pp"] = row["rec_def_ringfit3_pp"]
            if row["burst_def_ringfit3_pp"] > row["burst_def_cf_pp"] + 1e-9:
                f["h2"] = False
            for a in NEW_ARMS:
                if row[f"loop_vs_l_{a}_pp"] < -0.5:
                    f["h3"] = False
            for a in FIT3:
                if c[a].get("switches", 0) > 6:
                    f["h4"] = False
                if not c[a].get("acc", False):
                    f["acc"] = False
                row[f"adopt_{a}"] = {
                    "burst": choice_at(c[a]["tl"], cell["mid"]),
                    "rec": choice_at(c[a]["tl"], cell["mid"] + cell["d"]),
                    "sw": c[a]["switches"],
                    "sw_by": c[a]["choice_switches"],
                }
            rows.append(row)
    if f["h1"] and f["h2"] and f["h3"] and f["h4"] and f["acc"]:
        v = "RING-FIRST-CLASS"
    elif f["h2"] and f["h3"] and f["h4"] and f["acc"]:
        v = "IN_BURST_KEPT_VISIBLE"
    else:
        v = "HARM"
    return {"verdict": v, "bars": f, "rows": rows}


def stride_diag(grid):
    """Pure stream statistics (NO policy behavior) — exp686b sizing input."""
    diags = []
    seeds = np.random.default_rng(68120261103).spawn(2)
    base_seed = int(seeds[0].integers(0, 2**63))
    crowd_seed = int(seeds[1].integers(0, 2**63))
    for d in st.BURSTS:
        keys, seg, mid = st.crowd_stream(base_seed, crowd_seed, st.N, d)
        idx = np.nonzero(seg == 3)[0]
        crowd = [keys[i] for i in idx]
        pos = defaultdict(list)
        for j, k in enumerate(crowd):
            pos[k].append(j)
        rehit = {k: v for k, v in pos.items() if len(v) >= 2}
        gaps = [v[1] - v[0] for v in rehit.values()]
        loads = []
        for k in list(rehit)[::16]:
            v = rehit[k]
            distinct = len({crowd[t] for t in range(v[0] + 1, v[1])})
            loads.append(distinct)
        diags.append(
            {
                "d": d,
                "distinct": len(pos),
                "rehit_keys": len(rehit),
                "gap_mean": round(float(np.mean(gaps)), 2) if gaps else None,
                "gap_p90": round(float(np.percentile(gaps, 90)), 2)
                if gaps
                else None,
                "load_mean": round(float(np.mean(loads)), 1) if loads else None,
                "load_p90": round(float(np.percentile(loads, 90)), 1)
                if loads
                else None,
            }
        )
        print(f"stride d={d}: {diags[-1]}", flush=True)
    return diags


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
    res["repl"] = "exp681 costfit + exp685 ring10/ring25 exact"
    res["grid"] = g1
    res["stride_diag"] = stride_diag(g1)
    res["prediction"] = (
        "pre-registered: H1 FAILS at cap512 (sticky s3fifo "
        "lead, ring cannot clear MARGIN+GAMMA*sw_mass); "
        "uncertain: ring10 whole-loop lead 0.6805 vs s3fifo "
        "0.6768 could adopt at some boundary"
    )
    res["rules"] = {
        "streams": "exp681 exact (68120261103)",
        "arms": ARMS,
        "bars": {
            "REPL1": "center == exp681 costfit exact (exit 4)",
            "REPL2": "ring10+ring25 == exp685 exact (exit 5)",
            "H1": "ringfit3 rec_def <= 2.5pp @ d4096/cap512 (primary)",
            "H2": "ringfit3 burst_def <= center burst_def ALL cells",
            "H3": "new arms loop >= lcnu-0.5pp ALL cells",
            "H4": "ringfit3* switches <= 6; credited==loop; center loop2",
            "det": "byte-identical double grid (exit 2)",
        },
        "verdicts": "RING-FIRST-CLASS > IN_BURST_KEPT_VISIBLE > HARM",
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp686a-ringfit3.json").write_text(json.dumps(res, indent=1))
    print(
        "VERDICT:",
        res["verdict"],
        "| det:",
        res["det"],
        "| repl:",
        res["repl"],
        "| bars:",
        res["bars"],
    )
    for r in res["rows"]:
        print(
            "cell:",
            {
                k: r[k]
                for k in (
                    "d",
                    "cap",
                    "burst_def_cf_pp",
                    "burst_def_ringfit3_pp",
                    "rec_def_cf_pp",
                    "rec_def_ringfit3_pp",
                    "rec_def_lazy2_pp",
                    "loop_vs_l_lazy2_pp",
                    "loop_vs_l_ringfit3_pp",
                )
            },
        )
        for a in FIT3:
            print("  adopt:", a, r[f"adopt_{a}"])
    sys.exit(0 if res["det"] else 2)
