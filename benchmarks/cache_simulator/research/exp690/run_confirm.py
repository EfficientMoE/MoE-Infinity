#!/usr/bin/env python3
"""exp690: handoff timing — 2-way lcnu+ring and double-window confirmation.

Three convergent experiments (exp686a ringfit3, exp687 windowed, exp689
center_ring10) map the same frontier at cap512: the burst window wants
the LCNU inner (burst 0.0884 @ home), the recovery window wants the
RING (rec 0.6853), and every cumulative-rate scoreboard settles on
s3fifo in between — never best in either critical window. exp689's
center_zb refuted the barrier hypothesis; exp687 showed single-window
judges thrash (8-14 switches).

Registered levers (this run):
  T1  twoway_lr : 2-way composite, LCNU vs ring10 — no s3fifo to hide
                  on. If the scoreboard sits on lcnu through burst and
                  hands to ring in recovery, BOTH bars can pass without
                  any new judge.
  T2  twoway_cf : T1 + double-window confirmation (challenger must
                  clear the sealed barrier in TWO consecutive
                  re-arbitration windows; any non-clearing window
                  resets the pending challenge).
  T3  fit3_cf   : exp686a's RingFit3(0.10) + the same confirmation
                  (blocks exp687-style single-window thrash).

Arms: lcnu | center (REPL1 exp681, exit 4) | ring10 (REPL3 exp685,
exit 6) | fit3_10 (REPL4 exp686a, exit 7) | twoway_lr | twoway_cf |
fit3_cf.

Bars (per-arm conjunction on the three T-arms):
  RECOV  rec_def <= 3.5pp at d4096/cap512
  BURST  burst_def <= center's deficit ALL 8 cells
  SAFETY loop >= lcnu - 0.5pp ALL cells
  STRETCH burst_def <= 1.5pp at home
Verdicts: BOTH_PASS > REC_ONLY > BURST_ONLY > NO-SIGNAL > HARM.
Switch counts recorded (thrash check; exp687 windowed saw 8-14).
Exit codes: 0 settled | 2 det | 3 acc | 4 repl1 | 6 repl3 | 7 repl4.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
from benchmarks.cache_simulator.policies import build_policy  # noqa: E402
from benchmarks.cache_simulator.research.policies_costfit3 import (  # noqa: E402
    CostFit3Policy)

spec85 = importlib.util.spec_from_file_location(
    "x85", str(Path.home() / "warp-work/exp685-moei-ring/run_ring.py"))
x85 = importlib.util.module_from_spec(spec85)
sys.modules["x85"] = x85
spec85.loader.exec_module(x85)
st, rd = x85.st, x85.rd
RingProbationLCNU = x85.RingProbationLCNU
CostFitPolicy = x85.CostFitPolicy

spec86 = importlib.util.spec_from_file_location(
    "r86", str(Path.home() / "warp-work/exp686a-moei-ringfit3/run_ringfit3.py"))
r86 = importlib.util.module_from_spec(spec86)
sys.modules["r86"] = r86
spec86.loader.exec_module(r86)
RingFit3 = r86.RingFit3

D = Path(__file__).parent
ARMS = ["lcnu", "center", "ring10", "fit3_10",
        "twoway_lr", "twoway_cf", "fit3_cf"]
T_ARMS = ["twoway_lr", "twoway_cf", "fit3_cf"]


class TwoWayLR(CostFitPolicy):
    """Sealed 2-way composite with the s3fifo SLOT replaced by ring10.

    Sealed scoreboard/barrier machinery unedited; the 's3fifo' key now
    holds a RingProbationLCNU(ring_frac=0.10)."""

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self._inners["s3fifo"] = RingProbationLCNU(
            capacity, ring_frac=0.10, num_layers=num_layers, prior=prior)


class TwoWayConfirm(TwoWayLR):
    """T1 + double-window confirmation on the sealed 2-way barrier."""

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self._pending = None
        self._pending_n = 0

    def _re_arbitrate(self) -> None:
        inc = self.choice
        chg = "s3fifo" if inc == "lcnu" else "lcnu"
        ri = self._rate(inc)
        rc = self._rate(chg)
        barrier = self.MARGIN + self.GAMMA * self._sw_mass
        if ri is not None and rc is not None and rc - ri >= barrier:
            if self._pending == chg:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = chg, 1
            if self._pending_n >= 2:
                self.choice = chg
                self.switches += 1
                self._sw_mass = self._sw_mass * self._decay + 1.0
                self._pending, self._pending_n = None, 0
        else:
            self._pending, self._pending_n = None, 0


class Fit3Confirm(RingFit3):
    """exp686a's RingFit3(0.10) + double-window confirmation.

    Faithful copy of the sealed CostFit3Policy._re_arbitrate challenger
    scan, with the switch gated on the same challenger clearing the
    barrier in two consecutive windows."""

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16):
        super().__init__(capacity, num_layers=num_layers, prior=prior,
                         ring_frac=0.10)
        self._pending = None
        self._pending_n = 0

    def _re_arbitrate(self) -> None:
        inc = self.choice
        ri = self._rate(inc)
        chg, rc = None, None
        for name in sorted(self._inners):
            if name == inc:
                continue
            r = self._rate(name)
            if r is not None and (rc is None or r > rc):
                chg, rc = name, r
        barrier = self.MARGIN + self.GAMMA * self._sw_mass
        if ri is not None and chg is not None and rc is not None \
                and rc - ri >= barrier:
            if self._pending == chg:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = chg, 1
            if self._pending_n >= 2:
                self.choice = chg
                self.switches += 1
                self.choice_switches[chg] += 1
                self._sw_mass = self._sw_mass * self._decay + 1.0
                self._pending, self._pending_n = None, 0
        else:
            self._pending, self._pending_n = None, 0


def mk_arm(name, cap):
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "center":
        return build_policy("costfit", cap, **gk)
    if name == "ring10":
        return RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "fit3_10":
        return RingFit3(cap, ring_frac=0.10, **gk)
    if name == "twoway_lr":
        return TwoWayLR(cap, **gk)
    if name == "twoway_cf":
        return TwoWayConfirm(cap, **gk)
    if name == "fit3_cf":
        return Fit3Confirm(cap, **gk)
    raise KeyError(name)


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    out = {}
    for cap in st.CAPS:
        cell = {}
        for name in ARMS:
            pol = mk_arm(name, cap)
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            e = {"loop": float(h.mean()), "burst": float(h[burst].mean()),
                 "rec": float(h[rec].mean())}
            if hasattr(pol, "switches"):
                e["sw"] = int(pol.switches)
                if hasattr(pol, "choice_switches"):
                    e["sw_by"] = {k: int(v) for k, v in
                                  pol.choice_switches.items()}
            cell[name] = e
        c2 = build_policy("costfit", cap, num_layers=rd.L, prior=8 / rd.E)
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
    r5 = json.loads((Path.home() / "warp-work/exp685-moei-ring/"
                     "exp685-ring.json").read_text())
    r6 = json.loads((Path.home() / "warp-work/moei-granite3b/benchmarks/"
                     "cache_simulator/research/exp686a/"
                     "exp686a-ringfit3.json").read_text())
    ref1 = {c["d"]: c["caps"] for c in r1["grid"]}
    ref5 = {c["d"]: c["caps"] for c in r5["grid"]}
    ref6 = {c["d"]: c["caps"] for c in r6["grid"]}
    for cell in grid:
        for cap, c in cell["caps"].items():
            for ref, arm, code in ((ref1, "center", 4), (ref5, "ring10", 6),
                                   (ref6, "fit3_10", 7)):
                a = ref[cell["d"]][cap][arm]
                for m in ("loop", "burst", "rec"):
                    if c[arm][m] != a[m]:
                        return code, (f"repl d={cell['d']} cap={cap} {arm} "
                                      f"{m}: {c[arm][m]} != {a[m]}")
    return 0, None


def adjudicate(grid):
    rows = []
    arm = {a: {"recov": False, "burst_def": True, "stretch": False,
               "safety": True} for a in T_ARMS}
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["center"]["loop"] != c["center"]["loop2"]:
                return {"bad": "acc", "d": cell["d"], "cap": cap}
            bb = c["lcnu"]["burst"]
            row = {"d": cell["d"], "cap": int(cap)}
            for a in ARMS:
                row[a] = {m: round(c[a][m], 6) for m in ("loop", "burst", "rec")}
                if "sw" in c[a]:
                    row[a]["sw"] = c[a]["sw"]
            row["cf_burst_def_pp"] = round((bb - c["center"]["burst"]) * 100, 3)
            row["cf_rec_def_pp"] = round((c["lcnu"]["rec"]
                                          - c["center"]["rec"]) * 100, 3)
            for a in T_ARMS:
                f = arm[a]
                row[f"{a}_burst_def_pp"] = round((bb - c[a]["burst"]) * 100, 3)
                row[f"{a}_rec_def_pp"] = round(
                    (c["lcnu"]["rec"] - c[a]["rec"]) * 100, 3)
                if (c[a]["loop"] - c["lcnu"]["loop"]) * 100 < -0.5:
                    f["safety"] = False
                if row[f"{a}_burst_def_pp"] > row["cf_burst_def_pp"] + 1e-9:
                    f["burst_def"] = False
                if cell["d"] == 4096 and int(cap) == 512:
                    if row[f"{a}_rec_def_pp"] <= 3.5:
                        f["recov"] = True
                    if row[f"{a}_burst_def_pp"] <= 1.5:
                        f["stretch"] = True
            rows.append(row)
    def ok(a, *ks):
        return all(arm[a][k] for k in ks)
    winners = [a for a in T_ARMS if ok(a, "recov", "burst_def", "safety")]
    rec_only = [a for a in T_ARMS if ok(a, "recov", "safety")]
    burst_only = [a for a in T_ARMS if ok(a, "burst_def", "safety")]
    if winners:
        v = "BOTH_PASS"
    elif rec_only:
        v = "REC_ONLY"
    elif burst_only:
        v = "BURST_ONLY"
    elif all(arm[a]["safety"] for a in T_ARMS):
        v = "NO-SIGNAL"
    else:
        v = "HARM"
    return {"verdict": v, "winners": winners, "rec_only": rec_only,
            "burst_only": burst_only, "arm_bars": arm, "rows": rows}


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
    res["repl"] = "exp681+exp685+exp686a exact"
    res["grid"] = g1
    res["rules"] = {"streams": "exp681 exact (68120261103)",
                    "arms": ARMS,
                    "bars": {"REPL1": "center == exp681 exact (exit 4)",
                             "REPL3": "ring10 == exp685 exact (exit 6)",
                             "REPL4": "fit3_10 == exp686a exact (exit 7)",
                             "RECOV": "arm rec_def <= 3.5pp @d4096/cap512",
                             "BURST": "arm burst_def <= center's ALL cells",
                             "SAFETY": "loop >= lcnu-0.5pp ALL"}}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp690-confirm.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| winners:", res["winners"],
          "| det:", res["det"], "| repl:", res["repl"])
    for r in res["rows"]:
        print("cell:", {"d": r["d"], "cap": r["cap"],
                        **{f"{a}_rec": r[f"{a}_rec_def_pp"] for a in T_ARMS},
                        "sw": {a: r[a].get("sw") for a in T_ARMS}})
    sys.exit(0 if res["det"] else 2)
