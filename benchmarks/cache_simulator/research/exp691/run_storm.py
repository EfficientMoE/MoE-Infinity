#!/usr/bin/env python3
"""exp691: miss-signal arbitration — storm-revert and adoption freeze.

exp690 (REC_ONLY) proved any rate-scored menu containing ring adopts it
before the burst and pays in-burst; removing s3fifo doesn't help; the
burst edge belongs to the LCNU inner itself. exp686a/687/689/690
together rule out: barrier tuning (zb refuted), windowing, 2-way menus,
double-window confirmation. Remaining registered lever (exp690's
REC_ONLY branch): make the arbiter react to the OBSERVABLE burst
signature — a miss-rate storm on the composite's own credited stream —
with NO segment labels.

Arms:
  lcnu | center (REPL1 exp681, exit 4) | ring10 (REPL3 exp685, exit 6)
  | twoway_lr (CONTROL == exp690, exit 8)
  | freeze_50, freeze_65  — adoption-block-only: while the incumbent's
    windowed miss rate > TH, skip arbitration (challenger can never be
    adopted during a storm; no retreat).
  | storm_50, storm_65, storm_80 — storm-revert: while windowed miss
    rate > TH, force choice to lcnu (the burst-best inner); when the
    storm passes, sealed barrier rules resume (ring must clear MARGIN
    + GAMMA*sw_mass to be re-adopted).

Window = the sealed RESEGMENT (4096 accesses), the same cadence the
arbiter already uses; storm signal = 1 - windowed credited hit rate,
computed exactly at boundaries from cumulative credited counters.

Bars (per-arm conjunction on the 5 T-arms):
  RECOV  rec_def <= 3.5pp at d4096/cap512
  BURST  burst_def <= center's deficit ALL 8 cells
  STRETCH burst_def <= 1.5pp at home
  SAFETY loop >= lcnu - 0.5pp ALL cells
Verdicts: BOTH_PASS > REC_ONLY > BURST_ONLY > NO-SIGNAL > HARM.
Registered risk: d=256 bursts are shorter than one detection window —
small-d cells may keep BURST_ONLY even if large-d passes.
Exit codes: 0 settled | 2 det | 3 acc | 4 repl1 | 6 repl3 | 8 replc.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

spec85 = importlib.util.spec_from_file_location(
    "x85", str(Path.home() / "warp-work/exp685-moei-ring/run_ring.py")
)
x85 = importlib.util.module_from_spec(spec85)
sys.modules["x85"] = x85
spec85.loader.exec_module(x85)
st, rd = x85.st, x85.rd
RingProbationLCNU = x85.RingProbationLCNU
CostFitPolicy = x85.CostFitPolicy

D = Path(__file__).parent
ARMS = [
    "lcnu",
    "center",
    "ring10",
    "twoway_lr",
    "freeze_50",
    "freeze_65",
    "storm_50",
    "storm_65",
    "storm_80",
]
T_ARMS = ARMS[4:]


class TwoWayLR(CostFitPolicy):
    """Sealed 2-way composite, s3fifo SLOT -> RingProbationLCNU(0.10).

    Identical to exp690's twoway_lr (control replication gate)."""

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self._inners["s3fifo"] = RingProbationLCNU(
            capacity, ring_frac=0.10, num_layers=num_layers, prior=prior
        )


class _StormBase(TwoWayLR):
    def __init__(self, capacity, th, num_layers=1, prior=1.0 / 16):
        super().__init__(capacity, num_layers=num_layers, prior=prior)
        self.TH = th
        self._ph = self._pt = 0
        self.storm_windows = 0

    def _window_miss(self):
        h, t = self._credited
        dh, dt = h - self._ph, t - self._pt
        self._ph, self._pt = h, t
        return 1.0 - (dh / dt) if dt else 0.0


class FreezeOnly(_StormBase):
    """Adoption-block: storms freeze arbitration (incumbent keeps)."""

    def _re_arbitrate(self) -> None:
        if self._window_miss() > self.TH:
            self.storm_windows += 1
            return
        super()._re_arbitrate()


class StormRevert(_StormBase):
    """Storm-revert: storms force choice to lcnu (burst-best inner)."""

    def _re_arbitrate(self) -> None:
        if self._window_miss() > self.TH:
            self.storm_windows += 1
            if self.choice != "lcnu":
                self.choice = "lcnu"
                self.switches += 1
            return
        super()._re_arbitrate()


def mk_arm(name, cap):
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "center":
        return build_policy("costfit", cap, **gk)
    if name == "ring10":
        return RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "twoway_lr":
        return TwoWayLR(cap, **gk)
    if name.startswith("freeze_"):
        return FreezeOnly(cap, th=float(name[7:]) / 100, **gk)
    if name.startswith("storm_"):
        return StormRevert(cap, th=float(name[6:]) / 100, **gk)
    raise KeyError(name)


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    out = {}
    for cap in st.CAPS:
        cell = {}
        for name in ARMS:
            pol = mk_arm(name, cap)
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            e = {
                "loop": float(h.mean()),
                "burst": float(h[burst].mean()),
                "rec": float(h[rec].mean()),
            }
            if hasattr(pol, "switches"):
                e["sw"] = int(pol.switches)
            if hasattr(pol, "storm_windows"):
                e["storm_w"] = int(pol.storm_windows)
            cell[name] = e
        c2 = build_policy("costfit", cap, num_layers=rd.L, prior=8 / rd.E)
        h2 = np.array([c2.on_access(k) for k in keys], dtype=bool)
        cell["center"]["loop2"] = float(h2.mean())
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


def replication_check(grid):
    refs = []
    for path, arm in (
        ("warp-work/exp681-moei-crowd/exp681-crowd.json", None),
        ("warp-work/exp685-moei-ring/exp685-ring.json", None),
        ("warp-work/exp690-moei-confirm/exp690-confirm.json", None),
    ):
        refs.append(json.loads((Path.home() / path).read_text()))
    ref1 = {c["d"]: c["caps"] for c in refs[0]["grid"]}
    ref5 = {c["d"]: c["caps"] for c in refs[1]["grid"]}
    ref0 = {c["d"]: c["caps"] for c in refs[2]["grid"]}
    for cell in grid:
        for cap, c in cell["caps"].items():
            for ref, arm, refname, code in (
                (ref1, "center", "costfit", 4),
                (ref5, "ring10", "ring10", 6),
                (ref0, "twoway_lr", "twoway_lr", 8),
            ):
                a = ref[cell["d"]][cap][refname]
                for m in ("loop", "burst", "rec"):
                    if c[arm][m] != a[m]:
                        return code, (
                            f"repl d={cell['d']} cap={cap} {arm} "
                            f"{m}: {c[arm][m]} != {a[m]}"
                        )
    return 0, None


def adjudicate(grid):
    rows = []
    arm = {
        a: {"recov": False, "burst_def": True, "stretch": False, "safety": True}
        for a in T_ARMS
    }
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["center"]["loop"] != c["center"]["loop2"]:
                return {"bad": "acc", "d": cell["d"], "cap": cap}
            bb = c["lcnu"]["burst"]
            row = {"d": cell["d"], "cap": int(cap)}
            for a in ARMS:
                row[a] = {
                    m: round(c[a][m], 6) for m in ("loop", "burst", "rec")
                }
                if "sw" in c[a]:
                    row[a]["sw"] = c[a]["sw"]
                if "storm_w" in c[a]:
                    row[a]["storm_w"] = c[a]["storm_w"]
            row["cf_burst_def_pp"] = round((bb - c["center"]["burst"]) * 100, 3)
            row["cf_rec_def_pp"] = round(
                (c["lcnu"]["rec"] - c["center"]["rec"]) * 100, 3
            )
            for a in T_ARMS:
                f = arm[a]
                row[f"{a}_burst_def_pp"] = round((bb - c[a]["burst"]) * 100, 3)
                row[f"{a}_rec_def_pp"] = round(
                    (c["lcnu"]["rec"] - c[a]["rec"]) * 100, 3
                )
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
    return {
        "verdict": v,
        "winners": winners,
        "rec_only": rec_only,
        "burst_only": burst_only,
        "arm_bars": arm,
        "rows": rows,
    }


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
    res["repl"] = "exp681+exp685+exp690 exact"
    res["grid"] = g1
    res["rules"] = {
        "streams": "exp681 exact (68120261103)",
        "arms": ARMS,
        "bars": {
            "REPL1": "center == exp681 exact (exit 4)",
            "REPL3": "ring10 == exp685 exact (exit 6)",
            "REPLC": "twoway_lr == exp690 exact (exit 8)",
            "RECOV": "arm rec_def <= 3.5pp @d4096/cap512",
            "BURST": "arm burst_def <= center's ALL cells",
            "SAFETY": "loop >= lcnu-0.5pp ALL",
        },
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp691-storm.json").write_text(json.dumps(res, indent=1))
    print(
        "VERDICT:",
        res["verdict"],
        "| winners:",
        res["winners"],
        "| det:",
        res["det"],
        "| repl:",
        res["repl"],
    )
    for r in res["rows"]:
        print(
            "cell:",
            {
                "d": r["d"],
                "cap": r["cap"],
                **{f"{a}_rec": r[f"{a}_rec_def_pp"] for a in T_ARMS},
                "sw": {a: r[a].get("sw") for a in T_ARMS},
                "storm": {a: r[a].get("storm_w") for a in T_ARMS},
            },
        )
    sys.exit(0 if res["det"] else 2)
