#!/usr/bin/env python3
"""exp687: windowed challenger scoreboard (arbitration-evidence axis).

exp686a verdict IN_BURST_KEPT_VISIBLE: rf3_10 (ring as 3rd candidate,
decayed cumulative scoreboard) ADOPTS ring at all cap512 cells — home
cell rec_def -0.73pp (better than lcnu) BUT burst_def +5.96pp vs center
+1.07pp: "no arm holds both bars; decayed-cumulative scoreboard cannot
hand off between burst and recovery windows."

exp687 (registered branch of exp686a): keep the sealed CostFit3 barrier
family (HALF_LIFE 32768, MARGIN 0.005, GAMMA 0.0025, RESEGMENT 4096,
initial lcnu — ALL sealed params inherited unedited from
policies_costfit3.CostFit3Policy) and change ONLY the challenger's
evidence basis: the incumbent is judged on its decayed cumulative rate
(sealed behavior), each CHALLENGER on its last-4096 window rate. Victim
selection and inners identical to exp686a RingFit3 ({lcnu, s3fifo,
ring}); sealed victim-selection policies untouched.

STRUCTURAL PRE-ANALYSIS (pre-registered, frozen before the grid):
mid = 319488 = 78*4096 is window-ALIGNED. For d=4096 the burst fills
exactly one resegment window and the recovery the next; the first
pure-recovery window COMPLETES only at the recovery's final boundary
(adoption there benefits only post-recovery accesses), and mid-burst
escape is impossible (challenger burst-window rates 0.029-0.088 vs
incumbent decayed ~0.6+barrier). So at the home cell NO windowed
arbitration with these alignments can pass H1 AND H2 simultaneously;
the measured run adjudicates WHICH side fails (pre-burst ring
re-adoption via window noise -> H2 fails, rf3_10 tradeoff; ring never
re-adopted -> H1 fails, center-like) and measures asymmetric-evidence
stability (switch counts) across all 8 cells.

Arms on exp681's EXACT streams (seeds 68120261103 spawn scheme), 8 cells:
  lcnu | s3fifo | center (REPL1 vs exp681 costfit)
  ring10 | ring25 (REPL2 vs exp685)
  ringfit3_10 (REPL3 vs exp686a: proves sealed base untouched)
  windowfit3_10 (PRIMARY: windowed challenger, ring .10)
  windowfit3_25 (dose: ring .25)

Bars (pre-committed; exit codes):
  REPL1 center == exp681 costfit per-cell EXACT            (exit 4)
  REPL2 ring10+ring25 == exp685 per-cell EXACT             (exit 5)
  REPL3 ringfit3_10 == exp686a per-cell EXACT
        incl. switches + choice timeline                   (exit 6)
  UNIT window-vs-decayed decision bites on hand-set state  (exit 7)
  H1  windowfit3_10 rec_def @ d=4096/cap=512 <= 2.5pp      (primary)
  H2  windowfit3_10 burst_def <= center burst_def ALL cells
      (Michael's ladder: keep exp685/686a in-burst protection)
  H3  windowfit3* loop >= lcnu loop - 0.5pp ALL cells      (safety)
  H4  windowfit3* switches <= 6 ALL cells; acc credited==loop
      every composite; center double-sim identical         (exit 3)
  H0  structural sub-adjudication (measured, not gated):
      at home cell record incumbent-at-mid, switch indices in
      [mid-4096, mid+8192], first ring adoption >= mid.
  det byte-identical double grid                           (exit 2)
Verdicts: WINDOWED-HANDOFF > RECOVERY-ONLY > STRUCTURAL-SPLIT >
NO-GAIN > HARM/THRASH.
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
    "ringfit3_10",
    "windowfit3_10",
    "windowfit3_25",
]
NEW_ARMS = ["windowfit3_10", "windowfit3_25"]


class RingFit3(CostFit3Policy):
    """Sealed CostFit3 barrier with the nursery ring as 3rd candidate.

    Identical to exp686a's RingFit3 (ring .10 configuration): every
    parameter inherited UNEDITED from the sealed class; only _inners
    differs (rothead -> ring). Redeclared here so REPL3 anchors THIS
    file's base against exp686a published numbers.
    """

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16, ring_frac=0.10):
        self.capacity = capacity
        self.choice = "lcnu"
        self.switches = 0
        self.choice_switches = {"lcnu": 0, "s3fifo": 0, "ring": 0}
        self._since = 0
        self._decay = 0.5 ** (1.0 / self.HALF_LIFE)
        self._inners = {
            "lcnu": build_policy(
                "lcnu", capacity, num_layers=num_layers, prior=prior
            ),
            "s3fifo": build_policy("s3fifo", capacity),
            "ring": RingProbationLCNU(
                capacity,
                num_layers=num_layers,
                prior=prior,
                ring_frac=ring_frac,
            ),
        }
        self._mass = {n: [0.0, 0.0] for n in self._inners}
        self._sw_mass = 0.0
        self._credited = [0, 0]


class WindowFit3(RingFit3):
    """Windowed challenger scoreboard on the sealed barrier family.

    On every access: sealed decayed-mass update for ALL inners (the
    incumbent's evidence basis is EXACTLY sealed) PLUS raw window
    counters [hits, accesses] per inner over the current 4096-window.
    At each sealed RESEGMENT boundary:
      incumbent rate  = sealed decayed cumulative rate (unchanged code)
      challenger      = highest LAST-WINDOW rate among non-incumbents
                        (deterministic sorted-name order, strict >)
      barrier         = sealed MARGIN + GAMMA * decayed switch mass
    Window counters reset after arbitration. Nothing else changes.
    """

    def __init__(self, capacity, **k):
        super().__init__(capacity, **k)
        self._win = {n: [0, 0] for n in self._inners}

    def on_access(self, key):
        hit = False
        d = self._decay
        for name, pol in self._inners.items():
            h = pol.on_access(key)
            m = self._mass[name]
            m[0] = m[0] * d + (1.0 if h else 0.0)
            m[1] = m[1] * d + 1.0
            w = self._win[name]
            w[0] += 1 if h else 0
            w[1] += 1
            if name == self.choice:
                hit = h
                self._credited[0] += 1 if h else 0
                self._credited[1] += 1
        self._since += 1
        if self._since >= self.RESEGMENT:
            self._since = 0
            self._re_arbitrate()
        return hit

    def _win_rate(self, name):
        hm, am = self._win[name]
        return hm / am if am > 0 else None

    def _re_arbitrate(self) -> None:
        self._decide()
        for n in self._win:
            self._win[n] = [0, 0]

    def _decide(self) -> None:
        """Pure decision step (unit-testable on hand-set state)."""
        inc = self.choice
        ri = self._rate(inc)  # sealed decayed cumulative, incumbent only
        chg, rc = None, None
        for name in sorted(self._inners):
            if name == inc:
                continue
            r = self._win_rate(name)
            if r is not None and (rc is None or r > rc):
                chg, rc = name, r
        barrier = self.MARGIN + self.GAMMA * self._sw_mass
        if (
            ri is not None
            and chg is not None
            and rc is not None
            and rc - ri >= barrier
        ):
            self.choice = chg
            self.switches += 1
            self.choice_switches[chg] += 1
            self._sw_mass = self._sw_mass * self._decay + 1.0


def unit_checks():
    """Instrument bite: the window path (not decayed) drives challenges.

    Hand-set state, no stream: with incumbent decayed 0.5 and challenger
    window 0.9 -> switch; window barely above (0.504, gap < MARGIN) ->
    no switch; challenger DECAYED 0.9 but window 0.5 -> NO switch (the
    single-edit discriminator: windowed decision ignores decayed lead).
    """

    def fresh():
        p = WindowFit3(512, num_layers=rd.L, prior=8 / rd.E, ring_frac=0.10)
        p._mass = {
            "lcnu": [500.0, 1000.0],
            "s3fifo": [0.0, 0.0],
            "ring": [0.0, 0.0],
        }
        p._win = {"lcnu": [0, 0], "s3fifo": [0, 0], "ring": [0, 0]}
        return p

    p = fresh()
    p._win["ring"] = [921, 1024]  # 0.899 vs decayed incumbent 0.5
    p._decide()
    if p.choice != "ring" or p.switches != 1:
        return "bite1: expected switch to ring on window 0.899, got " + p.choice

    p = fresh()
    p._win["ring"] = [516, 1024]  # 0.504: gap 0.004 < MARGIN 0.005
    p._decide()
    if p.choice != "lcnu" or p.switches != 0:
        return "bite2: expected NO switch at gap 0.004 < barrier"

    p = fresh()
    p._mass["ring"] = [900.0, 1000.0]  # decayed 0.9 but window 0.5
    p._win["ring"] = [512, 1024]
    p._decide()
    if p.choice != "lcnu" or p.switches != 0:
        return "bite3: windowed decision must IGNORE decayed challenger lead"
    return None


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
    if name == "ringfit3_10":
        return RingFit3(cap, ring_frac=0.10, **gk)
    if name == "windowfit3_10":
        return WindowFit3(cap, ring_frac=0.10, **gk)
    if name == "windowfit3_25":
        return WindowFit3(cap, ring_frac=0.25, **gk)
    raise KeyError(name)


def choice_at(tl, i):
    cur = None
    for idx, ch in tl:
        if idx <= i:
            cur = ch
        else:
            break
    return cur


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


def run_all(bursts=None):
    grid, t0 = [], time.time()
    seeds = np.random.default_rng(68120261103).spawn(2)  # exp681 scheme, FRESH
    base_seed = int(seeds[0].integers(0, 2**63))
    crowd_seed = int(seeds[1].integers(0, 2**63))
    for d in bursts if bursts else st.BURSTS:
        keys, seg, mid = st.crowd_stream(base_seed, crowd_seed, st.N, d)
        grid.append({"d": d, "mid": mid, "caps": evaluate(keys, seg, mid, d)})
        print(f"d={d} done [{time.time() - t0:.0f}s]", flush=True)
    return grid


def repl_check_cell(cell, ref1, ref5, ref6a):
    """Anchor checks on ONE grid cell (used by smoke and full)."""
    for cap, c in cell["caps"].items():
        a = ref1[cell["d"]][cap]["costfit"]
        for m in ("loop", "burst", "rec"):
            if c["center"][m] != a[m]:
                return (
                    4,
                    f"repl1 d={cell['d']} cap={cap} {m}: {c['center'][m]} != {a[m]}",
                )
        for arm in ("ring10", "ring25"):
            b = ref5[cell["d"]][cap][arm]
            for m in ("loop", "burst", "rec"):
                if c[arm][m] != b[m]:
                    return (
                        5,
                        f"repl2 d={cell['d']} cap={cap} {arm} {m}: {c[arm][m]} != {b[m]}",
                    )
        g = ref6a[cell["d"]][cap]["ringfit3_10"]
        e = c["ringfit3_10"]
        for m in ("loop", "burst", "rec"):
            if e[m] != g[m]:
                return 6, f"repl3 d={cell['d']} cap={cap} {m}: {e[m]} != {g[m]}"
        tl_e = [[int(i), str(ch)] for i, ch in e["tl"]]
        if e["switches"] != g["switches"] or tl_e != g["tl"]:
            return 6, f"repl3 d={cell['d']} cap={cap} tl/sw mismatch"
    return 0, None


def load_refs():
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
    r6a = json.loads(
        (
            Path.home()
            / "warp-work/exp686a-moei-ringfit3/exp686a-ringfit3.json"
        ).read_text()
    )
    return (
        {c["d"]: c["caps"] for c in r1["grid"]},
        {c["d"]: c["caps"] for c in r5["grid"]},
        {c["d"]: c["caps"] for c in r6a["grid"]},
    )


def adjudicate(grid):
    rows = []
    f = {"h1": None, "h2": True, "h3": True, "h4": True, "acc": True}
    h0 = {}
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
                if row[f"loop_vs_l_{a}_pp"] < -0.5:
                    f["h3"] = False
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
            if cell["d"] == 4096 and int(cap) == 512:
                f["h1"] = bool(row["rec_def_windowfit3_10_pp"] <= 2.5)
                row["h1_rec_def_pp"] = row["rec_def_windowfit3_10_pp"]
                # H0 structural sub-adjudication (measured, not gated)
                tl = c["windowfit3_10"]["tl"]
                inc_mid = choice_at(tl, cell["mid"] - 1)
                sw_idx = [i for i, _ in tl]
                window = [
                    i
                    for i in sw_idx
                    if cell["mid"] - 4096 <= i <= cell["mid"] + 8192
                ]
                first_ring_after_mid = next(
                    (i for i, ch in tl if i >= cell["mid"] and ch == "ring"),
                    None,
                )
                h0 = {
                    "incumbent_at_mid": inc_mid,
                    "switch_indices_near_mid": window,
                    "first_ring_adoption_gte_mid": first_ring_after_mid,
                    "recovery_end": cell["mid"] + cell["d"] + st.WIN,
                    "tl_full": [[int(i), ch] for i, ch in tl],
                }
            if (
                row["burst_def_windowfit3_10_pp"]
                > row["burst_def_cf_pp"] + 1e-9
            ):
                f["h2"] = False
            rows.append(row)
    if f["h1"] and f["h2"] and f["h3"] and f["h4"] and f["acc"]:
        v = "WINDOWED-HANDOFF"
    elif f["h1"] and f["h3"] and f["h4"] and f["acc"]:
        v = "RECOVERY-ONLY"
    elif f["h3"] and f["h4"] and f["acc"] and not f["h1"] and f["h2"]:
        # H2 holds everywhere incl. home cell while H1 fails there:
        # split by adoption geometry, or plain no-gain — decided by h0
        v = "STRUCTURAL-SPLIT-OR-NO-GAIN"
    else:
        v = "HARM/THRASH"
    return {"verdict": v, "bars": f, "h0": h0, "rows": rows}


def refine_verdict(res):
    """Apply the pre-registered H0 discriminator (no new measurement)."""
    if res["verdict"] != "STRUCTURAL-SPLIT-OR-NO-GAIN":
        return res
    h0 = res["h0"]
    mid, rec_end = h0["mid"], h0["recovery_end"]
    fra = h0["first_ring_adoption_gte_mid"]
    inc = h0["incumbent_at_mid"]
    # mechanism confirmed when the incumbent entering the burst is
    # lcnu/s3fifo AND ring never serves inside the recovery window:
    # the first pure-recovery window completes only AT recovery end
    # (window-aligned d=4096 burst), so adoption before rec_end is
    # impossible and H1 fails by boundary arithmetic, not by tuning.
    adopted_in_recovery = any(
        ch == "ring" and mid <= i < rec_end for i, ch in h0["tl_full"]
    )
    if inc in ("lcnu", "s3fifo") and not adopted_in_recovery:
        res["verdict"] = "STRUCTURAL-SPLIT"
        res["h0"]["mechanism"] = (
            "ring first adoptable only at/after recovery end "
            f"(first ring >= mid: {fra}, recovery ends {rec_end}); "
            "window-aligned burst makes mid-recovery adoption impossible"
        )
        return res
    res["verdict"] = "NO-GAIN"
    res["h0"]["mechanism"] = (
        "windowed scoreboard changed adoption timing but not the "
        "H1 outcome through some path other than boundary alignment"
    )
    return res


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    err = unit_checks()
    if err:
        print("UNIT FAIL:", err)
        sys.exit(7)
    print(
        "unit: 3/3 bite (window drives challenge; decayed lead ignored)",
        flush=True,
    )
    ref1, ref5, ref6a = load_refs()
    if smoke:
        seeds = np.random.default_rng(68120261103).spawn(2)
        base_seed = int(seeds[0].integers(0, 2**63))
        crowd_seed = int(seeds[1].integers(0, 2**63))
        keys, seg, mid = st.crowd_stream(base_seed, crowd_seed, st.N, 256)
        cell = {"d": 256, "mid": mid, "caps": evaluate(keys, seg, mid, 256)}
        rc, msg = repl_check_cell(cell, ref1, ref5, ref6a)
        if rc:
            print(msg)
            sys.exit(rc)
        for cap, c in cell["caps"].items():
            print(
                f"smoke d256/cap{cap}: anchors OK; "
                f"windowfit3_10 loop={c['windowfit3_10']['loop']:.6f} "
                f"burst={c['windowfit3_10']['burst']:.6f} "
                f"rec={c['windowfit3_10']['rec']:.6f} "
                f"sw={c['windowfit3_10']['switches']} "
                f"tl={c['windowfit3_10']['tl']}",
                flush=True,
            )
        print("SMOKE OK")
        sys.exit(0)

    g1 = run_all()
    res = adjudicate(g1)
    if "bad" in res:
        print("ACC FAIL:", res)
        sys.exit(3)
    for cell in g1:
        rc, msg = repl_check_cell(cell, ref1, ref5, ref6a)
        if rc:
            print(msg)
            sys.exit(rc)
    res = refine_verdict(res)
    g2 = run_all()
    res["det"] = json.dumps(g1) == json.dumps(g2)
    res["repl"] = (
        "exp681 costfit + exp685 ring10/ring25 + exp686a ringfit3_10 exact"
    )
    res["grid"] = g1
    res["prediction"] = (
        "pre-registered: STRUCTURAL-SPLIT or NO-GAIN at home cell (mid "
        "319488 = 78*4096 window-ALIGNED; d=4096 burst fills exactly one "
        "window, first pure-recovery window completes at recovery END, "
        "mid-burst escape impossible at challenger burst-window rates "
        "0.029-0.088 vs incumbent decayed ~0.6+barrier); genuine "
        "uncertainty: pre-burst ring re-adoption via window noise (-> H2 "
        "fails instead, rf3_10 tradeoff) and asymmetric-evidence switch "
        "stability (H4 risk) across the 8 cells"
    )
    res["rules"] = {
        "streams": "exp681 exact (68120261103)",
        "arms": ARMS,
        "bars": {
            "REPL1": "center == exp681 costfit exact (exit 4)",
            "REPL2": "ring10+ring25 == exp685 exact (exit 5)",
            "REPL3": "ringfit3_10 == exp686a exact incl tl/sw (exit 6)",
            "UNIT": "window-vs-decayed decision bites x3 (exit 7)",
            "H1": "windowfit3_10 rec_def <= 2.5pp @ d4096/cap512 (primary)",
            "H2": "windowfit3_10 burst_def <= center burst_def ALL cells",
            "H3": "windowfit3* loop >= lcnu-0.5pp ALL cells",
            "H4": "windowfit3* switches <= 6; credited==loop; center loop2",
            "H0": "structural sub-adjudication at home cell (measured)",
            "det": "byte-identical double grid (exit 2)",
        },
        "verdicts": "WINDOWED-HANDOFF > RECOVERY-ONLY > "
        "STRUCTURAL-SPLIT > NO-GAIN > HARM/THRASH",
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp687-windowed.json").write_text(json.dumps(res, indent=1))
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
                    "burst_def_windowfit3_10_pp",
                    "rec_def_cf_pp",
                    "rec_def_windowfit3_10_pp",
                    "loop_vs_l_windowfit3_10_pp",
                )
            },
        )
        for a in NEW_ARMS:
            print("  adopt:", a, r[f"adopt_{a}"])
    print("h0:", res.get("h0"))
    sys.exit(0 if res["det"] else 2)
