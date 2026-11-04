#!/usr/bin/env python3
"""exp686a: make the ring visible — ring10 inner + lazy promotion + ghost history.

exp685 measured: the composite's ONLY failed gate was recovery at
d4096/cap512 (+6.982pp == exp681 center), and center_ring25 was
BYTE-EQUAL to center there — the ring inner lost arbitration (decayed
loop 0.6788 < s3fifo 0.6800 at d1024/cap512), so its resident set was
never used. But ring10's standalone loop was 0.6837 > 0.6800: a ring10
inner should WIN arbitration at cap512 cells, becoming visible and
carrying its recovery protection (+0.73pp vs lcnu at the home cell)
into the composite without paying the door's burst tax.

Arms (exp681 exact streams, 8 cells):
  lcnu           best standalone (ref)
  center         frozen composite (REPL1 vs exp681)
  ring10         RingProbationLCNU 0.10 (REPL3 control vs exp685 exact)
  center_ring10  composite, ring10 as the lcnu inner   <- visibility test
  center_lazy    ring10 inner + lazy promotion (promote on 2nd rehit;
                 1-rehit keys stay ring-evictable)     <- frozen branch
  center_ghost   ring10 inner + ghost history (one-timers evicted from
                 the ring are remembered G=4096 accesses and re-admit
                 as citizens, never re-churning the ring)

Bars (pre-committed):
  REPL1  center == exp681 costfit per-cell exact            (exit 4)
  REPL3  ring10  == exp685 published ring10 per-cell exact  (exit 6)
  VIS    center_ring10 NOT byte-equal to center at cap512 cells AND
         rec_def <= 3.5pp at d=4096/cap=512 (retain >= half of the
         6.98pp recovery gap; the gate exp685 failed)
  BURST  center_ring10 burst_def <= center's deficit ALL cells
         (keep exp685's erased-catastrophe win)
  STRETCH center_ring10 burst_def <= 1.5pp at d=4096/cap=512
  SAFETY all arms loop >= lcnu - 0.5pp ALL cells; acc (exit 3); det (exit 2)
Verdicts: BOTH_PASS > VIS_ONLY > IN_BURST_ONLY > NO-VIS > HARM.
Exit codes: 0 settled | 2 det | 3 acc | 4 repl1 | 6 repl3.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
spec = importlib.util.spec_from_file_location(
    "x85", str(Path(__file__).parent.parent / "exp685-moei-ring/run_ring.py"))
x85 = importlib.util.module_from_spec(spec)
sys.modules["x85"] = x85
spec.loader.exec_module(x85)
st = x85.st

D = Path(__file__).parent
ARMS = ["lcnu", "center", "ring10", "center_ring10",
        "center_lazy", "center_ghost", "center_zb"]


class LazyRing(x85.RingProbationLCNU):
    """Promotion needs k rehits; 1-rehit keys stay ring-evictable."""

    def __init__(self, capacity, ring_frac=0.10, promote_k=2, **k_):
        super().__init__(capacity, ring_frac=ring_frac, **k_)
        self._promote_k = promote_k
        self._hits = {}  # key -> rehit count while ring member

    def _on_hit(self, key):
        if key in self._ring:
            n = self._hits.get(key, 0) + 1
            self._hits[key] = n
            if n >= self._promote_k:
                self._ring.pop(key, None)  # promotion (base pop is then no-op)
                self._hits.pop(key, None)
            return  # under-k rehit: STAY in the ring (base would promote)
        super()._on_hit(key)  # citizen: base hook is a no-op pop

    def _on_miss(self, key):
        super()._on_miss(key)
        self._hits.pop(key, None)


_RING_BASE = x85.RingProbationLCNU


class GhostRing(x85.RingProbationLCNU):
    """Evicted one-timers remembered G accesses; re-admit as citizens."""

    def __init__(self, capacity, ring_frac=0.10, ghost=4096, **k_):
        super().__init__(capacity, ring_frac=ring_frac, **k_)
        self._ghost = {}  # key -> access index at ring eviction
        self._t = 0
        self._ghost_n = ghost

    def _evict(self, k):
        if k in self._ring:
            self._ghost[k] = self._t  # ring-evicted one-timer -> ghost
        return super()._evict(k)

    def _on_hit(self, key):
        self._t += 1
        self._ghost.pop(key, None)  # a rehit clears ghost status
        super()._on_hit(key)

    def _on_miss(self, key):
        self._t += 1
        stale = [g for g, t in self._ghost.items() if self._t - t >= self._ghost_n]
        for g in stale:
            del self._ghost[g]
        if key in self._ghost:  # known one-timer: citizen admission, no ring
            self._ghost.pop(key)
            super(_RING_BASE, self)._on_miss(key)
            return
        super()._on_miss(key)


class VisRingCostFit(x85.RingCostFit):
    """RingCostFit with a pluggable inner policy class (ring_frac=0.10)."""

    def __init__(self, capacity, inner_cls, num_layers=1, prior=1.0 / 16):
        x85.RingCostFit.__init__(self, capacity, num_layers=num_layers,
                                 prior=prior, ring_frac=0.10)
        self._inners["lcnu"] = inner_cls(
            capacity, ring_frac=0.10, prior=prior, num_layers=num_layers)


class ZeroBarrierRingCostFit(x85.RingCostFit):
    """Smoke finding: center_ring10 == center byte-equal at cap512 — the
    arbitration BARRIER (MARGIN + GAMMA*switch_mass) blocks the ring inner
    from ever being chosen where it would protect recovery. This arm drops
    the barrier (MARGIN=GAMMA=0): arbitrate purely on decayed hit-rate."""

    MARGIN = 0.0
    GAMMA = 0.0

    def __init__(self, capacity, num_layers=1, prior=1.0 / 16):
        x85.RingCostFit.__init__(self, capacity, num_layers=num_layers,
                                 prior=prior, ring_frac=0.10)


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    gk = {"num_layers": x85.rd.L, "prior": 8 / x85.rd.E}

    def center(cap):
        return x85.build_policy("costfit", cap, **gk)

    out = {}
    for cap in st.CAPS:
        cell = {}
        builders = {
            "lcnu": lambda: x85.build_policy("lcnu", cap, **gk),
            "center": lambda: center(cap),
            "ring10": lambda: x85.RingProbationLCNU(cap, ring_frac=0.10, **gk),
            "center_ring10": lambda: x85.RingCostFit(
                cap, ring_frac=0.10, **gk),
            "center_lazy": lambda: VisRingCostFit(cap, LazyRing, **gk),
            "center_ghost": lambda: VisRingCostFit(cap, GhostRing, **gk),
            "center_zb": lambda: ZeroBarrierRingCostFit(cap, **gk),
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
    r5 = json.loads((Path.home() / "warp-work/exp685-moei-ring/"
                     "exp685-ring.json").read_text())
    ref1 = {c["d"]: c["caps"] for c in r1["grid"]}
    ref5 = {c["d"]: c["caps"] for c in r5["grid"]}
    for cell in grid:
        for cap, c in cell["caps"].items():
            a = ref1[cell["d"]][cap]["costfit"]
            for m in ("loop", "burst", "rec"):
                if c["center"][m] != a[m]:
                    return 4, (f"repl1 d={cell['d']} cap={cap} {m}: "
                               f"{c['center'][m]} != {a[m]}")
            b = ref5[cell["d"]][cap]["ring10"]
            for m in ("loop", "burst", "rec"):
                if c["ring10"][m] != b[m]:
                    return 6, (f"repl3 d={cell['d']} cap={cap} {m}: "
                               f"{c['ring10'][m]} != {b[m]}")
    return 0, None


def adjudicate(grid):
    """Per-arm conjunction: a verdict arm must ITSELF pass vis+recov+burst+
    safety (no mixing metrics across arms)."""
    VIS_ARMS = ["center_ring10", "center_lazy", "center_ghost", "center_zb"]
    rows = []
    arm = {a: {"vis": True, "recov": False, "recov_partial": False,
               "burst": True, "burst_def": True, "stretch": False,
               "safety": True} for a in VIS_ARMS}
    for cell in grid:
        for cap, c in cell["caps"].items():
            if c["center"]["loop"] != c["center"]["loop2"]:
                return {"bad": "acc", "d": cell["d"], "cap": cap}
            bb = c["lcnu"]["burst"]
            row = {"d": cell["d"], "cap": int(cap)}
            for a in ARMS:
                row[a] = {m: round(c[a][m], 6) for m in ("loop", "burst", "rec")}
            row["cf_burst_def_pp"] = round((bb - c["center"]["burst"]) * 100, 3)
            row["cf_rec_def_pp"] = round((c["lcnu"]["rec"]
                                          - c["center"]["rec"]) * 100, 3)
            for a in VIS_ARMS:
                f = arm[a]
                row[f"{a}_burst_def_pp"] = round(
                    (bb - c[a]["burst"]) * 100, 3)
                row[f"{a}_rec_def_pp"] = round(
                    (c["lcnu"]["rec"] - c[a]["rec"]) * 100, 3)
                if (c[a]["loop"] - c["lcnu"]["loop"]) * 100 < -0.5:
                    f["safety"] = False
                if row[f"{a}_burst_def_pp"] > row["cf_burst_def_pp"] + 1e-9:
                    f["burst_def"] = False
                if int(cap) == 512:
                    if c[a]["burst"] == c["center"]["burst"] \
                            and c[a]["rec"] == c["center"]["rec"]:
                        f["vis"] = False  # byte-equal => still invisible
                if cell["d"] == 4096 and int(cap) == 512:
                    if row[f"{a}_rec_def_pp"] <= 3.5:
                        f["recov"] = True
                    if row[f"{a}_rec_def_pp"] <= 5.5:
                        f["recov_partial"] = True
                    if row[f"{a}_burst_def_pp"] <= 1.5:
                        f["stretch"] = True
            rows.append(row)
    def ok(a, *ks):
        return all(arm[a][k] for k in ks)
    winners = [a for a in VIS_ARMS
               if ok(a, "vis", "recov", "burst_def", "safety")]
    partial = [a for a in VIS_ARMS if ok(a, "vis", "recov_partial", "safety")]
    inburst = [a for a in VIS_ARMS if ok(a, "vis", "burst_def", "safety")]
    if winners:
        v = "BOTH_PASS"
    elif partial:
        v = "VIS_ONLY"
    elif inburst:
        v = "IN_BURST_ONLY"
    elif all(not arm[a]["vis"] for a in VIS_ARMS) \
            and all(arm[a]["safety"] for a in VIS_ARMS):
        v = "NO-VIS"
    else:
        v = "HARM"
    return {"verdict": v, "winners": winners, "arm_bars": arm, "rows": rows}


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
    res["repl"] = "exp681+exp685 exact"
    res["grid"] = g1
    res["rules"] = {"streams": "exp681 exact (68120261103)",
                    "arms": ARMS,
                    "bars": {"REPL1": "center == exp681 exact (exit 4)",
                             "REPL3": "ring10 == exp685 exact (exit 6)",
                             "VIS": "cr10 != center at cap512 AND rec_def <= 3.5pp @d4096/cap512",
                             "BURST": "cr10 burst_def <= center's ALL cells",
                             "STRETCH": "cr10 burst_def <= 1.5pp @d4096/cap512",
                             "SAFETY": "loops >= lcnu-0.5pp ALL"}}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp686a-ringvis.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| winners:", res["winners"],
          "| det:", res["det"], "| repl:", res["repl"])
    for r in res["rows"]:
        print("cell:", {"d": r["d"], "cap": r["cap"],
                        **{f"{a}_rec": r[f"{a}_rec_def_pp"]
                           for a in ("center_ring10", "center_lazy",
                                     "center_ghost", "center_zb")}})
    sys.exit(0 if res["det"] else 2)
