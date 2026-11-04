#!/usr/bin/env python3
"""exp693: SecondSightRing — revisit-only annex coexistence (no arbiter).

Family closure (exp686a/687/689/690/691): burst is shorter than any
natural arbitration window, so no rate signal can both detect and act
inside it; burst-vs-recovery at cap512 is a capacity frontier. exp691's
registered fallback: coexistence — a SINGLE policy holding both
concerns, no timing needed.

Mechanism under test: annex membership is granted ONLY to revisits
(keys seen at least once before, tracked in an ever-admitted set).
- First-sight keys (the burst crowd) are admitted WITHOUT annex
  protection: they compete by LCNU score and evict each other, exactly
  like plain lcnu — burst behavior preserved.
- Revisits (the recovery working set, re-accessed after an eviction)
  are admitted WITH annex protection: _victim scores only non-annex
  residents, so annex members survive burst floods. Annex is FIFO-
  bounded (frac x capacity); when full, the oldest member is exposed
  (stays resident, loses protection). Annex hit promotes out of the
  annex (frees protection for future revisits; matches ring10's
  promotion semantics).
- Prediction: burst_def stays near lcnu's own; rec approaches ring10's
  0.6853 at home because the recovery set survives the burst in the
  annex instead of being evicted.

Arms: lcnu | center (REPL1 exp681, exit 4) | ring10 (REPL3 exp685,
exit 6) | ssr_05 | ssr_10 | ssr_15 (annex_frac sweep 5/10/15%).

Bars (per-arm conjunction on ssr_*):
  RECOV  rec_def <= 3.5pp at d4096/cap512
  BURST  burst_def <= center's deficit ALL 8 cells
  STRETCH burst_def <= 1.5pp at home
  SAFETY loop >= lcnu - 0.5pp ALL cells
Verdicts: BOTH_PASS > REC_ONLY > BURST_ONLY > NO-SIGNAL > HARM.
Exit codes: 0 settled | 2 det | 3 acc | 4 repl1 | 6 repl3.
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
    "x85", str(Path.home() / "warp-work/exp685-moei-ring/run_ring.py"))
x85 = importlib.util.module_from_spec(spec85)
sys.modules["x85"] = x85
spec85.loader.exec_module(x85)
st, rd = x85.st, x85.rd
RingProbationLCNU = x85.RingProbationLCNU
LayerCycleNextUsePolicy = x85.LayerCycleNextUsePolicy

D = Path(__file__).parent
ARMS = ["lcnu", "center", "ring10", "ssr_05", "ssr_10", "ssr_15"]
T_ARMS = ARMS[3:]


class SecondSightLCNU(LayerCycleNextUsePolicy):
    """Revisit-only spared annex inside one resident set.

    Base LCNU machinery unedited; hooks mirror exp685's
    RingProbationLCNU (_on_hit/_on_miss/_victim), but annex membership
    is granted on REVISIT admission only — first-sights are never
    protected, so burst crowds cannot pollute the annex."""

    def __init__(self, capacity, annex_frac=0.10, **k):
        super().__init__(capacity, **k)
        if not 0.0 < annex_frac < 1.0:
            raise ValueError("annex_frac in (0,1)")
        self._annex_cap = max(1, int(round(annex_frac * capacity)))
        self._annex = {}   # key -> None, insertion order = admission order
        self._ever = set()  # keys admitted at least once (revisit memory)

    def _on_hit(self, key):
        if key in self._annex:
            del self._annex[key]  # promotion: frees protection slot

    def _on_miss(self, key):
        revisit = key in self._ever
        super()._on_miss(key)
        self._ever.add(key)
        if revisit:
            if len(self._annex) >= self._annex_cap:
                oldest = next(iter(self._annex))
                del self._annex[oldest]  # exposed, stays resident
            self._annex[key] = None

    def _victim(self):
        for k in [k for k in self._annex if k not in self.resident]:
            del self._annex[k]  # lazy cleanup of evicted members
        if len(self.resident) - len(self._annex) <= 0:
            k = next(iter(self._annex))  # everything protected: drain oldest
            return k[0], k
        # Batch exclusion (REPAIR of the frozen sequential fence — same
        # victim, O(1) recomputes): temporarily remove ALL annex members
        # from _by_layer + mark their layers stale, take ONE base pick
        # (the max-score unprotected key, deterministic tie-break), then
        # restore. Equivalent to fencing members one-by-one and
        # re-asking, which converges to the same pick but costs
        # O(annex) candidate recomputes per miss.
        saved = []
        for k in list(self._annex):
            self._by_layer[k[0]].discard(k)
            self._stale.add(k[0])
            saved.append(k)
        layer, k = super()._victim()
        for fk in saved:
            self._by_layer[fk[0]].add(fk)
            self._stale.add(fk[0])
        if k is None:
            # scoring found no candidate: drain oldest annex
            k = next(iter(self._annex))
            layer = k[0]
        return layer, k


def mk_arm(name, cap):
    gk = {"num_layers": rd.L, "prior": 8 / rd.E}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "center":
        return build_policy("costfit", cap, **gk)
    if name == "ring10":
        return RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name.startswith("ssr_"):
        return SecondSightLCNU(cap, annex_frac=float(name[4:]) / 100, **gk)
    raise KeyError(name)


def evaluate(keys, seg):
    burst, rec = (seg == 3), (seg == 4)
    out = {}
    for cap in st.CAPS:
        cell = {}
        for name in ARMS:
            pol = mk_arm(name, cap)
            h = np.array([pol.on_access(k) for k in keys], dtype=bool)
            cell[name] = {"loop": float(h.mean()),
                          "burst": float(h[burst].mean()),
                          "rec": float(h[rec].mean())}
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
    ref1 = {c["d"]: c["caps"] for c in r1["grid"]}
    ref5 = {c["d"]: c["caps"] for c in r5["grid"]}
    for cell in grid:
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
            row["cf_burst_def_pp"] = round((bb - c["center"]["burst"]) * 100, 3)
            row["cf_rec_def_pp"] = round(
                (c["lcnu"]["rec"] - c["center"]["rec"]) * 100, 3)
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
    res["repl"] = "exp681+exp685 exact"
    res["grid"] = g1
    res["rules"] = {"streams": "exp681 exact (68120261103)",
                    "arms": ARMS,
                    "bars": {"REPL1": "center == exp681 exact (exit 4)",
                             "REPL3": "ring10 == exp685 exact (exit 6)",
                             "RECOV": "arm rec_def <= 3.5pp @d4096/cap512",
                             "BURST": "arm burst_def <= center's ALL cells",
                             "SAFETY": "loop >= lcnu-0.5pp ALL"}}
    res["elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp693-ssr.json").write_text(json.dumps(res, indent=1))
    print("VERDICT:", res["verdict"], "| winners:", res["winners"],
          "| det:", res["det"], "| repl:", res["repl"])
    for r in res["rows"]:
        print("cell:", {"d": r["d"], "cap": r["cap"],
                        **{f"{a}_rec": r[f"{a}_rec_def_pp"] for a in T_ARMS},
                        "burst_def": {a: r[f"{a}_burst_def_pp"] for a in T_ARMS}})
    sys.exit(0 if res["det"] else 2)
