#!/usr/bin/env python3
"""exp696: composite admission policy — revisit annex (ssr) + churn
quarantine ring (ring) on one LCNU base.

Evidence chain: exp693 ssr = free lunch vs lcnu (equal-or-better both
axes everywhere); exp694 ring10 = steep rec-for-burst trade (+2.34pp
rec / -8.86pp burst at cap1024; neither dominates ssr); exp695 burst
window is 99.9% re-accesses of keys seen in the prefix (a=6, b=4,
c=4086 of 4096). The two mechanisms act on disjoint populations:
  ring  quarantines FIRST-SIGHTS (churn among themselves, hot set
        shielded) -> rec gain, burst cost (burst keys evicted before
        their in-window re-access);
  ssr   protects REVISITS (scoring cannot evict annex members) ->
        burst re-entries stay resident, rec untouched.
Composite SringLCNU: first-sight miss -> admit into ring (quarantine);
revisit miss -> admit into annex (protected); hit -> promotion out of
both. _victim: ring-full -> drain oldest ring member; else
batch-exclude annex from scoring; empty-candidate fallback drains
annex.

Pre-registered bars (d4096, caps 512-8192; PP units):
  HARM gate (must hold at ALL caps, else verdict HARM):
    sring rec >= lcnu rec - 0.10 AND sring burst >= lcnu burst - 0.50.
  PRIMARY COMPOSITE_DOMINATES: exists cap in {1024, 2048}:
    sring rec >= ring rec + 0.10 AND sring burst >= ring burst + 0.10
    (outright Pareto over the fence).
  SECONDARY ADDITIVE_REFUND: else exists cap in {1024, 2048}:
    sring rec >= ring rec - 0.10 AND sring burst >= ring burst + 1.00.
  else TRADE_UNCHANGED.
  MECH (report-only): sring class-c hits >= ring class-c hits at
  1024/2048 (refund carried by carry-in re-entries).

Gates: repl lcnu/ring10/ssr_10/center == exp694-capmap.json exactly
(both depths, all caps; exit 4); det sring cap2048 d4096 double-pass
(exit 2); consistency: class counts identical across arms (exit 3).
Exit 0 settled.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
spec94 = importlib.util.spec_from_file_location(
    "x94", str(HOME / "warp-work/exp694-moei-capmap/run_capmap.py"))
x94 = importlib.util.module_from_spec(spec94)
sys.modules["x94"] = x94
spec94.loader.exec_module(x94)
from benchmarks.cache_simulator.policies import (  # noqa: E402
    LayerCycleNextUsePolicy)

D = Path(__file__).parent
CAPS = [512, 1024, 2048, 4096, 8192]
DEPTHS = [4096, 1024]
BASE_ARMS = ["lcnu", "ring10", "ssr_10", "center"]


class SringLCNU(LayerCycleNextUsePolicy):
    """Composite: revisit annex + first-sight churn ring on LCNU.

    First-sight miss -> admit + ring (quarantine; ring-full drains
    oldest ring member). Revisit miss -> admit + annex (protected from
    scoring). Hit -> promotion out of both structures. Victim order:
    lazy-clean; ring-full -> oldest ring member; else batch-exclude
    annex from scoring (O(1) restore); empty candidate -> drain oldest
    annex member.
    """

    def __init__(self, capacity, annex_frac=0.10, ring_frac=0.10, **k):
        super().__init__(capacity, **k)
        self._annex_cap = max(1, int(round(annex_frac * capacity)))
        self._ring_cap = max(1, int(round(ring_frac * capacity)))
        self._annex = {}
        self._ring = {}
        self._ever = set()

    def _on_hit(self, key):
        self._ring.pop(key, None)
        if key in self._annex:
            del self._annex[key]  # promotion frees protection slot

    def _on_miss(self, key):
        revisit = key in self._ever
        super()._on_miss(key)
        self._ever.add(key)
        if revisit:
            if len(self._annex) >= self._annex_cap:
                oldest = next(iter(self._annex))
                del self._annex[oldest]
            self._annex[key] = None
        else:
            if len(self._ring) >= self._ring_cap:
                pass  # ring stays bounded; victim drains it on demand
            self._ring[key] = None

    def _victim(self):
        for k in [k for k in self._annex if k not in self.resident]:
            del self._annex[k]
        for k in [k for k in self._ring if k not in self.resident]:
            del self._ring[k]
        if len(self._ring) >= self._ring_cap:
            k = next(iter(self._ring))
            return k[0], k
        if len(self.resident) - len(self._annex) <= 0:
            k = next(iter(self._annex))
            return k[0], k
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
            k = next(iter(self._annex))
            layer = k[0]
        return layer, k


def burst_runs(seg):
    rid = np.zeros(len(seg), dtype=np.int64)
    cur, in_run = 0, False
    for i, s in enumerate(seg):
        if s == 3:
            if not in_run:
                cur += 1
                in_run = True
            rid[i] = cur
        else:
            in_run = False
    return rid


def pass_metrics(pol, keys, seg, rid):
    seen = {}
    st = {"a": [0, 0], "b": [0, 0], "c": [0, 0], "rec": [0, 0]}
    for i in range(len(keys)):
        k = keys[i]
        hit = pol.on_access(k)
        s = seg[i]
        if s == 3:
            r = rid[i]
            if k not in seen:
                cls = "a"
            elif seen[k] == r and r > 0:
                cls = "b"
            else:
                cls = "c"
            st[cls][0] += 1
            st[cls][1] += hit
        elif s == 4:
            st["rec"][0] += 1
            st["rec"][1] += hit
        if k not in seen:
            seen[k] = rid[i]
    n_b = st["a"][0] + st["b"][0] + st["c"][0]
    burst = (st["a"][1] + st["b"][1] + st["c"][1]) / n_b
    rec = st["rec"][1] / st["rec"][0]
    return {"burst": burst, "rec": rec, "cls": st}


def mk_arm(arm, cap):
    if arm == "sring_10":
        gk = {"num_layers": x94.st.rd.L, "prior": 8 / x94.st.rd.E}
        return SringLCNU(cap, annex_frac=0.10, ring_frac=0.10, **gk)
    return x94.mk_arm(arm, cap)


def main():
    t0 = time.time()
    ref = json.loads((HOME / "warp-work/exp694-moei-capmap/"
                      "exp694-capmap.json").read_text())
    ref_cells = {c["d"]: c["caps"] for c in ref["map_grid"]}
    out = {"exp": "exp696-sring", "cells": []}
    repl_ok, det_ok = True, True
    for d in DEPTHS:
        keys, seg, _ = x94.stream_for(d)
        rid = burst_runs(seg)
        cell = {"d": d, "caps": {}}
        base_counts = None
        for cap in CAPS:
            cc = {}
            for arm in BASE_ARMS + ["sring_10"]:
                m = pass_metrics(mk_arm(arm, cap), keys, seg, rid)
                cc[arm] = m
                if arm in ref_cells[d][str(cap)]:
                    r = ref_cells[d][str(cap)][arm]
                    if (abs(m["burst"] - r["burst"]) > 1e-12
                            or abs(m["rec"] - r["rec"]) > 1e-12):
                        print(f"REPL FAIL d={d} cap={cap} {arm}")
                        repl_ok = False
                if arm == "sring_10" and d == 4096 and cap == 2048:
                    det_ref = m
            counts = cc["sring_10"]["cls"]
            sig = tuple(counts[x][0] for x in ("a", "b", "c"))
            if base_counts is None:
                base_counts = sig
            elif sig != base_counts:
                print(f"CONSISTENCY FAIL d={d} cap={cap}")
                sys.exit(3)
            cell["caps"][str(cap)] = {a: {"burst": cc[a]["burst"],
                                          "rec": cc[a]["rec"],
                                          "cls": {x: cc[a]["cls"][x]
                                                  for x in
                                                  ("a", "b", "c")}}
                                      for a in cc}
        out["cells"].append(cell)
        print(f"d={d} done [{time.time()-t0:.0f}s]", flush=True)
    if not repl_ok:
        sys.exit(4)
    keys, seg, _ = x94.stream_for(4096)
    rid = burst_runs(seg)
    m2 = pass_metrics(mk_arm("sring_10", 2048), keys, seg, rid)
    if (m2["cls"] != det_ref["cls"]) or abs(
            m2["burst"] - det_ref["burst"]) > 1e-12:
        det_ok = False
    m = out["cells"][0]["caps"]
    def g(cap, arm, f):
        return m[str(cap)][arm][f]
    def pp(x):
        return x * 100
    harm = all(g(c, "sring_10", "rec") >= g(c, "lcnu", "rec") - 0.001
               and g(c, "sring_10", "burst") >= g(c, "lcnu", "burst") - 0.005
               for c in CAPS)
    primary = any(
        g(c, "sring_10", "rec") >= g(c, "ring10", "rec") + 0.001
        and g(c, "sring_10", "burst") >= g(c, "ring10", "burst") + 0.001
        for c in (1024, 2048))
    secondary = any(
        g(c, "sring_10", "rec") >= g(c, "ring10", "rec") - 0.001
        and g(c, "sring_10", "burst") >= g(c, "ring10", "burst") + 0.010
        for c in (1024, 2048))
    if not harm:
        verdict = "HARM"
    elif primary:
        verdict = "COMPOSITE_DOMINATES"
    elif secondary:
        verdict = "ADDITIVE_REFUND"
    else:
        verdict = "TRADE_UNCHANGED"
    mech = {str(c): (g(c, "sring_10", "cls")["c"][1],
                     g(c, "ring10", "cls")["c"][1]) for c in (1024, 2048)}
    out.update({"verdict": verdict, "harm": harm, "primary": primary,
                "secondary": secondary, "det": det_ok, "mech_c_hits": mech,
                "elapsed_s": round(time.time() - t0, 1)})
    (D / "exp696-sring.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", verdict, "| primary", primary, "| secondary",
          secondary, "| harm-ok", harm, "| det", det_ok)
    for cap in CAPS:
        print(f"cap{cap:5d} " + " ".join(
            f"{a}:{g(cap,a,'burst'):.4f}/{g(cap,a,'rec'):.4f}"
            for a in ("lcnu", "ring10", "ssr_10", "sring_10")))
    sys.exit(0 if det_ok else 2)


if __name__ == "__main__":
    main()
