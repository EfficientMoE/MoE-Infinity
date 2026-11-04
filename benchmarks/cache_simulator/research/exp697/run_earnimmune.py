#!/usr/bin/env python3
"""exp697: earn-then-immune (ETI) — the frontier's interior point.

exp696 pinned the conflict: the burst/recovery frontier IS the
revisit-miss routing rule. Poles: ssr (protect at re-sight admission
-> burst win) vs ring (every admission churns until it survives to a
re-hit -> rec win). ETI is the interior: re-entry revisits join the
ring and must EARN by re-hitting once; the re-hit promotes them to
the scorer-immune annex (ssr's protection, one re-sight later).

Churn arithmetic (pre-registered expectation): during the burst,
~2600 distinct keys flood a ~cap/10 ring; drain spacing means the
median carry-in is evicted before its in-window re-access (ring burst
0.0488 == this arithmetic). ETI's earn event (a HIT) can therefore
rarely fire for carry-ins -> expect COLLAPSES_TO_RING on the burst
axis while rec keeps ring's gain. If burst instead recovers >= 1pp,
the churn model is wrong — that would be the finding.

Policy ETILCNU (LayerCycleNextUsePolicy):
  miss -> admit + ring (EVERY miss, ring semantics preserved)
  hit  -> ring.pop + grant annex membership (scorer-immune); annex
          overflow drains oldest member (exposed, stays resident)
  victim: lazy-clean both; ring-full -> drain oldest ring member NOT
          in annex (annex = earned immunity, ring-drain-proof); if
          all ring members annexed -> batch-exclude annex from scorer
          and score; empty candidate -> drain oldest annex member.

Bars (d4096, caps 512-8192; PP):
  HARM gate (all caps, else HARM): eti rec >= lcnu rec - 0.10 AND
  eti burst >= lcnu burst - 0.50.
  STRONG DOMINATES_RING: cap in {1024, 2048}: eti rec >= ring rec
  + 0.10 AND eti burst >= ring burst + 0.10.
  PRIMARY MIDDLE_POINT: cap in {1024, 2048}: eti rec >= ring rec
  - 0.10 AND eti burst >= ring burst + 1.00.
  SECONDARY COLLAPSES_TO_RING: |eti burst - ring burst| < 1.00 at
  BOTH 1024 and 2048 AND eti rec >= ring rec - 0.10 (earn event
  cannot fire under churn; supports the drain-spacing model).
  else MIXED.
  MECH (report-only): eti class-c hits vs ring @1024/2048.

Gates: repl lcnu/ring10/ssr_10 == exp694 exactly (d4096 all caps,
exit 4); det eti cap2048 d4096 double-pass (exit 2); consistency:
class counts identical across arms (exit 3). Exit 0 settled.
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
BASE_ARMS = ["lcnu", "ring10", "ssr_10"]


class ETILCNU(LayerCycleNextUsePolicy):
    """Earn-then-immune: churn until first re-hit, then scorer-immune."""

    def __init__(self, capacity, annex_frac=0.10, ring_frac=0.10, **k):
        super().__init__(capacity, **k)
        self._annex_cap = max(1, int(round(annex_frac * capacity)))
        self._ring_cap = max(1, int(round(ring_frac * capacity)))
        self._annex = {}
        self._ring = {}

    def _on_hit(self, key):
        self._ring.pop(key, None)
        if key not in self._annex:
            if len(self._annex) >= self._annex_cap:
                oldest = next(iter(self._annex))
                del self._annex[oldest]
            self._annex[key] = None

    def _on_miss(self, key):
        super()._on_miss(key)
        self._ring[key] = None

    def _victim(self):
        for k in [k for k in self._annex if k not in self.resident]:
            del self._annex[k]
        for k in [k for k in self._ring if k not in self.resident]:
            del self._ring[k]
        if len(self._ring) >= self._ring_cap:
            victim = None
            for k in self._ring:
                if k not in self._annex:
                    victim = k
                    break
            if victim is not None:
                return victim[0], victim
            # all ring members earned immunity: score with annex excluded
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
    return {"burst": (st["a"][1] + st["b"][1] + st["c"][1]) / n_b,
            "rec": st["rec"][1] / st["rec"][0], "cls": st}


def mk_arm(arm, cap):
    if arm == "eti_10":
        gk = {"num_layers": x94.st.rd.L, "prior": 8 / x94.st.rd.E}
        return ETILCNU(cap, annex_frac=0.10, ring_frac=0.10, **gk)
    return x94.mk_arm(arm, cap)


def main():
    t0 = time.time()
    ref = json.loads((HOME / "warp-work/exp694-moei-capmap/"
                      "exp694-capmap.json").read_text())
    ref_cells = {c["d"]: c["caps"] for c in ref["map_grid"]}
    out = {"exp": "exp697-earnimmune", "cells": []}
    repl_ok, det_ok = True, True
    det_ref = None
    for d in (4096, 1024):
        keys, seg, _ = x94.stream_for(d)
        rid = burst_runs(seg)
        cell = {"d": d, "caps": {}}
        base_counts = None
        for cap in CAPS:
            cc = {}
            for arm in BASE_ARMS + ["eti_10"]:
                m = pass_metrics(mk_arm(arm, cap), keys, seg, rid)
                cc[arm] = m
                if arm in ref_cells[d][str(cap)]:
                    r = ref_cells[d][str(cap)][arm]
                    if (abs(m["burst"] - r["burst"]) > 1e-12
                            or abs(m["rec"] - r["rec"]) > 1e-12):
                        print(f"REPL FAIL d={d} cap={cap} {arm}")
                        repl_ok = False
                if arm == "eti_10" and d == 4096 and cap == 2048:
                    det_ref = m
            counts = cc["eti_10"]["cls"]
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
    m2 = pass_metrics(mk_arm("eti_10", 2048), keys, seg, rid)
    if m2 != det_ref:
        det_ok = False
    m = out["cells"][0]["caps"]
    def g(cap, arm, f):
        return m[str(cap)][arm][f]
    harm = all(g(c, "eti_10", "rec") >= g(c, "lcnu", "rec") - 0.001
               and g(c, "eti_10", "burst") >= g(c, "lcnu", "burst") - 0.005
               for c in CAPS)
    strong = any(
        g(c, "eti_10", "rec") >= g(c, "ring10", "rec") + 0.001
        and g(c, "eti_10", "burst") >= g(c, "ring10", "burst") + 0.001
        for c in (1024, 2048))
    primary = any(
        g(c, "eti_10", "rec") >= g(c, "ring10", "rec") - 0.001
        and g(c, "eti_10", "burst") >= g(c, "ring10", "burst") + 0.010
        for c in (1024, 2048))
    coll_ring = (all(
        abs(g(c, "eti_10", "burst") - g(c, "ring10", "burst")) < 0.010
        for c in (1024, 2048))
        and all(g(c, "eti_10", "rec") >= g(c, "ring10", "rec") - 0.001
                for c in (1024, 2048)))
    if not harm:
        verdict = "HARM"
    elif strong:
        verdict = "DOMINATES_RING"
    elif primary:
        verdict = "MIDDLE_POINT"
    elif coll_ring:
        verdict = "COLLAPSES_TO_RING"
    else:
        verdict = "MIXED"
    mech = {str(c): (g(c, "eti_10", "cls")["c"][1],
                     g(c, "ring10", "cls")["c"][1]) for c in (1024, 2048)}
    out.update({"verdict": verdict, "harm": harm, "strong": strong,
                "primary": primary, "collapses_to_ring": coll_ring,
                "det": det_ok, "mech_c_hits": mech,
                "elapsed_s": round(time.time() - t0, 1)})
    (D / "exp697-earnimmune.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", verdict, "| strong", strong, "| primary", primary,
          "| coll_ring", coll_ring, "| harm-ok", harm, "| det", det_ok)
    for cap in CAPS:
        print(f"cap{cap:5d} " + " ".join(
            f"{a}:{g(cap,a,'burst'):.4f}/{g(cap,a,'rec'):.4f}"
            for a in ("lcnu", "ring10", "ssr_10", "eti_10")))
    sys.exit(0 if det_ok else 2)


if __name__ == "__main__":
    main()
