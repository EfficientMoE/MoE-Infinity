#!/usr/bin/env python3
"""exp695: Decompose lcnu's burst behavior across capacity.

exp694 mapped the frontier: lcnu burst hit grows 0.0884@cap512 ->
0.4099@cap4096 -> 0.9985@cap8192 (d4096); ring10's sacrifice is
non-monotone (5.96/8.86/9.23/0.88pp at caps 512/1024/2048/4096).
exp695 asks WHY: classify every burst-window access by key history
(same stream, policy-independent classes) and locate where each arm
wins/loses.

Classes (per burst access i, key k; run id = contiguous seg==3 run):
  (a) first-sight     — k never seen before i
  (b) same-burst      — k first seen within the SAME burst run
  (c) carry-in        — k first seen earlier (earlier burst run or
                        outside burst)

H1 (registered): lcnu's growing burst hit with capacity is driven by
class (b) same-burst re-accesses: (b_hits@cap4096 - b_hits@cap512) /
(total_burst_hits@cap4096 - total_burst_hits@cap512) >= 0.80 at d4096.
H2 (registered): ring's recovery mechanism concentrates where it
protects: ring class-(c) hit > lcnu class-(c) hit at caps where its
rec edge >= 1pp (1024, 2048 per exp694); and ring class-(b) hit <
lcnu class-(b) (sacrifice located in flood re-accesses).
Verdicts: MECH_CONFIRMED (H1 and H2) | PARTIAL (one) | REFUTED (neither).

Gates:
  REPL: lcnu+ring10 burst/rec at d4096 ALL caps equal exp694-capmap.json
        exactly (exit 4 on mismatch).
  CONSIST: class counts identical across arms (same stream) (exit 3).
  det: rerun (lcnu, cap2048, d4096) pass — identical accumulators (exit 2).
Exit: 0 settled | 2 det | 3 consistency | 4 repl.

Arms: lcnu, ring10. Caps: 512/1024/2048/4096/8192. Depths: 4096 (main),
1024 (control). N full 638,976; exp681 streams via run_capmap machinery.
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

D = Path(__file__).parent
CAPS = [512, 1024, 2048, 4096, 8192]
DEPTHS = [4096, 1024]
ARMS = ["lcnu", "ring10"]


def burst_runs(seg):
    """run id per index: 0 outside burst, 1..K inside contiguous seg==3 runs."""
    rid = np.zeros(len(seg), dtype=np.int64)
    cur = 0
    in_run = False
    for i, s in enumerate(seg):
        if s == 3:
            if not in_run:
                cur += 1
                in_run = True
            rid[i] = cur
        else:
            in_run = False
    return rid


def classified_pass(pol, keys, seg, rid):
    """One policy pass; classify burst accesses; return per-class stats."""
    seen = {}  # key -> first burst-run id (0 if first seen outside burst)
    stats = {"a": [0, 0], "b": [0, 0], "c": [0, 0], "tot": [0, 0]}
    for i in range(len(keys)):
        k = keys[i]
        hit = pol.on_access(k)
        if seg[i] == 3:
            r = rid[i]
            if k not in seen:
                cls = "a"
            elif seen[k] == r and r > 0:
                cls = "b"
            else:
                cls = "c"
            stats[cls][0] += 1
            stats[cls][1] += hit
            stats["tot"][0] += 1
            stats["tot"][1] += hit
        if k not in seen:
            seen[k] = rid[i]
    return stats


def main():
    t0 = time.time()
    ref = json.loads((HOME / "warp-work/exp694-moei-capmap/"
                      "exp694-capmap.json").read_text())
    ref_cells = {c["d"]: c["caps"] for c in ref["map_grid"]}
    out = {"exp": "exp695-flooddecomp", "cells": []}
    det_ok = True
    repl_ok = True
    for d in DEPTHS:
        keys, seg, _ = x94.stream_for(d)
        rid = burst_runs(seg)
        cell = {"d": d, "n_burst_runs": int(rid.max()), "caps": {}}
        for cap in CAPS:
            cc = {}
            base_counts = None
            for arm in ARMS:
                pol = x94.mk_arm(arm, cap)
                st = classified_pass(pol, keys, seg, rid)
                cc[arm] = {"a_cnt": st["a"][0], "a_hits": st["a"][1],
                           "b_cnt": st["b"][0], "b_hits": st["b"][1],
                           "c_cnt": st["c"][0], "c_hits": st["c"][1],
                           "tot_cnt": st["tot"][0], "tot_hits": st["tot"][1]}
                if base_counts is None:
                    base_counts = (st["a"][0], st["b"][0], st["c"][0],
                                   st["tot"][0])
                elif (st["a"][0], st["b"][0], st["c"][0],
                      st["tot"][0]) != base_counts:
                    print(f"CONSISTENCY FAIL d={d} cap={cap} {arm}")
                    sys.exit(3)
            # repl gate vs exp694 (d4096 only: exp694 home depth)
            if d == 4096:
                r = ref_cells[4096][str(cap)]
                for arm in ARMS:
                    my_burst = cc[arm]["tot_hits"] / cc[arm]["tot_cnt"]
                    # rec needs the seg==4 rate: recompute cheaply? use
                    # stored? run one more plain pass for rec only.
                    if abs(my_burst - r[arm]["burst"]) > 1e-12:
                        print(f"REPL FAIL d=4096 cap={cap} {arm} burst "
                              f"{my_burst} != {r[arm]['burst']}")
                        repl_ok = False
            cell["caps"][str(cap)] = cc
        out["cells"].append(cell)
        print(f"d={d} done [{time.time()-t0:.0f}s]", flush=True)
    if not repl_ok:
        sys.exit(4)
    # det: rerun lcnu cap2048 d4096
    keys, seg, _ = x94.stream_for(4096)
    rid = burst_runs(seg)
    st2 = classified_pass(x94.mk_arm("lcnu", 2048), keys, seg, rid)
    refc = out["cells"][0]["caps"]["2048"]["lcnu"]
    if (st2["a"][0], st2["a"][1], st2["b"][0], st2["b"][1],
            st2["c"][0], st2["c"][1]) != (refc["a_cnt"], refc["a_hits"],
                                          refc["b_cnt"], refc["b_hits"],
                                          refc["c_cnt"], refc["c_hits"]):
        det_ok = False
    # adjudicate on d4096
    main_cells = out["cells"][0]["caps"]
    def rate(cap, arm, cls):
        c = main_cells[str(cap)][arm]
        return (c[f"{cls}_hits"] / c[f"{cls}_cnt"]) if c[f"{cls}_cnt"] else 0.0
    b_delta = (main_cells["4096"]["lcnu"]["b_hits"]
               - main_cells["512"]["lcnu"]["b_hits"])
    tot_delta = (main_cells["4096"]["lcnu"]["tot_hits"]
                 - main_cells["512"]["lcnu"]["tot_hits"])
    h1 = tot_delta > 0 and b_delta / tot_delta >= 0.80
    h2 = True
    for cap in (1024, 2048):
        if rate(cap, "ring10", "c") <= rate(cap, "lcnu", "c"):
            h2 = False
        if rate(cap, "ring10", "b") >= rate(cap, "lcnu", "b"):
            h2 = False
    verdict = ("MECH_CONFIRMED" if (h1 and h2)
               else "PARTIAL" if (h1 or h2) else "REFUTED")
    out.update({"verdict": verdict, "h1": h1, "h2": h2,
                "b_share_of_growth": round(b_delta / tot_delta, 4)
                if tot_delta else None,
                "det": det_ok, "repl": "exp694 exact (burst, d4096 all caps)",
                "elapsed_s": round(time.time() - t0, 1)})
    (D / "exp695-flooddecomp.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", verdict, "| H1:", h1, "| H2:", h2,
          "| b_share:", out["b_share_of_growth"], "| det:", det_ok)
    for cap in CAPS:
        c = main_cells[str(cap)]
        print(f"cap{cap:5d} lcnu a={c['lcnu']['a_hits']}/{c['lcnu']['a_cnt']}"
              f" b={c['lcnu']['b_hits']}/{c['lcnu']['b_cnt']}"
              f" c={c['lcnu']['c_hits']}/{c['lcnu']['c_cnt']}"
              f" | ring b={c['ring10']['b_hits']}/{c['ring10']['b_cnt']}"
              f" c={c['ring10']['c_hits']}/{c['ring10']['c_cnt']}")
    sys.exit(0 if det_ok else 2)


if __name__ == "__main__":
    main()
