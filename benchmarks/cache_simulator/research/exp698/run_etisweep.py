#!/usr/bin/env python3
"""exp698: ETI fraction sweep — trace the achievable frontier curve.

exp697: ETI(10,10) is a strict interior point with a ~0.3pp-rec-per-
pp-burst exchange vs ring (5-10x flatter than lcnu->ring). exp698
sweeps annex_frac x ring_frac in {0.05, 0.10, 0.20}^2 at caps
512-4096, d4096, with lcnu/ring10/ssr_10 anchors and the exp695
classifier instrumented.

Questions (pre-registered):
- FLAT_FRONTIER: exchange rate <= 1.0pp rec per pp burst (vs the
  ring10 point) at >= 2 of {1024, 2048} for the best-burst variant.
- DOMINANT_POINT: any variant with burst >= ring10 + 0.10pp AND
  rec >= ring10 + 0.10pp at cap 1024 or 2048 (outright domination).
- HARM gate (re-calibrated vs exp697's artifact, per-arm): variant
  rec >= lcnu - 0.10pp AND variant burst >= ring10 - 0.50pp at all
  caps. Any variant failing -> flagged, excluded from PRIMARY.
Verdicts: DOMINANT_POINT > FLAT_FRONTIER > STEEPENS (else).
Gates: repl anchors == exp694 exactly (exit 4); det eti(10,10)@2048
double-pass must equal exp697's eti_10 cell exactly (exit 2);
consistency class counts (exit 3). Exit 0 settled.
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
spec97 = importlib.util.spec_from_file_location(
    "x97", str(HOME / "warp-work/exp697-moei-earnimmune/run_earnimmune.py"))
x97 = importlib.util.module_from_spec(spec97)
sys.modules["x97"] = x97
spec97.loader.exec_module(x97)

D = Path(__file__).parent
CAPS = [512, 1024, 2048, 4096]
ANCHORS = ["lcnu", "ring10", "ssr_10"]
FRACS = [0.05, 0.10, 0.20]
VARIANTS = [f"eti_{int(a*100):02d}{int(r*100):02d}" for a in FRACS
            for r in FRACS]


def mk_arm(arm, cap):
    if arm.startswith("eti_"):
        af = int(arm[4:6]) / 100.0
        rf = int(arm[6:8]) / 100.0
        gk = {"num_layers": x94.st.rd.L, "prior": 8 / x94.st.rd.E}
        return x97.ETILCNU(cap, annex_frac=af, ring_frac=rf, **gk)
    return x94.mk_arm(arm, cap)


def main():
    t0 = time.time()
    ref = json.loads((HOME / "warp-work/exp694-moei-capmap/"
                      "exp694-capmap.json").read_text())
    ref_cells = {c["d"]: c["caps"] for c in ref["map_grid"]}
    ref97 = json.loads((HOME / "warp-work/exp697-moei-earnimmune/"
                        "exp697-earnimmune.json").read_text())
    eti_cell = [c["caps"] for c in ref97["cells"] if c["d"] == 4096][0]
    keys, seg, _ = x94.stream_for(4096)
    rid = x97.burst_runs(seg)
    out = {"exp": "exp698-etisweep", "caps": {}}
    repl_ok, det_ok = True, True
    base_counts = None
    for cap in CAPS:
        cc = {}
        for arm in ANCHORS + VARIANTS:
            m = x97.pass_metrics(mk_arm(arm, cap), keys, seg, rid)
            cc[arm] = {"burst": m["burst"], "rec": m["rec"], "cls": m["cls"]}
            if arm in ref_cells[4096][str(cap)]:
                r = ref_cells[4096][str(cap)][arm]
                if (abs(m["burst"] - r["burst"]) > 1e-12
                        or abs(m["rec"] - r["rec"]) > 1e-12):
                    print(f"REPL FAIL cap={cap} {arm}")
                    repl_ok = False
            if arm == "eti_1010" and cap == 2048:
                r = eti_cell["2048"]["eti_10"]
                if (abs(m["burst"] - r["burst"]) > 1e-12
                        or abs(m["rec"] - r["rec"]) > 1e-12):
                    print("DET/REPL97 FAIL eti_1010@2048")
                    det_ok = False
        counts = cc["eti_1010"]["cls"]
        sig = tuple(counts[x][0] for x in ("a", "b", "c"))
        if base_counts is None:
            base_counts = sig
        elif sig != base_counts:
            print(f"CONSISTENCY FAIL cap={cap}")
            sys.exit(3)
        out["caps"][str(cap)] = cc
        print(f"cap{cap} done [{time.time()-t0:.0f}s]", flush=True)
    if not repl_ok:
        sys.exit(4)
    # det double-pass
    m2 = x97.pass_metrics(mk_arm("eti_1010", 2048), keys, seg, rid)
    r = out["caps"]["2048"]["eti_1010"]
    if m2["burst"] != r["burst"] or m2["cls"] != r["cls"]:
        det_ok = False
    if not det_ok:
        print("DET FAIL")
        sys.exit(2)
    # adjudicate
    def g(cap, arm, f):
        return out["caps"][str(cap)][arm][f]
    harm = {}
    for v in VARIANTS:
        harm[v] = all(g(c, v, "rec") >= g(c, "lcnu", "rec") - 0.001
                      and g(c, v, "burst") >= g(c, "ring10", "burst") - 0.005
                      for c in CAPS)
    eligible = [v for v in VARIANTS if harm[v]]
    dominant = [v for v in VARIANTS if any(
        g(c, v, "burst") >= g(c, "ring10", "burst") + 0.001
        and g(c, v, "rec") >= g(c, "ring10", "rec") + 0.001
        for c in (1024, 2048))]
    flat = False
    for v in eligible:
        hits = 0
        for c in (1024, 2048):
            db = g(c, v, "burst") - g(c, "ring10", "burst")
            dr = g(c, v, "rec") - g(c, "ring10", "rec")
            if db > 0.001 and dr / db >= -0.01:  # flat: lose <=1pp rec per pp burst
                hits += 1
        if hits >= 2:
            flat = True
            break
    verdict = ("DOMINANT_POINT" if dominant else
               "FLAT_FRONTIER" if flat else "STEEPENS")
    out.update({"verdict": verdict, "dominant": dominant, "flat": flat,
                "harm": harm, "det": True,
                "elapsed_s": round(time.time() - t0, 1)})
    (D / "exp698-etisweep.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", verdict, "| dominant:", dominant, "| flat:", flat,
          "| harm-excluded:", [v for v in VARIANTS if not harm[v]])
    for cap in CAPS:
        print(f"cap{cap:5d} ring10:{g(cap,'ring10','burst'):.4f}/"
              f"{g(cap,'ring10','rec'):.4f} " + " ".join(
                  f"{v}:{g(cap,v,'burst'):.4f}/{g(cap,v,'rec'):.4f}"
                  for v in VARIANTS))
    sys.exit(0)


if __name__ == "__main__":
    main()
