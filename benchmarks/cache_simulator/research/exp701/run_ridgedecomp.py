#!/usr/bin/env python3
"""exp701: ridge decomposition part 2 — off-grid probes.

Part 1 (analysis of sealed exp700 JSON, done): ring-vs-ssr is the
max-min pair in 13/15 cells (PAIR_DOMINANT pass, 87%); the crest
cell (d4096 x cap2048, 11.91pp) is carried by ring10-vs-eti_2020.

Part 2 (this runner): off-grid probes d in {3072, 5120} x cap in
{1536, 2048, 2560} x arms {lcnu, ring10, ssr_10, eti_2020}. Is
11.91pp the crest, or does spread keep rising past d4096?

Bars (pre-registered):
- PRIMARY PAIR_DOMINANT: one pair is the max-min pair in >= 70% of
  the 15 exp700 cells (part 1, recorded: ring10-vs-ssr_10 13/15).
- SECONDARY RIDGE_CREST: all off-grid spreads <= 11.91 + 0.5pp.
  Else RIDGE_RISING.
Gates: repl — the recomputed d4096 x cap2048 leg (all 4 arms) must
match exp700 exactly (exit 4); det double-pass one off-grid cell
(exit 2); consistency within-depth totals equal across caps,
sum == d (exit 3). Exit 0 settled.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

HOME = Path.home()
spec94 = importlib.util.spec_from_file_location(
    "x94", str(HOME / "warp-work/exp694-moei-capmap/run_capmap.py")
)
x94 = importlib.util.module_from_spec(spec94)
sys.modules["x94"] = x94
spec94.loader.exec_module(x94)
spec97 = importlib.util.spec_from_file_location(
    "x97", str(HOME / "warp-work/exp697-moei-earnimmune/run_earnimmune.py")
)
x97 = importlib.util.module_from_spec(spec97)
sys.modules["x97"] = x97
spec97.loader.exec_module(x97)

D = Path(__file__).parent
ARMS = ["lcnu", "ring10", "ssr_10", "eti_2020"]
# d=5120 crashed the sealed generator (burst-draw array capped at
# 4096 = BURSTS axis ceiling; IndexError, disclosed in LEDGER).
# In-range re-scope: d3072 row + full cap cross-section at d4096.
PROBE_D = [3072]
PROBE_CAP = [1536, 2048, 2560]
XSEC_CAP = [1280, 1536, 2560, 3072, 5120]  # cap profile at d4096


def mk_arm(arm, cap):
    if arm.startswith("eti_"):
        af = int(arm[4:6]) / 100.0
        rf = int(arm[6:8]) / 100.0
        gk = {"num_layers": x94.st.rd.L, "prior": 8 / x94.st.rd.E}
        return x97.ETILCNU(cap, annex_frac=af, ring_frac=rf, **gk)
    return x94.mk_arm(arm, cap)


def leg_for(d, caps):
    keys, seg, _ = x94.stream_for(d)
    rid = x97.burst_runs(seg)
    leg, sig0 = {}, None
    for cap in caps:
        cc = {}
        for arm in ARMS:
            m = x97.pass_metrics(mk_arm(arm, cap), keys, seg, rid)
            cc[arm] = {"burst": m["burst"], "rec": m["rec"], "cls": m["cls"]}
        sig = tuple(cc["eti_2020"]["cls"][x][0] for x in ("a", "b", "c"))
        if sig0 is None:
            sig0 = sig
        elif sig != sig0 or sum(sig) != d:
            print(f"CONSISTENCY FAIL d={d} cap={cap}")
            sys.exit(3)
        leg[str(cap)] = cc
    return leg


def spread(cell):
    vals = [cell[a]["burst"] for a in ARMS]
    return (max(vals) - min(vals)) * 100.0


def main():
    t0 = time.time()
    ref = json.loads(
        (
            HOME / "warp-work/exp700-moei-phaseboundary/"
            "exp700-phaseboundary.json"
        ).read_text()
    )
    crest = ref["spread_pp"]["4096x2048"]
    out = {
        "exp": "exp701-ridgedecomp",
        "crest_pp": crest,
        "part1_pair_dominant": {"ring10-vs-ssr_10": "13/15"},
        "probes": {},
    }
    # repl leg: d4096 x cap2048 recomputed must match exp700 exactly
    keys, seg, _ = x94.stream_for(4096)
    rid = x97.burst_runs(seg)
    for arm in ARMS:
        m = x97.pass_metrics(mk_arm(arm, 2048), keys, seg, rid)
        r = ref["grid"]["4096"]["2048"][arm]
        if abs(m["burst"] - r["burst"]) > 1e-12:
            print(f"REPL FAIL d4096 cap2048 {arm}")
            sys.exit(4)
    print(f"repl leg done [{time.time()-t0:.0f}s]", flush=True)
    # off-grid probes: d3072 row + d4096 cap cross-section
    repl_probe = None
    for d in PROBE_D:
        leg = leg_for(d, PROBE_CAP)
        out["probes"][str(d)] = leg
        if repl_probe is None:
            repl_probe = (d, leg)
        print(f"d={d} done [{time.time()-t0:.0f}s]", flush=True)
    xsec = leg_for(4096, XSEC_CAP)
    out["xsec_d4096"] = xsec
    print(f"d=4096 xsec done [{time.time()-t0:.0f}s]", flush=True)
    # det double-pass on first probe cell
    d0 = PROBE_D[0]
    keys, seg, _ = x94.stream_for(d0)
    rid = x97.burst_runs(seg)
    m2 = x97.pass_metrics(mk_arm("eti_2020", 2048), keys, seg, rid)
    r = out["probes"][str(d0)]["2048"]["eti_2020"]
    if m2["burst"] != r["burst"] or m2["cls"] != r["cls"]:
        print("DET FAIL")
        sys.exit(2)
    # adjudicate
    spreads = {
        f"{d}x{c}": round(spread(out["probes"][str(d)][str(c)]), 3)
        for d in PROBE_D
        for c in PROBE_CAP
    }
    spreads.update(
        {f"4096x{c}": round(spread(xsec[str(c)]), 3) for c in XSEC_CAP}
    )
    crest_holds = all(v <= crest + 0.5 for v in spreads.values())
    out.update(
        {
            "verdict": ("RIDGE_CREST" if crest_holds else "RIDGE_RISING"),
            "probe_spread_pp": spreads,
            "det": True,
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    (D / "exp701-ridgedecomp.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", out["verdict"], "| crest:", crest)
    for k, v in spreads.items():
        print(f"  {k}: {v:.2f}")
    g = xsec["2048"]
    print("d4096 c2048:", " ".join(f"{a}:{g[a]['burst']:.4f}" for a in ARMS))
    sys.exit(0)


if __name__ == "__main__":
    main()
