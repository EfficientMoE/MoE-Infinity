#!/usr/bin/env python3
"""exp700: phase-boundary sweep — where do burst policies diverge?

exp699: all arms converge at d256/d1024; divergence at d4096.
exp695: burst traffic is 99.9% carry-in re-accesses of the base
crowd. Model: policy spread scales with burst depth d and capacity
headroom (cap vs working set). Sweep the surface.

Arms: lcnu, ring10, ssr_10, eti_2020.
Depths: 256, 512, 1024, 2048, 4096. Caps: 1024, 2048, 4096.

Bars (pre-registered):
- PRIMARY: (d, cap) matrix of max-min burst spread across arms;
  log-log least-squares fit spread ~ a + b1*log(d) + b2*log(cap);
  R^2 >= 0.8 => MODEL_HOLDS.
- 0.5pp contour monotone in d at fixed cap => MONOTONE_BOUNDARY.
- Verdicts: MODEL_HOLDS+MONOTONE_BOUNDARY > MODEL_HOLDS >
  STRUCTURE_UNEXPLAINED.
Gates: REPL anchors == exp694 at d4096 (caps 1024/2048/4096) and ==
exp699 grid B at d1024/d256 cap4096 (exit 4); DET double-pass one
cell (eti_2020 @ d2048 cap2048) (exit 2); consistency: within-depth
class totals equal across caps, sum == d (gate v2 form) (exit 3).
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
DEPTHS = [256, 512, 1024, 2048, 4096]
CAPS = [1024, 2048, 4096]


def mk_arm(arm, cap):
    if arm.startswith("eti_"):
        af = int(arm[4:6]) / 100.0
        rf = int(arm[6:8]) / 100.0
        gk = {"num_layers": x94.st.rd.L, "prior": 8 / x94.st.rd.E}
        return x97.ETILCNU(cap, annex_frac=af, ring_frac=rf, **gk)
    return x94.mk_arm(arm, cap)


def main():
    t0 = time.time()
    ref94 = json.loads(
        (
            HOME / "warp-work/exp694-moei-capmap/" "exp694-capmap.json"
        ).read_text()
    )
    ref94c = {c["d"]: c["caps"] for c in ref94["map_grid"]}
    ref99 = json.loads(
        (
            HOME / "warp-work/exp699-moei-beyondpole/" "exp699-beyondpole.json"
        ).read_text()
    )
    out = {"exp": "exp700-phaseboundary", "grid": {}}
    repl_ok = True
    for d in DEPTHS:
        keys, seg, _ = x94.stream_for(d)
        rid = x97.burst_runs(seg)
        leg = {}
        totals_sig = None
        for cap in CAPS:
            cc = {}
            for arm in ARMS:
                m = x97.pass_metrics(mk_arm(arm, cap), keys, seg, rid)
                cc[arm] = {
                    "burst": m["burst"],
                    "rec": m["rec"],
                    "cls": m["cls"],
                }
                if (
                    d == 4096
                    and str(cap) in ref94c[4096]
                    and arm in ref94c[4096][str(cap)]
                ):
                    r = ref94c[4096][str(cap)][arm]
                    if (
                        abs(m["burst"] - r["burst"]) > 1e-12
                        or abs(m["rec"] - r["rec"]) > 1e-12
                    ):
                        print(f"REPL94 FAIL cap={cap} {arm}")
                        repl_ok = False
                if (
                    d in (256, 1024)
                    and cap == 4096
                    and arm in ref99["gridB"][str(d)]["4096"]
                ):
                    r = ref99["gridB"][str(d)]["4096"][arm]
                    if abs(m["burst"] - r["burst"]) > 1e-12:
                        print(f"REPL99 FAIL d={d} {arm}")
                        repl_ok = False
            sig = tuple(cc["eti_2020"]["cls"][x][0] for x in ("a", "b", "c"))
            if totals_sig is None:
                totals_sig = sig
            elif sig != totals_sig or sum(sig) != d:
                print(f"CONSISTENCY FAIL d={d} cap={cap}")
                sys.exit(3)
            leg[str(cap)] = cc
        out["grid"][str(d)] = leg
        print(f"d={d} done [{time.time()-t0:.0f}s]", flush=True)
    if not repl_ok:
        sys.exit(4)
    # det double-pass
    keys, seg, _ = x94.stream_for(2048)
    rid = x97.burst_runs(seg)
    m2 = x97.pass_metrics(mk_arm("eti_2020", 2048), keys, seg, rid)
    r = out["grid"]["2048"]["2048"]["eti_2020"]
    if m2["burst"] != r["burst"] or m2["cls"] != r["cls"]:
        print("DET FAIL")
        sys.exit(2)
    # adjudicate: spread matrix + log-log fit
    spread = {}
    rows = []
    for d in DEPTHS:
        for cap in CAPS:
            g = out["grid"][str(d)][str(cap)]
            vals = [g[a]["burst"] for a in ARMS]
            sp = (max(vals) - min(vals)) * 100.0  # pp
            spread[f"{d}x{cap}"] = round(sp, 3)
            rows.append((np.log(d), np.log(cap), np.log(max(sp, 1e-4))))
    X = np.array([(1.0, r0, c0) for r0, c0, _ in rows])
    y = np.array([v for _, _, v in rows])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    model_holds = r2 >= 0.8
    # monotone boundary: for each cap, the set of d with spread>0.5
    # must be an upper set (once crossed, stays crossed)
    monotone = True
    for cap in CAPS:
        crossed = [spread[f"{d}x{cap}"] > 0.5 for d in DEPTHS]
        if any(
            crossed[i] and not crossed[i + 1] for i in range(len(crossed) - 1)
        ):
            monotone = False
    verdict = (
        "MODEL_HOLDS+MONOTONE_BOUNDARY"
        if model_holds and monotone
        else "MODEL_HOLDS"
        if model_holds
        else "STRUCTURE_UNEXPLAINED"
    )
    out.update(
        {
            "verdict": verdict,
            "spread_pp": spread,
            "r2": round(r2, 4),
            "beta": [round(b, 4) for b in beta],
            "monotone_boundary": monotone,
            "det": True,
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    (D / "exp700-phaseboundary.json").write_text(json.dumps(out, indent=1))
    print(
        "VERDICT:",
        verdict,
        "| R2 =",
        round(r2, 4),
        "| beta(logd,logcap) =",
        [round(b, 3) for b in beta[1:]],
        "| monotone:",
        monotone,
    )
    hdr = "d\\cap " + " ".join(f"{c:7d}" for c in CAPS)
    print(hdr)
    for d in DEPTHS:
        print(f"{d:5d} " + " ".join(f"{spread[f'{d}x{c}']:7.2f}" for c in CAPS))
    sys.exit(0)


if __name__ == "__main__":
    main()
