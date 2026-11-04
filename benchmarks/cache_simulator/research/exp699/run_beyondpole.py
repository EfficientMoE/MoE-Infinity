#!/usr/bin/env python3
"""exp699: beyond-pole existence test for the earn-then-immune family.

exp698: eti_2020 beat the ssr pole by +1.57pp burst at equal rec
(cap4096, d4096). exp699 tests whether beyond-pole points (a) survive
a finer annex/ring grid at saturation, and (b) generalize across
burst depths d in {256, 1024, 4096}.

Grid A (fine, cap4096, d4096): ssr_10, ring10, eti_2010, eti_2020,
eti_3020, eti_4020, eti_2030, eti_2040.
Grid B (d-generalization, caps 2048+4096, d in 256/1024/4096):
lcnu, ring10, ssr_10, eti_0510, eti_1010, eti_2020.

Bars (pre-registered):
- PRIMARY BEYOND_POLE_HOLDS: some eti variant with burst >= ssr_10
  burst + 0.005 AND rec >= ssr_10 rec - 0.001 in Grid A AND in Grid
  B's d4096 legs (both caps 2048 and 4096).
- SECONDARY FLAT_GENERALIZES: eti_0510 exchange vs ring <= 1.0pp/pp
  (dr/db >= -1.0 in rate units — the exp698 unit bug's correct form)
  at EVERY d in Grid B, at cap2048 and cap4096.
- Tertiary: argmax burst s.t. rec >= ssr_10 - 0.001 in Grid A
  (annex-optimum location).
Gates: anchor repl == exp694 at d4096 legs (exit 4); det double-pass
eti_2020@cap4096 (exit 2); consistency class counts (exit 3).
Exit 0 settled.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

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
GRID_A = ["ssr_10", "ring10", "eti_2010", "eti_2020", "eti_3020",
          "eti_4020", "eti_2030", "eti_2040"]
GRID_B = ["lcnu", "ring10", "ssr_10", "eti_0510", "eti_1010", "eti_2020"]
DEPTHS = [256, 1024, 4096]


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
    out = {"exp": "exp699-beyondpole", "gridA": {}, "gridB": {}}
    repl_ok, det_ok = True, True

    def run_grid(arms, d, caps, store):
        keys, seg, _ = x94.stream_for(d)
        rid = x97.burst_runs(seg)
        for cap in caps:
            cc = {}
            for arm in arms:
                m = x97.pass_metrics(mk_arm(arm, cap), keys, seg, rid)
                cc[arm] = {"burst": m["burst"], "rec": m["rec"],
                           "cls": m["cls"]}
                if (d in ref_cells and str(cap) in ref_cells[d]
                        and arm in ref_cells[d][str(cap)]):
                    r = ref_cells[d][str(cap)][arm]
                    if (abs(m["burst"] - r["burst"]) > 1e-12
                            or abs(m["rec"] - r["rec"]) > 1e-12):
                        print(f"REPL FAIL d={d} cap={cap} {arm}")
                        repl_ok = False
            store[str(cap)] = cc
            print(f"d={d} cap{cap} done [{time.time()-t0:.0f}s]",
                  flush=True)
        return rid, keys, seg

    # Grid A
    _, _, _ = run_grid(GRID_A, 4096, [4096], out["gridA"])
    # Grid B
    legs = {}
    for d in DEPTHS:
        legs[d] = {}
        run_grid(GRID_B, d, [2048, 4096], legs[d])
    out["gridB"] = {str(d): legs[d] for d in DEPTHS}
    if not repl_ok:
        sys.exit(4)
    # det double-pass on eti_2020 @ cap4096 d4096
    keys, seg, _ = x94.stream_for(4096)
    rid = x97.burst_runs(seg)
    m2 = x97.pass_metrics(mk_arm("eti_2020", 4096), keys, seg, rid)
    r = out["gridA"]["4096"]["eti_2020"]
    if m2["burst"] != r["burst"] or m2["cls"] != r["cls"]:
        det_ok = False
    if not det_ok:
        print("DET FAIL")
        sys.exit(2)
    # consistency: class TOTALS are stream-level constants — identical
    # across caps within a depth leg, and sum to the burst size d.
    # (Cross-depth equality is impossible: totals scale with d. Run 1
    # aborted exit 3 on that impossible gate; corrected v2, disclosed.)
    for d in DEPTHS:
        s2048 = tuple(out["gridB"][str(d)]["2048"]["eti_2020"]["cls"][x][0]
                      for x in ("a", "b", "c"))
        s4096 = tuple(out["gridB"][str(d)]["4096"]["eti_2020"]["cls"][x][0]
                      for x in ("a", "b", "c"))
        if s2048 != s4096 or sum(s2048) != d:
            print(f"CONSISTENCY FAIL d={d}", s2048, s4096)
            sys.exit(3)

    # adjudicate
    def A(arm, f):
        return out["gridA"]["4096"][arm][f]
    beyond_a = [v for v in GRID_A if v.startswith("eti_")
                and A(v, "burst") >= A("ssr_10", "burst") + 0.005
                and A(v, "rec") >= A("ssr_10", "rec") - 0.001]
    beyond_b = [v for v in GRID_B if v.startswith("eti_")
                and all(out["gridB"]["4096"][str(cap)][v]["burst"]
                        >= out["gridB"]["4096"][str(cap)]["ssr_10"]
                        ["burst"] + 0.005
                        and out["gridB"]["4096"][str(cap)][v]["rec"]
                        >= out["gridB"]["4096"][str(cap)]["ssr_10"]
                        ["rec"] - 0.001
                        for cap in (2048, 4096))]
    beyond_holds = bool(beyond_a) and bool(beyond_b)
    flat_gen = True
    for d in DEPTHS:
        for cap in (2048, 4096):
            v = out["gridB"][str(d)][str(cap)]["eti_0510"]
            rg = out["gridB"][str(d)][str(cap)]["ring10"]
            db = v["burst"] - rg["burst"]
            dr = v["rec"] - rg["rec"]
            if not (db > 0.001 and dr / db >= -1.0):
                flat_gen = False
    # annex optimum in Grid A
    elig = [v for v in GRID_A if v.startswith("eti_")
            and A(v, "rec") >= A("ssr_10", "rec") - 0.001]
    opt = max(elig, key=lambda v: A(v, "burst")) if elig else None
    verdict = ("BEYOND_POLE_HOLDS" if beyond_holds else
               "FLAT_ONLY" if flat_gen else "NEITHER")
    out.update({"verdict": verdict, "beyond_a": beyond_a,
                "beyond_b": beyond_b, "flat_generalizes": flat_gen,
                "annex_optimum": opt, "det": True,
                "elapsed_s": round(time.time() - t0, 1)})
    (D / "exp699-beyondpole.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", verdict, "| beyond_a:", beyond_a, "| beyond_b:",
          beyond_b, "| flat_gen:", flat_gen, "| opt:", opt)
    print("A@4096:", " ".join(f"{v}:{A(v,'burst'):.4f}/{A(v,'rec'):.4f}"
                              for v in GRID_A))
    for d in DEPTHS:
        g = out["gridB"][str(d)]
        print(f"B d{d} c4096:", " ".join(
            f"{v}:{g['4096'][v]['burst']:.4f}/{g['4096'][v]['rec']:.4f}"
            for v in GRID_B))
    sys.exit(0)


if __name__ == "__main__":
    main()
