#!/usr/bin/env python3
"""exp702: real-trace carry-in decomposition — the bridge experiment.

Replays the frozen real granite-3b decode trace (exp650 asset) and
asks whether the synthetic chain's central mechanism — burst
traffic is carry-in re-accesses (exp695: 99.9%) — exists in real
traffic, and whether the policy arms it motivates (ring/ssr/ETI)
actually diverge on real burst windows.

Bars/gates pre-registered in LEDGER.md before freeze. Exit codes:
0 settled | 2 det | 3 structure | 5 exp650 band.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HOME = Path.home()
sys.path.insert(0, str(HOME / "warp-work/moei-granite3b"))

spec85 = importlib.util.spec_from_file_location(
    "x85", str(HOME / "warp-work/exp685-moei-ring/run_ring.py")
)
x85 = importlib.util.module_from_spec(spec85)
sys.modules["x85"] = x85
spec85.loader.exec_module(x85)
spec93 = importlib.util.spec_from_file_location(
    "x93", str(HOME / "warp-work/exp693-moei-ssr/run_ssr.py")
)
x93 = importlib.util.module_from_spec(spec93)
sys.modules["x93"] = x93
spec93.loader.exec_module(x93)
spec97 = importlib.util.spec_from_file_location(
    "x97", str(HOME / "warp-work/exp697-moei-earnimmune/run_earnimmune.py")
)
x97 = importlib.util.module_from_spec(spec97)
sys.modules["x97"] = x97
spec97.loader.exec_module(x97)

from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

D = Path(__file__).parent
PT = HOME / "warp-work/exp650-moei-granite3b-trace-pr/stream_granite3b.pt"
CONV = HOME / "warp-work/exp650-moei-granite3b-trace-pr/convert-report.json"
NL, W = 32, 4096
CAPS = [256, 512]
ARMS = ["lcnu", "ring10", "ssr_10", "eti_2020"]
BURST_N = 16
EXP650_LCNU_256 = 0.468164
BAND = 0.005
EXPECT_PROMPTS = [313, 512, 336, 245, 169, 197, 512, 209]


def load_stream():
    s = torch.load(str(PT), map_location="cpu", weights_only=False)
    keys, session, per_prompt, starts = [], -1, [], []
    for r in s["records"]:
        t = r["topk_ids_of_layer"]
        if r["forward_mode"] == 1:
            session += 1
            per_prompt.append(0)
            starts.append(len(keys))
        else:
            per_prompt[session] += 1
            for l in range(NL):
                for e in t[l, 0].tolist():
                    keys.append((l, e))
    return keys, per_prompt, starts


def mk_arm(name, cap):
    gk = {"num_layers": NL, "prior": 8 / 40}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "ring10":
        return x85.RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "ssr_10":
        return x93.SecondSightLCNU(cap, annex_frac=0.10, **gk)
    if name == "eti_2020":
        return x97.ETILCNU(cap, annex_frac=0.20, ring_frac=0.20, **gk)
    raise KeyError(name)


def main():
    t0 = time.time()
    keys, per_prompt, starts = load_stream()
    n = len(keys)
    n_win = n // W
    if not (
        n == 638208
        and len(set(keys)) == 1262
        and per_prompt == EXPECT_PROMPTS
        and n_win == 155
    ):
        print("STRUCTURE FAIL", n, len(set(keys)), per_prompt, n_win)
        sys.exit(3)
    conv = json.loads(CONV.read_text())
    if [p["decode_steps"] for p in conv["prompts"]] != per_prompt:
        print("STRUCTURE FAIL convert-report mismatch")
        sys.exit(3)
    print(f"stream ok [{time.time()-t0:.0f}s]", flush=True)

    first_seen = {}
    for i, k in enumerate(keys):
        if k not in first_seen:
            first_seen[k] = i

    def classify(w):
        a = b = c = 0
        ws = w * W
        for i in range(ws, ws + W):
            fs = first_seen[keys[i]]
            if fs == i:
                a += 1
            elif fs >= ws:
                b += 1
            else:
                c += 1
        return a, b, c

    cls = {w: classify(w) for w in range(1, n_win)}  # window 0 excluded
    nov = {w: cls[w][0] / W for w in cls}
    order = sorted(nov, key=lambda w: -nov[w])
    burst = set(order[:BURST_N])
    med = sorted(nov.values())[len(nov) // 2]
    base = {w for w in nov if nov[w] < med}
    c_burst = np.mean([cls[w][2] / W for w in burst])
    a_burst = np.mean([cls[w][0] / W for w in burst])
    a_base = np.mean([cls[w][0] / W for w in base])
    print(
        f"classifier ok; c_burst={c_burst:.3f} a_burst={a_burst:.3f} "
        f"a_base={a_base:.3f} [{time.time()-t0:.0f}s]",
        flush=True,
    )

    start_wins = {i // W for i in starts}

    def policy_cell(arm, cap):
        pol = mk_arm(arm, cap)
        h = np.zeros(n, dtype=bool)
        for i, k in enumerate(keys):
            h[i] = pol.on_access(k)
        wh = h[: n_win * W].reshape(n_win, W)
        bh = wh[sorted(burst)].mean()
        sh = wh[sorted(base)].mean()
        return {"whole": float(h.mean()), "burst": float(bh), "base": float(sh)}

    grid = {}
    for cap in CAPS:
        grid[str(cap)] = {arm: policy_cell(arm, cap) for arm in ARMS}
        print(f"cap {cap} done [{time.time()-t0:.0f}s]", flush=True)

    # det: classifier window 1 + LCNU@256 rerun byte-identical
    if classify(1) != cls[1]:
        print("DET FAIL classifier")
        sys.exit(2)
    det_cell = policy_cell("lcnu", 256)
    ref_cell = grid["256"]["lcnu"]
    if (
        det_cell["whole"] != ref_cell["whole"]
        or det_cell["burst"] != ref_cell["burst"]
    ):
        print("DET FAIL policy rerun")
        sys.exit(2)
    print(f"det ok [{time.time()-t0:.0f}s]", flush=True)

    band_ok = abs(grid["256"]["lcnu"]["whole"] - EXP650_LCNU_256) <= BAND

    h1 = c_burst >= 0.80
    h2 = a_burst >= 5 * a_base
    spread = {}
    h3 = False
    for cap in CAPS:
        bvals = [grid[str(cap)][a]["burst"] for a in ARMS]
        spread[str(cap)] = round((max(bvals) - min(bvals)) * 100, 3)
        if spread[str(cap)] >= 1.0:
            h3 = True

    res = {
        "exp": "exp702-realcarryin",
        "N": n,
        "distinct": 1262,
        "windows": n_win,
        "burst_windows": sorted(burst),
        "burst_with_session_start": sorted(burst & start_wins),
        "session_start_windows": sorted(start_wins),
        "median_novelty": med,
        "classifier": {
            "c_share_burst": float(c_burst),
            "a_share_burst": float(a_burst),
            "a_share_base": float(a_base),
            "per_window": {str(w): cls[w] for w in sorted(cls)},
        },
        "H1_REAL_CARRYIN": bool(h1),
        "H2_NOVELTY_SPIKES": bool(h2),
        "H3_POLICY_SPREAD": bool(h3),
        "spread_pp": spread,
        "policy_grid": grid,
        "exp650_band": {
            "lcnu256_whole": grid["256"]["lcnu"]["whole"],
            "ref": EXP650_LCNU_256,
            "ok": bool(band_ok),
        },
        "det": True,
        "elapsed_s": round(time.time() - t0, 1),
    }
    res["verdict"] = "REAL_CARRYIN" if h1 else "NO_CARRYIN"
    (D / "exp702-realcarryin.json").write_text(json.dumps(res, indent=1))
    print(
        "VERDICT:",
        res["verdict"],
        "| H1 c_burst=%.3f" % c_burst,
        "| H2",
        res["H2_NOVELTY_SPIKES"],
        "| H3",
        res["H3_POLICY_SPREAD"],
        spread,
    )
    print(
        "band ok:",
        band_ok,
        "| lcnu256 whole:",
        round(grid["256"]["lcnu"]["whole"], 6),
    )
    print("burst windows:", res["burst_windows"])
    print("burst w/ session start:", res["burst_with_session_start"])
    if not band_ok:
        sys.exit(5)
    sys.exit(0)


if __name__ == "__main__":
    main()
