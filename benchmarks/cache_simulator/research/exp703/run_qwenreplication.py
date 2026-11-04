#!/usr/bin/env python3
"""exp703: qwen-30b replication of exp702 (carry-in, front-loading, LCNU-inversion).

Two frozen contrast streams (exp656 assets): hi (high Gini), lo
(low Gini). Bars/gates pre-registered in LEDGER.md before freeze.
Exit codes: 0 settled | 2 det | 3 structure | 5 anchor.
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
    "x85", str(HOME / "warp-work/exp685-moei-ring/run_ring.py"))
x85 = importlib.util.module_from_spec(spec85)
sys.modules["x85"] = x85
spec85.loader.exec_module(x85)
spec93 = importlib.util.spec_from_file_location(
    "x93", str(HOME / "warp-work/exp693-moei-ssr/run_ssr.py"))
x93 = importlib.util.module_from_spec(spec93)
sys.modules["x93"] = x93
spec93.loader.exec_module(x93)
spec97 = importlib.util.spec_from_file_location(
    "x97", str(HOME / "warp-work/exp697-moei-earnimmune/run_earnimmune.py"))
x97 = importlib.util.module_from_spec(spec97)
sys.modules["x97"] = x97
spec97.loader.exec_module(x97)

from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

D = Path(__file__).parent
W, CAPS = 4096, [1024, 2048]
ARMS = ["lcnu", "ring10", "ssr_10", "eti_2020"]
STREAMS = {
    "hi": {
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_hi.pt",
        "N": 786432, "distinct": 5097, "prompts": [512, 512, 512, 512],
        "n_win": 192, "burst_n": 20, "anchor": 0.704076,
    },
    "lo": {
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_lo.pt",
        "N": 638208, "distinct": 5842, "prompts": [512, 512, 126, 512],
        "n_win": 155, "burst_n": 16, "anchor": 0.624469,
    },
}
NL = 48
BAND = 0.005


def load_stream(pt):
    s = torch.load(str(pt), map_location="cpu", weights_only=False)
    keys, session, per_prompt, starts = [], -1, [], []
    for r in s["records"]:
        t = r["topk_ids_of_layer"]
        if r["forward_mode"] == 1:
            session += 1
            per_prompt.append(0)
            starts.append(len(keys))
        else:
            per_prompt[session] += 1
            for l in range(t.shape[0]):
                for e in t[l, 0].tolist():
                    keys.append((l, e))
    return keys, per_prompt, starts


def mk_arm(name, cap):
    gk = {"num_layers": NL, "prior": 8 / 128}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "ring10":
        return x85.RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "ssr_10":
        return x93.SecondSightLCNU(cap, annex_frac=0.10, **gk)
    if name == "eti_2020":
        return x97.ETILCNU(cap, annex_frac=0.20, ring_frac=0.20, **gk)
    raise KeyError(name)


def run_stream(tag, cfg):
    t0 = time.time()
    keys, per_prompt, starts = load_stream(cfg["pt"])
    n = len(keys)
    n_win = n // W
    if not (n == cfg["N"] and len(set(keys)) == cfg["distinct"]
            and per_prompt == cfg["prompts"] and n_win == cfg["n_win"]):
        print(f"STRUCTURE FAIL {tag}", n, len(set(keys)), per_prompt, n_win)
        sys.exit(3)
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

    cls = {w: classify(w) for w in range(1, n_win)}
    nov = {w: cls[w][0] / W for w in cls}
    order = sorted(nov, key=lambda w: -nov[w])
    burst = set(order[:cfg["burst_n"]])
    c_burst = float(np.mean([cls[w][2] / W for w in burst]))
    # S1: window index reaching 90% of all first-sights
    total_a = sum(cls[w][0] for w in cls)
    cum, w90 = 0, None
    for w in sorted(cls):
        cum += cls[w][0]
        if cum >= 0.9 * total_a:
            w90 = w
            break
    print(f"{tag}: c_burst={c_burst:.3f} w90={w90}/{n_win} [{time.time()-t0:.0f}s]",
          flush=True)

    start_wins = {i // W for i in starts}

    def policy_cell(arm, cap):
        pol = mk_arm(arm, cap)
        h = np.zeros(n, dtype=bool)
        for i, k in enumerate(keys):
            h[i] = pol.on_access(k)
        wh = h[: n_win * W].reshape(n_win, W)
        return {"whole": float(h.mean()),
                "burst": float(wh[sorted(burst)].mean())}

    grid = {str(cap): {arm: policy_cell(arm, cap) for arm in ARMS}
            for cap in CAPS}
    print(f"{tag}: policies done [{time.time()-t0:.0f}s]", flush=True)
    if classify(1) != cls[1]:
        print(f"DET FAIL classifier {tag}")
        sys.exit(2)
    det_cell = policy_cell("lcnu", 1024)
    ref_cell = grid["1024"]["lcnu"]
    if det_cell != ref_cell:
        print(f"DET FAIL policy {tag}")
        sys.exit(2)
    anchor_ok = abs(grid["1024"]["lcnu"]["whole"] - cfg["anchor"]) <= BAND

    spread = {}
    for cap in CAPS:
        bvals = [grid[str(cap)][a]["burst"] for a in ARMS]
        spread[str(cap)] = round((max(bvals) - min(bvals)) * 100, 3)
    lcnu_best = all(
        grid[str(cap)]["lcnu"]["burst"]
        == max(grid[str(cap)][a]["burst"] for a in ARMS)
        for cap in CAPS)
    return {
        "N": n, "windows": n_win, "total_first_sights": total_a,
        "burst_windows": sorted(burst),
        "burst_with_session_start": sorted(burst & start_wins),
        "session_start_windows": sorted(start_wins),
        "c_share_burst": c_burst,
        "w90_index": w90,
        "H1": c_burst >= 0.80,
        "S1_FRONT_LOADED": w90 is not None and w90 <= n_win // 2,
        "S2_LCNU_BEST": bool(lcnu_best),
        "S3_spread_pp": spread,
        "policy_grid": grid,
        "anchor": {"lcnu1024_whole": grid["1024"]["lcnu"]["whole"],
                   "ref": cfg["anchor"], "ok": bool(anchor_ok)},
        "elapsed_s": round(time.time() - t0, 1),
    }, anchor_ok


def main():
    t0 = time.time()
    out = {"exp": "exp703-qwenreplication", "streams": {}}
    anchors_ok = True
    for tag, cfg in STREAMS.items():
        res, ok = run_stream(tag, cfg)
        out["streams"][tag] = res
        anchors_ok = anchors_ok and ok
        print(f"{tag} settled [{time.time()-t0:.0f}s]", flush=True)
    h1 = [out["streams"][t]["H1"] for t in STREAMS]
    out["verdict"] = ("REPLICATED" if all(h1)
                      else "PARTIAL" if any(h1) else "REFUTED")
    out["anchors_ok"] = anchors_ok
    out["total_elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp703-qwenreplication.json").write_text(json.dumps(out, indent=1))
    print("VERDICT:", out["verdict"])
    for t in STREAMS:
        s = out["streams"][t]
        print(f"  {t}: c={s['c_share_burst']:.3f} w90={s['w90_index']}"
              f"/{s['windows']} LCNU_BEST={s['S2_LCNU_BEST']}"
              f" spread={s['S3_spread_pp']}")
    if not anchors_ok:
        sys.exit(5)
    sys.exit(0)


if __name__ == "__main__":
    main()
