#!/usr/bin/env python3
"""exp704: real-trace capacity fine-map (spread-collapse curves, 3 traces).

Frozen traces: granite (exp650 asset), qwen30b hi/lo (exp656
assets). Bars/gates pre-registered in LEDGER.md before freeze.
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

mods = {}
for nm, p in [
    ("x85", "exp685-moei-ring/run_ring.py"),
    ("x93", "exp693-moei-ssr/run_ssr.py"),
    ("x97", "exp697-moei-earnimmune/run_earnimmune.py"),
]:
    s = importlib.util.spec_from_file_location(nm, str(HOME / "warp-work" / p))
    m = importlib.util.module_from_spec(s)
    sys.modules[nm] = m
    s.loader.exec_module(m)
    mods[nm] = m

from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

D = Path(__file__).parent
W = 4096
ARMS = ["lcnu", "ring10", "ssr_10", "eti_2020"]
TRACES = {
    "granite": {
        "pt": HOME / "warp-work/exp650-moei-granite3b-trace-pr/stream_granite3b.pt",
        "nl": 32, "prior": 0.2,
        "N": 638208, "distinct": 1262, "prompts": [313, 512, 336, 245, 169, 197, 512, 209],
        "n_win": 155, "burst_n": 16,
        "caps": [128, 192, 256, 384, 512, 640, 768, 896, 1024],
        "anchors": {128: 0.283762, 256: 0.468164, 512: 0.728697,
                    768: 0.886415, 1024: 0.970839},
        "det_cell": ("lcnu", 256),
    },
    "hi": {
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_hi.pt",
        "nl": 48, "prior": 8 / 128,
        "N": 786432, "distinct": 5097, "prompts": [512, 512, 512, 512],
        "n_win": 192, "burst_n": 20,
        "caps": [768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048],
        "anchors": {896: 0.665581, 1024: 0.704076, 1152: 0.738525,
                    1280: 0.770724},
        "det_cell": ("lcnu", 1024),
    },
    "lo": {
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_lo.pt",
        "nl": 48, "prior": 8 / 128,
        "N": 638208, "distinct": 5842, "prompts": [512, 512, 126, 512],
        "n_win": 155, "burst_n": 16,
        "caps": [768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048],
        "anchors": {896: 0.590768, 1024: 0.624469, 1152: 0.656018,
                    1280: 0.686120},
        "det_cell": ("lcnu", 1024),
    },
}
BAND = 0.005


def load_stream(pt):
    s = torch.load(str(pt), map_location="cpu", weights_only=False)
    keys, session, per_prompt = [], -1, []
    for r in s["records"]:
        t = r["topk_ids_of_layer"]
        if r["forward_mode"] == 1:
            session += 1
            per_prompt.append(0)
        else:
            per_prompt[session] += 1
            for l in range(t.shape[0]):
                for e in t[l, 0].tolist():
                    keys.append((l, e))
    return keys, per_prompt


def mk_arm(name, cap, nl, prior):
    gk = {"num_layers": nl, "prior": prior}
    if name == "lcnu":
        return build_policy("lcnu", cap, **gk)
    if name == "ring10":
        return mods["x85"].RingProbationLCNU(cap, ring_frac=0.10, **gk)
    if name == "ssr_10":
        return mods["x93"].SecondSightLCNU(cap, annex_frac=0.10, **gk)
    if name == "eti_2020":
        return mods["x97"].ETILCNU(cap, annex_frac=0.20, ring_frac=0.20, **gk)
    raise KeyError(name)


def run_trace(tag, cfg):
    t0 = time.time()
    keys, per_prompt = load_stream(cfg["pt"])
    n = len(keys)
    n_win = n // W
    if not (n == cfg["N"] and len(set(keys)) == cfg["distinct"]
            and per_prompt == cfg["prompts"] and n_win == cfg["n_win"]):
        print(f"STRUCTURE FAIL {tag}", n, len(set(keys)), per_prompt, n_win,
              flush=True)
        sys.exit(3)
    first_seen = {}
    for i, k in enumerate(keys):
        if k not in first_seen:
            first_seen[k] = i
    nov = {}
    for w in range(1, n_win):
        ws = w * W
        nov[w] = sum(1 for i in range(ws, ws + W) if first_seen[keys[i]] == i)
    order = sorted(nov, key=lambda w: -nov[w])
    burst = sorted(order[: cfg["burst_n"]])
    print(f"{tag}: stream ok, burst={burst[:6]}... [{time.time()-t0:.0f}s]",
          flush=True)

    def policy_cell(arm, cap):
        pol = mk_arm(arm, cap, cfg["nl"], cfg["prior"])
        h = np.zeros(n, dtype=bool)
        for i, k in enumerate(keys):
            h[i] = pol.on_access(k)
        wh = h[: n_win * W].reshape(n_win, W)
        return {"whole": float(h.mean()),
                "burst": float(wh[burst].mean())}

    grid = {}
    for cap in cfg["caps"]:
        grid[str(cap)] = {arm: policy_cell(arm, cap) for arm in ARMS}
        print(f"{tag}: cap {cap} done [{time.time()-t0:.0f}s]", flush=True)

    spread = {str(cap): round(
        (max(grid[str(cap)][a]["burst"] for a in ARMS)
         - min(grid[str(cap)][a]["burst"] for a in ARMS)) * 100, 3)
        for cap in cfg["caps"]}
    ring_never = all(
        grid[str(cap)]["ring10"]["burst"] < grid[str(cap)]["lcnu"]["burst"]
        for cap in cfg["caps"])
    anchors = {str(cap): {
        "mine": grid[str(cap)]["lcnu"]["whole"], "ref": ref,
        "ok": bool(abs(grid[str(cap)]["lcnu"]["whole"] - ref) <= BAND)}
        for cap, ref in cfg["anchors"].items()}

    da, dc = cfg["det_cell"]
    if policy_cell(da, dc) != grid[str(dc)][da]:
        print(f"DET FAIL {tag} {da}@{dc}", flush=True)
        sys.exit(2)

    caps_s = [str(c) for c in cfg["caps"]]
    cross1 = next((c for c in caps_s if spread[c] < 1.0), None)
    return {
        "burst_windows": burst,
        "spread_pp": spread,
        "policy_grid": grid,
        "B1_DECAY": spread[caps_s[-1]] < spread[caps_s[0]],
        "B2_CROSS1": cross1 if cross1 is not None else "NOT_REACHED",
        "B3_RING_NEVER_WINS": bool(ring_never),
        "anchors": anchors,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    t0 = time.time()
    out = {"exp": "exp704-capacitymap", "traces": {}}
    anchor_fail = False
    for tag, cfg in TRACES.items():
        res = run_trace(tag, cfg)
        out["traces"][tag] = res
        anchor_fail = anchor_fail or any(
            not a["ok"] for a in res["anchors"].values())
        print(f"{tag} settled B1={res['B1_DECAY']} "
              f"CROSS1={res['B2_CROSS1']} RING={res['B3_RING_NEVER_WINS']} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    out["verdict"] = "CAPACITY_MAP"
    out["B3_all"] = all(t["B3_RING_NEVER_WINS"] for t in out["traces"].values())
    out["B1_all"] = all(t["B1_DECAY"] for t in out["traces"].values())
    out["total_elapsed_s"] = round(time.time() - t0, 1)
    (D / "exp704-capacitymap.json").write_text(json.dumps(out, indent=1))
    print("VERDICT: CAPACITY_MAP B1_all=", out["B1_all"],
          "B3_all=", out["B3_all"], flush=True)
    for t in out["traces"]:
        print(f"  {t}: spread={out['traces'][t]['spread_pp']}")
    if anchor_fail:
        sys.exit(5)
    sys.exit(0)


if __name__ == "__main__":
    main()
