#!/usr/bin/env python3
"""exp709: beyond-law ceiling probe on the remaining 2 traces (hi + granite).

Completes exp708 (lo: BEYOND_FLAT) to 3/3 traces: does best-vs-worst policy
convergence persist from CROSS1 up to cap == WS20 on qwen30b-hi and granite?
Mirrors exp708's sealed run_beyondlaw.py exactly (arms, loader, novelty,
burst windows, spread definition); per-trace CFG from sealed exp704.
Anchors parsed at runtime from sealed exp704/exp705 JSON and pinned (PREREG.md).
Exit codes: 0 settled | 2 det | 3 structure | 5 anchor.
--smoke: synthetic mini-trace through the same machinery (no result JSON).
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
SEALED704 = HOME / "warp-work/exp704-moei-capacitymap/exp704-capacitymap.json"
SEALED705 = (
    HOME
    / "warp-work/moei-granite3b-rebuilt/benchmarks/cache_simulator"
    / "research/exp705/exp705-costcurves.json"
)

TRACES = {
    "hi": {
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_hi.pt",
        "nl": 48,
        "prior": 8 / 128,
        "N": 786432,
        "distinct": 5097,
        "prompts": [512, 512, 512, 512],
        "n_win": 192,
        "burst_n": 20,
        "caps": [2048, 2304, 2560, 2816, 3072, 3396],
        "anchor_cap": "2048",
        "pin_sp": {"2048": 0.029},
        "pin_lcnu_whole": {"2048": 0.9110196431477865},
        "pin_ring_whole": {"2048": 0.9100151062011719},
        "pin_ws20": 3396,
        "det_cell": ("lcnu", 2304),
    },
    "granite": {
        "pt": HOME
        / "warp-work/exp650-moei-granite3b-trace-pr/stream_granite3b.pt",
        "nl": 32,
        "prior": 0.2,
        "N": 638208,
        "distinct": 1262,
        "prompts": [313, 512, 336, 245, 169, 197, 512, 209],
        "n_win": 155,
        "burst_n": 16,
        "caps": [1024, 1088, 1152, 1228],
        "anchor_cap": "1024",
        "pin_sp": {"1024": 0.16},
        "pin_lcnu_whole": {"1024": 0.9708386607501003},
        "pin_ring_whole": {"1024": 0.9708355269755314},
        "pin_ws20": 1228,
        "det_cell": ("lcnu", 1088),
    },
}
ORDER = ["hi", "granite"]
FLAT_PP = 1.0  # C2 threshold (identical to exp708 B2)
WS20_CELL_PP = 0.1  # C3 report-only threshold (identical to exp708 B3)


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


def burst_windows(keys, n_win, burst_n):
    first_seen = {}
    for i, k in enumerate(keys):
        if k not in first_seen:
            first_seen[k] = i
    nov = {}
    for w in range(1, n_win):
        ws = w * W
        nov[w] = sum(1 for i in range(ws, ws + W) if first_seen[keys[i]] == i)
    order = sorted(nov, key=lambda w: -nov[w])
    return sorted(order[:burst_n])


def policy_cell(arm, cap, keys, n_win, burst, nl, prior):
    pol = mk_arm(arm, cap, nl, prior)
    h = np.zeros(len(keys), dtype=bool)
    for i, k in enumerate(keys):
        h[i] = pol.on_access(k)
    wh = h[: n_win * W].reshape(n_win, W)
    return {"whole": float(h.mean()), "burst": float(wh[burst].mean())}


def check_structure(tag, cfg, keys, per_prompt):
    n = len(keys)
    n_win = n // W
    if not (
        n == cfg["N"]
        and len(set(keys)) == cfg["distinct"]
        and per_prompt == cfg["prompts"]
        and n_win == cfg["n_win"]
    ):
        print(
            f"STRUCTURE FAIL {tag}",
            n,
            len(set(keys)),
            per_prompt,
            n_win,
            flush=True,
        )
        sys.exit(3)


def run_grid(tag, cfg, keys, t0):
    n_win = cfg["n_win"]
    burst = burst_windows(keys, n_win, cfg["burst_n"])
    print(
        f"{tag}: stream ok, burst={burst[:6]}... [{time.time()-t0:.0f}s]",
        flush=True,
    )
    grid = {}
    for cap in cfg["caps"]:
        grid[str(cap)] = {
            arm: policy_cell(
                arm, cap, keys, n_win, burst, cfg["nl"], cfg["prior"]
            )
            for arm in ARMS
        }
        print(f"{tag}: cap {cap} done [{time.time()-t0:.0f}s]", flush=True)
    return grid, burst


def spread_of(grid, caps):
    return {
        str(cap): round(
            (
                max(grid[str(cap)][a]["burst"] for a in ARMS)
                - min(grid[str(cap)][a]["burst"] for a in ARMS)
            )
            * 100,
            3,
        )
        for cap in caps
    }


def smoke():
    # Disclosed smoke: synthetic mini-trace through the SAME machinery
    # (records, loader, structure check, novelty, burst, policy_cell, spread).
    # No result JSON.
    t0 = time.time()
    L, K, POOL = 4, 2, 24
    rng = np.random.default_rng(7)
    records = []
    for p in range(3):
        records.append(
            {
                "forward_mode": 1,
                "topk_ids_of_layer": torch.tensor(
                    rng.integers(0, POOL, (L, 8, K)), dtype=torch.int32
                ),
            }
        )
        for _ in range(1600 // 3 + 1):
            records.append(
                {
                    "forward_mode": 0,
                    "topk_ids_of_layer": torch.tensor(
                        rng.integers(0, POOL, (L, 1, K)), dtype=torch.int32
                    ),
                }
            )
    smoke_pt = D / "smoke_stream.pt"
    torch.save({"records": records[: 1 + 3 * (1600 // 3 + 1)]}, str(smoke_pt))
    keys, per_prompt = load_stream(smoke_pt)
    scfg = {
        "pt": smoke_pt,
        "nl": L,
        "prior": 0.5,
        "N": len(keys),
        "distinct": len(set(keys)),
        "prompts": per_prompt,
        "n_win": len(keys) // W,
        "burst_n": 2,
        "caps": [16, 32],
    }
    assert scfg["n_win"] == 3, f"smoke n_win {scfg['n_win']}"
    check_structure("smoke", scfg, keys, per_prompt)
    grid, burst = run_grid("smoke", scfg, keys, t0)
    sp = spread_of(grid, scfg["caps"])
    keys2, _ = load_stream(smoke_pt)
    det_ok = (
        policy_cell("lcnu", 16, keys2, scfg["n_win"], burst, L, 0.5)
        == grid["16"]["lcnu"]
    )
    sane = all(
        0.0 <= c["whole"] <= 1.0 and 0.0 <= c["burst"] <= 1.0
        for cell in grid.values()
        for c in cell.values()
    )
    sane = sane and all(0.0 <= v <= 100.0 for v in sp.values())
    smoke_pt.unlink()
    ok = det_ok and sane
    print(
        f"SMOKE det={det_ok} sane={sane} N={len(keys)} "
        f"distinct={scfg['distinct']} spread={sp}"
    )
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 2


def main():
    if "--smoke" in sys.argv:
        sys.exit(smoke())

    t0 = time.time()
    J704 = json.loads(SEALED704.read_text())["traces"]
    J705 = json.loads(SEALED705.read_text())["traces"]

    # ---- pins: runtime-parsed sealed refs must equal frozen pins (exit 5) --
    refs = {}
    for tag in ORDER:
        cfg = TRACES[tag]
        c = cfg["anchor_cap"]
        ref = {
            "sp": {c: J704[tag]["spread_pp"][c]},
            "lw": {c: J704[tag]["policy_grid"][c]["lcnu"]["whole"]},
            "rw": {c: J704[tag]["policy_grid"][c]["ring10"]["whole"]},
            "burst": J704[tag]["burst_windows"],
            "ws20": J705[tag]["WS20"],
        }
        if not (
            ref["sp"] == cfg["pin_sp"]
            and ref["lw"] == cfg["pin_lcnu_whole"]
            and ref["rw"] == cfg["pin_ring_whole"]
            and ref["ws20"] == cfg["pin_ws20"]
            and cfg["caps"][-1] == ref["ws20"]
        ):
            print(f"PIN MISMATCH {tag} vs sealed artifacts:", ref, flush=True)
            sys.exit(5)
        refs[tag] = ref

    # ---- structure first for BOTH traces (exit 3 before any measurement) --
    streams = {}
    for tag in ORDER:
        keys, per_prompt = load_stream(TRACES[tag]["pt"])
        check_structure(tag, TRACES[tag], keys, per_prompt)
        streams[tag] = keys
        print(f"{tag}: structure ok [{time.time()-t0:.0f}s]", flush=True)

    out_tr = {}
    anchors_ok, det_all = True, True
    for tag in ORDER:
        cfg, ref = TRACES[tag], refs[tag]
        keys = streams[tag]
        grid, burst = run_grid(tag, cfg, keys, t0)
        spread = spread_of(grid, cfg["caps"])
        c = cfg["anchor_cap"]
        a_ok = (
            spread[c] == ref["sp"][c]
            and grid[c]["lcnu"]["whole"] == ref["lw"][c]
            and grid[c]["ring10"]["whole"] == ref["rw"][c]
            and burst == ref["burst"]
        )
        anchor = {
            "cap": int(c),
            "spread_mine": spread[c],
            "spread_ref": ref["sp"][c],
            "lcnu_whole_mine": grid[c]["lcnu"]["whole"],
            "lcnu_whole_ref": ref["lw"][c],
            "ring_whole_mine": grid[c]["ring10"]["whole"],
            "ring_whole_ref": ref["rw"][c],
            "burst_equal": burst == ref["burst"],
            "ok": a_ok,
        }
        beyond = [x for x in cfg["caps"] if x > int(c)]
        s_max = max(spread[str(x)] for x in beyond)
        first_rise = next((x for x in beyond if spread[str(x)] > FLAT_PP), None)
        ws20 = ref["ws20"]
        ws20_cell = {
            "cap": ws20,
            "spread_pp": spread[str(ws20)],
            "le_0p1": spread[str(ws20)] <= WS20_CELL_PP,
        }
        da, dc = cfg["det_cell"]
        keys2, _ = load_stream(cfg["pt"])
        det_ok = (
            policy_cell(
                da, dc, keys2, cfg["n_win"], burst, cfg["nl"], cfg["prior"]
            )
            == grid[str(dc)][da]
        )
        del keys2
        anchors_ok &= a_ok
        det_all &= det_ok
        out_tr[tag] = {
            "burst_windows": burst,
            "spread_pp": spread,
            "policy_grid": grid,
            "C1_anchor": anchor,
            "C2_flat": {
                "s_max_pp": s_max,
                "first_rise_cap": first_rise,
                "flat": first_rise is None,
                "threshold_pp": FLAT_PP,
                "beyond_caps": beyond,
            },
            "C3_ws20_cell": ws20_cell,
            "C4_det": det_ok,
            "WS20": ws20,
            "cap_over_WS20": {str(x): round(x / ws20, 4) for x in cfg["caps"]},
        }
        print(
            f"{tag}: s_max={s_max} first_rise={first_rise} "
            f"ws20_cell={ws20_cell['spread_pp']} anchor={a_ok} det={det_ok} "
            f"[{time.time()-t0:.0f}s]",
            flush=True,
        )

    rises = [t for t in ORDER if not out_tr[t]["C2_flat"]["flat"]]
    verdict = (
        "CEILING_FLAT_BOTH"
        if not rises
        else "CEILING_RISE_" + "_".join(x.upper() for x in rises)
    )
    out = {
        "exp": "exp709-ceiling2",
        "traces": out_tr,
        "C1_all": anchors_ok,
        "C4_all": det_all,
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": verdict,
    }
    (D / "exp709-ceiling2.json").write_text(json.dumps(out, indent=1))
    print(
        f"exp709 settled verdict={verdict} anchors={anchors_ok} "
        f"det={det_all}",
        flush=True,
    )
    if not anchors_ok:
        sys.exit(5)
    if not det_all:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
