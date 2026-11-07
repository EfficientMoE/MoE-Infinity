#!/usr/bin/env python3
"""exp706: qwen30b-lo CROSS1 location — capacity-grid extension past 2048.

Mirrors exp704's sealed run_capacitymap.py exactly (arms, loader, novelty,
burst windows, spread definition); lo trace only; caps extended to 3328.
Anchors parsed at runtime from sealed exp704 JSON and pinned (PREREG.md).
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

CFG = {
    "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_lo.pt",
    "nl": 48,
    "prior": 8 / 128,
    "N": 638208,
    "distinct": 5842,
    "prompts": [512, 512, 126, 512],
    "n_win": 155,
    "burst_n": 16,
    "caps": [1792, 2048, 2176, 2304, 2432, 2560, 2816, 3072, 3328],
    "pin_sp": {"1792": 4.211, "2048": 2.116},
    "pin_lcnu_whole": {"1792": 0.7852596645607701, "2048": 0.8228962971319695},
    "det_cell": ("lcnu", 2304),
}
PIN_WS20_LO = 4662
LAW_LO, LAW_HI = 0.50, 0.66


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


def run_grid(tag, cfg, log):
    t0 = time.time()
    keys, per_prompt = load_stream(cfg["pt"])
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
    return grid, burst, t0


def main():
    t0 = time.time()
    cfg = dict(CFG)
    if "--smoke" in sys.argv:
        # Disclosed smoke: synthetic mini-trace through the SAME machinery
        # (records, loader, novelty, burst, policy_cell). No result JSON.
        L, K, POOL = 4, 2, 24
        rng = np.random.default_rng(7)
        records = []
        for p in range(3):  # 3 prompts: prefill then decodes
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
        torch.save(
            {"records": records[: 1 + 3 * (1600 // 3 + 1)]}, str(smoke_pt)
        )
        scfg = {
            "pt": smoke_pt,
            "nl": L,
            "prior": 0.5,
            "N": 12800 * 0 + 0,
            "distinct": 0,
            "prompts": None,
            "n_win": 3,
            "burst_n": 2,
            "caps": [16, 32],
            "det_cell": ("lcnu", 16),
        }
        # structure expectations for the synthetic stream
        keys, per_prompt = load_stream(scfg["pt"])
        scfg["N"], scfg["distinct"], scfg["prompts"] = (
            len(keys),
            len(set(keys)),
            per_prompt,
        )
        n = len(keys)
        assert n // W == scfg["n_win"], f"smoke n_win {n // W}"
        grid, burst, _ = run_grid("smoke", scfg, print)
        det_ok = (
            policy_cell("lcnu", 16, keys, scfg["n_win"], burst, L, 0.5)
            == grid["16"]["lcnu"]
        )
        sane = all(
            0.0 <= c["whole"] <= 1.0 and 0.0 <= c["burst"] <= 1.0
            for cell in grid.values()
            for c in cell.values()
        )
        smoke_pt.unlink()
        ok = det_ok and sane
        print(
            f"SMOKE det={det_ok} sane={sane} N={n} distinct={scfg['distinct']}"
        )
        print("SMOKE_OK" if ok else "SMOKE_FAIL")
        return 0 if ok else 2

    # ---- real run ----
    J704 = json.loads(SEALED704.read_text())["traces"]["lo"]
    J705 = json.loads(SEALED705.read_text())["traces"]["lo"]
    # anchors: runtime-parsed AND pinned (PREREG)
    ref_sp = {c: J704["spread_pp"][c] for c in ("1792", "2048")}
    ref_lw = {
        c: J704["policy_grid"][c]["lcnu"]["whole"] for c in ("1792", "2048")
    }
    ws20 = J705["WS20"]
    pin_ok = (
        ref_sp == cfg["pin_sp"]
        and ref_lw == cfg["pin_lcnu_whole"]
        and ws20 == PIN_WS20_LO
    )
    if not pin_ok:
        print(
            "PIN MISMATCH vs sealed artifacts:",
            ref_sp,
            ref_lw,
            ws20,
            flush=True,
        )
        sys.exit(5)

    grid, burst, t0 = run_grid("lo", cfg, print)
    spread = {
        str(cap): round(
            (
                max(grid[str(cap)][a]["burst"] for a in ARMS)
                - min(grid[str(cap)][a]["burst"] for a in ARMS)
            )
            * 100,
            3,
        )
        for cap in cfg["caps"]
    }

    # L3 replication-anchor (exact)
    anchor = {
        c: {
            "mine": spread[c],
            "ref": ref_sp[c],
            "lcnu_whole_mine": grid[c]["lcnu"]["whole"],
            "lcnu_whole_ref": ref_lw[c],
            "ok": spread[c] == ref_sp[c]
            and grid[c]["lcnu"]["whole"] == ref_lw[c],
        }
        for c in ("1792", "2048")
    }
    anchor_ok = all(a["ok"] for a in anchor.values())
    consistency_2048 = spread["2048"] >= 1.0  # sealed NOT_REACHED contract

    # L4 det
    da, dc = cfg["det_cell"]
    keys2, _ = load_stream(cfg["pt"])
    det_ok = (
        policy_cell(da, dc, keys2, cfg["n_win"], burst, cfg["nl"], cfg["prior"])
        == grid[str(dc)][da]
    )

    caps_s = [str(c) for c in cfg["caps"]]
    cross1 = next((c for c in caps_s if spread[c] < 1.0), None)
    ratio = (
        round(int(cross1) / ws20, 4)
        if cross1
        else round(cfg["caps"][-1] / ws20, 4)
    )
    if cross1 is not None:
        l2 = LAW_LO <= int(cross1) / ws20 <= LAW_HI
    else:
        l2 = False  # NOT_REACHED(<=3328): 3328/4662=0.713 > 0.66 refutes interval
    ring_never = all(
        grid[c]["ring10"]["burst"] < grid[c]["lcnu"]["burst"] for c in caps_s
    )
    decay = spread[caps_s[-1]] < spread[caps_s[0]]

    out = {
        "exp": "exp706-locross",
        "burst_windows": burst,
        "spread_pp": spread,
        "policy_grid": grid,
        "L3_anchor": anchor,
        "L3_all": anchor_ok,
        "L1_CROSS1": cross1 if cross1 is not None else "NOT_REACHED(<=3328)",
        "L1_consistency_2048_ge_1pp": consistency_2048,
        "L2_law_transfer": l2,
        "CROSS1_over_WS20": ratio,
        "WS20_lo": ws20,
        "law_interval": [LAW_LO, LAW_HI],
        "L4_det": det_ok,
        "continuity_ring_never_wins": ring_never,
        "continuity_decay": decay,
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": "LOCROSS_MAP",
    }
    (D / "exp706-locross.json").write_text(json.dumps(out, indent=1))
    print(
        f"exp706 settled CROSS1={out['L1_CROSS1']} ratio={ratio} "
        f"L2={l2} anchors={anchor_ok} det={det_ok}",
        flush=True,
    )
    if not anchor_ok:
        sys.exit(5)
    if not det_ok:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
