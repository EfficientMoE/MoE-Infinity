#!/usr/bin/env python3
"""exp710: HL-only region probe to the BO search ceiling (adopts exp679 design v1).

Streams: exactly exp670/exp674's 11 (seed scheme + sha-assert against
research/exp670-synth.json). Arms: lcnu, s3fifo, and costfit at center
MARGIN/GAMMA with HALF_LIFE in HLS. H0: HL_REPL arm == exp674 cf_hlonly and
HL 32768 arm == exp674 cf_center, per-cell rate/prop/switches EXACTLY.
Adjudication per frozen PREREG. No hand-transcribed values.

Modes: default = full frozen run.  SMOKE=1 env = disclosed machinery smoke
(reduced grid + full-N H0 machinery replication on 1 stream x 2 caps with
ONLY lcnu/s3fifo + the two exp674-sealed replica arms (HL_REPL, 32768) ->
no new-arm outcome data observed; reduced-grid S4 uses N=60k unpinned
streams, not the measured streams).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warp-work/moei-granite3b"))
from benchmarks.cache_simulator.policies import build_policy  # noqa: E402

D = Path(__file__).parent
REPO = Path.home() / "warp-work/moei-granite3b"
N_ACC = 638_208
H_RECENCY, R_WINDOW = 2048, 4096
SKEWS = [0.8, 1.1, 1.4]
LOCALITIES = [0.0, 0.3, 0.6]
CAPS = [128, 256, 512, 1024, 2048, 4096]
CENTER_M, CENTER_G = 0.005, 0.0025
EXP670 = REPO / "benchmarks/cache_simulator/research/exp670-synth.json"
EXP674 = (
    Path.home() / "warp-work/exp674-moei-hl-transfer/exp674-hl-transfer.json"
)
# replication bookend = exp674 cf_hlonly half-life, parsed at runtime (exact float, not rounded)
HL_REPL = float(json.loads(EXP674.read_text())["bo_params"]["half_life"])
HLS = [32768, 65536, 131072, HL_REPL, 262144]
SAFETY_PP, SWITCH_MAX, INFORMATIVE_PP = 0.5, 6, 0.1


def arm_name(hl):
    return f"cf_hl{round(hl)}"


def gen_stream(seed, E, L, s, lam, n_acc=N_ACC):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, E + 1, dtype=np.float64)
    zipf_p = ranks ** (-s)
    zipf_p /= zipf_p.sum()
    experts = rng.choice(E, size=n_acc, p=zipf_p)
    layers = rng.integers(0, L, size=n_acc)
    keys = list(zip(layers.tolist(), experts.tolist()))
    n_replay = int(lam * n_acc)
    idx = rng.choice(np.arange(R_WINDOW, n_acc), size=n_replay, replace=False)
    w = 0.5 ** (np.arange(1, R_WINDOW + 1) / H_RECENCY)
    w /= w.sum()
    ages = rng.choice(R_WINDOW, size=n_replay, p=w) + 1
    for i, a in zip(idx.tolist(), ages.tolist()):
        keys[i] = keys[i - a]
    return keys


def stream_sha(keys):
    raw = "\x00".join(f"{l}:{e}" for l, e in keys[:50000])
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def make_costfit(cap, gk, hl):
    pol = build_policy("costfit", cap, **gk)
    pol.MARGIN, pol.GAMMA, pol.HALF_LIFE = CENTER_M, CENTER_G, float(hl)
    pol._decay = 0.5 ** (1.0 / float(hl))
    return pol


def evaluate(keys, E, L, caps, hls, top_k=8):
    out = {}
    gk = {"num_layers": L, "prior": top_k / E}
    n = len(keys)
    for cap in caps:
        cell = {}
        for name in ("lcnu", "s3fifo"):
            pol = build_policy(name, cap)
            hits = sum(1 for k in keys if pol.on_access(k))
            cell[name] = {
                "rate": round(hits / n, 6),
                "prop": round(pol.effective_hit_rate, 6),
            }
        for hl in hls:
            pol = make_costfit(cap, gk, hl)
            assert pol.HALF_LIFE == float(hl) and pol._decay == 0.5 ** (
                1.0 / float(hl)
            )
            hits = sum(1 for k in keys if pol.on_access(k))
            cell[arm_name(hl)] = {
                "rate": round(hits / n, 6),
                "prop": round(pol.effective_hit_rate, 6),
                "switches": getattr(pol, "switches", 0),
            }
        out[str(cap)] = cell
    return out


def stream_cfgs():
    streams = [
        {"s": s, "lam": lam, "E": 128, "L": 48}
        for s in SKEWS
        for lam in LOCALITIES
    ]
    streams.append({"s": 1.1, "lam": 0.3, "E": 32, "L": 32})
    streams.append({"s": 1.4, "lam": 0.6, "E": 32, "L": 32})
    return streams


def run_all(
    expected_shas,
    idxs=None,
    caps=CAPS,
    hls=HLS,
    n_acc=N_ACC,
    check_sha=True,
    seed_perturb=0,
):
    streams = stream_cfgs()
    seeds = np.random.default_rng(67020261104).spawn(11)
    t0 = time.time()
    grid = []
    for i, cfg in enumerate(streams):
        if idxs is not None and i not in idxs:
            continue
        keys = gen_stream(
            int(seeds[i].integers(0, 2**63)) + seed_perturb,
            cfg["E"],
            cfg["L"],
            cfg["s"],
            cfg["lam"],
            n_acc,
        )
        sha = stream_sha(keys)
        if check_sha:
            assert (
                sha == expected_shas[i]
            ), f"stream {i} sha {sha} != exp670 {expected_shas[i]}"
        res = evaluate(keys, cfg["E"], cfg["L"], caps, hls)
        grid.append({"cfg": cfg, "seed": sha, "caps": res})
        print(
            f"stream {i} done [{time.time()-t0:.0f}s]", flush=True
        )  # timings only until settlement
    return grid


def h0_replication(grid, ref_grid, pairs=None):
    """Exact per-cell equality vs sealed exp674: HL_REPL arm == cf_hlonly and
    HL 32768 arm == cf_center (both center M/G). Only pairs present are checked."""
    pairs = pairs or [(HL_REPL, "cf_hlonly"), (32768, "cf_center")]
    ref = {
        (json.dumps(st["cfg"], sort_keys=True), cap): cell
        for st in ref_grid
        for cap, cell in st["caps"].items()
    }
    mism, n = [], 0
    for st in grid:
        for cap, cell in st["caps"].items():
            r = ref[(json.dumps(st["cfg"], sort_keys=True), cap)]
            for hl, refarm in pairs:
                if arm_name(hl) not in cell:
                    continue
                n += 1
                if cell[arm_name(hl)] != r[refarm]:
                    mism.append(
                        {
                            "cfg": st["cfg"],
                            "cap": cap,
                            "arm": arm_name(hl),
                            "ref": refarm,
                        }
                    )
    return {"cells": n, "mismatches": mism, "ok": n > 0 and not mism}


def adjudicate(grid, h0):
    informative, acct = 0, []
    per = {
        hl: {
            "worst": None,
            "safety_cells": 0,
            "sw_cells": 0,
            "max_sw": 0,
            "deltas": [],
        }
        for hl in HLS
    }
    for st in grid:
        for cap, cell in st["caps"].items():
            for hl in HLS:
                a = cell[arm_name(hl)]
                if a["rate"] != a["prop"]:
                    acct.append(
                        {"cfg": st["cfg"], "cap": cap, "arm": arm_name(hl)}
                    )
            lc, s3 = cell["lcnu"]["rate"], cell["s3fifo"]["rate"]
            if abs(lc - s3) * 100 < INFORMATIVE_PP:
                continue
            informative += 1
            best = max(lc, s3)
            for hl in HLS:
                a = cell[arm_name(hl)]
                p = per[hl]
                d = (best - a["rate"]) * 100
                p["deltas"].append(
                    (a["rate"] - cell[arm_name(32768)]["rate"]) * 100
                )
                p["max_sw"] = max(p["max_sw"], a["switches"])
                if p["worst"] is None or d > p["worst"]["deficit_pp"]:
                    p["worst"] = {
                        "cfg": st["cfg"],
                        "cap": cap,
                        "deficit_pp": round(d, 3),
                        "switches": a["switches"],
                    }
                if d > SAFETY_PP:
                    p["safety_cells"] += 1
                if a["switches"] > SWITCH_MAX:
                    p["sw_cells"] += 1
    arms = {}
    for hl in HLS:
        p = per[hl]
        arms[str(round(hl))] = {
            "H1_safe": p["safety_cells"] == 0 and p["sw_cells"] == 0,
            "worst": p["worst"],
            "cells_over_bar": p["safety_cells"],
            "cells_over_switch": p["sw_cells"],
            "max_switches": p["max_sw"],
            "mean_delta_vs_hl32768_pp": round(
                sum(p["deltas"]) / len(p["deltas"]), 4
            )
            if p["deltas"]
            else None,
        }
    h4 = not acct
    if not (h0["ok"] and h4):
        verdict = "ACCOUNTING-FAIL"
    else:
        failing = [hl for hl in HLS if not arms[str(round(hl))]["H1_safe"]]
        verdict = (
            "HL-CEILING-SAFE"
            if not failing
            else f"HL-BOUNDARY@{round(min(failing))}"
        )
    safe_sorted = [round(hl) for hl in HLS if arms[str(round(hl))]["H1_safe"]]
    largest_safe_prefix = None
    for hl in HLS:
        if arms[str(round(hl))]["H1_safe"]:
            largest_safe_prefix = round(hl)
        else:
            break
    return {
        "verdict": verdict,
        "H0_replication": h0,
        "H4_accounting": h4,
        "accounting_violations": acct,
        "informative_cells": informative,
        "arms": arms,
        "safe_hls": safe_sorted,
        "certified_hl_upper_edge": largest_safe_prefix,
    }


def smoke():
    exp670 = json.loads(EXP670.read_text())
    shas = [st["seed"] for st in exp670["grid"]]
    ref = json.loads(EXP674.read_text())["grid"]
    # S1 sha assertion bites on a perturbed seed (full-N stream 0)
    try:
        run_all(shas, idxs={0}, caps=[], hls=[], seed_perturb=1)
        print("SMOKE S1 FAIL: perturbed seed not caught")
        return 1
    except AssertionError:
        print("SMOKE S1 ok: sha assertion bites on perturbed seed")
    # S2 full-N H0 machinery replication: stream 4 (s=1.1 lam=0.3 E128), caps 512+2048, ONLY the two sealed-replica arms (HL_REPL, 32768)
    g = run_all(shas, idxs={4}, caps=[512, 2048], hls=[HL_REPL, 32768])
    h0 = h0_replication(g, ref)
    assert h0["cells"] == 4, h0
    print(
        "SMOKE S2 H0 machinery replication cells", h0["cells"], "ok", h0["ok"]
    )
    # S3 H0 logic negative control: perturb one value -> must detect
    bad = json.loads(json.dumps(g))
    bad[0]["caps"]["512"][arm_name(HL_REPL)]["rate"] += 1e-6
    print(
        "SMOKE S3 H0 negative control detects",
        not h0_replication(bad, ref)["ok"],
    )
    # S4 reduced grid all 7 arms: attrs, accounting, adjudicate well-formed, double-gen identical
    r1 = run_all(
        shas, idxs={0, 9}, caps=[128, 256], n_acc=60_000, check_sha=False
    )
    r2 = run_all(
        shas, idxs={0, 9}, caps=[128, 256], n_acc=60_000, check_sha=False
    )
    det = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    adj = adjudicate(r1, {"ok": True, "cells": 0, "mismatches": []})
    wf = set(adj["arms"]) == {str(round(h)) for h in HLS} and adj[
        "H4_accounting"
    ] in (True, False)
    print(
        "SMOKE S4 det",
        det,
        "adjudicate well-formed",
        wf,
        "accounting",
        adj["H4_accounting"],
    )
    ok = h0["ok"] and det and wf and adj["H4_accounting"]
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


def main():
    if os.environ.get("SMOKE") == "1":
        sys.exit(smoke())
    exp670 = json.loads(EXP670.read_text())
    shas = [st["seed"] for st in exp670["grid"]]
    ref = json.loads(EXP674.read_text())["grid"]
    t0 = time.time()
    g1 = run_all(shas)
    g2 = run_all(shas)
    assert json.dumps(g1, sort_keys=True) == json.dumps(
        g2, sort_keys=True
    ), "nondeterministic"
    v = adjudicate(g1, h0_replication(g1, ref))
    out = {
        **v,
        "determinism_two_runs": True,
        "elapsed_s": round(time.time() - t0, 1),
        "grid": g1,
        "rules": {
            "caps": CAPS,
            "skews": SKEWS,
            "localities": LOCALITIES,
            "n_acc": N_ACC,
            "center_m": CENTER_M,
            "center_g": CENTER_G,
            "hls": HLS,
            "hl_repl_source": "exp674 bo_params.half_life",
            "safety_pp": SAFETY_PP,
            "switch_max": SWITCH_MAX,
            "informative_pp": INFORMATIVE_PP,
        },
    }
    (D / "exp710-hlregion.json").write_text(json.dumps(out, indent=1))
    print("settled verdict=" + v["verdict"], flush=True)


if __name__ == "__main__":
    main()
