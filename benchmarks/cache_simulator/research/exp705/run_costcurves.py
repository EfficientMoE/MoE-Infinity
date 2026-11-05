#!/usr/bin/env python3
"""exp705: cost-integration — byte-denominated cache-sizing curves.

Transforms sealed exp704 capacity-map curves into GiB / miss-traffic-per-token
tables (frozen byte model cited in PREREG.md) and tests the working-set
mechanism behind CROSS1 (WS20 computed fresh from frozen traces).
Read-only over sealed inputs; writes only in its own dir.
Exit codes: 0 settled | 2 det | 3 structure | 5 anchor.
--smoke: disclosed smoke run on a synthetic mini-stream (no result JSON).
"""
import json
import sys
import time
from pathlib import Path

HOME = Path.home()
D = Path(__file__).parent
SEALED = HOME / "warp-work/exp704-moei-capacitymap/exp704-capacitymap.json"

# ---- frozen byte model (PREREG.md, fetched 2026-11-04) ----
FAM = {
    "granite": {
        "hidden": 1536, "inter": 512, "E": 40, "L": 32, "k": 8,
        "src": "ibm-granite/granite-3.1-3b-a800m-instruct",
        "pt": HOME / "warp-work/exp650-moei-granite3b-trace-pr/stream_granite3b.pt",
    },
    "hi": {
        "hidden": 2048, "inter": 768, "E": 128, "L": 48, "k": 8,
        "src": "Qwen/Qwen3-30B-A3B",
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_hi.pt",
    },
    "lo": {
        "hidden": 2048, "inter": 768, "E": 128, "L": 48, "k": 8,
        "src": "Qwen/Qwen3-30B-A3B",
        "pt": HOME / "warp-work/exp656-moei-within/stream_qwen30b_lo.pt",
    },
}


def b_e(fam):
    return 3 * fam["hidden"] * fam["inter"] * 2  # bf16 bytes per expert


def load_decode_keys(pt):
    """Decode-access keys in order, mirroring exp704's loader (read-only).

    DEVIATIONS D1: decode records store (L, seq_pos=1, k); reshape to
    (L, -1) exactly as run_policies_real.py:169-174 does. Keys keep
    multiplicity (each of the k routed experts per layer is one access).
    """
    import torch
    s = torch.load(str(pt), map_location="cpu", weights_only=False)
    keys = []
    for r in s["records"]:
        if r["forward_mode"] == 1:      # prefill
            continue
        tensor = r["topk_ids_of_layer"]
        num_layers = tensor.shape[0]
        flattened = tensor.reshape(num_layers, -1)
        for l in range(num_layers):
            for e in flattened[l].tolist():
                keys.append((l, int(e)))
    return keys


def ws20(keys):
    """Distinct keys in the final 20% of the stream."""
    cut = int(len(keys) * 0.8)
    return len(set(keys[cut:]))


def spread_recompute(trace_json):
    """max-min of burst-window hits over arms, in pp, per cap.

    DEVIATIONS D2: sealed exp704 spread_pp is defined over BURST hits
    (run_capacitymap.py:138-141), not whole hits. Frozen C4c assumed
    whole; the frozen-track run caught this (ANCHOR-FAIL, exit 5).
    Repaired anchor uses the sealed definition; tolerance unchanged
    (0.01) though exact-0 reproduction was pre-verified at all 27 caps.
    """
    out = {}
    for cap, cell in trace_json["policy_grid"].items():
        burst = [a["burst"] for a in cell.values()]
        out[int(cap)] = (max(burst) - min(burst)) * 100.0
    return out


def main(smoke=False):
    t0 = time.time()
    if smoke:
        synth = [(i % 97, i % 11) for i in range(5000)]
        w1, w2 = ws20(synth), ws20(synth)
        det = w1 == w2 == min(1000, 97 * 11)  # cycle period 1067; tail is 1000 accesses
        print(f"SMOKE ws20 det={w1 == w2} val={w1} | b_e granite={b_e(FAM['granite'])} "
              f"qwen={b_e(FAM['hi'])}")
        print("SMOKE_OK" if det else "SMOKE_FAIL")
        return 0 if det else 2

    J = json.loads(SEALED.read_text())
    res = {"exp": "exp705-costcurves", "traces": {}, "elapsed_s": 0}
    c4 = True

    for name, fam in FAM.items():
        be = b_e(fam)
        keys = load_decode_keys(fam["pt"])
        # Structure gate: decode-access count must equal the frozen trace N.
        n_reg = {"granite": 638208, "hi": 786432, "lo": 638208}[name]
        c4b = len(keys) == n_reg
        apt = fam["L"] * fam["k"]
        w1 = ws20(keys)
        w2 = ws20(keys)
        c4d = w1 == w2

        t = J["traces"][name]
        sp = {int(c): v for c, v in t["spread_pp"].items()}
        caps = sorted(sp)
        rec = spread_recompute(t)
        probe = [caps[0], caps[len(caps) // 2], caps[-1]]
        c4c = all(abs(rec[c] - sp[c]) <= 0.01 for c in probe)

        cross1 = t["B2_CROSS1"]
        cross1 = int(cross1) if str(cross1).isdigit() else None
        total_slots = fam["E"] * fam["L"]
        table = []
        for c in caps:
            table.append({
                "cap": c,
                "cache_GiB": round(c * be / 2**30, 4),
                "spread_pp": sp[c],
                "premium_B_per_token": round(sp[c] / 100 * apt * be, 1),
                "lcnu_miss_GiB_per_token": round((1 - t["policy_grid"][str(c)]["lcnu"]["whole"]) * apt * be / 2**30, 5),
            })
        cross1_gib = None if cross1 is None else round(cross1 * be / 2**30, 4)
        below = [sp[c] for c in caps if cross1 and c < cross1]
        c3 = None
        if cross1:
            c3 = max(below) >= 10 * sp[cross1]
        lo_bound_gib = round(2048 * be / 2**30, 4)

        if name == "granite":
            c1 = cross1_gib is not None and cross1_gib <= 4.0
        elif name == "hi":
            c1 = cross1_gib is not None and cross1_gib >= 15.0
        else:
            c1 = None  # lo: report lower bound only
        if cross1 is not None:
            c2 = abs(cross1 - w1) / w1 <= 0.30
        else:
            c2 = w1 >= 2048

        res["traces"][name] = {
            "b_e_bytes": be, "b_e_MiB": round(be / 2**20, 3),
            "src": fam["src"], "N_decode_accesses": len(keys),
            "N_matches_sealed": c4b, "APT": apt,
            "WS20": w1, "WS20_det": c4d,
            "CROSS1_slots": cross1, "CROSS1_GiB": cross1_gib,
            "CROSS1_pct_keyspace": None if cross1 is None else round(100 * cross1 / total_slots, 1),
            "lo_bound_GiB_2048": lo_bound_gib if cross1 is None else None,
            "total_expert_GiB": round(total_slots * be / 2**30, 3),
            "spread_recompute_match": c4c,
            "C1_hardware": c1, "C2_working_set": c2,
            "C2_ratio": None if cross1 is None else round((cross1 - w1) / w1, 3),
            "C3_cliff": c3,
            "table": table,
        }
        c4 = c4 and c4b and c4c and c4d

    res["C4_all"] = bool(c4)
    res["verdict"] = "COST_CURVES" if c4 else "ANCHOR-FAIL"
    res["elapsed_s"] = round(time.time() - t0, 1)
    out = D / "exp705-costcurves.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"exp705 settled verdict={res['verdict']} C4={c4} "
          f"-> {out.name} ({res['elapsed_s']}s)")
    return 0 if c4 else 5


if __name__ == "__main__":
    sys.exit(main(smoke="--smoke" in sys.argv))
