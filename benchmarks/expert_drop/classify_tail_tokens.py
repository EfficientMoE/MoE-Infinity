# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Per-token tail classification for the drop-on-miss debate (#234).

Joins a MOE_EXPERT_DROP_TRACE binary dump with the per-prompt decode ITLs
recorded by trace_capture.py in the SAME run, replays the mass-budget
selector on every decode record, and classifies every routed non-resident
expert (the fetch-causing set) of every decode token as one of:

  - dropped        selector dropped it (no fetch)
  - mustkeep_mass  kept because its cumulative candidate mass exceeds the
                   budget (the policy never drops it)
  - kept_mink      inside the mass budget but kept by the min_k clamp
  - kept_flat      kept because the whole row took the flatness bypass
  - kept_guard     kept because a row guard failed (count <= min_k,
                   non-finite / negative weights, zero max weight)

Tokens are then bucketed by their measured ITL percentile band
([0,p50) / [p50,p90) / [p90,p99) / [p99,max]) and the per-band fetch
composition, routing-mass ranks, per-layer fetch depth, and fetch bytes are
reported. This is the discriminating log requested on issue #234: if the
tail band fetches predominantly mustkeep_mass experts, re-scoping the gate
to p50/p90 is mechanism-backed and a tail-targeting policy is the only tail
lever; if it fetches droppable-but-kept experts, a policy change addresses
it directly; if its fetches are individually cheap but deep/congested,
post-selection fetch priority is the cheaper lever.

Alignment contract (greedy, min_new_tokens == max_new_tokens, batch 1):
token_times in trace_capture.py records one timestamp per sampled token;
itl_ms_per_prompt[i] is the interval that covers decode forward i+1, so
decode forward f (0-based, f >= 1) of a prompt pairs with itl[f-1].
Forward 0 of each prompt has no measured ITL and is excluded.

The replay is cross-checked against the Python reference selector
(moe_infinity.runtime.expert_drop.select_expert_drops) on a sample of
records (--crosscheck N, 0 disables; --crosscheck -1 checks everything).

Residency caveat: the trace snapshot is taken at route time, microseconds
before the fused selector reads residency itself; an in-flight prefetch can
flip an expert to resident between the two reads. Treat single-expert
discrepancies near admission as noise, not signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from benchmarks.expert_drop.trace_parser import parse_trace
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trace_parser import parse_trace

CLASSES = ("dropped", "mustkeep_mass", "kept_mink", "kept_flat", "kept_guard")
BANDS = ("p0_50", "p50_90", "p90_99", "p99_100")


def replay_row(weights, mask, resident, min_k, mass_budget, flatness_floor):
    """Replay SelectExpertDrops on one row; classify non-resident routed.

    Returns (per-expert class dict {expert_id: class}, row_tag) where
    row_tag in {"normal", "flat", "guard"}.
    """
    sel = mask.astype(bool)
    nonres = sel & ~resident.astype(bool)
    nonres_ids = np.nonzero(nonres)[0]
    selected_count = int(sel.sum())
    w_sel = weights[sel]

    guards_ok = (
        selected_count > min_k
        and bool(np.isfinite(w_sel).all())
        and not bool((w_sel < 0).any())
        and float(w_sel.max()) > 0.0
    )
    if not guards_ok:
        return {int(e): "kept_guard" for e in nonres_ids}, "guard"
    if float(w_sel.min()) / float(w_sel.max()) > flatness_floor:
        return {int(e): "kept_flat" for e in nonres_ids}, "flat"

    # Reference order: descending-id prefill + stable ascending-weight sort
    # (lower weight first, higher id on ties).
    cands = nonres_ids[::-1]
    order = np.argsort(weights[cands], kind="stable")
    cands = cands[order]
    cum = np.cumsum(weights[cands].astype(np.float64))
    budget_count = int((cum <= mass_budget).sum())
    dropped_here = min(budget_count, selected_count - min_k)

    out = {}
    for idx, e in enumerate(cands):
        if idx < dropped_here:
            out[int(e)] = "dropped"
        elif idx < budget_count:
            out[int(e)] = "kept_mink"
        else:
            out[int(e)] = "mustkeep_mass"
    return out, "normal"


def crosscheck(records, min_k, mass_budget, flatness_floor, sample, rng):
    """Bit-check replay_row drop sets against the Python reference."""
    import torch

    from moe_infinity.runtime.expert_drop import select_expert_drops

    idx = np.arange(len(records))
    if 0 <= sample < len(idx):
        idx = rng.choice(idx, size=sample, replace=False)
    mismatches = 0
    for i in idx:
        weights, mask, resident = records[i]
        cls, _ = replay_row(
            weights, mask, resident, min_k, mass_budget, flatness_floor
        )
        mine = {e for e, c in cls.items() if c == "dropped"}
        ref_mask, _, _ = select_expert_drops(
            torch.from_numpy(mask.astype(bool)).reshape(1, -1),
            torch.from_numpy(weights.astype(np.float32)).reshape(1, -1),
            torch.from_numpy(resident.astype(bool)),
            min_k=min_k,
            mass_budget=mass_budget,
            flatness_floor=flatness_floor,
        )
        ref_dropped = {
            int(e)
            for e in np.nonzero(
                mask.astype(bool) & ~ref_mask.numpy().reshape(-1)
            )[0]
        }
        if mine != ref_dropped:
            mismatches += 1
    return len(idx), mismatches


def group_decode_forwards(trace):
    """Split trace records into (prompt, forward) -> list of record indices.

    Prefill records have rows > 1; each contiguous prefill block starts a
    new prompt. Decode records (rows == 1) are grouped into forwards by
    layer wrap-around (a non-increasing layer index starts a new forward).
    """
    prompts = []  # list of list of forwards; forward = list of record idx
    current_prompt = None
    current_forward = None
    prev_layer = None
    in_prefill = False
    n = len(trace["rows"])
    order = np.argsort(trace["seq"], kind="stable")
    for i in order:
        if trace["rows"][i] > 1:
            if not in_prefill:
                # New prompt begins at its prefill block.
                current_prompt = []
                prompts.append(current_prompt)
                in_prefill = True
                current_forward = None
                prev_layer = None
            continue
        in_prefill = False
        if current_prompt is None:
            # Trace started mid-stream (no prefill seen); skip until the
            # first prompt boundary so alignment stays trustworthy.
            continue
        layer = int(trace["layer"][i])
        if current_forward is None or (
            prev_layer is not None and layer <= prev_layer
        ):
            current_forward = []
            current_prompt.append(current_forward)
        current_forward.append(int(i))
        prev_layer = layer
    return prompts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trace", required=True, help="MOE_EXPERT_DROP_TRACE dump")
    ap.add_argument(
        "--latency",
        required=True,
        help="trace_capture.py --latency-out json from the SAME run "
        "(must contain itl_ms_per_prompt)",
    )
    ap.add_argument("--min-k", type=int, default=None)
    ap.add_argument("--mass-budget", type=float, default=None)
    ap.add_argument("--flatness-floor", type=float, default=0.5)
    ap.add_argument(
        "--crosscheck",
        type=int,
        default=2000,
        help="records to bit-check against the Python reference selector "
        "(0 disables, -1 checks all)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="", help="write full JSON report here")
    args = ap.parse_args()

    with open(args.latency) as fh:
        lat = json.load(fh)
    if "itl_ms_per_prompt" not in lat:
        raise SystemExit(
            "latency json lacks itl_ms_per_prompt; re-run trace_capture.py "
            "from a tree that records raw per-prompt ITLs"
        )
    if float(lat.get("head_budget", 0.0)) or float(
        lat.get("adaptive_slope", 0.0)
    ):
        raise SystemExit(
            "classifier replays the plain mass policy only; run had "
            "head_budget/adaptive_slope set"
        )
    min_k = args.min_k if args.min_k is not None else int(lat.get("min_k", 1))
    mass_budget = (
        args.mass_budget
        if args.mass_budget is not None
        else float(lat["mass_budget"])
    )
    counterfactual = lat.get("policy", "on_miss") != "on_miss"

    trace = parse_trace(args.trace)
    ne = trace["num_experts"]
    byte_size = trace["byte_size"]
    prompts = group_decode_forwards(trace)
    itl_pp = lat["itl_ms_per_prompt"]
    if len(prompts) != len(itl_pp):
        raise SystemExit(
            f"alignment failure: trace has {len(prompts)} prompts, latency "
            f"json has {len(itl_pp)}"
        )

    # Per measured token: join forward f (f>=1) with itl[f-1].
    tokens = []
    skipped_residency = 0
    records_for_check = []
    for p, (forwards, itls) in enumerate(zip(prompts, itl_pp)):
        if len(forwards) - 1 != len(itls):
            raise SystemExit(
                f"alignment failure: prompt {p} has {len(forwards)} decode "
                f"forwards but {len(itls)} ITLs (want forwards-1)"
            )
        for f in range(1, len(forwards)):
            itl = itls[f - 1]
            per_class = defaultdict(int)
            per_class_bytes = defaultdict(int)
            mass_ranks = defaultdict(list)
            layer_depths = []
            usable_layers = 0
            for ridx in forwards[f]:
                if trace["residency_known"][ridx] != 1:
                    skipped_residency += 1
                    continue
                usable_layers += 1
                layer = int(trace["layer"][ridx])
                weights = trace["weights"][ridx][0]
                mask = trace["mask"][ridx][0]
                resident = trace["resident"][ridx]
                records_for_check.append((weights, mask, resident))
                cls, _ = replay_row(
                    weights,
                    mask,
                    resident,
                    min_k,
                    mass_budget,
                    args.flatness_floor,
                )
                # Mass rank among the routed set: 1 = highest weight.
                sel_ids = np.nonzero(mask.astype(bool))[0]
                rank_order = sel_ids[
                    np.argsort(-weights[sel_ids], kind="stable")
                ]
                rank_of = {int(e): r + 1 for r, e in enumerate(rank_order)}
                depth = 0
                for e, c in cls.items():
                    per_class[c] += 1
                    if c != "dropped":
                        depth += 1
                        per_class_bytes[c] += int(byte_size[layer, e])
                        mass_ranks[c].append(rank_of[e])
                if depth:
                    layer_depths.append(depth)
            tokens.append(
                {
                    "prompt": p,
                    "forward": f,
                    "itl_ms": itl,
                    "usable_layers": usable_layers,
                    "per_class": dict(per_class),
                    "per_class_bytes": dict(per_class_bytes),
                    "mass_ranks": {k: v for k, v in mass_ranks.items()},
                    "fetch_depth_max": max(layer_depths, default=0),
                    "fetch_depth_mean": (
                        float(np.mean(layer_depths)) if layer_depths else 0.0
                    ),
                    "layers_with_fetch": len(layer_depths),
                }
            )

    if args.crosscheck:
        rng = np.random.default_rng(args.seed)
        sample = args.crosscheck if args.crosscheck > 0 else -1
        checked, mismatches = crosscheck(
            records_for_check,
            min_k,
            mass_budget,
            args.flatness_floor,
            sample,
            rng,
        )
        if mismatches:
            raise SystemExit(
                f"replay crosscheck FAILED: {mismatches}/{checked} records "
                "disagree with select_expert_drops"
            )
        print(f"crosscheck: {checked} records match the reference selector")

    itls = np.array([t["itl_ms"] for t in tokens])
    p50, p90, p99 = np.percentile(itls, [50, 90, 99])

    def band_of(itl):
        if itl >= p99:
            return "p99_100"
        if itl >= p90:
            return "p90_99"
        if itl >= p50:
            return "p50_90"
        return "p0_50"

    bands = {b: [] for b in BANDS}
    for t in tokens:
        bands[band_of(t["itl_ms"])].append(t)

    fetched_classes = [c for c in CLASSES if c != "dropped"]
    report = {
        "trace": args.trace,
        "latency": args.latency,
        "policy": lat.get("policy"),
        "counterfactual": counterfactual,
        "min_k": min_k,
        "mass_budget": mass_budget,
        "flatness_floor": args.flatness_floor,
        "num_tokens": len(tokens),
        "skipped_residency_unknown_records": skipped_residency,
        "itl_percentiles_ms": {"p50": p50, "p90": p90, "p99": p99},
        "bands": {},
    }
    for b in BANDS:
        ts = bands[b]
        if not ts:
            report["bands"][b] = {"tokens": 0}
            continue
        agg = {c: sum(t["per_class"].get(c, 0) for t in ts) for c in CLASSES}
        fetched = sum(agg[c] for c in fetched_classes)
        bytes_total = sum(sum(t["per_class_bytes"].values()) for t in ts)
        ranks_mustkeep = [
            r for t in ts for r in t["mass_ranks"].get("mustkeep_mass", [])
        ]
        ranks_droppable = [
            r
            for t in ts
            for c in ("kept_mink", "kept_flat", "kept_guard")
            for r in t["mass_ranks"].get(c, [])
        ]
        report["bands"][b] = {
            "tokens": len(ts),
            "itl_ms_mean": float(np.mean([t["itl_ms"] for t in ts])),
            "itl_ms_max": float(np.max([t["itl_ms"] for t in ts])),
            "per_token_mean": {c: agg[c] / len(ts) for c in CLASSES},
            "fetched_per_token_mean": fetched / len(ts),
            "fetched_share_mustkeep": (
                agg["mustkeep_mass"] / fetched if fetched else 0.0
            ),
            "fetched_share_droppable_kept": (
                (agg["kept_mink"] + agg["kept_flat"] + agg["kept_guard"])
                / fetched
                if fetched
                else 0.0
            ),
            "mass_rank_mustkeep_mean": (
                float(np.mean(ranks_mustkeep)) if ranks_mustkeep else None
            ),
            "mass_rank_droppable_kept_mean": (
                float(np.mean(ranks_droppable)) if ranks_droppable else None
            ),
            "fetch_depth_max_mean": float(
                np.mean([t["fetch_depth_max"] for t in ts])
            ),
            "fetch_depth_mean_mean": float(
                np.mean([t["fetch_depth_mean"] for t in ts])
            ),
            "layers_with_fetch_mean": float(
                np.mean([t["layers_with_fetch"] for t in ts])
            ),
            "fetch_mb_per_token_mean": bytes_total / len(ts) / 1e6,
        }

    drop_word = "would-drop" if counterfactual else "dropped"
    print(
        f"\npolicy={lat.get('policy')} mass_budget={mass_budget} "
        f"min_k={min_k} tokens={len(tokens)} "
        f"p50/p90/p99 = {p50:.1f}/{p90:.1f}/{p99:.1f} ms"
    )
    header = (
        f"{'band':8} {'tok':>5} {'itl':>7} {drop_word:>10} {'mustkeep':>9} "
        f"{'minK':>6} {'flat':>6} {'guard':>6} {'mk%':>6} {'depth':>6} "
        f"{'lyrs':>5} {'MB/tok':>7}"
    )
    print(header)
    for b in BANDS:
        r = report["bands"][b]
        if not r["tokens"]:
            continue
        m = r["per_token_mean"]
        print(
            f"{b:8} {r['tokens']:>5} {r['itl_ms_mean']:>7.1f} "
            f"{m['dropped']:>10.2f} {m['mustkeep_mass']:>9.2f} "
            f"{m['kept_mink']:>6.2f} {m['kept_flat']:>6.2f} "
            f"{m['kept_guard']:>6.2f} {r['fetched_share_mustkeep']:>6.1%} "
            f"{r['fetch_depth_max_mean']:>6.2f} "
            f"{r['layers_with_fetch_mean']:>5.1f} "
            f"{r['fetch_mb_per_token_mean']:>7.1f}"
        )

    tail = report["bands"]["p99_100"]
    if tail.get("tokens"):
        mk = tail["fetched_share_mustkeep"]
        print("\ntail (>=p99) verdict inputs:")
        print(
            f"  fetched experts/token {tail['fetched_per_token_mean']:.1f}, "
            f"must-keep share {mk:.1%}, droppable-but-kept share "
            f"{tail['fetched_share_droppable_kept']:.1%}, max in-layer fetch "
            f"depth {tail['fetch_depth_max_mean']:.2f}, "
            f"{tail['fetch_mb_per_token_mean']:.1f} MB/token"
        )

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
