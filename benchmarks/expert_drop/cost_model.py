"""Offline fetch-cost model for the tail-policy evaluation (Phase 1).

Predicts decode inter-token-latency (ITL) percentiles from a captured
off-policy routing/residency trace, for the no-drop baseline and for any
policy whose kept-expert set is known. Per decode step the critical-path
model is

    latency_step = base + t_fetch * A + q * B

summed over the step's MoE layers, where A = number of layers needing at
least one non-resident fetch and B = number of extra non-resident fetches
beyond the first in each layer. This encodes "overlapped fetches => one full
fetch latency plus a per-extra queue term" from the plan. The three
parameters (base, t_fetch, q) are calibrated to reproduce the measured
off/fused ITL percentiles; acceptance is predicted p99 within +/-10% of
measured for both anchors.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

QS = (0.5, 0.9, 0.99)


def load_npz(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def group_steps(layer: np.ndarray, seq: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(seq, kind="stable")
    steps: list[list[int]] = []
    cur: list[int] = []
    prev = None
    for pos in order:
        ly = int(layer[pos])
        if prev is not None and ly <= prev:
            steps.append(np.asarray(cur, dtype=np.int64))
            cur = []
        cur.append(int(pos))
        prev = ly
    if cur:
        steps.append(np.asarray(cur, dtype=np.int64))
    return steps


def nonres_counts_off(mask: np.ndarray, resident: np.ndarray) -> np.ndarray:
    return ((mask != 0) & (resident == 0)).sum(axis=1).astype(np.int64)


def nonres_counts_policy(
    mask: np.ndarray,
    weights: np.ndarray,
    resident: np.ndarray,
    *,
    min_k: int,
    mass_budget: float,
    flatness_floor: float,
) -> np.ndarray:
    import torch

    from moe_infinity.runtime.expert_drop import select_expert_drops

    n = mask.shape[0]
    out = np.empty(n, dtype=np.int64)
    mt = torch.from_numpy(np.ascontiguousarray(mask)).bool()
    wt = torch.from_numpy(np.ascontiguousarray(weights.astype(np.float32)))
    rt = torch.from_numpy(np.ascontiguousarray(resident)).bool()
    for i in range(n):
        kept_mask, _kept_w, _counts = select_expert_drops(
            mt[i : i + 1],
            wt[i : i + 1],
            rt[i],
            min_k=min_k,
            mass_budget=mass_budget,
            flatness_floor=flatness_floor,
        )
        kept = kept_mask[0].numpy().astype(bool)
        out[i] = int((kept & (resident[i] == 0)).sum())
    return out


def step_features(steps: list[np.ndarray], counts: np.ndarray):
    a = np.empty(len(steps), dtype=np.float64)
    b = np.empty(len(steps), dtype=np.float64)
    for s, idx in enumerate(steps):
        f = counts[idx]
        a[s] = float((f >= 1).sum())
        b[s] = float(np.maximum(f - 1, 0).sum())
    return a, b


def predict(a: np.ndarray, b: np.ndarray, theta, qs=QS) -> dict:
    base, t_fetch, q = theta
    lat = base + t_fetch * a + q * b
    return {p: float(np.quantile(lat, p)) for p in qs}


def _rel_err(pred: dict, meas: dict) -> float:
    e = 0.0
    n = 0
    for p in QS:
        key = f"p{int(p * 100)}_ms"
        if meas and key in meas and meas[key] > 0:
            e += ((pred[p] - meas[key]) / meas[key]) ** 2
            n += 1
    return e, n


def calibrate(a_off, b_off, a_fused, b_fused, meas_off, meas_fused):
    best = [None, None]

    def _search(base_r, tf_r, q_r):
        for base in base_r:
            for tf in tf_r:
                for q in q_r:
                    theta = (float(base), float(tf), float(q))
                    po = predict(a_off, b_off, theta)
                    e, n = _rel_err(po, meas_off)
                    if a_fused is not None and meas_fused:
                        pf = predict(a_fused, b_fused, theta)
                        ef, nf = _rel_err(pf, meas_fused)
                        e += ef
                        n += nf
                    score = e / max(n, 1)
                    if best[0] is None or score < best[0]:
                        best[0] = score
                        best[1] = theta

    _search(
        np.linspace(0, 150, 31),
        np.linspace(0, 20, 41),
        np.linspace(0, 20, 41),
    )
    b0 = best[1]
    _search(
        np.linspace(max(0.0, b0[0] - 10), b0[0] + 10, 21),
        np.linspace(max(0.0, b0[1] - 1), b0[1] + 1, 21),
        np.linspace(max(0.0, b0[2] - 1), b0[2] + 1, 21),
    )
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--measured-off", required=True)
    ap.add_argument("--measured-fused", default="")
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--mass-budget", type=float, default=0.20)
    ap.add_argument("--flatness-floor", type=float, default=0.5)
    ap.add_argument("--model", default="model")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    d = load_npz(args.npz)
    steps = group_steps(d["layer"], d["seq"])
    counts_off = nonres_counts_off(d["mask"], d["resident"])
    a_off, b_off = step_features(steps, counts_off)

    meas_off = json.load(open(args.measured_off))
    meas_fused = (
        json.load(open(args.measured_fused)) if args.measured_fused else None
    )

    a_fused = b_fused = None
    if meas_fused is not None:
        counts_fused = nonres_counts_policy(
            d["mask"],
            d["weights"],
            d["resident"],
            min_k=args.min_k,
            mass_budget=args.mass_budget,
            flatness_floor=args.flatness_floor,
        )
        a_fused, b_fused = step_features(steps, counts_fused)

    theta, score = calibrate(
        a_off, b_off, a_fused, b_fused, meas_off, meas_fused
    )
    pred_off = predict(a_off, b_off, theta)
    pred_fused = (
        predict(a_fused, b_fused, theta) if a_fused is not None else None
    )

    def _row(label, pred, meas):
        out = {"arm": label}
        for p in QS:
            key = f"p{int(p * 100)}_ms"
            m = meas.get(key) if meas else None
            out[f"pred_{key}"] = round(pred[p], 3)
            out[f"meas_{key}"] = m
            if m:
                out[f"err_{key}_pct"] = round(100.0 * (pred[p] - m) / m, 2)
        return out

    report = {
        "model": args.model,
        "npz": args.npz,
        "num_steps": len(steps),
        "theta_base_ms": round(theta[0], 4),
        "theta_t_fetch_ms": round(theta[1], 4),
        "theta_q_ms": round(theta[2], 4),
        "fit_score": score,
        "rows": [_row("off", pred_off, meas_off)],
    }
    if pred_fused is not None:
        report["rows"].append(_row("fused", pred_fused, meas_fused))

    def _p99_ok(pred, meas):
        if not meas or not meas.get("p99_ms"):
            return None
        return abs(pred[0.99] - meas["p99_ms"]) / meas["p99_ms"] <= 0.10

    report["p99_off_within_10pct"] = _p99_ok(pred_off, meas_off)
    if pred_fused is not None:
        report["p99_fused_within_10pct"] = _p99_ok(pred_fused, meas_fused)
    passed = report["p99_off_within_10pct"] and (
        pred_fused is None or report.get("p99_fused_within_10pct")
    )
    report["PASS"] = bool(passed)

    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
