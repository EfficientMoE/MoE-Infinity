"""Phase 2 offline policy simulator + coarse Pareto (six tail-drop candidates).

Operates on a captured off-policy trace (trace_parser .npz) and implements the
six candidates from 2026-10-02-tail-targeting-drop-policies.md:

  mass : shipped mass-budget tail drop (= select_expert_drops)
  A    : deadline-aware high-mass drop (drop costliest non-resident past L_us)
  B    : nearest-resident mass redirect (move high-mass miss onto a resident)
  C    : head-budget hybrid (mass drop + one bounded high-mass drop)
  D    : residency-adaptive budget (budget grows with this token's miss count)
  E    : deadline partial-wait (keep only experts that arrive by L_us)

For each policy/knob it reports, aggregated over decode steps: a COARSE
predicted p99 from the off-calibrated cost model (REFERENCE ONLY -- the Phase-1
gate showed the model is not a +/-10% oracle for tail-acting arms), routing
total-variation fidelity proxies, and drop/redirect counts.

C and D reduce EXACTLY to select_expert_drops when their extra lever is off
(head_budget=0 / slope=0). verify_parity() asserts this and that the vectorized
base_drop_np matches the scalar reference select_expert_drops.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:
    from benchmarks.expert_drop import cost_model as cm
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    import cost_model as cm


def base_drop_np(W, M, R, *, min_k, mass_budget, flatness_floor):
    """Vectorized port of select_expert_drops with per-row residency."""
    W = W.astype(np.float64)
    M = M.astype(bool)
    R = R.astype(bool)
    n, ne = W.shape
    if mass_budget == 0:
        return M.copy(), W.astype(np.float32), np.zeros(n, dtype=bool)
    sel_count = M.sum(1)
    Wm = np.where(M, W, 0.0)
    max_w = np.where(M, W, -np.inf).max(1)
    min_w = np.where(M, W, np.inf).min(1)
    finite_ok = np.isfinite(Wm).all(1)
    nonneg_ok = ~((Wm < 0).any(1))
    base_ok = (sel_count > min_k) & finite_ok & nonneg_ok & (max_w > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(max_w > 0, min_w / max_w, np.inf)
    flat_bypass = base_ok & (ratio > flatness_floor)
    eligible = base_ok & ~flat_bypass

    candidates = M & ~R
    key = np.where(candidates, W, np.inf)
    key_rev = key[:, ::-1]
    order_rev = np.argsort(key_rev, axis=1, kind="stable")
    sorted_ids = (ne - 1) - order_rev
    sorted_is_cand = np.take_along_axis(candidates, sorted_ids, axis=1)
    sorted_w = np.where(
        sorted_is_cand, np.take_along_axis(W, sorted_ids, axis=1), 0.0
    )
    cum = np.cumsum(sorted_w, axis=1)
    budget_count = (sorted_is_cand & (cum <= mass_budget)).sum(1)
    allowed = np.maximum(sel_count - min_k, 0)
    dropped_here = np.minimum(budget_count, allowed)
    dropped_here = np.where(eligible, dropped_here, 0)

    position = np.arange(ne)[None, :]
    drop_sorted = position < dropped_here[:, None]
    drop_mask = np.zeros_like(M)
    np.put_along_axis(drop_mask, sorted_ids, drop_sorted, axis=1)
    changed = dropped_here > 0
    new_M = M & ~drop_mask
    stripped = np.where(drop_mask, 0.0, W)
    surv_sum = np.where(new_M, stripped, 0.0).sum(1, keepdims=True)
    safe = np.where(surv_sum == 0, 1.0, surv_sum)
    renormed = np.where(new_M & (surv_sum != 0), stripped / safe, stripped)
    out_M = np.where(changed[:, None], new_M, M)
    out_W = np.where(changed[:, None], renormed, W)
    return out_M, out_W.astype(np.float32), changed


def _renorm(W, M):
    W = np.where(M, W, 0.0).astype(np.float64)
    s = W.sum(1, keepdims=True)
    safe = np.where(s == 0, 1.0, s)
    return np.where(M & (s != 0), W / safe, W).astype(np.float32)


def policy_off(W, M, R, byte, **_):
    return M.copy(), W.copy()


def policy_mass(W, M, R, byte, *, min_k, mass_budget, flatness_floor, **_):
    km, kw, _ = base_drop_np(
        W, M, R, min_k=min_k, mass_budget=mass_budget, flatness_floor=flatness_floor
    )
    return km, kw


def policy_C(W, M, R, byte, *, min_k, mass_budget, flatness_floor, head_budget, **_):
    km, kw, _ = base_drop_np(
        W, M, R, min_k=min_k, mass_budget=mass_budget, flatness_floor=flatness_floor
    )
    if head_budget <= 0:
        return km, kw
    kept_nonres = km & ~R.astype(bool)
    nonres_count = kept_nonres.sum(1)
    kept_count = km.sum(1)
    w_masked = np.where(kept_nonres, kw.astype(np.float64), np.inf)
    sole_idx = np.argmin(w_masked, axis=1)
    sole_w = np.take_along_axis(kw, sole_idx[:, None], axis=1)[:, 0]
    act = (nonres_count == 1) & (kept_count > min_k) & (sole_w <= head_budget)
    if act.any():
        drop = np.zeros_like(km)
        rows = np.where(act)[0]
        drop[rows, sole_idx[rows]] = True
        km = km & ~drop
        kw = _renorm(kw, km)
    return km, kw


def policy_D(W, M, R, byte, *, min_k, mass_budget, flatness_floor, slope, miss0, **_):
    miss = (M.astype(bool) & ~R.astype(bool)).sum(1)
    eff = np.clip(mass_budget + slope * np.maximum(miss - miss0, 0), 0.0, 1.0)
    km = M.astype(bool).copy()
    kw = W.copy()
    for b in np.unique(eff):
        rows = np.where(eff == b)[0]
        m2, w2, _ = base_drop_np(
            W[rows], M[rows], R[rows],
            min_k=min_k, mass_budget=float(b), flatness_floor=flatness_floor,
        )
        km[rows] = m2
        kw[rows] = w2
    return km, kw


def policy_A(W, M, R, byte, *, min_k, mass_budget, flatness_floor, l_us, t_fetch_ms,
            q_ms, **_):
    km, kw, _ = base_drop_np(
        W, M, R, min_k=min_k, mass_budget=mass_budget, flatness_floor=flatness_floor
    )
    km = km.copy()
    kw = kw.copy()
    budget_ms = l_us / 1000.0
    R = R.astype(bool)
    nonres = km & ~R
    cost = byte.astype(np.float64)[None, :] if byte.ndim == 1 else byte.astype(np.float64)
    for _ in range(int(M.shape[1])):
        k = nonres.sum(1)
        proj = np.where(k > 0, t_fetch_ms + q_ms * np.maximum(k - 1, 0), 0.0)
        act = (proj > budget_ms) & (km.sum(1) > min_k) & (k > 0)
        if not act.any():
            break
        key = np.where(nonres, cost + 1e-9 * kw.astype(np.float64), -np.inf)
        victim = np.argmax(key, axis=1)
        drop = np.zeros_like(km)
        rows = np.where(act)[0]
        drop[rows, victim[rows]] = True
        km = km & ~drop
        nonres = km & ~R
    kw = _renorm(kw, km)
    return km, kw


def policy_B(W, M, R, byte, *, min_k, mass_budget, flatness_floor, redirect_min, **_):
    km, kw, _ = base_drop_np(
        W, M, R, min_k=min_k, mass_budget=mass_budget, flatness_floor=flatness_floor
    )
    R = R.astype(bool)
    km = km.copy()
    kw = kw.astype(np.float64).copy()
    kept_nonres = km & ~R
    move = kept_nonres & (kw >= redirect_min)
    kept_res = km & R
    tgt_key = np.where(kept_res, kw, -np.inf)
    target = np.argmax(tgt_key, axis=1)
    has_res = kept_res.any(1)
    rows = np.where(move.any(1) & has_res)[0]
    for r in rows:
        tgt = target[r]
        moved = kw[r, move[r]].sum()
        kw[r, move[r]] = 0.0
        km[r, move[r]] = False
        kw[r, tgt] += moved
    return km, kw.astype(np.float32)


def policy_E(W, M, R, byte, *, min_k, l_us, t_fetch_ms, **_):
    R = R.astype(bool)
    M = M.astype(bool)
    n, ne = W.shape
    t_eff = max(t_fetch_ms, 0.05)
    n_arr = int(max(0, np.floor((l_us / 1000.0) / t_eff)))
    nonres = M & ~R
    order = np.argsort(-np.where(nonres, W, -np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(ne), (n, ne)).copy(), 1)
    arrived = nonres & (rank < n_arr)
    km = (M & R) | arrived
    deficit = np.maximum(min_k - km.sum(1), 0)
    if deficit.any():
        avail = M & ~km
        av_order = np.argsort(-np.where(avail, W, -np.inf), axis=1, kind="stable")
        for r in np.where(deficit > 0)[0]:
            add = av_order[r, : deficit[r]]
            km[r, add] = True
    kw = _renorm(W, km)
    return km, kw


POLICIES = {
    "off": policy_off,
    "mass": policy_mass,
    "A": policy_A,
    "B": policy_B,
    "C": policy_C,
    "D": policy_D,
    "E": policy_E,
}


def tv_distance(P_w, P_m, Q_w, Q_m):
    P = _renorm(P_w, P_m).astype(np.float64)
    Q = Q_w.astype(np.float64)
    qs = np.where(Q_m, Q, 0.0).sum(1, keepdims=True)
    Q = np.where(Q_m & (qs != 0), np.where(Q_m, Q, 0.0) / np.where(qs == 0, 1, qs), 0.0)
    return 0.5 * np.abs(P - Q).sum(1)


def aggregate(trace, steps, theta, name, policy, params, pred_off):
    W, M, R = trace["weights"], trace["mask"], trace["resident"]
    nl = int(trace["num_layers"])
    byte = trace["byte_size"].astype(np.float64)
    layer = trace["layer"]
    byte_per_row = byte[np.clip(layer, 0, nl - 1)]
    km, kw = policy(W, M, R, byte_per_row, **params)
    kept_nonres = (km & ~R.astype(bool)).sum(1).astype(np.int64)
    a, b = cm.step_features(steps, kept_nonres)
    pred = cm.predict(a, b, theta)
    tv = tv_distance(W, M.astype(bool), kw, km)
    dropped = (M.astype(bool) & ~km).sum()
    orig_routed = M.astype(bool).sum()
    return {
        "policy": name,
        "params": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in params.items()
                   if k not in ("t_fetch_ms", "q_ms")},
        "pred_p99_ms": round(pred[0.99], 2),
        "pred_p99_ratio_off": round(pred[0.99] / pred_off[0.99], 3),
        "pred_p50_ms": round(pred[0.5], 2),
        "tv_mean": round(float(tv.mean()), 4),
        "tv_p99": round(float(np.quantile(tv, 0.99)), 4),
        "experts_dropped": int(dropped),
        "frac_experts_dropped": round(float(dropped / max(orig_routed, 1)), 4),
        "tokens_changed": int((tv > 1e-6).sum()),
    }


def verify_parity(trace, min_k, flatness_floor, sample=4000, seed=0):
    import torch

    from moe_infinity.runtime.expert_drop import select_expert_drops

    W, M, R = trace["weights"], trace["mask"], trace["resident"]
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    for budget in (0.05, 0.1, 0.2):
        km, kw, _ = base_drop_np(
            W[idx], M[idx], R[idx],
            min_k=min_k, mass_budget=budget, flatness_floor=flatness_floor,
        )
        mt = torch.from_numpy(M[idx].astype(np.uint8)).bool()
        wt = torch.from_numpy(W[idx].astype(np.float32))
        rt = torch.from_numpy(R[idx].astype(np.uint8)).bool()
        for j in range(len(idx)):
            rm, rw, _ = select_expert_drops(
                mt[j : j + 1], wt[j : j + 1], rt[j],
                min_k=min_k, mass_budget=budget, flatness_floor=flatness_floor,
            )
            ref_m = rm[0].numpy().astype(bool)
            if not np.array_equal(ref_m, km[j]):
                raise AssertionError(f"base_drop parity mask mismatch budget={budget} row={j}")
            ref_w = rw[0].numpy()
            if not np.allclose(ref_w[ref_m], kw[j][km[j]], atol=1e-5):
                raise AssertionError(f"base_drop parity weight mismatch budget={budget} row={j}")
    byte = np.zeros(W.shape[1])
    base = policy_mass(W, M, R, byte, min_k=min_k, mass_budget=0.2,
                       flatness_floor=flatness_floor)
    c0 = policy_C(W, M, R, byte, min_k=min_k, mass_budget=0.2,
                  flatness_floor=flatness_floor, head_budget=0.0)
    d0 = policy_D(W, M, R, byte, min_k=min_k, mass_budget=0.2,
                  flatness_floor=flatness_floor, slope=0.0, miss0=0)
    assert np.array_equal(base[0], c0[0]), "C(head_budget=0) != base"
    assert np.array_equal(base[0], d0[0]), "D(slope=0) != base"
    return len(idx)


def build_configs(theta):
    tf, q = theta[1], theta[2]
    cfgs = []
    cfgs.append(("off", {}))
    for mb in (0.05, 0.10, 0.20):
        cfgs.append(("mass", {"mass_budget": mb}))
    for hb in (0.0, 0.05, 0.10, 0.15, 0.20):
        cfgs.append(("C", {"mass_budget": 0.20, "head_budget": hb}))
    for sl in (0.0, 0.01, 0.02, 0.05):
        cfgs.append(("D", {"mass_budget": 0.05, "slope": sl, "miss0": 2}))
    for lus in (1e9, 2000.0, 1000.0, 500.0):
        cfgs.append(("A", {"mass_budget": 0.20, "l_us": lus,
                           "t_fetch_ms": tf, "q_ms": q}))
    for rm in (0.05, 0.10, 0.20):
        cfgs.append(("B", {"mass_budget": 0.20, "redirect_min": rm}))
    for lus in (1e9, 600.0, 300.0, 150.0):
        cfgs.append(("E", {"l_us": lus, "t_fetch_ms": tf}))
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--model", default="model")
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--flatness-floor", type=float, default=0.5)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    trace = cm.load_npz(args.npz)
    calib = json.load(open(args.calib))
    theta = (calib["theta_base_ms"], calib["theta_t_fetch_ms"], calib["theta_q_ms"])
    steps = cm.group_steps(trace["layer"], trace["seq"])

    checked = verify_parity(trace, args.min_k, args.flatness_floor)
    print(f"parity OK on {checked} sampled rows (base==select_expert_drops; C0,D0==base)")

    W, M, R = trace["weights"], trace["mask"], trace["resident"]
    off_nonres = (M.astype(bool) & ~R.astype(bool)).sum(1).astype(np.int64)
    a0, b0 = cm.step_features(steps, off_nonres)
    pred_off = cm.predict(a0, b0, theta)

    rows = []
    for name, extra in build_configs(theta):
        params = dict(min_k=args.min_k, flatness_floor=args.flatness_floor)
        params.update(extra)
        rows.append(
            aggregate(trace, steps, theta, name, POLICIES[name], params, pred_off)
        )

    report = {
        "model": args.model,
        "npz": args.npz,
        "theta": {"base_ms": theta[0], "t_fetch_ms": theta[1], "q_ms": theta[2]},
        "pred_p99_off_ms": round(pred_off[0.99], 2),
        "note": "pred_p99 is the off-calibrated cost model; REFERENCE ONLY per "
                "the Phase-1 fork (not a +/-10% oracle for tail-acting arms). "
                "Fidelity = routing total-variation proxy; true greedy-agreement "
                "comes from the real C+D A/B gate.",
        "rows": rows,
    }
    print(json.dumps(report, indent=2))
    if args.out_json:
        json.dump(report, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
