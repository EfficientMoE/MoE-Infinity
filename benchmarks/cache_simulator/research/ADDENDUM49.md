# ADDENDUM 49 — exp705: cost-integration, byte-denominated cache-sizing curves

Verdict (repaired track, r3): **COST_CURVES** — C4 anchors/determinism ALL
PASS; C1 PASS; C2 **FAIL (honest negative, kept)**; C3 PASS. Frozen-track run
returned ANCHOR-FAIL exit 5 (preserved verbatim: `exp705-costcurves.frozen-track.json`);
both repairs disclosed in DEVIATIONS.md (D1 loader shape, D2 anchor definition),
bars C1-C4 unchanged throughout. Exit 0. 1.3s.

## Byte model (frozen, HF configs fetched 2026-11-04, bf16 gate+up+down)
- granite-3.1-3b-a800m: b_e = 3*1536*512*2 = 4,718,592 B = 4.5 MiB/expert;
  E*L = 1280 experts, total 5.625 GiB.
- Qwen3-30B-A3B: b_e = 3*2048*768*2 = 9,437,184 B = 9.0 MiB/expert;
  E*L = 6144 experts, total 54.0 GiB.
- Constants exact (C4a); APT = L*k exact 3/3 (256/384); decode-N structure
  exact 3/3 (638,208 / 786,432 / 638,208 — matches registered constants).

## C1 hardware mapping — PASS
- granite CROSS1 = 768 slots = **3.375 GiB** (<= 4.0): the 3B-class model's
  policy-sensitive zone ends inside a small GPU.
- qwen hi CROSS1 = 1792 slots = **15.75 GiB** (>= 15.0): the 30B-class trace
  needs >= 15.75 GiB of pure expert cache to exit the policy-sensitive zone.
- lo: NOT_REACHED by 2048 slots => lower bound **18.0 GiB**.

## C2 working-set mechanism — FAIL (negative kept, sharpens the law)
- WS20 (distinct keys in final 20% of decode stream, fresh from frozen
  traces, independently reproduced by a second implementation): granite
  1228, hi 3396, lo 4662.
- |CROSS1 - WS20|/WS20: granite 0.375, hi 0.472 — both outside the frozen
  0.30 band. CROSS1/WS20 = **0.625 (granite) / 0.528 (hi)**.
- Mechanism conclusion (replaces exp704's working-set phrasing): policy
  insensitivity does NOT arrive when the cache holds the steady-state
  working set — best-vs-worst convergence occurs with only ~53-63% of the
  final-20% working set resident. Nor is CROSS1 a fixed fraction of key
  space (60.0% granite vs 29.2% hi). lo's WS20 = 4662 >= 2048 is consistent
  with its NOT_REACHED (C2 lo PASS).

## C3 cliff shape — PASS
- max spread below CROSS1 >= 10x spread at CROSS1, both reached traces
  (granite: 38.2pp hump vs 0.377pp; hi: 24.9pp vs ~0.9pp band).

## C4 anchors/determinism (repaired definition) — PASS
- Burst-spread recompute from sealed policy_grid reproduces sealed
  spread_pp **exactly (0.0 diff)** at probe cells; pre-verified at ALL 27
  trace-cap cells — an independent internal-consistency re-verification of
  exp704's sealed artifact. The frozen-track whole-hit anchor failure was a
  prereg misread of exp704's definition (burst, run_capacitymap.py:138-141),
  not an exp704 defect.
- WS20 double-compute byte-identical 3/3; independent second implementation
  (record-index slicing + dict.fromkeys) agrees exactly.

## Policy-premium pricing (miss-traffic difference best-vs-worst per decoded token)
- granite @128 slots (0.5625 GiB): 344.5 MB/token; @192 (0.84 GiB): 461.9
  MB/token (hump); @768 (CROSS1): 4.55 MB/token; @1024: 1.93 MB/token.
- LCNU absolute miss traffic (whole): granite 0.806 -> 0.033 GiB/token from
  cap 128 -> 1024; sizing table per trace in exp705-costcurves.json.

## Tier & limitations
- **Analysis tier** (pure transform of sealed exp704 data + fresh WS20 on
  frozen traces; no new capture, no engine/timing claims). NOT countable
  toward 1001-native; candidate for coordinator ledger intake.
- Hit-derived ratios carry over from exp704 by construction (per-family
  byte multiplier constant); new content = absolute GiB tables, per-token
  miss-traffic pricing, C1 hardware claim, C2 mechanism test.
- Byte model assumes bf16 dense FFN experts per HF configs; CROSS1 uses the
  sealed burst-spread definition.

Data: exp705-costcurves.json (+ frozen-track ANCHOR-FAIL preserved);
DEVIATIONS.md D1/D2; ARTIFACT-r2/r3 sha256. Freeze 1a8514a -> r3 runner.
