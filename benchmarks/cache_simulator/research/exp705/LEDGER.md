# exp705 LEDGER — cost-integration (byte-denominated cache-sizing curves)

## Status
- 2026-11-04 17:33 ET — PREREG.md written (gates C1-C4 frozen; byte model
  cited: granite-3.1-3b-a800m-instruct + Qwen3-30B-A3B configs, fetched
  2026-11-04). Runner run_costcurves.py written. Namespace exp705 verified
  free (~/warp-work + queue tail; lane-0026 coordination respected).
- Pending: disclosed smoke -> ARTIFACT.sha256 freeze -> real run.

## Bars (frozen in PREREG.md)
- C1 HARDWARE: CROSS1_GiB(granite) <= 4.0 AND CROSS1_GiB(hi) >= 15.0
- C2 WORKING-SET: |CROSS1-WS20|/WS20 <= 0.30 (granite, hi); WS20(lo) >= 2048
- C3 CLIFF: max spread below CROSS1 >= 10x spread at CROSS1 (reached traces)
- C4 ANCHORS/DET: byte constants exact; N decode accesses == frozen N;
  spread recompute 3-cell match <= 0.01; WS20 double-compute byte-identical
- 2026-11-04 17:35 ET — disclosed smoke PASSED (SMOKE_OK; ws20 det, val=1000 as predicted). Freeze: shasum -a 256 run_costcurves.py -> ARTIFACT.sha256 BEFORE real run.
- 2026-11-04 22:4xZ — REAL RUN EXECUTED (this fire):
  - Frozen track (r1, original runner): ANCHOR-FAIL exit 5 — C4c whole-hit
    anchor mismatch (prereg misread sealed exp704 spread definition);
    output preserved exp705-costcurves.frozen-track.json. D1 loader-shape
    crash (exit 1, zero measurements) repaired first as r2.
  - Repaired track (r3): COST_CURVES exit 0. C1 PASS (3.375 GiB / 15.75 GiB
    / lo bound 18.0 GiB); C2 FAIL honest negative (CROSS1/WS20 0.625/0.528 —
    convergence at ~53-63% of final-20% working set, not full WS, not fixed
    keyspace fraction); C3 PASS; C4 PASS (N exact 3/3; burst anchor 0.0 diff
    all probe cells, pre-verified all 27 cells; WS20 det 3/3 + independent
    second implementation exact). Tier: analysis (NOT 1001-native-countable).
    ADDENDUM49.md. See DEVIATIONS.md D1/D2.
