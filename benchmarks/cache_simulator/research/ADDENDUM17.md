## Addendum 17 — exp668: CostFit robustness is a region, not a point (VERDICT: ROBUST, 9/9)

exp667 (Addendum 16) passed at one parameter point. exp668 sweeps the two
control parameters 4x each way around it — MARGIN {0.0025, 0.005, 0.010} x
GAMMA {0.00125, 0.0025, 0.005}, HALF_LIFE fixed 32768 — over both traces at
all canonical caps, with the frozen pass rule: >=7/9 cells passing
H1 (near-dominance 0.5pp, both traces), H2 (<=6 switches), H3 (loop==property
+ zero-switch caps equal baseline CSVs exactly).

**Result: ROBUST — 9/9 cells pass.**
Two full grid runs byte-identical (determinism asserted in runner).
Center cell (MARGIN 0.005, GAMMA 0.0025) reproduces the exp667 published
CSV rates exactly at all 15 caps (cross-artifact anchor).

Cell summary (worst deficit pp / max switches):

| MARGIN\GAMMA | 0.00125 | 0.0025 | 0.005 |
|---|---|---|---|
| 0.0025 | PASS: 0.115pp/sw6, 0.065pp/sw2 | PASS: 0.072pp/sw4, 0.070pp/sw1 | PASS: 0.086pp/sw3, 0.070pp/sw1 |
| 0.005 | PASS: 0.096pp/sw4, 0.112pp/sw1 | PASS: 0.113pp/sw4, 0.112pp/sw1 | PASS: 0.116pp/sw3, 0.112pp/sw1 |
| 0.01 | PASS: 0.332pp/sw2, 0.220pp/sw1 | PASS: 0.332pp/sw2, 0.220pp/sw1 | PASS: 0.332pp/sw2, 0.220pp/sw1 |

Failure-mode boundary map: worst deficit 0.332pp at MARGIN 0.01
(qwen30b cap 4096) — the exp665 late-switch direction; max switches 6
at MARGIN 0.0025/GAMMA 0.00125 (qwen30b cap 1024)
— the exp666 oscillation direction, exactly at the H2 boundary. The pass
region is bounded by both refuted failure modes on opposite edges and the
entire tested 4x4 region stays inside: CostFit needs no per-deployment
parameter tuning across this range.

Artifacts: exp668-grid.json (per-cell per-cap rates, switches, H-flags);
prereg frozen 81079e60 before the grid ran; instance-attribute cell override
(sealed exp667 policy source unedited, sha 4bfff5bd unchanged).
