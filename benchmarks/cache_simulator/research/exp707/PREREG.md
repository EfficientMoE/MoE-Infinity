# exp707 PREREG — cost-layer completion: lo GiB sizing table + 3-trace deployable sizing rule

Program STACK-1001, lane 79990. ANALYSIS tier (pure programmatic transform of
sealed artifacts; no new capture, no simulation, no engine/timing claims —
same tier as exp705; NOT countable toward 1001-native).

## Motivation
exp706 (ledger #229) located qwen-lo CROSS1 = 2816 slots (24.75 GiB), L2 PASS
(CROSS1/WS20 = 0.604 in [0.50, 0.66]) — the convergence law is now 3-for-3
(granite 0.625 / hi 0.528 / lo 0.604). exp706 next_hint: "cost layer
(deployable sizing rule cap~0.6xWS20 + lo GiB table completion)". exp705's lo
cost table stops at cap 2048 with CROSS1/C1/C3 null.

## Frozen inputs (read-only; parsed at runtime AND asserted)
- Sealed exp706 result (~/warp-work/exp706-moei-locross/exp706-locross.json):
  lo policy_grid caps 1792-3328, spread_pp, L1_CROSS1=2816, L3 anchors ok.
- Sealed exp705 result (research/exp705/exp705-costcurves.json): byte
  constants (b_e 4,718,592 / 9,437,184 B), APT (256/384), WS20
  (1228/3396/4662), granite/hi CROSS1 (768/1792), all three existing tables.
- Sealed exp704 lo grid (~/warp-work/exp704-moei-capacitymap/exp704-capacitymap.json):
  lo table rows 768-1536 (below exp706's overlap floor) + spread source.
- Formulas (identical expressions to exp705 run_costcurves.py:130-131):
  premium_B_per_token = round(spread_pp/100 * apt * be, 1);
  lcnu_miss_GiB_per_token = round((1 - lcnu_whole) * apt * be / 2**30, 5).

## Pre-registered gates (frozen before any real run)
- S1 ANCHOR-EXACT (hard, exit 5): every granite/hi row mirrored byte-equal to
  exp705's table (same dicts, same rounding); lo rows at 1792/2048 recomputed
  from exp706 grid == exp705's rows exactly (proves exp706 grid and exp704
  agree through the shared formula — third determinism witness).
- S2 LAW-BAND: ratios {granite, hi, lo} = CROSS1/WS20 all inside [0.50, 0.66];
  band width (max-min) <= 0.10.
- S3 C1-COMPLETION: lo CROSS1_GiB == 2816 * 9437184 / 2**30 exactly (24.75);
  lo C1_hardware = (CROSS1_GiB >= 15.0); granite/hi C1 mirror exp705.
- S4 SAVINGS-IDENTITY: per trace, savings_vs_WS20 == 1 - CROSS1/WS20 to 1e-9
  (definitional identity — catches any arithmetic/table bug).
- S5 DETERMINISM: in-process double run -> byte-equal JSON serialization.
- Verdict: SIZRULE_COMPLETE if S1-S5 all pass (exit 0); exit 5 anchor, exit 3
  structure (missing keys/shape), exit 2 determinism.
- Output sizing_rule block: per-trace {WS20, CROSS1_slots, CROSS1_GiB,
  WS20_GiB, ratio, savings}, rule band [min, max] ratios, deployable rule
  cap ~ 0.6 x WS20, GiB per 1000 WS20 keys per family.

## Scope honesty
Analysis tier over three sealed artifacts; the only new numbers are lo table
rows 2176-3328 (formula applied to sealed exp706 measurements), lo
C1/C2-completion, and the consolidated rule table. No new policy runs, no
tuning, nothing measured twice. 3 ratio points is a band claim, not a fitted
law; per-family variance (hi 0.528 vs lo 0.604, same model) is reported, not
explained.

## Disclosures
- Smoke: synthetic mini-tables through the SAME row-mirror/row-build/gate
  machinery (validates formula/rounding/anchor-compare code paths); disclosed
  in LEDGER.md before freeze.
- If the real run crashes pre-measurement, one bounded relaunch allowed only
  if runner sha still matches ARTIFACT.sha256 (standing rule).
