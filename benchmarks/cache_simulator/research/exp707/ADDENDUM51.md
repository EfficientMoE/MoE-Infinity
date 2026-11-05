# ADDENDUM 51 — exp707: cost-layer completion (lo GiB sizing table + 3-trace deployable sizing rule)

Verdict: **SIZRULE_COMPLETE** — S1 ANCHOR-EXACT (granite/hi tables byte-equal to
sealed exp705; lo rows at 1792/2048 recomputed from the sealed exp706 grid equal
exp705's rows exactly), S2 LAW-BAND (all three ratios in [0.50, 0.66], band
width 0.0977 <= 0.10), S3 C1-COMPLETION (lo CROSS1_GiB = 24.75 exact;
C1 3/3), S4 SAVINGS-IDENTITY, S5 DETERMINISM (in-process double run byte-equal).
Exit 0, analysis tier (~4 s pure transform). Freeze commit 36f665a preceded all
run artifacts.

## Deployable sizing rule (the cost layer's output)
| trace | WS20 slots | WS20 GiB | CROSS1 slots | CROSS1 GiB | CROSS1/WS20 | cache savings |
|---|---|---|---|---|---|---|
| granite | 1228 | 5.396 | 768 | 3.375 | 0.6254 | 37.5% |
| hi | 3396 | 29.85 | 1792 | 15.75 | 0.5277 | 47.2% |
| lo | 4662 | 40.97 | 2816 | 24.75 | 0.6040 | 39.6% |

- Rule: **cap ~ 0.6 x WS20 expert slots** (observed band 0.528-0.625;
  width 0.098). Sizing a re-access cache at
  ~0.6 x the final-20% working set captures best-vs-worst policy convergence on
  all three traces measured.
- Byte cost: 5.27 GiB per 1000 WS20 keys (30B-class,
  b_e 9.0 MiB); 2.64 GiB per 1000 WS20 keys (3B-class, b_e 4.5 MiB).
- Same-model variance is real (hi 0.528 vs lo 0.604, both
  qwen-30b traces): the rule is a band over 3 points, not a fitted constant.

## Completed lo cost table (768-3328 slots; rows <1792 mirrored exp705; 1792-3328 = sealed exp706 grid through the frozen formula)
| cap | cache GiB | spread pp | premium MB/token | lcnu miss GiB/token |
|---|---|---|---|---|
| 768 | 6.75 | 21.042 | 762.5 | 1.51212 |
| 896 | 7.875 | 20.213 | 732.5 | 1.38116 |
| 1024 | 9 | 19.562 | 708.9 | 1.26742 |
| 1152 | 10.12 | 16.763 | 607.5 | 1.16094 |
| 1280 | 11.25 | 14.172 | 513.6 | 1.05935 |
| 1408 | 12.38 | 13.095 | 474.5 | 0.96390 |
| 1536 | 13.5 | 10.851 | 393.2 | 0.87709 |
| 1792 | 15.75 | 4.211 | 152.6 | 0.72475 |
| 2048 | 18 | 2.116 | 76.7 | 0.59772 |
| 2176 | 19.12 | 1.714 | 62.1 | 0.54332 |
| 2304 | 20.25 | 1.553 | 56.3 | 0.49568 |
| 2432 | 21.38 | 1.321 | 47.9 | 0.45054 |
| 2560 | 22.5 | 1.076 | 39.0 | 0.40849 |
| 2816 | 24.75 | 0.717 | 26.0 | 0.33424 |
| 3072 | 27 | 0.156 | 5.7 | 0.27328 |
| 3328 | 29.25 | 0.055 | 2.0 | 0.22308 |

Policy premium decays 21.042pp -> 0.055pp;
below CROSS1 (2816 slots) the policy choice still moves burst hit rates by up
to 21.0pp; at the law point spread is 0.717pp.

## Anchors & determinism (third witness, now cross-artifact)
- S1 recomputation: lo@1792/2048 rebuilt from the sealed exp706 grid via the
  frozen formula (identical expressions to exp705 run_costcurves.py:130-131) are
  byte-equal to exp705's published rows; the same rows rebuilt from the sealed
  exp704 grid are byte-equal too (verified programmatically here and by the
  follow-up controller's independent adjudicator). Three sealed artifacts —
  exp704, exp705, exp706 — agree through one shared formula: cross-artifact
  determinism witnessed a third time.
- Constants parsed at runtime and asserted: b_e 4,718,592/9,437,184 B;
  APT 256/384; WS20 1228/3396/4662
  (exp706 WS20_lo cross-checked equal).
- S3: 2816 x 9,437,184 / 2^30 = 24.7500 GiB exactly; lo C1_hardware = (>= 15.0 GiB) true.
- Integrity: runner sha256 7f1fbb71f35cf0f8... matches ARTIFACT.sha256.

## Tier & limitations
- Analysis tier: pure programmatic transform of sealed exp704/exp705/exp706
  artifacts — NOT countable toward 1001-native. The only new numbers are lo
  table rows 2176-3328 (frozen formula applied to sealed exp706 measurements),
  lo C1 completion, and the consolidated rule table.
- 3 ratio points is a band claim, not a fitted law; per-family variance
  (hi 0.528 vs lo 0.604) is reported, not explained.
- No data beyond 3328 slots (0.714 x WS20(lo)): whether convergence
  persists, flattens, or reverses between 0.77-1.0 x WS20 is unmeasured — the
  designated next probe (lo grid 3584-4662).

Data: exp707/exp707-sizrule.json; PREREG.md/ARTIFACT.sha256/LEDGER.md/run_sizrule.py
in exp707/ (freeze commit 36f665a); sealed refs exp704-capacitymap.json,
exp705-costcurves.json, exp706-locross.json. Follow-up controller re-adjudicated
every gate programmatically from primary sources with an independently written
adjudicator (39/39 checks, runner sha intact, never re-executed): verdict
SIZRULE_COMPLETE stands.
