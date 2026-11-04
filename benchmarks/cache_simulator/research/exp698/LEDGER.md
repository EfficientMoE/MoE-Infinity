# exp698 — ETI fraction sweep: trace the achievable frontier curve

Pre-registered before run. Freeze hash after commit.

## Chain
exp697: ETI(10,10) strict interior point, exchange ~0.3pp rec per pp
burst vs ring (5-10x flatter than lcnu->ring). Sweep: annex_frac x
ring_frac in {0.05, 0.10, 0.20}^2 at caps 512/1024/2048/4096, d4096,
N full, anchors lcnu/ring10/ssr_10, exp695 classifier instrumented.

## Bars (PP)
- HARM gate (per variant, all caps; failing variants excluded from
  PRIMARY): rec >= lcnu - 0.10 AND burst >= ring10 - 0.50.
- DOMINANT_POINT: any variant burst >= ring10 + 0.10 AND rec >=
  ring10 + 0.10 at cap 1024 or 2048.
- FLAT_FRONTIER: eligible variant with (rec loss)/(burst gain) vs
  ring <= 1.0pp/pp at BOTH 1024 and 2048.
- else STEEPENS.

## Gates
REPL anchors == exp694 exactly (d4096 all caps; exit 4).
REPL97 eti_1010@2048 == exp697's eti_10 cell exactly (exit 2 —
cross-experiment determinism).
CONSIST class counts identical (exit 3). Exit 0 settled.

## Smoke (mini pre-freeze)
eti_1020 rec 0.6589 vs ring 0.6536 (fraction plumbing distinct);
eti_0505/eti_2005 == ring at mini (earn event needs full-run dynamic
range — same as exp697 mini behavior). Anchor repls exact.

- STATUS: DONE (exit 0; code verdict superseded by binding-spec re-adjudication, disclosed)
- **Result:**
```
CODE VERDICT: STEEPENS — WRONG (unit bug in flat check, self-caught at adjudication review; disclosed) | BINDING-SPEC VERDICT: FLAT_FRONTIER | det True | REPL anchors == exp694 exact | REPL97 eti_1010@2048 == exp697 exact (cross-experiment determinism PASS) | consistency PASS | elapsed 1441s | freeze 71ef7db | UNIT BUG: code threshold dr/db >= -0.01 implements 0.01pp-per-pp (100x stricter than registered 1.0pp-per-pp = dr/db >= -1.0); sealed JSON verdict field says STEEPENS, NOT edited; correction disclosed here per programmatic-addendum discipline | FLAT carriers (harm-eligible only: eti_0510, eti_1010): eti_0510 exchange vs ring = 0.45pp/pp @1024 (0.0552/0.7869 vs 0.0488/0.7898), 0.17 @2048 (0.1047/0.8801 vs 0.0957/0.8816); eti_1010 = 0.50 @1024, 0.20 @2048 | NO DOMINANT_POINT (no variant beats ring both axes — honest negative) | STRUCTURE: annex_frac drives burst (monotone), ring_frac drives rec; 2-param family traces ring->ssr interior | BEYOND-POLE: cap4096 eti_2020 0.4285/0.9517 vs ssr_10 0.4128/0.9514 — beats the ssr pole +1.57pp burst at +0.03pp rec; also eti_2010 0.4187/0.9521 beats ssr burst at rec above ring-family mean | exp699 registered: beyond-pole existence test (finer annex at cap4096 + d-generalization of the flat exchange)
```

- FREEZE: 71ef7db (prereg + smoke)
