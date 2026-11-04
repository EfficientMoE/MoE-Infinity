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

- STATUS: NEXT
