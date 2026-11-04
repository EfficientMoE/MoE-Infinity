# exp701 — ridge decomposition (pairwise + off-grid probes)

Pre-registered before run. Freeze hash after commit.

## Chain
exp700: ridge at d~2xcap, max 11.91pp @d4096/cap2048. Decompose.

## Part 1 (analysis of sealed exp700 JSON — no simulation)
Pairwise |burst_i - burst_j| for all 6 arm pairs across the 15 grid
cells. RESULT (recorded pre-part-2): ring10-vs-ssr_10 is the max-min
pair in 13/15 cells (87%) — PAIR_DOMINANT (>=70%) PASS. The crest
cell (11.91pp) itself is ring10-vs-eti_2020 (eti_2020 burst 0.2148
vs ring 0.0957 at cap2048 — matches exp698 exactly; its rec cost
0.8518 vs ssr 0.8686 is why exp699's beyond_b bar failed there).

## Part 2 (this runner)
Off-grid probes: d in {3072, 5120} x cap in {1536, 2048, 2560} x 4
arms. Is 11.91pp the crest (all probes <= crest+0.5pp) or does the
ridge keep rising (RIDGE_RISING)?

## Bars (PP)
- PRIMARY PAIR_DOMINANT: >=70% of 15 cells — ring-vs-ssr 13/15 PASS
  (part 1, sealed data).
- SECONDARY RIDGE_CREST: all off-grid spreads <= 11.91+0.5pp, else
  RIDGE_RISING.

## Gates
REPL: recomputed d4096 x cap2048 leg (4 arms) == exp700 exactly
(exit 4). DET double-pass eti_2020@d3072/cap2048 (exit 2).
CONSISTENCY within-depth totals equal across caps + sum == d
(exit 3). Exit 0 settled.

## Smoke
Runner imports and repl-leg logic exercised at mini scale in-line
(part-1 numbers reproduced from sealed JSON before any new compute).

- STATUS: NEXT
