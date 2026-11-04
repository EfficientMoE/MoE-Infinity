# exp699 — beyond-pole existence test

Pre-registered before run. Freeze hash after commit.

## Chain
exp698: eti_2020 beat the ssr pole +1.57pp burst at equal rec
(cap4096, d4096). Test survival under finer grids and d-generalization.

## Grids
- A (fine, cap4096, d4096): ssr_10, ring10, eti_2010/2020/3020/4020/
  2030/2040 — annex/ring axes extended past 20%.
- B (d-generalization, caps 2048+4096, d in 256/1024/4096): lcnu,
  ring10, ssr_10, eti_0510, eti_1010, eti_2020.

## Bars (PP)
- PRIMARY BEYOND_POLE_HOLDS: some eti variant burst >= ssr_10 +0.50pp
  AND rec >= ssr_10 - 0.10pp in Grid A AND in Grid B's d4096 legs
  (both caps).
- SECONDARY FLAT_GENERALIZES: eti_0510 exchange vs ring <= 1.0pp/pp
  (dr/db >= -1.0 rate units — exp698's unit bug corrected) at EVERY
  d in Grid B, both caps.
- Tertiary: argmax burst s.t. rec >= ssr_10 - 0.10 in Grid A.
- Verdicts: BEYOND_POLE_HOLDS > FLAT_ONLY > NEITHER.

## Gates
REPL anchors == exp694 d4096 legs (exit 4). DET double-pass
eti_2020@cap4096-d4096 (exit 2). CONSISTENCY class-count signature
across depth legs (exit 3). Exit 0 settled.

## Smoke (mini pre-freeze, N=8192)
Depth legs d256/d1024 execute; saturation arms coincide at mini
(expected; full N differentiates per exp697/698 pattern). eti_0510
d1024 cap2048 = 0.2236/0.7874 (sane).

- STATUS: NEXT
- FREEZE: 839c7e2 (prereg + smoke)

- RUN1: exit 3 — consistency gate as registered was impossible (class totals scale with d; signatures 6+4+4086=4096, 1+0+255=256, 2+0+1022=1024 prove data correct). No results observed before abort (adjudication/metrics print after the gate). GATE V2: within-depth totals equality across caps + sum==d. Bars unchanged.
