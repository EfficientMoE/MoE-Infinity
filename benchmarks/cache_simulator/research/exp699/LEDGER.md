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

- STATUS: DONE (run 2 exit 0; run 1 gate-error disclosed above)
- **Result:**
```
VERDICT: NEITHER (run 2, gate v2 freeze c0c7dfe) | det True | REPL anchors == exp694 all legs | cross-exp determinism: eti_2020@cap4096 == exp698 exact (0.4285/0.9517); consistency v2 PASS (within-depth totals equal, sum==d) | elapsed 2230s | RUN 1 DISCLOSURE: exit 3 on impossible cross-depth totals gate (totals scale with d; 6+4+4086=4096, 1+0+255=256, 2+0+1022=1024 proved data correct; abort pre-adjudication, no results observed, data discarded — no selection possible) | BEYOND-POLE: real but NARROW — Grid A cap4096/d4096: 6 variants beat ssr (eti_2010 +0.59pp through eti_2040 +1.57pp at rec>=ssr-0.03pp); optimum eti_4020 0.4292/0.9517 (plateau from annex 20%+; ring axis 30/40% identical to 20% — inert); Grid B: FAILS at cap2048 and at d256/d1024 entirely | FLAT_GENERALIZES: FALSE — eti_0510 == ring10 EXACTLY at cap4096 every depth (0.3750/0.9531 d256, 0.3945/0.9487 d1024, 0.4011/0.9539 d4096); flat exchange exists only in d4096 trade-band caps 1024-2048 | d256/d1024: ALL policies converge (burst 0.3750 uniform d256; spread 0.49pp d1024) — carry-in mechanism (exp695, 99.9% c-class) needs deep bursts to matter | MODEL: burst depth x capacity jointly set which population dominates arbitration; family advantages regime-specific (interior: d4096 caps 1024-2048; beyond-pole: d4096 cap4096; elsewhere: convergence) | exp700: pin the phase boundary — burst-depth sweep
```

- FREEZE: 839c7e2 (prereg + smoke)

- RUN1: exit 3 — consistency gate as registered was impossible (class totals scale with d; signatures 6+4+4086=4096, 1+0+255=256, 2+0+1022=1024 prove data correct). No results observed before abort (adjudication/metrics print after the gate). GATE V2: within-depth totals equality across caps + sum==d. Bars unchanged.