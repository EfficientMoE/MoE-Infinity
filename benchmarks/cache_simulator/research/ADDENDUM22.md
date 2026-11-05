# ADDENDUM 22 — exp676: drift stress of the certified CostFit center

**Verdict: DRIFT-SAFE** (H1 True, H2 True, H3 True;
determinism byte-identical double-run;
accounting loop==property in all 16 cells).

## Question

All prior barrier results (exp668/670/673/674) test stationary streams. exp676
asks whether the certified center (MARGIN=0.005, GAMMA=0.0025, HALF_LIFE=32768)
bounds harm through a MID-STREAM regime change — during the transition window
itself (H1, bar 1.0pp), after recovery (H2, bar 0.5pp), and without thrash
(H3, bar 3 switches per 4096-access window).

## Design (frozen pre-run: PREREG 2a94267e, runner 156fedc3)

8 two-regime blockwise streams (B=4096, N=638976, switch at 50%):
skew 0.8->1.4, 1.4->0.8 (locality 0.3); locality 0.0->0.6, 0.6->0.0 (skew 1.1);
transition widths W={4096 hard, 81920 gradual}; caps {512, 2048}; arms
lcnu, s3fifo, costfit@center; seeds default_rng(67620261103).spawn(8).
{d['cells']} cells; windows: transition = lerp blocks, post-steady = last 25%.

## Result

- H1 TRANSITION-BOUNDED PASS: worst transition deficit 0.781pp
  (bar 1.0pp) at s1.1,l0.6->s1.1,l0.0, W=4096, cap=2048.
- H2 RECOVERY PASS: worst post-steady deficit 0.264pp (bar 0.5pp)
  — recovery is near-immediate once the new regime stabilizes.
- H3 NO-THRASH PASS: max switches in any 4096-access window = 1
  across all cells (bar 3); the arbitration never oscillates through a shift.
- Pressure profile: locality shifts are harder than skew shifts
  (mean transition deficit 0.268pp vs 0.158pp);
  hard switches 0.165pp vs gradual 0.261pp.

## Reading

The barrier is NOT a steady-state-only result: through 4 regime-change
directions x 2 transition speeds x 2 caps, the center never exceeds 0.78pp
deficit vs the best standalone during the shift itself, recovers to <=0.26pp
after, and switches at most once in any window. Combined with exp674 (BO point
rejected: worse on average AND safety-broken), the center's certification now
covers stationary synthetic families, cross-model transfer, BO-optimality
attacks, and mid-stream drift.

## Disclosures

- Smoke caught four defects pre-freeze (two runner: empty replay range,
  window double-count; two smoke-expectation) and the first launch died on the
  divisibility guard (N corrected 638,208->638,976 pre-run, zero data under
  old N; disclosed in PREREG).
- Run wall time ~15 min/pass (est. was 4-5h); single host, no concurrent
  memory-heavy work; disk exception scoped in PREREG (<50 MB writes).
- Worst transition deficits sit at 78% of the H1 bar — locality-drop hard
  switches are the closest to failing; that is where the next probe pushes.

## Follow-up (registered)

W sweep at the failing edge: locality-drop direction, W in {1024, 2048, 4096,
8192} with caps {2048, 4096} to find the transition speed where H1 breaks
(if any), then barrier cooldown design if it does.
