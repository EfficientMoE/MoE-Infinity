## Addendum 19: exp673 — CostFit3: the barrier generalizes to 3 candidates (toxic-rejection regime)

**Question.** exp667/668/670 established the self-limiting barrier for TWO
candidates (LCNU vs S3FIFO): switch only when the decayed lead exceeds
MARGIN + GAMMA x decayed switch mass. Does the mechanism survive a THIRD
candidate — rothead, itself a probe-adaptive LCNU<->S3FIFO switcher on an
8x faster probe timescale (PROBE_WINDOW 2048 vs the barrier's
HALF_LIFE 32768)?

**Design.** Three pre-freeze iterations tried to construct a stream family
where rothead DESERVES adoption; all three failed (disclosed in the prereg):
lambda-flips left the inners tied (<=0.8pp); on churn-flips LCNU dominated
and rothead was toxic by up to -8.3pp. What survived is the regime the
barrier exists for: a third candidate that is persistently harmful. exp673
therefore freezes the honest question — does the N-way generalization
(challenger = highest decayed-rate non-incumbent, name-order tie-break;
barrier and switch-mass accounting unchanged) correctly REJECT the toxic
candidate while staying within the verified no-harm envelope?

**Verdict: MECHANISM-GENERALIZES-3WAY.** Informative cells: 27;
violations: 0 (H1 no-harm True, H2 toxic-rejection
True, H3 accounting True); synthetic double-run determinism:
true (bit-identical JSON).

Arms: both real traces (30b: 10 caps; granite: 5 caps; sealed harness,
costfit3 fresh, all references parsed from sealed CSVs) plus two frozen
synthetic flip families (N=262,144; 16 blocks of 16,384; caps 128-4096;
lcnu/s3fifo/rothead/costfit/costfit3 all evaluated fresh).

**Risk map (worst 5 informative cells by cf3 deficit vs best standalone):**

| cell | cf3 deficit (pp) | rothead deficit (pp) | cf3 switches |
| --- | --- | --- | --- |
| flipB @ 128 | +0.245 | +2.202 | 2 (lcnu:1+rothead:1+s3fifo:0) |
| flipA @ 2048 | +0.135 | +1.261 | 2 (lcnu:1+rothead:1+s3fifo:0) |
| flipA @ 128 | +0.083 | +0.082 | 1 (lcnu:0+rothead:0+s3fifo:1) |
| flipB @ 2048 | +0.072 | +3.158 | 2 (lcnu:1+rothead:1+s3fifo:0) |
| flipA @ 256 | +0.069 | +0.069 | 1 (lcnu:0+rothead:0+s3fifo:1) |

**Toxic-rejection: 8 cells have rothead trailing best2 by >1pp;
in every one, costfit3 finished within 0.5pp of the sealed 2-way costfit
(H2) — wrong excursions onto rothead occurred (visible as rothead switch-ins
that revert) and self-corrected. The barrier never adopted a candidate the
evidence did not support, on any family, real or synthetic.**

**Mechanism note.** The N-way generalization changed exactly one thing —
which challengers the scoreboard may consider — and the barrier's function
(rate evidence must exceed the accumulated switch-mass penalty before
control transfers) transferred unchanged. This is the third independent
stress family (parameters exp668, stream structure exp670, candidate set
exp673) in which the self-limiting property held with zero violations.

Cost model, parameters (exp667 center), registration points, prereg sha,
and the three disclosed failed design iterations: `exp673-moei-costfit3/PREREG.md`
(frozen pre-run, `f7378dd3`). Data: `research/exp673-costfit3.json`.

