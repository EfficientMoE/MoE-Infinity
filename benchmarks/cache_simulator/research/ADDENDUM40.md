# Addendum 40 — exp682-flashcrowd: sub-resegment flash-duration sweep (CROWD-SAFE — H4 refuted)

Designed + frozen 2026-11-03 (runner sha e3a7269c, prereg sha eccac5a2,
smoke clean), launch-blocked solely on the 15 GiB Data-volume gate, and
launched at the first fire past that gate (96 GiB available,
2026-11-04). Same frozen runner bytes as smoked; no edits between smoke,
freeze, and run.

## Question

exp677's W-sweep put costfit's crowd-window win at 0.4395pp on the
captured qwen3-30b trace (deterministic CPU replay). H4 asked whether
that cost is duration-sensitive: if crowds shorter than the resegment
period (D in {256, 512, 1024, 2048} vs PERIOD 16384, locality pushed
0.6 -> 0.95) slip through — either cheaper because they end before the
window turns over (DURATION-SENSITIVE) or equally costly at any length
(CROWD-SAFE). Caps 2048/4096; costfit vs lcnu and s3fifo; per-cell
accounting (crowd-phase delta, post-phase delta, switch-count
reconciliation).

## Result

- VERDICT: CROWD-SAFE. Zero violations in all 8 duration x cap cells:
  crowd-phase cost deltas <= 0.054pp, post-phase <= 0.158pp, switch
  accounting exact (acc true everywhere).
- H4 REFUTED: duration effect 0.074pp < the 0.1pp preregistered bar.
  A crowd 64x shorter than the window costs the same as a full-window
  one; there is no "short enough to hide" regime.
- Determinism: full grid run twice, byte-identical.
- Control: replication of exp677's peak cell is exact to the bit
  (d_center_trans 0.439453125 == exp677-wsweep.json).

## Mechanism read

Duration-invariance is the third independent confirmation of the
competition mechanism (after exp695's carry-in decomposition and
exp696's population-routing collapse): crowd harm is set by the arrival
concentration of competing first-sights — how many cold experts fight
for the same slots — not by how long the crowd stays resident, and not
by burst mass or timing. Shortening crowds does not relieve pressure;
only routing/reordering of that population does.

## Registered follow-up (unchanged, per PREREG)

CROWD-SAFE -> sub-window crowds are not the harm channel; inner-policy
per-access response confirmed as the residual lever -> inner-window
adaptation design. (DURATION-SENSITIVE and CROWD-BROKEN branches moot.)

## Scope statement

CPU-only deterministic replay of the real captured qwen3-30b routing
trace (N=638,976 events), cache-simulator policy comparison. Not GPU
inference and not an end-to-end model run.
