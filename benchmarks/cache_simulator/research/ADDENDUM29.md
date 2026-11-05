# Addendum 29 — exp683 doorkeeper under flash crowds (DOOR-CROWD-HARM)

Home-field test for the doorkeeper: exp681's EXACT crowd streams
(anchor cell reproduced to the digit in smoke), arms lcnu/s3fifo/
center/door512/door4096, 8 cells. Pre-registered consequence fires:
**admission filtering is dead at the wrapper level** for this policy
family.

Verdict: **DOOR-CROWD-HARM** (det byte-identical, acc exact).
- H1 (burst deficit shrinks): only 3/16 combos.
- H3 (recovery deficit shrinks): 5/16.
- H2 safety violated in 11/16 (loop hit > 0.5pp below
  center).
- Worst in-burst harm: door512 at d=2048/cap=2048
  burst deficit 10.89pp vs center's
  4.88pp.

## The mechanistic split (why this is still a positive result)

- Recovery FLIPS TO ADVANTAGE in 4 combos: at
  d=4096/cap=512 door4096 turns the center's 6.98pp recovery deficit
  into a -1.00pp advantage — hot-set
  protection works exactly as designed when one-timers are blocked.
- But in-burst harm grows everywhere (door delays the admission of
  repeating crowd keys during the burst window).
- Combined with exp681 (sw=1 aligned control still lost 10.86pp) and
  exp682 (stationary harm): **the lever is admission SPEED with
  post-admission demotion, not admission delay.** Admit immediately;
  evict never-rehit keys first.

## Pre-registered branch taken

HARM -> wrapper-level admission filtering closed; exp684 targets the
inner admission/eviction priority: probation subclass (admitted on
first sight at highest eviction priority until first re-hit).

Cells 8 | 5 arms | det True | elapsed 763.3s

Generated programmatically from exp683-doorcrowd.json
(gen_addendum29.py).
