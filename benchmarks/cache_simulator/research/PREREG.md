# exp683 PREREG — doorkeeper under flash crowds (frozen before launch)

## Question

exp681 proved one-touch crowd admissions cost up to 10.86pp in-burst /
6.98pp recovery. exp682 proved the doorkeeper harms stationary streams
(no crowd to block; constant double-miss tax). Does the door pay on the
crowd testbed — the only place its blocked event occurs? If it still
harms, admission filtering is dead for this policy family (pre-registered
consequence) and the fix must live inside inner admission.

## Design (frozen)

- Streams: exp681's EXACT crowd streams (seeds default_rng(68120261103)
  .spawn(2), fresh spawn inside run_all; base skew 1.1 lam 0.3; crowd
  uniform ranks [64,128) at midpoint; d in {256,1024,2048,4096}; N, B,
  WIN as exp681). Replication anchor: center burst hit at the smoke cell
  must equal exp681's (0.0234375) — verified in smoke.
- Arms: lcnu, s3fifo, costfit@frozen center, door512, door4096 (center +
  doorkeeper, K-LRU, exp682's mk_door). Caps {512, 2048}; 8 cells.
- Wrapper-only; sealed files untouched. center double-sim for acc.

## Bars (frozen; verdict from pass-1 grid)

- H1 (mechanism): door burst deficit vs best standalone < center's in
  >= half of door-cell combos.
- H3: door recovery deficit < center's in >= half.
- H2 (safety): door loop hit >= center loop - 0.5pp everywhere.
- acc: center loop == loop2. det: pass-2 byte-identical.
- Verdict: DOOR-CROWD-WINS > DOOR-CROWD-NULL > DOOR-CROWD-HARM.

## Pre-freeze smoke (disclosed)

- exp681 anchor EXACT (costfit burst 0.0234375 at d=512/cap=512 reduced
  stream); acc exact; all five arms run with burst < loop; double-eval
  identical; adjudicate path exercised.
- Honest mini-signal (d=512/cap=512, reduced N): DOOR-CROWD-HARM —
  door burst deficit 6.445pp vs center 5.859pp (in-burst WORSE: crowd
  keys pay double-miss before admission), loop -0.77pp; BUT recovery
  improves (door4096 rec 0.6958 vs center 0.6792 — hot-set protection
  appears exactly where the mechanism predicts). Full-length streams
  have proportionally more hot set to protect; they decide.
- Disk ~6 GiB (> 5 kill); writes < 5 MB; scoped exception per series.

## Next (pre-registered branches)

- DOOR-CROWD-WINS -> dose K + combined door/fast-admit hybrid (block
  one-touch, admit on second sight FAST) as upstream candidate.
- DOOR-CROWD-HARM/NULL -> admission filtering dead at wrapper level;
  fix must live inside inner admission (exp684: inner admission-latency
  lever, e.g., lcnu admission decay under crowd pressure).
