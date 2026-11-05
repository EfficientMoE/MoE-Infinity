# exp706 LEDGER — qwen30b-lo CROSS1 location (capacity-grid extension)

## Status
- 2026-11-04 19:12 ET — PREREG.md written (gates L1-L4 + continuity frozen;
  anchors runtime-parsed from sealed exp704 lo JSON and pinned; WS20 lo=4662
  parsed from sealed exp705; law interval [0.50, 0.66]). Runner run_locross.py
  written (mirrors exp704 run_capacitymap.py arms/loader/burst semantics
  exactly; grid 1792-3328 with 1792/2048 overlap-replication cells).
  Namespace exp706 verified free (~/warp-work + queue tail; lane-0026
  coordination respected).
- Pending: disclosed smoke -> ARTIFACT.sha256 freeze -> real run.

## Bars (frozen in PREREG.md)
- L3 REPLICATION-ANCHOR: spread@1792 == 4.211, spread@2048 == 2.116 EXACT;
  lcnu whole@{1792,2048} == sealed EXACT (exit 5 on mismatch).
- L1 LOCATE: first cap with spread < 1.0 (exp704 semantics); consistency:
  spread@2048 >= 1.0.
- L2 LAW-TRANSFER: 0.50 <= CROSS1_lo/4662 <= 0.66 if found; NOT_REACHED
  refutes the interval (3328/4662 = 0.713).
- L4 DET: lcnu@2304 recomputed == grid cell exact (exit 2).
- Structure: N=638208, distinct=5842, prompts=[512,512,126,512], n_win=155
  (exit 3).
- 2026-11-04 19:14 ET — disclosed smoke PASSED (SMOKE_OK: synthetic mini-trace through same loader/novelty/policy_cell machinery; det exact; N=12800 distinct=96). Freeze: shasum -a 256 run_locross.py -> ARTIFACT.sha256 BEFORE real run.
- 2026-11-04 19:13 ET — REAL RUN LAUNCHED (pid 69474, background, run.log): structure gate passed (stream ok, N=638208 distinct=5842 prompts=[512,512,126,512] n_win=155); burst=[[1, 2, 3, 4, 5, 6]... [0s]]. Freeze commit (prereg+runner+ARTIFACT+LEDGER under research/exp706/) pushed minutes AFTER launch — sha-freeze on disk preceded launch; disclosed deviation from exp699-705 pattern (freeze-commit-before-launch) due to controller sequencing; substance unchanged (prereg+bars+sha immutable pre-run).
