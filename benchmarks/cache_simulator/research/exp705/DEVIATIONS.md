# exp705 DEVIATIONS

## D1 — loader shape defect in frozen run_costcurves.py (repaired before first measurement)

- **Discovered:** 2026-11-04T22:41Z, real-run attempt #1. Runner crashed with
  `ValueError: not enough values to unpack (expected 2, got 1)` at
  `load_decode_keys()` line 52, elapsed 0.6 s, **exit 1, zero gates evaluated,
  zero measurements performed, no result JSON written.**
- **Cause:** frozen runner assumed `topk_ids_of_layer` rows unpack as
  `(layer, experts)` — i.e. a (L, k) tensor. Actual trace format (all three
  frozen traces, verified by direct inspection this fire): decode records store
  **(L, seq_pos=1, k)** int32 tensors; prefill records (L, seq, k).
- **Evidence the prereg semantics were still exactly implementable:**
  registered structure-N constants match raw per-record access counts with
  multiplicity, no dedup:
  - granite: 2493 decode records × L32 × k8 = 638,208 (registered 638,208)
  - hi: 2048 × L48 × k8 = 786,432 (registered 786,432)
  - lo: 1662 × L48 × k8 = 638,208 (registered 638,208)
- **Repair (minimal, mirrors exp704's committed loader `run_policies_real.py`
  lines 169-174):** `flattened = tensor.reshape(num_layers, -1)` then enumerate
  `(layer, expert)` with multiplicity. Decode records have seq_pos=1, so this
  yields exactly the prereg's keys and APT=L*k semantics.
- **Unchanged:** all prereg bars C1-C4, byte model, formulas, inputs, exit-code
  contract. No bar, threshold, or registered constant was touched.
- **Freeze discipline:** original frozen runner preserved verbatim in git
  (commit 1a8514a, sha256 c956d868…); repaired runner re-frozen BEFORE the
  first real measurement as ARTIFACT-r2.sha256. Smoke re-run after repair
  (disclosed): SMOKE_OK.
- Precedent: exp638 D1/D3 repaired-track disclosure (bars unchanged).

## D2 — C4c anchor misread sealed spread definition (repaired after frozen track failed honestly)

- **Discovered:** 2026-11-04T22:5xZ. Frozen-track real run (output preserved
  verbatim as `exp705-costcurves.frozen-track.json`) evaluated honestly and
  returned **ANCHOR-FAIL, exit 5**: C4c `max-min of WHOLE hits` vs sealed
  `spread_pp` mismatched at every probe cell (granite 0.14–1.26 pp; hi/lo up
  to 12.7 pp). N-structure (c4b) and WS20 determinism (c4d) PASSED all traces.
- **Cause:** prereg misread exp704's sealed artifact. Sealed `spread_pp` is
  max−min over arms of **BURST**-window hit rates ×100
  (`run_capacitymap.py` lines 138-141, commit 53fedca), not whole-stream hits.
  exp704 itself is internally consistent — this fire's recompute of
  burst-spread from its own policy_grid reproduces sealed values **exactly
  (diff 0.0 at all 27 trace-cap cells)**.
- **Repair:** `spread_recompute()` switched to burst hits. Tolerance and all
  bars (C1-C4), byte model, formulas, inputs unchanged. Artifact re-frozen as
  ARTIFACT-r3.sha256 before the repaired real run; smoke re-run post-patch.
- **Interpretation discipline:** the frozen-track ANCHOR-FAIL remains the
  verdict of the frozen protocol; the repaired track is reported separately
  (exp638 precedent). The anchor failure is itself evidence the frozen
  prereg's upstream-artifact assumption was wrong, not that exp704 is wrong.
