# exp708 LEDGER — beyond-law flat check (lo grid 3328-4662, cap ceiling = WS20)

## Status
- 2026-11-04 19:56 ET — PREREG.md written (gates B1-B4 frozen; inputs = sealed
  exp706 anchor cell @3328 + sealed exp705 WS20(lo)=4662, parsed at runtime AND
  pinned; grid [3328, 3584, 3840, 4096, 4352, 4662]; machinery mirrors exp706's
  sealed run_locross.py exactly). Runner run_beyondlaw.py written. Namespace
  exp708 verified free (~/warp-work ls + queue tail = exp707; lane-0026
  coordination respected — no exp708 in either).
- Pending: disclosed smoke -> ARTIFACT.sha256 freeze -> freeze commit -> real run.
- 2026-11-04 20:01 ET — disclosed smoke PASSED (SMOKE_OK: synthetic mini-trace
  through same loader/novelty/burst/policy_cell machinery; det exact; N=12800
  distinct=96; runtime = exp650 lane venv torch 2.15.0 — same interpreter
  family as exp706 per its __pycache__ cpython-314). Freeze: shasum -a 256
  run_beyondlaw.py -> ARTIFACT.sha256 BEFORE real run; freeze commit to
  research/exp708/ precedes launch (exp705/exp707 pattern).
