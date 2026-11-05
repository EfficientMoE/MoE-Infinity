# exp707 LEDGER — cost-layer completion (lo GiB table + 3-trace sizing rule)

## Status
- 2026-11-04 19:40 ET — PREREG.md written (gates S1-S5 frozen; inputs =
  sealed exp704/exp705/exp706 parsed at runtime and asserted; formulas
  byte-identical to exp705 run_costcurves.py:130-131). Runner run_sizrule.py
  written (pure transform; no simulation). Namespace exp707 verified free
  (~/warp-work ls + queue tail = exp706; lane-0026 coordination respected).
- Pending: disclosed smoke -> ARTIFACT.sha256 freeze -> real run.
- 2026-11-04 19:41 ET — disclosed smoke run 1 crashed pre-freeze (KeyError: synthetic smoke dicts used int keys where json.load gives string keys — exactly the record-shape class the smoke exists to catch; repaired input to mirror json.load shape; second smoke crash: S1 anchor caps hardcoded 1792/2048, now derived from input overlap — identical machinery for smoke and real; zero real measurements involved, all pre-freeze). SMOKE_OK: S1 anchor exact (synthetic overlap cap 8), S2 law-band, S4 savings-identity, determinism byte-equal (S3 is real-input-only by design, False on synthetic).
- Freeze: shasum -a 256 run_sizrule.py -> ARTIFACT.sha256 BEFORE real run.
