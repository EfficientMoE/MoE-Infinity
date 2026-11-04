# ADDENDUM26 — exp675a: Shared-state CostFit (structural sharing across composed cache instances)

Date: 2026-11-03/04 · Branch: exp651/qwen30b-results · Prereg: PREREG.md sha256 0c34d992facd17aaa3bcafa5daa68085e191c7de48ef8c758457b0e1aeba0e1b (frozen pre-measurement)

## Question

CostFit N-way compositions (costfit3 = LCNU+S3FIFO+RotHead heads over costfit core) hold N independent
per-layer index tables: costfit = 5 tables, costfit3 = 10 tables of per-key dicts. Can the composed
variants hold ONE shared key table (N-slot records) with **bit-identical replacement decisions**, and
what are the measured memory/time economics?

## Implementation

`research/policies_shared_costfit.py`: `SharedKeyTable`/`SharedFieldView`/`SharedLCNU`/`SharedS3FIFO`/
`SharedRotHead`/`SharedCostFitPolicy`/`SharedCostFit3Policy` — structural sharing via subclass
injection (sealed policy files untouched; additive registration in `policies.py` +
`research/run_policies_real.py`, mirroring the costfit3 pattern). Per-key state moves from N×(dict of
scalars) to 1×(dict of N-slot records); bookkeeping written through per-layer field views.

## Hosts & identity

- Implementation host: Mack (Darwin arm64, python 3.14.8, torch 2.15.0) — exp650 venv.
- Measurement host: Winnie WSL Ubuntu (12 cores, 94 GB RAM; python 3.12.3, numpy 2.5.4, torch
  2.15.0+cpu). Source shas recorded in artifacts/identity.log; unit tests rerun on host: 143 passed
  (89 independent shared-costfit tests + 54 pre-existing).
- Full-precision comparator executed on Mack (Winnie session churn killed two attempts mid-log;
  Winnie's 10 completed cells matched Mack's 12/12).

## Measurements (all rc=0; frozen prereg criteria)

Sweep grid: caps 128–4096 (10 values) × traces {qwen3-30b-a3b decode 1,424,640 acc / granite3.1-moe-3b
decode 638,208 acc} × policies {costfit, shared_costfit, costfit3, shared_costfit3}.

- **H1 parity (hard gate): PASS.** 40/40 sweep cells bit-identical (hit_rate, effective_hit_rate)
  both traces; full-precision comparator (hits, switches, choice_switches, victim sequences) 12/12 OK.
- **H2a (2-way economics): PASS** — struct ×0.52–0.83 (6/6 cells < 0.90); terminal bytes ×0.81–1.10
  (5/6 cells ≤ 1.05; the miss is granite cap=256 terminal ×1.105 — union table exceeds 2 small
  per-layer tables at low overlap).
- **H2b (3-way economics): PASS** — struct ×0.34–0.57 (6/6); terminal ×0.64–0.92 (6/6 < 1.00).
- **H3 time (informational): ns/access ×1.13–1.67** (mean ≈ ×1.33) — indirection through shared views
  costs ~13–67% per access; no free lunch.
- **H4 determinism: PASS** — detached reruns reproduce sweep rows exactly (4/4 cells).

Example (qwen30b, cap 4096): costfit3 8.20 MB terminal → shared_costfit3 6.33 MB (×0.771) with struct
×0.342 at identical hit 0.988616.

## Verdict

`SHARED-STATE-VALIDATED` — parity is exact; savings scale with composition depth (N=2 modest, N=3
halves structure); per-access time penalty is real and must be weighed per workload.

## Limitations

- CPU simulation tier (trace replay of real decode traces; NOT exllama/native-serving parity — no
  CUDA, no serving stack).
- Time ratios measured under concurrent foreign load on Winnie (loadavg ~23) — directions robust,
  magnitudes noisy.
- Census cells report 3-repeat means for ns/access; byte counters exact.

## Artifacts

- `research/policies_shared_costfit.py`, `research/census_shared_costfit.py`
- `tests/python/unit/test_shared_costfit_parity.py` (89 tests, independent author)
- `research/results_real_exp675-{qwen30b,granite}.csv/.md`, `results_real_exp675-det-*.csv`,
  `census_exp675-{qwen30b,granite}.csv`, `comparator_exp675.log`, `results_real_exp675-smoke.*`
- Evidence dir: `~/warp-work/exp675-moei-shared-costfit/` (PREREG.md, AUTHORSHIP.sha256, verdict_exp675a.py, artifacts/)
