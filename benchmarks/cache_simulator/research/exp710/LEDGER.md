# exp710 LEDGER (lane 79990, STACK-1001)
- 2026-11-04 21:11 EST — registered; namespace exp710 free (ls ~/warp-work + queue tail exp709 + full grep). Adopts never-frozen exp679 design v1 (no runner/seals/data existed there) + second H0 bookend (HL 32768 == exp674 cf_center). gen_stream/stream_sha AST-identical to exp674 (checked).
- Disclosed smoke PASSED (smoke.log: S1 sha bites, S2 H0 full-N replication 4/4 cells exact vs sealed exp674, S3 negative control detects, S4 det/accounting/well-formed; SMOKE_OK). No runner edits caused by smoke.
- Freeze: shasum -a 256 run_hlregion.py PREREG.md -> ARTIFACT.sha256 BEFORE real run; freeze commit to research/exp710/ pushed before launch. Interpreter /opt/homebrew/bin/python3 (CPython 3.14.8, numpy 2.4.2).
