# ADDENDUM 54 — exp710: HL-only region probe to the BO search ceiling (costfit center M/G; HL 32768 -> 262144)

Verdict: **HL-BOUNDARY@262144** — every HL arm up to and including the exp669 BO point
HL=160,975 stays inside the frozen no-harm bar (worst informative-cell deficit <=
0.5pp, switches <= 6); the search ceiling HL=262,144 breaks it on
3/59 informative cells (worst 0.523pp). Certified HL upper edge
(largest HL with every smaller HL also safe) = **160,975**. H0 two-bookend replication
132/132 cells exact vs sealed exp674; H4 accounting 330/330 + two full runs
bit-identical. Exit 0, 6624s.

## Per-arm results (59 informative cells; bars frozen from exp674)
| HL | sealed replica? | worst deficit pp | cells > 0.5pp | max switches | mean delta vs HL32768 pp | H1 |
|---|---|---|---|---|---|---|
| 32768 | yes (bookend: exp674 cf_center) | 0.176 | 0 | 2 | +0.0000 | PASS |
| 65536 | no | 0.198 | 0 | 2 | -0.0270 | PASS |
| 131072 | no | 0.351 | 0 | 2 | -0.0544 | PASS |
| 160975 | yes (bookend: exp674 cf_hlonly) | 0.367 | 0 | 2 | -0.0613 | PASS |
| 262144 | no | 0.523 | 3 | 2 | -0.0783 | **FAIL** |

Ceiling-arm (HL 262,144) cells over the bar:
| stream | cap | deficit pp | switches | also an exp674 joint-point violation? |
|---|---|---|---|---|
| s=1.1 lam=0.3 E=128/L=48 | 2048 | 0.523 | 2 | yes |
| s=1.4 lam=0.3 E=128/L=48 | 1024 | 0.509 | 2 | yes |
| s=1.4 lam=0.6 E=128/L=48 | 512 | 0.507 | 1 | yes |

## What this settles
- **Half-life is a monotone harm dial with a soft edge, not a cliff.** Worst deficit
  rises 0.176 -> 0.198 -> 0.351 -> 0.367 -> 0.523pp
  while mean paired delta drifts only +0.0000 -> -0.0783pp: longer memory
  costs a little everywhere and a lot nowhere until the ceiling. Switches never exceed
  2 on any arm (bar 6) — the switch bound never binds; every failure is a pure hit-rate deficit.
- **exp674's break is NOT an HL effect.** At the same HL (160,975), HL-only worst is
  0.367pp (safe) while the joint (M,G,HL) BO point's worst is 0.779pp (8 cells over).
  3/3 of the ceiling arm's failing cells coincide with exp674 joint-point violation
  cells — the same stream regions are fragile, but at the BO HL it is the (M,G) move that
  pushes them over. Attribution: (M,G) or the (M,G)xHL interaction; separating those needs
  the M/G-only arm at center HL (registered next as exp711).
- Pre-registered consequence (HL-BOUNDARY branch): certified HL band frozen at
  [32,768, 160,975]; HL exploration stops there. Interval (160,975, 262,144) is untested
  between grid points, so the true edge lies somewhere inside it (PREREG grid resolution).

## Integrity, determinism, disclosures
- Runner sha256 fce715888df41948... and PREREG sha256 2d519de9545cdd92... == ARTIFACT.sha256 ==
  freeze-commit 814406c blobs; freeze pushed before launch (21:11 ET) and precedes the
  result mtime. Disclosed smoke (S1-S4) passed pre-freeze; no smoke-caused runner edits.
- Published exp674-hl-transfer.json was reformatted by a later pre-commit pass (byte sha
  differs from the local sealed copy the runner read); parse-equality asserted at
  generation time — the data are identical.
- Synthetic streams only (11 cfgs from exp670); no real-trace or serving claim.

Data: exp710/exp710-hlregion.json (run.log/smoke.log local; *.log gitignored except the
frozen smoke.log); PREREG.md / ARTIFACT.sha256 / LEDGER.md in exp710/. Follow-up
controller re-adjudicated every gate with an independently written adjudicator
(16/16 checks): verdict HL-BOUNDARY@262144 stands.
