# ADDENDUM 55 — exp711: 2^3 factorial attribution of exp674's SAFETY-BROKEN BO point (M, G, HL)

Verdict: **MG-BREAK:G-DRIVEN** — the break lives in GAMMA. At center HL, moving G alone to the
exp669 BO value (gamma 0.00998964, 4.00x center) breaks the frozen no-harm bar on
6/59 informative cells (worst 1.145pp); moving M alone to BO (margin
0.00161648) is safe and in fact *improves* the worst cell (0.055pp vs center
0.176pp). The joint (M,G) move at center HL still fails (2 cells, worst
0.526pp), so the ladder's first rung fires with M safe / G failing. H0 three-bookend
replication 330/330 cells exact vs sealed exp674 (cf_center, cf_hlonly, cf_bo + standalones);
H4 accounting 528/528 + two full runs bit-identical. Exit 0, 9323s.

## Per-corner results (59 informative cells; bars frozen from exp674)
| corner | knobs | sealed replica? | worst deficit pp | cells > 0.5pp | max switches | mean delta vs 000 pp | H1 |
|---|---|---|---|---|---|---|---|
| 000 | M=ctr G=ctr HL=ctr | exp674 cf_center | 0.176 | 0 | 2 | +0.0000 | PASS |
| M00 | M=BO G=ctr HL=ctr | new | 0.055 | 0 | 2 | +0.0417 | PASS |
| 0G0 | M=ctr G=BO HL=ctr | new | 1.145 | 6 | 2 | -0.1095 | **FAIL** |
| MG0 | M=BO G=BO HL=ctr | new | 0.526 | 2 | 2 | -0.0599 | **FAIL** |
| 00H | M=ctr G=ctr HL=BO | exp674 cf_hlonly | 0.367 | 0 | 2 | -0.0613 | PASS |
| M0H | M=BO G=ctr HL=BO | new | 0.149 | 0 | 2 | -0.0053 | PASS |
| 0GH | M=ctr G=BO HL=BO | new | 1.145 | 11 | 2 | -0.1705 | **FAIL** |
| MGH | M=BO G=BO HL=BO | exp674 cf_bo | 0.779 | 8 | 2 | -0.1234 | **FAIL** |

## 2^3 effects (descriptive, Yates /4; no gate)
- Worst-cell deficit: M -0.3310 | G +0.7120 | H +0.1345 | MG -0.1615 | MH +0.0390 | GH -0.0080 | MGH +0.0875
- Mean paired delta: M +0.0486 | G -0.1096 | H -0.0582 | MG -0.0002 | MH +0.0030 | GH -0.0040 | MGH -0.0042

G is the only positive main effect on worst-cell harm (+0.712pp); M is protective
(-0.331pp) and the MxG interaction is negative (-0.162pp): a smaller margin
partially cancels the gamma damage (0G0 1.145 -> MG0 0.526pp at center HL;
0GH 1.145 -> MGH 0.779pp at BO HL). HL adds a modest main effect
(+0.135pp, consistent with exp710's monotone HL dial) and widens G's footprint
(0G0 6 -> 0GH 11 cells over the bar). Switches never exceed 2 on any
corner (bar 6): every failure is a pure hit-rate deficit.

G-only (0G0) cells over the bar:
| stream | cap | deficit pp | also over in MG0? | also an exp674 joint-point violation? |
|---|---|---|---|---|
| s=1.1 lam=0.3 E=128/L=48 | 512 | 0.779 | no | yes |
| s=1.1 lam=0.3 E=128/L=48 | 2048 | 0.522 | no | yes |
| s=1.1 lam=0.6 E=128/L=48 | 512 | 0.753 | no | yes |
| s=1.4 lam=0.3 E=128/L=48 | 1024 | 0.540 | yes | yes |
| s=1.4 lam=0.6 E=128/L=48 | 512 | 0.507 | yes | yes |
| s=1.4 lam=0.6 E=32/L=32 | 256 | 1.145 | no | yes |

## What this settles
- exp674's SAFETY-BROKEN BO point is a **gamma** problem. exp710 had already cleared HL up to
  the BO value; exp711 shows M is benign-to-helpful and G carries the harm (alone and in every
  context it was tested). The MGH bookend reproduces exp674 exactly: worst
  0.779pp and the identical 8-cell violation set.
- Pre-registered consequence (MG-BREAK:G-DRIVEN): next = single-knob GAMMA dose line at center
  M/HL to find gamma's certified band (as exp710 did for HL); registered as exp712.
- Union of G-carrying failures (0G0 ∪ 0GH) covers 7/8 of exp674's violation cells.

## Integrity, determinism, disclosures
- run_mgfactorial.py sha256 a13d401e88545d62..., PREREG.md 10c34724ec315c45..., smoke.log
  8e1b948be8a2c177... == ARTIFACT.sha256 == freeze-commit 4a015bc blobs == local run dir;
  freeze pushed before launch and precedes the result mtime. Disclosed smoke (S1-S4) passed
  pre-freeze; no smoke-caused runner edits.
- Published exp674-hl-transfer.json asserted parse-equal to the local sealed copy the runner read.
- Synthetic streams only (11 cfgs from exp670); no real-trace or serving claim.

Data: exp711/exp711-mgfactorial.json (run.log local, gitignored); PREREG.md / ARTIFACT.sha256 /
smoke.log / LEDGER.md in exp711/. Follow-up controller re-adjudicated every gate with an
independently written adjudicator (38/38 checks): verdict MG-BREAK:G-DRIVEN stands.
