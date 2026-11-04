# Addendum 45 — exp701: ridge decomposition (RIDGE_CREST by 0.012pp; crest off-grid at ratio 1.6)

## Verdicts
- PRIMARY PAIR_DOMINANT: PASS — ring10-vs-ssr_10 is the max-min
  pair in 13/15 exp700 cells (87% >= 70%). The general divergence
  surface IS the ssr-vs-ring trade. (Part 1, pure analysis of
  sealed exp700 JSON — no new compute.)
- SECONDARY RIDGE_CREST: PASS AT THE TOLERANCE EDGE — max off-grid
  probe 12.402pp <= 11.914 + 0.5 = 12.414 (margin 0.012pp).
  Disclosed: this is a pass by 0.012pp, and the probe that set it
  EXCEEDS the exp700 crest itself.

## The refinement that matters
exp700's grid put the ridge max at (d4096, cap2048) = 11.91pp and
we read "d ~ 2 x cap". The off-grid cross-section at d4096:

  cap:   1280  1536 [2048] 2560  3072  5120
  pp:     9.79  9.89 [11.91] 12.40 10.01  0.17

The surface maximum is at cap ~ 2560 — d/cap ~ 1.6, not 2.0 — and
it then collapses hard (3072: 10.01; 5120: 0.17 — saturation
washout). The d3072 row rises monotonically through cap2560
(9.77 / 10.22 / 10.61 at caps 1536/2048/2560): the cap-direction
peak drifts right as d grows. Crest attribution (part 1): the
crest cell itself is ring10-vs-eti_2020 (0.2148 vs 0.0957), i.e.
eti_2020's burst surge with ring's floor — not the ssr trade that
carries the rest of the surface.

## Process disclosure
Run 1 crashed at d=5120: the sealed generator's burst-draw array
is sized 4096 (BURSTS axis ceiling) — d > 4096 is out of design
range; extending the generator is deferred to its own experiment.
Rescope v2 (d3072 row + d4096 cap cross-section) pre-registered,
freeze d77a4ab. Run 2 completed all gates (REPL d4096xcap2048 leg
== exp700 exact; DET double-pass eti_2020@d3072/cap2048; depth
totals consistent; exit path: adjudication + JSON write SUCCEEDED,
then a cosmetic print crashed post-JSON on xsec["2048"] — key not
in cross-section list; KeyError, no data impact; sealed JSON not
edited).

Data: exp701-ridgedecomp.json (verdict RIDGE_CREST; det True;
elapsed 1433s). Freeze d77a4ab. exp700 anchors exact.
