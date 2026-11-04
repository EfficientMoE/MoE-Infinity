# exp703 — qwen-30b replication: carry-in, front-loading, LCNU-inversion

Pre-registered before run. Freeze hash after commit.

## Chain
exp702 (granite-3b real trace): carry-in mechanism REAL (99.24%
c-share on burst windows) but the synthetic policy lesson INVERTS
— plain LCNU dominates, ring/ETI pay 14-21pp; real novelty is
front-loaded in the startup ramp; session switches are
zero-novelty warm reshuffles. Single-model evidence. exp703
replicates on the second frozen model family (qwen3-30b-a3b,
E=128/L=48/k=8, key space 6,144) with TWO contrast streams:
- hi (high Gini imbalance prompts {2,4,6,7}): N=786,432 = 192
  full 4096-windows exactly; distinct 5,097 (83.0%).
- lo (low Gini {0,1,3,5}): N=638,208; 155 full windows (+3,328
  remainder dropped); distinct 5,842 (95.1%).
Assets frozen by exp656 (derived from trace 08732f27; shas in
exp656 DATA-FREEZE). Coverage 83-95% >> granite's 98.4%-reached-
fast — novelty structure is the open question, not a foregone
conclusion.

## Protocol (exp702's exactly, adapted params)
Windows W=4096 aligned at 0; window 0 excluded (no pre-window
history); classifier a = global first-sight, b = first seen
within same window, c = carry-in (seen before window start).
BURST = top decile by novelty (hi: 20 of windows 1..191; lo: 16
of 1..154). Arms lcnu / ring10 / ssr_10 / eti_2020, params
num_layers=48, prior=8/128; full-stream stateful pass; caps
{1024, 2048} (mid-slope for 6,144 keys; anchors exist at 1024).

## Bars (pre-registered)
- PRIMARY H1 REAL_CARRYIN: c-share >= 0.80 on BURST windows,
  per stream. Verdict: REPLICATED (both) / PARTIAL (one) /
  REFUTED (neither).
- SECONDARY S1 FRONT_LOADED: window index reaching 90% of all
  first-sights <= half the window span (<= 96 hi / <= 78 lo).
  (Replaces exp702's degenerate H2 ratio — that bar produced an
  empty base set on granite; disclosed there.)
- SECONDARY S2 LCNU_BEST: lcnu burst-window hit == max of the 4
  arms at both caps, both streams (inversion confirmation).
- SECONDARY S3 SPREAD: max-min burst spread >= 1.0pp per
  stream/cap.
- DESCRIPTIVE: burst-window ids vs session starts; per-arm whole
  hit table.

## Gates (exit codes)
- 3 STRUCTURE: per stream N, distinct, per-prompt decode counts
  (hi [512,512,512,512]; lo [512,512,126,512]), window count.
- 2 DET: classifier(window 1) + full lcnu@1024 pass rerun
  byte-identical, per stream.
- 5 ANCHOR: |whole-stream lcnu@1024 - repo CSV| <= 0.005 per
  stream (hi 0.704076, lo 0.624469; exp702 hit this band
  byte-exact).
- Exit 0 settled.

## Smoke (pre-freeze, structure only)
Both streams load; dims 48x8 decode confirmed; N/distinct/
per-prompt/window counts verified above; policies accept
num_layers=48 prior=1/16 (same constructor path as exp702).

- STATUS: NEXT
