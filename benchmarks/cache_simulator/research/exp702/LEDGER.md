# exp702 — real-trace carry-in decomposition (the bridge experiment)

Pre-registered before run. Freeze hash after commit.

## Chain
exp695 (synthetic): burst traffic = 99.9% carry-in re-accesses.
exp696-701 map the policy frontier this creates (ring/ssr/ETI,
ridge at d/cap~1.6). ALL of it rests on the synthetic crowd
stream. exp650 bridged the REAL granite-3b trace for
lru/lcnu/belady hit rates (LCNU wins 7/8 caps; cap256 0.468164)
but never ran the carry-in classifier or the ring/ssr/ETI arms.
exp702 asks: does the carry-in mechanism exist in real traffic?

## Stream (frozen asset, provenance-sealed)
stream_granite3b.pt (exp650 convert of exp646 llama.cpp trace;
DATA-FREEZE sha e2735ff2...). Decode records only, file order,
keys = (layer, expert) in step/layer/slot order. N=638,208;
distinct 1,262; 8 sessions (prefill boundaries at access idx 0,
80128, 211200, 297216, 359936, 403200, 453632, 584704 = windows
0/19/51/72/87/98/110/142); per-prompt decode steps 313/512/336/
245/169/197/512/209 (== convert-report).

## Windows
W=4096, aligned at 0; windows 1..154 (155 full; window 0 has no
pre-window history — c is undefined-by-construction there —
EXCLUDED from burst/base sets, disclosed). Remainder 3,328
dropped from windowed stats.

## Sets (no free parameters)
- novelty(w) = a-count / W (a = key's first appearance in the
  whole stream).
- BURST = top 16 novelty among windows 1..154 (top decile).
- BASE = novelty below median among windows 1..154 (77 windows).

## Bars (pre-registered)
- PRIMARY H1 REAL_CARRYIN: mean c-share over BURST windows
  >= 0.80. c = key first seen before window start.
- SECONDARY H2 NOVELTY_SPIKES: mean a-share(BURST) >= 5x
  mean a-share(BASE).
- SECONDARY H3 POLICY_SPREAD: max-min arm hit spread within
  BURST windows >= 1.0pp at cap 256 or 512, arms {lcnu, ring10,
  ssr_10, eti_2020}, granite params (num_layers=32, prior=0.20),
  full-stream stateful pass, per-window hit aggregation.
- DESCRIPTIVE: which BURST windows contain session starts;
  whole-stream hit table; per-arm burst-vs-base hit deltas.

## Gates (exit codes)
- 3 STRUCTURE: N==638208; distinct==1262; per-prompt decode
  counts == convert-report; full windows==155.
- 2 DET: classifier(window 1) + full LCNU@256 pass rerun must be
  byte-identical.
- 5 EXP650_BAND: |whole-stream lcnu@256 - 0.468164| <= 0.005
  (different simulator internals — band, not byte-exact; failure
  = disclose + settle without verdict, not silent pass).
- Exit 0 settled.

## Smoke (done pre-freeze, structure only)
N/distinct/per-prompt/session-starts verified exact; window0
a/b/c = 991/3105/0 (cold start: no carry-in possible, c==0 by
construction — basis for the window-0 exclusion); policies accept
granite params; ring10@256 cold window 0.293 vs lcnu 0.444
(ring's burst cost is VISIBLE in real traffic — motivating H3).

- STATUS: NEXT
