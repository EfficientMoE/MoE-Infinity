# Addendum 47 — exp703: qwen-30b replication (mechanism now a 2-family law)

## Verdict: REPLICATED (primary H1) — with three layered findings

- H1 REAL_CARRYIN: PASS both streams — c-share on burst windows:
  **hi 94.8%, lo 93.1%** (granite was 99.2%). Re-access dominance
  is now confirmed on two frozen real model families. The core
  premise of the synthetic mechanism program survives its own
  bridge test.
- S2 LCNU_BEST: effectively confirmed. On lo, LCNU is top burst
  arm at both caps. On hi, ssr_10 edges lcnu by **0.0003**
  (0.6772 vs 0.6769 @1024) — a dead heat, not a defeat; at 2048
  four arms tie within 0.03pp. No arm beats LCNU on whole-stream
  in ANY cell. Strict registered bar (==max) reads hi=False;
  disclosed as a 0.03pp tiebreak.
- The inversion DEEPENS on qwen: ring10 whole-stream @1024 =
  **0.391 (hi, −31.3pp vs lcnu)** and **0.284 (lo, −34.1pp)** —
  worse than granite's −20.7pp. eti_2020 also loses big @1024
  (−7.5pp hi / −10.1pp lo). ssr_10 is neutral everywhere
  (≤0.1pp). Synthetic-trained burst-arbitration is harmful on
  both real families.
- S1 FRONT_LOADED: SPLIT — lo PASS (w90=64 ≤ 78), hi FAIL by 2
  windows (98 > 96). Front-loading degree is stream-dependent,
  and the earlier granite claim needs revision: on qwen, SESSION
  SWITCHES CARRY NOVELTY BURSTS (burst∩session-starts = {48,144}
  hi; {48} lo). Granite's "switches are zero-novelty warm
  reshuffles" is model-specific, not a law.
- S3 SPREAD: 18.7pp (hi) / 19.6pp (lo) @1024 = cap 1/6 of key
  space; collapses to 0.03pp (hi) / 2.1pp (lo) @2048 = 1/3 of
  key space. Capacity is the divergence knob: policy choice
  matters in a band, vanishes above it — and ring never wins
  anywhere on real traffic.

## Replication anchors (byte-exact, second time)
lcnu@1024 whole: hi **0.704076 == repo 0.704076**; lo
**0.624469 == repo 0.624469** (band was ±0.005). Research
harness ≡ repo simulator on both families. det True (classifier
+ full policy rerun per stream); structure gates exact (N
786,432/638,208; distinct 5,097/5,842; per-prompt decode counts
verified). 518s, exit 0.

## Program consequence
Carry-in law: universal. Policy lesson: inversion universal and
sharper on qwen. Structure lesson: front-loading is NOT universal
— session-switch novelty is model-dependent; capacity positions
the divergence band. Next: exp704 real-trace capacity fine-map
(collapse curves of spread vs cap on all three frozen traces) —
feeds the cost-integration layer (cache-sizing) with real-traffic
curves.

Data: exp703-qwenreplication.json. Assets: exp656 hi/lo frozen
streams (08732f27 lineage). Freeze c599dab.
