# Addendum 46 — exp702: real-trace carry-in (mechanism REAL, policy lesson INVERTS)

## Verdict: REAL_CARRYIN — and the synthetic policy frontier does not transfer

- H1 REAL_CARRYIN: PASS — c-share on burst windows = **99.24%**
  (synthetic exp695: 99.9%). Re-access dominance is real traffic
  structure, not a synthetic artifact.
- H3 POLICY_SPREAD: PASS at **20.65pp (cap256) / 14.26pp (cap512)**
  — larger than the synthetic ridge max (12.40pp) — but the
  direction INVERTS: plain LCNU is the BEST arm on real traffic
  (burst 0.4677 / whole 0.4682 @cap256); ring10 is the WORST
  (0.2612/0.2385); eti_2020 also loses (−20.6pp @256, −4.8pp
  @512); ssr_10 is near-neutral (−1.8pp whole @256, −0.3pp @512).
  In the synthetic world ring10 BOUGHT +6–9pp burst; on the real
  trace it PAYS −20.7pp.
- H2 NOVELTY_SPIKES: UNEVALUABLE — median novelty = 0.0 makes the
  strict `below-median` base set empty (nan). Registered bar
  produced a degenerate comparison; with `<=` it would pass
  trivially (a_burst 0.0037 vs ~0). Disclosed, not re-run.

## Why the inversion: real novelty is FRONT-LOADED
Top-decile novelty windows are 1–23 — the STARTUP RAMP of the
first session (novelty up to 84 new keys/window). Windows 24–154
contain **17 first-sights in total**. All 6 later session switches
(windows 51/72/87/98/110/142) carry ZERO novelty — pure warm
re-access reshuffles, exactly the opposite of the synthetic
cold-crowd model of session switches. The synthetic frontier
(ring/ssr/ETI trading burst for recovery) was built on mid-stream
COLD crowds; real single-user decode has no such events — the only
divergence regime is the ramp, where probation/annex gates
suppress admission and MISS the ramp's own re-accesses.

## Replication anchor (stronger than registered)
Whole-stream lcnu@256 = **0.468164 — byte-equal** to exp650's
repo-simulator number (registered band was ±0.5pp). Research
harness LCNU ≡ repo LCNU on this trace. det True (classifier +
policy rerun byte-identical). Structure gates exact (N=638,208,
1,262 distinct, per-prompt decode counts == convert-report).

## Program consequence
Synthetic-driven burst-arbitration policy work is REFUTED for
this traffic class by its own bridge experiment. The live regimes
on real decode: (1) startup ramp — policy-divergent, LCNU-favoring;
(2) warm steady state — novelty-free, all-keys-seen. Next: does
the inversion + ramp-front-loading generalize to a second model
family (qwen-30b, E=128 key space 6,144) — and what does the ramp
regime actually reward (admission speed vs. holdback)?

Data: exp702-realcarryin.json (exit 0; det True; 60s).
Asset: exp650 frozen stream (sha e2735ff2 lineage). Freeze 062b413.
