# PREREG — exp682: sub-block flash-crowd probe

Date: 2026-11-03. exp682 free (exp681 = phiboundary, designed-launch-blocked).
Builds on exp677 (#202: peak harm at W=RESEGMENT 0.439pp) and exp680 (#204:
STAG-SAFE-NO-GAIN — cadence refuted, barrier never fires at the peak; harm
flows through inner-policy transition response). Residual axis named by
exp676/677 conclusions: sub-block intra-window shifts.

## Question

Do transient expert-demand flash crowds — locality spikes (lam 0.6 -> 0.95
-> back) whose duration D << RESEGMENT arrives and reverts entirely INSIDE
one arbitration window, so arbitration cannot react by construction —
create hit-rate deficit vs standalone policies, and does harm scale with D?

## Design (frozen)

- Streams: exp677's exact block machinery (Zipf skew 1.1, recency-weighted
  replay, E=128, L=48, B=1024). Stationary base (s1.1, lam0.6); every
  PERIOD=16384 accesses a crowd run of D accesses at lam0.95, first crowd
  at access 8192 (8-block warm-up). D in {256, 512, 1024, 2048}; crowd +
  recovery window (B) never overlap the next crowd (PERIOD >> max D + B).
  N=638,976; caps {2048, 4096}; 8 cells + 1 control.
- Seeds: default_rng(68220261103).spawn(4), index i -> DURS[i], drawn
  fresh inside run_grid (exp680's det-bug lesson encoded).
- Arms: lcnu, s3fifo, costfit@center (M=0.005, G=0.0025, HL=32768,
  RESEGMENT 4096 default). No staggered arm (cadence refuted by exp680).
- Masks: crowd = the D accesses per crowd run; post = 1024 accesses after
  each crowd run ends.
- Control cell: exp677's exact W=4096 stream (default_rng(67720261103)
  .spawn(4)[2], cap 4096); center trans deficit must reproduce exp677's
  published peak (parsed from exp677-wsweep.json at runtime, never
  transcribed) within 0.005pp — cross-experiment replication gate.

## Hypotheses (frozen bars)

- H1 crowd deficit (best standalone crowd rate - costfit crowd rate) <=
  1.0pp in all 8 crowd cells.
- H2 post-crowd deficit <= 0.5pp in all 8 cells.
- H3 costfit switches <= 3 per 4096-access window.
- H4 (headline, duration sensitivity): max over caps of
  (d_crowd at D=2048 - d_crowd at D=256) > 0.1pp.
- Accounting: loop == effective_hit_rate for costfit, every cell.
- Determinism: double full-grid run byte-identical (streams regenerated).
- Verdict: DURATION-SENSITIVE iff safety(H1-H3,acc) + H4 + replication;
  CROWD-SAFE iff safety + replication, no H4; REPLICATION-FAIL iff safety
  holds but control deviates from exp677; CROWD-BROKEN otherwise.

## Pre-freeze smoke (disclosed, 2026-11-03)

Reduced stream N/4=159,744, D=1024, both caps, 9.4s:
crowd accesses 10,240 == 10 crowds x 1024; post 10,240; overlap 0;
cap 2048 crowd rates lcnu 0.9708 / s3fifo 0.9722 / costfit 0.9727
(costfit best), post cf 0.9042, sw 1; cap 4096 0.9852/0.9853/0.9852,
post 0.9507, sw 0; accounting loop==prop both caps; stream double-gen
determinism True. No asserts failed. Smoke is reduced scale — no
adjudication, no verdict claim from it. Disk at smoke: 6 GiB avail
(< 15 GiB launch gate; smoke writes < 1 MB).

## Launch status

LAUNCH BLOCKED at freeze time: Data volume 6 GiB < 15 GiB gate (trend
13 -> 11 -> 6 GiB across the day; idle>6h scratch reclaim ~39 MB; the 22G
callnotes-goal-20261102 scratch is ACTIVE work and stays). Runner +
prereg frozen in ARTIFACT.sha256 BEFORE any real run. Launch on a
controller run with disk >= 15 GiB: `nohup python3 run_flashcrowd.py`
in this dir, expected ~20-40 min, register in running.json at launch.

## Follow-up (registered)

DURATION-SENSITIVE -> burst-adaptive margin candidate; ties into exp678
COST-H1 anti-clustering + exp681 (phi,cap) boundary map. CROWD-SAFE ->
sub-window crowds are not the harm channel; inner-policy per-access
response confirmed as the residual lever -> inner-window adaptation
design. CROWD-BROKEN -> sub-window cooldown/barrier redesign.
