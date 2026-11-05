# exp705 PREREG — cost-integration: byte-denominated cache-sizing curves (exp704 follow-up)

Program STACK-1001, lane 79990. Analysis tier (pure transform of sealed exp704
data + new working-set computation on frozen traces). No new capture.

## Motivation
exp704 (ledger #227, admitted #53/1001) produced slot-denominated spread-collapse
curves: CROSS1 (smallest capacity where best-vs-worst policy spread < 1pp) =
granite 768 / qwen hi 1792 / qwen lo NOT_REACHED by 2048 slots. next_hint:
"curves feed cost-integration (cache sizing where policy choice stops
mattering)". Nobody in the arc has priced slots in physical bytes yet — exp678
used abstract cost units only. This experiment converts the curves to GiB and
tests the working-set mechanism behind CROSS1.

## Frozen inputs (read-only)
- Sealed verdict: ~/warp-work/exp704-moei-capacitymap/exp704-capacitymap.json
  (policy_grid whole/burst hits per arm/cap/trace, spread_pp, B2_CROSS1).
- Traces: exp650 stream_granite3b.pt; exp656 stream_qwen30b_hi.pt,
  stream_qwen30b_lo.pt (for accesses-per-token assert + WS20 computation).
- Byte model, frozen at prereg time (fetched 2026-11-04):
  - granite = ibm-granite/granite-3.1-3b-a800m-instruct config.json
    (huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct/raw/main/config.json):
    hidden_size=1536, intermediate_size=512, num_local_experts=40,
    num_hidden_layers=32, num_experts_per_tok=8, torch_dtype=bfloat16.
    (Matches ollama granite3.1-moe:3b traced in exp650; E=40/L=32/k=8 == trace.)
  - qwen = Qwen/Qwen3-30B-A3B config.json
    (huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json):
    hidden_size=2048, moe_intermediate_size=768, num_experts=128,
    num_hidden_layers=48, num_experts_per_tok=8, torch_dtype=bfloat16.

## Formulas (frozen)
- b_e (bytes/expert, bf16) = 3 * hidden * inter * 2
  (gate+up+down FFN matrices; 2 bytes/param).
  granite: 3*1536*512*2 = 4,718,592 B = 4.5 MiB.
  qwen:    3*2048*768*2 = 9,437,184 B = 9.0 MiB.
- cache_GiB(cap) = cap * b_e / 2^30. Total expert bytes = E*L*b_e.
- accesses_per_token (APT): computed from each trace's decode records;
  asserted == L*k (32*8=256 granite, 48*8=384 qwen).
- premium_bytes_per_token(cap) = spread_pp(cap)/100 * APT * b_e.
  (Miss-traffic difference between best and worst policy per decoded token.)
- WS20 = number of DISTINCT keys in the final 20% of the decode access stream
  (per trace, computed from the frozen .pt; keys = (layer, expert) pairs as in
  exp704's loader).
- CROSS1: taken verbatim from sealed B2_CROSS1 (768 / 1792 / NOT_REACHED).
  CROSS1_GiB = CROSS1 * b_e / 2^30. lo lower bound = 2048 * b_e / 2^30.

## Pre-registered gates (frozen before any real run)
- C1 HARDWARE-MAPPING: CROSS1_GiB(granite) <= 4.0 AND CROSS1_GiB(hi) >= 15.0.
  Claim: the 3B-class model's policy-sensitive zone ends inside a small GPU;
  the 30B-class trace needs >= 15 GiB of pure expert cache to exit it.
- C2 WORKING-SET-MECHANISM: for granite and hi:
  |CROSS1_slots - WS20| / WS20 <= 0.30; for lo: WS20 >= 2048
  (consistency with NOT_REACHED). Claim: CROSS1 tracks the steady-state
  working set, not a fixed fraction of key space.
- C3 CLIFF-SHAPE: per family with CROSS1 reached: max spread_pp over caps
  strictly below CROSS1 >= 10 * spread_pp at CROSS1. (lo: report only.)
- C4 ANCHORS/DETERMINISM: (a) computed byte constants equal the frozen values
  above (exact); (b) APT == L*k per trace (exact); (c) spread_pp recomputed at
  first/mid/last cap from sealed policy_grid (max-min of whole hits, x100)
  matches sealed spread_pp within 0.01 for all 3 traces; (d) WS20 computed
  twice -> byte-identical.
- Verdict: COST_CURVES if C4 holds and C1&C2&C3 evaluated honestly (any gate
  may fail honestly; verdict records per-gate PASS/FAIL). Exit codes:
  0 settled | 2 determinism | 3 structure | 5 anchor.

## Scope honesty
Hit-derived RATIOS carry over from exp704 by construction (per-family byte
multiplier is constant); new measured content = (a) absolute byte/GiB tables +
per-token miss-traffic pricing, (b) C1 hardware claim, (c) C2 working-set
mechanism computed fresh from the frozen traces. Single-workstation analysis;
no engine claims.
