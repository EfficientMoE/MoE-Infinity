# exp704 — real-trace capacity fine-map: spread-collapse curves (3 traces)

Pre-registered before run. Freeze hash after commit.

## Chain
exp702 (granite) + exp703 (qwen hi/lo): carry-in is a 2-family
law (93-99% on burst windows); the synthetic policy inversion is
universal and sharper on qwen (ring −31pp whole @1024); spread
collapses from ~19pp at cap≈1/6 of key space to ≈0-2pp at 1/3.
exp704 maps that collapse finely on all three frozen real traces
— the curve that says at what cache capacity policy choice stops
mattering on real traffic. Direct input to the cost-integration
layer (cache sizing ↔ policy ↔ hardware budget).

## Protocol
Traces (all frozen, structure-gated as in exp702/703):
- granite3.1-moe-3b (E=40/L=32, prior 8/40; key space 1,280),
  caps {128,192,256,384,512,640,768,896,1024}.
- qwen3-30b-a3b hi (E=128/L=48, prior 8/128; key space 6,144),
  caps {768,896,1024,1152,1280,1408,1536,1792,2048}.
- qwen3-30b-a3b lo, same caps.
Arms: lcnu / ring10 / ssr_10 / eti_2020 (same constructors,
params per family). Burst windows recomputed per trace (same
algorithm as exp702/703: top decile by novelty, window 0
excluded). Per cell: whole + burst hit over full stateful pass.

## Bars (pre-registered)
- B1 DECAY: spread_pp at highest cap < spread_pp at lowest cap,
  per trace (collapse exists).
- B2 CROSS1 (descriptive + boundary): smallest cap per trace with
  spread_pp < 1.0; if none by top cap -> NOT_REACHED recorded.
- B3 RING_NEVER_WINS: ring10 burst < lcnu burst at every cap on
  every trace (running law from exp702/703).
- B4 ANCHORS: |lcnu whole − repo CSV| <= 0.005 at every
  overlapping cap: granite 128=0.283762, 256=0.468164,
  512=0.728697, 768=0.886415, 1024=0.970839; hi/lo
  896/1024/1152/1280 = (0.665581/0.704076/0.738525/0.770724) and
  (0.590768/0.624469/0.656018/0.686120).

## Gates (exit codes)
- 3 STRUCTURE: per trace N, distinct, per-prompt decode counts,
  window count (constants from exp702/703 verified runs).
- 2 DET: one full policy cell per trace rerun byte-identical
  (granite lcnu@256; hi lcnu@1024; lo lcnu@1024).
- 5 ANCHOR: B4 violated.
- Exit 0 settled.

## Budget
36 granite + 72 qwen + 3 det passes ≈ 80-85 min single process.

- STATUS: NEXT
