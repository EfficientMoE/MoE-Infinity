# Addendum 32 — exp686a ring as a first-class arbitration candidate (IN_BURST_KEPT_VISIBLE)

exp685's failed gate was the ring inner being INVISIBLE to 2-way
arbitration at small cap (composite rec_def == center's exactly).
exp686a made the ring a THIRD CANDIDATE of the sealed CostFit3
barrier (HALF_LIFE 32768, MARGIN 0.005, GAMMA 0.0025, RESEGMENT
4096, initial lcnu — all inherited unedited; exp681's exact streams;
8 cells; LEDGER safe-settle v2 freeze 2735a8e; dual anchors
byte-exact in smoke). Pre-registered prediction: the ring CANNOT
displace s3fifo at the primary cell d4096/cap512 -> H1 fails.

## Verdict: IN_BURST_KEPT_VISIBLE (det True; repl exact 24 cells)

The prediction failed in the most informative way. On the registered
primary arm (ring 0.25) it held: ringfit3 sat on s3fifo through the
primary cell (rec_def +6.98pp == center's +6.98pp; byte-equal behavior
8/8 cells) and the ladder passed everywhere
(H2). But the ring 0.10 variant ADOPTED the ring at all four cap512
cells — 4/4 — and at the primary cell its recovery is BETTER than lcnu's: rec_def -0.73pp (vs center's +6.98 collapse), loop vs lcnu +0.89pp, sw 3, sw_by {'lcnu': 0, 's3fifo': 1, 'ring': 2}. First-class candidacy
alone CAN recover the recovery collapse — when the ring is small
enough to fit the working set.

## The tradeoff that remains (why not RING-FIRST-CLASS)

No arm holds BOTH bars at cap512. ringfit3_10 pays for its recovery
win in-burst: burst_def +5.96pp vs center's +1.07pp at the primary cell (ladder fails 6/8 cells: worst +9.23pp over center). Riding
the ring through the BURST gives up the s3fifo inner's in-burst
edge; riding s3fifo through the RECOVERY collapses. Under a decayed
cumulative-rate scoreboard the two windows need different judges —
the handoff Michael's ladder wants cannot happen at re-segment
boundaries alone.

## Secondary dose + stride diagnostic (for exp687 / exp686b)

- lazy2 standalone (promotion on 2nd rehit): home rec_def +0.64pp, burst_def +4.03pp — strictly worse than eager ring25 at the home cell; lazy promotion is a dead dose (kept as a
  negative finding).
- Stride (pure stream stats, no policy behavior): d4096 crowd rehit gap mean 1216.1/p90 2555.0 accesses, concurrent-probation load mean 932.0 — yet ring_cap 51 (frac .10 @cap512, far below gap p90) wins recovery
while ring_cap 128 (frac .25) is never adopted: ring sizing tracks
the one-timer share, not the reuse stride; exp686b sweep input.

## Branch consequence (frozen)

IN_BURST_KEPT_VISIBLE -> exp687: arbitration-layer redesign —
windowed challenger scoreboard (challenger judged on last-4096 rate
vs incumbent decayed cumulative, same barrier family) so the
composite can sit s3fifo for the burst and hand off to the ring for
recovery; sealed victim-selection policies untouched.

Data: exp686a-ringfit3.json (elapsed 1471.5s; 2 passes identical; exit 0; settle re-executed with quoting fix, disclosed).
