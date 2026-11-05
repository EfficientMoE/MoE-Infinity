# Addendum 48 — exp704: real-trace capacity fine-map (policy premium curves)

## Verdict: CAPACITY_MAP — B1 PASS 3/3, anchors 13/13, B3 fails only on saturation ties

- B1 DECAY: spread collapses with capacity on all three frozen
  traces. granite 28.5pp@128 → hump 38.2pp@192 → 0.05pp@896;
  hi 24.9pp@768 → 0.03pp@2048; lo 21.0pp@768 → 2.1pp@2048.
- B2 CROSS1 (smallest cap with spread < 1pp): granite **768**
  (60% of its 1,280 key space), hi **1792** (29% of 6,144),
  lo **NOT_REACHED** by 2048 (2.116pp at 33%). Low-imbalance
  (diverse-footprint) traffic keeps policy sensitivity longer
  and decays slower (13.1pp@1408 vs hi 5.8pp).
- B3 RING_NEVER_WINS: FAILS as registered on strict inequality —
  but only on saturation ties: granite@896 ring burst edges lcnu
  by 0.0045pp (whole: lcnu still better, 0.9357 vs 0.9356) and
  hi@2048 is an exact tie (both 0.8927; spread there 0.029pp).
  In the divergence band (spread >= 1pp, 27/27 trace-cap cells),
  ring NEVER wins. Both readings recorded.
- B4 ANCHORS: 13/13 within band — all deltas <= 4.3e-07, i.e.
  byte-equal to the repo CSVs' 6-decimal print precision
  (granite 5/5, hi 4/4, lo 4/4). Cross-simulator replication
  now holds at 13 capacity points across 2 model families.
- Low-cap microstructure: granite spread PEAKS at cap 192
  (38.2pp) — below that, even LCNU is starved and all arms
  converge low; the premium band is widest just above starvation.

## The engineering artifact (cost-integration input)
Per-trace "policy premium curve": below CROSS1, cache-policy
choice is worth up to 38.2pp of burst hit (LCNU vs ring);
above CROSS1, policy choice is immaterial (<1pp) and capacity
alone decides. Sizing rule surfaced by the map: CROSS1 tracks
the steady-state working set (hi crosses at 1792 where whole
hit ~0.91; lo's diverse working set keeps it above 1pp through
2048). For hardware budgeting: the marginal value of smart
policy is bounded by the spread curve; the marginal value of
capacity is the hit-rate curve — exp704 measures both on REAL
traffic, per family.

Data: exp704-capacitymap.json (exit 0; det 3/3 byte-identical;
structure gates exact; 2376s). Freeze 29e8d18.
