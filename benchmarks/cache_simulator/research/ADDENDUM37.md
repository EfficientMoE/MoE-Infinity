# Addendum 37 — exp694: capacity-scaling map of the burst/recovery frontier

exp693 closed the arbitration axis (null capture: the burst-vs-recovery
trade at cap512 is a capacity frontier). exp694 maps WHERE that
frontier sits from cap512 to cap8192.

## Verdict: DUAL_STABLE (det True; repl exp681+exp685 exact)

Table at d4096 (home depth), ring10 vs lcnu:

- cap 512: burst_adv +5.96pp, rec_adv +0.73pp — DUAL
- cap 1024: burst_adv +8.86pp, rec_adv +2.34pp — DUAL (rec edge TRIPLES)
- cap 2048: burst_adv +9.23pp, rec_adv +1.44pp — DUAL
- cap 4096: burst_adv +0.88pp, rec_adv +0.20pp — lcnu catches
- cap 8192: 0.00 / 0.00 — total convergence (every arm 0.9985-1.0;
  the working set fits and policy choice stops mattering)

Findings:

1. The frontier is a MID-CAP phenomenon. It is worst at cap1024-2048
   (~1.5-4x the home working set) and dissolves at saturation.
   Ring's probation/fenced reservation delivers its largest combined
   value exactly where real KV-cache budgets sit.
2. lcnu's ABSOLUTE burst penalty GROWS with capacity (0.0884 at cap512
   -> 0.4099 at cap4096): a bigger cache retains more flood keys, and
   the eviction pressure on the recovery set is not relieved until
   near-saturation.
3. CostFit arbitration is a no-op beyond cap512: center converges
   byte-equal to lcnu at cap>=1024 (both depths) — sealed re-arb
   every 4096 accesses never differentiates when capacity slack
   absorbs the burst.
4. ssr_10 never beats lcnu at burst at any cap (confirms exp693's
   cap2048 negative as depth-specific); safety holds everywhere.

Deployment hint (pre-registered branch DUAL_STABLE): if you can
afford cap >= ~2x working set, policy choice is moot; between 1x and
4x, a probation ring buys up to +9.2pp burst and +2.3pp recovery
over plain LCNU scoring.

exp695 registered: why does lcnu pay burst at ALL caps —
victim-ranking pathology under flood.

Data: exp694-capmap.json (elapsed 822s; exit 0; freeze f3b4795).

## Erratum (same day, pre-any-read): sign convention in the deployment hint

The table and verdict stand (pre-registered bars used
ring_burst_adv = lcnu_burst - ring_burst, i.e. POSITIVE means ring is
WORSE than lcnu during burst — ring SACRIFICES burst hit rate to buy
recovery). The deployment-hint paragraph misread this as a gain:
"a probation ring buys up to +9.2pp burst" is WRONG. Correct reading:

- At cap1024-2048 ring pays its LARGEST burst sacrifice (-8.9pp and
  -9.2pp burst hit rate vs lcnu) to buy its largest recovery gain
  (+2.34pp / +1.44pp). The frontier (trade) is widest there.
- Deployment: if burst-window throughput matters, plain lcnu serves
  the flood better everywhere below saturation; ring only pays off
  when recovery dominates. At cap>=4096 (approaching working-set
  size) the trade nearly vanishes; at 8192 policy is moot.
- Verdict DUAL_STABLE re-read: the TRADE persists at >=3 of 5 caps
  (ring remains measurably different on BOTH axes in its
  characteristic directions), not "ring wins both".
