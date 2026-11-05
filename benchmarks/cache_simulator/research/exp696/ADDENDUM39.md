# Addendum 39 — exp696: composition test (TRADE_UNCHANGED — conflict pinned)

SringLCNU composed ssr's revisit annex with ring's churn ring on one
LCNU base. Result: the composite is IDENTICAL to ssr (4 decimal places,
burst and rec, every cap 512–8192, both depths).

## What that identity proves

The composite's only free choice was routing REVISIT-misses to the
annex (protection) instead of the ring (quarantine). That single
choice determined everything:

- First-sight quarantine (all the ring did in the composite) delivers
  ZERO rec gain: sring rec 0.7666 == lcnu 0.7664 at cap1024, vs pure
  ring's 0.7898 (+2.34pp).
- And zero burst change: sring burst == ssr burst everywhere.

Therefore ring's ENTIRE recovery win comes from quarantining re-entry
REVISITS — making them churn until they earn a slot by re-hitting.
But re-entry revisits are precisely the population ssr protects
(exp695: the burst window is 99.9% revisits; recovery re-entries are
revisits by definition). The two mechanisms are not disjoint — they
are opposite answers to ONE binary question about ONE population:

  protect revisits on first re-sight (burst win)  vs
  make revisits earn under churn (rec win)

Composition at the population level is impossible. The Pareto frontier
between burst and recovery IS the revisit-miss routing rule. Any
policy claiming both must find a rule strictly between these two
poles.

## The untested middle point

Earn-then-immune: re-entry revisits join the ring (churn risk, ring's
rec semantics) but on their first re-hit they are promoted into the
scorer-immune annex (ssr's protection, one step later). Prediction:
rec keeps ring-level gain (re-entries still earn under churn) while
in-burst repeat accesses (4096-window density) reach annex protection
before eviction → burst recovers part of ssr's edge.

Registered as exp697.

Data: exp696-sring.json (elapsed 1269s + det; exit 0; freeze 0c2e7fb;
REPL exact vs exp694 both depths; det True; consistency PASS).
