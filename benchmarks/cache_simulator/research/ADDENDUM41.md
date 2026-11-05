# Addendum 41 — exp697: earn-then-immune (HARM by gate artifact; interior point found)

## Verdict: HARM — registration artifact, disclosed and self-caught

The pre-registered HARM gate demanded eti burst >= lcnu burst - 0.50pp
at every cap. ring10 itself fails that clause by construction (its
burst is -5.96pp vs lcnu at cap512); the gate was written for
free-lunch composites, not rec-trading policies. Registration error,
not a policy failure. Bars below use ring as the honest anchor.

## The substantive result: the frontier's interior is NOT empty

exp696's churn arithmetic predicted ETI collapses onto ring (earn
event cannot fire under flood churn). REFUTED — d4096:

- cap1024: eti 0.0623 / 0.7830 vs ring 0.0488 / 0.7898
  (burst +1.35pp, rec -0.68pp; rec still +1.66pp above lcnu)
- cap2048: eti 0.1216 / 0.8765 vs ring 0.0957 / 0.8816
  (burst +2.59pp, rec -0.51pp; +0.93pp above lcnu)
- cap512: +0.73pp burst / -0.71pp rec vs ring
- cap4096: +0.37pp burst / +0.02pp rec (tiny rec win as well)
- cap8192: convergence, all arms equal

Exchange rate: ~0.27–0.30pp rec per pp of burst — 5–10x flatter than
the lcnu->ring trade (~3.8pp burst per pp rec). ETI traces a strictly
better curve than moving the ring fraction: it recovers burst while
keeping most of the rec gain.

## Mechanism (exact arithmetic)

The burst recovery is carried ENTIRELY by class-c carry-in hits:
eti 254 vs ring 199 at cap1024 (+55/4086 = +1.35pp, exact);
497 vs 391 at cap2048 (+106/4086 = +2.59pp, exact). b-class hits
identical (1/4). The earn event fires: a carry-in that survives ring
drain to its first in-window re-hit is annex-protected, so its LATER
in-window re-accesses hit. The earn-then-immune rule converts exactly
the carry-in population ring was sacrificing.

## Next

exp698 registered: ETI fraction sweep (annex x ring in {5,10,20}%) at
the trade-band caps to trace the achievable frontier curve and test
whether the flat exchange rate persists or steepens toward ssr.

Data: exp697-earnimmune.json (exit 0; det True; REPL exact vs exp694
both depths; freeze c8bb520).
