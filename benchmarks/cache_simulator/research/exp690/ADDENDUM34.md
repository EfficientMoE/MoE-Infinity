# Addendum 34 — exp690: 2-way lcnu+ring and double-window confirmation (REC_ONLY)

Four experiments (exp686a ringfit3, exp687 windowed, exp689 visibility
arms, this run) now triangulate the same frontier at cap512: the burst
window wants the LCNU inner, the recovery window wants the RING, and
every rate-based arbiter so far parks on one of them permanently.

## Design

- T1 twoway_lr: sealed 2-way CostFit with the s3fifo SLOT replaced by
  RingProbationLCNU(0.10) — no s3fifo to hide on.
- T2 twoway_cf: T1 + double-window confirmation (challenger must clear
  the sealed barrier two consecutive windows; exp687's anti-thrash
  prescription).
- T3 fit3_cf: exp686a's RingFit3(0.10) + the same confirmation.
- Triple replication gates: center==exp681, ring10==exp685,
  fit3_10==exp686a, per-cell exact (all passed; det grid identical;
  repair freeze c50a310 fixed only reference arm-name mapping).

## Verdict: REC_ONLY (det True; triple repl exact)

- ALL THREE T-arms converge to the standalone ring10 endpoint at the
  home cell: burst 0.0288, rec 0.6853,
  loop 0.6804. Removing s3fifo does NOT keep
  lcnu through burst — the scoreboard switches to ring once (switch
  counts {'twoway_lr': [1, 2], 'twoway_cf': [1], 'fit3_cf': [1, 2]}) and never returns: ring's decayed cumulative rate
  dominates after the loop phase, so the burst ALWAYS inherits the
  ring (burst_def 5.96pp at home, worst
  9.23pp across cells).
- Double-window confirmation is a no-op here: there is nothing to
  confirm (single early switch, zero thrash — exp687's prescription
  solves a problem this menu doesn't have).
- Recovery bar passes everywhere (home rec_def
  -0.73pp) and safety holds for all
  T-arms.

## Conclusion (registered REC_ONLY branch)

Arbitration across rate-scored menus cannot time the burst/recovery
handoff: any menu containing ring adopts it before the burst and pays
in-burst; any menu excluding ring collapses recovery. The burst edge
belongs to the LCNU inner itself (0.0884 vs ring
0.0288 at home). Next (exp691, registered):
miss-signal adoption freeze — block challenger adoption while the
composite's miss rate over the arbitration window exceeds a threshold
(deployable: no segment labels, reacts to the observable burst
signature). If lcnu is held through the burst and ring is adopted only
after the miss rate settles, BOTH bars can pass for the first time.

Data: exp690-confirm.json (elapsed 1545.0s; exit 0).
