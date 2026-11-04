# exp694 — Capacity-scaling map of the burst/recovery frontier

Pre-registered before run. Freeze hash recorded after commit.

## Motivation (registered chain)
exp693 (BOTH_PASS, null capture) closed the arbitration axis: timing,
judging, hysteresis, and selective sparing all fail to move recovery
at cap512 without paying burst — a capacity frontier. ADDENDUM36
registered exp694: map WHERE the frontier sits as capacity grows.

## Question
Does ring10's dual advantage over lcnu at d4096/cap512 (burst +5.96pp,
rec +0.73pp) survive, shrink, or invert across caps 1024..8192? Does
lcnu's burst penalty dissolve with capacity? Where does ssr_10's
negative burst_def (cap2048, exp693) go at higher caps?

## Protocol
exp693's machinery exactly (streams: exp681 spawn scheme 68120261103
spawn(2); single-pass seg metrics; center loop2 acc check; SecondSightLCNU
batch-exclusion repair). Only the CAP grid is new.
- repl_grid: st.BURSTS(256..4096) x st.CAPS(512,2048) x {center, ring10}
  — exact equality vs exp681/exp685 JSONs (exit 4 / 6).
- map_grid: d {1024,4096} x caps {512,1024,2048,4096,8192} x
  arms {lcnu, center, ring10, ssr_10}.

## Bars (d4096 home depth, per cap)
- ring_burst_adv = (lcnu_burst - ring_burst)*100
- ring_rec_adv = (ring_rec - lcnu_rec)*100
- DUAL at C: burst_adv >= 1.0 AND rec_adv >= 0.30
- LCNU_CATCH at C: burst_adv <= 1.0 AND rec_adv < 0.30
Verdicts: DUAL_STABLE (>=3/5 DUAL) | LCNU_CATCHES (>=3/5 catch —
frontier dissolves: capacity alone restores burst AND erases ring's
recovery edge) | MIXED (report transition cap).
Advisory SAFETY: ssr_10 loop >= lcnu - 0.5pp all map cells.
det: d4096 caps {512,8192} rerun byte-identical (exit 2).
Exit: 0 settled | 2 det | 4 repl1 | 6 repl3.

## Smoke (mini, pre-freeze)
N=8192 d=1024 caps {64,512}: arms distinct, ring leads burst+rec,
ssr tracks lcnu, determinism micro True. Placeholder-loop bug fixed
pre-freeze (cell_eval nested dict).

- STATUS: NEXT
