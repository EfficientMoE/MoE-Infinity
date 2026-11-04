# exp696 — Composite admission: revisit annex + churn ring (sring)

Pre-registered before run. Freeze hash after commit.

## Chain
exp693 ssr free lunch vs lcnu; exp694 ring10 steep trade (+2.34/-8.86
@cap1024), neither dominates ssr; exp695 burst = 99.9% re-accesses of
prefix-seen keys (a=6 b=4 c=4086). The two mechanisms act on disjoint
populations: ring quarantines FIRST-SIGHTS (hot set shielded -> rec
gain, burst cost); ssr protects REVISITS (burst re-entries resident ->
burst win, rec neutral). exp696 composes them: SringLCNU (annex 10% +
ring 10% on LCNU; first-sight miss -> ring, revisit miss -> annex,
hit -> promotion; victim: ring-full drain, else batch-exclude annex,
fallback annex drain).

## Bars (d4096; PP; caps 512/1024/2048/4096/8192)
- HARM gate (all caps, else HARM): sring rec >= lcnu rec - 0.10 AND
  sring burst >= lcnu burst - 0.50.
- PRIMARY COMPOSITE_DOMINATES: cap in {1024, 2048}: sring rec >= ring
  +0.10 AND sring burst >= ring +0.10.
- SECONDARY ADDITIVE_REFUND: cap in {1024, 2048}: sring rec >= ring
  -0.10 AND sring burst >= ring +1.00.
- else TRADE_UNCHANGED. MECH (report-only): sring class-c hits >= ring
  class-c hits @1024/2048.

## Gates
REPL lcnu/ring10/ssr_10/center == exp694 exactly (both depths, all
caps; exit 4). CONSIST class counts equal across arms (exit 3).
det sring cap2048 d4096 double-pass (exit 2).

## Smoke (mini N=8192 d1024 cap512, pre-freeze)
lcnu .1104/.6035; ring10 .0537/.6536; ssr_10 .1143/.6038;
sring_10 .0498/.6361; c-hits sring 26 > ring 20 (annex refund visible
at mini); classes policy-independent; det micro True; 4 arms distinct.

- STATUS: NEXT
