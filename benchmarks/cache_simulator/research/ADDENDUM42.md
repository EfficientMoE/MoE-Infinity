# Addendum 42 — exp698: ETI fraction sweep (FLAT_FRONTIER; beyond-pole points exist)

## Verdict: FLAT_FRONTIER — with a disclosed adjudication-code bug

The runner's sealed JSON verdict field reads STEEPENS. That is wrong:
the flat-check implemented dr/db >= -0.01 (0.01pp rec per pp burst —
100x stricter than the pre-registered bar "exchange <= 1.0pp/pp",
which is dr/db >= -1.0 in rate units). The sealed JSON is not edited;
this correction follows the programmatic-addendum discipline used for
ADDENDUM37's erratum. Re-adjudicated per the binding LEDGER spec.

## The frontier family (d4096, anchors: lcnu / ring10 / ssr_10)

- No variant dominates ring10 (honest negative): rec always pays.
- Annex fraction drives burst monotonically; ring fraction gates rec.
  The (annex, ring) sweep traces a smooth interior curve between the
  ring and ssr poles.
- HARM-eligible variants (rec >= lcnu-0.10, burst >= ring-0.50 at all
  caps): eti_0510, eti_1010.
- FLAT exchange carried by BOTH eligible variants:
  - eti_0510: +0.64pp burst / -0.29pp rec at cap1024 (0.45pp/pp);
    +0.90 / -0.15 at cap2048 (0.17pp/pp)
  - eti_1010: +1.35 / -0.68 (0.50); +2.59 / -0.51 (0.20)
  Confirms exp697's ~0.3pp/pp flat rate along the family.

## Beyond-pole finding (cap4096)

eti_2020: 0.4285 / 0.9517 vs ssr_10: 0.4128 / 0.9514 — beats the ssr
pole by +1.57pp burst at +0.03pp rec. eti_2010 (0.4187/0.9521) also
exceeds ssr's burst with ring-level rec. The earn-then-immune family
does not just fill the interior — near saturation it EXCEEDS both
poles. The (burst, rec) achievable set is larger than the
exp693/694 pole map suggested.

## Next

exp699 registered: beyond-pole existence test — finer annex grid
(20-40%) at cap4096 + generalization of the flat exchange across
burst depths d in {256, 1024, 4096} at the two eligible cells.

Data: exp698-etisweep.json (exit 0; det True; anchors == exp694
exactly; eti_1010@2048 == exp697 exactly — cross-experiment
determinism; freeze 71ef7db; 1441s).
