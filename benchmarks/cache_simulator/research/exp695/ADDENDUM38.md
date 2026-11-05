# Addendum 38 — exp695: flood decomposition (REFUTED — mechanism rewritten)

Both pre-registered hypotheses failed, and the decomposition shows why
— replacing the working model of the burst/recovery frontier.

## Verdict: REFUTED (det True; repl exp694 exact; consistency PASS)

Class counts at d4096 (per 4096-access burst window, policy-independent):
- (a) first-sight: 6 accesses
- (b) same-burst re-access: 4 accesses
- (c) carry-in: 4086 accesses — 99.9%

Bursts do NOT inject novelty. They RE-ACCESS the crowd region that was
seeded during the calm phase. Hypothesis H1 (growth via same-burst
re-accesses) is dead: b_share = 0.0008.

Per-class hits by capacity (d4096):

- lcnu carry-in hits: 359 / 559 / 766 / 1675 / 4086 at caps
  512/1024/2048/4096/8192 — this IS lcnu's entire burst hit rate.
  Growth with capacity = more crowd keys still resident when their
  burst lands.
- ring10 carry-in hits: 118 / 199 / 391 / 1641 / 4086 — the probation
  ring holds HOME keys, so crowd carry-ins miss; H2's direction was
  inverted (ring does not protect carry-ins; carry-ins are its
  victims). Converges to lcnu at cap8192.
- same-burst hits: lcnu 4/4 at cap>=1024 vs ring 1-2/4 — ring's small
  extra sacrifice.

## The rewritten mechanism

burst-vs-recovery is SLOT COMPETITION between two re-access
populations: crowd carry-ins (recently seeded, 4096-window dense) and
home revisits (older, recovery-phase dense). LCNU scoring favors the
crowd (recency); ring fences slots for home. The trade dissolves when
both populations fit (cap8192 -> both 100%).

Actionable consequence: the decisive admission decisions happen in
the CALM SEEDING PHASE (when crowd keys first arrive), not during the
burst. Burst-time arbitration was always too late (exp691); the open
axis is admission-time population policy — throttle/deprioritize
crowd seeding by recognizing novel-population admission rate.

exp696 registered: admission-time population throttling (recognize
seeding phase; bound crowd admissions per window; measure whether
recovery gain survives WITHOUT burst sacrifice, vs the exp694 map).

Data: exp695-flooddecomp.json (elapsed 141s; exit 0; freeze ad3b362).
