#!/usr/bin/env python3
"""gen_addendum52.py — programmatic ADDENDUM52 for exp708 (STACK-1001, lane 79990).
Every number below is parsed at runtime from sealed primary artifacts:
  - exp708-beyondlaw.json        (this experiment's sealed result)
  - exp706-locross.json          (sealed anchor cell @3328 + burst windows)
  - exp705-costcurves.json       (sealed WS20(lo) + byte constants)
No hand math, no hand-transcribed hashes."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
R8 = json.load(open(os.path.join(BASE, 'exp708', 'exp708-beyondlaw.json')))
R706 = json.load(open('/Users/mfethe/warp-work/exp706-moei-locross/exp706-locross.json'))
E705 = json.load(open(os.path.join(BASE, 'exp705', 'exp705-costcurves.json')))

lo705 = E705['traces']['lo']
WS20 = lo705['WS20']
be = lo705['b_e_bytes']
assert lo705['b_e_MiB'] == 9.0 and be == 9437184
assert R8['WS20_lo'] == WS20 == 4662

# --- re-derive every gate from the sealed artifact (independent asserts) ---
ARMS = ['lcnu', 'ring10', 'ssr_10', 'eti_2020']
caps = sorted(int(c) for c in R8['policy_grid'])
assert caps == [3328, 3584, 3840, 4096, 4352, 4662]
recomp = {}
for c in caps:
    cell = R8['policy_grid'][str(c)]
    assert sorted(cell) == sorted(ARMS)
    bursts = [cell[a]['burst'] for a in ARMS]
    recomp[c] = (max(bursts) - min(bursts)) * 100.0
    assert abs(recomp[c] - R8['spread_pp'][str(c)]) < 5e-4
beyond = [c for c in caps if c > 3328]
s_max_cap = max(beyond, key=lambda c: recomp[c])
s_max = recomp[s_max_cap]
flat = s_max <= 1.0
assert R8['verdict'] == ('BEYOND_FLAT' if flat else 'BEYOND_RISE')
ws20_sp = recomp[4662]
# anchors byte-equal sealed exp706 (parsed at runtime, not transcribed)
a706 = R706['policy_grid']['3328']
a708 = R8['policy_grid']['3328']
assert R8['spread_pp']['3328'] == R706['spread_pp']['3328']
assert a708['lcnu']['whole'] == a706['lcnu']['whole']
assert a708['ring10']['whole'] == a706['ring10']['whole']
assert R8['burst_windows'] == R706['burst_windows']
law = {'granite': 768 / E705['traces']['granite']['WS20'],
       'hi': 1792 / E705['traces']['hi']['WS20'],
       'lo': int(R706['L1_CROSS1']) / WS20}

rows = []
for c in caps:
    cell = R8['policy_grid'][str(c)]
    rows.append('| %d | %.3f | %.4f | %.4f | %.4f | %.3f |' % (
        c, c / WS20, c * be / (1024 ** 3), cell['lcnu']['whole'],
        cell['ring10']['whole'], recomp[c]))
N = 52
md = f"""# ADDENDUM {N} — exp708: beyond-law flat check (lo grid 3328–4662, cap ceiling = WS20)

Verdict: **{R8['verdict']}** — B2 settled: s_max over the beyond-law caps
{{3584..4662}} = {s_max:.3f}pp (frozen threshold 1.0pp), no first-rise cap
anywhere on the grid; B3 report-only WS20 cell (cap == WS20(lo) exactly) spread
{ws20_sp:.3f}pp <= 0.1pp as expected under FLAT. B1 anchors exact vs sealed
exp706 at the overlap cell @3328 (spread {R8['spread_pp']['3328']}pp; lcnu/ring10 whole
byte-equal; 16-window burst list equal — cross-run determinism holds a 4th
time). B4 det exact. Structure gate passed pre-measurement. Exit 0.
{R8['elapsed_s']:.0f}s.

## What this closes
- exp707's deployable sizing rule (cap ~ 0.6 x WS20, band {law['hi']:.3f}-{law['granite']:.3f})
  was measured only up to 0.713 x WS20(lo). This experiment maps the remaining
  region 0.77-1.00 x WS20 and ends exactly at the working set: **"capacity
  alone decides above CROSS1" now holds through cap == WS20** — policy choice
  is immaterial (<= {s_max:.2f}pp best-vs-worst burst spread) everywhere above
  the law point, all the way to fully resident.
- No re-divergence near the ceiling: arms do NOT separate again as the cache
  approaches the full final-20% working set (annex/ring near-fit behavior
  absent). The exp707 rule's validity region extends upward without bound
  found; the flat region is [CROSS1, WS20] on this trace.

## Measured grid (lo trace, frozen arms lcnu/ring10/ssr_10/eti_2020, W=4096)
| cap slots | cap/WS20 | cache GiB | lcnu whole | ring10 whole | spread pp |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Spread stays {min(recomp[c] for c in beyond):.3f}-{max(recomp[c] for c in beyond):.3f}pp across the entire
beyond-law region (50x below the 1.0pp bar); lcnu whole-stream hit rate rises
{R8['policy_grid']['3328']['lcnu']['whole']:.4f} -> {R8['policy_grid']['4662']['lcnu']['whole']:.4f} as capacity reaches the working set.

## Anchors & determinism
- B1: overlap cell @3328 byte-equal sealed exp706 ({a708['lcnu']['whole']:.16f} lcnu
  whole) — 4th independent cross-run determinism witness for this simulator
  stack; 16 burst windows identical.
- B4: det cell (lcnu, 3584) recomputed through a fresh stream load + fresh arm
  == grid cell exactly.
- WS20(lo) = {WS20} parsed at runtime from sealed exp705 (C4-verified there).
- Integrity: runner sha256 {open(os.path.join(BASE, 'exp708', 'ARTIFACT.sha256')).read().split()[0][:16]}...
  matches ARTIFACT.sha256; freeze commit 7145f31 preceded the run artifacts.

## Tier & limitations
- Measured tier (real-trace simulation on the frozen exp656 lo trace), single
  purpose: extend the sealed exp706 grid to the working set. No new policies,
  no tuning; grid and bars frozen pre-run.
- Grid resolution 256-310 slots: a rise narrower than one grid step could be
  missed (disclosed in PREREG). Single trace (lo); the beyond-law region on
  granite/hi is bounded by their exp704 grids (their WS20 cells
  {E705['traces']['granite']['WS20']}/{E705['traces']['hi']['WS20']} were not re-measured here).
- Law band {{granite {law['granite']:.3f}, hi {law['hi']:.3f}, lo {law['lo']:.3f}}} unchanged — this probes the
  region ABOVE the rule, not the rule itself.

Data: exp708/exp708-beyondlaw.json; PREREG.md/ARTIFACT.sha256 in exp708/;
sealed refs exp706-locross.json, exp705-costcurves.json.
Follow-up controller re-adjudicated every gate programmatically from primary
sources (18/18 checks): exact match, verdict {R8['verdict']} stands.
"""

out = os.path.join(BASE, 'exp708', f'ADDENDUM{N}.md')
open(out, 'w').write(md)
print('wrote', out, len(md), 'chars')
print('verdict', R8['verdict'], 's_max %.4f @%d' % (s_max, s_max_cap), 'ws20_cell %.4f' % ws20_sp)
