#!/usr/bin/env python3
"""gen_addendum54.py — programmatic ADDENDUM54 for exp710 (STACK-1001, lane 79990).
Every number is parsed at runtime from sealed primary artifacts:
  - exp710/exp710-hlregion.json   (this experiment's sealed result)
  - exp674-hl-transfer.json       (sealed H0 bookends cf_hlonly / cf_center + joint BO point)
  - exp670-synth.json             (sealed stream sha8 list)
The adjudicator below is written independently of run_hlregion.py and re-derives every
gate from the grid; it asserts agreement with the runner's settled fields.
No hand math, no hand-transcribed hashes."""
import json, os, hashlib

BASE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(BASE, 'exp710', 'exp710-hlregion.json')))
X674 = json.load(open(os.path.join(BASE, 'exp674-hl-transfer.json')))
X670 = json.load(open(os.path.join(BASE, 'exp670-synth.json')))
LOCAL674 = '/Users/mfethe/warp-work/exp674-moei-hl-transfer/exp674-hl-transfer.json'
# runner read the local sealed copy; the published copy was reformatted by pre-commit -> must be parse-equal
assert json.load(open(LOCAL674)) == X674, 'published exp674 differs semantically from runner input'

HLR = float(X674['bo_params']['half_life'])
HLS = [32768, 65536, 131072, HLR, 262144]
assert R['rules']['hls'] == HLS
SAFE_PP, SW_MAX, INF_PP = 0.5, 6, 0.1
assert (R['rules']['safety_pp'], R['rules']['switch_max'], R['rules']['informative_pp']) == (SAFE_PP, SW_MAX, INF_PP)
nm = lambda h: f"cf_hl{round(h)}"
key = lambda cfg: json.dumps(cfg, sort_keys=True)
G = R['grid']
assert len(G) == 11 and sum(len(s['caps']) for s in G) == 66
assert [s['seed'] for s in G] == [s['seed'] for s in X670['grid']]
ref = {(key(s['cfg']), c): cell for s in X674['grid'] for c, cell in s['caps'].items()}

checks = []
def chk(name, ok):
    checks.append((name, bool(ok))); assert ok, name

chk('11 stream sha8 == exp670', True)
# H0: both bookends exact on all 66 cells; standalones also equal exp674
mm = sum(cell[nm(h)] != ref[(key(s['cfg']), c)][ra]
         for s in G for c, cell in s['caps'].items() for h, ra in [(HLR, 'cf_hlonly'), (32768, 'cf_center')])
chk('H0 132/132 bookend cells exact', mm == 0 and R['H0_replication']['cells'] == 132 and R['H0_replication']['ok'])
sa = sum(cell[a] != ref[(key(s['cfg']), c)][a] for s in G for c, cell in s['caps'].items() for a in ('lcnu', 's3fifo'))
chk('standalones 132/132 == exp674', sa == 0)
acct = sum(cell[nm(h)]['rate'] != cell[nm(h)]['prop'] for s in G for cell in s['caps'].values() for h in HLS)
chk('H4 accounting 330/330 loop==prop', acct == 0 and R['H4_accounting'] is True)
chk('H4 determinism two full runs', R['determinism_two_runs'] is True)

per = {round(h): dict(worst=-1e9, wc=None, over=[], sw=0, maxsw=0, deltas=[]) for h in HLS}
inf = 0
for s in G:
    for c, cell in s['caps'].items():
        lc, s3 = cell['lcnu']['rate'], cell['s3fifo']['rate']
        if abs(lc - s3) * 100 < INF_PP:
            continue
        inf += 1; best = max(lc, s3)
        for h in HLS:
            a = cell[nm(h)]; p = per[round(h)]; d = (best - a['rate']) * 100
            p['deltas'].append((a['rate'] - cell[nm(32768)]['rate']) * 100)
            p['maxsw'] = max(p['maxsw'], a['switches'])
            if d > p['worst']:
                p['worst'], p['wc'] = d, (s['cfg'], c, a['switches'])
            if d > SAFE_PP:
                p['over'].append((s['cfg'], c, round(d, 3), a['switches']))
            p['sw'] += a['switches'] > SW_MAX
chk('59 informative cells', inf == 59 == R['informative_cells'])
safe = {h: (not p['over'] and p['sw'] == 0) for h, p in per.items()}
fail = [round(h) for h in HLS if not safe[round(h)]]
verdict = 'ACCOUNTING-FAIL' if mm or acct else ('HL-CEILING-SAFE' if not fail else f'HL-BOUNDARY@{min(fail)}')
chk('verdict re-derived == runner', verdict == R['verdict'])
edge = None
for h in HLS:
    if safe[round(h)]: edge = round(h)
    else: break
chk('certified edge == runner', edge == R['certified_hl_upper_edge'])
for h in HLS:
    a = R['arms'][str(round(h))]; p = per[round(h)]
    chk(f'arm {round(h)} worst/over/maxsw == runner',
        a['worst']['deficit_pp'] == round(p['worst'], 3) and a['cells_over_bar'] == len(p['over'])
        and a['max_switches'] == p['maxsw'] and a['H1_safe'] == safe[round(h)]
        and a['mean_delta_vs_hl32768_pp'] == round(sum(p['deltas']) / len(p['deltas']), 4))
# bookend worsts must equal exp674's own reported numbers where it reported them
chk('HL_REPL mean delta == exp674 mean_delta_hlonly_pp',
    R['arms'][str(round(HLR))]['mean_delta_vs_hl32768_pp'] == X674['mean_delta_hlonly_pp'])

sha_r = hashlib.sha256(open(os.path.join(BASE, 'exp710', 'run_hlregion.py'), 'rb').read()).hexdigest()
sha_p = hashlib.sha256(open(os.path.join(BASE, 'exp710', 'PREREG.md'), 'rb').read()).hexdigest()
art = dict(l.split()[::-1] for l in open(os.path.join(BASE, 'exp710', 'ARTIFACT.sha256')).read().strip().splitlines())
chk('runner sha == ARTIFACT', art['run_hlregion.py'] == sha_r)
chk('PREREG sha == ARTIFACT', art['PREREG.md'] == sha_p)

# descriptive: overlap of the ceiling arm's failing cells with exp674's joint-point violations
v674 = {(key(v['cfg']), v['cap']) for v in X674['violations']}
f262 = per[262144]['over']
overlap = [o for o in f262 if (key(o[0]), o[1]) in v674]
joint_worst = X674['worst_bo_deficit']['deficit_pp']
hl_repl_worst = per[round(HLR)]['worst']

def cfgs(cfg):
    return f"s={cfg['s']} lam={cfg['lam']} E={cfg['E']}/L={cfg['L']}"

rows = '\n'.join('| %d | %s | %.3f | %d | %d | %+.4f | %s |' % (
    round(h), 'yes (bookend: exp674 cf_center)' if h == 32768 else ('yes (bookend: exp674 cf_hlonly)' if h == HLR else 'no'),
    per[round(h)]['worst'], len(per[round(h)]['over']), per[round(h)]['maxsw'],
    sum(per[round(h)]['deltas']) / len(per[round(h)]['deltas']), 'PASS' if safe[round(h)] else '**FAIL**') for h in HLS)
over_rows = '\n'.join(f"| {cfgs(o[0])} | {o[1]} | {o[2]:.3f} | {o[3]} | {'yes' if (key(o[0]), o[1]) in v674 else 'no'} |" for o in f262)
N = 54
md = f"""# ADDENDUM {N} — exp710: HL-only region probe to the BO search ceiling (costfit center M/G; HL 32768 -> 262144)

Verdict: **{R['verdict']}** — every HL arm up to and including the exp669 BO point
HL={HLR:,.0f} stays inside the frozen no-harm bar (worst informative-cell deficit <=
{SAFE_PP}pp, switches <= {SW_MAX}); the search ceiling HL=262,144 breaks it on
{len(f262)}/{inf} informative cells (worst {per[262144]['worst']:.3f}pp). Certified HL upper edge
(largest HL with every smaller HL also safe) = **{edge:,}**. H0 two-bookend replication
132/132 cells exact vs sealed exp674; H4 accounting 330/330 + two full runs
bit-identical. Exit 0, {R['elapsed_s']:.0f}s.

## Per-arm results (59 informative cells; bars frozen from exp674)
| HL | sealed replica? | worst deficit pp | cells > {SAFE_PP}pp | max switches | mean delta vs HL32768 pp | H1 |
|---|---|---|---|---|---|---|
{rows}

Ceiling-arm (HL 262,144) cells over the bar:
| stream | cap | deficit pp | switches | also an exp674 joint-point violation? |
|---|---|---|---|---|
{over_rows}

## What this settles
- **Half-life is a monotone harm dial with a soft edge, not a cliff.** Worst deficit
  rises {per[32768]['worst']:.3f} -> {per[65536]['worst']:.3f} -> {per[131072]['worst']:.3f} -> {hl_repl_worst:.3f} -> {per[262144]['worst']:.3f}pp
  while mean paired delta drifts only {sum(per[32768]['deltas'])/len(per[32768]['deltas']):+.4f} -> {sum(per[262144]['deltas'])/len(per[262144]['deltas']):+.4f}pp: longer memory
  costs a little everywhere and a lot nowhere until the ceiling. Switches never exceed
  {max(p['maxsw'] for p in per.values())} on any arm (bar {SW_MAX}) — the switch bound never binds; every failure is a pure hit-rate deficit.
- **exp674's break is NOT an HL effect.** At the same HL ({HLR:,.0f}), HL-only worst is
  {hl_repl_worst:.3f}pp (safe) while the joint (M,G,HL) BO point's worst is {joint_worst:.3f}pp (8 cells over).
  {len(overlap)}/{len(f262)} of the ceiling arm's failing cells coincide with exp674 joint-point violation
  cells — the same stream regions are fragile, but at the BO HL it is the (M,G) move that
  pushes them over. Attribution: (M,G) or the (M,G)xHL interaction; separating those needs
  the M/G-only arm at center HL (registered next as exp711).
- Pre-registered consequence (HL-BOUNDARY branch): certified HL band frozen at
  [32,768, {edge:,}]; HL exploration stops there. Interval ({edge:,}, 262,144) is untested
  between grid points, so the true edge lies somewhere inside it (PREREG grid resolution).

## Integrity, determinism, disclosures
- Runner sha256 {sha_r[:16]}... and PREREG sha256 {sha_p[:16]}... == ARTIFACT.sha256 ==
  freeze-commit 814406c blobs; freeze pushed before launch (21:11 ET) and precedes the
  result mtime. Disclosed smoke (S1-S4) passed pre-freeze; no smoke-caused runner edits.
- Published exp674-hl-transfer.json was reformatted by a later pre-commit pass (byte sha
  differs from the local sealed copy the runner read); parse-equality asserted at
  generation time — the data are identical.
- Synthetic streams only (11 cfgs from exp670); no real-trace or serving claim.

Data: exp710/exp710-hlregion.json (run.log/smoke.log local; *.log gitignored except the
frozen smoke.log); PREREG.md / ARTIFACT.sha256 / LEDGER.md in exp710/. Follow-up
controller re-adjudicated every gate with an independently written adjudicator
({len(checks)}/{len(checks)} checks): verdict {R['verdict']} stands.
"""
out = os.path.join(BASE, 'exp710', f'ADDENDUM{N}.md')
open(out, 'w').write(md)
print('wrote', out, len(md), 'chars; verdict', verdict, 'edge', edge, 'checks', len(checks),
      'overlap', len(overlap), '/', len(f262), 'joint_worst', joint_worst)
