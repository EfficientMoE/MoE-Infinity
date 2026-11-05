#!/usr/bin/env python3
"""gen_addendum55.py — programmatic ADDENDUM55 for exp711 (STACK-1001, lane 79990).
Every number is parsed at runtime from sealed primary artifacts:
  - exp711/exp711-mgfactorial.json  (this experiment's sealed result)
  - exp674-hl-transfer.json         (sealed H0 bookends cf_center / cf_hlonly / cf_bo + bo_params)
  - exp670-synth.json               (sealed stream sha8 list)
  - exp711/ARTIFACT.sha256 + freeze commit 4a015bc blobs (git show)
The adjudicator below is written independently of run_mgfactorial.py and re-derives every
gate and the attribution ladder from the grid; it asserts agreement with the runner.
No hand math, no hand-transcribed hashes."""
import hashlib
import itertools
import json
import os
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(BASE, 'exp711')
R = json.load(open(os.path.join(EXP, 'exp711-mgfactorial.json')))
X674 = json.load(open(os.path.join(BASE, 'exp674-hl-transfer.json')))
X670 = json.load(open(os.path.join(BASE, 'exp670-synth.json')))
LOCAL674 = '/Users/mfethe/warp-work/exp674-moei-hl-transfer/exp674-hl-transfer.json'
LOCALRES = '/Users/mfethe/warp-work/exp711-moei-mgfactorial/exp711-mgfactorial.json'
FREEZE = '4a015bc'
assert json.load(open(LOCAL674)) == X674, 'published exp674 differs semantically from runner input'

checks = []


def chk(name, ok):
    checks.append((name, bool(ok)))
    assert ok, name


sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()  # noqa: E731
chk('published result byte-equal local sealed result', sha(os.path.join(EXP, 'exp711-mgfactorial.json')) == sha(LOCALRES))

CENTER = (0.005, 0.0025, 32768.0)
BO = tuple(float(X674['bo_params'][k]) for k in ('margin', 'gamma', 'half_life'))
chk('rules center/bo == exp674 bo_params (runtime parse)', tuple(R['rules']['center']) == CENTER and tuple(R['rules']['bo']) == BO)
SAFE_PP, SW_MAX, INF_PP = 0.5, 6, 0.1
chk('bars frozen', (R['rules']['safety_pp'], R['rules']['switch_max'], R['rules']['informative_pp']) == (SAFE_PP, SW_MAX, INF_PP))

# corners enumerated independently (itertools over factor presence)
CORNERS = [('M' if m else '0') + ('G' if g else '0') + ('H' if h else '0') for h, g, m in itertools.product((0, 1), repeat=3)]
chk('8 corners == runner set', sorted(CORNERS) == sorted(R['rules']['corners']) and len(set(CORNERS)) == 8)
params = {c: tuple(BO[i] if c[i] != '0' else CENTER[i] for i in range(3)) for c in CORNERS}
for c in CORNERS:
    chk(f'corner {c} params == runner', tuple(R['arms'][c]['params']) == params[c])

key = lambda cfg: json.dumps(cfg, sort_keys=True)  # noqa: E731
G = R['grid']
chk('11 streams x 6 caps', len(G) == 11 and sum(len(s['caps']) for s in G) == 66)
chk('11 stream sha8 == exp670', [s['seed'] for s in G] == [s['seed'] for s in X670['grid']])
ref = {(key(s['cfg']), c): cell for s in X674['grid'] for c, cell in s['caps'].items()}

BOOK = {'000': 'cf_center', '00H': 'cf_hlonly', 'MGH': 'cf_bo'}
mm = sum(cell['cf_' + c] != ref[(key(s['cfg']), cap)][ra] for s in G for cap, cell in s['caps'].items() for c, ra in BOOK.items())
sa = sum(cell[a] != ref[(key(s['cfg']), cap)][a] for s in G for cap, cell in s['caps'].items() for a in ('lcnu', 's3fifo'))
chk('H0 198/198 bookend cells exact vs exp674', mm == 0)
chk('H0 standalones 132/132 exact vs exp674', sa == 0)
chk('H0 runner 330 cells ok', R['H0_replication']['cells'] == 330 and R['H0_replication']['ok'] and not R['H0_replication']['mismatches'])
acct = sum(cell['cf_' + c]['rate'] != cell['cf_' + c]['prop'] for s in G for cell in s['caps'].values() for c in CORNERS)
chk('H4 accounting 528/528 loop==prop', acct == 0 and R['H4_accounting'] is True)
chk('H4 determinism two full runs', R['determinism_two_runs'] is True)

per = {c: dict(worst=None, wc=None, over=[], sw=0, maxsw=0, deltas=[]) for c in CORNERS}
inf = 0
for s in G:
    for cap, cell in s['caps'].items():
        lc, s3 = cell['lcnu']['rate'], cell['s3fifo']['rate']
        if abs(lc - s3) * 100 < INF_PP:
            continue
        inf += 1
        best = max(lc, s3)
        for c in CORNERS:
            a = cell['cf_' + c]
            p = per[c]
            d = (best - a['rate']) * 100
            p['deltas'].append((a['rate'] - cell['cf_000']['rate']) * 100)
            p['maxsw'] = max(p['maxsw'], a['switches'])
            if p['worst'] is None or d > p['worst']:
                p['worst'], p['wc'] = d, (s['cfg'], cap)
            if d > SAFE_PP:
                p['over'].append((s['cfg'], cap, round(d, 3)))
            p['sw'] += a['switches'] > SW_MAX
chk('59 informative cells', inf == 59 == R['informative_cells'])
safe = {c: not per[c]['over'] and per[c]['sw'] == 0 for c in CORNERS}
mean = {c: sum(per[c]['deltas']) / len(per[c]['deltas']) for c in CORNERS}
for c in CORNERS:
    a = R['arms'][c]
    p = per[c]
    chk(f'corner {c} worst/over/maxsw/mean/H1 == runner',
        a['worst']['deficit_pp'] == round(p['worst'], 3) and a['cells_over_bar'] == len(p['over'])
        and a['cells_over_switch'] == p['sw'] and a['max_switches'] == p['maxsw'] and a['H1_safe'] == safe[c]
        and a['mean_delta_vs_000_pp'] == round(mean[c], 4))

# independent ladder (re-written from PREREG text, not imported)
if not safe['MG0']:
    tag = {(False, True): 'M-DRIVEN', (True, False): 'G-DRIVEN', (False, False): 'BOTH-KNOBS', (True, True): 'SYNERGY'}
    verdict = 'MG-BREAK:' + tag[(safe['M00'], safe['0G0'])]
elif safe['MGH']:
    verdict = 'NO-BREAK'
else:
    tag = {(False, True): 'MxH', (True, False): 'GxH', (False, False): 'BOTH', (True, True): 'THREE-WAY'}
    verdict = 'HL-INTERACTION:' + tag[(safe['M0H'], safe['0GH'])]
if mm or sa or acct:
    verdict = 'ACCOUNTING-FAIL'
chk('verdict re-derived == runner', verdict == R['verdict'])


def effects(vals):
    out = {}
    for fs in ('M', 'G', 'H', 'MG', 'MH', 'GH', 'MGH'):
        tot = 0.0
        for c in CORNERS:
            sgn = 1
            for f in fs:
                sgn *= 1 if f in c else -1
            tot += sgn * vals[c]
        out[fs] = round(tot / 4.0, 4)
    return out


eff_w = effects({c: round(per[c]['worst'], 3) for c in CORNERS})
eff_m = effects({c: round(mean[c], 4) for c in CORNERS})
chk('Yates effects (worst) == runner', eff_w == R['effects']['worst_deficit_pp'])
chk('Yates effects (mean delta) == runner', eff_m == R['effects']['mean_delta_pp'])

# exp674 cross-checks
chk('MGH worst == exp674 worst_bo_deficit', round(per['MGH']['worst'], 3) == X674['worst_bo_deficit']['deficit_pp'])
v674 = {(key(v['cfg']), v['cap']) for v in X674['violations']}
mgh_over = {(key(o[0]), o[1]) for o in per['MGH']['over']}
chk('MGH over-bar cells == exp674 violation cells', mgh_over == v674)
chk('00H mean delta == exp674 mean_delta_hlonly_pp', round(mean['00H'], 4) == X674['mean_delta_hlonly_pp'])

# integrity: ARTIFACT == working files == freeze-commit blobs; freeze precedes result
art = dict(l.split()[::-1] for l in open(os.path.join(EXP, 'ARTIFACT.sha256')).read().strip().splitlines())
REPO = subprocess.check_output(['git', '-C', BASE, 'rev-parse', '--show-toplevel'], text=True).strip()
rel = os.path.relpath(EXP, REPO)
for f in ('run_mgfactorial.py', 'PREREG.md', 'smoke.log'):
    blob = subprocess.check_output(['git', '-C', REPO, 'show', f'{FREEZE}:{rel}/{f}'])
    chk(f'{f} sha == ARTIFACT == freeze blob', art[f] == sha(os.path.join(EXP, f)) == hashlib.sha256(blob).hexdigest()
        == sha(os.path.join('/Users/mfethe/warp-work/exp711-moei-mgfactorial', f)))
freeze_t = int(subprocess.check_output(['git', '-C', REPO, 'log', '-1', '--format=%ct', FREEZE], text=True).strip())
chk('freeze commit precedes result mtime', freeze_t < os.path.getmtime(LOCALRES))

# descriptive
def cfgs(cfg):
    return f"s={cfg['s']} lam={cfg['lam']} E={cfg['E']}/L={cfg['L']}"


over_sets = {c: {(key(o[0]), o[1]) for o in per[c]['over']} for c in CORNERS}
g_union = over_sets['0G0'] | over_sets['0GH']
rows = '\n'.join('| %s | M=%s G=%s HL=%s | %s | %.3f | %d | %d | %+.4f | %s |' % (
    c, 'BO' if c[0] == 'M' else 'ctr', 'BO' if c[1] == 'G' else 'ctr', 'BO' if c[2] == 'H' else 'ctr',
    {'000': 'exp674 cf_center', '00H': 'exp674 cf_hlonly', 'MGH': 'exp674 cf_bo'}.get(c, 'new'),
    per[c]['worst'], len(per[c]['over']), per[c]['maxsw'], mean[c], 'PASS' if safe[c] else '**FAIL**') for c in CORNERS)
erow = lambda e: ' | '.join(f'{k} {v:+.4f}' for k, v in e.items())  # noqa: E731
og = '\n'.join(f"| {cfgs(o[0])} | {o[1]} | {o[2]:.3f} | {'yes' if (key(o[0]), o[1]) in over_sets['MG0'] else 'no'} | {'yes' if (key(o[0]), o[1]) in v674 else 'no'} |" for o in per['0G0']['over'])
N = 55
md = f"""# ADDENDUM {N} — exp711: 2^3 factorial attribution of exp674's SAFETY-BROKEN BO point (M, G, HL)

Verdict: **{R['verdict']}** — the break lives in GAMMA. At center HL, moving G alone to the
exp669 BO value (gamma {BO[1]:.6g}, {BO[1]/CENTER[1]:.2f}x center) breaks the frozen no-harm bar on
{len(per['0G0']['over'])}/{inf} informative cells (worst {per['0G0']['worst']:.3f}pp); moving M alone to BO (margin
{BO[0]:.6g}) is safe and in fact *improves* the worst cell ({per['M00']['worst']:.3f}pp vs center
{per['000']['worst']:.3f}pp). The joint (M,G) move at center HL still fails ({len(per['MG0']['over'])} cells, worst
{per['MG0']['worst']:.3f}pp), so the ladder's first rung fires with M safe / G failing. H0 three-bookend
replication 330/330 cells exact vs sealed exp674 (cf_center, cf_hlonly, cf_bo + standalones);
H4 accounting 528/528 + two full runs bit-identical. Exit 0, {R['elapsed_s']:.0f}s.

## Per-corner results (59 informative cells; bars frozen from exp674)
| corner | knobs | sealed replica? | worst deficit pp | cells > {SAFE_PP}pp | max switches | mean delta vs 000 pp | H1 |
|---|---|---|---|---|---|---|---|
{rows}

## 2^3 effects (descriptive, Yates /4; no gate)
- Worst-cell deficit: {erow(eff_w)}
- Mean paired delta: {erow(eff_m)}

G is the only positive main effect on worst-cell harm ({eff_w['G']:+.3f}pp); M is protective
({eff_w['M']:+.3f}pp) and the MxG interaction is negative ({eff_w['MG']:+.3f}pp): a smaller margin
partially cancels the gamma damage (0G0 {per['0G0']['worst']:.3f} -> MG0 {per['MG0']['worst']:.3f}pp at center HL;
0GH {per['0GH']['worst']:.3f} -> MGH {per['MGH']['worst']:.3f}pp at BO HL). HL adds a modest main effect
({eff_w['H']:+.3f}pp, consistent with exp710's monotone HL dial) and widens G's footprint
(0G0 {len(per['0G0']['over'])} -> 0GH {len(per['0GH']['over'])} cells over the bar). Switches never exceed {max(p['maxsw'] for p in per.values())} on any
corner (bar {SW_MAX}): every failure is a pure hit-rate deficit.

G-only (0G0) cells over the bar:
| stream | cap | deficit pp | also over in MG0? | also an exp674 joint-point violation? |
|---|---|---|---|---|
{og}

## What this settles
- exp674's SAFETY-BROKEN BO point is a **gamma** problem. exp710 had already cleared HL up to
  the BO value; exp711 shows M is benign-to-helpful and G carries the harm (alone and in every
  context it was tested). The MGH bookend reproduces exp674 exactly: worst
  {per['MGH']['worst']:.3f}pp and the identical {len(mgh_over)}-cell violation set.
- Pre-registered consequence (MG-BREAK:G-DRIVEN): next = single-knob GAMMA dose line at center
  M/HL to find gamma's certified band (as exp710 did for HL); registered as exp712.
- Union of G-carrying failures (0G0 ∪ 0GH) covers {len(g_union & v674)}/{len(v674)} of exp674's violation cells.

## Integrity, determinism, disclosures
- run_mgfactorial.py sha256 {art['run_mgfactorial.py'][:16]}..., PREREG.md {art['PREREG.md'][:16]}..., smoke.log
  {art['smoke.log'][:16]}... == ARTIFACT.sha256 == freeze-commit {FREEZE} blobs == local run dir;
  freeze pushed before launch and precedes the result mtime. Disclosed smoke (S1-S4) passed
  pre-freeze; no smoke-caused runner edits.
- Published exp674-hl-transfer.json asserted parse-equal to the local sealed copy the runner read.
- Synthetic streams only (11 cfgs from exp670); no real-trace or serving claim.

Data: exp711/exp711-mgfactorial.json (run.log local, gitignored); PREREG.md / ARTIFACT.sha256 /
smoke.log / LEDGER.md in exp711/. Follow-up controller re-adjudicated every gate with an
independently written adjudicator ({len(checks)}/{len(checks)} checks): verdict {R['verdict']} stands.
"""
out = os.path.join(EXP, f'ADDENDUM{N}.md')
open(out, 'w').write(md)
print('wrote', out, len(md), 'chars; verdict', verdict, 'checks', len(checks), 'eff_w', eff_w,
      'g_union_cover', len(g_union & v674), '/', len(v674))
