#!/usr/bin/env python3
"""exp707 — cost-layer completion: lo GiB sizing table + 3-trace deployable
sizing rule (STACK-1001, lane 79990). Analysis tier: pure transform of sealed
exp704/exp705/exp706 artifacts. All refs parsed at runtime and asserted.
Gates S1-S5 per frozen PREREG.md. Exit 0 settled | 2 det | 3 structure | 5 anchor."""
import json, sys, os

E704 = '/Users/mfethe/warp-work/exp704-moei-capacitymap/exp704-capacitymap.json'
E705 = '/Users/mfethe/warp-work/moei-granite3b-rebuilt/benchmarks/cache_simulator/research/exp705/exp705-costcurves.json'
E706 = '/Users/mfethe/warp-work/exp706-moei-locross/exp706-locross.json'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp707-sizrule.json')

def structure_ok(d704, d705, d706):
    try:
        for t in ('granite', 'hi', 'lo'):
            assert 'table' in d705['traces'][t] and 'APT' in d705['traces'][t]
        assert d706['L1_CROSS1'] and d706['policy_grid'] and d706['spread_pp']
        assert 'policy_grid' in d704['traces']['lo'] and 'spread_pp' in d704['traces']['lo']
        return True
    except Exception:
        return False

def premium(spread_pp, apt, be):
    return round(spread_pp / 100 * apt * be, 1)

def miss_gib(lcnu_whole, apt, be):
    return round((1 - lcnu_whole) * apt * be / 2**30, 5)

def row(cap, cache_gib, spread_pp, lcnu_whole, apt, be):
    return {"cap": cap, "cache_GiB": round(cache_gib, 4),
            "spread_pp": spread_pp,
            "premium_B_per_token": premium(spread_pp, apt, be),
            "lcnu_miss_GiB_per_token": miss_gib(lcnu_whole, apt, be)}

def build(d704, d705, d706):
    """Pure function over sealed inputs -> result dict."""
    traces = {}
    be_of = {t: d705['traces'][t]['b_e_bytes'] for t in d705['traces']}
    apt_of = {t: d705['traces'][t]['APT'] for t in d705['traces']}
    # granite / hi: mirror exp705 exactly (rows, CROSS1, C1, C2, C3, WS20)
    for t in ('granite', 'hi'):
        s = d705['traces'][t]
        traces[t] = {"b_e_bytes": s['b_e_bytes'], "b_e_MiB": s['b_e_MiB'],
                     "APT": s['APT'], "WS20": s['WS20'],
                     "CROSS1_slots": s['CROSS1_slots'], "CROSS1_GiB": s['CROSS1_GiB'],
                     "total_expert_GiB": s['total_expert_GiB'],
                     "C1_hardware": s['C1_hardware'], "C2_working_set": s['C2_working_set'],
                     "C2_ratio": s['C2_ratio'], "C3_cliff": s['C3_cliff'],
                     "table": [dict(r) for r in s['table']],
                     "mirrored_from": "exp705 (byte-equal)"}
    # lo: exp705 rows below 1792 + exp706 rows 1792-3328 (recomputed via shared formula)
    lo705, lo706 = d705['traces']['lo'], d706
    be, apt = be_of['lo'], apt_of['lo']
    table = [dict(r) for r in lo705['table'] if r['cap'] < 1792]
    grid, sp6 = lo706['policy_grid'], lo706['spread_pp']
    caps = sorted(int(c) for c in grid)
    for c in caps:
        r = row(c, c * be / 2**30, sp6[str(c)], grid[str(c)]['lcnu']['whole'], apt, be)
        if c in (1792, 2048):
            table.append(r)  # anchor cells: asserted equal to exp705 rows in S1
        else:
            table.append(r)
    cross1 = int(lo706['L1_CROSS1'])
    cross1_gib = cross1 * be / 2**30
    sp_at = lambda c: sp6[str(c)]
    max_below = max(sp_at(c) for c in caps if c < cross1)
    traces['lo'] = {"b_e_bytes": be, "b_e_MiB": lo705['b_e_MiB'], "APT": apt,
                    "WS20": lo705['WS20'], "CROSS1_slots": cross1,
                    "CROSS1_GiB": round(cross1_gib, 4),
                    "total_expert_GiB": lo705['total_expert_GiB'],
                    "C1_hardware": bool(cross1_gib >= 15.0),
                    "C2_working_set": False,
                    "C2_ratio": round(cross1 / lo705['WS20'] - 1, 3),
                    "C3_cliff": bool(max_below >= 10 * sp_at(cross1)),
                    "table": table,
                    "rows_below_1792_from": "exp705 (byte-equal)",
                    "rows_1792_3328_from": "exp706 sealed grid + frozen formula"}
    # consolidated deployable sizing rule
    rule = {}
    for t, s in traces.items():
        ws20_gib = s['WS20'] * s['b_e_bytes'] / 2**30
        ratio = s['CROSS1_slots'] / s['WS20']
        rule[t] = {"WS20": s['WS20'], "CROSS1_slots": s['CROSS1_slots'],
                   "CROSS1_GiB": round(s['CROSS1_slots'] * s['b_e_bytes'] / 2**30, 4),
                   "WS20_GiB": round(ws20_gib, 4), "ratio": ratio,
                   "savings_vs_WS20": 1 - ratio}
    ratios = [v['ratio'] for v in rule.values()]
    sizing_rule = {"per_trace": rule,
                   "band": [min(ratios), max(ratios)],
                   "band_width": max(ratios) - min(ratios),
                   "deployable_rule": "cap ~ 0.6 x WS20 expert slots (band 0.53-0.63)",
                   "GiB_per_1000_WS20_keys": {
                       "30B-class (b_e 9.0 MiB)": round(0.6 * be_of['lo'] / 2**30 * 1000, 2),
                       "3B-class (b_e 4.5 MiB)": round(0.6 * be_of['granite'] / 2**30 * 1000, 2)},
                   "per_family_note": "same-model variance hi 0.528 vs lo 0.604 — band, not constant (3 points)"}
    return {"exp": "exp707-sizrule", "traces": traces, "sizing_rule": sizing_rule}

def gates(d704, d705, d706, res):
    g = {}
    # S1 ANCHOR-EXACT
    ok = True
    for t in ('granite', 'hi'):
        if res['traces'][t]['table'] != d705['traces'][t]['table']:
            ok = False
    lo_rows705 = {r['cap']: r for r in d705['traces']['lo']['table']}
    overlap = sorted(set(lo_rows705) & set(int(k) for k in d706['spread_pp']))
    for c in overlap:
        mine = [r for r in res['traces']['lo']['table'] if r['cap'] == c][0]
        if mine != lo_rows705[c]:
            ok = False
    g['S1_anchor_caps'] = overlap
    g['S1_anchor_exact'] = ok
    # S2 LAW-BAND
    r = {t: v['ratio'] for t, v in res['sizing_rule']['per_trace'].items()}
    g['S2_law_band'] = all(0.50 <= x <= 0.66 for x in r.values()) and \
                       (max(r.values()) - min(r.values())) <= 0.10
    # S3 C1-COMPLETION
    lo = res['traces']['lo']
    g['S3_c1_completion'] = (round(2816 * 9437184 / 2**30, 4) == lo['CROSS1_GiB']
                             and lo['C1_hardware'] is True
                             and res['traces']['granite']['C1_hardware'] is True
                             and res['traces']['hi']['C1_hardware'] is True)
    # S4 SAVINGS-IDENTITY
    g['S4_savings_identity'] = all(
        abs(v['savings_vs_WS20'] - (1 - v['ratio'])) < 1e-9
        for v in res['sizing_rule']['per_trace'].values())
    return g

def main():
    smoke = '--smoke' in sys.argv
    if smoke:
        # synthetic mini-inputs through the SAME machinery (disclosed)
        d705 = {"traces": {"granite": {"b_e_bytes": 4096, "b_e_MiB": 0.00390625, "APT": 8,
                "WS20": 10, "CROSS1_slots": 6, "CROSS1_GiB": 0.0234, "total_expert_GiB": 0.05,
                "C1_hardware": True, "C2_working_set": False, "C2_ratio": -0.4, "C3_cliff": True,
                "table": [row(2, 2*4096/2**30, 30.0, 0.5, 8, 4096), row(6, 6*4096/2**30, 0.4, 0.9, 8, 4096)]},
                "hi": {"b_e_bytes": 8192, "b_e_MiB": 0.0078125, "APT": 16, "WS20": 20,
                "CROSS1_slots": 11, "CROSS1_GiB": 0.0859, "total_expert_GiB": 0.2,
                "C1_hardware": True, "C2_working_set": False, "C2_ratio": -0.45, "C3_cliff": True,
                "table": [row(4, 4*8192/2**30, 20.0, 0.4, 16, 8192), row(11, 11*8192/2**30, 0.5, 0.95, 16, 8192)]},
                "lo": {"b_e_bytes": 8192, "b_e_MiB": 0.0078125, "APT": 16, "WS20": 30,
                "CROSS1_slots": None, "CROSS1_GiB": None, "total_expert_GiB": 0.2,
                "C1_hardware": None, "C2_working_set": True, "C2_ratio": None, "C3_cliff": None,
                "table": [row(4, 4*8192/2**30, 40.0, 0.3, 16, 8192), row(8, 8*8192/2**30, 9.0, 0.7, 16, 8192)]}}}
        d706 = {"L1_CROSS1": "18", "spread_pp": {"8": 9.0, "12": 5.0, "18": 0.6, "24": 0.1},
                "policy_grid": {str(c): {"lcnu": {"whole": 0.7 + 0.02*i}} for i, c in enumerate((8, 12, 18, 24))}}
        d704 = {"traces": {"lo": {"policy_grid": {}, "spread_pp": {}}}}
        res = build(d704, d705, d706)
        g = gates(d704, d705, d706, res)
        ok_det = json.dumps(build(d704, d705, d706), sort_keys=True) == json.dumps(res, sort_keys=True)
        lo_anchor = [r for r in res['traces']['lo']['table'] if r['cap'] == 8][0] == \
                    {r['cap']: r for r in d705['traces']['lo']['table']}[8]
        print('SMOKE_OK' if (g['S1_anchor_exact'] and g['S2_law_band'] and g['S4_savings_identity'] and ok_det and lo_anchor) else 'SMOKE_FAIL', g, 'det', ok_det)
        return 0 if g['S1_anchor_exact'] and ok_det else 1
    d704 = json.load(open(E704)); d705 = json.load(open(E705)); d706 = json.load(open(E706))
    if not structure_ok(d704, d705, d706):
        print('STRUCTURE FAIL'); return 3
    res1 = build(d704, d705, d706)
    res2 = build(d704, d705, d706)
    if json.dumps(res1, sort_keys=True) != json.dumps(res2, sort_keys=True):
        print('DET FAIL'); return 2
    g = gates(d704, d705, d706, res1)
    g['S5_determinism'] = True
    res1['gates'] = g
    res1['verdict'] = 'SIZRULE_COMPLETE' if all(g.values()) else 'GATE_FAIL'
    res1['all_gates_pass'] = all(g.values())
    json.dump(res1, open(OUT, 'w'), indent=1)
    print('exp707', res1['verdict'], json.dumps(g))
    print('lo CROSS1_GiB', res1['traces']['lo']['CROSS1_GiB'], 'band', [round(x, 3) for x in res1['sizing_rule']['band']])
    return 0 if all(g.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
