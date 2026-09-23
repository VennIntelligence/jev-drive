"""Markdown tables for the lateral v2 report from analysis summary.json (plain Python).

    python3 report_tables.py results/analysis-v1/summary.json > results/analysis-v1/tables.md
"""
import json
import sys

ARMS = ['prod-k0', 'prod-kfix', 'slipack-k0', 'slipack-kfix', 'prod-truth', 'slipack-truth']


def f(x, nd=3):
    return '—' if x is None or x != x else ('%.*f' % (nd, x))


def ci(c, nd=3):
    return '[%s, %s]' % (f(c[0], nd), f(c[1], nd))


def main():
    s = json.load(open(sys.argv[1]))
    names = [w['name'] for w in s['windows']]
    print('### Window CTE RMS, 10-seed median [95% CI of median] (m)\n')
    print('| window | ' + ' | '.join(ARMS) + ' |')
    print('|---|' + '---:|' * len(ARMS))
    for n in names:
        cells = []
        for a in ARMS:
            m = s['per_arm'][n + '|' + a]['cte_rms']
            cells.append('%s %s' % (f(m['median']), ci(m['ci'])))
        print('| %s | %s |' % (n, ' | '.join(cells)))
    for cand, base in [('slipack-kfix', 'prod-kfix'), ('slipack-k0', 'prod-k0'), ('slipack-truth', 'prod-truth')]:
        print('\n### Paired %s − %s: mean [95%% CI], n with increase / 10\n' % (cand, base))
        print('| window | CTE RMS m | CTE P95 m | CTE max m | course P95 ° | body P95 ° | speed ref RMS m/s | '
              'speed drop m/s | lat acc P95 rel | steer rate P95 rel |')
        print('|---|' + '---:|' * 9)
        for n in names:
            e = s['paired']['%s|%s-vs-%s' % (n, cand, base)]
            row = [e['cte_rms'], e['cte_p95'], e['cte_max'], e['course_p95_deg'], e['body_p95_deg'],
                   e['speed_ref_err_rms'], e['mean_speed_drop'], e['lat_acc_p95_ratio'], e['steer_rate_p95_ratio']]
            print('| %s | %s |' % (n, ' | '.join('%s %s %d' % (f(r['mean']), ci(r['ci']), r['n_positive']) for r in row)))
    print('\n### Heading on each window, candidate (slipack-kfix) vs production (prod-kfix), medians (°)\n')
    print('| window | body P95 prod / cand | course P95 prod / cand | β P95 prod / cand | β̂ P95 cand | body−β̂ P95 prod / cand |')
    print('|---|---:|---:|---:|---:|---:|')
    for n in names:
        p, c = s['per_arm'][n + '|prod-kfix'], s['per_arm'][n + '|slipack-kfix']
        g = lambda d, k: f(d[k]['median'], 2)
        print('| %s | %s / %s | %s / %s | %s / %s | %s | %s / %s |' % (
            n, g(p, 'body_p95_deg'), g(c, 'body_p95_deg'), g(p, 'course_p95_deg'), g(c, 'course_p95_deg'),
            g(p, 'beta_p95_deg'), g(c, 'beta_p95_deg'), g(c, 'beta_hat_p95_deg'),
            g(p, 'body_minus_beta_hat_p95_deg'), g(c, 'body_minus_beta_hat_p95_deg')))
    print('\n### Gates (slipack-kfix vs prod-kfix)\n')
    print('| window | check | value | limit | 95% CI | pass | note |')
    print('|---|---|---:|---:|---|---|---|')
    for c in s['gates']['checks']:
        value = c['value'] if isinstance(c['value'], (int, float)) or c['value'] is None else None
        print('| %s | %s | %s | %s | %s | %s | %s |' % (c['window'], c['check'], f(value, 4) if value is not None else c['value'],
                                                    c['limit'], ci(c['ci'], 4) if c.get('ci') else '', 'yes' if c['passed'] else '**no**',
                                                    c.get('note') or ''))
    print('\n### Frozen body-heading guard, informational only\n')
    print('| window | body P95 increment ° | 95% CI | would pass (≤ 1°) |')
    print('|---|---:|---|---|')
    for r in s['gates']['info']:
        print('| %s | %s | %s | %s |' % (r['window'], f(r['value'], 2), ci(r['ci'], 2), 'yes' if r['would_pass'] else 'no'))
    print('\nVerdict: ' + json.dumps(s['gates']['verdict']))
    fails = [g for g in s['g2'] if not g['gate_pass']]
    print('\nG2 failures (all arms, all perturbations): %d of %d' % (len(fails), len(s['g2'])))
    for g in fails:
        print('- %s %s %s: %s %s' % (g['perturbation'], g['route'], g['arm'], g['status'], g['failed']))
    print('\nMissing cases: %d' % len(s['missing_cases']))
    anchor = s['anchor_cte_rms']
    print('\nAnchor p00 window CTE RMS: ' + ', '.join('%s %s' % (k, f(v, 6)) for k, v in sorted(anchor.items())
                                                       if k.split('|')[1] in ('prod-k0', 'prod-kfix') and k.split('|')[0] in
                                                       ('24240', '26966', '17563-S1', '17563-S2')))


if __name__ == '__main__':
    main()
