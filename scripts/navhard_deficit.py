"""Navhard two-stage EPDMS deficit breakdown for openpilot Cinque (CPU only, reads existing per-token score CSVs).

Inputs (copied from the GPU box eval dirs into --data, names fixed):
  nh_cin.csv / nh_lc.csv   v2_navhard_two_stage_opi_lb_navhard_gimm-cinque{,.lc_m1.0}__base/*/*.csv
  nh_ztrs.csv nh_cv.csv nh_hydra.csv nh_cls.csv nh_warp.csv   other models' navhard per-token CSVs (same devkit)
  nt_cin.csv               v1_navtest_opi_lb_navtest_gimm-cinque__base/*/*.csv
  meta.json                runs/op_lb/lb_navhard/meta.json (cmd, speed per token)
  navhard_two_stage.yaml   navsim config train_test_split/navhard_two_stage.yaml (reactive_all_mapping)
  gimm-cinque__base.npz    runs/op_lb/lb_navhard/preds (8 x 3 poses per token)
The HF leaderboard sub-scores come from todos/2026-09-26-top10-intersection/raw_snapshots (in this repo).

Combined EPDMS is reproduced with uniform stage-two weights: the devkit weights stage-two scenes by a Gaussian kernel
(sigma^2 = 0.1) around the stage-one endpoint, which we cannot recompute from the CSVs. Uniform weights reproduce the
official number within 0.1 for Cinque (33.43 vs 33.33) but under-count planners with accurate endpoints (ZTRS 44.89 vs 48.15).
"""
import argparse, io, json, re
from pathlib import Path
import numpy as np, pandas as pd, yaml

SUBS = ['no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance', 'traffic_light_compliance',
        'ego_progress', 'time_to_collision_within_bound', 'lane_keeping', 'history_comfort', 'two_frame_extended_comfort']
MULT, WT = SUBS[:4], dict(ego_progress=5, time_to_collision_within_bound=5, lane_keeping=2, history_comfort=2, two_frame_extended_comfort=2)
SH = dict(zip(SUBS, ['NC', 'DAC', 'DDC', 'TLC', 'EP', 'TTC', 'LK', 'HC', 'EC']))
HF = Path(__file__).resolve().parents[1] / 'todos/2026-09-26-top10-intersection/raw_snapshots/agc2025-e2e-driving-navhard.public.md'


def load(f):
    d = pd.read_csv(f)
    d = d[~d.token.astype(str).str.startswith('extended')]
    st = np.where(d.token.str.len() == 16, 1, 2)
    o = pd.DataFrame({'token': d.token, 'stage': st, 'score': d.score})
    for s in SUBS:
        o[s] = d[s + '_stage_one'].where(st == 1, d[s + '_stage_two'])
    return o.set_index('token')


def rescore(df, fix=()):
    x = df.copy()
    for f in fix:
        x[f] = 1.0
    return np.prod([x[c] for c in MULT], axis=0) * sum(WT[k] * x[k] for k in WT) / 16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); D, O = Path(a.data), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    M = yaml.safe_load(open(D / 'navhard_two_stage.yaml'))['reactive_all_mapping']
    G = [(o, [p[0] for p in pr]) for o, _, pr in M] + [(p_, [p[1] for p in pr]) for _, p_, pr in M]

    def comb(df, score=None):
        d = df if score is None else df.assign(score=score)
        s1 = np.array([d.loc[x, 'score'] for x, _ in G]); s2 = np.array([d.loc[y, 'score'].mean() for _, y in G])
        return 100 * (s1 * s2).mean(), 100 * s1.mean(), 100 * s2.mean(), s1 * s2

    names = dict(cin='nh_cin.csv', lc='nh_lc.csv', ztrs='nh_ztrs.csv', cv='nh_cv.csv', hydra='nh_hydra.csv', cls='nh_cls.csv', warp='nh_warp.csv')
    Dm = {k: load(D / v) for k, v in names.items()}
    c = Dm['cin']; idx = c.index
    Dm = {k: v.reindex(idx) for k, v in Dm.items()}
    base = comb(c)[0]

    # 1. sub-score table (ours from the devkit summary rows, references from the HF board snapshot)
    def ours(f):
        r = pd.read_csv(D / f).query("token == 'extended_pdm_score_combined'").iloc[0]
        return pd.Series({'EPDMS': 100 * r.score, **{s + '_s1': 100 * r[s + '_stage_one'] for s in SUBS}, **{s + '_s2': 100 * r[s + '_stage_two'] for s in SUBS}})
    rows = {'Cinque none': ours(names['cin']), 'Cinque lc@-1.0': ours(names['lc']), 'CV (ours)': ours(names['cv']), 'ZTRS (ours, repro)': ours(names['ztrs'])}
    L = open(HF).read().split('\n'); hdr = [x.strip() for x in L[0].strip('|').split('|')]
    for l in L[2:]:
        cs = [x.strip() for x in l.strip('|').split('|')]
        if len(cs) != len(hdr): continue
        nm = re.sub(r'\*|\[|\]|\(.*?\)|<[^>]+>', '', cs[1]).replace('Baseline: ', '').strip()
        v = [float(x) for x in cs[2:-1]]
        r = dict(zip(hdr[2:-1], v))
        rows['HF ' + nm] = pd.Series({'EPDMS': r['extended_pdm_score_combined'], **{s + '_s1': r[s + '_stage_one'] for s in SUBS}, **{s + '_s2': r[s + '_stage_two'] for s in SUBS}})
    pd.DataFrame(rows).T.round(2).to_csv(O / 'sub_scores.csv')

    # 2. exact per-token substitution of ZTRS terms into Cinque (uniform stage-two weights)
    z = Dm['ztrs']; tgt = comb(z)[0]; res = []
    def subst(cols, stage=None):
        x = c.copy(); m = np.ones(len(x), bool) if stage is None else (x.stage == stage).values
        for t in cols: x.loc[m, t] = z.loc[m, t]
        return comb(x, rescore(x))[0] - base
    for t in SUBS: res.append(dict(term=SH[t], all_stages=subst([t]), stage1=subst([t], 1), stage2=subst([t], 2)))
    res.append(dict(term='ALL multiplicative', all_stages=subst(MULT), stage1=subst(MULT, 1), stage2=subst(MULT, 2)))
    res.append(dict(term='ALL weighted', all_stages=subst(list(WT)), stage1=subst(list(WT), 1), stage2=subst(list(WT), 2)))
    res.append(dict(term='ALL terms', all_stages=subst(SUBS), stage1=subst(SUBS, 1), stage2=subst(SUBS, 2)))
    pd.DataFrame(res).round(2).to_csv(O / 'substitution_vs_ztrs.csv', index=False)

    # 2b. marginal points if a term were perfect (fix to 1), Cinque / ZTRS / CV
    mg = {}
    for k in ['cin', 'ztrs', 'cv']:
        d = Dm[k]; b = comb(d)[0]
        mg[k] = {SH[t]: comb(d, rescore(d, (t,)))[0] - b for t in SUBS} | {'ALL multiplicative': comb(d, rescore(d, MULT))[0] - b, 'ALL weighted': comb(d, rescore(d, list(WT)))[0] - b, 'base (uniform w)': b}
    pd.DataFrame(mg).round(2).to_csv(O / 'marginal_if_perfect.csv')

    # 2c. aggregate log-share attribution of the official gap for the HF references
    sub = pd.read_csv(O / 'sub_scores.csv', index_col=0)
    def logshare(ref):
        r, cn = sub.loc[ref], sub.loc['Cinque none']; d = {}
        for st in ('s1', 's2'):
            for s in MULT: d[(SH[s], st)] = np.log(r[f'{s}_{st}'] / cn[f'{s}_{st}'])
            wc = sum(WT[k] * cn[f'{k}_{st}'] for k in WT) / 1600; wr = sum(WT[k] * r[f'{k}_{st}'] for k in WT) / 1600
            lin = {k: WT[k] * (r[f'{k}_{st}'] - cn[f'{k}_{st}']) for k in WT}; sl = sum(lin.values())
            for k in WT: d[(SH[k], st)] = np.log(wr / wc) * lin[k] / sl
        d = pd.Series(d); pts = d / d.sum() * (r.EPDMS - cn.EPDMS)
        return pts.unstack()
    out = []
    for ref in ['HF ZTRS', 'HF DrivoR', 'HF DriveZero', 'HF EABOT.AI&NJU', 'HF LEAD-LTFv6', 'HF LTF']:
        p = logshare(ref); p['total'] = p.sum(axis=1); p['ref'] = ref; p['gap'] = sub.loc[ref, 'EPDMS'] - sub.loc['Cinque none', 'EPDMS']; out.append(p.reset_index().rename(columns={'index': 'term'}))
    pd.concat(out).round(2).to_csv(O / 'logshare_vs_references.csv', index=False)

    # 3. slices
    meta = json.load(open(D / 'meta.json')); mt = pd.DataFrame({'token': meta['names'], 'cmd': meta['cmd'], 'speed': meta['speed']}).set_index('token').reindex(idx)
    mt['cmd'] = np.array(['left', 'straight', 'right', 'unk'])[mt.cmd.astype(int)]; mt['speed_bin'] = pd.cut(mt.speed, [-1, 1, 3, 6, 9, 12, 40]).astype(str)
    x = c.join(mt); x['zero'] = x.score == 0; x['DAC_fail'] = x.drivable_area_compliance == 0; x['NC_lt1'] = x.no_at_fault_collisions < 1; x['DDC_lt1'] = x.driving_direction_compliance < 1
    zs = Dm['ztrs']; x['ztrs_zero'] = zs.score == 0
    sl = []
    for key in ['cmd', 'speed_bin']:
        g = x.groupby(['stage', key]).agg(n=('score', 'size'), score=('score', 'mean'), zero=('zero', 'mean'), ztrs_zero=('ztrs_zero', 'mean'), DAC_fail=('DAC_fail', 'mean'), NC_lt1=('NC_lt1', 'mean'),
                                         DDC_lt1=('DDC_lt1', 'mean'), EP=('ego_progress', 'mean'), LK=('lane_keeping', 'mean'), EC=('two_frame_extended_comfort', 'mean')).reset_index().rename(columns={key: 'value'}); g.insert(1, 'slice', key); sl.append(g)
    pd.concat(sl).round(3).to_csv(O / 'slices.csv', index=False)

    # plan geometry vs DAC failure (from the plan file) and stage-one shortfall
    z_ = np.load(D / 'gimm-cinque__base.npz'); P = dict(zip(z_['tokens'].tolist(), z_['poses'])); pp = np.stack([P[t] for t in idx])
    x['x4'], x['y4'], x['h4'] = pp[:, -1, 0], pp[:, -1, 1], pp[:, -1, 2]
    x['ay_bin'] = pd.cut(x.y4.abs(), [-.1, .5, 1, 2, 3, 5, 50]).astype(str)
    x['ah_bin'] = pd.cut(x.h4.abs(), [-.1, .05, .15, .3, .6, 4]).astype(str)
    gm = []
    for key in ['ay_bin', 'ah_bin']:
        g = x[x.speed > 1].groupby(['stage', key]).agg(n=('score', 'size'), DAC_fail=('DAC_fail', 'mean'), zero=('zero', 'mean'), score=('score', 'mean')).reset_index().rename(columns={key: 'value'}); g.insert(1, 'slice', key); gm.append(g)
    pd.concat(gm).round(3).to_csv(O / 'plan_geometry_vs_dac.csv', index=False)
    f = np.load(D / 'navhard_two_stage_future.npz'); Gt = dict(zip(f['tokens'].tolist(), f['poses'])); t1 = [t for t in idx if t in Gt]
    g = np.stack([Gt[t] for t in t1]); p = np.stack([P[t] for t in t1]); dd = np.hypot(*(p[:, -1, :2] - g[:, -1, :2]).T)
    sh = pd.DataFrame({'lon4': p[:, -1, 0] - g[:, -1, 0], 'speed': mt.loc[t1, 'speed']})
    sh['bin'] = pd.cut(sh.speed, [-1, 1, 3, 6, 9, 40]).astype(str)
    sh.groupby('bin').lon4.agg(['size', 'mean']).round(2).to_csv(O / 'stage1_lon_shortfall.csv')
    pd.DataFrame([dict(n=len(t1), fde_median=np.median(dd), share_within_1p92m=(dd < 1.92).mean(), share_within_1m=(dd < 1).mean(), lon4_mean=sh.lon4.mean())]).round(3).to_csv(O / 'stage1_endpoint_vs_human.csv', index=False)

    # stage-one failure kills the whole group in the combined score
    s1 = np.array([c.loc[a_, 'score'] for a_, _ in G]); s2 = np.array([c.loc[b_, 'score'].mean() for _, b_ in G])
    rng = np.random.default_rng(0); gp = lambda d: comb(d)[3].reshape(-1, 2).mean(1)
    A_, B_, Z_ = gp(c), gp(Dm['lc']), gp(z); n = len(A_)
    ci = lambda v: (100 * v.mean(), *[100 * np.percentile([v[rng.integers(0, n, n)].mean() for _ in range(5000)], q) for q in (2.5, 97.5)])
    cirows = [dict(name=k, mean=m, lo=lo, hi=hi) for k, (m, lo, hi) in {'cin (uniform w)': ci(A_), 'lc - cin': ci(B_ - A_), 'ztrs - cin': ci(Z_ - A_)}.items()]
    pd.DataFrame(cirows).round(2).to_csv(O / 'group_bootstrap_ci.csv', index=False)
    fx = [dict(item='groups with stage-one score 0', value=(s1 == 0).mean()), dict(item='combined points lost to stage-one-zero groups (their stage-two mean x share)', value=100 * ((s1 == 0) * s2).mean()),
          dict(item='combined if stage one -> 1 (mean stage two)', value=100 * s2.mean()), dict(item='combined if stage two -> 1 (mean stage one)', value=100 * s1.mean())]

    # 4. fix estimates (in-sample, existing per-token data; none is a measured result)
    ft = []
    yy, hh = x.y4.abs().values, x.h4.abs().values
    for fb in ['cv', 'hydra', 'cls']:
        for Y, H in [(5, 9), (3, 9), (9, .6), (5, .6), (2, .4)]:
            m = (yy > Y) | (hh > H); s = np.where(m, Dm[fb].score.values, c.score.values)
            ft.append(dict(fix=f'geometry rule: |y4|>{Y} m or |yaw4|>{H} rad (9 = off) -> {fb}', share_swapped=m.mean(), gain=comb(c, s)[0] - base))
    def gain(**kw):
        y = c.copy()
        for k, v in kw.items(): y[k] = v
        return comb(y, rescore(y))[0] - base
    for k in ['warp', 'ztrs', 'hydra']: ft.append(dict(fix=f'EC substituted from {k} (per token)', gain=gain(two_frame_extended_comfort=Dm[k].two_frame_extended_comfort)))
    ft.append(dict(fix='EC set to 0.75 everywhere (assumed smoothing outcome)', gain=gain(two_frame_extended_comfort=0.75)))
    lo = (mt.speed < 1).values; y = c.copy(); y.loc[lo, 'ego_progress'] = np.maximum(c.ego_progress[lo], .8)
    ft.append(dict(fix='EP >= 0.8 on the v0 < 1 m/s tokens (standstill start)', share_swapped=lo.mean(), gain=comb(y, rescore(y))[0] - base))
    for f_ in (1.1, 1.2): y = c.copy(); y['ego_progress'] = np.minimum(1, c.ego_progress * f_); ft.append(dict(fix=f'EP x {f_} all tokens (speed stretch, no safety change assumed)', gain=comb(y, rescore(y))[0] - base))
    for keys in [['cin', 'lc'], ['cin', 'hydra'], ['cin', 'cls'], ['cin', 'lc', 'hydra', 'cls'], ['cin', 'lc', 'hydra', 'cls', 'cv']]:
        S = np.max([Dm[k].score.values for k in keys], axis=0); ft.append(dict(fix='per-token ORACLE over ' + '+'.join(keys), gain=comb(c, S)[0] - base))
    pd.DataFrame(ft).round(2).to_csv(O / 'fix_estimates.csv', index=False); pd.DataFrame(fx).round(3).to_csv(O / 'stage_coupling.csv', index=False)

    # navtest v1 decomposition
    t = open(D / 'nt_cin.csv').read(); nt = pd.read_csv(io.StringIO(t[t.rindex(',token,valid'):])); nt = nt[nt.token.astype(str).str.len() == 16]
    f5 = lambda q: q.no_at_fault_collisions * q.drivable_area_compliance * (5 * q.ego_progress + 5 * q.time_to_collision_within_bound + 2 * q.comfort) / 12
    nb = 100 * nt.score.mean(); r = dict(PDMS=nb, **{k: 100 * nt[k].mean() for k in ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance']})
    rows = []
    for k in ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']:
        y = nt.copy(); y[k] = 1.0; rows.append(dict(term=k, mean=r[k], marginal_if_perfect=100 * f5(y).mean() - nb))
    pd.DataFrame(rows).round(2).to_csv(O / 'navtest_v1_terms.csv', index=False)
    print(pd.read_csv(O / 'fix_estimates.csv').to_string())


if __name__ == '__main__':
    main()
