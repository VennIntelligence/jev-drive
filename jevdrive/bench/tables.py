"""Standard readout tables: per-arm metrics, paired contrasts with cluster-bootstrap CIs (jevdrive.stats), Shapley attribution of
the navtest EPDMS gap to its sub-scores, and the standard strata.

Arms: `label=spec+spec+...` (seeds; every unit takes the mean over the seeds, then the arm mean is taken over units) or one model
spec. Units and clusters: navtest token / log; navhard scene-mapping group / log of its stage-1 token (the mean over groups is
the official number); HUGSIM scenario / scenario.

Sources, in order: the bench run dir ($DATA_DIR/runs/bench/<bench>/<key>/units.csv), then the lanes' stored results (op_parity
navtest CSVs, navhard harness dirs, HUGSIM results.csv; WA-JEPA's stored navtest / navhard / HUGSIM runs). The table says which.

Shapley (experiments/op_parity/scripts/pp_gap_tables.py): the v2 token score is
    NC * DAC * DDC * TLC * (5 EP + 5 TTC + 2 LK + 2 HC + 2 EC) / (14 + 2 [EC present]);
for the gap ref - arm every sub-score is a player and v(S) is the score with the terms in S at the ref's values; the exact Shapley
value over the 512 coalitions splits each token's gap additively, so the per-term columns sum to the EPDMS gap.
Strata (experiments/op_probe/scripts/opj_build.py / opj_figs.py): logged 4 s heading change bins <5 / 5-20 / 20-45 / >45 deg,
manoeuvre (stationary, launch, stop, left / right turn > 20 deg, lane change, curve, straight), speed at t0 <2 / 2-5 / 5-10 / >=10
m/s, city; HUGSIM: dataset, difficulty, turning route (heading range >= 30 deg).
"""
from __future__ import annotations

import json
from math import factorial
from pathlib import Path

import numpy as np

from .models import REPO, Model, data_dir, resolve
from .navsim import SUBS, read_devkit_csv
from . import runner as R

TERMS = list(SUBS)                              # NC DAC DDC TLC EP TTC LK HC EC


# ---------------------------------------------------------------- EPDMS algebra
def score_of(X):
    """X (..., 9) sub-scores in TERMS order (EC may be NaN) -> the token's v2 EPDMS."""
    ec = X[..., 8]
    num = 5 * X[..., 4] + 5 * X[..., 5] + 2 * X[..., 6] + 2 * X[..., 7] + 2 * np.nan_to_num(ec)
    return np.prod(X[..., :4], -1) * num / (14 + 2 * np.isfinite(ec))


def shapley(A, B):
    """Exact Shapley split of score(B) - score(A) over the 9 terms; A, B (n, 9) -> (n, 9), rows sum to the gap."""
    n, k = A.shape
    sub = np.arange(1 << k)
    bits = ((sub[:, None] >> np.arange(k)) & 1).astype(bool)
    v = np.empty((n, 1 << k))
    for s in sub:
        v[:, s] = score_of(np.where(bits[s], B, A))
    pop = bits.sum(1)
    w = np.array([factorial(p) * factorial(k - p - 1) / factorial(k) if p < k else 0 for p in pop])
    phi = np.zeros((n, k))
    for i in range(k):
        s0 = sub[~bits[:, i]]
        phi[:, i] = (w[s0] * (v[:, s0 | (1 << i)] - v[:, s0])).sum(1)
    return phi


# ---------------------------------------------------------------- loading units
def _latest(pattern_dir: Path, glob: str):
    fs = sorted(pattern_dir.glob(glob)) if pattern_dir.exists() else []
    return fs[-1] if fs else None


def _navtest_legacy(m: Model):
    D = data_dir()
    if m.family == "wajepa":
        return D / m.stored["navtest"]
    if m.family == "parity":
        st = f"{m.frames}@cinque_PP{m.name}" + (f"_{m.opt}" if m.opt else "")
    else:
        st = f"{m.frames}@{m.base}" + (f"_O{m.name}" if m.onnx else "")
    return _latest(D / "runs/navsim/eval", f"v2_navtest_opi_lb_navtest_{st.replace('@', '-')}__base/*/*.csv")


def _navhard_legacy(m: Model):
    D = data_dir()
    if m.family == "wajepa":
        return D / m.stored["navhard"]
    if m.family == "parity":
        return D / ("runs/op_parity/navhard_gimm/harness" if m.frames == "gimm" else "runs/op_parity/navhard/harness") / m.name
    return D / "runs/op_guard" / ("shipped" if m.name == "cinque" and not m.onnx else m.name) / "nav/navhard"


def _hugsim_legacy_rows(m: Model, preset: str) -> list:
    import csv
    if m.family == "wajepa":
        f, tag = (REPO / m.stored["hugsim:exam"]).as_posix().split("#")
        return [r for r in csv.DictReader(open(f)) if r["tag"] == tag] if preset == "exam" else []
    if m.family == "parity":
        prefix = {"exam": "pp-", "spec": "pp-spec-", "spec_plan": "pp-specplan-",
                  "spec_plan_smooth": "pp-specplansmooth-", "spec_plan_mpc": "pp-specplanmpc-"}.get(preset)
        if prefix is None:
            return []
        tag = prefix + m.name
        f = data_dir() / "runs/op_parity/hugsim/results.csv"
        return [r for r in csv.DictReader(open(f)) if r["tag"] == tag] if f.exists() else []
    return []


def tokens_meta(bench: str):
    from .sets import navsim_tokens
    toks, logs = navsim_tokens(bench)
    return dict(zip(toks, logs))


def load(bench: str, spec: str, preset: str = "exam", behaviour: bool = True, **kw):
    """(units DataFrame indexed by unit, source string) of one model; None if nothing is stored."""
    import pandas as pd
    from . import run_dir
    explicit = spec.startswith("run:")
    if explicit:
        key = spec[4:]
        if not key or Path(key).name != key or key in (".", ".."):
            raise ValueError(f"invalid bench run reference {spec!r}")
        d = R.bench_root(bench, key)
    else:
        m = resolve(spec)
        d = run_dir(spec, bench, preset, **kw)
    if (d / "units.csv").exists():
        u = pd.read_csv(d / "units.csv")
        idx = {"navtest": "token", "navhard": "group", "hugsim": "scenario"}[bench]
        return u.set_index(idx), f"bench {d}"
    if explicit:
        return None, ""
    if bench == "navtest":
        f = _navtest_legacy(m)
        if f is None or not Path(f).exists():
            return None, ""
        t, _ = read_devkit_csv(f)
        lg = tokens_meta("navtest")
        u = pd.DataFrame({"log": [lg.get(x, "?") for x in t.index], "score": t["score"].to_numpy(float)}, index=t.index.rename("token"))
        for k, c in SUBS.items():
            u[k] = t[c].to_numpy(float)
        return u, f"stored {f}"
    if bench == "navhard":
        h = _navhard_legacy(m)
        if not (h / "harness_groups.csv").exists():
            return None, ""
        g = pd.read_csv(h / "harness_groups.csv")
        lg = tokens_meta("navhard")
        return g.assign(log=[lg.get(o, "?") for o in g.orig]).set_index("group"), f"stored {h}"
    from . import hugsim as H
    if H.run_key(m, preset, **kw) != H.run_key(m, preset):
        return None, ""                    # custom rules/repeats must never fall back to baseline rows
    preset = H.configuration(preset)["preset"]
    rows = _hugsim_legacy_rows(m, preset)
    if not rows:
        return None, ""
    if behaviour:
        try:
            rts = H.routes()
        except Exception:
            rts, behaviour = {}, False
    u = []
    for r in H.units_from_rows(rows, rts if behaviour else {}):
        u.append(r)
    return pd.DataFrame(u).set_index("scenario"), "stored runs/op_parity/hugsim/results.csv" if m.family != "wajepa" else "stored " + m.stored["hugsim:exam"]


def parse_arm(a: str) -> tuple:
    lab, _, specs = a.partition("=")
    if not specs:
        return a, [a]
    return lab, [s for s in specs.split("+") if s]


# ---------------------------------------------------------------- tables
def _arm_units(bench, specs, preset, need, **kw):
    """Per-seed unit frames of one arm, aligned on their common units."""
    got = []
    for s in specs:
        u, src = load(bench, s, preset, **kw)
        if u is None:
            raise SystemExit(f"{bench}: no result for {s} (preset {preset}); run `python -m jevdrive.bench run --model {s} --bench {bench}`")
        got.append((s, u, src))
    common = sorted(set.intersection(*[set(u.index) for _, u, _ in got]))
    return [(s, u.loc[common], src) for s, u, src in got], common


def report(bench: str, arms: list, vs: list = (), preset: str = "exam", out: str = "", strata: bool = True, units_subset=None, **kw) -> dict:
    """Write <out>/<bench>_{arms,paired[,shapley,strata]}.{md,csv}; returns the DataFrames."""
    import pandas as pd
    from .. import stats
    allarms = [parse_arm(a) for a in list(arms) + [v for v in vs if v not in arms]]
    arm_data = {}
    for lab, specs in allarms:
        arm_data[lab], _ = _arm_units(bench, specs, preset, None, **kw)
    common = sorted(set.intersection(*[set(v[0][1].index) for v in arm_data.values()]))
    if units_subset is not None:
        common = [c for c in common if c in set(units_subset)]
    outd = Path(out or (REPO / "tmp" / "bench"))
    tag = f"{bench}" + (f"_{preset}" if bench == "hugsim" else "")
    res = {}
    rows, metric = [], {"navtest": "score", "navhard": "combined", "hugsim": "hdscore"}[bench]
    scale = 100.0 if bench == "navtest" else 1.0
    seedmean = {}
    for lab, v in arm_data.items():
        U = [u.loc[common] for _, u, _ in v]
        num = [c for c in U[0].columns if pd.api.types.is_numeric_dtype(U[0][c])]
        M = sum(u[num].astype(float) for u in U) / len(U)
        seedmean[lab] = (M, U)
        r = {"arm": lab, "seeds": len(U), "n": len(common)}
        if bench == "navtest":
            r["EPDMS"] = 100 * M.score.mean()
            r["per seed"] = " / ".join(f"{100 * u.score.mean():.2f}" for u in U)
            r |= {k: 100 * np.nanmean(M[k]) for k in TERMS}
        elif bench == "navhard":
            r |= {k: M[k].mean() for k in ("combined", "stage1", "stage2")}
            r["per seed"] = " / ".join(f"{u.combined.mean():.2f}" for u in U)
        else:
            r |= {k: M[k].mean() for k in ("hdscore", "rc", "nc", "dac", "ttc", "c", "pdms")}
            r["per seed"] = " / ".join(f"{u.hdscore.mean():.3f}" for u in U)
            for c in ("complete", "fg_coll", "bg_coll", "off_route", "spin"):
                if "cls" in U[0]:
                    r[c] = np.mean([(u.cls == c).sum() for u in U])
            r["stuck (max_steps end)"] = np.mean([(u.end == "max_steps").sum() for u in U])
            if "launch_stall" in U[0]:
                r["launch stall"] = np.mean([u.launch_stall.astype(bool).sum() for u in U])
                r["spin (any end)"] = np.mean([u.spin.fillna(False).astype(bool).sum() for u in U])
        r["source"] = "; ".join(sorted({src.split(" ")[0] for _, _, src in v}))
        rows.append(r)
    note = {"navtest": "navtest, devkit navsim main @0a380a9 v2 EPDMS x 100 (token mean); seed means per token",
            "navhard": "navhard_two_stage v2 EPDMS, devkit two-stage aggregation per scene-mapping group (official = mean over groups)",
            "hugsim": f"HUGSIM, preset {preset}; HD-Score mean over scenarios; class = spin, else the runner's end; counts are seed means"}[bench]
    res["arms"] = stats.write_table(rows, outd / f"{tag}_arms", floatfmt=".3f" if bench == "hugsim" else ".2f", note=note)
    # paired contrasts
    first = next(iter(arm_data.values()))[0][1]
    groups = None if bench == "hugsim" else first.loc[common, "log"].astype(str).to_numpy()
    refs = [parse_arm(v)[0] for v in vs]
    pairs = [(a, b) for a, _ in allarms for b in refs if a != b and a not in refs] + \
            [(a, b) for i, a in enumerate(refs) for b in refs[i + 1:]]
    pr = []
    for a, b in pairs:
        A, B = seedmean[a][0], seedmean[b][0]
        r = stats.paired(scale * A[metric].to_numpy(float), scale * B[metric].to_numpy(float), groups=groups)
        row = {"pair": f"{a} - {b}", **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")}}
        if bench == "navtest":
            row |= {f"d{k}": 100 * (np.nanmean(A[k]) - np.nanmean(B[k])) for k in ("EP", "DAC", "NC", "TTC", "EC")}
        if bench == "navhard":
            for k in ("stage1", "stage2"):
                rk = stats.paired(A[k].to_numpy(float), B[k].to_numpy(float), groups=groups)
                row[f"d{k}"] = f"{rk['mean']:.2f} [{rk['lo']:.2f}, {rk['hi']:.2f}]"
        pr.append(row)
    unit = {"navtest": "per-token EPDMS x 100, cluster bootstrap over navtest logs", "navhard": "per-group EPDMS, cluster bootstrap over the "
            "stage-1 logs of the groups", "hugsim": "per-scenario HD-Score, bootstrap over scenarios"}[bench]
    if pr:
        res["paired"] = stats.write_table(pr, outd / f"{tag}_paired", floatfmt=".3f" if bench == "hugsim" else ".2f",
                                          note=f"{unit} (B 10000, seed 0); seeds averaged per unit first")
    if bench == "navtest" and refs:
        res["shapley"] = shapley_table(arm_data, refs, common, groups, outd / f"{tag}_shapley")
    if strata and refs:
        res["strata"] = strata_table(bench, seedmean, refs, common, metric, scale, outd / f"{tag}_strata")
    print(f"tables -> {outd}/{tag}_*.md")
    return res


def shapley_table(arm_data, refs, common, groups, stem):
    from .. import stats
    rows = []
    for ref in refs:
        Xr = np.mean([u.loc[common, TERMS].to_numpy(float) for _, u, _ in arm_data[ref]], 0)
        for lab, v in arm_data.items():
            if lab == ref:
                continue
            phi = np.mean([shapley(u.loc[common, TERMS].to_numpy(float), Xr) for _, u, _ in v], 0) * 100
            gap = stats.bootstrap(phi.sum(1), groups=groups)
            r = {"gap": f"{ref} - {lab}", "EPDMS gap": gap["mean"], "lo": gap["lo"], "hi": gap["hi"]}
            for j, t in enumerate(TERMS):
                b = stats.bootstrap(phi[:, j], groups=groups)
                r[t] = f"{b['mean']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]"
            rows.append(r)
    return stats.write_table(rows, stem, floatfmt=".2f", note="exact Shapley split of the per-token EPDMS gap (x 100) over the 9 sub-scores; "
                             "columns sum to the gap; cluster bootstrap over logs")


# ---------------------------------------------------------------- strata
TURN_BINS, TURN_LABELS = [0, 5, 20, 45, 400], ["<5", "5-20", "20-45", ">45"]
CITY = {"us-nv-las-vegas-strip": "Las Vegas", "us-ma-boston": "Boston", "us-pa-pittsburgh-hazelwood": "Pittsburgh", "sg-one-north": "Singapore"}


def motion(fut, v0):
    """opj_build.motion: logged-motion class of the 4 s future (t0 rear-axle frame) -> (dyaw deg, manoeuvre)."""
    if np.isnan(fut[0, 0]):
        return np.nan, "unknown"
    P = np.vstack([[0, 0, 0], fut])
    dyaw = float(np.degrees(np.unwrap(P[:, 2])[-1]))
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    v_end, L = float(seg[-1] / 0.5), float(seg.sum())
    if v0 < 0.5 and L < 1.0:
        man = "stationary"
    elif v0 < 1.5 and v_end > 2.5:
        man = "launch"
    elif v0 > 2.5 and v_end < 0.5:
        man = "stop"
    elif dyaw > 20:
        man = "left turn"
    elif dyaw < -20:
        man = "right turn"
    elif abs(dyaw) < 8 and abs(fut[-1, 1]) > 2.0 and L > 10:
        man = "lane change"
    elif abs(dyaw) >= 8:
        man = "curve"
    else:
        man = "straight"
    return dyaw, man


def navtest_strata():
    """token -> turn bin / manoeuvre / speed bin / city (cached in runs/bench/strata/navtest.csv)."""
    import pandas as pd
    f = R.bench_root("strata", "navtest.csv")
    if f.exists():
        return pd.read_csv(f).set_index("token")
    tab = np.load(data_dir() / "runs/op_parity/cache/lb_navtest/tab.npz")
    from .. import navsim_zs as Z
    mp = {e["token"]: e["map"] for e in Z.load_index("navtest", slim=True)}
    rows = []
    for t, fut, v0 in zip(tab["names"], tab["fut"], tab["speed"]):
        dyaw, man = motion(np.asarray(fut, float), float(v0))
        rows.append(dict(token=t, dyaw=dyaw, maneuver=man, v0=float(v0), city=CITY.get(mp.get(t, ""), mp.get(t, "?"))))
    d = pd.DataFrame(rows)
    d["turn"] = pd.cut(d.dyaw.abs(), TURN_BINS, right=False, labels=TURN_LABELS).astype(str)
    d["speed"] = pd.cut(d.v0, [0, 2, 5, 10, 99], right=False, labels=["<2", "2-5", "5-10", ">=10"]).astype(str)
    f.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(f, index=False)
    return d.set_index("token")


def hugsim_strata(units):
    import pandas as pd
    from .sets import route_turn_deg
    from . import hugsim as H
    try:
        rt = route_turn_deg(H.routes())
    except Exception:
        rt = {}
    d = pd.DataFrame(index=units.index)
    d["dataset"], d["difficulty"] = units.dataset, units.difficulty
    d["turning"] = ["turning route (>= 30 deg)" if rt.get(s, 0) >= 30 else "straight route" for s in units.scene]
    return d


def strata_table(bench, seedmean, refs, common, metric, scale, stem):
    import pandas as pd
    from .. import stats
    if bench == "navtest":
        S = navtest_strata().loc[common]
        dims = {"turn (logged |dyaw|, deg)": S.turn, "manoeuvre": S.maneuver, "speed at t0 (m/s)": S.speed, "city": S.city}
    elif bench == "hugsim":
        S = hugsim_strata(next(iter(seedmean.values()))[1][0].loc[common])
        dims = {"dataset": S.dataset, "difficulty": S.difficulty, "route": S.turning}
    else:
        return None
    first = next(iter(seedmean.values()))[1][0]
    rows = []
    for dim, lab in dims.items():
        for val in sorted(pd.unique(lab)):
            msk = (lab == val).to_numpy()
            g = None if bench == "hugsim" else first.loc[common, "log"].astype(str).to_numpy()[msk]
            r = {"stratum": dim, "value": val, "n": int(msk.sum())}
            for a, (M, _) in seedmean.items():
                r[a] = scale * M[metric].to_numpy(float)[msk].mean()
            for ref in refs:
                for a, (M, _) in seedmean.items():
                    if a in refs:
                        continue
                    p = stats.paired(scale * M[metric].to_numpy(float)[msk], scale * seedmean[ref][0][metric].to_numpy(float)[msk], groups=g)
                    r[f"{a} - {ref}"] = f"{p['mean']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]" if bench != "hugsim" else \
                        f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
            rows.append(r)
    return stats.write_table(rows, stem, floatfmt=".2f" if bench != "hugsim" else ".3f",
                             note="strata of the units in common; paired CIs as in the paired table, within the stratum")


def summary_json(bench, spec, preset="exam", **kw) -> dict:
    from . import run_dir
    if spec.startswith("run:") and (not spec[4:] or Path(spec[4:]).name != spec[4:] or spec[4:] in (".", "..")):
        raise ValueError(f"invalid bench run reference {spec!r}")
    d = R.bench_root(bench, spec[4:]) if spec.startswith("run:") else run_dir(spec, bench, preset, **kw)
    f = d / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}
