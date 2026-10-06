"""op_probe tables for results/dac-localize.md (no model runs; op-train or jevdrive env, CPU):

  python experiments/op_probe/scripts/opb_report.py [--small]

Reads runs/op_probe/{probe,decode}[-small]/, score/*.csv, sets/ and op_parity's gap tables; writes results/<stem>.md / .csv:
  probe[_small]       M1 / M2 per stage (corridor MAE on F / PP / F-plan, skill vs E, F - PP excess, AUC of the footprint margin)
  decode[_small]      decoder DAC pass rate on F / F-plan, DAC and per-token score on PP-1500 (opb_score.py)
  ablate              P2 input ablations: DAC pass on F / F-plan / R / FF, PP DAC and score, with P2-F-s1 (seed control) from the devkit
  fsplit              the F failure kinds (raw plan footprint out vs LQR-only, depth)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_pl.Path(__file__).parent)]
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive import stats  # noqa: E402
import opb_probe as P  # noqa: E402

OUT = _R / "experiments" / "op_probe" / "results"


def sets_frame():
    S = np.load(P.SETS / "navtest_sets.npz")
    df = pd.DataFrame({k: S[k] for k in ("tokens", "log", "F", "PP", "R", "FF", "F_both_seeds")}).set_index("tokens")
    full = pd.read_csv(P.ROOT / "score" / "ablate_P2.csv")
    full = full[full.key == "full"].set_index("token")
    df["raw_out"] = full.raw_out.reindex(df.index)
    df["out_depth"] = full.out_depth.reindex(df.index)
    df["Fplan"] = df.F & (df.raw_out == True)  # noqa: E712
    df["Fcore"] = df.Fplan & (df.out_depth >= 0.3)
    tb = np.load(P.CACHE / "lb_navtest" / "tab.npz")
    yaw4 = pd.Series(np.abs(np.degrees(tb["fut"][:, -1, 2])), index=tb["names"])
    df["PPturn"] = df.PP & (yaw4.reindex(df.index).values > 20)
    return df


def strat(v, sets, groups, B=2000, seed=0):
    """Navtest-wide mean of v from the scored strata: F / R / FF tokens weight 1, PP tokens weight n_PP / n_PP scored; log-cluster bootstrap CI."""
    S = np.load(P.SETS / "navtest_sets.npz")
    w = np.where(sets["PP"].values, S["PP"].sum() / max(sets["PP"].sum(), 1), 1.0)
    v = np.asarray(v, float)
    ug, inv = np.unique(groups, return_inverse=True)
    sw, sv = np.bincount(inv, w, len(ug)), np.bincount(inv, w * v, len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), (B, len(ug)))
    b = sv[idx].sum(1) / sw[idx].sum(1)
    return f"{100 * sv.sum() / sw.sum():.2f} [{100 * np.percentile(b, 2.5):.2f}, {100 * np.percentile(b, 97.5):.2f}]"


def strat_paired(a, b, sets, groups, B=2000, seed=0):
    """Navtest-wide (stratified as `strat`) mean of a - b over the same tokens, log-cluster bootstrap CI (x100)."""
    S = np.load(P.SETS / "navtest_sets.npz")
    w = np.where(sets["PP"].values, S["PP"].sum() / max(sets["PP"].sum(), 1), 1.0)
    v = np.asarray(a, float) - np.asarray(b, float)
    ug, inv = np.unique(groups, return_inverse=True)
    sw, sv = np.bincount(inv, w, len(ug)), np.bincount(inv, w * v, len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), (B, len(ug)))
    bs = sv[idx].sum(1) / sw[idx].sum(1)
    return f"{100 * sv.sum() / sw.sum():+.2f} [{100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}]"


def t_decode_paired(sf, name="decode", extra=()):
    f = P.ROOT / "score" / f"{name}.csv"
    if not f.exists():
        return
    df = pd.concat([pd.read_csv(f)] + [pd.read_csv(P.ROOT / "score" / f"{e}.csv").assign(key=lambda d, e=e: d.key + f"@{e}") for e in extra
                                       if (P.ROOT / "score" / f"{e}.csv").exists()])
    ab = pd.read_csv(P.ROOT / "score" / "ablate_P2.csv")
    df = pd.concat([df, ab[ab.key == "full"].assign(key="P2 (itself)")])
    fail = df.assign(fail=1 - df.drivable_area_compliance).pivot_table(index="token", columns="key", values="fail")
    score = df.pivot_table(index="token", columns="key", values="score")
    ss = sf.loc[fail.index]
    pairs = [("P2-V|imit", "WA-Cf|imit"), ("P2-V|hinge", "WA-Cf|hinge"), ("P2-V|imit", "WA-Ca|imit"), ("P2-V|hinge", "WA-Ca|hinge"),
             ("P2-V|imit", "E|imit"), ("WA-Cf|imit", "E|imit"), ("P2-H|imit", "P2 (itself)"), ("P2-H|hinge", "P2 (itself)"),
             ("P2-T|hinge", "P2 (itself)"), ("WA-Cf|hinge", "P2 (itself)"), ("WA-Ca|hinge", "P2 (itself)"), ("P2-V|hinge", "P2-H|hinge"),
             ("WA-Cf|hinge", "WA-H|hinge")]
    pairs += [(k, k.split("@")[0]) for k in fail.columns if "@" in k]
    pairs += [("P2-H|hinge@decode_h10", "P2 (itself)"), ("P2-T|hinge@decode_h10", "P2 (itself)"), ("P2-V|hinge@decode_h10", "WA-Cf|hinge@decode_h10"),
              ("P2-V|hinge@decode_h10", "WA-Ca|hinge@decode_h10"), ("P2-H|hinge@decode_h10", "WA-H|hinge@decode_h10"),
              ("WA-Cf|hinge@decode_h10", "P2 (itself)"), ("WA-H|hinge@decode_h10", "P2 (itself)")]
    rows = []
    for x, y in pairs:
        if x in fail and y in fail:
            rows.append({"a - b": f"{x} - {y}", "DAC fail pp (stratified navtest)": strat_paired(fail[x], fail[y], ss, ss.log.values),
                         "score x100 (stratified)": strat_paired(score[x], score[y], ss, ss.log.values)})
    stats.write_table(rows, OUT / f"{name}_paired", note="paired over the same tokens (F, R, FF all, PP 1 500 reweighted to 11 494), log-cluster bootstrap")


def rate(v, groups):
    r = stats.bootstrap(v, groups=groups)
    return f"{100 * r['mean']:.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def t_fsplit(sf):
    F = sf[sf.F]
    rows = [dict(set="F (P2-s0 DAC fail, WA pass)", n=len(F), raw_plan_out=F.raw_out.mean(), lqr_only=(~F.raw_out.astype(bool)).mean(),
                 depth_lt_0_3=(F.out_depth < 0.3).mean(), depth_median_m=F.out_depth.median(), depth_p90_m=F.out_depth.quantile(0.9)),
            dict(set="F-plan (raw plan footprint out)", n=int(sf.Fplan.sum()), raw_plan_out=1.0, lqr_only=0.0,
                 depth_lt_0_3=(sf[sf.Fplan].out_depth < 0.3).mean(), depth_median_m=sf[sf.Fplan].out_depth.median(),
                 depth_p90_m=sf[sf.Fplan].out_depth.quantile(0.9)),
            dict(set="F-core (F-plan, depth >= 0.3 m)", n=int(sf.Fcore.sum()))]
    stats.write_table(rows, OUT / "fsplit", floatfmt=".3f", note="opb_score.py on P2-F-s0's navtest plans (scorer reproduces the devkit DAC on all checked tokens)")


def t_ablate(sf):
    df = pd.read_csv(P.ROOT / "score" / "ablate_P2.csv")
    z = np.load(P.GAP, allow_pickle=True)
    pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
    s1 = pd.DataFrame({"key": "seed1 (P2-F-s1, devkit)", "token": list(pos), "drivable_area_compliance": z["X2"][1][:, P.J_DAC]})
    df = pd.concat([df, s1[s1.token.isin(df.token.unique())]])
    rows = []
    fs = df[df.key == "full"].set_index("token")
    for k, g in df.groupby("key", sort=False):
        g = g.set_index("token")
        r = {"variant": k}
        ss = sf.loc[g.index]
        if ss.PP.any() and ss.R.any():
            r["DAC fail % navtest (stratified)"] = strat(1 - g.drivable_area_compliance.values, ss, ss.log.values)
        for c in ("F", "Fplan", "Fcore", "R", "FF", "PP", "PPturn"):
            m = sf.loc[g.index, c].values.astype(bool)
            r[f"DAC {c} %"] = rate(g.drivable_area_compliance.values[m], sf.loc[g.index[m], "log"].values)
        m = sf.loc[g.index, "PP"].values.astype(bool)
        if "score" in g:
            d = g.score.values[m] - fs.loc[g.index[m], "score"].values
            rr = stats.bootstrap(d, groups=sf.loc[g.index[m], "log"].values)
            r["PP score delta x100"] = f"{100 * rr['mean']:+.2f} [{100 * rr['lo']:+.2f}, {100 * rr['hi']:+.2f}]"
        rows.append(r)
    stats.write_table(rows, OUT / "ablate", note="P2-F-s0 on navtest W frames; variants of its inputs (opb_feats.py ablate), scored with opb_score.py; "
                      "DAC pass rate % with log-cluster CI; PP = 1 500 random both-pass tokens; score = per-token EPDMS without EC")


def t_probe(small):
    d = P.ROOT / ("probe-small" if small else "probe")
    m = pd.read_csv(d / "metrics.csv")
    stats.write_table(m, OUT / ("probe_small" if small else "probe"), floatfmt=".3f",
                      note="ridge probes on [stage, E]; corridor MAE in m (6 free distances, capped 15 m); skill = 1 - MAE / MAE(E); "
                           "AUC of -margin (raster probe sampled on P2's own plan footprint) for F vs PP; CIs log-clustered")


def t_decode(small, sf, name=None):
    f = P.ROOT / "score" / (name or ("decode_small.csv" if small else "decode.csv"))
    if not f.exists():
        return
    df = pd.read_csv(f)
    rows = []
    for k, g in df.groupby("key", sort=False):
        g = g.set_index("token")
        r = {"decoder": k}
        ss = sf.loc[g.index]
        if ss.PP.any() and ss.R.any():
            r["DAC fail % navtest (stratified)"] = strat(1 - g.drivable_area_compliance.values, ss, ss.log.values)
            r["score x100 navtest (stratified)"] = strat(g.score.values, ss, ss.log.values)
        for c in ("F", "Fplan", "R", "FF", "PP", "PPturn"):
            m = sf.loc[g.index, c].values.astype(bool)
            r[f"DAC {c} %"] = rate(g.drivable_area_compliance.values[m], sf.loc[g.index[m], "log"].values)
        m = sf.loc[g.index, "PP"].values.astype(bool)
        r["PP score x100"] = 100 * g.score.values[m].mean()
        r["F score x100"] = 100 * g.score.values[sf.loc[g.index, "F"].values.astype(bool)].mean()
        rows.append(r)
    stats.write_table(rows, OUT / (f.stem if name else ("decode_small" if small else "decode")), note="MLP decoders on [stage, E] trained on navtrain (imit: log poses; "
                      "hinge: + drivable hinge on the footprint, true SDF); navtest scored with opb_score.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    sf = sets_frame()
    t_fsplit(sf)
    t_ablate(sf)
    t_probe(a.small)
    t_decode(a.small, sf)
    if not a.small:
        t_decode(False, sf, "decode_h10.csv")
        t_decode_paired(sf, extra=("decode_h10",))
