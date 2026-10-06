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
    return df


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
        for c in ("F", "Fplan", "Fcore", "R", "FF", "PP"):
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


def t_decode(small, sf):
    f = P.ROOT / "score" / ("decode_small.csv" if small else "decode.csv")
    if not f.exists():
        return
    df = pd.read_csv(f)
    rows = []
    for k, g in df.groupby("key", sort=False):
        g = g.set_index("token")
        r = {"decoder": k}
        for c in ("F", "Fplan", "PP"):
            m = sf.loc[g.index, c].values.astype(bool)
            r[f"DAC {c} %"] = rate(g.drivable_area_compliance.values[m], sf.loc[g.index[m], "log"].values)
        m = sf.loc[g.index, "PP"].values.astype(bool)
        r["PP score x100"] = 100 * g.score.values[m].mean()
        r["F score x100"] = 100 * g.score.values[sf.loc[g.index, "F"].values.astype(bool)].mean()
        rows.append(r)
    stats.write_table(rows, OUT / ("decode_small" if small else "decode"), note="MLP decoders on [stage, E] trained on navtrain (imit: log poses; "
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
