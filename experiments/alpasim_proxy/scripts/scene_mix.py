#!/usr/bin/env python3
"""SCENE-MIX N2: human-state statistics of 5.5 s windows sampled uniformly in time on the raw nuPlan test logs against the windows of
the 1491 public AlpaSim scenes (pre-registration: experiments/alpasim_proxy/plans/2026-10-16-scene-mix-prereg.md).

  scene_mix.py extract [--step 0.5]                                   (box, CPU; one pmap unit per log, about a minute)
  scene_mix.py report <windows.csv> --out DIR [--gs <go-straight per_scene.csv>]   (Mac)

extract writes $DATA_DIR/runs/alpasim_proxy/scene_mix/n2/<ts>/windows.csv: one row per window (kind uniform | public), ego at the
lidar_pc instants (20 Hz): v0 / v_end (|vx, vy|), path (10 Hz polyline), maxdisp (max distance from the start), dyaw (heading change),
tag_stat (nuPlan `stationary` scenario tag at the start frame). report: class shares with log-clustered CIs (N2a), the stratum
reweighting of stored public scores (N2b), and stored outcomes on the static / near-stationary public scenes (N2c).
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
RAW = DATA / "datasets/nuplan_raw/test"
CFG = DATA / "datasets/alpasim_nuplan/navtest/configs"
WIN = 5.5
PRIVATE = {"APY10m10-AB-s0": 0.640, "YR10m10-F-s0": 0.800, "LTF-sample": 0.738, "DD-sample": 0.772, "G0014": 0.848, "GS": 0.799}


def windows(job):
    """All uniform windows (start grid `step` s) and the public scenes' windows of one log."""
    db, ids, step = job
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = ("select lower(hex(l.token)), l.timestamp, e.x, e.y, e.qw, e.qz, e.qx, e.qy, e.vx, e.vy from lidar_pc l "
         "join ego_pose e on e.token = l.ego_pose_token order by l.timestamp")
    rows = c.execute(q).fetchall()
    stat = {r[0] for r in c.execute("select lower(hex(lidar_pc_token)) from scenario_tag where type = 'stationary'")}
    tok = [r[0] for r in rows]
    a = np.array([r[1:] for r in rows], float)
    t, xy = (a[:, 0] - a[0, 0]) / 1e6, a[:, 1:3]
    qw, qz, qx, qy = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    v = np.hypot(a[:, 7], a[:, 8])
    idx = {k: i for i, k in enumerate(tok)}
    log = Path(db).stem

    def one(i0, kind, scene):
        i1 = int(np.searchsorted(t, t[i0] + WIN, side="right")) - 1
        if t[i1] - t[i0] < WIN - 0.06:
            return None
        p = xy[i0:i1 + 1:2]
        dy = np.angle(np.exp(1j * (yaw[i1] - yaw[i0])))
        return dict(log=log, kind=kind, scene=scene, t0=t[i0], v0=v[i0], v_end=v[i1], path=np.linalg.norm(np.diff(p, axis=0), axis=1).sum(),
                    maxdisp=np.linalg.norm(xy[i0:i1 + 1] - xy[i0], axis=1).max(), dyaw=np.degrees(abs(dy)), tag_stat=tok[i0] in stat)

    out = [one(int(np.searchsorted(t, s)), "uniform", "") for s in np.arange(0, t[-1], step)]
    for sid in ids:
        out.append(one(idx[sid.rsplit("-", 1)[1]], "public", sid) if sid.rsplit("-", 1)[1] in idx else dict(log=log, kind="public", scene=sid))
    return [r for r in out if r]


def extract(a):
    from jevdrive import par
    from jevdrive.run import Run
    ids = sorted(p.stem for p in CFG.glob("*.yaml"))
    by = {}
    for s in ids:
        by.setdefault(s.rsplit("-", 1)[0], []).append(s)
    dbs = sorted(RAW.glob("*.db"))
    with Run("alpasim_proxy", "scene_mix/n2", config=vars(a)) as run:
        miss = sorted(set(by) - {d.stem for d in dbs})
        run.info(f"{len(dbs)} logs, {len(ids)} public ids, logs without a db: {miss}")
        res = par.pmap(windows, [(str(d), by.get(d.stem, []), a.step) for d in dbs], run=run, desc="logs")
        res.raise_if_failed()
        W = pd.DataFrame([r for v in res.values for r in v])
        W.to_csv(run.path("windows.csv"), index=False)
        run.summary.update(n_uniform=int((W.kind == "uniform").sum()), n_public=int((W.kind == "public").sum()),
                           public_unmatched=int(W[W.kind == "public"].v0.isna().sum()))


def classes(W):
    return pd.DataFrame({"S": W.maxdisp < 1, "E": W.v_end < 0.5, "N": (W.v0 < 1) & (W.path < 5),
                         "C": (W.v0 >= 5) & ((W.v_end - W.v0).abs() < 1) & (W.dyaw < 5), "T": W.tag_stat.astype(bool)}, index=W.index)


def stratum(K):
    return np.select([K.S, K.N, K.E], ["S", "N", "E"], "rest")


def report(a):
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = pd.read_csv(a.windows)
    U, P = W[W.kind == "uniform"].reset_index(drop=True), W[W.kind == "public"].dropna(subset=["v0"]).reset_index(drop=True)
    KU, KP = classes(U), classes(P)
    lines = [f"# SCENE-MIX N2\n\n{len(U)} uniform windows on {U.log.nunique()} logs (0.5 s grid, overlapping), {len(P)} public scenes "
             f"({(W.kind == 'public').sum() - len(P)} unmatched). CIs: log-clustered bootstrap.\n"]

    # alignment check against AUD1's per-scene table
    m = pd.read_csv(REPO / "experiments/alpasim/results/aud1/per_scene.csv", usecols=["scene", "v0", "APY10m10-AB-s0|gt_dist_traveled_m"])
    m = P.merge(m.rename(columns={"v0": "v0_aud1", "APY10m10-AB-s0|gt_dist_traveled_m": "log_m"}), on="scene")
    ok = ((m.path - m.log_m).abs() < 1) & ((m.v0 - m.v0_aud1).abs() < 0.5)
    n262 = int(((m.log_m < 5) & (m.v0_aud1 < 1)).sum())
    lines.append(f"## Alignment check\n\n{len(m)} public scenes matched to AUD1: |path - gt_dist_traveled| < 1 m and |v0 - v0_aud1| < 0.5 m/s on "
                 f"{ok.mean():.1%} (median |dpath| {(m.path - m.log_m).abs().median():.2f} m); class N here {int(KP.N.sum())}, d262 count from "
                 f"AUD1 {n262}. Line >= 95 %: {'pass' if ok.mean() >= 0.95 else 'FAIL'}.\n")

    rows = []
    for c, name in [("N", "near-stationary start (primary)"), ("S", "static whole window"), ("E", "ends stopped"), ("C", "cruise"),
                    ("T", "nuPlan `stationary` tag at start")]:
        ru, rp = stats.bootstrap(KU[c].astype(float), groups=U.log), stats.bootstrap(KP[c].astype(float), groups=P.log)
        rows.append(dict(cls=c, name=name, uniform=ru["mean"], u_lo=ru["lo"], u_hi=ru["hi"], public=rp["mean"], p_lo=rp["lo"], p_hi=rp["hi"],
                         n_public=int(KP[c].sum()), ratio=ru["mean"] / rp["mean"] if rp["mean"] else np.inf))
    T = pd.DataFrame(rows)
    T.to_csv(out / "n2a_shares.csv", index=False)
    r = T.set_index("cls").ratio["N"]
    verdict = "supported" if r >= 3 else "refuted" if r <= 1.5 else "undetermined"
    lines.append("## N2a: class shares\n\n| class | uniform [95 % CI] | public [95 % CI] | public n | ratio |\n|---|--:|--:|--:|--:|\n" + "\n".join(
        f"| {x.cls} {x.name} | {x.uniform:.3f} [{x.u_lo:.3f}, {x.u_hi:.3f}] | {x.public:.3f} [{x.p_lo:.3f}, {x.p_hi:.3f}] | {x.n_public} | {x.ratio:.1f} |"
        for x in T.itertuples()) + f"\n\nPrimary (N): ratio {r:.2f} against lines >= 3 supported / <= 1.5 refuted: **{verdict}**.\n")

    # N2b / N2c on stored public rollouts
    S = pd.concat([pd.read_csv(REPO / "tmp/aud1_drift/per_scene.csv")] + [pd.read_csv(REPO / f"experiments/alpasim_proxy/results/anchors/{f}.csv")
                                                                          for f in ("ltf", "dd", "g0014")]
                  + ([pd.read_csv(a.gs).assign(driver="GS")] if a.gs else []), ignore_index=True)
    S = S.drop_duplicates(["driver", "scene"])
    P["stratum"] = stratum(KP)
    U["stratum"] = stratum(KU)
    wu = U.stratum.value_counts(normalize=True)
    S = S.merge(P[["scene", "stratum"]], on="scene", how="left")
    S["stratum"] = S.stratum.fillna("rest")
    S["score"] = S.score.fillna(0)
    S["zero"] = S.score == 0
    S["corr"] = S.left_corridor_laterally.fillna(0) > 0
    S["coll"] = S.collision_at_fault.fillna(0) > 0
    S["off"] = S.offroad.fillna(0) > 0
    g = S.groupby(["driver", "stratum"]).agg(n=("score", "size"), score=("score", "mean"), zeros=("zero", "sum"), corridor=("corr", "sum"),
                                              collision=("coll", "sum"), offroad=("off", "sum"), driven_m=("dist_traveled_m", "mean")).reset_index()
    g.to_csv(out / "n2c_strata.csv", index=False)
    pub = S.groupby("driver").score.mean()
    sc = g.pivot(index="driver", columns="stratum", values="score")
    npub = P.stratum.value_counts()
    thin = [s for s in ("S", "N", "E") if npub.get(s, 0) < 5]
    pred = {d: sum(wu.get(s, 0) * (sc.loc[d, s] if s in sc.columns and s not in thin else sc.loc[d, "rest"]) for s in wu.index) for d in sc.index}
    B = pd.DataFrame({"public": pub, "pred_private": pd.Series(pred), "private": pd.Series(PRIVATE)}).dropna(subset=["public"])
    B["pred_change"], B["obs_change"] = B.pred_private - B.public, B.private - B.public
    B.to_csv(out / "n2b_reweight.csv")
    lines.append(f"## N2b: public scores reweighted to the uniform-window strata\n\nStrata (exclusive, S > N > E > rest): uniform shares "
                 + ", ".join(f"{s} {wu.get(s, 0):.3f}" for s in ("S", "N", "E", "rest")) + "; public n " + ", ".join(f"{s} {npub.get(s, 0)}" for s in ("S", "N", "E", "rest"))
                 + (f"; too thin (< 5 public scenes, scored as `rest`): {thin}" if thin else "") + ".\n\n| driver | public | predicted private | predicted change | private | observed change |\n|---|--:|--:|--:|--:|--:|\n"
                 + "\n".join(f"| {d} | {x.public:.3f} | {x.pred_private:.3f} | {x.pred_change:+.3f} | {x.private:.3f} | {x.obs_change:+.3f} |" for d, x in B.iterrows()) + "\n")
    lines.append("## N2c: stored public outcomes by stratum\n\n| driver | stratum | n | score | zeros | corridor | collision at fault | offroad | driven m |\n|---|---|--:|--:|--:|--:|--:|--:|--:|\n"
                 + "\n".join(f"| {x.driver} | {x.stratum} | {x.n} | {x.score:.3f} | {x.zeros} | {x.corridor} | {x.collision} | {x.offroad} | {x.driven_m:.1f} |"
                             for x in g[g.stratum != "rest"].itertuples()) + "\n")
    P.merge(KP, left_index=True, right_index=True).to_csv(out / "public_windows.csv", index=False)
    (out / "n2.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--step", type=float, default=0.5)
    r = sp.add_parser("report")
    r.add_argument("windows")
    r.add_argument("--out", required=True)
    r.add_argument("--gs")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else report(a)


if __name__ == "__main__":
    main()
