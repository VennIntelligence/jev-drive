#!/usr/bin/env python3
"""SCENE-MIX N2: human-state statistics of 5.5 s windows sampled uniformly in time on the raw nuPlan test logs against the windows of
the 1491 public AlpaSim scenes (pre-registration: experiments/alpasim_proxy/plans/2026-10-16-scene-mix-prereg.md).

  scene_mix.py extract [--step 0.5]                                   (box, CPU; one pmap unit per log, about a minute)
  scene_mix.py report <windows.csv> --out DIR [--gs <go-straight per_scene.csv>]   (Mac)
  scene_mix.py sample --out DIR                                                  (Mac: amendment 3 scene lists)
  scene_mix.py n1 <go-straight per_scene.csv> --out DIR                          (Mac, after report and sample)

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
OFFSET = -1.5   # a public scene starts 1.5 s before its central (navtest) token: amendment 1, fitted to AUD1 gt_dist_traveled / v0
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
        out.append(one(int(np.searchsorted(t, t[idx[sid.rsplit("-", 1)[1]]] + OFFSET)), "public", sid) if sid.rsplit("-", 1)[1] in idx else dict(log=log, kind="public", scene=sid))
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


SEED = 20261016


def sample(a):
    """Amendment 3: log-stratified random sample of the public scenes (largest-remainder allocation over AUD1's 44 logs, at least one
    per log, fixed seed) plus the 25 near-stationary scenes (d262: logged path < 5 m and v0 < 1 m/s), written as two scene lists."""
    out = Path(a.out)
    m = pd.read_csv(REPO / "experiments/alpasim/results/aud1/per_scene.csv", usecols=["scene", "log", "v0", "APY10m10-AB-s0|gt_dist_traveled_m"])
    q = m.groupby("log").size() * a.n / len(m)
    k = np.maximum(1, np.floor(q)).astype(int)
    k[(q - np.floor(q)).sort_values(ascending=False).index[:max(0, a.n - k.sum())]] += 1
    rng = np.random.default_rng(SEED)
    pick = sorted(s for lg, x in m.groupby("log") for s in rng.choice(sorted(x.scene), k[lg], replace=False))
    ns = sorted(m.scene[(m["APY10m10-AB-s0|gt_dist_traveled_m"] < 5) & (m.v0 < 1)])
    (out / "n1_sample.txt").write_text("\n".join(pick) + "\n")
    (out / "n1_nearstat.txt").write_text("\n".join(ns) + "\n")
    (out / "n1_run.txt").write_text("\n".join(sorted(set(pick) | set(ns))) + "\n")
    print(f"sample {len(pick)} over {m.log.nunique()} logs, near-stationary {len(ns)}, overlap {len(set(pick) & set(ns))}, run list {len(set(pick) | set(ns))}")


def metrics(x):
    ok, sc = x.dist_traveled_m.notna(), x.score.fillna(0)
    z = x[sc == 0]
    fold, coll = z.dist_traveled_m.isna(), z.collision_at_fault.fillna(0) > 0
    off = ~fold & ~coll & (z.offroad.fillna(0) > 0)
    cor = ~fold & ~coll & ~off & (z.left_corridor_laterally.fillna(0) > 0)
    return dict(n=len(x), score=sc.mean(), km=x.dist_traveled_m.sum() / max(x.offroad_or_collision_at_fault.sum(), 1e-9) / 1000,
                gt=x.dist_to_gt_trajectory[ok].mean(), zeros=len(z), z_fold=int(fold.sum()), z_coll=int(coll.sum()), z_off=int(off.sum()),
                z_corr=int(cor.sum()), driven_m=x.dist_traveled_m.mean())


def n1(a):
    """N1 (amendment 3): go-straight on the log-stratified sample (primary, log-clustered CI) and on the 25 near-stationary scenes; every
    stored driver on the same scenes and on all 1491 for reference. Lines: <= 0.70 supported, >= 0.78 refuted (private 0.799)."""
    from jevdrive import stats
    out = Path(a.out)
    m = pd.read_csv(REPO / "experiments/alpasim/results/aud1/per_scene.csv", usecols=["scene", "log"])
    S = pd.concat([pd.read_csv(a.gs).assign(driver="GS"), pd.read_csv(REPO / "tmp/aud1_drift/per_scene.csv")]
                  + [pd.read_csv(REPO / f"experiments/alpasim_proxy/results/anchors/{f}.csv") for f in ("ltf", "dd", "g0014")], ignore_index=True)
    S = S.drop_duplicates(["driver", "scene"]).merge(m, on="scene", how="left")
    sets = {k: set((out / f"n1_{k}.txt").read_text().split()) for k in ("sample", "nearstat")}
    gs = set(S.scene[S.driver == "GS"])
    miss = (sets["sample"] | sets["nearstat"]) - gs
    rows = []
    for d, x in S.groupby("driver", sort=False):
        for k, sc in [*sets.items(), ("all", None)]:
            y = x if sc is None else x[x.scene.isin(sc)]
            if sc is None and len(y) < 1491:
                continue
            r = dict(driver=d, set=k, **metrics(y))
            b = stats.bootstrap(y.score.fillna(0).to_numpy(float), groups=y.log.to_numpy())
            r.update(lo=b["lo"], hi=b["hi"])
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "n1_table.csv", index=False)
    g = T[(T.driver == "GS") & (T.set == "sample")].iloc[0]
    v = "supported" if g["hi"] <= 0.70 else "refuted" if g["lo"] >= 0.78 else (
        "point supported, CI straddles 0.70" if g["score"] <= 0.70 else "point refuted, CI straddles 0.78" if g["score"] >= 0.78 else "undetermined")
    md = [f"# SCENE-MIX N1: go-straight (starter kit, unmodified), public sample (amendment 3)\n\nGo-straight on the {g['n']}-scene log-stratified "
          f"sample: score {g['score']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}] (log-clustered), at-fault {g['km']:.3f} km, GT {g['gt']:.2f} m; private 0.799 / "
          f"0.258 km / 1.15 m. Lines <= 0.70 supported, >= 0.78 refuted: **{v}**. Scenes missing from the go-straight run: {len(miss)}.\n",
          "| driver | set | n | score [95 % CI] | at-fault km | GT m | zeros | route fold | collision at fault | offroad | corridor | driven m |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    md += [f"| {x.driver} | {x.set} | {x.n} | {x.score:.3f} [{x.lo:.3f}, {x.hi:.3f}] | {x.km:.3f} | {x.gt:.2f} | {x.zeros} | {x.z_fold} | {x.z_coll} | "
           f"{x.z_off} | {x.z_corr} | {x.driven_m:.1f} |" for x in T.sort_values(["set", "driver"]).itertuples()]
    (out / "n1.md").write_text("\n".join(md) + "\n\nScore = mean over the set, failed scenes 0. Zero types exclusive in this order: route fold "
                               "(no metrics), collision at fault, offroad, corridor.\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    smp = sp.add_parser("sample")
    smp.add_argument("--n", type=int, default=300)
    smp.add_argument("--out", required=True)
    o = sp.add_parser("n1")
    o.add_argument("gs")
    o.add_argument("--out", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--step", type=float, default=0.5)
    r = sp.add_parser("report")
    r.add_argument("windows")
    r.add_argument("--out", required=True)
    r.add_argument("--gs")
    a = ap.parse_args()
    {"extract": extract, "report": report, "sample": sample, "n1": n1}[a.cmd](a)


if __name__ == "__main__":
    main()
