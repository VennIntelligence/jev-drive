#!/usr/bin/env python3
"""ARENA-RESID: where RSG-L18-s0 + JEV_HOLD=stop3 still loses points on the arena's held windows (results/arena_hold/residual.md).
CPU only, stored reads. Box, repo root, AlpaSim's venv ($P). $AR = $DATA_DIR/runs/alpasim_proxy/arena; outputs under $AR/resid.

  $P arena_resid.py extract --read TAG=DIR [TAG=DIR ...] --out DIR [--set F] [--workers N]
        per read (DIR holds c<n>/run): DIR/steps_<TAG>.parquet = every scorer metric per metric step and window
        (aggregate/metrics_unprocessed.parquet, wide), DIR/calls_<TAG>.parquet = one row per driver call (drive.jsonl: route command,
        route point, ego anchor (x, y, yaw, local frame), plan poses; hold.jsonl joined when present). --set keeps the held windows
  $P arena_resid.py offroad --arena AR [--tag stop3] [--workers N]   per offroad zero of the read: the human's footprint (its logged
        track of the window, Pacifica footprint around the rear axle) against the drivable surface (lane and road-area polygons, the
        scorer's own polygon builders via route_follow's rf_offroad.Road), the ego footprint at the first offroad step (ego anchors
        interpolated), the location class (rf_offroad.rclass on the human's track) -> AR/resid/offroad_<tag>.parquet
  $P arena_resid.py report --arena AR --out F.md   the point accounting and the lever estimates (residual.md's tables)

Scorer facts used (third_party alpasim src/eval, read 2026-10-18): score = 0 on at-fault (front / lateral) collision, offroad or a
lateral corridor exit; else min(clamp(progress_rel_to_total, 0, 1) / 0.8, 1), or 1 when the human travels < 5 m. progress_rel_to_total
= the ego centroid's normalised projection onto the human's window track (scaled to the reachable part), taken at the last kept metric
step. Kept steps: from the first policy-relevant step (eval_relevant, 1.017 s; the force-GT steps are dropped) up to and including the
first offroad / any collision / d >= 4 m step. Any collision (rear included) therefore freezes progress there.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
MODELS = ("RSG", "NR1F", "NR1HV2", "YR", "JTN_off", "JTN_stop3", "JTB_stop3")      # unheld or held reads compared on stop3's offroad zeros


# ---------------------------------------------------------------- extract
def _steps(run):
    import polars as pl
    f = Path(run) / "aggregate/metrics_unprocessed.parquet"
    if not f.exists():
        return None
    m = pl.read_parquet(f, columns=["name", "timestamps_us", "values", "clipgt_id"]).with_columns(pl.col("timestamps_us").cast(pl.Int64))
    w = m.pivot(on="name", index=["clipgt_id", "timestamps_us"], values="values", aggregate_function="first").sort(["clipgt_id", "timestamps_us"])
    return w.rename({"clipgt_id": "id", "timestamps_us": "t_us"}).to_pandas()


def _calls(run):
    import pandas as pd
    rows = []
    f = Path(run) / "driver-logs/drive.jsonl"
    if not f.exists():
        return None
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") != "drive":
            continue
        P = np.asarray(json.loads(r["poses"]) if isinstance(r["poses"], str) else r["poses"], float)
        p = np.r_[np.zeros((1, 3)), P[:, :3]]
        seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
        rows.append(dict(id=r["scene"], session=r["session"], t0=int(r["t0"]), k=int(r["k"]), cmd=r.get("cmd"),
                         route_x=(r.get("route0") or [np.nan, np.nan])[0], route_y=(r.get("route0") or [np.nan, np.nan])[1],
                         line=json.dumps(r.get("line")), motion=r.get("motion"), ax=r["anchor"][0], ay=r["anchor"][1], ayaw=r["anchor"][2],
                         arc2=float(seg[:4].sum()), arc4=float(seg.sum()), y2=float(P[3, 1]), y4=float(P[-1, 1]), yaw4=float(P[-1, 2]),
                         poses=json.dumps(np.round(P, 3).tolist())))
    C = pd.DataFrame(rows)
    h = Path(run) / "driver-logs/hold.jsonl"
    if h.exists() and len(C):
        H = pd.DataFrame([json.loads(x) for x in open(h)]).drop(columns=["poses", "scene"], errors="ignore")
        C = C.merge(H, on=["session", "t0"], how="left")
    return C


def _one(job):
    tag, run, ids = job
    s, c = _steps(run), _calls(run)
    if ids is not None:
        s = s[s.id.isin(ids)] if s is not None else None
        c = c[c.id.isin(ids)] if c is not None else None
    for x in (s, c):
        if x is not None:
            x.insert(0, "tag", tag)
    return tag, s, c


def cmd_extract(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = None
    if a.set:
        S = pd.read_csv(a.set)
        ids = set(S[S.split == "held"].id)
    jobs = [(t, r, ids) for x in a.read for t, d in [x.split("=", 1)] for r in sorted(glob.glob(f"{d}/c*/run"))]
    with Pool(a.workers) as p:
        res = p.map(_one, jobs)
    for t in dict.fromkeys(j[0] for j in jobs):
        S = pd.concat([s for tt, s, _ in res if tt == t and s is not None], ignore_index=True)
        C = pd.concat([c for tt, _, c in res if tt == t and c is not None], ignore_index=True)
        S.to_parquet(out / f"steps_{t}.parquet")
        C.to_parquet(out / f"calls_{t}.parquet")
        print(json.dumps(dict(tag=t, windows=int(S.id.nunique()), steps=len(S), calls=len(C))))


# ---------------------------------------------------------------- shared tables
def held(AR):
    """The held windows the hold reports use (route-fail dropped), the stored per-window rows of the hold reads and the arena reads."""
    import pandas as pd
    S = pd.read_csv(f"{AR}/set/set.csv.gz").set_index("id")
    R = pd.read_csv(f"{AR}/hold/report2/rows.csv.gz")
    R0 = pd.read_csv(f"{AR}/hold/report2/ref_rows.csv.gz")
    bad = set(R0[(R0.tag == "RSG-L18-s0") & (R0.type == "route-fail")].id) | set(R[R.type == "route-fail"].id)
    return S[(S.split == "held") & ~S.index.isin(bad)].copy(), R, R0


def steps(AR, tag, ids):
    import pandas as pd
    St = pd.read_parquet(f"{AR}/resid/steps_{tag}.parquet")
    St = St[St.id.isin(ids)].copy()
    St["t"] = (St.t_us - St.groupby("id").t_us.transform("min")) / 1e6
    return St


def offinfo(St):
    """Per window with an offroad step after engagement: offroad already during force-GT (off_pre: the ego sits on the human's pose
    there), the first policy-relevant offroad step and the ego's distance to the human's track / current human pose there."""
    import pandas as pd
    o = []
    for i, g in St.sort_values("t").groupby("id"):
        off, rel = g.offroad.values > 0, g.eval_relevant.values > 0
        if not (off & rel).any():
            continue
        k = int(np.argmax(off & rel))
        o.append(dict(id=i, off_pre=bool((off & ~rel).any()), t_off=g.t.values[k], t_off_us=int(g.t_us.values[k]),
                      d_off=g.dist_to_gt_trajectory.values[k], dloc_off=g.dist_to_gt_location.values[k], trav_off=g.dist_traveled_m.values[k]))
    return pd.DataFrame(o, columns=["id", "off_pre", "t_off", "t_off_us", "d_off", "dloc_off", "trav_off"]).set_index("id")


def window_table(AR, tag="stop3"):
    """stop3's held windows: score / type, progress at the last kept step, obstacle distance, latch class, the human's end state."""
    import pandas as pd
    H, R, R0 = held(AR)
    W = H.join(R[R.tag == tag].drop_duplicates("id").set_index("id")[["score", "type", "dist", "gt"]].rename(columns={"gt": "gtd"}))
    r0 = R0[R0.tag == "RSG-L18-s0"].drop_duplicates("id").set_index("id")
    W["rsg_score"], W["rsg_dist"], W["rsg_type"] = r0.score, r0.dist, r0.type
    St = steps(AR, tag, W.index)
    g = St[St.eval_relevant > 0].copy()
    g["ev"] = (g.collision_any > 0) | (g.offroad > 0) | (g.dist_to_gt_trajectory >= 4)
    first = g[g.ev].groupby("id").t.min()
    k = g[g.t <= g.id.map(first).fillna(99)]
    W["prog"] = k.groupby("id").progress_rel_to_total.last().clip(0, 1)
    W["obs_min"] = k.groupby("id").min_distance_to_obstacle_m.min()
    W["dmax"] = g.groupby("id").dist_to_gt_trajectory.max()
    C = pd.read_parquet(f"{AR}/resid/calls_{tag}.parquet")
    C = C[C.id.isin(W.index)]
    f = C[C.state.isin(["held", "free"])].sort_values("t0").drop_duplicates("id").set_index("id").reindex(W.index)
    W["latch"] = np.where(f.state == "held", np.where(f.stop_a.notna(), "stop", "static"), "free")
    hd = pd.read_csv(f"{AR}/corridor/hand.csv.gz").set_index("id")
    W = W.join(hd[["arc_after_hand", "gt"]].rename(columns={"gt": "gt_h"}))
    W["still"] = W.index.map(pd.read_csv(f"{AR}/corridor/gtend_arena.csv.gz").set_index("id").seg_last_mm) < 1
    W["loss"] = 1 - W.score
    pr = np.where(W.gtd < 5, 1.0, np.minimum(W.prog / 0.8, 1.0))
    nz = (W.score > 0).values
    assert np.abs(pr[nz] - W.score.values[nz]).max() < 1e-6, "progress reconstruction"
    return W


# ---------------------------------------------------------------- offroad geometry
def _geo_one(job):
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    import rf_offroad as RF
    import shapely
    from ao_e1 import loader
    sc, cfg, anc, t_off = job
    try:
        ds = loader(cfg).get_data_source(sc)
        tr = ds.rig.trajectory
        P, Y = np.asarray(tr.positions, float)[:, :2], np.asarray(tr.yaws, float)
        road = RF.Road(ds.map, np.r_[P, anc[:, 1:3]])
        corn = np.array(RF.CORN)
        hfp = [RF.RM.to_local((*P[i], Y[i]), corn) for i in range(len(P))]
        hcov = np.array([road.U.covers(shapely.Polygon(x)) for x in hfp])
        hmarg = np.array([road.margin(x).min() for x in hfp])
        tk = anc[:, 0]
        z = np.array([np.interp(t_off, tk, anc[:, 1]), np.interp(t_off, tk, anc[:, 2])])
        zy = float(np.interp(t_off, tk, np.unwrap(anc[:, 3])))
        efp = RF.RM.to_local((z[0], z[1], zy), corn)
        j = int(np.argmin(np.hypot(*(P - z).T)))
        gl = float(np.hypot(*np.diff(P, axis=0).T).sum())
        out = dict(id=sc, h_off_frac=float((~hcov).mean()), h_off_any=bool((~hcov).any()), h_min_margin=float(hmarg.min()),
                   e_cov=bool(road.U.covers(shapely.Polygon(efp))), e_margin=float(road.margin(efp).min()), near_d=float(np.hypot(*(P[j] - z))),
                   h_cov_near=bool(hcov[j]), h_margin_near=float(hmarg[j]), dyaw_near=float(np.degrees(RF._wrap(zy - Y[j]))),
                   junction=bool(road.junction(z)), gt_len=gl)
        try:
            out["cls"] = RF.rclass(road, P, z) if gl > 5 else "short"
        except Exception:  # noqa: BLE001
            out["cls"] = "n/a"
        return out
    except Exception as e:  # noqa: BLE001
        return dict(id=sc, err=repr(e))


def cmd_offroad(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    AR = a.arena
    H, R, _ = held(AR)
    R = R[R.tag == a.tag].drop_duplicates("id").set_index("id")
    z = R.index[(R.score == 0) & R.type.str.contains("offroad") & R.index.isin(H.index)]
    oi = offinfo(steps(AR, a.tag, z))
    C = pd.read_parquet(f"{AR}/resid/calls_{a.tag}.parquet")
    C = C[C.id.isin(z)].sort_values("t0").drop_duplicates(["id", "t0"], keep="last")
    jobs = [(i, f"{AR}/hold/reads/{a.tag}/{R.chunk[i]}/run/generated-user-config-0.yaml", C[C.id == i][["t0", "ax", "ay", "ayaw"]].values.astype(float),
             float(oi.t_off_us[i])) for i in z]
    with ProcessPoolExecutor(a.workers) as ex:
        G = pd.DataFrame(list(ex.map(_geo_one, jobs))).set_index("id").join(oi)
    for t in MODELS:
        G[f"off_{t}"] = G.index.isin(offinfo(steps(AR, t, z)).index)
    G.to_parquet(f"{AR}/resid/offroad_{a.tag}.parquet")
    print(json.dumps(dict(zeros=len(z), errors=int(G.get("err", pd.Series(dtype=object)).notna().sum()), ego_uncovered=int((~G.e_cov).sum()))))


# ---------------------------------------------------------------- report
def _auc(pos, neg):
    from scipy.stats import rankdata
    x = np.r_[pos, neg]
    r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _summ(d):
    import pandas as pd
    o = {}
    for f in glob.glob(f"{d}/c*/run/aggregate/results-summary.json"):
        for r in json.load(open(f))["rollouts"]:
            o[r["clipgt_id"]] = float(r.get("score") or 0)
    return pd.Series(o, dtype=float)


def cmd_report(a):
    import pandas as pd
    AR = a.arena
    W = window_table(AR)
    N = len(W)
    G = pd.read_parquet(f"{AR}/resid/offroad_stop3.parquet").join(W[["stratum"]])
    G["grp"] = np.select([G.off_pre, G.h_off_any, G.stratum.eq("static") & ~G.h_off_any],
                         ["offroad: human pose offroad at engagement", "offroad: human track offroad in the window", "offroad: held static roll"],
                         "offroad: policy, straight / near edge")
    G.loc[(G.grp == "offroad: policy, straight / near edge") & (G.cls.isin(["junction", "curve", "approach"]) | G.junction), "grp"] = "offroad: policy, turn / junction"
    zc = lambda t: "AF-front" if t.startswith("AF-front") else "AF-side" if t.startswith("AF-") else "corridor" if t == "corridor" else t  # noqa: E731
    W["cls"] = np.where(W.score == 0, W.type.map(zc), "")
    W.loc[G.index, "cls"] = G.grp
    part = (W.score > 0) & (W.score < 1)
    W.loc[part, "cls"] = np.where(W.latch[part] == "static", "partial: static latch (" + W.type[part] + ")",
                                  np.where(W.latch[part] == "stop", "partial: stop latch", "partial: free (" + W.type[part].replace({"ok": "slow"}) + ")"))
    W.loc[part & (W.latch == "static") & (W.type == "ok"), "cls"] = "partial: static latch (slow)"
    L = W[W.loss > 0]
    T = L.pivot_table(index="cls", columns="stratum", values="loss", aggfunc="sum", fill_value=0.0)
    T["points"], T["windows"] = T.sum(axis=1), L.groupby("cls").size()
    T["per window"] = T.points / N
    T["share of gap"] = T.points / L.loss.sum()
    T = T.sort_values("points", ascending=False)
    T.loc["all"] = T.sum()
    T.loc["all", "share of gap"] = 1.0
    o = ["# ARENA-RESID tables (arena_resid.py report)", "",
         f"stop3 on {N} held windows: mean {W.score.mean():.4f}, gap to 1.0 = {L.loss.sum():.1f} points ({L.loss.sum() / N:.4f}).", "",
         "## Point accounting (points = summed 1 - score; per window = points / held windows)", "",
         "| class | " + " | ".join(c for c in T.columns[:-4]) + " | points | windows | per window | share |", "|---|" + "--:|" * (len(T.columns))]
    for c, r in T.iterrows():
        o.append(f"| {c} | " + " | ".join(f"{r[s]:.1f}" for s in T.columns[:-4]) + f" | {r.points:.1f} | {int(r.windows)} | {r['per window']:.4f} | {r['share of gap']:.2f} |")
    # offroad
    o += ["", "## Offroad zeros", "", "| group | windows | first offroad step (s), median | ego to the human track there (m), median | "
          "|ego - human heading| there (deg), median | human footprint min margin (m), median | ego footprint uncovered | also offroad in "
          + " / ".join(MODELS) + " (sum) | all 7 |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    G["n_mod"] = G[[f"off_{t}" for t in MODELS]].sum(axis=1)
    for gname, g in G.groupby("grp"):
        o.append(f"| {gname} | {len(g)} | {g.t_off.median():.2f} | {g.d_off.median():.2f} | {g.dyaw_near.abs().median():.1f} | {g.h_min_margin.median():+.2f} | "
                 f"{int((~g.e_cov).sum())} | " + " / ".join(str(int(g[f'off_{t}'].sum())) for t in MODELS) + f" | {int((g.n_mod == 7).sum())} |")
    o.append(f"\nLocation class of the policy groups: {G[G.grp.str.contains('policy')].cls.value_counts().to_dict()}; logs of all 43: "
             f"{G.index.str[:38].value_counts().head(5).to_dict()}.")
    hum = G.index[G.grp.str.contains("human")]
    keep = W.index.difference(hum)
    o.append(f"\nWithout the {len(hum)} windows whose human track is itself offroad (a human replay would score 0 there; prereg amendment 1 drops "
             f"replay-zero windows when a replay read exists): stop3 {W.score[keep].mean():.4f}, RSG {W.rsg_score[keep].mean():.4f} on {len(keep)} windows.")
    # free partials
    F = W[(W.latch == "free") & (W.score > 0) & (W.score < 1)].copy()
    F["obs"] = pd.cut(F.obs_min.clip(upper=1e9), [0, 5, 10, 20, 1e9], labels=["< 5 m", "5-10 m", "10-20 m", ">= 20 m (or none)"])
    o += ["", "## Free-session partials (no latch): points by the nearest obstacle over the kept steps (scorer min_distance_to_obstacle_m)", "",
          "| nearest obstacle | windows | points | median distance ratio ego / human | median progress / 0.8 |", "|---|--:|--:|--:|--:|"]
    for b, g in F.groupby("obs", observed=False):
        o.append(f"| {b} | {len(g)} | {g.loss.sum():.1f} | {(g.dist / g.gtd).median():.2f} | {(g.prog / 0.8).median():.2f} |")
    o.append(f"| all | {len(F)} | {F.loss.sum():.1f} | {(F.dist / F.gtd).median():.2f} | {(F.prog / 0.8).median():.2f} |")
    o.append(f"\nBy stratum: {F.groupby('stratum').loss.sum().round(1).to_dict()}; by type: {F.groupby('type').loss.sum().round(1).to_dict()}.")
    # leash sweep
    S = W[W.latch == "static"].copy()
    pre = S.gt_h - S.arc_after_hand
    reach = (S.rsg_dist - pre).clip(lower=0)

    def est(Lm):
        return np.where(S.gt_h < 5, 1.0, np.minimum((pre + np.minimum(reach, Lm)) / (0.8 * S.gt_h), 1.0))
    p3, ok = est(3.0), ((S.type != "rear-hit") & (S.score > 0)).values
    o += ["", "## Static leash length (offline, the 1 004 static-latched sessions)", "",
          f"Still-end latched windows under stop3: largest distance to the human's rest point {S[S.still].dmax.max():.2f} m (leash 3 m; a zero needs 4). "
          "Progress: launch windows credited min(pre + min(RSG reach, leash), ...) as diag.md (rear-hit and zero windows unchanged); new front hits: "
          "latched windows with a logged stopped lead whose gap lies in (3.05 m, leash + 0.05 m] and RSG's reach past it (the 11 leash3 hits were at "
          "gap + 0.03..0.13 m). Calibration at 3 m: estimate "
          f"{p3[ok & (S.stratum == 'launch').values].sum():.1f} vs read {S.score[ok & (S.stratum == 'launch').values].sum():.1f} on the launch windows.", "",
          "| leash (m) | launch progress (points) | new front hits | their current points | net | per window |", "|--:|--:|--:|--:|--:|--:|"]
    for Lm in (3.3, 3.5, 3.7, 3.9):
        d = np.where(ok, est(Lm) - p3, 0.0).sum()
        hit = (S.lead == True) & (S.lead_gap > 3.05) & (S.lead_gap <= Lm + 0.05) & (reach > S.lead_gap - 0.1) & (S.score > 0)  # noqa: E712
        o.append(f"| {Lm} | {d:+.1f} | {int(hit.sum())} | -{S.score[hit].sum():.1f} | {d - S.score[hit].sum():+.1f} | {(d - S.score[hit].sum()) / N:+.4f} |")
    # launch observable: the held plan of RSG (stop3) and of JTN (stop3)
    JD = Path(AR).parents[1] / "defect_fix/jt/arena"
    W["jtn_off"], W["jtn_s3"] = _summ(JD / "JTN-L18-s0@off").reindex(W.index), _summ(JD / "JTN-L18-s0@stop3").reindex(W.index)
    pos_i = W.index[(W.latch == "static") & (W.stratum == "launch") & (W.gt_h >= 5)]
    neg_i = W.index[(W.latch == "static") & W.still & (W.stratum == "static")]
    o += ["", "## Launch: does a held plan separate launch (human >= 5 m) from still-end static windows?", "",
          f"Means on the {len(pos_i)} latched launch / {len(neg_i)} still-end windows: stop3 {W.score[pos_i].mean():.3f} / {W.score[neg_i].mean():.3f}, "
          f"RSG unheld {W.rsg_score[pos_i].mean():.3f} / {W.rsg_score[neg_i].mean():.3f}, JTN unheld {W.jtn_off[pos_i].mean():.3f} / "
          f"{W.jtn_off[neg_i].mean():.3f}, JTN + stop3 {W.jtn_s3[pos_i].mean():.3f} / {W.jtn_s3[neg_i].mean():.3f} (all held: stop3 {W.score.mean():.4f}, "
          f"JTN + stop3 {W.jtn_s3.mean():.4f}).", "", "| read | AUC, plan 2 s arc max while held (calls from 1 s) | release at arc >= 2 m: launch / still-end released | arc >= 3 m |",
          "|---|--:|--:|--:|"]
    for tag in ("stop3", "JTN_stop3"):
        C = pd.read_parquet(f"{AR}/resid/calls_{tag}.parquet")
        f = C[C.id.isin(W.index) & (C.t_s >= 1) & (C.state == "held")].groupby("id").arc_model.max()
        p, n = f.reindex(pos_i).dropna(), f.reindex(neg_i).dropna()
        o.append(f"| {tag} | {_auc(p.values, n.values):.2f} | {(p >= 2).mean():.2f} / {(n >= 2).mean():.2f} | {(p >= 3).mean():.2f} / {(n >= 3).mean():.2f} |")
    # the rest
    cz = W[W.cls == "corridor"]
    o += ["", f"Corridor zeros: {len(cz)}, latch {cz.latch.value_counts().to_dict()}, strata {cz.stratum.value_counts().to_dict()}, human end speed "
          f"0 in {int((cz.v_end < 0.3).sum())}; RSG zero there too in {int((cz.rsg_score == 0).sum())}.",
          f"AF zeros: {W.cls.isin(['AF-front', 'AF-side']).sum()}: static-latched with a stopped lead at "
          f"{W[(W.cls == 'AF-front') & (W.latch == 'static')].lead_gap.min():.2f}-{W[(W.cls == 'AF-front') & (W.latch == 'static')].lead_gap.max():.2f} m "
          f"{int(((W.cls == 'AF-front') & (W.latch == 'static')).sum())} (diag.md (b)), free {int(((W.cls.isin(['AF-front', 'AF-side'])) & (W.latch == 'free')).sum())}."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--read", nargs="+", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=16)
    p.add_argument("--set", help="the arena set (set.csv.gz): keep its held windows only")
    p = sp.add_parser("offroad")
    p.add_argument("--arena", required=True), p.add_argument("--tag", default="stop3"), p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("report")
    p.add_argument("--arena", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(extract=cmd_extract, offroad=cmd_offroad, report=cmd_report)[a.cmd](a)
