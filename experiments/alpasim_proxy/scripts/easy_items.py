#!/usr/bin/env python3
"""EASY-ITEMS: which arena held windows the strong (garage-type) pool aces, why JTN-L18-s1 native misses them, and what fixing each cause is
worth on the frozen board-mix PCS (decisions 382 / 384). Diagnostic, CPU only, no new rollouts.

  $AP easy_items.py extract                     (box, AlpaSim venv, CPU) per-step scorer metrics of the reads that have none in
        native_gap's steps.parquet (garage 0014, LTF, JTN-L18-s1 + stop3, RSG + stop3), held windows only -> $O/steps_extra.parquet
  tmp/pcs_local/.venv/bin/python easy_items.py report [--extra tmp/easy_items/steps_extra.parquet]
        (Mac) every table of results/easy_items.md -> results/easy_items.json + results/easy_items/windows.csv.gz

Scorer facts used (alpasim eval 0bb4c4b, scene_score.py / ground_truth.py / processing.py): score = 0 on at-fault collision (front or lateral),
offroad, or leaving the corridor sideways; otherwise min(progress_clipped_rel / 0.8, 1), progress_clipped_rel = the ego polygon centroid's
projection on the full logged path at the last kept step, scaled by the path reachable in the simulated time; 1 if the logged path is < 5 m.
Steps after any collision (rear included) or offroad are dropped. There are no comfort / TTC / lane-margin terms, so every non-1, non-0
score is a progress shortfall: near-1 (0.98-0.999) = progress 0.784-0.799 of the reachable human arc.

Easy item: a window in the frozen board-mix item set (2 106 held windows) where garage 0014 scores exactly 1 (>= 1 - 1e-4). Variant
"easy-hi": easy and board-mix discrimination a >= 2 (the top half of easy items by a).
Causes of a JTN-s1 native miss on an easy window (first match, from the arena type, the per-step metrics and d384's decision table):
  zero: corridor (static window / moving window), offroad, at-fault collision;
  partial: struck from behind (rear collision truncates the rollout), late launch (human starts < 1 m/s), stop / lock, following a lead,
           free-road ratchet (last same-station speed ratio < 0.9), near-1 (0.98-0.999, any of the above at the margin, split by sub-cause),
           other.
PCS of a cause = frozen PCS with that class set to exactly 1 minus as read (board-mix and g15), CI = log-clustered bootstrap (2 000 draws)
of the paired difference with items fixed (slow_rescore.Frozen).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
R = DATA / "runs"
EXTRA = {"garage": R / "alpasim_proxy/arena/reads/G0014", "LTF": R / "alpasim_proxy/arena/reads/LTF-sample",
         "JTN1S3": R / "defect_fix/jt/arena/JTN-L18-s1@stop3", "RSGS3": R / "alpasim_proxy/arena/hold/reads/stop3"}
O = R / "alpasim_proxy/easy_items"
KEEP = ["t_us", "collision_front", "collision_rear", "collision_lateral", "offroad", "min_distance_to_lane_boundary_m",
        "min_distance_to_obstacle_m", "progress_rel_to_total", "dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory",
        "left_corridor_laterally", "dist_to_gt_location", "dist_traveled_m", "gt_dist_traveled_m", "eval_relevant"]


def _one(job):
    import arena_resid as RZ
    tag, run, held = job
    s = RZ._steps(run)
    if s is None:
        return None
    s = s[s.id.isin(held)]
    s = s[["id"] + [c for c in KEEP if c in s.columns]].copy()
    s.insert(0, "tag", tag)
    return s


def cmd_extract(a):
    import pandas as pd
    from multiprocessing import Pool
    S = pd.read_csv(R / "alpasim_proxy/arena/set/set.csv.gz")
    held = set(S.id[S.split == "held"])
    jobs = [(t, r, held) for t, d in EXTRA.items() for r in sorted(glob.glob(f"{d}/c*/run"))]
    with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
        out = [x for x in p.map(_one, jobs) if x is not None]
    O.mkdir(parents=True, exist_ok=True)
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(O / "steps_extra.parquet")
    print(D.groupby("tag").id.nunique().to_dict(), len(D))


# ---------------------------------------------------------------- report (Mac)
SUBJ = {"JTN1": "JTN-s1", "JTN1S3": "JTN-s1+stop3", "RSG": "RSG", "RSGS3": "RSG+stop3"}


def per_window(st):
    """per-window kept-step summary of one read's metrics (after eval_relevant, up to and including the first collision / offroad)."""
    import pandas as pd
    st = st.sort_values(["id", "t_us"])
    st = st[st.eval_relevant > 0] if "eval_relevant" in st else st
    ev = ((st.collision_front > 0) | (st.collision_rear > 0) | (st.collision_lateral > 0) | (st.offroad > 0)).astype(int)
    st = st.assign(_ev=ev.groupby(st.id).cumsum().groupby(st.id).shift(1, fill_value=0))
    k = st[st._ev == 0]
    g = k.groupby("id")
    out = pd.DataFrame(dict(prog=g.progress_rel_to_total.last(), trav=g.dist_traveled_m.last(), gtlen=g.gt_dist_traveled_m.last(),
                            lat_med=g.lateral_dist_to_gt_trajectory.median(), lat_max=g.lateral_dist_to_gt_trajectory.max(),
                            dloc_end=g.dist_to_gt_location.last(), lane_min=g.min_distance_to_lane_boundary_m.min(),
                            obst_min=g.min_distance_to_obstacle_m.min(), rear=g.collision_rear.max(), n=g.size(),
                            t_end=g.t_us.last(), t0=g.t_us.first()))
    return out


def speed_profile(st, ids, nb=10):
    """ego speed along normalised time, from dist_traveled_m (median over windows) -> list of nb values."""
    import pandas as pd
    st = st[st.id.isin(ids)].sort_values(["id", "t_us"])
    st = st[st.eval_relevant > 0]
    rows = []
    for i, g in st.groupby("id"):
        t = (g.t_us.values - g.t_us.values[0]) / 1e6
        if len(t) < 5 or t[-1] <= 0:
            continue
        v = np.gradient(g.dist_traveled_m.values, t)
        b = np.minimum((t / t[-1] * nb).astype(int), nb - 1)
        rows.append(pd.Series(v).groupby(b).median().reindex(range(nb)).values)
    return np.nanmedian(np.array(rows), 0) if rows else np.full(nb, np.nan)


def cmd_report(a):
    import pandas as pd
    import slow_rescore as SR
    import pcs_local as PL
    import native_gap_report as NG
    res = REPO / "experiments/alpasim_proxy/results"
    out = res / "easy_items"; out.mkdir(parents=True, exist_ok=True)
    sc, ty, w = PL.load(res / "pcs_local")
    items = list(pd.read_csv(res / "pcs_local/items_boardmix.csv.gz", index_col=0).index)
    it = pd.read_csv(res / "pcs_local/items_boardmix.csv.gz", index_col=0)
    FZ = {k: SR.Frozen(p, sc) for k, p in SR.FITS.items()}
    assert list(FZ["boardmix"].items) == items
    S = pd.read_csv(Path(a.native_gap) / "set.csv.gz").drop_duplicates("id").set_index("id")
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO / "tmp/pcs_local/rows_*.csv.gz")))])
    G = sc.loc["garage", items]
    one = lambda y: y >= 1 - 1e-4  # noqa: E731
    zero = lambda y: y <= 1e-4  # noqa: E731
    easy = G.index[one(G)]
    hi = it.loc[easy].a >= 2.0
    easy_hi = easy[hi.values]
    J = {"main": dict(n_items=len(items), n_easy=len(easy), n_easy_hi=len(easy_hi))}
    strong3 = sc.loc[["garage", "LTF", "DD"], items]
    J["main"]["easy_all3"] = int(one(strong3).all(0).sum())
    J["main"]["a_median_easy"] = float(it.loc[easy].a.median()); J["main"]["a_median_other"] = float(it.drop(easy).a.median())

    # 1. scores of our reads on easy windows
    T1 = []
    for s in ["JTN-s1", "JTN-s1+stop3", "RSG", "RSG+stop3", "NTV0-s0", "garage", "LTF", "DD"]:
        for nm, ids in [("easy", easy), ("easy-hi", easy_hi), ("other", G.index.difference(easy))]:
            y = sc.loc[s, ids]
            T1.append(dict(subject=s, set=nm, n=len(ids), mean=float(y.mean()), ones=int(one(y).sum()), zeros=int(zero(y).sum()),
                           partial=int((~one(y) & ~zero(y)).sum()), near1=int(((y >= 0.98) & ~one(y)).sum())))
    J["scores"] = T1
    # by cell
    J["cells"] = [dict(cell=c, n=int((w.loc[easy].cell == c).sum()), jtn1_miss=int((~one(sc.loc["JTN-s1", easy]) & (w.loc[easy].cell == c)).sum()),
                       jtn1s3_miss=int((~one(sc.loc["JTN-s1+stop3", easy]) & (w.loc[easy].cell == c)).sum()))
                  for c in sorted(w.loc[easy].cell.unique())]

    # 2. causes of JTN-s1 native misses
    import pandas as pd  # noqa: F811
    st = pd.read_parquet(Path(a.native_gap) / "steps.parquet")
    ex = pd.read_parquet(a.extra) if Path(a.extra).exists() else None
    PW = {t: per_window(g) for t, g in st.groupby("tag")}
    if ex is not None:
        PW |= {t: per_window(g) for t, g in ex.groupby("tag")}
    D, _, _, _ = NG.load(Path(a.native_gap))
    causes = {}
    for tag in ["JTN1", "JTN1S3"]:
        subj = SUBJ[tag]
        rt = rows[rows.tag == {"JTN1": "JTN-L18-s1@off", "JTN1S3": "JTN-L18-s1@stop3"}[tag]].drop_duplicates("id").set_index("id")
        y = sc.loc[subj, items]
        miss = easy[~one(y.loc[easy]).values]
        pw = PW.get(tag)
        Dt = D[(D.tag == tag)] if tag in set(D.tag) else None
        C = []
        for i in miss:
            v = float(y[i]); typ = str(rt.type.get(i, "")); stt = S.loc[i, "stratum"]
            p = pw.loc[i] if pw is not None and i in pw.index else None
            rear = bool(rt.rear.get(i, 0)) or (p is not None and p.rear > 0)
            sub = None
            if Dt is not None and not zero(v):
                g = Dt[(Dt.id == i) & Dt.valid & (Dt.k >= 2)].sort_values("k")
                mv = g[g.vh >= 2]
                stop = bool(((g.vh >= 2) & (g.v < 0.3) & (g.v.cummax().shift(1) >= 1)).any())
                lead = (mv.lead_gap < 25).mean() if len(mv) else np.nan
                last = (mv.v / mv.vh).iloc[-1] if len(mv) else np.nan
                sub = ("late launch" if S.loc[i, "v0"] < 1.0 else "stop / lock" if stop else "following a lead" if len(mv) and lead >= 0.5
                       else "free-road ratchet" if len(mv) and last < 0.9 else "other (ratio >= 0.9 or no moving decision)")
            elif not zero(v):
                sub = "late launch" if S.loc[i, "v0"] < 1.0 else "progress short (no decision table)"
            if zero(v):
                c = {"corridor": "zero: corridor, static window" if stt == "static" else "zero: corridor, moving window", "offroad": "zero: offroad"}.get(
                    typ, "zero: at-fault collision" if typ.startswith("AF") else f"zero: {typ}")
            elif rear:
                c = "partial: struck from behind"
            elif v >= 0.98:
                c = "partial: near-1 (0.98-0.999)"
            else:
                c = f"partial: {sub}"
            C.append(dict(id=i, score=v, type=typ, stratum=stt, cell=w.loc[i, "cell"], cause=c, sub=sub, rear=rear, a=float(it.loc[i, "a"]),
                          prog=float(p.prog) if p is not None else np.nan, trav=float(p.trav) if p is not None else np.nan,
                          gtlen=float(p.gtlen) if p is not None else np.nan, lat_med=float(p.lat_med) if p is not None else np.nan,
                          lat_max=float(p.lat_max) if p is not None else np.nan))
        C = pd.DataFrame(C)
        C.to_csv(out / f"misses_{tag}.csv.gz", index=False, float_format="%.4f")
        causes[tag] = pcs_table(C, y.to_numpy(float), items, FZ, S, subj)
    J["causes"] = causes

    # 3. garage vs ours on easy windows (per-step metrics)
    if "garage" in PW:
        cmp = []
        for tag in ["JTN1", "JTN1S3", "RSG"]:
            if tag not in PW:
                continue
            subj = SUBJ[tag]
            y = sc.loc[subj, easy]
            for nm, ids in [("all easy", easy), ("JTN miss, partial", easy[(~one(y) & ~zero(y)).values]), ("JTN = 1", easy[one(y).values])]:
                ids = ids.intersection(PW[tag].index).intersection(PW["garage"].index)
                a_, g_ = PW[tag].loc[ids], PW["garage"].loc[ids]
                cmp.append(dict(subject=subj, set=nm, n=len(ids),
                                prog_ours=float(a_.prog.median()), prog_garage=float(g_.prog.median()),
                                trav_ratio_ours=float((a_.trav / a_.gtlen).median()), trav_ratio_garage=float((g_.trav / g_.gtlen).median()),
                                lat_med_ours=float(a_.lat_med.median()), lat_med_garage=float(g_.lat_med.median()),
                                lat_max_ours=float(a_.lat_max.median()), lat_max_garage=float(g_.lat_max.median()),
                                dloc_end_ours=float(a_.dloc_end.median()), dloc_end_garage=float(g_.dloc_end.median()),
                                lane_min_ours=float(a_.lane_min.median()), lane_min_garage=float(g_.lane_min.median()),
                                v_ours=speed_profile(st[st.tag == tag] if tag in set(st.tag) else ex[ex.tag == tag], ids).round(2).tolist(),
                                v_garage=speed_profile(ex[ex.tag == "garage"], ids).round(2).tolist()))
        J["garage_cmp"] = cmp
        # longitudinal vs lateral decomposition of the progress gap on JTN1's partial misses:
        # progress = projection of the centroid; lateral offset alone cannot shorten it (projection), so shortfall = arc short of human
        Cj = pd.read_csv(out / "misses_JTN1.csv.gz")
        pj = Cj[~Cj.cause.str.startswith("zero")].set_index("id")
        g_ = PW["garage"].reindex(pj.index)
        J["lon_lat"] = dict(n=len(pj), prog_ours=float(pj.prog.median()), prog_garage=float(g_.prog.median()),
                            trav_ours=float((pj.trav / pj.gtlen).median()), trav_garage=float((g_.trav / g_.gtlen).median()),
                            lat_med_ours=float(pj.lat_med.median()), lat_med_garage=float(g_.lat_med.median()),
                            share_lat_over_1m=float((pj.lat_max > 1.0).mean()), share_lat_over_1m_garage=float((g_.lat_max > 1.0).mean()))
        # board-style GT distance (mean over all windows of dist_to_gt... at end): use arena 'gt' column? report per-window median dloc_end
    json.dump(J, open(res / "easy_items.json", "w"), indent=1, default=float)
    print(json.dumps(J, indent=1, default=float)[:20000])


def pcs_table(C, y, items, FZ, S, subj):
    """PCS gain of setting each cause class (and unions) to exactly 1, board-mix and g15, log-clustered bootstrap CI."""
    import pandas as pd
    logs = S.log.reindex(items).to_numpy()
    codes, uniq = pd.factorize(logs)
    rng = np.random.default_rng(0)
    draw = rng.integers(len(uniq), size=(SR_NB, len(uniq)))
    cnt = np.stack([np.bincount(d, minlength=len(uniq)) for d in draw], 1)
    W = cnt[codes].astype(float)
    groups = [(c, C.id[C.cause == c]) for c in C.cause.value_counts().index]
    groups += [("all partials", C.id[~C.cause.str.startswith("zero")]), ("all zeros", C.id[C.cause.str.startswith("zero")]),
               ("all misses", C.id)]
    res = []
    for k, fz in FZ.items():
        L0 = fz.L(y)
        p0 = fz.pcs(L0, np.ones(len(items))); p0b = fz.pcs(L0, W)
        for nm, ids in groups:
            y2 = y.copy(); m = np.isin(items, ids); y2[m] = 1.0
            L2 = L0.copy(); L2[:, m] = _Lsub(fz, y2, m)
            d = fz.pcs(L2, np.ones(len(items))) - p0
            db = fz.pcs(L2, W) - p0b
            lo, hi = np.quantile(db, [0.025, 0.975])
            sel = C[C.id.isin(ids)]
            res.append(dict(fit=k, base=float(p0), cause=nm, windows=len(sel), points=float((1 - sel.score).sum()), pcs=float(d), lo=float(lo), hi=float(hi)))
    return res


SR_NB = 2000


def _Lsub(fz, y2, m):
    """per-item log-likelihood columns for the changed items only, at the frozen items."""
    full = np.full(len(y2), np.nan); full[m] = y2[m]
    Lm = fz.L(full)
    return Lm[:, m]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("extract")
    r = sp.add_parser("report")
    r.add_argument("--native-gap", default=str(REPO / "tmp/native_gap"))
    r.add_argument("--extra", default=str(REPO / "tmp/easy_items/steps_extra.parquet"))
    a = ap.parse_args()
    {"extract": cmd_extract, "report": cmd_report}[a.cmd](a)
