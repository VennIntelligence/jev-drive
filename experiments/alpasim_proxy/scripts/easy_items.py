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
                            gtd=g.dist_to_gt_trajectory.max(),
                            dloc_end=g.dist_to_gt_location.last(), lane_min=g.min_distance_to_lane_boundary_m.min(),
                            obst_min=g.min_distance_to_obstacle_m.min(), rear=g.collision_rear.max(), n=g.size(),
                            t_end=g.t_us.last(), t0=g.t_us.first()))
    return out


def speed_profile(st, ids, grid=np.arange(0.25, 3.76, 0.5)):
    """median ego speed (m/s) at fixed times after the first scored step, from dist_traveled_m (0.5 s steps, kept steps only)."""
    st = st[st.id.isin(ids) & (st.eval_relevant > 0)].sort_values(["id", "t_us"])
    rows = []
    for _, g in st.groupby("id"):
        t = (g.t_us.values - g.t_us.values[0]) / 1e6
        v = np.diff(g.dist_traveled_m.values) / np.maximum(np.diff(t), 1e-3)
        rows.append(np.interp(grid, 0.5 * (t[1:] + t[:-1]), v, right=np.nan) if len(v) else np.full(len(grid), np.nan))
    return np.nanmedian(np.array(rows), 0) if rows else np.full(len(grid), np.nan)


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
    causes, natsub = {}, {}
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
            elif not zero(v):                                  # stop3: same model, same frames until the hold engages -> native's sub-cause
                sub = natsub.get(i, "stop3 hold, native = 1")
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
        if tag == "JTN1":
            natsub = C.set_index("id")["sub"].dropna().to_dict()
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
        # board GT distance = per-scene MAX of dist_to_gt_trajectory (distance to the whole logged path; past the path end it is the
        # Euclidean distance to the end point, while lateral_dist_to_gt_trajectory keeps only the part across the final heading)
        gd = {}
        for tag in ["JTN1", "JTN1S3", "RSG", "garage", "LTF"]:
            q = PW[tag].reindex(items)
            gd[tag] = dict(mean=float(q.gtd.mean()), lateral=float(q.lat_max.mean()), overrun=float((q.gtd - q.lat_max).mean()),
                           mean_easy=float(q.gtd.reindex(easy).mean()), mean_other=float(q.gtd.drop(easy).mean()))
        cj = Cj.set_index("id").cause
        for tag in ["JTN1", "JTN1S3"]:
            q, g_ = PW[tag].reindex(items), PW["garage"].reindex(items)
            cls = pd.Series("garage not 1", index=items)
            cls[easy] = "easy, ours = 1"
            cls[cj.index] = cj.where(~cj.str.startswith("partial"), "partial (any)")
            d = (q.gtd - g_.gtd) / len(items)
            dl = (q.lat_max - g_.lat_max) / len(items)
            gd[f"{tag}-garage by JTN1 outcome"] = {k: dict(n=int((cls == k).sum()), d_gtd=float(d[cls == k].sum()), d_lateral=float(dl[cls == k].sum()))
                                                    for k in cls.unique()}
        J["gt_distance"] = gd
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
               ("all misses", C.id),
               ("union: free-road ratchet incl. its near-1s", C.id[(C["sub"] == "free-road ratchet") & ~C.cause.str.startswith("zero") & ~C.rear]),
               ("union: every moving-window progress partial (not rear-hit)", C.id[~C.cause.str.startswith("zero") & ~C.rear]),
               ("union: lateral zeros (offroad, moving corridor)", C.id[C.cause.isin(["zero: offroad", "zero: corridor, moving window",
                                                                                     "zero: offroad+corridor"])])]
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


def cmd_pai(a):
    """PAI 281 proxy: no strong-model local PAI read exists, so 'easy' = scenes the other two family candidates (RSG-s0, JTN-s1, both
    + ACC + GAP) both score exactly 1; A0 (NTV0-s0 + ACC + GAP) losses by d381 class inside / outside that set. Points only (no PCS pool)."""
    import pandas as pd
    s = pd.read_csv(a.scenes)
    one = lambda x: x >= 1 - 1e-4  # noqa: E731
    s["fam"] = np.where(one(s["RSG-s0"]) & one(s["JTN-s1"]), "easy (RSG and JTN-s1 = 1)",
                        np.where(one(s["RSG-s0"]) | one(s["JTN-s1"]), "one of RSG / JTN-s1 = 1", "neither = 1"))
    s["all3"] = one(s["RSG-s0"]) & one(s["JTN-s1"]) & one(s["NTV0-s0"])
    s["klass"] = np.where(s.cls == "zero corridor", "zero corridor: " + s["sub"].fillna("?"), s.cls)
    t = s[s.loss > 0].groupby(["klass", "fam"]).loss.agg(["size", "sum"]).unstack("fam").fillna(0)
    out = dict(n=len(s), all3=int(s.all3.sum()), fam=s.fam.value_counts().to_dict(),
               loss_by_fam=s.groupby("fam").loss.sum().round(2).to_dict(),
               table=[dict(klass=k, **{f"{f}|{c}": float(t.loc[k, (c, f)]) for c, f in t.columns}) for k in t.index])
    res = REPO / "experiments/alpasim_proxy/results"
    J = json.load(open(res / "easy_items.json")); J["pai"] = out
    json.dump(J, open(res / "easy_items.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------- subject: where a new read's remaining PCS is (Mac)
POOL8 = ["garage", "LTF", "DD", "go-straight", "APY", "YR", "JTN-s1", "JTN-s1+stop3"]   # pcs_local.BOARD_MIX's subjects


class Pricer:
    """Frozen PCS of a response vector and of paired changes, with the log-clustered bootstrap (items fixed) of easy_items' pcs_table."""

    def __init__(self, fz, logs):
        import pandas as pd
        self.fz = fz
        codes, uniq = pd.factorize(logs)
        draw = np.random.default_rng(0).integers(len(uniq), size=(SR_NB, len(uniq)))
        self.W = np.stack([np.bincount(d, minlength=len(uniq)) for d in draw], 1)[codes].astype(float)
        self.one = np.ones(len(logs))

    def L(self, y):
        return self.fz.L(np.asarray(y, float))

    def pcs(self, y):
        return float(self.fz.pcs(self.L(y), self.one))

    def diff(self, y0, y1, L0=None):
        L0 = self.L(y0) if L0 is None else L0
        L1 = self.L(y1)
        d = float(self.fz.pcs(L1, self.one) - self.fz.pcs(L0, self.one))
        lo, hi = np.quantile(self.fz.pcs(L1, self.W) - self.fz.pcs(L0, self.W), [0.025, 0.975])
        return d, float(lo), float(hi)


def classify(y, typ, ids, S, cue, hoff, dec=None, rear=None, natsub=None):
    """cause class of every non-1 item (first match). y / typ: pd.Series over the items; dec: native_gap_report.load's decision table of
    this read (None: take the sub-cause from natsub, a dict id -> sub of the same model's native read, as easy_items does for stop3)."""
    import pandas as pd
    out = []
    for i in ids:
        v, t, stt = float(y[i]), str(typ.get(i, "")), S.loc[i, "stratum"]
        if v >= 1 - 1e-4:
            continue
        sub, r9 = None, np.nan
        if dec is not None:
            g = dec.get(i)
            if g is not None:
                mv = g[g.vh >= 2]
                stop = bool(((g.vh >= 2) & (g.v < 0.3) & (g.v.cummax().shift(1) >= 1)).any())
                lead = (mv.lead_gap < 25).mean() if len(mv) else np.nan
                last = (mv.v / mv.vh).iloc[-1] if len(mv) else np.nan
                r9 = float((mv.v / mv.vh).median()) if len(mv) else np.nan
                sub = ("late launch" if S.loc[i, "v0"] < 1.0 else "stop / lock" if stop else "following a lead" if len(mv) and lead >= 0.5
                       else "free-road ratchet" if len(mv) and last < 0.9 else "other (ratio >= 0.9 or no moving decision)")
        elif natsub is not None:
            sub = natsub.get(i, "no native sub-cause (native = 1)")
        if i in hoff:
            c = "unsolvable: human track offroad (d354)"
        elif v <= 1e-4:
            if t.startswith("AF-front"):
                c = "zero: AF-front collision"
            elif t.startswith("AF"):
                c = "zero: AF-side collision"
            elif t == "corridor" and stt == "static":
                c = "zero: static, red line, no lead" if cue[i] == "red light at the line, no lead" else "zero: static, other cue"
            elif stt == "turn":
                c = "zero: turn window, offroad / corridor"
            elif "offroad" in t:
                c = "zero: offroad, non-turn"
            else:
                c = "zero: corridor, moving non-turn"
        elif t == "rear-hit" or (rear is not None and bool(rear.get(i, False))):
            c = "partial: rear-ended"
        elif v >= 0.98:
            c = "partial: near-miss 0.98-0.999"
        elif stt == "launch" or S.loc[i, "v0"] < 1.0 or t in ("no-launch", "creep"):
            c = "partial: launch late / not launching"
        elif sub == "free-road ratchet":
            c = "partial: free-road ratchet / slow"
        elif sub in ("following a lead", "stop / lock"):
            c = f"partial: {sub}"
        else:
            c = "partial: other slow"
        out.append(dict(id=i, score=v, type=t, stratum=stt, cue=cue[i] if stt == "static" else "", cause=c, sub=sub, ratio_med=r9))
    return pd.DataFrame(out)


def cmd_subject(a):
    import pandas as pd
    import slow_rescore as SR
    import pcs_local as PL
    import native_gap_report as NG
    res = REPO / "experiments/alpasim_proxy/results"
    sc, ty, w = PL.load(res / "pcs_local")
    it = pd.read_csv(res / "pcs_local/items_boardmix.csv.gz", index_col=0)
    items = list(it.index)
    S = pd.read_csv(REPO / "tmp/native_gap/set.csv.gz").drop_duplicates("id").set_index("id")
    Cn = pd.read_parquet(REPO / "tmp/native_gap/set_census.parquet").drop_duplicates("id").set_index("id")
    cue = NG.cue(S.loc[items], Cn)
    o = pd.read_parquet(REPO / "tmp/native_gap/offroad_stop3.parquet")
    hoff = set(o.index[o.h_off_any]) & set(items)
    R = pd.read_csv(a.rows)
    rd = {t: g.drop_duplicates("id").set_index("id") for t, g in R.groupby("tag")}
    yB = rd[a.tag].score.reindex(items); tB = rd[a.tag].type.reindex(items)
    assert yB.notna().all()
    D, _, St, _ = NG.load(Path(a.dec))
    Dv = D[D.valid & (D.k >= 2)].sort_values("k")
    rearB = (St[St.tag == a.tag].groupby("id").collision_rear.max() > 0) if "collision_rear" in St else None
    decB = {i: g for i, g in Dv[Dv.tag == a.tag].groupby("id")}
    decJ = {i: g for i, g in Dv[Dv.tag == a.ref].groupby("id")}
    rearJ = (St[St.tag == a.ref].groupby("id").collision_rear.max() > 0) if "collision_rear" in St else None
    yJ, tJ = sc.loc["JTN-s1", items], ty.loc["JTN-s1", items]
    yJ3, tJ3 = sc.loc["JTN-s1+stop3", items], ty.loc["JTN-s1+stop3", items]
    G = sc.loc["garage", items]
    one = lambda x: x >= 1 - 1e-4  # noqa: E731
    zero = lambda x: x <= 1e-4  # noqa: E731
    logs = S.log.reindex(items).to_numpy()
    PR = {k: Pricer(SR.Frozen(p, sc), logs) for k, p in SR.FITS.items()}
    J = dict(tag=a.tag, n_items=len(items), human_offroad=len(hoff))
    base = {k: p.pcs(yB.to_numpy(float)) for k, p in PR.items()}
    J["base"] = base | dict(mean=float(yB.mean()), zeros=int(zero(yB).sum()), partials=int((~one(yB) & ~zero(yB)).sum()))
    J["ref_pcs"] = {s: {k: p.pcs(sc.loc[s, items].to_numpy(float)) for k, p in PR.items()} for s in ["garage", "JTN-s1", "JTN-s1+stop3"]}
    J["oracle"] = {k: p.pcs(np.ones(len(items))) for k, p in PR.items()}
    L0 = {k: p.L(yB.to_numpy(float)) for k, p in PR.items()}

    def fix(mask, val=1.0, y=yB):
        y1 = y.to_numpy(float).copy(); m = np.asarray(mask, bool)
        y1[m] = val[m] if isinstance(val, np.ndarray) else val
        return {k: p.diff(y.to_numpy(float), y1, L0[k] if y is yB else None) for k, p in PR.items()}

    # 1. strata
    pool1 = one(sc.loc[POOL8, items]).any(axis=0)
    anyread = one(sc.loc[list(sc.index), items]).any(axis=0)
    strict = one(sc.loc[["garage", "LTF", "DD"], items]).all(axis=0)
    hm = pd.Series(np.isin(items, list(hoff)), index=items)
    stratum = pd.Series(np.select([hm, one(G), pool1], ["d: human track offroad", "a: easy (garage = 1)", "b: garage < 1, a pool subject = 1"],
                                  "c: no pool subject = 1"), index=items)
    J["strata"] = []
    for nm, m in [*[(s, stratum == s) for s in sorted(stratum.unique())], ("a-strict: garage, LTF, DD all = 1 (not d)", strict & ~hm),
                  ("c, but some stored read of ours = 1", (stratum == "c: no pool subject = 1") & anyread),
                  ("all items", pd.Series(True, index=items))]:
        r = dict(stratum=nm, items=int(m.sum()), B_zeros=int((m & zero(yB)).sum()), B_partials=int((m & ~one(yB) & ~zero(yB)).sum()),
                 B_lost=float((1 - yB[m]).sum()), garage_lost=float((1 - G[m]).sum()), JTN1_lost=float((1 - yJ[m]).sum()),
                 a_median=float(it.a[m].median()), fix=fix(m))
        J["strata"].append(r)

    # 2. premise: B against garage
    win, loss = yB > G + 1e-4, G > yB + 1e-4
    J["premise"] = dict(win=int(win.sum()), loss=int(loss.sum()), tie=int((~win & ~loss).sum()),
                        win_points=float((yB - G)[win].sum()), loss_points=float((G - yB)[loss].sum()),
                        win_by_cell=w.loc[items].cell[win].value_counts().to_dict(), loss_by_cell=w.loc[items].cell[loss].value_counts().to_dict(),
                        win_by_stratum=stratum[win].value_counts().to_dict(), loss_by_stratum=stratum[loss].value_counts().to_dict(),
                        a_median_win=float(it.a[win].median()), a_median_loss=float(it.a[loss].median()),
                        a_median_easy=float(it.a[one(G)].median()), a_median_other=float(it.a[~one(G)].median()),
                        give_back_wins=fix(win, G.to_numpy(float)), take_garage_on_losses=fix(loss, G.to_numpy(float)),
                        take_garage_on_losses_easy=fix(loss & one(G), G.to_numpy(float)))
    # per-window marginal PCS (board-mix, no CI): a win given back, a loss fixed to 1
    fz = PR["boardmix"].fz; y0 = yB.to_numpy(float); tot0 = L0["boardmix"].sum(1); p0 = base["boardmix"]

    def marg(j, v):
        yy = np.full(len(items), np.nan); yy[j] = v
        c = fz.L(yy)[:, j]
        return float(fz.pcs((tot0 - L0["boardmix"][:, j] + c)[:, None], np.ones(1)) - p0)
    idx = {i: n for n, i in enumerate(items)}
    mw = pd.Series({i: marg(idx[i], float(G[i])) for i in win[win].index})
    ml = pd.Series({i: marg(idx[i], 1.0) for i in yB.index[~one(yB)]})
    J["marginal"] = dict(win_give_back_median=float(mw.median()), win_give_back_by_cell=mw.groupby(w.loc[mw.index].cell).median().round(2).to_dict(),
                         loss_to1_median_by_stratum=ml.groupby(stratum[ml.index]).median().round(2).to_dict(),
                         loss_to1_sum=float(ml.sum()))

    # 3. causes (B, JTN-s1 native, JTN-s1 + stop3)
    CB = classify(yB, tB, items, S, cue, hoff, decB, rearB)
    CJ = classify(yJ, tJ, items, S, cue, hoff, decJ, rearJ)
    CJ3 = classify(yJ3, tJ3, items, S, cue, hoff, None, None, CJ.set_index("id")["sub"].dropna().to_dict())
    CB["strat"] = stratum.reindex(CB.id).values
    CB["garage"] = G.reindex(CB.id).values
    CB["JTN1"] = yJ.reindex(CB.id).values; CB["JTN1S3"] = yJ3.reindex(CB.id).values
    CB["a"] = it.a.reindex(CB.id).values
    CB.to_csv(res / "easy_items/misses_STANDFMX.csv.gz", index=False, float_format="%.4f")
    classes = list(CB.cause.value_counts().index)
    T = []
    for c in classes + sorted(set(CJ.cause) - set(classes)):
        r = dict(cause=c)
        for nm, C, y in [("B", CB, yB), ("JTN1", CJ, yJ), ("JTN1S3", CJ3, yJ3)]:
            ids = C.id[C.cause == c]
            m = np.isin(items, ids)
            r[nm] = dict(windows=int(m.sum()), zeros=int((m & zero(y)).sum()), partials=int((m & ~one(y) & ~zero(y)).sum()),
                         points=float((1 - y[m]).sum()))
            if m.any():
                r[nm]["fix"] = {k: p.diff(y.to_numpy(float), np.where(m, 1.0, y.to_numpy(float)), L0[k] if nm == "B" else None) for k, p in PR.items()}
        sel = CB[CB.cause == c]
        r["B_by_stratum"] = sel.strat.value_counts().to_dict(); r["B_garage1"] = int(one(sel.garage).sum())
        r["B_ratio_med"] = float(sel.ratio_med.median()) if len(sel) else np.nan
        r["B_cells"] = w.loc[sel.id].cell.value_counts().to_dict() if len(sel) else {}
        T.append(r)
    J["causes"] = T
    # oracle minus one class (only this class left unfixed): the non-additivity
    J["leave_one"] = {}
    for c in classes:
        m = np.isin(items, CB.id[CB.cause == c])
        y1 = np.where(m, y0, 1.0)
        J["leave_one"][c] = {k: p.pcs(y1) for k, p in PR.items()}
    # 4. greedy curve over B's classes (each fit separately)
    J["greedy"] = {}
    for k, p in PR.items():
        cur, left, path = y0.copy(), list(classes), [dict(step=0, cause="as read", pcs=base[k], mean=float(y0.mean()))]
        while left:
            best = None
            for c in left:
                m = np.isin(items, CB.id[CB.cause == c]); y1 = np.where(m, 1.0, cur); g = p.pcs(y1)
                n = int(m.sum()); score = (g - path[-1]["pcs"]) / n
                if best is None or score > best[0]:
                    best = (score, c, y1, g, n)
            _, c, cur, g, n = best
            left.remove(c)
            path.append(dict(step=len(path), cause=c, windows=n, pcs=g, per_window=(g - path[-1]["pcs"]) / n, mean=float(cur.mean())))
        J["greedy"][k] = path
    # window-level curve: fix B's misses one at a time in order of item discrimination a (highest first), and in order of a within easy
    J["by_a"] = {}
    for k, p in PR.items():
        L1 = p.L(np.ones(len(items))); Lc = L0[k].copy()
        order = CB.assign(j=CB.id.map(idx)).sort_values("a", ascending=False)
        pts = []
        for n, (j, c, a_, s_) in enumerate(zip(order.j, order.cause, order.a, order.strat), 1):
            Lc[:, j] = L1[:, j]
            pts.append(dict(n=n, a=float(a_), cause=c, stratum=s_, pcs=float(p.fz.pcs(Lc, p.one))))
        J["by_a"][k] = pts
    J["theta"] = {k: float((v - PR[k].fz.A[1]) / PR[k].fz.A[0]) for k, v in base.items()}
    J["theta_cap_pcs"] = {k: float(np.polyval(p.fz.A, 8.0)) for k, p in PR.items()}
    # each class fixed on its easy windows only
    J["class_easy_only"] = {c: fix(np.isin(items, CB.id[(CB.cause == c) & (CB.strat == "a: easy (garage = 1)")]))
                            for c in classes if ((CB.cause == c) & (CB.strat == "a: easy (garage = 1)")).any()}
    # each class split by item stratum, every class x stratum cell fixed alone (the figure's bar segments)
    J["class_stratum"] = {c: {st: dict(windows=int(m.sum()), fix=fix(m)) for st in sorted(CB.strat.unique())
                              for m in [np.isin(items, CB.id[(CB.cause == c) & (CB.strat == st)])] if m.any()} for c in classes}
    # B with garage's score on every easy item (garage = 1 there), B elsewhere; and max(B, garage) everywhere
    J["garage_on_easy"] = fix(one(G)); J["max_B_garage"] = fix(loss, G.to_numpy(float))
    J["zeros_to_1"] = fix(zero(yB)); J["partials_to_1"] = fix(~one(yB) & ~zero(yB))
    # the same strata / premise for JTN-s1 native (what B already took)
    J["JTN1_by_stratum"] = {s: dict(zeros=int(((stratum == s) & zero(yJ)).sum()), partials=int(((stratum == s) & ~one(yJ) & ~zero(yJ)).sum()),
                                    lost=float((1 - yJ[stratum == s]).sum())) for s in sorted(stratum.unique())}
    winJ, lossJ = yJ > G + 1e-4, G > yJ + 1e-4
    J["JTN1_premise"] = dict(win=int(winJ.sum()), loss=int(lossJ.sum()), win_points=float((yJ - G)[winJ].sum()), loss_points=float((G - yJ)[lossJ].sum()))
    out = res / "easy_items_B.json"
    json.dump(J, open(out, "w"), indent=1, default=float)
    print(json.dumps(J, indent=1, default=float))


BOARD1 = 1899      # private board #1 PCS, nuPlan, 2026-10-19 (results/leaderboard_nuplan_20261019.json)


def cmd_fig(a):
    """results/figs/easy_items_B.{png,pdf} from results/easy_items_B.json: the greedy class curve and each class's PCS fixed alone (1600 px wide
    diagnostic figure, Okabe-Ito colours of research/plot_style.py; not the CVPR column format)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = REPO / "experiments/alpasim_proxy/results"
    J = json.load(open(res / "easy_items_B.json"))
    STC = {"a: easy (garage = 1)": ("#0072B2", "easy (garage = 1)"), "b: garage < 1, a pool subject = 1": ("#D55E00", "hard, a pool subject solves"),
           "c: no pool subject = 1": ("#009E73", "nobody in the pool solves"), "d: human track offroad": ("#E69F00", "human log fails (d354)")}
    short = lambda c: c.replace("unsolvable: ", "").replace(" (d354)", "").replace("partial: ", "P: ").replace("zero: ", "0: ")  # noqa: E731
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 8.4), dpi=100, gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.patch.set_facecolor("#fcfcfb")
    path = J["greedy"]["boardmix"]
    xs = np.cumsum([0] + [p["windows"] for p in path[1:]])
    ys = [p["pcs"] for p in path]
    ax.plot(xs, ys, color="#0072B2", lw=2, marker="o", ms=8, label="board-mix, greedy by PCS per window")
    g = J["greedy"]["g15"]
    ax.plot(np.cumsum([0] + [p["windows"] for p in g[1:]]), [p["pcs"] for p in g], color="#CC79A7", lw=2, ls="--", marker="s", ms=7,
            label="g15 (garage weight 15), its own greedy order")
    lines = []
    for k, (x, y, p) in enumerate(zip(xs[1:], ys[1:], path[1:]), 1):
        if p["pcs"] - path[k - 1]["pcs"] > 0.5:
            ax.annotate(str(k), (x, y), xytext=(-12, 4), textcoords="offset points", fontsize=10, fontweight="bold", color="#0072B2")
            lines.append(f"{k}. {short(p['cause'])} ({p['windows']} w) -> {p['pcs']:.0f}")
    lines.append(f"{len(lines) + 1}-{len(path) - 1}. the other {len(path) - 1 - len(lines)} classes: +0 (already at the cap)")
    fig.text(0.015, 0.01, "Left, board-mix greedy steps (windows) -> PCS:  " + ";  ".join(lines[:4]) + "\n" + ";  ".join(lines[4:]), fontsize=10,
             color="#0b0b0b", va="bottom")
    refs = [(J["ref_pcs"]["JTN-s1"]["boardmix"], "JTN-s1 native"), (J["ref_pcs"]["JTN-s1+stop3"]["boardmix"], "JTN-s1 + stop3"),
            (1553, "garage 0014 (pool fit; MAP 1525)"), (BOARD1, "board #1, PRIVATE board (not local)"),
            (J["theta_cap_pcs"]["boardmix"], "oracle (all 1) = the fit's grid cap, theta 8:")]
    for v, nm in refs:
        ax.axhline(v, color="#52514e" if "PRIVATE" not in nm else "#000000", lw=1, ls=":" if "PRIVATE" not in nm else "-.", zorder=0)
        ax.text(2 if "cap" in nm else xs[-1], v - 30 if "cap" in nm else v + 8, f"{nm} {v:.0f}", ha="left" if "cap" in nm else "right", fontsize=10, color="#52514e" if "PRIVATE" not in nm else "#000000")
    ax.set_xlabel("B's lost windows set to 1 (cumulative, by cause class)")
    ax.set_ylabel("frozen-pool PCS (board-mapped)")
    ax.text(0, 1.02, "(a) cumulative PCS as B's cause classes are fixed greedily", transform=ax.transAxes, fontsize=12)
    ax.set_ylim(1200, 2320); ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9), frameon=False, fontsize=10)
    # right: each class fixed alone, segments = the class's windows in one stratum fixed alone
    T = sorted(J["causes"], key=lambda r: -r["B"].get("fix", {}).get("boardmix", [0])[0])
    T = [r for r in T if r["B"]["windows"]]
    cs = J["class_stratum"]
    yy = np.arange(len(T))[::-1]
    for y, r in zip(yy, T):
        left = 0.0
        for st, (col, _) in STC.items():
            c = cs.get(r["cause"], {}).get(st)
            if c is None:
                continue
            v = max(c["fix"]["boardmix"][0], 0)
            bx.barh(y, v, left=left, color=col, height=0.62, edgecolor="#fcfcfb", lw=2)
            left += v
        tot, lo, hi = r["B"]["fix"]["boardmix"]
        bx.errorbar(tot, y, xerr=[[tot - lo], [hi - tot]], fmt="D", color="#0b0b0b", ms=6, capsize=3, lw=1)
        bx.text(max(hi, left) + 2, y, f"{tot:+.0f}  ({r['B']['windows']} w, g15 {r['B']['fix']['g15'][0]:+.0f})", va="center", fontsize=9.5,
                color="#52514e")
    bx.set_yticks(yy, [short(r["cause"]) for r in T], fontsize=10.5)
    bx.set_xlabel("PCS gain, class alone set to 1 (board-mix); diamond + bar = whole class [95 % CI]")
    bx.text(0, 1.02, "(b) each cause class fixed alone, split by item stratum", transform=bx.transAxes, fontsize=12)
    bx.set_xlim(0, 175); bx.grid(axis="x", color="#e6e5e1", lw=0.8)
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(color=c, label=l) for c, l in STC.values()], loc="lower right", frameon=False, fontsize=10,
              title="segment = that stratum's windows fixed alone", title_fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    (res / "figs").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(res / f"figs/easy_items_B.{ext}", facecolor=fig.get_facecolor())
    print(res / "figs/easy_items_B.png")


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
    q = sp.add_parser("pai")
    q.add_argument("--scenes", default=str(REPO / "tmp/easy_items/pai/scenes.csv"))
    u = sp.add_parser("subject")
    u.add_argument("--rows", default=str(REPO / "experiments/defect_fix/results/stand_b/arena_windows_full.csv.gz"))
    u.add_argument("--tag", default="STANDFMX-L18-s0")
    u.add_argument("--ref", default="JTN-L18-s1", help="the native reference's tag in --dec (its sub-causes also label its stop3 read)")
    u.add_argument("--dec", default=str(REPO / "tmp/easy_items_B"), help="native_gap extract dir of the read (box: stand_b/ext_full)")
    sp.add_parser("fig")
    a = ap.parse_args()
    {"extract": cmd_extract, "report": cmd_report, "pai": cmd_pai, "subject": cmd_subject, "fig": cmd_fig}[a.cmd](a)
