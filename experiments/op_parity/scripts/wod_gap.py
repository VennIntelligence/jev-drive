"""Where WP2 loses RFS on WOD-E2E val relative to the top-rated rater trajectory and the log, and what is recoverable offline
(results/wod_gap.md; plan plans/2026-10-07-wod-gap-prereg.md). Offline on stored val predictions; no training, no serving.

  python experiments/op_parity/scripts/wod_gap.py            (jevdrive env, CPU, < 1 min)

A  gap map: per-frame RFS parts (horizon, floor, mode), type vs the top-rated trajectory (ahead / behind / lateral / both), O1 / O2 swaps
   (pp_wod_diag.retime), strata, the same map for the log, the ranked failure table, BEV figures
B  seed ensemble (trajectory mean of WP2 s0 / s1; that mean averaged with shipped)
C  arc-length scaling of WP2's plan, global and stratified, chosen by sequence-level 5-fold x 20 repeats (out-of-fold next to in-sample)

Tables -> experiments/op_parity/results/wod_gap/, figures -> experiments/op_parity/figs/wod_gap/.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B, GRID = 4000, np.concatenate([[0.0], np.round(np.arange(0.50, 1.501, 0.02), 2)])
FOLDS, REPEATS = 5, 20
TYPES = ("ahead", "behind", "lateral", "both")
OUT, FIG = _R / "experiments/op_parity/results/wod_gap", _R / "experiments/op_parity/figs/wod_gap"


# ---------------------------------------------------------------- RFS parts
def parts(pred, traj, sc, v0):
    """Per-frame decomposition of jevdrive.waymo.rater_feedback_score (same constants and operand order) for one trajectory per frame."""
    from jevdrive import waymo as W
    pred, traj, sc = (np.asarray(a, np.float64) for a in (pred, traj, sc))
    lng, lat = W._rater_frames(traj)
    v = pred[:, None] - traj                                                      # (n, 3, 20, 2)
    k = np.array(W.RFS_HORIZONS) * W.RFS_FREQ - 1
    e_lon, e_lat = (lng * v).sum(-1)[..., k], (lat * v).sum(-1)[..., k]           # signed, (n, 3, 2): + = further along / left of the rater
    scale = np.clip(0.5 + 0.5 * (np.asarray(v0, np.float64) - 1.4) / (11 - 1.4), 0.5, 1.0)[:, None]
    base = np.array(W.RFS_BASE_THRESHOLDS)
    lat_thr, lng_thr = (scale * (base * m) for m in W.RFS_MULTIPLIERS)            # (n, 2)
    norm = np.maximum(np.abs(e_lon) / lng_thr[:, None], np.abs(e_lat) / lat_thr[:, None])
    inside_r = (norm <= 1.0).all(-1)                                              # (n, 3)
    inside = inside_r.any(1)
    sh = (sc[..., None] * W.RFS_DECAY ** np.maximum(norm - 1.0, 0.0)).max(1)      # (n, 2) per-horizon component
    raw = sh.mean(-1)
    score = np.where(inside, raw, np.maximum(raw, W.RFS_FLOOR))
    return dict(score=score, raw=raw, sh=sh, inside=inside, inside_r=inside_r, e_lon=e_lon, e_lat=e_lat, lng_thr=lng_thr, lat_thr=lat_thr)


def typ(p, top):
    """Type of a trajectory against the top-rated rater trajectory, at the horizon with the larger normalised miss."""
    i = np.arange(len(top))
    nl, nt = np.abs(p["e_lon"][i, top]) / p["lng_thr"], np.abs(p["e_lat"][i, top]) / p["lat_thr"]
    h = np.maximum(nl, nt).argmax(1)
    lo, la, fwd = nl[i, h] > 1, nt[i, h] > 1, p["e_lon"][i, top, h] > 0
    return np.where(~(lo | la), "inside", np.where(lo & la, "both", np.where(la, "lateral", np.where(fwd, "ahead", "behind"))))


def arc(p):
    return np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros_like(p[:, :1]), p], 1), axis=1), axis=-1), 1)


# ---------------------------------------------------------------- main
def main(a):
    from jevdrive import stats
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.data import splits
    from jevdrive.run import Run
    from pp_wod import load_preds
    from pp_wod_diag import retime
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "wod-gap", seed=0, config=dict(B=B, folds=FOLDS, repeats=REPEATS, grid=GRID.tolist())) as run:
        val = splits.load("wod/val")
        run.use_split(val)
        S = Z.load_sets()
        r, x = S["rater"], S["extra"]
        n = len(r["name"])
        assert set(r["sequence"].astype(str)) == set(val.members) and len(set(r["sequence"])) == n, "rater frames != wod/val sequences"
        names = np.concatenate([r["name"], x["name"]]).astype(str)
        seq = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
        fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
        traj, sc, v0 = r["traj"].astype(np.float64), r["scores"].astype(np.float64), W.init_speed(r["past"]).astype(np.float64)
        cl = r["cluster"].astype(str)
        ccode, cu = pd.factorize(pd.Series(cl))
        nc = len(cu)
        M = np.eye(nc)[ccode]                                                     # (n, nc) one-hot
        w = 1.0 / (np.bincount(ccode)[ccode] * nc)                                # frame weights: sum(w * f) = cluster-mean of f
        ii = np.arange(n)
        top = sc.argmax(1)
        order = np.argsort(-sc, axis=1, kind="stable")
        PA = {t: load_preds(t, names) for t in ("shipped", "WP2-full-s0", "WP2-full-s1", "WP1-full-s0", "WP1-full-s1")}
        T = {"shipped": [PA["shipped"][:n]], "WP2": [PA["WP2-full-s0"][:n], PA["WP2-full-s1"][:n]], "WP1": [PA["WP1-full-s0"][:n], PA["WP1-full-s1"][:n]],
             "log": [fut[:n]], "top": [traj[ii, top]], "second": [traj[ii, order[:, 1]]], "worst": [traj[ii, order[:, 2]]]}

        def rfs(p):
            return np.asarray(W.rater_feedback_score(p, traj, sc, v0), float)

        def cm(f, rows=None):                                                     # leaderboard aggregation on a subset
            m = M if rows is None else M * np.asarray(rows, float)[:, None]
            cnt = m.sum(0)
            return float(((m * f[:, None]).sum(0)[cnt > 0] / cnt[cnt > 0]).mean())
        K = np.stack([np.bincount(d, minlength=n) for d in np.random.default_rng(0).integers(n, size=(B, n))]).astype(float)  # one frame per sequence

        def ci(d, rows=None):
            """Cluster-mean of the per-frame difference d on `rows`: point, lo, hi (paired bootstrap over sequences)."""
            m = M if rows is None else M * np.asarray(rows, float)[:, None]
            cnt, sm = K @ m, K @ (m * d[:, None])
            with np.errstate(invalid="ignore", divide="ignore"):
                b = np.nanmean(np.where(cnt > 0, sm / cnt, np.nan), 1)
            return (cm(d, rows), *np.nanpercentile(b, [2.5, 97.5]))

        def f3(t):
            return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        Pp = {k: [parts(p, traj, sc, v0) for p in v] for k, v in T.items()}
        for k, v in T.items():                                                    # the decomposition reproduces the port
            for p, q in zip(v, Pp[k]):
                assert np.abs(q["score"] - rfs(p)).max() < 1e-9, k
        sco = {k: np.mean([q["score"] for q in v], 0) for k, v in Pp.items()}
        assert np.abs(sco["top"] - sc.max(1)).max() < 1e-9
        run.info("reproduce: shipped %.3f WP2 %.3f WP1 %.3f log %.3f top %.3f | WP2 - shipped %s", *(cm(sco[k]) for k in ("shipped", "WP2", "WP1", "log", "top")),
                 f3(ci(sco["WP2"] - sco["shipped"])))
        self_err = float(np.abs(retime(T["shipped"][0], T["shipped"][0]) - T["shipped"][0]).max())

        # ---- strata
        lead = np.array([float(np.asarray(np.load(Z.root("preds", "op_cinque") / f"{nm}.npz")["lead_prob"]).reshape(-1)[0]) for nm in names[:n]])
        logd, intent = np.linalg.norm(fut[:n, -1], axis=-1), r["intent"]
        lum = pd.read_csv(_R / "experiments/leaderboard_audit/results/night_gap/seq_lum.csv").set_index("sequence").l.reindex(seq[:n]).to_numpy()
        vbin = np.where(v0 < 0.5, "stopped", np.where(v0 < 5, "slow", np.where(v0 < 12, "mid", "fast")))
        st = {"all": np.ones(n, bool), "stopped v<0.5": v0 < 0.5, "stopped, log stays (<1 m @5s)": (v0 < 0.5) & (logd < 1),
              "stopped, log moves (>=1 m @5s)": (v0 < 0.5) & (logd >= 1), "launch v<2 & log5s>5m": (v0 < 2) & (logd > 5),
              "slow 0.5-5": vbin == "slow", "mid 5-12": vbin == "mid", "fast >=12": vbin == "fast", "turn intent L/R": intent >= 2,
              "straight intent": intent == 1, "lead_prob>0.5 (shipped)": lead > 0.5, "lead_prob<=0.5": lead <= 0.5,
              "night (luma < 50)": lum < 50, "day (luma >= 120)": lum >= 120} | {"cluster " + c: cl == c for c in cu}

        # ---- swaps: O1 = A's path at R's arc length, O2 = R's path at A's arc length
        def swap(A, R):
            o1 = [rfs(retime(pa, T[R][0])) for pa in T[A]]
            o2 = [rfs(retime(T[R][0], pa)) for pa in T[A]]
            return o1, o2
        SW = {(A, R): swap(A, R) for A, R in (("WP2", "top"), ("WP2", "log"), ("shipped", "top"), ("log", "top"), ("WP2", "shipped"), ("WP1", "top"))}
        rows = []
        for (A, R), (o1, o2) in SW.items():
            g = cm(sco[R] - sco[A])
            m1, m2 = np.mean(o1, 0), np.mean(o2, 0)
            c1, c2 = ci(m1 - sco[A]), ci(m2 - sco[A])
            lab = lambda s, c: "carries" if s >= 0.5 and c[1] > 0 else "part" if s >= 0.25 and c[1] > 0 else "not"  # noqa: E731
            rows.append({"A": A, "R": R, "RFS A": cm(sco[A]), "RFS R": cm(sco[R]), "gap": g, "RFS O1 (A path, R speed)": cm(m1), "O1 - A": c1[0], "O1_lo": c1[1],
                         "O1_hi": c1[2], "lon share": c1[0] / g, "speed": lab(c1[0] / g, c1), "RFS O2 (R path, A speed)": cm(m2), "O2 - A": c2[0], "O2_lo": c2[1],
                         "O2_hi": c2[2], "lat share": c2[0] / g, "path": lab(c2[0] / g, c2),
                         "O1 - A per seed": "/".join(f"{cm(o - q['score']):+.3f}" for o, q in zip(o1, Pp[A])),
                         "O2 - A per seed": "/".join(f"{cm(o - q['score']):+.3f}" for o, q in zip(o2, Pp[A]))})
        stats.write_table(rows, OUT / "swaps")
        run.info("swaps:\n%s", pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- per-arm map against the top-rated trajectory (units = (frame, seed), weight w / K)
        def units(A):
            k = len(T[A])
            o1, o2 = SW[(A, "top")] if (A, "top") in SW else ([np.full(n, np.nan)] * k,) * 2
            u = []
            for s, (p, q) in enumerate(zip(T[A], Pp[A])):
                gap = sco["top"] - q["score"]
                h = np.maximum(np.abs(q["e_lon"][ii, top]) / q["lng_thr"], np.abs(q["e_lat"][ii, top]) / q["lat_thr"]).argmax(1)
                r1, r2 = (o1[s] - q["score"]) >= gap / 2, (o2[s] - q["score"]) >= gap / 2
                u.append(pd.DataFrame({
                    "frame": ii, "seed": s, "w": w / k, "gap": gap, "score": q["score"], "type": typ(q, top), "ctx": np.where(v0 < 0.5, "stopped", "moving"),
                    "vbin": vbin, "lead": lead > 0.5, "floored": q["score"] <= W.RFS_FLOOR + 1e-9, "floor_binds": ~q["inside"] & (q["raw"] < W.RFS_FLOOR),
                    "mode": np.where(q["inside_r"][ii, top], "in top", np.where(q["inside"], "in lower-rated", "outside all")),
                    "loss3": (sco["top"] - q["sh"][:, 0]) / 2, "loss5": (sco["top"] - q["sh"][:, 1]) / 2, "floor_credit": q["score"] - q["raw"],
                    "e_lon3": q["e_lon"][ii, top, 0], "e_lon5": q["e_lon"][ii, top, 1], "e_lat3": q["e_lat"][ii, top, 0], "e_lat5": q["e_lat"][ii, top, 1],
                    "worst_h": np.where(h == 0, "3s", "5s"),
                    "label": np.where(gap < 0.5, "small", np.where(r1 & ~r2, "speed", np.where(r2 & ~r1, "path", np.where(r1 & r2, "either", "joint")))),
                    "d5": np.linalg.norm(p[:, -1], axis=-1), "d1": arc(p)[:, 3], "len5": arc(p)[:, -1]}))
            return pd.concat(u, ignore_index=True)
        U = {A: units(A) for A in ("WP2", "shipped", "log", "WP1")}
        ws = lambda d, c: float((d.w * d[c]).sum())  # noqa: E731
        arows = []
        for A in ("WP2", "shipped", "WP1", "log", "second", "worst"):
            row = {"arm": A, "RFS": cm(sco[A]), "RFS frame mean": float(sco[A].mean()), "gap to top": cm(sco["top"] - sco[A])}
            c = ci(sco[A] - sco["log"])
            row |= {"d vs log": c[0], "d_log_lo": c[1], "d_log_hi": c[2]}
            c = ci(sco[A] - sco["shipped"])
            row |= {"d vs shipped": c[0], "d_shipped_lo": c[1], "d_shipped_hi": c[2]}
            if A in U:
                u = U[A]
                k = len(T[A])
                row |= {"frames gap>0": float((u.gap > 1e-9).sum() / k), "floored frac": float(u.floored.mean()), "floored pts": ws(u[u.floored], "gap"),
                        "floored share of gap": ws(u[u.floored], "gap") / ws(u, "gap"), "floor binds frac": float(u.floor_binds.mean()),
                        "outside all frac": float((u["mode"] == "outside all").mean()), "outside all pts": ws(u[u["mode"] == "outside all"], "gap"),
                        "in lower-rated frac": float((u["mode"] == "in lower-rated").mean()), "in lower-rated pts": ws(u[u["mode"] == "in lower-rated"], "gap"),
                        "in top frac": float((u["mode"] == "in top").mean()), "loss 3s pts": ws(u, "loss3"), "loss 5s pts": ws(u, "loss5"),
                        "floor credit pts": ws(u, "floor_credit"), "worst horizon 5s frac (gap>0)": float((u[u.gap > 1e-9].worst_h == "5s").mean())}
                for t in ("inside",) + TYPES:
                    row[f"type {t} frac"] = float((u.type == t).mean())
                    row[f"type {t} pts"] = ws(u[u.type == t], "gap")
            arows.append(row)
        stats.write_table(arows, OUT / "arms")
        run.info("arms:\n%s", pd.DataFrame(arows).T.to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- ranked failure table per arm: type x context
        def ranked(A, by=("type", "ctx")):
            u = U[A][U[A].gap > 1e-9]
            k, tot = len(T[A]), ws(U[A], "gap")
            out = []
            for key, d in u.groupby(list(by)):
                f, big = d.frame.to_numpy(), d[d.label != "small"]
                row = dict(zip(by, key)) | {"frames": len(d) / k, "points lost": ws(d, "gap"), "share of gap": ws(d, "gap") / tot,
                                             "mean gap": float(d.gap.mean()), "floored frac": float(d.floored.mean()),
                                             "outside all frac": float((d["mode"] == "outside all").mean()),
                                             "frames gap>=0.5": len(big) / k}
                for lb in ("speed", "path", "either", "joint"):
                    row[f"label {lb} pts"] = ws(big[big.label == lb], "gap")
                row |= {"median e_lon 3s (m)": float(d.e_lon3.median()), "median e_lon 5s (m)": float(d.e_lon5.median()),
                        "median |e_lat| 5s (m)": float(d.e_lat5.abs().median()), "worst horizon 5s frac": float((d.worst_h == "5s").mean())}
                for O in ("WP2", "shipped", "log"):                              # the other arms on the same (frame, seed-weight) units
                    if O == A:
                        continue
                    row[f"{O} pts on same frames"] = float((d.w * (sco["top"] - sco[O])[f]).sum())
                    t0 = U[O][U[O].seed == 0].set_index("frame")
                    row[f"{O} same type frac"] = float(((t0.type.reindex(f).to_numpy() == d.type.to_numpy()) & (t0.gap.reindex(f).to_numpy() > 1e-9)).mean())
                row[f"{A} - log pts"] = float((d.w * (d.score.to_numpy() - sco["log"][f])).sum())
                out.append(row)
            return pd.DataFrame(out).sort_values("points lost", ascending=False).reset_index(drop=True)
        RK = {A: ranked(A) for A in ("WP2", "shipped", "log")}
        for A, d in RK.items():
            stats.write_table(d, OUT / f"ranked_{A}")
            run.info("ranked %s (total gap %.3f):\n%s", A, ws(U[A], "gap"), d.to_string(float_format=lambda v: f"{v:.3f}"))
        stats.write_table(ranked("WP2", ("type", "vbin", "lead")), OUT / "ranked_WP2_fine")
        stats.write_table(ranked("log", ("type", "vbin", "lead")), OUT / "ranked_log_fine")

        # ---- strata table
        srows = []
        (o1w, o2w), (o1l, o2l) = SW[("WP2", "top")], SW[("log", "top")]
        for nm, msk in st.items():
            msk = np.asarray(msk, bool)
            if msk.sum() < 8:
                continue
            uw, ul, us = (U[A][msk[U[A].frame.to_numpy()]] for A in ("WP2", "log", "shipped"))
            row = {"stratum": nm, "n": int(msk.sum())} | {f"RFS {A}": cm(sco[A], msk) for A in ("top", "log", "WP2", "shipped")}
            row |= {"WP2 pts lost": ws(uw, "gap"), "WP2 share of gap": ws(uw, "gap") / ws(U["WP2"], "gap"), "log pts lost": ws(ul, "gap"),
                    "shipped pts lost": ws(us, "gap")}
            for nm2, d in (("WP2 - log", sco["WP2"] - sco["log"]), ("WP2 - shipped", sco["WP2"] - sco["shipped"]), ("log - top", sco["log"] - sco["top"]),
                           ("WP2 - top", sco["WP2"] - sco["top"])):
                c = ci(d, msk)
                row |= {nm2: c[0], nm2 + " lo": c[1], nm2 + " hi": c[2]}
            row |= {"WP2 floored frac": float(uw.floored.mean()), "log floored frac": float(ul.floored.mean()), "shipped floored frac": float(us.floored.mean()),
                    "WP2 O1 - WP2 (top speed)": cm(np.mean(o1w, 0) - sco["WP2"], msk), "WP2 O2 - WP2 (top path)": cm(np.mean(o2w, 0) - sco["WP2"], msk),
                    "log O1 - log (top speed)": cm(o1l[0] - sco["log"], msk), "log O2 - log (top path)": cm(o2l[0] - sco["log"], msk),
                    "WP2 median e_lon 5s (m)": float(uw.e_lon5.median()), "log median e_lon 5s (m)": float(ul.e_lon5.median()),
                    "WP2 mean e_lon 5s (m)": float(uw.e_lon5.mean()), "log mean e_lon 5s (m)": float(ul.e_lon5.mean())}
            for t in TYPES:
                row[f"WP2 {t} pts"] = ws(uw[uw.type == t], "gap")
            srows.append(row)
        sdf = stats.write_table(srows, OUT / "strata")
        pd.set_option("display.width", 320, "display.max_columns", 60, "display.max_rows", 200)
        run.info("strata:\n%s", sdf.T.to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- what the raters prefer where the log is outside the top-rated region
        ul = U["log"].set_index("frame")
        tp, lg = T["top"][0], T["log"][0]
        at, al = arc(tp), arc(lg)
        near = np.linalg.norm(traj - lg[:, None], axis=-1).mean(-1).argmin(1)        # rater trajectory closest to the log (post hoc, descriptive)
        pref = pd.DataFrame({"type": ul.type, "ctx": ul.ctx, "vbin": ul.vbin, "lead": ul.lead, "w": ul.w, "gap": ul.gap, "v0": v0, "e_lon3": ul.e_lon3,
                             "e_lon5": ul.e_lon5, "abs_lat3": ul.e_lat3.abs(), "abs_lat5": ul.e_lat5.abs(), "len5_top": at[:, -1], "len5_log": al[:, -1],
                             "len1_top": at[:, 3], "len1_log": al[:, 3], "len3_top": at[:, 11], "len3_log": al[:, 11],
                             "acc_top": 2 * (at[:, -1] - v0 * 5) / 25, "acc_log": 2 * (al[:, -1] - v0 * 5) / 25,
                             "top_score": sc.max(1), "near_is_top": near == top, "near_score": sc[ii, near],
                             "near_ade": np.linalg.norm(traj[ii, near] - lg, axis=-1).mean(-1), "cluster": cl, "floored": ul.floored})
        prows = []
        for key, d in [(("all frames", ""), pref), (("log outside top region", ""), pref[pref.type != "inside"]), (("log gap > 0", ""), pref[pref.gap > 1e-9])] + \
                [((t, c), pref[(pref.type == t) & (pref.ctx == c) & (pref.gap > 1e-9)]) for t in TYPES for c in ("stopped", "moving")] + \
                [((t, f"{vb}{' lead' if ld else ''}"), pref[(pref.type == t) & (pref.vbin == vb) & (pref.lead == ld) & (pref.gap > 1e-9)])
                 for t in ("ahead", "behind") for vb in ("slow", "mid", "fast") for ld in (True, False)]:
            if len(d) < 5:
                continue
            mv = d.len5_log > 2
            prows.append({"log type vs top": key[0], "context": key[1], "frames": len(d), "points lost": float((d.w * d.gap).sum()), "mean gap": float(d.gap.mean()),
                          "floored frac": float(d.floored.mean()), "mean v0 (m/s)": float(d.v0.mean()), "lead frac": float(d.lead.mean()),
                          "median e_lon 3s (m, log - top)": float(d.e_lon3.median()), "median e_lon 5s (m, log - top)": float(d.e_lon5.median()),
                          "median |e_lat| 3s (m)": float(d.abs_lat3.median()), "median |e_lat| 5s (m)": float(d.abs_lat5.median()),
                          "median len5 top / log": float((d.len5_top / d.len5_log)[mv].median()) if mv.sum() >= 3 else np.nan,
                          "median len1 top / log": float((d.len1_top / d.len1_log)[d.len1_log > 0.5].median()) if (d.len1_log > 0.5).sum() >= 3 else np.nan,
                          "mean len5 log (m)": float(d.len5_log.mean()), "mean len5 top (m)": float(d.len5_top.mean()),
                          "mean accel top - log (m/s2)": float((d.acc_top - d.acc_log).mean()), "mean accel log (m/s2)": float(d.acc_log.mean()),
                          "mean accel top (m/s2)": float(d.acc_top.mean()),
                          "log-nearest rater is top frac": float(d.near_is_top.mean()), "log-nearest rater score": float(d.near_score.mean()),
                          "top score": float(d.top_score.mean())})
        stats.write_table(prows, OUT / "rater_pref")
        run.info("rater preference:\n%s", pd.DataFrame(prows).T.to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- per-frame table
        u0, u1 = (U["WP2"][U["WP2"].seed == s].set_index("frame") for s in (0, 1))
        us = U["shipped"].set_index("frame")
        fr = pd.DataFrame({"name": names[:n], "cluster": cl, "v0": v0, "intent": intent, "lead_prob": lead, "luma": lum, "log_d5": logd, "w": w,
                           "top_score": sc.max(1), "scores": [" ".join(f"{s:.0f}" for s in q) for q in sc],
                           "rfs_WP2": sco["WP2"], "rfs_WP2_s0": Pp["WP2"][0]["score"], "rfs_WP2_s1": Pp["WP2"][1]["score"], "rfs_shipped": sco["shipped"],
                           "rfs_WP1": sco["WP1"], "rfs_log": sco["log"], "type_WP2_s0": u0.type, "type_WP2_s1": u1.type, "type_shipped": us.type,
                           "type_log": ul.type, "label_WP2_s0": u0.label, "label_log": ul.label, "label_shipped": us.label,
                           "e_lon5_WP2_s0": u0.e_lon5, "e_lat5_WP2_s0": u0.e_lat5, "e_lon5_log": ul.e_lon5, "e_lat5_log": ul.e_lat5,
                           "e_lon5_shipped": us.e_lon5, "e_lat5_shipped": us.e_lat5,
                           "rfs_WP2_top_speed": np.mean(o1w, 0), "rfs_WP2_top_path": np.mean(o2w, 0), "log_nearest_rater_score": sc[ii, near]})
        fr.to_csv(OUT / "frames.csv", index=False, float_format="%.4f")

        # ---- B. ensembles
        ens2 = (PA["WP2-full-s0"] + PA["WP2-full-s1"]) / 2
        ens3 = (ens2 + PA["shipped"]) / 2
        topx = T["top"][0]
        err = {k: np.linalg.norm(p - fut, axis=-1) for k, p in PA.items()} | {"ENS2": np.linalg.norm(ens2 - fut, axis=-1), "ENS3": np.linalg.norm(ens3 - fut, axis=-1)}
        ade = {"WP2": {m: np.mean([f(err[f"WP2-full-s{s}"]) for s in (0, 1)], 0) for m, f in (("ade3", lambda e: e[:, :12].mean(1)), ("ade5", lambda e: e.mean(1)))}}
        for k in ("shipped", "ENS2", "ENS3"):
            ade[k] = {"ade3": err[k][:, :12].mean(1), "ade5": err[k].mean(1)}
        adr = {"WP2": np.mean([np.linalg.norm(p - topx, axis=-1).mean(1) for p in T["WP2"]], 0), "shipped": np.linalg.norm(T["shipped"][0] - topx, axis=-1).mean(1),
               "ENS2": np.linalg.norm(ens2[:n] - topx, axis=-1).mean(1), "ENS3": np.linalg.norm(ens3[:n] - topx, axis=-1).mean(1)}
        sco["ENS2"], sco["ENS3"] = rfs(ens2[:n]), rfs(ens3[:n])
        brows = []
        for k in ("WP2", "shipped", "ENS2", "ENS3"):
            row = {"arm": k, "RFS": cm(sco[k]), "ADE@3s log": float(ade[k]["ade3"].mean()), "ADE@5s log": float(ade[k]["ade5"].mean()), "ADE@5s top-rated": float(adr[k].mean())}
            for ref in ("WP2", "shipped"):
                if ref == k:
                    continue
                c = ci(sco[k] - sco[ref])
                row |= {f"d RFS vs {ref}": c[0], f"d RFS vs {ref} lo": c[1], f"d RFS vs {ref} hi": c[2]}
                for m in ("ade3", "ade5"):
                    q = stats.paired(ade[k][m], ade[ref][m], groups=seq, n_boot=B)
                    row |= {f"d {m} vs {ref}": q["mean"], f"d {m} vs {ref} lo": q["lo"], f"d {m} vs {ref} hi": q["hi"]}
            brows.append(row)
        stats.write_table(brows, OUT / "ensemble")
        run.info("ensemble:\n%s", pd.DataFrame(brows).T.to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- C. arc-length scaling, out-of-fold
        def at_arc(path, sq):
            dummy = np.zeros_like(path)                                           # a profile whose arc length is sq: points on the x axis
            dummy[..., 0] = sq
            return retime(path, dummy)
        assert np.abs(at_arc(T["WP2"][0], arc(T["WP2"][0])) - T["WP2"][0]).max() < 1e-9
        SK = np.stack([np.mean([rfs(at_arc(p, k * arc(p))) for p in T["WP2"]], 0) for k in GRID])       # (nk, n) seed-mean RFS at each k
        i1 = int(np.flatnonzero(GRID == 1.0)[0])
        assert np.abs(SK[i1] - sco["WP2"]).max() < 1e-9
        SKs = np.stack([rfs(at_arc(T["shipped"][0], k * arc(T["shipped"][0]))) for k in GRID])
        SKl = np.stack([rfs(at_arc(T["log"][0], k * arc(T["log"][0]))) for k in GRID])
        pd.DataFrame({"k": GRID, "RFS WP2": [cm(s) for s in SK], "RFS shipped": [cm(s) for s in SKs], "RFS log": [cm(s) for s in SKl]}
                     | {f"WP2 {vb}": [cm(s, vbin == vb) for s in SK] for vb in ("stopped", "slow", "mid", "fast")}
                     | {f"log {vb}": [cm(s, vbin == vb) for s in SKl] for vb in ("stopped", "slow", "mid", "fast")}).to_csv(OUT / "scale_curve.csv", index=False, float_format="%.4f")
        cells = {"C1 global": np.zeros(n, int), "C2a speed bin": pd.factorize(vbin)[0], "C2b speed bin x lead": pd.factorize(pd.Series(vbin) + np.where(lead > 0.5, "+lead", ""))[0]}

        def fit(SKm, cell, rows):                                                 # per cell: argmax of the weighted sum on rows, ties to the k nearest 1
            ks = np.full(cell.max() + 1, i1)
            for c in range(cell.max() + 1):
                m = rows & (cell == c)
                if m.any():
                    tot = (SKm[:, m] * w[m]).sum(1)
                    best = np.flatnonzero(tot >= tot.max() - 1e-12)
                    ks[c] = best[np.abs(GRID[best] - 1).argmin()]
            return ks

        def oof(SKm, cell):
            acc, picks = np.zeros(n), []
            for rep in range(REPEATS):
                fold = np.random.default_rng(rep).permutation(n) % FOLDS
                for f in range(FOLDS):
                    ks = fit(SKm, cell, fold != f)
                    te = fold == f
                    acc[te] += SKm[ks[cell[te]], np.flatnonzero(te)]
                    picks.append(GRID[ks])
            return acc / REPEATS, np.array(picks)
        crows = []
        for arm, SKm, base in (("WP2", SK, sco["WP2"]), ("shipped", SKs, sco["shipped"]), ("log", SKl, sco["log"])):
            for nm, cell in cells.items():
                ks = fit(SKm, cell, np.ones(n, bool))
                ins = SKm[ks[cell], ii]
                oo, picks = oof(SKm, cell)
                cname = (["all"] if nm == "C1 global" else list(pd.factorize(vbin)[1]) if nm == "C2a speed bin"
                         else list(pd.factorize(pd.Series(vbin) + np.where(lead > 0.5, "+lead", ""))[1]))
                ci_in, ci_oo = ci(ins - base), ci(oo - base)
                row = {"arm": arm, "variant": nm, "RFS base": cm(base), "k in-sample": ", ".join(f"{c}={GRID[k]:.2f}" for c, k in zip(cname, ks)),
                       "RFS in-sample": cm(ins), "d in-sample": ci_in[0], "d_in_lo": ci_in[1], "d_in_hi": ci_in[2], "RFS out-of-fold": cm(oo),
                       "d out-of-fold": ci_oo[0], "d_oof_lo": ci_oo[1], "d_oof_hi": ci_oo[2], "optimism": ci_in[0] - ci_oo[0],
                       "k over folds (median [min, max])": ", ".join(f"{c}={np.median(picks[:, j]):.2f} [{picks[:, j].min():.2f}, {picks[:, j].max():.2f}]"
                                                                     for j, c in enumerate(cname)),
                       "recoverable": bool(ci_oo[1] > 0)}
                if arm == "WP2":
                    c = ci(oo - sco["shipped"])
                    row |= {"oof vs shipped": c[0], "oof_vs_shipped_lo": c[1], "oof_vs_shipped_hi": c[2]}
                    sco[f"{nm} oof"] = oo
                crows.append(row)
        stats.write_table(crows, OUT / "scale")
        run.info("scale:\n%s", pd.DataFrame(crows).T.to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- figures: per top type, the frame with the largest loss and the one at the type's median loss (seed-mean gap, both seeds same type)
        picks = []
        gapw = sco["top"] - sco["WP2"]
        agree = (u0.type == u1.type).to_numpy()
        for _, rk in RK["WP2"].head(5).iterrows():
            m = np.flatnonzero(agree & (u0.type.to_numpy() == rk["type"]) & (u0.ctx.to_numpy() == rk["ctx"]) & (gapw > 1e-9))
            if len(m) < 2:
                continue
            o = m[np.argsort(-(w * gapw)[m], kind="stable")]
            for tag, f in (("max", o[0]), ("median", o[len(o) // 2])):
                if f not in [p[0] for p in picks]:
                    picks.append((int(f), rk["type"], rk["ctx"], tag))
        figs = []
        for j, (f, t, c, tag) in enumerate(picks[:10]):
            fn = f"{j:02d}_{t}_{c}_{tag}_{names[f][:8]}.png"
            bev(FIG / fn, f, names[f], cl[f], v0[f], sc[f], traj[f], T, Pp, sco, t, c, tag, lead[f])
            figs.append({"file": fn, "frame": names[f], "type": t, "ctx": c, "pick": tag, "cluster": cl[f], "v0": v0[f], "lead_prob": lead[f],
                         "scores": " ".join(f"{s:.0f}" for s in sc[f]), "rfs_WP2": sco["WP2"][f], "rfs_shipped": sco["shipped"][f], "rfs_log": sco["log"][f],
                         "e_lon5_WP2_s0": u0.e_lon5[f], "e_lat5_WP2_s0": u0.e_lat5[f], "e_lon5_log": ul.e_lon5[f], "e_lat5_log": ul.e_lat5[f],
                         "w_gap": (w * gapw)[f]})
        stats.write_table(figs, OUT / "figures")
        run.info("figures:\n%s", pd.DataFrame(figs).to_string(float_format=lambda v: f"{v:.3f}"))
        meta = dict(n=n, B=B, clusters=dict(zip(cu, np.bincount(ccode).tolist())), split=val.id, retime_self_err=self_err, grid=GRID.tolist(), folds=FOLDS,
                    repeats=REPEATS, reproduce={k: cm(sco[k]) for k in ("shipped", "WP2", "WP1", "log", "top", "second", "worst")},
                    wp2_minus_shipped=ci(sco["WP2"] - sco["shipped"]), per_seed={f"WP2-s{s}": cm(Pp["WP2"][s]["score"]) for s in (0, 1)},
                    strata_n={k: int(np.asarray(v).sum()) for k, v in st.items()})
        (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        run.summary.update(rfs_wp2=meta["reproduce"]["WP2"], gap_top=cm(sco["top"] - sco["WP2"]), out=str(OUT))


def bev(path, f, name, cluster, v0, sc, traj, T, Pp, sco, t, c, tag, lead):
    """One frame, ego at the origin heading up: rater trajectories with scores, the top-rated trust regions at 3 s / 5 s, log, shipped, WP2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
    col = {"WP2": "#2a78d6", "shipped": "#eb6834", "log": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    q, top = Pp["WP2"][0], int(np.argmax(sc))
    from jevdrive import waymo as W
    lng, lat = W._rater_frames(traj[None])
    for h, k in enumerate((11, 19)):                                              # trust region of the top-rated trajectory
        ctr, a, b = traj[top, k], lng[0, top, k] * q["lng_thr"][f, h], lat[0, top, k] * q["lat_thr"][f, h]
        box = np.array([ctr + a + b, ctr + a - b, ctr - a - b, ctr - a + b])
        ax.add_patch(Polygon(np.c_[-box[:, 1], box[:, 0]], closed=True, fc="#f0efec", ec=MUTED, lw=0.8, zorder=1))
    xy = lambda p: (-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]])  # noqa: E731  (left is left, forward is up)
    for j in np.argsort(sc):
        ax.plot(*xy(traj[j]), color=INK if j == top else MUTED, lw=2 if j == top else 1.2, ls="-" if j == top else (0, (4, 2)), zorder=2,
                label=f"rater, score {sc[j]:.0f}" + (" (top)" if j == top else ""))
        ax.plot(-traj[j, [11, 19], 1], traj[j, [11, 19], 0], "o", ms=5, color=INK if j == top else MUTED, mec=SURF, mew=1, zorder=3)
    for nm, ps, ls in (("log", T["log"], ["-"]), ("shipped", T["shipped"], ["-"]), ("WP2", T["WP2"], ["-", (0, (1, 1))])):
        for s, p in enumerate(ps):
            ax.plot(*xy(p[f]), color=col[nm], lw=2, ls=ls[s], zorder=4,
                    label=f"{nm}{' s' + str(s) if len(ps) > 1 else ''}, RFS {Pp[nm][s]['score'][f]:.2f}")
            ax.plot(-p[f, [11, 19], 1], p[f, [11, 19], 0], "o", ms=6, color=col[nm], mec=SURF, mew=1, zorder=5)
    ax.plot(0, 0, "^", ms=9, color=INK, zorder=6)
    ally = np.concatenate([traj.reshape(-1, 2)] + [p[f] for k in ("log", "shipped", "WP2") for p in T[k]])
    half = max(4.0, np.abs(ally[:, 1]).max() * 1.25 + 1.5)
    ax.set_xlim(-half, half)
    ax.set_ylim(min(-1.0, ally[:, 0].min() - 1), max(6.0, ally[:, 0].max() * 1.08 + 1))
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("lateral (m, left is left; scale differs from the forward axis)", color="#52514e", fontsize=8)
    ax.set_ylabel("forward (m)", color="#52514e", fontsize=8)
    ax.set_title(f"{name}  |  {cluster}  |  v0 {v0:.1f} m/s  |  lead_prob {lead:.2f}\nWP2 type: {t}, {c} ({tag} loss of the type); dots = 3 s and 5 s; "
                 f"grey boxes = top-rated trust region", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor="#52514e")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main(argparse.ArgumentParser().parse_args())
