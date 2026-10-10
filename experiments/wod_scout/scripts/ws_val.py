"""WODSCOUT Q2 / Q3 / Q4c on the 479 WOD-E2E val rater frames (plans/2026-10-10-wodscout-prereg.md). Offline on stored WLG predictions;
no training, no serving, nothing submitted. Rater trajectories are analysis-only labels.

  python experiments/wod_scout/scripts/ws_val.py            (jevdrive env, CPU, box, about 1 min)

Q2  non-longitudinal budget of WLG against the top-rated trajectory (swaps of decision 164: O1 = own path at top-rated arc length, O2 = top-rated
    path at own arc length), by cluster and situation; the 30 frames with the largest path-only points, typed by the pre-registered path rubric, with sheets
Q3  strata (night, Pedestrian, the clusters that lose on test): points, speed / path label, WLG - shipped, ADE; val cluster scores against the test ones
Q4c pairwise agreement of unfitted, map-free kinematic quantities with the rater ordering of the 3 rated trajectories per frame

Tables -> experiments/wod_scout/results/{q2_path_budget,q3_strata,q4_label_supply}/, figures -> experiments/wod_scout/figs/.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_P = _R / "experiments/op_parity/scripts"
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_P), str(_R / "research")]
import argparse, io, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B, NSHEET = 4000, 30
RES, FIG = _R / "experiments/wod_scout/results", _R / "experiments/wod_scout/figs"
TAGS = ("WLG-full-s0", "WLG-full-s1")
TEST = {"Construction": 8.725, "Single-Lane Maneuvers": 8.377, "Cut_ins": 8.353, "Special Vehicles": 8.289, "Foreign Object Debris": 8.265,
        "Multi-Lane Maneuvers": 8.079, "Interections": 8.073, "Cyclist": 8.007, "Pedestrian": 7.957, "Others": 7.616}   # decision 180 (official page), Spotlight 7.342
LEADER_GAP = {"Others": -0.369, "Pedestrian": -0.270, "Single-Lane Maneuvers": -0.241, "Interections": -0.222, "Special Vehicles": -0.155,
              "Multi-Lane Maneuvers": 0.244, "Cut_ins": 0.116, "Cyclist": 0.040}                                           # decision 180 point 5
G0 = dict(rfs=8.187, gap=1.400, o1=0.657, o2=0.290)                               # decisions 169 / 218 amendment


def arc(p):
    return np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros_like(p[:, :1]), p], 1), axis=1), axis=-1), 1)


def heading_change(p):
    """Heading (deg) of the last moving 1 m of each trajectory relative to +x (the heading at t = 0); 0 for a trajectory shorter than 1 m."""
    out = np.zeros(len(p))
    for i, q in enumerate(p):
        q = np.vstack([[0, 0], q])
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        if s[-1] < 1.0:
            continue
        j = np.searchsorted(s, s[-1] - 1.0, side="right") - 1
        d = q[-1] - q[j]
        out[i] = np.degrees(np.arctan2(d[1], d[0]))
    return out


def kin(tr):
    """Map-free kinematics of (n, 20, 2) trajectories at 4 Hz: 5 s arc length, min longitudinal accel, max |lateral accel|, max |jerk|."""
    p = np.concatenate([np.zeros_like(tr[:, :1]), tr], 1)
    v = np.diff(p, axis=1) * 4.0                                                  # (n, 20, 2)
    sp = np.linalg.norm(v, axis=-1)
    a = np.diff(v, axis=1) * 4.0                                                  # (n, 19, 2)
    u = v[:, :-1] / np.maximum(sp[:, :-1, None], 1e-3)
    alon = (a * u).sum(-1)
    alat = a[..., 1] * u[..., 0] - a[..., 0] * u[..., 1]
    jerk = np.linalg.norm(np.diff(a, axis=1) * 4.0, axis=-1)
    return dict(arc5=sp.sum(1) / 4.0, min_alon=alon.min(1), max_alat=np.abs(alat).max(1), max_jerk=jerk.max(1))


def main(a):
    from jevdrive import stats
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.data import splits
    from jevdrive.run import Run
    from pp_wod import load_preds
    from pp_wod_diag import retime
    O2, O3, O4 = RES / "q2_path_budget", RES / "q3_strata", RES / "q4_label_supply"
    for d in (O2, O3, O4, FIG):
        d.mkdir(parents=True, exist_ok=True)
    with Run("wod_scout", "val", seed=0, config=dict(B=B, tags=TAGS, nsheet=NSHEET)) as run:
        val = splits.load("wod/val")
        run.use_split(val)
        S = Z.load_sets()
        r, x = S["rater"], S["extra"]
        n = len(r["name"])
        assert set(r["sequence"].astype(str)) <= set(val.members)
        scode, su = pd.factorize(pd.Series(r["sequence"].astype(str)))
        ns = len(su)
        names = np.concatenate([r["name"], x["name"]]).astype(str)
        seq = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
        fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
        traj, sc, v0 = r["traj"].astype(np.float64), r["scores"].astype(np.float64), W.init_speed(r["past"]).astype(np.float64)
        cl, intent = r["cluster"].astype(str), r["intent"]
        ccode, cu = pd.factorize(pd.Series(cl))
        nc = len(cu)
        M = np.eye(nc)[ccode]
        w = 1.0 / (np.bincount(ccode)[ccode] * nc)
        ii = np.arange(n)
        top = sc.argmax(1)
        PA = {t: load_preds(t, names) for t in ("shipped",) + TAGS}
        own = [PA[t][:n] for t in TAGS]
        ens = np.mean(own, 0)                                                     # the submitted trajectory (mean of the two seeds)
        tp, lg, sh = traj[ii, top], fut[:n], PA["shipped"][:n]

        def rfs(p):
            return np.asarray(W.rater_feedback_score(p, traj, sc, v0), float)

        def cm(f, rows=None):
            m = M if rows is None else M * np.asarray(rows, float)[:, None]
            cnt = m.sum(0)
            return float(((m * f[:, None]).sum(0)[cnt > 0] / cnt[cnt > 0]).mean())
        K = np.stack([np.bincount(d, minlength=ns) for d in np.random.default_rng(0).integers(ns, size=(B, ns))]).astype(float)[:, scode]

        def ci(d, rows=None):
            """Cluster-mean of d on `rows` (stratum RFS difference): point, lo, hi; paired bootstrap over sequences."""
            m = M if rows is None else M * np.asarray(rows, float)[:, None]
            cnt, sm = K @ m, K @ (m * d[:, None])
            with np.errstate(invalid="ignore", divide="ignore"):
                b = np.nanmean(np.where(cnt > 0, sm / cnt, np.nan), 1)
            return (cm(d, rows), *np.nanpercentile(b, [2.5, 97.5]))

        def pts(d, rows):
            """Points of the all-frame cluster mean carried by `rows` (additive over disjoint strata), with CI."""
            return ci(d * np.asarray(rows, float))

        def f3(t):
            return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        s_own = [rfs(p) for p in own]
        A = np.mean(s_own, 0)
        s_top, s_log, s_sh, s_ens = sc.max(1), rfs(lg), rfs(sh), rfs(ens)
        o1 = np.mean([rfs(retime(p, tp)) - s for p, s in zip(own, s_own)], 0)     # speed-only gain per frame
        o2 = np.mean([rfs(retime(tp, p)) - s for p, s in zip(own, s_own)], 0)     # path-only gain per frame
        gap = s_top - A
        joint, nonlon = gap - o1 - o2, gap - o1
        got = dict(rfs=cm(A), gap=cm(gap), o1=cm(o1), o2=cm(o2))
        run.info("G0 reproduce: %s (want %s); shipped %.3f log %.3f top %.3f ens %.3f", got, G0, cm(s_sh), cm(s_log), cm(s_top), cm(s_ens))
        for k, tol in (("rfs", 0.002), ("gap", 0.005), ("o1", 0.005), ("o2", 0.005)):
            assert abs(got[k] - G0[k]) <= tol, f"G0 fails on {k}: {got[k]:.4f} vs {G0[k]}"

        # ---- strata
        lum = pd.read_csv(_R / "experiments/leaderboard_audit/results/night_gap/seq_lum.csv").set_index("sequence").l
        lum_r, lum_all = lum.reindex(seq[:n]).to_numpy(), lum.reindex(seq).to_numpy()
        turn = intent >= 2
        C = {c: cl == c for c in cu}
        situ = {"cut-in (Cut_ins)": C["Cut_ins"],
                "obstacle bypass (FOD + Construction + Special Vehicles)": C["Foreign Object Debris"] | C["Construction"] | C["Special Vehicles"],
                "lane change (Multi-Lane Maneuvers)": C["Multi-Lane Maneuvers"], "intersection turn (Interections, intent L/R)": C["Interections"] & turn,
                "intersection straight (Interections, intent straight)": C["Interections"] & ~turn, "pedestrians (Pedestrian)": C["Pedestrian"],
                "cyclists (Cyclist)": C["Cyclist"], "single lane (Single-Lane Maneuvers)": C["Single-Lane Maneuvers"], "others (Others)": C["Others"]}
        assert np.stack(list(situ.values())).sum(0).max() == 1 and np.stack(list(situ.values())).sum(0).min() == 1      # a partition of the 479 frames
        cross = {"night (luma < 50)": lum_r < 50, "day (luma >= 120)": lum_r >= 120, "turn intent L/R": turn, "straight intent": ~turn,
                 "stopped v0 < 0.5": v0 < 0.5, "moving v0 >= 0.5": v0 >= 0.5}

        def row(name, m, kind):
            g, s1, s2 = pts(gap, m), pts(o1, m), pts(o2, m)
            d1, d2 = ci(o1, m), ci(o2, m)
            lab = ("speed" if s1[1] > 0 and s1[0] >= 2 * s2[0] else "path" if s2[1] > 0 and s2[0] >= 2 * s1[0] else
                   "both" if s1[1] > 0 and s2[1] > 0 else "undetermined")
            nl, jt, dsh = pts(nonlon, m), pts(joint, m), ci(A - s_sh, m)
            return {"stratum": name, "kind": kind, "n": int(m.sum()), "RFS top": cm(s_top, m), "RFS log": cm(s_log, m), "RFS WLG": cm(A, m), "RFS shipped": cm(s_sh, m),
                    "gap pts": g[0], "gap_lo": g[1], "gap_hi": g[2], "share of gap": g[0] / cm(gap),
                    "speed-only pts (O1)": s1[0], "o1_lo": s1[1], "o1_hi": s1[2], "path-only pts (O2)": s2[0], "o2_lo": s2[1], "o2_hi": s2[2],
                    "joint pts": jt[0], "joint_lo": jt[1], "joint_hi": jt[2], "non-lon upper pts (gap - O1)": nl[0], "nl_lo": nl[1], "nl_hi": nl[2],
                    "path-only share of stratum gap": s2[0] / g[0] if g[0] > 1e-9 else np.nan, "O1 dRFS": d1[0], "O1d_lo": d1[1], "O1d_hi": d1[2],
                    "O2 dRFS": d2[0], "O2d_lo": d2[1], "O2d_hi": d2[2], "label": lab, "WLG - shipped": dsh[0], "dsh_lo": dsh[1], "dsh_hi": dsh[2]}
        rows = [row("all", np.ones(n, bool), "all")] + [row(k, m, "situation") for k, m in situ.items()] + [row(k, m, "cross-cut") for k, m in cross.items()] + \
               [row("cluster " + c, m, "cluster") for c, m in C.items()]
        T2 = pd.DataFrame(rows)
        stats.write_table(rows, O2 / "budget")
        run.info("budget:\n%s", T2[["stratum", "n", "RFS WLG", "gap pts", "speed-only pts (O1)", "path-only pts (O2)", "joint pts", "non-lon upper pts (gap - O1)",
                                    "label"]].to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- Q2: the frames with the largest path-only points, typed by the path rubric (geometry on the submitted mean trajectory)
        wp = w * o2
        order = np.lexsort((names[:n], -wp))
        pick = order[:NSHEET]
        o2traj = retime(tp, ens)                                                  # top-rated path at own arc length
        d = o2traj[:, -1] - ens[:, -1]
        hd_own, hd_top = heading_change(ens), heading_change(tp)
        uh = np.stack([np.cos(np.radians(hd_own)), np.sin(np.radians(hd_own))], -1)
        slat = d[:, 0] * -uh[:, 1] + d[:, 1] * uh[:, 0]                           # top-rated minus own at 5 s, along own left normal
        lat5 = np.abs(slat)                                                       # lateral separation at 5 s in the own heading frame
        own_inside = -slat * np.sign(hd_own)                                      # > 0: own ends further towards its own turn side than the top-rated path
        tstep = np.linalg.norm(np.diff(np.concatenate([np.zeros((n, 1, 2)), tp], 1), axis=1), axis=-1) * W.RFS_FREQ
        nmov = (tstep > 1e-6).cumsum(1).argmax(1)
        padded = (nmov < 19) & (tstep[ii, nmov] > 1.0)
        arc_own, arc_top = arc(ens)[:, -1], arc(tp)[:, -1]
        dh = np.abs((hd_own - hd_top + 180) % 360 - 180)
        to, tt = np.abs(hd_own) >= 25, np.abs(hd_top) >= 25
        top_straight = np.abs(tp[:, :, 1]).max(1) <= 0.5
        ptype = np.where(padded | (arc_top < 2.0), "P6 label artefact",
                 np.where(to & tt, "P3 turn geometry", np.where((to ^ tt) & (dh >= 25), "P4 different route",
                  np.where(top_straight & (np.abs(ens[:, -1, 1]) >= 0.5), "P5 own drifts, top straight",
                   np.where((dh < 15) & (lat5 >= 2.5), "P1 other lane", np.where((dh < 15) & (lat5 >= 0.5), "P2 in-lane offset", "P7 other"))))))
        fr = pd.DataFrame({"rank": np.argsort(order) + 1, "name": names[:n], "cluster": cl, "v0": v0, "intent": intent, "luma": lum_r, "w": w, "rfs_WLG": A, "rfs_top": s_top,
                           "rfs_log": s_log, "rfs_shipped": s_sh, "gap": gap, "o1_gain": o1, "o2_gain": o2, "joint": joint, "w_gap": w * gap, "w_o1": w * o1, "w_o2": wp,
                           "arc_own": arc_own, "arc_top": arc_top, "head_own": hd_own, "head_top": hd_top, "lat5": lat5, "own_inside_m": own_inside, "top_padded": padded, "auto_type": ptype,
                           "scores": [" ".join(f"{s:.0f}" for s in q) for q in sc]})
        fr.to_csv(O2 / "frames.csv", index=False)
        tot = float(wp.sum())
        conc = [{"top k": k, "path-only pts": float(wp[order[:k]].sum()), "share of all path-only pts": float(wp[order[:k]].sum() / tot)} for k in (10, 30, 50, 100)]
        conc.append({"top k": "frames with O2 gain > 0", "path-only pts": float((o2 > 1e-9).sum()), "share of all path-only pts": float((o2 > 1e-9).mean())})
        stats.write_table(conc, O2 / "concentration")
        typ = []
        for scope, mask in (("top 30", np.isin(ii, pick)), ("all frames with O2 gain >= 0.5", o2 >= 0.5), ("all frames", np.ones(n, bool))):
            for t in sorted(set(ptype)):
                m = mask & (ptype == t)
                typ.append({"scope": scope, "auto type": t, "frames": int(m.sum()), "path-only pts": float(wp[m].sum()), "share of scope": float(wp[m].sum() / max(wp[mask].sum(), 1e-12)),
                            "gap pts": float((w * gap)[m].sum()), "median lat5 (m)": float(np.median(lat5[m])) if m.any() else np.nan})
        stats.write_table(typ, O2 / "path_types_auto")
        man = O2 / "top30_manual.csv"                                              # written by hand after looking at the sheets (rank, manual_type, note)
        if man.exists():
            mt = pd.read_csv(man).set_index("rank")
            fr["manual_type"] = fr["rank"].map(mt.manual_type)
            fr["note"] = fr["rank"].map(mt.note)
            t30 = fr.loc[pick]
            mrows = [{"manual type": k, "frames": len(g), "path-only pts": float(g.w_o2.sum()), "share of the top 30": float(g.w_o2.sum() / t30.w_o2.sum()),
                      "auto type agrees": int((g.auto_type.str[:2] == k[:2]).sum()), "turn intent": int((g.intent >= 2).sum()), "night (luma < 50)": int((g.luma < 50).sum()),
                      "v0 < 5 m/s": int((g.v0 < 5).sum()), "WLG at the 4.0 floor": int((g.rfs_WLG <= 4.0 + 1e-9).sum()),
                      "own on the inside (m, median; turn types)": float(g.own_inside_m.median())} for k, g in t30.groupby("manual_type")]
            stats.write_table(mrows, O2 / "path_types_manual")
            run.info("manual types:\n%s", pd.DataFrame(mrows).to_string(float_format=lambda v: f"{v:.3f}"))
        stats.write_table(fr.loc[pick].drop(columns=["w"]).to_dict("records"), O2 / "top30")
        run.info("top30:\n%s", fr.loc[pick, ["rank", "name", "cluster", "v0", "rfs_WLG", "rfs_top", "o1_gain", "o2_gain", "lat5", "head_own", "head_top", "auto_type"]]
                 .to_string(float_format=lambda v: f"{v:.2f}"))
        if not a.no_sheets:
            sheets(pick, names, cl, v0, sc, traj, own, lg, sh, o2traj, A, s_top, o2, ptype, fr)

        # ---- Q3: night and the val -> test clusters
        gf = gap                                                                  # frame-level quantities for the night contrast (frame means, not cluster means)
        vb = np.digitize(v0, [0.5, 5, 12])
        Kf = np.stack([np.bincount(d_, minlength=ns) for d_ in np.random.default_rng(1).integers(ns, size=(B, ns))]).astype(float)[:, scode]

        def night_diff(q, kw):
            ng, dy = lum_r < 50, lum_r >= 120
            def one(k):
                if kw == "unmatched":
                    return (k * ng * q).sum() / (k * ng).sum() - (k * dy * q).sum() / (k * dy).sum()
                num = den = 0.0
                for b_ in range(4):
                    mn, md = k * ng * (vb == b_), k * dy * (vb == b_)
                    if mn.sum() > 0 and md.sum() > 0:
                        num += mn.sum() * ((mn * q).sum() / mn.sum() - (md * q).sum() / md.sum())
                        den += mn.sum()
                return num / den
            bs = np.array([one(k) for k in Kf])
            return (one(np.ones(n)), *np.percentile(bs, [2.5, 97.5]))
        nrows = []
        for nm, q in (("gap to top-rated (RFS, frame mean)", gf), ("speed-only gain O1", o1), ("path-only gain O2", o2), ("WLG RFS", A), ("shipped RFS", s_sh),
                      ("log RFS", s_log), ("WLG - shipped", A - s_sh)):
            for kw in ("unmatched", "v0-bin matched"):
                t_ = night_diff(q, kw)
                nrows.append({"quantity": nm, "contrast": f"night - day, {kw}", "diff": t_[0], "lo": t_[1], "hi": t_[2]})
        # ADE@3s against the log on rater + extra frames, night - day, by-sequence bootstrap
        sa, sua = pd.factorize(pd.Series(seq))
        ade = {k: np.linalg.norm(p[:, :12] - fut[:, :12], axis=-1).mean(1) for k, p in (("WLG", np.mean([PA[t] for t in TAGS], 0)), ("shipped", PA["shipped"]))}
        ng, dy = lum_all < 50, lum_all >= 120
        rng = np.random.default_rng(2)
        cnt_n, cnt_d = np.bincount(sa, ng.astype(float), len(sua)), np.bincount(sa, dy.astype(float), len(sua))
        for k, e in ade.items():
            sn, sd = np.bincount(sa, e * ng, len(sua)), np.bincount(sa, e * dy, len(sua))
            bs = []
            for _ in range(B):
                j = rng.integers(len(sua), size=len(sua))
                bs.append(sn[j].sum() / cnt_n[j].sum() - sd[j].sum() / cnt_d[j].sum())
            nrows.append({"quantity": f"ADE@3s vs log, {k} (m; {int(ng.sum())} night / {int(dy.sum())} day frames)", "contrast": "night - day, unmatched",
                          "diff": float(e[ng].mean() - e[dy].mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                          "night": float(e[ng].mean()), "day": float(e[dy].mean())})
        stats.write_table(nrows, O3 / "night")
        run.info("night:\n%s", pd.DataFrame(nrows).to_string(float_format=lambda v: f"{v:.3f}"))
        vt = []
        for c in cu:
            m = C[c]
            t_ = ci(s_ens, m)
            te = TEST[c]
            vt.append({"cluster": c, "n val": int(m.sum()), "val RFS (submitted mean trajectory)": t_[0], "lo": t_[1], "hi": t_[2], "test RFS": te, "test - val": te - t_[0],
                       "test inside val CI": bool(t_[1] <= te <= t_[2]), "test: WLG - leader": LEADER_GAP.get(c, np.nan)})
        t_ = ci(s_ens)
        vt.append({"cluster": "mean of the 10", "n val": n, "val RFS (submitted mean trajectory)": t_[0], "lo": t_[1], "hi": t_[2], "test RFS": float(np.mean(list(TEST.values()))),
                   "test - val": float(np.mean(list(TEST.values()))) - t_[0], "test inside val CI": bool(t_[1] <= np.mean(list(TEST.values())) <= t_[2])})
        stats.write_table(vt, O3 / "val_vs_test")
        run.info("val vs test:\n%s", pd.DataFrame(vt).to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- Q4c: unfitted kinematic quantities against the rater ordering (pairs of rated trajectories within a frame with different scores)
        flat = traj.reshape(-1, 20, 2)
        kq = kin(flat)
        kq["ade_log"] = np.linalg.norm(flat - np.repeat(lg, 3, 0), axis=-1).mean(1)
        sign = {"arc5": +1, "min_alon": +1, "max_alat": -1, "max_jerk": -1, "ade_log": -1}   # +1: the higher-scored trajectory is expected to have the larger value
        desc = {"arc5": "5 s arc length (progress): higher-scored goes further", "min_alon": "min longitudinal accel: higher-scored brakes less hard",
                "max_alat": "max |lateral accel|: higher-scored is gentler", "max_jerk": "max |jerk|: higher-scored is smoother", "ade_log": "ADE to the logged future: higher-scored is closer to the log"}
        pi, pj = [], []
        for f in range(n):
            for i in range(3):
                for j in range(3):
                    if sc[f, i] > sc[f, j] and sc[f, j] >= 0:
                        pi.append(3 * f + i), pj.append(3 * f + j)
        pi, pj = np.array(pi), np.array(pj)
        pf = pi // 3
        sub = {"all pairs": np.ones(len(pi), bool), "frames stopped v0 < 0.5": v0[pf] < 0.5, "frames moving": v0[pf] >= 0.5, "pairs with the top-rated": sc[pf, pi % 3] == s_top[pf],
               "score difference >= 2": (sc.reshape(-1)[pi] - sc.reshape(-1)[pj]) >= 2}
        crow = []
        rngc = np.random.default_rng(3)
        draws = rngc.integers(ns, size=(B, ns))
        for k, sg in sign.items():
            dv = sg * (kq[k][pi] - kq[k][pj])
            val_ = np.where(np.abs(dv) < 1e-9, 0.5, (dv > 0).astype(float))
            for sn_, m in sub.items():
                num, den = np.bincount(scode[pf[m]], val_[m], ns), np.bincount(scode[pf[m]], None, ns).astype(float)
                bs = num[draws].sum(1) / den[draws].sum(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                pt = float(val_[m].mean())
                crow.append({"quantity": desc[k], "pairs": sn_, "n pairs": int(m.sum()), "concordance": pt, "lo": lo, "hi": hi,
                             "reading": "same direction" if lo >= 0.70 else "weak" if lo > 0.5 else "opposite" if hi < 0.5 else "unrelated"})
        stats.write_table(crow, O4 / "kinematic_concordance")
        run.info("concordance:\n%s", pd.DataFrame(crow).to_string(float_format=lambda v: f"{v:.3f}"))
        figure(T2)
        meta = dict(n=n, n_sequences=ns, B=B, split=val.id, reproduce=got | dict(shipped=cm(s_sh), log=cm(s_log), top=cm(s_top), ens=cm(s_ens)),
                    n_night=int((lum_r < 50).sum()), n_day=int((lum_r >= 120).sum()), n_pairs=int(len(pi)), path_only_total=tot,
                    frames_o2_pos=int((o2 > 1e-9).sum()), frames_o2_ge_half=int((o2 >= 0.5).sum()), frames_o2_neg=int((o2 < -1e-9).sum()))
        (RES / "val_meta.json").write_text(json.dumps(meta, indent=1, default=float))
        run.summary.update(meta["reproduce"])


def figure(T2):
    """Stacked points of WLG's gap by situation: speed-only, path-only, joint remainder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    d = T2[T2.kind == "situation"].sort_values("gap pts", ascending=True)
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 2.9))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for col, c, lab in (("speed-only pts (O1)", ps.PALETTE["blue"], "speed only (own path, top-rated speed)"),
                        ("path-only pts (O2)", ps.PALETTE["vermillion"], "path only (top-rated path, own speed)"), ("joint pts", ps.BASELINE, "needs both")):
        v = d[col].to_numpy()
        ax.barh(y, v, left=left, color=c, label=lab, height=0.62)
        left = left + v
    ax.errorbar(d["gap pts"], y, xerr=[d["gap pts"] - d["gap_lo"], d["gap_hi"] - d["gap pts"]], fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s.split(' (')[0]} (n = {k})" for s, k in zip(d.stratum, d.n)])
    ax.set_xlabel("RFS points of the gap to the top-rated trajectory (cluster-mean units; bar = 95% CI of the total)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    ps.save(fig, FIG / "q2_budget_by_situation")
    plt.close(fig)


def sheets(pick, names, cl, v0, sc, traj, own, lg, sh, o2traj, A, s_top, o2, ptype, fr, per=6):
    """Review sheets of the picked frames: front camera crop next to a BEV (ego at the origin heading up) with the rated trajectories, log, WLG and the O2 swap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from jevdrive import waymo as W
    df = W.load_index()
    key = {nm: i for i, nm in enumerate(W.frame_names(df))}
    xy = lambda p: (-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]])  # noqa: E731

    def img(nm, cam):
        rw = df.iloc[key[nm]]
        with open(W.shard_dir() / rw.shard, "rb") as fh:
            fh.seek(int(rw[f"{cam}_off"]))
            return Image.open(io.BytesIO(fh.read(int(rw[f"{cam}_len"])))).convert("RGB")
    for s0 in range(0, len(pick), per):
        blk = pick[s0:s0 + per]
        fig, axs = plt.subplots(len(blk), 2, figsize=(12.5, 3.5 * len(blk)), gridspec_kw=dict(width_ratios=[2.5, 1]))
        for (ax_i, ax_b), f in zip(np.atleast_2d(axs), blk):
            nm = names[f]
            ims = [img(nm, c) for c in ("front_left", "front", "front_right")]
            h = min(i.height for i in ims)
            strip = np.concatenate([np.asarray(i.resize((int(i.width * h / i.height), h)))[int(h * 0.25):int(h * 0.80)] for i in ims], 1)
            ax_i.imshow(strip)
            ax_i.axis("off")
            ax_i.set_title(f"#{int(fr['rank'][f])}  {nm}  |  {cl[f]}  |  v0 {v0[f]:.1f} m/s  |  WLG {A[f]:.2f}  top {s_top[f]:.0f}  |  path-only gain {o2[f]:+.2f}  |  {ptype[f]}",
                           fontsize=8.5, loc="left")
            t = int(np.argmax(sc[f]))
            for j in np.argsort(sc[f]):
                ax_b.plot(*xy(traj[f, j]), color="#0b0b0b" if j == t else "#898781", lw=2.2 if j == t else 1.1, ls="-" if j == t else (0, (4, 2)),
                          label=f"rater {sc[f, j]:.0f}" + (" (top)" if j == t else ""))
            ax_b.plot(*xy(lg[f]), color="#1baf7a", lw=1.6, label="log")
            ax_b.plot(*xy(sh[f]), color="#eb6834", lw=1.2, label="shipped")
            for s, p in enumerate(own):
                ax_b.plot(*xy(p[f]), color="#2a78d6", lw=2, ls="-" if s == 0 else (0, (1, 1)), label=f"WLG s{s}")
            ax_b.plot(*xy(o2traj[f]), color="#c026d3", lw=1.2, ls=(0, (3, 1)), label="top path, own speed")
            ax_b.plot(0, 0, "^", ms=7, color="#0b0b0b")
            ally = np.concatenate([traj[f].reshape(-1, 2), lg[f], own[0][f], own[1][f]])
            half = max(4.0, np.abs(ally[:, 1]).max() * 1.2 + 1.0)
            ax_b.set_xlim(-half, half)
            ax_b.set_ylim(min(-1.0, ally[:, 0].min() - 1), max(6.0, ally[:, 0].max() * 1.08 + 1))
            ax_b.grid(color="#e4e3df", lw=0.5)
            ax_b.tick_params(labelsize=7)
            ax_b.legend(fontsize=6, frameon=False, loc="best")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, format="png")
        plt.close(fig)
        Image.open(buf).convert("RGB").save(FIG / f"q2_top30_sheet{s0 // per + 1}.webp", quality=72, method=6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sheets", action="store_true")
    main(ap.parse_args())
