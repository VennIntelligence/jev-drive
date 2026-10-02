"""Tables for the navhard off-road diagnosis (plan: experiments/skill_pack/plans/2026-10-03-navhard-offroad-diagnosis-plan.md).

navsim2 env, CPU. Reads features.pkl (offroad_replay_cf.py), the slim index, the cached plans, the per-token official CSVs and
writes CSV / JSON tables to <results>/tables and the one-line numbers to <results>/numbers.json.
  Q1  opposite-side rates by stage / command / map, history yaw rate vs plan / reference side
  Q2  failure statistics, classes, lag measures
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

MODELS = ["native", "n4"]
CLASSES = ["start_outside", "wrong_direction", "early_clip", "junction", "late_undershoot", "other"]


def build():
    """One row per token with the features of both models."""
    F = pickle.load(open(L.OUT / "features.pkl", "rb"))
    idx = {e["token"]: e for e in L.index()}
    rows = []
    for t, r in F.items():
        e = idx[t]
        ref = r["ref"]
        pose, vel = e["pose"], e["vel"]
        row = dict(token=t, stage=e["stage"], map=e["map"], cmd=CMD[int(np.argmax(e["cmd"][-1]))], speed=float(np.linalg.norm(vel[-1])),
                   yaw_rate_t0=float((pose[3, 2] - pose[2, 2]) / 0.5), yaw_hist=float(pose[3, 2] - pose[0, 2]),
                   center_off=r["start"]["center_off"], center_head_err=r["start"]["center_head_err"],
                   ref_end_y=float(ref[-1, 1]), ref_end_x=float(ref[-1, 0]), ref_end_yaw=float(ref[-1, 2]), ref_y0=float(ref[0, 1]),
                   ref_yaw1=float(ref[10, 2]), ref_y1=float(ref[10, 1]))
        for m in MODELS + ["pdm"]:
            f = r["feat"][m]
            d = f["raw_dense"]
            row.update({f"{m}_dac": f["dac"], f"{m}_start_outside": f["start_outside"], f"{m}_first": f["first"] if f["first"] is not None else -1,
                        f"{m}_d_first": f["d_first"], f"{m}_d_worst": f["d_worst"], f"{m}_junction": f["junction"], f"{m}_side": f["side_first"],
                        f"{m}_end_y": float(d[-1, 1]), f"{m}_end_x": float(d[-1, 0]), f"{m}_end_yaw": float(d[-1, 2]),
                        f"{m}_yaw1": float(d[10, 2]), f"{m}_y1": float(d[10, 1])})
            for tt, v in f["lat_ref"].items():
                row[f"{m}_latref_{tt:g}"] = v
            first = f["first"]
            if first is not None:
                row[f"{m}_y_at_dep"] = float(f["sim_ego"][first, 1])
                row[f"{m}_y_ref_at_dep"] = float(ref[first, 1])
            for tt in (0.5, 1.0, 2.0, 4.0):
                i = int(round(tt / 0.1))
                row[f"{m}_sim_y_{tt:g}"] = float(f["sim_ego"][i, 1])
                row[f"{m}_raw_y_{tt:g}"] = float(d[i, 1])
                row[f"{m}_track_lat_{tt:g}"] = float(f["sim_ego"][i, 1] - d[i, 1])     # simulated minus commanded (tracker)
        rows.append(row)
    return pd.DataFrame(rows)


CMD = ["left", "straight", "right", "unknown"]


def opposite(df, m):
    y, r = df[f"{m}_end_y"], df.ref_end_y
    return (y * r < 0) & (y.abs() > 1) & (r.abs() > 1) & ((y - r).abs() > 2)


def classify(df, m):
    """Plan classes of the stage-2 DAC failures of model m (order as the plan)."""
    fail = (df.stage == "two") & (df[f"{m}_dac"] == 0)
    cls = pd.Series("", index=df.index)
    opp = opposite(df, m)
    first = df[f"{m}_first"]
    s = np.sign(df.ref_end_yaw)
    under = (s * (df[f"{m}_y_at_dep"] - df[f"{m}_y_ref_at_dep"]) < 0) & (np.sign(df[f"{m}_side"]) == -s) & (df.ref_end_yaw.abs() >= 0.3)
    early = (first >= 0) & (first <= 15) & ((df.center_off.abs() >= 0.5) | (df.center_head_err.abs() >= 0.1))
    cls[fail] = "other"
    cls[fail & under & (first > 15)] = "late_undershoot"
    cls[fail & df[f"{m}_junction"]] = "junction"
    cls[fail & early] = "early_clip"
    cls[fail & opp] = "wrong_direction"
    cls[fail & df[f"{m}_start_outside"]] = "start_outside"
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/skill_pack/results/navhard-offroad"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    T = out / "tables"
    df = build()
    df.to_pickle(L.OUT / "token_table.pkl")
    num = {}

    # ------------------------------------------------------------ Q1: opposite side
    rows = []
    for m in MODELS:
        df[f"{m}_opp"] = opposite(df, m)
        for key in ("stage", "cmd", "map"):
            for (st, k), g in df.groupby(["stage", key]):
                rows.append(dict(model=m, stage=st, by=key, group=k, n=len(g), n_opp=int(g[f"{m}_opp"].sum()), rate=float(g[f"{m}_opp"].mean()),
                                 dac_fail_among_opp=float((g.loc[g[f"{m}_opp"], f"{m}_dac"] == 0).mean()) if g[f"{m}_opp"].any() else np.nan,
                                 dac_fail_rate=float((g[f"{m}_dac"] == 0).mean())))
        for st, g in df.groupby("stage"):
            rows.append(dict(model=m, stage=st, by="all", group="all", n=len(g), n_opp=int(g[f"{m}_opp"].sum()), rate=float(g[f"{m}_opp"].mean()),
                             dac_fail_among_opp=float((g.loc[g[f"{m}_opp"], f"{m}_dac"] == 0).mean()) if g[f"{m}_opp"].any() else np.nan,
                             dac_fail_rate=float((g[f"{m}_dac"] == 0).mean())))
    q1 = pd.DataFrame(rows)
    q1.to_csv(T / "q1_opposite_side.csv", index=False)
    # mirror symmetry and the history cue
    sym = []
    for m in MODELS:
        for st, g in df.groupby("stage"):
            o = g[g[f"{m}_opp"]]
            sym.append(dict(model=m, stage=st, n_opp=len(o), ref_left_plan_right=int(((o.ref_end_y > 0) & (o[f"{m}_end_y"] < 0)).sum()),
                            ref_right_plan_left=int(((o.ref_end_y < 0) & (o[f"{m}_end_y"] > 0)).sum()),
                            plan_follows_history_yaw=float((np.sign(o[f"{m}_end_y"]) == np.sign(o.yaw_rate_t0)).mean()) if len(o) else np.nan,
                            ref_follows_history_yaw=float((np.sign(o.ref_end_y) == np.sign(o.yaw_rate_t0)).mean()) if len(o) else np.nan))
    pd.DataFrame(sym).to_csv(T / "q1_symmetry.csv", index=False)
    # history yaw rate vs the reference's turn, stage 1 vs stage 2 (is the synthesized history's curvature sign the reference's?)
    h = []
    for st, g in df.groupby("stage"):
        mv = g[(g.speed > 2) & (g.ref_end_yaw.abs() > 0.3)]
        mv2 = mv[mv.yaw_rate_t0.abs() > 0.05]
        h.append(dict(stage=st, n=len(g), n_turning_ref=len(mv), n_hist_turning=len(mv2),
                      hist_yaw_sign_equals_ref_turn=float((np.sign(mv2.yaw_rate_t0) == np.sign(mv2.ref_end_yaw)).mean()),
                      corr_plan_end_y_vs_hist_yawrate_native=float(np.corrcoef(g.native_end_y, g.yaw_rate_t0 * g.speed)[0, 1]),
                      corr_ref_end_y_vs_hist_yawrate=float(np.corrcoef(g.ref_end_y, g.yaw_rate_t0 * g.speed)[0, 1]),
                      corr_plan_end_y_vs_ref_end_y_native=float(np.corrcoef(g.native_end_y, g.ref_end_y)[0, 1]),
                      corr_plan_end_y_vs_ref_end_y_n4=float(np.corrcoef(g.n4_end_y, g.ref_end_y)[0, 1])))
    pd.DataFrame(h).to_csv(T / "q1_history_cue.csv", index=False)
    # sign agreement of plan heading at 4 s and reference heading at 4 s (convention check), tokens with a clear turn
    conv = []
    for m in MODELS:
        for st, g in df.groupby("stage"):
            c = g[(g.ref_end_yaw.abs() > 0.4) & (g[f"{m}_end_yaw"].abs() > 0.2)]
            conv.append(dict(model=m, stage=st, n=len(c), same_sign=float((np.sign(c[f"{m}_end_yaw"]) == np.sign(c.ref_end_yaw)).mean())))
    pd.DataFrame(conv).to_csv(T / "q1_sign_agreement.csv", index=False)

    # ------------------------------------------------------------ Q2: failure statistics
    st2 = df[df.stage == "two"].copy()
    rows, crow = [], []
    for m in MODELS:
        st2[f"{m}_class"] = classify(st2, m)
        fail = st2[st2[f"{m}_dac"] == 0]
        nf = len(fail)
        num[f"{m}_stage2_dac_fail"] = nf
        num[f"{m}_stage2_n"] = len(st2)
        dep = fail[~fail[f"{m}_start_outside"]]
        bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.01]
        h, _ = np.histogram(dep[f"{m}_first"] * 0.1, bins=bins)
        for lo, hi, c in zip(bins[:-1], bins[1:], h):
            rows.append(dict(model=m, stat="first_departure_s", bin=f"({lo:g},{min(hi, 4):g}]", count=int(c), share=float(c / max(1, len(dep)))))
        for k, nm in (("d_first", "outside_cm_first"), ("d_worst", "outside_cm_worst")):
            q = (dep[f"{m}_{k}"] * 100).quantile([.25, .5, .75, .9]).round(1).tolist()
            rows.append(dict(model=m, stat=nm, bin="q25/50/75/90", count=len(dep), share=np.nan, q=str(q)))
        for tt in (0.0, 0.5, 1.0, 2.0, 4.0):
            for nm, g in (("fail", fail), ("pass", st2[st2[f"{m}_dac"] == 1])):
                v = g[f"{m}_latref_{tt:g}"]
                rows.append(dict(model=m, stat=f"lat_vs_pdm_ref_{tt:g}s_{nm}", bin="median [q25,q75] m", count=len(v), share=np.nan,
                                 q=f"{v.median():.2f} [{v.quantile(.25):.2f},{v.quantile(.75):.2f}]"))
        for nm, g in (("fail", fail), ("pass", st2[st2[f"{m}_dac"] == 1])):
            rows.append(dict(model=m, stat=f"start_offset_{nm}", bin="|off| m, |head err| rad (median)", count=len(g), share=np.nan,
                             q=f"{g.center_off.abs().median():.2f} / {g.center_head_err.abs().median():.3f}"))
            rows.append(dict(model=m, stat=f"start_offset_ge0.5m_or_0.1rad_{nm}", bin="share", count=len(g),
                             share=float(((g.center_off.abs() >= .5) | (g.center_head_err.abs() >= .1)).mean())))
        for c in CLASSES:
            crow.append(dict(model=m, cls=c, n=int((fail[f"{m}_class"] == c).sum()) if f"{m}_class" in fail else int((st2.loc[fail.index, f"{m}_class"] == c).sum()),
                             share=float((st2.loc[fail.index, f"{m}_class"] == c).mean())))
    pd.DataFrame(rows).to_csv(T / "q2_failure_stats.csv", index=False)
    pd.DataFrame(crow).to_csv(T / "q2_classes.csv", index=False)
    st2.to_pickle(L.OUT / "stage2_table.pkl")
    # lateral gap to the PDM reference closed by the simulated car (tokens with a clear initial gap), and the share of the commanded
    # lateral displacement the tracker realises
    gc = []
    for m in MODELS:
        for nm, g in (("fail", st2[st2[f"{m}_dac"] == 0]), ("pass", st2[st2[f"{m}_dac"] == 1])):
            g = g[g[f"{m}_latref_0"].abs() > 0.5]
            r = dict(model=m, set=nm, n=len(g))
            for tt in (0.5, 1.0, 2.0, 4.0):
                r[f"gap_closed_{tt:g}s_median"] = float((1 - g[f"{m}_latref_{tt:g}"].abs() / g[f"{m}_latref_0"].abs()).median())
                r[f"abs_lat_vs_ref_{tt:g}s_median"] = float(g[f"{m}_latref_{tt:g}"].abs().median())
            mv = st2[(st2[f"{m}_dac"] == 0 if nm == "fail" else st2[f"{m}_dac"] == 1) & (st2.speed > 3)]
            for tt in (1.0, 2.0):
                k = mv[mv[f"{m}_raw_y_{tt:g}"].abs() > 0.3]
                r[f"tracker_realised_share_{tt:g}s_median"] = float((k[f"{m}_sim_y_{tt:g}"] / k[f"{m}_raw_y_{tt:g}"]).median())
            gc.append(r)
    pd.DataFrame(gc).to_csv(T / "q2_gap_closure.csv", index=False)
    # by command / map for the failures
    gm = []
    for m in MODELS:
        for key in ("cmd", "map"):
            for k, g in st2.groupby(key):
                gm.append(dict(model=m, by=key, group=k, n=len(g), dac_fail_rate=float((g[f"{m}_dac"] == 0).mean()),
                               start_outside_rate=float(g[f"{m}_start_outside"].mean())))
    pd.DataFrame(gm).to_csv(T / "q2_fail_by_cmd_map.csv", index=False)
    # agreement of the two models' failures
    both = int(((st2.native_dac == 0) & (st2.n4_dac == 0)).sum())
    num["both_fail"], num["native_only_fail"], num["n4_only_fail"] = both, int(((st2.native_dac == 0) & (st2.n4_dac == 1)).sum()), int(((st2.native_dac == 1) & (st2.n4_dac == 0)).sum())
    num["pdm_ref_dac_fail_stage2"] = int((st2.pdm_dac == 0).sum())
    num["start_outside_stage2"] = int(st2.native_start_outside.sum())

    # ------------------------------------------------------------ Q2: lag measures
    lag = []
    z = np.load(L.NATIVE_PLANS)
    from jevdrive import op_interp as I
    nh = {t: i for i, t in enumerate(z["names"].tolist())}
    idx = {e["token"]: e for e in L.index()}

    def plan_metrics(pos, yaw, cam):
        r = I.to_rear(pos, yaw, I.T_IDXS, cam[:2], I.T_IDXS, "lever", "linear")      # (33, 3)
        f = lambda t, k: float(np.interp(t, I.T_IDXS, r[:, k]))  # noqa: E731
        s1 = float(np.hypot(f(1.0, 0), f(1.0, 1)))
        return dict(psi_0p5=f(0.5, 2), psi_1=f(1.0, 2), y_0p5=f(0.5, 1), y_1=f(1.0, 1), y_2=f(2.0, 1), s_1=s1,
                    kappa_1=f(1.0, 2) / max(s1, 1.0))

    nav = [(t, i) for t, i in nh.items()]
    rows_h = []
    for t, i in nav:
        e = idx[t]
        cam = np.asarray(e["cams"][-1]["CAM_F0"]["t"], float)
        d = plan_metrics(z["plan_pos"][i], z["plan_yaw"][i], cam)
        d.update(token=t, stage=e["stage"], speed=float(np.linalg.norm(e["vel"][-1])),
                 kappa_hist=float((e["pose"][3, 2] - e["pose"][2, 2]) / 0.5 / max(float(np.linalg.norm(e["vel"][-1])), 1.0)))
        rows_h.append(d)
    H = pd.DataFrame(rows_h).set_index("token")
    # navtest reference plans
    zt = np.load(L.D / "runs/op_lb/lb_navtest/plans/gimm@cinque.npz")
    it = {e["token"]: e for e in __import__("pickle").load(open(L.D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
    nt_rows = []
    for k, t in enumerate(zt["names"].tolist()):
        e = it[t]
        sp = float(np.linalg.norm(e["vel"][-1]))
        if sp <= 3:
            continue
        cam = np.asarray(e["cams"][-1]["CAM_F0"]["t"], float)
        d = plan_metrics(zt["plan_pos"][k], zt["plan_yaw"][k], cam)
        d.update(token=t, speed=sp, kappa_hist=float((e["pose"][3, 2] - e["pose"][2, 2]) / 0.5 / max(sp, 1.0)))
        nt_rows.append(d)
    NT = pd.DataFrame(nt_rows).set_index("token")
    futt = np.load(L.D / "runs/navsim_zs/index/navtest_future.npz")
    gt = dict(zip(futt["tokens"].tolist(), futt["poses"]))
    NT["human_psi_1"] = [float(gt[t][1, 2]) for t in NT.index]
    NT["human_y_1"] = [float(gt[t][1, 1]) for t in NT.index]
    NT["human_y_2"] = [float(gt[t][3, 1]) for t in NT.index]
    NT.to_pickle(L.OUT / "navtest_lag.pkl")
    H.to_pickle(L.OUT / "navhard_lag.pkl")

    def summ(name, g, ref_psi=None, ref_y1=None, ref_y2=None):
        r = dict(set=name, n=len(g), kappa1_over_kappa_hist=np.nan, psi1_over_ref=np.nan, y1_over_ref=np.nan, y2_over_ref=np.nan,
                 frac_psi1_lt_half_ref=np.nan)
        m = g.kappa_hist.abs() > 0.01
        if m.sum() > 20:
            r["kappa1_over_kappa_hist"] = float(np.median(g.kappa_1[m] * np.sign(g.kappa_hist[m]) / g.kappa_hist.abs()[m]))
        if ref_psi is not None:
            mm = ref_psi.abs() > 0.05
            if mm.sum() > 20:
                q = (g.psi_1[mm] * np.sign(ref_psi[mm]) / ref_psi.abs()[mm])
                r["psi1_over_ref"], r["frac_psi1_lt_half_ref"] = float(q.median()), float((q < 0.5).mean())
                r["n_ref_turning_1s"] = int(mm.sum())
            my = ref_y1.abs() > 0.2
            if my.sum() > 20:
                r["y1_over_ref"] = float((g.y_1[my] * np.sign(ref_y1[my]) / ref_y1.abs()[my]).median())
        r.update(psi_0p5_abs_med=float(g.psi_0p5.abs().median()), psi_1_abs_med=float(g.psi_1.abs().median()), y_1_abs_med=float(g.y_1.abs().median()))
        return r

    lag.append(summ("navtest moving (>3 m/s), human reference", NT, NT.human_psi_1, NT.human_y_1, NT.human_y_2))
    s1 = df[(df.stage == "one") & (df.speed > 3)].set_index("token")
    lag.append(summ("navhard stage 1 moving, PDM reference", H.loc[s1.index], s1.ref_yaw1, s1.ref_y1, None))
    s2 = df[(df.stage == "two") & (df.speed > 3)].set_index("token")
    lag.append(summ("navhard stage 2 moving, all, PDM reference", H.loc[s2.index], s2.ref_yaw1, s2.ref_y1, None))
    f2 = s2[s2.native_dac == 0]
    lag.append(summ("navhard stage 2 moving, native DAC failures", H.loc[f2.index], f2.ref_yaw1, f2.ref_y1, None))
    p2 = s2[s2.native_dac == 1]
    lag.append(summ("navhard stage 2 moving, native passes", H.loc[p2.index], p2.ref_yaw1, p2.ref_y1, None))
    # PDM reference vs human future at stage 1 (is the reference's own first second comparable?)
    pd.DataFrame(lag).to_csv(T / "q2_lag_first_second.csv", index=False)
    # tracker lag and conversion: commanded (8-pose, linearly interpolated) vs simulated lateral at 0.5 / 1 / 2 s, moving stage-2 failures
    tl = []
    for m in MODELS:
        for nm, g in (("failures", st2[(st2[f"{m}_dac"] == 0) & (st2.speed > 3)]), ("passes", st2[(st2[f"{m}_dac"] == 1) & (st2.speed > 3)])):
            r = dict(model=m, set=nm, n=len(g))
            for tt in (0.5, 1.0, 2.0):
                r[f"tracker_lat_minus_cmd_{tt:g}s_med_abs"] = float(g[f"{m}_track_lat_{tt:g}"].abs().median())
                r[f"cmd_lat_{tt:g}s_med_abs"] = float(g[f"{m}_raw_y_{tt:g}"].abs().median())
            tl.append(r)
    pd.DataFrame(tl).to_csv(T / "q2_tracker_lag.csv", index=False)
    # raw 33-knot plan vs the 8-pose file at the pose times (conversion check)
    cmax = 0.0
    for t in list(nh)[:500]:
        e = idx[t]
        cam = np.asarray(e["cams"][-1]["CAM_F0"]["t"], float)
        r = I.to_rear(z["plan_pos"][nh[t]], z["plan_yaw"][nh[t]], I.T_IDXS, cam[:2], I.T_IDXS, "lever", "linear")
        a8 = I.to_rear(z["plan_pos"][nh[t]], z["plan_yaw"][nh[t]], I.T_IDXS, cam[:2], L.T_POSE, "lever", "linear")
        dense = np.stack([np.interp(np.arange(0.1, 4.0001, 0.1), I.T_IDXS, r[:, k]) for k in range(2)], 1)
        lin = np.stack([np.interp(np.arange(0.1, 4.0001, 0.1), np.r_[0, L.T_POSE], np.r_[0, a8[:, k]]) for k in range(2)], 1)
        cmax = max(cmax, float(np.abs(dense - lin).max()))
    num["conversion_max_lateral_gap_8pose_vs_33knot_m_first500"] = cmax
    (out / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print(json.dumps(num, indent=1, default=float))
    for f in sorted(T.glob("*.csv")):
        print("==", f.name)
        print(pd.read_csv(f).round(3).to_string(index=False)[:6000])


if __name__ == "__main__":
    main()
