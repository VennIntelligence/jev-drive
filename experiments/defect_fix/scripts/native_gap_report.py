#!/usr/bin/env python3
"""NATIVE-GAP report (results/native_gap.md tables): Mac, .venv, on the box outputs pulled to tmp/native_gap/ (native_gap.py's extract and cf,
plus the arena set and arena_resid's offroad_stop3.parquet for the human-track offroad windows).

  .venv/bin/python experiments/defect_fix/scripts/native_gap_report.py [--src tmp/native_gap] [--out experiments/defect_fix/results/native_gap_tables.md]

Definitions: native_gap.py's docstring. Strata of a moving decision (first match): after-stop (the ego was below 0.3 m/s at an earlier decision of
the window), turn (the human track turns >= 15 deg over the next 20 m), lead (a logged object in the human-track corridor within 25 m), free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402

TAGS = ["RSG", "NTV0", "JTN0", "JTN1"]
NAME = {"RSG": "RSG-L18-s0", "NTV0": "NTV0-L18-s0", "JTN0": "JTN-L18-s0", "JTN1": "JTN-L18-s1"}
OUT = []


def p(*x):
    OUT.append(" ".join(str(y) for y in x))


def table(df, fmt="{:.3f}"):
    cols = list(df.columns)
    p("| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |")
    p("|---|" + "--:|" * len(cols))
    for i, r in df.iterrows():
        p("| " + " | ".join([str(i)] + [(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("-" if isinstance(v, (float, np.floating)) else str(v))) for v in r]) + " |")
    p("")


def load(src):
    D = pd.read_parquet(src / "dec.parquet")
    W = pd.read_parquet(src / "win.parquet")
    St = pd.read_parquet(src / "steps.parquet")
    C = pd.read_parquet(src / "set_census.parquet").drop_duplicates("id").set_index("id")
    P = np.stack(D.poses.map(lambda x: np.asarray(x, float).reshape(8, 3)).values)
    D["arc4"] = np.hypot(*np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P[:, :, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
    D["arc2"] = np.hypot(*np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P[:, :4, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
    D["vp05"] = np.hypot(P[:, 1, 0], P[:, 1, 1]) / 1.0                       # central difference at 0.5 s: |p(1.0) - p(0)| / 1 s
    D["vp10"] = np.hypot(*(P[:, 2, :2] - P[:, 0, :2]).T) / 1.0
    D["p05x"], D["p10x"], D["p20x"] = P[:, 0, 0], P[:, 1, 0], P[:, 3, 0]
    D = D.sort_values(["tag", "id", "k"]).reset_index(drop=True)
    # executed motion: the next decisions' anchors in this decision's frame
    g = D.groupby(["tag", "id"])
    for h in (1, 2, 4):
        for c in ("x", "y", "v", "vh", "t0"):
            D[f"{c}_{h}"] = g[c].shift(-h)
        dx, dy = D[f"x_{h}"] - D.x, D[f"y_{h}"] - D.y
        D[f"ex_{h}"] = np.cos(D.yaw) * dx + np.sin(D.yaw) * dy
    # validity: before the first kept-step event
    s = St[St.eval_relevant > 0]
    ev = s[(s.collision_any > 0) | (s.offroad > 0) | (s.dist_to_gt_trajectory >= 4)].groupby(["tag", "id"]).t_us.min()
    D["t_ev"] = pd.MultiIndex.from_frame(D[["tag", "id"]]).map(ev.to_dict()).astype(float)
    D["valid"] = ~(D.t0 >= D.t_ev)
    D["min_prev_v"] = g.v.transform(lambda x: x.shift(1).cummin())
    D["last_stop_t"] = D.t0.where(D.v < 0.3)
    D["last_stop_t"] = D.groupby(["tag", "id"]).last_stop_t.transform(lambda x: x.shift(1).ffill())
    return D, W, St, C


def common(W, C, S):
    X = W.pivot_table(index="id", columns="tag", values="score")
    T = W.pivot_table(index="id", columns="tag", values="type", aggfunc="first")
    ok = X[TAGS].notna().all(axis=1) & ~(T[TAGS] == "route-fail").any(axis=1) & X.index.isin(S.index[S.split == "held"])
    return X.index[ok]


def stratum_dec(D):
    return np.select([D.min_prev_v < 0.3, D.turn20 >= 15, D.lead_gap < 25], ["after-stop", "turn", "lead"], "free")


def part_a(D, ids, S):
    p("## A. The speed ratchet on moving windows")
    p("")
    M = D[D.id.isin(ids) & D.valid & (D.k >= 2) & (D.vh >= 2)].copy()
    M["r"] = M.v / M.vh
    M["st"] = stratum_dec(M)
    p("Ego speed / human speed at the same station (decisions k >= 2, valid, human >= 2 m/s there), median by decision index:")
    p("")
    t = M.pivot_table(index="k", columns="tag", values="r", aggfunc="median")[TAGS]
    t.index.name = "k"
    table(t)
    t = M.pivot_table(index="st", columns="tag", values="r", aggfunc="median")[TAGS]
    t["decisions (RSG)"] = M[M.tag == "RSG"].groupby("st").size()
    t.index.name = "decision stratum"
    p("By decision stratum (median ratio):")
    p("")
    table(t)
    still = M[M.v < 0.3]
    p("Decisions where the ego stands (< 0.3 m/s) at a station the human passed at >= 2 m/s (the launch lag; excluded from the next table): "
      + ", ".join(f"{t_} {int((still.tag == t_).sum())}" for t_ in TAGS) + ".")
    p("")
    a = M[(M.st == "after-stop") & (M.v >= 0.3)].copy()
    a["since"] = pd.cut((a.t0 - a.last_stop_t) / 1e6, [0, 1, 2, 3, 9], labels=["0-1 s", "1-2 s", "2-3 s", "> 3 s"])
    t = a.pivot_table(index="since", columns="tag", values="r", aggfunc="median", observed=False)[TAGS]
    t.index.name = "time since the ego's last stop"
    p("After a stop, the ego moving again (>= 0.3 m/s), by time since its last stop (median ratio):")
    p("")
    table(t)
    # step decomposition
    Q = D[D.id.isin(ids) & D.valid & (D.k >= 2) & (D.k <= 8) & (D.vh >= 2) & (D.vh_1 >= 2) & (D.v >= 1) & (D.vp05 > 0.3)].copy()
    Q = Q[Q.t0_1.notna() & (Q.t0_1 < Q.t_ev.fillna(np.inf))]
    Q["st"] = stratum_dec(Q)
    Q["plan"] = np.log(Q.vp05 / Q.v) - np.log(Q.vh_1 / Q.vh)
    Q["exec"] = np.log(Q.v_1 / Q.vp05)
    Q["total"] = Q.plan + Q.exec
    Q["plan_ms"] = (Q.vp05 - Q.v) - (Q.vh_1 - Q.vh)
    Q["exec_ms"] = Q.v_1 - Q.vp05
    Q["lag05"] = Q.p05x - Q.ex_1                                              # executed forward motion behind the plan's 0.5 s pose (m)
    Q["lag10"] = (Q.p10x - Q.ex_2).where(Q.t0_2 < Q.t_ev.fillna(np.inf))
    Q["lag20"] = (Q.p20x - Q.ex_4).where(Q.t0_4 < Q.t_ev.fillna(np.inf))
    p("Step decomposition (k -> k+1, k = 2..8, both valid; human >= 2 m/s at both stations, ego >= 1 m/s). Change of log(ego / human speed at "
      "the same station) = **plan term** log(vp(0.5 s) / v) - log(vh[k+1] / vh[k]) + **execution term** log(v[k+1] / vp(0.5 s)). Sums over all "
      "steps (= the summed drop of the log ratio), share of the total, and the same in m/s:")
    p("")
    rows = []
    for tag in TAGS:
        for st in ["all", "free", "lead", "turn", "after-stop"]:
            q = Q[(Q.tag == tag) & ((Q.st == st) if st != "all" else True)]
            if not len(q):
                continue
            rows.append(dict(model=tag, stratum=st, steps=len(q), per_step_total=q.total.mean(), per_step_plan=q.plan.mean(), per_step_exec=q.exec.mean(),
                             plan_share=q.plan.sum() / q.total.sum(), plan_ms=q.plan_ms.mean(), exec_ms=q.exec_ms.mean(), lag05_m=q.lag05.median(),
                             lag10_m=q.lag10.median(), lag20_m=q.lag20.median()))
    t = pd.DataFrame(rows).set_index("model")
    t.index.name = "model"
    table(t, "{:+.4f}")
    # the losing steps only
    L = Q[Q.total < 0]
    t = L.groupby("tag").apply(lambda q: pd.Series(dict(steps=len(q), share_of_steps=len(q) / (Q.tag == q.name).sum(), plan=q.plan.sum() / q.total.sum(),
                                                     exec=q.exec.sum() / q.total.sum())), include_groups=False).loc[TAGS]
    t.index.name = "losing steps (total < 0)"
    p("Losing steps only (shares of their summed loss):")
    p("")
    table(t)
    # per window: ratio drop k2 -> k9 and the window's score
    R = M.pivot_table(index=["tag", "id"], columns="k", values="r")
    R = R[[2, 9]].dropna() if 9 in R.columns else R
    p(f"Window ratio at k = 2 / k = 9 (median, windows valid to the end): " + ", ".join(
        f"{t_} {R.loc[t_][2].median():.3f} / {R.loc[t_][9].median():.3f}" for t_ in TAGS if t_ in R.index.get_level_values(0)))
    p("")
    return Q


def part_cf(src, D):
    p("## A2. Open-loop counterfactual: the same frames, the human's speed history")
    p("")
    rows = []
    for tag in TAGS:
        f = src / "cf" / f"{tag}.parquet"
        if not f.exists():
            continue
        X = pd.read_parquet(f)
        F = np.stack(X.F.map(lambda x: np.asarray(x).reshape(8, 3))).astype(float)
        H = np.stack(X.H.map(lambda x: np.asarray(x).reshape(8, 3))).astype(float)
        vh = np.stack(X.vh_keys.map(np.asarray))[:, -1]
        arc = lambda P, n: np.hypot(*np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P[:, :n, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)  # noqa: E731
        X["vF"], X["vH"] = arc(F, 4) / 2, arc(H, 4) / 2
        X["vF1"], X["vH1"] = np.hypot(F[:, 1, 0], F[:, 1, 1]), np.hypot(H[:, 1, 0], H[:, 1, 1])
        X["vh"] = vh
        g = X[(X.k >= 2) & (X.vh >= 2)]
        p(f"{tag}: gate (replayed F plan vs the dumped plan, max |dxy| over the 8 poses) max {X.gate.max():.4f} m, p99 {X.gate.quantile(0.99):.4f} m; "
          f"{len(X)} decisions, {len(g)} with k >= 2 and the human >= 2 m/s")
        for nm, q in (("all", g), ("ego < 0.9 x human", g[g.v < 0.9 * g.vh]), ("ego < 0.7 x human", g[g.v < 0.7 * g.vh])):
            rF, rH = (q.vF / q.vh).median(), (q.vH / q.vh).median()
            dv = q.vh - q.v
            m = dv.abs() > 0.5
            gam = ((q.vH - q.vF)[m] / dv[m]).median()
            rows.append(dict(model=tag, decisions=nm, n=len(q), ego_over_human=(q.v / q.vh).median(), plan_F=rF, plan_H=rH,
                             recovery=(rH - rF) / (1 - rF) if rF < 1 else np.nan, gain=gam,
                             plan_F_05=(q.vF1 / q.vh).median(), plan_H_05=(q.vH1 / q.vh).median()))
    if rows:
        p("")
        p("Plan speed over 0-2 s (arc2 / 2 s) / the human's speed at the ego's station; F = as served, H = speed history = the human's; "
          "recovery = (H - F) / (1 - F) of the medians; gain = median (vp_H - vp_F) / (vh - v) per decision (|vh - v| > 0.5 m/s; 1 = the plan "
          "moves one for one with the fed speed, 0 = it ignores it); plan 0.5 s = the central speed at 0.5 s:")
        p("")
        t = pd.DataFrame(rows).set_index("model")
        table(t)


def cue(S, C):
    red = C.red_stop.fillna(False).astype(bool) & (C.red_cls.fillna("") == "C") & C.red_start.fillna(False).astype(bool)
    lead = S.lead.fillna(False).astype(bool)
    gap = S.lead_gap.astype(float)
    tl = C.tl.fillna(False).astype(bool)
    red_any = C.red_stop.fillna(False).astype(bool) & (C.red_cls.fillna("") == "C")
    return pd.Series(np.select([lead & (gap <= 8), lead, red.reindex(S.index).fillna(False), red_any.reindex(S.index).fillna(False), tl.reindex(S.index).fillna(False)],
                               ["lead <= 8 m", "lead 8-30 m", "red light, no lead (red at start)", "red light, no lead (red later)", "other light, no lead"],
                               "no light, no lead"), index=S.index)


def part_b(D, X, T, ids, S, C):
    p("## B. Static windows natively")
    p("")
    st = ids[S.loc[ids, "stratum"].values == "static"]
    cu = cue(S.loc[st], C)
    Z = D[D.id.isin(st)].copy()
    Z["disp"] = np.hypot(Z.x - Z.groupby(["tag", "id"]).x.transform("first"), Z.y - Z.groupby(["tag", "id"]).y.transform("first"))
    rows = []
    per = {}
    for tag in TAGS:
        z = Z[Z.tag == tag]
        A = z.pivot_table(index="id", columns="k", values="arc4")
        V = z.pivot_table(index="id", columns="k", values="v")
        Dp = z.pivot_table(index="id", columns="k", values="disp")
        roll = (V.loc[:, 2:] > 0.2).idxmax(axis=1).where((V.loc[:, 2:] > 0.2).any(axis=1))
        w = pd.DataFrame(dict(score=X.loc[st, tag], type=T.loc[st, tag], cue=cu, a0=A[0], a2=A[2], a3=A[3], amax=A.loc[:, 3:].max(axis=1), v3=V[3], v5=V[5],
                              roll=roll, disp9=Dp[9]))
        w["zero"] = w.score == 0
        per[tag] = w
        for c in ["all"] + sorted(cu.unique()):
            q = w if c == "all" else w[w.cue == c]
            rows.append(dict(model=tag, cue=c, windows=len(q), zeros=int(q.zero.sum()), corridor=int((q.type == "corridor").sum()), mean=q.score.mean(),
                             arc4_k0=q.a0.median(), arc4_k2=q.a2.median(), arc4_max_k3=q.amax.median(), roll_k=q.roll.median(),
                             rolled=float(q.roll.notna().mean()), disp_end=q.disp9.median()))
    t = pd.DataFrame(rows).set_index("model")
    p("Per cue type (cue: the arena set's lead at the window start, then the census light labels; arc4 = the plan's 4 s arc, m, median; "
      "roll k = first decision k >= 2 with the ego above 0.2 m/s, median over the windows that roll; rolled = share that roll; "
      "disp end = the ego's distance from its start at k = 9):")
    p("")
    table(t)
    for tag in ("JTN1", "RSG"):
        w = per[tag]
        okw, zw = w[~w.zero], w[w.zero]
        p(f"{tag} static successes ({len(okw)}) vs zeros ({len(zw)}): arc4 at k0 {okw.a0.median():.2f} / {zw.a0.median():.2f} m, k2 "
          f"{okw.a2.median():.2f} / {zw.a2.median():.2f}, k3 {okw.a3.median():.2f} / {zw.a3.median():.2f}, max k>=3 {okw.amax.median():.2f} / "
          f"{zw.amax.median():.2f}; v at k3 {okw.v3.median():.2f} / {zw.v3.median():.2f} m/s, k5 {okw.v5.median():.2f} / {zw.v5.median():.2f}; "
          f"rolled {okw.roll.notna().mean():.2f} / {zw.roll.notna().mean():.2f}; zero roll k {zw.roll.value_counts().sort_index().to_dict()}")
        y = w.zero.values.astype(float)
        for c in ("a0", "a2", "a3"):
            x = w[c].values
            m = np.isfinite(x)
            from scipy.stats import rankdata
            r = rankdata(x[m])
            npos, nneg = y[m].sum(), (1 - y[m]).sum()
            auc = (r[y[m] == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            p(f"  AUC of {c} (plan 4 s arc at that decision) for the window's zero: {auc:.3f}")
        p("")
    # creep amplification on JTN1 zeros: arc4 by k
    for tag in ("JTN1", "RSG"):
        z = Z[(Z.tag == tag) & Z.id.isin(per[tag].index[per[tag].zero])]
        o = z.groupby("k")[["arc4", "v", "disp"]].median()
        zs = Z[(Z.tag == tag) & Z.id.isin(per[tag].index[~per[tag].zero])].groupby("k")[["arc4", "v", "disp"]].median()
        p(f"{tag} by decision (zeros | successes), median plan arc4 m / ego v m/s / displacement m:")
        p("")
        p("| k | " + " | ".join(str(k) for k in o.index) + " |")
        p("|---|" + "--:|" * len(o))
        for c in ("arc4", "v", "disp"):
            p(f"| {c} zeros | " + " | ".join(f"{x:.2f}" for x in o[c]) + " |")
            p(f"| {c} ok | " + " | ".join(f"{x:.2f}" for x in zs[c]) + " |")
        p("")
    # JTN1 vs RSG on the same static windows
    a, b = per["JTN1"], per["RSG"]
    p(f"Static windows, JTN1 vs RSG: both zero {int((a.zero & b.zero).sum())}, JTN1 only {int((a.zero & ~b.zero).sum())}, RSG only {int((~a.zero & b.zero).sum())}.")
    p("")
    return per


def part_c(D, X, T, ids, S, C, hum_off, per):
    p("## C. Points to 0.93")
    p("")
    N = len(ids)
    cu = cue(S.loc[ids], C)
    stt = S.loc[ids, "stratum"]
    best = X.loc[ids, TAGS].max(axis=1)
    rows = []
    cls_of = {}
    for tag in TAGS:
        sc, ty = X.loc[ids, tag], T.loc[ids, tag]
        c = pd.Series("", index=ids, dtype=object)
        zero = sc == 0
        c[zero & ids.isin(hum_off)] = "human-track offroad (ceiling)"
        m = (c == "") & (stt == "static")
        c[m & zero] = "static zero: " + cu[m & zero]
        c[m & ~zero & (sc < 1)] = "static partial"
        m = (c == "") & zero
        c[m & ty.str.startswith("AF-front")] = "AF-front zero"
        c[m & ty.str.startswith("AF-side")] = "AF-side zero"
        c[m & ty.str.contains("offroad") & (stt == "turn")] = "offroad zero: turn stratum"
        c[m & ty.str.contains("offroad") & (stt != "turn") & (c == "")] = "offroad zero: other strata"
        c[m & (ty == "corridor") & (c == "")] = "corridor zero: " + stt[m & (ty == "corridor") & (c == "")]
        c[m & (c == "")] = "zero other"
        m = (c == "") & (sc < 1)
        c[m & (stt == "launch")] = "launch partial (late / slow launch)"
        c[m & (ty == "rear-hit") & (c == "")] = "rear-hit partial"
        c[m & (c == "")] = "slow partial: " + stt[m & (c == "")]
        cls_of[tag] = c
        loss = 1 - sc
        for k, g in loss.groupby(c):
            if k == "":
                continue
            rows.append(dict(model=tag, cls=k, windows=int((c == k).sum()), points=g.sum(), best_of_4=(1 - best[c == k]).sum()))
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index="cls", columns="model", values="points", aggfunc="sum", fill_value=0)[TAGS]
    piv["JTN1 windows"] = t[t.model == "JTN1"].set_index("cls").windows
    piv["best-of-4 on JTN1's windows"] = t[t.model == "JTN1"].set_index("cls").best_of_4
    piv = piv.sort_values("JTN1", ascending=False)
    piv.loc["all"] = piv.sum()
    piv.index.name = "loss class (points = summed 1 - score)"
    p(f"{N} common held windows; mean: " + ", ".join(f"{tg} {X.loc[ids, tg].mean():.4f}" for tg in TAGS) + f"; best-of-4 per window {best.mean():.4f}; "
      f"0.93 needs {0.93 * N:.0f} points, JTN1 has {X.loc[ids, 'JTN1'].sum():.1f} (gap {0.93 * N - X.loc[ids, 'JTN1'].sum():.1f}).")
    p("")
    table(piv, "{:.1f}")
    ceil = 1 - len(set(hum_off) & set(ids)) / N
    p(f"Ceiling: {len(set(hum_off) & set(ids))} human-track offroad windows -> {ceil:.4f}.")
    p("")
    return cls_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "tmp/native_gap"))
    ap.add_argument("--out", default=str(REPO / "experiments/defect_fix/results/native_gap_tables.md"))
    a = ap.parse_args()
    src = Path(a.src)
    D, W, St, C = load(src)
    S = pd.read_csv(src / "set.csv.gz").set_index("id")
    ids = common(W, C, S)
    X = W.pivot_table(index="id", columns="tag", values="score")
    T = W.pivot_table(index="id", columns="tag", values="type", aggfunc="first")
    G = pd.read_parquet(src / "offroad_stop3.parquet")
    hum_off = set(G.index[G.off_pre | G.h_off_any])
    p("# NATIVE-GAP tables (native_gap_report.py)")
    p("")
    p(f"Gate: common held windows {len(ids)}; means " + ", ".join(f"{NAME[t]} {X.loc[ids, t].mean():.4f}" for t in TAGS) + "; decisions per window "
      f"{D.groupby(['tag', 'id']).size().median():.0f}; ego on the human track at k = 0: lateral median {D[D.k == 0].lat_e.median():.3f} m, "
      f"p99 {D[D.k == 0].lat_e.quantile(0.99):.3f} m; |v - vh_now| at k = 0 median {(D[D.k == 0].v - D[D.k == 0].vh_now).abs().median():.3f} m/s.")
    p("")
    part_a(D, ids, S)
    part_cf(src, D)
    per = part_b(D, X, T, ids, S, C)
    part_c(D, X, T, ids, S, C, hum_off, per)
    Path(a.out).write_text("\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
