"""EXPLORE-G report (Mac, .venv): nx_turns.py report --out DIR --res RESULTS_DIR [--dev DEVDIR]. DIR holds the box outputs (scenes / dec /
replay / lanes .parquet, bev.pkl, replay_identity_*.json); DEVDIR nx.py hist's dev-row plans (<tag>.npz). Writes RESULTS_DIR/explore_g.json
(every number of explore_g.md) and figures RESULTS_DIR/explore_g/*.png. Rules: nx_turns.py's docstring; the report adds (exploratory):
  pairs      J vs C on the same scenes: held RTB1O-L18-s0 / RTB1ARO-L18-s0 vs RSG-L18-s0, RTB1OB1-L18-s1 vs RSG-L18-s1; public the two s0 pairs.
             The bend (route, stretch) does not depend on the label, so every bend is paired.
  bins       |bend| 15-45 / 45-90 / >= 90 deg; r_bend < 10 / 10-20 / 20-40 / >= 40 m; w_mid < 3.3 / 3.3-4.0 / >= 4.0 m; "large narrow" =
             |bend| >= 60 and r_bend < 15 m; hard = outcome offroad | contact | opposite
  plan gain  decisions in [entry - 20, exit + 10] before the zero step with speed >= 1 m/s and req2 >= 0.15 rad toward the bend: g2 = plan
             heading change at 2 s / req2 (1 = the plan turns what the route asks over the plan's own arc; clipped to [-0.5, 2.5] for means).
             OLS y2 = req2 (b0 + b1 J + b2 (v - 5) + b3 J (v - 5)) + c out_lat, log-clustered bootstrap
  matched    per pair and bend, stations every 2 m over [entry - 20, exit + 10]: each label's nearest decision within 3 m (each decision pair
             once); speed-matched = |speed difference| <= 0.5 m/s
  same input replayed decisions (same window and filters on the stored plan's req2): J and C plans on the same (group, scene, decision);
             speed-free path geometry at fixed arcs: out_q = lateral offset from the route at arc q signed to the outside, lag_q = heading
             behind the route's turn at arc q (both + = turns too little)
  dev rows   nx.py hist plans (speed history x 1) of the 1 789 gate-4 dev rows (logged frames, the human's state): turning rows = human
             heading change at 4 s >= 15 deg and speed >= 1 m/s; at fixed arcs of both paths: lateral offset of the plan from the human path
             (outside +), heading lag; at 4 s (time-indexed): cross-track offset outside +
CIs: log-clustered percentile bootstrap (jevdrive.stats, 2 000 draws) of mean paired differences, log = scene id without its hash.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "research")]
from jevdrive import stats  # noqa: E402

C0, C1 = "RSG-L18-s0", "RSG-L18-s1"
SHORT = {"RSG-L18-s0": "RSG", "RTB1O-L18-s0": "RTB1", "RTB1ARO-L18-s0": "V2", "RSG-L18-s1": "RSG-s1", "RTB1OB1-L18-s1": "RTB1-s1"}
PAIRS = dict(held=[("RTB1O-L18-s0", C0), ("RTB1ARO-L18-s0", C0), ("RTB1OB1-L18-s1", C1)], pub=[("RTB1O-L18-s0", C0), ("RTB1ARO-L18-s0", C0)])
SEEDSET = {"RSG-L18-s0": "s0", "RTB1O-L18-s0": "s0", "RTB1ARO-L18-s0": "s0", "RSG-L18-s1": "s1", "RTB1OB1-L18-s1": "s1"}
HARD = ("offroad", "contact", "opposite")
OUTC = ("completed", "offroad", "contact", "opposite", "diverged", "zero_before", "incomplete", "untestable")
HB = [(15, 45), (45, 90), (90, 400)]
RB = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
WB = [(0, 3.3), (3.3, 4.0), (4.0, 99)]
QS = (5, 10, 15, 20)
NB = 2000
EXAMPLES = {"9a675656": "wide", "53d37873": "wide", "ddc1271e": "wide", "1f544304": "wide", "9d1d720d": "wide",
            "81a72614": "inside", "365c0937": "inside", "a7fa8bcc": "inside", "d33c6db3": "opposite", "a9609780": "opposite"}


def ci(x, g=None):
    r = stats.bootstrap(np.asarray(x, float), None if g is None else np.asarray(g), n_boot=NB)
    return [r["mean"], r["lo"], r["hi"], int(r["n"])]


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def ols_cluster(X, y, g, names):
    X, y, g = np.asarray(X, float), np.asarray(y, float), np.asarray(g)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y, g = X[ok], y[ok], g[ok]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    U, inv = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(U))]
    rng = np.random.default_rng(0)
    bs = np.array([np.linalg.lstsq(X[r], y[r], rcond=None)[0] for r in (np.concatenate([idx[i] for i in rng.integers(0, len(U), len(U))])
                                                                         for _ in range(NB))])
    return {n: [float(b[i]), float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))] for i, n in enumerate(names)} | \
        dict(n=int(len(y)), logs=int(len(U)))


def pname(j, c):
    return f"{SHORT[j]} vs {SHORT[c]}"


# ---------------------------------------------------------------- closed loop: outcomes by geometry
def pair_frame(S, st, j, c):
    a, b = S[(S.set == st) & (S.label == j)], S[(S.set == st) & (S.label == c)]
    return a.merge(b, on=["scene", "log"], suffixes=("_J", "_C"))


def bins_of(P):
    a, r, w = P.bend_deg_J.abs(), P.r_bend_J, P.w_mid_J
    sel = {"all bends": a >= 15}
    sel |= {f"bend {lo}-{hi if hi < 400 else 'inf'} deg": (a >= lo) & (a < hi) for lo, hi in HB}
    sel |= {f"r_bend {lo}-{hi if hi < 1e9 else 'inf'} m": (a >= 15) & (r >= lo) & (r < hi) for lo, hi in RB}
    sel |= {f"lane {lo}-{hi if hi < 99 else 'inf'} m": (a >= 15) & (w >= lo) & (w < hi) for lo, hi in WB}
    sel["large narrow (>= 60 deg, r_bend < 15 m)"] = (a >= 60) & (r < 15)
    return sel


def bins_table(S):
    res = {}
    for st, pairs in PAIRS.items():
        res[st] = {}
        for j, c in pairs:
            P = pair_frame(S, st, j, c)
            R = {}
            for name, m in bins_of(P).items():
                X = P[m]
                T = X[X.testable_J]
                r = dict(bends=int(len(X)), testable=int(len(T)), logs=int(X.log.nunique()))
                if not len(X):
                    R[name] = r
                    continue
                for nm, Y, fJ, fC in (("completed", T, T.completed_J.astype(float), T.completed_C.astype(float)),
                                      ("hard", X, X.outcome_J.isin(HARD).astype(float), X.outcome_C.isin(HARD).astype(float)),
                                      *((o, X, (X.outcome_J == o).astype(float), (X.outcome_C == o).astype(float)) for o in HARD + ("diverged",))):
                    r[nm] = dict(J=float(fJ.mean()) if len(Y) else None, C=float(fC.mean()) if len(Y) else None, nJ=int(fJ.sum()), nC=int(fC.sum()),
                                 diff=ci(fJ.to_numpy() - fC.to_numpy(), Y.log) if len(Y) else None)
                for v in ("max_out", "max_in", "max_lat"):
                    a_, b_ = X[f"{v}_J"].to_numpy(float), X[f"{v}_C"].to_numpy(float)
                    r[v] = dict(medJ=med(a_), medC=med(b_), diff=ci(a_ - b_, X.log))
                for v in ("v_entry", "v_mid"):
                    a_, b_, h_ = X[f"{v}_J"].to_numpy(float), X[f"{v}_C"].to_numpy(float), X[f"{v}_h_J"].to_numpy(float)
                    ok = np.isfinite(a_) & np.isfinite(b_)
                    r[v] = dict(J=float(np.nanmean(a_[ok])) if ok.any() else None, C=float(np.nanmean(b_[ok])) if ok.any() else None,
                                human=float(np.nanmean(h_[ok])) if ok.any() else None, diff=ci(a_ - b_, X.log))
                R[name] = r
            res[st][pname(j, c)] = R
    return res


def discordance(S):
    """Paired flips per pair: J-only vs C-only for a hard outcome in a bend, an offroad / zero rollout anywhere; exact two-sided sign test."""
    from math import comb
    sign_p = lambda a, b: float(min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))) if a + b else 1.0  # noqa: E731
    out = {}
    for st, pairs in PAIRS.items():
        for j, c in pairs:
            P = pair_frame(S, st, j, c)
            B_ = P[P.bend_deg_J.abs() >= 15]
            r = {}
            for nm, fj, fc in (("hard in a bend", B_.outcome_J.isin(HARD), B_.outcome_C.isin(HARD)),
                               ("offroad in a bend", B_.outcome_J == "offroad", B_.outcome_C == "offroad"),
                               ("opposite in a bend", B_.outcome_J == "opposite", B_.outcome_C == "opposite"),
                               ("contact in a bend", B_.outcome_J == "contact", B_.outcome_C == "contact"),
                               ("completed (testable)", B_.completed_J & B_.testable_J, B_.completed_C & B_.testable_C),
                               ("offroad rollout", P.offroad_J, P.offroad_C), ("zero rollout", P.score_J == 0, P.score_C == 0)):
                a, b = int((fj & ~fc).sum()), int((fc & ~fj).sum())
                r[nm] = dict(J=int(fj.sum()), C=int(fc.sum()), J_only=a, C_only=b, both=int((fj & fc).sum()), sign_p=sign_p(a, b))
            out[f"{st}: {pname(j, c)}"] = r
    return out


def outcome_counts(S):
    X = S[S.bend_deg.abs() >= 15]
    return {f"{st}/{SHORT[lab]}": {o: int((g.outcome == o).sum()) for o in OUTC} | dict(bends=int(len(g)), zeros=int((g.score == 0).sum()))
            for (st, lab), g in X.groupby(["set", "label"])}


# ---------------------------------------------------------------- closed loop: plan gain at the same speed
def decs(D):
    X = D[D.inbend & D.before & (D.v >= 1.0)].copy()
    X["req2"], X["req1"] = X.sg * (X.dhr2 - X.herr), X.sg * (X.dhr1 - X.herr)
    X["y2"], X["y1"] = X.sg * X.yaw2, X.sg * X.yaw1
    X["out_lat"] = -X.sg * X.lat
    X["pout2"], X["pout4"] = -X.sg * X.plat2, -X.sg * X.plat4
    X["g2"] = (X.y2 / X.req2).clip(-0.5, 2.5)
    X["log"] = X.scene.str.rsplit("-", n=1).str[0]
    X["ex1"] = X.sg * X.exec_dyaw1
    X["exg"] = (X.ex1 / X.y1).where(X.y1.abs() >= 0.05).clip(-0.5, 2.5)
    return X


def gain_reg(X):
    out = {}
    for st, pairs in PAIRS.items():
        for j, c in pairs:
            Y = X[(X.set == st) & X.label.isin((j, c)) & (X.req2 >= 0.15)]
            J = (Y.label == j).astype(float)
            v5 = Y.v - 5.0
            M = np.c_[Y.req2, Y.req2 * J, Y.req2 * v5, Y.req2 * J * v5, Y.out_lat]
            r = ols_cluster(M, Y.y2, Y.log, ["gain", "gain_J", "gain_per_ms", "gain_J_per_ms", "out_lat"])
            sp = {}
            for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 10), (10, 30)):
                Z = Y[(Y.v >= lo) & (Y.v < hi)]
                sp[f"{lo}-{hi}"] = {SHORT[l_]: dict(n=int((Z.label == l_).sum()), g2_med=med(Z.g2[Z.label == l_]), pout2_med=med(Z.pout2[Z.label == l_]))
                                    for l_ in (j, c)}
            lv = {SHORT[l_]: dict(n=int(len(z)), v_med=med(z.v), g2_med=med(z.g2), g2_mean=float(z.g2.mean()), pout2_med=med(z.pout2),
                                  deficit_med=med(z.deficit), exg_med=med(z.exg)) for l_, z in ((j, Y[Y.label == j]), (c, Y[Y.label == c]))}
            out[f"{st}: {pname(j, c)}"] = dict(reg=r, by_speed=sp, labels=lv)
    return out


def matched(X, S):
    rows = []
    for st, pairs in PAIRS.items():
        Sx = S[S.set == st].set_index(["label", "scene"])
        for j, c in pairs:
            A_, C_ = X[(X.set == st) & (X.label == j)], X[(X.set == st) & (X.label == c)]
            ga, gc = dict(tuple(A_.groupby("scene"))), dict(tuple(C_.groupby("scene")))
            for sc in sorted(set(ga) & set(gc)):
                b = Sx.loc[(j, sc)]
                if not np.isfinite(b.entry):
                    continue
                a_, c_, seen = ga[sc], gc[sc], set()
                for s_ in np.arange(b.entry - 20, b.exit + 10.01, 2.0):
                    ia, ic = (a_.st - s_).abs(), (c_.st - s_).abs()
                    if ia.min() > 3 or ic.min() > 3:
                        continue
                    ra, rc = a_.loc[ia.idxmin()], c_.loc[ic.idxmin()]
                    if (ra.k, rc.k) in seen:
                        continue
                    seen.add((ra.k, rc.k))
                    rows.append(dict(set=st, pair=pname(j, c), scene=sc, log=ra.log, s=s_, adeg=abs(b.bend_deg), r_bend=b.r_bend, vJ=ra.v, vC=rc.v,
                                     gJ=ra.g2 if ra.req2 >= 0.15 else np.nan, gC=rc.g2 if rc.req2 >= 0.15 else np.nan,
                                     poJ=ra.pout2, poC=rc.pout2, oJ=ra.out_lat, oC=rc.out_lat, exJ=ra.exg, exC=rc.exg, dfJ=ra.deficit, dfC=rc.deficit))
    M = pd.DataFrame(rows)
    out = {}
    for (st, pr), Y in M.groupby(["set", "pair"]):
        mm = Y[(Y.vJ - Y.vC).abs() <= 0.5]
        r = dict(points=int(len(Y)), scenes=int(Y.scene.nunique()), dv=ci(Y.vJ - Y.vC, Y.log), dg_all=ci(Y.gJ - Y.gC, Y.log),
                 dg_matched=ci(mm.gJ - mm.gC, mm.log), matched_points=int(len(mm)), matched_scenes=int(mm.scene.nunique()),
                 dpout_all=ci(Y.poJ - Y.poC, Y.log), dpout_matched=ci(mm.poJ - mm.poC, mm.log),
                 dout_all=ci(Y.oJ - Y.oC, Y.log), dout_matched=ci(mm.oJ - mm.oC, mm.log), dexg_matched=ci(mm.exJ - mm.exC, mm.log),
                 ddeficit_all=ci(Y.dfJ - Y.dfC, Y.log))
        Z = Y[np.isfinite(Y.oJ - Y.oC) & np.isfinite(Y.vJ - Y.vC)]
        if len(Z) > 20:
            r["dout_on_dv"] = ols_cluster(np.c_[np.ones(len(Z)), Z.vJ - Z.vC], Z.oJ - Z.oC, Z.log, ["intercept", "per_ms"])
        Z = Y[np.isfinite(Y.gJ - Y.gC)]
        if len(Z) > 20:
            r["dg_on_dv"] = ols_cluster(np.c_[np.ones(len(Z)), Z.vJ - Z.vC], Z.gJ - Z.gC, Z.log, ["intercept", "per_ms"])
        T = Y[(Y.adeg >= 60) & (Y.r_bend < 15)]
        tm = T[(T.vJ - T.vC).abs() <= 0.5]
        r["large_narrow"] = dict(points=int(len(T)), scenes=int(T.scene.nunique()), dv=ci(T.vJ - T.vC, T.log), dg_all=ci(T.gJ - T.gC, T.log),
                                 dg_matched=ci(tm.gJ - tm.gC, tm.log), matched_points=int(len(tm)), dout_all=ci(T.oJ - T.oC, T.log))
        out[f"{st}: {pr}"] = r
    return M, out


# ---------------------------------------------------------------- same inputs: the replayed zero rollouts
def olframe(O):
    O = O.copy()
    O["before"] = (O.kz < 0) | (O.k < O.kz)
    O["inbend"] = np.isfinite(O.entry) & (O.st >= O.entry - 20.0) & (O.st <= O.exit + 10.0)
    O["req2"] = O.sg * (O.dhr2 - O.herr)
    O["y2"] = O.sg * O.yaw2
    O["g2"] = (O.y2 / O.req2).clip(-0.5, 2.5)
    for q in QS:
        O[f"out{q}"], O[f"lag{q}"] = -O.sg * O[f"pa{q}"], -O.sg * O[f"ha{q}"]
    O["pout4"] = -O.sg * O.plat4
    O["log"] = O.scene.str.rsplit("-", n=1).str[0]
    return O


def same_input(O):
    O = olframe(O)
    st_ = O[O.ckpt == "stored"].set_index(["group", "scene", "k"])
    keep = st_[st_.inbend & st_.before & (st_.v >= 1.0) & (st_.req2 >= 0.15)].index
    out, frames = {}, {}
    for j, c in [p for p in PAIRS["held"]]:
        A = O[O.ckpt == j].set_index(["group", "scene", "k"])
        Cc = O[O.ckpt == c].set_index(["group", "scene", "k"])
        idx = keep.intersection(A.index).intersection(Cc.index)
        a, b = A.loc[idx], Cc.loc[idx]
        lg = a.log.to_numpy()
        frames[pname(j, c)] = pd.DataFrame(dict(group=[i[0] for i in idx], scene=[i[1] for i in idx], inputs=a.inputs.to_numpy(), log=lg, gJ=a.g2.to_numpy(), gC=b.g2.to_numpy(),
                                                vJ=a.v.to_numpy(), arcJ=a.arc2.to_numpy(), arcC=b.arc2.to_numpy(), adeg=a.bend_deg.abs().to_numpy(),
                                                r_bend=a.r_bend.to_numpy(), **{f"out{q}J": a[f"out{q}"].to_numpy() for q in QS},
                                                **{f"out{q}C": b[f"out{q}"].to_numpy() for q in QS},
                                                **{f"lag{q}J": a[f"lag{q}"].to_numpy() for q in QS}, **{f"lag{q}C": b[f"lag{q}"].to_numpy() for q in QS}))
        F = frames[pname(j, c)]

        def summ(F):
            return dict(decisions=int(len(F)), rollouts=int(F.group.str.cat(F.scene).nunique()), g2_medJ=med(F.gJ), g2_medC=med(F.gC),
                        dg2=ci(F.gJ - F.gC, F.log), darc2=ci(F.arcJ - F.arcC, F.log),
                        **{f"dout{q}": ci(F[f"out{q}J"] - F[f"out{q}C"], F.log) for q in QS},
                        **{f"dlag{q}_deg": ci(np.degrees(F[f"lag{q}J"] - F[f"lag{q}C"]), F.log) for q in QS},
                        out10_medJ=med(F.out10J), out10_medC=med(F.out10C), lag10_medJ_deg=med(np.degrees(F.lag10J)), lag10_medC_deg=med(np.degrees(F.lag10C)))
        r = dict(all=summ(F))
        for src, m in (("inputs J", F.inputs == j), ("inputs C", F.inputs == c), ("inputs other", ~F.inputs.isin((j, c))),
                       ("held", F.group.str.startswith("held")), ("pub", F.group.str.startswith("pub")),
                       ("large narrow", (F.adeg >= 60) & (F.r_bend < 15)), ("|bend| >= 45", F.adeg >= 45)):
            if m.sum() >= 5:
                r[src] = summ(F[m])
        out[pname(j, c)] = r
    return out, frames


# ---------------------------------------------------------------- same inputs: logged dev rows (the human's state and speed)
def path_at(P, q):
    """(n, 8, 3) poses from the rear axle -> (n, 3) x, y, yaw at arc q (nan beyond the 4 s pose)."""
    X = np.concatenate([np.zeros((len(P), 1, 3)), P[..., :3]], 1)
    a = np.concatenate([np.zeros((len(P), 1)), np.cumsum(np.hypot(*np.diff(X[..., :2], axis=1).transpose(2, 0, 1)), 1)], 1)
    o = np.full((len(P), 3), np.nan)
    for i in range(len(P)):
        if q <= a[i, -1]:
            o[i] = [np.interp(q, a[i], X[i, :, j]) for j in range(3)]
    return o


def dev_rows(devdir):
    H = {t: np.load(Path(devdir) / f"{t}.npz") for t in SHORT if (Path(devdir) / f"{t}.npz").exists()}
    if C0 not in H:
        return None, None
    common = sorted(set.intersection(*(set(h["names"].tolist()) for h in H.values())))
    pos = {t: {n: i for i, n in enumerate(h["names"].tolist())} for t, h in H.items()}
    ix = {t: np.array([pos[t][n] for n in common]) for t in H}                    # align every tag on the reference's row names
    ref = H[C0]
    fut, spd, log = ref["fut"][ix[C0]].astype(float), ref["speed"][ix[C0]].astype(float), ref["log"][ix[C0]]
    th = fut[:, 7, 2]
    turn = (np.abs(th) >= np.radians(15)) & (spd >= 1.0)
    sg = np.sign(th)
    hum = {q: path_at(fut, q) for q in QS}
    nrm = lambda y: np.c_[-np.sin(y), np.cos(y)]                                   # noqa: E731
    per = {}
    for t, h in H.items():
        P = h["plan_1.0"][ix[t]].astype(float)
        d = {}
        for q in QS:
            p = path_at(P, q)
            d[f"out{q}"] = -sg * ((p[:, :2] - hum[q][:, :2]) * nrm(hum[q][:, 2])).sum(1)
            d[f"lag{q}"] = -sg * (p[:, 2] - hum[q][:, 2])
        d["xt4"] = -sg * ((P[:, 7, :2] - fut[:, 7, :2]) * nrm(fut[:, 7, 2])).sum(1)
        d["arc4"] = np.hypot(P[:, 0, 0], P[:, 0, 1]) + np.hypot(*np.diff(P[..., :2], axis=1).transpose(2, 0, 1)).sum(1)
        d["ade"] = np.hypot(*(P[..., :2] - fut[..., :2]).transpose(2, 0, 1)).mean(1)
        per[t] = d
    harc4 = np.hypot(fut[:, 0, 0], fut[:, 0, 1]) + np.hypot(*np.diff(fut[..., :2], axis=1).transpose(2, 0, 1)).sum(1)
    out = dict(rows=int(len(fut)), rows_per_tag={SHORT[t]: int(len(h["names"])) for t, h in H.items()}, logs=int(len(set(log))), turning=int(turn.sum()), turning_logs=int(len(set(log[turn]))),
               turn_deg_med=float(np.degrees(np.median(np.abs(th[turn])))), human_arc4_med=float(np.median(harc4[turn])),
               labels={SHORT[t]: dict(ade_turn=float(np.nanmean(d["ade"][turn])), ade_all=float(np.nanmean(d["ade"])),
                                      arc4_turn_med=float(np.nanmedian(d["arc4"][turn])), xt4_med=float(np.nanmedian(d["xt4"][turn])),
                                      **{f"out{q}_med": float(np.nanmedian(d[f"out{q}"][turn])) for q in QS},
                                      **{f"lag{q}_med_deg": float(np.degrees(np.nanmedian(d[f"lag{q}"][turn]))) for q in QS}) for t, d in per.items()})
    pairs = {}
    sharp = turn & (np.abs(th) >= np.radians(45))
    out["sharp_rows"], out["sharp_logs"] = int(sharp.sum()), int(len(set(log[sharp])))
    for j, c in PAIRS["held"]:
        if j in per and c in per:
            a, b = per[j], per[c]
            for nm, m in (("turning", turn), ("sharp >= 45 deg", sharp)):
                pairs[f"{pname(j, c)}, {nm}"] = dict(dade=ci((a["ade"] - b["ade"])[m], log[m]), dxt4=ci((a["xt4"] - b["xt4"])[m], log[m]),
                                                     darc4=ci((a["arc4"] - b["arc4"])[m], log[m]),
                                                     **{f"dout{q}": ci((a[f"out{q}"] - b[f"out{q}"])[m], log[m]) for q in QS},
                                                     **{f"dlag{q}_deg": ci(np.degrees(a[f"lag{q}"] - b[f"lag{q}"])[m], log[m]) for q in QS})
            pairs[f"{pname(j, c)}, straight"] = dict(dade=ci((a["ade"] - b["ade"])[~turn & (spd >= 1)], log[~turn & (spd >= 1)]))
    out["pairs"] = pairs
    return out, dict(turn=turn, per=per, th=th)


# ---------------------------------------------------------------- lane lines, route bias
def lane_sample(L, S):
    """ctrl decisions (k >= 2) and replayed own-checkpoint decisions >= 3 s before the zero step."""
    kz = S.set_index(["set", "label", "scene"]).kz
    L = L.copy()
    L["kz"] = [(-1 if g == "ctrl" else int(kz.get((g.split("_", 1)[0], g.split("_", 1)[1], sc), -1))) for g, sc in zip(L.group, L.scene)]
    keep = ((L.group == "ctrl") & (L.k >= 2)) | ((L.group != "ctrl") & (L.k >= 2) & ((L.kz < 0) | (L.k <= L.kz - 6)))
    return L[keep]


def lane_acc(L):
    out = {}
    L = L.copy()
    L["present"] = (L.P0_p1 > 0.5) & (L.P0_p2 > 0.5)
    for samp, Z0 in (("all", L), ("ctrl", L[L.group == "ctrl"]), ("zero rollouts", L[L.group != "ctrl"])):
        o = {}
        for seg in ("straight", "irregular", "junction", "none"):
            Z = Z0[Z0.seg3 == seg]
            r = dict(decisions=int(len(Z)), scenes=int(Z.scene.nunique()), present=float(Z.present.mean()) if len(Z) else None)
            if seg != "none" and len(Z):
                for d in (10, 20, 30):
                    c = Z[f"c{d}"]
                    for H in ("P0", "FT"):
                        for nm, col in (("raw", f"{H}_c{d}"), ("scaled", f"{H}_cs{d}")):
                            e = Z[col] - c
                            ep = e[Z.present]
                            r[f"{H}_{nm}_{d}"] = dict(n=int(e.notna().sum()), med_abs=float(e.abs().median()), p90_abs=float(e.abs().quantile(0.9)),
                                                     med=float(e.median()), share_lt_0p5=float((e.abs() < 0.5).mean()),
                                                     present_med_abs=float(ep.abs().median()) if len(ep) else None,
                                                     present_share_lt_0p5=float((ep.abs() < 0.5).mean()) if len(ep) else None)
                    r[f"width_ratio_{d}"] = float((Z[f"P0_w{d}"] / Z[f"w{d}"]).median())
                    r[f"width_ratio_scaled_{d}"] = float((Z[f"P0_ws{d}"] / Z[f"w{d}"]).median())
                    r[f"map_width_med_{d}"] = float(Z[f"w{d}"].median())
                # heading of the lane read (centre at 30 m - 10 m over 20 m) against the map's
                eh = np.degrees(np.arctan2(Z.P0_cs30 - Z.P0_cs10, 20.0) - np.arctan2(Z.c30 - Z.c10, 20.0))
                r["heading_err_deg"] = dict(med_abs=float(eh.abs().median()), p90_abs=float(eh.abs().quantile(0.9)))
            o[seg] = r
        out[samp] = o
    return out


def route_bias(D):
    X = D[(D.seg == "straight") & D.before].copy()
    out = {}
    for d in (10, 20, 30):
        sg = np.sign(X[f"rc{d}"] - X[f"c{d}"])
        for nm, col in (("route_mem", f"ry{d}"), ("route_rebuilt", f"gy{d}"), ("plan", f"py{d}")):
            X[f"{nm}_b{d}"] = (X[col] - X[f"c{d}"]) * sg
    X["log"] = X.scene.str.rsplit("-", n=1).str[0]
    for (st, lab), Y in X.groupby(["set", "label"]):
        r = {}
        for d in (10, 20, 30):
            for nm in ("route_mem", "route_rebuilt", "plan"):
                v = Y[f"{nm}_b{d}"]
                r[f"{nm}_{d}"] = dict(n=int(v.notna().sum()), med=med(v), mean=ci(v, Y.log), gt0p5=float((v > 0.5).mean()))
        out[f"{st}/{SHORT[lab]}"] = r
    return out


# ---------------------------------------------------------------- the user's example scenes
def examples(S, X, O):
    out = []
    Ofr = olframe(O) if O is not None else None
    for pre, kind in EXAMPLES.items():
        Z = S[S.scene.str.rsplit("-", n=1).str[1].str.startswith(pre)]
        for (st, sc), g in Z.groupby(["set", "scene"]):
            for _, r in g.sort_values("label").iterrows():
                x = X[(X.set == st) & (X.label == r.label) & (X.scene == sc)]
                o = dict(example=pre, kind=kind, set=st, label=SHORT[r.label], score=float(r.score), outcome=r.outcome, z_cause=r.z_cause,
                         bend_deg=float(r.bend_deg), r_bend=float(r.r_bend), w_mid=float(r.get("w_mid", np.nan)), v_entry=float(r.v_entry),
                         v_entry_h=float(r.v_entry_h), max_out=float(r.max_out), max_in=float(r.max_in), opp=bool(r.opp),
                         g2_med=med(x.g2[x.req2 >= 0.15]), n_dec=int((x.req2 >= 0.15).sum()))
                if Ofr is not None:
                    w = Ofr[(Ofr.scene == sc) & (Ofr.inputs == r.label) & Ofr.inbend & Ofr.before & (Ofr.v >= 1)]
                    if len(w):
                        o["same_input_g2_med"] = {SHORT.get(ck, ck): med(z.g2[z.req2 >= 0.15]) for ck, z in w.groupby("ckpt")}
                        o["same_input_out10_med"] = {SHORT.get(ck, ck): med(z.out10) for ck, z in w.groupby("ckpt")}
                out.append(o)
    return out


# ---------------------------------------------------------------- main
def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PS
    PS.apply()
    d, res = Path(a.out), Path(a.res)
    fd = res / "explore_g"
    fd.mkdir(parents=True, exist_ok=True)
    S, D = pd.read_parquet(d / "scenes.parquet"), pd.read_parquet(d / "dec.parquet")
    for c in ("testable", "completed", "offroad", "opp"):
        S[c] = S[c].astype("boolean").fillna(False).astype(bool)
    D["log"] = D.scene.str.rsplit("-", n=1).str[0]
    out = dict(counts={f"{s}/{SHORT[l]}": int(len(g)) for (s, l), g in S.groupby(["set", "label"])},
               bends={f"{s}/{SHORT[l]}": dict(bends=int((g.bend_deg.abs() >= 15).sum()), testable=int(g.testable.sum())) for (s, l), g in S.groupby(["set", "label"])},
               geo_errors=len(json.loads((d / "geo_errors.json").read_text())) if (d / "geo_errors.json").exists() else None)
    out["outcomes"] = outcome_counts(S)
    out["discordance"] = discordance(S)
    out["bins"] = bins_table(S)
    X = decs(D)
    out["gain"] = gain_reg(X)
    M, out["matched"] = matched(X, S)
    out["replay_identity"] = {f.stem.replace("replay_identity_", ""): json.loads(f.read_text()) for f in sorted(d.glob("replay_identity_*.json"))}
    O = pd.read_parquet(d / "replay.parquet") if (d / "replay.parquet").exists() else None
    F = {}
    if O is not None:
        out["same_input"], F = same_input(O)
    dv, DV = dev_rows(a.dev) if a.dev else (None, None)
    out["dev_rows"] = dv
    if (d / "lanes.parquet").exists():
        L = lane_sample(pd.read_parquet(d / "lanes.parquet"), S)
        out["lanes"] = lane_acc(L)
    else:
        L = None
    out["route_bias"] = route_bias(D)
    out["examples"] = examples(S, X, O)
    (res / "explore_g.json").write_text(json.dumps(out, indent=1, default=float))

    # ---- figures
    col = {"RSG": PS.PALETTE["vermillion"], "RTB1": PS.PALETTE["blue"], "V2": PS.PALETTE["green"], "RSG-s1": PS.PALETTE["orange"],
           "RTB1-s1": PS.PALETTE["sky_blue"], "human": PS.BASELINE}
    # F1: outcome by bend angle (held + public), the three s0 labels
    fig, ax = plt.subplots(1, 3, figsize=(PS.DOUBLE_COLUMN_IN, 2.2))
    Sb = S[S.bend_deg.abs() >= 15].copy()
    Sb["abin"] = pd.cut(Sb.bend_deg.abs(), [15, 45, 90, 400], right=False, labels=["15-45°", "45-90°", "≥90°"])
    labs = ("RSG-L18-s0", "RTB1O-L18-s0", "RTB1ARO-L18-s0")
    xx = np.arange(3)
    for i, (nm, f, tt) in enumerate((("hard", lambda g: g.outcome.isin(HARD).mean(), "offroad / contact / opposite in the bend"),
                                     ("completed", lambda g: g[g.testable].completed.mean(), "completed (testable bends)"),
                                     ("v", lambda g: g.v_entry.mean(), "speed 10 m before entry (m/s)"))):
        for k_, l_ in enumerate(labs):
            Z = Sb[Sb.label == l_]
            ax[i].bar(xx + (k_ - 1) * 0.22, [f(Z[Z.abin == b]) for b in Z.abin.cat.categories], 0.22, color=col[SHORT[l_]], label=SHORT[l_])
        if nm == "v":
            Z = Sb[Sb.label == C0]
            ax[i].plot(xx, [Z[Z.abin == b].v_entry_h.mean() for b in Z.abin.cat.categories], "o", color="k", ms=3, label="human")
        ax[i].set_xticks(xx, ["15-45°", "45-90°", "≥90°"])
        ax[i].set_title(tt, fontsize=7.5)
        PS.bars(ax[i])
    ax[0].legend(fontsize=6.5)
    ax[2].legend(fontsize=6.5)
    fig.text(0.5, 0.005, "bend angle (held 460 @18 + public 1491 @10 pooled)", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(fd / "outcomes.png", dpi=200)
    plt.close(fig)
    # F2: plan gain vs speed (closed loop), station-matched dg vs dv, same-input gains, dev rows
    fig, ax = plt.subplots(1, 4, figsize=(PS.DOUBLE_COLUMN_IN, 2.0))
    Y = X[X.label.isin(labs) & (X.req2 >= 0.15)]
    for l_ in labs:
        z = Y[Y.label == l_]
        bb = np.arange(1, 13, 2)
        ax[0].plot(bb + 1, [z.g2[(z.v >= lo) & (z.v < lo + 2)].median() for lo in bb], "-o", ms=2.5, color=col[SHORT[l_]], label=SHORT[l_])
    ax[0].axhline(1, color="k", lw=0.5, ls=":")
    ax[0].set_xlabel("ego speed (m/s)")
    ax[0].set_ylabel("plan turn / route turn (2 s)")
    ax[0].set_title("(a) closed loop, bend decisions", fontsize=7.5)
    ax[0].legend(fontsize=6)
    for pr, cc in (("RTB1 vs RSG", col["RTB1"]), ("V2 vs RSG", col["V2"])):
        Mh = M[M.pair == pr]
        ax[1].scatter(Mh.vJ - Mh.vC, (Mh.gJ - Mh.gC).clip(-1.5, 1.5), s=2, color=cc, alpha=0.4, lw=0, label=pr)
    ax[1].axvline(0, color="k", lw=0.5)
    PS.zero_line(ax[1])
    ax[1].set_xlabel("speed J − RSG, same station (m/s)")
    ax[1].set_ylabel("plan gain J − RSG")
    ax[1].set_title("(b) station-matched", fontsize=7.5)
    ax[1].legend(fontsize=5.5, markerscale=3)
    for pr, cc in (("RTB1 vs RSG", col["RTB1"]), ("V2 vs RSG", col["V2"])):
        if pr in F:
            Fp = F[pr]
            ax[2].scatter(Fp.out10C.clip(-2, 2), Fp.out10J.clip(-2, 2), s=3, color=cc, alpha=0.5, lw=0, label=pr)
    ax[2].plot([-2, 2], [-2, 2], color="k", lw=0.5)
    ax[2].set_xlabel("RSG: outside at 10 m (m)")
    ax[2].set_ylabel("J: outside at 10 m (m)")
    ax[2].set_title("(c) same inputs (replay)", fontsize=7.5)
    ax[2].legend(fontsize=5.5, markerscale=3)
    if DV is not None:
        t = DV["turn"]
        for l_ in labs:
            if l_ in DV["per"]:
                v = DV["per"][l_]["out10"][t]
                v = np.sort(v[np.isfinite(v)])
                ax[3].plot(v, np.linspace(0, 1, len(v)), color=col[SHORT[l_]], label=SHORT[l_])
        ax[3].axvline(0, color="k", lw=0.5)
        ax[3].set_xlim(-2, 2)
        ax[3].set_xlabel("outside the human path, 10 m (m)")
        ax[3].set_ylabel("cumulative share")
        ax[3].set_title(f"(d) dev rows, {int(t.sum())} turning", fontsize=7.5)
        ax[3].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(fd / "gain.png", dpi=200)
    plt.close(fig)
    # F3: BEV of the user's example scenes (RSG, RTB1, V2 controller traces; route dashed)
    bev = pickle.load(open(d / "bev.pkl", "rb"))
    keys = []
    for pre in EXAMPLES:
        ks = [k for k in bev if k[2].rsplit("-", 1)[1].startswith(pre)]
        st = "held" if any(k[0] == "held" for k in ks) else "pub"
        if ks:
            keys.append((pre, st, [k for k in ks if k[0] == st]))
    n = len(keys)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(PS.DOUBLE_COLUMN_IN, 3.6))
    for x, (pre, st, ks) in zip(axs.flat, keys):
        ref = bev[next((k for k in ks if k[1] == C0), ks[0])]
        for e in ref["edges"]:
            x.plot(e[:, 0], e[:, 1], color="#CCCCCC", lw=0.35)
        for e in ref["road"]:
            x.plot(e[:, 0], e[:, 1], color="k", lw=0.5)
        x.plot(ref["route"][:, 0], ref["route"][:, 1], color=PS.BASELINE, lw=1.0, ls="--")
        for l_ in labs:
            k = next((k for k in ks if k[1] == l_), None)
            if k is None or bev[k]["ego"] is None:
                continue
            E_ = bev[k]["ego"]
            w = E_[:, 0] <= min(bev[k]["tz"], E_[-1, 0])
            x.plot(E_[w, 1], E_[w, 2], color=col[SHORT[l_]], lw=1.1, label=SHORT[l_])
            if np.isfinite(bev[k]["tz"]):
                x.plot(E_[w, 1][-1:], E_[w, 2][-1:], "x", color=col[SHORT[l_]], ms=4)
        A_ = ref["anchors"]
        lo, hi = A_[:, :2].min(0) - 12, A_[:, :2].max(0) + 12
        span = max(hi - lo) / 2
        cc = (lo + hi) / 2
        x.set_xlim(cc[0] - span, cc[0] + span)
        x.set_ylim(cc[1] - span, cc[1] + span)
        x.set_aspect("equal")
        x.set_xticks([]), x.set_yticks([])
        x.set_title(f"{pre} ({st}, {EXAMPLES[pre]})", fontsize=6.5)
    for x in list(axs.flat)[n:]:
        x.axis("off")
    axs.flat[0].legend(fontsize=5.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(fd / "bev_examples.png", dpi=220)
    plt.close(fig)
    # F4: lane-centre error by segment class and range
    if L is not None:
        fig, ax = plt.subplots(1, 3, figsize=(PS.DOUBLE_COLUMN_IN, 2.1))
        for i, seg in enumerate(("straight", "irregular", "junction")):
            r = out["lanes"]["all"][seg]
            xx = np.arange(3)
            ax[i].bar(xx - 0.2, [r[f"P0_raw_{dd}"]["med_abs"] for dd in (10, 20, 30)], 0.4, color=PS.BASELINE, label="raw")
            ax[i].bar(xx + 0.2, [r[f"P0_scaled_{dd}"]["med_abs"] for dd in (10, 20, 30)], 0.4, color=PS.PALETTE["blue"], label="scale undone")
            ax[i].errorbar(xx + 0.2, [r[f"P0_scaled_{dd}"]["med_abs"] for dd in (10, 20, 30)],
                           yerr=[np.zeros(3), [r[f"P0_scaled_{dd}"]["p90_abs"] - r[f"P0_scaled_{dd}"]["med_abs"] for dd in (10, 20, 30)]],
                           fmt="none", color="k", lw=0.6)
            ax[i].set_xticks(xx, ["10 m", "20 m", "30 m"])
            ax[i].set_title(f"{seg} (n {r['decisions']}, both lines p>0.5: {r['present']:.2f})", fontsize=7)
            PS.bars(ax[i])
        ax[0].set_ylabel("abs. lane-centre error (m): median, p90")
        for x in ax:
            x.set_yscale("log")
            x.set_ylim(0.05, 60)
        ax[0].legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(fd / "lane_err.png", dpi=200)
        plt.close(fig)
    print(json.dumps({k: out[k] for k in ("counts", "outcomes", "replay_identity")}, indent=1, default=float)[:4000])
