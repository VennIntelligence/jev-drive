#!/usr/bin/env python3
"""Lane SWV1, parts B and C (plans/2026-10-10-swv1-prereg.md).

B: kinematic counterfactuals on the executed path with recorded object motion: lateral-acceleration cap (non-privileged), lead-sweep stop
   (non-privileged: the plan's swept ego box against a box at the model's own lead output), footprint-aware stop (privileged ceiling:
   the simulator's boxes), the shipped weights' plan on the same tokens, the logged human path (COL1's columns).
C: per-decision non-privileged signals against the label "the served plan's sweep intersects an object box".

  swv1_bc.py [--dom nuplan pai]   (after swv1_a.py)  -> results/swerve/{cf_<dom>.csv, signals_<dom>.csv, bc_<dom>.md}, tmp/swv1/sig_<dom>.pkl
Simulator boxes, the logged path and recorded object motion are labels / oracle inputs; the arms that use them are ceilings."""
import argparse
import csv
import pickle
from collections import defaultdict

import numpy as np
import shapely

import swv1_lib as S

CL = S.CL
OUT = S.RES / "swerve"
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
ALIMS = (1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0)
A_BRAKE_CAP, A_STOP, A_UP, MARGIN = 2.5, 4.0, 2.0, 0.5
SPEED_BINS = np.array([1.0, 3.0, 6.0, 10.0])
sig = lambda z: 1 / (1 + np.exp(-np.clip(np.asarray(z, float), -30, 30)))  # noqa: E731


# ---------------------------------------------------------------- model heads per decision
def heads(r, i):
    """-> {"FT": {...}, "P0": {...}, "cam": (x, y)} raw heads of decision i, or None."""
    rp = r.rp
    if rp is None:
        return None
    if r.dom == "nuplan":
        j = next((j for j, x in enumerate(rp) if x["now"] == r.now[i]), None)
        return None if j is None else dict(FT={k: v.astype(np.float32) for k, v in rp[j]["FT"].items()}, P0={k: v.astype(np.float32) for k, v in rp[j]["P0"].items()}, cam=rp[j]["cam"])
    j = np.flatnonzero(rp["now"] == r.now[i])
    if not len(j):
        return None
    j = j[0]
    return dict(FT=dict(lead=rp["lead_ft"][j], lead_prob=rp["lp_ft"][j]), P0=dict(lead=rp["lead_p0"][j], lead_prob=rp["lp_p0"][j]),
                cam=np.asarray(r.o["start"]["cam_t"][:2], float))


def lead_sweep(r, i, h, src):
    """The served plan's swept ego box against a 4.8 x 2.0 m box at the model's own lead output (selection 0 at t = 0: x from the camera to
    the lead's rear, y right-positive, moved along the ego heading at the reported lead speed). -> (p, bumper gap d, v_l, y_left, clr, first
    intersecting sample or -1)."""
    ld = h[src]["lead"][:72].reshape(3, 6, 4)
    p = float(sig(h[src]["lead_prob"][0]))
    x0, y0, vl = float(ld[0, 0, 0]), float(ld[0, 0, 1]), max(float(ld[0, 0, 2]), 0.0)
    cam = h["cam"]
    plan = r.plan[i]
    e0 = plan[0]
    lx = cam[0] + x0 + S.LEAD_BOX[0] / 2 + vl * S.TAU
    ly = np.full(41, cam[1] - y0)
    c, s = np.cos(e0[2]), np.sin(e0[2])
    lw = np.c_[e0[0] + c * lx - s * ly, e0[1] + s * lx + c * ly, np.full(41, e0[2])]
    d = shapely.distance(S.boxes(S.centre(plan, r.off), r.Le, r.We), S.boxes(lw, *S.LEAD_BOX))
    hit = np.flatnonzero(d <= 0)
    return p, x0 - (r.off + r.Le / 2 - cam[0]), vl, float(cam[1] - y0), float(d.min()), int(hit[0]) if len(hit) else -1


def geo_signals(r, i, h):
    """Signals from lane lines, road edges, meta, desire, plan spread (nuPlan replay only)."""
    out = {}
    plan = r.plan[i]
    cs = np.concatenate([CL.into(plan[0], CL.corners(*p, r.Le, r.We)) for p in S.centre(plan, r.off)[::2]])       # ego-box corners in the ego frame
    cs = cs[cs[:, 0] > 0]
    cam = h["cam"]
    for src in ("P0", "FT"):
        H = h[src]
        if "road_edges" not in H:
            return out
        re = H["road_edges"][:132].reshape(2, 33, 2)[..., 0]
        ex = X_IDXS + cam[0]
        yl, yr = cam[1] - re[0], cam[1] - re[1]
        if len(cs):
            out[f"edge_margin_{src}"] = -float(min(np.min(np.interp(cs[:, 0], ex, yl) - cs[:, 1]), np.min(cs[:, 1] - np.interp(cs[:, 0], ex, yr))))
            ll = H["lane_lines"][:264].reshape(4, 33, 2)[..., 0]
            lp = sig(H["lane_lines_prob"])[1::2]
            exc = [-5.0]
            if lp[1] >= 0.5:
                exc.append(float(np.max(cs[:, 1] - np.interp(cs[:, 0], ex, cam[1] - ll[1]))))
            if lp[2] >= 0.5:
                exc.append(float(np.max(np.interp(cs[:, 0], ex, cam[1] - ll[2]) - cs[:, 1])))
            out[f"lane_exc_{src}"] = max(exc)
        else:
            out[f"edge_margin_{src}"], out[f"lane_exc_{src}"] = -5.0, -5.0
        m = sig(H["meta"])
        out[f"meta_brake_{src}"] = float(max(m[4:31:6].max(), m[5:31:6].max(), m[6:31:6].max()))
        out[f"meta_dis_{src}"] = float(max(m[1:31:6].max(), m[2:31:6].max()))
        out[f"meta_steer_{src}"] = float(m[3:31:6].max())
        out[f"meta_brakepress_{src}"] = float(m[32:55:4].max())
        ds = np.exp(H["desire_state"] - H["desire_state"].max())
        ds /= ds.sum()
        dp = H["desire_pred"].reshape(4, 8)
        dp = np.exp(dp - dp.max(1, keepdims=True))
        dp /= dp.sum(1, keepdims=True)
        out[f"desire_lc_{src}"] = float(max(ds[3], ds[4], dp[:, 3].max(), dp[:, 4].max()))
        sd = np.exp(np.minimum(H["plan"][495:].reshape(33, 15)[:, 1], 11))
        out[f"plan_std2_{src}"], out[f"plan_std4_{src}"] = float(np.interp(2.0, T_IDXS, sd)), float(np.interp(4.0, T_IDXS, sd))
    return out


def signals(R, rows):
    """Decision rows (swv1_a.py) + the non-privileged signals."""
    by = {(r.set, r.scene): r for r in R}
    out = []
    for d in rows:
        r = by[d["set"], d["scene"]]
        h = heads(r, d["k"])
        if h is None:
            continue
        d = dict(d)
        for src in ("P0", "FT"):
            p, gap, vl, yl, clr, j = lead_sweep(r, d["k"], h, src)
            cum = np.r_[0, np.cumsum(np.hypot(*np.diff(r.plan[d["k"]][:, :2], axis=0).T))]
            d[f"lead_p_{src}"], d[f"lead_d_{src}"], d[f"lead_v_{src}"], d[f"lead_y_{src}"] = p, gap, vl, yl
            d[f"lead_sweep_{src}"] = -(clr if p >= 0.5 else 20.0)
            d[f"lead_hit_{src}"] = bool(p >= 0.5 and j >= 0)
            d[f"lead_dstar_{src}"] = float(cum[j]) if p >= 0.5 and j >= 0 else None
            d[f"lead_need_{src}"] = float(max(d["v_e"] - vl, 0.0) ** 2 / (2 * max(gap - 4.0, 0.5))) if p >= 0.5 else 0.0
            d[f"lead_near_{src}"] = -gap if p >= 0.5 else -100.0
        d.update(geo_signals(r, d["k"], h))
        d["base_len"] = d["arc4"] / max(d.get("arc4_p0", np.nan), 0.5)
        d["abs_dpsi4"] = abs(d["dpsi4"])
        d["frozen_ref"] = -d["clr_frozen"]
        out.append(d)
    return out


# ---------------------------------------------------------------- B: kinematic counterfactuals
def counterfactual(r, D, arm, a_lim=3.0, src="P0", margin=MARGIN):
    """Re-time the executed path. D: this rollout's decision rows with signals (time order). arm: "alat" | "lead" | "priv".
    -> dict(contact with the struck object, avoided (strict), deferred, stopped, progress loss, triggered)."""
    if not D:
        return None
    t = np.arange(r.T0 + D[0]["now"] * 1e6, r.T1 + 1, 1e5)
    rig = r.ego_rig(t)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(rig[:, :2], axis=0).T))]
    v = np.gradient(s, 0.1) if len(s) > 1 else np.zeros(1)
    su, iu = np.unique(np.maximum.accumulate(s), return_index=True)
    v_arc = lambda x: float(np.interp(x + 0.5, su, v[iu]))  # noqa: E731  (original speed on first reaching the arc 0.5 m ahead: no standstill trap)
    now = np.array([r.T0 + d["now"] * 1e6 for d in D])
    s_k = np.interp(now, t, s)
    kin = [S.plan_kin(r.plan[d["k"]]) for d in D]

    def allowed(sp, ti):
        if arm == "alat":
            k = max(int(np.searchsorted(s_k, sp + 1e-6, "right")) - 1, 0)
            vs, kap = kin[k][0], kin[k][1]
            dj = np.r_[0, np.cumsum(vs * 0.5)][:-1] - (sp - s_k[k])
            cap = np.sqrt(a_lim / np.maximum(np.abs(kap), 1e-6))
            ok = dj + vs * 0.5 > 0
            return float(np.min(np.sqrt(cap[ok] ** 2 + 2 * A_BRAKE_CAP * np.maximum(dj[ok], 0)))) if ok.any() else np.inf
        k = max(int(np.searchsorted(now, ti, "right")) - 1, 0)
        ds = D[k].get(f"lead_dstar_{src}") if arm == "lead" else (D[k]["di_any"] if D[k]["hit_any"] else None)
        if ds is None:
            return np.inf
        return float(np.sqrt(2 * A_STOP * max(s_k[k] + ds - margin - sp, 0.0)))

    sp, vp, trig = 0.0, float(v[0]), False
    S_, V_ = [0.0], [vp]
    dec = A_BRAKE_CAP if arm == "alat" else A_STOP
    for i in range(1, len(t)):
        lag = sp < s[i - 1] - 0.05
        free = max(v[i], v_arc(sp)) if lag else float(v[i])            # unconstrained: the original timing; behind it: catch up along the same path
        al = allowed(sp, t[i - 1])
        if al < free - 0.05:
            trig = True
        vc = min(free, al)
        vp = vc if not lag and al >= free else float(np.clip(vc, vp - dec * 0.1, vp + A_UP * 0.1))
        vp = max(vp, 0.0)
        sp = min(sp + vp * 0.1, s[i])
        S_.append(sp), V_.append(vp)
    S_ = np.array(S_)
    pose = np.c_[np.interp(S_, s, rig[:, 0]), np.interp(S_, s, rig[:, 1]), np.interp(S_, s, np.unwrap(rig[:, 2]))]
    res = dict(trig=bool(trig), loss=float(1 - S_[-1] / s[-1]) if s[-1] > 0.5 else 0.0, loss_m=float(s[-1] - S_[-1]), v_end=float(V_[-1]))
    res["lose10"] = bool(res["loss"] > 0.10 and res["loss_m"] > 0.5)
    if r.struck:
        k = r.struck
        eb = S.boxes(S.centre(pose, r.off), r.Le, r.We)
        p, ok = r.obj_pose(k, t)
        hit = shapely.intersects(eb, S.boxes(p, *r.size(k))) & ok & (t <= r.T1)
        res["contact"] = bool(hit.any())
        res["hit_standing"] = bool(hit.any() and V_[int(np.argmax(hit))] <= 0.3)      # touched while standing: the object drove into the ego
        stopped = V_[-1] <= 0.3
        s_ev = float(np.interp(r.t_ev, t, s))
        rest = np.linspace(S_[-1], max(s_ev + 1.0, S_[-1]), 40)
        rp_ = np.c_[np.interp(rest, s, rig[:, 0]), np.interp(rest, s, rig[:, 1]), np.interp(rest, s, np.unwrap(rig[:, 2]))]
        last = S.boxes(r.obj_pose(k, r.T1)[0], *r.size(k))[0]
        later = bool(shapely.intersects(S.boxes(S.centre(rp_, r.off), r.Le, r.We), last).any())
        res["deferred"] = bool(not res["contact"] and not stopped and later)
        res["avoided"] = bool(not res["contact"] and not res["deferred"])
        res["stopped"] = bool(stopped)
    return res


# ---------------------------------------------------------------- C: separation
def speed_w(v_pos, v_neg):
    """Weights for the negatives so that their ego-speed histogram matches the positives'."""
    bp, bn = np.digitize(v_pos, SPEED_BINS), np.digitize(v_neg, SPEED_BINS)
    w = np.zeros(len(v_neg))
    for b in range(5):
        if (bn == b).any():
            w[bn == b] = (bp == b).mean() / (bn == b).mean()
    return w


def read_auc(pos, neg, name, n_boot=1000):
    """-> dict(auc, lo, hi, thr10, n_pos, n_neg) for signal `name` (rows lacking it are dropped)."""
    pos, neg = [d for d in pos if d.get(name) is not None and np.isfinite(d[name])], [d for d in neg if d.get(name) is not None and np.isfinite(d[name])]
    if len(pos) < 5 or len(neg) < 20:
        return None
    y = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    x = np.array([d[name] for d in pos + neg], float)
    v = np.array([d["v_e"] for d in pos + neg])
    g = np.array([d["log"] for d in pos + neg])

    def fn(idx):
        yy, xx, vv = y[idx], x[idx], v[idx]
        if yy.sum() < 3 or (~yy).sum() < 5:
            return np.nan
        w = np.ones(len(idx))
        w[~yy] = speed_w(vv[yy], vv[~yy])
        return S.wauc(yy, xx, w) if w[~yy].sum() > 0 else np.nan
    a, lo, hi = S.boot_ci(fn, g, n_boot)
    w = speed_w(v[y], v[~y])
    o = np.argsort(-x[~y])
    cw = np.cumsum(w[o]) / w.sum()
    thr = x[~y][o][min(np.searchsorted(cw, 0.10), len(o) - 1)]
    return dict(auc=a, lo=lo, hi=hi, thr10=float(thr), n_pos=len(pos), n_neg=len(neg))


def combo(pos, neg, names):
    """Out-of-fold (by log, 5 folds) logistic regression on the listed signals -> the rows with `combo` filled."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = pos + neg
    X = np.array([[d.get(n, np.nan) for n in names] for d in rows], float)
    ok = np.isfinite(X).all(1)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    g = np.array([d["log"] for d in rows])
    sc = np.full(len(rows), np.nan)
    idx = np.flatnonzero(ok)
    if len(np.unique(g[idx])) < 5 or y[idx].sum() < 8:
        return
    for tr, te in GroupKFold(5).split(X[idx], y[idx], g[idx]):
        if y[idx][tr].sum() < 3:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")).fit(np.clip(X[idx][tr], -50, 50), y[idx][tr])
        sc[idx[te]] = m.decision_function(np.clip(X[idx][te], -50, 50))
    for d, v in zip(rows, sc):
        d["combo"] = None if not np.isfinite(v) else float(v)


SIG_LEAD = ["lead_sweep_P0", "lead_sweep_FT", "lead_need_P0", "lead_p_P0", "lead_near_P0"]
SIG_GEO = ["edge_margin_P0", "edge_margin_FT", "lane_exc_P0", "lane_exc_FT", "meta_brake_P0", "meta_brake_FT", "meta_dis_P0", "meta_steer_P0", "meta_brakepress_P0",
           "desire_lc_P0", "desire_lc_FT", "plan_std2_FT", "plan_std4_FT", "plan_std2_P0", "plan_std4_P0"]
SIG_BASE = ["base_gap", "base_len"]
SIG_PLAN = ["v_e", "alat", "abs_dpsi4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", nargs="*", default=["nuplan", "pai"])
    a = ap.parse_args()
    cls = {(r["set"], r["scene"]): r["cls"] for r in csv.DictReader(open(OUT / "classes.csv"))}
    for dom in a.dom:
        R = [r for r in (S.nuplan() if dom == "nuplan" else S.pai("base")) if r.grp != "X"]
        rows = signals(R, pickle.load(open(S.TMP / f"dec_{dom}.pkl", "rb")))
        if dom == "pai":                                   # controls: decisions up to the first step more than 4 m from the logged path (COL1)
            far = {}
            for d in rows:
                if d["grp"] == "N" and abs(d["lat_off"]) > 4.0:
                    far[d["scene"]] = min(far.get(d["scene"], 1e9), d["now"])
            for d in rows:
                d["ok"] = d["grp"] == "C" or d["now"] < far.get(d["scene"], 1e9)
        else:
            for d in rows:
                d["ok"] = True
        by = defaultdict(list)
        for d in rows:
            by[d["set"], d["scene"]].append(d)
        md = [f"# SWV1 parts B and C, {dom} (generated by scripts/swv1_bc.py)", ""]
        # ---------------- B
        sets = sorted({r.set for r in R if r.grp == "C"})
        arms = [("alat", dict(a_lim=x), f"lat-acc cap {x}") for x in ALIMS] + [("lead", dict(src="P0"), "lead-sweep stop (P0)"), ("lead", dict(src="FT"), "lead-sweep stop (FT)"),
                                                                              ("lead", dict(src="P0", margin=2.0), "lead-sweep stop (P0), 2 m margin"),
                                                                              ("priv", {}, "footprint stop (privileged ceiling)"), ("priv", dict(margin=2.0), "footprint stop (privileged ceiling), 2 m margin")]
        cf = []
        for r in R:
            for arm, kw, name in arms:
                x = counterfactual(r, by.get((r.set, r.scene), []), arm, **kw)
                if x:
                    cf.append(dict(dom=dom, set=r.set, scene=r.scene, log=r.log, grp=r.grp, cls=cls.get((r.set, r.scene), ""), arm=name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}))
        keys = list(dict.fromkeys(k for d in cf for k in d))
        with open(OUT / f"cf_{dom}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader(), w.writerows(cf)
        N = [c for c in cf if c["grp"] == "N"]
        md += ["## B. Counterfactuals (executed path re-timed, recorded object motion)", "",
               "Avoided = no contact with the struck object to the end of the rollout and not merely deferred. Clean rollouts: share losing more than 10 % (and 0.5 m) of progress.", "",
               "| arm | collision set | n | avoided | touched only while standing | deferred only | by class i / ii / iii / iv / v (avoided / n) | ii + iii avoided | clean rollouts triggered | clean losing > 10 % | mean clean progress lost |",
               "|:--|:--|--:|--:|--:|--:|:--|:--|:--|:--|--:|"]
        for arm, kw, name in arms:
            n = [c for c in N if c["arm"] == name]
            for st in ([("P2H10-F (s0 + s1)", ("P2H10-F-s0", "P2H10-F-s1")), ("APY10m10-AB (s0 + s1)", ("APY10m10-AB-s0", "APY10m10-AB-s1"))] if dom == "nuplan" else [("P2H10-F-s0", ("pai-base",))]):
                c = [x for x in cf if x["arm"] == name and x["set"] in st[1] and x["grp"] == "C"]
                pc = " / ".join(f"{sum(x['avoided'] for x in c if x['cls'] == k)}/{sum(x['cls'] == k for x in c)}" for k in ("i", "ii", "iii", "iv", "v"))
                m23 = [x for x in c if x["cls"] in ("ii", "iii")]
                md.append(f"| {name} | {st[0]} | {len(c)} | {sum(x['avoided'] for x in c)} | {sum(x['hit_standing'] for x in c)} | {sum(x['deferred'] for x in c)} | {pc} | {sum(x['avoided'] for x in m23)} / {len(m23)} | "
                          f"{sum(x['trig'] for x in n)} / {len(n)} | {sum(x['lose10'] for x in n)} / {len(n)} = {100 * np.mean([x['lose10'] for x in n]):.1f} % | {100 * np.mean([x['loss'] for x in n]):.1f} % |")
        # P0 plan and the human path
        md += ["", "Shipped weights' plan on the same tokens (open loop), last 3 s before impact (decisions within 0.5 s excluded):", "",
               "| collision set | n | served plan's sweep intersects the struck object at some decision | P0 plan's does | base-clear (no P0 plan intersects) | median share of decisions intersecting, served / P0 |", "|:--|--:|--:|--:|--:|:--|"]
        for st in sets:
            g = [[d for d in v if 0.5 < d["t_to_ev"] <= 3.0 and "hit_st_p0" in d] for (s_, sc), v in by.items() if s_ == st and v[0]["grp"] == "C"]
            g = [x for x in g if x]
            if g:
                md.append(f"| {st} | {len(g)} | {sum(any(d['hit_st'] for d in x) for x in g)} | {sum(any(d['hit_st_p0'] for d in x) for x in g)} | {sum(not any(d['hit_st_p0'] for d in x) for x in g)} | "
                          f"{np.median([np.mean([d['hit_st'] for d in x]) for x in g]):.2f} / {np.median([np.mean([d['hit_st_p0'] for d in x]) for x in g]):.2f} |")
        # ---------------- C
        names = SIG_LEAD + (SIG_GEO if dom == "nuplan" else []) + SIG_BASE + SIG_PLAN
        md += ["", "## C. Signals against the label (the served plan's sweep intersects an object box)", ""]
        negs = [d for d in rows if d["grp"] == "N" and d["ok"] and not d["hit_any"]]
        nall = [d for d in rows if d["grp"] == "N" and d["ok"]]
        md += [f"Clean rollouts: {len({(d['set'], d['scene']) for d in nall})}, decisions {len(nall)}, of which the planned sweep intersects an object at {sum(d['hit_any'] for d in nall)} "
               f"({100 * np.mean([d['hit_any'] for d in nall]):.1f} %; objects frozen at the decision time: {100 * np.mean([d['hit_frozen'] for d in nall]):.1f} %; sweep truncated at the end of the record: "
               f"{100 * np.mean([d['hit_trunc'] for d in nall]):.1f} %); clean rollouts with such a decision: {len({(d['set'], d['scene']) for d in nall if d['hit_any']})}.", ""]
        sig_rows = []
        groups = [("P2H10-F (s0 + s1)", ("P2H10-F-s0", "P2H10-F-s1")), ("APY10m10-AB (s0 + s1; P2H10 controls, flagged)", ("APY10m10-AB-s0", "APY10m10-AB-s1"))] if dom == "nuplan" else [("P2H10-F-s0", ("pai-base",))]
        for gname, st in groups:
            pos = [d for d in rows if d["grp"] == "C" and d["set"] in st and d["t_to_ev"] <= 8.0 and d["hit_st"]]
            combo(pos, negs, [n for n in names if n not in ("frozen_ref",)])
            md += [f"### Primary read, {gname}: {len(pos)} positive decisions in {len({d['scene'] + d['set'] for d in pos})} rollouts ({len({d['log'] for d in pos})} logs) against {len(negs)} clean decisions "
                   f"({len({d['log'] for d in negs})} logs), speed-matched", "",
                   "| signal | AUC [95 % by log] | line (>= 0.75, lower > 0.65) | rollouts flagged >= 1.5 s before impact at 10 % false positives | median lead time s |", "|:--|:--|:--|:--|--:|"]
            for n in names + ["combo", "frozen_ref"]:
                x = read_auc(pos, negs, n)
                if x is None:
                    continue
                lt = []
                for key in {(d["set"], d["scene"]) for d in pos}:
                    f = [d["t_to_ev"] for d in by[key] if d["t_to_ev"] <= 8.0 and d.get(n) is not None and d[n] > x["thr10"]]
                    lt.append(max(f) if f else 0.0)
                line = "privileged reference" if n == "frozen_ref" else "usable" if x["auc"] >= 0.75 and x["lo"] > 0.65 else "no"
                if line == "usable" and (x["auc"] < 0.78 or x["lo"] < 0.68):
                    line = "candidate (within 0.03)"
                md.append(f"| `{n}` | {x['auc']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] | {line} | {sum(v >= 1.5 for v in lt)} / {len(lt)} | {np.median(lt):.1f} |")
                sig_rows.append(dict(dom=dom, group=gname, read="primary", signal=n, **{k: round(v, 4) if isinstance(v, float) else v for k, v in x.items()}, line=line,
                                     flagged_1p5=sum(v >= 1.5 for v in lt), rollouts=len(lt), lead_med=float(np.median(lt))))
            md.append("")
        # secondary: all decisions Y = 1 against Y = 0 on the P2H10 sets
        st = groups[0][1]
        allr = [d for d in rows if d["ok"] and (d["grp"] == "N" or d["set"] in st)]
        p2, n2 = [d for d in allr if d["hit_any"]], [d for d in allr if not d["hit_any"]]
        md += [f"### Secondary read: every decision, Y = 1 ({len(p2)}) against Y = 0 ({len(n2)}), regardless of rollout", "", "| signal | AUC [95 % by log] |", "|:--|:--|"]
        for n in names + ["frozen_ref"]:
            x = read_auc(p2, n2, n)
            if x:
                md.append(f"| `{n}` | {x['auc']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] |")
                sig_rows.append(dict(dom=dom, group=groups[0][0], read="secondary", signal=n, **{k: round(v, 4) if isinstance(v, float) else v for k, v in x.items()}))
        keys = list(dict.fromkeys(k for d in sig_rows for k in d))
        with open(OUT / f"signals_{dom}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader(), w.writerows(sig_rows)
        pickle.dump(rows, open(S.TMP / f"sig_{dom}.pkl", "wb"))
        (OUT / f"bc_{dom}.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))


if __name__ == "__main__":
    main()
