#!/usr/bin/env python3
"""LOCAL-MAP report (plans/2026-10-12-local-map-prereg.md), Mac .venv: gates G-sign / G-d0 on top of the box gates, the registered tables
(Z: the 98 road-exit zero windows; H: held turns 0-20 m before a junction), the verdict, the plan check and two figures, from a copy of
the box read dir (lm.py read: est_{zero,held}.parquet, labels_{zero,held}.parquet, gates.json, bev.pkl).

  python experiments/local_map/scripts/lm_report.py --read tmp/lm_read     -> results/lm_report.md, results/*.csv, figs/lm_*.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/road_probe/scripts"), str(REPO / "research")]
import rp as RP  # noqa: E402

OUT, FIGS = REPO / "experiments/local_map/results", REPO / "experiments/local_map/figs"
DS, DLIST, CARRY = (10, 20, 40), (0, 2, 4, 6, 8), (2, 4, 6, 8)
RULES = {"L1": [str(d) for d in DLIST], "L2": [str(d) for d in DLIST] + ["n"], "L3": [str(d) for d in DLIST] + ["n"]}
NAME = {"L1": "L1 carried probe", "L2": "L2 carried route (centre)", "L3": "L3 route centre + probe width"}
REF = dict(r3=1.2551398536573624, r3_eval=45, j010=1.1492006381375899, j1020=1.68332965109287)    # decision 275 (rp.json, junction.csv)
COVER = 23
B = 2000


def f3(x, d=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def md(cols, rows):
    return ["| " + " | ".join(cols) + " |", "|:--" + "|--:" * (len(cols) - 1) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


def rule_lab(r):
    return "newest covering" if r == "n" else "newest, start-behind (exploratory)" if r == "b" else f"d = {r} ({int(r) / 2:g} s)" if r != "0" else "d = 0 (now)"


def errs(E, Lb, line, r, D):
    """-> (left error, right error) per decision; L2 gives the centre error on both."""
    if line == "L2":
        e = np.abs(E[f"L2_{r}_c{D}"].values - 0.5 * (Lb[f"L{D}"].values + Lb[f"R{D}"].values))
        return e, e
    return np.abs(E[f"{line}_{r}_L{D}"].values - Lb[f"L{D}"].values), np.abs(E[f"{line}_{r}_R{D}"].values - Lb[f"R{D}"].values)


def zsum(W_):
    s = RP.r3_summary(W_, B=B)
    w = np.array([x["werr"] for x in W_], float)
    s["p90"] = float(np.nanquantile(w, 0.9)) if np.isfinite(w).any() else np.nan
    return s


def main(a):
    R = Path(a.read)
    G = json.loads((R / "gates.json").read_text())
    Ez, Eh = pd.read_parquet(R / "est_zero.parquet"), pd.read_parquet(R / "est_held.parquet")
    Lz, Lh = pd.read_parquet(R / "labels_zero.parquet"), pd.read_parquet(R / "labels_held.parquet")
    assert (Ez.row.values == Lz.row.values).all() and (Eh.row.values == Lh.row.values).all()
    ev = RP.zero_events()
    st = (Lh.turn40 < 5).values
    tu = (Lh.turn40 >= 20).values
    jd = Lh.jdist.values
    pre = tu & (jd > 0) & (jd <= 20)
    cl = Lh.scene.to_numpy(dtype=str)
    # Mac gates
    G["G_sign"] = dict(L20=float(np.nanmedian(Eh["L1_2_L20"][st])), R20=float(np.nanmedian(Eh["L1_2_R20"][st])))
    G["G_sign"]["pass"] = bool(G["G_sign"]["L20"] > 0 > G["G_sign"]["R20"])
    G["G_sign_mirror"]["pass"] = bool(G["G_sign_mirror"]["ratio"] > 10)
    w0 = zsum(RP.r3_windows(*errs(Ez, Lz, "L1", "0", 20), Lz, ev))
    eh0 = np.vstack(errs(Eh, Lh, "L1", "0", 20))
    j = {b: float(np.nanmedian(eh0[:, tu & m].ravel())) for b, m in (("j010", (jd > 0) & (jd <= 10)), ("j1020", (jd > 10) & (jd <= 20)))}
    G["G_d0"] = dict(r3=w0["median"], r3_eval=w0["eval"], **j)
    G["G_d0"]["pass"] = bool(abs(w0["median"] - REF["r3"]) < 0.005 and w0["eval"] == REF["r3_eval"] and all(abs(j[b] - REF[b]) < 0.005 for b in j))
    G["pass"] = all(v["pass"] for k, v in G.items() if k.startswith("G_") and isinstance(v, dict) and "pass" in v)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps(G, indent=1, default=float))
    L = ["# LOCAL-MAP: would a carried local map have saved the frozen probe at the road-exit zeros?", "",
         "Pre-registration: [../plans/2026-10-12-local-map-prereg.md](../plans/2026-10-12-local-map-prereg.md) (pushed before any read; section 7 "
         "amended G-carry before the read). Estimates made d decisions earlier (d = 2 / 4 / 6 / 8 = 1 / 2 / 3 / 4 s) are carried into the current "
         "decision's rear-axle frame with the egomotion the driver receives and read at x = camera + D. **L1** = decision 275's probe (two-seed mean) "
         "ego-lane boundaries (three points per side at 10 / 20 / 40 m, joined, no extrapolation); **L2** = the route waypoints the driver received "
         "(lane-centre points ~4.2 m apart starting ~40 m ahead), error against the map lane centre (L + R) / 2; **L3** = L2's centre with L1's width "
         "(same d, else the current probe). d = 0 is the current decision (L1 d = 0 = decision 275's probe). Labels are decision 275's, unchanged: the "
         "lane the logged route occupies at D. Note that the route and the label both follow the logged path's lane, so L2 measures how well the "
         "*received* route carries that lane into the zero window's frame, not an independent map.", "", "## Gates", ""]
    gl = []
    for k, v in G.items():
        if k.startswith("G_"):
            gl.append([k, ", ".join(f"{q} {f3(x, 4) if isinstance(x, float) else x}" for q, x in v.items() if q != "pass"), "pass" if v["pass"] else "FAIL"])
    L += md(["gate", "reading", ""], gl) + [""]
    if not G["pass"] and not a.force:
        L += ["**A gate failed: no registered read.**"]
        (OUT / "lm_report.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    # Z
    Z, Zw = [], {}
    for line, rules in RULES.items():
        for r in rules:
            for D in DS:
                W_ = RP.r3_windows(*errs(Ez, Lz, line, r, D), Lz, ev)
                s = zsum(W_)
                Zw[line, r, D] = np.array([x["werr"] for x in W_], float)
                Z.append(dict(line=line, rule=r, D=D, **s))
        for D in DS:                                                     # oracle: best carried d per window
            M = np.stack([Zw[line, str(d), D] for d in CARRY])
            ok = np.isfinite(M).any(0)
            w = np.where(ok, np.nanmin(np.where(np.isfinite(M), M, np.inf), 0), np.nan)
            rng = np.random.default_rng(0)
            idx = np.flatnonzero(ok)
            bm = [np.median(w[rng.choice(idx, len(idx))]) for _ in range(B)] if len(idx) else [np.nan]
            Z.append(dict(line=line, rule="oracle", D=D, windows=len(w), eval=int(ok.sum()), median=float(np.nanmedian(w)) if ok.any() else np.nan,
                          med_lo=float(np.quantile(bm, 0.025)), med_hi=float(np.quantile(bm, 0.975)), p90=float(np.nanquantile(w, 0.9)) if ok.any() else np.nan,
                          share=float((w[ok] <= 0.5).mean()) if ok.any() else np.nan))
    Z = pd.DataFrame(Z)
    Z.to_csv(OUT / "z_windows.csv", index=False)
    # verdict
    z20 = Z[(Z.D == 20) & (Z.rule != "oracle") & (Z.rule != "0")]
    ok = z20[z20["eval"] >= COVER]
    works = ok[ok["median"] < 0.5]
    if len(works):
        verdict = "local map works: " + "; ".join(f"{r.line} {rule_lab(r.rule)} {f3(r['median'])} m ({int(r['eval'])} windows)" for _, r in works.iterrows())
        vshort = "local map works"
    elif len(ok) and (ok["median"] > 1.0).all():
        verdict = vshort = "never seen clearly"
    else:
        verdict = vshort = "undetermined"
    L += ["## Z (registered): the 98 road-exit zero windows, D = 20 m", "",
          f"**Verdict: {verdict}.** Line: a fixed rule (d in 2-8 or newest covering) with window median < 0.5 m on >= {COVER} covered windows; "
          "every covered fixed rule > 1.0 m = never seen clearly. Window error as decision 275's R3 (per-side window medians, mean of the sides; "
          "L2: the centre error); CI = window bootstrap (B 2000).", ""]
    rows = []
    for line in RULES:
        for r in RULES[line] + ["oracle"]:
            x = Z[(Z.line == line) & (Z.rule == r) & (Z.D == 20)].iloc[0]
            rows.append([NAME[line], "oracle: best d (2-8) per window, not a rule" if r == "oracle" else rule_lab(r), f"{f3(x['median'])} [{f3(x.med_lo)}, {f3(x.med_hi)}]",
                         f3(x.p90), f3(x.share), int(x["eval"])])
    L += md(["line", "rule", "window median (m)", "p90 (m)", "share <= 0.5 m", "covered windows"], rows) + [""]
    L += ["Other distances (window median m / covered windows):", ""]
    L += md(["line", "rule", "D = 10 m", "D = 40 m"], [[line, rule_lab(r) if r != "oracle" else "oracle"] +
                                                       [f"{f3(x['median'])} / {int(x['eval'])}" for D in (10, 40) for x in [Z[(Z.line == line) & (Z.rule == r) & (Z.D == D)].iloc[0]]]
                                                       for line in RULES for r in RULES[line] + ["oracle"]]) + [""]
    # L2 secondary (point-to-polyline distance) and the newest route's age
    sec = []
    for r in ("0", "2", "4", "6", "8"):
        e = Ez[f"L2_{r}_P20"].values
        W_ = RP.r3_windows(e, e, Lz, ev)
        s = zsum(W_)
        sec.append([rule_lab(r), f3(s["median"]), int(s["eval"])])
    wm = np.zeros(len(Lz), bool)
    for e in ev:
        wm |= ((Lz.tag == e["group"]) & (Lz.scene == e["scene"]) & (Lz.k >= max(0, int(e["k_ev"]) - 4)) & (Lz.k <= int(e["k_ev"]))).values
    age = Ez.L2_n_d20.values[wm]
    L += [f"L2 secondary (distance from the map centre point at 20 m to the carried route polyline): " +
          ", ".join(f"{a_} {b_} m ({c_} windows)" for a_, b_, c_ in sec) +
          f". Age of the newest route covering 20 m on window decisions: median {f3(float(np.nanmedian(age)), 1)} decisions, "
          f"none on {f3(float(np.isnan(age).mean()))} of them.", ""]
    # H
    H = []
    for line, rules in RULES.items():
        for r in rules + (["b"] if line != "L1" else []):
            el, er = errs(Eh, Lh, line, r, 20)
            for sub, m in (("pre-junction turns 0-20 m", pre), ("turns in junction", tu & (jd == 0)), ("turns 20-60 m before", tu & (jd > 20) & (jd <= 60)),
                           ("straight", st)):
                e = np.r_[el[m], er[m]] if line != "L2" else el[m]
                c = np.r_[cl[m], cl[m]] if line != "L2" else cl[m]
                md_, p9 = RP.boot_q(e, c, 0.5, B=B if sub.startswith("pre") else 300), RP.boot_q(e, c, 0.9, B=B if sub.startswith("pre") else 300)
                H.append(dict(line=line, rule=r, subset=sub, decisions=int(m.sum()), n=md_["n"], median=md_["v"], lo=md_["lo"], hi=md_["hi"], p90=p9["v"],
                              p90_lo=p9["lo"], p90_hi=p9["hi"]))
    H = pd.DataFrame(H)
    H.to_csv(OUT / "h_turns.csv", index=False)
    L += ["## H (registered): held turn decisions (route turn >= 20 deg) 0-20 m before a junction, D = 20 m", "",
          f"{int(pre.sum())} decisions; per decision both sides pooled (L2: centre error); scene-cluster bootstrap (B 2000).", ""]
    L += md(["line", "rule", "median (m)", "p90 (m)", "n (points)"],
            [[NAME[x.line], rule_lab(x.rule), f"{f3(x['median'])} [{f3(x.lo)}, {f3(x.hi)}]", f"{f3(x.p90)} [{f3(x.p90_lo)}, {f3(x.p90_hi)}]", int(x.n)]
             for _, x in H[H.subset.str.startswith("pre") & (H.rule != "b")].iterrows()]) + [""]
    L += ["Descriptive, other held subsets (median m / points; CI B 300):", ""]
    subs = ["turns in junction", "turns 20-60 m before", "straight"]
    L += md(["line", "rule"] + subs, [[x0, rule_lab(r)] + [f"{f3(y['median'])} / {int(y.n)}" for s in subs for y in [H[(H.line == x0) & (H.rule == r) & (H.subset == s)].iloc[0]]]
                                      for x0 in RULES for r in RULES[x0]]) + [""]
    # plan check
    PC = []
    for line in ("L1", "L3"):
        for r in RULES[line]:
            for e in ev:
                m = ((Lz.tag == e["group"]) & (Lz.scene == e["scene"]) & (Lz.k >= max(0, int(e["k_ev"]) - 4)) & (Lz.k <= int(e["k_ev"]))).values
                ks = Lz.k.values[m]
                oc, om, evc, evm = np.zeros(m.sum(), bool), np.zeros(m.sum(), bool), np.zeros(m.sum(), bool), np.zeros(m.sum(), bool)
                for D in (10, 20):
                    y = Ez[f"plan{D}"].values[m]
                    lh, rh = Ez[f"{line}_{r}_L{D}"].values[m], Ez[f"{line}_{r}_R{D}"].values[m]
                    lt, rt = Lz[f"L{D}"].values[m], Lz[f"R{D}"].values[m]
                    vc = np.isfinite(y) & np.isfinite(lh) & np.isfinite(rh)
                    vm = np.isfinite(y) & np.isfinite(lt) & np.isfinite(rt)
                    oc |= vc & ((y > lh) | (y < rh))
                    om |= vm & ((y > lt) | (y < rt))
                    evc |= vc
                    evm |= vm
                PC.append(dict(line=line, rule=r, group=e["group"], scene=e["scene"], k_ev=int(e["k_ev"]), eval=bool(evc.any()), carried_out=bool(oc.any()),
                               map_eval=bool(evm.any()), map_out=bool(om.any()), lead=int(e["k_ev"]) - int(ks[oc].min()) if oc.any() else np.nan))
    PC = pd.DataFrame(PC)
    PC.to_csv(OUT / "plan_check.csv", index=False)
    rows = []
    for (line, r), x in PC.groupby(["line", "rule"], sort=False):
        b = x[x["eval"] & x.map_eval]
        rows.append([NAME[line], rule_lab(r), int(x["eval"].sum()), int(x.carried_out.sum()), int(b.map_out.sum()), int((b.carried_out & b.map_out).sum()),
                     int((b.carried_out & ~b.map_out).sum()), f3(float(x.lead.median()), 1)])
    L += ["## Plan check (registered): would a lane-keeping rule on the carried map have disagreed with the served plan before the zero?", "",
          "Per window: the served plan's rear-axle path at 10 or 20 m outside the carried lane [R, L] on any window decision. 'map says out' uses the "
          "true lane labels on the same decisions (windows where both are evaluable); lead = decisions from the first disagreement to k_ev.", ""]
    L += md(["line", "rule", "evaluable windows", "carried map says out", "map says out (both eval.)", "both say out", "carried out, map in",
             "median lead (decisions)"], rows) + [""]
    nm = int(PC[(PC.line == "L1") & (PC.rule == "0")].map_eval.sum())
    no = int(PC[(PC.line == "L1") & (PC.rule == "0")].map_out.sum())
    L += [f"Reference: on the true map the served plan leaves the route lane at 10 or 20 m in {no} of the {nm} windows where the map lane is defined.", ""]
    L += explore(Ez, Lz, Eh, Lh, ev, pre, cl)
    (OUT / "lm_report.md").write_text("\n".join(L) + "\n")
    S = dict(verdict=vshort, verdict_text=verdict, gates=G, pre_n=int(pre.sum()))
    (OUT / "lm.json").write_text(json.dumps(S, indent=1, default=float))
    figures(R, Z, H, Ez, Lz, ev)
    print("\n".join(L))


def win_mask(Lz, ev):
    m = np.zeros(len(Lz), bool)
    for e in ev:
        m |= ((Lz.tag == e["group"]) & (Lz.scene == e["scene"]) & (Lz.k >= max(0, int(e["k_ev"]) - 4)) & (Lz.k <= int(e["k_ev"]))).values
    return m


def plan_out(E, Lb, line, r):
    """Per decision: (carried map says the plan is out at 10 or 20 m, evaluable; map says out, evaluable)."""
    oc = ec = om = em = np.zeros(len(E), bool)
    for D in (10, 20):
        y = E[f"plan{D}"].values
        if line == "map":
            lh, rh = Lb[f"L{D}"].values, Lb[f"R{D}"].values
        else:
            lh, rh = E[f"{line}_{r}_L{D}"].values, E[f"{line}_{r}_R{D}"].values
        v = np.isfinite(y) & np.isfinite(lh) & np.isfinite(rh)
        oc, ec = oc | (v & ((y > lh) | (y < rh))), ec | v
    return oc, ec


def explore(Ez, Lz, Eh, Lh, ev, pre, cl):
    """Post-read, EXPLORATORY (not registered)."""
    out = ["## Exploratory (decided after the read, not registered)", ""]
    rows = []
    for line, r in (("L2", "n"), ("L2", "b"), ("L3", "n"), ("L3", "b")):
        s = zsum(RP.r3_windows(*errs(Ez, Lz, line, r, 20), Lz, ev))
        el, er = errs(Eh, Lh, line, r, 20)
        e, c = (el[pre], cl[pre]) if line == "L2" else (np.r_[el[pre], er[pre]], np.r_[cl[pre], cl[pre]])
        h = RP.boot_q(e, c, 0.5, B=B)
        rows.append([NAME[line], "newest covering" if r == "n" else "newest covering, route starts at or behind the line",
                     f"{f3(s['median'])} [{f3(s['med_lo'])}, {f3(s['med_hi'])}]", f3(s["p90"]), int(s["eval"]), f"{f3(h['v'])} [{f3(h['lo'])}, {f3(h['hi'])}]", h["n"]])
    out += ["- **Wrong-leg crossings.** The registered first-crossing rule also takes a route that starts beyond the 20 m line and bends back across "
            "it (a route received before a turn crosses x = camera + 20 m on the far leg, tens of metres to the side): that is the 30-50 m tail of L2 / L3 "
            "at d = 0-4 and on held turns. A label-free fix, a route counts only if its carried start lies at or behind the line:", ""]
    out += md(["line", "rule", "Z window median (m)", "Z p90", "Z windows", "H pre-junction median (m)", "H points"], rows) + [""]
    wm = win_mask(Lz, ev)
    out += [f"- **Why coverage is low.** In the window decisions the car covers a median {f3(float(np.nanmedian(Ez.trav_4[wm])), 1)} m in 2 s and "
            f"{f3(float(np.nanmedian(Ez.trav_8[wm])), 1)} m in 4 s (slow, junction speeds); a route received 4 s earlier starts a median "
            f"{f3(float(np.nanmedian(Ez.x0_8[wm] - Lz.cam_x[wm])), 1)} m ahead of the camera, so it reaches the 20 m line only when the car has moved "
            f"~20 m since. {f3(float(np.isnan(Ez.L2_b_c20[wm]).mean()))} of the window decisions have no received route (of any age) covering "
            f"20 m by the start-behind rule (and {f3(float((Lz.k[wm] < 8).mean()))} of all window decisions fall in the first 8 decisions of a rollout). The probe's own "
            "memory (L1) reaches 20 m only through its 40 m point, which is the probe's weakest output on turns, and gets worse with d.", ""]
    rows = []
    for nm, line, r in (("probe now (L1 d = 0)", "L1", "0"), ("probe carried 2 s (L1 d = 4)", "L1", "4"), ("L3 newest, start-behind", "L3", "b"),
                        ("map labels", "map", None)):
        rr = [nm]
        for E, Lb, m in ((Ez, Lz, wm), (Eh, Lh, np.ones(len(Eh), bool)), (Eh, Lh, (Lh.turn40 < 5).values)):
            o, v = plan_out(E, Lb, line, r)
            rr.append(f"{f3(float(o[m & v].mean()))} ({int((m & v).sum())})")
        rows.append(rr)
    out += ["- **Base rate of the plan check on ordinary driving.** Share of decisions where the rule says the served plan's rear axle is outside "
            "the lane at 10 or 20 m (evaluable decisions in brackets). On ordinary held driving the rule fires on 7-25 % of decisions, and the true map "
            "itself says 'out' on 16 % (the served plan crosses the route lane's lines in normal driving, and the label lane is the logged route's "
            "lane): a disagreement in the window is informative only against that base rate, and the carried-probe versions fire more often than "
            "the current probe.", ""]
    out += md(["lane source", "zero-window decisions", "held, all decisions", "held straight"], rows) + [""]
    return out


def figures(R, Z, H, Ez, Lz, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIGS.mkdir(parents=True, exist_ok=True)
    col = {"L1": ps.PALETTE["blue"], "L2": ps.PALETTE["orange"], "L3": ps.PALETTE["green"]}
    fig, ax = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.0))
    xs = list(DLIST) + [10]
    for line in RULES:
        for p_, (src, sub) in enumerate(((Z, None), (Z, "cov"), (H, "pre"))):
            y, lo, hi, xx = [], [], [], []
            for r, x in zip(RULES[line], xs):
                if src is Z:
                    q = Z[(Z.line == line) & (Z.rule == r) & (Z.D == 20)].iloc[0]
                    v = (q["eval"], q["eval"], q["eval"]) if sub == "cov" else (q["median"], q.med_lo, q.med_hi)
                else:
                    q = H[(H.line == line) & (H.rule == r) & H.subset.str.startswith("pre")].iloc[0]
                    v = (q["median"], q.lo, q.hi)
                y.append(v[0]), lo.append(v[0] - v[1]), hi.append(v[2] - v[0]), xx.append(x)
            y, lo, hi = np.array(y, float), np.array(lo, float), np.array(hi, float)
            k = len(DLIST)
            ax[p_].errorbar(np.array(xx[:k]) + (list(RULES).index(line) - 1) * 0.15, y[:k], yerr=None if sub == "cov" else [lo[:k], hi[:k]], color=col[line],
                            marker="o", ms=2.5, lw=0.9, capsize=1.5, label=NAME[line])
            if len(xx) > k:
                ax[p_].errorbar([xx[k] + (list(RULES).index(line) - 1) * 0.15], y[k:], yerr=None if sub == "cov" else [lo[k:], hi[k:]], color=col[line],
                                marker="s", ms=3, ls="none", capsize=1.5)
    for line in ("L2", "L3"):                                          # exploratory start-behind rule, hollow
        q = zsum(RP.r3_windows(*errs(Ez, Lz, line, "b", 20), Lz, ev))
        ax[0].plot([10.3 + (list(RULES).index(line) - 1) * 0.15], [q["median"]], marker="s", ms=3.5, mfc="none", color=col[line], ls="none")
        ax[1].plot([10.3], [q["eval"]], marker="s", ms=3.5, mfc="none", color=col[line], ls="none")
        q = H[(H.line == line) & (H.rule == "b")]
        if len(q):
            ax[2].plot([10.3 + (list(RULES).index(line) - 1) * 0.15], [q.iloc[0]["median"]], marker="s", ms=3.5, mfc="none", color=col[line], ls="none")
    for p_ in (0, 2):
        ax[p_].axhline(0.5, color="#999999", lw=0.5, ls=":"), ax[p_].axhline(1.0, color="#999999", lw=0.5, ls=":")
        ax[p_].set_yscale("log"), ax[p_].set_ylim(0.05, 80), ax[p_].set_ylabel("20 m error (m)")
    ax[1].axhline(COVER, color="#999999", lw=0.5, ls=":"), ax[1].set_ylabel("covered windows")
    for p_, t in enumerate(("(a) zero windows, median", "(b) zero windows, coverage", "(c) held turns 0-20 m before junction")):
        ax[p_].set_xticks(xs, ["0", "2", "4", "6", "8", "new"]), ax[p_].set_xlabel("carried from d decisions earlier")
        ps.panel(ax[p_], t)
    ax[1].legend(fontsize=6, loc="upper right")
    fig.tight_layout(w_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"lm_errors.{ext}", bbox_inches="tight")
    plt.close(fig)
    # BEV: 4 windows at the 10 / 40 / 70 / 95 % quantiles of the L3-newest window error at 20 m (event decision)
    bev = pickle.load(open(R / "bev.pkl", "rb"))
    w = pd.DataFrame(RP.r3_windows(*errs(Ez, Lz, "L3", "n", 20), Lz, ev)).dropna(subset=["werr"]).sort_values("werr").reset_index(drop=True)
    picks = [w.iloc[int(q * (len(w) - 1))] for q in (0.1, 0.4, 0.7, 0.95)]
    fig, ax = plt.subplots(1, 4, figsize=(ps.DOUBLE_COLUMN_IN, 3.1))
    for a_, p in zip(ax, picks):
        m = ((Lz.tag == p.group) & (Lz.scene == p.scene) & (Lz.k >= max(0, int(p.k_ev) - 4)) & (Lz.k <= int(p.k_ev)) & Lz.L20.notna() & Lz.R20.notna()).values
        kk = int(Lz.k[m].max()) if m.any() else int(p.k_ev)               # the last window decision with a defined map lane at 20 m
        key = (p.group, p.scene, kk)
        b = bev[key]
        for q in b["edges"]:
            a_.plot(-q[:, 1], q[:, 0], color="#BBBBBB", lw=0.5)
        for q in b["road"]:
            a_.plot(-q[:, 1], q[:, 0], color="#222222", lw=0.8)
        a_.plot(-b["log"][:, 1], b["log"][:, 0], color=ps.PALETTE["green"], lw=1.2, label="human log")
        pl = np.r_[[[0, 0]], b["plan"][:, :2]]
        a_.plot(-pl[:, 1], pl[:, 0], color=ps.PALETTE["vermillion"], lw=1.1, label="served plan")
        row = Ez[(Ez.tag == p.group) & (Ez.scene == p.scene) & (Ez.k == kk)].iloc[0]
        dn = row.L2_n_d20
        if np.isfinite(dn) and int(dn) in b["route"]:
            q = b["route"][int(dn)]
            a_.plot(-q[:, 1], q[:, 0], color=ps.PALETTE["orange"], lw=0.9, marker=".", ms=2.5, label="carried route (newest)")
        for d, c_, ls in ((0, ps.PALETTE["sky_blue"], ":"), (4, ps.PALETTE["blue"], "-")):
            if d in b["probe"]:
                for s in "LR":
                    q = b["probe"][d][s]
                    a_.plot(-q[:, 1], q[:, 0], color=c_, lw=0.9, ls=ls, marker="o", ms=2, label=f"probe {'now' if d == 0 else 'carried 2 s'}" if s == "L" else None)
        x20 = b["cam"][0] + 20
        lab = Lz[(Lz.tag == p.group) & (Lz.scene == p.scene) & (Lz.k == kk)].iloc[0]
        a_.plot([-lab.L20, -lab.R20], [x20, x20], ls="none", marker="x", ms=3.5, color="#000000")
        a_.set_xlim(-30, 30), a_.set_ylim(-10, 70), a_.set_aspect("equal"), a_.tick_params(labelsize=6.5)
        a_.text(0.02, 0.98, f"{p.type}, k = {kk} (zero at {int(p.k_ev)})\nL3 window {p.werr:.2f} m", transform=a_.transAxes, va="top", fontsize=6.5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax[0].set_ylabel("ahead (m)"), ax[0].set_xlabel("lateral (m, right +)")
    ax[0].legend(fontsize=5.5, loc="lower left")
    fig.tight_layout(w_pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"lm_bev.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    (OUT / "bev_picks.json").write_text(json.dumps([[p.group, p.scene, int(p.k_ev), float(p.werr)] for p in picks], indent=1))
    _ = R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True), ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
    _ = shutil
