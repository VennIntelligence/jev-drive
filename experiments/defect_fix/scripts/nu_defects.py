#!/usr/bin/env python3
"""NU-DEFECTS (results/nu_defects/): do the PAI plan defects (INERTIA d404, LEAD-BREAK d407) show on nuPlan, on B2's own stored native
arena held states? CPU only, no forward pass, no rollout: the stored plans of the native reads (stand_b extract, `dec.parquet`).

  .venv/bin/python nu_defects.py --src <ext dir> [--tags A,B,..] --out experiments/defect_fix/results/nu_defects

The counterfactual gates themselves (inertia.py cf, lead_break.py cf) need the decision's frames, which the arena reads did not dump;
this is their stored-plan form:
  STOP approach   the human comes to rest inside the window (< 0.5 m/s for >= 1 s after >= 1 m/s); H = its rest station on its track.
                  Approach decision: valid (before the window's first collision / offroad / corridor step), ego >= 1 m/s, 0 < H - s_e <= 40 m,
                  no lead closer than H - s_e + 5 m. Per decision: plan has a stop (a 0.5 s segment < 0.5 m/s), E - H (E = s_e + 4 s arc,
                  or the plan's stop station), plan end speed. Per window (>= 4 approach decisions, s_e spread >= 3 m): slope of E on s_e
                  (>= 0.7 the end recedes with the ego, <= 0.3 pinned in the world; inertia.py supplement 2).
  copy elasticity d log A4 / d log v by OLS with distance-to-H bins as fixed effects (STOP) against open road (no stop, no lead,
                  |turn20| < 15 deg): the observational analogue of d404's ratio (it is not a counterfactual; reported as such).
  BRK (lead)      valid, ego >= 2 m/s, lead gap <= 40 m, an ego holding its speed reaches the lead's tail within 4 s under a constant-speed
                  lead. Clear = the plan's arc stays <= gap + v_lead t at every 0.5 s pose (lead_break.py's "stays behind", lead future
                  approximated at constant speed). FOL control: lead 5-30 m, |v - v_lead| < 1 m/s.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

V_STOP = 0.5
TT = np.arange(1, 9) * 0.5


def human_rest(h):
    T, P = np.asarray(h["T"]) * 1e-6, np.asarray(h["P"])
    S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    V = np.gradient(S, T)
    moved = np.flatnonzero(V >= 1.0)
    if not len(moved):
        return np.nan, np.nan
    for i in range(moved[0], len(V)):
        if V[i] < V_STOP and np.all(V[i:min(i + 10, len(V))] < V_STOP):
            return float(S[i]), float(T[i])
    return np.nan, np.nan


def plan_metrics(poses):
    P = np.asarray(poses, np.float64).reshape(-1, 8, 3)[..., :2]
    xy = np.concatenate([np.zeros((len(P), 1, 2)), P], 1)
    seg = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    arc = np.cumsum(seg, 1)
    slow = seg / 0.5 < V_STOP
    has = slow.any(1)
    i = np.argmax(slow, 1)
    stop = np.where(has, np.take_along_axis(np.concatenate([np.zeros((len(P), 1)), arc], 1), i[:, None], 1)[:, 0], arc[:, -1])
    return arc, has, stop, seg[:, -1] / 0.5


def first_event(st):
    e = st[(st.collision_any > 0) | (st.offroad > 0) | (st.left_corridor_laterally > 0)]
    return e.groupby(["tag", "id"]).t_us.min()


def boot(x, g, f=np.mean, n=2000, seed=0):
    x, g = np.asarray(x, float), np.asarray(g)
    u, inv = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(inv == j) for j in range(len(u))]
    r = np.random.default_rng(seed)
    b = [f(np.concatenate([x[idx[j]] for j in r.integers(0, len(u), len(u))])) for _ in range(n)]
    return float(f(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def elasticity(d, bins):
    """OLS of log A4 on log v with bin fixed effects -> slope."""
    y, x = np.log(np.maximum(d.A4.to_numpy(), 0.1)), np.log(d.v.to_numpy())
    for b in np.unique(bins):
        m = bins == b
        y[m] -= y[m].mean()
        x[m] -= x[m].mean()
    return float((x * y).sum() / max((x * x).sum(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tags", default="STANDFMXH-L18-s1,STANDFMXH-L18-s0,JTN-L18-s1,JTN-L18-s0,RSG-L18-s0")
    ap.add_argument("--misses", default="experiments/alpasim_proxy/results/easy_items/misses_STANDFMXH-s1.csv.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tags = a.tags.split(",")
    D = pd.read_parquet(src / "dec.parquet")
    D = D[D.tag.isin(tags)].reset_index(drop=True)
    W = pd.read_parquet(src / "win.parquet").set_index(["tag", "id"])
    hum = pickle.load(open(src / "hum.pkl", "rb"))
    ev = first_event(pd.read_parquet(src / "steps.parquet"))
    rest = {w: human_rest(h) for w, h in hum.items() if h is not None}
    arc, has, stop, vend = plan_metrics(np.stack(D.poses.to_numpy()))
    D["A4"], D["has"], D["stop"], D["vend"] = arc[:, -1], has, stop, vend
    D["a02"] = (arc[:, 3] - 2.0 * D.v) / 2.0                 # constant-acceleration fit of the plan's 2 s arc from the fed speed
    D["t_ev"] = [ev.get((t, i), np.inf) for t, i in zip(D.tag, D.id)]
    D["valid"] = D.t0 < D.t_ev
    D["H"] = D.id.map(lambda w: rest.get(w, (np.nan,))[0])
    D["score"] = [W.score.get((t, i), np.nan) for t, i in zip(D.tag, D.id)]
    D["wtype"] = [W.type.get((t, i), "") for t, i in zip(D.tag, D.id)]
    D["dH"] = D.H - D.s_e
    D["E"] = D.s_e + np.where(D.has, D.stop, D.A4)
    lead_ok = ~(D.lead_gap < D.dH + 5)
    R = {"src": str(src), "tags": tags, "def": __doc__.split("\n\n")[1]}

    # ---------------- STOP approaches
    S = D[D.valid & (D.k >= 2) & (D.v >= 1) & (D.dH > 0) & (D.dH <= 40) & lead_ok].copy()
    O = D[D.valid & (D.k >= 2) & (D.v >= 2) & D.H.isna() & D.lead_gap.isna() & (D.turn20 < 15)].copy()
    S["bin"] = np.digitize(S.dH, [5, 10, 20, 30])
    stop_rows, per_win = {}, []
    for t in tags:
        s, o = S[S.tag == t], O[O.tag == t]
        el_s, el_o = elasticity(s, s.bin.to_numpy()), elasticity(o, np.zeros(len(o)))
        r = dict(decisions=int(len(s)), windows=int(s.id.nunique()), plan_has_stop=boot(s.has, s.id), E_minus_H_median=float((s.E - s.H).median()),
                 E_past_H_share=boot((s.E - s.H) > 2.0, s.id), vend_median=float(s.vend.median()), v_median=float(s.v.median()),
                 elast_stop=el_s, elast_open=el_o, ratio=el_s / el_o if el_o else np.nan, open_decisions=int(len(o)))
        by = {}
        for b, lab in enumerate(["0-5", "5-10", "10-20", "20-30", "30-40"]):
            x = s[s.bin == b]
            by[lab] = dict(n=int(len(x)), v=float(x.v.median()) if len(x) else np.nan, has=float(x.has.mean()) if len(x) else np.nan,
                           E_minus_H=float((x.E - x.H).median()) if len(x) else np.nan)
        r["by_dist"] = by
        cls = {"recedes": 0, "between": 0, "pinned": 0, "short": 0}
        for w, g in s.groupby("id"):
            g = g.sort_values("k")
            if len(g) < 4 or np.ptp(g.s_e) < 3:
                c, b = "short", np.nan
            else:
                b = float(np.polyfit(g.s_e, g.E, 1)[0])
                c = "recedes" if b >= 0.7 else "pinned" if b <= 0.3 else "between"
            cls[c] += 1
            per_win.append(dict(tag=t, id=w, cls=c, slope=b, n=len(g), has=float(g.has.mean()), E_minus_H=float((g.E - g.H).median()),
                                v=float(g.v.median()), score=float(g.score.iloc[0]), type=g.wtype.iloc[0]))
        r["window_class"] = cls
        stop_rows[t] = r
    PW = pd.DataFrame(per_win)
    out_tab = {}
    for t in tags:
        p = PW[PW.tag == t]
        out_tab[t] = {grp: dict(windows=int(len(x)), **{c: int((x.cls == c).sum()) for c in ("recedes", "between", "pinned", "short")},
                                has=float(x.has.mean()) if len(x) else np.nan, E_minus_H=float(x.E_minus_H.median()) if len(x) else np.nan,
                                v=float(x.v.median()) if len(x) else np.nan)
                      for grp, x in (("fail (score 0)", p[p.score == 0]), ("partial", p[(p.score > 0) & (p.score < 1)]), ("pass (1)", p[p.score >= 1]))}
    R["stop"], R["stop_by_outcome"] = stop_rows, out_tab
    PW.to_csv(out / "stop_windows.csv.gz", index=False, float_format="%.3f")

    # ---------------- BRK / FOL (lead)
    L = D[D.valid & (D.k >= 2) & (D.lead_gap <= 40)].copy()
    lv = L.lead_v.fillna(0).to_numpy()
    reach = L.lead_gap.to_numpy()[:, None] + lv[:, None] * TT[None]
    L["brk"] = (L.v >= 2) & (L.v * 4 > L.lead_gap + lv * 4)
    L["clear"] = (arc[L.index] <= reach + 1e-6).all(1)
    L["ttc"] = L.lead_gap / np.maximum(L.v - lv, 1e-3)
    L["fol"] = (L.lead_gap >= 5) & (L.lead_gap <= 30) & ((L.v - lv).abs() < 1) & (L.v >= 2)
    brk_rows = {}
    for t in tags:
        b, f = L[(L.tag == t) & L.brk], L[(L.tag == t) & L.fol]
        first3 = b.sort_values("k").groupby("id").head(3)
        r = dict(decisions=int(len(b)), windows=int(b.id.nunique()), clear=boot(b.clear, b.id), clear_first3=boot(first3.clear, first3.id),
                 ttc_first3_median=float(first3.ttc.median()), a02_median=float(b.a02.median()), v_median=float(b.v.median()),
                 fol_decisions=int(len(f)), fol_clear=float(f.clear.mean()) if len(f) else np.nan, fol_a02=float(f.a02.median()) if len(f) else np.nan)
        r["by_ttc"] = {lab: dict(n=int(m.sum()), clear=float(b.clear[m].mean()) if m.sum() else np.nan)
                       for lab, m in (("0-1", b.ttc < 1), ("1-2", (b.ttc >= 1) & (b.ttc < 2)), ("2-3", (b.ttc >= 2) & (b.ttc < 3)), ("3+", b.ttc >= 3))}
        wt = W.loc[t]
        r["outcomes"] = dict(AF_front=int((wt.type == "AF-front").sum()), rear_hit=int((wt.type == "rear-hit").sum()),
                             brk_windows_zero=int(b.groupby("id").score.first().eq(0).sum()))
        brk_rows[t] = r
    R["brk"] = brk_rows

    # ---------------- loss table (frozen items, the subject's misses file)
    if Path(a.misses).exists():
        m = pd.read_csv(a.misses)
        R["loss_by_cause"] = (m.assign(lost=1 - m.score).groupby("cause").agg(windows=("id", "size"), lost=("lost", "sum"),
                              easy=("strat", lambda s: int((s.str.startswith("a")).sum()))).sort_values("lost", ascending=False).round(2).reset_index().to_dict("records"))
    (out / "report.json").write_text(json.dumps(R, indent=1, default=float))
    lines = ["| tag | STOP dec (win) | plan has stop | E - H med (m) | E past H + 2 m | v med | elast stop / open (ratio) | windows recede / between / pinned |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in stop_rows.items():
        c = r["window_class"]
        lines.append(f"| {t} | {r['decisions']} ({r['windows']}) | {r['plan_has_stop'][0]:.3f} | {r['E_minus_H_median']:+.1f} | {r['E_past_H_share'][0]:.2f} "
                     f"[{r['E_past_H_share'][1]:.2f}, {r['E_past_H_share'][2]:.2f}] | {r['v_median']:.1f} | {r['elast_stop']:.2f} / {r['elast_open']:.2f} ({r['ratio']:.2f}) "
                     f"| {c['recedes']} / {c['between']} / {c['pinned']} |")
    lines += ["", "| tag | BRK dec (win) | clear [CI] | clear first 3 [CI] | TTC first 3 | a02 | FOL n / clear / a02 | AF-front / rear-hit windows |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in brk_rows.items():
        lines.append(f"| {t} | {r['decisions']} ({r['windows']}) | {r['clear'][0]:.3f} [{r['clear'][1]:.2f}, {r['clear'][2]:.2f}] | {r['clear_first3'][0]:.3f} "
                     f"[{r['clear_first3'][1]:.2f}, {r['clear_first3'][2]:.2f}] | {r['ttc_first3_median']:.1f} | {r['a02_median']:+.2f} | {r['fol_decisions']} / "
                     f"{r['fol_clear']:.2f} / {r['fol_a02']:+.2f} | {r['outcomes']['AF_front']} / {r['outcomes']['rear_hit']} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(out_tab, indent=1, default=float))


if __name__ == "__main__":
    main()
