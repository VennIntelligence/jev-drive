#!/usr/bin/env python3
"""Prereg 补记 D read 1: decompose the WAIT pilots' dev-ADE rise (gate 4) on the gate's own dev rows (wt.py devall -> explore/devall.npz).

  wt_devdecomp.py <devall.npz> <out stem>  -> <stem>.json, <stem>.md

Per arm J (WTJ2P / WTJ5P) against the control C (WTCP) and P0 (shipped Cinque's stored teacher plan): along / cross-track split of the ADE
rise (error projected on the human path's tangent; the two orders of replacing C's component averaged, so the parts sum to the rise), signed
arc-length bias at 2 s / 4 s (plan - human), by the human's v0 bin, behaviour class and agent context, the share carried by the top 5 % of
rows, and plan-wait / human-go flips. CIs: log-clustered bootstrap (jevdrive.stats).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import stats  # noqa: E402

C, P0 = "WTCP-L18-s0", "P0"
VBINS = [(0, 0.5), (0.5, 3), (3, 8), (8, 99)]


def arc(P):
    """(n, 8, >=2) poses from the rear-axle origin -> (n, 8) cumulative arc length."""
    seg = np.hypot(*np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[..., :2]], 1), axis=1).transpose(2, 0, 1))
    return np.cumsum(seg, 1)


def tangent(G):
    """Unit tangent of the human path at each of its 8 poses (central differences from the origin); stationary stretches -> +x."""
    Z = np.concatenate([np.zeros_like(G[:, :1, :2]), G[..., :2]], 1)
    d = np.concatenate([Z[:, 2:] - Z[:, :-2], Z[:, -1:] - Z[:, -2:-1]], 1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(n > 0.2, d / np.maximum(n, 1e-9), np.array([1.0, 0.0]))


def parts(P, G, T):
    e = P[..., :2] - G[..., :2]
    a = (e * T).sum(-1)
    c = e[..., 0] * -T[..., 1] + e[..., 1] * T[..., 0]
    return a, c


def ade(a, c):
    return np.hypot(a, c).mean(1)


def main(src, stem):
    z = np.load(src, allow_pickle=False)
    tags = [t for t in z["tags"].tolist() if t not in (C, P0)]
    G, v0, logs = z["fut"].astype(np.float64), z["speed"].astype(np.float64), z["log"]
    plans = {t: z[f"plan_{t}"].astype(np.float64) for t in z["tags"].tolist()}
    T = tangent(G)
    AC = {t: parts(P, G, T) for t, P in plans.items()}
    E = {t: ade(*AC[t]) for t in plans}
    sG = arc(G)
    S = {t: arc(P) for t, P in plans.items()}
    vend = np.hypot(*(G[:, 7, :2] - G[:, 6, :2]).T) / 0.5
    beh = np.select([(v0 < 0.5) & (sG[:, 7] < 1), (v0 < 2) & (vend >= v0 + 1), vend <= v0 - 1.5], ["stopped", "starting", "braking"], "cruising")
    closing = v0 + z["lead_close"]                      # ego speed + (lead x(t0) - x(0.5 s)) / 0.5 in the t0 frame = closing speed
    near = z["lead_gap"] < 20
    lowttc = near & (closing > 0.1) & (z["lead_gap"] / np.maximum(closing, 1e-3) < 4)
    ped = z["ped"].astype(bool)
    ctx = np.select([lowttc, near, ped], ["low TTC", "lead near", "pedestrian"], "free road")
    ctx = np.where(z["ctx_ok"], ctx, "no label")
    n = len(G)
    out = dict(rows=n, logs=int(len(set(logs.tolist()))), ade={t: float(E[t].mean()) for t in plans},
               behaviour_counts={k: int((beh == k).sum()) for k in np.unique(beh)}, context_counts={k: int((ctx == k).sum()) for k in np.unique(ctx)}, arms={})
    for t in tags:
        r = {}
        for ref in (C, P0):
            aJ, cJ = AC[t]
            aR, cR = AC[ref]
            d_along = 0.5 * ((ade(aJ, cJ) - ade(aR, cJ)) + (ade(aJ, cR) - ade(aR, cR)))      # per row; d_along + d_cross = E[t] - E[ref]
            d_cross = (E[t] - E[ref]) - d_along
            r[f"vs_{ref}"] = dict(d_ade=stats.fmt(stats.paired(E[t], E[ref], groups=logs), ".4f"), d_ade_mean=float((E[t] - E[ref]).mean()),
                                  along=float(d_along.mean()), cross=float(d_cross.mean()), along_share=float(d_along.mean() / (E[t] - E[ref]).mean()),
                                  mean_abs_along=[float(np.abs(aJ).mean()), float(np.abs(aR).mean())], mean_abs_cross=[float(np.abs(cJ).mean()), float(np.abs(cR).mean())])
        dE = E[t] - E[C]
        tot = dE.sum()
        bias = {f"{k}s": dict(J=float((S[t][:, i] - sG[:, i]).mean()), C=float((S[C][:, i] - sG[:, i]).mean()), P0=float((S[P0][:, i] - sG[:, i]).mean()),
                              diff=stats.fmt(stats.paired(S[t][:, i] - sG[:, i], S[C][:, i] - sG[:, i], groups=logs), ".3f")) for k, i in ((2, 3), (4, 7))}
        def by(lab):
            res = {}
            for k in np.unique(lab):
                m = lab == k
                res[str(k)] = dict(n=int(m.sum()), ade_J=float(E[t][m].mean()), ade_C=float(E[C][m].mean()), ade_P0=float(E[P0][m].mean()),
                                   share_of_rise=float(dE[m].sum() / tot), bias4_J=float((S[t][m, 7] - sG[m, 7]).mean()), bias4_C=float((S[C][m, 7] - sG[m, 7]).mean()),
                                   bias2_J=float((S[t][m, 3] - sG[m, 3]).mean()), bias2_C=float((S[C][m, 3] - sG[m, 3]).mean()))
            return res
        vb = np.select([(v0 >= lo) & (v0 < hi) for lo, hi in VBINS], [f"{lo}-{hi}" for lo, hi in VBINS], "?")
        cross = np.char.add(np.char.add(beh.astype(str), " / "), np.where(ctx == "free road", "free", "wait-like"))
        k = max(1, int(round(0.05 * n)))
        top = np.argsort(-dE)[:k]
        tw = (S[t][:, 7] < 1) & (sG[:, 7] >= 3)                                      # plan waits (4 s arc < 1 m) where the human goes >= 3 m
        cw = (S[C][:, 7] < 1) & (sG[:, 7] >= 3)
        r |= dict(bias=bias, by_v0=by(vb), by_behaviour=by(beh), by_context=by(ctx), by_behaviour_context=by(cross),
                  top5=dict(rows=k, share_of_rise=float(dE[top].sum() / tot), logs=int(len(set(logs[top].tolist()))),
                            behaviour={b: int((beh[top] == b).sum()) for b in np.unique(beh[top])}, context={c: int((ctx[top] == c).sum()) for c in np.unique(ctx[top])},
                            v0_med=float(np.median(v0[top])), human_arc4_med=float(np.median(sG[top, 7])), plan_arc4_med_J=float(np.median(S[t][top, 7])),
                            plan_arc4_med_C=float(np.median(S[C][top, 7])), plan_short_share=float(np.mean(S[t][top, 7] < S[C][top, 7])),
                            plan_wait_human_go=int(tw[top].sum()),
                            examples=[dict(name=str(z["names"][i]), log=str(logs[i]), v0=round(float(v0[i]), 2), beh=str(beh[i]), ctx=str(ctx[i]),
                                           human_arc4=round(float(sG[i, 7]), 2), J_arc4=round(float(S[t][i, 7]), 2), C_arc4=round(float(S[C][i, 7]), 2),
                                           dADE=round(float(dE[i]), 2), lead_gap=round(float(z["lead_gap"][i]), 1)) for i in top[:15]]),
                  plan_wait_human_go=dict(J=int(tw.sum()), C=int(cw.sum()), share_of_rise=float(dE[tw].sum() / tot)),
                  rows_worse_share=float((dE > 0).mean()), rise_from_rows_with_dE_gt_1m=float(dE[dE > 1].sum() / tot))
        aJ, cJ = AC[t]
        aR, cR = AC[C]
        dal = 0.5 * ((ade(aJ, cJ) - ade(aR, cJ)) + (ade(aJ, cR) - ade(aR, cR)))
        r["top5"]["along_share"] = float(dal[top].sum() / dE[top].sum())         # along-track part of the top rows' rise
        out["arms"][t] = r
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.json").write_text(json.dumps(out, indent=1))
    L = [f"# dev-ADE decomposition (补记 D read 1): {n} gate-4 dev rows, {out['logs']} logs", "",
         "ADE: " + ", ".join(f"{t} {v:.4f}" for t, v in out["ade"].items()), ""]
    for t, r in out["arms"].items():
        v = r[f"vs_{C}"]
        L += [f"## {t}", "", f"- vs C: dADE {v['d_ade']} (along {v['along']:+.4f}, cross {v['cross']:+.4f}; along share {v['along_share']:.2f}); "
              f"vs P0: dADE {r['vs_P0']['d_ade']} (along share {r['vs_P0']['along_share']:.2f})",
              "- signed arc bias (plan - human, m): " + "; ".join(f"{k}: J {b['J']:+.3f}, C {b['C']:+.3f}, P0 {b['P0']:+.3f}, J - C {b['diff']}" for k, b in r["bias"].items()),
              f"- top 5 % rows ({r['top5']['rows']}, {r['top5']['logs']} logs) carry {r['top5']['share_of_rise']:.2f} of the rise (along share {r['top5']['along_share']:.2f}); "
              f"behaviour {r['top5']['behaviour']}, context {r['top5']['context']}; v0 med {r['top5']['v0_med']:.2f}, human 4 s arc med {r['top5']['human_arc4_med']:.1f}, "
              f"J {r['top5']['plan_arc4_med_J']:.1f}, C {r['top5']['plan_arc4_med_C']:.1f}; plan shorter than C in {r['top5']['plan_short_share']:.2f}",
              f"- plan waits (4 s arc < 1 m) where the human goes (>= 3 m): J {r['plan_wait_human_go']['J']} rows vs C {r['plan_wait_human_go']['C']}, "
              f"{r['plan_wait_human_go']['share_of_rise']:.2f} of the rise; rows worse than C {r['rows_worse_share']:.2f}; rows with dADE > 1 m carry {r['rise_from_rows_with_dE_gt_1m']:.2f}", ""]
        for name in ("by_v0", "by_behaviour", "by_context", "by_behaviour_context"):
            L += [f"| {name[3:]} | n | ADE J | ADE C | ADE P0 | share of rise | 2 s bias J / C | 4 s bias J / C |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
            for k, b in r[name].items():
                L.append(f"| {k} | {b['n']} | {b['ade_J']:.3f} | {b['ade_C']:.3f} | {b['ade_P0']:.3f} | {b['share_of_rise']:.2f} | {b['bias2_J']:+.2f} / {b['bias2_C']:+.2f} | "
                         f"{b['bias4_J']:+.2f} / {b['bias4_C']:+.2f} |")
            L.append("")
        L += ["top rows (by dADE vs C):", "", "| log | v0 | behaviour | context | lead gap | human 4 s arc | J 4 s arc | C 4 s arc | dADE |", "|---|--:|---|---|--:|--:|--:|--:|--:|"]
        L += [f"| {e['log']} | {e['v0']} | {e['beh']} | {e['ctx']} | {e['lead_gap']} | {e['human_arc4']} | {e['J_arc4']} | {e['C_arc4']} | {e['dADE']} |" for e in r["top5"]["examples"]]
        L.append("")
    Path(f"{stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
