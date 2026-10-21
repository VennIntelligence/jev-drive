#!/usr/bin/env python3
"""ROUND6 line C, FIX (plans/2026-10-21-fix-rows.md): one joint data-filter arm from the OFFROAD-MV anatomy (round5 results/offroad_mv.md §4).
R6FIX = R5LC-L18-s<seed>'s recorded argv (MV-LC) with one more draw-file change, draw R6FIX = R5LC with
  F1  ph15 (stretch f = 1.5) rows whose label peak lateral acceleration > 3 m/s^2 or whose fed v0 > 30 m/s,
  F2  PAI human rows (ph / ph05 / ph15) slowing without a lead and without stopping: v0 >= 4, 0.5 <= v_end < 0.6 v0, max |heading| < 20 deg,
      no r5 LEAD flag
at weight 0, each PAI part renormalised to its R5LC share (still 32 PAI human rows a batch). The features are offroad_mv.label_feats (same code
as the diagnosis), the LEAD flag is round 5's lead_rows.npz. Lane script: scripts/fix.sh. Box, repo root, envs/op-train python.

  fix.py rows                 F1 / F2 masks on the three PAI parts; R6FIX and R6FIXI (the same code, nothing dropped, asserted bit-equal to
                              R5LC) -> $DATA_DIR/runs/round6/fix/w/<out>.npz + results/fix/w_<out>.json; offline tables (removed rows per part /
                              class, F1 x F2 overlap, the ph15 lateral-acceleration distribution before / after, route-turn / junction rows
                              removed) -> results/fix/fix_rows.{md,json}
  fix.py smoke --tag T        round 5's smoke checks for a draw-file-only arm (r5.py smoke --arm L) -> results/fix/smoke_R6FIX.json
  fix.py verdict --json J --md M --pcs Q --read K=FACTS.csv ..   the registered rule, the summary table and the 9 stable OFFROAD-MV scenes
                              (FIX - R5LC per seed) -> appended to M, verdict into J
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/round3/scripts"), str(REPO / "experiments/round5/scripts")]
import r3  # noqa: E402
import r5  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round6/fix"
W6 = O / "w"
CR = DATA / "runs/op_parity/cache"
RES = REPO / "experiments/round6/results/fix"
A_LAT, V_MAX = 3.0, 30.0                                  # F1
F2_V0, F2_VSTOP, F2_FRAC, F2_H = 4.0, 0.5, 0.6, 20.0      # F2
JT_DEG = 45.0                                             # junction turn (offroad_mv STATES)
# offroad_mv.md §1: the 9 scenes offroad in >= 3 of the 4 MV reads (R3MV s0 / s1, MV-LC s0 / s1) where NTV0 is not offroad, with the mechanism
STABLE = {"6e7b42a3": "over-speed drift", "dc401449": "slow-down drift", "a2b415d5": "straight: lateral offset", "596af0c7": "curve: cut inside",
          "94c1d722": "turn: cut inside", "e1c6ae9f": "curve (fork): cut inside", "ea468790": "turn: cut inside", "0f4b8753": "slow-down drift",
          "7750265d": "straight: lateral offset"}


def wq(x, w, p):
    """draw-weighted quantile."""
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return float(x[i][min(np.searchsorted(c, p), len(c) - 1)])


def cmd_rows(a):
    from offroad_mv import label_feats
    B = np.load(r5.W5 / "R5LC.npz")
    names, w0, grp = B["names"].astype(str), B["w"].astype(np.float64), B["grp"]
    w3 = np.load(r5.W3 / "R3MV.npz")["w"].astype(np.float64)
    assert np.array_equal(np.load(r5.W3 / "R3MV.npz")["names"].astype(str), names), "R3MV rows != R5LC rows"
    assert np.array_equal(np.load(CR / "on1_JTPH/tab.npz")["names"].astype(str), names), "R5LC rows != on1_JTPH"
    idx = {x: i for i, x in enumerate(names)}
    LR = np.load(DATA / "runs/round5/r5/N/lead_rows.npz")
    EV = r3.pai_events()                                  # route turn ahead (stitched human path, own clock) per part row
    n = len(names)
    Z = {k: np.zeros(n, bool) for k in ("pai", "f1", "f1a", "f1b", "f2", "jt_label", "jt_ahead", "turn_ahead", "curve", "fast30", "slow_nostop")}
    alat, v0a, cls_all, part_of = np.zeros(n), np.zeros(n), np.full(n, "", dtype="U13"), np.full(n, "", dtype="U4")
    parts = {}
    for p in r3.PARTS:
        t = r3.load_part(p)
        nm = t["names"].astype(str)
        ix = np.array([idx[f"on:{p}:{x[3:]}"] for x in nm])
        v0, fut = t["speed"].astype(np.float64), t["fut"].astype(np.float64)
        F = label_feats(v0, fut)
        lp = {x: i for i, x in enumerate(LR[f"{p}_names"].astype(str))}
        lead = LR[f"{p}_lead"][np.array([lp[f"on:{p}:{x[3:]}"] for x in nm])].astype(bool)
        f1a = (F["alat"] > A_LAT) & (p == "ph15")
        f1b = (v0 > V_MAX) & (p == "ph15")
        f2 = (v0 >= F2_V0) & (F["vend"] >= F2_VSTOP) & (F["vend"] < F2_FRAC * v0) & (F["hmax"] < F2_H) & ~lead
        assert len(EV[p]["turn"]) == len(nm), f"{p}: pai_events rows != part rows"
        for k, m in (("f1a", f1a), ("f1b", f1b), ("f1", f1a | f1b), ("f2", f2), ("jt_label", F["hmax"] >= JT_DEG),
                     ("jt_ahead", EV[p]["dhead"] >= JT_DEG), ("turn_ahead", EV[p]["turn"]), ("curve", (F["hmax"] >= 15) & (F["hmax"] < JT_DEG)),
                     ("fast30", v0 > V_MAX), ("slow_nostop", (v0 >= F2_V0) & (F["vend"] >= F2_VSTOP) & (F["vend"] < F2_FRAC * v0) & (F["hmax"] < F2_H))):
            Z[k][ix] = m
        Z["pai"][ix] = True
        alat[ix], v0a[ix], part_of[ix] = F["alat"], v0, p
        cls_all[ix] = r3.label_class(v0, fut)[0]
        parts[p] = (ix, (Z["f1"][ix] | Z["f2"][ix]) & (w0[ix] > 0))
    W6.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    rec = dict(rule=dict(F1=f"ph15 rows: label peak lateral acceleration > {A_LAT} m/s^2 or fed v0 > {V_MAX} m/s (offroad_mv.label_feats)",
                         F2=f"ph / ph05 / ph15 rows: v0 >= {F2_V0}, {F2_VSTOP} <= v_end < {F2_FRAC} v0, max |heading| < {F2_H} deg, no r5 LEAD flag",
                         base="R5LC (MV-LC's draw); each PAI part renormalised to its R5LC share"))
    for out_, nodrop in (("R6FIX", False), ("R6FIXI", True)):
        w = r5.part_drop(w0, parts, nodrop)
        if nodrop:
            assert np.array_equal(w, w0), "R6FIXI != R5LC"
        gs = Counter()
        for g_, x in zip(grp.astype(str), w):
            gs[g_] += x
        np.savez(W6 / f"{out_}.npz", names=B["names"], w=w, grp=grp)
        r = dict(base="R5LC", out=out_, rule=("R5LC with F1 (ph15) and F2 (all PAI parts) rows -> 0, each part renormalised to its R5LC share" if not nodrop
                                              else "identity: R6FIX's code with no row dropped (bit-equal to R5LC)"),
                 group_share={k: float(v) for k, v in sorted(gs.items()) if v > 0}, pai_drawable=int(sum((w[v[0]] > 0).sum() for v in parts.values())))
        r3.dump(RES / f"w_{out_}.json", r)
        rec[out_] = r
    wf = np.load(W6 / "R6FIX.npz")["w"].astype(np.float64)
    K = r3.K_ON
    # per part: removed rows by rule (drawn under R5LC), overlap, class, junction / route-turn rows
    rec["parts"] = {}
    for p, (ix, drop) in parts.items():
        dr = w0[ix] > 0
        g = lambda k: Z[k][ix]  # noqa: E731
        f1, f2 = g("f1") & dr, g("f2") & dr
        rr = dict(drawable=int(dr.sum()), f1=int(f1.sum()), f1a=int((g("f1a") & dr).sum()), f1b=int((g("f1b") & dr).sum()), f2=int(f2.sum()),
                  both=int((f1 & f2).sum()), removed=int(drop.sum()), removed_share_draw=float(w0[ix][drop].sum() / w0[ix][dr].sum()),
                  removed_per_batch=float(K * w0[ix][drop].sum()), f1_per_batch=float(K * w0[ix][f1].sum()), f2_per_batch=float(K * w0[ix][f2].sum()),
                  r3mv_f1=int((g("f1") & (w3[ix] > 0)).sum()), r3mv_f2=int((g("f2") & (w3[ix] > 0)).sum()),
                  by_class={c: dict(drawable=int((dr & (cls_all[ix] == c)).sum()), removed=int((drop & (cls_all[ix] == c)).sum())) for c in r3.CLASSES},
                  scenes_removed=int(len({x.split(":")[3] for x in names[ix][drop]})), scenes_drawable=int(len({x.split(":")[3] for x in names[ix][dr]})))
        for k in ("jt_label", "jt_ahead", "turn_ahead", "curve"):
            m = g(k) & dr
            rr[k] = dict(drawable=int(m.sum()), removed=int((m & drop).sum()), removed_f1=int((m & f1).sum()), removed_f2=int((m & f2).sum()))
        rec["parts"][p] = rr
    pm = Z["pai"]
    mv = pm & (v0a >= r3.V_MV)
    # rows a batch by state (PAI human rows; 64 pool rows x draw mass) under R3MV, R5LC, R6FIX
    st = dict(all=pm, junction_label=Z["jt_label"], junction_ahead=Z["jt_ahead"], route_turn_ahead=Z["turn_ahead"], curve=Z["curve"],
              fast30=Z["fast30"], slow_nostop=Z["slow_nostop"], slow_nostop_nolead=Z["f2"], f1=Z["f1"])
    rec["per_batch"] = {k: {s: float(K * ww[pm & m].sum()) for s, m in st.items()} for k, ww in (("R3MV", w3), ("R5LC", w0), ("R6FIX", wf))}
    rec["per_batch_part"] = {k: {p: {s: float(K * ww[(part_of == p) & m].sum()) for s, m in st.items()} for p in r3.PARTS}
                             for k, ww in (("R5LC", w0), ("R6FIX", wf))}
    rec["unique_pai_rows"] = {k: int((ww[pm] > 0).sum()) for k, ww in (("R3MV", w3), ("R5LC", w0), ("R6FIX", wf))}
    # lateral acceleration of the drawn moving rows per part, before / after (draw-weighted)
    rec["alat"] = {}
    for p in r3.PARTS:
        for k, ww in (("R5LC", w0), ("R6FIX", wf)):
            for sname, sm in (("moving", mv), ("curve", mv & Z["curve"]), ("junction", mv & Z["jt_label"])):
                m = sm & (part_of == p) & (ww > 0)
                if m.sum() < 10:
                    rec["alat"][f"{p} {sname} {k}"] = dict(n=int(m.sum()))
                    continue
                x, wx = alat[m], ww[m]
                rec["alat"][f"{p} {sname} {k}"] = dict(n=int(m.sum()), q50=wq(x, wx, .5), q90=wq(x, wx, .9), q99=wq(x, wx, .99), max=float(x.max()),
                                                       gt2=float(wx[x > 2].sum() / wx.sum()), gt3=float(wx[x > 3].sum() / wx.sum()),
                                                       v_gt30=float(wx[v0a[m] > V_MAX].sum() / wx.sum()), v50=wq(v0a[m], wx, .5))
    r3.dump(RES / "fix_rows.json", rec)
    (RES / "fix_rows.md").write_text(rows_md(rec) + "\n")
    print(rows_md(rec))


def rows_md(rec):
    P = rec["parts"]
    L = ["# FIX rows (round 6 line C): F1 + F2 on MV-LC's draw (offline, before training)", "",
         f"- F1: {rec['rule']['F1']}.", f"- F2: {rec['rule']['F2']}.", f"- Base: {rec['rule']['base']} (ph 0.25, ph05 0.125, ph15 0.125 of the 64 pool rows: "
         "32 PAI human rows a batch).", "",
         "## Removed rows (drawn under R5LC)", "",
         "| part | drawable | F1 | F1a (a_lat > 3) | F1b (v0 > 30) | F2 | F1 and F2 | removed | of part (draw-weighted) | rows a batch removed | "
         "F1 / F2 a batch | scenes removed / drawable | under R3MV: F1 / F2 (offroad_mv: 23 693 / 9 735) |", "|:--|" + "--:|" * 12]
    for p, r in P.items():
        L.append(f"| {p} | {r['drawable']} | {r['f1']} | {r['f1a']} | {r['f1b']} | {r['f2']} | {r['both']} | {r['removed']} | {r['removed_share_draw']:.1%} | "
                 f"{r['removed_per_batch']:.2f} | {r['f1_per_batch']:.2f} / {r['f2_per_batch']:.2f} | {r['scenes_removed']} / {r['scenes_drawable']} | {r['r3mv_f1']} / {r['r3mv_f2']} |")
    tot = {k: sum(r[k] for r in P.values()) for k in ("drawable", "f1", "f1a", "f1b", "f2", "both", "removed", "r3mv_f1", "r3mv_f2")}
    tb = {k: sum(r[k] for r in P.values()) for k in ("removed_per_batch", "f1_per_batch", "f2_per_batch")}
    L.append(f"| all PAI | {tot['drawable']} | {tot['f1']} | {tot['f1a']} | {tot['f1b']} | {tot['f2']} | {tot['both']} | {tot['removed']} | | "
             f"{tb['removed_per_batch']:.2f} | {tb['f1_per_batch']:.2f} / {tb['f2_per_batch']:.2f} | | {tot['r3mv_f1']} / {tot['r3mv_f2']} |")
    L += ["", "Removed / drawable by label class (r3.label_class):", "", "| part | " + " | ".join(r3.CLASSES) + " |", "|:--|" + "--:|" * len(r3.CLASSES)]
    L += [f"| {p} | " + " | ".join(f"{r['by_class'][c]['removed']} / {r['by_class'][c]['drawable']}" for c in r3.CLASSES) + " |" for p, r in P.items()]
    L += ["", "## Route-turn and junction rows removed (risk to the turn fix)", "",
          f"`junction (label)` = the 4 s label turns >= {JT_DEG:.0f} deg; `junction ahead` = the human's stitched path turns >= {JT_DEG:.0f} deg within 4 f s "
          "or 20 m (r3.pai_events, own clock); `route turn ahead` = the same >= 15 deg (MVE's turn event); `curve` = label 15-45 deg. Removed / drawable "
          "(of which by F1 / by F2):", "",
          "| part | junction (label) | junction ahead | route turn ahead | curve |", "|:--|--:|--:|--:|--:|"]
    for p, r in P.items():
        L.append(f"| {p} | " + " | ".join(f"{r[k]['removed']} / {r[k]['drawable']} ({r[k]['removed_f1']} / {r[k]['removed_f2']})"
                                            for k in ("jt_label", "jt_ahead", "turn_ahead", "curve")) + " |")
    pb = rec["per_batch"]
    cols = ("all", "junction_label", "junction_ahead", "route_turn_ahead", "curve", "fast30", "slow_nostop", "slow_nostop_nolead", "f1")
    L += ["", "PAI human rows a batch by state (64 pool rows x draw mass; F2 = slow-down, no stop, no lead; F1 = the F1 rows of ph15):", "",
          "| draw | " + " | ".join(cols) + " | unique PAI rows |", "|:--|" + "--:|" * (len(cols) + 1)]
    L += [f"| {k} | " + " | ".join(f"{v[c]:.2f}" for c in cols) + f" | {rec['unique_pai_rows'][k]} |" for k, v in pb.items()]
    L += ["", "Per part, rows a batch (R5LC -> R6FIX):", "", "| part | " + " | ".join(cols[1:6]) + " |", "|:--|" + "--:|" * 5]
    for p in r3.PARTS:
        a_, b_ = rec["per_batch_part"]["R5LC"][p], rec["per_batch_part"]["R6FIX"][p]
        L.append(f"| {p} | " + " | ".join(f"{a_[c]:.2f} -> {b_[c]:.2f}" for c in cols[1:6]) + " |")
    L += ["", "## Label peak lateral acceleration (drawn moving rows, draw-weighted; m/s^2)", "",
          "| part / state / draw | n | v50 | a50 | a90 | a99 | max | > 2 | > 3 | v0 > 30 |", "|:--|" + "--:|" * 9]
    for k, r in rec["alat"].items():
        if "q50" not in r:
            L.append(f"| {k} | {r['n']} | | | | | | | | |")
            continue
        L.append(f"| {k} | {r['n']} | {r['v50']:.1f} | {r['q50']:.2f} | {r['q90']:.2f} | {r['q99']:.2f} | {r['max']:.1f} | {r['gt2']:.1%} | {r['gt3']:.1%} | "
                 f"{r['v_gt30']:.1%} |")
    L += ["", "navtrain reference (offroad_mv.md §4, uniform): all moving a50 0.35 / a90 1.58 / > 3 0.3 %; curve 1.07 / 1.95 / 0.3 %; junction 1.69 / 2.51 / 2.0 %; "
          "no row >= 25 m/s.", "", "Identity: R6FIXI (the same code, nothing dropped) = R5LC.npz bit for bit; non-PAI weights unchanged (r5.part_drop asserts). "
          "Files: fix_rows.json, w_R6FIX.json, w_R6FIXI.json."]
    return "\n".join(L)


def cmd_smoke(a):
    r5.RES = RES
    r5.cmd_smoke(argparse.Namespace(tag=a.tag, arm="L", wfile="R6FIX", k_s3=0, suffix="", index=""))


def cmd_verdict(a):
    import pandas as pd
    J = json.loads(Path(a.json).read_text())
    Q = json.loads(Path(a.pcs).read_text())
    R, P = J["reads"], J["pairs"]
    lat = lambda k: R[k]["z_offroad"] + R[k]["z_zero-other"]  # noqa: E731
    cap = lat("NTV0-ag") + 3
    rows, ok = [], True
    for s in ("0", "1"):
        k, q = f"FIX-s{s}-ag", Q[f"R6FIX-s{s}+AG"]
        d = P[f"{k}:NTV0-ag"]["mean"]
        b, l_, r_ = d["mean"] >= 0, lat(k) <= cap, q["diff_ci"][1] >= 0
        ok &= b and l_ and r_
        rows.append(f"| {k} | {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'yes' if b else 'no'} | {lat(k)} (offroad {R[k]['z_offroad']}, "
                    f"zero-other {R[k]['z_zero-other']}) | {'yes' if l_ else 'no'} | {q['diff_median']:+.0f} [{q['diff_ci'][0]:+.0f}, {q['diff_ci'][1]:+.0f}] | "
                    f"{'yes' if r_ else 'no'} |")
    L = ["", "## Registered rule (plans/2026-10-21-fix-rows.md)", "",
         f"adopt-candidate = both FIX seeds: mean - NTV0 >= 0 (paired point estimate), offroad + zero-other zeros <= NTV0's + 3 = {cap}, local PCS ruler "
         "difference to NTV0 with a 95 % CI upper bound >= 0.", "",
         "| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | ruler - NTV0 [95 % CI] | CI upper >= 0 |", "|:--|--:|:-:|--:|:-:|--:|:-:|",
         *rows, "", f"**FIX: {'adopt-candidate' if ok else 'not an adopt-candidate'}.**", "",
         f"## Summary ({J['scenes']}-scene common set; new = on the scenes NTV0 scores 1.0)", "",
         "| read | mean | - R5LC same seed [95 % CI] | offroad | zero-other | route-turn | rear-hit | slow | launch | new zeros / partials | ruler PCS - NTV0 |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    nb = J.get("newbreaks", {}).get("reads", {})
    for k, lab in (("NTV0-ag", "NTV0-s0+AG"), ("R5LC-s0-ag", "R5LC-s0+AG"), ("R5LC-s1-ag", "R5LC-s1+AG"), ("FIX-s0-ag", "R6FIX-s0+AG"), ("FIX-s1-ag", "R6FIX-s1+AG")):
        r = R[k]
        pr = P.get(f"{k}:R5LC-{k.split('-')[1]}-ag", {}).get("mean") if k.startswith("FIX") else None
        q = Q.get(lab, {})
        L.append(f"| {k} | {r['mean']:.3f} | " + (f"{pr['mean']:+.3f} [{pr['lo']:+.3f}, {pr['hi']:+.3f}]" if pr else "") + f" | {r['z_offroad']} | "
                 f"{r['z_zero-other']} | {r['z_route-turn']} | {r['p_rear-hit']} | {r['p_slow']} | {r['launch']} | "
                 + (f"{nb[k]['zeros']} / {nb[k]['partials']}" if k in nb else "") + " | "
                 + (f"{q['diff_median']:+.0f} [{q['diff_ci'][0]:+.0f}, {q['diff_ci'][1]:+.0f}]" if "diff_ci" in q else "ref") + " |")
    # the 9 stable OFFROAD-MV scenes
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    key = lambda s: s.split("-")[1][:8] if s.startswith("clipgt-") else s[:8]  # noqa: E731
    sc = {key(s): s for s in F["NTV0-ag"].index}
    ks = list(F)
    L += ["", "## The 9 stable OFFROAD-MV scenes (offroad_mv.md §1): score / fclass per read", "",
          "| scene | mechanism | " + " | ".join(ks) + " | FIX - R5LC s0 | FIX - R5LC s1 |", "|:--|:--|" + ":-:|" * len(ks) + "--:|--:|"]
    tot, offr, st = {0: 0.0, 1: 0.0}, Counter(), {}
    for h, mech in STABLE.items():
        s = sc.get(h)
        if s is None:
            L.append(f"| {h} | {mech} | missing |")
            continue
        c = {k: (float(F[k].score.get(s, np.nan)), str(F[k].fclass.get(s, ""))) for k in ks}
        for k, (_, fc) in c.items():
            offr[k] += fc == "offroad"
        dd = {i: c[f"FIX-s{i}-ag"][0] - c[f"R5LC-s{i}-ag"][0] for i in (0, 1)}
        for i in (0, 1):
            tot[i] += np.nan_to_num(dd[i])
        st[h] = dict(mech=mech, reads={k: dict(score=v[0], fclass=v[1]) for k, v in c.items()}, fix_minus_r5lc={str(i): dd[i] for i in (0, 1)})
        L.append(f"| {h} | {mech} | " + " | ".join(f"{v[0]:.2f} {v[1]}" for v in c.values()) + f" | {dd[0]:+.2f} | {dd[1]:+.2f} |")
    L += ["", "Offroad among the 9: " + ", ".join(f"{k} {offr[k]}" for k in ks) + f". Score sum FIX - R5LC: s0 {tot[0]:+.2f}, s1 {tot[1]:+.2f}."]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J["verdict"] = dict(adopt_candidate=bool(ok), cap=cap, stable9=st, stable9_offroad=dict(offr), stable9_sum={str(i): v for i, v in tot.items()})
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("rows")
    p = sp.add_parser("smoke"); p.add_argument("--tag", required=True)
    p = sp.add_parser("verdict"); p.add_argument("--json", required=True); p.add_argument("--md", required=True); p.add_argument("--pcs", required=True)
    p.add_argument("--read", nargs="+", required=True, help="K=FACTS.csv: NTV0-ag, R5LC-s0/s1-ag, FIX-s0/s1-ag")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
