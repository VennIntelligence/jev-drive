#!/usr/bin/env python3
"""ROAD-PROBE report (plans/2026-10-12-road-probe-prereg.md), Mac .venv: the registered tables, lines, verdict, learning curve and
figures from a copy of the box read dir (rp.py read; gates.json, rp.json, r12.csv, nearest.csv, curve.csv, junction.csv,
r3_windows.csv, dec_held.parquet, dec_zero.parquet, bev.pkl).

  python experiments/road_probe/scripts/rp_report.py picks  --read tmp/rp_read     -> tmp/rp_read/picks.json (BEV decisions, for rp.py bev)
  python experiments/road_probe/scripts/rp_report.py report --read tmp/rp_read     -> results/rp_report.md, results/*.csv, figs/*.png
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
OUT, FIGS = REPO / "experiments/road_probe/results", REPO / "experiments/road_probe/figs"
NAMES = {"P": "probe (2 seeds, mean)", "P-p1000-s0": "probe seed 0", "P-p1000-s1": "probe seed 1", "lin-p1000-s0": "linear floor",
         "blind-p1000-s0": "blind floor (ego + command)", "P0": "shipped heads", "P0K": "shipped heads, 0.689 undone"}
SHOW = ["P", "P-p1000-s0", "P-p1000-s1", "lin-p1000-s0", "blind-p1000-s0", "P0", "P0K"]


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(r, a, lo, hi, d=3):
    return f"{f3(r[a], d)} [{f3(r[lo], d)}, {f3(r[hi], d)}]"


def md(cols, rows):
    return ["| " + " | ".join(cols) + " |", "|:--" + "|--:" * (len(cols) - 1) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


def picks(a):
    R = Path(a.read)
    W = pd.read_csv(R / "r3_windows.csv").dropna(subset=["werr"]).sort_values("werr").reset_index(drop=True)
    sel = [W.iloc[int(q * (len(W) - 1))] for q in (0.1, 0.4, 0.7, 0.95)]
    out = [["zero", r.scene, int(r.k_ev), r.group] for r in sel]
    H = pd.read_parquet(R / "dec_held.parquet")
    t = H[(H.turn40 >= 45) & H.L20.notna() & H.R20.notna()].copy()
    t["e"] = (t["P:L20"] - t.L20).abs() + (t["P:R20"] - t.R20).abs()
    t = t.sort_values("e").reset_index(drop=True)
    for q in (0.5, 0.9):
        r = t.iloc[int(q * (len(t) - 1))]
        out.append(["held", r.scene, int(r.k), "YS-L18-s1"])
    (R / "picks.json").write_text(json.dumps([o[:3] for o in out]))
    print(json.dumps(out, indent=1))


def report(a):
    R = Path(a.read)
    G, S = json.loads((R / "gates.json").read_text()), json.loads((R / "rp.json").read_text())
    T, NB, LC, JB = (pd.read_csv(R / f) for f in ("r12.csv", "nearest.csv", "curve.csv", "junction.csv"))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("r12.csv", "nearest.csv", "curve.csv", "junction.csv", "r3_windows.csv", "gates.json", "rp.json"):
        shutil.copyfile(R / f, OUT / f)
    c = S["counts"]
    r3 = S["r3"]
    L = ["# ROAD-PROBE: is city road geometry in the frozen Cinque vision tokens?", "",
         "Pre-registration: [../plans/2026-10-12-road-probe-prereg.md](../plans/2026-10-12-road-probe-prereg.md) (pushed before any read). "
         "Probe = decision 243's HEAD1 head on the frozen tokens (8 slots x 32 x 512) + the 20-dim ego (command included), trained on 69 419 on-policy "
         "rows of 1020 train scenes (28 log units); read on a fresh held run of `YS-L18-s1` (every decision of the held scenes) and on decision 269's 98 "
         "road-exit zero windows (served tokens by replay). Errors are |prediction - map label| in metres at x = camera + d in the rear-axle frame "
         "(ev.py's frame); the ego-lane label is the lane the logged route occupies at that distance. Medians with a scene-cluster bootstrap CI (B 2000).", "",
         f"Held: {c['held']} decisions on {c['held_scenes']} scenes, {c['straight']} straight (route turn < 5 deg over 40 m), {c['turn']} turns (>= 20 deg); "
         f"zero group: {c['zero']} decisions, {c['windows']} registered windows.", "", "## Gates", ""]
    gl = []
    for k, v in G.items():
        if k.startswith("G_"):
            gl.append([k, ", ".join(f"{q} {f3(x) if isinstance(x, float) else x}" for q, x in v.items() if q != "pass"), "pass" if v["pass"] else "FAIL"])
    L += md(["gate", "reading", ""], gl) + [""]
    # R3 and verdict
    P = r3["P"]
    L += ["## R3 (registered verdict): the 98 road-exit zero windows, 20 m ego-lane boundary error", "",
          f"**Verdict: {S['verdict']}.** Probe (2-seed mean) window median {ci(P, 'median', 'med_lo', 'med_hi')} m (line < 0.3 repairable, > 1.0 "
          f"do not see), share of windows with both boundaries within 0.5 m {ci(P, 'share', 'share_lo', 'share_hi')} (line >= 0.6); "
          f"{P['eval']} windows evaluable on at least one side, {P['both_eval']} on both.", ""]
    L += md(["predictor", "window median (m)", "share both <= 0.5 m", "windows (one side / both)"],
            [[NAMES.get(k, k), ci(r3[k], "median", "med_lo", "med_hi"), ci(r3[k], "share", "share_lo", "share_hi"), f"{r3[k]['eval']} / {r3[k]['both_eval']}"]
             for k in SHOW if k in r3]) + [""]
    # R1 / R2
    for sub, title in (("straight", "R1: held straight decisions"), ("turn", "R2: held turn decisions (route turn >= 20 deg)")):
        L += [f"## {title}, 20 m", ""]
        rows = []
        for k in SHOW:
            x = T[(T.subset == sub) & (T.pred == k)].set_index("q")
            rows.append([NAMES[k]] + [ci(x.loc[q], "median", "med_lo", "med_hi") for q in ("L20", "R20", "LR20")] + [ci(x.loc["LR20"], "p90", "p90_lo", "p90_hi"), int(x.loc["LR20", "n"])])
        L += md(["predictor", "left median (m)", "right median (m)", "both, median (m)", "both, p90 (m)", "n (boundary points)"], rows) + [""]
    x = T[(T.subset == "straight") & (T.pred == "P") & (T.q == "LR20")].iloc[0]
    L += [f"Secondary line (decision 269's 'eyes usable' (ii)-(iii), straight, both boundaries pooled): probe median {f3(x['median'])} (< 0.30) and p90 {f3(x['p90'])} "
          f"(< 0.75): **{'met' if x['median'] < 0.30 and x['p90'] < 0.75 else 'not met'}**. Shipped heads: both ego lines present (prob > 0.5) on "
          f"{f3(S['presence_P0']['straight'])} of straight and {f3(S['presence_P0']['turn'])} of turn decisions of this held run.", ""]
    L += ["### Decision 269's measure (distance to the nearest map lane boundary, 20 m, left and right predictions pooled)", ""]
    L += md(["subset", "predictor", "median (m)", "p90 (m)", "n"], [[r.subset, NAMES[r.pred], f"{f3(r.median)} [{f3(r.lo)}, {f3(r.hi)}]", f3(r.p90), int(r.n)]
                                                                   for r in NB.itertuples() if r.pred in SHOW]) + [""]
    L += ["### Descriptive: other distances, road edges, lane heading (medians; heading in degrees)", ""]
    qs = ("LR10", "LR20", "LR40", "E10", "E20", "E40", "h20_deg")
    for sub in ("straight", "gentle", "turn"):
        L += [f"{sub}:", ""] + md(["predictor"] + list(qs), [[NAMES[k]] + [f3(T[(T.subset == sub) & (T.pred == k) & (T.q == q)]["median"].iloc[0]) for q in qs] for k in SHOW]) + [""]
    # learning curve
    L += ["## Learning curve (probe seed 0, nested shares of the fit log units)", ""]
    L += md(["share", "log units", "scenes", "rows", "held straight 20 m median", "held turn 20 m median", "R3 window median", "R3 share"],
            [[f3(r.frac, 3), int(r.logs), int(r.scenes), int(r.rows), f3(r.straight), f3(r.turn), f3(r.r3), f3(r.r3_share)] for r in LC.itertuples()]) + [""]
    for k, v in S["slope"].items():
        L += [f"- {k}: slope {f3(v['per_doubling'])} m per doubling of log units [{f3(v['lo'])}, {f3(v['hi'])}] -> "
              f"**{'more data still helps' if v['hi'] < 0 else 'no reliable gain from more data'}** (registered rule: CI below 0)."]
    L += ["", "## Turn decisions by distance to the junction (held; ego-lane boundary error at 20 m, both sides pooled, median m)", ""]
    bins = list(dict.fromkeys(JB.bin))
    for sub in ("turn", "straight"):
        L += [f"{sub}:", ""] + md(["predictor"] + bins, [[NAMES[k]] + [f"{f3(JB[(JB.bin == b) & (JB.subset == sub) & (JB.pred == k)]['median'].iloc[0])} "
                                                                       f"({int(JB[(JB.bin == b) & (JB.subset == sub) & (JB.pred == k)]['n'].iloc[0])})" for b in bins]
                                                           for k in ("P", "lin-p1000-s0", "blind-p1000-s0", "P0", "P0K")]) + [""]
    L += ["## Exploratory: is the ego in an oncoming lane (decision 268 H3), AUC", ""]
    L += md(["predictor", "held AUC (pos / neg)", "zero rollouts AUC (pos / neg)"],
            [[NAMES[k], f"{f3(v['held']['auc'])} ({v['held']['pos']} / {v['held']['neg']})", f"{f3(v['zero']['auc'])} ({v['zero']['pos']} / {v['zero']['neg']})"]
             for k, v in S["onc"].items()]) + [""]
    (OUT / "rp_report.md").write_text("\n".join(L) + "\n")
    figures(R, T, LC, JB, S)
    print("\n".join(L))


def figures(R, T, LC, JB, S):
    sys.path.insert(0, str(REPO / "research"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIGS.mkdir(parents=True, exist_ok=True)
    col = {"P": ps.PALETTE["blue"], "lin-p1000-s0": ps.PALETTE["green"], "blind-p1000-s0": ps.PALETTE["purple"], "P0": ps.BASELINE, "P0K": ps.PALETTE["orange"]}
    lab = {"P": "probe", "lin-p1000-s0": "linear", "blind-p1000-s0": "blind", "P0": "shipped", "P0K": "shipped / 0.689"}
    # figure 1: error by distance (straight, turn), by scene type at 20 m, learning curve
    fig, ax = plt.subplots(1, 4, figsize=(ps.DOUBLE_COLUMN_IN, 1.9))
    for j, sub in enumerate(("straight", "turn")):
        for k in col:
            y = [T[(T.subset == sub) & (T.pred == k) & (T.q == q)].iloc[0] for q in ("LR10", "LR20", "LR40")]
            ax[j].errorbar([10, 20, 40], [r["median"] for r in y], yerr=[[r["median"] - r["med_lo"] for r in y], [r["med_hi"] - r["median"] for r in y]],
                           color=col[k], marker="o", ms=2.5, capsize=1.5, lw=0.9, label=lab[k])
        ax[j].set_xticks([10, 20, 40]), ax[j].set_xlabel("distance ahead (m)"), ax[j].set_ylim(0, None)
        ps.panel(ax[j], f"({'ab'[j]}) held {sub}")
    ax[0].set_ylabel("ego-lane boundary error (m)")
    ax[0].legend(fontsize=6.5, loc="upper left")
    W3 = pd.read_csv(R / "r3_windows.csv")
    groups = [("straight", "LR20"), ("gentle", "LR20"), ("turn", "LR20")]
    xs = np.arange(4)
    for i, k in enumerate(col):
        v = [T[(T.subset == s) & (T.pred == k) & (T.q == q)]["median"].iloc[0] for s, q in groups] + [S["r3"][k]["median"]]
        ax[2].bar(xs + (i - 2) * 0.16, v, 0.16, color=col[k])
    ax[2].axhline(0.3, color="#999999", lw=0.5, ls=":")
    ax[2].axhline(1.0, color="#999999", lw=0.5, ls=":")
    ax[2].set_xticks(xs, ["straight", "gentle", "turn", "R3 zeros"], fontsize=7)
    ps.panel(ax[2], "(c) 20 m, by scene type")
    ax[3].plot(LC.logs, LC.straight, "o-", color=col["P"], ms=2.5, label="held straight")
    ax[3].plot(LC.logs, LC.turn, "s-", color=col["P"], ms=2.5, alpha=0.6, label="held turn")
    ax[3].plot(LC.logs, LC.r3, "^-", color=ps.PALETTE["vermillion"], ms=2.5, label="R3 windows")
    ax[3].set_xscale("log", base=2), ax[3].set_xlabel("training log units"), ax[3].set_ylim(0, None)
    ax[3].legend(fontsize=6.5)
    ps.panel(ax[3], "(d) learning curve, 20 m")
    _ = W3
    fig.tight_layout(w_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"rp_errors.{ext}", bbox_inches="tight")
    plt.close(fig)
    # figure 2: error by distance to the junction on turn decisions
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 1.9))
    bins = list(dict.fromkeys(JB.bin))
    for k in col:
        x = JB[(JB.subset == "turn") & (JB.pred == k)].set_index("bin").loc[bins]
        ax.plot(range(len(bins)), x["median"], "o-", color=col[k], ms=2.5, label=lab[k])
    ax.set_xticks(range(len(bins)), [b.replace(" within 60 m", "").replace("in junction", "inside") for b in bins], fontsize=6.5, rotation=20)
    ax.set_ylabel("20 m error, turns (m)"), ax.set_xlabel("distance to the junction along the route"), ax.set_ylim(0, None)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"rp_junction.{ext}", bbox_inches="tight")
    plt.close(fig)
    # figure 3: BEV examples
    if not (R / "bev.pkl").exists():
        return
    bev = pickle.load(open(R / "bev.pkl", "rb"))
    picks = json.loads((R / "picks.json").read_text())
    dec = {g: pd.read_parquet(R / f"dec_{g}.parquet") for g in ("held", "zero")}
    W3 = W3.set_index("scene")
    fig, ax = plt.subplots(2, 3, figsize=(ps.DOUBLE_COLUMN_IN, 4.6))
    for a_, (g, scene, k) in zip(ax.ravel(), picks):
        b = bev[(g, scene, k)]
        for p in b["edges"]:
            a_.plot(-p[:, 1], p[:, 0], color="#BBBBBB", lw=0.5)
        for p in b["road"]:
            a_.plot(-p[:, 1], p[:, 0], color="#222222", lw=0.8)
        a_.plot(-b["log"][:, 1], b["log"][:, 0], color=ps.PALETTE["green"], lw=1.2)
        for i, (pr, ls) in enumerate(zip(b["pr"][1:3], ("--", "--"))):
            a_.plot(-b["ll"][i + 1], b["xs"], color=ps.BASELINE, lw=0.9, ls=ls)
            yk = b["cam"][1] + (b["ll"][i + 1] - b["cam"][1]) / 0.689
            a_.plot(-yk, b["xs"], color=ps.PALETTE["orange"], lw=0.7, ls=":")
        a_.plot(-b["plan"][:, 1], b["plan"][:, 0], color=ps.PALETTE["vermillion"], lw=1.1)
        D = dec[g]
        r = D[(D.scene == scene) & (D.k == k)].iloc[0]
        x = b["cam"][0] + np.array([10, 20, 40])
        for s, m in (("L", "o"), ("R", "o"), ("EL", "s"), ("ER", "s")):
            yp = np.array([r[f"P:{s}{d}"] for d in (10, 20, 40)])
            a_.plot(-yp, x, color=col["P"] if s in ("L", "R") else ps.PALETTE["sky_blue"], marker=m, ms=2.8, lw=0.8)
            yt = np.array([r[f"{s}{d}"] for d in (10, 20, 40)])
            a_.plot(-yt, x, ls="none", marker="x", ms=3, color="#000000" if s in ("L", "R") else "#666666")
        a_.set_xlim(-20, 20), a_.set_ylim(-10, 55), a_.set_aspect("equal")
        a_.tick_params(labelsize=6.5)
        tl = (f"zero ({W3.loc[scene, 'type']}), k = {k}" if g == "zero" else f"held turn {r.turn40:.0f} deg, k = {k}")
        a_.text(0.02, 0.98, tl, transform=a_.transAxes, va="top", fontsize=6.5)
    ax[1, 0].set_xlabel("lateral (m, right +)"), ax[0, 0].set_ylabel("ahead (m)")
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"rp_bev.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("picks", "report"))
    ap.add_argument("--read", required=True)
    a = ap.parse_args()
    dict(picks=picks, report=report)[a.cmd](a)
