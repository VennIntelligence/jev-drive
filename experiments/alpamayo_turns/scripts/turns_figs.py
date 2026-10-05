#!/usr/bin/env python
"""Figures for alpamayo_turns (runs on the Mac over a pulled copy of the cache).

  python turns_figs.py <cache dir with alp/*.npz and op/preds.npz>

  figs/turn_<arm>.png  the camera images the arm actually fed (last of the 4 frames per camera; openpilot: its road / wide
                       luma frames) and the predicted vs logged trajectory and heading on one turn (the pilot case with the
                       median logged heading change, primary t0; rule fixed before looking at outputs)
  figs/summary.png     heading gain A_H and lag50 per case and arm, and the paired contrasts with 95 % CIs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC.parents[1]))
T = np.arange(1, 65) * 0.1
CAM_NAMES = {0: "cross-left 120°", 1: "front-wide 120°", 2: "cross-right 120°", 6: "front-tele 30°"}
ARM_LABEL = {"A": "A: 4 cameras", "B": "B: front-wide + tele", "B1": "B1: front-wide only", "An": "An: 4 cameras + nav",
             "C": "C: 3 cameras, no tele (post-hoc)", "B1n": "B1n: front-wide only + nav (post-hoc)", "Tblk": "Tblk: 4 cameras, tele frames black (post-hoc)", "B1t": "B1t: front-wide + black tele (post-hoc)",
             "Bn": "Bn: front-wide + tele + nav", "OP": "OP: openpilot Cinque (front-wide reprojected)"}
COL = {"A": "#1f6fb4", "B": "#d9822b", "B1": "#b85450", "An": "#2a9d8f", "Bn": "#e9c46a", "OP": "#6c757d", "C": "#7b5ea7",
       "B1n": "#c9a0a0", "Tblk": "#264653", "B1t": "#f4a261"}


def showcase(cases, alp):
    d = [(abs(np.degrees(np.load(alp / f"{c['clip']}_p.npz")["gt_yaw"][-1])), c) for c in cases
         if (alp / f"{c['clip']}_p.npz").exists()]
    d.sort(key=lambda x: x[0])
    return d[len(d) // 2][1]


def bev(ax, z, xy, gxy, col):
    h = z["hist_xyz"]
    ax.plot(-h[:, 1], h[:, 0], color="0.6", lw=2, label="history 1.6 s")
    for k, p in enumerate(xy):
        ax.plot(-p[:, 1], p[:, 0], color=col, lw=1.2, alpha=0.75, label="prediction (6 samples)" if k == 0 else None)
    ax.plot(-gxy[:, 1], gxy[:, 0], "k--", lw=2, label="logged 6.4 s")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("lateral (m, right +)")
    ax.set_ylabel("forward (m)")
    ax.legend(fontsize=7, loc="best")


def heading(ax, yaw, gyaw, col):
    for k, y in enumerate(yaw):
        ax.plot(T, np.degrees(y), color=col, lw=1.2, alpha=0.75)
    ax.plot(T, np.degrees(gyaw), "k--", lw=2)
    ax.axhline(np.degrees(gyaw[-1]) / 2, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("time after t0 (s)")
    ax.set_ylabel("heading change (deg, left +)")


def fig_arm(arm, c, z, op, opk, out):
    import matplotlib.pyplot as plt
    gxy, gyaw = z["gt_xy"], z["gt_yaw"]
    if arm == "OP":
        i = opk[f"{c['clip']}_p"]
        imgs = [(op["luma"][i][0], "openpilot road frame (31°)"), (op["luma"][i][1], "openpilot wide frame (58.7°)")]
        xy, yaw, txt = op["xy"][i][None], op["yaw"][i][None], "no route input; plan heading (one deterministic plan)"
    else:
        imgs = [(np.transpose(im, (1, 2, 0)), CAM_NAMES.get(int(ci), str(ci))) for im, ci in zip(z[f"{arm}_img"], z[f"{arm}_cams"])]
        xy, yaw = z[f"{arm}_xyz"][..., :2], z[f"{arm}_yaw"]
        nav = str(z[f"{arm}_nav"])
        txt = (f"nav: \"{nav}\"   " if nav else "no nav   ") + f"CoC[0]: {str(z[f'{arm}_cot'][0])[:90]}"
    A = float(((yaw * gyaw).sum(-1) / (gyaw ** 2).sum()).mean())
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.7])
    for k, (im, name) in enumerate(imgs):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    bev(fig.add_subplot(gs[1, :2]), z, xy, gxy, COL[arm])
    heading(fig.add_subplot(gs[1, 2:]), yaw, gyaw, COL[arm])
    fig.suptitle(f"{ARM_LABEL[arm]} | clip {c['clip'][:8]} ({c['side']}, logged {np.degrees(gyaw[-1]):+.0f}° in 6.4 s, "
                 f"{c['v_med']:.1f} m/s) | heading gain A_H {A:.2f}\n{txt}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_summary(df, eff, out):
    import matplotlib.pyplot as plt
    d = df[df.t0 == "p"]
    arms = [a for a in ("A", "Tblk", "C", "B", "B1t", "B1", "An", "Bn", "B1n", "OP") if a in set(d.arm)]
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.8))
    for ax, m, lab in ((axs[0], "A_H", "heading gain A_H over time (1 = logged)"), (axs[1], "lag50", "lag50 (s, + = late)"),
                       (axs[3], "A_S", "post-hoc: heading gain over distance A_S")):
        piv = d.pivot(index="case", columns="arm", values=m)[arms]
        for _, row in piv.iterrows():
            ax.plot(range(len(arms)), row.to_numpy(), color="0.8", lw=0.8, zorder=1)
        for j, a in enumerate(arms):
            ax.scatter(np.full(len(piv), j), piv[a], color=COL[a], s=18, zorder=2)
            r = eff[(eff.t0 == "p") & (eff.contrast == a) & (eff.metric == m)].iloc[0]
            ax.errorbar(j + 0.18, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", color="k", ms=5)
        ax.set_xticks(range(len(arms)), arms)
        ax.set_ylabel(lab)
        ax.axhline(1.0 if m == "A_H" else 0.0, color="k", lw=0.6, ls=":")
    cons = [c for c in ("An-Bn", "A-B", "A-C", "A-Tblk", "Tblk-C", "B-B1", "B-B1t", "B1t-B1", "An-A", "A-OP", "An-OP", "B-OP") if c in set(eff.contrast)]
    e = eff[(eff.t0 == "p") & (eff.metric == "A_H")].set_index("contrast").loc[cons]
    y = np.arange(len(cons))[::-1]
    axs[2].errorbar(e["mean"], y, xerr=[e["mean"] - e["lo"], e["hi"] - e["mean"]], fmt="o", color="k")
    axs[2].axvline(0, color="k", lw=0.6)
    for v in (-0.10, 0.10):
        axs[2].axvline(v, color="0.6", lw=0.6, ls="--")
    axs[2].set_yticks(y, cons)
    axs[2].set_xlabel("paired ΔA_H, 95 % CI over cases (primary t0)")
    fig.suptitle(f"Alpamayo 1.5 on {d.case.nunique()} real sharp turns (PhysicalAI-AV test): cross cameras on / off, nav on / off, vs openpilot",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_cases(df, alp, cases, out):
    """All pilot turns, primary t0: logged path and the mean of each arm's 6 sampled paths (A, B, B1, An, Bn)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.4))
    for ax, c in zip(axs.ravel(), cases):
        z = np.load(alp / f"{c['clip']}_p.npz", allow_pickle=True)
        for arm in ("A", "C", "B", "B1", "An", "Bn"):
            if f"{arm}_xyz" in z.files:
                for p in z[f"{arm}_xyz"][..., :2]:
                    ax.plot(-p[:, 1], p[:, 0], color=COL[arm], lw=0.8, alpha=0.45)
                ax.plot([], [], color=COL[arm], label=arm)
        g = z["gt_xy"]
        ax.plot(-g[:, 1], g[:, 0], "k--", lw=2, label="logged")
        a = df[(df.case == c["case"]) & (df.t0 == "p")].set_index("arm").A_H
        ax.set_title(f"{c['clip'][:8]} {c['side']} {np.degrees(z['gt_yaw'][-1]):+.0f}° | A_H A {a.get('A', np.nan):.2f} "
                     f"B {a.get('B', np.nan):.2f} B1 {a.get('B1', np.nan):.2f}", fontsize=8)
        ax.set_aspect("equal", "datalim")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Pilot turns (primary t0): logged 6.4 s path (dashed) and all 6 samples per arm; x = lateral (right +), y = forward (m)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main(src):
    import matplotlib
    matplotlib.use("Agg")
    src = Path(src)
    cases = json.loads((TOPIC / "results" / "cases.json").read_text())
    df = pd.read_csv(TOPIC / "results" / "per_case.csv")
    eff = pd.read_csv(TOPIC / "results" / "effects.csv")
    op = np.load(src / "op" / "preds.npz") if (src / "op" / "preds.npz").exists() else None
    opk = {k: i for i, k in enumerate(op["keys"])} if op is not None else {}
    (TOPIC / "figs").mkdir(exist_ok=True)
    c = showcase(cases, src / "alp")
    z = np.load(src / "alp" / f"{c['clip']}_p.npz", allow_pickle=True)
    for arm in ("A", "Tblk", "C", "B", "B1t", "B1", "An", "Bn", "B1n", "OP"):
        if arm == "OP" and f"{c['clip']}_p" not in opk or arm != "OP" and f"{arm}_xyz" not in z.files:
            continue
        fig_arm(arm, c, z, op, opk, TOPIC / "figs" / f"turn_{arm}.png")
    fig_summary(df, eff, TOPIC / "figs" / "summary.png")
    fig_cases(df, src / "alp", cases, TOPIC / "figs" / "cases_bev.png")
    print("showcase", c["case"])


if __name__ == "__main__":
    main(sys.argv[1])
