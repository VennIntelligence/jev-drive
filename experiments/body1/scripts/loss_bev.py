#!/usr/bin/env python3
"""BODY1 arm 4.3 review strips (loss arm; no serving switch, so no shift / contact panels): one PNG per case, bird's-eye panels over the rollout + lateral offset and per-decision traces.
Box, AlpaSim's venv ($DATA_DIR/third_party/alpasim/.venv/bin/python); reads rollout.asl through experiments/alpasim/scripts/c1_extract.py.

  loss_bev.py --cases cases.csv --out DIR
cases.csv columns: id, kind, scene, on (run dir with the switch; needs rollout.asl of the scene), off (baseline run dir; its rollout.asl when it
is there, else the controller trace), look (one sentence for the figure's doc).
Panels (logged heading at t = 0 up, centred on the new-checkpoint ego): other vehicles at that time (grey; they replay the log), the logged ego
(dashed outline), the baseline run's ego (orange outline), the new-checkpoint ego (blue), the trajectory returned at the newest decision (green);
at a re-planned decision also the plan before the shift (green dotted). Bottom left: lateral offset of the new-checkpoint ego and of the baseline
run's ego from the logged path (+ = left), re-planned decisions as vertical lines. Bottom right: served shift per decision (bars) and the
plan's two contact probabilities (agent, boundary). Objects and the logged path are privileged: for the reader only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2] / "alpasim" / "scripts")]
import c1_extract as X  # noqa: E402
import c1_lib as L  # noqa: E402
import stop_bev as SB  # noqa: E402

BLUE, RED, GREEN, ORANGE, GREY = SB.BLUE, SB.RED, SB.GREEN, SB.ORANGE, SB.GREY
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def lat(path, tr, ts):
    """Signed lateral offset (+ = left) of the track tr (n, 4) at times ts from the polyline path (n, 4)."""
    P, h = path[:, 1:3], np.unwrap(path[:, 3])
    out = []
    for t in ts:
        p = L.interp_pose(tr, t)[:2]
        j = int(np.argmin(((P - p) ** 2).sum(1)))
        out.append(-np.sin(h[j]) * (p[0] - P[j, 0]) + np.cos(h[j]) * (p[1] - P[j, 1]))
    return np.array(out)


def render(c, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    scene = c["scene"]
    o = X.one_log((c["on"], scene))[1]
    o["summary"] = next(r for r in json.loads((Path(c["on"]) / "aggregate/results-summary.json").read_text())["rollouts"] if r["clipgt_id"] == scene)
    R_, e, g, b = SB.recs(c["on"], scene), L.ego(o), L.gt(o), SB.off_track(c["off"], scene)
    Sb = next(r for r in json.loads((Path(c["off"]) / "aggregate/results-summary.json").read_text())["rollouts"] if r["clipgt_id"] == scene)
    fl = next((f for f in L.FLAGS if o["summary"]["score_metrics"].get(f)), None)
    t_ev = L.first_event(o, fl) if fl else None
    now = np.array([r["now"] for r in R_])
    t_end = e[-1, 0]
    ts = np.linspace(0.5e6, t_end, 6)
    Rv = L.rot(np.pi / 2 - g[0, 3])
    W = 16.0
    fig = plt.figure(figsize=(15.5, 6.3), dpi=110)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.62], left=0.035, right=0.99, top=0.9, bottom=0.085, hspace=0.22, wspace=0.08)
    for j, t in enumerate(ts):
        ax = fig.add_subplot(gs[0, j])
        ctr = L.interp_pose(e, t)[:2]
        tf = lambda xy: (np.atleast_2d(xy) - ctr) @ Rv.T  # noqa: E731
        poly = lambda p, **kw: ax.add_patch(MP(tf(np.array(L.box(*p).exterior.coords)), **kw))  # noqa: E731
        ax.plot(*tf(g[:, 1:3]).T, "--", color=GREY, lw=0.9, zorder=1)
        for a, tr in o["actors"].items():
            if a == "EGO" or not (tr[0, 0] <= t <= tr[-1, 0]):
                continue
            p = L.interp_pose(tr, t)
            if np.hypot(*(p[:2] - ctr)) > 1.6 * W:
                continue
            sx, sy = o["size"][a][:2]
            ax.add_patch(MP(tf(np.array(L.box(*p, sx, sy).exterior.coords)), fc="#cfcfcf", ec="#777777", lw=0.6, zorder=2))
        poly(L.interp_pose(g, t), fc="none", ec=GREY, lw=0.9, ls="--", zorder=3)
        poly(L.interp_pose(b, t), fc="none", ec=ORANGE, lw=1.4, zorder=4)
        hit = t_ev is not None and t >= t_ev
        poly(L.interp_pose(e, t), fc=RED if hit else BLUE, ec="k", lw=0.5, alpha=0.75, zorder=5)
        k = max(int(np.searchsorted(now, t, side="right")) - 1, 0)
        tr = o["drive"][k].get("traj") if k < len(o["drive"]) else None
        if tr is not None and len(tr):
            ax.plot(*tf(L.to_center(tr[:, 1:4])[:, :2]).T, color=GREEN, lw=1.6, zorder=6)
        ax.set_xlim(-W, W), ax.set_ylim(-W * 0.7, W * 1.3), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"t = {t * 1e-6:.1f} s, decision {k}" + (" (event)" if hit else ""), fontsize=8.5, color=RED if hit else "k")
    ax = fig.add_subplot(gs[1, :])
    fine = np.arange(1e5, t_end, 1e5)
    ax.plot(fine * 1e-6, lat(g, e, fine), color=BLUE, lw=1.6, label="new checkpoint")
    ax.plot(fine * 1e-6, lat(g, b, fine), color=ORANGE, lw=1.3, label="baseline run (P2H10-F)")
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    if t_ev:
        ax.axvline(t_ev * 1e-6, color="k", lw=1.2, ls="--")
    ax.set_xlabel("simulated time (s)"), ax.set_ylabel("lateral offset from the logged path (m, + left)"), ax.legend(fontsize=7.5, loc="best"), ax.grid(alpha=0.25)
    zs = lambda r: next((L.SHORT[f] for f in L.FLAGS if r["score_metrics"].get(f)), "")  # noqa: E731
    fig.suptitle(f"{c['id']} ({c['kind']}): {scene[-16:]}; new score {o['summary']['score']:.2f} {zs(o['summary'])}, baseline run {Sb['score']:.2f} {zs(Sb)}", fontsize=10)
    fig.savefig(Path(out) / f"{c['id']}.png")
    plt.close(fig)
    print(c["id"], scene, "on", o["summary"]["score"], "off", Sb["score"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in csv.DictReader(open(a.cases)):
        render(c, a.out)


if __name__ == "__main__":
    main()
