#!/usr/bin/env python3
"""BODY1 arm 4.1 review strips: one PNG per case, bird's-eye panels over the rollout + speed and contact-probability traces.
Box, AlpaSim's venv ($DATA_DIR/third_party/alpasim/.venv/bin/python); reads rollout.asl through experiments/alpasim/scripts/c1_extract.py.

  stop_bev.py --cases cases.csv --out DIR
cases.csv columns: id, kind, scene, on (run dir with the switch; needs rollout.asl of the scene), off (baseline run dir; its rollout.asl when it
is there, else the controller trace), look (one sentence for the figure's doc).
Panels (logged heading at t = 0 up, centred on the switch-on ego): other vehicles at that time (grey; they replay the log), the logged ego
(dashed outline), the baseline run's ego (orange outline), the switch-on ego (blue), the trajectory returned at the newest decision (green);
at a flagged decision also the plan before re-timing (green dotted) and the stop point (red cross). Bottom: ego speed (switch on, baseline
run, log), flagged decisions (red lines), p(contact) per decision (bars). Objects and the logged path are privileged: for the reader only.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "alpasim" / "scripts"))
import c1_extract as X  # noqa: E402
import c1_lib as L  # noqa: E402

BLUE, RED, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#8a8a8a"


def recs(run, scene):
    out = []
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        if '"kind": "drive"' in line and scene in line:
            out.append(json.loads(line))
    return sorted(out, key=lambda r: r["k"])


def off_track(run, scene):
    """Baseline ego as AABB-centre poses (n, 4) t, x, y, yaw: from its rollout.asl when kept, else from the controller trace."""
    if glob.glob(f"{run}/rollouts/{scene}/*/rollout.asl"):
        return L.ego(X.one_log((run, scene))[1])
    S = json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]
    sid = next(r["rollout_id"] for r in S if r["clipgt_id"] == scene)
    c = np.genfromtxt(Path(run) / "controller" / f"alpasim_controller_{sid}.csv", delimiter=",", names=True)
    yaw = 2 * np.arctan2(c["qz"], c["qw"])
    return np.c_[c["timestamp_us"], L.to_center(np.c_[c["x"], c["y"], yaw])]


def to_local(anchor, p):
    """Poses (n, 3) in the ego frame of the decision -> AABB-centre poses in the rollout's local frame."""
    x0, y0, a = anchor
    xy = np.asarray(p)[:, :2] @ L.rot(a).T + [x0, y0]
    return L.to_center(np.c_[xy, np.asarray(p)[:, 2] + a])


def render(c, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    scene = c["scene"]
    o = X.one_log((c["on"], scene))[1]
    o["summary"] = next(r for r in json.loads((Path(c["on"]) / "aggregate/results-summary.json").read_text())["rollouts"] if r["clipgt_id"] == scene)
    R_, e, g, b = recs(c["on"], scene), L.ego(o), L.gt(o), off_track(c["off"], scene)
    Sb = next(r for r in json.loads((Path(c["off"]) / "aggregate/results-summary.json").read_text())["rollouts"] if r["clipgt_id"] == scene)
    t_ev = L.first_event(o, "collision_at_fault") if o["summary"]["score_metrics"].get("collision_at_fault") else None
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
        bd = R_[k].get("body") if k < len(R_) else None
        if bd and bd["flag"]:
            pl = to_local(R_[k]["anchor"], np.r_[[[0, 0, 0]], bd["plan"]])
            ax.plot(*tf(pl[:, :2]).T, ":", color=GREEN, lw=1.2, zorder=6)
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(pl[:, :2], axis=0).T))]
            sp = [np.interp(min(bd["D"], s[-1]), s, pl[:, i]) for i in (0, 1)]
            ax.plot(*tf(sp).T, "x", color=RED, ms=9, mew=2.2, zorder=7)
        ax.set_xlim(-W, W), ax.set_ylim(-W * 0.7, W * 1.3), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"t = {t * 1e-6:.1f} s" + (f", decision {k}: p = {bd['p']:.2f}" + (" FLAG" if bd["flag"] else "") if bd else "") + (" (contact)" if hit else ""),
                     fontsize=8.5, color=RED if (bd and bd["flag"]) or hit else "k")
    ax = fig.add_subplot(gs[1, 0:4])
    fine = np.arange(1e5, t_end, 1e5)
    ax.plot(fine * 1e-6, [L.speed_at(e, t) for t in fine], color=BLUE, lw=1.6, label="switch on")
    ax.plot(fine * 1e-6, [L.speed_at(b, t) for t in fine], color=ORANGE, lw=1.3, label="baseline run (switch off)")
    ax.plot(fine * 1e-6, [L.speed_at(g, t) for t in fine], "--", color=GREY, lw=1.0, label="log")
    for r in R_:
        if r.get("body") and r["body"]["flag"]:
            ax.axvline(r["now"] * 1e-6, color=RED, lw=1.0, alpha=0.8)
    if t_ev:
        ax.axvline(t_ev * 1e-6, color="k", lw=1.2, ls="--")
    ax.set_xlabel("simulated time (s)"), ax.set_ylabel("ego speed (m/s)"), ax.legend(fontsize=7.5, loc="best"), ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[1, 4:6])
    p = [r["body"]["p"] if r.get("body") else np.nan for r in R_]
    ax.bar(now * 1e-6, p, width=0.35, color=[RED if (r.get("body") and r["body"]["flag"]) else "#9db7cc" for r in R_])
    ax.axhline(0.6006, color=RED, lw=0.8, ls=":")
    ax.set_ylim(0, 1), ax.set_xlabel("decision time (s)"), ax.set_ylabel("p(contact) of the served plan"), ax.grid(alpha=0.25)
    zs = lambda r: next((L.SHORT[f] for f in L.FLAGS if r["score_metrics"].get(f)), "")  # noqa: E731
    fig.suptitle(f"{c['id']} ({c['kind']}): {scene[-16:]}; switch on score {o['summary']['score']:.2f} {zs(o['summary'])}, baseline run {Sb['score']:.2f} {zs(Sb)}", fontsize=10)
    fig.savefig(Path(out) / f"{c['id']}.png")
    plt.close(fig)
    print(c["id"], scene, "on", o["summary"]["score"], "off", Sb["score"], "flags", sum(bool(r.get("body") and r["body"]["flag"]) for r in R_), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in csv.DictReader(open(a.cases)):
        render(c, a.out)


if __name__ == "__main__":
    main()
