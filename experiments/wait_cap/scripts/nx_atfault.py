#!/usr/bin/env python3
"""RETAIN prereg 补记 1 addition (exploratory): the at-fault collisions of RTB1O-L18-s0 on held 460 @18 against RSG-L18-s0 (9 vs 4).
Box, AlpaSim venv (polars, the rollout.asl reader), CPU. For every scene with an at-fault collision in either tag: the collision time
(metrics collision_any), the ego speed (driver_ego_trajectory) and the struck actor (the nearest actor box to the ego at contact, its speed and
its position in the ego frame over the last 6 s), per tag where the rollout.asl survived (a scored rollout's is pruned: then the drive log's
measured speed and the evaluator's obstacle gap only). Class per contact, rule fixed in code before any output:
  launch       ego speed < 0.5 m/s at some point in the 4 s before contact, and the struck actor was within 15 m ahead then;
  late stop    not launch; the struck actor is ahead (x > 0, |y| < 2.5 m) and moving < 1.0 m/s for the last 2 s, ego decelerating (speed 2 s
               before contact > speed at contact + 0.5);
  no braking   not launch; the struck actor ahead and slow (< 1.0 m/s) but the ego does not decelerate;
  other        anything else (the actor moves into the ego's path, lateral, cut-in, struck while turning).
  nx_atfault.py --j RTB1O-L18-s0 --c RSG-L18-s0 --out DIR   -> DIR/atfault.{json,md}, DIR/atfault_traces.png
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "experiments/alpasim/scripts"), str(REPO)]
DATA = Path("/root/autodl-tmp/ujs")
READ = DATA / "runs/route_mem/read"


def summary(tag):
    """scene -> (run dir, metrics dict, score) over the six held chunks."""
    o = {}
    for d in sorted(glob.glob(f"{READ}/{tag}/c*/run")):
        for r in json.loads(Path(d, "aggregate/results-summary.json").read_text())["rollouts"]:
            o.setdefault(r["clipgt_id"], (d, r.get("metrics") or {}, float(r.get("score") or 0.0)))
    return o


def series(run, scene):
    import polars as pl
    m = pl.read_parquet(glob.glob(f"{run}/rollouts/{scene}/*/metrics.parquet")[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    return {n[0]: np.array([g["timestamps_us"].to_list(), g["values"].to_list()], np.float64).T for n, g in m.group_by("name")}


def drive_rec(run, scene):
    rec = []
    for line in open(f"{run}/driver-logs/drive.jsonl"):
        if f'"{scene}"' in line:
            r = json.loads(line)
            if r["kind"] == "drive" and r["scene"] == scene:
                rec.append(r)
    return sorted(rec, key=lambda r: r["k"])


def arc4(P):
    P = np.asarray(P, float)[:, :2]
    return float(np.hypot(*P[0]) + np.hypot(*np.diff(P, axis=0).T).sum())


def one(tag, run, scene):
    M = series(run, scene)
    ca = M["collision_any"]
    tc = float(ca[ca[:, 1] > 0][0, 0]) if (ca[:, 1] > 0).any() else None
    rec = drive_rec(run, scene)
    t00 = rec[0]["t0"] if rec else ca[0, 0]
    out = dict(tag=tag, t_contact=None if tc is None else (tc - t00) / 1e6,
               dec=[dict(t=(r["t0"] - t00) / 1e6, v=10 * float(r["ego"][4]), arc4=arc4(r["poses"])) for r in rec],
               gap=[[(t - t00) / 1e6, v] for t, v in M.get("min_distance_to_obstacle_m", np.zeros((0, 2)))],
               kinds={k: bool((M[k][:, 1] > 0).any()) for k in ("collision_front", "collision_lateral", "collision_rear") if k in M})
    asl = glob.glob(f"{run}/rollouts/{scene}/*/rollout.asl")
    if not asl or tc is None:
        return out
    import c1_extract as X
    _, o = X.one_log((run, scene))
    E = np.array([[q[0], q[1], q[2], q[3]] for e in o["ego_obs"] for q in e["traj"][-1:]])        # (t, x, y, yaw), last pose per update
    E = E[np.argsort(E[:, 0])]
    vt = np.array([[e["traj"][-1][0], np.hypot(*e["v"])] for e in o["ego_obs"] if e["v"] is not None and len(e["traj"])])
    vt = vt[np.argsort(vt[:, 0])]
    out["ego_v"] = [[(t - t00) / 1e6, v] for t, v in vt]
    i = int(np.argmin(np.abs(E[:, 0] - tc)))
    ex, ey = E[i, 1:3]
    best = None
    for aid, A in o["actors"].items():
        if aid == "EGO":                                               # the ego's own box (rear axle -> centre 1.5 m)
            continue
        j = int(np.argmin(np.abs(A[:, 0] - tc)))
        if abs(A[j, 0] - tc) > 2e5:
            continue
        d = float(np.hypot(A[j, 1] - ex, A[j, 2] - ey))
        if best is None or d < best[0]:
            best = (d, aid)
    if best is None:
        return out
    A = o["actors"][best[1]]
    rel = []
    for t, x, y, *_ in A:
        if tc - 6e6 <= t <= tc + 2e5:
            k = int(np.argmin(np.abs(E[:, 0] - t)))
            dx, dy = x - E[k, 1], y - E[k, 2]
            ck, sk = np.cos(E[k, 3]), np.sin(E[k, 3])
            rel.append([(t - t00) / 1e6, ck * dx + sk * dy, -sk * dx + ck * dy])
    rel = np.array(rel)
    av = np.hypot(*np.gradient(A[:, 1:3], A[:, 0] / 1e6, axis=0).T)
    out["struck"] = dict(id=str(best[1]), label=str(o.get("size", {}).get(best[1], ("", "", ""))[2]), dist_at_contact=best[0],
                         rel=rel.tolist(), v=[[(t - t00) / 1e6, float(v)] for t, v in zip(A[:, 0], av) if tc - 6e6 <= t <= tc + 2e5])
    return out


def classify(r):
    if "ego_v" not in r or "struck" not in r:
        return "unread (asl pruned)"
    tc = r["t_contact"]
    ev = np.array(r["ego_v"])
    rel, av = np.array(r["struck"]["rel"]), np.array(r["struck"]["v"])
    v_at = lambda T: float(np.interp(T, ev[:, 0], ev[:, 1]))          # noqa: E731
    w = ev[(ev[:, 0] >= tc - 4) & (ev[:, 0] <= tc)]
    if len(w) and w[:, 1].min() < 0.5:
        t_min = w[np.argmin(w[:, 1]), 0]
        k = int(np.argmin(np.abs(rel[:, 0] - t_min)))
        if 0 < rel[k, 1] < 15 and abs(rel[k, 2]) < 2.5:
            return "launch"
    last = rel[rel[:, 0] >= tc - 2]
    ahead = len(last) and (last[:, 1] > 0).all() and (np.abs(last[:, 2]) < 2.5).all()
    slow = len(av) and av[av[:, 0] >= tc - 2][:, 1].max(initial=0) < 1.0
    if ahead and slow:
        return "late stop" if v_at(tc - 2) > v_at(tc) + 0.5 else "no braking"
    return "other"


def main(a):
    J, C = summary(a.j), summary(a.c)
    af = lambda S, s: (S[s][1].get("collision_at_fault") or 0) > 0     # noqa: E731
    scenes = sorted(s for s in set(J) & set(C) if af(J, s) or af(C, s))
    res = []
    for s in scenes:
        rj, rc = one(a.j, J[s][0], s), one(a.c, C[s][0], s)
        for r in (rj, rc):
            r["class"] = classify(r) if r["t_contact"] is not None else "no contact"
        res.append(dict(scene=s, log=s.split("_")[0] + "_" + s.split("_")[1], at_fault={a.j: af(J, s), a.c: af(C, s)},
                        score={a.j: J[s][2], a.c: C[s][2]}, dist={a.j: J[s][1].get("dist_traveled_m"), a.c: C[s][1].get("dist_traveled_m")},
                        human_dist=J[s][1].get("gt_dist_traveled_m"), J=rj, C=rc))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "atfault.json").write_text(json.dumps(res, indent=1, default=float))
    L = [f"# At-fault collisions on held 460 @18: {a.j} vs {a.c}", "",
         "| scene | log | at-fault J / C | contact J (s) | class J | struck (label, x / y m at contact) | J speed 2 s before / at contact | "
         "C contact (s) | C dist (m) | J dist (m) | human dist (m) |", "|---|---|:-:|--:|---|---|--:|--:|--:|--:|--:|"]
    for r in res:
        j = r["J"]
        st = j.get("struck")
        rl = np.array(st["rel"]) if st else None
        pos = f"{st['label']}, {rl[-1, 1]:.1f} / {rl[-1, 2]:.1f}" if st and len(rl) else "-"
        if "ego_v" in j and j["t_contact"] is not None:
            ev = np.array(j["ego_v"])
            sp = f"{np.interp(j['t_contact'] - 2, ev[:, 0], ev[:, 1]):.1f} / {np.interp(j['t_contact'], ev[:, 0], ev[:, 1]):.1f}"
        else:
            sp = "-"
        f = lambda x: "-" if x is None else f"{x:.1f}"                  # noqa: E731
        L.append(f"| {r['scene'][-16:]} | {r['log']} | {'Y' if r['at_fault'][a.j] else '-'} / {'Y' if r['at_fault'][a.c] else '-'} | {f(j['t_contact'])} | "
                 f"{j['class']} | {pos} | {sp} | {f(r['C']['t_contact'])} | {f(r['dist'][a.c])} | {f(r['dist'][a.j])} | {f(r['human_dist'])} |")
    (out / "atfault.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figs(res, a, out)


def figs(res, a, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [r for r in res if r["at_fault"][a.j] and not r["at_fault"][a.c]]
    if not R:
        return
    fig, ax = plt.subplots(len(R), 2, figsize=(11, 2.4 * len(R)), squeeze=False)
    for i, r in enumerate(R):
        j, c = r["J"], r["C"]
        for x, lab, col in ((j, a.j, "C0"), (c, a.c, "C7")):
            d = np.array([[q["t"], q["v"]] for q in x["dec"]])
            if len(d):
                ax[i, 0].plot(d[:, 0], d[:, 1], "o-", ms=3, color=col, label=f"{lab} speed (decisions)")
            g = np.array(x["gap"])
            if len(g):
                ax[i, 1].plot(g[:, 0], g[:, 1], color=col, label=f"{lab} min gap")
            if x["t_contact"] is not None:
                for k in (0, 1):
                    ax[i, k].axvline(x["t_contact"], color=col, ls="--", lw=0.8)
        if "struck" in j:
            sv = np.array(j["struck"]["v"])
            ax[i, 0].plot(sv[:, 0], sv[:, 1], color="C3", lw=1, label="struck actor speed")
            rl = np.array(j["struck"]["rel"])
            ax[i, 1].plot(rl[:, 0], rl[:, 1], color="C3", lw=1, label="struck actor x in ego frame")
        ax[i, 0].set_ylabel("m/s")
        ax[i, 1].set_ylabel("m")
        ax[i, 0].set_title(f"{r['scene'][:40]}  [{j['class']}]", fontsize=8)
        ax[i, 1].set_ylim(-1, 25)
    ax[0, 0].legend(fontsize=6)
    ax[0, 1].legend(fontsize=6)
    for k in (0, 1):
        ax[-1, k].set_xlabel("s since the first decision")
    fig.tight_layout()
    fig.savefig(out / "atfault_traces.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--j", required=True), ap.add_argument("--c", required=True), ap.add_argument("--out", required=True)
    main(ap.parse_args())
