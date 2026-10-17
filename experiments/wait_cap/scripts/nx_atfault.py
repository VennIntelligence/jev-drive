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
  nx_atfault.py --geo --j .. --c .. --out DIR   (after the above) per contact, from the scene map (c1_extract map's loader) and the controller
                CSVs (ego pose every 0.1 s, never pruned): the struck actor's heading against the ego (head-on |dpsi| > 135 deg, crossing 45-135,
                same direction < 45) and velocity; the ego box's lanes at contact and over the 2 s before (a corner in a lane whose direction is
                > 90 deg off the ego heading = opposite lane; no corner in any lane = off the lanes); the ego's lateral offset from the human path
                and the other tag's offset at the same station along the human path (and its lane state there); the struck actor's lane
                direction against the ego heading -> DIR/atfault_geo.json, DIR/atfault.md (columns added)
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


# ---------------------------------------------------------------- geometry against the scene map
L_FRONT, L_REAR, HALF_W = 3.9, 1.1, 1.15        # ego box from the rear axle (nuPlan Pacifica: 5.18 x 2.30 m, rear overhang ~1.1 m)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def scene_map(run, scene, center, radius=150.0):
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_runtime.scene_loader import build_scene_loader
    from omegaconf import OmegaConf
    user = OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(f"{run}/generated-user-config-0.yaml"))
    vm = build_scene_loader(user).get_data_source(scene).map
    from matplotlib.path import Path as MP
    lanes = []
    for ln in vm.get_lanes_within(np.r_[center, 0.0], radius):
        c = np.asarray(ln.center.points[:, :2], float)
        if ln.left_edge is None or ln.right_edge is None or len(c) < 2:
            continue
        poly = np.r_[np.asarray(ln.left_edge.points[:, :2], float), np.asarray(ln.right_edge.points[:, :2], float)[::-1]]
        lanes.append(dict(id=str(ln.id), c=c, path=MP(poly)))
    return lanes


def lane_dir(ln, xy):
    c = ln["c"]
    i = int(np.argmin(np.hypot(*(c - xy).T)))
    j = min(i + 1, len(c) - 1)
    i = j - 1
    return float(np.arctan2(*(c[j] - c[i])[::-1]))


def box(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[x + c * dx - s * dy, y + s * dx + c * dy] for dx in (L_FRONT, -L_REAR) for dy in (HALF_W, -HALF_W)])


def lane_state(lanes, x, y, yaw):
    """'own' (every corner in a lane within 90 deg of the heading), 'opposite' (a corner only in lanes > 90 deg off), 'off lanes' (a corner in
    no lane), worst first; plus the largest heading gap of the lanes under the box."""
    worst, gap = "own", 0.0
    for q in box(x, y, yaw):
        d = [abs(wrap(lane_dir(ln, q) - yaw)) for ln in lanes if ln["path"].contains_point(q)]
        if not d:
            worst = "off lanes" if worst == "own" else worst
        else:
            gap = max(gap, min(d))
            if min(d) > np.pi / 2:
                worst = "opposite"
    return worst, float(np.degrees(gap))


def ctrl_path(run, scene):
    """Ego (t_us, x, y, yaw) every 0.1 s from the controller CSV of the scene's session."""
    import csv
    for line in open(f"{run}/driver-logs/drive.jsonl"):
        if '"kind": "start"' in line and f'"{scene}"' in line:
            ses = json.loads(line)["session"]
            break
    else:
        return None
    f = Path(run) / f"controller/alpasim_controller_{ses}.csv"
    if not f.exists():
        return None
    R = list(csv.DictReader(open(f)))
    return np.array([[float(r["timestamp_us"]), float(r["x"]), float(r["y"]), 2 * np.arctan2(float(r["qz"]), float(r["qw"]))] for r in R])


def station(G, xy):
    """Human path G (n, >=3: t, x, y) -> (station, signed lateral offset, left +) of a point."""
    P = G[:, 1:3]
    seg = np.diff(P, axis=0)
    L = np.hypot(*seg.T)
    keep = L > 1e-3
    P0, seg, L = P[:-1][keep], seg[keep], L[keep]
    s0 = np.r_[0, np.cumsum(L)][:-1]
    u = np.clip(((xy - P0) * seg).sum(1) / L ** 2, 0, 1)
    q = P0 + u[:, None] * seg
    d = np.hypot(*(xy - q).T)
    k = int(np.argmin(d))
    sign = np.sign(seg[k, 0] * (xy[1] - P0[k, 1]) - seg[k, 1] * (xy[0] - P0[k, 0]))
    return float(s0[k] + u[k] * L[k]), float(sign * d[k])


def geo(a):
    import c1_extract as X
    out = Path(a.out)
    res = json.loads((out / "atfault.json").read_text())
    J, C = summary(a.j), summary(a.c)
    G = []
    for r in res:
        s = r["scene"]
        runs = {a.j: J[s][0], a.c: C[s][0]}
        asl = {t: bool(glob.glob(f"{d}/rollouts/{s}/*/rollout.asl")) for t, d in runs.items()}
        src = next(t for t in (a.j, a.c) if asl[t])
        _, o = X.one_log((runs[src], s))
        human = np.asarray(o["logged"][0]["traj"])
        lanes = scene_map(runs[src], s, human[len(human) // 2, 1:3])
        paths = {t: ctrl_path(d, s) for t, d in runs.items()}
        for t, P in paths.items():                                      # frame check: the controller pose against the log at its first sample
            if P is not None:
                i = int(np.argmin(np.abs(human[:, 0] - P[0, 0])))
                print(f"  frame check {t}: ctrl vs log at t {P[0, 0] / 1e6:.1f} s: {np.hypot(*(P[0, 1:3] - human[i, 1:3])):.2f} m", flush=True)
        g = dict(scene=s, lanes=len(lanes), map_from=src, tags={})
        for t, other in ((a.j, a.c), (a.c, a.j)):
            x = r["J"] if t == a.j else r["C"]
            P = paths[t]
            if x["t_contact"] is None or P is None:
                continue
            rec = drive_rec(runs[t], s)
            tc = rec[0]["t0"] + 1e6 * x["t_contact"]
            k = int(np.argmin(np.abs(P[:, 0] - tc)))
            ex, ey, eyaw = P[k, 1:]
            st_c = lane_state(lanes, ex, ey, eyaw)
            win = P[(P[:, 0] >= tc - 2e6) & (P[:, 0] <= tc)]
            states = [lane_state(lanes, *q[1:]) for q in win]
            sta, off = station(human, np.array([ex, ey]))
            e = dict(t_contact=x["t_contact"], lane_at_contact=st_c[0], lane_gap_deg=st_c[1],
                     opposite_share_2s=float(np.mean([q[0] == "opposite" for q in states])) if states else None,
                     off_share_2s=float(np.mean([q[0] == "off lanes" for q in states])) if states else None,
                     max_lane_gap_2s=max([q[1] for q in states], default=None), station=sta, offset_human=off)
            Q = paths[other]
            if Q is not None:
                so = np.array([station(human, q[1:3]) for q in Q])
                near = np.argsort(np.abs(so[:, 0] - sta))[:1]
                if len(near) and abs(so[near[0], 0] - sta) < 1.0:
                    qo = Q[near[0]]
                    e["other_offset_human"] = float(so[near[0], 1])
                    e["other_lane_state"] = lane_state(lanes, *qo[1:])[0]
                    e["other_passes_station"] = True
                else:
                    e["other_passes_station"] = False
            if asl[t]:                                                                    # struck actor heading / velocity / lane
                _, ot = (None, o) if t == src else X.one_log((runs[t], s))
                st = x.get("struck") or {}
                A = ot["actors"].get(st.get("id")) if st else None
                if A is not None:
                    i = int(np.argmin(np.abs(A[:, 0] - tc)))
                    v = np.gradient(A[:, 1:3], A[:, 0] / 1e6, axis=0)[i]
                    dpsi = float(np.degrees(wrap(A[i, 3] - eyaw)))
                    sp = float(np.hypot(*v))
                    vdir = float(np.degrees(wrap(np.arctan2(v[1], v[0]) - eyaw))) if sp > 0.5 else None
                    kind = "head-on" if abs(dpsi) > 135 else ("crossing" if abs(dpsi) > 45 else "same direction")
                    ad = [abs(wrap(lane_dir(ln, A[i, 1:3]) - eyaw)) for ln in lanes if ln["path"].contains_point(A[i, 1:3])]
                    e["struck"] = dict(dpsi_deg=dpsi, kind=kind, speed=sp, vel_dir_deg=vdir,
                                       lane_vs_ego_deg=float(np.degrees(min(ad))) if ad else None,
                                       lane_faces_ego=bool(ad and min(ad) > np.pi * 3 / 4))
            g["tags"][t] = e
        G.append(g)
        print(s, json.dumps(g["tags"], default=float)[:600], flush=True)
    (out / "atfault_geo.json").write_text(json.dumps(G, indent=1, default=float))
    md = (out / "atfault.md").read_text().split("\n")
    md = [m for m in md if not m.startswith("## Geometry")]
    f = lambda v, q=".1f": "-" if v is None else format(v, q)  # noqa: E731
    L = ["", "## Geometry against the scene map (per contact)", "",
         "| scene | tag | struck: dpsi (deg) / kind / speed (m/s) | struck's lane vs ego heading (deg; faces ego) | ego lane at contact (max lane gap deg) | "
         "2 s before: opposite / off-lane share | ego offset from human (m) | other tag at same station: offset / lane |", "|---|---|---|---|---|--:|--:|---|"]
    for g in G:
        for t, e in g["tags"].items():
            st = e.get("struck")
            sd = f"{st['dpsi_deg']:.0f} / {st['kind']} / {st['speed']:.1f}" if st else "- (asl pruned)"
            sl = (f"{f(st['lane_vs_ego_deg'], '.0f')} ({'yes' if st['lane_faces_ego'] else 'no'})") if st else "-"
            ot = (f"{e['other_offset_human']:+.2f} / {e['other_lane_state']}" if e.get("other_passes_station") else "not reached")
            L.append(f"| {g['scene'][-16:]} | {t} | {sd} | {sl} | {e['lane_at_contact']} ({e['lane_gap_deg']:.0f}) | "
                     f"{f(e['opposite_share_2s'], '.2f')} / {f(e['off_share_2s'], '.2f')} | {e['offset_human']:+.2f} | {ot} |")
    (out / "atfault.md").write_text("\n".join(md).rstrip("\n") + "\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--j", required=True), ap.add_argument("--c", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--geo", action="store_true")
    a = ap.parse_args()
    geo(a) if a.geo else main(a)
