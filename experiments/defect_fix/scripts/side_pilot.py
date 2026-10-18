#!/usr/bin/env python3
"""Supplement 11 (plans/2026-10-17-nr1-prereg.md): readout of the side-buffer mechanism check on held 460 @18.

  $AP side_pilot.py --ctl JTNS1K-L18-s0 --arm JTNN1K-L18-s0 --out results/jtb/side_pilot_data.json   (box; AlpaSim venv: .asl reader + jevdrive.stats)

Per arm: zeros, offroad zeros, lateral zeros (offroad / corridor / AF-side), at-fault front / side collisions, AF-side zeros, mean score, progress_rel.
Paired (arm minus ctl, clustered by log session): the same as indicator means x n, so the CI of a count difference is the CI of the mean x n.
Offroad zeros: the scenes that flip in either direction, and for every offroad-zero rollout the distance to the nearest other actor on the side opposite
to the departure (ego frame, |lon| <= 10 m, centre to centre) at the last pose and the minimum over the last 2 s. Departure side = sign of the ego's
offset from the logged path at the last pose (+ left); rollouts end at the failure, so the last poses are the event.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OK_SCORE = 0.999


def failure_type(r):
    """= nx_regress.failure_type for zero scores (the only ones used here)."""
    m, sc = r.get("metrics") or {}, float(r.get("score") or 0.0)
    pos = lambda k: (m.get(k) or 0) > 0  # noqa: E731
    if sc != 0:
        return "nonzero"
    z = []
    if pos("collision_at_fault"):
        z.append("AF-front" if pos("collision_front") else "AF-side" if pos("collision_lateral") else "AF-rear")
    z += [n for k, n in (("offroad", "offroad"), ("left_corridor_laterally", "corridor")) if pos(k)]
    return "+".join(z) if z else "route-fail" if str(r.get("failure_reason") or "").startswith("Waypoint") else "zero-other"


def load(tag):
    R = {}
    for d in sorted(glob.glob(str(DATA / f"runs/route_mem/read/{tag}/c*/run"))):
        f = Path(d) / "aggregate/results-summary.json"
        for r in json.loads(f.read_text())["rollouts"]:
            R.setdefault(r["clipgt_id"], dict(r, run=d))
    return R


def pose(t, p):
    q = p.quat
    return [t, p.vec.x, p.vec.y, float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))]


def geometry(r):
    """Last-pose geometry of one rollout from its rollout.asl (zero scores keep it)."""
    from alpasim_utils.logs import async_read_pb_log
    fs = glob.glob(f"{r['run']}/rollouts/{r['clipgt_id']}/*/rollout.asl")
    if not fs:
        return dict(error="no rollout.asl")

    async def go():
        ego, oth, log = [], {}, None
        async for e in async_read_pb_log(fs[0]):
            k = e.WhichOneof("log_entry")
            if k == "actor_poses":
                t = e.actor_poses.timestamp_us
                for x in e.actor_poses.actor_poses:
                    (ego.append(pose(t, x.actor_pose)) if x.actor_id == "EGO" else oth.setdefault(x.actor_id, []).append(pose(t, x.actor_pose)))
            elif k == "traffic_session_request" and log is None:
                o = e.traffic_session_request.logged_object_trajectories[0]
                log = np.array([pose(q.timestamp_us, q.pose) for q in o.trajectory.poses], np.float64)
        return np.array(ego, np.float64), {k: np.array(v, np.float64) for k, v in oth.items()}, log
    ego, oth, log = asyncio.run(go())
    if not len(ego) or log is None:
        return dict(error="no ego / log")
    t = ego[:, 0]
    # departure side: sign of the offset of the last ego pose from the logged polyline (+ left of its direction)
    a, d = log[:-1, 1:3], np.diff(log[:, 1:3], axis=0)
    keep = (d ** 2).sum(1) > 1e-8
    a, d = a[keep], d[keep]
    p = ego[-1, 1:3]
    rr = p - a
    u = np.clip((rr * d).sum(1) / (d ** 2).sum(1), 0, 1)
    e = rr - u[:, None] * d
    k = (e ** 2).sum(1).argmin()
    dep = float(np.sign(d[k, 0] * e[k, 1] - d[k, 1] * e[k, 0]) * np.linalg.norm(e[k]))
    out = dict(dur_s=round((t[-1] - t[0]) * 1e-6, 1), depart_lat_m=round(dep, 2), per_step=[])
    best = []
    for i in np.flatnonzero(t >= t[-1] - 2e6):
        c, s = np.cos(ego[i, 3]), np.sin(ego[i, 3])
        row = []
        for aid, A in oth.items():
            j = np.searchsorted(A[:, 0], t[i])
            if j >= len(A) or abs(A[j, 0] - t[i]) > 2e5:
                continue
            dx, dy = A[j, 1] - ego[i, 1], A[j, 2] - ego[i, 2]
            lon, lat = c * dx + s * dy, -s * dx + c * dy               # ego frame: x forward, y left
            if abs(lon) <= 10:
                row.append((float(np.hypot(lon, lat)), float(lon), float(lat), aid))
        best.append(row)
    sgn = 1.0 if dep >= 0 else -1.0                                     # opposite side = lat with the other sign
    for name, rows in (("last", best[-1:]), ("last2s", best)):
        cand = [(h, lon, lat, aid) for row in rows for (h, lon, lat, aid) in row if lat * sgn < 0]
        any_ = [(h, lon, lat, aid) for row in rows for (h, lon, lat, aid) in row]
        out[f"opp_{name}_m"] = None if not cand else round(min(cand)[0], 1)
        out[f"opp_{name}_lat_m"] = None if not cand else round(abs(min(cand)[2]), 1)
        out[f"any_{name}_m"] = None if not any_ else round(min(any_)[0], 1)
    out.pop("per_step")
    return out


def summarize(R):
    ids = sorted(R)
    sc = np.array([float(R[i].get("score") or 0) for i in ids])
    ft = [failure_type(R[i]) for i in ids]
    m = [R[i].get("metrics") or {} for i in ids]
    pos = lambda i, k: (m[i].get(k) or 0) > 0  # noqa: E731
    z = sc == 0
    f = lambda k: np.array([k(i) for i in range(len(ids))], float)  # noqa: E731
    ind = dict(zeros=z.astype(float),
               offroad_zeros=f(lambda i: z[i] and "offroad" in ft[i]),
               lateral_zeros=f(lambda i: z[i] and any(k in ft[i] for k in ("offroad", "corridor", "AF-side"))),
               af_front=f(lambda i: pos(i, "collision_at_fault") and pos(i, "collision_front")),
               af_side=f(lambda i: pos(i, "collision_at_fault") and pos(i, "collision_lateral") and not pos(i, "collision_front")),
               af_side_zeros=f(lambda i: z[i] and "AF-side" in ft[i]),
               score=sc, progress_rel=np.array([float(x.get("progress_rel", np.nan)) for x in m]))
    return ids, ind, ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctl", required=True), ap.add_argument("--arm", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    RC, RA = load(a.ctl), load(a.arm)
    ids = sorted(set(RC) & set(RA))
    RC, RA = {i: RC[i] for i in ids}, {i: RA[i] for i in ids}
    _, IC, FC = summarize(RC)
    _, IA, FA = summarize(RA)
    grp = np.array(["_".join(i.split("_")[:2]) for i in ids])
    res = dict(n=len(ids), ctl=a.ctl, arm=a.arm, per_arm={}, paired={}, flips=[], offroad_events={})
    for nm, I in ((a.ctl, IC), (a.arm, IA)):
        res["per_arm"][nm] = {k: (float(np.nansum(v)) if k not in ("score", "progress_rel") else float(np.nanmean(v))) for k, v in I.items()}
    for k in IC:
        p = stats.paired(IA[k], IC[k], groups=grp)
        s = 1.0 if k in ("score", "progress_rel") else len(ids)
        res["paired"][k] = dict(diff=p["mean"] * s, lo=p["lo"] * s, hi=p["hi"] * s)
    oc, oa = IC["offroad_zeros"] > 0, IA["offroad_zeros"] > 0
    for j, i in enumerate(ids):
        if oc[j] != oa[j]:
            res["flips"].append(dict(scene=i, direction="removed (offroad zero in ctl only)" if oc[j] else "added (offroad zero in no-side only)",
                                     ctl=f"{float(RC[i].get('score') or 0):.2f} {FC[j]}", arm=f"{float(RA[i].get('score') or 0):.2f} {FA[j]}"))
    for nm, R, I in ((a.ctl, RC, IC), (a.arm, RA, IA)):
        res["offroad_events"][nm] = {i: geometry(R[i]) for j, i in enumerate(ids) if I["offroad_zeros"][j] > 0}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "per_arm", "paired")}, indent=1))


if __name__ == "__main__":
    main()
