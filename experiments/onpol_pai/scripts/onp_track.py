"""How closely the simulated ego of a PAI run followed the scene's logged path, from rollout.asl alone (the check of pass 2, and of
pass 1 as the reference of an ego that is free to leave the log). Run through onp_env.sh (the .asl reader is AlpaSim's).

  onp_env.sh scripts/onp_track.py <run dir of pai_run.sh>... [--out <json>] [--jobs N]

Simulated ego = the EGO entries of `actor_poses`; log = the first logged object trajectory of `traffic_session_request` (the recorded
ego, same reference point: both are the ego box, checked by the force-GT start where the runtime holds the ego on the log). Per scene,
over the closed-loop part (after the run's simulation_config.force_gt_duration_us) and over the whole rollout:
  same  distance to the log's pose of the same time            lon   its component along the log's heading (+ = ego ahead)
  lat   signed offset from the logged path, any time (+ = left of it): nearest point of the polyline
  yaw   heading difference to the log at the same time, deg
plus the score, dist_to_gt_trajectory and decisions of the scene (results-summary.json, driver/drive.jsonl) and, for a log-replay
run, the driver's own measurement (`log_now` of drive.jsonl) and its error counters.
"""
import argparse
import asyncio
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def p4(t, p) -> list:
    q = p.quat
    return [t, p.vec.x, p.vec.y, float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))]


def lateral(xy, path) -> np.ndarray:
    """Signed offset of points (n, 2) from the polyline path (m, 2): + = left of its direction."""
    a, d = path[:-1], np.diff(path, axis=0)
    keep = (d ** 2).sum(1) > 1e-8
    a, d = a[keep], d[keep]
    if not len(a):                                           # a parked log: the distance to its point
        return np.linalg.norm(xy - path[0], axis=1)
    r = xy[:, None] - a[None]
    u = np.clip((r * d).sum(2) / (d ** 2).sum(1), 0, 1)
    e = r - u[..., None] * d
    k = (e ** 2).sum(2).argmin(1)
    i = np.arange(len(xy))
    return np.sign(d[k, 0] * e[i, k, 1] - d[k, 1] * e[i, k, 0]) * np.linalg.norm(e[i, k], axis=1)


def stats(x) -> dict:
    x = np.abs(np.asarray(x, np.float64))
    return {"med": round(float(np.median(x)), 4), "p95": round(float(np.percentile(x, 95)), 4), "max": round(float(x.max()), 4)} if len(x) else {}


def one(a) -> dict:
    from alpasim_utils.logs import async_read_pb_log
    f, force = a

    async def go():
        ego, log, ids = [], None, set()
        async for e in async_read_pb_log(f):
            k = e.WhichOneof("log_entry")
            if k == "actor_poses":
                for x in e.actor_poses.actor_poses:
                    ids.add(x.actor_id)
                    if x.actor_id == "EGO":
                        ego.append(p4(e.actor_poses.timestamp_us, x.actor_pose))
            elif k == "traffic_session_request" and log is None:
                o = e.traffic_session_request.logged_object_trajectories[0]
                log = (o.object_id, np.array([p4(q.timestamp_us, q.pose) for q in o.trajectory.poses], np.float64))
        return np.array(ego, np.float64), log, ids
    ego, log, ids = asyncio.run(go())
    o = {"scene": f.split("/rollouts/")[1].split("/")[0], "n_ego": len(ego), "force_gt_s": force * 1e-6}
    if not len(ego) or log is None:
        return {**o, "error": f"ego poses {len(ego)}, logged trajectory {log is not None}, actor ids {sorted(ids)[:5]}"}
    lid, L = log
    t = ego[:, 0]
    ref = np.stack([np.interp(t, L[:, 0], L[:, 1]), np.interp(t, L[:, 0], L[:, 2]), np.interp(t, L[:, 0], np.unwrap(L[:, 3]))], 1)
    d = ego[:, 1:3] - ref[:, :2]
    same, lon = np.linalg.norm(d, axis=1), d[:, 0] * np.cos(ref[:, 2]) + d[:, 1] * np.sin(ref[:, 2])
    lat, yaw = lateral(ego[:, 1:3], L[:, 1:3]), np.degrees((ego[:, 3] - ref[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    cl, inside = t > t[0] + force + 1, (t >= L[0, 0]) & (t <= L[-1, 0])
    o.update(logged_id=lid, dur_s=round((t[-1] - t[0]) * 1e-6, 2), log_s=[round((L[0, 0] - t[0]) * 1e-6, 2), round((L[-1, 0] - t[0]) * 1e-6, 2)],
             log_len_m=round(float(np.linalg.norm(np.diff(L[:, 1:3], axis=0), axis=1).sum()), 1), outside_log=int((~inside).sum()))
    for name, m in (("closed", cl & inside), ("all", inside), ("force_gt", ~cl & inside), ("last2s", inside & (t > t[-1] - 2e6))):
        o[name] = {"n": int(m.sum()), "same": stats(same[m]), "lat": stats(lat[m]), "lon": stats(lon[m]), "yaw": stats(yaw[m]),
                   "lon_mean": round(float(lon[m].mean()), 4) if m.any() else None, "lat_mean": round(float(lat[m].mean()), 4) if m.any() else None}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        run = run.rstrip("/")
        S = {r["clipgt_id"]: r for r in json.load(open(f"{run}/sim/aggregate/results-summary.json"))["rollouts"]}
        force = int(re.search(r"force_gt_duration_us: (\d+)", open(f"{run}/sim/generated-user-config-0.yaml").read()).group(1))
        drv = {}
        for line in open(f"{run}/driver/drive.jsonl"):
            r = json.loads(line)
            v = drv.setdefault(r["scene"], {"drive": 0, "log": 0, "ext": 0, "log_now": [], "close": None, "t": []})
            if r["kind"] == "close":
                v["close"] = {k: r.get(k) for k in ("drive", "inference", "inference_error", "input_error", "log_error", "cold", "images")}
            else:
                v["drive"] += 1
                v["t"].append(r["t"])
                if r.get("ret") == "log":
                    v["log"] += 1
                    v["ext"] += int(r["ret_ext"] > 0)
                    v["log_now"].append(r["log_now"])
        dumps = {}
        for p in glob.glob(f"{run}/driver/dump/*.npz"):
            dumps[os.path.basename(p)[:3]] = dumps.get(os.path.basename(p)[:3], 0) + 1
        with ProcessPoolExecutor(a.jobs) as ex:
            for o in ex.map(one, [(f, force) for f in sorted(glob.glob(f"{run}/sim/rollouts/*/*/rollout.asl"))]):
                s, v = S.get(o["scene"], {}), drv.get(o["scene"])
                o.update(run=run, score=s.get("score"), passed=s.get("passed"), failure=s.get("failure_reason"),
                         dist_to_gt_trajectory=(s.get("metrics") or {}).get("dist_to_gt_trajectory"))
                if v:
                    g = np.array(v["log_now"], np.float64).reshape(-1, 3)
                    o.update(decisions=v["drive"], returned_log=v["log"], with_extrapolation=v["ext"], close=v["close"],
                             wall_s=round(max(v["t"]) - min(v["t"]), 1),
                             driver_log_now={"lon": stats(g[:, 0]), "lat": stats(g[:, 1]), "yaw_deg": stats(np.degrees(g[:, 2]))} if len(g) else None)
                rows.append(o)
                c = o.get("closed", {})
                print(f"{o['scene'][-12:]} score {o['score']} decisions {o.get('decisions')} log-returned {o.get('returned_log')} "
                      f"err {o.get('close')} | closed loop n {c.get('n')}: same-time {c.get('same')} lateral {c.get('lat')} "
                      f"lon {c.get('lon')} yaw {c.get('yaw')} | force-GT part same {o.get('force_gt', {}).get('same')} "
                      f"| last 2 s same {o.get('last2s', {}).get('same')} lat {o.get('last2s', {}).get('lat')} | {o.get('error', '')}", flush=True)
        print(f"{run}: dumps per session {sorted(dumps.values())}")
    ok = [o for o in rows if "closed" in o and o["closed"]["n"]]
    if ok:
        f = lambda k, q: [o["closed"][k][q] for o in ok]  # noqa: E731
        print(f"{len(ok)} scenes, closed loop: same-time median of medians {np.median(f('same', 'med')):.3f} m, worst max {max(f('same', 'max')):.3f} m; "
              f"lateral median of medians {np.median(f('lat', 'med')):.3f} m, worst max {max(f('lat', 'max')):.3f} m")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
