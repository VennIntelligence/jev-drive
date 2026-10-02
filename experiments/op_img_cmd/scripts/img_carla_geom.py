"""CARLA half of the image-command test, step 3 (envs/carla, CPU, carla.Map from the xodr, no server): route geometry
and t0 selection on the recordings of img_carla_agent.py, in the schema of navtrain's geom/nav.pkl (img_geom_nav.py).

Per recorded route (done/<id>.json -> its attempt directory): the approach path is the entry lane walked back from the
start of the connector the route takes (APPROACH_BACK_M, straightest predecessor at merges); each branch (lane D's
(class, connecting road, lane) list, the taken connector first within its class) is the connector plus the following
lanes to EXT_M past its end (straightest successor). Centrelines from carla waypoints every STEP_M, boundaries
centre +- lane_width / 2. t0: the 5 Hz camera frames nearest to 20 / 10 / 5 / 1 m (rear axle, along the approach
centreline) before the connector start, plus the last stationary frame within STOP_WIN_M of it if the ego stops there;
each needs HIST_N earlier frames without a gap and 4 s of logged future.

Frames: CARLA world is left-handed (x fwd-east, y right, yaw clockwise in degrees); everything stored is right-handed
(Y = -y, yaw = -yaw) and then in the t0 rear-axle frame (x fwd, y left, z up). The hero's transform is the actor origin
(Lincoln MKZ 2020: bounding-box centre on the ground); the rear axle is REAR_AXLE_X along the heading.

Output $DATA_DIR/runs/op_img_cmd/carla/geom.pkl: nav.pkl keys (token, log, kind, v, cmd, dist, classes, taken, map,
approach, node, branches, future) plus town, tag, files (JPEG triplets of the HIST_N + 1 frames, oldest first),
frame_t (s, relative to t0), pose ((k, 3) rear-axle x, y, yaw in the t0 frame), pitch ((k,) CARLA pitch, deg), cam.
img_carla_frames.py renders the packed model frames and writes carla.pkl.

  $DATA_DIR/envs/carla/bin/python experiments/op_img_cmd/scripts/img_carla_geom.py [--run rec] [--workers 16]
"""
import argparse, json, math, os, pickle, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "op_common_cause" / "scripts"))
import pair_inv_carla as PI  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
ROOT = DATA / "runs" / "op_img_cmd" / "carla"
REAR_AXLE_X = -1.388633220          # p4_carla_agent.REAR_AXLE_X
CAM = (1.519, 0.026, 1.806)         # model-frame origin in the rear-axle frame: the front camera (jevdrive.p5_openpilot.RIG)
STEP_M, EXT_M, APPROACH_BACK_M = 0.5, 45.0, 110.0
TARGETS = {"d20": 20.0, "d10": 10.0, "d5": 5.0, "d1": 1.0}
TOL_M, OFF_M, STOP_WIN_M, STOP_V = 1.5, 3.0, 8.0, 0.1
HIST_N, FUT_N, TICK = 8, 8, 0.05     # 8 earlier 5 Hz frames (1.6 s), 8 future points at 0.5 s


def rh(x, y, yaw_deg):
    return np.array([x, -y]), -math.radians(yaw_deg)


def wp_rh(w):
    t = w.transform
    p, yaw = rh(t.location.x, t.location.y, t.rotation.yaw)
    return p, yaw, w.lane_width


def straightest(cands, yaw_deg):
    return min(cands, key=lambda w: abs((w.transform.rotation.yaw - yaw_deg + 180) % 360 - 180))


def walk(w, length, back=False):
    """Waypoints from w over `length` metres forward (or backward), staying on w's road while it continues."""
    out, L = [w], 0.0
    while L < length:
        c = w.previous(STEP_M) if back else w.next(STEP_M)
        if not c:
            break
        same = [v for v in c if v.road_id == w.road_id and v.lane_id == w.lane_id]
        w = same[0] if same else straightest(c, w.transform.rotation.yaw)
        out.append(w)
        L += STEP_M
    return out[::-1] if back else out


def segs_of(wps):
    """Group consecutive waypoints by (road, lane) into segs {c, l, r, id} (right-handed world)."""
    out, cur = [], None
    for w in wps:
        k = (w.road_id, w.lane_id)
        p, yaw, wd = wp_rh(w)
        n = np.array([-math.sin(yaw), math.cos(yaw)])
        if cur is None or cur[0] != k:
            if cur is not None:      # share the joint point so chains have no gap
                for a, b in zip((cur[1], cur[2], cur[3]), (p, p + n * wd / 2, p - n * wd / 2)):
                    a.append(b)
            cur = (k, [], [], [])
            out.append(cur)
        cur[1].append(p)
        cur[2].append(p + n * wd / 2)
        cur[3].append(p - n * wd / 2)
    return [dict(c=np.array(c), l=np.array(l), r=np.array(r), id="%d:%d" % k) for k, c, l, r in out]


def conn_start(m, town, road, lane):
    L = PI.TAB[town][2][road]
    return m.get_waypoint_xodr(road, lane, 0.05 if lane < 0 else L - 0.05)


def to_local(P, t, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(P, float) - t
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def seg_local(segs, t, yaw):
    return [{k: (to_local(v, t, yaw).astype(np.float32) if k != "id" else v) for k, v in s.items()} for s in segs]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def process(job):
    rid, adir, trav, stop = job
    m = PI.get_map(trav["town"])
    adir = Path(adir)
    pose = [json.loads(x) for x in open(adir / "pose.jsonl")]
    fr = sorted((json.loads(x) for x in open(adir / "frames.jsonl")), key=lambda r: r["frame"])
    route = json.loads((adir / "route.json").read_text())
    bf = {p["frame"]: p for p in pose}
    # geometry (world, right-handed)
    taken_c = tuple(stop["conn"])
    brs = sorted(trav["branches"], key=lambda b: (b[1], b[2]) != taken_c)
    w0 = conn_start(m, trav["town"], *taken_c)
    app_w = walk(w0, APPROACH_BACK_M, back=True)[:-1]
    if (app_w[-1].road_id, app_w[-1].lane_id) != tuple(trav["entry"]):
        return rid, [], "entry lane mismatch"
    app = [s for s in segs_of(app_w + [w0]) if len(s["c"]) >= 2]     # ends at the connector start
    branches = []
    for cls, road, lane in brs:
        ws = conn_start(m, trav["town"], road, lane)
        branches.append(dict(cls=cls, segs=segs_of(walk(ws, PI.TAB[trav["town"]][2][road] + EXT_M))))
    ac = np.concatenate([s["c"] for s in app])
    acum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ac, axis=0), axis=1))]
    entry_xy = ac[-1]
    # command: the route's RoadOption inside the junction
    R = np.array([rh(r["x"], r["y"], 0)[0] for r in route])
    ex = np.array(rh(*stop["exit"], 0)[0])
    i0, i1 = int(np.argmin(np.linalg.norm(R - entry_xy, axis=1))), int(np.argmin(np.linalg.norm(R - ex, axis=1)))
    opts = [route[i]["option_name"] for i in range(i0, max(i1, i0) + 1)]
    turn = [o for o in opts if o in ("LEFT", "RIGHT", "STRAIGHT")]
    cmd = turn[0].lower() if turn else "straight"

    def rear(p):
        t, yaw = rh(p["x"], p["y"], p["yaw"])
        return t + REAR_AXLE_X * np.array([math.cos(yaw), math.sin(yaw)]), yaw

    def dist_of(p):
        t, _ = rear(p)
        d = np.linalg.norm(ac - t, axis=1)
        k = int(np.argmin(d))
        return float(acum[-1] - acum[k]), float(d[k])

    fnum = [r["frame"] for r in fr]
    D = [dist_of(bf[f]) if f in bf else (np.nan, np.nan) for f in fnum]
    v = [math.hypot(bf[f]["vx"], bf[f]["vy"]) if f in bf else np.nan for f in fnum]
    entered = next((k for k, (d, off) in enumerate(D) if d < 0.3), len(D))   # first frame at / past the entry
    pick = {}
    for tag, tgt in TARGETS.items():
        ks = [k for k in range(entered) if abs(D[k][0] - tgt) <= TOL_M and D[k][1] <= OFF_M]
        if ks:
            pick[tag] = min(ks, key=lambda k: abs(D[k][0] - tgt))
    st = [k for k in range(entered) if D[k][0] <= STOP_WIN_M and D[k][1] <= OFF_M and v[k] < STOP_V]
    if st:
        pick["stop"] = st[-1]
    out, why = [], Counter()
    last_tick = max(bf)
    for tag, k in pick.items():
        if k < HIST_N or any(fnum[j + 1] - fnum[j] != 4 for j in range(k - HIST_N, k)):
            why["history gap"] += 1
            continue
        f0 = fnum[k]
        if f0 + FUT_N * 10 > last_tick:
            why["short future"] += 1
            continue
        t0, y0 = rear(bf[f0])
        hp = []
        for j in range(k - HIST_N, k + 1):
            t, y = rear(bf[fnum[j]])
            hp.append([*to_local(t, t0, y0), wrap(y - y0)])
        fut = []
        for q in range(1, FUT_N + 1):
            t, y = rear(bf[f0 + 10 * q])
            fut.append([*to_local(t, t0, y0), wrap(y - y0)])
        out.append(dict(token=f"{rid}-{tag}", log=rid, kind="junction", v=float(v[k]), cmd=cmd, dist=D[k][0],
                        classes=",".join(sorted({b["cls"] for b in branches})), taken=trav["taken"], map=trav["town"],
                        town=trav["town"], tag=tag, approach=seg_local(app, t0, y0), node=app[-1]["id"],
                        branches=[dict(cls=b["cls"], segs=seg_local(b["segs"], t0, y0)) for b in branches],
                        future=np.array(fut, np.float32),
                        files=[[str(adir / r["files"][c]) for c in ("front", "front_left", "front_right")] for r in fr[k - HIST_N: k + 1]],
                        frame_t=np.round((np.array(fnum[k - HIST_N: k + 1]) - f0) * TICK, 3),
                        pose=np.array(hp, np.float32), pitch=np.array([bf[fnum[j]]["pitch"] for j in range(k - HIST_N, k + 1)], np.float32),
                        cam=np.array(CAM, np.float32), frame=f0))
    return rid, out, dict(why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rec")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    run = ROOT / a.run
    T = {t["route"]: t for t in json.load(open(REPO / "experiments" / "op_common_cause" / "results" / "carla_traversals.json"))}
    stops = json.loads((ROOT / "plan" / "stops.json").read_text())
    jobs = []
    for f in sorted((run / "done").glob("*.json")):
        rid, rec = f.stem, json.loads(f.read_text())
        att = sorted((run / "attempts" / rid).glob("*"), key=lambda p: int(p.name))
        adir = next((p for p in reversed(att) if (p / "frames.jsonl").exists() and (p / "p4_summary.json").exists()), None)
        if adir is not None:
            jobs.append((rid, str(adir), T[rid], stops[rid]))
    for town in {j[2]["town"] for j in jobs}:
        PI.TAB[town] = PI.xodr_tables(town)
    jobs.sort(key=lambda j: j[2]["town"])
    res, notes = [], {}
    with Pool(a.workers) as pool:
        for rid, out, why in pool.imap_unordered(process, jobs):
            res += out
            notes[rid] = why
    res.sort(key=lambda s: s["token"])
    pickle.dump(res, open(ROOT / f"geom_{a.run}.pkl", "wb"))
    summ = dict(routes=len(jobs), routes_with_samples=len({s["log"] for s in res}), samples=len(res),
                by_tag=Counter(s["tag"] for s in res), by_taken=Counter(s["taken"] for s in res),
                by_speed=Counter("stop" if s["v"] < 0.5 else "low" if s["v"] < 3 else "moving" for s in res),
                dropped=dict(sum((Counter(w) for w in notes.values() if isinstance(w, dict)), Counter())),
                route_errors={r: w for r, w in notes.items() if isinstance(w, str)})
    (ROOT / f"geom_{a.run}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
