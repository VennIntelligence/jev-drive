"""Route geometry for the image-command test on navtrain (envs/navsim2, CPU; nuPlan map API).

For every sample token: the approach path (ego lane through single-exit connectors to the branching lane) and, per exit
of the branching lane, the branch path (connector + following lanes to >= EXT_M past the connector end), each as
centreline + left / right boundary polylines in the t0 rear-axle frame (x fwd, y left). Junction samples are the
pair-inventory frames (decision 93) that are also in op_lb's lb_navtrain pool (their 4 keyframes and GIMM frames are on
disk); straight samples are lb_navtrain frames with no branching lane within 30 m and command straight (lane keeping
control). Also the logged future (8 x 0.5 s) in the t0 frame.

  NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
  OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim $DATA_DIR/envs/navsim2/bin/python \
      experiments/op_img_cmd/scripts/img_geom_nav.py --workers 24 [--limit 10]
Output: $DATA_DIR/runs/op_img_cmd/geom/nav.pkl (list of dicts), nav_summary.json.
"""
import argparse, json, math, os, pickle, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))   # JEV_REPO: dev copies outside the repo
sys.path.insert(0, str(REPO / "experiments" / "op_common_cause" / "scripts"))
import pair_inv_navtrain as PI  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
OUT = DATA / "runs" / "op_img_cmd" / "geom"
EXT_M = 45.0          # branch path length past the connector end
STRAIGHT_N = 300      # lane-keeping control frames
BACK_M = 25.0         # approach path kept behind ego (history frames see it)


def xy(path):
    return np.array([[p.x, p.y] for p in path.discrete_path], float)


def seg(obj):
    """centre / left / right polylines of a lane or connector (global frame)."""
    return dict(c=xy(obj.baseline_path), l=xy(obj.left_boundary), r=xy(obj.right_boundary), id=str(obj.id))


def straightest(objs, h0):
    return min(objs, key=lambda o: abs(PI.wrap(o.baseline_path.discrete_path[-1].heading - h0)))


def extend(obj, need):
    """Follow outgoing edges (straightest at each fork) until `need` metres are collected after obj."""
    out, L = [], 0.0
    while L < need and obj.outgoing_edges:
        obj = straightest(obj.outgoing_edges, obj.baseline_path.discrete_path[-1].heading)
        out.append(seg(obj))
        L += obj.baseline_path.length
    return out


def approach_path(lane, pt):
    """Lanes / connectors from the ego lane to the branching lane (as PI.approach), plus the branching lane object."""
    path, node, cum = [seg(lane)], lane, lane.baseline_path.length - lane.baseline_path.get_nearest_arc_length_from_position(pt)
    for _ in range(PI.MAX_DEPTH):
        outs = node.outgoing_edges
        if len(outs) >= 2 or len(outs) == 0 or cum > PI.WINDOW_M:
            break
        nxt = outs[0].outgoing_edges
        if len(nxt) != 1:
            break
        path += [seg(outs[0]), seg(nxt[0])]
        cum += outs[0].baseline_path.length + nxt[0].baseline_path.length
        node = nxt[0]
    return path, node


def back_path(lane):
    """Up to BACK_M of predecessors (straightest) so history frames have the approach drawn too."""
    out, L, obj = [], 0.0, lane
    while L < BACK_M and obj.incoming_edges:
        obj = straightest(obj.incoming_edges, obj.baseline_path.discrete_path[0].heading)
        out.insert(0, seg(obj))
        L += obj.baseline_path.length
    return out


def to_local(segs, t, yaw):
    R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    return [{k: ((v - t) @ R.T).astype(np.float32) if k != "id" else v for k, v in s.items()} for s in segs]


def process_log(job):
    log, rows = job
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    frames = pickle.load(open(Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / "trainval" / f"{log}.pkl", "rb"))
    m = PI.get_map(frames[0]["map_location"])
    tok = {f["token"]: i for i, f in enumerate(frames)}
    out = []
    for r in rows:
        i = tok[r["token"]]
        f = frames[i]
        t = np.array(f["ego2global_translation"][:2], float)
        yaw = PI.quat_yaw(f["ego2global_rotation"])
        pt = Point2D(*t)
        st, lane = PI.find_lane(m, pt, yaw)
        if lane is None:
            continue
        app, node = approach_path(lane, pt)
        app = back_path(lane) + app
        br = []
        if r["kind"] == "junction":
            for cls, cid, _ in PI.branches_of(node):
                c = m.get_map_object(cid, L.LANE_CONNECTOR)
                br.append(dict(cls=cls, segs=[seg(c)] + extend(c, EXT_M)))
            if len({b["cls"] for b in br}) < 2:
                continue
        else:
            br.append(dict(cls="straight", segs=extend(node, 60.0)))
        fut = []
        for j in range(i + 1, min(len(frames), i + 9)):
            g = np.array(frames[j]["ego2global_translation"][:2], float)
            fut.append(np.r_[to_local([dict(c=g[None])], t, yaw)[0]["c"][0], PI.wrap(PI.quat_yaw(frames[j]["ego2global_rotation"]) - yaw)])
        out.append(dict(r, map=f["map_location"], approach=to_local(app, t, yaw), node=str(node.id),
                        branches=[dict(cls=b["cls"], segs=to_local(b["segs"], t, yaw)) for b in br],
                        future=np.array(fut, np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet(DATA / "processed" / "op_common_cause" / "pair_inventory" / "navtrain_frames.parquet")
    mt = json.loads((DATA / "runs" / "op_lb" / "lb_navtrain" / "meta.json").read_text())
    row_of = {n: k for k, n in enumerate(mt["names"])}
    df = df[df.token.isin(row_of)]
    J = df[(df.status == "branch") & df.classes.str.contains(",") & (df.taken != "")].assign(kind="junction")
    S = df[(df.status == "no_junction_30m") & (df.cmd == "straight") & (df.v >= 0.5)]
    S = S.sample(min(STRAIGHT_N, len(S)), random_state=0).assign(kind="straight")
    D = pd.concat([J, S])
    if a.limit:
        D = D.groupby("kind").head(a.limit)
    D = D.assign(row=D.token.map(row_of))
    keep = ["token", "log", "kind", "v", "cmd", "dist", "classes", "taken", "row"]
    jobs = [(log, g[keep].to_dict("records")) for log, g in D.groupby("log")]
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(process_log, jobs)):
            res += r
            if k % 50 == 0:
                print(k, len(jobs), len(res), flush=True)
    res.sort(key=lambda s: (s["kind"], s["token"]))
    name = "nav" + ("_limit" if a.limit else "")
    pickle.dump(res, open(OUT / f"{name}.pkl", "wb"))
    summ = dict(n_in=len(D), n_out=len(res), by_kind=pd.Series([s["kind"] for s in res]).value_counts().to_dict(),
                n_branch=pd.Series([len(s["branches"]) for s in res]).value_counts().to_dict())
    json.dump(summ, open(OUT / f"{name}_summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
