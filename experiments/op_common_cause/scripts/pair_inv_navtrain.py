"""Pair inventory, navtrain part: junction-approach frames with an alternative branch available from the nuPlan map.

Run with envs/navsim2 on the box (CPU only):
  NUPLAN_MAPS_ROOT=... NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=... \
  python pair_inv_navtrain.py --out <dir> --workers 20
Writes <out>/navtrain_frames.parquet (one row per navtrain frame) and <out>/navtrain_summary.json.
Definitions are in results/pair_inventory.md. No navhard / test logs are read (navtrain logs live in navsim_logs/trainval).
"""
import argparse, json, math, os, pickle, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_M = 30.0            # junction counts as "ahead" if the branching point is within this arc length of ego
TURN_DEG = 25.0            # |heading change| over the connector above this = left / right
MAX_DEPTH = 4              # lanes followed through single-exit connectors while looking for a branch point
TAKEN_FRAMES = 20          # frames (0.5 s each) scanned ahead to see which branch ego actually took
SPEED_BINS = [0, 1, 3, 6, 10, 15, 1e9]
SPEED_LAB = ["<1", "1-3", "3-6", "6-10", "10-15", ">=15"]
CMD = ["left", "straight", "right", "unknown"]  # driving_command one-hot order in OpenScene logs

_maps = {}


def get_map(loc):
    if loc not in _maps:
        from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
        db = get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))
        _maps[loc] = NuPlanMapFactory(db).build_map_from_name(loc)
    return _maps[loc]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def cls_of(dyaw):
    d = math.degrees(dyaw)
    if d > 135 or d < -135:
        return "uturn"
    return "left" if d > TURN_DEG else "right" if d < -TURN_DEG else "straight"


def end_heading(obj):
    return obj.baseline_path.discrete_path[-1].heading


def branches_of(lane):
    """Distinct exits of a lane: list of (class, connector_id, next_lane_id). Class from heading change over the connector."""
    h0 = end_heading(lane)
    out, seen = [], set()
    for c in lane.outgoing_edges:
        nxt = tuple(sorted(x.id for x in c.outgoing_edges))
        if nxt in seen:
            continue
        seen.add(nxt)
        out.append((cls_of(wrap(end_heading(c) - h0)), c.id, nxt))
    return out


def find_lane(m, pt, yaw):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    lanes = m.get_all_map_objects(pt, L.LANE)
    if not lanes:
        return ("in_junction" if m.get_all_map_objects(pt, L.LANE_CONNECTOR) else "off_map"), None
    best, bd = None, 9
    for ln in lanes:
        h = ln.baseline_path.get_nearest_pose_from_position(pt).heading
        d = abs(wrap(h - yaw))
        if d < bd:
            best, bd = ln, d
    return ("ok", best) if bd < math.radians(60) else ("heading_mismatch", None)


def approach(lane, pt):
    """Walk single-exit connectors; return (dist to branching point, node lane, branches) for the first lane with >1 exit
    within WINDOW_M, else (None, None, [])."""
    cum = lane.baseline_path.length - lane.baseline_path.get_nearest_arc_length_from_position(pt)
    node = lane
    for _ in range(MAX_DEPTH):
        if cum > WINDOW_M:
            break
        outs = node.outgoing_edges
        if len(outs) >= 2:
            br = branches_of(node)
            if len(br) >= 2:
                return cum, node, br
            return None, None, []
        if len(outs) == 0:
            break
        nxt = outs[0].outgoing_edges
        if len(nxt) != 1:
            break
        cum += outs[0].baseline_path.length + nxt[0].baseline_path.length
        node = nxt[0]
    return None, None, []


def rb_classes(m, lane):
    """Union of branch classes over all lanes of the ego roadblock (depth 0): tier B, may need a lane change."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    rb = m.get_map_object(lane.get_roadblock_id(), L.ROADBLOCK)
    cs = set()
    if rb is not None:
        for ln in rb.interior_edges:
            cs |= {b[0] for b in branches_of(ln)}
    return cs


def process_log(path):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.state_representation import Point2D
    frames = pickle.load(open(path, "rb"))
    keep = TOKENS.get(Path(path).stem)  # None -> whole log is in navtrain, but only listed tokens count
    m = get_map(frames[0]["map_location"])
    n = len(frames)
    xs = [(f["ego2global_translation"][0], f["ego2global_translation"][1], quat_yaw(f["ego2global_rotation"])) for f in frames]
    conn_cache = {}

    def conns_at(i):
        """connector id -> |heading difference| between ego and the connector baseline at ego position."""
        if i not in conn_cache:
            p = Point2D(xs[i][0], xs[i][1])
            conn_cache[i] = {c.id: abs(wrap(c.baseline_path.get_nearest_pose_from_position(p).heading - xs[i][2]))
                             for c in m.get_all_map_objects(p, L.LANE_CONNECTOR)}
        return conn_cache[i]

    rows = []
    for i, f in enumerate(frames):
        if f["token"] not in keep:
            continue
        x, y, yaw = xs[i]
        pt = Point2D(x, y)
        ed = f["ego_dynamic_state"]
        v = math.hypot(ed[0], ed[1])
        cmd = CMD[int(np.argmax(f["driving_command"]))]
        # contiguous history: frames are 0.5 s apart inside a log pickle
        hist = i
        r = dict(log=Path(path).stem, token=f["token"], scene=f["scene_token"], frame_idx=i, map=f["map_location"], v=v, cmd=cmd,
                 hist_frames=hist, hist_s=hist * 0.5, status="", dist=np.nan, node="", classes="", n_branch=0, taken="",
                 taken_cmd_agree=np.nan, classes_rb="")
        st, lane = find_lane(m, pt, yaw)
        r["status"] = st
        if lane is not None:
            d, node, br = approach(lane, pt)
            if lane.baseline_path.length - lane.baseline_path.get_nearest_arc_length_from_position(pt) < 1.0 and conns_at(i):
                r["status"] = "in_junction"   # at the lane end and already inside a connector polygon
            elif d is None:
                r["status"] = "no_junction_30m"
                r["classes_rb"] = ",".join(sorted(rb_classes(m, lane))) if lane.baseline_path.length - lane.baseline_path.get_nearest_arc_length_from_position(pt) <= WINDOW_M else ""
            else:
                r["status"] = "branch"
                r["dist"] = d
                r["node"] = node.id
                r["n_branch"] = len(br)
                r["classes"] = ",".join(sorted({b[0] for b in br}))
                cids = {b[1]: b[0] for b in br}
                # connector polygons overlap at junctions: pick the connector best aligned with ego over the frames it contains
                acc = defaultdict(list)
                for j in range(i, min(n, i + TAKEN_FRAMES + 1)):
                    for cid, dh in conns_at(j).items():
                        if cid in cids:
                            acc[cid].append(dh)
                if acc:
                    multi = {c: v for c, v in acc.items() if len(v) >= 2}
                    pool_ = multi or acc
                    cid = min(pool_, key=lambda c: np.mean(pool_[c]))
                    if np.mean(pool_[cid]) < math.radians(30):
                        r["taken"] = cids[cid]
                if r["taken"]:
                    r["taken_cmd_agree"] = float(r["taken"] == cmd)
        rows.append(r)
    return rows


TOKENS = {}


def init(tokens):
    global TOKENS
    TOKENS = tokens


def summarize(df):
    S = {}
    S["n_frames"] = len(df)
    S["n_logs"] = df.log.nunique()
    S["n_scenes"] = df.scene.nunique()
    S["status"] = df.status.value_counts().to_dict()
    S["hist_s_min"] = float(df.hist_s.min())
    S["frames_hist_ge_2s"] = int((df.hist_s >= 2).sum())
    S["frames_hist_ge_4s"] = int((df.hist_s >= 4).sum())
    br = df[df.status == "branch"].copy()
    br["multi"] = br.classes.str.contains(",")
    A = br[br.multi]
    def block(d):
        o = {"frames": len(d), "logs": d.log.nunique(), "scenes": d.scene.nunique(),
             "segments": d.groupby(["log", "node"]).ngroups, "junctions_unique": d.groupby(["map", "node"]).ngroups}
        return o
    S["branch_any"] = block(br)
    S["alt_available"] = block(A)
    S["alt_class_sets"] = {k: block(g) for k, g in A.groupby("classes")}
    T = A[A.taken != ""]
    S["alt_taken_known"] = block(T)
    S["taken_unknown"] = block(A[A.taken == ""])
    S["pair_eligible"] = block(T)  # taken known and >=1 alternative class
    S["taken_class_counts"] = T.taken.value_counts().to_dict()
    Ts = T[T.dist >= 2.0]
    S["pair_eligible_strict_dist_2_30m"] = block(Ts)       # drops frames already at the lane end (< 2 m)
    seg = T.groupby(["log", "node"]).agg(taken=("taken", lambda x: x.mode().iat[0]), v=("v", "median"), n=("v", "size"))
    S["segments_by_taken_class"] = seg.taken.value_counts().to_dict()
    S["segments_by_median_speed"] = pd.cut(seg.v, SPEED_BINS, labels=SPEED_LAB, right=False).value_counts().reindex(SPEED_LAB).astype(int).to_dict()
    S["segment_frames_median"] = float(seg.n.median())
    S["frames_per_log_pair_eligible_max_share"] = float(T.groupby("log").size().max() / len(T))
    alt_rows = []
    for _, r in T.iterrows():
        for c in r.classes.split(","):
            if c != r.taken:
                alt_rows.append((r.taken, c, r.v, r.log, r.node))
    ad = pd.DataFrame(alt_rows, columns=["taken", "alt", "v", "log", "node"])
    S["alt_pairs_total"] = len(ad)
    S["alt_pairs_by_taken_alt"] = {f"{a}->{b}": int(c) for (a, b), c in ad.groupby(["taken", "alt"]).size().items()}
    S["alt_pairs_by_alt"] = ad.alt.value_counts().to_dict()
    sp = pd.cut(T.v, SPEED_BINS, labels=SPEED_LAB, right=False)
    S["pair_eligible_frames_by_speed"] = sp.value_counts().reindex(SPEED_LAB).astype(int).to_dict()
    S["pair_eligible_by_speed_and_alt"] = {}
    if len(ad):
        spa = pd.cut(ad.v, SPEED_BINS, labels=SPEED_LAB, right=False)
        S["alt_pairs_by_speed"] = spa.value_counts().reindex(SPEED_LAB).astype(int).to_dict()
        S["alt_pairs_by_speed_alt"] = {f"{s}|{a}": int(c) for (s, a), c in pd.crosstab(spa, ad.alt).stack().items() if c}
    S["dist_to_branch_quantiles_alt"] = A.dist.quantile([.05, .25, .5, .75, .95]).round(1).to_dict()
    S["taken_vs_logged_command_agree"] = float(T.taken_cmd_agree.mean()) if len(T) else None
    S["logged_cmd_counts_pair_eligible"] = T.cmd.value_counts().to_dict()
    # tier B: roadblock-level alternatives for approach frames whose own lane has a single exit
    nb = df[(df.status == "no_junction_30m") & (df.classes_rb != "")]
    nb2 = nb[nb.classes_rb.str.contains(",")]
    S["tierB_roadblock_multi_class_frames_own_lane_single_exit"] = block(nb2.assign(node=nb2.log)) if len(nb2) else {"frames": 0}
    # all-frame speed histogram and perturbation eligibility
    S["all_frames_by_speed"] = pd.cut(df.v, SPEED_BINS, labels=SPEED_LAB, right=False).value_counts().reindex(SPEED_LAB).astype(int).to_dict()
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import yaml
    nav = os.environ.get("NAVSIM_DEVKIT", os.path.expanduser("~/data/third_party/navsim"))
    cfg = yaml.safe_load(open(f"{nav}/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"))
    logs = cfg["log_names"]
    tok = cfg["tokens"]
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / "trainval"
    assert not any("navhard" in str(p) for p in [root])
    # tokens are globally unique; group by log lazily: keep set of all tokens per log after loading
    toks = set(tok)
    paths = [root / f"{l}.pkl" for l in logs if (root / f"{l}.pkl").exists()]
    print("logs", len(logs), "on disk", len(paths), "tokens", len(toks), flush=True)
    if a.limit:
        paths = paths[:a.limit]
    TK = {p.stem: toks for p in paths}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.workers, initializer=init, initargs=(TK,)) as pool:
        for k, r in enumerate(pool.imap_unordered(process_log, [str(p) for p in paths], chunksize=1)):
            rows += r
            if k % 50 == 0:
                print(k, len(paths), len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "navtrain_frames.parquet")
    S = summarize(df)
    json.dump(S, open(out / ("navtrain_summary.json" if not a.limit else "navtrain_summary_limit.json"), "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
