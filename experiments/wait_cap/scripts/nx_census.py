#!/usr/bin/env python3
"""Red-light and rear-end census of the AlpaSim nuPlan reads (handoff 2026-10-17, section 4 step 0, items 1-2). CPU only, read-only on
stored runs and on the raw nuPlan test logs; no simulator. Box: $DATA_DIR/envs/navsim2/bin/python (nuPlan map API, pyarrow), repo root.

  nx_census.py scenes  [--workers N]   every public scene (1491; the held 460 are a subset of them, read at 18 decisions): the light /
                                       lead context of the scene window from the raw log, the human's behaviour, and per model and read the
                                       outcome, stop / launch / line-pass times and every rear contact (struck actor from the logged boxes,
                                       speed delta, cause class, the score without the rear-contact truncation)
                                       -> $DATA_DIR/runs/wait_cap/census/scenes/<ts>/scenes.json.gz
  nx_census.py windows [--workers N]   decision 325's uniform 5.5 s windows on the 136 raw test logs (0.5 s grid) and the 1491 public
                                       windows: the same light / lead features, plus a hypothetical red-light runner replayed against the
                                       logged traffic and the scorer's corridor rule -> .../census/windows/<ts>/windows.parquet
  nx_census.py report --scenes F --windows F --out DIR   tables (Markdown fragments + CSV) for results/next/census_red_rear.md

Definitions, fixed before any output:
  frames       raw lidar_pc frames (20 Hz); ego pose = rear axle (nuPlan ego_pose). The AlpaSim nuPlan ego box: front bumper 4.049 m ahead
               of the rear axle, rear 1.127 m behind, width 2.297 m; the scorer's ego point is the box centroid, 1.461 m ahead.
  scene        <log>-<token> starts 30 frames (1.5 s) before its token (decision 325, amendment 1); public reads cover 5.5 s (10 decisions),
               held reads 9.5 s (18). The scene frame is the world frame translated by the ego's first pose (alpasim_utils
               trajdata_data_source), so controller pose + logged ego position at the scene start = world pose.
  route        the human's hindsight route from a frame: its driven path over the next 120 s (2 Hz vertices) plus 15 m straight along the
               final heading. Arcs are rear-axle arcs along it.
  light line   a map stop polygon of type 2 crossed by the route whose lane connector (the one the route takes after it, stoplbl_label's
               match: mean distance of the route 4 / 8 / 12 m past the entry < 3 m) is found; entry arc d. It is ahead while the rear axle
               is less than 1 m into it (so a car stopped with its nose over the near edge is still at the line). Light = the connector's
               traffic_light_status (nearest status within +-0.5 s): red / green / unknown. "Passed" = rear axle 1 m into the polygon.
  red stop     a frame with human speed < 0.3 m/s, the line not yet passed, the front bumper at most 80 m before the near edge, light red.
               Class C: no launch (> 1 m/s) after the first red stop within the window; G: launches within the window.
  lead         a logged vehicle box whose centre is within 1.8 m of the route, whose rear is ahead of the ego's front bumper (-0.5 m), at most
               30 m ahead and at most 2 m past the line's near edge (queued at the light), at the first red-stop frame.
  runner       for red-stop windows without a lead: an ego that replays the human for 0.5 s (force-GT), then accelerates at 1.5 m/s^2 to
               max(8 m/s, its speed) along the human's route, ignoring the light. Contacts with logged vehicle boxes every 0.1 s (both boxes
               shrunk 2 % and corner-rounded as eval.vehicle: roundness 0.5, shrink 0.02), classed by the ego's bumper lines as the scorer
               does (front / rear / lateral; at fault = front or lateral). Corridor: the scorer's _lateral_distance_to_gt (GT = the human's
               rear-axle path over the window at 10 Hz, ego point = box centroid) >= 4 m at a 0.5 s step. Score as score_rollout. The same
               contact check on the human's own poses is the false-positive control.
  rear class   per rear contact (first collision_rear onset in metrics.parquet, scene time t): with the ego's stops (< 0.3 m/s, gaps < 0.5 s
               merged) and hard brakes (>= 2.5 m/s drop in 1 s) in [t - 4, t] and the human's stops (station ranges along the route, +-5 m):
               slow launch      the ego stopped where the human also stopped and the human had left first (human > 1 m/s at t, or the
                                human's stop ended before the ego's);
               no-reason brake  the ego stopped or braked hard where the human did not stop and moved > 2 m/s;
               log stop         the ego stopped where the human stopped and the human had not left by t;
               other            none of these (slower than the log while moving, or at log speed).
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sqlite3
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/lowboard_diag/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RAW = DATA / "datasets/nuplan_raw/test"
CFG = DATA / "datasets/alpasim_nuplan/navtest/configs"

FRONT, REAR, HALF_W = 4.049, 1.127, 2.297 / 2
CEN = (FRONT - REAR) / 2
HZ, OFF = 20, 30
WIN = dict(pub=5.5, held=9.5)
V_STOP, V_GO, V_HMOVE = 0.3, 1.0, 2.0
HOR, EXT = 120.0, 15.0
LINE_IN, LINE_MAX, LEAD_LAT, LEAD_GAP = 1.0, 80.0, 1.8, 30.0
CORR, ROUND, SHRINK = 4.0, 0.5, 0.02
A_RUN, V_RUN = 1.5, 8.0
TAGS = {"RSG": "RSG-L18-s0", "V2": "RTB1ARO-L18-s0", "a": "RTB1O-L18-s0"}


def read_dirs(read, tag):
    if read == "held":
        return sorted(glob.glob(f"{DATA}/runs/route_mem/read/{tag}/c*/run"))
    if tag == "RSG-L18-s0":
        return sorted(glob.glob(f"{DATA}/runs/route_follow/guard/read10/{tag}/b*"))
    return sorted(glob.glob(f"{DATA}/runs/wait_cap/next/cont/wide/pub/{tag}/b*"))


def index_read(read, tag):
    """scene -> run dir, score, failure reason, summary metrics, session (first rollout per scene over the read's chunks)."""
    o = {}
    for d in read_dirs(read, tag):
        f = Path(d, "aggregate/results-summary.json")
        if not f.exists():
            continue
        ses = {}
        for line in open(Path(d, "driver-logs/drive.jsonl")):
            if '"kind": "start"' in line:
                r = json.loads(line)
                ses.setdefault(r["scene"], r["session"])
        for r in json.loads(f.read_text())["rollouts"]:
            o.setdefault(r["clipgt_id"], dict(dir=d, score=float(r.get("score") or 0.0), fail=r.get("failure_reason"),
                                              m=r.get("metrics") or {}, ses=ses.get(r["clipgt_id"])))
    return o


# ---------------------------------------------------------------- raw log, map, boxes
class Log:
    def __init__(self, db):
        self.c = c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = c.execute("select l.token, l.timestamp, e.x, e.y, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy from lidar_pc l "
                         "join ego_pose e on e.token = l.ego_pose_token order by l.timestamp").fetchall()
        self.tok = [r[0].hex() for r in rows]
        a = np.array([r[1:] for r in rows], float)
        self.t = (a[:, 0] - a[0, 0]) / 1e6
        self.xy = a[:, 1:3]
        qw, qx, qy, qz = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
        self.yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.v = np.hypot(a[:, 7], a[:, 8])
        self.idx = {k: i for i, k in enumerate(self.tok)}
        self.loc = c.execute("select map_version from log").fetchone()[0]
        self.tl = {}
        for lp, cid, st in c.execute("select lidar_pc_token, lane_connector_id, status from traffic_light_status"):
            i = self.idx.get(lp.hex())
            if i is not None:
                self.tl.setdefault(i, {})[str(cid)] = 1 if st == "red" else 0 if st == "green" else -1
        self._boxes = None

    def status(self, k, cid, w=10):
        """1 red, 0 green, -1 unknown: the connector's status at frame k, else the nearest within +-w frames."""
        for dk in range(w + 1):
            for kk in ((k,) if dk == 0 else (k - dk, k + dk)):
                s = self.tl.get(kk, {}).get(cid)
                if s is not None and s >= 0:
                    return s
        return -1

    def boxes(self, k):
        """Logged vehicle boxes at frame k: (n, 8) x, y, yaw, length, width, vx, vy, track id (int per log)."""
        if self._boxes is None:
            cat = self.c.execute("select token from category where name = 'vehicle'").fetchone()[0]
            q = ("select b.lidar_pc_token, b.track_token, b.x, b.y, b.yaw, b.length, b.width, b.vx, b.vy from lidar_box b "
                 "join track t on t.token = b.track_token where t.category_token = ?")
            fr, tr, val = [], {}, []
            for lp, tt, *rest in self.c.execute(q, (cat,)):
                i = self.idx.get(lp.hex())
                if i is None:
                    continue
                fr.append(i)
                val.append(rest + [tr.setdefault(tt, len(tr))])
            fr, val = np.asarray(fr), np.asarray(val, float).reshape(-1, 8)
            o = np.argsort(fr, kind="stable")
            fr, val = fr[o], val[o]
            cut = np.searchsorted(fr, np.arange(len(self.t) + 1))
            self._boxes = (val, cut)
        val, cut = self._boxes
        return val[cut[k]:cut[k + 1]] if 0 <= k < len(self.t) else val[:0]


_MAPS = {}


def map_of(loc):
    if loc not in _MAPS:
        import stoplbl_label as SL
        _MAPS[loc] = SL.get_map(loc)
    return _MAPS[loc]


def route(L, i0):
    """The human's rear-axle path: the last <= 30 m before frame i0 (at most 60 s back), then the next 120 s, 2 Hz vertices, + 15 m straight."""
    from shapely.geometry import LineString
    j = int(np.searchsorted(L.t, L.t[i0] + HOR, "right")) - 1
    back = np.arange(i0, max(int(np.searchsorted(L.t, L.t[i0] - 60.0)), 0) - 1, -10)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(L.xy[back], axis=0).T))]
    p = L.xy[np.r_[back[cum <= 30.0][::-1], np.arange(i0 + 10, j + 1, 10)]]
    p = p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 0.05]]
    h = L.yaw[j]
    tail = p[-1] + EXT * np.array([np.cos(h), np.sin(h)])
    return LineString(np.vstack([p, tail]))


def arcs(line, xy):
    import shapely
    return shapely.line_locate_point(line, shapely.points(np.asarray(xy, float)))


def heading_on(line, s):
    a = np.asarray([line.interpolate(max(x - 0.5, 0)).coords[0] for x in np.atleast_1d(s)])
    b = np.asarray([line.interpolate(min(x + 0.5, line.length)).coords[0] for x in np.atleast_1d(s)])
    return np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])


def light_lines(line, mp):
    """Stop polygons of type 1 / 2 crossed by the route: dicts (type, entry arc d, exit arc, connector id or None, match distance)."""
    import shapely
    out = []
    for c in mp["tree"].query(line.buffer(3.0)):
        ty = int(mp["ty"][c])
        if ty not in (1, 2):
            continue
        x = line.intersection(mp["sp"][c])
        if x.is_empty:
            continue
        a = arcs(line, shapely.get_coordinates(x))
        d, e = float(a.min()), float(a.max())
        cid, bd = None, np.inf
        if ty == 2:
            for cc in mp["sp_conn"].get(mp["sp_ids"][c], []):
                g = mp["lcg"][cc]
                dd = float(np.mean([g.distance(line.interpolate(min(d + q, line.length))) for q in (4, 8, 12)]))
                if dd < bd:
                    cid, bd = str(cc), dd
            if bd >= 3.0:
                cid = None
        out.append(dict(ty=ty, d=d, e=e, cid=cid, bd=bd))
    return sorted(out, key=lambda r: r["d"])


def episodes(t, v, merge=0.5):
    """[[start, end or None]] of v < V_STOP; motion gaps < merge s are merged."""
    ep, st = [], None
    for k in range(len(t)):
        if v[k] < V_STOP and st is None:
            st = t[k]
        elif v[k] >= V_STOP and st is not None:
            ep.append([st, t[k]])
            st = None
    if st is not None:
        ep.append([st, None])
    out = []
    for e in ep:
        if out and out[-1][1] is not None and e[0] - out[-1][1] < merge:
            out[-1][1] = e[1]
        else:
            out.append(e)
    return out


# ---------------------------------------------------------------- geometry of the scorer
def ego_poly(x, y, h):
    import shapely
    lf, lr, hw = FRONT * (1 - SHRINK), REAR * (1 - SHRINK), HALF_W * (1 - SHRINK)
    r = ROUND * min(lf + lr, 2 * hw) / 2
    c, s = np.cos(h), np.sin(h)
    pts = [(lf - r, hw - r), (-lr + r, hw - r), (-lr + r, -hw + r), (lf - r, -hw + r)]
    poly = shapely.Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in pts]).buffer(r, quad_segs=4)
    bump = lambda a: shapely.LineString([(x + c * a - s * b, y + s * a + c * b) for b in (hw, -hw)])  # noqa: E731
    return poly, bump(lf), bump(-lr)


def box_polys(B):
    import shapely
    if not len(B):
        return np.array([], dtype=object)
    L, W = B[:, 3] * (1 - SHRINK) / 2, B[:, 4] * (1 - SHRINK) / 2
    r = ROUND * np.minimum(L, W)
    c, s = np.cos(B[:, 2]), np.sin(B[:, 2])
    corners = []
    for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        da, db = a * (L - r), b * (W - r)
        corners.append(np.stack([B[:, 0] + c * da - s * db, B[:, 1] + s * da + c * db], -1))
    P = shapely.polygons(np.stack(corners + corners[:1], 1))
    return shapely.buffer(P, r, quad_segs=4)


def contact(B, ego, skip=()):
    """First intersecting logged box against an ego (poly, front line, rear line): (kind, row) or None. kind front / rear / lateral."""
    import shapely
    if not len(B):
        return None
    keep = ~np.isin(B[:, 7], list(skip)) if skip else np.ones(len(B), bool)
    B = B[keep]
    if not len(B):
        return None
    P = box_polys(B)
    hit = np.flatnonzero(shapely.intersects(P, ego[0]))
    if not len(hit):
        return None
    i = hit[np.argmax(shapely.area(shapely.intersection(P[hit], ego[0])))]
    kind = "front" if P[i].intersects(ego[1]) else "rear" if P[i].intersects(ego[2]) else "lateral"
    return kind, B[i]


def lateral_to_gt(gt, gt_len, end_dir, pt):
    """eval.scorers.ground_truth._lateral_distance_to_gt (0bb4c4b), unchanged."""
    if end_dir is None or gt_len <= 0:
        return float(gt.distance(pt))
    if gt.project(pt) < gt_len - 1e-9:
        return float(gt.distance(pt))
    end = np.asarray(gt.coords, float)[-1, :2]
    off = np.array([pt.x, pt.y]) - end
    return float(abs(off[0] * end_dir[1] - off[1] * end_dir[0]))


def gt_of(L, i0, k1):
    from shapely.geometry import LineString
    p = L.xy[i0:k1 + 1:2]
    gt = LineString(p) if len(p) > 1 else LineString([p[0], p[0]])
    c = np.asarray(gt.coords, float)
    d = c[-1] - c[-2] if len(c) > 1 else np.zeros(2)
    n = float(np.hypot(*d))
    return gt, float(gt.length), (d / n if n > 0 else None)


def replay(L, i0, k1, poses, W):
    """poses: (t, x, y, h) every 0.1 s. Contacts with the logged vehicles and the scorer's corridor / progress rule -> dict."""
    from shapely.geometry import Point
    import shapely
    gt, gl, ed = gt_of(L, i0, k1)
    B0 = L.boxes(i0)
    skip = set(B0[shapely.intersects(box_polys(B0), ego_poly(*L.xy[i0], L.yaw[i0])[0]), 7]) if len(B0) else set()
    hit, cor, t_end = None, None, W
    for t, x, y, h in poses:
        k = i0 + int(round(t * HZ))
        if t >= 0.5 - 1e-9 and hit is None:
            c = contact(L.boxes(k), ego_poly(x, y, h), skip)
            if c:
                hit = (round(float(t), 2), c[0], float(np.hypot(c[1][5], c[1][6])))
        on_step = abs(t * 2 - round(t * 2)) < 1e-6 and t >= 0.5 - 1e-9
        if on_step and cor is None:
            if lateral_to_gt(gt, gl, ed, Point(x + CEN * np.cos(h), y + CEN * np.sin(h))) >= CORR:
                cor = round(float(t), 2)
        if hit is not None:                                     # the scorer drops every step after the first contact
            t_end = hit[0]
            break
    x, y, h = poses[poses[:, 0] <= t_end + 1e-9][-1, 1:]
    prog = gt.project(Point(x + CEN * np.cos(h), y + CEN * np.sin(h))) / gl if gl > 0 else 1.0
    zero = (hit is not None and hit[1] != "rear") or (cor is not None and (hit is None or cor <= hit[0]))
    sc = 0.0 if zero else (1.0 if gl < 5.0 else min(max(prog, 0.0), 1.0) / 0.8)
    return dict(contact_t=hit[0] if hit else None, contact=hit[1] if hit else None, contact_v_other=hit[2] if hit else None,
                corridor_t=cor, score=min(sc, 1.0), gt_len=gl)


def runner(L, line, i0, k1, s, W):
    th = L.t[i0:k1 + 1] - L.t[i0]
    tg = np.round(np.arange(0, W + 1e-6, 0.1), 2)
    sr = np.interp(tg, th, s)
    vv = float(np.interp(0.5, th, L.v[i0:k1 + 1]))
    vmax = max(V_RUN, vv)
    for j in range(1, len(tg)):
        if tg[j] > 0.5 + 1e-9:
            vv = min(vmax, vv + A_RUN * 0.1)
            sr[j] = sr[j - 1] + vv * 0.1
    xy = np.asarray([line.interpolate(x).coords[0] for x in sr])
    h = heading_on(line, sr)
    r = replay(L, i0, k1, np.c_[tg, xy, h], W)
    kh = np.minimum(i0 + np.round(tg * HZ).astype(int), k1)
    hum = replay(L, i0, k1, np.c_[tg, L.xy[kh], L.yaw[kh]], W)
    return {f"run_{k}": v for k, v in r.items()} | {f"hum_{k}": v for k, v in hum.items() if k in ("contact_t", "contact", "score")}


def lead_gap(L, line, k, s_k, d):
    import shapely
    B = L.boxes(k)
    if not len(B):
        return np.nan
    pts = shapely.points(B[:, :2])
    a = shapely.line_locate_point(line, pts)
    lat = shapely.distance(line, pts)
    rear = a - B[:, 3] / 2
    gap = rear - (s_k + FRONT)
    m = (lat <= LEAD_LAT) & (gap >= -0.5) & (gap <= LEAD_GAP) & (rear <= d + 2.0)
    return float(gap[m].min()) if m.any() else np.nan


def feats(L, mp, i0, W, line=None, run_red=False, detail=False):
    k1 = int(np.searchsorted(L.t, L.t[i0] + W, "right")) - 1
    if i0 < 0 or L.t[k1] - L.t[i0] < W - 0.06:
        return None
    line = line or route(L, i0)
    ks = np.arange(i0, k1 + 1)
    t, v = L.t[ks] - L.t[i0], L.v[ks]
    s = arcs(line, L.xy[ks])
    p = L.xy[i0:k1 + 1:2]
    f = dict(v0=float(v[0]), v_end=float(v[-1]), path=float(np.hypot(*np.diff(p, axis=0).T).sum()),
             maxdisp=float(np.hypot(*(L.xy[ks] - L.xy[i0]).T).max()))
    f["static"] = f["maxdisp"] < 1.0
    go = np.flatnonzero(v > V_GO)
    f["h_launch_t"] = float(t[go[0]]) if v[0] < V_STOP and len(go) else np.nan
    lines = [x for x in light_lines(line, mp) if x["d"] > s[0] - LINE_IN]
    ss = next((x for x in lines if x["ty"] == 1), None)
    f["ss_d_front"] = ss["d"] - s[0] - FRONT if ss else np.nan
    tl = next((x for x in lines if x["ty"] == 2 and x["cid"] is not None), None)
    f["tl"] = tl is not None
    if detail:
        f["v_str"] = " ".join(f"{x:.1f}" for x in v[::10])
        f["h_stops"] = [[float(a_), None if b_ is None else float(b_)] for a_, b_ in episodes(t, v)]
        f["s"] = s[::2].round(2).tolist()
    if tl is None:
        return f
    d, cid = tl["d"], tl["cid"]
    st = np.array([L.status(k, cid) for k in ks])
    passed = s >= d + LINE_IN
    kp = int(np.argmax(passed)) if passed.any() else None
    dfront = d - s - FRONT
    pre = np.arange(len(ks)) < (kp if kp is not None else len(ks))
    f.update(tl_d_front=float(dfront[0]), tl_d=float(d), tl_cid=cid, tl_bd=float(tl["bd"]), st0=int(st[0]),
             red_frac=float((st == 1).mean()), green_frac=float((st == 0).mean()), h_pass_t=float(t[kp]) if kp is not None else np.nan,
             h_pass_st=int(st[kp]) if kp is not None else -2)
    f["red_start"] = bool(st[0] == 1 and dfront[0] <= LINE_MAX and not passed[0])
    g = np.flatnonzero((st == 0) & (np.cumsum(st == 1) > 0))
    f["t_green"] = float(t[g[0]]) if len(g) else np.nan
    rs = (v < V_STOP) & pre & (st == 1) & (dfront <= LINE_MAX)
    f["red_stop"] = bool(rs.any())
    f["red_pass_nostop"] = bool(kp is not None and st[kp] == 1 and not rs.any())
    if detail:
        f["st_str"] = "".join({1: "R", 0: "G", -1: "."}[x] for x in st[::10])
    kr = int(np.argmax(rs)) if rs.any() else (0 if f["red_start"] else None)
    if kr is not None:
        after = np.flatnonzero((v > V_GO) & (np.arange(len(v)) > kr))
        f.update(red_t=float(t[kr]), red_dfront=float(dfront[kr]), red_go_t=float(t[after[0]]) if len(after) else np.nan,
                 red_go_st=int(st[after[0]]) if len(after) else -2, lead=lead_gap(L, line, int(ks[kr]), float(s[kr]), d))
        f["red_cls"] = ("G" if len(after) else "C") if rs.any() else ""
        if run_red and rs.any() and not np.isfinite(f["lead"]):
            f |= runner(L, line, i0, k1, s, W)
    return f


# ---------------------------------------------------------------- one model rollout against the log
def ctrl_of(d, ses):
    f = Path(d, f"controller/alpasim_controller_{ses}.csv")
    if not ses or not f.exists():
        return None
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    c = lambda k: A[:, h[k]]  # noqa: E731
    qx, qy, qz, qw = c("qx"), c("qy"), c("qz"), c("qw")
    return dict(t=c("timestamp_us") / 1e6, x=c("x"), y=c("y"), yaw=np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
                v=np.hypot(c("vx"), c("vy")))


SERIES = ("collision_rear", "collision_front", "collision_lateral", "collision_any", "offroad", "left_corridor_laterally",
          "progress_rel_to_total")


def cmd_metrics(a):
    """AlpaSim venv (polars reads the int128 column pyarrow refuses): the per-step series of every rollout of the six reads."""
    import polars as pl
    from jevdrive.run import Run
    with Run("wait_cap", "census/metrics", config=vars(a)) as run:
        out = {}
        for rd in ("pub", "held"):
            for tag in TAGS.values():
                for d in read_dirs(rd, tag):
                    for f in sorted(glob.glob(f"{d}/rollouts/*/*/metrics.parquet")):
                        m = pl.read_parquet(f, columns=["name", "timestamps_us", "values"]).filter(pl.col("name").is_in(SERIES))
                        out[f"{d}|{Path(f).parent.parent.name}"] = {
                            n: [(g["timestamps_us"].cast(pl.Float64) / 1e6).to_list(), g["values"].cast(pl.Float64).to_list()]
                            for (n,), g in m.sort("timestamps_us").group_by("name", maintain_order=True)}
        with gzip.open(run.path("metrics.json.gz"), "wt") as fh:
            json.dump(out, fh)
        run.summary.update(rollouts=len(out))


def mseries(MS, d, scene):
    x = MS.get(f"{d}|{scene}")
    return {n: np.array(v, float).T for n, v in x.items()} if x else {}


def onset(M, n):
    x = M.get(n)
    return float(x[x[:, 1] > 0][0, 0]) if x is not None and (x[:, 1] > 0).any() else None


def rear_class(tc, E, s_e_of, s_h_stops, v_h_at):
    """Cause class of a rear contact at scene time tc (module docstring)."""
    near = lambda s_: any(lo - 5 <= s_ <= hi + 5 for lo, hi, _, _ in s_h_stops)  # noqa: E731
    h_left = lambda s_: any(lo - 5 <= s_ <= hi + 5 and t1 is not None and t1 <= tc for lo, hi, _, t1 in s_h_stops)  # noqa: E731
    eps = [e for e in episodes(E["t"], E["v"]) if e[0] <= tc and (e[1] is None or e[1] >= tc - 4)]
    vh = v_h_at(tc)
    for a_, b_ in reversed(eps):
        se = s_e_of(a_)
        if near(se):
            ended = [t1 for lo, hi, _, t1 in s_h_stops if lo - 5 <= se <= hi + 5 and t1 is not None]
            if vh > V_GO or (ended and (b_ is None or min(ended) < b_)):
                return "slow launch", f"ego stopped {a_:.1f}-{'end' if b_ is None else f'{b_:.1f}'} s where the human stopped; human {vh:.1f} m/s at contact"
            return "log stop", f"ego stopped {a_:.1f} s where the human stopped; human {vh:.1f} m/s at contact"
        if max(v_h_at(a_), vh) > V_HMOVE:
            return "no-reason brake", f"ego stopped at {a_:.1f} s where the human did not stop (human {v_h_at(a_):.1f} m/s then)"
    t, v = E["t"], E["v"]
    for k in np.flatnonzero((t >= tc - 4) & (t <= tc)):
        j = int(np.searchsorted(t, t[k] + 1.0))
        if j < len(t) and v[k] - v[j] >= 2.5:
            se = s_e_of(t[k])
            if near(se):
                return ("slow launch" if h_left(se) else "log stop"), f"hard brake at {t[k]:.1f} s where the human stopped"
            if max(v_h_at(t[k]), vh) > V_HMOVE:
                return "no-reason brake", f"hard brake {v[k]:.1f} -> {v[j]:.1f} m/s at {t[k]:.1f} s, human {v_h_at(t[k]):.1f} m/s"
    ve = float(np.interp(tc, t, v))
    return "other", ("slower than the log while moving" if ve < vh - 1.0 else "at about log speed") + f" (ego {ve:.1f}, human {vh:.1f} m/s)"


def model(L, line, i0, f, info, scene, W, MS):
    m = info["m"]
    r = dict(score=info["score"], fail=info["fail"], **{k: m.get(k) for k in (
        "collision_rear", "collision_front", "collision_lateral", "collision_at_fault", "offroad", "left_corridor_laterally",
        "dist_traveled_m", "gt_dist_traveled_m", "progress_clipped_rel")})
    E = ctrl_of(info["dir"], info["ses"])
    if E is None:
        return r
    wx, wy = E["x"] + L.xy[i0, 0], E["y"] + L.xy[i0, 1]
    se = arcs(line, np.c_[wx, wy])
    r["v_str"] = " ".join(f"{x:.1f}" for x in np.interp(np.arange(0.5, W + 1e-6, 0.5), E["t"], E["v"]))
    r["stops"] = [[round(float(a_), 2), None if b_ is None else round(float(b_), 2)] for a_, b_ in episodes(E["t"], E["v"])]
    go = np.flatnonzero(E["v"] > V_GO)
    r["launch_t"] = float(E["t"][go[0]]) if E["v"][0] < V_STOP and len(go) else (np.nan if E["v"][0] < V_STOP else 0.0)
    r["dist_end"] = float(se[-1] - se[0])
    if f.get("tl"):
        ps = np.flatnonzero(se >= f["tl_d"] + LINE_IN)
        r["pass_t"] = float(E["t"][ps[0]]) if len(ps) else np.nan
        r["pass_st"] = L.status(i0 + int(round(E["t"][ps[0]] * HZ)), f["tl_cid"]) if len(ps) else -2
    M = mseries(MS, info["dir"], scene)
    if not M:
        return r
    tr = onset(M, "collision_rear")
    if (m.get("collision_rear") or 0) > 0 and tr is not None:
        k = i0 + int(round(tr * HZ))
        yw = np.unwrap(E["yaw"])
        pose = lambda tt: tuple(float(np.interp(tt, E["t"], q)) for q in (wx, wy, yw))  # noqa: E731
        c, kc = None, k
        for kk in range(k - 10, k + 3):                 # the metric step is 0.5 s: the first overlap in the 0.5 s up to the onset
            c = contact(L.boxes(kk), ego_poly(*pose((kk - i0) / HZ)))
            if c is not None:
                kc = kk
                break
        if c is None:                                   # replayed tracks are resampled / smoothed: allow 0.5 m at the onset frame
            P_ = ego_poly(*pose(tr))
            c = contact(L.boxes(k), tuple(g_.buffer(0.5) for g_ in P_))
        x, y, h = pose((kc - i0) / HZ)
        ve = float(np.interp((kc - i0) / HZ, E["t"], E["v"]))
        o = dict(t=tr, t_first=(kc - i0) / HZ, v_ego=ve, v_human=float(L.v[min(kc, len(L.v) - 1)]))
        k = kc
        if c is not None:
            B = c[1]
            tid = B[7]
            q0, q1 = [L.boxes(kk) for kk in (k - 4, k + 4)]
            a0, a1 = q0[q0[:, 7] == tid], q1[q1[:, 7] == tid]
            vel = (a1[0, :2] - a0[0, :2]) / 0.4 if len(a0) and len(a1) else B[5:7]
            ch, sh = np.cos(h), np.sin(h)
            dx, dy = B[0] - x, B[1] - y
            o.update(kind=c[0], v_other_long=float(vel[0] * ch + vel[1] * sh), other_x=float(ch * dx + sh * dy),
                     other_y=float(-sh * dx + ch * dy), dpsi=float(np.degrees(np.angle(np.exp(1j * (B[2] - h))))))
            o["dv"] = o["v_other_long"] - ve
        # the human's stops as station ranges along the route; human speed by scene time
        th = L.t[i0:] - L.t[i0]
        hs = arcs(line, L.xy[i0:i0 + int((W + 1) * HZ)])
        th_w = th[:len(hs)]
        hst = [(float(np.interp(a_, th_w, hs)), float(np.interp(b_ if b_ is not None else th_w[-1], th_w, hs)), a_, b_)
               for a_, b_ in episodes(th_w, L.v[i0:i0 + len(hs)])]
        o["class"], o["why"] = rear_class(tr, E, lambda tt: float(np.interp(tt, E["t"], se)), hst,
                                          lambda tt: float(np.interp(tt, th_w, L.v[i0:i0 + len(hs)])))
        # the score the rollout would have had without the truncation at the rear contact (raw per-step metrics after it)
        later = [n for n in ("collision_front", "collision_lateral", "offroad", "left_corridor_laterally")
                 if n in M and (M[n][(M[n][:, 0] > tr), 1] > 0).any()]
        pr = M.get("progress_rel_to_total")
        gtd = m.get("gt_dist_traveled_m") or 0
        p_at = float(pr[pr[:, 0] <= tr + 1e-6][-1, 1]) if pr is not None else np.nan
        p_end = float(pr[-1, 1]) if pr is not None else np.nan
        ps = lambda p_: 1.0 if gtd < 5.0 else min(max(p_, 0.0), 1.0) / 0.8  # noqa: E731
        o.update(later=later, p_at=p_at, p_end=p_end, gt_short=bool(gtd < 5.0), score_at_contact=min(ps(p_at), 1.0),
                 score_untruncated=0.0 if later else min(ps(p_end), 1.0),
                 score_nohit=0.0 if {"offroad", "left_corridor_laterally"} & set(later) else min(ps(p_end), 1.0))
        r["rear"] = o
    tl_, tf_ = onset(M, "collision_lateral"), onset(M, "collision_front")
    if (m.get("collision_at_fault") or 0) > 0 and tl_ is not None and (tf_ is None or tl_ < tf_):
        r["lat"] = struck(L, i0, E, wx, wy, tl_)       # the first at-fault contact is lateral: who touched the flank, from where
    return r


def struck(L, i0, E, wx, wy, t0):
    """The logged vehicle box that first overlaps the ego in the 0.5 s up to a contact onset t0: kind, position in the ego frame,
    heading difference, speeds along the ego heading."""
    k = i0 + int(round(t0 * HZ))
    yw = np.unwrap(E["yaw"])
    pose = lambda tt: tuple(float(np.interp(tt, E["t"], q)) for q in (wx, wy, yw))  # noqa: E731
    for kk in range(k - 10, k + 3):
        c = contact(L.boxes(kk), ego_poly(*pose((kk - i0) / HZ)))
        if c is not None:
            break
    if c is None:
        return dict(t=t0)
    x, y, h = pose((kk - i0) / HZ)
    B, ve = c[1], float(np.interp((kk - i0) / HZ, E["t"], E["v"]))
    q0, q1 = [L.boxes(j) for j in (kk - 4, kk + 4)]
    a0, a1 = q0[q0[:, 7] == B[7]], q1[q1[:, 7] == B[7]]
    vel = (a1[0, :2] - a0[0, :2]) / 0.4 if len(a0) and len(a1) else B[5:7]
    ch, sh = np.cos(h), np.sin(h)
    vo = float(vel[0] * ch + vel[1] * sh)
    return dict(t=t0, t_first=(kk - i0) / HZ, kind=c[0], other_x=float(ch * (B[0] - x) + sh * (B[1] - y)),
                other_y=float(-sh * (B[0] - x) + ch * (B[1] - y)), dpsi=float(np.degrees(np.angle(np.exp(1j * (B[2] - h))))),
                v_other_long=vo, v_ego=ve, dv=vo - ve)


# ---------------------------------------------------------------- commands
def scenes_log(job):
    log, items, reads, MS = job
    L = Log(RAW / f"{log}.db")
    mp = map_of(L.loc)
    out = []
    for scene, held in items:
        ic = L.idx.get(scene.rsplit("-", 1)[1])
        if ic is None:
            out.append(dict(scene=scene, log=log, err="token not in the log"))
            continue
        i0 = ic - OFF
        line = route(L, i0)
        r = dict(scene=scene, log=log, held=held, t0=float(L.t[i0]))
        for rd in ("pub", "held") if held else ("pub",):
            f = feats(L, mp, i0, WIN[rd], line=line, detail=True)
            if f is None:
                continue
            for key, tag in TAGS.items():
                info = reads.get((rd, key), {}).get(scene)
                if info:
                    f[key] = model(L, line, i0, f, info, scene, WIN[rd], MS)
            r[rd] = f
        out.append(r)
    return out


def cmd_scenes(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("wait_cap", "census/scenes", config=vars(a)) as run:
        held_split, all_split = splits.load("alpasim_nuplan/onpol-held"), splits.load("alpasim_nuplan/onpol-all")
        run.use_split(held_split)
        run.use_split(all_split)
        ids = sorted(p.stem for p in CFG.glob("*.yaml"))
        held = set(held_split.members)
        reads = {(rd, k): index_read(rd, t) for rd in ("pub", "held") for k, t in TAGS.items()}
        run.info("reads: " + ", ".join(f"{rd}/{k} {len(v)}" for (rd, k), v in reads.items()))
        with gzip.open(a.metrics, "rt") as fh:
            MS = json.load(fh)
        by = {}
        for s in ids:
            by.setdefault(s.rsplit("-", 1)[0], []).append((s, s in held))
        jobs = []
        for lg, it in sorted(by.items()):
            sub = {k: {s: v[s] for s, _ in it if s in v} for k, v in reads.items()}
            keys = {f"{x['dir']}|{s}" for v in sub.values() for s, x in v.items()}
            jobs.append((lg, it, sub, {k: MS[k] for k in keys if k in MS}))
        res = par.pmap(scenes_log, jobs, run=run, workers=min(a.workers, len(jobs)), desc="logs")
        res.raise_if_failed()
        rows = [x for v in res.values for x in v]
        with gzip.open(run.path("scenes.json.gz"), "wt") as fh:
            json.dump(rows, fh, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else float(o)
                      if isinstance(o, (np.floating, np.integer)) else bool(o) if isinstance(o, np.bool_) else str(o))
        run.summary.update(scenes=len(rows), errors=sum("err" in r for r in rows), held=sum(bool(r.get("held")) for r in rows))


def windows_log(job):
    db, pub, step = job
    L = Log(db)
    mp = map_of(L.loc)
    starts = [("uniform", "", int(np.searchsorted(L.t, s))) for s in np.arange(0, L.t[-1], step)]
    for sid in pub:
        ic = L.idx.get(sid.rsplit("-", 1)[1])
        if ic is not None:
            starts.append(("public", sid, int(np.searchsorted(L.t, L.t[ic] - OFF / HZ))))
    rows = []
    for kind, sid, i0 in starts:
        f = feats(L, mp, i0, 5.5, run_red=True)
        if f is not None:
            rows.append(dict(log=Path(db).stem, kind=kind, scene=sid, t0=float(L.t[i0]), **f))
    return rows


def cmd_windows(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("wait_cap", "census/windows", config=vars(a)) as run:
        run.use_split(splits.load("alpasim_nuplan/onpol-all"))
        ids = sorted(p.stem for p in CFG.glob("*.yaml"))
        by = {}
        for s in ids:
            by.setdefault(s.rsplit("-", 1)[0], []).append(s)
        dbs = sorted(RAW.glob("*.db"))
        jobs = [(str(d), by.get(d.stem, []), a.step) for d in dbs]
        res = par.pmap(windows_log, jobs, run=run, workers=min(a.workers, len(jobs)), desc="logs")
        res.raise_if_failed()
        W = pd.DataFrame([r for v in res.values for r in v])
        W.to_parquet(run.path("windows.parquet"))
        run.summary.update(uniform=int((W.kind == "uniform").sum()), public=int((W.kind == "public").sum()),
                           red_stop_uniform=int(W[W.kind == "uniform"].red_stop.fillna(False).sum()))


# ---------------------------------------------------------------- report (Mac or box; pandas)
def human_class(r):
    """Scene-level human behaviour at the light: C waits (stops at red, no launch in the window), G stops at red then launches,
    pass-red / pass-green (passes the line without stopping, label at the pass), approach (red ahead, neither stops nor passes)."""
    if r.get("red_stop"):
        return r.get("red_cls") or "C"
    if r.get("red_start"):
        st = r.get("h_pass_st")
        return "pass-red" if st == 1 else "pass-green" if st == 0 else "pass-unknown" if st == -1 else "approach"
    return ""


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.scenes, "rt") as fh:
        S = json.load(fh)
    rows = []
    for r in S:
        for rd in ("pub", "held"):
            f = r.get(rd)
            if not isinstance(f, dict):                     # r["held"] is the split flag (bool) when the scene has no held read
                continue
            base = dict(read=rd, scene=r["scene"], log=r["log"], hcls=human_class(f), path=f.get("path"),
                        **{k: f.get(k) for k in ("v0", "static", "tl", "tl_d_front", "st0", "st_str", "red_start", "red_stop", "red_cls",
                                                 "red_t", "red_go_t", "red_go_st", "lead", "h_launch_t", "h_pass_t", "h_pass_st",
                                                 "t_green", "v_str")})
            for key in TAGS:
                m = f.get(key)
                if not m:
                    continue
                rr, la = m.get("rear") or {}, m.get("lat") or {}
                rows.append(base | dict(lat=bool(la), lat_x=la.get("other_x"), lat_dpsi=la.get("dpsi"), lat_dv=la.get("dv"), lat_t=la.get("t"),
                    model=key, score=m["score"], fail=m.get("fail"), launch_t=m.get("launch_t"), pass_t=m.get("pass_t"),
                    pass_st=m.get("pass_st"), m_v_str=m.get("v_str"), dist=m.get("dist_traveled_m"), gt_m=m.get("gt_dist_traveled_m"),
                    rear=bool(rr), rear_t=rr.get("t"), rear_dv=rr.get("dv"), rear_v_ego=rr.get("v_ego"), rear_v_other=rr.get("v_other_long"),
                    rear_v_human=rr.get("v_human"), rear_kind=rr.get("kind"), rear_class=rr.get("class"), rear_why=rr.get("why"),
                    rear_later=",".join(rr.get("later") or []) if rr else None, score_at_contact=rr.get("score_at_contact"),
                    score_untrunc=rr.get("score_untruncated"), score_nohit=rr.get("score_nohit"), rear_dpsi=rr.get("dpsi"),
                    rear_t_first=rr.get("t_first"), dist_end=m.get("dist_end")))
    T = pd.DataFrame(rows)
    T.to_csv(out / "per_scene_model.csv.gz", index=False)
    md = []
    f3 = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"  # noqa: E731
    f1 = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1f}"  # noqa: E731

    # alignment: logged path over the scene window against the reads' gt_dist_traveled_m
    al = T.dropna(subset=["gt_m", "path"]).drop_duplicates(["read", "scene"])
    md += ["## Alignment", "", "| read | scenes | logged path within 1 m of gt_dist_traveled_m | median abs diff (m) |", "|---|--:|--:|--:|"]
    for rd, g in al.groupby("read"):
        d = (g["path"] - g["gt_m"]).abs()
        md.append(f"| {rd} | {len(g)} | {(d < 1).mean():.1%} | {d.median():.2f} |")

    # D2: red-light scenes among zeros / lows
    T["red"] = T.red_start.fillna(False).astype(bool) | T.red_stop.fillna(False).astype(bool)
    T["lost"] = 1 - T.score
    md += ["", "## D2 counts", "", "| read | model | scenes | zeros | lows (0-0.5) | red-light scenes | zeros with red | lows with red | "
           "points lost: all / red scenes | red by human class: C / G / pass-red / pass-green / approach (points lost) |",
           "|---|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    for (rd, key), g in T.groupby(["read", "model"], sort=False):
        z, lo, rd_ = g.score == 0, (g.score > 0) & (g.score < 0.5), g.red
        by = " / ".join(f"{(g.hcls == c).sum()} ({g.lost[g.hcls == c].sum():.2f})" for c in ("C", "G", "pass-red", "pass-green", "approach"))
        md.append(f"| {rd} | {key} | {len(g)} | {z.sum()} | {lo.sum()} | {rd_.sum()} | {(z & rd_).sum()} | {(lo & rd_).sum()} | "
                  f"{g.lost.sum():.2f} / {g.lost[rd_].sum():.2f} | {by} |")
    W_ = T.pivot_table(index=["read", "scene"], columns="model", values="score")
    red_sc = T[T.red].drop_duplicates(["read", "scene"]).set_index(["read", "scene"])
    bad = [ix for ix in red_sc.index if ix in W_.index and (W_.loc[ix] < 0.5).any()]
    md += ["", "## D2 red-light scenes where a model scores < 0.5", "",
           "| read | scene | human: class / launch / pass (label) / label green / lead (m) | human v (0.5 s) | " +
           " | ".join(f"{k}: score / launch / pass (label) / fail / rear t, dv" for k in TAGS) + " |",
           "|---|---|---|---|" + "---|" * len(TAGS)]
    lab = {1: "R", 0: "G", -1: "?", -2: "-"}
    for rd, sc in sorted(bad):
        h = red_sc.loc[(rd, sc)]
        cells = []
        for key in TAGS:
            m = T[(T.read == rd) & (T.scene == sc) & (T.model == key)]
            if not len(m):
                cells.append("-")
                continue
            m = m.iloc[0]
            cells.append(f"{m.score:.2f} / {f1(m.launch_t)} / {f1(m.pass_t)} ({lab.get(m.pass_st, '-') if pd.notna(m.pass_t) else '-'}) / "
                         f"{m.fail or '-'} / " + (f"{m.rear_t:.1f}, {f1(m.rear_dv)}" if m.rear else "-"))
        md.append(f"| {rd} | {sc[-16:-8]} | {h.hcls} / {f1(h.h_launch_t)} / {f1(h.h_pass_t)} ({lab.get(h.h_pass_st, '-') if pd.notna(h.h_pass_t) else '-'}) / "
                  f"{f1(h.t_green)} / {f1(h.lead)} | {h.v_str} | " + " | ".join(cells) + " |")
    T[T.red].to_csv(out / "red_scenes.csv", index=False)

    # D3: windows
    W = pd.read_parquet(a.windows)
    for c in ("red_start", "red_stop", "red_pass_nostop", "tl", "static"):
        W[c] = W[c].fillna(False).astype(bool)
    W["has_lead"] = np.isfinite(W.lead.astype(float))
    W["C"] = W.red_stop & (W.red_cls == "C")
    W["G"] = W.red_stop & (W.red_cls == "G")
    U, P = W[W.kind == "uniform"], W[W.kind == "public"]
    cls = [("light line ahead (<= 80 m, connector matched)", lambda D: D.tl),
           ("red label at the window start", lambda D: D.red_start),
           ("stopped at a red light (any)", lambda D: D.red_stop),
           ("  first in line (no lead)", lambda D: D.red_stop & ~D.has_lead),
           ("  queued behind a lead", lambda D: D.red_stop & D.has_lead),
           ("  C: no launch in the window", lambda D: D.C),
           ("    C, static the whole window", lambda D: D.C & D.static),
           ("    C, first in line", lambda D: D.C & ~D.has_lead),
           ("  G: launches within the window", lambda D: D.G),
           ("    G, first in line", lambda D: D.G & ~D.has_lead),
           ("    G, launch while the label is still red", lambda D: D.G & (D.red_go_st == 1)),
           ("passes the line on a red label without stopping", lambda D: D.red_pass_nostop),
           ("human static the whole window (decision 325 S)", lambda D: D.static),
           ("  S and stopped at a red light", lambda D: D.static & D.red_stop),
           ("  S, at a light line, not red (green / unknown)", lambda D: D.static & D.tl & ~D.red_stop),
           ("  S, stop sign within 15 m, no light", lambda D: D.static & ~D.tl & (D.ss_d_front.astype(float) <= 15)),
           ("  S, none of these", lambda D: D.static & ~D.tl & ~(D.ss_d_front.astype(float) <= 15))]
    md += ["", "## D3 window shares", "", f"{len(U)} uniform windows on {U.log.nunique()} logs, {len(P)} public windows. CIs: log-clustered bootstrap.", "",
           "| class | uniform [95 % CI] | public [95 % CI] | public n | ratio |", "|---|--:|--:|--:|--:|"]
    sh = []
    for name, fn in cls:
        ru, rp = stats.bootstrap(fn(U).astype(float), groups=U.log), stats.bootstrap(fn(P).astype(float), groups=P.log)
        sh.append(dict(cls=name.strip(), uniform=ru["mean"], u_lo=ru["lo"], u_hi=ru["hi"], public=rp["mean"], p_lo=rp["lo"], p_hi=rp["hi"],
                       n_public=int(fn(P).sum())))
        md.append(f"| {name.replace('  ', '&nbsp;&nbsp;')} | {ru['mean']:.3f} [{ru['lo']:.3f}, {ru['hi']:.3f}] | {rp['mean']:.3f} [{rp['lo']:.3f}, {rp['hi']:.3f}] | "
                  f"{int(fn(P).sum())} | {ru['mean'] / rp['mean'] if rp['mean'] else np.inf:.1f} |")
    pd.DataFrame(sh).to_csv(out / "window_shares.csv", index=False)
    R = W[W.red_stop & ~W.has_lead & W.run_score.notna()].copy()
    R["atfault"] = R.run_contact.isin(["front", "lateral"])
    R["cor0"] = R.run_corridor_t.notna() & ~R.atfault
    md += ["", "## D3 runner (red-stop windows, first in line)", "",
           "| set | class | windows | runner at-fault contact | runner corridor zero | runner mean score | human-replay contact (false positives) |",
           "|---|---|--:|--:|--:|--:|--:|"]
    for kind, g0 in R.groupby("kind"):
        for name, g in (("C static", g0[g0.C & g0.static]), ("C moving", g0[g0.C & ~g0.static]), ("G", g0[g0.G])):
            if len(g):
                md.append(f"| {kind} | {name} | {len(g)} | {g.atfault.mean():.3f} | {g.cor0.mean():.3f} | {g.run_score.mean():.3f} | "
                          f"{g.hum_contact.notna().mean():.3f} |")

    # E: rear contacts
    E = T[T.rear].copy()
    E["sub"] = np.where(E.rear_class == "other", np.where(E.rear_why.fillna("").str.startswith("slower"), "slower than the log", "at log speed"),
                        E.rear_class)
    E["same"] = E.rear_dpsi.abs() < 45
    E["masked"] = E.rear_later.fillna("").str.contains("collision")
    subs = ("slow launch", "no-reason brake", "log stop", "slower than the log", "at log speed")
    md += ["", "## E rear contacts", "", "| read | model | scenes | rear contacts | actor found: same direction / crossing or oncoming | "
           "slow launch / no-reason brake / log stop / slower than the log / at log speed | dv median [max] m/s (same direction) | "
           "ego v at contact median | contact time median (s) | score of rear-ended scenes (mean) | progress the truncation cost | "
           "later pass-through collisions masked | max abs(score at contact - summary) |", "|---|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|"]
    for (rd, key), g in T.groupby(["read", "model"], sort=False):
        e = E[(E.read == rd) & (E.model == key)]
        cc = " / ".join(str((e["sub"] == c).sum()) for c in subs)
        sd = e[e.same]
        cost = (e.score_nohit - e.score)[~e.masked].sum()
        md.append(f"| {rd} | {key} | {len(g)} | {len(e)} | {int(e.same.sum())} / {int((e.rear_dpsi.notna() & ~e.same).sum())} | {cc} | "
                  f"{f1(sd.rear_dv.median())} [{f1(sd.rear_dv.max())}] | {f1(e.rear_v_ego.median())} | {f1(e.rear_t.median())} | {f3(e.score.mean())} | "
                  f"{cost:.2f} | {int(e.masked.sum())} | {(e.score_at_contact - e.score).abs().max():.4f} |")
    E.to_csv(out / "rear_contacts.csv", index=False)
    md += ["", "| read | scene | " + " | ".join(f"{k}: score / t / dv / class" for k in TAGS) + " | human v (0.5 s) |", "|---|---|" + "---|" * len(TAGS) + "---|"]
    for (rd, sc), g in E.groupby(["read", "scene"]):
        cells = []
        for key in TAGS:
            m = T[(T.read == rd) & (T.scene == sc) & (T.model == key)]
            m = m.iloc[0] if len(m) else None
            x = E[(E.read == rd) & (E.scene == sc) & (E.model == key)]
            cells.append("-" if m is None else f"{m.score:.2f}" + (f" / {m.rear_t:.1f} / {f1(m.rear_dv)} / {x.iloc[0]['sub']}" if m.rear else ""))
        md.append(f"| {rd} | {sc[-16:-8]} | " + " | ".join(cells) + f" | {g.iloc[0].v_str} |")

    A = T[T.lat].copy()
    A["behind"] = (A.lat_x < CEN) & (A.lat_dpsi.abs() < 45) & (A.lat_dv > 0.5)
    md += ["", "## E2 at-fault zeros whose first contact is lateral", "",
           "| read | model | lateral-first at-fault zeros | actor found | side-swiped from behind (centre behind the ego's, same direction, closing > 0.5 m/s) |",
           "|---|---|--:|--:|--:|"]
    for (rd, key), g in A.groupby(["read", "model"], sort=False):
        md.append(f"| {rd} | {key} | {len(g)} | {int(g.lat_x.notna().sum())} | {int(g.behind.sum())} "
                  f"({', '.join(s_[-16:-8] for s_ in g[g.behind].scene)}) |")

    # D2b: launches from a standstill, at a red light against elsewhere; D2c: the human waits at red the whole window
    L0 = T[(T.v0 < V_STOP) & T.h_launch_t.notna()].copy()
    L0["at_red"] = L0.red_stop.fillna(False).astype(bool)
    L0["never"] = L0.launch_t.isna()
    L0["delay"] = L0.launch_t - L0.h_launch_t
    md += ["", "## D2b launches from a standstill (human starts < 0.3 m/s, launches > 1 m/s in the window)", "",
           "| read | model | where | scenes | model never launches | launch delay median (s) | launch delay > 1 s | mean score | points lost |",
           "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    for (rd, key), g in L0.groupby(["read", "model"], sort=False):
        for name, x in (("red light", g[g.at_red]), ("elsewhere", g[~g.at_red])):
            if len(x):
                md.append(f"| {rd} | {key} | {name} | {len(x)} | {int(x.never.sum())} | {f1(x.delay.median())} | {int(((x.delay > 1) | x.never).sum())} | "
                          f"{x.score.mean():.3f} | {(1 - x.score).sum():.2f} |")
    C0 = T[T.hcls == "C"]
    md += ["", "## D2c the human stops at red and does not launch in the window", "",
           "| read | model | scenes | mean score | zeros (reasons) | model drives > 3 m | median distance (m) |", "|---|---|--:|--:|---|--:|--:|"]
    for (rd, key), g in C0.groupby(["read", "model"], sort=False):
        z = g[g.score == 0]
        md.append(f"| {rd} | {key} | {len(g)} | {g.score.mean():.3f} | {len(z)} ({', '.join(z.fail.fillna('-'))}) | {int((g.dist_end > 3).sum())} | "
                  f"{f1(g.dist_end.median())} |")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("metrics")
    p = sp.add_parser("scenes")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--metrics", required=True, help="metrics.json.gz of the metrics step")
    p = sp.add_parser("windows")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--step", type=float, default=0.5)
    p = sp.add_parser("report")
    p.add_argument("--scenes", required=True), p.add_argument("--windows", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(metrics=cmd_metrics, scenes=cmd_scenes, windows=cmd_windows, report=lambda a: cmd_report(a))[a.cmd](a)
