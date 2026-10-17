#!/usr/bin/env python3
"""EXPLORE-G (diagnosis after decisions 337 / 339, exploratory, no training): (Q1) does the retain line (RTB1O-L18-s0, V2 = RTB1ARO-L18-s0,
RTB1OB1-L18-s1) track turns worse than RSG-L18 because its lateral pathway is lost (a) or because of its speed into the turn (b);
(Q2) how accurate are the shipped model's lane lines on continuous straight roads, at junctions and on irregular roads; the route-point bias.
Stored reads, plus pool replays of the zero rollouts (their logged driver messages through every checkpoint of the same seed).

  geo     (box, AlpaSim venv, CPU)  nx_turns.py geo --out DIR [--limit N]
          every rollout of the held 460 @18 reads of HELD and the public 1491 @10 reads of PUB -> DIR/scenes.parquet (bend geometry + outcome
          per rollout), DIR/dec.parquet (per decision: speed, route station / lateral / heading error, plan and executed heading change
          against the route's, lane class, lane-centre / road-centre / route-memory / plan offsets at 10 / 20 / 30 m), DIR/trace.parquet
          (controller samples in the bend), DIR/bev.pkl (HELD7 and the user's EXAMPLES)
  msgs    (box, AlpaSim venv, CPU)  nx_turns.py msgs --out DIR       driver-side messages of every zero rollout with its rollout.asl kept
  replay  (box, envs/op-train, one GPU: one pool job per checkpoint)  nx_turns.py replay --out DIR [--ckpts T ..]
          each msgs group through the real driver class with every checkpoint of its seed set (CKSET; the group's own checkpoint gives the
          identity against the stored plans): served plan, fed route memory, and the raw lane-line / road-edge / lead heads of the served
          checkpoint (FT) and of the shipped policy on the same vision tokens (P0) -> DIR/replay/<group>__<ckpt>.pkl,
          DIR/replay_identity_<ckpt>.json
  olg     (box, AlpaSim venv, CPU)  nx_turns.py olg --out DIR
          every replayed plan (and the stored one) against the rebuilt route of its decision -> DIR/replay.parquet
  lanes   (box, AlpaSim venv, CPU)  nx_turns.py lanes --out DIR      the replayed decisions (own checkpoint) and eyes_vs_head's ctrl group
          (200 public scenes, P2H10-F-s0) against the map -> DIR/lanes.parquet
  report  (Mac, .venv)  nx_turns.py report --out DIR --res RESULTS_DIR [--dev DIR]   tables and figures -> RESULTS_DIR/explore_g.json, figs;
          --dev: nx.py hist's dev-row plans (<tag>.npz) of the same checkpoints (the open-loop read on logged frames)

Definitions (fixed before any output was read):
  route        AlpaSim's MAP route rebuilt at every logged anchor (rf_follow.regen_full): the human's lane centres; heading h(s) = chord heading over
               +-5 m at a 1 m step (rf_follow.heading), curvature = dh/ds, radius = 1 / |curvature|
  stretch      the route between the first anchor's station s0 and the log's own station at the last decision of the read (s_end; k 17 held, 9 public)
  bend         the 40 m window of the stretch with the largest |net heading change| (bend_deg, signed, left +); entry / exit = 10 % / 90 % of the
               change over [a - 10, a + 50] m (only when |bend_deg| >= 15; else no bend); r_min = smallest radius on the stretch. Testable: exit + 10 m <= s_end
  outcome      completed = the controller trace (10 Hz) crosses exit + 10 m before the zero step (the first 2 Hz step with offroad, a front / lateral
               contact, or dist_to_gt >= 4 m; long_read.one's rule) within 2.5 m of the route; max_lat = largest |lateral| from the route over
               [entry - 10, exit + 10] before the zero step; out = lateral signed to the outside of the bend
  plan turn    plan heading change at 1 / 2 s (pose yaw) against req = route heading change over the plan's own arc minus the current heading
               error (the turn the plan must make to be aligned with the route there); executed = the controller's heading change over the next
               1 s against the plan's at 1 s and against the route's over the arc driven
  lane refs    own lane = the map lane containing the car centre (rear axle + 1.4 m) within 60 deg of the heading, chained forward (the successor
               nearest the route, else the straightest); c_map(d) = its centre line's y at x = d (ego frame, left +; d from the rear axle in geo,
               from the camera in lanes); road centre = midpoint of the road-area boundary crossings bracketing c_map(d); seg = junction (the car
               centre or the own-lane centre at 10 / 20 m lies in two lane polygons > 30 deg apart: ev.Geo.junction) | straight (not junction,
               own-lane and route heading change over 40 m both < 10 deg) | curved (the rest) | none (no own lane)
  heads        ego lane centre from the lane lines = (y_1 + y_2) / 2 at x = cam + d (lines 1 / 2 = left / right ego lines; ev.heads); scaled =
               the same about the camera with decision 104's lateral scale undone (/ 0.689); present = both line probs > 0.5

Added 2026-10-17 after the first read (exploratory; the first replay fed an empty route memory, see replay):
  lane class   zeros_mech's Lanes.cls on the car centre and heading (the evaluator's own lane polygons and centre-line yaw): same = a containing
               lane within 60 deg, else opposite = one beyond 120 deg, else cross; none = no lane
  outcome      (bends only) the first in time of: the zero step while the car's station is in [entry - 20, exit + 10] (offroad | contact = front /
               lateral collision | diverged = dist_to_gt >= 4 m only) and the first opposite-lane sample (controller trace every 0.2 s in the same
               window, before the zero step) -> offroad | contact | opposite | diverged; else zero_before (zero step before the window),
               completed, incomplete (testable, not completed), untestable
  lane width   at route points every 2 m over [entry, exit]: the containing lane within 60 deg of the route heading, distance to its left edge +
               right edge; w_mid at the bend's mid station, w_min the smallest; r_bend = smallest radius over [entry, exit]
  seg3         junction (as seg) | straight = continuous straight (seg straight, the own lane reaches 30 m and its width changes < 0.5 m from 10 to
               30 m) | irregular (curved, merging / splitting / tapering, or the own lane ends before 30 m) | none
  replay       the driver's route tracker as sh30_driver.main() sets it (RMEM = rmem, SERVE_ORDER from the checkpoint); identity = the group's own
               checkpoint against the stored plan (pose distance at 1 / 2 / 4 s) and the stored fed memory (max |difference|)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("geo", "msgs", "lanes", "olg"):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(p) for p in (REPO, REPO / "lib", REPO / "experiments/route_mem/lib", REPO / "experiments/route_mem/scripts",
                                 REPO / "experiments/alpasim/scripts", REPO / "experiments/alpasim/lib", REPO / "experiments/anchor_offset/scripts",
                                 REPO / "experiments/eyes_vs_head/scripts", REPO / "experiments/alpasim_long/scripts",
                                 REPO / "experiments/route_follow/scripts", REPO / "experiments/op_parity/scripts",
                                 REPO / "experiments/zeros_mech/scripts", HERE)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
READ = DATA / "runs/route_mem/read"
NX = DATA / "runs/wait_cap/next"
HELD = ("RTB1O-L18-s0", "RTB1ARO-L18-s0", "RTB1OB1-L18-s1", "RSG-L18-s0", "RSG-L18-s1")
PUB = {"RTB1O-L18-s0": NX / "cont/wide/pub/RTB1O-L18-s0", "RTB1ARO-L18-s0": NX / "cont/wide/pub/RTB1ARO-L18-s0",
       "RSG-L18-s0": DATA / "runs/route_follow/guard/read10/RSG-L18-s0"}
CKSET = {"s0": ("RSG-L18-s0", "RTB1O-L18-s0", "RTB1ARO-L18-s0"), "s1": ("RSG-L18-s1", "RTB1OB1-L18-s1")}
KEND = dict(held=17, pub=9)
T0, DT = 17_000, 500_000
DS = (10.0, 20.0, 30.0)
WIN, LAT_OK, CAR_C, SCALE, BEND_MIN = 40.0, 2.5, 1.4, 0.689, 15.0
SER = ("dist_to_gt_trajectory", "offroad", "collision_front", "collision_lateral", "collision_rear", "wrong_lane")
HELD7 = ("53d3787367c75240", "de4ace24443b5b2d", "6d2b59f78d995d9c", "9d1d720d0e2e511e", "9a675656c3c85f4c", "ddc1271ea57154bc",
         "1f54430427975391")
EXAMPLES = ("9a675656", "53d37873", "ddc1271e", "1f544304", "9d1d720d",          # the user's video judgements (handoff 3 B): wide / outside,
            "81a72614", "365c0937", "a7fa8bcc", "d33c6db3", "a9609780")          # cuts inside, opposite lane


def ckset(label):
    return CKSET[label[-2:]]


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def in_ego(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, float).reshape(-1, 2) - np.asarray(a[:2], float)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def to_local(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.asarray(a[:2], float) + np.c_[c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]]


def y_at(Pe, x):
    """Ego-frame polyline (n, 2), ordered forward -> y of its first crossing of the line x (nan if none)."""
    if Pe is None or len(Pe) < 2:
        return np.nan
    m = np.flatnonzero((Pe[:-1, 0] - x) * (Pe[1:, 0] - x) <= 0)
    if not len(m):
        return np.nan
    i = m[0]
    dx = Pe[i + 1, 0] - Pe[i, 0]
    return float(Pe[i, 1] if abs(dx) < 1e-9 else Pe[i, 1] + (x - Pe[i, 0]) / dx * (Pe[i + 1, 1] - Pe[i, 1]))


def arcs(P):
    P = np.asarray(P, float)[:, :2]
    return np.cumsum(np.r_[np.hypot(*P[0]), np.hypot(*np.diff(P, axis=0).T)])


def runs_of(st, label):
    return sorted(glob.glob(f"{READ}/{label}/c*/run")) if st == "held" else sorted(glob.glob(f"{PUB[label]}/b*"))


def summaries(dirs):
    o = {}
    for d in dirs:
        for r in json.loads(Path(d, "aggregate/results-summary.json").read_text())["rollouts"]:
            o.setdefault(r["clipgt_id"], (d, dict(score=float(r.get("score") or 0.0), **(r.get("metrics") or {}))))
    return o


def ctrl_path(run, scene):
    """Ego (t_us, x, y, yaw, speed) every 0.1 s from the controller CSV of the scene's last session (nx_atfault.ctrl_path)."""
    import csv
    ses = None
    for line in open(f"{run}/driver-logs/drive.jsonl"):
        if '"kind": "start"' in line and f'"{scene}"' in line:
            ses = json.loads(line)["session"]
    f = Path(run) / f"controller/alpasim_controller_{ses}.csv"
    if ses is None or not f.exists():
        return None
    R = list(csv.DictReader(open(f)))
    P = np.array([[float(r["timestamp_us"]), float(r["x"]), float(r["y"]), 2 * np.arctan2(float(r["qz"]), float(r["qw"])),
                   float(np.hypot(float(r["vx"]), float(r["vy"])))] for r in R])
    P = P[np.argsort(P[:, 0])]
    P[:, 3] = np.unwrap(P[:, 3])
    return P


def steps(D, scene):
    """The evaluator's 2 Hz series of one rollout on the decision grid (long_read.one's file) -> {k: dict(gt, off, front, side, rear, wrong)}."""
    import polars as pl
    f = sorted(glob.glob(f"{D}/rollouts/{scene}/*/metrics.parquet"))
    if not f:
        return {}
    m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(SER)).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    ser = {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}
    g = lambda n, t: (ser.get(n, {}).get(t) or 0.0) > 0                                     # noqa: E731
    return {int((t - T0) // DT): dict(gt=float(ser["dist_to_gt_trajectory"][t] or 0.0), off=g("offroad", t), front=g("collision_front", t),
                                      side=g("collision_lateral", t), rear=g("collision_rear", t), wrong=g("wrong_lane", t))
            for t in sorted(ser.get("dist_to_gt_trajectory", {})) if t >= T0 and (t - T0) % DT == 0}


# ---------------------------------------------------------------- map: own lane, junction, road boundary
class LaneMap:
    def __init__(self, ds, bbox):
        import shapely
        from shapely.ops import unary_union
        from trajdata.maps.vec_map_elements import MapElementType as MT
        import ev as EV
        lo, hi = bbox
        inb = lambda P: bool(((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1])).any())  # noqa: E731
        self.L, ids, polys = {}, [], []
        for lid, ln in ds.map.elements.get(MT.ROAD_LANE, {}).items():
            L, R, C = (np.asarray(e.points, float)[:, :2] if e is not None else None for e in (ln.left_edge, ln.right_edge, ln.center))
            if L is None or R is None or C is None or len(C) < 2 or not any(inb(p) for p in (L, R, C)):
                continue
            poly = shapely.Polygon(np.r_[L, R[::-1]]).buffer(0)
            self.L[str(lid)] = dict(C=C, Le=L, Re=R, nxt=[str(x) for x in (getattr(ln, "next_lanes", None) or ())], poly=poly)
            ids.append(str(lid))
            polys.append(poly)
        self.ids, self.tree, self.sh = ids, (shapely.STRtree(polys) if polys else None), shapely
        areas = []
        for ra in ds.map.elements.get(MT.ROAD_AREA, {}).values():
            E = np.asarray(ra.exterior_polygon.points, float)[:, :2]
            if len(E) > 2 and inb(E):
                areas.append(shapely.Polygon(E, [np.asarray(h.points, float)[:, :2] for h in ra.interior_holes]).buffer(0))
        b = unary_union(areas).boundary if areas else None
        geoms = [] if b is None else (list(b.geoms) if hasattr(b, "geoms") else [b])
        self.road = [np.asarray(g.coords)[:, :2] for g in geoms]
        self.SR = EV.segs_of(self.road)
        self.cross = EV.cross

    @staticmethod
    def _dir(C, i):
        j = min(max(i, 0) + 1, len(C) - 1)
        i = j - 1
        return float(np.arctan2(C[j, 1] - C[i, 1], C[j, 0] - C[i, 0]))

    def containing(self, p):
        if self.tree is None:
            return []
        return [self.ids[int(i)] for i in self.tree.query(self.sh.Point(*p), predicate="within")]

    def junction(self, p):
        ys = []
        for lid in self.containing(p):
            C = self.L[lid]["C"]
            ys.append(self._dir(C, int(np.argmin(np.hypot(*(C - p).T)))))
        ys = np.array(ys)
        return bool(len(ys) > 1 and np.abs(wrap(ys[:, None] - ys[None])).max() > np.radians(30))

    def own(self, p, yaw):
        best = None
        for lid in self.containing(p):
            C = self.L[lid]["C"]
            i = int(np.argmin(np.hypot(*(C - p).T)))
            d = abs(float(wrap(self._dir(C, i) - yaw)))
            if d < np.radians(60) and (best is None or d < best[0]):
                best = (d, lid, i)
        return best

    def width(self, p, yaw):
        """Width of the lane containing p within 60 deg of yaw: distance to its left edge + right edge (nan without one)."""
        o = self.own(p, yaw)
        if o is None:
            return np.nan
        ln, pt = self.L[o[1]], self.sh.Point(*p)
        return float(self.sh.LineString(ln["Le"]).distance(pt) + self.sh.LineString(ln["Re"]).distance(pt))

    def chain(self, p, yaw, route=None, length=70.0):
        """-> (centre, left edge, right edge) local polylines of the own lane from the car centre forward, or None."""
        o = self.own(p, yaw)
        if o is None:
            return None
        _, lid, i = o
        ln = self.L[lid]
        C, Le, Re = [ln["C"][max(i - 1, 0):]], [ln["Le"]], [ln["Re"]]
        tot, seen = float(np.hypot(*np.diff(C[0], axis=0).T).sum()), {lid}
        while tot < length:
            nx = [n for n in ln["nxt"] if n in self.L and n not in seen]
            if not nx:
                break
            end = self._dir(ln["C"], len(ln["C"]) - 2)

            def score(n):
                Cn = self.L[n]["C"]
                if route is not None:
                    q = Cn[: max(2, min(len(Cn), 10))]
                    dmin = np.min(np.hypot(q[:, None, 0] - route[None, :, 0], q[:, None, 1] - route[None, :, 1]), 1).mean()
                    if dmin < 1.0:
                        return dmin - 10.0
                return abs(float(wrap(self._dir(Cn, 0) - end)))
            lid = min(nx, key=score)
            seen.add(lid)
            ln = self.L[lid]
            C.append(ln["C"])
            Le.append(ln["Le"])
            Re.append(ln["Re"])
            tot += float(np.hypot(*np.diff(ln["C"], axis=0).T).sum())
        return np.vstack(C), np.vstack(Le), np.vstack(Re)


def lane_refs(M, a, route_l, x0=0.0, Hr=None):
    """Lane references of one decision at x = x0 + d (ego frame of pose a): c_map, lane width, road centre, the ego's offset from the own-lane
    centre at the rear axle, segment type (seg, seg3)."""
    p = np.asarray(a[:2], float) + CAR_C * np.array([np.cos(a[2]), np.sin(a[2])])
    ch = M.chain(p, a[2], route_l)
    o = dict(seg="none", seg3="none")
    if ch is None:
        return o
    C, Le, Re = ch
    Ce, Le_e, Re_e = in_ego(a, C), in_ego(a, Le), in_ego(a, Re)
    import ev as EV
    o["ego_off"] = EV.signed_off(C, np.asarray(a[:2], float))                   # left + offset of the rear axle from the own-lane centre
    J = M.junction(p)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
    hC = np.unwrap(np.arctan2(*np.diff(C, axis=0).T[::-1]))
    k0 = int(np.argmin(np.hypot(*(C - p).T)))
    s_p = S[k0]
    for dd in (10.0, 20.0):
        q = np.array([np.interp(s_p + dd, S, C[:, 0]), np.interp(s_p + dd, S, C[:, 1])])
        J = J or M.junction(q)
    j40 = int(np.searchsorted(S, s_p + 40.0))
    lane_dh = float(np.degrees(abs(hC[min(j40, len(hC) - 1)] - hC[min(k0, len(hC) - 1)]))) if len(hC) else np.nan
    o.update(junction=bool(J), lane_dh40=lane_dh, lane_len=float(S[-1] - s_p))
    for d in DS:
        x = x0 + d
        c = y_at(Ce, x)
        yl, yr = M.cross(Le_e[:-1], Le_e[1:], (0.0, 0.0, 0.0), x), M.cross(Re_e[:-1], Re_e[1:], (0.0, 0.0, 0.0), x)
        w = np.nan
        if np.isfinite(c) and len(yl) and len(yr):
            w = float(yl[np.argmin(np.abs(yl - c))] - yr[np.argmin(np.abs(yr - c))])
        R = M.cross(*M.SR, a, x)
        rc = np.nan
        if np.isfinite(c) and len(R):
            L_, R_ = R[R > c], R[R < c]
            if len(L_) and len(R_):
                rc = float((L_.min() + R_.max()) / 2)
        o[f"c{int(d)}"], o[f"w{int(d)}"], o[f"rc{int(d)}"] = c, w, rc
    straight = lane_dh < 10 and (Hr is None or Hr < 10)
    o["seg"] = "junction" if J else ("straight" if straight else "curved")
    cont = bool(np.isfinite(o["c30"]) and np.isfinite(o["w10"]) and np.isfinite(o["w30"]) and abs(o["w30"] - o["w10"]) < 0.5)
    o["seg3"] = "junction" if J else ("straight" if straight and cont else "irregular")
    return o


def rmem_y(rm, x):
    """Served route memory (200,) -> y of its valid points at x (rear-axle frame), nan outside."""
    if rm is None:
        return np.nan
    F = np.asarray(rm, float).reshape(40, 5)
    F = F[F[:, 0] > 0.5]
    if len(F) < 2:
        return np.nan
    return y_at(F[:, 1:3] * 20.0, x)


# ---------------------------------------------------------------- geo (CPU)
def bend_of(s, h, kap, s0, s_end):
    i0, i1 = int(np.searchsorted(s, s0)), int(np.searchsorted(s, s_end, side="right")) - 1
    i1 = max(i1, i0 + 1)
    w = int(round(WIN / (s[1] - s[0])))
    if i1 - i0 <= w:
        a, b = i0, i1
    else:
        dd = np.abs(h[i0 + w:i1 + 1] - h[i0:i1 + 1 - w])
        a = i0 + int(np.argmax(dd))
        b = a + w
    th = float(h[b] - h[a])
    j0, j1 = max(0, a - 10), min(len(s) - 1, b + 10)
    tot = float(h[j1] - h[j0])
    ent = ext = np.nan
    if abs(np.degrees(th)) >= BEND_MIN:                                  # a bend of >= 15 deg in 40 m; straighter stretches have none
        f = (h[j0:j1 + 1] - h[j0]) / tot
        k10, k90 = np.flatnonzero(f >= 0.1), np.flatnonzero(f >= 0.9)
        if len(k10) and len(k90):
            ent, ext = float(s[j0 + k10[0]]), float(s[j0 + k90[0]])
    seg = np.abs(kap[i0:i1 + 1])
    r_bend = np.nan
    if np.isfinite(ent):
        kb = np.abs(kap[(s >= ent) & (s <= ext)])
        r_bend = float(1.0 / max(kb.max(), 1e-3)) if len(kb) else np.nan
    return dict(bend_deg=float(np.degrees(th)), entry=ent, exit=ext, r_min=float(1.0 / max(seg.max(), 1e-3)), r_bend=r_bend,
                abs_deg=float(np.degrees(np.abs(np.diff(h[i0:i1 + 1])).sum())), net_deg=float(np.degrees(h[i1] - h[i0])))


def one(job):
    st_, label, D, scene, recs, summ = job
    import rf_follow as RF
    import ev as EV
    import zm_read as ZM
    from ao_e1 import loader, project, stations
    ev = steps(D, scene)
    kz = min([k for k, r in ev.items() if r["off"] or r["front"] or r["side"] or r["gt"] >= 4.0], default=10 ** 6)
    tz = T0 + DT * kz if kz < 10 ** 6 else np.inf
    zc = ""
    if kz < 10 ** 6:
        e = ev[kz]
        zc = "offroad" if e["off"] else ("contact" if e["front"] or e["side"] else "diverged")
    W, A, G0, Rl, _ = RF.regen_full(scene, recs, EV.CFG)
    Rl = Rl[np.r_[True, np.hypot(*np.diff(Rl, axis=0).T) > 1e-6]]
    S = stations(Rl)
    s, h = RF.heading(Rl, S)
    kap = np.gradient(h, s)
    ds = loader(str(EV.CFG)).get_data_source(scene)
    tr = ds.rig.trajectory
    Tt, P3 = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
    Pg = P3[:, :2]
    kend = KEND[st_]
    t_end = T0 + DT * kend
    log_end = np.array([np.interp(t_end, Tt, Pg[:, 0]), np.interp(t_end, Tt, Pg[:, 1])])
    h_at = lambda q: float(np.interp(q, s, h))                                              # noqa: E731
    r_at = lambda q: np.array([np.interp(q, S, Rl[:, 0]), np.interp(q, S, Rl[:, 1])])       # noqa: E731
    # decisions: stations along the route
    dec, hint = [], None
    for r in recs:
        a = np.asarray(r["anchor"], float)
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        stn, lat = project(Rl, S, a[:2], lo, hi)
        hint = stn
        dec.append((r, a, stn, lat))
    s0 = dec[0][2]
    s_end = project(Rl, S, log_end, s0 - 10.0, s0 + 150.0)[0]
    B = bend_of(s, h, kap, s0, s_end)
    sg = np.sign(B["bend_deg"]) or 1.0
    allp = np.vstack([Pg, np.array([d[1][:2] for d in dec])])
    M = LaneMap(ds, (allp.min(0) - 60, allp.max(0) + 60))
    LN = ZM.Lanes(ds.map)

    def lcls(x, y, yaw):                                                                    # zeros_mech lane class of the car centre
        c = np.array([x, y]) + CAR_C * np.array([np.cos(yaw), np.sin(yaw)])
        z = float(P3[int(np.argmin(np.hypot(*(Pg - c).T))), 2])
        return int(LN.cls(c[0], c[1], z, yaw)[0])
    if np.isfinite(B["entry"]):
        ws = [M.width(r_at(q), h_at(q)) for q in np.arange(B["entry"], B["exit"] + 1e-6, 2.0)]
        mid = (B["entry"] + B["exit"]) / 2
        B.update(w_mid=M.width(r_at(mid), h_at(mid)), w_min=float(np.nanmin(ws)) if np.isfinite(ws).any() else np.nan)
    # the human on the route (speed at station)
    hs, hint = [], None
    for p in Pg[::2]:
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 20.0, hint + 40.0)
        hint = project(Rl, S, p, lo, hi)[0]
        hs.append(hint)
    hs = np.maximum.accumulate(np.array(hs))
    hv = np.gradient(hs, Tt[::2] * 1e-6)
    vh_at = lambda q: float(np.interp(q, hs, hv)) if hs[0] <= q <= hs[-1] else np.nan       # noqa: E731
    # controller trace
    P = ctrl_path(D, scene)
    tst = np.full(0, np.nan)
    trace = []
    if P is not None:
        q, hint = [], None
        for x in P:
            lo, hi = (-np.inf, np.inf) if hint is None else (hint - 20.0, hint + 40.0)
            sn, la = project(Rl, S, x[1:3], lo, hi)
            hint = sn
            q.append((sn, la))
        q = np.array(q)
        tst, tlat = q[:, 0], q[:, 1]
    row = dict(set=st_, label=label, scene=scene, log=scene.rsplit("-", 1)[0], score=summ["score"], kz=kz if kz < 10 ** 6 else -1,
               offroad=float(summ.get("offroad") or 0) > 0, at_fault=float(summ.get("collision_at_fault") or 0) > 0,
               corridor=float(summ.get("left_corridor_laterally") or 0) > 0, z_cause=zc, s0=s0, s_end=s_end, sg=sg, **B,
               rear=any(r["rear"] for k, r in ev.items() if k <= min(kz, KEND[st_])),
               wrong_eval=any(r["wrong"] for k, r in ev.items() if k <= min(kz, KEND[st_])))
    row["testable"] = bool(np.isfinite(B["exit"]) and B["exit"] + 10.0 <= s_end)
    z_st = opp_t = np.inf
    if len(tst):
        before = P[:, 0] < tz
        iz = np.flatnonzero(P[:, 0] >= tz)
        z_st = float(tst[iz[0]]) if len(iz) else (float(tst[-1]) if kz < 10 ** 6 else np.inf)
        inb = before & (tst >= B["entry"] - 10.0) & (tst <= B["exit"] + 10.0) if np.isfinite(B["entry"]) else np.zeros(len(P), bool)
        row["max_lat"] = float(np.abs(tlat[inb]).max()) if inb.any() else np.nan
        row["max_out"] = float((-sg * tlat[inb]).max()) if inb.any() else np.nan
        row["max_in"] = float((sg * tlat[inb]).max()) if inb.any() else np.nan
        row["max_lat_all"] = float(np.abs(tlat[before & (tst <= s_end)]).max()) if (before & (tst <= s_end)).any() else np.nan
        if np.isfinite(B["exit"]):
            ix = np.flatnonzero(tst >= B["exit"] + 10.0)
            row["completed"] = bool(len(ix) and P[ix[0], 0] < tz and abs(tlat[ix[0]]) <= LAT_OK)
        for nm, q_ in (("v_entry", B["entry"] - 10.0), ("v_mid", (B["entry"] + B["exit"]) / 2)):
            j = np.flatnonzero(tst >= q_) if np.isfinite(q_) else []
            row[nm] = float(P[j[0], 4]) if len(j) and P[j[0], 0] < tz else np.nan
            row[nm + "_h"] = vh_at(q_) if np.isfinite(q_) else np.nan
        if np.isfinite(B["entry"]):
            win = before & (tst >= B["entry"] - 20.0) & (tst <= B["exit"] + 10.0)
            n_opp = 0
            for i in np.flatnonzero(win)[::2]:                                              # every 0.2 s
                if lcls(P[i, 1], P[i, 2], P[i, 3]) == 2:
                    n_opp += 1
                    opp_t = min(opp_t, P[i, 0])
            row.update(opp_n=n_opp, opp_share=n_opp / max(len(np.flatnonzero(win)[::2]), 1))
            for i in np.flatnonzero((tst >= B["entry"] - 30.0) & (tst <= B["exit"] + 20.0)):
                trace.append(dict(set=st_, label=label, scene=scene, t=(P[i, 0] - T0) * 1e-6, st=tst[i], lat=tlat[i], v=P[i, 4],
                                  herr=float(wrap(P[i, 3] - h_at(tst[i]))), before=bool(P[i, 0] < tz)))
    row["opp"] = bool(np.isfinite(opp_t))
    row["z_st"] = z_st if np.isfinite(z_st) else np.nan
    out = "nobend"
    if np.isfinite(B["entry"]):
        lo, hi = B["entry"] - 20.0, B["exit"] + 10.0
        evs = ([(tz, zc)] if kz < 10 ** 6 and lo <= z_st <= hi else []) + ([(opp_t, "opposite")] if np.isfinite(opp_t) else [])
        if evs:
            out = min(evs)[1]
        elif kz < 10 ** 6 and z_st < lo:
            out = "zero_before"
        elif row.get("completed"):
            out = "completed"
        else:
            out = "incomplete" if row["testable"] else "untestable"
    row["outcome"] = out
    # per decision
    rows = []
    for r, a, stn, lat in dec:
        k = int((r["t0"] - T0) // DT)
        e = np.asarray(r["ego"], float)
        v = float(np.hypot(e[4], e[5]) * 10.0)
        Pp = np.asarray(r["poses"], float)
        ar = arcs(Pp)
        hr = h_at(stn)
        herr = float(wrap(a[2] - hr))
        d = dict(set=st_, label=label, scene=scene, k=k, st=stn, lat=lat, v=v, herr=herr, before=k < kz,
                 inbend=bool(np.isfinite(B["entry"]) and B["entry"] - 20.0 <= stn <= B["exit"] + 10.0), sg=sg,
                 v_h=vh_at(stn), arc1=ar[1], arc2=ar[3], arc4=ar[7], deficit=ar[3] - 2.0 * v, kap_route=float(np.interp(stn, s, kap)),
                 lcls=lcls(a[0], a[1], a[2]))
        for n, i in (("1", 1), ("2", 3), ("4", 7)):
            d["yaw" + n] = float(Pp[i, 2])
            d["dhr" + n] = h_at(stn + ar[i]) - hr
            lp = to_local(a, Pp[i, :2])[0]
            d["plat" + n] = project(Rl, S, lp, stn - 5.0, stn + 90.0)[1]
        if P is not None:
            w_ = (P[:, 0] >= r["t0"]) & (P[:, 0] <= r["t0"] + 1e6)
            if w_.sum() >= 5:
                Q = P[w_]
                d["exec_dyaw1"] = float(Q[-1, 3] - Q[0, 3])
                d["exec_arc1"] = float(np.hypot(*np.diff(Q[:, 1:3], axis=0).T).sum())
                i_ = np.flatnonzero(w_)
                d["exec_dhr1"] = h_at(tst[i_[-1]]) - h_at(tst[i_[0]]) if len(tst) else np.nan
        # lane refs, route memory, plan offsets at 10 / 20 / 30 m from the rear axle
        Hr = float(np.degrees(abs(h_at(stn + 40.0) - hr)))
        d.update(lane_refs(M, a, Rl, 0.0, Hr))
        d["route_dh40"] = Hr
        rm = r.get("rmem")
        for x in DS:
            d[f"ry{int(x)}"] = rmem_y(rm, x)
            d[f"py{int(x)}"] = y_at(np.r_[[[0.0, 0.0]], Pp[:, :2]], x)
            d[f"gy{int(x)}"] = y_at(in_ego(a, Rl), x)                    # the rebuilt route (the human's lane centres) at x
        rows.append(d)
    bev = None
    hsh = scene.rsplit("-", 1)[1]
    if hsh in HELD7 or hsh[:8] in EXAMPLES:
        bb = (allp.min(0) - 25, allp.max(0) + 25)
        ins = lambda X: bool(((X[:, 0] >= bb[0][0]) & (X[:, 0] <= bb[1][0]) & (X[:, 1] >= bb[0][1]) & (X[:, 1] <= bb[1][1])).any())  # noqa: E731
        bev = dict(edges=[x for ln in M.L.values() for x in (ln["Le"], ln["Re"]) if ins(x)], road=[x for x in M.road if ins(x)], route=Rl,
                   human=Pg, ego=None if P is None else P[:, :3], tz=tz, plans=[to_local(a, np.asarray(r["poses"], float)[:, :2]) for r, a, _, _ in dec],
                   anchors=np.array([a for _, a, _, _ in dec]), bend=B)
    return row, rows, trace, bev


def geo(a):
    import pandas as pd
    import ev as EV  # noqa: F401
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held = splits.load("alpasim_nuplan/onpol-held")
    with Run("wait_cap", "explore_g_geo", config=vars(a)) as run:
        run.use_split(held)
        hm = set(held.members)
        jobs = []
        for st_, labels in (("held", HELD), ("pub", tuple(PUB))):
            for lab in labels:
                dirs = runs_of(st_, lab)
                summ, n = summaries(dirs), 0
                for D in dirs:
                    for sc, rr in sorted(drive_records(D).items()):
                        if sc not in summ or summ[sc][0] != D or (st_ == "held" and sc not in hm) or (a.limit and n >= a.limit):
                            continue
                        for r in rr:
                            r.pop("ms", None), r.pop("hist", None)
                        jobs.append((st_, lab, D, sc, rr, summ[sc][1]))
                        n += 1
                run.info(f"{st_} {lab}: {n} rollouts from {len(dirs)} dirs")
        res = par.pmap(one, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="rollouts")
        err = {jobs[i][1] + " " + jobs[i][3]: str(e)[-300:] for i, e in res.errors.items()}
        run.info(f"{len(err)} rollouts failed: {list(err.items())[:3]}")
        V = [v for v in res.values if v is not None]
        pd.DataFrame([v[0] for v in V]).to_parquet(out / "scenes.parquet", index=False)
        pd.DataFrame([d for v in V for d in v[1]]).to_parquet(out / "dec.parquet", index=False)
        pd.DataFrame([d for v in V for d in v[2]]).to_parquet(out / "trace.parquet", index=False)
        pickle.dump({(v[0]["set"], v[0]["label"], v[0]["scene"]): v[3] for v in V if v[3] is not None}, open(out / "bev.pkl", "wb"), protocol=4)
        (out / "geo_errors.json").write_text(json.dumps(err, indent=1))
        run.summary.update(rollouts=len(jobs), errors=len(err))


# ---------------------------------------------------------------- msgs (CPU) / replay (GPU) / olg, lanes (CPU)
def zero_runs():
    """-> [(group, run dir, scene)] of every zero rollout whose rollout.asl was kept (held: every HELD label; pub: every PUB label)."""
    o = []
    for st_, labels in (("held", HELD), ("pub", tuple(PUB))):
        for lab in labels:
            for sc, (D, m) in sorted(summaries(runs_of(st_, lab)).items()):
                if m["score"] == 0 and glob.glob(f"{D}/rollouts/{sc}/*/rollout.asl"):
                    o.append((f"{st_}_{lab}", D, sc))
    return o


def _msgs(job):
    import c1_extract as X
    return X.one_msgs(job)


def msgs(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    with Run("wait_cap", "explore_g_msgs", config=vars(a)) as run:
        Z = zero_runs()
        jobs = []
        for g, D, sc in Z:
            (Path(a.out) / "msgs" / g).mkdir(parents=True, exist_ok=True)
            jobs.append((D, sc, str(Path(a.out) / "msgs" / g)))
        res = par.pmap(_msgs, jobs, workers=min(32, n_cpus()), run=run, desc="msgs", skip=lambda j: Path(j[2], f"{j[1]}.pkl").exists())
        cnt = {}
        for g, _, _ in Z:
            cnt[g] = cnt.get(g, 0) + 1
        run.info(f"{len(jobs)} zero rollouts with asl: {cnt}; {len(res.errors)} errors")
        run.summary.update(groups=cnt, errors=len(res.errors))


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def replay(a):
    import tempfile
    import torch
    import pp_train as T
    import rmem as RM
    import sh30_driver as D
    from ao_e1 import drive_records
    from jevdrive.run import Run
    HEADS = ("lane_lines", "lane_lines_prob", "road_edges", "lead", "lead_prob")
    od = Path(a.out) / "replay"
    od.mkdir(parents=True, exist_ok=True)
    groups = sorted(p.name for p in (Path(a.out) / "msgs").iterdir() if p.is_dir())
    D.RMEM = RM                                  # what sh30_driver.main() sets: without it a Session has no route tracker and feeds zeros
    with Run("wait_cap", "explore_g_replay", config=vars(a)) as run:
        p0 = T.load_pmodel("P0", torch.device("cuda"))
        pb, ctx = D.egodriver_pb2, Ctx()
        for ck in a.ckpts or HELD:
            todo = [g for g in groups if ck in ckset(g.split("_", 1)[1]) and (a.force or not (od / f"{g}__{ck}.pkl").exists())]
            if not todo:
                continue
            core = D.C.Core(ck, "cuda", "backwarp", 1.0, "gpu")
            assert core.rm, f"{ck}: not a ROUTE-MEM checkpoint"
            RM.SERVE_ORDER = bool(getattr(core, "rm_order", False))
            run.info(f"{ck}: route memory layout {'ordered' if RM.SERVE_ORDER else '-6..+72'}, groups {todo}")
            sl, cap, last = core.model.net.slices, {}, {}
            core.model.register_forward_hook(lambda m, args, out: cap.update(H=args[0], ego=args[1], tc=args[2], out=out))
            orig = core.plan

            def plan(*x, _o=orig, **k):
                last["o"] = _o(*x, **k)
                return last["o"]
            core.plan = plan
            fi = Path(a.out) / f"replay_identity_{ck}.json"
            ident = json.loads(fi.read_text()) if fi.exists() else {}
            for g in todo:
                st_, lab = g.split("_", 1)
                logd = Path(tempfile.mkdtemp())
                drv = D.Driver(core, logd, 0, False)
                md = Path(a.out) / "msgs" / g
                stored = {}
                if ck == lab:
                    for Dd in runs_of(st_, lab):
                        stored.update(drive_records(Dd))
                out = {}
                for scene in sorted(p.stem for p in md.glob("*.pkl"))[: a.limit or None]:
                    rec, uuid = [], None
                    for kind, raw in pickle.load(open(md / f"{scene}.pkl", "rb")):
                        if kind == "driver_session_request":
                            req = pb.DriveSessionRequest.FromString(raw)
                            uuid = req.session_uuid
                            drv.start_session(req, ctx)
                        elif kind == "driver_camera_image":
                            drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                        elif kind == "driver_ego_trajectory":
                            drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                        elif kind == "route_request":
                            drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                        elif kind == "driver_request":
                            req = pb.DriveRequest.FromString(raw)
                            try:
                                drv.drive(req, ctx)
                            except RuntimeError as e:
                                run.info(f"{g} {ck} {scene} t {req.time_now_us}: {e}")
                                continue
                            with torch.no_grad():
                                o0 = p0(cap["H"], cap["ego"][:, :20], cap["tc"]).float()[0]
                            ft = cap["out"][0] if isinstance(cap["out"], tuple) else cap["out"]
                            ft = ft.float()[0]
                            cam = np.asarray(drv.sessions[uuid].cam["t"], np.float64)
                            r = dict(now=int(req.time_now_us), poses=np.asarray(last["o"]["poses"], np.float32).copy(), cam=cam.copy(),
                                     rmem=None if "rmem" not in last["o"] else np.asarray(last["o"]["rmem"], np.float32).copy())
                            for name, v in (("FT", ft), ("P0", o0)):
                                r[name] = {k: v[sl[k]].cpu().numpy().astype(np.float32) for k in HEADS}
                            rec.append(r)
                    drv.sessions.pop(uuid, None)
                    out[scene] = rec
                drv.out.flush()
                drives = [json.loads(x) for x in open(logd / "drive.jsonl") if '"kind": "drive"' in x]
                flat = [r for sc in sorted(out) for r in out[sc]]
                assert len(drives) == len(flat), f"{g} {ck}: {len(drives)} records for {len(flat)} decisions"
                for r, d in zip(flat, drives):
                    r.update(anchor=np.asarray(d["anchor"], float), k=int(d["k"]), t0=int(d["t0"]))
                if stored:
                    dd, dm = [], []
                    for sc, rr in out.items():
                        by_k = {int(x["k"]): x for x in stored.get(sc, [])}
                        for r in rr:
                            x = by_k.get(r["k"])
                            if x is None:
                                continue
                            dd.append(np.hypot(*(np.asarray(x["poses"], float)[[1, 3, 7], :2] - r["poses"][[1, 3, 7], :2]).T))
                            if x.get("rmem") is not None and r["rmem"] is not None:
                                dm.append(float(np.abs(np.asarray(x["rmem"], float) - r["rmem"]).max()))
                    dd, dm = np.array(dd).reshape(-1, 3), np.array(dm)
                    ident[g] = dict(n=int(len(dd)), median_m={h: float(np.median(dd[:, i])) for i, h in enumerate(("1s", "2s", "4s"))} if len(dd) else None,
                                    p95_4s=float(np.percentile(dd[:, 2], 95)) if len(dd) else None,
                                    share_4s_lt_5cm=float((dd[:, 2] < 0.05).mean()) if len(dd) else None,
                                    rmem_maxdiff_p99=float(np.percentile(dm, 99)) if len(dm) else None, rmem_n=int(len(dm)))
                    run.info(f"identity {g} {ck}: {ident[g]}")
                pickle.dump(out, open(od / f"{g}__{ck}.pkl", "wb"), protocol=4)
                run.info(f"replay {g} with {ck}: {len(out)} scenes, {len(flat)} decisions")
            fi.write_text(json.dumps(ident, indent=1))
            del core
            torch.cuda.empty_cache()
        run.summary.update(ckpts=list(a.ckpts or HELD))


def olg_unit(job):
    """One replayed scene: every checkpoint's plan (and the stored one) against the rebuilt route of each decision (geo's rules)."""
    g, scene, recs, R, bend = job
    import rf_follow as RF
    import ev as EV
    from ao_e1 import project, stations
    W, A, G0, Rl, _ = RF.regen_full(scene, recs, EV.CFG)
    Rl = Rl[np.r_[True, np.hypot(*np.diff(Rl, axis=0).T) > 1e-6]]
    S = stations(Rl)
    s, h = RF.heading(Rl, S)
    h_at = lambda q: float(np.interp(q, s, h))                                              # noqa: E731
    plans = dict(stored={int(x["k"]): np.asarray(x["poses"], float) for x in recs}, **R)
    rows, hint = [], None
    for r in recs:
        k, a = int(r["k"]), np.asarray(r["anchor"], float)
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        stn, lat = project(Rl, S, a[:2], lo, hi)
        hint = stn
        e = np.asarray(r["ego"], float)
        base = dict(group=g, scene=scene, k=k, st=stn, lat=lat, v=float(np.hypot(e[4], e[5]) * 10.0), herr=float(wrap(a[2] - h_at(stn))), **bend)
        for ck, Pk in plans.items():
            if k not in Pk:
                continue
            P = Pk[k]
            ar = arcs(P)
            d = dict(base, ckpt=ck)
            for n, i in (("1", 1), ("2", 3), ("4", 7)):
                d["yaw" + n], d["arc" + n], d["dhr" + n] = float(P[i, 2]), float(ar[i]), h_at(stn + ar[i]) - h_at(stn)
                d["plat" + n] = project(Rl, S, to_local(a, P[i, :2])[0], stn - 5.0, stn + 90.0)[1]
            X = np.r_[[[0.0, 0.0, 0.0]], P[:, :3]]                                       # the path through the poses, by arc length
            ax = np.r_[0.0, np.cumsum(np.hypot(*np.diff(X[:, :2], axis=0).T))]
            for q in (5, 10, 15, 20):                                                     # path geometry at a fixed arc (speed-free)
                d[f"pa{q}"] = d[f"ha{q}"] = np.nan
                if q <= ax[-1]:
                    p = np.array([np.interp(q, ax, X[:, j]) for j in range(3)])
                    sq, d[f"pa{q}"] = project(Rl, S, to_local(a, p[:2])[0], stn - 5.0, stn + 60.0)
                    d[f"ha{q}"] = float(wrap(a[2] + p[2] - h_at(sq)))
            rows.append(d)
    return rows


def olg(a):
    import pandas as pd
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    out = Path(a.out)
    Sc = pd.read_parquet(out / "scenes.parquet").set_index(["set", "label", "scene"])
    with Run("wait_cap", "explore_g_olg", config=vars(a)) as run:
        jobs = []
        for g in sorted(p.name for p in (out / "msgs").iterdir() if p.is_dir()):
            st_, lab = g.split("_", 1)
            stored = {}
            for Dd in runs_of(st_, lab):
                stored.update(drive_records(Dd))
            R = {ck: pickle.load(open(out / "replay" / f"{g}__{ck}.pkl", "rb")) for ck in ckset(lab)}
            for sc in sorted(R[lab]):
                if (st_, lab, sc) not in Sc.index or sc not in stored:
                    continue
                b = Sc.loc[(st_, lab, sc)]
                bend = dict(set=st_, inputs=lab, sg=float(b.sg), entry=float(b.entry), exit=float(b.exit), kz=int(b.kz), bend_deg=float(b.bend_deg),
                            r_bend=float(b.r_bend), outcome=str(b.outcome))
                rr = [dict(x) for x in stored[sc]]
                for x in rr:
                    x.pop("ms", None), x.pop("hist", None)
                jobs.append((g, sc, rr, {ck: {int(x["k"]): np.asarray(x["poses"], float) for x in R[ck].get(sc, [])} for ck in R}, bend))
        res = par.pmap(olg_unit, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="scenes")
        run.info(f"{len(jobs)} scenes, {len(res.errors)} errors: {[str(e)[-200:] for e in list(res.errors.values())[:3]]}")
        pd.DataFrame([x for v in res.values if v for x in v]).to_parquet(out / "replay.parquet", index=False)
        run.summary.update(scenes=len(jobs), errors=len(res.errors))


def lanes_unit(job):
    g, ck, scene, recs = job
    import ev as EV
    from ao_e1 import loader
    ds = loader(str(EV.CFG)).get_data_source(scene)
    tr = ds.rig.trajectory
    Pg = np.asarray(tr.positions, float)[:, :2]
    A = np.array([r["anchor"] for r in recs])
    allp = np.vstack([Pg, A[:, :2]])
    M = LaneMap(ds, (allp.min(0) - 60, allp.max(0) + 60))
    LP = np.vstack([Pg, Pg[-1] + 60.0 * (Pg[-1] - Pg[-2]) / max(np.hypot(*(Pg[-1] - Pg[-2])), 1e-6)])
    rows = []
    for r in recs:
        a, cam = np.asarray(r["anchor"], float), np.asarray(r["cam"], float)
        # route heading change over 40 m ahead from the logged path (the human's lanes): the "Hr" of the segment rule
        Le = in_ego(a, LP)
        ahead = Le[(Le[:, 0] > 0) & (Le[:, 0] < 60)]
        Hr = float(np.degrees(abs(np.arctan2(*(ahead[-1] - ahead[0])[::-1])))) if len(ahead) > 2 else None
        o = dict(group=g, ckpt=ck, scene=scene, k=int(r["k"]), cam_x=cam[0], cam_y=cam[1], **lane_refs(M, a, LP, cam[0], Hr))
        P = np.asarray(r["poses"], float)
        for H in ("P0", "FT"):
            h = EV.heads(r[H], cam)
            o[f"{H}_p1"], o[f"{H}_p2"] = float(h["pr"][1]), float(h["pr"][2])
            for d in DS:
                y1, y2 = float(EV.at(h["ll"][1], cam[0] + d, cam[0])), float(EV.at(h["ll"][2], cam[0] + d, cam[0]))
                c = (y1 + y2) / 2
                o[f"{H}_c{int(d)}"], o[f"{H}_w{int(d)}"] = c, y1 - y2
                o[f"{H}_cs{int(d)}"], o[f"{H}_ws{int(d)}"] = cam[1] + (c - cam[1]) / SCALE, (y1 - y2) / SCALE
                r0, r1 = float(EV.at(h["re"][0], cam[0] + d, cam[0])), float(EV.at(h["re"][1], cam[0] + d, cam[0]))
                o[f"{H}_rc{int(d)}"] = (r0 + r1) / 2
        for d in DS:
            o[f"py{int(d)}"] = y_at(np.r_[[[0.0, 0.0]], P[:, :2]], cam[0] + d)
            o[f"ry{int(d)}"] = rmem_y(r.get("rmem"), cam[0] + d)
        rows.append(o)
    return rows


def lanes(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    with Run("wait_cap", "explore_g_lanes", config=vars(a)) as run:
        jobs = []
        for f in sorted((Path(a.out) / "replay").glob("*.pkl")):
            g, ck = f.stem.split("__")
            if g.split("_", 1)[1] != ck:
                continue
            for sc, rr in pickle.load(open(f, "rb")).items():
                if rr:
                    jobs.append((g, ck, sc, rr))
        ctrl = DATA / "runs/eyes_vs_head/replay/ctrl.pkl"
        for sc, rr in pickle.load(open(ctrl, "rb")).items():
            if rr:
                jobs.append(("ctrl", "P2H10-F-s0", sc, [dict(r, rmem=None) for r in rr]))
        res = par.pmap(lanes_unit, jobs, run=run, workers=min(64, n_cpus(), len(jobs)), desc="rollouts")
        run.info(f"{len(jobs)} rollouts, {len(res.errors)} errors: {[str(e)[-200:] for e in list(res.errors.values())[:3]]}")
        pd.DataFrame([x for v in res.values if v for x in v]).to_parquet(Path(a.out) / "lanes.parquet", index=False)
        run.summary.update(rollouts=len(jobs), errors=len(res.errors))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("geo", "msgs", "replay", "olg", "lanes", "report"):
        p = sp.add_parser(n)
        p.add_argument("--out", default=str(NX / "explore_g")), p.add_argument("--limit", type=int, default=0)
        p.add_argument("--workers", type=int, default=0), p.add_argument("--force", action="store_true"), p.add_argument("--res", default="")
        p.add_argument("--ckpts", nargs="*", default=None), p.add_argument("--dev", default="")
    a = ap.parse_args()
    if a.cmd == "report":
        import nx_turns_report as RP
        RP.main(a)
    else:
        globals()[a.cmd](a)
