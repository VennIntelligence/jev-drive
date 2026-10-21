#!/usr/bin/env python3
"""ROUND6 line A, DAGGER-MAP (plans/2026-10-21-dagger-map.md): DAgger rows for the LATERAL part of the plan. The student (R5LC-L18-s0, MV-LC)
is rolled under the submission serve on the 719 non-held curated train scenes; at every own state a privileged map teacher gives the path back
onto the route lane, and the label puts that path's points at the arc lengths of the student's own served plan (only the shape changes).
dag.sh runs the chain; this file holds the teacher, the gates and the tables.

Teacher (AlpaSim venv: the runtime's scene loader and route generator):
  route   AlpaSim's MAP route generator (alpasim_runtime.route_generator.RouteGeneratorMap, route_start_offset 40 m, as the rollouts) rebuilt
          on the scene's recorded rig trajectory = the polyline the driver's route messages are cut from (probe: logged waypoints within 5 cm
          of it). Lanes: for every recorded pose the lanes within 4 m and pi / 8 of its heading; the previous lane is kept while it stays a
          candidate, else its unique successor among the candidates, else the candidate common to the most upcoming poses (the generator's
          own rule): through a junction the route takes the connector lane the human drove. The poses are projected onto their lane's centre
          line. Where the route switches to a lane that is not a successor (a lateral switch), the step is blended over clip(3 s x human
          speed, 15, 60) m between the two lane centre lines (quintic smoothstep). Past the log the generator's own lane following extends it
          80 m. Resampled every 0.25 m, Gaussian sigma 1 m.
  path    at a student pose: the nearest reference point whose heading is within 90 deg -> station s0, signed lateral offset d0 (left +),
          heading error psi0. Frenet quintic d(u) from (d0, tan(psi0) (1 - kappa d0), 0) to (0, 0, 0) over L = clip(2.0 s x v0, 6, 80) m, then
          along the reference. No label when |d0| > 5 m, |psi0| > 60 deg or no heading-consistent point.
  label   the 8 points at the arc lengths of the student's served plan at 0.5 .. 4 s (from the decision's own origin), heading = the path
          tangent, in the rig frame at the decision anchor.

  dag.py lists --root O                        chunk lists O/lists/c000.txt (6 scenes, every class) + c001.. (25 a chunk, classes interleaved)
  dag.py t1 --cfg-glob G [--limit N]           AlpaSim venv: T1, the teacher on HUMAN states (log, every 0.5 s, v >= 2 m/s, 4 s future) with the
                                               human's own arc lengths -> lateral error at 2 / 4 s -> results/dagger/t1.{md,json}
  dag.py t2                                    AlpaSim venv: T2, the MV-family PAI 281 offroad zeros: the teacher at the decision 2.5 s before the
                                               event, its footprint on the road_pai raster -> results/dagger/t2.{md,json,csv}
  dag.py teach --rows DIR --cfg-glob G --out DIR [--chunk-list F]   AlpaSim venv: teacher labels for every decision of the row files
  dag.py chk1 --root O                         stage-1 check on c000 (scored, rows, teacher validity, projected card-hours); exit 1 = stop
  dag.py label --root O --pool on1_PAID        the pool's tab.fut := teacher label on drawable rows (v0 >= 2 m/s, teacher ok), served plan
                                               elsewhere (weight 0); on1_PAID/dagger.npz; T4 + row tables -> results/dagger/rows.{md,json}
  dag.py r1in --root O --pool on1_PAID --fwd D --tag T --control W   the trainer forward of T on the pool's tokens vs the logged model plan
  dag.py mfix --pool on1_JTPHD --ref on1_JTPH  the zero-copy merge's prefix equals the reference (s3.py mfix, own check dir)
  dag.py weights --pool on1_JTPHD --base R5LC --k-on 64 --k-dag 8 --out R6DAG   draw file (k-dag 0: R6D0, the identity)
  dag.py smoke --tag X --wfile R6DAG --k-dag 8 the 60-step smoke's checks -> results/dagger/smoke.json
  dag.py verdict --json J --md M --read K=F ..    the registered rule on r2.py r281's json, appended to M
  dag.py pcs --pcs Q --json J --md M            the local PCS ruler rows and the ruler rule, appended to M
  dag.py report --root O                        rows, disk and card-hours -> results/dagger/build.{md,json}
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/op_probe/scripts"),
                str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/round3/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O_DEF = DATA / "runs/round6/dag"
CR = DATA / "runs/op_parity/cache"
W5, W6 = DATA / "runs/round5/r5/w", DATA / "runs/round6/dag/w"
RES = REPO / "experiments/round6/results/dagger"
TSV = REPO / "experiments/round5/results/s3_scenes.tsv"
CLS = ("stop_nolead", "turn_sharp", "launch", "other")
STUDENT = "R5LC-L18-s0"
CARD_STACKS = 3

# teacher (registered in the plan before any roll)
T_MERGE, L_MIN, L_MAX = 2.0, 6.0, 80.0          # merge length = clip(T_MERGE x v0, L_MIN, L_MAX) m
D_MAX, PSI_MAX = 5.0, np.radians(60.0)          # no label beyond these
V_ROW = 2.0                                     # DAgger rows: fed speed >= 2 m/s
SW_T, SW_MIN, SW_MAX = 3.0, 15.0, 60.0          # lateral lane switch in the route: blend length = clip(3 s x human speed, 15, 60) m
DS, SIG = 0.25, 1.0                             # reference resampling and Gaussian smoothing (m)
ROUTE_OFFSET = 40.0                             # the rollouts' route_start_offset_m
T_J = 0.5 * np.arange(1, 9)                     # label times (s)
A_LAT = 4.0                                     # T4 line (m/s^2, 99 %)
TURN_DEG, LC_V, LC_Y, LC_YAW = 30.0, 5.0, 2.5, 12.0
T2_LEAD = 2.5
H_SIG, H_CLIP = 10.0, 0.5                       # amendment 1 (b): human lateral offset from the lane centre, Gaussian sigma 10 m along the station,
                                                # |offset| > 0.5 m -> 0 (lane change / transient: plain lane centre), the mask smoothed the same way
TEACHER_V = "b1"                                # teacher version stamped into every teach file (label refuses other versions)


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))


def read_tsv(f):
    return list(csv.DictReader(open(f), delimiter="\t"))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def smooth5(t):
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


# ---------------------------------------------------------------- reference line and teacher path (numpy only)
class Ref:
    """Resampled, smoothed route polyline: s (m,), x, y, th (unwrapped), kap; s_rec = station of the last recorded-log point."""

    def __init__(self, xy, s_rec=np.inf):
        from scipy.ndimage import gaussian_filter1d
        xy = np.asarray(xy, np.float64)[:, :2]
        keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-6]
        xy = xy[keep]
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        self.s = np.arange(0.0, s[-1] + 1e-9, DS)
        x, y = np.interp(self.s, s, xy[:, 0]), np.interp(self.s, s, xy[:, 1])
        sg = SIG / DS
        self.x, self.y = gaussian_filter1d(x, sg, mode="nearest"), gaussian_filter1d(y, sg, mode="nearest")
        self.th = np.unwrap(np.arctan2(np.gradient(self.y), np.gradient(self.x)))
        self.kap = gaussian_filter1d(np.gradient(self.th) / DS, 2.0, mode="nearest")
        self.s_rec = float(np.interp(s_rec, s, s)) if np.isfinite(s_rec) else float(self.s[-1])

    def at(self, s):
        return np.interp(s, self.s, self.x), np.interp(s, self.s, self.y), np.interp(s, self.s, self.th)

    def project(self, pose):
        """pose (x, y, yaw) -> (s0, d0, psi0) or None (no point within 90 deg of the heading)."""
        d2 = (self.x - pose[0]) ** 2 + (self.y - pose[1]) ** 2
        d2 = np.where(np.abs(wrap(self.th - pose[2])) < np.pi / 2, d2, np.inf)
        i = int(np.argmin(d2))
        if not np.isfinite(d2[i]):
            return None
        c, sn = np.cos(self.th[i]), np.sin(self.th[i])
        dx, dy = pose[0] - self.x[i], pose[1] - self.y[i]
        s0 = self.s[i] + c * dx + sn * dy
        d0 = c * dy - sn * dx
        return float(s0), float(d0), float(wrap(pose[2] - np.interp(s0, self.s, self.th)))


def teacher_path(ref, pose, v0, amax):
    """-> (arc (m,), X, Y, TH, kp, info) of the teacher path from the pose, at least amax long (linear past the reference's end), or None."""
    pr = ref.project(pose)
    if pr is None:
        return None, dict(why="no_proj")
    s0, d0, psi0 = pr
    info = dict(s0=s0, d0=d0, psi0=psi0, why="")
    if abs(d0) > D_MAX:
        info["why"] = "far"
    elif abs(psi0) > PSI_MAX:
        info["why"] = "heading"
    if info["why"]:
        return None, info
    L = float(np.clip(T_MERGE * v0, L_MIN, L_MAX))
    k0 = float(np.interp(s0, ref.s, ref.kap))
    m0 = np.tan(psi0) * (1 - k0 * d0)
    u = np.arange(0.0, max(amax, L) + 12.0, DS)
    tau = np.clip(u / L, 0, 1)
    d = d0 * (1 - 10 * tau ** 3 + 15 * tau ** 4 - 6 * tau ** 5) + m0 * L * (tau - 6 * tau ** 3 + 8 * tau ** 4 - 3 * tau ** 5)
    su = s0 + u
    past = su > ref.s[-1]
    xr, yr, tr = ref.at(np.minimum(su, ref.s[-1]))
    if past.any():                                                              # past the reference: straight along its last heading
        e = su[past] - ref.s[-1]
        xr[past] += e * np.cos(ref.th[-1])
        yr[past] += e * np.sin(ref.th[-1])
    X, Y = xr - d * np.sin(tr), yr + d * np.cos(tr)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]
    TH = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
    from scipy.ndimage import gaussian_filter1d
    kp = gaussian_filter1d(np.gradient(TH) / np.maximum(np.gradient(arc), 1e-6), 4.0, mode="nearest")
    kr = np.interp(np.minimum(su, ref.s[-1]), ref.s, ref.kap)
    info |= dict(L=L, m0=float(m0), past_ref=bool(past[np.searchsorted(arc, amax).clip(0, len(arc) - 1)]),
                 past_rec=bool(su[np.searchsorted(arc, amax).clip(0, len(arc) - 1)] > ref.s_rec))
    return (arc, X, Y, TH, kp, kr), info


def arcs_of(fut):
    """fut (8, >= 2) poses from the origin -> (8,) cumulative path length."""
    P = np.concatenate([np.zeros((1, 2)), np.asarray(fut, np.float64)[:, :2]])
    return np.cumsum(np.hypot(*np.diff(P, axis=0).T))


def place(path, a, anchor):
    """path from teacher_path, arc lengths a (8,), anchor (3,) -> (fut (8, 3) rig frame, world points (8, 2), headings (8,))."""
    arc, X, Y, TH = path[:4]
    px, py, ph = np.interp(a, arc, X), np.interp(a, arc, Y), np.interp(a, arc, TH)
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    dx, dy = px - anchor[0], py - anchor[1]
    fut = np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(ph - anchor[2])], -1)
    return fut, np.c_[px, py], ph


def lat_acc(path, a, v0):
    """Max lateral acceleration over the label: path curvature x the served plan's speed at that arc (m/s^2) and the route-only part."""
    arc, kp, kr = path[0], path[4], path[5]
    v = np.r_[v0, np.gradient(np.r_[0, a], 0.5)[1:]]
    m = arc <= a[-1]
    vv = np.interp(arc[m], np.r_[0, a], v)
    return float(np.max(vv ** 2 * np.abs(kp[m]))), float(np.max(vv ** 2 * np.abs(kr[m])))


# ---------------------------------------------------------------- route of a scene (AlpaSim venv)
def _proj_ext(C, p):
    """p (2,) onto the polyline C (m, 2), its first / last segments extended without end -> nearest point (2,)."""
    a, b = C[:-1], C[1:]
    ab = b - a
    t = ((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12)
    lo, hi = np.zeros(len(t)), np.ones(len(t))
    lo[0], hi[-1] = -np.inf, np.inf
    q = a + np.clip(t, lo, hi)[:, None] * ab
    return q[np.argmin(((q - p) ** 2).sum(1))]


def traj_of(ds):
    tr = ds.rig.trajectory
    P = np.asarray(tr.positions, np.float64)
    yaw = np.unwrap(np.asarray(tr.yaws, np.float64)) if hasattr(tr, "yaws") else None
    ts = np.asarray(tr.timestamps_us, np.int64) if hasattr(tr, "timestamps_us") else None
    return P, yaw, ts


def route_ref(ds):
    """The scene's route as the rollouts' driver received it (RouteGeneratorMap on the recorded rig trajectory) -> (Ref, info)."""
    from alpasim_runtime.route_generator import RouteGeneratorMap
    P, yaw, ts = traj_of(ds)
    vm = ds.map
    g = RouteGeneratorMap(P, vm, route_start_offset_m=ROUTE_OFFSET)
    R0 = np.asarray(g.route_polyline_in_local.points, np.float64)
    g._ensure_sufficient_waypoints(R0[-1])
    R1 = np.asarray(g.route_polyline_in_local.points, np.float64)
    ids, hd = [], 0.0                                                           # the generator's lane per recorded pose (its own code path)
    for i in range(len(P)):
        if i < len(P) - 1:
            dd = (P[i + 1] - P[i])[:2]
            if np.linalg.norm(dd) > 1e-2:
                hd = float(np.arctan2(dd[1], dd[0]))
        ids.append([ln.id for ln in vm.get_current_lane(np.r_[P[i], hd], max_dist=4.0)] or [vm.get_closest_lane(P[i]).id])
    best = g._determine_best_lane_sequence(ids)
    C = {i: np.asarray(vm.get_road_lane(i).center.points, np.float64)[:, :2] for i in set(best)}
    W = np.array([np.asarray(vm.get_road_lane(best[i]).center.project_onto(P[i].reshape(1, 3)))[0, :2] for i in range(len(P))])
    dmax = float(np.sqrt(((R0[:, None, :2] - W[None]) ** 2).sum(-1)).min(1).max())   # the generator's points are a downsample of W
    sP = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    tsec = ts * 1e-6 if ts is not None and len(ts) == len(P) else np.arange(len(P)) * 0.1
    v = np.gradient(sP, tsec)
    sw = [i for i in range(1, len(best)) if best[i] != best[i - 1] and best[i] not in vm.get_road_lane(best[i - 1]).next_lanes]
    Wb, steps = W.copy(), []
    for i in sw:
        A, B = C[best[i - 1]], C[best[i]]
        steps.append(float(np.hypot(*(_proj_ext(A, P[i, :2]) - _proj_ext(B, P[i, :2])))))
        L = float(np.clip(SW_T * v[i], SW_MIN, SW_MAX))
        for j in np.flatnonzero(np.abs(sP - sP[i]) <= L / 2):
            w = smooth5((sP[j] - (sP[i] - L / 2)) / L)
            frm, to = (Wb[j], _proj_ext(B, P[j, :2])) if j < i else (_proj_ext(A, P[j, :2]), Wb[j])
            Wb[j] = (1 - w) * frm + w * to
    xy = np.concatenate([Wb, R1[len(R0):, :2]])
    s_rec = float(np.sum(np.hypot(*np.diff(Wb, axis=0).T)))
    ref = Ref(xy, s_rec)
    refh, hinfo = human_ref(ref, P, yaw)
    return (ref, refh), dict(n_traj=len(P), n_route=len(R0), n_ext=len(R1) - len(R0), lanes=len(set(best)), switches=len(sw), steps=steps,
                             gen_vs_lanes_max=dmax, **hinfo)


def human_ref(ref, P, yaw):
    """Amendment 1 (b): the lane-centre reference shifted by the human's own lateral offset from it at the matched station: the log poses projected
    onto ref, offset interpolated on the station grid inside the recorded span (0 outside), Gaussian H_SIG; where |offset| > H_CLIP the offset is 0
    (lane change / transient -> plain lane centre) and that mask is smoothed the same way. -> (Ref, info with the clip share)."""
    from scipy.ndimage import gaussian_filter1d
    if yaw is None:
        yaw = np.unwrap(np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])))
    pr = [ref.project(np.r_[P[i, :2], yaw[i]]) for i in range(len(P))]
    sd = np.array([(p[0], p[1]) for p in pr if p is not None]).reshape(-1, 2)
    if len(sd) < 2:                                                             # no heading-consistent log pose (e.g. a parked log): lane centre
        rh = Ref(np.c_[ref.x, ref.y], ref.s_rec)
        rh.b = np.zeros(len(ref.s))
        return rh, dict(h_offset_med=float("nan"), h_clip_share=0.0, h_offset_abs_med_used=0.0, h_no_projection=True)
    o = np.argsort(sd[:, 0])
    s_, d_ = sd[o, 0], sd[o, 1]
    inside = (ref.s >= s_[0]) & (ref.s <= s_[-1])
    b = np.where(inside, np.interp(ref.s, s_, d_), 0.0)
    sg = H_SIG / DS
    b = gaussian_filter1d(b, sg, mode="nearest")
    keep = (np.abs(b) <= H_CLIP) & inside
    b = gaussian_filter1d(np.where(keep, b, 0.0), sg, mode="nearest")
    X, Y = ref.x - b * np.sin(ref.th), ref.y + b * np.cos(ref.th)
    rh = Ref(np.c_[X, Y], ref.s_rec)
    rh.b = b
    return rh, dict(h_offset_med=float(np.median(d_)), h_clip_share=float(1 - keep[inside].mean()) if inside.any() else 0.0,
                    h_offset_abs_med_used=float(np.median(np.abs(b[inside]))) if inside.any() else 0.0)


def scene_cfgs(patterns):
    """scene id -> the first generated AlpaSim config (comma-separated globs) that lists it."""
    out = {}
    for f in [f for p in patterns.split(",") for f in sorted(glob.glob(p))]:
        for s in re.findall(r"scene_id: (clipgt-[0-9a-f-]+)", open(f).read()):
            out.setdefault(s, f)
    return out


def load_ds(scene, cfg):
    import rf_tax as RT
    return RT.loader(cfg).get_data_source(scene)


# ---------------------------------------------------------------- lists
def cmd_lists(a):
    S = read_tsv(TSV)
    by = defaultdict(list)
    for r in S:
        by[r["cls"]].append(r["scene_id"])
    first = [by[c].pop(0) for c in CLS] + [by["turn_sharp"].pop(0), by["other"].pop(0)]
    inter = []
    for i in range(max(map(len, by.values()))):
        inter += [by[c][i] for c in CLS if i < len(by[c])]
    L = Path(a.root) / "lists"
    L.mkdir(parents=True, exist_ok=True)
    (L / "c000.txt").write_text("\n".join(first) + "\n")
    for i in range(0, len(inter), 25):
        (L / f"c{i // 25 + 1:03d}.txt").write_text("\n".join(inter[i:i + 25]) + "\n")
    print(len(first) + len(inter), "scenes in", 1 + (len(inter) + 24) // 25, "chunks")


# ---------------------------------------------------------------- T1: the teacher on human states
def _t1_one(job):
    scene, cfg = job
    ds = load_ds(scene, cfg)
    (ref, refh), rinfo = route_ref(ds)
    P, yaw, ts = traj_of(ds)
    t = ts * 1e-6
    out = []
    for i in range(0, len(P), 5):
        if t[i] + 4.0 > t[-1]:
            break
        tq = t[i] + T_J
        H = np.c_[np.interp(tq, t, P[:, 0]), np.interp(tq, t, P[:, 1])]
        hy = np.interp(tq, t, yaw)
        v = float(np.hypot(*(np.interp(t[i] + 0.1, t, P[:, 0]) - np.interp(t[i] - 0.1, t, P[:, 0]),
                             np.interp(t[i] + 0.1, t, P[:, 1]) - np.interp(t[i] - 0.1, t, P[:, 1]))) / 0.2)
        if v < V_ROW:
            continue
        pose = np.r_[P[i, :2], yaw[i]]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rig = np.stack([c * (H[:, 0] - pose[0]) + s * (H[:, 1] - pose[1]), -s * (H[:, 0] - pose[0]) + c * (H[:, 1] - pose[1]), wrap(hy - pose[2])], -1)
        a_h = arcs_of(rig)
        path, info = teacher_path(ref, pose, v, a_h[-1])
        rec = dict(scene=scene, t=round(float(t[i] - t[0]), 2), v=round(v, 2), d0=round(info.get("d0", np.nan), 3), why=info["why"])
        dpsi = float(np.degrees(abs(wrap(hy[-1] - pose[2]))))
        lc = v > LC_V and abs(rig[-1, 1]) > LC_Y and abs(np.degrees(rig[-1, 2])) < LC_YAW
        rec["cls"] = "turn" if dpsi >= TURN_DEG else ("lane-change" if lc else "other")
        if path is not None:
            _, pts, ph = place(path, a_h, pose)
            e = ((H - pts) * np.c_[-np.sin(ph), np.cos(ph)]).sum(1)
            rec |= dict(e2=round(float(e[3]), 3), e4=round(float(e[7]), 3), past_rec=info["past_rec"])
        path, info = teacher_path(refh, pose, v, a_h[-1])                       # amendment 1 (b): the teacher used for the rows (construction check)
        if path is not None:
            _, pts, ph = place(path, a_h, pose)
            e = ((H - pts) * np.c_[-np.sin(ph), np.cos(ph)]).sum(1)
            rec |= dict(e2h=round(float(e[3]), 3), e4h=round(float(e[7]), 3))
        out.append(rec)
    return dict(states=out, route=rinfo)


def cmd_t1(a):
    from jevdrive import par
    S = read_tsv(TSV)
    cfg = scene_cfgs(a.cfg_glob)
    jobs = [(r["scene_id"], cfg[r["scene_id"]]) for r in S if r["scene_id"] in cfg][:a.limit or None]
    res = par.pmap(_t1_one, jobs, workers=a.workers or None, desc="t1 scenes")
    fail = {jobs[i][0]: e.strip().splitlines()[-1][:200] for i, e in res.errors.items()}
    st = [x for v in res.values if v for x in v["states"]]
    ri = [v["route"] for v in res.values if v]
    ok = [x for x in st if "e2" in x]
    tab = {}
    for c in ("all", "turn", "lane-change", "other", "all but lane-change"):
        m = [x for x in ok if c == "all" or x["cls"] == c or (c == "all but lane-change" and x["cls"] != "lane-change")]
        n_all = sum(1 for x in st if c == "all" or x["cls"] == c or (c == "all but lane-change" and x["cls"] != "lane-change"))
        if m:
            e2, e4 = np.abs([x["e2"] for x in m]), np.abs([x["e4"] for x in m])
            mh = [x for x in m if "e2h" in x]
            tab[c] = dict(states=n_all, labelled=len(m), e2_med=float(np.median(e2)), e4_med=float(np.median(e4)), e2_p90=float(np.percentile(e2, 90)),
                          e4_p90=float(np.percentile(e4, 90)), d0_med=float(np.median(np.abs([x["d0"] for x in m]))),
                          e2h_med=float(np.median(np.abs([x["e2h"] for x in mh]))) if mh else None,
                          e4h_med=float(np.median(np.abs([x["e4h"] for x in mh]))) if mh else None)
    why = Counter(x["why"] or "ok" for x in st)
    passed = dict(all=bool(tab["all"]["e2_med"] <= 0.3 and tab["all"]["e4_med"] <= 0.5),
                  turn_inspect=bool(tab.get("turn", {}).get("e2_med", 0) > 0.6 or tab.get("turn", {}).get("e4_med", 0) > 1.0))
    rec = dict(scenes=len(jobs), scenes_ok=len(ri), failed=fail, states=len(st), why=dict(why), table=tab, passed=passed,
               route=dict(switch_scenes=sum(r["switches"] > 0 for r in ri), switches=sum(r["switches"] for r in ri),
                          steps_med=float(np.median([s for r in ri for s in r["steps"]])) if any(r["steps"] for r in ri) else None,
                          gen_vs_lanes_max=float(max(r["gen_vs_lanes_max"] for r in ri)) if ri else None,
                          ext_points_med=float(np.median([r["n_ext"] for r in ri])) if ri else None,
                          h_clip_share_med=float(np.median([r["h_clip_share"] for r in ri])) if ri else None,
                          h_clip_share_mean=float(np.mean([r["h_clip_share"] for r in ri])) if ri else None,
                          h_offset_med=float(np.median([r["h_offset_med"] for r in ri])) if ri else None))
    dump(RES / "t1.json", rec)
    import pandas as pd
    pd.DataFrame(st).to_csv(RES / "t1_states.csv.gz", index=False)
    L = ["# T1: the map teacher on HUMAN states (generated by dag.py t1)", "",
         f"Human states: the recorded log of the {len(ri)} non-held curated train scenes, every 0.5 s with speed >= {V_ROW} m/s and a full 4 s future. "
         "Teacher path from the human pose; label points at the human's own arc lengths (the T1 analogue of the student's served plan). "
         "Error = the human's future point minus the teacher point at the same time, along the teacher path's left normal.", "",
         "Classes: turn = the human turns >= 30 deg in 4 s; lane-change = D1's shape (v > 5 m/s, |y4| > 2.5 m, |yaw4| < 12 deg in the human's "
         "frame; the teacher keeps the route lane, so these are expected to differ); other = the rest.", "",
         "Two teachers: the registered lane-centre teacher (the teacher-geometry check), and amendment 1 (b), lane centre + the human's own "
         f"lateral offset (sigma {H_SIG:.0f} m, |offset| > {H_CLIP} m -> lane centre), the teacher the rows use. On human states (b) is close to the "
         "human by construction: its column is a construction check, not a pass.", "",
         "| states | n | labelled | lane centre: median abs lat at 2 s (m) | at 4 s | p90 at 2 s | p90 at 4 s | median abs d0 | (b): at 2 s | at 4 s |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for c, v in tab.items():
        L.append(f"| {c} | {v['states']} | {v['labelled']} | {v['e2_med']:.3f} | {v['e4_med']:.3f} | {v['e2_p90']:.2f} | {v['e4_p90']:.2f} | {v['d0_med']:.3f} | "
                 f"{f3(v['e2h_med'])} | {f3(v['e4h_med'])} |")
    L += ["", f"Line (plan, lane-centre teacher): all states median <= 0.30 m at 2 s and <= 0.50 m at 4 s -> **{'PASS' if passed['all'] else 'FAIL'}** "
          "(the miss is the human's offset from the NuRec lane centre, signed median left; amendment 1). "
          f"Turns over 2x the line (inspect before rows): {passed['turn_inspect']}.", "",
          f"Amendment 1 (b): per-scene human offset median {rec['route']['h_offset_med']}; share of the recorded route stations clipped to the plain "
          f"lane centre (|offset| > {H_CLIP} m): median {rec['route']['h_clip_share_med']}, mean {rec['route']['h_clip_share_mean']}.", "",
          f"No label: {dict(why)}. Route: lateral lane switches in {rec['route']['switch_scenes']} scenes ({rec['route']['switches']}; step median "
          f"{rec['route']['steps_med']} m); the generator's points vs the replayed lane projections max {rec['route']['gen_vs_lanes_max']} m. "
          f"Failed scenes: {len(fail)}."]
    (RES / "t1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if fail:
        print("failed:", json.dumps(fail)[:2000])


# ---------------------------------------------------------------- T2: the stored MV-family offroad zeros
T2_READS = {"R3MV-s0": ("R3MV-L18-s0-281ag", "R3MV-s0-281ag"), "R3MV-s1": ("R3MV-L18-s1-281ag", "R3MV-s1-281ag"),
            "MVLC-s0": ("R5LC-L18-s0-281ag", "R5LC-s0-281ag"), "MVLC-s1": ("R5LC-L18-s1-281ag", "R5LC-s1-281ag")}


def _t2_one(job):
    import road_pai as RP
    scene, cfg, evs = job
    ds = load_ds(scene, cfg)
    (_, ref), _ = route_ref(ds)
    g = RP.scene_geom(scene, cfg, np.array([e["pose"][:2] for e in evs]))
    out = []
    for e in evs:
        a = arcs_of(e["ps"])
        path, info = teacher_path(ref, e["pose"], e["v0"], a[-1])
        fut = place(path, a, e["pose"])[0] if path is not None else np.full((8, 3), np.nan)
        sdf = RP.pai_raster(g, e["pose"])[0]
        out.append(dict(e, fut=fut, sdf=sdf, info=info, empty=g["U"] is None and not g["E"]))
    return out


def cmd_t2(a):
    import pickle
    import pandas as pd
    import torch
    from drivable_hinge import Hinge
    import road_pai as RP
    from jevdrive import par
    torch.set_num_threads(4)
    ntv0 = pd.read_csv(DATA / "runs/pai_fix/facts/NTV0-L18-s0-281ag.csv").set_index("scene")
    cfg = scene_cfgs(str(DATA / "runs/round5/r5/pai281/*-ag/runs/*/sim/generated-user-config-0.yaml"))
    evs = []
    for rd, (facts, dec) in T2_READS.items():
        F = pd.read_csv(DATA / f"runs/pai_fix/facts/{facts}.csv")
        R = {o["scene"]: o for o in pickle.load(open(DATA / f"runs/pai_fix/tax/{facts}.pkl", "rb"))["rolls"] if "ctrl" in o}
        Dc = pickle.load(open(DATA / f"runs/round5/offroad_mv/dec/dec_{dec}.pkl", "rb"))
        for _, r in F[(F.fclass == "offroad") & ~F.rpc.fillna(False).astype(bool)].iterrows():
            s = r.scene
            if s not in R or s not in Dc:
                continue
            o, d = R[s], Dc[s]
            te = RP.first_event(o, "offroad")
            if not np.isfinite(te):
                te = RP.first_event(o, "left_corridor_laterally")
            if not np.isfinite(te):
                continue
            k = min(max(int(np.searchsorted(d["t0"], te - T2_LEAD * 1e6)), 0), len(d["t0"]) - 1)
            evs.append(dict(read=rd, scene=s, k=k, to_event_s=round((te - float(d["t0"][k])) * 1e-6, 2), pose=RP.ego_at(o["ctrl"], float(d["t0"][k])),
                            ps=np.asarray(d["ps"][k], np.float64), pm=np.asarray(d["pm"][k], np.float64), v0=float(d["v0"][k]),
                            ntv0_off=str(ntv0.fclass.get(s, "")) == "offroad", ntv0_score=float(ntv0.score.get(s, np.nan))))
    by = defaultdict(list)
    for e in evs:
        by[e["scene"]].append(e)
    jobs = [(s, cfg[s], v) for s, v in sorted(by.items())]
    res = par.pmap(_t2_one, jobs, workers=a.workers or None, desc="t2 scenes")
    res.raise_if_failed()
    E = [x for v in res.values for x in v]
    tok = np.array([f"{e['read']}:{e['scene']}:{e['k']}" for e in E])
    f = O_DEF / "t2_rasters.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, tokens=tok, ok=np.array([not e["empty"] for e in E]), sdf=np.stack([e["sdf"] for e in E]))
    H = Hinge(f, tok, "cpu", RP.MARGIN)
    rows = np.arange(len(E))
    has = np.array([np.isfinite(e["fut"]).all() for e in E])
    T = np.stack([np.nan_to_num(e["fut"]) for e in E]).astype(np.float32)
    S = np.stack([e["ps"] for e in E]).astype(np.float32)
    M = np.stack([e["pm"] for e in E]).astype(np.float32)
    with torch.no_grad():
        z = torch.zeros(len(E), 8)
        m0 = H.margins(z, z, z, torch.as_tensor(rows)).amin(1).numpy()
        mg = {k: H.margins(torch.as_tensor(X[..., 0]), torch.as_tensor(X[..., 1]), torch.as_tensor(X[..., 2]), torch.as_tensor(rows)).amin(1).numpy()
              for k, X in (("teacher", T), ("served", S), ("model", M))}
    incan = np.array([(e["fut"][:, 0] <= 54).all() if h else False for e, h in zip(E, has)])
    rec_rows = []
    for i, e in enumerate(E):
        rec_rows.append(dict(read=e["read"], scene=e["scene"], k=e["k"], to_event_s=e["to_event_s"], v0=round(e["v0"], 2), ntv0_off=e["ntv0_off"],
                             d0=round(e["info"].get("d0", np.nan), 3), psi0_deg=round(float(np.degrees(e["info"].get("psi0", np.nan))), 1),
                             why=e["info"]["why"], m_t0=round(float(m0[i]), 3), m_teacher=round(float(mg["teacher"][i]), 3) if has[i] else "",
                             m_served=round(float(mg["served"][i]), 3), m_model=round(float(mg["model"][i]), 3), label_in_canvas=bool(incan[i]),
                             no_surface=bool(e["empty"])))
    pd.DataFrame(rec_rows).to_csv(RES / "t2.csv", index=False)
    D = pd.DataFrame(rec_rows)
    base = D[(D.m_t0 >= 0) & ~D.no_surface]
    tin = base[base.why == ""]
    inside = (pd.to_numeric(tin.m_teacher) >= 0)
    res_ = {}
    for nm, X in (("all", D), ("NTV0 not offroad", D[~D.ntv0_off])):
        b = X[(X.m_t0 >= 0) & ~X.no_surface]
        t_ = b[b.why == ""]
        res_[nm] = dict(events=len(X), t0_inside=len(b), labelled=len(t_), teacher_inside=int((pd.to_numeric(t_.m_teacher) >= 0).sum()),
                        teacher_inside_share=float((pd.to_numeric(t_.m_teacher) >= 0).mean()) if len(t_) else None,
                        served_inside=int((b.m_served >= 0).sum()), model_inside=int((b.m_model >= 0).sum()),
                        no_label=dict(Counter(b.why[b.why != ""])), label_in_canvas=int(t_.label_in_canvas.sum()),
                        teacher_m_med=float(pd.to_numeric(t_.m_teacher).median()) if len(t_) else None, served_m_med=float(b.m_served.median()) if len(b) else None)
    share = len(tin) and float(inside.mean())
    passed = bool(len(tin) >= 0.8 * len(base) and share >= 0.9)
    tc = tin[tin.label_in_canvas]                                              # amendment 1 (c): label poses all inside the raster canvas
    canvas = dict(events=len(tc), teacher_inside=int((pd.to_numeric(tc.m_teacher) >= 0).sum()), served_inside=int((tc.m_served >= 0).sum()),
                  beyond=len(tin) - len(tc))
    canvas["share"] = canvas["teacher_inside"] / max(len(tc), 1)
    canvas["passed"] = bool(canvas["share"] >= 0.9)
    rec = dict(events=len(E), lead_s=T2_LEAD, results=res_, in_canvas=canvas, line="teacher label footprint min SDF >= 0 on >= 90 % of the events whose t0 footprint is inside "
               "(and >= 80 % of those labelled)", passed=passed)
    dump(RES / "t2.json", rec)
    L = ["# T2: the map teacher on the MV-family PAI 281 offroad zeros (generated by dag.py t2)", "",
         f"Events: every offroad zero of R3MV s0 / s1 and MV-LC s0 / s1 (PAI 281 ACC + GAP) with a scorer event; the decision {T2_LEAD} s before the first "
         "offroad (else left-corridor) event; anchor = the controller pose at that decision (road_pai g2's rule); teacher label timed by the "
         "decision's served plan. Footprint = Pacifica, interpolated at 0.1 s (drivable_hinge.Hinge), on the road_pai raster at the anchor (the "
         "evaluator's PAI offroad rule; -8 .. 56 m ahead: poses beyond read the border).", "",
         "| events | n | t0 inside | teacher labelled | teacher inside | share | served plan inside | model plan inside | median margin teacher / served (m) | no label |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|"]
    for nm, v in res_.items():
        sh = "-" if v["teacher_inside_share"] is None else f"{v['teacher_inside_share']:.0%}"
        L.append(f"| {nm} | {v['events']} | {v['t0_inside']} | {v['labelled']} | {v['teacher_inside']} | {sh} | "
                 f"{v['served_inside']} | {v['model_inside']} | {v['teacher_m_med']} / {v['served_m_med']} | {v['no_label']} |")
    L += ["", f"Line (plan): {rec['line']} -> **{'PASS' if passed else 'FAIL'}** (raw). Amendment 1 (c), the registered measurement fix: events "
          f"whose label poses all lie inside the raster canvas (x <= 54 m; beyond it the margin reads border padding): teacher inside "
          f"**{canvas['teacher_inside']} / {canvas['events']}** ({canvas['share']:.0%}; served plan {canvas['served_inside']}); {canvas['beyond']} labelled events "
          f"reach past the canvas. Teacher = amendment 1 (b). Per event: t2.csv."]
    (RES / "t2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- teacher labels for the student's decisions
def _teach_one(job):
    scene, cfg, rowf, out = job
    keys = ("k", "anchor", "poses", "v0", "after_event", "after_end")
    with np.load(rowf) as z:
        Z = {k: z[k] for k in keys}
    ds = load_ds(scene, cfg)
    (_, ref), rinfo = route_ref(ds)
    n = len(Z["k"])
    fut, ok = np.full((n, 8, 3), np.nan, np.float32), np.zeros(n, bool)
    why = np.full(n, "", "U8")
    d0, psi0, L, alat, aref = (np.full(n, np.nan, np.float32) for _ in range(5))
    past_rec, past_ref = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        A = Z["anchor"][i].astype(np.float64)
        a = arcs_of(Z["poses"][i])
        path, info = teacher_path(ref, A, float(Z["v0"][i]), a[-1])
        why[i] = info["why"]
        d0[i], psi0[i] = info.get("d0", np.nan), info.get("psi0", np.nan)
        if path is None:
            continue
        fut[i] = place(path, a, A)[0]
        L[i] = info["L"]
        alat[i], aref[i] = lat_acc(path, a, float(Z["v0"][i]))
        past_rec[i], past_ref[i] = info["past_rec"], info["past_ref"]
        ok[i] = np.isfinite(fut[i]).all()
    tmp = Path(out) / f"{scene}.tmp.npz"
    np.savez(tmp, k=Z["k"], fut=fut, ok=ok, why=why, d0=d0, psi0=psi0, L=L, alat=alat, alat_ref=aref, past_rec=past_rec, past_ref=past_ref,
             v0=Z["v0"], served=Z["poses"], anchor=Z["anchor"], pre=~Z["after_event"] & ~Z["after_end"], route=json.dumps(rinfo),
             version=TEACHER_V)
    os.replace(tmp, Path(out) / f"{scene}.npz")
    return dict(scene=scene, n=n, ok=int(ok.sum()), switches=rinfo["switches"])


def cmd_teach(a):
    from jevdrive import par
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = scene_cfgs(a.cfg_glob)
    want = set(Path(a.chunk_list).read_text().split()) if a.chunk_list else None
    files = sorted(Path(a.rows).glob("clipgt-*.npz"))
    jobs = [(f.stem, cfg[f.stem], str(f), str(out)) for f in files if (want is None or f.stem in want) and f.stem in cfg]
    miss = [f.stem for f in files if (want is None or f.stem in want) and f.stem not in cfg]
    def fresh(j):                                                              # --stale-only: skip scenes already taught by this teacher version
        f = out / f"{j[0]}.npz"
        if not (a.stale_only and f.exists()):
            return False
        with np.load(f) as z:
            return "version" in z.files and str(z["version"]) == TEACHER_V
    res = par.pmap(_teach_one, jobs, workers=a.workers or None, desc="teach scenes", skip=fresh)
    res.raise_if_failed()
    v = [x for x in res.values if x]
    print(json.dumps(dict(scenes=len(v), rows=sum(x["n"] for x in v), ok=sum(x["ok"] for x in v), no_cfg=miss)))


# ---------------------------------------------------------------- stage-1 check
def runtimes(root):
    out = []
    for sim in sorted(glob.glob(f"{root}/runs/*/sim")):
        f, log = Path(sim) / "aggregate/results-summary.json", Path(sim).parent / "pool/log.txt"
        if f.exists() and log.exists():
            m = re.findall(r"'runtime_s': ([0-9.]+)", log.read_text())
            if m:
                out.append((float(m[-1]), len(json.loads(f.read_text())["rollouts"])))
    return out


def cmd_chk1(a):
    O = Path(a.root)
    want = (O / "lists/c000.txt").read_text().split()
    sm = sorted(glob.glob(str(O / "roll/runs/*_c000/sim/aggregate/results-summary.json")))
    scored = {r["clipgt_id"]: r for f in sm for r in json.loads(Path(f).read_text())["rollouts"]}
    rows = sorted(p.stem for p in (O / "rows/PAID").glob("clipgt-*.npz"))
    T = [np.load(O / "teach" / f"{s}.npz") for s in want if (O / "teach" / f"{s}.npz").exists()]
    v0 = np.concatenate([t["v0"] for t in T]) if T else np.zeros(0)
    pre = np.concatenate([t["pre"] for t in T]) if T else np.zeros(0, bool)
    okt = np.concatenate([t["ok"] for t in T]) if T else np.zeros(0, bool)
    al = np.concatenate([t["alat"] for t in T]) if T else np.zeros(0)
    d0 = np.concatenate([t["d0"] for t in T]) if T else np.zeros(0)
    fast = (v0 >= V_ROW) & pre
    rt = runtimes(O / "roll")
    sps = float(np.mean([t / max(k, 1) for t, k in rt])) if rt else float("nan")
    n_all = len(read_tsv(TSV))
    rec = dict(listed=len(want), scored=sum(s in scored for s in want), rows=len(rows), teach=len(T), decisions=int(len(v0)), fast=int(fast.sum()),
               teacher_ok_of_fast=float(okt[fast].mean()) if fast.any() else None, alat_p99=float(np.nanpercentile(al[fast & okt], 99)) if (fast & okt).any() else None,
               d0_abs_med=float(np.nanmedian(np.abs(d0[fast]))) if fast.any() else None, d0_abs_p90=float(np.nanpercentile(np.abs(d0[fast]), 90)) if fast.any() else None,
               s_per_scene=sps, card_h_projected=sps * n_all / 3600 / CARD_STACKS)
    why = []
    if rec["scored"] < len(want) - 1:
        why.append(f"{rec['scored']} of {len(want)} scored")
    if rec["rows"] < len(want) - 1:
        why.append(f"{rec['rows']} row files")
    if not (rec["teacher_ok_of_fast"] or 0) >= 0.8:
        why.append(f"teacher ok on {rec['teacher_ok_of_fast']} of the >= 2 m/s decisions (line 0.8)")
    if not rec["card_h_projected"] <= 6.0:
        why.append(f"projected {rec['card_h_projected']:.1f} card-h (> 6)")
    rec["ok"], rec["why"] = not why, why
    dump(O / "checks/c000.json", rec)
    print(json.dumps(rec))
    if why:
        (O / "VERDICT").write_text(f"STOP c000 check: {'; '.join(why)}\n")
        sys.exit(1)


# ---------------------------------------------------------------- the pool's labels
def pool_rows(pool):
    tb = dict(np.load(CR / pool / "tab.npz"))
    nm = tb["names"].astype(str)
    return tb, np.array([n.split(":")[2] for n in nm]), np.array([int(n.split(":")[3]) for n in nm])


def cmd_label(a):
    O = Path(a.root)
    tb, sc, kk = pool_rows(a.pool)
    n = len(sc)
    cls = {r["scene_id"]: r["cls"] for r in read_tsv(TSV)}
    X = dict(names=tb["names"], fut_t=np.full((n, 8, 3), np.nan, np.float32), ok=np.zeros(n, bool), why=np.full(n, "", "U8"),
             d0=np.full(n, np.nan, np.float32), psi0=np.full(n, np.nan, np.float32), alat=np.full(n, np.nan, np.float32),
             alat_ref=np.full(n, np.nan, np.float32), past_rec=np.zeros(n, bool), cls=np.array([cls.get(s, "?") for s in sc]))
    dv0 = 0.0
    for s in np.unique(sc):
        j = np.flatnonzero(sc == s)
        with np.load(O / "teach" / f"{s}.npz") as z:
            assert "version" in z.files and str(z["version"]) == TEACHER_V, f"{s}: teacher file is not version {TEACHER_V}"
            pos = {int(k): i for i, k in enumerate(z["k"])}
            i = np.array([pos[int(k)] for k in kk[j]])
            for key in ("ok", "why", "d0", "psi0", "alat", "alat_ref", "past_rec"):
                X[key][j] = z[key][i]
            X["fut_t"][j] = z["fut"][i]
            dv0 = max(dv0, float(np.abs(z["v0"][i] - tb["speed"][j]).max()), float(np.abs(z["served"][i] - tb["served"][j]).max()))
    v0 = tb["speed"].astype(np.float64)
    draw = X["ok"] & (v0 >= V_ROW) & np.isfinite(X["fut_t"]).reshape(n, -1).all(1)
    X["draw"] = draw
    sv = tb["served"].astype(np.float64)
    fut = None
    lat = X["fut_t"][:, :, 1] - tb["served"][:, :, 1]
    lon = np.abs(arcs_of_n(X["fut_t"]) - arcs_of_n(sv))
    al = X["alat"][draw]
    t4 = float(np.percentile(al, 99)) if draw.any() else float("inf")
    hi = draw & (X["alat"] > A_LAT)
    t4_fallback = bool(t4 > A_LAT and hi.sum() <= 0.03 * draw.sum())             # plan section 4: over-line rows <= 3 % -> weight 0, go on
    if t4_fallback:
        draw &= ~hi
        X["draw"] = draw
    d0a = np.abs(X["d0"])
    tabs = {}
    for c in CLS + ("all",):
        m = draw & ((X["cls"] == c) if c != "all" else True)
        tabs[c] = dict(rows=int(((X["cls"] == c) if c != "all" else np.ones(n, bool)).sum()), drawable=int(m.sum()),
                       d0_med=float(np.median(d0a[m])) if m.any() else None, d0_p90=float(np.percentile(d0a[m], 90)) if m.any() else None,
                       d0_gt_05=int((m & (d0a > 0.5)).sum()), d0_gt_1=int((m & (d0a > 1.0)).sum()),
                       lat4_med=float(np.median(np.abs(lat[m, 7]))) if m.any() else None, lat4_p90=float(np.percentile(np.abs(lat[m, 7]), 90)) if m.any() else None,
                       alat_p99=float(np.percentile(X["alat"][m], 99)) if m.any() else None)
    rec = dict(pool=a.pool, rows=n, scenes=int(len(np.unique(sc))), drawable=int(draw.sum()), slow=int((v0 < V_ROW).sum()),
               no_label=dict(Counter(X["why"][(v0 >= V_ROW) & ~X["ok"]].tolist())), past_rec=int((draw & X["past_rec"]).sum()),
               lon_arc_err_max=float(np.nanmax(lon[draw])) if draw.any() else None, teach_vs_pool_max=dv0,
               T4=dict(line=A_LAT, alat_p99=t4, alat_ref_p99=float(np.percentile(X["alat_ref"][draw], 99)) if draw.any() else None,
                       over=int((al > A_LAT).sum()), passed=bool(t4 <= A_LAT), fallback_rows_dropped=int(hi.sum()) if t4_fallback else 0,
                       go_on=bool(t4 <= A_LAT or t4_fallback)), classes=tabs)
    assert dv0 < 1e-4, f"teacher inputs differ from the pool's speed / served ({dv0})"
    fut = np.where(draw[:, None, None], X["fut_t"], tb["served"]).astype(np.float32)
    np.savez(CR / a.pool / "dagger.npz", **X)
    np.savez(CR / a.pool / "tab.npz", **(tb | dict(fut=fut)))
    dump(RES / "rows.json", rec)
    dump(O / "checks/label.json", rec)
    L = [f"# DAgger rows: pool {a.pool} (generated by dag.py label)", "",
         f"Rows = the student's ({STUDENT}) usable 10 Hz decisions before its first scorer event and the log's end (rf_pai_pool's rule) on "
         f"{rec['scenes']} scenes: {n}. Drawable = fed speed >= {V_ROW} m/s and a teacher label: **{rec['drawable']}** "
         f"({rec['slow']} slow; no label {rec['no_label']}). Label = the teacher path at the served plan's arc lengths "
         f"(arc-length error vs the served plan max {rec['lon_arc_err_max']:.3g} m); non-drawable rows keep fut = served and weight 0.", "",
         "| class | rows | drawable | median abs d0 | p90 | abs d0 > 0.5 m | > 1 m | median abs label - served y at 4 s | p90 | lat. acc. p99 |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    for c, v in tabs.items():
        L.append(f"| {c} | {v['rows']} | {v['drawable']} | {f2(v['d0_med'])} | {f2(v['d0_p90'])} | {v['d0_gt_05']} | {v['d0_gt_1']} | "
                 f"{f2(v['lat4_med'])} | {f2(v['lat4_p90'])} | {f2(v['alat_p99'])} |")
    L += ["", f"**T4** (label lateral acceleration at the served speeds, p99 <= {A_LAT} m/s^2): p99 {t4:.2f} (route curvature alone "
          f"{rec['T4']['alat_ref_p99']:.2f}), {rec['T4']['over']} rows over -> **{'PASS' if rec['T4']['passed'] else 'FAIL'}**"
          f"{f' (fallback: {hi.sum()} rows <= 3 % set to weight 0)' if t4_fallback else ''}. "
          f"Labels reaching past the recorded route (generator extension): {rec['past_rec']}."]
    (RES / "rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not rec["T4"]["go_on"]:
        (O / "VERDICT").write_text(f"STOP T4: lateral acceleration p99 {t4:.2f} > {A_LAT}\n")
        sys.exit(1)


def arcs_of_n(fut):
    P = np.concatenate([np.zeros((len(fut), 1, 2)), np.asarray(fut, np.float64)[:, :, :2]], 1)
    return np.cumsum(np.linalg.norm(np.diff(P, axis=1), axis=-1), 1)


# ---------------------------------------------------------------- input path check, merge check, weights, smoke
def cmd_r1in(a):
    """The trainer-path forward of --tag on the pool's tokens + ego (rf_recov fwd) against the model's own plan logged in the row files."""
    O = Path(a.root)
    tb, sc, kk = pool_rows(a.pool)
    ref = np.full((len(sc), 8, 2), np.nan)
    for s in np.unique(sc):
        j = np.flatnonzero(sc == s)
        with np.load(O / "rows/PAID" / f"{s}.npz") as z:
            pos = {int(k): i for i, k in enumerate(z["k"])}
            ref[j] = z["poses_model"][[pos[int(k)] for k in kk[j]]][:, :, :2]
    tol = np.maximum(4 * 2.0 ** (np.floor(np.log2(np.maximum(np.abs(ref), 2.0 ** -14))) - 10), 0.03)
    res = {}
    for t in (a.tag, a.control):
        z = np.load(Path(a.fwd) / f"plans_{t}.npz")
        P = np.full(ref.shape, np.nan)
        P[z["rows"]] = z["poses"][:, :8, :2]
        d = np.abs(P - ref).reshape(len(ref), -1).max(1)
        over = (np.abs(P - ref) > tol).reshape(len(ref), -1).any(1)
        res[t] = dict(share_over=float(over.mean()), median_m=float(np.nanmedian(d)), p99_m=float(np.nanpercentile(d, 99)), rows=int(len(d)), missing=int(np.isnan(d).sum()))
    out = dict(pool=a.pool, tag=a.tag, control=a.control, model=res[a.tag], wrong=res[a.control])
    out["ok"] = bool(res[a.tag]["missing"] == 0 and res[a.tag]["median_m"] <= 0.02 and res[a.tag]["share_over"] <= 0.10 and res[a.control]["share_over"] >= 0.9)
    dump(O / "checks/r1in.json", out)
    print(json.dumps(out))
    if not out["ok"]:
        (O / "VERDICT").write_text(f"STOP input path: {out}\n")
        sys.exit(1)


def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def cmd_mfix(a):
    """s3.py mfix with this lane's check dir: the merged prefix equals --ref in every file; agent ok of the prefix taken from --ref."""
    from jevdrive.tokens import open_front
    P, R = CR / a.pool, CR / a.ref
    tp, tr = dict(np.load(P / "tab.npz")), np.load(R / "tab.npz")
    n = len(tr["names"])
    bad = [k for k in tr.files if k in tp and not same(tp[k][:n], tr[k])] + [f"tab.{k} missing" for k in tr.files if k not in tp]
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        if (R / f).exists():
            zp, zr = np.load(P / f), np.load(R / f)
            bad += [f"{f}.{k}" for k in zr.files if not (k == "ok" and f == "agent.npz") and zr[k].ndim and len(zr[k]) == n and not same(zp[k][:n], zr[k])]
    feat = DATA / "runs/route_mem/feat"
    fp, fr = np.load(feat / f"{a.pool}.npz"), np.load(feat / f"{a.ref}.npz")
    bad += [f"feat.{k}" for k in fr.files if not same(fp[k][:n], fr[k])]
    zp, zr = np.load(CR / f"{a.pool}@warp/teacher.npz"), np.load(CR / f"{a.ref}@warp/teacher.npz")
    bad += [f"teacher.{k}" for k in zr.files if not same(zp[k][:n] if zr[k].ndim and len(zr[k]) == n else zp[k], zr[k])]
    i = np.sort(np.random.default_rng(0).choice(n, 4000, replace=False))
    if not same(open_front(CR / f"{a.pool}@warp/front.npy")[i], open_front(CR / f"{a.ref}@warp/front.npy")[i]):
        bad.append("front sample")
    ag, agr = dict(np.load(P / "agent.npz")), np.load(R / "agent.npz")
    before = int((ag["ok"][:n] != agr["ok"]).sum())
    ag["ok"] = np.concatenate([agr["ok"], ag["ok"][n:]])
    np.savez(P / "agent.npz", **ag)
    part = tp["pool"].astype(str)
    rec = dict(pool=a.pool, ref=a.ref, ref_rows=n, rows=len(part), parts={p: int((part == p).sum()) for p in np.unique(part)}, agent_ok_fixed=before,
               new_rows_mode=sorted(set(tp["mode"][n:].astype(str).tolist())), new_agent_ok=int(ag["ok"][n:].sum()), mismatch=bad)
    dump(Path(a.root) / "checks" / f"mfix_{a.pool}.json", rec)
    print(json.dumps(rec))
    if bad:
        sys.exit(2)


def cmd_weights(a):
    """R5LC's draw x k_on / (k_on + k_dag) on the merged pool's on1_JTPH prefix (bit-equal names); the drawable DAgger rows k_dag / (k_on + k_dag),
    uniform. k_dag 0 = the identity file (R5LC's weights, DAgger rows at 0)."""
    B = np.load(W5 / f"{a.base}.npz")
    names = np.load(CR / a.pool / "tab.npz")["names"].astype(str)
    n0 = len(B["names"])
    assert np.array_equal(names[:n0], B["names"].astype(str)), f"{a.pool}: prefix != {a.base}'s rows"
    src = a.pool.replace("JTPH", "PA") if not a.src else a.src
    X = np.load(CR / src / "dagger.npz")
    assert len(names) == n0 + len(X["names"]) and all(x.endswith(y[2:]) for x, y in zip(names[n0:], X["names"].astype(str))), "tail != the DAgger pool"
    k_all = a.k_on + a.k_dag
    w = np.zeros(len(names))
    w[:n0] = B["w"].astype(np.float64) * (a.k_on / k_all if a.k_dag else 1.0)
    dr = np.flatnonzero(X["draw"])
    if a.k_dag:
        w[n0 + dr] = a.k_dag / k_all / len(dr)
    grp = np.concatenate([B["grp"].astype(str), np.full(len(names) - n0, "pd")])
    assert abs(w.sum() - 1) < 1e-9, w.sum()
    if not a.k_dag:
        assert np.array_equal(w[:n0], B["w"]), "identity file: prefix weights changed"
    gs = defaultdict(float)
    for g_, x in zip(grp, w):
        gs[str(g_)] += x
    rec = dict(base=a.base, pool=a.pool, k_on=a.k_on, k_dag=a.k_dag, k_all=k_all, dag_rows=int(len(dr)), group_share={k: float(v) for k, v in sorted(gs.items()) if v > 0},
               by_cls={c: int((X["cls"][dr] == c).sum()) for c in CLS}, out=a.out)
    W6.mkdir(parents=True, exist_ok=True)
    np.savez(W6 / f"{a.out}.npz", names=names, w=w, grp=grp)
    dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_smoke(a):
    import r3
    d = r3.run_dir(a.tag)
    S = defaultdict(list)
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S[e["tag"]].append(e["value"])
    done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    ds = done.get("draw_shares") or {}
    mean = lambda k: float(np.mean(S[k])) if k in S else float("nan")  # noqa: E731
    ex = json.loads((RES / f"w_{a.wfile}.json").read_text())
    k_on = 64 + a.k_dag
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy", (d / "log.txt").read_text())
    k_im = 32 + a.k_dag
    ck = dict(done=(bool(done), True, bool(done)), finite=(None, None, all(np.isfinite(v).all() for v in S.values())),
              no_flat=(mean("loss/flat/rows"), 0, np.nan_to_num(mean("loss/flat/rows")) == 0 and not done.get("flat_brk")),
              wfile=(ds.get("on_wfile"), a.wfile, f"/{a.wfile}.npz" in str(ds.get("on_wfile"))),
              batch_rows=(b.groups() if b else None, ("51", "13", str(k_on)), bool(b) and b.groups() == ("51", "13", str(k_on))),
              pool_imit_rows=(mean("loss/on/n_imit"), k_im, 0.75 * k_im <= mean("loss/on/n_imit") <= 1.25 * k_im))
    gs = ds.get("group_share", {})
    ck |= {f"share_{p}": (gs.get(p), v, abs(gs.get(p, -1) - v) < 1e-6) for p, v in ex["group_share"].items()}
    ok = bool(all(v[2] for v in ck.values()))
    res = dict(tag=a.tag, run=str(d), pass_=ok, checks=ck, steps=len(S.get("loss/total", [])), draw_shares=ds)
    dump(RES / "smoke.json", res)
    print(json.dumps(res, indent=1, default=float))
    if not ok:
        raise SystemExit(2)


# ---------------------------------------------------------------- verdict and ruler rows
def cmd_verdict(a):
    """The registered rule (plan section 6) on r2.py r281's json + new zeros / partials on the scenes NTV0 scores 1.0 -> appended to the md."""
    import pandas as pd
    J = json.loads(Path(a.json).read_text())
    R, P = J["reads"], J["pairs"]
    lat = lambda k: R[k]["z_offroad"] + R[k]["z_zero-other"]  # noqa: E731
    cap = lat("NTV0-ag") + 3
    F = {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in a.read)}
    com = sorted(set.intersection(*[set(X.index[X.score.notna() & ~X.rpc.fillna(False).astype(bool)]) for X in F.values()]))
    solved = (F["NTV0-ag"].score.reindex(com) >= 1 - 1e-4).to_numpy()
    sol = {}
    for k, X in F.items():
        y, c = X.score.reindex(com)[solved], X.fclass.reindex(com)[solved]
        sol[k] = dict(zeros=int((y <= 1e-4).sum()), partials=int(((y > 1e-4) & (y < 1 - 1e-4)).sum()),
                      by_class={str(q): int(n) for q, n in c[y < 1 - 1e-4].value_counts().items()})
    rows, ok = [], True
    for s in ("s0", "s1"):
        k = f"DAG-{s}-ag"
        if k not in R:
            continue
        d = P[f"{k}:NTV0-ag"]["mean"]
        b, l_ = d["mean"] >= 0, lat(k) <= cap
        ok &= b and l_
        m = P.get(f"{k}:R5LC-{s}-ag", {}).get("mean", {})
        rows.append(f"| {k} | {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'yes' if b else 'no'} | {lat(k)} (offroad {R[k]['z_offroad']}, "
                    f"zero-other {R[k]['z_zero-other']}) | {'yes' if l_ else 'no'} | {m.get('mean', float('nan')):+.3f} [{m.get('lo', float('nan')):+.3f}, "
                    f"{m.get('hi', float('nan')):+.3f}] | {lat(k) - lat(f'R5LC-{s}-ag'):+d} | {sol[k]['zeros']} / {sol[k]['partials']} |")
    L = ["", "## Registered rule (plan section 6)", "",
         f"adopt-candidate (primary) = both DAG seeds: mean - NTV0 >= 0 (paired point estimate) and offroad + zero-other <= NTV0's + 3 = {cap}; "
         "the ruler rule (CI upper bound of PCS - NTV0 >= 0) is appended by `dag.py pcs`.", "",
         "| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | - MV-LC same seed | offroad + zero-other - MV-LC | new zeros / partials on NTV0 1.0 |",
         "|:--|--:|:-:|--:|:-:|--:|--:|--:|", *rows, "",
         f"**Primary rule: {'both seeds pass' if ok else 'not both seeds'}.**", "",
         f"### New zeros / partials on the {int(solved.sum())} of {len(com)} common scenes NTV0 scores 1.0", "",
         "| read | new zeros | new partials | by class (zeros + partials) |", "|:--|--:|--:|:--|"]
    L += [f"| {k} | {v['zeros']} | {v['partials']} | " + ", ".join(f"{q} {n}" for q, n in sorted(v["by_class"].items(), key=lambda t: -t[1])) + " |"
          for k, v in sol.items() if k != "NTV0-ag"]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J["verdict"] = dict(primary=bool(ok), cap=cap, ntv0_solved=dict(scenes=int(solved.sum()), reads=sol))
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def cmd_pcs(a):
    Q = json.loads(Path(a.pcs).read_text())
    J = json.loads(Path(a.json).read_text())
    L = ["", "### Local PAI PCS ruler (d392 pool / anchors, out of pool, paired vs NTV0-s0+AG on the same bootstrap draws)", "",
         f"NTV0-s0+AG {Q['NTV0-s0+AG']['pcs']:.0f} [{Q['NTV0-s0+AG']['ci'][0]:.0f}, {Q['NTV0-s0+AG']['ci'][1]:.0f}].", "",
         "| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |", "|:--|--:|--:|--:|"]
    for k in ("R5LC-s0+AG", "R5LC-s1+AG", "DAG-s0+AG", "DAG-s1+AG"):
        if k in Q:
            q = Q[k]
            L.append(f"| {k} | {q['pcs']:.0f} [{q['ci'][0]:.0f}, {q['ci'][1]:.0f}] | {q['diff_median']:+.0f} ({q['diff_point']:+.0f}) "
                     f"[{q['diff_ci'][0]:+.0f}, {q['diff_ci'][1]:+.0f}] | {q['p_pos']:.2f} |")
    sp = {k: Q[k]["diff_ci"][1] >= 0 for k in ("DAG-s0+AG", "DAG-s1+AG") if k in Q}
    prim = J.get("verdict", {}).get("primary", False)
    cand = bool(prim and len(sp) == 2 and all(sp.values()))
    better = {}
    for s in ("s0", "s1"):
        k, m = f"DAG-{s}+AG", f"R5LC-{s}+AG"
        if k in Q and m in Q:
            r = J["reads"]
            dl = (r[f"DAG-{s}-ag"]["z_offroad"] + r[f"DAG-{s}-ag"]["z_zero-other"]) - (r[f"R5LC-{s}-ag"]["z_offroad"] + r[f"R5LC-{s}-ag"]["z_zero-other"])
            better[s] = bool(dl <= -5 and Q[k]["diff_point"] - Q[m]["diff_point"] >= 100)
    clearly = bool(len(better) == 2 and all(better.values()))
    L += ["", "Ruler rule: each DAG seed's ruler difference to NTV0 has a 95 % CI upper bound >= 0: " + ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in sp.items()),
          "", f"**DAG: {'adopt-candidate' if cand else 'not an adopt-candidate'}** (primary rule {'pass' if prim else 'fail'}; ruler rule "
          f"{'pass' if sp and all(sp.values()) else 'fail'}). Clearly better than MV-LC (both seeds: offroad + zero-other <= MV-LC - 5 and ruler "
          f"point >= MV-LC + 100): {better} -> {'yes' if clearly else 'no'}."]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J.setdefault("verdict", {})["pcs"] = dict({k: Q[k] for k in Q if k.endswith("+AG")}, ruler_pass=sp)
    J["verdict"] |= dict(adopt_candidate=cand, clearly_better_than_mvlc=clearly, better_by_seed=better)
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


# ---------------------------------------------------------------- report
def du(p):
    p = Path(p)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def cmd_report(a):
    O = Path(a.root)
    disk = {k: du(v) / 2 ** 30 for k, v in dict(roll=O / "roll", rows=O / "rows", teach=O / "teach", cache=O / "cache", pool=CR / "on1_PAID",
                                                 pool_warp=CR / "on1_PAID@warp", merged=CR / "on1_JTPHD", merged_warp=CR / "on1_JTPHD@warp").items()}
    h = sum(t for t, _ in runtimes(O / "roll")) / 3600
    ok = lambda d: next((d[k] for k in ("ok", "pass_", "passed", "identical") if k in d), None)  # noqa: E731
    rec = dict(disk_gb=disk, roll=dict(stack_h=h, card_h=h / CARD_STACKS), checks={f.stem: ok(json.loads(f.read_text())) for f in sorted((O / "checks").glob("*.json"))})
    dump(RES / "build.json", rec)
    L = ["# DAgger build: disk, cost, checks (generated by dag.py report)", "",
         "Disk (GiB): " + ", ".join(f"{k} {v:.1f}" for k, v in disk.items()),
         f"Roll: {rec['roll']['card_h']:.2f} card-h (3 stacks a card).", "Checks: " + ", ".join(f"{k} {v}" for k, v in rec["checks"].items())]
    (RES / "build.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, args in dict(lists=["root"], t1=["cfg_glob", "limit:int", "workers:int"], t2=["workers:int"],
                           teach=["rows", "cfg_glob", "out", "chunk_list?", "workers:int", "stale_only:int"], chk1=["root"], label=["root", "pool"],
                           r1in=["root", "pool", "fwd", "tag", "control"], mfix=["root", "pool", "ref"],
                           weights=["pool", "base", "k_on:int", "k_dag:int", "out", "src?"], smoke=["tag", "wfile", "k_dag:int"],
                           verdict=["json", "md", "read+"], pcs=["pcs", "json", "md"], report=["root"]).items():
        p = sp.add_parser(name)
        for x in args:
            k, _, t = x.partition(":")
            if k.endswith("+"):
                p.add_argument(f"--{k[:-1].replace('_', '-')}", dest=k[:-1], nargs="+", required=True)
            elif t == "int":
                p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int, default=0)
            else:
                p.add_argument(f"--{k.rstrip('?').replace('_', '-')}", dest=k.rstrip("?"), required=not k.endswith("?"), default="")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
