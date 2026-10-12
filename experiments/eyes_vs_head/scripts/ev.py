#!/usr/bin/env python3
"""EYES-VS-HEAD (plans/2026-10-12-eyes-vs-head-prereg.md): when the driver scores zero on the stored 18-decision held rollouts, did the frozen
openpilot model's own perception heads (lane lines, road edges, lead) read the scene right while the plan still left the road ("eyes right,
head wrong"), or were the heads wrong on the reconstructed images ("eyes wrong")?

  msgs    (box, AlpaSim venv)  ev.py msgs
      the driver-side messages of every zero-score rollout of LABELS (rollout.asl, CAM_F0 only; c1_extract.one_msgs)
      -> $DATA_DIR/runs/eyes_vs_head/msgs/<label>/<scene>.pkl
  replay  (box, envs/op-train, one GPU: a pool job)  ev.py replay [--groups ...]
      every decision again through the real driver class (sh30_driver.Driver, the label's own checkpoint, held-read settings, not compiled);
      per decision the served plan, anchor, camera, command and the raw heads of the served checkpoint (FT) and of the shipped policy weights
      without the adapter on the same vision tokens (P0); group ctrl = col1's P2H10-F-s0 control re-run (200 public scenes, all asl kept)
      -> $DATA_DIR/runs/eyes_vs_head/replay/<group>.pkl
  read    (box, AlpaSim venv)  ev.py read --replay DIR
      the run's scene loader (map lane edges, road-area boundary, logged path, logged traffic) against the heads: per-decision table,
      per-event classification, gates, BEV payloads -> $DATA_DIR/runs/eyes_vs_head/read/<ts>/ dec.parquet, ev.parquet, gates.json, bev.pkl
  report  (Mac, .venv)  ev.py report --read DIR
      the registered tables, lines and verdicts, the per-event table, figures -> results/, figs/

Frames: rollout local frame; ego = rear axle, x forward, y LEFT. openpilot heads: x = X_IDXS ahead of the camera, y RIGHT-positive; a head
point (X, y) is (cam_x + X, cam_y - y) in the ego frame (G-sign checks it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("msgs", "read"):     # one thread per worker (zeros_mech: 64 workers x BLAS / polars / rayon pools
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):  # hit the limit)
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/anchor_offset/scripts"), str(REPO / "experiments/alpasim/scripts"),
                str(REPO / "experiments/alpasim/lib"), str(REPO / "experiments/op_parity/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = DATA / "runs/eyes_vs_head"
RUNS = {"YR10m10-F-s0": DATA / "runs/alpasim_long/held/h18/YR10m10-F-s0", "YR10m10-F-s1": DATA / "runs/alpasim_long/held/h18/YR10m10-F-s1",
        **{f"{f}-L18-s{s}": DATA / f"runs/onpol_long/read/{f}-L18-s{s}/run" for f in ("YS", "YS2G") for s in (0, 1)}}
CTRL_MSGS, CTRL_TAG = DATA / "runs/alpasim/col1/lead/msgs/ctrl", "P2H10-F-s0"
ZM = DATA / "runs/zeros_mech/collect/20261012-155215"
CFG = RUNS["YR10m10-F-s0"] / "generated-user-config-0.yaml"          # the after160 rebuilt cache: every public scene, one local frame
T0, DT = 17_000, 500_000
X_IDXS = 192.0 * (np.arange(33) / 32.0) ** 2
DS = (10.0, 20.0, 40.0)
HEADS = ("lane_lines", "lane_lines_prob", "road_edges", "lead", "lead_prob")
MERGE, HALF, R_RIGHT, R_WRONG = 0.3, 15.0, 0.5, 1.0                  # registered: crossings merged within 0.3 m, |y| <= 15 m, 0.5 / 1.0 m
LEAD_POS, LEAD_V = 2.0, 2.0


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


def zeros_of(run):
    """-> {scene: rollout dict} of the zero-score rollouts whose asl was kept."""
    out = {}
    for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]:
        if r.get("score") is not None and float(r["score"]) == 0 and glob.glob(f"{run}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
            out[r["clipgt_id"]] = r
    return out


# ---------------------------------------------------------------- msgs (AlpaSim venv)
def _msgs(job):
    import c1_extract as X
    return X.one_msgs(job)


def msgs(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    held = splits.load("alpasim_nuplan/onpol-held")
    with Run("eyes_vs_head", "msgs", config=vars(a)) as run:
        run.use_split(held)
        jobs, n_zero = [], {}
        for lab, D in RUNS.items():
            n_all = sum(1 for r in json.loads((D / "aggregate/results-summary.json").read_text())["rollouts"]
                        if r["clipgt_id"] in set(held.members) and r.get("score") is not None and float(r["score"]) == 0)
            z = {s: r for s, r in zeros_of(D).items() if s in set(held.members)}
            n_zero[lab] = dict(zeros=n_all, with_asl=len(z))
            (OUT / "msgs" / lab).mkdir(parents=True, exist_ok=True)
            jobs += [(str(D), s, str(OUT / "msgs" / lab)) for s in sorted(z)]
        run.info(f"{len(jobs)} zero rollouts with asl: {n_zero}")
        res = par.pmap(_msgs, jobs, workers=a.workers or min(32, n_cpus()), run=run, desc="msgs",
                       skip=lambda j: Path(j[2], f"{j[1]}.pkl").exists())
        g = dict(n_zero=n_zero, jobs=len(jobs), errors=len(res.errors))
        g["G_msg"] = dict(share=1 - len(res.errors) / max(len(jobs), 1), asl_share=sum(v["with_asl"] for v in n_zero.values()) /
                          max(sum(v["zeros"] for v in n_zero.values()), 1))
        g["G_msg"]["pass"] = bool(g["G_msg"]["share"] >= 0.99 and g["G_msg"]["asl_share"] >= 0.99)
        (OUT / "msgs/gates.json").write_text(json.dumps(g, indent=1))
        run.summary.update(g)


# ---------------------------------------------------------------- replay (envs/op-train, GPU)
class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def replay(a):
    import tempfile
    import torch
    import pp_train as T
    import sh30_driver as D
    from jevdrive.run import Run
    I, Z = D.C.I, D.C.Z
    groups = a.groups or [*RUNS, "ctrl"]
    (OUT / "replay").mkdir(parents=True, exist_ok=True)
    with Run("eyes_vs_head", "replay", config=vars(a)) as run:
        p0 = T.load_pmodel("P0", torch.device("cuda"))
        pb, ctx = D.egodriver_pb2, Ctx()
        for g in groups:
            f = OUT / "replay" / f"{g}.pkl"
            if f.exists() and not a.force:
                continue
            tag, md = (CTRL_TAG, CTRL_MSGS) if g == "ctrl" else (g, OUT / "msgs" / g)
            core = D.C.Core(tag, "cuda", "backwarp", 1.0, "gpu")
            logd = Path(tempfile.mkdtemp())
            drv = D.Driver(core, logd, 0, False)
            sl, cap, last = core.model.net.slices, {}, {}
            core.model.register_forward_hook(lambda m, args, out: cap.update(H=args[0], ego=args[1], tc=args[2], out=out))
            orig = core.plan

            def plan(*x, _o=orig, **k):
                last["o"] = _o(*x, **k)
                return last["o"]
            core.plan = plan
            out, scenes = {}, sorted(p.stem for p in md.glob("*.pkl"))[: a.limit or None]
            for i, scene in enumerate(run.tqdm(scenes, desc=g)):
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
                        except RuntimeError as e:                       # a decision the driver aborted when served (logged, skipped)
                            run.info(f"{g} {scene} t {req.time_now_us}: {e}")
                            continue
                        with torch.no_grad():
                            o0 = p0(cap["H"], cap["ego"], cap["tc"]).float()[0]
                        ft = cap["out"].float()[0]
                        cam = np.asarray(drv.sessions[uuid].cam["t"], np.float64)
                        m0 = o0[core.pi].reshape(33, 15).cpu().numpy()
                        r = dict(now=int(req.time_now_us), poses=np.asarray(last["o"]["poses"], np.float32).copy(),
                                 poses_p0=I.to_rear(m0[:, 0:3], m0[:, 11], I.T_IDXS, cam[:2], Z.T_OUT, "lever").astype(np.float32),
                                 cmd=int(np.argmax(drv.sessions[uuid].cmd)), cam=cam.copy())
                        for name, v in (("FT", ft), ("P0", o0)):
                            r[name] = {k: v[sl[k]].cpu().numpy().astype(np.float32) for k in HEADS}
                        rec.append(r)
                drv.sessions.pop(uuid, None)
                out[scene] = rec
            drv.out.flush()
            drives = [json.loads(x) for x in open(logd / "drive.jsonl") if '"kind": "drive"' in x]
            flat = [r for s in scenes for r in out[s]]
            assert len(drives) == len(flat), f"{g}: {len(drives)} driver records for {len(flat)} decisions"
            for r, d in zip(flat, drives):
                assert int(d["now"]) == r["now"]
                r.update(anchor=np.asarray(d["anchor"], float), k=int(d["k"]), t0=int(d["t0"]))
            pickle.dump(out, open(f, "wb"), protocol=4)
            run.info(f"replay {g}: {len(out)} scenes, {len(flat)} decisions")
            del core, drv
            torch.cuda.empty_cache()
        run.summary.update(groups=groups)


# ---------------------------------------------------------------- read (AlpaSim venv)
def segs_of(lines):
    """Polylines [(n, 2)] -> segment endpoints (m, 2) x 2."""
    A = [np.asarray(p, float)[:-1, :2] for p in lines if len(p) > 1]
    B = [np.asarray(p, float)[1:, :2] for p in lines if len(p) > 1]
    return (np.concatenate(A), np.concatenate(B)) if A else (np.zeros((0, 2)), np.zeros((0, 2)))


def cross(SA, SB, a, xs):
    """Ego-frame y (sorted, merged within MERGE, |y| <= HALF) where the segments cross the line x = xs of the frame of pose a."""
    if not len(SA):
        return np.zeros(0)
    P, Q = in_ego(a, SA), in_ego(a, SB)
    m = ((P[:, 0] - xs) * (Q[:, 0] - xs) <= 0) & (np.abs(Q[:, 0] - P[:, 0]) > 1e-9)
    y = P[m, 1] + (xs - P[m, 0]) / (Q[m, 0] - P[m, 0]) * (Q[m, 1] - P[m, 1])
    y = np.sort(y[np.abs(y) <= HALF])
    if len(y) < 2:
        return y
    grp = np.r_[0, np.cumsum(np.diff(y) > MERGE)]
    return np.array([y[grp == i].mean() for i in range(grp[-1] + 1)])


def near(ys, y):
    """-> (|error|, index) of the nearest crossing; (nan, -1) if none."""
    if not len(ys):
        return np.nan, -1
    i = int(np.argmin(np.abs(ys - y)))
    return float(abs(ys[i] - y)), i


def heads(raw, cam):
    """Raw head slices -> ego-frame (left-positive) lane lines (4, 33), probs (4,), road edges (2, 33), lead (x, y, v) in the ego frame, lead p."""
    ll = raw["lane_lines"][: 264].reshape(4, 33, 2)[..., 0]
    re = raw["road_edges"][: 132].reshape(2, 33, 2)[..., 0]
    pr = 1 / (1 + np.exp(-np.clip(raw["lane_lines_prob"], -11, None)))[1::2]
    ld = raw["lead"][: 72].reshape(3, 6, 4)[0, 0]
    lp = float(1 / (1 + np.exp(-max(float(raw["lead_prob"][0]), -11))))
    return dict(ll=cam[1] - ll, pr=pr, re=cam[1] - re, lead=np.array([cam[0] + ld[0], cam[1] - ld[1], ld[2]]), lp=lp)


def at(line, X, cam_x):
    """Head line (33,) at ego-frame x (array) -> y; nan outside 0 .. 192 m ahead of the camera."""
    X = np.asarray(X, float) - cam_x
    y = np.interp(X, X_IDXS, line)
    return np.where((X >= 0) & (X <= X_IDXS[-1]), y, np.nan)


class Geo:
    def __init__(self, ds, bbox):
        import shapely
        from shapely.ops import unary_union
        from trajdata.maps.vec_map_elements import MapElementType as MT
        lo, hi = bbox
        inb = lambda P: bool(((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1])).any())  # noqa: E731
        self.edges, self.lanes = [], []
        for ln in ds.map.elements.get(MT.ROAD_LANE, {}).values():
            L, R, C = (np.asarray(e.points, float)[:, :2] if e is not None else None for e in (ln.left_edge, ln.right_edge, ln.center))
            if not any(p is not None and inb(p) for p in (L, R, C)):
                continue
            self.edges += [p for p in (L, R) if p is not None]
            if L is not None and R is not None and C is not None and len(C) > 1:
                poly = shapely.Polygon(np.r_[L, R[::-1]]).buffer(0)
                self.lanes.append((poly, C))
        areas = []
        for ra in ds.map.elements.get(MT.ROAD_AREA, {}).values():
            E = np.asarray(ra.exterior_polygon.points, float)[:, :2]
            if len(E) > 2 and inb(E):
                areas.append(shapely.Polygon(E, [np.asarray(h.points, float)[:, :2] for h in ra.interior_holes]).buffer(0))
        b = unary_union(areas).boundary if areas else None
        geoms = [] if b is None else (list(b.geoms) if hasattr(b, "geoms") else [b])
        self.road = [np.asarray(g.coords) for g in geoms]
        self.SE, self.SR = segs_of(self.edges), segs_of(self.road)
        self.tree = shapely.STRtree([p for p, _ in self.lanes]) if self.lanes else None
        self.sh = shapely

    def junction(self, x, y):
        """Rear axle in two lane polygons whose centre lines differ by > 30 deg."""
        if self.tree is None:
            return False
        pt = self.sh.Point(x, y)
        yaws = []
        for i in self.tree.query(pt, predicate="within"):
            C = self.lanes[int(i)][1]
            j = int(np.argmin(np.hypot(C[:, 0] - x, C[:, 1] - y)))
            j = min(j, len(C) - 2)
            yaws.append(np.arctan2(C[j + 1, 1] - C[j, 1], C[j + 1, 0] - C[j, 0]))
        yaws = np.array(yaws)
        return bool(len(yaws) > 1 and np.abs(wrap(yaws[:, None] - yaws[None])).max() > np.radians(30))


def log_y(LP, a, xs):
    """Ego-frame y of the logged path where it crosses x = xs (the first crossing ahead); nan if none."""
    P = in_ego(a, LP)
    m = np.flatnonzero((P[:-1, 0] - xs) * (P[1:, 0] - xs) <= 0)
    if not len(m):
        return np.nan
    i = m[0]
    dx = P[i + 1, 0] - P[i, 0]
    return float(P[i, 1] if abs(dx) < 1e-9 else P[i, 1] + (xs - P[i, 0]) / dx * (P[i + 1, 1] - P[i, 1]))


def signed_off(LP, p):
    """Signed lateral offset (left +) of point p from polyline LP."""
    d = LP[1:] - LP[:-1]
    t = np.clip(((p - LP[:-1]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    q = LP[:-1] + t[:, None] * d
    i = int(np.argmin(np.hypot(*(q - p).T)))
    return float(np.sign(d[i, 0] * (p[1] - LP[i, 1]) - d[i, 1] * (p[0] - LP[i, 0])) * np.hypot(*(q[i] - p)))


def obj_at(o, t):
    T, P, Y = o["T"], o["P"], o["Y"]
    if t < T[0] - 1 or t > T[-1] + 1:
        return None
    p = np.array([np.interp(t, T, P[:, 0]), np.interp(t, T, P[:, 1])])
    t1, t2 = max(t - 2.5e5, T[0]), min(t + 2.5e5, T[-1])
    v = (np.array([np.interp(t2, T, P[:, 0]), np.interp(t2, T, P[:, 1])]) - np.array([np.interp(t1, T, P[:, 0]), np.interp(t1, T, P[:, 1])])) / max((t2 - t1) * 1e-6, 1e-3)
    return p, float(np.interp(t, T, Y)), v


def read_unit(job):
    """(group, scene, replay records, event dict or None) -> dict(dec=[...], bev={k: payload})."""
    g, scene, recs, ev = job
    from ao_e1 import loader
    ds = loader(str(CFG)).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
    LP = np.vstack([Pg, Pg[-1] + 60.0 * np.array([np.cos(Yg[-1]), np.sin(Yg[-1])])])
    A = np.array([r["anchor"] for r in recs]) if recs else np.zeros((0, 3))
    allp = np.vstack([Pg, A[:, :2]]) if len(A) else Pg
    G = Geo(ds, (allp.min(0) - 90, allp.max(0) + 90))
    loc = str(getattr(ds.scene, "location", "") or "")
    hit = None
    if ev and ev.get("hit_id"):
        for oid in ds.traffic_objects.keys():
            if str(oid) == ev["hit_id"]:
                o = ds.traffic_objects[oid]
                t_ = o.trajectory
                hit = dict(T=np.asarray(t_.timestamps_us, float), P=np.asarray(t_.positions, float)[:, :2], Y=np.unwrap(np.asarray(t_.yaws, float)),
                           L=float(o.aabb.x), W=float(o.aabb.y))
    dec, bev = [], {}
    for r in recs:
        a, k, cam = np.asarray(r["anchor"], float), int(r["k"]), np.asarray(r["cam"], float)
        t = T0 + DT * k
        row = dict(group=g, scene=scene, loc=loc, k=k, t0=r["t0"], cmd=r["cmd"], x=a[0], y=a[1], yaw=a[2],
                   frame_err=float(np.hypot(a[0] - np.interp(r["t0"], Tg, Pg[:, 0]), a[1] - np.interp(r["t0"], Tg, Pg[:, 1]))),
                   e_log=signed_off(LP, a[:2]), junc=G.junction(a[0], a[1]), cam_x=cam[0], cam_y=cam[1])
        B = {d: cross(*G.SE, a, cam[0] + d) for d in DS}
        R = {d: cross(*G.SR, a, cam[0] + d) for d in DS}
        for d in DS:
            row[f"nB{int(d)}"], row[f"nR{int(d)}"] = len(B[d]), len(R[d])
            row[f"ylog{int(d)}"] = log_y(LP, a, cam[0] + d)
        P = np.asarray(r["poses"], float)
        sel = (P[:, 0] - cam[0] >= 0) & (P[:, 0] - cam[0] <= 40)
        for H in ("P0", "FT"):
            h = heads(r[H], cam)
            for i in range(4):
                row[f"{H}_p{i}"] = float(h["pr"][i])
            for d in DS:
                yi = [float(at(h["ll"][i], cam[0] + d, cam[0])) for i in range(4)]
                for i in range(4):
                    row[f"{H}_y{i}_{int(d)}"] = yi[i]
                    row[f"{H}_e{i}_{int(d)}"], _ = near(B[d], yi[i])
                    row[f"{H}_s{i}_{int(d)}"] = float(yi[i] - B[d][near(B[d], yi[i])[1]]) if len(B[d]) else np.nan
                for j in range(2):
                    ye = float(at(h["re"][j], cam[0] + d, cam[0]))
                    row[f"{H}_r{j}_{int(d)}"] = ye
                    row[f"{H}_re{j}_{int(d)}"], _ = near(R[d], ye)
            # ego lane framed at 10 m: both lines matched within 0.5 m to adjacent crossings
            y1, y2 = row[f"{H}_y1_10"], row[f"{H}_y2_10"]
            e1, i1 = near(B[10.0], y1)
            e2, i2 = near(B[10.0], y2)
            row[f"{H}_framed"] = bool(h["pr"][1] > 0.5 and h["pr"][2] > 0.5 and e1 <= R_RIGHT and e2 <= R_RIGHT and abs(i1 - i2) == 1)
            # the served plan against this head's lines and edges (left-positive: line 1 / edge 0 on the left)
            X = P[sel, 0]
            if len(X):
                l1, l2 = at(h["ll"][1], X, cam[0]), at(h["ll"][2], X, cam[0])
                r0, r1 = at(h["re"][0], X, cam[0]), at(h["re"][1], X, cam[0])
                py = P[sel, 1]
                row[f"{H}_xL"] = bool(np.nanmax(py - l1) > 0)
                row[f"{H}_xR"] = bool(np.nanmax(l2 - py) > 0)
                row[f"{H}_xeL"] = bool(np.nanmax(py - r0) > 0)
                row[f"{H}_xeR"] = bool(np.nanmax(r1 - py) > 0)
                row[f"{H}_mL"], row[f"{H}_mR"] = float(np.nanmax(py - l1)), float(np.nanmax(l2 - py))
            else:
                row.update({f"{H}_{q}": False for q in ("xL", "xR", "xeL", "xeR")}, **{f"{H}_mL": np.nan, f"{H}_mR": np.nan})
            row[f"{H}_lp"], (row[f"{H}_lx"], row[f"{H}_ly"], row[f"{H}_lv"]) = h["lp"], h["lead"]
            if H == "P0":
                hp = h
        if hit is not None:
            s = obj_at(hit, t)
            if s is not None:
                p, yaw, v = s
                c, si = np.cos(yaw), np.sin(yaw)
                C = p + np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * np.array([hit["L"] / 2, hit["W"] / 2]) @ np.array([[c, si], [-si, c]])
                Ce = in_ego(a, C)
                # nearest point of the box (edges sampled) to the camera
                pts = np.concatenate([Ce[i] + np.linspace(0, 1, 21)[:, None] * (Ce[(i + 1) % 4] - Ce[i]) for i in range(4)])
                q = pts[np.argmin(np.hypot(pts[:, 0] - cam[0], pts[:, 1] - cam[1]))]
                row.update(hit_x=q[0], hit_y=q[1], hit_v=float(v @ np.array([np.cos(a[2]), np.sin(a[2])])), hit_dpsi=float(np.degrees(wrap(yaw - a[2]))))
        dec.append(row)
        if ev is not None:                                  # BEV payload of every decision of a zero rollout (the report picks)
            Le = [in_ego(a, p) for p in G.edges]
            Le = [p for p in Le if ((np.abs(p[:, 1]) < 25) & (p[:, 0] > -15) & (p[:, 0] < 60)).any()]
            Rd = [in_ego(a, p) for p in G.road]
            Rd = [p for p in Rd if ((np.abs(p[:, 1]) < 25) & (p[:, 0] > -15) & (p[:, 0] < 60)).any()]
            Lg = in_ego(a, LP)
            Lg = Lg[(Lg[:, 0] > -15) & (Lg[:, 0] < 60) & (np.abs(Lg[:, 1]) < 25)]
            xs = cam[0] + X_IDXS[X_IDXS <= 60]
            bev[k] = dict(edges=[p.astype(np.float32) for p in Le], road=[p.astype(np.float32) for p in Rd], log=Lg.astype(np.float32),
                          plan=P.astype(np.float32), xs=xs, ll=hp["ll"][:, : len(xs)], pr=hp["pr"], re=hp["re"][:, : len(xs)], cam=cam)
    return dict(dec=dec, bev=bev, scene=scene, group=g)


def read(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    from ao_e1 import drive_records
    held = splits.load("alpasim_nuplan/onpol-held")
    Rz = pd.read_parquet(ZM / "per_rollout.parquet")
    with Run("eyes_vs_head", "read", config=vars(a)) as run:
        run.use_split(held)
        jobs, grep = [], []
        for f in sorted(Path(a.replay).glob("*.pkl")):
            g = f.stem
            rp = pickle.load(open(f, "rb"))
            stored = drive_records(RUNS[g]) if g in RUNS else {}
            for scene, recs in sorted(rp.items()):
                ev = None
                if g in RUNS:
                    z = Rz[(Rz.label == g) & (Rz.scene == scene)]
                    if len(z):
                        z = z.iloc[0]
                        ev = dict(type=z.type, k_ev=int(z.k_ev), k_c=int(z.k_c), hit_id=(z.hit_id if isinstance(z.get("hit_id"), str) else ""),
                                  k_end=int(z.k_end))
                    by_k = {int(r["k"]): r for r in stored.get(scene, [])}
                    for r in recs:
                        s = by_k.get(int(r["k"]))
                        grep.append(dict(group=g, scene=scene, k=int(r["k"]), d4=float(np.hypot(*(np.asarray(s["poses"], float)[7, :2] - r["poses"][7, :2])))
                                         if s else np.nan))
                else:
                    ev = None
                jobs.append((g, scene, recs, ev))
        run.info(f"{len(jobs)} rollouts")
        res = par.pmap(read_unit, jobs, workers=a.workers or min(64, n_cpus()), run=run, desc="rollouts")
        out = Path(run.dir)
        vals = [v for v in res.values if v]
        D = pd.DataFrame([x for v in vals for x in v["dec"]])
        evs = {(j[0], j[1]): j[3] for j in jobs if j[3]}
        E = pd.DataFrame([dict(group=g, scene=s, **e) for (g, s), e in evs.items()])
        D.to_parquet(out / "dec.parquet", index=False)
        E.to_parquet(out / "ev.parquet", index=False)
        pd.DataFrame(grep).to_parquet(out / "grep.parquet", index=False)
        pickle.dump({(v["group"], v["scene"]): v["bev"] for v in vals if v["bev"]}, open(out / "bev.pkl", "wb"), protocol=4)
        Gr = pd.DataFrame(grep)
        f0 = D[D.k == 0].groupby(["group", "scene"]).frame_err.first()
        G = dict(rollouts=len(jobs), errors=len(res.errors), err_first=[res.errors[i].strip().splitlines()[-1][:300] for i in sorted(res.errors)[:5]],
                 G_rep=dict(decisions=int(Gr.d4.notna().sum()), share=float((Gr.d4 <= 0.10).mean()), median=float(Gr.d4.median()),
                            missing=int(Gr.d4.isna().sum())),
                 G_frame=dict(rollouts=int(len(f0)), share=float((f0 <= 0.10).mean()), max=float(f0.max())))
        G["G_rep"]["pass"] = bool(G["G_rep"]["share"] >= 0.95)
        G["G_frame"]["pass"] = bool(G["G_frame"]["share"] >= 0.99)
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.summary.update(G)
        run.info(json.dumps(G, default=float))


# ---------------------------------------------------------------- report (Mac)
def report(a):
    import ev_report
    ev_report.main(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("msgs")
    p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("replay")
    p.add_argument("--groups", nargs="*"), p.add_argument("--limit", type=int, default=0), p.add_argument("--force", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--replay", default=str(OUT / "replay")), p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--read", required=True), p.add_argument("--out", default=str(REPO / "experiments/eyes_vs_head/results")),
    p.add_argument("--figs", default=str(REPO / "experiments/eyes_vs_head/figs"))
    a = ap.parse_args()
    dict(msgs=msgs, replay=replay, read=read, report=report)[a.cmd](a)
