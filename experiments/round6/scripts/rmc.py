#!/usr/bin/env python3
"""ROUND6 ROUTE-CURV: is the route input a lateral intent in disguise? Diagnosis, stored reads only, no GPU (pool CPU jobs, rmc.sh).

What the model is fed as route (NTV0 / R3MV / R5LC: sidecar route_mem.json without `order` / `anchor` = the PROJECTED layout of
experiments/route_mem/lib/rmem.py, plus the NAVSIM command in the ego row): rmem.window on the merged route messages, 40 points at route-arc
offsets -6 .. +72 m (2 m) from the ego's projection, each [valid, x / 20, y / 20, cos h, sin h] in the current rear-axle frame (x forward,
y left), and ego[1:4] = command_from_route (first waypoint >= 5 m ahead, i.e. the message's first point ~42 m ahead: y > 2 m LEFT, y < -2 m
RIGHT, else STRAIGHT). The route messages are AlpaSim's RouteGeneratorMap: the log's poses projected onto the centre line of a lane chosen
per pose among the lanes within 4 m and pi / 8 of the heading (the candidate that persists longest; kept while it stays a candidate).

  extract (box, repo .venv)   the five stored PAI 281 ACC + GAP reads' drive.jsonl decisions (t0, anchor, command, fed rmem, model plan,
                              v0) + facts + the tax extract's first controller time -> $O/dec.pkl
  scenes  (box, AlpaSim venv) per scene, the NuRec vector map: for every state the ego's current lane (nearest lane centre within 4 m and
                              pi / 8) chained forward along the successors nearest the route -> Frenet reference; the route's lateral offset
                              at 10 / 20 / 40 m of route arc split into pose (ego offset + heading against the lane), curvature (the lane
                              centre's own bend) and lane offset (route point minus lane centre); the 4 s plan / label against both.
                              States: human log every 0.5 s (v >= 2 m/s, 4 s future) on the 281 test scenes and the 719 curated train
                              scenes (the route = the generator's polyline windowed by rmem.window), every decision of the five reads (the
                              route = the fed rmem) -> $O/hum_{test,train}.parquet, $O/dec.parquet
  rows    (box, repo .venv)   map-free training-row check: PAI human pools (on1_PAIH / 05 / 15) vs navtrain: command shares and the label's
                              4 s point against the fed route at the label's arc -> $O/rows.json
  report  (box, repo .venv)   tables -> experiments/round6/results/route_mem_curv.md + route_mem_curv_*.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "lib"), str(REPO / "experiments/route_mem/lib"), str(REPO / "experiments/route_follow/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round6/route_curv"
RES = REPO / "experiments/round6/results"
READS = {"NTV0": ("NTV0-L18-s0", "defect_fix/pai_accgap/NTV0-L18-s0"), "R3MV-s0": ("R3MV-L18-s0", "round5/r5_281/pai281/R3MV-L18-s0-ag"),
         "R3MV-s1": ("R3MV-L18-s1", "round5/r5/pai281/R3MV-L18-s1-ag"), "MVLC-s0": ("R5LC-L18-s0", "round5/r5/pai281/R5LC-L18-s0-ag"),
         "MVLC-s1": ("R5LC-L18-s1", "round5/r5/pai281/R5LC-L18-s1-ag")}
MV = [k for k in READS if k != "NTV0"]
TCFG = (f"{DATA}/runs/round5/s3/roll/A/runs/*/sim/generated-user-config-0.yaml,"
        f"{DATA}/runs/round5/d3/roll/A/runs/*/sim/generated-user-config-0.yaml")
TSV = REPO / "experiments/round5/results/s3_scenes.tsv"
S_AT = (10, 20, 40)                         # route arcs read out (m)
S_CMD = 42                                  # the slot nearest the route message's first point (command_from_route's target)
LANE_D, LANE_H = 4.0, np.pi / 8             # current-lane candidates (the generator's own radius and heading tolerance)
AHEAD = 110.0                               # lane chain length ahead of the ego (m)
LEADS = (3.0, 2.0, 1.0)


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


def _arr(x, dt=np.float32):
    return np.asarray(json.loads(x) if isinstance(x, str) else x, dt)


# ---------------------------------------------------------------- extract (box, .venv)
def _one_log(f):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(f):
        if '"infer": true' not in line or '"kind": "drive"' not in line:
            continue
        r = json.loads(line)
        if "rmem" not in r:
            continue
        fx = r.get("fix") or {}
        S = out[r["scene"]]
        S["t0"].append(int(r["t0"]))
        S["anchor"].append(np.asarray(r["anchor"], np.float64)[:3])
        S["cmd"].append(int(r["cmd"]))
        S["rm"].append(_arr(r["rmem"]).reshape(40, 5))
        S["pm"].append(_arr(r.get("poses_model") or r["poses"])[:8, :3])
        S["v0"].append(float(fx.get("v0", _arr(r["ego"])[4] * 10)))
    return {s: {k: np.asarray(v) for k, v in d.items()} for s, d in out.items()}


def cmd_extract(a):
    import pandas as pd
    from jevdrive import par
    out = {}
    for k, (tag, root) in READS.items():
        fs = sorted(glob.glob(f"{DATA}/runs/{root}/runs/*/driver/drive.jsonl"))
        res = par.pmap(_one_log, fs, workers=min(len(fs), a.workers), desc=f"logs {k}")
        res.raise_if_failed()
        D = {}
        for p in res.values:
            for s, d in p.items():
                if s not in D or len(d["t0"]) > len(D[s]["t0"]):
                    D[s] = d
        for s, d in D.items():
            i = np.argsort(d["t0"], kind="stable")
            D[s] = {x: v[i] for x, v in d.items()}
        F = pd.read_csv(DATA / f"runs/pai_fix/facts/{tag}-281ag.csv").set_index("scene")
        tax = pickle.load(open(DATA / f"runs/pai_fix/tax/{tag}-281ag.pkl", "rb"))["rolls"]
        c0 = {o["scene"]: float(o["ctrl"]["t"][0]) for o in tax if "ctrl" in o}
        cfg = {}
        for f in glob.glob(f"{DATA}/runs/{root}/runs/*/sim/generated-user-config-0.yaml"):
            for s in re.findall(r"scene_id: (clipgt-[0-9a-f-]+)", open(f).read()):
                cfg.setdefault(s, f)
        out[k] = dict(tag=tag, dec=D, facts=F.reset_index().to_dict("records"), c0=c0, cfg=cfg)
        print(k, len(fs), "logs", len(D), "scenes", sum(len(d["t0"]) for d in D.values()), "decisions", len(c0), "ctrl", flush=True)
    O.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(O / "dec.pkl", "wb"), protocol=4)


# ---------------------------------------------------------------- geometry (numpy; the map calls need the AlpaSim venv)
def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def in_ego(p, X):
    c, s = np.cos(p[2]), np.sin(p[2])
    d = np.asarray(X, np.float64).reshape(-1, 2) - np.asarray(p[:2], np.float64)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def to_local(p, X):
    c, s = np.cos(p[2]), np.sin(p[2])
    X = np.asarray(X, np.float64).reshape(-1, 2)
    return np.c_[p[0] + c * X[:, 0] - s * X[:, 1], p[1] + s * X[:, 0] + c * X[:, 1]]


class Lanes:
    """The scene's vector map: current lane of a pose and its forward chain (successor nearest the route at a fork) -> dag.Ref (cached)."""

    def __init__(self, vm, route_xy):
        self.vm, self.R, self.cache = vm, np.asarray(route_xy, np.float64)[:, :2], {}

    def _dist_route(self, X):
        d = ((X[:, None, :] - self.R[None, ::2]) ** 2).sum(-1).min(1)
        return float(np.sqrt(d).mean())

    def current(self, pose, z):
        """-> (lane, n candidates, lateral distance of the pose to its centre) or (None, 0, nan)."""
        c = self.vm.get_current_lane(np.r_[pose[0], pose[1], z, pose[2]], max_dist=LANE_D, max_heading_error=LANE_H)
        if not c:
            return None, 0, np.nan
        best, bd = None, np.inf
        for ln in c:
            C = np.asarray(ln.center.points, np.float64)[:, :2]
            a, b = C[:-1], C[1:]
            ab = b - a
            t = np.clip(((pose[:2] - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
            d = float(np.hypot(*(a + t[:, None] * ab - pose[:2]).T).min())
            if d < bd:
                best, bd = ln, d
        return best, len(c), bd

    def ref(self, lane, pose):
        import dag
        key = lane.id
        if key in self.cache:
            return self.cache[key]
        C = np.asarray(lane.center.points, np.float64)[:, :2]
        pts, seen, cur = [C], {lane.id}, lane
        total = float(np.hypot(*np.diff(C, axis=0).T).sum())
        need = total + AHEAD                                                    # AHEAD beyond the first lane's end, wherever the ego is on it
        while total < need and cur.next_lanes:
            nx = [self.vm.get_road_lane(i) for i in sorted(cur.next_lanes) if i not in seen]
            if not nx:
                break
            if len(nx) > 1:
                nx.sort(key=lambda ln: self._dist_route(np.asarray(ln.center.points, np.float64)[:12, :2]))
            cur = nx[0]
            seen.add(cur.id)
            X = np.asarray(cur.center.points, np.float64)[:, :2]
            pts.append(X[1:] if np.hypot(*(X[0] - pts[-1][-1])) < 0.5 else X)
            total += float(np.hypot(*np.diff(X, axis=0).T).sum())
        P = np.concatenate(pts)
        u = P[1] - P[0]
        u = u / max(np.hypot(*u), 1e-9)
        P = np.vstack([P[0] - 15.0 * u, P])                                    # 15 m behind the lane start: the projection never falls off
        r = dag.Ref(P)
        self.cache[key] = r
        return r


def frenet(ref, X, H):
    """local points X (n, 2), headings H (n,) -> (station, signed lateral, left +) per point (nan without a heading-consistent point)."""
    out = np.full((len(X), 2), np.nan)
    for i, (x, h) in enumerate(zip(X, H)):
        pr = ref.project(np.r_[x, h])
        if pr is not None:
            out[i] = pr[:2]
    return out


def arc_interp(F, a):
    """fed point block F (40, 5) (ego frame), route arc a (m) from the ego's projection -> ego-frame (x, y), heading (rad) or nans."""
    import rmem as RM
    v = F[:, 0] > 0.5
    if v.sum() < 2 or a < RM.OFFS[v][0] or a > RM.OFFS[v][-1]:
        return np.array([np.nan, np.nan]), np.nan
    o = RM.OFFS[v]
    return np.array([np.interp(a, o, F[v, 1] * 20), np.interp(a, o, F[v, 2] * 20)]), float(np.interp(a, o, np.arctan2(F[v, 4], F[v, 3])))


def state_metrics(L, pose, z, F, fut=None):
    """pose (x, y, yaw) local; F (40, 5) the route block in the pose's frame; fut = (8, >= 2) ego-frame plan / label (4 s point used).
    -> dict of the read-outs (module docstring of `scenes`)."""
    import rmem as RM
    r = dict()
    ln, nc, dl = L.current(pose, z)
    r.update(n_lanes=nc, lane_d=dl)
    if ln is None:
        return r
    ref = L.ref(ln, pose)
    pr = ref.project(np.asarray(pose, np.float64))
    if pr is None:
        return r
    s_e, d_e, psi = pr
    r.update(d_ego=d_e, psi_deg=float(np.degrees(psi)), lane_id=str(ln.id))
    x0, y0, t0 = ref.at(s_e)
    r["dth40"] = float(np.degrees(np.interp(s_e + 40, ref.s, ref.th) - t0))
    r["kap0"] = float(np.interp(s_e, ref.s, ref.kap))
    for S in list(S_AT) + [S_CMD]:
        i = int(np.flatnonzero(RM.OFFS == S)[0])
        if F[i, 0] < 0.5:
            continue
        q = F[i, 1:3].astype(np.float64) * 20
        hq = pose[2] + np.arctan2(F[i, 4], F[i, 3])
        ql = to_local(pose, q)
        fr = frenet(ref, ql, [hq])[0]
        xl, yl, _ = ref.at(s_e + S)
        y_ln = float(in_ego(pose, [xl, yl])[0, 1])
        y_ps = float(in_ego(pose, [x0 + S * np.cos(t0), y0 + S * np.sin(t0)])[0, 1])
        r.update({f"y_rt{S}": float(q[1]), f"d_rt{S}": float(fr[1]), f"y_ln{S}": y_ln, f"y_ps{S}": y_ps, f"y_cv{S}": y_ln - y_ps,
                  f"y_of{S}": float(q[1]) - y_ln})
    if fut is not None:
        f4 = np.asarray(fut, np.float64)[7]
        arc4 = float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], np.asarray(fut, np.float64)[:, :2]], axis=0).T).sum())
        p4 = to_local(pose, f4[:2])[0]
        fr = frenet(ref, p4[None], [pose[2] + (f4[2] if len(f4) > 2 else 0.0)])[0]
        r.update(arc4=arc4, y_f4=float(f4[1]), d_f4=float(fr[1]), yaw4_deg=float(np.degrees(f4[2])) if len(f4) > 2 else np.nan)
        q, hq = arc_interp(F, arc4)
        if np.isfinite(q).all():
            fr = frenet(ref, to_local(pose, q), [pose[2] + hq])[0]
            xl, yl, _ = ref.at(s_e + arc4)
            r.update(y_rt4=float(q[1]), d_rt4=float(fr[1]), y_ln4=float(in_ego(pose, [xl, yl])[0, 1]))
    return r


# ---------------------------------------------------------------- scenes (box, AlpaSim venv)
def _scene(job):
    import dag
    import rmem as RM
    from alpasim_runtime.route_generator import RouteGeneratorMap
    scene, cfg, decs = job
    ds = dag.load_ds(scene, cfg)
    P, yaw, ts = dag.traj_of(ds)
    if yaw is None:
        yaw = np.unwrap(np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])))
    g = RouteGeneratorMap(P, ds.map, route_start_offset_m=40.0)
    R0 = np.asarray(g.route_polyline_in_local.points, np.float64)
    g._ensure_sufficient_waypoints(R0[-1])
    R = np.asarray(g.route_polyline_in_local.points, np.float64)[:, :2]
    L = Lanes(ds.map, R)
    t = ts * 1e-6
    sl = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    zz = lambda xy: float(P[int(np.argmin(((P[:, :2] - xy[:2]) ** 2).sum(1))), 2])  # noqa: E731
    hum = []
    for i in range(0, len(P), 5):                                               # human states: the log every 0.5 s
        if t[i] + 4.0 > t[-1]:
            break
        v = float((np.interp(t[i] + 0.1, t, sl) - np.interp(t[i] - 0.1, t, sl)) / 0.2)
        if v < 2.0:
            continue
        pose = np.r_[P[i, :2], yaw[i]]
        tq = t[i] + 0.5 * np.arange(1, 9)
        H = np.c_[np.interp(tq, t, P[:, 0]), np.interp(tq, t, P[:, 1])]
        fut = np.c_[in_ego(pose, H), wrap(np.interp(tq, t, yaw) - pose[2])]
        F = RM.window(in_ego(pose, R))
        r = dict(scene=scene, t=round(float(t[i] - t[0]), 2), v0=v, st_log=float(sl[i]), **state_metrics(L, pose, float(P[i, 2]), F, fut))
        y42 = r.get(f"y_rt{S_CMD}")
        r["cmd"] = -1 if y42 is None else (0 if y42 > 2 else 2 if y42 < -2 else 1)
        hum.append(r)
    dec = []
    if decs:
        import dag as _d
        lref = _d.Ref(P[:, :2])                                                 # station of a decision's ego on the human log
        for k, d in decs.items():
            G = R
            for j in range(len(d["t0"])):
                pose = d["anchor"][j]
                F = d["rm"][j].astype(np.float64)
                r = dict(read=k, scene=scene, j=j, t0=int(d["t0"][j]), v0=float(d["v0"][j]), cmd=int(d["cmd"][j]))
                pr = lref.project(np.asarray(pose, np.float64))
                r["st_log"], r["d_log"] = (pr[0], pr[1]) if pr is not None else (np.nan, np.nan)
                r.update(state_metrics(L, pose, zz(pose), F, d["pm"][j]))
                if F[23, 0] > 0.5:                                              # fed +40 m point vs the generator's polyline (sanity)
                    q = to_local(pose, F[23, 1:3] * 20)[0]
                    r["fed_vs_gen"] = float(np.sqrt(((G - q) ** 2).sum(1)).min())
                # the human log's lateral against the ego's lane at the plan's 4 s station (does the human stay in this lane there)
                if "arc4" in r and "d_ego" in r:
                    ln, _, _ = L.current(pose, zz(pose))
                    ref = L.ref(ln, pose)
                    s_e = ref.project(np.asarray(pose, np.float64))[0]
                    m = np.abs(sl - (r["st_log"] + r["arc4"])) < 3.0
                    if m.any():
                        fr = frenet(ref, P[m, :2], yaw[m])
                        ok = np.isfinite(fr[:, 0])
                        if ok.any():
                            jj = int(np.argmin(np.abs(fr[ok, 0] - (s_e + r["arc4"]))))
                            r["d_hl4"] = float(fr[ok][jj, 1])
                dec.append(r)
    return dict(hum=hum, dec=dec)


def cmd_scenes(a):
    import pandas as pd
    import dag
    from jevdrive import par
    jobs = []
    if a.split in ("test", "all"):
        D = pickle.load(open(O / "dec.pkl", "rb"))
        cfg = D["NTV0"]["cfg"]
        for s in sorted(cfg)[:a.limit or None]:
            jobs.append((s, cfg[s], {k: D[k]["dec"][s] for k in READS if s in D[k]["dec"]}))
        del D
    if a.split in ("train", "all"):
        import csv
        cfg = dag.scene_cfgs(TCFG)
        S = [r["scene_id"] for r in csv.DictReader(open(TSV), delimiter="\t")]
        jobs += [(s, cfg[s], None) for s in S if s in cfg][:a.limit or None]
    res = par.pmap(_scene, jobs, workers=a.workers, desc="scenes")
    print(res.summary(), flush=True)
    for i, e in list(res.errors.items())[:5]:
        print(jobs[i][0], e.strip().splitlines()[-1][:300], flush=True)
    sfx = f".{a.tag}" if a.tag else ""
    tr = {j[0] for j in jobs if j[2] is None}
    H = [x for v in res.values if v for x in v["hum"]]
    Dd = [x for v in res.values if v for x in v["dec"]]
    O.mkdir(parents=True, exist_ok=True)
    hdf = pd.DataFrame(H)
    if len(hdf):
        hdf["split"] = np.where(hdf.scene.isin(tr), "train", "test")
        for sp in ("test", "train"):
            if (hdf.split == sp).any():
                hdf[hdf.split == sp].to_parquet(O / f"hum_{sp}{sfx}.parquet")
    if Dd:
        pd.DataFrame(Dd).to_parquet(O / f"dec{sfx}.parquet")
    dump(O / f"scenes{sfx}.json", dict(jobs=len(jobs), failed={jobs[i][0]: e.strip().splitlines()[-1][:300] for i, e in res.errors.items()},
                                        hum=len(H), dec=len(Dd)))


# ---------------------------------------------------------------- rows (box, .venv): map-free training-row check
def cmd_rows(a):
    import rmem as RM
    CR, FE = DATA / "runs/op_parity/cache", DATA / "runs/route_mem/feat"
    parts = {"ph": "on1_PAIH", "ph05": "on1_PAIH05", "ph15": "on1_PAIH15"} | {f"nav{i}": f"navtrain_full.s{i}of12" for i in range(12)}
    out = {}
    for k, n in parts.items():
        T, Fz = np.load(CR / n / "tab.npz"), np.load(FE / f"{n}.npz")
        assert (T["names"] == Fz["names"]).all(), n
        ego, fut, F = T["ego"], T["fut"].astype(np.float64), Fz["f"].astype(np.float32).reshape(-1, RM.N_PT, RM.NF)
        v0 = ego[:, 4] * 10
        cmd = np.where(ego[:, 1:4].max(1) > 0.5, ego[:, 1:4].argmax(1), 3)
        arc4 = np.hypot(*np.diff(np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
        i40, i42 = int(np.flatnonzero(RM.OFFS == 40)[0]), int(np.flatnonzero(RM.OFFS == 42)[0])
        v = F[:, :, 0] > 0.5
        yr4 = np.full(len(F), np.nan)
        for r in np.flatnonzero(v.sum(1) >= 2):
            o = RM.OFFS[v[r]]
            if o[0] <= arc4[r] <= o[-1]:
                yr4[r] = np.interp(arc4[r], o, F[r, v[r], 2] * 20)
        y40 = np.where(v[:, i40], F[:, i40, 2] * 20, np.nan)
        y42 = np.where(v[:, i42], F[:, i42, 2] * 20, np.nan)
        h40 = np.where(v[:, i40], np.degrees(np.arctan2(F[:, i40, 4], F[:, i40, 3])), np.nan)
        out[k] = dict(v0=v0, cmd=cmd, y4=fut[:, 7, 1], yaw4=np.degrees(fut[:, 7, 2]), arc4=arc4, yr4=yr4, y40=y40, y42=y42, h40=h40)
        print(k, len(v0), flush=True)
    nav = {x: np.concatenate([out[f"nav{i}"][x] for i in range(12)]) for x in out["nav0"]}
    out = {"ph": out["ph"], "ph05": out["ph05"], "ph15": out["ph15"], "navtrain": nav}
    pickle.dump(out, open(O / "rows.pkl", "wb"), protocol=4)


# ---------------------------------------------------------------- report (box, .venv)
def md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(":--" if df[c].dtype == object else "--:" for c in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return "\n".join(L)


def road_cls(dth):
    a = np.abs(dth)
    return np.select([a < 3, a < 15, a < 45], ["straight (<3 deg / 40 m)", "gentle curve (3-15)", "bend (15-45)"], "turn (>=45)")


CLS_ORDER = ["straight (<3 deg / 40 m)", "gentle curve (3-15)", "bend (15-45)", "turn (>=45)"]


def offset_table(df, by):
    """route lateral at 40 m and its parts, per group."""
    rows = []
    for g, x in df.groupby(by, sort=False):
        x = x[np.isfinite(x.y_rt40)]
        if not len(x):
            continue
        q = lambda c, p=50: float(np.nanpercentile(np.abs(x[c]), p))  # noqa: E731
        rows.append({by if isinstance(by, str) else "group": g, "n": len(x), "|y_rt40| p50": q("y_rt40"), "p90": q("y_rt40", 90),
                     "|pose| p50": q("y_ps40"), "|curv| p50": q("y_cv40"), "|curv| p90": q("y_cv40", 90), "|lane off| p50": q("d_rt40"),
                     "|lane off| p90": q("d_rt40", 90), "off>1 m %": 100 * float((np.abs(x.d_rt40) > 1).mean()),
                     "off>2.5 m %": 100 * float((np.abs(x.d_rt40) > 2.5).mean()),
                     "off<-1 (right) %": 100 * float((x.d_rt40 < -1).mean()), "off>1 (left) %": 100 * float((x.d_rt40 > 1).mean()),
                     "cmd L %": 100 * float((x.cmd == 0).mean()), "cmd R %": 100 * float((x.cmd == 2).mean())})
    return pd_().DataFrame(rows)


def pd_():
    import pandas as pd
    return pd


def cmd_cause(df):
    """turn commands (L / R) on non-turn states: what puts the route's ~42 m point beyond 2 m (dominant part)."""
    x = df[(df.cmd.isin([0, 2])) & np.isfinite(df.y_rt42) & (np.abs(df.dth40) < 15)].copy()
    if not len(x):
        return pd_().DataFrame()
    parts = np.c_[x.y_ps42, x.y_cv42, x.y_of42]
    sgn = np.sign(x.y_rt42.to_numpy())[:, None]
    dom = np.array(["pose (ego offset / heading)", "curvature (lane bends)", "lane offset (route not on the ego's lane centre)"])[np.argmax(parts * sgn, 1)]
    x["cause"] = dom
    x["side"] = np.where(x.cmd == 2, "R", "L")
    t = x.groupby(["cls", "side", "cause"]).size().rename("n").reset_index()
    return t


def follow(df, fcol, label):
    """lane-relative 4 s offset of the plan / label (fcol) by the route's lane-relative offset at the same arc (d_rt4) and by curvature."""
    x = df[np.isfinite(df.get("d_rt4", np.nan)) & np.isfinite(df[fcol]) & (df.v0 >= 4) & (np.abs(df.dth40) < 15)]
    rows = []
    bins = [(-99, -2.5, "route >= 2.5 m right of lane"), (-2.5, -1.0, "1-2.5 m right"), (-1.0, 1.0, "within 1 m (lane-centred)"),
            (1.0, 2.5, "1-2.5 m left"), (2.5, 99, ">= 2.5 m left")]
    for lo, hi, nm in bins:
        m = x[(x.d_rt4 >= lo) & (x.d_rt4 < hi)]
        if len(m):
            rows.append(dict(who=label, route_vs_lane_at_4s=nm, n=len(m), d_route_med=float(m.d_rt4.median()), d_f4_med=float(m[fcol].median()),
                             follow_ratio=float(m[fcol].median() / m.d_rt4.median()) if abs(m.d_rt4.median()) > 0.5 else np.nan,
                             stays_in_lane_pct=100 * float((np.abs(m[fcol]) < 1.0).mean()),
                             goes_to_route_pct=100 * float((np.abs(m[fcol] - m.d_rt4) < 1.0).mean())))
    return rows


def curv_follow(df, fcol, label):
    x = df[np.isfinite(df.get("y_cv40", np.nan)) & np.isfinite(df[fcol]) & (df.v0 >= 4) & (np.abs(df.d_rt40) < 1.0)]
    rows = []
    for lo, hi, nm in [(-99, -3, "right bend (curv part < -3 m at 40 m)"), (-3, -1, "gentle right (-3..-1)"), (-1, 1, "straight (|curv| < 1)"),
                       (1, 3, "gentle left (1..3)"), (3, 99, "left bend (> 3)")]:
        m = x[(x.y_cv40 >= lo) & (x.y_cv40 < hi)]
        if len(m):
            rows.append(dict(who=label, lane_bend=nm, n=len(m), d_f4_med=float(m[fcol].median()), d_f4_p10=float(m[fcol].quantile(0.1)),
                             d_f4_p90=float(m[fcol].quantile(0.9)), right_of_lane_gt1_pct=100 * float((m[fcol] < -1).mean()),
                             left_of_lane_gt1_pct=100 * float((m[fcol] > 1).mean())))
    return rows


def cmd_report(a):
    import pandas as pd
    D = pickle.load(open(O / "dec.pkl", "rb"))
    HT, HR = pd.read_parquet(O / "hum_test.parquet"), pd.read_parquet(O / "hum_train.parquet")
    DC = pd.read_parquet(O / "dec.parquet")
    for x in (HT, HR, DC):
        x["cls"] = road_cls(x.dth40.fillna(0))
    RES.mkdir(parents=True, exist_ok=True)
    L = ["# ROUTE-CURV: is the route input a lateral intent in disguise? (ROUND6, 2026-10-21)", "",
         "Diagnosis, stored reads only, no GPU (pool CPU jobs). Code `experiments/round6/scripts/rmc.py` (`extract`, `scenes`, `rows`, "
         "`report`), lane `rmc.sh`. Generated tables; the reading follows at the end.", ""]
    # ---- 0. sanity
    fv = DC.fed_vs_gen.dropna() if "fed_vs_gen" in DC else pd.Series(dtype=float)
    L += ["## 0. Sanity", "",
          f"Decisions read: {len(DC)} ({', '.join(f'{k} {int((DC.read == k).sum())}' for k in READS)}); human states: test {len(HT)}, "
          f"train {len(HR)}. Fed +40 m route point vs AlpaSim's generator polyline rebuilt on the log: median {fv.median():.3f} m, "
          f"p99 {fv.quantile(.99):.3f} m, > 0.5 m on {100 * (fv > 0.5).mean():.2f} % (n {len(fv)}). Current lane found on "
          f"{100 * DC.d_ego.notna().mean():.1f} % of decisions, {100 * HT.d_ego.notna().mean():.1f} / {100 * HR.d_ego.notna().mean():.1f} % "
          "of human states (test / train).", ""]
    # ---- 1. human states
    L += ["## 1. The route against the lane on the human log", "",
          "Every 0.5 s of the human log (v >= 2 m/s); route = AlpaSim's map route (what the messages are cut from) windowed as the fed memory. "
          "`y_rt40` = the route point 40 m of route arc ahead, lateral in the ego heading frame (+ left); parts: `pose` = the ego's lane offset and "
          "heading error carried 40 m straight, `curv` = the lane centre's own bend over those 40 m, `lane off` = the route point's signed Frenet "
          "offset from the ego's current lane centre (the human's lane here). `off > 2.5 m` = the route sits in another lane. cmd = the command "
          "rule on the route's ~42 m point.", ""]
    for nm, H in (("test 281", HT), ("train 719 (the PAI human rows' scenes)", HR)):
        t = offset_table(H, "cls").set_index("cls").reindex(CLS_ORDER).dropna(how="all").reset_index()
        t.to_csv(RES / f"route_mem_curv_hum_{nm.split()[0]}.csv", index=False, float_format="%.3f")
        L += [f"**{nm}**", "", md(t), ""]
    # route off-lane while the human stays in lane
    for nm, H in (("test", HT), ("train", HR)):
        x = H[np.isfinite(H.d_rt40) & np.isfinite(H.d_f4)]
        off = np.abs(x.d_rt40) > 1.75
        stay = np.abs(x.d_f4) < 1.0
        go = np.abs(x.d_f4 - x.d_rt4) < 1.0
        L.append(f"- {nm}: route >= 1.75 m off the human's lane centre at 40 m on {100 * off.mean():.1f} % of states "
                 f"({int(off.sum())}; right {100 * (x.d_rt40 < -1.75).mean():.1f} %, left {100 * (x.d_rt40 > 1.75).mean():.1f} %); of these the "
                 f"human's 4 s point stays within 1 m of its lane centre on {100 * stay[off].mean():.1f} % and ends within 1 m of the route on "
                 f"{100 * go[off].mean():.1f} %; scenes with any such state {x[off].scene.nunique()} / {x.scene.nunique()}.")
    L.append("")
    cc = pd.concat([cmd_cause(HT).assign(split="test"), cmd_cause(HR).assign(split="train")])
    if len(cc):
        L += ["Turn commands (L / R) on non-turn states (|lane bend over 40 m| < 15 deg): the part that puts the ~42 m point beyond 2 m "
              "(largest signed contribution):", "", md(cc.pivot_table(index=["split", "side", "cause"], columns="cls", values="n", aggfunc="sum",
                                                                        fill_value=0).reset_index(), "{:.0f}"), ""]
    # ---- 2. decisions per read
    L += ["## 2. The fed route at the decisions of the five reads", "",
          "Every inference decision (10 Hz) of the stored PAI 281 ACC + GAP reads; route = the fed memory itself; lane = the ego's own current lane.", ""]
    t = offset_table(DC.assign(g=DC.read + " / " + DC.cls), "g")
    t.to_csv(RES / "route_mem_curv_dec.csv", index=False, float_format="%.3f")
    t = t[t.g.str.contains("straight|gentle")]
    L += [md(t), ""]
    # ---- 3. events
    ev = []
    FA = {k: pd.DataFrame(D[k]["facts"]).set_index("scene") for k in READS}
    FN = FA["NTV0"]
    E = pd.read_csv(REPO / "experiments/round5/results/offroad_mv_events.csv")
    mech = {(r.read, r.scene): r.mech for r in E.itertuples()}
    for k in MV:
        F = FA[k]
        for s in F.index:
            fc = F.fclass[s]
            if fc not in ("offroad", "zero-other") or (s in FN.index and FN.fclass[s] == fc) or not np.isfinite(F.t_ev_s[s]):
                continue
            if s not in D[k]["c0"]:
                continue
            ev.append(dict(read=k, scene=s, fclass=fc, te=D[k]["c0"][s] + float(F.t_ev_s[s]) * 1e6, mech=mech.get((k, s), "")))
    ev = pd.DataFrame(ev)
    cols = ["y_rt40", "d_rt40", "y_cv40", "y_ps40", "cmd", "d_f4", "y_f4", "d_rt4", "y_rt4", "v0", "dth40", "d_ego", "d_hl4"]
    by = {(r, s): g.sort_values("t0") for (r, s), g in DC.groupby(["read", "scene"])}
    rows = []
    for e in ev.itertuples():
        g = by.get((e.read, e.scene))
        n = by.get(("NTV0", e.scene))
        if g is None:
            continue
        for Ld in LEADS:
            kk = min(int(np.searchsorted(g.t0.to_numpy(), e.te - Ld * 1e6)), len(g) - 1)
            r = g.iloc[kk]
            rows.append(dict(read=e.read, scene=e.scene[7:15], fclass=e.fclass, mech=e.mech, lead=Ld, who="MV", **{c: r.get(c, np.nan) for c in cols}))
            if n is not None and np.isfinite(r.st_log):
                dd = np.abs(n.st_log.to_numpy() - r.st_log)
                if np.nanmin(dd) < 3.0:
                    rn = n.iloc[int(np.nanargmin(dd))]
                    rows.append(dict(read=e.read, scene=e.scene[7:15], fclass=e.fclass, mech=e.mech, lead=Ld, who="NTV0 same station",
                                     **{c: rn.get(c, np.nan) for c in cols}))
    EV = pd.DataFrame(rows)
    EV.to_csv(RES / "route_mem_curv_events.csv", index=False, float_format="%.3f")
    # clean-scene baselines (moving decisions, v0 >= 4)
    def share(x, nm):
        x = x[np.isfinite(x.y_rt40)]
        return dict(set=nm, n=len(x), y_rt40_med=float(x.y_rt40.median()), d_rt40_med=float(x.d_rt40.median()),
                    y_cv40_med=float(x.y_cv40.median()), right_route_pct=100 * float((x.y_rt40 < -1.5).mean()),
                    right_laneoff_pct=100 * float((x.d_rt40 < -1.0).mean()), cmd_R_pct=100 * float((x.cmd == 2).mean()),
                    cmd_L_pct=100 * float((x.cmd == 0).mean()), d_f4_med=float(x.d_f4.median()) if "d_f4" in x else np.nan,
                    plan_right_of_lane_pct=100 * float((x.d_f4 < -1.0).mean()))
    B = []
    for Ld in LEADS:
        for who in ("MV", "NTV0 same station"):
            for fc in ("offroad", "zero-other"):
                x = EV[(EV.lead == Ld) & (EV.who == who) & (EV.fclass == fc)]
                if len(x):
                    B.append(share(x, f"{fc} events, -{Ld:.0f} s, {who}"))
    for k in ("NTV0", "MV"):
        rd = [k] if k == "NTV0" else MV
        cl = DC[DC.read.isin(rd) & (DC.v0 >= 4)]
        ok = cl[[FA[r].fclass.get(s, "") == "ok" for r, s in zip(cl.read, cl.scene)]]
        B.append(share(ok, f"{k} clean scenes (score 1), all moving decisions"))
        B.append(share(ok[np.abs(ok.dth40) < 15], f"{k} clean scenes, non-turn moving decisions"))
    B = pd.DataFrame(B)
    B.to_csv(RES / "route_mem_curv_failure_link.csv", index=False, float_format="%.3f")
    L += ["## 3. The route before the MV family's offroad / zero-other events", "",
          f"Events: every offroad or zero-other zero of R3MV s0 / s1 and MV-LC s0 / s1 on a scene where NTV0 does not have that class "
          f"({len(ev)} events: {', '.join(f'{c} {int((ev.fclass == c).sum())}' for c in ('offroad', 'zero-other'))}); decision = the first one at "
          "or after t_ev - 3 / 2 / 1 s (offroad_mv's rule); NTV0 = its decision nearest the same human-log station (< 3 m). `right route` = "
          "y_rt40 < -1.5 m, `right lane off` = d_rt40 < -1 m, `plan right of lane` = the plan's 4 s point > 1 m right of the ego lane's centre.", "",
          md(B), ""]
    S = EV[EV.who == "MV"].groupby(["scene", "fclass", "mech"]).agg(reads=("read", "nunique"), y_rt40=("y_rt40", "median"),
                                                                     d_rt40=("d_rt40", "median"), y_cv40=("y_cv40", "median"),
                                                                     cmd_R=("cmd", lambda c: float((c == 2).mean())), d_f4=("d_f4", "median"),
                                                                     d_hl4=("d_hl4", "median")).reset_index()
    L += ["Per event scene (median over its reads and the three leads; `d_hl4` = the human log's offset from the ego lane's centre at the "
          "plan's 4 s station):", "", md(S.sort_values(["fclass", "reads"], ascending=[True, False])), ""]
    # ---- 4. who follows the route
    FO = []
    for k in READS:
        FO += follow(DC[DC.read == k], "d_f4", k + " plan")
    FO += follow(HT, "d_f4", "human test") + follow(HR, "d_f4", "human train")
    FO = pd.DataFrame(FO)
    FO.to_csv(RES / "route_mem_curv_follow.csv", index=False, float_format="%.3f")
    L += ["## 4. Does the plan / the human label follow the route's lateral offset or the lane?", "",
          "Moving (v0 >= 4), non-turn states; bins by the route's offset from the ego's lane centre at the plan's (label's) own 4 s arc; "
          "`d_f4` = the plan / label 4 s point's offset from the same lane centre; follow ratio = median d_f4 / median route offset; "
          "`stays` = |d_f4| < 1 m, `goes to route` = within 1 m of the route point.", "", md(FO), ""]
    CF = []
    for k in READS:
        CF += curv_follow(DC[DC.read == k], "d_f4", k + " plan")
    CF += curv_follow(HT, "d_f4", "human test") + curv_follow(HR, "d_f4", "human train")
    CF = pd.DataFrame(CF)
    CF.to_csv(RES / "route_mem_curv_bend.csv", index=False, float_format="%.3f")
    L += ["Lane-centred route (|lane off| < 1 m at 40 m), by the lane's own bend (the curvature part of y_rt40): the plan / label 4 s point "
          "against the lane centre (negative = right).", "", md(CF), ""]
    # ---- 5. training rows (map-free)
    if (O / "rows.pkl").exists():
        R = pickle.load(open(O / "rows.pkl", "rb"))
        TR = []
        for k, x in R.items():
            m = (x["v0"] >= 4) & np.isfinite(x["yr4"])
            nt = m & (np.abs(x["yaw4"]) < 10) & (np.abs(np.nan_to_num(x["h40"])) < 10)
            for nm, mm in (("moving", m), ("moving, label and route heading < 10 deg", nt)):
                e = x["y4"][mm] - x["yr4"][mm]
                big = np.abs(x["yr4"][mm]) > 1.75
                TR.append(dict(rows=k, set=nm, n=int(mm.sum()), cmd_L_pct=100 * float((x["cmd"][mm] == 0).mean()),
                               cmd_R_pct=100 * float((x["cmd"][mm] == 2).mean()), cmd_unk_pct=100 * float((x["cmd"][mm] == 3).mean()),
                               route_lat4_gt175_pct=100 * float(big.mean()), label_minus_route_abs_med=float(np.median(np.abs(e))),
                               label_off_route_gt175_pct=100 * float((np.abs(e) > 1.75).mean()),
                               when_route_gt175_label_follows_pct=100 * float((np.abs(e[big]) < 1.0).mean()) if big.any() else np.nan,
                               when_route_gt175_label_straight_pct=100 * float((np.abs(x["y4"][mm][big]) < 1.0).mean()) if big.any() else np.nan))
        TR = pd.DataFrame(TR)
        TR.to_csv(RES / "route_mem_curv_rows.csv", index=False, float_format="%.3f")
        L += ["## 5. Training rows (map-free): does the label go where the fed route points?", "",
              "Rows fed v0 >= 4 m/s with the route covering the label's 4 s arc; `route lat4` = the fed route's lateral (ego frame) at the label's "
              "4 s arc; `label - route` = the label's 4 s lateral minus it. PAI parts unweighted (not the draw weights).", "", md(TR), ""]
    (RES / "route_mem_curv.md").write_text("\n".join(L) + "\n<!-- generated end -->\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("extract")
    q.add_argument("--workers", type=int, default=8)
    q = sp.add_parser("scenes")
    q.add_argument("--split", default="all", choices=["test", "train", "all"])
    q.add_argument("--workers", type=int, default=None)
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--tag", default="")
    sp.add_parser("rows")
    sp.add_parser("report")
    a = ap.parse_args()
    dict(extract=cmd_extract, scenes=cmd_scenes, rows=cmd_rows, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
