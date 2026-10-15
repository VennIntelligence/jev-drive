#!/usr/bin/env python3
"""ROUTE-LANE CORRIDOR labels and the offline corridor check (plans/2026-10-15-corridor-check-prereg.md; research/next-round/turn-night.md
section 6). The corridor of a decision = the route's own lane sequence (entry lane -> junction connector -> exit lane; the target lane at a
lane change) + 0.3 m; outside it a point is an adjacent same-direction lane (soft), oncoming lane or off-road (hard), or other drivable
(crossing lane / road area, soft). CPU only; stored reads are read, never written.

  lanes  (box: AlpaSim venv; Tokyo: rf_corridor.sh lanes-tokyo, inside alpasim-base:0.89.0)
         rf_corridor.py lanes --runs <held run dir | PAI run root> ... --out <pkl> [--workers N]
         per scene: lanes within 150 m of the logged path (centre, left / right edge, the evaluator's lane polygon as WKB, next / prev /
         adjacent ids), road-area polygons (WKB), the logged path.
  plans  (host python3 + numpy)  rf_corridor.py plans --runs <held run dir | PAI run root> ... --out <pkl>
         {session: t0 (n,), anchor (n, 3), poses (n, 8, 3)}: the served plan of every decision (PAI: `infer` records only).
  build  (CPU, repo .venv; DATA_DIR for the Run dir)
         rf_corridor.py build --set NAME=<tax pkl>,<plans pkl>,<lanes pkl>,<tax csv>,<follow csv>,<follow label> ... --out DIR
         -> DIR/labels_<NAME>.pkl ({rid: t0, grid (n, 48, 48) uint8 codes, corridor polygon WKB in the anchor frame, found, amb}),
            DIR/dec.parquet (one row per decision: plan / human / ego footprint class masks), DIR/roll.parquet (one row per rollout:
            zero class, t_f, strict turn), DIR/coverage.{md,json}
  check  (CPU, repo .venv)  rf_corridor.py check --build DIR --out RESDIR [--fig PNG --set ... --picks NAME:scene8@t,..]
         -> RESDIR/check.{md,json}, zeros.csv, turns.csv, coverage.{md,json} (copied); the BEV figure.

Codes: 0 corridor, 1 adjacent same-direction lane, 2 oncoming lane, 3 off-road, 4 other drivable; 255 = no corridor. Hard = {2, 3}.
Mask columns hold one bit per code (bit c = some footprint point in code c) for each of the plan's 1-4 s poses (m1 .. m7 = 1.0 .. 4.0 s),
the human log's own poses at t0 + 1.0 .. 4.0 s (h1 .. h7), and the ego's driven footprint at t0 (eg).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + ([str(HERE.parents[2]), str(HERE.parents[1] / "junction_eyes" / "lib")] if len(HERE.parents) > 2 else [])

R_MAP, STEP, NEAR_M, CHANGE_M = 150.0, 1.0, 1.0, 6.0
# registered: corridor margin 0.3 m, full rig box footprint, grids stored. Exploratory variants (Amendment 4) via the environment:
# COR_MARGIN (m), COR_FP=axle (rear-axle and front-bumper centre points only), COR_GRID=0 (no grids / labels, check columns only).
MARGIN, FP_MODE, GRID_ON = float(os.environ.get("COR_MARGIN", 0.3)), os.environ.get("COR_FP", "box"), os.environ.get("COR_GRID", "1") == "1"
HEAD_MAX, SAME, ONC, AMB_PSI = np.radians(60.0), np.radians(60.0), np.radians(120.0), np.radians(30.0)
AMB_DC, AMB_SEP = 0.5, 1.0
WIN_LO, WIN_HI, FOUND_D, FOUND_SHARE = 15.0, 100.0, 40.0, 0.9
LC_PAST, LC_BACK = 10.0, 30.0
GX = np.arange(48) * 0.5 + 0.25
GY = np.arange(48) * 0.5 - 11.75
GRID = np.stack(np.meshgrid(GX, GY, indexing="ij"), -1).reshape(-1, 2)          # [x index, y index] like je_ruler.rb_label
POSE_T = np.arange(1.0, 4.01, 0.5)                                              # plan poses 1..7 of 8 at 0.5 s
HARD, SOFT = (1 << 2) | (1 << 3), (1 << 1) | (1 << 4)
NOCOR = 255


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- lanes (AlpaSim env)
def _sims(root):
    s = sorted(glob.glob(os.path.join(root, "runs", "*", "sim")))
    return s if s else [root]


def lanes_job(job):
    scene, cfg = job
    import logging
    import rf_tax as TX
    logging.disable(logging.WARNING)
    return scene, lanes_of(TX.loader(cfg).get_data_source(scene))


def lanes_of(ds):
    """A loaded scene data source -> the lanes extract of one scene (lanes_job)."""
    import shapely
    from eval.scorers.offroad import _get_lane_polygon, _repair_polygonal_geometry
    tr = ds.rig.trajectory
    T, P, Y = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
    Q = P[::5]
    ed = lambda e: None if e is None else np.asarray(e.points, float)[:, :2].astype(np.float32)  # noqa: E731
    lanes = {}
    for ln in ds.map.lanes:
        c = np.asarray(ln.center.points, float)[:, :2]
        if len(c) < 2 or np.hypot(*(c[:, None] - Q[None]).transpose(2, 0, 1)).min() > R_MAP:
            continue
        lanes[str(ln.id)] = dict(c=c.astype(np.float32), l=ed(ln.left_edge), r=ed(ln.right_edge), poly=shapely.to_wkb(_get_lane_polygon(ln)),
                                 nx=sorted(map(str, ln.next_lanes)), pv=sorted(map(str, ln.prev_lanes)),
                                 al=sorted(map(str, ln.adj_lanes_left)), ar=sorted(map(str, ln.adj_lanes_right)))
    areas = []
    lo, hi = P.min(0) - R_MAP, P.max(0) + R_MAP
    try:
        for ra in ds.map.get_road_areas_within(np.r_[(lo + hi) / 2, 0.0], float(np.hypot(*(hi - lo)) / 2)):
            areas.append(shapely.to_wkb(_repair_polygonal_geometry(ds.map.get_road_area_polygon_2d(ra.id))))
    except Exception:  # noqa: BLE001  PAI maps have no road-area layer
        pass
    return dict(lanes=lanes, areas=areas, log=dict(T=T, P=P, Y=Y))


def lanes_extract(a):
    from concurrent.futures import ProcessPoolExecutor
    jobs = {}
    for root in a.runs:
        for sim in _sims(root):
            f = Path(sim) / "aggregate/results-summary.json"
            if f.exists():
                for r in json.loads(f.read_text())["rollouts"]:
                    jobs.setdefault(r["clipgt_id"], str(Path(sim) / "generated-user-config-0.yaml"))
    print(len(jobs), "scenes", flush=True)
    res, err = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(lanes_job, j): j[0] for j in sorted(jobs.items())}
        for i, f in enumerate(futs):
            try:
                k, v = f.result()
                res[k] = v
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:300]))
            if i % 50 == 0:
                print(i, "done", len(err), "errors", flush=True)
    pickle.dump(dict(scenes=res, errors=err), open(a.out, "wb"), protocol=4)
    print(len(res), "scenes,", len(err), "errors ->", a.out, flush=True)


# ---------------------------------------------------------------- plans (host python3)
def plans_extract(a):
    out = {}
    for root in a.runs:
        fs = sorted(glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl"))) or [os.path.join(root, "driver-logs", "drive.jsonl")]
        for f in fs:
            for line in open(f):
                if '"kind": "drive"' not in line:
                    continue
                r = json.loads(line)
                if r.get("infer") is False or "poses" not in r or "anchor" not in r:
                    continue
                out.setdefault(r["session"], []).append((int(r["t0"]), r["anchor"], r["poses"], r.get("cmd")))
    res = {}
    for k, v in out.items():
        v.sort(key=lambda x: x[0])
        res[k] = dict(t0=np.array([x[0] for x in v], np.int64), anchor=np.array([x[1] for x in v], float),
                      poses=np.array([x[2] for x in v], np.float32), cmd=np.array([-1 if x[3] is None else x[3] for x in v], np.int8))
    pickle.dump(res, open(a.out, "wb"), protocol=4)
    print(len(res), "sessions,", sum(len(v["t0"]) for v in res.values()), "decisions ->", a.out)


# ---------------------------------------------------------------- geometry
def to_world(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.asarray(a[:2], float) + np.c_[c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]]


def in_frame(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, float).reshape(-1, 2) - np.asarray(a[:2], float)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def stations(C):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]


def arc_proj(C, Sc, X):
    """Points X (m, 2) on polyline C with arc Sc -> (arc, heading, distance)."""
    A, d = C[:-1], C[1:] - C[:-1]
    dd = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip((((X[:, None, :] - A[None]) * d[None]).sum(-1)) / dd[None], 0, 1)
    Q = A[None] + t[..., None] * d[None]
    e = ((Q - X[:, None, :]) ** 2).sum(-1)
    j = np.argmin(e, 1)
    r = np.arange(len(X))
    return Sc[j] + t[r, j] * np.sqrt(dd[j]), np.arctan2(d[j, 1], d[j, 0]), np.sqrt(e[r, j])


def densify(X, step=0.5):
    S = stations(X)
    if S[-1] <= 0:
        return X
    q = np.r_[np.arange(0.0, S[-1], step), S[-1]]
    return np.c_[np.interp(q, S, X[:, 0]), np.interp(q, S, X[:, 1])]


def footprint(dims, n=6):
    rear, front, hw = dims
    if FP_MODE == "axle":
        return np.array([[0.0, 0.0], [front, 0.0]])
    u = np.linspace(rear, front, n)
    return np.r_[np.c_[u, np.full(n, hw)], np.c_[u, np.full(n, -hw)], [[rear, 0.0], [front, 0.0]]]


def fp_world(P, yaw, FP):
    """Poses (k, 2) + yaw (k,) -> footprint points (k, len(FP), 2)."""
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([P[:, None, 0] + c * FP[None, :, 0] - s * FP[None, :, 1], P[:, None, 1] + s * FP[None, :, 0] + c * FP[None, :, 1]], -1)


class SceneMap:
    """One scene's lanes (+ road areas) from the lanes extract: evaluator polygons, centre / edges, successor and adjacency sets."""

    def __init__(self, sc):
        import shapely
        self.sh = shapely
        self.ids = sorted(sc["lanes"])
        L = [sc["lanes"][k] for k in self.ids]
        self.ix = {k: i for i, k in enumerate(self.ids)}
        self.C = [x["c"].astype(float) for x in L]
        self.Sc = [stations(c) for c in self.C]
        self.E = [(None if x["l"] is None or x["r"] is None else (x["l"].astype(float), x["r"].astype(float))) for x in L]
        self.poly = [shapely.make_valid(shapely.from_wkb(x["poly"])) for x in L]
        self.nxt = [set(x["nx"]) for x in L]
        self.adj = [set(x["al"]) | set(x["ar"]) for x in L]
        self.tree = shapely.STRtree(self.poly)
        A = [shapely.make_valid(shapely.from_wkb(w)) for w in sc.get("areas", [])]
        self.area = shapely.union_all(A) if A else None
        if self.area is not None:
            shapely.prepare(self.area)

    # ------------------------------------------------------------ route-lane chain (je_ruler.LocalMap.route's rules, on the route)
    def match(self, R, S, LP):
        sh = self.sh
        from je_ruler import proj
        sg = np.arange(0.0, S[-1], STEP)
        P = np.c_[np.interp(sg, S, R[:, 0]), np.interp(sg, S, R[:, 1])]
        q0, q1 = np.maximum(sg - 5.0, 0), np.minimum(sg + 5.0, S[-1])
        d = np.c_[np.interp(q1, S, R[:, 0]) - np.interp(q0, S, R[:, 0]), np.interp(q1, S, R[:, 1]) - np.interp(q0, S, R[:, 1])]
        h = np.arctan2(d[:, 1], d[:, 0])
        n = len(sg)
        lane, amb = np.full(n, -1), np.zeros(n, bool)
        cand = [[] for _ in range(n)]
        pts = sh.points(P)
        ii, jj = self.tree.query(pts, predicate="dwithin", distance=NEAR_M)
        for j in np.unique(jj):
            m = ii[jj == j]
            _, hd, dc = arc_proj(self.C[j], self.Sc[j], P[m])
            dh = np.abs(wrap(hd - h[m]))
            sc = dh + 0.5 * sh.distance(self.poly[j], pts[m])
            Q = P[m]
            for k in np.flatnonzero(dh < HEAD_MAX):
                cand[m[k]].append((float(sc[k]), int(j), float(dc[k]), float(dh[k]), Q[k]))
        for i, cs in enumerate(cand):
            if not cs:
                continue
            cs.sort(key=lambda c: c[0])
            b = cs[0]
            opts = [b]
            if b[3] < AMB_PSI:
                for c in cs[1:]:
                    if c[3] < AMB_PSI and abs(c[2] - b[2]) < AMB_DC:
                        cb = self.C[b[1]][np.argmin(np.hypot(*(self.C[b[1]] - P[i]).T))]
                        cc = self.C[c[1]][np.argmin(np.hypot(*(self.C[c[1]] - P[i]).T))]
                        _, _, eb = arc_proj(self.C[b[1]], self.Sc[b[1]], P[i][None])
                        _, _, ec = arc_proj(self.C[c[1]], self.Sc[c[1]], P[i][None])
                        if np.hypot(*(cb - cc)) > AMB_SEP and abs(eb[0] - ec[0]) < AMB_DC:
                            opts.append(c)
            if len(opts) > 1:                                       # two distinct lanes: the one whose centre is nearer the human's path
                dl = []
                for c in opts:
                    a_, _, _ = arc_proj(self.C[c[1]], self.Sc[c[1]], P[i][None])
                    cp = np.array([np.interp(a_[0], self.Sc[c[1]], self.C[c[1]][:, 0]), np.interp(a_[0], self.Sc[c[1]], self.C[c[1]][:, 1])])
                    dl.append(proj(LP, cp)[2])
                b = opts[int(np.argmin(dl))]
                amb[i] = True
            lane[i] = b[1]
        E = []
        for i in np.flatnonzero(lane >= 0):
            if E and E[-1][0] == lane[i]:
                E[-1][2] = i
            else:
                E.append([int(lane[i]), int(i), int(i)])
        k = 1
        while k < len(E) - 1:                                        # flicker A, B, A with B < 2 samples -> A
            if E[k - 1][0] == E[k + 1][0] and E[k][2] - E[k][1] < 2:
                E[k - 1][2] = E[k + 1][2]
                del E[k:k + 2]
            else:
                k += 1
        F = []
        for e in E:                                                  # one missing connector filled from the successor graph
            if F:
                x, y = F[-1][0], e[0]
                if self.ids[y] not in self.nxt[x]:
                    zs = [self.ix[z] for z in self.nxt[x] if z in self.ix and self.ids[y] in self.nxt[self.ix[z]]]
                    if zs:
                        gap = P[F[-1][2]:e[1] + 1]
                        F.append([min(zs, key=lambda z: float(np.mean(arc_proj(self.C[z], self.Sc[z], gap)[2]))), None, None])
            F.append(list(e))
        trans = []
        for k in range(len(F) - 1):
            x, y = F[k][0], F[k + 1][0]
            if self.ids[y] in self.nxt[x]:
                trans.append("succ")
            else:
                q = P[F[k + 1][1]] if F[k + 1][1] is not None else self.C[y][0]
                trans.append("change" if (self.ids[y] in self.adj[x] or arc_proj(self.C[x], self.Sc[x], q[None])[2][0] <= CHANGE_M) else "break")
        larc = lambda j, p: float(arc_proj(self.C[j], self.Sc[j], p[None])[0][0])  # noqa: E731
        pieces, s_prev = [], 0.0
        for k, (j, i0, i1) in enumerate(F):
            Sc = self.Sc[j]
            tin = trans[k - 1] if k else "break"
            tout = trans[k] if k < len(trans) else "break"
            if i0 is None:
                a0, a1, s0 = 0.0, Sc[-1], s_prev
            else:
                ap0 = larc(j, P[i0])
                a0 = 0.0 if tin == "succ" else ap0
                if tout == "succ":
                    a1 = Sc[-1]
                elif tout == "change":
                    nx = F[k + 1]
                    a1 = larc(j, P[nx[1]] if nx[1] is not None else self.C[nx[0]][0])
                else:
                    a1 = larc(j, P[i1])
                s0 = sg[i0] - (ap0 - a0)
            a1 = max(a1, a0)
            pieces.append(dict(j=j, a0=a0, a1=a1, s0=s0, s1=s0 + a1 - a0, tin=tin, tout=tout))
            s_prev = s0 + a1 - a0
        for k in range(len(pieces) - 1):                              # lane change: both lanes in [cut - 30, cut + 10]
            if pieces[k]["tout"] == "change":
                p, q = pieces[k], pieces[k + 1]
                ext = max(min(LC_PAST, self.Sc[p["j"]][-1] - p["a1"]), 0.0)
                p["a1"] += ext
                p["s1"] += ext
                back = max(min(LC_BACK, q["a0"]), 0.0)
                q["a0"] -= back
                q["s0"] -= back
        for p in pieces:
            p["poly"] = self.piece_poly(p)
            a = np.r_[np.arange(p["a0"], p["a1"], 1.0), p["a1"]]
            C, Sc = self.C[p["j"]], self.Sc[p["j"]]
            seg = np.clip(np.searchsorted(Sc, a, side="right") - 1, 0, len(C) - 2)
            p["ref"] = np.c_[np.interp(a, Sc, C[:, 0]), np.interp(a, Sc, C[:, 1]),
                             np.arctan2(C[seg + 1, 1] - C[seg, 1], C[seg + 1, 0] - C[seg, 0]), p["s0"] + (a - p["a0"])]
        cov = np.zeros(n, bool)
        for p in pieces:
            cov |= (sg >= p["s0"] - 0.5) & (sg <= p["s1"] + 0.5)
        return dict(sg=sg, cov=cov, amb=amb, pieces=pieces, n_amb=int(amb.sum()), trans=trans)

    def piece_poly(self, p):
        sh, j = self.sh, p["j"]
        Sc = self.Sc[j]
        if p["a0"] <= 0.05 and p["a1"] >= Sc[-1] - 0.05:
            return self.poly[j]
        if p["a1"] - p["a0"] < 0.5:
            return None
        if self.E[j] is None:
            a = np.r_[np.arange(p["a0"], p["a1"], 0.5), p["a1"]]
            return sh.LineString(np.c_[np.interp(a, Sc, self.C[j][:, 0]), np.interp(a, Sc, self.C[j][:, 1])]).buffer(1.85, cap_style="flat")
        L, R = densify(self.E[j][0]), densify(self.E[j][1])
        aL, aR = arc_proj(self.C[j], Sc, L)[0], arc_proj(self.C[j], Sc, R)[0]
        Ls, Rs = L[(aL >= p["a0"]) & (aL <= p["a1"])], R[(aR >= p["a0"]) & (aR <= p["a1"])]
        if len(Ls) < 2 or len(Rs) < 2:
            return None
        g = sh.make_valid(sh.Polygon(np.r_[Ls, Rs[::-1]]))
        return g if not g.is_empty else None

    # ------------------------------------------------------------ classes
    def classify(self, K, ref, X):
        """World points X (m, 2) against corridor K (prepared polygon) with centre samples ref (x, y, heading, s) -> codes (m,) uint8."""
        sh = self.sh
        code = np.full(len(X), 3, np.uint8)
        if K is None or ref is None or not len(ref):
            code[:] = NOCOR
            return code
        inK = sh.contains_xy(K, X[:, 0], X[:, 1])
        code[inK] = 0
        r = np.flatnonzero(~inK)
        if not len(r):
            return code
        Xr = X[r]
        href = ref[np.argmin(((Xr[:, None, :] - ref[None, :, :2]) ** 2).sum(-1), 1), 2]
        ii, jj = self.tree.query(sh.points(Xr), predicate="within")
        same, onc, lan = (np.zeros(len(r), bool) for _ in range(3))
        for j in np.unique(jj):
            m = ii[jj == j]
            hd = arc_proj(self.C[j], self.Sc[j], Xr[m])[1]
            d = np.abs(wrap(hd - href[m]))
            lan[m] = True
            same[m[d < SAME]] = True
            onc[m[d > ONC]] = True
        area = sh.contains_xy(self.area, Xr[:, 0], Xr[:, 1]) if self.area is not None else np.zeros(len(r), bool)
        code[r] = np.where(same, 1, np.where(onc, 2, np.where(lan | area, 4, 3)))
        return code


def mask(codes):
    """codes (k, n) -> one bitmask per row (bit c = some point in code c)."""
    m = np.zeros(len(codes), np.int64)
    for c in (0, 1, 2, 3, 4):
        m |= np.where((codes == c).any(1), 1 << c, 0)
    return np.where((codes == NOCOR).any(1), -1, m)


class Corridors:
    """Corridor polygons of a chain, cached by the set of pieces in the station window."""

    def __init__(self, M, ch):
        self.M, self.ch, self.cache = M, ch, {}

    def at(self, s):
        sh = self.M.sh
        P = self.ch["pieces"]
        key = tuple(k for k, p in enumerate(P) if p["s1"] >= s - WIN_LO and p["s0"] <= s + WIN_HI and p["poly"] is not None)
        if key not in self.cache:
            if not key:
                self.cache[key] = (None, None, None)
            else:
                K = sh.union_all([P[k]["poly"] for k in key]).buffer(MARGIN)
                sh.prepare(K)
                self.cache[key] = (K, np.concatenate([P[k]["ref"] for k in key]), key)
        return self.cache[key]

    def found(self, s):
        sg, cov = self.ch["sg"], self.ch["cov"]
        if not len(sg):
            return False, False
        m = (sg >= s) & (sg <= min(s + FOUND_D, sg[-1]))
        lo, hi = s - WIN_LO, s + WIN_HI
        return bool(m.any() and cov[m].mean() >= FOUND_SHARE), bool(self.ch["amb"][(sg >= lo) & (sg <= hi)].any())


# ---------------------------------------------------------------- build
def _route_ego(o):
    import rf_progress as RP
    PF, TX = RP._imports()
    return RP, PF, TX, RP.Ego(o, None, PF, TX)


def roll_job(job):
    name, o, pl, sc, tax, frow, keep = job
    RP, PF, TX, e = _route_ego(o)
    M = SceneMap(sc)
    L = sc["log"]
    LT, LPp, LY = np.asarray(L["T"], float), np.asarray(L["P"], float), np.asarray(L["Y"], float)
    LS = stations(LPp)
    ls0, _ = PF.project(LPp, LS, e.R[0])                           # the route starts ahead of the log start: prefix the log before it
    pre = LPp[LS < ls0 - 0.5]
    Rx = PF.dedup(np.r_[pre, e.R]) if len(pre) else e.R
    Sx = PF.stations(Rx)
    off = float(Sx[len(Rx) - len(e.R)]) if len(pre) else 0.0
    ch = M.match(Rx, Sx, LPp)
    CR = Corridors(M, ch)
    FP = footprint(o["dims"])
    t0 = pl["t0"].astype(float)
    A, Pz = pl["anchor"], pl["poses"].astype(float)
    n = len(t0)
    s_e = np.interp(t0, e.t, e.st)                                  # route stations (rf_follow's entry / exit)
    s_x = np.interp(t0, e.t, PF.track(Rx, Sx, e.P)[0])              # stations on the prefixed route (corridor windows)
    s_end = max((p["s1"] for p in ch["pieces"]), default=0.0)        # where the route's lane chain ends
    grids, cpoly, rows = np.full((n, 48, 48), NOCOR, np.uint8), [], []
    hx = lambda t: np.c_[np.interp(t, LT, LPp[:, 0]), np.interp(t, LT, LPp[:, 1])]  # noqa: E731
    for i in range(n):
        a = A[i]
        fd, am = CR.found(s_x[i])
        K, ref, key = CR.at(s_x[i])
        r = dict(set=name, scene=o["scene"], rid=o["rid"], k=i, t0=t0[i], s_e=float(s_e[i]), s_x=float(s_x[i]), found=fd, amb=am, npieces=0 if key is None else len(key),
                 v=float(np.hypot(*(Pz[i, 1, :2] - Pz[i, 0, :2])) / 0.5), plan_len=float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], Pz[i, :, :2]], axis=0).T).sum()),
                 cor_left=float(s_end - s_x[i]))
        if GRID_ON:
            grids[i] = M.classify(K, ref, to_world(a, GRID)).reshape(48, 48)
            cpoly.append(None if K is None else M.sh.to_wkb(_to_frame(K, a)))
        pw = to_world(a, Pz[i, 1:, :2])
        cd = M.classify(K, ref, fp_world(pw, a[2] + Pz[i, 1:, 2], FP).reshape(-1, 2)).reshape(len(pw), -1)
        for q, mm in enumerate(mask(cd), 1):
            r[f"m{q}"] = int(mm)
        ie = e.i_at(t0[i])
        cd = M.classify(K, ref, fp_world(e.P[ie][None], np.array([e.yaw[ie]]), FP).reshape(-1, 2))
        r["eg"] = int(mask(cd[None])[0])
        th = t0[i] + POSE_T * 1e6                                   # the human: own station's corridor, own future footprint
        if t0[i] >= LT[0] and t0[i] <= LT[-1]:
            h0 = hx(t0[i])[0]
            s_h, _ = PF.project(Rx, Sx, h0)
            Kh, refh, _ = CR.at(s_h)
            r["s_h"], r["found_h"] = float(s_h - off), CR.found(s_h)[0]
            r["h0"] = int(M.classify(Kh, refh, h0[None])[0])
            ok = th <= LT[-1]
            if ok.any():
                cd = M.classify(Kh, refh, fp_world(hx(th[ok]), np.interp(th[ok], LT, LY), FP).reshape(-1, 2)).reshape(int(ok.sum()), -1)
                hm = mask(cd)
            for q in range(1, 8):
                r[f"h{q}"] = int(hm[int(ok[:q].sum()) - 1]) if ok[q - 1] else -2
        else:
            r.update(s_h=np.nan, found_h=False, h0=-2, **{f"h{q}": -2 for q in range(1, 8)})
        rows.append(r)
    roll = dict(set=name, scene=o["scene"], rid=o["rid"], score=o["score"], n_dec=n, n_pieces=len(ch["pieces"]), n_amb_samples=ch["n_amb"],
                n_change=ch["trans"].count("change"), n_break=ch["trans"].count("break"), route_m=float(e.S[-1]), prefix_m=off, t_first=float(e.t[0]),
                t_last=float(e.t[-1]), dt0=float(t0[0] - e.t[0]) if n else np.nan)
    if tax is not None:
        roll.update(zero_type=tax.get("zero_type"), primary=tax.get("primary"))
        zt = tax.get("zero_type")
        if isinstance(zt, str) and zt in ("collision", "offroad", "corridor") and tax.get("primary") not in RP.DROP:
            z = RP.zero_job((name, o, tax, None, frow))
            roll.update({k: z[k] for k in ("t_f", "turn_zero", "near_turn", "turn_entry", "turn_exit", "turn_deg", "cls", "rule_cls", "shortcut", "v_f")})
            tf = z["t_f"]
            if np.isfinite(z["turn_entry"]):
                stg = np.interp(o["tg"], e.t, e.st)
                w = (stg >= z["turn_entry"] - PF.PRE) & (stg <= z["turn_exit"] + PF.PAST) & (o["tg"] < tf)
            else:
                w = (o["tg"] >= tf - 5e6) & (o["tg"] < tf)
            roll["onc_touch"] = bool((np.asarray(o["lane"])[w] == 2).any()) if w.any() else False
    if frow is not None and bool(frow.get("turn")):
        ent, ext = float(frow["entry"]), float(frow["exit"])
        j = np.flatnonzero(e.st >= ext + PF.PAST)
        roll.update(turn=True, entry=ent, exit=ext, side=frow.get("side"), turn_deg_f=frow.get("turn_deg"), outcome=frow.get("outcome"),
                    t_exit=float(e.t[j[0]]) if len(j) else np.nan)
    out = dict(rows=rows, roll=roll, labels=dict(t0=pl["t0"], grid=grids, cpoly=cpoly, found=np.array([r["found"] for r in rows]),
                                                amb=np.array([r["amb"] for r in rows])))
    if keep:
        out["geo"] = dict(M=M, ch=ch, CR=CR, e=e, o=o, pl=pl)
    return out


def _to_frame(K, a):
    """World polygon -> the anchor's rear-axle frame (x forward, y left), simplified at 5 cm."""
    import shapely
    import shapely.affinity as AF
    c, s = np.cos(a[2]), np.sin(a[2])
    return shapely.simplify(AF.affine_transform(K, [c, s, -s, c, -(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1])]), 0.05)


def load_sets(specs):
    """NAME=<tax pkl>,<plans pkl>,<lanes pkl>,<tax csv>,<follow csv>,<follow label> -> per set (name, rolls, plans, lanes, tax df, follow df)."""
    import pandas as pd
    out = []
    lanes_cache = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        fp, fpl, fl, fc, ff, lab = rest.split(",")
        rolls = [o for o in pickle.load(open(fp, "rb"))["rolls"] if "ctrl" in o]
        pl = pickle.load(open(fpl, "rb"))
        if fl not in lanes_cache:
            lanes_cache[fl] = pickle.load(open(fl, "rb"))["scenes"]
        tax = pd.read_csv(fc).set_index("scene")
        fo = pd.read_csv(ff)
        fo = fo[fo.label == lab].set_index("scene")
        out.append((name, rolls, pl, lanes_cache[fl], tax, fo))
    return out


def jobs_of(sets, keep=None):
    J, miss = [], {}
    for name, rolls, pl, lanes, tax, fo in sets:
        for o in rolls:
            if keep is not None and (name, o["scene"]) not in keep:
                continue
            p, sc = pl.get(o["rid"]), lanes.get(o["scene"])
            if p is None or sc is None or not len(p["t0"]):
                miss[name] = miss.get(name, 0) + 1
                continue
            t = tax.loc[o["scene"]].to_dict() if o["scene"] in tax.index else None
            f = fo.loc[o["scene"]].to_dict() if o["scene"] in fo.index else None
            J.append((name, o, p, sc, t, f, keep is not None))
    return J, miss


def build(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "corridor_build", config=dict(vars(a), margin=MARGIN, fp=FP_MODE, grid=GRID_ON)) as run:
        sets = load_sets(a.set)
        J, miss = jobs_of(sets)
        run.info(f"{len(J)} rollouts; missing plans / lanes {miss}")
        res = par.pmap(roll_job, J, run=run, desc="rollouts", workers=a.workers or None)
        if res.errors:
            run.info(f"{len(res.errors)} errors: {list(res.errors.values())[:2]}")
        D, R, lab = [], [], {}
        for v in res.values:
            if v is None:
                continue
            D += v["rows"]
            R.append(v["roll"])
            lab.setdefault(v["roll"]["set"], {})[v["roll"]["rid"]] = v["labels"]
        D, R = pd.DataFrame(D), pd.DataFrame(R)
        D.to_parquet(out / "dec.parquet")
        R.to_parquet(out / "roll.parquet")
        for k, v in (lab.items() if GRID_ON else ()):
            pickle.dump(v, open(out / f"labels_{k}.pkl", "wb"), protocol=4)
        cov = coverage(D, R)
        cov["missing"], cov["errors"] = miss, len(res.errors)
        (out / "coverage.json").write_text(json.dumps(cov, indent=1, default=float) + "\n")
        (out / "coverage.md").write_text(coverage_md(cov))
        print((out / "coverage.md").read_text())
        run.summary.update(rollouts=len(R), decisions=len(D))


def turn_window(D, R, who="e"):
    """Decisions of strict-turn rollouts with the ego (who = e) or the human (h) station in [entry - 20, exit + 10] (ego: before t_exit)."""
    T = R[R.get("turn", pd_false(R)).fillna(False).astype(bool)][["set", "rid", "entry", "exit", "outcome", "t_exit", "side"]]
    X = D.merge(T, on=["set", "rid"])
    s = X.s_e if who == "e" else X.s_h
    m = (s >= X.entry - 20.0) & (s <= X.exit + 10.0)
    if who == "e":
        m &= ~(X.t0 >= X.t_exit)
    return X[m]


def pd_false(R):
    import pandas as pd
    return pd.Series(False, index=R.index)


def coverage(D, R):
    src = lambda s: "pai" if s.startswith("pai") else "held"  # noqa: E731
    D = D.assign(src=D.set.map(src))
    W = turn_window(D, R).assign(src=lambda x: x.set.map(src))
    out = {}
    for nm, X in (("all decisions", D), ("strict-turn window (ego)", W)):
        for g, x in [(s, x) for s, x in X.groupby("src")] + [(f"set {s}", x) for s, x in X.groupby("set")]:
            out[f"{nm}|{g}"] = dict(n=int(len(x)), found=float(x.found.mean()), amb=float(x.amb.mean()), found_h=float(x.found_h.mean()),
                                    rollouts=int(x.rid.nunique()), scenes=int(x.scene.nunique()))
    out["rollouts"] = {s: dict(n=int(len(x)), with_change=int((x.n_change > 0).sum()), with_break=int((x.n_break > 0).sum()),
                               with_amb=int((x.n_amb_samples > 0).sum()), dt0_med_s=float(x.dt0.median() / 1e6)) for s, x in R.groupby("set")}
    return out


def coverage_md(c):
    md = ["# Route-lane corridor labels: coverage (rf_corridor.py build)", "",
          "Found = the route's lane chain covers >= 90 % of the 1 m route samples in [s_e, s_e + 40] m (s_e = the ego's route station at the "
          "decision); ambiguous = a route sample in the corridor window had two distinct candidate lanes (resolved by the human's path); "
          "found_h = the same at the human's own station (used by the label sanity check).", "",
          "| population | source | decisions | rollouts | scenes | found | ambiguous | found (human station) |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for k, v in c.items():
        if "|" not in k:
            continue
        p, s = k.split("|")
        md.append(f"| {p} | {s} | {v['n']} | {v['rollouts']} | {v['scenes']} | {v['found']:.3f} | {v['amb']:.3f} | {v['found_h']:.3f} |")
    md += ["", "| set | rollouts | with a lane change in the chain | with a break | with ambiguous samples | first decision - first controller sample (s, median) |",
           "|:--|--:|--:|--:|--:|--:|"]
    for s, v in c["rollouts"].items():
        md.append(f"| {s} | {v['n']} | {v['with_change']} | {v['with_break']} | {v['with_amb']} | {v['dt0_med_s']:+.2f} |")
    md += ["", f"Rollouts without plans or lanes: {c.get('missing')}; job errors: {c.get('errors')}."]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- check
def hard(m):
    return (m >= 0) & ((m & HARD) > 0)


def plan_cols(X, prefix="m"):
    M = np.nan_to_num(np.stack([X[f"{prefix}{q}"].to_numpy(float) for q in range(1, 8)], 1), nan=-1).astype(np.int64)
    ok = (M >= 0).any(1)
    H = np.where(M >= 0, M, 0)
    return dict(hard=np.where(ok, ((H & HARD) > 0).any(1), np.nan), onc=np.where(ok, ((H & 4) > 0).any(1), np.nan),
                off=np.where(ok, ((H & 8) > 0).any(1), np.nan), soft=np.where(ok, ((H & SOFT) > 0).any(1) & ~((H & HARD) > 0).any(1), np.nan),
                first_hard=np.where(((H & HARD) > 0).any(1), POSE_T[np.argmax((H & HARD) > 0, 1)], np.nan))


def probes(D, R):
    """One row per (zero, probe D in 1, 2, 3 s): the last decision with t0 <= t_f - D and > t_f - D - 1 s."""
    Z = R[R.t_f.notna()] if "t_f" in R else R.iloc[:0]
    out = []
    G = {k: x.sort_values("t0") for k, x in D.groupby(["set", "rid"])}
    for _, z in Z.iterrows():
        x = G.get((z.set, z.rid))
        for dd in (1, 2, 3):
            r = dict(set=z.set, rid=z.rid, probe=dd)
            if x is not None:
                y = x[(x.t0 <= z.t_f - dd * 1e6) & (x.t0 > z.t_f - (dd + 1) * 1e6)]
                if len(y):
                    r.update(y.iloc[-1].to_dict())
                    r["probe"] = dd
            out.append(r)
    import pandas as pd
    P = pd.DataFrame(out)
    if "m1" in P:
        for k, v in plan_cols(P.fillna({f"m{q}": -1 for q in range(1, 8)})).items():
            P[k] = v
        P.loc[P.t0.isna(), ["hard", "onc", "off", "soft"]] = np.nan
    return P


def check(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.run import Run
    b, out = Path(a.build), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "corridor_check", config=vars(a)) as run:
        D, R = pd.read_parquet(b / "dec.parquet"), pd.read_parquet(b / "roll.parquet")
        src = lambda s: "pai" if s.startswith("pai") else "held"  # noqa: E731
        clu = lambda X: X.set.map(src) + ":" + X.scene  # noqa: E731
        for f in ("coverage.md", "coverage.json"):
            (out / f).write_text((b / f).read_text())
        res = dict(gh={}, c1={}, c2={}, c3={}, sec={}, counts={})
        # --- G-H: the human's own footprint in the strict-turn windows (human station), per source, dedup by scene x t0 (held)
        H = turn_window(D, R, who="h")
        H = H[H.h0 != -2].assign(src=lambda x: x.set.map(src))
        H = H.drop_duplicates(["src", "scene", "t0"])
        hp = plan_cols(H, "h")
        H = H.assign(h_hard=hp["hard"], h_soft=hp["soft"], h_axle_in=(H.h0 == 0).astype(float).where(H.h0 >= 0), h_onc=hp["onc"], h_off=hp["off"])
        for s, x in H.groupby("src"):
            r = dict(hard=stats.bootstrap(x.h_hard, groups=x.scene), axle_in=stats.bootstrap(x.h_axle_in, groups=x.scene),
                     onc=stats.bootstrap(x.h_onc, groups=x.scene), off=stats.bootstrap(x.h_off, groups=x.scene), soft=stats.bootstrap(x.h_soft, groups=x.scene))
            r["pass"] = bool(r["hard"]["mean"] <= 0.05)
            res["gh"][s] = r
        Hall = D[(D.h0 != -2)].assign(src=lambda x: x.set.map(src)).drop_duplicates(["src", "scene", "t0"])
        hq = plan_cols(Hall, "h")
        for s, x in Hall.assign(h_hard=hq["hard"]).groupby("src"):
            res["gh"][f"{s} (all decisions)"] = dict(hard=stats.bootstrap(x.h_hard, groups=x.scene))
        void = {s for s, r in res["gh"].items() if "(" not in s and not r["pass"]}
        # --- zeros at the probes
        P = probes(D, R)
        Z = R[R.t_f.notna() & R.turn_zero.fillna(False).astype(bool)].copy()
        Z["c1"] = (Z.cls == "over_turn") | ((Z.zero_type == "corridor") & Z.onc_touch.fillna(False).astype(bool))
        hit = P.groupby(["set", "rid"]).hard.max().rename("hit")                  # any defined probe hard (NaN = none defined)
        Z = Z.merge(hit, left_on=["set", "rid"], right_index=True, how="left")
        for dd in (1, 2, 3):
            q = P[P.probe == dd].set_index(["set", "rid"])
            Z = Z.merge(q[["hard", "onc", "off", "soft", "first_hard"]].add_suffix(f"_{dd}"), left_on=["set", "rid"], right_index=True, how="left")
            Z = Z.merge(q[["eg"]].rename(columns={"eg": f"eg_{dd}"}), left_on=["set", "rid"], right_index=True, how="left")
        Gd = {k: x for k, x in D.groupby(["set", "rid"])}                         # secondary: any decision in [t_f - 3, t_f - 1]
        Z["hit_win"] = [np.nan if (x := Gd.get((z.set, z.rid))) is None or not len(y := x[(x.t0 >= z.t_f - 3e6) & (x.t0 <= z.t_f - 1e6)])
                        else float(hard(np.stack([y[f"m{q}"].to_numpy(np.int64) for q in range(1, 8)], 1)).any()) for _, z in Z.iterrows()]
        Z["eg_onc"] = [(int(z.eg_1) & 4) > 0 if np.isfinite(z.eg_1) and z.eg_1 >= 0 else np.nan for _, z in Z.iterrows()]
        Z["src"], Z["cluster"] = Z.set.map(src), clu(Z)
        Z.to_csv(out / "zeros.csv", index=False)

        def share(x, col="hit"):
            y = x[x[col].notna()]
            return dict(stats.bootstrap(y[col].astype(float), groups=y.cluster), n_pop=int(len(x)), n_undef=int(x[col].isna().sum()),
                        hits=int(y[col].astype(float).sum()))
        C1 = Z[Z.c1]
        res["c1"]["pooled"] = share(C1[~C1.src.isin(void)])
        res["c1"]["pooled"]["holds"] = bool(res["c1"]["pooled"]["mean"] >= 0.70)
        res["c1"]["pooled"]["robust"] = bool(res["c1"]["pooled"]["lo"] >= 0.70 or res["c1"]["pooled"]["hi"] < 0.70)
        for s, x in C1.groupby("set"):
            res["c1"][s] = share(x)
        for nm, m in (("over_turn", C1.cls == "over_turn"), ("corridor + oncoming touch (not over_turn)", C1.cls != "over_turn")):
            res["c1"][f"class {nm}"] = share(C1[m])
        for col in ("hit_win", "hard_1", "hard_2", "hard_3"):
            res["sec"][f"c1 {col}"] = share(C1, col)
        # --- C3: other turn-zero classes
        for c, x in Z[~Z.c1].groupby("cls"):
            res["c3"][f"class {c}"] = share(x)
            for s, y in x.groupby("set"):
                res["c3"][f"class {c}|{s}"] = share(y)
        # --- C2: completed strict turns, decisions in the ego's turn window
        W = turn_window(D, R)
        pc = plan_cols(W)
        W = W.assign(hard=pc["hard"], onc=pc["onc"], off=pc["off"], soft=pc["soft"], src=lambda x: x.set.map(src))
        W["cluster"] = clu(W)
        W.groupby(["set", "scene", "rid", "side", "outcome"]).agg(decisions=("hard", "size"), hard=("hard", "mean"), onc=("onc", "mean"), off=("off", "mean"),
                                                                 soft=("soft", "mean")).reset_index().to_csv(out / "turns.csv", index=False)   # one row per strict turn
        Wc = W[W.outcome == "completed"]
        res["c2"]["pooled"] = dict(stats.bootstrap(Wc[~Wc.src.isin(void)].hard, groups=Wc[~Wc.src.isin(void)].cluster),
                                   turns=int(Wc[~Wc.src.isin(void)].groupby(["set", "rid"]).ngroups))
        res["c2"]["pooled"]["holds"] = bool(res["c2"]["pooled"]["mean"] <= 0.10)
        res["c2"]["pooled"]["robust"] = bool(res["c2"]["pooled"]["hi"] <= 0.10 or res["c2"]["pooled"]["lo"] > 0.10)
        for s, x in Wc.groupby("set"):
            res["c2"][s] = dict(stats.bootstrap(x.hard, groups=x.cluster), turns=int(x.rid.nunique()), onc=stats.bootstrap(x.onc, groups=x.cluster)["mean"],
                                off=stats.bootstrap(x.off, groups=x.cluster)["mean"], soft=stats.bootstrap(x.soft, groups=x.cluster)["mean"])
        tr = Wc.groupby(["set", "rid"]).hard.max()
        res["sec"]["c2 completed turns with any hard decision"] = dict(n=int(len(tr)), k=int(tr.sum()))
        for s, x in W[W.outcome != "completed"].groupby("set"):
            res["c3"][f"failed strict turns (all window decisions)|{s}"] = dict(stats.bootstrap(x.hard, groups=x.cluster), turns=int(x.rid.nunique()))
        res["counts"] = dict(zeros=int(R.t_f.notna().sum()), turn_zeros=int(len(Z)), c1=int(len(C1)), void_sources=sorted(void),
                             by_set={s: dict(turn_zeros=int((Z.set == s).sum()), c1=int((C1.set == s).sum())) for s in sorted(R.set.unique())})
        res["verdict"] = dict(C1=res["c1"]["pooled"]["holds"] if res["c1"]["pooled"]["n"] else "void", C2=res["c2"]["pooled"]["holds"] if res["c2"]["pooled"]["n"] else "void",
                              GH={s: r["pass"] for s, r in res["gh"].items() if "pass" in r})
        res["c1"]["all sources (descriptive, ignores G-H)"] = share(C1)                # exploratory: the same numbers if the labels were trusted
        res["c2"]["all sources (descriptive, ignores G-H)"] = dict(stats.bootstrap(Wc.hard, groups=Wc.cluster), turns=int(Wc.groupby(["set", "rid"]).ngroups),
                                                                    onc=np.nan, off=np.nan, soft=np.nan)
        (out / "check.json").write_text(json.dumps(res, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
        (out / "check.md").write_text(check_md(res, Z))
        print((out / "check.md").read_text())
        run.summary.update(c1=res["c1"]["pooled"]["mean"], c2=res["c2"]["pooled"]["mean"])
        if a.fig:
            fig_bev(a, D, R, Z, W)


def check_md(res, Z):
    f = lambda r: f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]" if r.get("n") else "-"  # noqa: E731
    v = res["verdict"]
    c1, c2 = res["c1"]["pooled"], res["c2"]["pooled"]
    vd = lambda ok, r: "VOID (G-H failed on every source: no labelled population left)" if not r.get("n") else \
        ("holds" if ok else "fails") + (" (robust)" if r.get("robust") else "")  # noqa: E731
    md = ["# Route-lane corridor check (rf_corridor.py check)", "",
          "Prereg `plans/2026-10-15-corridor-check-prereg.md`. Plan hard exit = a footprint point of a served 1-4 s pose in the oncoming lane or "
          "off the road (outside the route-lane corridor + 0.3 m); probes = the decisions at t_f - 1 / 2 / 3 s. Shares with scene-clustered "
          "bootstrap 95 % CI (B = 10 000).", "",
          f"**C1 (>= 0.70 of inside / oncoming / over-turn turn zeros have a hard-exit plan at t_f - 1 s or earlier): {f(c1)}, {c1['hits']} / "
          f"{c1['n']} defined ({c1['n_undef']} without a probe) -> {vd(v['C1'], c1)}.**", "",
          f"**C2 (<= 0.10 of the completed strict turns' window decisions have a hard-exit plan): {f(c2)} over {c2['n']} decisions of "
          f"{c2['turns']} turns -> {vd(v['C2'], c2)}.**", "",
          "**G-H (label sanity: the human's own 1-4 s footprint in the strict-turn windows, hard exit <= 0.05): "
          + "; ".join(f"{s} {f(r['hard'])} ({r['hard']['n']} decisions) -> {'pass' if r['pass'] else 'FAIL, labels void'}" for s, r in res["gh"].items() if "pass" in r)
          + f"**. Void sources: {res['counts']['void_sources'] or 'none'}.", "",
          "## G-H detail (human)", "", "| source / population | hard exit | oncoming | off-road | soft only | rear axle inside the corridor |", "|:--|:--|:--|:--|:--|:--|"]
    for s, r in res["gh"].items():
        md.append(f"| {s} | {f(r['hard'])} | {f(r['onc']) if 'onc' in r else '-'} | {f(r['off']) if 'off' in r else '-'} | "
                  f"{f(r['soft']) if 'soft' in r else '-'} | {f(r['axle_in']) if 'axle_in' in r else '-'} |")
    md += ["", "## C1 / C3: turn zeros, plan hard exit at a probe", "", "| population | hit share [CI] | hits / defined | no probe |", "|:--|:--|--:|--:|"]
    for grp in ("c1", "c3"):
        for k, r in res[grp].items():
            if "mean" in r and "n_pop" in r:
                md.append(f"| {grp.upper()} {k} | {f(r)} | {r['hits']} / {r['n']} | {r['n_undef']} |")
    md += ["", "Secondary (not registered):", ""]
    for k, r in res["sec"].items():
        md.append(f"- {k}: " + (f"{f(r)} ({r['hits']} / {r['n']})" if "mean" in r else json.dumps(r)))
    md += ["", "## C2: completed strict turns, window decisions", "", "| set | hard exit [CI] | decisions | turns | oncoming | off-road | soft only |",
           "|:--|:--|--:|--:|--:|--:|--:|"]
    for k, r in res["c2"].items():
        if k == "pooled" or k.startswith("all sources"):
            md.append(f"| {k} | {f(r)} | {r['n']} | {r['turns']} | | | |")
        else:
            md.append(f"| {k} | {f(r)} | {r['n']} | {r['turns']} | {r['onc']:.2f} | {r['off']:.2f} | {r['soft']:.2f} |")
    for k, r in res["c3"].items():
        if k.startswith("failed"):
            md.append(f"| {k} | {f(r)} | {r['n']} | {r['turns']} | | | |")
    md += ["", "## Turn zeros (per zero)", "", "| set | scene | class | zero type | C1 | onc touch | hit | t_f-1 | t_f-2 | t_f-3 | first hard pose (s) at t_f-1 / -2 / -3 |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    b = lambda x: "-" if not np.isfinite(x) else ("H" if x else ".")  # noqa: E731
    for _, z in Z.sort_values(["c1", "set", "cls"], ascending=[False, True, True]).iterrows():
        sc = z.scene[7:15] if z.scene.startswith("clipgt") else z.scene[-16:]
        fh = " / ".join("-" if not np.isfinite(z[f"first_hard_{d}"]) else f"{z[f'first_hard_{d}']:.1f}" for d in (1, 2, 3))
        md.append(f"| {z.set} | {sc} | {z.cls} | {z.zero_type} | {'C1' if z.c1 else ''} | {bool(z.onc_touch)} | {b(z.hit)} | {b(z.hard_1)} | {b(z.hard_2)} | {b(z.hard_3)} | {fh} |")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- figure
CMAP = {0: "#cfe8cf", 1: "#fff1b8", 2: "#f4b6b6", 3: "#9e9e9e", 4: "#dcdcf0"}


def fig_bev(a, D, R, Z, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    picks = [p.split(":") for p in a.picks.split(",")]            # set:scene8[:completed]
    sets = load_sets(a.set)
    keep, want = set(), []
    for p in picks:
        nm, sc8 = p[0], p[1]
        for name, rolls, *_ in sets:
            if name == nm:
                o = next((o for o in rolls if sc8 in o["scene"]), None)
                if o is not None:
                    keep.add((name, o["scene"]))
                    want.append((name, o["scene"], len(p) > 2))
    J, _ = jobs_of(sets, keep)
    geo = {(j[0], j[1]["scene"]): roll_job(j) for j in J}
    n = len(want)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(6.2 * ((n + 1) // 2), 12.5))
    cm = ListedColormap([CMAP[i] for i in range(5)])
    for ax, (name, sc, done) in zip(np.ravel(axs), want):
        g = geo[(name, sc)]
        e, M, CR, rl = g["geo"]["e"], g["geo"]["M"], g["geo"]["CR"], g["roll"]
        o, pl = g["geo"]["o"], g["geo"]["pl"]
        rows = {r["k"]: r for r in g["rows"]}
        t0 = np.array([r["t0"] for r in g["rows"]])
        if done:
            j = np.flatnonzero(e.st >= rl["entry"])
            t_ref = e.t[j[0]] if len(j) else e.t[len(e.t) // 2]
            tt, title = t_ref + 2e6, f"completed strict turn ({rl.get('side')})"   # completed: the decision 2 s past the entry crossing
        else:
            t_ref = rl["t_f"]
            tt, title = t_ref - 2e6, f"{rl.get('cls')} zero ({rl.get('zero_type')})"
        k = int(np.clip(np.searchsorted(t0, tt, side="right") - 1, 0, len(t0) - 1))
        r = rows[k]
        K, ref, _ = CR.at(r["s_x"])
        anc, pz = pl["anchor"][k], pl["poses"][k].astype(float)
        FP = footprint(o["dims"])
        c = e.P[e.i_at(t0[k])]
        half = 32.0
        xs, ys = np.arange(c[0] - half, c[0] + half, 0.4), np.arange(c[1] - half, c[1] + half, 0.4)
        X = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        code = M.classify(K, ref, X).reshape(len(ys), len(xs)).astype(float)
        code[code == NOCOR] = np.nan
        ax.imshow(code, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=cm, vmin=-0.5, vmax=4.5, interpolation="nearest")
        if K is not None:
            for gg in getattr(K, "geoms", [K]):
                ax.plot(*np.asarray(gg.exterior.coords).T, color="#2e7d32", lw=0.9)
        ax.plot(e.LP[:, 0], e.LP[:, 1], color="#1565c0", lw=1.6, label="human log path")
        tend = rl["t_f"] if not done and np.isfinite(rl.get("t_f", np.nan)) else e.t[-1]
        m = e.t <= tend
        ax.plot(e.P[m, 0], e.P[m, 1], color="#e65100", lw=2.0, label="driven path")
        pw = to_world(anc, pz[:, :2])
        ax.plot(np.r_[anc[0], pw[:, 0]], np.r_[anc[1], pw[:, 1]], "-o", color="#6a1b9a", ms=3, lw=1.4, label="served plan (0.5-4 s)")
        for q in range(1, 8):
            ft = fp_world(pw[q][None], np.array([anc[2] + pz[q, 2]]), FP)[0]
            hh = (r[f"m{q}"] >= 0) and (r[f"m{q}"] & HARD)
            ax.plot(*np.r_[ft[[0, 5, 11, 6, 0]]].T, color="#b71c1c" if hh else "#6a1b9a", lw=1.0 if hh else 0.6)
        ax.plot(*anc[:2], marker="s", color="black", ms=6, zorder=6)
        if not done:
            ax.plot(*e.P[e.i_at(rl["t_f"])], marker="X", color="#b71c1c", ms=11, zorder=6)
        hs = ",".join(f"{q * 0.5 + 0.5:.1f}" for q in range(1, 8) if r[f"m{q}"] >= 0 and r[f"m{q}"] & HARD)
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        lab = sc[7:15] if sc.startswith("clipgt") else sc[-16:]
        dt = (t0[k] - t_ref) / 1e6
        ax.set_title(f"{name} {lab} score {rl['score']:.2f}\n{title}; decision at {'t_f' if not done else 'entry'} {dt:+.1f} s; "
                     f"hard-exit poses: {hs or 'none'}", fontsize=9)
    for ax in np.ravel(axs)[n:]:
        ax.axis("off")
    hdl = [Patch(color=CMAP[i], label=l) for i, l in enumerate(("corridor (route lane + 0.3 m)", "adjacent same-direction lane (soft)",
                                                                  "oncoming lane (hard)", "off-road (hard)", "other drivable: crossing lane / road area (soft)"))]
    h2, l2 = np.ravel(axs)[0].get_legend_handles_labels()
    fig.legend(handles=hdl + h2, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Route-lane corridor classes at the decision 2 s before the zero (completed turn: 2 s past the entry); square = the decision's "
                 "anchor, X = the zero event;\nplan footprints in red = a 1-4 s pose with a footprint point in a hard class (the hinge would "
                 "push there), purple = not", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("lanes")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("plans")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("build")
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=0)
    p = sp.add_parser("check")
    p.add_argument("--build", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fig", default="")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--picks", default="")
    a = ap.parse_args()
    {"lanes": lanes_extract, "plans": plans_extract, "build": build, "check": check}[a.cmd](a)
