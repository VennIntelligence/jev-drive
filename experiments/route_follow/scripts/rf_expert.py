#!/usr/bin/env python3
"""RULE EXPERT (plans/2026-10-15-rule-expert-prereg.md): the per-scene packs the expert drives on (experiments/alpasim/lib/rule_expert.py),
the BEV figure of the expert's chain against the human and the driven paths, and the per-turn failure list. CPU only; run dirs are read,
never written.

  pack  (box: AlpaSim venv; Tokyo: rf_expert.sh pack-tokyo, inside alpasim-base:0.89.0)
        rf_expert.py pack --runs <held run dir | PAI run root> ... --out DIR [--scenes FILE] [--workers N]
        per scene DIR/<scene>.npz (numpy only, what the driver reads: path, s, turns, vlog, dims, replayed boxes, log) and
        DIR/geo/<scene>.pkl (lanes within 60 m of the path, road areas as WKB, the matched chain's pieces: figures only); DIR/index.json
        (per scene: pieces, ambiguous samples, lane changes, breaks, fallback, path length, turns, objects; errors)
        Chain: rf_corridor.SceneMap.match on the log before the route start + AlpaSim's MAP route (40 m) generated along the log's 0.5 s
        poses, ambiguous candidates resolved toward the route points; pieces joined at the midpoint between their station ranges, a lane
        change blended (smoothstep) over [mid - 15, mid + 15] m inside both pieces' overlap; Amendment 1: where the chain lies > 2.5 m from
        the route at the same station (or is absent: the log prefix before a chain that starts late), the route polyline is used instead
        (dilated 5 m, weight smoothed over 10 m; index `on_route` = that share); resampled at 0.5 m; 30 m straight before the start,
        150 m after the end. No piece -> the route polyline itself (fallback). Amendment 1: shifted sideways around parked replayed
        boxes that intrude on the ego's width (nudge(); index `nudges`).
  fig   (repo .venv / host python3 with matplotlib, shapely)
        rf_expert.py fig --pack DIR --runs LABEL=<run root | run dir>,.. ... --scenes s1,s2,.. --out PNG
        one panel per scene: road area / lanes (grey), the human log (blue), the expert chain (black dashed, turn entry / exit ticks),
        each label's driven rear-axle path from drive.jsonl anchors (first label red, second orange), the first zero-score position (x)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + ([str(HERE.parents[1] / "junction_eyes" / "lib")] if len(HERE.parents) > 2 else [])

STEP, PRE_M, POST_M, BLEND, ROUTE_TOL = 0.5, 30.0, 150.0, 15.0, 2.5
TURN_WIN, TURN_DEG, CHORD = 30.0, 45.0, 5.0
NUDGE_M, NUDGE_MAX, NUDGE_PRE, NUDGE_POST, NUDGE_RAMP, STATIC_M = 0.3, 1.8, 12.0, 8.0, 10.0, 1.0
GEO_R = 60.0


def stations(X):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(X, axis=0).T))]


def dedup(X, eps=1e-3):
    return X[np.r_[True, np.hypot(*np.diff(X, axis=0).T) > eps]]


def resample(X, step=STEP):
    S = stations(X)
    q = np.r_[np.arange(0.0, S[-1], step), S[-1]]
    return np.c_[np.interp(q, S, X[:, 0]), np.interp(q, S, X[:, 1])]


def proj_seq(P, S, X, win=(-5.0, 30.0)):
    """Points X (in travel order) on polyline P -> stations, each searched near the previous one (the first: globally)."""
    out, h = np.zeros(len(X)), None
    for i, x in enumerate(X):
        m = np.ones(len(P) - 1, bool) if h is None else (S[1:] >= h + win[0]) & (S[:-1] <= h + win[1])
        if not m.any():
            m[:] = True
        j = np.flatnonzero(m)
        A, d = P[j], P[j + 1] - P[j]
        L2 = np.maximum((d * d).sum(1), 1e-12)
        u = np.clip(((x - A) * d).sum(1) / L2, 0, 1)
        e = ((A + u[:, None] * d - x) ** 2).sum(1)
        k = int(np.argmin(e))
        h = out[i] = S[j[k]] + u[k] * np.sqrt(L2[k])
    return out


# ---------------------------------------------------------------- the chain -> one path
def assemble(pieces):
    """rf_corridor.SceneMap.match pieces (ref: x, y, heading, route station) -> path (n, 2) in route-station order, lane changes blended."""
    P = [p for p in pieces if p.get("ref") is not None and len(p["ref"]) >= 2]
    if not P:
        return None, None, 0
    lo, hi = P[0]["ref"][0, 3], P[-1]["ref"][-1, 3]
    g = np.arange(lo, hi, STEP)
    bnd = np.maximum.accumulate(np.array([0.5 * (P[k]["s1"] + P[k + 1]["s0"]) for k in range(len(P) - 1)]))
    k = np.searchsorted(bnd, g)
    xy = np.zeros((len(g), 2))
    f = lambda p, s: np.c_[np.interp(s, p["ref"][:, 3], p["ref"][:, 0]), np.interp(s, p["ref"][:, 3], p["ref"][:, 1])]  # noqa: E731
    for i, p in enumerate(P):
        m = k == i
        xy[m] = f(p, g[m])
    nlc = 0
    for i in range(len(P) - 1):
        if P[i]["tout"] != "change":
            continue
        a, b = P[i], P[i + 1]
        w0, w1 = max(b["ref"][0, 3], bnd[i] - BLEND), min(a["ref"][-1, 3], bnd[i] + BLEND)
        m = (g >= w0) & (g <= w1)
        if w1 - w0 < 2.0 or not m.any():
            continue
        u = (g[m] - w0) / (w1 - w0)
        w = (u * u * (3 - 2 * u))[:, None]
        xy[m] = (1 - w) * f(a, g[m]) + w * f(b, g[m])
        nlc += 1
    return g, xy, nlc


def merge_route(g, xy, Rx, Sx):
    """Amendment 1: the chain where it lies within ROUTE_TOL of the route at the same route station, else the route polyline itself (the
    log prefix before the route start); the bad stretches dilated by 5 m, the weight smoothed over 10 m. -> path, share of stations on the
    route."""
    q = np.arange(0.0, Sx[-1], STEP)
    R = np.c_[np.interp(q, Sx, Rx[:, 0]), np.interp(q, Sx, Rx[:, 1])]
    C = R.copy()
    good = np.zeros(len(q), bool)
    if g is not None and len(g):
        m = (q >= g[0]) & (q <= g[-1])
        C[m] = np.c_[np.interp(q[m], g, xy[:, 0]), np.interp(q[m], g, xy[:, 1])]
        good[m] = np.hypot(*(C[m] - R[m]).T) <= ROUTE_TOL
    n = int(round(5.0 / STEP))
    bad = np.convolve((~good).astype(float), np.ones(2 * n + 1), mode="same") > 0
    w = np.convolve(np.pad((~bad).astype(float), 2 * n, mode="edge"), np.ones(4 * n + 1) / (4 * n + 1), mode="valid")
    out = w[:, None] * C + (1 - w[:, None]) * R
    return dedup(out, 0.05), float((w < 0.5).mean())


def extend(X):
    """30 m straight before the start, 150 m after the end, along the end headings; resampled at STEP."""
    def head(a, b):
        d = b - a
        return d / max(np.hypot(*d), 1e-9)
    S = stations(X)
    a = X[0] - head(X[0], X[min(np.searchsorted(S, 3.0), len(X) - 1)]) * PRE_M
    b = X[-1] + head(X[max(np.searchsorted(S, S[-1] - 3.0) - 1, 0)], X[-1]) * POST_M
    return resample(dedup(np.r_[[a], X, [b]]))


def heading(P, S):
    """Chord heading over +-CHORD m, unwrapped."""
    lo, hi = np.maximum(S - CHORD, 0), np.minimum(S + CHORD, S[-1])
    d = np.c_[np.interp(hi, S, P[:, 0]) - np.interp(lo, S, P[:, 0]), np.interp(hi, S, P[:, 1]) - np.interp(lo, S, P[:, 1])]
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def turns(P, S, h=None):
    """Turns on the path: runs of stations a with |h(a + 30) - h(a)| >= 45 deg (same sign); the change TH over [a0, a1 + 30]; entry / exit
    where 10 % / 90 % of TH is reached. -> (k, 3): entry, exit, TH deg (left +)."""
    h = heading(P, S) if h is None else h
    dh = np.interp(S + TURN_WIN, S, h) - h
    dh[S + TURN_WIN > S[-1]] = 0.0
    big = np.abs(dh) >= np.radians(TURN_DEG)
    out, i = [], 0
    while i < len(S):
        if not big[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(S) and big[j + 1] and np.sign(dh[j + 1]) == np.sign(dh[i]):
            j += 1
        e = min(S[j] + TURN_WIN, S[-1])
        m = (S >= S[i]) & (S <= e)
        rel = h[m] - h[i]
        th = float(rel[np.argmax(np.abs(rel))])
        sm = S[m]
        en = float(sm[np.argmax(np.abs(rel) >= 0.1 * abs(th))])
        ex = float(sm[np.argmax(np.abs(rel) >= 0.9 * abs(th))])
        out.append((en, ex, np.degrees(th)))
        i = int(np.searchsorted(S, e)) + 1
    return np.array(out, float).reshape(-1, 3)


def vlog_of(P, S, T, LP):
    """The log's speed when it first passed each path station (smoothed over +-2.5 m); NaN outside the log's stretch."""
    v = np.r_[0.0, np.hypot(*np.diff(LP, axis=0).T) / np.maximum(np.diff(T) * 1e-6, 1e-3)]
    v = np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    sl = np.maximum.accumulate(proj_seq(P, S, LP))
    out = np.full(len(S), np.nan)
    m = (S >= sl[0]) & (S <= sl[-1])
    idx = np.minimum(np.searchsorted(sl, S[m]), len(sl) - 1)
    n = int(round(2.5 / STEP))
    if m.sum() > 2 * n:
        out[m] = np.convolve(np.pad(v[idx], n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), mode="valid")
    else:
        out[m] = v[idx]
    return out


def nudge(P, objs, hw):
    """Amendment 1: the path shifted sideways around parked boxes. A box is parked when its logged centre moves < STATIC_M m over its
    whole track; it intrudes when its corners, in the path frame, reach within hw + NUDGE_M of the path. Offset o = the smaller of the
    left / right shifts that clear it by NUDGE_M (none when both exceed NUDGE_MAX: the lane is blocked and the yield rule stops),
    held over [s_min - NUDGE_PRE, s_max + NUDGE_POST] with cosine ramps of NUDGE_RAMP m; the largest |o| wins per station.
    -> shifted path (resampled), list of (object id, s_min, offset)."""
    S = stations(P)
    d = np.gradient(P, axis=0)
    N = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
    off, used = np.zeros(len(S)), []
    for k, o in objs.items():
        X = np.asarray(o["P"], float)
        if not len(X) or np.hypot(*(X - X[0]).T).max() >= STATIC_M:
            continue
        c, h = X.mean(0), float(np.asarray(o["Y"], float).mean())
        u, v = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
        Q = np.array([c + a * u * o["L"] / 2 + b * v * o["W"] / 2 for a in (-1, 1) for b in (-1, 1)])
        if dist_to(P, c[None])[0] > 8.0:
            continue
        st = proj_seq(P, S, Q)
        i = np.clip(np.searchsorted(S, st), 0, len(S) - 1)
        lat = ((Q - P[i]) * N[i]).sum(1)
        lim = hw + NUDGE_M
        if lat.min() > lim or lat.max() < -lim:
            continue
        left, right = lat.max() + lim, lat.min() - lim
        o_ = left if abs(left) <= abs(right) else right
        if abs(o_) > NUDGE_MAX:
            continue
        a0, a1 = st.min() - NUDGE_PRE, st.max() + NUDGE_POST
        w = np.clip(np.minimum((S - (a0 - NUDGE_RAMP)) / NUDGE_RAMP, ((a1 + NUDGE_RAMP) - S) / NUDGE_RAMP), 0, 1)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        cand = w * o_
        off = np.where(np.abs(cand) > np.abs(off), cand, off)
        used.append([str(k), round(float(st.min()), 1), round(float(o_), 2)])
    if not used:
        return P, used
    return resample(dedup(P + off[:, None] * N, 0.05)), used


def dist_to(P, X):
    """Distance of every point X (m, 2) to the polyline P (n, 2)."""
    A, d = P[:-1], P[1:] - P[:-1]
    L2 = np.maximum((d * d).sum(1), 1e-12)
    out = np.empty(len(X))
    for i in range(0, len(X), 256):
        x = X[i:i + 256, None, :]
        u = np.clip(((x - A) * d).sum(-1) / L2, 0, 1)
        out[i:i + 256] = np.sqrt((((A + u[..., None] * d) - x) ** 2).sum(-1).min(1))
    return out


def chain_check(P, S, tn, Rx):
    """The route's own turns (same rule on the route polyline) against the chain: per route turn [entry, exit, deg, max distance of the
    route to the chain over [entry - 20, exit + 10] m, a chain turn of the same sign whose entry lies within 15 m of the route entry's
    projection]."""
    Rr = resample(dedup(Rx))
    Sr = stations(Rr)
    out = []
    for en, ex, deg in turns(Rr, Sr):
        m = (Sr >= en - 20) & (Sr <= ex + 10)
        dev = float(dist_to(P, Rr[m]).max()) if m.any() else np.nan
        q = np.array([np.interp(en, Sr, Rr[:, 0]), np.interp(en, Sr, Rr[:, 1])])
        sc = proj_seq(P, S, q[None])[0]
        ok = bool(any(np.sign(t[2]) == np.sign(deg) and abs(t[0] - sc) < 15.0 for t in tn))
        out.append([round(float(en), 1), round(float(ex), 1), round(float(deg), 1), round(dev, 2), ok, round(float(sc), 1)])
    return out


def route_along_log(ds, T, P3, Y):
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    g = RouteGenerator.create(P3, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
    for i in range(0, len(T), 5):
        g.generate_route(int(T[i]), Pose(P3[i], np.array([0.0, 0.0, np.sin(Y[i] / 2), np.cos(Y[i] / 2)])))
    return np.asarray(g.route_polyline_in_local.points, float)[:, :2]


def pack_job(job):
    scene, cfg, out = job
    import logging
    import shapely
    import rf_corridor as CO
    import rf_tax as TX
    logging.disable(logging.WARNING)
    ds = TX.loader(cfg).get_data_source(scene)
    sc = CO.lanes_of(ds)
    tr = ds.rig.trajectory
    T, P3, Y = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
    LP = P3[:, :2]
    vc = ds.rig.vehicle_config
    dims = np.array([vc.aabb_x_offset_m, vc.aabb_x_offset_m + vc.aabb_x_m, vc.aabb_y_m / 2], float)
    meta = dict(scene=scene)
    try:
        R = dedup(route_along_log(ds, T, P3, Y))
        meta["route_err"] = 0
    except Exception as e:  # noqa: BLE001
        R, meta["route_err"] = dedup(LP), repr(e)[:200]
    LS = stations(LP)
    ls0 = proj_seq(LP, LS, R[:1])[0]
    pre = LP[LS < ls0 - 0.5]
    Rx = dedup(np.r_[pre, R]) if len(pre) else R
    Sx = stations(Rx)
    M = CO.SceneMap(sc)
    ch = M.match(Rx, Sx, Rx)
    g, xy, nlc = assemble(ch["pieces"])
    X, on_route = merge_route(g, xy, Rx, Sx)
    meta.update(pieces=len(ch["pieces"]), amb=int(ch["n_amb"]), change=ch["trans"].count("change"), brk=ch["trans"].count("break"),
                blended=nlc, cov=float(ch["cov"].mean()) if len(ch["cov"]) else 0.0, fallback=g is None, on_route=round(on_route, 3))
    objs = TX.logged_objects(ds)
    P, nudges = nudge(extend(X), objs, float(dims[2]))
    S = stations(P)
    h = heading(P, S)
    tn = turns(P, S, h)
    vl = vlog_of(P, S, T, LP)
    keep = [k for k, o in objs.items() if len(o["T"]) and np.hypot(*(o["P"][:, None, :] - LP[None, ::10]).transpose(2, 0, 1)).min() < 150.0]
    off = np.r_[0, np.cumsum([len(objs[k]["T"]) for k in keep])].astype(np.int64)
    ot = np.concatenate([np.asarray(objs[k]["T"], float) for k in keep]) if keep else np.zeros(0)
    oxyh = np.concatenate([np.c_[objs[k]["P"], objs[k]["Y"]] for k in keep]) if keep else np.zeros((0, 3))
    np.savez_compressed(Path(out) / f"{scene}.npz", path=P.astype(np.float64), s=S, turns=tn, vlog=vl, dims=dims, log_t=T, log_xy=LP, log_yaw=Y,
                        obj_id=np.array(keep, str), obj_off=off, obj_t=ot, obj_xyh=oxyh,
                        obj_lw=np.array([[objs[k]["L"], objs[k]["W"]] for k in keep], float).reshape(-1, 2),
                        obj_static=np.array([objs[k]["static"] for k in keep], bool), route=Rx)
    near = lambda c: np.hypot(*(np.asarray(c, float)[:, None, :2] - P[None, ::10]).transpose(2, 0, 1)).min() < GEO_R  # noqa: E731
    geo = dict(lanes={k: dict(c=v["c"], l=v["l"], r=v["r"]) for k, v in sc["lanes"].items() if near(v["c"])},
               areas=[w for w in sc["areas"] if shapely.distance(shapely.from_wkb(w), shapely.Point(*P[len(P) // 2])) < 400],
               pieces=[dict(j=M.ids[p["j"]], s0=p["s0"], s1=p["s1"], tin=p["tin"], tout=p["tout"]) for p in ch["pieces"]])
    pickle.dump(geo, open(Path(out) / "geo" / f"{scene}.pkl", "wb"), protocol=4)
    meta.update(path_m=float(S[-1]), turns=np.round(tn, 1).tolist(), objects=len(keep), vlog_known=float(np.isfinite(vl).mean()),
                route_turns=chain_check(P, S, tn, Rx), nudges=nudges, log_s=[round(float(x), 1) for x in proj_seq(P, S, LP[[0, -1]])])
    return meta


def scenes_of(roots, only=None):
    jobs = {}
    for root in roots:
        sims = sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))) or [root]
        for sim in sims:
            f = Path(sim) / "aggregate/results-summary.json"
            if f.exists():
                for r in json.loads(f.read_text())["rollouts"]:
                    if only is None or r["clipgt_id"] in only:
                        jobs.setdefault(r["clipgt_id"], str(Path(sim) / "generated-user-config-0.yaml"))
    return jobs


def pack(a):
    from concurrent.futures import ProcessPoolExecutor
    only = set(Path(a.scenes).read_text().split()) if a.scenes else None
    jobs = scenes_of(a.runs, only)
    out = Path(a.out)
    (out / "geo").mkdir(parents=True, exist_ok=True)
    print(len(jobs), "scenes", flush=True)
    meta, err = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(pack_job, (s, c, str(out))): s for s, c in sorted(jobs.items())}
        for i, f in enumerate(futs):
            try:
                meta.append(f.result())
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:300]))
            if i % 50 == 0:
                print(i, "done", len(err), "errors", flush=True)
    idx = dict(scenes={m["scene"]: m for m in meta}, errors=err)
    (out / "index.json").write_text(json.dumps(idx, indent=1))
    fb = sum(m["fallback"] for m in meta)
    print(f"{len(meta)} packs, {len(err)} errors, {fb} fallbacks, {sum(m['brk'] > 0 for m in meta)} with breaks, "
          f"{sum(m['amb'] > 0 for m in meta)} with ambiguous samples -> {out}", flush=True)


# ---------------------------------------------------------------- figure
def drive_paths(spec):
    """LABEL=<root | run dir>,.. -> {scene: anchors (n, 3)} of each scene's last session."""
    lab, roots = spec.split("=", 1)
    out = {}
    for root in roots.split(","):
        fs = (glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")) + glob.glob(os.path.join(root, "driver-logs", "drive.jsonl"))
              + glob.glob(os.path.join(root, "driver", "drive.jsonl")))
        for f in fs:
            S = {}
            for line in open(f):
                if '"kind": "drive"' not in line or '"anchor"' not in line:
                    continue
                r = json.loads(line)
                S.setdefault((r["scene"], r["session"]), []).append((r["t0"], r["anchor"], (r.get("expert") or {}).get("hits")))
            for (sc, _), v in S.items():
                v.sort(key=lambda x: x[0])
                out[sc] = dict(t=np.array([x[0] for x in v], float), A=np.array([x[1] for x in v], float),
                               yield_=np.array([bool(x[2]) for x in v]))
    return lab, out


def zero_xy(root_spec, scene, D):
    """The first zero-score event position: the evaluator's failure time, else None (from results-summary + the driven anchors)."""
    for root in root_spec.split("=", 1)[1].split(","):
        for f in glob.glob(os.path.join(root, "runs", "*", "sim", "aggregate", "results-summary.json")) + glob.glob(os.path.join(root, "aggregate", "results-summary.json")):
            for r in json.loads(open(f).read())["rollouts"]:
                if r["clipgt_id"] == scene:
                    return r.get("score"), r.get("failure_reason")
    return None, None


def fig(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shapely
    runs = [drive_paths(s) for s in a.runs]
    scenes = a.scenes.split(",")
    n = len(scenes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    f, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 6.2 * rows), squeeze=False)
    colors = ["#e03131", "#f08c00", "#2f9e44"]
    for ax, sc in zip(axs.flat, scenes):
        z = np.load(Path(a.pack) / f"{sc}.npz")
        geo = pickle.load(open(Path(a.pack) / "geo" / f"{sc}.pkl", "rb"))
        P, S, tn, LP = z["path"], z["s"], z["turns"], z["log_xy"]
        for w in geo["areas"]:
            g = shapely.from_wkb(w)
            for p in getattr(g, "geoms", [g]):
                if hasattr(p, "exterior"):
                    ax.fill(*p.exterior.xy, color="#e9ecef", lw=0, zorder=0)
        for ln in geo["lanes"].values():
            for e in (ln["l"], ln["r"]):
                if e is not None:
                    ax.plot(e[:, 0], e[:, 1], color="#adb5bd", lw=0.5, zorder=1)
        ax.plot(LP[:, 0], LP[:, 1], color="#1c7ed6", lw=2.2, alpha=0.6, label="human log", zorder=2)
        ax.plot(P[:, 0], P[:, 1], "k--", lw=1.0, label="expert chain", zorder=3)
        for en, ex, _ in tn:
            for s_, mk in ((en, "|"), (ex, "x")):
                q = [np.interp(s_, S, P[:, 0]), np.interp(s_, S, P[:, 1])]
                ax.plot(*q, marker="o" if mk == "|" else "s", ms=5, mfc="none", mec="k", zorder=4)
        xs = [LP[:, 0]]
        ys = [LP[:, 1]]
        title = [sc[7:15] if sc.startswith("clipgt") else sc[-8:]]
        for (lab, D), col in zip(runs, colors):
            if sc not in D:
                continue
            A = D[sc]["A"]
            ax.plot(A[:, 0], A[:, 1], color=col, lw=1.6, label=lab, zorder=5)
            yv = D[sc]["yield_"]
            if yv.any():
                ax.plot(A[yv, 0], A[yv, 1], "o", color=col, ms=2.5, zorder=6)
            xs.append(A[:, 0])
            ys.append(A[:, 1])
            spec = next(s for s in a.runs if s.startswith(lab + "="))
            scv, fr = zero_xy(spec, sc, D)
            title.append(f"{lab} {0 if scv is None else scv:.2f}{' ' + str(fr)[:22] if fr else ''}")
            if fr:
                ax.plot(A[-1, 0], A[-1, 1], "X", color=col, ms=10, zorder=7)
        x, y = np.concatenate(xs), np.concatenate(ys)
        cx, cy, r = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max()), 0.5 * max(np.ptp(x), np.ptp(y)) + 15
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_aspect("equal")
        ax.set_title("\n".join(title), fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7, loc="best")
    f.tight_layout()
    f.savefig(a.out, dpi=130)
    print("->", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("pack")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--scenes", default="")
    s.add_argument("--workers", type=int, default=8)
    s = sp.add_parser("fig")
    s.add_argument("--pack", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--scenes", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pack": pack, "fig": fig}[a.cmd](a)


if __name__ == "__main__":
    main()
