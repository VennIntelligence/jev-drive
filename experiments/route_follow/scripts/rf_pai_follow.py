#!/usr/bin/env python3
"""PAI-FOLLOW (research/route-follow/pai-follow.md): turn completion with the real route and the map topology of every corridor zero on stored
PAI 281 reads (pai_eval.sh run roots). CPU only, run dirs / maps read, never written. Self-contained (copied to Tokyo as is).

  extract  (Tokyo, inside alpasim-base:0.89.0 with /mnt/nre-data and /repo/src mounted; rf_pai_follow.sh)
      rf_pai_follow.py extract --runs LABEL=<run root> ... --out <pkl> [--workers N] [--limit N]
      Per scene (map loaded once): every lane (centre x, y, heading; next / prev / adjacent ids), the logged rig path; per label rollout:
      summary row, the controller trace (10 Hz), the first zero-score event time on the evaluator's 2 Hz series, the rear-contact time,
      the decisions (t0, the logged route memory's +20 m flag), and AlpaSim's route (RouteGenerator MAP, 40 m: the log projected on its
      lanes, extended through every lane the rollout triggered at the ego's 0.5 s poses, rf_tax's way).
  report   (Mac, .venv)  rf_pai_follow.py report --pkl <pkl> --tax <pai281_rsg.csv> --s48 <file> --out DIR [--figs DIR]
      -> DIR/pai_follow.json, turns.md, zeros.md, turns.csv, zeros.csv; with --figs one BEV sheet of 6 RSG route zeros.

Definitions (rf_follow.py ported; PAI clips are 20 s, so the window is the log's whole travel instead of 18 decisions):
  turn       the route stretch [s0, s_log] (s0 = the ego's first position projected on the route, s_log = the log's last pose) holds a 30 m
             window with net heading change (chord over +-5 m) >= 60 deg and change over [a - 10, a + 40] m >= 45 deg; entry / exit at 10 / 90 %
             of that change; exit + 10 m must lie inside s_log (the human finished the turn in the clip). The first such turn per scene.
  completed  the ego path (controller, 10 Hz, projections within -30 / +60 m of the previous station) crosses exit + 10 m before the first
             zero-score event (at-fault front / lateral collision, offroad, left_corridor_laterally) and <= 2.5 m from the route there.
             else event (zero event before the crossing) | off (crossed > 2.5 m off, or last seen > 2.5 m off) | short (still on the route
             when the rollout ended; split by rear contact, which truncates PAI rollouts, vs plain slow).
  variants   label@bend: a bend (45 deg / 60 m) counts as a turn; label@open: the log may end inside the turn (entry + 5 m inside s_log,
             target min(exit + 10 m, s_log): PAI clips often stop right at the exit, which rf_follow's rule drops).
  cover      the logged memory's +20 m point is valid at the last decision whose station is <= entry - 20 m (decision 0 if it starts closer).
  junction   decisions before the event with station in [entry - 20, exit + 10]; per-decision pairs A:B on (scene, k) split by cover in both.
Zero topology (corridor zeros; the RSG taxonomy's 22 route zeros = cor_straight_at_turn + cor_turn_under):
  s_reach    the last station the ego held within 2.5 m of the route before the event (where it left the route).
  turn       the first route turn (any, whole route) whose exit + 10 m > s_reach and entry <= s_reach + 40 m; if no strict turn (60 deg /
             30 m) is near, a bend (45 deg / 60 m, same rule; z_def = bend); none -> beyond_log_end (s_reach >= log end - 2 m) or other.
  ego lane   the lane under the ego (centre <= 2.5 m, heading within 45 deg; nearest) when it reached entry - 10 m (else at s_reach).
  paths      every successor chain from the ego lane's point at entry - 10 m, out to (exit - entry) + 20 m (or its end when the map cuts it,
             if >= 25 m); net heading change of each.
  class      lane_change: the ego lane has no path that turns the route's way (>= 60 % of the turn, same sign),
             or the route itself changes lane (an adjacent, not successor, transition) within entry - 80 .. entry and the ego is not on it;
             straight_exit: some path ends within 30 deg of straight (a road to charge into); straight_road: the ego lane only turns but
             another lane runs straight ahead (lane centre <= 2.5 m and within 30 deg at >= 80 % of the points 15 .. 10 + min(turn length, 40) m
             along the ego's heading: the ego is in a turn-only lane of a road that goes on); no_straight_exit: several paths, all turn, no road
             ahead (T / forced); curve_single: one path only, it turns, no road ahead (a turn-only lane or a curved road);
             other: the rest.
  speeds     ego vs log speed when each first crosses route station ref - 40, ref - 20, ref (ref = entry; s_reach for no-turn zeros).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

WIN, TURN_DEG, LAT_OK, PAST, PRE, HALF = 30.0, 60.0, 2.5, 10.0, 20.0, 5.0
STRAIGHT, LANE_D, LANE_PSI, NEAR = 30.0, 2.5, 45.0, 40.0
MIN_CHAIN = 25.0                                    # m: shortest successor chain (map extent cuts some) whose end heading counts
BEND_WIN, BEND_DEG = 60.0, 45.0                     # zero topology only: a gradual bend (>= 45 deg within 60 m) when no strict turn is near
I20 = 13                                            # rmem.OFFS == 20 m (N_PT 40, NF 5)
ZERO = ("collision_front", "collision_lateral", "offroad", "left_corridor_laterally")
N_BOOT = 10_000


# ---------------------------------------------------------------- extract (container)
def yawq(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def ctrl(sim, rid):
    import csv
    f = Path(sim) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists():
        return None
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    if not len(A):
        return None
    c = lambda k: A[:, h[k]]  # noqa: E731
    return dict(t=c("timestamp_us"), x=c("x"), y=c("y"), yaw=np.unwrap(yawq(c("qx"), c("qy"), c("qz"), c("qw"))), vx=c("vx"))


def events(sim, scene, rid):
    """(first zero-event time us, its flag, first rear-contact time) from the evaluator's series."""
    import polars as pl
    f = glob.glob(f"{sim}/rollouts/{scene}/{rid}/metrics.parquet")
    if not f:
        return None, None, None
    m = pl.read_parquet(f[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    first = {}
    for (n,), x in m.group_by("name"):
        if n in ZERO + ("collision_rear",):
            t, v = np.array(x["timestamps_us"].to_list(), float), np.array([np.nan if q is None else q for q in x["values"].to_list()], float)
            hit = t[np.nan_to_num(v) > 0]
            if len(hit):
                first[n] = float(hit.min())
    z = {k: v for k, v in first.items() if k in ZERO}
    k = min(z, key=z.get) if z else None
    return (z[k] if k else None), k, first.get("collision_rear")


def decisions(root):
    """{session: [(t0, k, +20 m memory flag or None)]} from every chunk's driver log."""
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")):
        for line in open(f):
            if '"infer": true' not in line:
                continue
            r = json.loads(line)
            rm = r.get("rmem")
            out.setdefault(r["session"], []).append((int(r["t0"]), int(r["k"]), None if rm is None else bool(rm[I20 * 5] > 0.5)))
    return {k: sorted(v) for k, v in out.items()}


def scene_job(job):
    scene, cfg, rolls = job
    from alpasim_runtime.config import RouteGeneratorType, UserSimulatorConfig
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_runtime.scene_loader import build_scene_loader
    from alpasim_utils.geometry import Pose
    from omegaconf import OmegaConf
    import logging
    logging.disable(logging.WARNING)
    ds = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(cfg))).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
    lanes = {str(l.id): dict(c=np.asarray(l.center.points, float)[:, [0, 1, 3]].astype(np.float32), nx=sorted(map(str, l.next_lanes)),
                             pv=sorted(map(str, l.prev_lanes)), al=sorted(map(str, l.adj_lanes_left)), ar=sorted(map(str, l.adj_lanes_right)))
             for l in ds.map.lanes}
    out = dict(scene=scene, lanes=lanes, log=dict(T=Tg, P=Pg[:, :2]), rolls={})
    zt = lambda t: float(np.interp(t, Tg, Pg[:, 2]))  # noqa: E731
    for r in rolls:
        o = dict(score=r["score"], metrics=r["metrics"], rid=r["rid"], dec=r["dec"])
        C = ctrl(r["sim"], r["rid"])
        o["tz"], o["zflag"], o["trear"] = events(r["sim"], scene, r["rid"])
        if C is not None:
            g = RouteGenerator.create(Pg, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
            o["route0"] = np.asarray(g.route_polyline_in_local.points, float)[:, :2]
            try:
                for i in range(min(4, len(C["t"]) - 1), len(C["t"]), 5):
                    yw = C["yaw"][i]
                    g.generate_route(int(C["t"][i]), Pose(np.array([C["x"][i], C["y"][i], zt(C["t"][i])]), np.array([0.0, 0.0, np.sin(yw / 2), np.cos(yw / 2)])))
                o["route"], o["route_err"] = np.asarray(g.route_polyline_in_local.points, float)[:, :2], 0
            except Exception:  # noqa: BLE001
                o["route"], o["route_err"] = o["route0"], 1
            o["ctrl"] = C
        out["rolls"][r["label"]] = o
    return out


def extract(a):
    from concurrent.futures import ProcessPoolExecutor
    jobs = {}
    for spec in a.runs:
        label, root = spec.split("=", 1)
        dec = decisions(root)
        for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
            f = Path(sim) / "aggregate/results-summary.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["rollouts"]:
                j = jobs.setdefault(r["clipgt_id"], [str(Path(sim) / "generated-user-config-0.yaml"), {}])
                j[1].setdefault(label, dict(label=label, sim=sim, rid=r["rollout_id"], score=r.get("score"), metrics=r.get("metrics") or {},
                                            dec=dec.get(r["rollout_id"], [])))
    units = [(s, c, list(v.values())) for s, (c, v) in sorted(jobs.items())][: a.limit or None]
    print(len(units), "scenes", flush=True)
    res, err = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(scene_job, u): u[0] for u in units}
        for i, f in enumerate(futs):
            try:
                res.append(f.result())
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:300]))
            if i % 50 == 0:
                print(i, "done", len(err), "errors", flush=True)
    pickle.dump(dict(scenes=res, errors=err), open(a.out, "wb"), protocol=4)
    print(len(res), "scenes,", len(err), "errors ->", a.out, flush=True)


# ---------------------------------------------------------------- geometry (report)
def stations(R):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]


def dedup(R):
    return R[np.r_[True, np.hypot(*np.diff(R, axis=0).T) > 1e-6]]


def project(R, S, p, lo=-np.inf, hi=np.inf):
    """Point p -> (station, signed lateral, left +) on polyline R within stations [lo, hi]."""
    a, b = R[:-1], R[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    u = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
    q = a + u[:, None] * d
    s = S[:-1] + u * np.sqrt(L2)
    dist = np.hypot(*(p - q).T)
    ok = (s >= lo) & (s <= hi)
    if not ok.any():
        ok[:] = True
    i = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
    cr = d[i, 0] * (p[1] - a[i, 1]) - d[i, 1] * (p[0] - a[i, 0])
    return float(s[i]), float(np.sign(cr) * dist[i])


def heading(R, S, step=1.0):
    s = np.arange(0.0, S[-1] + 1e-9, step)
    at = lambda q: np.c_[np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])]  # noqa: E731
    d = at(np.minimum(s + HALF, S[-1])) - at(np.maximum(s - HALF, 0.0))
    return s, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def find_turn(s, h, lo, hi, win=WIN, deg=TURN_DEG):
    """rf_follow.find_turn: first `win` m window in [lo, hi] with net heading change >= `deg` -> (entry, exit, TH rad) or None
    (defaults = rf_follow's 30 m / 60 deg; the change over [a - 10, a + win + 10] m must be >= 75 % of `deg`)."""
    i0, i1 = np.searchsorted(s, lo), np.searchsorted(s, hi, side="right")
    w = int(round(win / (s[1] - s[0]))) if len(s) > 1 else 0
    for a in range(i0, max(i0, i1 - w)):
        if np.degrees(abs(h[a + w] - h[a])) >= deg:
            j0, j1 = max(0, a - 10), min(len(s) - 1, a + w + 10)
            th = h[j1] - h[j0]
            if np.degrees(abs(th)) < 0.75 * deg:
                continue
            f = (h[j0:j1 + 1] - h[j0]) / th
            k10, k90 = np.flatnonzero(f >= 0.1), np.flatnonzero(f >= 0.9)
            return float(s[j0 + k10[0]]), float(s[j0 + k90[0]]), float(th)
    return None


def all_turns(s, h, lo, hi, **kw):
    out, x = [], lo
    while (t := find_turn(s, h, x, hi, **kw)) is not None:
        out.append(t)
        x = t[1] + 1.0
    return out


def track(R, S, P):
    """Path P (n, 2) -> stations, laterals with the -30 / +60 m local window."""
    st, la, hint = np.empty(len(P)), np.empty(len(P)), None
    for i, p in enumerate(P):
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        st[i], la[i] = project(R, S, p, lo, hi)
        hint = st[i]
    return st, la


def lane_at(lanes, p, psi, prefer=()):
    """Lane under point p with heading psi: centre <= LANE_D m, heading within LANE_PSI deg; a preferred id wins a near tie."""
    best = []
    for k, l in lanes.items():
        c = l["c"]
        if len(c) < 2:
            continue
        s = stations(c[:, :2].astype(float))
        st, lat = project(c[:, :2].astype(float), s, p)
        if abs(lat) > LANE_D:
            continue
        i = min(np.searchsorted(s, st), len(c) - 1)
        dpsi = np.degrees(abs((psi - c[i, 2] + np.pi) % (2 * np.pi) - np.pi))
        if dpsi <= LANE_PSI:
            best.append((abs(lat) - (0.5 if k in prefer else 0.0), k))
    return min(best)[1] if best else None


def chains(lanes, lid, p, length, cap=64):
    """Every successor chain from lane lid's point nearest p, cut at `length` m -> [(polyline, lane ids, reached full length)]."""
    c = lanes[lid]["c"][:, :2].astype(float)
    s = stations(c)
    st, _ = project(c, s, p)
    k = int(np.searchsorted(s, st))
    start = np.r_[[[np.interp(st, s, c[:, 0]), np.interp(st, s, c[:, 1])]], c[k:]]
    out, todo = [], [(start, [lid])]
    while todo and len(out) < cap:
        P, ids = todo.pop()
        P = dedup(P)
        L = stations(P)[-1] if len(P) > 1 else 0.0
        nx = [n for n in lanes[ids[-1]]["nx"] if n in lanes and n not in ids]
        if L >= length or not nx:
            out.append((P, ids, L >= length))
            continue
        for n in nx:
            todo.append((np.r_[P, lanes[n]["c"][:, :2].astype(float)], ids + [n]))
    return out


def road_ahead(lanes, p, psi, far, near=15.0, step=2.5, share=0.8):
    """Does some lane (any, not only the ego lane's successors) run straight ahead of p along heading psi: at >= 80 % of the points
    near .. far m ahead a lane centre within LANE_D m whose heading is within STRAIGHT deg of psi."""
    Q = []
    for l in lanes.values():
        c = l["c"].astype(float)
        if len(c) < 2:
            continue
        S = stations(c[:, :2])
        q = np.arange(0.0, S[-1], 1.0)
        hh = np.unwrap(c[:, 2])
        Q.append(np.c_[np.interp(q, S, c[:, 0]), np.interp(q, S, c[:, 1]), np.interp(q, S, hh)])
    if not Q:
        return False
    Q = np.concatenate(Q)
    u = np.array([np.cos(psi), np.sin(psi)])
    d = np.arange(near, far + 1e-9, step)
    hit = 0
    for x in d:
        g = p + x * u
        m = np.hypot(Q[:, 0] - g[0], Q[:, 1] - g[1]) <= LANE_D
        dpsi = np.degrees(np.abs((Q[m, 2] - psi + np.pi) % (2 * np.pi) - np.pi))
        hit += bool((dpsi <= STRAIGHT).any())
    return bool(len(d) and hit >= share * len(d))


def net_turn(P, d0, d1):
    """Net heading change (deg, left +) of polyline P between arc length d0 and d1 (chords over +-HALF m)."""
    S = stations(P)
    s, h = heading(P, S)
    return float(np.degrees(np.interp(min(d1, S[-1] - HALF), s, h) - np.interp(min(d0, S[-1]), s, h)))


def route_lanes(lanes, R, S, step=2.0):
    """Lane id at every `step` m of the route (continuity preferred: the previous lane, its successors and neighbours)."""
    s, h = heading(R, S, step)
    ids, prev = [], ()
    for q, psi in zip(s, h):
        p = np.array([np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])])
        lid = lane_at(lanes, p, psi, prefer=prev)
        ids.append(lid)
        if lid is not None:
            prev = (lid, *lanes[lid]["nx"])
    return s, ids


def speed_at(t, st, v, target, t_max=np.inf):
    """Speed when the station track first crosses `target` (before t_max) or nan."""
    i = np.flatnonzero((st >= target) & (t <= t_max))
    if not len(i):
        return np.nan
    i = int(i[0])
    if i == 0:
        return float(v[0])
    w = (target - st[i - 1]) / max(st[i] - st[i - 1], 1e-9)
    return float((1 - w) * v[i - 1] + w * v[i])


# ---------------------------------------------------------------- report
def kind_of(o):
    m = o["metrics"]
    if o["score"]:
        return ""
    return "collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else "corridor" if m.get("left_corridor_laterally") else "other"


def one(sc, label, o, tdef=None, zeros=True):
    """(scene record, label, rollout) -> row dict, per-decision rows, drawing data. tdef: find_turn kwargs (None = rf_follow's 30 m / 60 deg)."""
    tdef = dict(tdef or {})
    open_end = tdef.pop("open_end", False)              # the log ends inside the turn: entry + 5 m inside s_log, target min(exit + 10, s_log)
    inside = (lambda x: x[0] + 5.0 <= s_log) if open_end else (lambda x: x[1] + PAST <= s_log)  # noqa: E731
    lanes, LG = sc["lanes"], sc["log"]
    C = o["ctrl"]
    R = dedup(o["route"])
    S = stations(R)
    P = np.c_[C["x"], C["y"]]
    t = C["t"]
    tz = o["tz"] if o["tz"] is not None and not o["score"] else np.inf          # a flag that did not zero the score ends nothing
    st, lat = track(R, S, P)
    lt, _ = track(R, S, LG["P"])
    vlog = np.r_[0.0, np.hypot(*np.diff(LG["P"], axis=0).T) / np.maximum(np.diff(LG["T"]) / 1e6, 1e-3)]
    vlog = np.convolve(np.r_[vlog[:2], vlog, vlog[-2:]], np.ones(5) / 5, "valid")
    s0, s_log = st[0], float(lt.max())
    s, h = heading(R, S)
    row = dict(label=label, scene=sc["scene"], score=float(o["score"] or 0), kind=kind_of(o), zflag=o["zflag"] or "", rear=o["trear"] is not None,
               route_err=o["route_err"], s0=s0, s_log=s_log, route_m=float(S[-1]), turn=False)
    dec = []
    pre = t <= tz
    for (t0, k, c20) in o["dec"]:
        i = int(np.clip(np.searchsorted(t, t0), 0, len(t) - 1))
        dec.append(dict(label=label, scene=sc["scene"], k=k, t0=t0, st=float(st[i]), lat=float(lat[i]), c20=c20, pre=bool(t0 <= tz), junction=False))
    tt = next((x for x in all_turns(s, h, s0, s_log + (WIN if open_end else 0.0), **tdef) if inside(x)), None) if open_end else find_turn(s, h, s0, s_log, **tdef)
    if tt is not None and inside(tt):
        ent, ext, th = tt
        tgt = min(ext + PAST, s_log) if open_end else ext + PAST
        i = np.flatnonzero(st >= tgt)
        reached, lat_x, t_x = False, np.nan, None
        if len(i):
            i = int(i[0])
            w = 0.0 if i == 0 else (tgt - st[i - 1]) / max(st[i] - st[i - 1], 1e-9)
            lat_x = abs(lat[i] if i == 0 else (1 - w) * lat[i - 1] + w * lat[i])
            t_x = t[i]
            reached = t_x < tz
        ok = bool(reached and lat_x <= LAT_OK)
        last = np.flatnonzero(pre)
        lat_last = abs(lat[int(last[-1])]) if len(last) else np.nan
        ev = np.isfinite(tz) and (t_x is None or tz <= t_x)
        outcome = ("completed" if ok else "event" if ev else "off" if t_x is not None or lat_last > LAT_OK else "short_rear" if o["trear"] else "short_slow")
        dd = [d for d in dec if d["st"] <= ent - PRE]
        cov = (dd[-1] if dd else dec[0])["c20"] if dec else None
        for d in dec:
            d["junction"] = bool(ent - PRE <= d["st"] <= tgt and d["pre"])
        row.update(turn=True, side="L" if th > 0 else "R", turn_deg=float(np.degrees(th)), entry=ent, exit=ext, completed=ok, outcome=outcome,
                   lat_cross=lat_x, cover=cov, v_entry=speed_at(t, st, C["vx"], ent, tz), v_log_entry=speed_at(LG["T"], lt, vlog, ent))
    row["n_turns_route"] = len([x for x in all_turns(s, h, s0, s_log + (WIN if open_end else 0.0), **tdef) if inside(x)])
    draw = None
    if row["kind"] == "corridor" and zeros:
        z = topo(sc, o, R, S, s, h, P, t, st, lat, lt, vlog, tz, s_log)
        row.update(z)
        draw = dict(R=R, P=P[pre], L=LG["P"], lanes=lanes, z=z, pz=P[np.flatnonzero(pre)[-1]] if pre.any() else P[-1])
    return row, dec, draw


def topo(sc, o, R, S, s, h, P, t, st, lat, lt, vlog, tz, s_log):
    lanes = sc["lanes"]
    C = o["ctrl"]
    on = np.flatnonzero((t <= tz) & (np.abs(lat) <= LAT_OK))
    s_reach = float(st[on].max()) if len(on) else float(st[0])
    z = dict(s_reach=s_reach, beyond_log=bool(s_reach >= s_log - 2.0), z_def="")
    for nm, kw in (("strict", {}), ("bend", dict(win=BEND_WIN, deg=BEND_DEG))):
        turns = [x for x in all_turns(s, h, 0.0, S[-1], **kw) if x[1] + PAST > s_reach and x[0] <= s_reach + NEAR]
        if turns:
            z["z_def"] = nm
            break
    if turns:
        ent, ext, th = turns[0]
        ref = ent
        z.update(z_turn=True, z_entry=ent, z_exit=ext, z_turn_deg=float(np.degrees(th)))
    else:
        ref = s_reach
        z.update(z_turn=False)
    for d in (40, 20, 0):
        z[f"v_ego_{d}"] = speed_at(t, st, C["vx"], ref - d, tz)
        z[f"v_log_{d}"] = speed_at(sc["log"]["T"], lt, vlog, ref - d)
    if not turns:
        z["topo"] = "beyond_log_end" if z["beyond_log"] else "other_no_turn"
        return z
    a = ent - 10.0
    i = np.flatnonzero(st >= a)
    i = int(i[0]) if len(i) and t[int(i[0])] <= tz else int(np.flatnonzero(t <= tz)[-1]) if (t <= tz).any() else 0
    sr, ids = route_lanes(lanes, R, S)
    ra = ids[int(np.clip(np.searchsorted(sr, a), 0, len(ids) - 1))]
    p_r = np.array([np.interp(a, S, R[:, 0]), np.interp(a, S, R[:, 1])])
    ego = lane_at(lanes, P[i], C["yaw"][i], prefer=(ra,) if ra else ())
    lc_route = False
    for j in range(1, len(ids)):
        if a - 70.0 <= sr[j] <= ent and ids[j] and ids[j - 1] and ids[j] != ids[j - 1] and ids[j] not in lanes[ids[j - 1]]["nx"] and \
                ids[j] in lanes[ids[j - 1]]["al"] + lanes[ids[j - 1]]["ar"]:
            lc_route = True
    length = (ext - ent) + 20.0

    def paths(lid, p):
        if lid is None:
            return []
        out_ = []
        for Q, ids_, _ in chains(lanes, lid, p, 10.0 + length + HALF):         # a chain cut by the map's extent counts once it runs >= MIN_CHAIN
            L = stations(Q)[-1]
            out_.append((net_turn(Q, HALF, 10.0 + length), bool(L >= min(10.0 + length + HALF, MIN_CHAIN)), ids_))
        return out_
    pe, pr = paths(ego, P[i]), paths(ra, p_r)
    TH = np.degrees(th)
    turns_way = lambda pp: any(full and np.sign(d) == np.sign(TH) and abs(d) >= 0.6 * abs(TH) for d, full, _ in pp)  # noqa: E731
    straight = lambda pp: any(full and abs(d) < STRAIGHT for d, full, _ in pp)  # noqa: E731
    z.update(ego_lane=ego, route_lane=ra, ego_on_route_lane=bool(ego is not None and ego == ra), route_lc=lc_route,
             ego_paths=[round(d, 1) for d, full, _ in pe if full], route_paths=[round(d, 1) for d, full, _ in pr if full],
             ego_straight=straight(pe), route_straight=straight(pr), ego_reaches_turn=turns_way(pe), ego_xy=P[i].tolist(),
             ego_chains=[(ids_, round(d, 1), full) for d, full, ids_ in pe])
    psi = C["yaw"][i]
    z["road_ahead"] = road_ahead(lanes, P[i], psi, 10.0 + min(length, 40.0))
    z["probe"] = [P[i].tolist(), (P[i] + (10.0 + min(length, 40.0)) * np.array([np.cos(psi), np.sin(psi)])).tolist()]
    if ego is None:
        z["topo"] = "other_no_lane"
    elif not z["ego_reaches_turn"] or (lc_route and ego != ra):
        z["topo"] = "lane_change"
    elif z["ego_straight"]:
        z["topo"] = "straight_exit"
    elif z["road_ahead"]:
        z["topo"] = "straight_road"
    elif z["ego_reaches_turn"]:
        z["topo"] = "no_straight_exit" if len(set(tuple(c[0]) for c in z["ego_chains"] if c[2])) > 1 else "curve_single"
    else:
        z["topo"] = "other"
    return z


def boot(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float)
    if not len(x):
        return dict(n=0, rate=np.nan, lo=np.nan, hi=np.nan)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return dict(n=int(len(x)), rate=float(x.mean()), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)))


def dec_pair(Dx, A, B):
    """Junction-stretch decisions of turn scenes paired on (scene, k), split by the +20 m memory flag in both; scene-clustered bootstrap."""
    cols = ["scene", "k", "lat", "c20"]
    m = Dx[(Dx.label == A) & Dx.junction][cols].merge(Dx[(Dx.label == B) & Dx.junction][cols], on=["scene", "k"], suffixes=("_a", "_b"))
    m = m.dropna(subset=["c20_a", "c20_b"])
    ca, cb = m.c20_a.astype(bool), m.c20_b.astype(bool)
    out = dict(pairs=int(len(m)), discordant=int((ca != cb).sum()))
    rng = np.random.default_rng(0)
    for half, sel in (("covered", ca & cb), ("uncovered", ~ca & ~cb)):
        x = m[sel.to_numpy()]
        if not len(x):
            out[half] = dict(n=0)
            continue
        la, lb = x.lat_a.abs().to_numpy(), x.lat_b.abs().to_numpy()
        L, inv = np.unique(x.scene.to_numpy(), return_inverse=True)
        idx = [np.flatnonzero(inv == j) for j in range(len(L))]
        bd = np.empty(N_BOOT)
        for b in range(N_BOOT):
            r = np.concatenate([idx[j] for j in rng.integers(0, len(L), len(L))])
            bd[b] = np.median(la[r]) - np.median(lb[r])
        out[half] = dict(n=int(len(x)), scenes=int(len(L)), med_a=float(np.median(la)), med_b=float(np.median(lb)),
                         diff=float(np.median(la) - np.median(lb)), lo=float(np.quantile(bd, 0.025)), hi=float(np.quantile(bd, 0.975)))
    return out


def report(a):
    import pandas as pd
    D = pickle.load(open(a.pkl, "rb"))
    tax = pd.read_csv(a.tax).set_index("scene")
    s48 = {x for x in Path(a.s48).read_text().split() if x.startswith("clipgt-")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, decs, draws, err = [], [], {}, list(D["errors"])
    for sc in D["scenes"]:
        for label, o in sc["rolls"].items():
            if o.get("ctrl") is None:
                err.append((sc["scene"], label, "no controller trace"))
                continue
            try:
                r, d, dr = one(sc, label, o)
                rows.append(one(sc, label + "@bend", o, dict(win=BEND_WIN, deg=BEND_DEG), zeros=False)[0])
                rows.append(one(sc, label + "@open", o, dict(open_end=True), zeros=False)[0])
            except Exception as e:  # noqa: BLE001
                err.append((sc["scene"], label, repr(e)[:200]))
                continue
            rows.append(r)
            decs += d
            if dr is not None:
                draws[(label, sc["scene"])] = dr
    X, Dx = pd.DataFrame(rows), pd.DataFrame(decs)
    for c in ("side", "turn_deg", "entry", "exit", "completed", "outcome", "lat_cross", "cover", "v_entry", "v_log_entry", "topo", "z_turn", "z_turn_deg",
              "ego_paths", "ego_on_route_lane", "route_lc", *(f"v_{w}_{d}" for w in ("ego", "log") for d in (40, 20, 0))):
        if c not in X:
            X[c] = None
    X["short"] = X.scene.str[7:15]
    X["in48"] = X.scene.isin(s48)
    X["tax"] = X.scene.map(tax.primary).where(X.label == "rsg")
    X["route_zero"] = X.tax.isin(["cor_straight_at_turn", "cor_turn_under"])
    labels = [x for x in dict.fromkeys(X.label) if "@" not in x]
    tlabels = list(dict.fromkeys(X.label))
    J = dict(errors=err, labels={}, pairs={})
    for lab in tlabels:
        x = X[X.label == lab]
        T = x[x.turn]
        oc = {k: int((T.outcome == k).sum()) for k in ("completed", "event", "off", "short_slow", "short_rear")}
        ev = T[T.outcome == "event"].kind.value_counts().to_dict()
        tl = set(T.short)
        r = dict(rollouts=int(len(x)), mean=float(x.score.mean()), zeros=int((x.score == 0).sum()), turns=int(len(T)),
                 left=int((T.side == "L").sum()), right=int((T.side == "R").sum()), outcomes=oc, event_kinds=ev,
                 all=boot(T.completed.astype(float)), left_rate=boot(T[T.side == "L"].completed.astype(float)),
                 right_rate=boot(T[T.side == "R"].completed.astype(float)),
                 covered=boot(T[T.cover == True].completed.astype(float)), uncovered=boot(T[T.cover == False].completed.astype(float)),  # noqa: E712
                 turn_mean_score=float(T.score.mean()), multi_turn=int((x.n_turns_route >= 2).sum()),
                 overlap48=int(len(tl & {s[7:15] for s in s48})), only_here=sorted(tl - {s[7:15] for s in s48}),
                 only48=sorted({s[7:15] for s in s48} - tl), v_entry_ratio_median=float((T.v_entry / T.v_log_entry).median()))
        J["labels"][lab] = r
        if "@" in lab:
            continue
        Z = x[x.kind == "corridor"]
        r["corridor_zeros"] = int(len(Z))
        r["topo_all"] = Z.topo.value_counts().to_dict()
        r["topo_turn"] = Z[Z.z_turn == True].topo.value_counts().to_dict()  # noqa: E712
        if lab == "rsg":
            r["topo_route22"] = Z[Z.route_zero].topo.value_counts().to_dict()
            r["topo_by_tax"] = {k: v.topo.value_counts().to_dict() for k, v in Z.groupby("tax")}
        for g, ZZ in (("turn", Z[Z.z_turn == True]),) + ((("route22", Z[Z.route_zero]),) if lab == "rsg" else ()):  # noqa: E712
            r[f"speed_{g}"] = {f"{d}": dict(ego=float(ZZ[f"v_ego_{d}"].median()), log=float(ZZ[f"v_log_{d}"].median()),
                                             ratio=float((ZZ[f"v_ego_{d}"] / ZZ[f"v_log_{d}"].clip(lower=0.5)).median()),
                                             faster=int((ZZ[f"v_ego_{d}"] > ZZ[f"v_log_{d}"] + 1.0).sum()), n=int(ZZ[f"v_ego_{d}"].notna().sum()))
                               for d in (40, 20, 0)}
            pk = lambda w: np.fmax(ZZ[f"v_{w}_40"], ZZ[f"v_{w}_20"])  # noqa: E731
            ok = ZZ[[f"v_{w}_{d}" for w in ("ego", "log") for d in (40, 20, 0)]].notna().all(axis=1)
            r[f"speed_{g}"]["slowed"] = dict(n=int(ok.sum()), ego=int((ZZ.v_ego_0 < pk("ego") - 1.0)[ok].sum()), log=int((ZZ.v_log_0 < pk("log") - 1.0)[ok].sum()),
                                             ego_gain=float((ZZ.v_ego_0 - pk("ego"))[ok].median()), log_gain=float((ZZ.v_log_0 - pk("log"))[ok].median()))
        J["labels"][lab] = r
    if len(labels) == 2:
        A, B = labels[1], labels[0]
        J["pairs"][f"{A}:{B}"] = dec_pair(Dx, A, B)
    X.drop(columns=["ego_chains"], errors="ignore").to_csv(out / "scenes.csv", index=False)
    (out / "pai_follow.json").write_text(json.dumps(J, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    f = lambda r: "-" if not r or not r.get("n") else f"{100 * r['rate']:.1f} % ({r['n']}) [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"  # noqa: E731
    md = ["# PAI turn completion with the real route (rf_pai_follow.py)", "",
          f"Turn = AlpaSim route (MAP, the log on its lanes + extensions) net heading change >= {TURN_DEG:.0f} deg within {WIN:.0f} m, inside the log's travel, "
          f"exit + {PAST:.0f} m inside it; completed = the ego crosses exit + {PAST:.0f} m before any zero-score event, <= {LAT_OK} m from the route. "
          "event = zero event first; off = crossed > 2.5 m off / last seen > 2.5 m off; short = still on the route at the end (rear = truncated by a rear contact). "
          "Rates with scene-bootstrap 95 % CI; cover = logged memory +20 m valid at the decision 20 m before the entry.", "",
          "| label | rollouts / turn scenes (L / R) | completion all | left | right | covered | uncovered | completed / event / off / short slow / short rear | event kinds | ego / log speed at entry (median ratio) | in the 48 list |",
          "|:--|--:|:--|:--|:--|:--|:--|:--|:--|--:|--:|"]
    for lab, r in J["labels"].items():
        md.append(f"| {lab} | {r['rollouts']} / {r['turns']} ({r['left']} / {r['right']}) | {f(r['all'])} | {f(r['left_rate'])} | {f(r['right_rate'])} | {f(r['covered'])} | "
                  f"{f(r['uncovered'])} | {' / '.join(str(v) for v in r['outcomes'].values())} | {', '.join(f'{k} {v}' for k, v in r['event_kinds'].items())} | "
                  f"{r['v_entry_ratio_median']:.2f} | {r['overlap48']} of 48 |")
    for p, v in J["pairs"].items():
        md += ["", f"Per-decision junction lateral, pair {p} (|lat| median A / B, A - B [scene-clustered 95 % CI]); discordant {v['discordant']} of {v['pairs']} left out:", "",
               "| half | decisions (scenes) | median A / B (m) | A - B |", "|:--|--:|:--|:--|"]
        for hh in ("covered", "uncovered"):
            q = v[hh]
            md.append(f"| {hh} | {q['n']} ({q.get('scenes', 0)}) | " + (f"{q['med_a']:.2f} / {q['med_b']:.2f} | {q['diff']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] |" if q["n"] else "- | - |"))
    md += ["", "Per scene (turn scenes, both labels):", "", "| scene | side / deg | in 48 | " + " | ".join(f"{lab} outcome (score)" for lab in labels) + " |",
           "|:--|:--|:--|" + "--|" * len(labels)]
    TT = X[X.turn.astype(bool) & X.label.isin(labels)]
    for sc_, g in TT.groupby("scene"):
        r0 = g.iloc[0]
        md.append(f"| {r0.short} | {r0.side} {r0.turn_deg:+.0f} | {'y' if r0.in48 else ''} | " +
                  " | ".join((f"{q.outcome} ({q.score:.2f})" if len(q := g[g.label == lab]) and (q := q.iloc[0]) is not None else "-") for lab in labels) + " |")
    (out / "turns.md").write_text("\n".join(md) + "\n")
    names = dict(straight_exit="straight exit drivable: the ego lane's own successor", straight_road="straight exit drivable: another lane runs straight ahead (ego lane turn-only)",
                 no_straight_exit="T / no straight exit (branches, all turn)",
                 curve_single="no straight exit: turn-only lane, one chain, no straight lane ahead", lane_change="lane change needed",
                 beyond_log_end="no route turn: beyond the log's end", other_no_turn="no route turn: other", other="other (turn)", other_no_lane="other (no lane)")
    mz = ["# Corridor zeros by map topology at the failure (rf_pai_follow.py)", "",
          "Class rules in the script docstring. Route zeros = the RSG taxonomy's cor_straight_at_turn + cor_turn_under (22).", "",
          "| class | " + " | ".join(f"{lab} all corridor ({J['labels'][lab]['corridor_zeros']})" for lab in labels) + " | rsg route zeros (22) |", "|:--|" + "--:|" * (len(labels) + 1)]
    for k, nm in names.items():
        mz.append(f"| {nm} | " + " | ".join(str(J["labels"][lab]["topo_all"].get(k, 0)) for lab in labels) + f" | {J['labels'].get('rsg', {}).get('topo_route22', {}).get(k, 0)} |")
    for lab in labels:
        for g in ("route22", "turn"):
            if f"speed_{g}" not in J["labels"][lab]:
                continue
            sp = J["labels"][lab][f"speed_{g}"]
            mz += ["", f"{lab}, {g} zeros: speed at the junction (median m/s, ego / log, median ratio, ego faster than log by > 1 m/s):", "",
                   "| station | ego | log | ratio | faster / n |", "|:--|--:|--:|--:|--:|"]
            for d in ("40", "20", "0"):
                q = sp[d]
                mz.append(f"| entry - {d} m | {q['ego']:.1f} | {q['log']:.1f} | {q['ratio']:.2f} | {q['faster']} / {q['n']} |")
            q = sp["slowed"]
            mz.append(f"| slowed into the entry (v at 0 < max(v at -40, -20) - 1 m/s; median change) | {q['ego']} / {q['n']} ({q['ego_gain']:+.1f}) | "
                      f"{q['log']} / {q['n']} ({q['log_gain']:+.1f}) | | |")
    for lab in labels:
        Z = X[(X.label == lab) & (X.kind == "corridor")].sort_values(["route_zero", "topo"], ascending=[False, True])
        mz += ["", f"## {lab}: {len(Z)} corridor zeros", "",
               "| scene | taxonomy | class | turn deg | ego on route lane | ego paths (deg) | route LC | v ego / log at -40 | -20 | 0 m |",
               "|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|"]
        for _, r in Z.iterrows():
            v = lambda d: f"{r[f'v_ego_{d}']:.1f} / {r[f'v_log_{d}']:.1f}"  # noqa: E731
            td = f"{r.z_turn_deg:+.0f}" if r.get("z_turn") is True else "-"
            ep = r.get("ego_paths")
            mz.append(f"| {r.short} | {r.tax if isinstance(r.tax, str) else '-'}{' **R**' if r.route_zero else ''} | {r.topo} | {td} | "
                      f"{r.get('ego_on_route_lane', '') if r.get('z_turn') is True else ''} | {ep if isinstance(ep, list) else ''} | "
                      f"{r.get('route_lc', '') if r.get('z_turn') is True else ''} | {v(40)} | {v(20)} | {v(0)} |")
    if err:
        mz += ["", f"{len(err)} rollouts not read: " + "; ".join(f"{e[0][7:15]} {e[1]}" for e in err[:10])]
    (out / "zeros.md").write_text("\n".join(mz) + "\n")
    print("\n".join(md[:12] + [""] + mz[:16]))
    if a.figs:
        figs(X, draws, Path(a.figs), a.pick.split(",") if a.pick else None, a.fig_name)


def figs(X, draws, out, pick=None, name="pai_follow_zeros.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    want = pick
    Z = X[(X.label == "rsg") & X.route_zero]
    pick = []
    for cls in ("straight_exit", "straight_road", "no_straight_exit", "curve_single", "lane_change", "other", "other_no_lane", "beyond_log_end", "other_no_turn"):
        pick += list(Z[Z.topo == cls].scene)[: 3 if cls == "straight_exit" else 2]
    pick = (pick + [s for s in Z.scene if s not in pick])[:6] if not want else [s for s in X[X.label == "rsg"].scene.unique() if s[7:15] in want][:6]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for ax, sc in zip(axs.flat, pick):
        d = draws[("rsg", sc)]
        z, lanes = d["z"], d["lanes"]
        cx, cy = z.get("ego_xy", d["pz"])
        R_ = 55.0
        for l in lanes.values():
            ax.plot(l["c"][:, 0], l["c"][:, 1], color="0.82", lw=0.8, zorder=1)
        chain_ids = {i for ids, _, _ in z.get("ego_chains", []) for i in ids}
        for ids, dpsi, full in z.get("ego_chains", []):
            col = "tab:green" if full and abs(dpsi) < STRAIGHT else "tab:orange"
            for i in ids:
                c = lanes[i]["c"]
                ax.plot(c[:, 0], c[:, 1], color=col, lw=2.5, alpha=0.7, zorder=2)
                ax.annotate("", xy=c[-1, :2], xytext=c[max(0, len(c) - 3), :2], arrowprops=dict(arrowstyle="->", color=col, lw=1.5), zorder=2)
        if z.get("ego_lane") in lanes:
            c = lanes[z["ego_lane"]]["c"]
            ax.plot(c[:, 0], c[:, 1], color="purple", lw=3, zorder=3, label="ego lane")
        ax.plot(d["R"][:, 0], d["R"][:, 1], "b--", lw=1.6, zorder=4, label="route")
        ax.plot(d["L"][:, 0], d["L"][:, 1], "k-", lw=1.0, zorder=4, label="log")
        ax.plot(d["P"][:, 0], d["P"][:, 1], "r-", lw=2.0, zorder=5, label="ego (to the zero)")
        ax.plot(*d["pz"], "rx", ms=12, mew=3, zorder=6)
        ax.plot(cx, cy, "o", color="purple", ms=8, zorder=6)
        if z.get("probe"):
            (x0, y0), (x1, y1) = z["probe"]
            ax.plot([x0, x1], [y0, y1], ":", color="purple", lw=1.5, zorder=6)
        r = X[(X.label == "rsg") & (X.scene == sc)].iloc[0]
        ax.set_xlim(cx - R_, cx + R_)
        ax.set_ylim(cy - R_, cy + R_)
        ax.set_aspect("equal")
        ax.set_title(f"{sc[7:15]}  {r.tax}  ->  {r.topo}\nturn {r.get('z_turn_deg', np.nan):+.0f} deg, v ego / log at -40/-20/0 m: "
                     f"{r.v_ego_40:.0f}/{r.v_log_40:.0f}, {r.v_ego_20:.0f}/{r.v_log_20:.0f}, {r.v_ego_0:.0f}/{r.v_log_0:.0f} m/s", fontsize=10)
        _ = chain_ids
    axs.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("RSG-L18-s0, PAI 281 route zeros: green = successor chains from the ego lane ending < 30 deg from straight, orange = turning chains, "
                 "purple = ego lane / ego at entry - 10 m (dotted: straight-ahead probe), x = zero event", fontsize=11)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=110)
    print("figure ->", out / name, pick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--limit", type=int, default=0)
    r = sp.add_parser("report")
    r.add_argument("--pkl", required=True)
    r.add_argument("--tax", required=True)
    r.add_argument("--s48", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--figs", default="")
    r.add_argument("--pick", default="", help="comma list of 8-hex scene ids for the sheet (default: by class)")
    r.add_argument("--fig-name", default="pai_follow_zeros.png")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else report(a)
