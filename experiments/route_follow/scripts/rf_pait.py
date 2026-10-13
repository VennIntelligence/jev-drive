#!/usr/bin/env python3
"""Turning school, demonstration source 1: pool `on1_PAIT` = PAI human-log rows at turns, with the human's own timing (user decision
2026-10-13: the demonstration carries the longitudinal profile, not the shape only; research/next-round/turning-school-rows.md, prereg
experiments/route_follow/plans/2026-10-13-turning-school-prereg.md). Pool contract: experiments/onpol_nuplan/plans/on_pool_format.md.

Source rows: a log-following PAI pass (onpol_pai onp_pass2.sh MODE=driver: the driver returns the log, the stack's MPC tracks it, the model is
fed and recorded as usual) -> onp_rows.sh -> onp_cache.py build (tokens + teacher on the card, gate G-id-ulp). Scenes: PAI train160 and the
exam hard_train half (experiments/alpasim/results/pai_exam/hard_train.txt); never hard_exam / control.

  route   AlpaSim image (alpasim-base, CPU). Per scene of --rows: the route messages pai_driver would have received at those states, by the
          runtime's own generator in the runtime's own call order (route_mem rm_build.regen: RouteGenerator.create(rig trajectory, map, MAP, 40),
          one generate_route per decision at pose = the row's anchor, z = log z at t0, prepare_for_policy) and the driver's memory of them
          (rm_build.feats_of = rmem.Tracker on the same messages) -> <out>/route/<scene>.npz: t0_us, k, anchor, wp (N, 20, 2), f (N, 200),
          lab (N, 51, 2) = rm_build.label on the generator's polyline after decision k, poly (M, 2) the final polyline (local frame),
          route0_d (N,) = |wp[0] - the first route point the pass's driver logged (drive.jsonl route0)| (gate G-route).
          --check '<run glob>' (pai_driver runs that logged their messages, e.g. the RB0-F-s0 re-roll): regenerated wp vs the logged wp and the
          features vs the served `rmem` -> <out>/route_check.json
  pack    op-train env, CPU. Rows of the log-pass cache (--cache, onp_cache layout) of the given scenes that are usable, before any scorer
          event / end, with a full 4 s logged future, inside a turn window: turn = a stretch of the route polyline whose heading changes
          >= 60 deg within 30 m of arc (entry / exit = where the heading leaves the stretch's start / end heading by 5 deg); window = from 6 s
          before the log reaches the entry to the moment it reaches the exit (the rows' own projections on the polyline). Target `fut` = the
          row's `imit` (onp_rows: the logged path at arc s_ego + d_k with d_k the log's own advance over t0 + 0.5 k s, lateral offset e0 faded
          over max(8 m, 2 s x v0)): position AND timing, so the imitation loss (pp_train dist: x, y, yaw) carries the speed profile.
          -> cache/on1_<name>/{tab (src = "human"), agent (the rows' actor boxes, ok True), road (ok False: no PAI raster), route}.npz,
          cache/on1_<name>@warp/{front.npy, teacher.npz}, runs/route_mem/feat/on1_<name>.npz, gates <out>/gates_<name>.json, BEV sheet.
          Gates: G-r1 (pool inputs bit-equal to the cache rows; the cache's G-id-ulp read is the pass's), G-route (route0_d <= 0.05 m on
          >= 99 % of rows), G-road (demo polyline crosses no road-edge segment on >= 99 % of rows), G-speed (demo chord arc at 4 s within 10 %
          of the log's on >= 99 % of rows; log arc < 1 m: within 0.1 m), G-bev (6-row sheet).
  merge   pools -> one (names prefixed on:<prefix>:..., feat merged, absent label files filled ok False; `src` kept, absent = "human";
          string tags of one pool kept, "" elsewhere; `pool` = the prefix, so rm_train --on-weight pool=<prefix>:W balances the parts).
  r1      GPU (or CPU) read of the pool rows: the pass's driver checkpoint on the pool's tokens + ego vs its logged poses_model, per-coordinate
          tolerance max(0.03 m, 4 fp16 ulps), and a wrong checkpoint (onp_cache gid-ulp's working line) -> gates file updated.

  docker run --rm --network none --user 0:0 --entrypoint bash -e DATA_DIR=/data -e PYTHONPATH=/data/third_party/alpasim/src/utils:/jev \
      -v /data:/data:ro -v $OUT:$OUT -v $CODE:/jev:ro -w /repo alpasim-base:0.89.0 -c 'umask 0000; /repo/.venv/bin/python \
      /jev/experiments/route_follow/scripts/rf_pait.py route --rows /data/runs/onpol_pai/rows/log --scenes <list> --out $OUT'
  /data/envs/op-train/bin/python experiments/route_follow/scripts/rf_pait.py pack --name PAIT --cache /data/runs/onpol_pai/train/cache/log \
      --rows /data/runs/onpol_pai/rows/log --route $OUT/route --scenes <train160 + hard_train lists> --out $OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/route_follow/scripts"), str(R / "experiments/route_mem/lib"), str(R / "experiments/route_mem/scripts"), str(R / "experiments/anchor_offset/scripts")]
D = Path(os.environ.get("DATA_DIR", "/data"))
CR, FEAT = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
PRE_S, APPROACH = 6.0, 40.0
ARC, R0TOL = 0.10, 0.05


def scenes_of(files):
    out = []
    for f in files:
        out += [l.split("\t")[0] for l in Path(f).read_text().splitlines() if l.startswith("clipgt-")]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- route (AlpaSim image)
def route0_of(run, session):
    """{k: route0 (2,)} of one session from the run's drive.jsonl."""
    out = {}
    for f in glob.glob(os.path.join(str(run), "driver", "drive.jsonl")):
        for line in open(f):
            if session in line and '"route0"' in line:
                r = json.loads(line)
                if r.get("kind") == "drive" and r["session"] == session and r.get("route0") is not None:
                    out[int(r["k"])] = np.asarray(r["route0"], float)
    return out


def qmul(a, b):
    """Quaternion product a * b, xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def regen(scene, recs, cfg, full):
    """rm_build.regen with the full ego attitude: the log pose interpolated at t0 (z, roll, pitch), turned about z to the anchor's yaw and
    moved to the anchor's x, y (the log-following ego carries the log's attitude; yaw-only poses miss up to 0.1 m at the 40 m start on slopes)."""
    import logging
    from ao_e1 import loader
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    logging.disable(logging.WARNING)
    ds = loader(str(cfg)).get_data_source(scene)
    tr = ds.rig.trajectory
    g = RouteGenerator.create(np.asarray(tr.positions, float), vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
    W, A = {}, {}
    for r in recs:
        x, y, yaw = r["anchor"]
        lp = tr.interpolate_pose(int(r["t0"]))
        dz = yaw - float(lp.yaw())
        q = qmul(np.array([0.0, 0.0, np.sin(dz / 2), np.cos(dz / 2)]), np.asarray(lp.quat, float))
        pose = Pose(np.array([x, y, float(np.asarray(lp.vec3)[2])]), q)
        W[r["k"]] = np.asarray(RouteGenerator.prepare_for_policy(g.generate_route(int(r["t0"]), pose)).waypoints, float)[:, :2]
        full[r["k"]] = np.asarray(g.route_polyline_in_local.points, float)[:, :2].copy()
        A[r["k"]] = np.asarray(r["anchor"], float)
    return W, A, ds


def turn_topo(ds, P):
    """Strict route turns of the final polyline P (rf_pai_follow's rf_follow port: 30 m / 60 deg net, entry / exit at 10 / 90 %) and the
    lane-graph class of each, read from the route lane 10 m before the entry (the log-following ego is on it): rf_pai_follow.topo's rule
    with ego lane = route lane -> [(entry, exit, deg, class)], class in lane_change / straight_exit / straight_road / no_straight_exit /
    curve_single / other_no_lane."""
    import rf_pai_follow as PF
    lanes = {str(l.id): dict(c=np.asarray(l.center.points, float)[:, [0, 1, 3]].astype(np.float32), nx=sorted(map(str, l.next_lanes)),
                             pv=sorted(map(str, l.prev_lanes)), al=sorted(map(str, l.adj_lanes_left)), ar=sorted(map(str, l.adj_lanes_right)))
             for l in ds.map.lanes}
    R = PF.dedup(np.asarray(P, float))
    S = PF.stations(R)
    s, h = PF.heading(R, S)
    out = []
    TT = PF.all_turns(s, h, 0.0, S[-1])
    if not TT:
        return out
    sr, ids = PF.route_lanes(lanes, R, S)
    for ent, ext, th in TT:
        a = max(ent - 10.0, 0.0)
        ra = ids[int(np.clip(np.searchsorted(sr, a), 0, len(ids) - 1))]
        p = np.array([np.interp(a, S, R[:, 0]), np.interp(a, S, R[:, 1])])
        psi = float(np.interp(a, s, h))
        TH, length = np.degrees(th), (ext - ent) + 20.0
        if ra is None:
            out.append((ent, ext, float(TH), "other_no_lane"))
            continue
        pe = [(PF.net_turn(Q, PF.HALF, 10.0 + length), bool(PF.stations(Q)[-1] >= min(10.0 + length + PF.HALF, PF.MIN_CHAIN)), ids_)
              for Q, ids_, _ in PF.chains(lanes, ra, p, 10.0 + length + PF.HALF)]
        way = any(f and np.sign(d) == np.sign(TH) and abs(d) >= 0.6 * abs(TH) for d, f, _ in pe)
        if not way:
            c = "lane_change"
        elif any(f and abs(d) < PF.STRAIGHT for d, f, _ in pe):
            c = "straight_exit"
        elif PF.road_ahead(lanes, p, psi, 10.0 + min(length, 40.0)):
            c = "straight_road"
        else:
            c = "no_straight_exit" if len({tuple(i) for _, f, i in pe if f}) > 1 else "curve_single"
        out.append((ent, ext, float(TH), c))
    return out


def route_one(job):
    import rm_build as RB
    f, out = job
    with np.load(f) as z:
        sc, run, se = str(z["scene"]), str(z["run"]), str(z["session"])
        t0, kk, an = z["t0_us"].astype(np.int64), z["k"].astype(int), z["anchor"].astype(np.float64)
    o = np.argsort(kk)
    recs = [dict(k=int(kk[i]), t0=int(t0[i]), anchor=an[i].tolist()) for i in o]
    full = {}
    W, A, ds = regen(sc, recs, Path(run) / "sim" / "generated-user-config-0.yaml", full)
    TT = turn_topo(ds, full[recs[-1]["k"]])
    F = RB.feats_of(W, A)
    r0 = route0_of(run, se)
    ks = [r["k"] for r in recs]
    wp = np.stack([W[k] for k in ks]).astype(np.float32)
    d0 = np.array([np.hypot(*(W[k][0] - r0[k])) if k in r0 and np.isfinite(W[k][0]).all() else np.nan for k in ks], np.float32)
    np.savez(Path(out) / "route" / f"{sc}.npz", t0_us=t0[o], k=np.array(ks), anchor=an[o], wp=wp, f=np.stack([F[k] for k in ks]).astype(np.float32),
             lab=np.stack([RB.label(full[k], A[k]) for k in ks]).astype(np.float32), poly=full[ks[-1]].astype(np.float64), route0_d=d0,
             turns=np.array([t[:3] for t in TT], float).reshape(-1, 3), topo=np.array([t[3] for t in TT], "U20"))
    return dict(scene=sc, rows=len(ks), route0_n=int(np.isfinite(d0).sum()), route0_max=float(np.nanmax(d0)) if np.isfinite(d0).any() else None)


def check_one(job):
    """One pai_driver run's session: regenerated wp / features vs the logged messages and the served rmem."""
    import rm_build as RB
    scene, recs, msgs, served, cfg = job
    W, A, _ = regen(scene, recs, cfg, {})
    F = RB.feats_of(W, A)
    dw = [float(np.nanmax(np.abs(W[k] - msgs[k]))) for k in W if k in msgs and np.isfinite(msgs[k]).any()]
    df = [float(np.abs(F[k] - served[k]).max()) for k in F if k in served]
    def pos(f):
        f = f.reshape(-1, 5)
        return f[:, 0] > 0.5, f[:, 1:3] * 20
    dp, dv = [], []
    for k in F:
        if k in served:
            (va, pa), (vb, pb) = pos(F[k]), pos(served[k])
            dv.append(int((va != vb).sum()))
            dp.append(float(np.hypot(*(pa - pb)[va & vb].T).max()) if (va & vb).any() else 0.0)
    FL = RB.feats_of({k: msgs[k] for k in W if k in msgs}, A)                # the same Tracker on the LOGGED messages (rf_pai_pool's path)
    dl = [float(np.abs(FL[k] - served[k]).max()) for k in FL if k in served]
    return dict(scene=scene, n_wp=len(dw), wp_max=max(dw) if dw else None, n_f=len(df), f_max=max(df) if df else None,
                f_share_over_002=float(np.mean(np.array(df) > 0.02)) if df else None, f_p99=float(np.quantile(df, 0.99)) if df else None,
                f_logged_max=max(dl) if dl else None, pos_p50=float(np.median(dp)) if dp else None, pos_p99=float(np.quantile(dp, 0.99)) if dp else None,
                pos_max=max(dp) if dp else None, valid_flip_rows=float(np.mean(np.array(dv) > 0)) if dv else None)


def cmd_route(a):
    from concurrent.futures import ProcessPoolExecutor
    out = Path(a.out)
    (out / "route").mkdir(parents=True, exist_ok=True)
    if a.check:
        jobs = []
        for d in sorted({d for g in a.check for d in glob.glob(g)}):
            S = {}
            for line in open(Path(d) / "driver" / "drive.jsonl"):
                if '"route"' not in line and '"rmem"' not in line:
                    continue
                r = json.loads(line)
                s = S.setdefault(r["session"], dict(scene=r["scene"], recs={}, msgs={}, served={}))
                if r.get("kind") == "route":
                    s["msgs"][int(r["ts"])] = np.array(r["wp"], float)
                elif r.get("kind") == "drive" and "rmem" in r:
                    s["recs"][int(r["k"])] = dict(k=int(r["k"]), t0=int(r["t0"]), anchor=r["anchor"])
                    s["served"][int(r["k"])] = np.array(r["rmem"], np.float32)
            cfg = Path(d) / "sim" / "generated-user-config-0.yaml"
            for s in list(S.values())[: a.check_n]:
                recs = [s["recs"][k] for k in sorted(s["recs"])]
                msgs = {r["k"]: s["msgs"][r["t0"]] for r in recs if r["t0"] in s["msgs"]}
                jobs.append((s["scene"], recs, msgs, s["served"], cfg))
        jobs = jobs[: a.check_n]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(check_one, jobs))
        wm = [r["wp_max"] for r in res if r["wp_max"] is not None]
        fm = [r["f_max"] for r in res if r["f_max"] is not None]
        rec = dict(sessions=len(res), wp_max=max(wm) if wm else None, f_max=max(fm) if fm else None,
                   f_share_over_002=float(np.mean([r["f_share_over_002"] for r in res if r["n_f"]])),
                   f_logged_max=max([r["f_logged_max"] for r in res if r["f_logged_max"] is not None], default=None),
                   pos_p99_max=max([r["pos_p99"] for r in res if r["pos_p99"] is not None], default=None),
                   valid_flip_rows=float(np.mean([r["valid_flip_rows"] for r in res if r["valid_flip_rows"] is not None])), per=res)
        (out / "route_check.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: v for k, v in rec.items() if k != "per"}))
        return
    keep = set(scenes_of(a.scenes)) if a.scenes else None
    files = [f for f in sorted(Path(a.rows).glob("clipgt-*.npz")) if keep is None or f.stem in keep]
    files = [f for f in files if a.force or not (out / "route" / f.name).exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(route_one, [(f, out) for f in files]))
    print(json.dumps(dict(scenes=len(res), rows=sum(r["rows"] for r in res),
                          route0_max=max([r["route0_max"] for r in res if r["route0_max"] is not None], default=None))))


# ---------------------------------------------------------------- pack (op-train)
def arc_on(P, X):
    """Arc position of points X (n, 2) projected on polyline P (m, 2) (nearest segment)."""
    d = np.diff(P, axis=0)
    L = np.maximum(np.hypot(*d.T), 1e-9)
    cum = np.r_[0.0, np.cumsum(L)]
    out = np.empty(len(X))
    for q, x in enumerate(X):
        t = np.clip(((x - P[:-1]) * d).sum(1) / L ** 2, 0, 1)
        e = np.hypot(*(P[:-1] + t[:, None] * d - x).T)
        i = int(np.argmin(e))
        out[q] = cum[i] + t[i] * L[i]
    return out


def chord(F, o=None):
    """(n, 8, 3) poses -> (n,) chord arc length from o (n, 2) (default origin) through the 8 points."""
    o = np.zeros((len(F), 2)) if o is None else o
    X = np.concatenate([o[:, None], F[..., :2]], 1)
    return np.hypot(*np.diff(X, axis=1).transpose(2, 0, 1)).sum(1)


def crosses(F, E, ok):
    """Demo polylines (n, 8, 3) from the origin vs road-edge boxes E (n, K, 5) x, y, yaw, length (thin segments) -> (n,) any proper crossing."""
    A = np.concatenate([np.zeros((len(F), 1, 2)), F[..., :2]], 1)
    p, q = A[:, :-1, None], A[:, 1:, None]                           # (n, 8, 1, 2)
    c, s, hl = np.cos(E[..., 2]), np.sin(E[..., 2]), E[..., 3] / 2
    u = np.stack([E[..., 0] - c * hl, E[..., 1] - s * hl], -1)[:, None]   # (n, 1, K, 2)
    v = np.stack([E[..., 0] + c * hl, E[..., 1] + s * hl], -1)[:, None]

    def cr(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])
    x = (cr(p, q, u) * cr(p, q, v) < 0) & (cr(u, v, p) * cr(u, v, q) < 0)
    return (x & ok[:, None]).any((1, 2))


def bev(path, rows, T, Zs, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rmem as RM
    fig, axs = plt.subplots(2, 3, figsize=(13, 10.5))
    for ax, i, z, txt in zip(axs.flat, rows, Zs, info):
        E, eo = T["edge_box"][i], T["edge_ok"][i]
        for e in E[eo]:
            c, s, hl = np.cos(e[2]), np.sin(e[2]), e[3] / 2
            ax.plot([-(e[1] - s * hl), -(e[1] + s * hl)], [e[0] - c * hl, e[0] + c * hl], "-", color="#888888", lw=1.0)
        pp, po = T["path"][i], T["path_ok"][i]
        ax.plot(-pp[po, 1], pp[po, 0], "-", color="#1f77b4", lw=1.2, alpha=0.6, label="logged path")
        f = z.reshape(RM.N_PT, RM.NF)
        v = f[:, 0] > 0.5
        ax.plot(-f[v, 2] * 20, f[v, 1] * 20, ".", color="#ff7f0e", ms=3, label="route memory (input)")
        B, bo = T["box"][i, 0], T["box_ok"][i, 0]
        for b in B[bo]:
            c, s = np.cos(b[2]), np.sin(b[2])
            cx = np.array([1, 1, -1, -1, 1]) * b[3] / 2
            cy = np.array([1, -1, -1, 1, 1]) * b[4] / 2
            ax.plot(-(b[1] + s * cx + c * cy), b[0] + c * cx - s * cy, "-", color="#9467bd", lw=0.7)
        fu = T["imit"][i]
        ax.plot(np.r_[0, -fu[:, 1]], np.r_[0, fu[:, 0]], "s-", color="#d62728", ms=3, lw=1.4, label="demo = human future 4 s (target)")
        for k in range(8):
            ax.annotate(f"{(k + 1) * 0.5:g}", (-fu[k, 1], fu[k, 0]), fontsize=5.5, color="#d62728", xytext=(3, 0), textcoords="offset points")
        ax.plot([0], [0], "k^", ms=8)
        ax.set_xlim(-30, 30), ax.set_ylim(-10, 70), ax.set_aspect("equal")
        ax.set_title(txt, fontsize=7.5)
        ax.set_xlabel("-y (m, right +)"), ax.set_ylabel("x (m)")
    axs.flat[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_pack(a):
    T = np.load(Path(a.cache) / "tab.npz")
    scenes = T["scenes"][T["sid"]]
    want = scenes_of(a.scenes)
    rd = Path(a.route)
    have = [s for s in want if s in set(T["scenes"].tolist()) and (rd / f"{s}.npz").exists()]
    sel, info, F, LB, TW, TP, DE = [], {}, [], [], [], [], []
    r0d = []
    for sc in have:
        Z = np.load(rd / f"{sc}.npz")
        P = Z["poly"][:, :2]
        TT = [(float(e), float(x), float(d), str(c)) for (e, x, d), c in zip(Z["turns"], Z["topo"])]
        idx = np.flatnonzero(scenes == sc)
        idx = idx[np.argsort(T["k"][idx])]
        pos = {int(k): j for j, k in enumerate(Z["k"])}
        j = np.array([pos[int(k)] for k in T["k"][idx]])
        assert np.array_equal(Z["t0_us"][j], T["t0_us"][idx])
        arc = arc_on(P, Z["anchor"][j, :2])
        t0 = T["t0_us"][idx].astype(np.int64)
        good = T["usable"][idx] & ~T["after_event"][idx] & ~T["after_end"][idx] & T["imit_ok"][idx].all(1)
        m = np.zeros(len(idx), bool)
        tid, tp, de = np.full(len(idx), -1), np.full(len(idx), "", "U20"), np.full(len(idx), np.nan, np.float32)
        used = []
        for q, (s_in, s_out, deg, cl) in enumerate(TT):
            rin = np.flatnonzero(arc >= s_in)
            if not len(rin) or arc[0] > s_out:                       # never reached by the log, or behind it at the start
                continue
            t_in = t0[rin[0]]
            r40 = np.flatnonzero(arc >= s_in - APPROACH)
            t_a = min(t_in - PRE_S * 1e6, t0[r40[0]])                 # 6 s before the entry or the -40 m crossing, whichever is earlier
            rout = np.flatnonzero(arc >= s_out)
            t_out = t0[rout[0]] if len(rout) else t0[-1]
            w = (t0 >= t_a) & (t0 <= t_out) & good & (tid < 0)
            tid[w], tp[w], de[w] = q, cl, (s_in - arc)[w]
            m |= w
            vat = lambda d: float(np.interp(s_in - d, arc, T["v0"][idx])) if arc[0] <= s_in - d else float("nan")  # noqa: E731
            used.append(dict(s_in=s_in, s_out=s_out, deg=deg, topo=cl, t_in=float((t_in - t0[0]) * 1e-6), t_out=float((t_out - t0[0]) * 1e-6),
                             v_log_40=vat(40), v_log_20=vat(20), v_log_0=vat(0), rows=int(w.sum())))
        info[sc] = dict(turns=TT, used=used, rows=int(m.sum()))
        sel.append(idx[m])
        F.append(Z["f"][j[m]])
        LB.append(Z["lab"][j[m]])
        TW.append(tid[m])
        TP.append(tp[m])
        DE.append(de[m])
        r0d.append(Z["route0_d"][j[m]])
    sel = np.concatenate(sel)
    F, LB, TW, r0d, TP, DE = (np.concatenate(x) for x in (F, LB, TW, r0d, TP, DE))
    n = len(sel)
    sc, kk = scenes[sel], T["k"][sel].astype(int)
    names = np.array([f"on:pait:{s}:{k}" for s, k in zip(sc, kk)])
    assert len(set(names.tolist())) == n
    fut = T["imit"][sel].astype(np.float32)
    assert np.isfinite(fut).all()
    hard = set(scenes_of([a.hard])) if a.hard else set()
    part = np.array(["hard_train" if s in hard else "train160" for s in sc])
    # ---- gates
    arc_d, arc_l = chord(fut), chord(T["log_fut"][sel], T["log0"][sel][:, :2])
    arc_ok = np.where(arc_l >= 1.0, np.abs(arc_d / np.maximum(arc_l, 1e-6) - 1) <= ARC, np.abs(arc_d - arc_l) <= 0.1)
    E, eo = T["edge_box"][sel], T["edge_ok"][sel]
    xd, xl = crosses(fut, E, eo), crosses(T["log_fut"][sel], E, eo)
    v4 = np.diff(np.concatenate([np.zeros((n, 1, 2)), fut[..., :2]], 1), axis=1)
    vd = np.hypot(*v4.transpose(2, 0, 1)) / 0.5
    G = dict(name=a.name, scenes_asked=len(want), scenes_with_rows=len(have), scenes_missing=[s for s in want if s not in have][:400],
             rows=n, scenes=int(len(np.unique(sc))), turns=int(sum(len(v["used"]) for v in info.values())),
             rows_by_part={p: int((part == p).sum()) for p in ("train160", "hard_train")},
             rows_by_topo={c: int((TP == c).sum()) for c in np.unique(TP)},
             turns_by_topo={c: int(sum(u["topo"] == c for v in info.values() for u in v["used"])) for c in np.unique(TP)},
             rows_by_d_entry={"> 40 m": int((DE > 40).sum()), "40 .. 20": int(((DE <= 40) & (DE > 20)).sum()), "20 .. 0": int(((DE <= 20) & (DE > 0)).sum()),
                              "in junction (<= 0)": int((DE <= 0).sum())},
             human_entry_speed_median={f"v_log_{d}": float(np.nanmedian([u[f"v_log_{d}"] for v in info.values() for u in v["used"]] or [np.nan]))
                                       for d in (40, 20, 0)},
             G_route=dict(rows_with_route0=int(np.isfinite(r0d).sum()), share_le_005=float((r0d[np.isfinite(r0d)] <= R0TOL).mean()),
                          max=float(np.nanmax(r0d)), feat_ok=float((F[:, 0::5].sum(1) > 0).mean())),
             G_road=dict(demo_cross_share=float(xd.mean()), log_cross_share=float(xl.mean()), rows_with_edges=int(eo.any(1).sum())),
             G_speed=dict(within_10pct=float(arc_ok.mean()), median_ratio=float(np.median(arc_d[arc_l >= 1] / arc_l[arc_l >= 1])),
                          stopped_rows=int((arc_l < 1).sum()), v0_median=float(np.median(T["v0"][sel])), demo_v_mean_median=float(np.median(vd.mean(1))),
                          v0_ge_12_share=float((T["v0"][sel] >= 12).mean())),
             tracking=dict(e0_abs_p95=float(np.quantile(np.abs(T["e0"][sel]), 0.95)), ds0_abs_p95=float(np.quantile(np.abs(T["ds0"][sel]), 0.95))),
             per_scene=info)
    G["G_route"]["pass"] = G["G_route"]["share_le_005"] >= 0.99
    G["G_road"]["pass"] = G["G_road"]["demo_cross_share"] <= 0.01
    G["G_speed"]["pass"] = G["G_speed"]["within_10pct"] >= 0.99
    if not a.gates_only:
        pool, warp = CR / f"on1_{a.name}", CR / f"on1_{a.name}@warp"
        pool.mkdir(parents=True, exist_ok=True), warp.mkdir(parents=True, exist_ok=True), FEAT.mkdir(parents=True, exist_ok=True)
        np.savez(pool / "tab.npz", names=names, log=sc, ego=T["ego_in"][sel].astype(np.float32), fut=fut, cam=T["cam_t"][sel].astype(np.float32),
                 lht=T["lht"][sel].astype(bool), speed=T["v0"][sel].astype(np.float32), served=T["poses"][sel].astype(np.float32),
                 src=np.full(n, "human"), part=part, topo=TP, d_entry=DE)
        tok = np.load(Path(a.cache) / "tok.npy", mmap_mode="r")
        fr = np.lib.format.open_memmap(warp / "front.npy", "w+", np.float16, (n, 8) + tok.shape[1:])
        tix = T["tix"][sel]
        for i in range(0, n, 512):
            fr[i:i + 512] = tok[tix[i:i + 512]]
        fr.flush()
        del fr
        np.savez(warp / "teacher.npz", out=T["t_out"][sel].astype(np.float32), plan=np.zeros((n, 33, 15), np.float32), di=T["di"], pi=T["pi"])
        np.savez(pool / "agent.npz", tokens=names, ok=np.ones(n, bool), box=T["box"][sel].astype(np.float32), valid=T["box_ok"][sel].astype(bool))
        np.savez(pool / "road.npz", tokens=names, ok=np.zeros(n, bool), sdf=np.zeros((n, 128, 96), np.float16))
        seg = np.linalg.norm(np.diff(LB, axis=1), axis=-1).astype(np.float32)
        fok = F[:, 0::5].sum(1) > 0
        import rmem as RM
        sm = RM.summary(F)
        np.savez(pool / "route.npz", tokens=names, ok=fok & (np.isfinite(LB).all(-1).sum(1) >= 2), poly=LB, seglen=seg, dz=np.ones(n, np.float32),
                 cov20=np.asarray(sm["c20"], bool) & fok)
        np.savez(FEAT / f"on1_{a.name}.npz", names=names, f=F.astype(np.float16), k=kk, ok=fok)
        # G-r1: the pool files are the cache rows bit for bit (front, ego, teacher)
        fr = np.load(warp / "front.npy", mmap_mode="r")
        rs = np.random.default_rng(0).choice(n, min(n, 256), replace=False)
        G["G_r1"] = dict(front_equal=bool(all(np.array_equal(fr[i], tok[tix[i]]) for i in rs)),
                         ego_equal=bool(np.array_equal(np.load(pool / "tab.npz")["ego"], T["ego_in"][sel].astype(np.float32))),
                         feat_f16_max_diff=float(np.abs(np.load(FEAT / f"on1_{a.name}.npz")["f"].astype(np.float32) - F).max()), pool=str(pool))
    figs = R / "experiments/route_follow/figs"
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pick = []
    for q in rng.permutation(len(np.unique(sc))):                    # one row per scene, at most 6, a random decision inside the window
        s = np.unique(sc)[q]
        ii = np.flatnonzero(sc == s)
        pick.append(int(rng.choice(ii)))
        if len(pick) == 6:
            break
    txt = [f"{sc[i][7:15]} k {kk[i]} v0 {T['v0'][sel][i]:.1f} m/s, arc4 log {arc_l[i]:.1f} / demo {arc_d[i]:.1f} m, part {part[i]}" for i in pick]
    bev(figs / f"pait_bev_{a.name}.png", [sel[i] for i in pick], T, [F[i] for i in pick], txt)
    G["G_bev"] = dict(fig=f"experiments/route_follow/figs/pait_bev_{a.name}.png", rows=[str(names[i]) for i in pick])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gates_{a.name}.json").write_text(json.dumps(G, indent=1))
    print(json.dumps({k: v for k, v in G.items() if k not in ("per_scene", "scenes_missing")}, indent=1))


# ---------------------------------------------------------------- merge
def cmd_merge(a):
    spec = [x.split(":") for x in a.pools]                            # pool:prefix[:mode] (mode -> tab `mode`, on_train --on-mode mixed)
    src = [(x[0], x[1]) for x in spec]
    modes = [x[2] if len(x) > 2 else "" for x in spec]
    out, warp = CR / f"on1_{a.out}", CR / f"on1_{a.out}@warp"
    out.mkdir(exist_ok=True), warp.mkdir(exist_ok=True)
    ren = lambda v, pre: np.array([f"on:{pre}:{t[3:]}" for t in v.tolist()])  # noqa: E731
    tabs = [dict(np.load(CR / p / "tab.npz")) for p, _ in src]
    ns = [len(t["names"]) for t in tabs]
    for t in tabs:
        t.setdefault("src", np.full(len(t["names"]), "human"))
    keys = set.intersection(*[set(t) for t in tabs])
    M = {k: np.concatenate([ren(t[k], pre) if k == "names" else t[k] for t, (_, pre) in zip(tabs, src)]) for k in keys}
    for k in set.union(*[set(t) for t in tabs]) - keys:                # per-row string tags of one pool (topo, part, cls): "" elsewhere
        if all(k not in t or (t[k].dtype.kind == "U" and len(t[k]) == n) for t, n in zip(tabs, ns)):
            M[k] = np.concatenate([t[k] if k in t else np.full(n, "") for t, n in zip(tabs, ns)]).astype("U20")
    M["src"] = M["src"].astype("U16")
    M["pool"] = np.concatenate([np.full(n, pre) for n, (_, pre) in zip(ns, src)]).astype("U16")   # --on-weight pool=<prefix>:W balances the parts
    if any(modes):
        M["mode"] = np.concatenate([np.full(n, md or "both") for n, md in zip(ns, modes)]).astype("U8")
    np.savez(out / "tab.npz", **M)
    names = M["names"]
    assert len(set(names.tolist())) == len(names)
    for f in ("agent.npz", "road.npz", "drivable.npz"):                # a pool without the file: zero rows, ok False
        Z = [dict(np.load(CR / p / f)) if (CR / p / f).exists() else None for p, _ in src]
        if all(z is None for z in Z):
            continue
        z0 = next(z for z in Z if z is not None)
        A = {"tokens": names}
        for k in set.intersection(*[set(z) for z in Z if z is not None]):   # keys every present file has (tokens, ok, box / valid, sdf)
            if k == "tokens" or len(z0[k]) != len(z0["tokens"]):
                continue
            A[k] = np.concatenate([z[k] if z is not None else np.zeros((n,) + z0[k].shape[1:], z0[k].dtype) for z, n in zip(Z, ns)])
        assert all(z is None or len(z["tokens"]) == n for z, n in zip(Z, ns)), f"{f}: rows"
        np.savez(out / f, **A)
    Z = [dict(np.load(CR / p / "route.npz")) for p, _ in src]
    np.savez(out / "route.npz", tokens=names, **{k: np.concatenate([z[k] for z in Z]) for k in ("ok", "poly", "seglen", "dz", "cov20")})
    Fz = [np.load(FEAT / f"{p}.npz") for p, _ in src]
    np.savez(FEAT / f"on1_{a.out}.npz", names=names, f=np.concatenate([z["f"] for z in Fz]).astype(np.float16), k=np.concatenate([z["k"] for z in Fz]),
             ok=np.concatenate([z["ok"] for z in Fz]))
    Fr = [np.load(CR / f"{p}@warp/front.npy", mmap_mode="r") for p, _ in src]
    o = np.lib.format.open_memmap(warp / "front.npy", "w+", np.float16, (len(names),) + Fr[0].shape[1:])
    i = 0
    for x in Fr:
        for j in range(0, len(x), 4096):
            o[i + j:i + min(j + 4096, len(x))] = x[j:j + 4096]
        i += len(x)
    o.flush()
    Tz = [dict(np.load(CR / f"{p}@warp/teacher.npz")) for p, _ in src]
    assert all(np.array_equal(t["di"], Tz[0]["di"]) and np.array_equal(t["pi"], Tz[0]["pi"]) for t in Tz)
    np.savez(warp / "teacher.npz", out=np.concatenate([t["out"] for t in Tz]), plan=np.concatenate([t["plan"] for t in Tz]), di=Tz[0]["di"], pi=Tz[0]["pi"])
    print(json.dumps(dict(pool=str(out), rows=len(names), parts={p: n for (p, _), n in zip(src, ns)}, modes=modes,
                          src={s: int((M["src"] == s).sum()) for s in np.unique(M["src"])})))


# ---------------------------------------------------------------- r1
def cmd_r1(a):
    import torch
    sys.path[:0] = [str(R / "experiments/op_parity/scripts"), str(R / "experiments/onpol_pai/lib"), str(R / "lib"), str(R / "experiments/body1/lib")]
    import onp_loss as OL
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    dev = torch.device(a.device)
    pool, warp = CR / f"on1_{a.name}", CR / f"on1_{a.name}@warp"
    tb = np.load(pool / "tab.npz")
    C = np.load(Path(a.cache) / "tab.npz")
    key = {(s, int(k)): i for i, (s, k) in enumerate(zip(C["scenes"][C["sid"]].tolist(), C["k"].tolist()))}
    ci = np.array([key[(n.split(":")[2], int(n.split(":")[3]))] for n in tb["names"].tolist()])
    ref = torch.as_tensor(C["poses_model"][ci][..., :2], dtype=torch.float64)
    ulp = torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(2.0 ** -14))) - 10)
    tol = torch.clamp(4 * ulp, min=0.03)
    fr = np.load(warp / "front.npy", mmap_mode="r")
    W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
    ego = torch.as_tensor(tb["ego"], device=dev)
    tc = torch.as_tensor(np.where(tb["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)
    cam = torch.as_tensor(tb["cam"][:, :2], device=dev)
    res = {}
    for tag in [a.tag] + a.control:
        m = T.load_pmodel(tag, dev)
        pi = torch.as_tensor(np.arange(m.net.slices["plan"].start, m.net.slices["plan"].start + 495), device=dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(ci), a.bs):
                r = slice(i, i + a.bs)
                o = m(torch.as_tensor(np.asarray(fr[r]), device=dev), ego[r], tc[r]).float()
                out.append(torch.stack(OL.rear_xy(o[:, pi].view(-1, 33, 15), cam[r], W)[:2], -1).double().cpu())
        d = (torch.cat(out) - ref).abs()
        res[tag] = dict(share_over=float((d > tol).flatten(1).any(1).float().mean()), share_over_fixed=float((d > 0.03).flatten(1).any(1).float().mean()),
                        max_m=float(d.max()), median_m=float(d.flatten(1).amax(1).median()))
    gp = Path(a.out) / f"gates_{a.name}.json"
    G = json.loads(gp.read_text())
    G["G_r1_read"] = dict(device=a.device, model=res[a.tag], control={c: res[c] for c in a.control},
                          pass_=bool(res[a.tag]["share_over"] <= 0.01 and all(res[c]["share_over"] >= 0.9 for c in a.control)))
    gp.write_text(json.dumps(G, indent=1))
    print(json.dumps(G["G_r1_read"], indent=1))
    raise SystemExit(0 if G["G_r1_read"]["pass_"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("route")
    p.add_argument("--rows", default=""), p.add_argument("--scenes", nargs="*", default=[]), p.add_argument("--out", required=True)
    p.add_argument("--check", nargs="*", default=[]), p.add_argument("--check-n", type=int, default=8)
    p.add_argument("--workers", type=int, default=8), p.add_argument("--force", action="store_true")
    p = sub.add_parser("pack")
    p.add_argument("--name", default="PAIT"), p.add_argument("--cache", required=True), p.add_argument("--route", required=True)
    p.add_argument("--scenes", nargs="+", required=True), p.add_argument("--hard", default=""), p.add_argument("--out", required=True)
    p.add_argument("--gates-only", action="store_true")
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True), p.add_argument("--pools", nargs="+", required=True)
    p = sub.add_parser("r1")
    p.add_argument("--name", default="PAIT"), p.add_argument("--cache", required=True), p.add_argument("--out", required=True)
    p.add_argument("--tag", default="P2H10S-F-s0"), p.add_argument("--control", nargs="+", default=["P2H10S-F-s1"])
    p.add_argument("--device", default="cpu"), p.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    dict(route=cmd_route, pack=cmd_pack, merge=cmd_merge, r1=cmd_r1)[a.cmd](a)
