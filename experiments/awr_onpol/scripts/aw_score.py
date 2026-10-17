#!/usr/bin/env python3
"""AWR-ONPOL outcome score and advantage per on-policy decision (plans/2026-10-17-awr-onpol-prereg.md, section 1). Box, CPU, AlpaSim's venv
(polars reads the evaluator's metrics.parquet; Int128 timestamps). No model, no expert, no label: only what the stored rollouts recorded.

  build [--h 3.0] [--workers N] [--limit S]   pool on1_RSGACC (6 policies x nuPlan onpol-train / PAI train160, rolled under ACC+GAP, d333)
        -> $DATA_DIR/runs/awr_onpol/pack/on1_RSGACC.npz (one entry per pool row, pool order) + results/score.{md,json} (coverage)
  check                                       known answers: rear(mu) of the default-serve pool rows (on1_L18RB0, served = model plan)
        equals their served poses; the window scorer on synthetic series (exit 2 on a mismatch)

Score of decision k (time t, window W = (t, t + H], H = --h, evaluator terms of eval/aggregation/scene_score.py at the window):
  walk the evaluator's timesteps T_j in W in order (nuPlan 0.5 s, PAI 0.1 s); the first terminal step (collision_any | offroad | dist_to_gt_trajectory
  >= 4 m, the evaluator's RemoveTimestepsAfterEvent) ends the window; it is a FAIL if at-fault there (collision_front | collision_lateral |
  offroad | left_corridor_laterally = the scene score's hard failures), else (rear collision only, or leaving the corridor forwards) the window is
  truncated without failure.
  s = 0 on FAIL, else progress p = min(1, ego_adv / (0.8 log_adv)) (the scene score's saturation 0.8), ego_adv = metres along the GT over the
  window from the evaluator's `progress` x gt length, log_adv = the log's own metres over the same window (nuPlan: rows meta D; PAI: log_s); p = 1
  when log_adv < 2 m x win / 3 s (the scene score's "short GT -> full score" rule at window scale). Comfort: not in the stored metrics.
  invalid: a terminal step at or before t (the evaluator stopped scoring), or no FAIL and a window shorter than 1 s (scene end).
Baseline b(state) = mean s of the OTHER policies' rows at the same (scene, decision time) whose ego state is near-identical: rear axle within
  0.5 m, heading within 3 deg, speed within 0.5 m/s (nuPlan map frame / PAI scene frame; the traffic replays the log, so ego state + time
  is the state). kind 1 = matched (>= 1 such peer); kind 2 = no near-identical peer: leave-one-out mean over the other policies at the same
  (scene, time) (the per-scene / per-time baseline; states differ); kind 0 = no valid peer at all -> A undefined (NaN). A = s - b.
Target (the action, prereg 2): the model's own raw plan at that decision (ACC is part of the environment): nuPlan = rear(mu) of the dump's plan
  mean (pp_train.rear on the 8 poses 0.5 .. 4 s, cam_x of the row), PAI = the rollout cache's poses_model. The ACC-served poses are not the action.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R_))
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = D / "runs/op_parity/cache"
POOL = "on1_RSGACC"
OUT = D / "runs/awr_onpol/pack"
RES = R_ / "experiments/awr_onpol/results"
NU_ROWS, PAI_ROWS = D / "runs/onpol_nuplan/rows", D / "runs/route_follow/rsgacc/rows/pai"
POL = {"a0": "RB0-F-s0", "a1": "RB0-F-s1", "g0": "RSG-L18-s0", "g1": "RSG-L18-s1", "l0": "RSGlr03n5000-L18-s0", "k0": "RSGk51w1-L18-s0"}
TOL_XY, TOL_YAW, TOL_V = 0.5, np.radians(3.0), 0.5
SAT, SHORT_M, MIN_WIN, CORRIDOR = 0.8, 2.0, 1.0, 4.0
HS = (2.0, 3.0, 4.0)                                                 # coverage table; --h is the pack's
T8 = 0.5 * np.arange(1, 9)
GRID = np.array([10.0 * (i / 32) ** 2 for i in range(33)])         # jevdrive.op_adapt.T_IDXS
MET = ("collision_any", "collision_front", "collision_lateral", "collision_rear", "offroad", "left_corridor_laterally",
       "dist_to_gt_trajectory", "progress", "gt_dist_traveled_m")


def t_weights(ts, grid=GRID):
    """= op_adapt_r2.t_weights: (len(ts), 33) linear interpolation weights of the plan grid."""
    W = np.zeros((len(ts), len(grid)), np.float32)
    for i, t in enumerate(ts):
        j = int(np.clip(np.searchsorted(grid, t) - 1, 0, len(grid) - 2))
        w = (t - grid[j]) / (grid[j + 1] - grid[j])
        W[i, j], W[i, j + 1] = 1 - w, w
    return W


def rear(mu, cam_x):
    """= pp_train.rear in numpy: plan (n, 33, 15) camera frame -> (n, 8, 3) rear-axle x, y (left +), yaw (left +) at 0.5 .. 4 s."""
    W = t_weights(T8).astype(np.float64)
    c = np.asarray(cam_x, np.float64)[:, None]
    x, y, psi = (mu[..., k].astype(np.float64) @ W.T for k in (0, 1, 11))
    psi = -psi
    return np.stack([x + c - c * np.cos(psi), -y - c * np.sin(psi), psi], -1).astype(np.float32)


def series(f):
    """metrics.parquet -> (T seconds (m,), {name: (m,)}) on the union of timestamps (missing -> 0)."""
    import polars as pl
    m = pl.read_parquet(f, columns=["name", "timestamps_us", "values"]).filter(pl.col("name").is_in(MET))
    m = m.with_columns(pl.col("timestamps_us").cast(pl.Int64))
    T = np.unique(m["timestamps_us"].to_numpy())
    S = {}
    for (n,), x in m.group_by("name"):
        v = np.zeros(len(T))
        v[np.searchsorted(T, x["timestamps_us"].to_numpy())] = np.nan_to_num(x["values"].to_numpy().astype(float))
        S[n] = v
    for n in MET:
        S.setdefault(n, np.zeros(len(T)))
    return T * 1e-6, S


def window(T, S, t, H, log_adv):
    """One decision at time t (s): -> (valid, fail, prog, score, win). log_adv(win) -> the log's metres over (t, t + win]."""
    term = (S["collision_any"] > 0) | (S["offroad"] > 0) | (S["dist_to_gt_trajectory"] >= CORRIDOR)
    atf = (S["collision_front"] > 0) | (S["collision_lateral"] > 0) | (S["offroad"] > 0) | (S["left_corridor_laterally"] > 0)
    if (term & (T <= t + 1e-6)).any():
        return False, False, np.nan, np.nan, 0.0
    j = np.flatnonzero((T > t + 1e-6) & (T <= t + H + 1e-6))
    hit = j[term[j]]
    if len(hit) and atf[hit[0]]:
        return True, True, 0.0, 0.0, float(T[hit[0]] - t)
    t1 = float(T[hit[0]]) if len(hit) else min(t + H, float(T[-1]))
    win = t1 - t
    if win < MIN_WIN - 1e-6:
        return False, False, np.nan, np.nan, win
    g = S["progress"] * S["gt_dist_traveled_m"]
    ego = float(np.interp(t1, T, g) - np.interp(t, T, g))
    la = float(log_adv(win))
    p = 1.0 if la < SHORT_M * win / 3.0 else float(np.clip(ego / (SAT * la), 0.0, 1.0))
    return True, False, p, p, win


def _nu_job(job):
    """One nuPlan (run, scene): rows (t_us, D) -> per H (valid, fail, prog, score, win)."""
    run, scene, t_us, Dm = job
    f = sorted(glob.glob(f"{run}/rollouts/{scene}/*/metrics.parquet"))
    out = np.full((len(t_us), len(HS), 5), np.nan)
    if not f:
        return out
    T, S = series(f[0])
    for i, (t, d) in enumerate(zip(t_us, Dm)):
        la = lambda w, d=d: np.interp(w, np.r_[0.0, T8], np.r_[0.0, d])  # noqa: E731
        for h, H in enumerate(HS):
            out[i, h] = window(T, S, t * 1e-6, H, la)
    return out


def _pai_job(job):
    """One PAI policy x scene rows file -> (k, t, pose, v, target, per-H window) of every decision in it."""
    path, = job
    z = np.load(path)
    scene, run = str(z["scene"]), str(z["run"])
    t, ls, lt = z["t0_us"].astype(np.int64), z["log_s"].astype(float), z["log_t_us"].astype(np.int64)
    out = np.full((len(t), len(HS), 5), np.nan)
    f = sorted(glob.glob(f"{run}/sim/rollouts/{scene}/*/metrics.parquet"))
    if f:
        T, S = series(f[0])
        for i, ti in enumerate(t):
            s0 = np.interp(ti, lt, ls)
            la = lambda w, ti=ti, s0=s0: np.interp(ti + w * 1e6, lt, ls) - s0  # noqa: E731
            for h, H in enumerate(HS):
                out[i, h] = window(T, S, ti * 1e-6, H, la)
    return scene, z["k"].astype(int), t, z["anchor"].astype(np.float64), z["v0"].astype(np.float64), z["poses_model"].astype(np.float32), out


def baseline(key, pol, xy, yaw, v, s):
    """-> b, n_peer, kind per row (module docstring)."""
    n = len(s)
    b, npeer, kind = np.full(n, np.nan), np.zeros(n, np.int16), np.zeros(n, np.int8)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cut = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, z in zip(cut[:-1], cut[1:]):
        g = order[a:z]
        g = g[np.isfinite(s[g])]
        if len(g) < 2:
            continue
        P = pol[g]
        dxy = np.hypot(xy[g, None, 0] - xy[None, g, 0], xy[g, None, 1] - xy[None, g, 1])
        dyw = np.abs((yaw[g, None] - yaw[None, g] + np.pi) % (2 * np.pi) - np.pi)
        dv = np.abs(v[g, None] - v[None, g])
        other = P[:, None] != P[None, :]
        near = other & (dxy <= TOL_XY) & (dyw <= TOL_YAW) & (dv <= TOL_V)
        sg = s[g]
        for i, r in enumerate(g):
            if near[i].any():
                b[r], npeer[r], kind[r] = sg[near[i]].mean(), near[i].sum(), 1
            elif other[i].any():
                b[r], npeer[r], kind[r] = sg[other[i]].mean(), other[i].sum(), 2
    return b, npeer, kind


def cmd_build(a):
    from jevdrive.common import n_cpus
    tb = np.load(CR / POOL / "tab.npz")
    names, pre = tb["names"].astype(str), tb["pool"].astype(str)
    n = len(names)
    pos = {x: i for i, x in enumerate(names.tolist())}
    F = dict(track=np.full(n, "", "<U3"), pol=np.full(n, "", "<U3"), scene=np.full(n, "", "<U55"), t=np.full(n, -1, np.int64),
             k=np.full(n, -1, np.int32), xy=np.full((n, 2), np.nan), yaw=np.full(n, np.nan), v=np.full(n, np.nan),
             target=np.full((n, 8, 3), np.nan, np.float32), w=np.full((n, len(HS), 5), np.nan))
    workers = a.workers or n_cpus()
    with ProcessPoolExecutor(workers) as ex:
        for x in POL:                                                # nuPlan pools: rows meta (the row's state, D, mu), metrics per (run, scene)
            m = np.load(NU_ROWS / f"L18A{x}" / "meta.npz")
            pn = np.array([f"on:n{x}:" + s[3:] for s in m["names"].astype(str).tolist()])
            idx = np.array([pos.get(s, -1) for s in pn.tolist()])
            assert (idx >= 0).mean() > 0.99, f"L18A{x}: {(idx < 0).sum()} meta rows not in {POOL}"
            keep = np.flatnonzero(idx >= 0)
            if a.limit:
                sc_ok = set(np.unique(m["scene"].astype(str))[:a.limit].tolist())
                keep = keep[np.isin(m["scene"].astype(str)[keep], list(sc_ok))]
            r = idx[keep]
            F["track"][r], F["pol"][r], F["scene"][r] = "nu", x, m["scene"].astype(str)[keep]
            F["t"][r], F["k"][r] = m["t_us"][keep], m["k"][keep]
            F["xy"][r], F["yaw"][r], F["v"][r] = m["pose_map"][keep, :2], m["pose_map"][keep, 2], m["speed"][keep]
            F["target"][r] = rear(m["mu"][keep], tb["cam"][r, 0])
            run, sc = m["run"].astype(str)[keep], m["scene"].astype(str)[keep]
            grp = {}
            for j, key in enumerate(zip(run.tolist(), sc.tolist())):
                grp.setdefault(key, []).append(j)
            jobs = [(ru, s, m["t_us"][keep[g]], m["D"][keep[g]]) for (ru, s), g in grp.items()]
            for (key, g), res in zip(grp.items(), ex.map(_nu_job, jobs, chunksize=8)):
                F["w"][r[g]] = res
            print(f"nuPlan {x}: {len(r)} rows, {len(grp)} rollouts", flush=True)
        for x in POL:                                                # PAI pools: one rows file per scene
            files = sorted((PAI_ROWS / f"PAIA{x}").glob("clipgt-*.npz"))[:a.limit or None]
            for scene, k, t, anc, v0, pm, res in ex.map(_pai_job, [(f,) for f in files]):
                pn = [f"on:p{x}:pai:{scene}:{q}" for q in k.tolist()]
                idx = np.array([pos.get(s, -1) for s in pn])
                j = np.flatnonzero(idx >= 0)
                r = idx[j]
                F["track"][r], F["pol"][r], F["scene"][r], F["t"][r], F["k"][r] = "pai", x, scene, t[j], k[j]
                F["xy"][r], F["yaw"][r], F["v"][r], F["target"][r], F["w"][r] = anc[j, :2], anc[j, 2], v0[j], pm[j], res[j]
            print(f"PAI {x}: {len(files)} scenes", flush=True)
    hi = HS.index(a.h)
    key = np.char.add(np.char.add(F["track"], "|"), np.char.add(F["scene"], np.char.add("|", F["t"].astype(str))))
    cover = {}
    for h, H in enumerate(HS):
        s = np.where(F["w"][:, h, 0] == 1, F["w"][:, h, 3], np.nan)
        b, npeer, kind = baseline(key, F["pol"], F["xy"], F["yaw"], F["v"], s)
        A = s - b
        cover[f"H{H:g}"] = coverage(F, tb, s, A, kind, F["w"][:, h, 1] == 1)
        if h == hi:
            pk = dict(s=s.astype(np.float32), b=b.astype(np.float32), A=A.astype(np.float32), n_peer=npeer, kind=kind,
                      fail=(F["w"][:, h, 1] == 1), prog=F["w"][:, h, 2].astype(np.float32), win=F["w"][:, h, 4].astype(np.float32))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{POOL}{'_lim' if a.limit else ''}.npz", names=names, pool=pre, dev=tb["dev"].astype(str), log=tb["log"].astype(str),
             track=F["track"], pol=F["pol"], scene=F["scene"], t=F["t"], k=F["k"], xy=F["xy"].astype(np.float32), yaw=F["yaw"].astype(np.float32),
             v=F["v"].astype(np.float32), target=F["target"], h=np.float32(a.h), **pk)
    report(cover, a)


def coverage(F, tb, s, A, kind, fail):
    dev = tb["dev"].astype(str) == "1"
    out = {}
    for trk in ("nu", "pai"):
        for part, m in (("train", ~dev), ("dev", dev)):
            sel = (F["track"] == trk) & m
            v = sel & np.isfinite(s)
            a = sel & np.isfinite(A)
            out[f"{trk}/{part}"] = dict(rows=int(sel.sum()), valid=int(v.sum()), fail=int((sel & fail).sum()), score_mean=float(np.nanmean(s[v])) if v.any() else None,
                                        A_def=int(a.sum()), matched=int((a & (kind == 1)).sum()), fallback=int((a & (kind == 2)).sum()),
                                        A_nonzero=int((a & (np.abs(A) > 1e-6)).sum()), A_pos=int((a & (A > 1e-6)).sum()), A_neg=int((a & (A < -1e-6)).sum()),
                                        A_pos_matched=int((a & (A > 1e-6) & (kind == 1)).sum()), A_ge05=int((a & (A >= 0.5)).sum()),
                                        A_ge05_matched=int((a & (A >= 0.5) & (kind == 1)).sum()), A_ge01_matched=int((a & (A >= 0.1) & (kind == 1)).sum()),
                                        A_le_m05=int((a & (A <= -0.5)).sum()),
                                        A_abs_mean=float(np.abs(A[a]).mean()) if a.any() else None,
                                        A_q=[float(q) for q in np.quantile(A[a], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])] if a.any() else None)
    return out


def report(cover, a):
    RES.mkdir(parents=True, exist_ok=True)
    L = [f"# AWR-ONPOL outcome score and advantage on {POOL} (pack H = {a.h:g} s{', LIMIT ' + str(a.limit) if a.limit else ''})", "",
         "Score: 0 on an at-fault window (collision front / lateral, offroad, left corridor laterally), else progress vs the log saturated at 0.8; "
         "baseline: other policies at the same (scene, time) with near-identical ego state (kind 1), else the per-(scene, time) mean (kind 2).", "",
         "| H | track / part | rows | valid | fail windows | mean score | A defined | matched | fallback | A != 0 | A > 0 | A > 0 matched | A >= 0.1 matched | A >= 0.5 | A >= 0.5 matched | A <= -0.5 | mean abs A |",
         "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for H, C in cover.items():
        for k, c in C.items():
            if not c["rows"]:
                continue
            L.append(f"| {H} | {k} | {c['rows']} | {c['valid']} | {c['fail']} | {(c['score_mean'] or 0):.3f} | {c['A_def']} | {c['matched']} | {c['fallback']} | "
                     f"{c['A_nonzero']} ({c['A_nonzero'] / max(c['A_def'], 1):.1%}) | {c['A_pos']} | {c['A_pos_matched']} | {c['A_ge01_matched']} | {c['A_ge05']} | "
                     f"{c['A_ge05_matched']} | {c['A_le_m05']} | {(c['A_abs_mean'] or 0):.3f} |")
    L += ["", "A quantiles (1, 5, 25, 50, 75, 95, 99 %) at the pack's H:"]
    for k, c in cover[f"H{a.h:g}"].items():
        if c["A_q"]:
            L.append(f"- {k}: " + ", ".join(f"{q:+.3f}" for q in c["A_q"]))
    tag = "_lim" if a.limit else ""
    (RES / f"score{tag}.md").write_text("\n".join(L) + "\n")
    (RES / f"score{tag}.json").write_text(json.dumps(cover, indent=1))
    print("\n".join(L), flush=True)


def cmd_check(a):
    """rear(mu) vs served poses on the default-serve pool (served = the model plan there; the dump's mu is fp16-rounded, ulp 0.016 - 0.031 m at
    16 - 64 m: line max < 0.05 m, median < 0.01 m), and the window scorer on synthetic series."""
    m = np.load(NU_ROWS / "L18RB0" / "meta.npz")
    tb = np.load(CR / "on1_L18RB0" / "tab.npz")
    pos = {x: i for i, x in enumerate(tb["names"].astype(str).tolist())}
    idx = np.array([pos.get(x, -1) for x in m["names"].astype(str).tolist()])
    j = np.flatnonzero(idx >= 0)[:4000]
    P = rear(m["mu"][j], tb["cam"][idx[j], 0])
    d = np.abs(P[..., :2] - m["poses"][j][..., :2].astype(np.float32)).max((1, 2))
    T = np.arange(0, 9.5, 0.5)
    z = np.zeros(len(T))
    S0 = {k: z.copy() for k in MET}
    S0["gt_dist_traveled_m"][:] = 100.0
    S0["progress"] = T / 10.0                                        # ego 10 m/s along a 100 m GT
    la = lambda w: 10.0 * w  # noqa: E731
    k1 = window(T, S0, 1.0, 3.0, la)                                 # log 10 m/s: p = 1
    k2 = window(T, S0, 1.0, 3.0, lambda w: 20.0 * w)                 # log 20 m/s: p = 0.5 / 0.8
    S1 = {k: v.copy() for k, v in S0.items()}
    S1["collision_any"][T == 3.0] = 1
    S1["collision_rear"][T == 3.0] = 1
    k3 = window(T, S1, 1.0, 3.0, la)                                 # rear only: truncated at 2 s, no fail
    S2 = {k: v.copy() for k, v in S1.items()}
    S2["collision_front"][T == 3.0] = 1
    k4 = window(T, S2, 1.0, 3.0, la)                                 # front: fail
    k5 = window(T, S2, 3.0, 3.0, la)                                 # at the event time: invalid
    ok = dict(rear_max_m=float(d.max()), rear_p99_m=float(np.quantile(d, 0.99)), rear_median_m=float(np.median(d)), rows=int(len(j)),
              k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)
    good = (d.max() < 0.05 and np.median(d) < 0.01 and abs(k1[3] - 1) < 1e-9 and abs(k2[3] - 0.625) < 1e-9 and k3[0] and not k3[1] and abs(k3[4] - 2.0) < 1e-9
            and k4[1] and k4[3] == 0 and not k5[0])
    print(json.dumps(dict(ok, pass_=bool(good)), indent=1, default=float))
    raise SystemExit(0 if good else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--h", type=float, default=3.0, choices=HS)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first S scenes per policy and track only (test run; pack *_lim)")
    sub.add_parser("check")
    a = ap.parse_args()
    dict(build=cmd_build, check=cmd_check)[a.cmd](a)
