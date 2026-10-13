#!/usr/bin/env python3
"""CRITIC-PILOT labels (experiments/route_follow/plans/2026-10-13-critic-pilot-prereg.md, "Labels"): the closed-loop consequence of every
stored on-policy decision, read from the same rollout's run dir. CPU only, read-only; AlpaSim venv (polars reads the Int128 series).

  pool   --name L18RB0 [...]      rows of pool on1_<name> (rows/<name>/meta.npz, aligned to <pool>/tab.npz)
  reads  --tag RBSG2-L18-s0 [...]  every decision of the stored reads runs/route_mem/read/<tag>/c*/run (drive.jsonl): inputs (ego, rmem,
                                   served poses) and the event / prog4 / wzpk / spin labels (no route label there, so no lat2)
      $DATA_DIR/third_party/alpasim/.venv/bin/python experiments/critic_pilot/scripts/cp_labels.py pool --name L18RB0 L18RSG L18RBSG L18RSGallheld

Out: $DATA_DIR/runs/critic_pilot/labels/<name>.npz (pool: names = tab names; reads: rd:<scene>:<k> plus the inputs) and labels.json (counts,
gate G0 = median |controller pose at t - driver anchor| <= 0.3 m per pool).
Columns: keep (row before the rollout's first zero event), ev (n, 3, 2) collision / offroad / corridor x T = 2, 4 s in {0, 1, nan},
anyz (n, 2), lat2 (signed m, left +, executed rear axle at t + 2 s against the route label polyline), prog4 (m), wzpk (rad/s), spin (0 / 1
/ nan), b_now / b_plan / b_log (hand-rule references), d_anchor (G0), log (log file = scene minus token).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
_R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_R), str(_R / "experiments/route_mem/lib")]
import numpy as np  # noqa: E402

TYPES, HOR = ("collision", "offroad", "corridor"), (2.0, 4.0)
SPIN_A = 6.0                                                    # m/s^2, rf_tax spin rule (3 consecutive 10 Hz samples)


def first_pos(ser, name):
    x = ser.get(name)
    if x is None or not len(x):
        return np.nan
    m = np.nan_to_num(x[:, 1]) > 0
    return float(x[m, 0].min()) if m.any() else np.nan


def series(run, scene, rid):
    import polars as pl
    f = glob.glob(f"{run}/rollouts/{scene}/{rid}/metrics.parquet")
    if not f:
        return None
    m = pl.read_parquet(f[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    return {n: np.array(sorted(zip(x["timestamps_us"].to_list(), [np.nan if v is None else v for v in x["values"].to_list()])), float).reshape(-1, 2)
            for (n,), x in m.group_by("name")}


def ctrl(run, rid):
    import csv
    f = Path(run) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists():
        return None
    rows = list(csv.reader(open(f)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    if len(A) < 2:
        return None
    c = lambda k: A[:, h[k]]  # noqa: E731
    qx, qy, qz, qw = c("qx"), c("qy"), c("qz"), c("qw")
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return dict(t=c("timestamp_us"), x=c("x"), y=c("y"), yaw=yaw, vx=c("vx"), wz=c("wz"))


def signed_off(P, Q):
    """Points Q (m, 2) -> signed lateral offset (left +) from polyline P (n, 2) (nearest segment)."""
    P = P[np.isfinite(P).all(1)]
    if len(P) < 2:
        return np.full(len(Q), np.nan)
    A, B = P[:-1], P[1:]
    d = B - A
    L2 = np.maximum((d ** 2).sum(1), 1e-9)
    out = np.empty(len(Q))
    for i, q in enumerate(Q):
        u = np.clip(((q - A) * d).sum(1) / L2, 0, 1)
        C = A + u[:, None] * d
        j = int(np.argmin(((q - C) ** 2).sum(1)))
        r = q - C[j]
        s = np.sign(d[j, 0] * r[1] - d[j, 1] * r[0])
        out[i] = s * np.hypot(*r)
    return out


def in_ego(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, float).reshape(-1, 2) - np.asarray(a[:2], float)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def roll_labels(job):
    """One rollout: run, scene, rid, flags, rows = list of (i, t_us, anchor (3), poly (51, 2) or None, served (8, 3)) -> list of (i, dict)."""
    run, scene, rid, flags, rows = job
    ser, C = series(run, scene, rid), ctrl(run, rid)
    if ser is None or C is None:
        return [(r[0], None) for r in rows]
    tf = {"collision": np.nanmin([first_pos(ser, "collision_front"), first_pos(ser, "collision_lateral")]),
          "offroad": first_pos(ser, "offroad"), "corridor": first_pos(ser, "left_corridor_laterally")}
    flag = {"collision": flags.get("collision_at_fault"), "offroad": flags.get("offroad"), "corridor": flags.get("left_corridor_laterally")}
    te = {k: (v if float(flag[k] or 0) > 0 and np.isfinite(v) else np.inf) for k, v in tf.items()}
    t_first = min(te.values())
    t_end = float(max(x[:, 0].max() for x in ser.values() if len(x)))
    ct, cmax = C["t"], C["t"][-1]
    av = np.abs(C["vx"] * C["wz"])
    out = []
    for i, t, anc, poly, served in rows:
        o = dict(keep=bool(t < t_first))
        ev = np.full((3, 2), np.nan)
        for a, k in enumerate(TYPES):
            for b, T in enumerate(HOR):
                e = te[k]
                ev[a, b] = 1.0 if t < e <= t + T * 1e6 else (np.nan if t + T * 1e6 > t_end else 0.0)
        anyz = np.where((ev == 1).any(0), 1.0, np.where(np.isnan(ev).any(0), np.nan, 0.0))
        o.update(ev=ev, anyz=anyz)
        p0 = np.array([np.interp(t, ct, C["x"]), np.interp(t, ct, C["y"])])
        o["d_anchor"] = float(np.hypot(*(p0 - np.asarray(anc[:2])))) if ct[0] <= t <= cmax else np.nan
        o["lat2"] = np.nan
        if poly is not None and t + 2e6 <= cmax:
            q = in_ego(anc, [[np.interp(t + 2e6, ct, C["x"]), np.interp(t + 2e6, ct, C["y"])]])
            o["lat2"] = float(signed_off(poly, q)[0])
            o["q2"] = q[0]
        if t + 4e6 <= cmax:
            tt = np.r_[t, ct[(ct > t) & (ct < t + 4e6)], t + 4e6]
            xy = np.c_[np.interp(tt, ct, C["x"]), np.interp(tt, ct, C["y"])]
            o["prog4"] = float(np.hypot(*np.diff(xy, axis=0).T).sum())
            w = (ct > t) & (ct <= t + 4e6)
            o["wzpk"] = float(np.abs(C["wz"][w]).max()) if w.any() else np.nan
            s = (av[w] >= SPIN_A).astype(int)
            o["spin"] = float(len(s) >= 3 and np.convolve(s, np.ones(3, int), "valid").max() >= 3)
        else:
            o["prog4"] = o["wzpk"] = np.nan
            w = (ct > t)
            s = (av[w] >= SPIN_A).astype(int)
            o["spin"] = 1.0 if len(s) >= 3 and np.convolve(s, np.ones(3, int), "valid").max() >= 3 else np.nan
        if poly is not None:
            o["b_now"] = float(abs(signed_off(poly, np.zeros((1, 2)))[0]))
            P = np.asarray(served, float)[:, :2]
            arc = np.cumsum(np.hypot(*np.diff(np.r_[np.zeros((1, 2)), P], axis=0).T))
            o["b_plan"] = float(abs(signed_off(poly, P[[int(np.argmin(np.abs(arc - 10.0)))]])[0]))
        out.append((i, o))
    return out


def summary_of(run):
    S = json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]
    return {r["clipgt_id"]: (r["rollout_id"], r.get("score_metrics") or {}, r.get("score")) for r in S}


def assemble(n, res):
    f = lambda k, shape=(): np.full((n, *shape), np.nan)  # noqa: E731
    L = dict(keep=np.zeros(n, bool), ev=f("ev", (3, 2)), anyz=f("anyz", (2,)), lat2=f("lat2"), q2=f("q2", (2,)), prog4=f("prog4"),
             wzpk=f("wzpk"), spin=f("spin"), b_now=f("b_now"), b_plan=f("b_plan"), d_anchor=f("d_anchor"), labelled=np.zeros(n, bool))
    for i, o in res:
        if o is None:
            continue
        L["labelled"][i] = True
        for k, v in o.items():
            L[k][i] = v
    return L


def counts(L, rows=None):
    k = L["keep"] & L["labelled"]
    c = dict(rows=int(len(k)), labelled=int(L["labelled"].sum()), keep=int(k.sum()),
             d_anchor_med=float(np.nanmedian(L["d_anchor"])), lat2_ok=int(np.isfinite(L["lat2"][k]).sum()))
    for a, t in enumerate(TYPES):
        for b, T in enumerate(HOR):
            c[f"{t}{int(T)}_pos"] = int((L["ev"][k, a, b] == 1).sum())
    for b, T in enumerate(HOR):
        c[f"any{int(T)}_pos"], c[f"any{int(T)}_n"] = int((L["anyz"][k, b] == 1).sum()), int(np.isfinite(L["anyz"][k, b]).sum())
    c["spin_pos"] = int((L["spin"][k] == 1).sum())
    return c


def main():
    from jevdrive import par
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pool", "reads"))
    ap.add_argument("--name", nargs="*", default=[])
    ap.add_argument("--tag", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rollouts only")
    a = ap.parse_args()
    D = data_dir()
    out = D / "runs/critic_pilot/labels"
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((out / "labels.json").read_text()) if (out / "labels.json").exists() else {}
    with Run("critic_pilot", f"labels-{a.mode}", config=vars(a)) as run:
        for name in (a.name if a.mode == "pool" else a.tag):
            if a.mode == "pool":
                M = np.load(D / f"runs/onpol_nuplan/rows/{name}/meta.npz")
                P = D / f"runs/op_parity/cache/on1_{name}"
                tb, rt = np.load(P / "tab.npz"), np.load(P / "route.npz")
                names = tb["names"]
                assert (M["names"] == names).all() and (rt["tokens"] == names).all(), f"{name}: row order differs"
                runs, scenes, t_us, anc, served = M["run"], M["scene"], M["t_us"], M["anchor"], tb["served"]
                poly = np.where(rt["ok"][:, None, None], rt["poly"], np.nan)
                extra = dict(e0=M["e0"])
            else:
                recs = []
                for rd in sorted(glob.glob(str(D / f"runs/route_mem/read/{name}/c*/run"))):
                    for line in open(Path(rd) / "driver-logs/drive.jsonl"):
                        r = json.loads(line)
                        if r.get("kind") == "drive" and "rmem" in r:
                            recs.append((rd, r))
                runs = np.array([rd for rd, _ in recs])
                scenes = np.array([r["scene"] for _, r in recs])
                t_us = np.array([int(r["t0"]) for _, r in recs])
                anc = np.array([r["anchor"] for _, r in recs], float)
                served = np.array([r["poses"] for _, r in recs], np.float32)
                names = np.array([f"rd:{r['scene']}:{r['k']}" for _, r in recs])
                poly = None
                extra = dict(ego=np.array([r["ego"] for _, r in recs], np.float32), rmem=np.array([r["rmem"] for _, r in recs], np.float32),
                             served=served, k=np.array([r["k"] for _, r in recs]), sess=np.array([r["session"] for _, r in recs]))
            n = len(names)
            sums = {rd: summary_of(rd) for rd in sorted(set(runs.tolist()))}
            jobs = {}
            for i in range(n):
                rid, flags, _ = sums[runs[i]].get(scenes[i], (None, {}, None))
                if rid is None:
                    continue
                jobs.setdefault((runs[i], scenes[i], rid), (flags, []))[1].append(
                    (i, int(t_us[i]), anc[i], None if poly is None else poly[i], served[i]))
            js = [(r, s, rid, fl, rows) for (r, s, rid), (fl, rows) in jobs.items()][: a.limit or None]
            run.info(f"{name}: {n} rows, {len(js)} rollouts")
            R = par.pmap(roll_labels, js, workers=a.workers, run=run, desc=name)
            run.info(f"{name}: {R.summary()}")
            res = [x for chunk in R.values if chunk for x in chunk]
            L = assemble(n, res)
            log = np.array([s.rsplit("-", 1)[0] for s in scenes])
            if a.mode == "pool":
                L["b_log"] = np.abs(extra.pop("e0"))
            np.savez(out / f"{name}.npz", names=names, log=log, scene=scenes, **L, **extra)
            c = counts(L)
            c["G0"] = bool(c["d_anchor_med"] <= 0.3)
            rep[name] = c
            run.info(f"{name}: {json.dumps(c)}")
        (out / "labels.json").write_text(json.dumps(rep, indent=1) + "\n")
        run.summary.update(rep)


if __name__ == "__main__":
    main()
