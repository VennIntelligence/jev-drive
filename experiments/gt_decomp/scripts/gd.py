#!/usr/bin/env python3
"""GT-DECOMP (plans/2026-10-12-gt-decomp-prereg.md): the leaderboard "Distance to GT Trajectory" split into along-track and cross-track parts,
our AlpaSim driver against the reference drivers, on the stored cap_audit rollouts. CPU only; run dirs are read, never written.

  collect (box, AlpaSim venv)   gd.py collect --set pub|held [--workers N] [--limit N]
      per scene: the evaluator's per-step series (metrics.parquet), its truncation and time aggregation re-applied (gate G1a), the GT path
      rebuilt from the scene loader and the ego box centre from the controller CSV (gate G1b), the along / cross split per decision and the
      timing counterfactual -> $DATA_DIR/runs/gt_decomp/collect-<set>/<ts>/ step.parquet, roll.parquet, runlevel.csv, gates.json
  report  (Mac, project venv)   gd.py report --pub DIR --held DIR [--scenes-pub DIR --scenes-held DIR --tags tags.csv]
      tables (results/gt_decomp.md + csv) and figures (figs/gd_split.png, figs/gd_lag.png)

Evaluator (AlpaSim 0bb4c4b, src/eval/src/eval): dist_to_gt_trajectory = shapely distance from the ego box centroid to the whole logged path
(box-centre frame), NOT time-aligned (scorers/ground_truth.py 59-60, 106-108); dist_to_gt_location = distance to the logged point at the same
time (99-105); lateral_dist_to_gt_trajectory = the same as the first except past the path end, where only the component perpendicular to the
last segment counts (20-40, 109-116). First two steps are 0. Aggregation: drop steps before eval_relevant, after the first collision_any or
offroad (aggregation/processing.py 37-64), after the first dist_to_gt_trajectory >= 4 m (aggregation/main.py 155-160; the event step stays,
modifiers.py 100-112); MAX over time for the distances, LAST for progress_rel_to_total; the run value is the plain mean over clips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "collect":            # one thread per worker process
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/cap_audit/scripts")]
T0, DT, CORR = 17_000, 500_000, 4.0
KSET = {"pub": 10, "held": 18}
NAMES = ("dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory", "dist_to_gt_location", "progress_rel_to_total", "eval_relevant",
         "collision_any", "collision_front", "collision_lateral", "offroad", "left_corridor_laterally", "gt_dist_traveled_m")
SCORE_MIN_GT, SCORE_SAT = 5.0, 0.8                          # eval-config.yaml scene_score (min_gt_distance_for_full_score_m, saturation)


# ---------------------------------------------------------------- evaluator re-implementation (per-step series -> rollout values)
def series(run, scene, rid):
    import polars as pl
    f = Path(run) / f"rollouts/{scene}/{rid}/metrics.parquet"
    m = pl.read_parquet(f).filter(pl.col("name").is_in(list(NAMES))).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    return {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}


def kept_steps(S, d=None):
    """Timestamps the evaluator keeps (processing.DEFAULT_MODIFIERS then main's >= 4 m rule). `d` overrides the distance series (counterfactual)."""
    g = lambda n, t: float(S.get(n, {}).get(t) or 0.0)                                       # noqa: E731
    ts = sorted(S["dist_to_gt_trajectory"])
    ev = [t for t in ts if g("eval_relevant", t) > 0]
    if ev:
        ts = [t for t in ts if t >= ev[0]]
    oc = next((t for t in ts if g("collision_any", t) > 0 or g("offroad", t) > 0), None)
    if oc is not None:
        ts = [t for t in ts if t <= oc]
    d = d if d is not None else S["dist_to_gt_trajectory"]
    c4 = next((t for t in ts if d[t] >= CORR), None)
    return [t for t in ts if t <= c4] if c4 is not None else ts


def rollout_values(S):
    ts = kept_steps(S)
    g = lambda n: np.array([float(S.get(n, {}).get(t) or 0.0) for t in ts])                   # noqa: E731
    d, lat = g("dist_to_gt_trajectory"), g("lateral_dist_to_gt_trajectory")
    i = int(d.argmax())
    out = dict(gt=float(d.max()), gt_lat_max=float(lat.max()), gt_lat=float(lat[i]), gt_over=float(d[i] - lat[i]), k_star=int((ts[i] - T0) // DT),
               loc=float(g("dist_to_gt_location").max()), prog=float(g("progress_rel_to_total")[-1]), gtlen=float(g("gt_dist_traveled_m")[-1]),
               caf=float(((g("collision_front") > 0) | (g("collision_lateral") > 0)).max()), off=float(g("offroad").max()),
               lcl=float(g("left_corridor_laterally").max()), n_kept=len(ts), t_last=int(ts[-1]), over_any=bool((d - lat > 1e-6).any()))
    ps = 1.0 if out["gtlen"] < SCORE_MIN_GT else min(min(max(out["prog"], 0.0), 1.0) / SCORE_SAT, 1.0)
    out["prog_score"] = ps
    out["score_re"] = 0.0 if (out["caf"] or out["off"] or out["lcl"]) else ps
    return out, ts


# ---------------------------------------------------------------- path frame
class Path2:
    """The logged path as the evaluator's linestring (box-centre frame) with arc-length helpers."""

    def __init__(self, G, T):
        import shapely
        self.G, self.T = G, T
        self.ls = shapely.LineString(G)
        self.L = self.ls.length
        self.S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(G, axis=0).T))]
        e = G[-1] - G[-2]
        self.ue = e / np.hypot(*e)
        b = G[1] - G[0]
        self.ub = b / max(np.hypot(*b), 1e-9)

    def at_time(self, t):
        return np.array([np.interp(t, self.T, self.G[:, 0]), np.interp(t, self.T, self.G[:, 1])])

    def frame(self, p):
        """Point -> (station along the path extended linearly past both ends, signed cross-track (left +), evaluator distance)."""
        import shapely
        pt = shapely.Point(*p)
        d = float(self.ls.distance(pt))
        s = float(self.ls.project(pt))
        if s >= self.L - 1e-9:
            v = p - self.G[-1]
            return self.L + float(v @ self.ue), float(self.ue[0] * v[1] - self.ue[1] * v[0]), d
        if s <= 1e-9:
            v = p - self.G[0]
            return float(v @ self.ub), float(self.ub[0] * v[1] - self.ub[1] * v[0]), d
        q0, q1 = self.ls.interpolate(max(s - 0.05, 0.0)), self.ls.interpolate(min(s + 0.05, self.L))
        tx, ty = q1.x - q0.x, q1.y - q0.y
        q = self.ls.interpolate(s)
        sg = np.sign(tx * (p[1] - q.y) - ty * (p[0] - q.x)) or 1.0
        return s, float(sg * d), d


def scene_job(job):
    scene, cfg, K, labs = job
    import ca
    from ao_e1 import loader
    ds = loader(cfg).get_data_source(scene)
    tr, vc = ds.rig.trajectory, ds.rig.vehicle_config
    Tg, Pg, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
    dx = float(vc.aabb_x_offset_m + vc.aabb_x_m / 2)                                        # rig -> box centre (x only; y offset 0)
    cen = lambda x, y, yaw: np.array([x + np.cos(yaw) * dx, y + np.sin(yaw) * dx])           # noqa: E731
    P = Path2(np.c_[Pg[:, 0] + np.cos(Yg) * dx, Pg[:, 1] + np.sin(Yg) * dx], Tg)
    tk = T0 + DT * np.arange(K + 1)
    hum = [P.frame(P.at_time(t)) for t in tk]                                              # the human at t_k: station, cross (~0)
    steps, rolls = [], []
    for lab, d in labs.items():
        if not d["rid"]:
            continue
        S = series(d["run"], scene, d["rid"])
        C = ca.ctrl(d["run"], d["rid"])
        rv, kept = rollout_values(S)
        st = d["metrics"]
        r = dict(label=lab, scene=scene, run=d["run"], score=d["score"], **rv,
                 st_gt=st.get("dist_to_gt_trajectory"), st_lat=st.get("lateral_dist_to_gt_trajectory"), st_loc=st.get("dist_to_gt_location"),
                 st_prog=st.get("progress_rel_to_total"), has_ctrl=C is not None)
        keptset = set(kept)
        oc_cut = max(kept)                                                                # collision / offroad / 4 m cut of the real rollout
        g = lambda n, t: float(S.get(n, {}).get(int(t)) or 0.0)                              # noqa: E731
        hard = next((t for t in sorted(S["dist_to_gt_trajectory"]) if g("collision_any", t) > 0 or g("offroad", t) > 0), None)
        if C is not None:
            # ego centre on the 10 Hz controller grid up to the hard cut: station / cross profile for the counterfactual
            tc = np.array(sorted(t for t in C if t <= (hard if hard is not None else tk[-1])))
            E = np.array([P.frame(cen(*C[t][:3])) for t in tc]) if len(tc) else np.zeros((0, 3))
            sr = np.maximum.accumulate(E[:, 0]) if len(E) else E[:, 0]
            for k in range(1, K + 1):
                t = int(tk[k])
                p = C.get(t)
                if p is None:
                    continue
                c = cen(*p[:3])
                s_e, x_e, d_g = P.frame(c)
                e_g = float(np.hypot(*(c - P.at_time(t))))
                s_h = hum[k][0]
                steps.append(dict(label=lab, scene=scene, k=k, kept=t in keptset, along=s_e - s_h, cross=x_e, d_geo=d_g, e_geo=e_g,
                                  d_ev=S["dist_to_gt_trajectory"].get(t), l_ev=S["lateral_dist_to_gt_trajectory"].get(t),
                                  e_ev=S["dist_to_gt_location"].get(t), s_h=s_h, s_e=s_e))
            # counterfactual: our path, the human's timing (station s_h(t_k)), cross-track as we had it at that station
            dcf, ext = {}, {}
            for k in range(1, K + 1):
                t = int(tk[k])
                if t not in S["dist_to_gt_trajectory"]:
                    continue
                if k == 1 or not len(E):                                                   # the evaluator's first scored step is 0
                    dcf[t], ext[t] = 0.0, False
                    continue
                s_h = hum[k][0]
                j = int(np.searchsorted(sr, s_h))
                if j >= len(sr):
                    dcf[t], ext[t] = abs(float(E[-1, 1])), True
                elif j == 0:
                    dcf[t], ext[t] = abs(float(E[0, 1])), False
                else:
                    w = (s_h - sr[j - 1]) / max(sr[j] - sr[j - 1], 1e-9)
                    dcf[t], ext[t] = abs(float((1 - w) * E[j - 1, 1] + w * E[j, 1])), False
            full = {t: dcf.get(t, 0.0) for t in S["dist_to_gt_trajectory"]}
            kc = [t for t in kept_steps(S, full) if t <= (hard if hard is not None else 10 ** 12)]
            kc_lb = [t for t in kc if not ext.get(t, False)]
            r |= dict(gt_cf=max(full[t] for t in kc), gt_cf_lb=max([full[t] for t in kc_lb] or [0.0]),
                      cf_ext=sum(ext.get(t, False) for t in kc), cf_n=len(kc))
            # the geometric reproduction of the rollout value (k = 1 forced to 0 as the evaluator does)
            dg = {int(tk[k]): (0.0 if k == 1 else P.frame(cen(*C[int(tk[k])][:3]))[2]) for k in range(1, K + 1) if int(tk[k]) in C}
            r["gt_geo"] = max([dg.get(t, 0.0) for t in kept] or [0.0])
        rolls.append(r)
    return dict(step=steps, roll=rolls, scene=dict(scene=scene, L=P.L, s_h_end=hum[-1][0], gt_end_t=float(Tg[-1])))


def runlevel(D):
    """Every run dir: the mean of its per-rollout metrics against its own metrics_results (gate G1a, run level)."""
    rows = []
    for lab, ds in D.items():
        for d in ds:
            J = json.loads((Path(d) / "aggregate/results-summary.json").read_text())
            mr = (J.get("metrics_results") or [{}])[0]
            for n in ("dist_to_gt_trajectory", "dist_to_gt_location", "lateral_dist_to_gt_trajectory", "progress_rel_to_total"):
                v = [r["metrics"][n] for r in J["rollouts"] if (r.get("metrics") or {}).get(n) is not None]
                if mr.get(n) is not None and v:
                    rows.append(dict(label=lab, run=d, metric=n, n=len(v), mean=float(np.mean(v)), stored=float(mr[n]), diff=float(np.mean(v) - mr[n])))
    return rows


def collect(a):
    import ca
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    K = KSET[a.set]
    keep = ca.held_scenes() if a.set == "held" else None
    D = ca.run_dirs(a.set)
    with Run("gt_decomp", f"collect-{a.set}", config=vars(a)) as run:
        if a.set == "held":
            from jevdrive.data import splits
            run.use_split(splits.load("alpasim_nuplan/onpol-held"))
        Pk = ca.pick_rollouts(D, keep)
        jobs = {}
        for lab, v in Pk.items():
            for s, x in v.items():
                jobs.setdefault(s, {})[lab] = x
        cfg = str(Path(D["YR10m10-F-s0"][0]) / "generated-user-config-0.yaml")
        units = [(s, cfg, K, v) for s, v in sorted(jobs.items()) if any(x["rid"] for x in v.values())][: a.limit or None]
        workers = a.workers or min(64, n_cpus())
        run.info(f"{a.set}: {len(units)} scenes x {len(Pk)} labels, K={K}, {workers} workers")
        res = par.pmap(scene_job, units, workers=workers, run=run, desc="scenes")
        out = Path(run.dir)
        err = [dict(scene=units[i][0], err=res.errors[i].strip().splitlines()[-1][:300]) for i in sorted(res.errors)]
        pd.DataFrame(err, columns=["scene", "err"]).to_csv(out / "errors.csv", index=False)
        vals = [v for v in res.values if v]
        St = pd.DataFrame([x for v in vals for x in v["step"]])
        Ro = pd.DataFrame([x for v in vals for x in v["roll"]])
        St.to_parquet(out / "step.parquet", index=False)
        Ro.to_parquet(out / "roll.parquet", index=False)
        pd.DataFrame([v["scene"] for v in vals]).to_parquet(out / "pathinfo.parquet", index=False)
        RL = pd.DataFrame(runlevel(D))
        RL.to_csv(out / "runlevel.csv", index=False)
        G = gates(Ro, St, RL)
        G["scene_errors"] = len(err)
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.summary.update(scenes=len(units), errors=len(err), passed=G["passed"])
        run.info(json.dumps(G, default=float))


def gates(Ro, St, RL):
    G = {}
    ok = lambda x, y: (np.abs(x.astype(float) - y.astype(float)) <= 1e-3)                    # noqa: E731
    per = {}
    for lab, x in Ro.groupby("label"):
        x = x[x.st_gt.notna()]
        per[lab] = dict(n=len(x), gt=float(ok(x.gt, x.st_gt).mean()), lat=float(ok(x.gt_lat_max, x.st_lat).mean()),
                        loc=float(ok(x["loc"], x.st_loc).mean()), prog=float(ok(x.prog, x.st_prog).mean()),
                        score=float(ok(x.score_re, x.score.astype(float)).mean()))
    G["G1a_rollout"] = dict(per_label=per, line=0.999, passed=bool(all(min(v[m] for m in ("gt", "lat", "loc", "prog", "score")) >= 0.999 for v in per.values())))
    G["G1a_run"] = dict(n=len(RL), max_abs=float(RL["diff"].abs().max()), passed=bool((RL["diff"].abs() <= 1e-3).all()))
    k = St[St.kept & (St.k >= 2) & St.d_ev.notna()]
    pg = {lab: dict(n=len(x), d=float(ok(x.d_geo, x.d_ev).mean()), e=float(ok(x.e_geo, x.e_ev).mean())) for lab, x in k.groupby("label")}
    G["G1b_geometry"] = dict(per_label=pg, line=0.99, passed=bool(all(min(v["d"], v["e"]) >= 0.99 for v in pg.values())))
    r = Ro[Ro.gt_geo.notna()]
    G["gt_geo_match"] = float(ok(r.gt_geo, r.gt).mean())
    G["passed"] = bool(G["G1a_rollout"]["passed"] and G["G1a_run"]["passed"] and G["G1b_geometry"]["passed"])
    return G


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--set", choices=("pub", "held"), required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--pub", required=True)
    p.add_argument("--held", required=True)
    p.add_argument("--cap-pub", required=True)
    p.add_argument("--cap-held", required=True)
    p.add_argument("--tags", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a)
    else:
        from gd_report import report
        report(a)
