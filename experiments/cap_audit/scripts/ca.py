#!/usr/bin/env python3
"""CAP-AUDIT (plans/2026-10-12-cap-audit-prereg.md): per-decision capability measures of our AlpaSim driver and the reference drivers on
every stored decision, against the human log and the privileged scene. CPU only; run dirs are read, never written.

  probe   (box, any python3)        ca.py probe                       label -> run dirs, rollouts, parquet / controller coverage
  tags    (box, any python3)        ca.py tags                        nuPlan scenario tags per public scene -> results/tags.csv
  collect (box, AlpaSim venv)       ca.py collect --set pub|held [--workers N] [--limit N]
      per scene through the run's own scene loader: logged path, lanes, logged traffic; per label: controller pose / speed at every
      decision, the evaluator's 2 Hz series, our served plans -> $DATA_DIR/runs/cap_audit/collect-<set>/<ts>/ dec.parquet, log.parquet,
      roll.parquet, scene.parquet, gates.json, errors.csv
  percep  (box, AlpaSim venv)       ca.py percep                      road_probe's held P0-head errors per decision -> results/percep_held.csv.gz
  bev     (box, AlpaSim venv)       ca.py bev --collect DIR --picks picks.json      BEV payloads -> <collect dir>/bev.pkl

Frames: rollout local frame; ego = rear axle, x forward, y left. Decision k at t_k = 17 000 + 500 000 k us; the controller CSV holds the ego
rear-axle pose and body velocity at 10 Hz (aud1_drift_collect.py checked it against the driver's anchor and the asl EGO pose).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("collect", "bev", "percep"):   # one thread per worker process (zeros_mech b3cfe86d)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/zeros_mech/scripts")]
import zm_read as ZM  # noqa: E402  (zeros_mech helpers, unedited: Lanes, objects, obj_state, rel, frames)
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUNS = DATA / "runs"
OUT = REPO / "experiments/cap_audit/results"
OUT.mkdir(parents=True, exist_ok=True)
T0, DT = 17_000, 500_000
KSET = {"pub": 10, "held": 18}
OURS = ("YR10m10-F-s0", "APY10m10-AB-s0", "YS2G-L18-s0", "YS2G-L18-s1", "YS-L18-s0", "YS-L18-s1")
LEAD_GAP, LEAD_Y, LEAD_PSI, THW_CAP = 30.0, 1.5, np.radians(30.0), 4.0
GAP_R, GAP_CAP = 40.0, 5.0


# ---------------------------------------------------------------- labels -> run dirs
def run_dirs(st):
    if st == "pub":
        ra = RUNS / "alpasim"
        o3, cf, c0c = (json.loads((ra / f).read_text()) for f in ("ot3/a/manifest.json", "cf1/a/manifest.json", "c0c/manifest.json"))
        d = {k: o3.get(k, []) + o3.get(k + "@new", []) + cf.get(k + "@new", []) + cf.get(k + "@fresh", []) for k in ("YR10m10-F-s0", "APY10m10-AB-s0")}
        for t in ("YS2G-L18-s0", "YS2G-L18-s1", "YS-L18-s0", "YS-L18-s1"):
            d[t] = sorted(glob.glob(str(RUNS / f"onpol_long/read10/{t}/*")))
        d["WA-JEPA"] = c0c["WA-JEPA (reference)"]
        an = RUNS / "alpasim_proxy/anchors"
        g = lambda *p: sorted(x for q in p for x in glob.glob(str(an / q)))                     # noqa: E731
        d |= dict(LTF=g("ltf/runs/LTF-*/2*", "ltf2/runs/LTF-*/2*"), DD=g("dd/runs/DD-*/2*"), G0014=g("garage/runs/G0014-*/2*", "garage2/runs/G0014-*/2*"))
        return d
    al, ol = RUNS / "alpasim_long", RUNS / "onpol_long/read"
    d = {"YR10m10-F-s0": [al / "held/h18/YR10m10-F-s0"], "APY10m10-AB-s0": [al / "held/a18/APY10m10-AB-s0"]}
    d |= {t: [ol / t / "run"] for t in ("YS2G-L18-s0", "YS2G-L18-s1", "YS-L18-s0", "YS-L18-s1")}
    d |= {"LTF": [al / "fam/ltf"], "DD": [al / "fam/dd"], "G0014": [al / "fam/garage"], "WA-JEPA": [al / f"fam/wajepa{i}" for i in range(4)]}
    return {k: [str(x) for x in v] for k, v in d.items()}


def pick_rollouts(dirs, keep=None):
    """label -> {scene: dict(run, rid, score, metrics)}; first run dir that holds a scene wins (aud1_collect.py's rule)."""
    out = {}
    for lab, ds in dirs.items():
        seen = {}
        for d in ds:
            for r in json.loads((Path(d) / "aggregate/results-summary.json").read_text())["rollouts"]:
                s = r["clipgt_id"]
                if s in seen or (keep is not None and s not in keep):
                    continue
                rid = r.get("rollout_id")
                if not rid or not (Path(d) / f"rollouts/{s}/{rid}/metrics.parquet").exists():
                    f = glob.glob(f"{d}/rollouts/{s}/*/metrics.parquet")
                    rid = f[0].split("/")[-2] if f else None
                seen[s] = dict(run=d, rid=rid, score=r.get("score"), metrics=r.get("metrics") or {})
        out[lab] = seen
    return out


def probe(a):
    for st in ("pub", "held"):
        keep = held_scenes() if st == "held" else None
        D = run_dirs(st)
        P = pick_rollouts(D, keep)
        for lab, v in P.items():
            par_ = sum(1 for x in v.values() if x["rid"])
            ctl = sum(1 for x in v.values() if x["rid"] and (Path(x["run"]) / f"controller/alpasim_controller_{x['rid']}.csv").exists())
            drv = sum(1 for d in D[lab] if (Path(d) / "driver-logs/drive.jsonl").exists())
            sc = sum(1 for x in v.values() if x["score"] is not None)
            print(f"{st:5s} {lab:16s} dirs {len(D[lab]):2d} (drive.jsonl {drv:2d})  rollouts {len(v):4d}  scored {sc:4d}  parquet {par_:4d}  controller {ctl:4d}")


def held_scenes():
    from jevdrive.data import splits
    return set(splits.load("alpasim_nuplan/onpol-held").members)


# ---------------------------------------------------------------- scenario tags (nuPlan raw logs)
def tags(a):
    import pandas as pd
    cfgd = DATA / "datasets/alpasim_nuplan/navtest/configs"
    raw = DATA / "datasets/nuplan_raw/test"
    rows, dbs = [], {}
    for f in sorted(cfgd.glob("*.yaml")):
        s = f.stem
        log, tok = s.rsplit("-", 1)
        db = dbs.setdefault(log, sqlite3.connect(f"file:{raw / (log + '.db')}?mode=ro", uri=True))
        t = db.execute("select timestamp from lidar_pc where token=?", (bytes.fromhex(tok),)).fetchone()
        if t is None:
            rows.append(dict(scene=s, log=log, found=False, tags="", central=""))
            continue
        w = db.execute("select distinct st.type from scenario_tag st join lidar_pc lp on st.lidar_pc_token = lp.token "
                       "where lp.timestamp between ? and ?", (t[0] - 1_500_000, t[0] + 4_000_000)).fetchall()
        c = db.execute("select distinct type from scenario_tag where lidar_pc_token=?", (bytes.fromhex(tok),)).fetchall()
        rows.append(dict(scene=s, log=log, found=True, tags="|".join(sorted(x[0] for x in w)), central="|".join(sorted(x[0] for x in c))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tags.csv", index=False)
    g = dict(scenes=len(T), found=int(T.found.sum()), tagged=int((T.tags != "").sum()))
    g["pass"] = bool(g["tagged"] >= 0.95 * g["scenes"])
    (OUT / "gate_tag.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g))


# ---------------------------------------------------------------- per-scene geometry (AlpaSim venv)
def lane_geom(L, x, y, z, psi, hull):
    """Point / heading / ego polygon against the vector map -> (class code as zm_read Lanes.cls, junction flag, signed offset from the nearest
    same-direction centre line (left +, NaN if none), heading minus that centre line (deg), margin = polygon distance to the boundary of the
    union of nearby lane polygons (0 if the union does not contain it; the evaluator's min_distance_to_lane_boundary_m))."""
    import shapely
    pt = shapely.Point(float(x), float(y))
    yaws, polys, best = [], [], None
    for ln in L.vm.get_lanes_within(np.array([x, y, z], float), ZM.LANE_R):
        poly, cl = L.lane(ln)
        polys.append(poly)
        if not poly.contains(pt):
            continue
        yw = float(L.yaw(cl, pt))
        yaws.append(yw)
        d = float(ZM.wrap(yw - psi))
        if abs(d) < ZM.SAME:
            c = cl.project(pt)
            p0 = np.asarray(cl.interpolate(c).coords[0][:2])
            p1 = np.asarray(cl.interpolate(min(c + 0.5, cl.length)).coords[0][:2])
            if np.hypot(*(p1 - p0)) < 1e-3:
                p1, p0 = p0, np.asarray(cl.interpolate(max(c - 0.5, 0)).coords[0][:2])
            t = p1 - p0
            v = np.array([x, y]) - np.asarray(cl.interpolate(c).coords[0][:2])
            off = float(np.sign(t[0] * v[1] - t[1] * v[0]) * np.hypot(*v))
            if best is None or abs(off) < abs(best[0]):
                best = (off, float(np.degrees(d)))
    if yaws:
        yw = np.array(yaws)
        dd = np.abs(ZM.wrap(yw - psi))
        cls = 1 if (dd < ZM.SAME).any() else 2 if (dd > ZM.ONC).any() else 3
        junc = bool(len(yw) > 1 and np.abs(ZM.wrap(yw[:, None] - yw[None])).max() > ZM.JUNC)
    else:
        cls, junc = 0, False
    U = shapely.union_all(polys) if polys else None
    margin = float(hull.distance(U.boundary)) if U is not None and U.contains(hull) else 0.0
    return cls, junc, (best[0] if best else np.nan), (best[1] if best else np.nan), margin


def traffic(st, a, hull, front):
    """Logged traffic at one instant against an ego pose -> (min polygon distance to any vehicle (m, capped), its centre in the ego frame,
    gap to the nearest in-lane lead (bumper to its rearmost corner; NaN if none))."""
    if not len(st["id"]):
        return GAP_CAP, np.nan, np.nan, np.nan
    E, dpsi, gap = ZM.rel(st, a, front)
    near = np.flatnonzero(np.hypot(E[:, 0], E[:, 1]) < GAP_R)
    gmin, gx, gy = GAP_CAP, np.nan, np.nan
    if len(near):
        dist = np.array([hull.distance(ZM.obj_poly(st, i)) for i in near])
        j = int(dist.argmin())
        gmin, gx, gy = min(float(dist[j]), GAP_CAP), float(E[near[j], 0]), float(E[near[j], 1])
    box = (gap > 0) & (gap <= LEAD_GAP) & (np.abs(E[:, 1]) < LEAD_Y) & (np.abs(dpsi) < LEAD_PSI)
    return gmin, gx, gy, (float(gap[box].min()) if box.any() else np.nan)


SER = ZM.SER + ("min_distance_to_obstacle_m", "min_distance_to_lane_boundary_m")


def series(run, scene, rid):
    """The evaluator's 2 Hz series of one rollout (zm_read.series with two more metrics) -> {name: {t_us: value}} or None."""
    import polars as pl
    f = Path(run) / f"rollouts/{scene}/{rid}/metrics.parquet"
    if not f.exists():
        return None
    m = pl.read_parquet(f).filter(pl.col("name").is_in(list(SER))).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    return {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}


def ctrl(run, rid):
    f = Path(run) / f"controller/alpasim_controller_{rid}.csv"
    if not f.exists():
        return None
    c = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 2, 6, 7, 8, 9), ndmin=2)
    return {int(r[0]): (r[1], r[2], 2 * np.arctan2(r[4], r[3]), float(np.hypot(r[5], r[6]))) for r in c}


def plan_at(P, a, dist):
    """Served plan (8 poses in the anchor frame) -> the point `dist` m of arc ahead in the local frame, and whether the plan is shorter."""
    X = np.vstack([a[:2], ZM.to_local(a, np.asarray(P, float)[:, :2])])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(X, axis=0).T))]
    if s[-1] < dist:
        return X[-1], True
    return np.array([np.interp(dist, s, X[:, 0]), np.interp(dist, s, X[:, 1])]), False


def scene_job(job):
    scene, cfg, K, labs = job
    from ao_e1 import loader, project, stations
    ds = loader(cfg).get_data_source(scene)
    tr = ds.rig.trajectory
    Tg, Pg, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
    vc = ds.rig.vehicle_config
    rear, front, hw = float(vc.aabb_x_offset_m), float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2
    loc = str(getattr(ds.scene, "location", "") or "")
    s_onc = -1 if any(x in loc.lower() for x in ZM.LHT) else 1
    Lp = ZM.dedupe(np.vstack([Pg[:, :2], Pg[-1, :2] + ZM.EXT * np.array([np.cos(Yg[-1]), np.sin(Yg[-1])])]))
    SL = stations(Lp)
    Sg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Pg[:, :2], axis=0).T))]
    vg = np.gradient(Sg, Tg * 1e-6) if len(Tg) > 1 else np.zeros(len(Tg))
    zt = lambda t: float(np.interp(t, Tg, Pg[:, 2]))                                    # noqa: E731
    vlog = lambda t: float(np.interp(t, Tg, vg)) if t <= Tg[-1] else np.nan              # noqa: E731
    path_yaw = lambda s: float(np.arctan2(*(np.diff(ZM.at_station(Lp, SL, [s - 0.5, s + 0.5]), axis=0)[0][::-1])))  # noqa: E731
    objs = ZM.objects(ds)
    tk = T0 + DT * np.arange(K + 1)
    states = [ZM.obj_state(objs, t) for t in tk]
    L = ZM.Lanes(ds.map)
    logr = []
    for k, t in enumerate(tk):
        a = np.array([np.interp(t, Tg, Pg[:, 0]), np.interp(t, Tg, Pg[:, 1]), np.interp(t, Tg, Yg)])
        hull = ZM.ego_poly(a, rear, front, hw)
        cls, junc, off, dh, mg = lane_geom(L, a[0], a[1], zt(t), a[2], hull)
        gmin, gx, gy, lgap = traffic(states[k], a, hull, front)
        v = vlog(t)
        logr.append(dict(scene=scene, k=k, cls=cls, junc=junc, lane_off=off, lane_dh=dh, margin=mg, gap=gmin, gap_x=gx, gap_y=gy, lead_gap=lgap,
                         v=v, v1=vlog(t + 1e6), s=float(np.interp(t, Tg, Sg))))
    sc = dict(scene=scene, loc=loc, s_onc=s_onc, K=K, t_end_log=float(Tg[-1]), v0=vlog(tk[0]),
              dpsi_log=float(np.degrees(np.interp(tk[-1], Tg, Yg) - np.interp(tk[0], Tg, Yg))),
              dpsi_log10=float(np.degrees(np.interp(tk[min(10, K)], Tg, Yg) - np.interp(tk[0], Tg, Yg))))
    dec, roll = [], []
    g = dict(pose_need=0, pose_have=0, lat_n=0, lat_ok=0, anchor_n=0, anchor_ok=0, gap_pairs=[])
    for lab, d in labs.items():
        if not d["rid"]:
            roll.append(dict(label=lab, scene=scene, has=False))
            continue
        ser = series(d["run"], scene, d["rid"])
        if ser is None:
            roll.append(dict(label=lab, scene=scene, has=False))
            continue
        C = ctrl(d["run"], d["rid"])
        at = lambda n, t: ser.get(n, {}).get(int(t))                             # noqa: E731
        ks = [k for k in range(1, K + 1) if at("dist_to_gt_trajectory", tk[k]) is not None]
        trunc = next((k for k in ks if (at("dist_to_gt_trajectory", tk[k]) or 0) >= 4.0 or (at("collision_any", tk[k]) or 0) > 0
                      or (at("offroad", tk[k]) or 0) > 0), ks[-1] if ks else 0)
        recs = {int((r["t0"] - T0) // DT): r for r in d.get("recs") or []}
        hint = None
        for k in ks:
            t = int(tk[k])
            kept = k <= trunc
            g["pose_need"] += kept
            p = C.get(t) if C else None
            if p is None:
                continue
            g["pose_have"] += kept
            a = np.array(p[:3], float)
            hull = ZM.ego_poly(a, rear, front, hw)
            lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30, hint + 60)
            s_e, lat = project(Lp, SL, a[:2], lo, hi)
            hint = s_e
            cls, junc, off, dh, mg = lane_geom(L, a[0], a[1], zt(t), a[2], hull)
            gmin, gx, gy, lgap = traffic(states[k], a, hull, front)
            p1 = C.get(t + 1_000_000)
            le = at("lateral_dist_to_gt_trajectory", t)
            row = dict(label=lab, scene=scene, k=k, kept=kept, x=a[0], y=a[1], yaw=a[2], v=p[3], v1=p1[3] if p1 else np.nan, s=s_e, lat=lat,
                       head_path=float(np.degrees(ZM.wrap(a[2] - path_yaw(s_e)))), cls=cls, junc=junc, lane_off=off, lane_dh=dh, margin=mg,
                       gap=gmin, gap_x=gx, gap_y=gy, lead_gap=lgap, lat_eval=le, dist_gt=at("dist_to_gt_trajectory", t),
                       offroad=at("offroad", t), coll=at("collision_any", t), wrong_lane=at("wrong_lane", t),
                       gap_eval=at("min_distance_to_obstacle_m", t), margin_eval=at("min_distance_to_lane_boundary_m", t))
            if le is not None and kept:
                g["lat_n"] += 1
                g["lat_ok"] += abs(abs(lat) - le) <= 0.10
            if row["gap_eval"] is not None and kept and len(g["gap_pairs"]) < 400:
                g["gap_pairs"].append((gmin, min(float(row["gap_eval"]), GAP_CAP)))
            r = recs.get(k)
            if r is not None:
                an = np.asarray(r["anchor"], float)
                g["anchor_n"] += 1
                g["anchor_ok"] += float(np.hypot(*(an[:2] - a[:2]))) <= 0.05
                P = np.asarray(r["poses"], float)
                q1 = ZM.to_local(an, P[1:2, :2])[0]
                q10, short = plan_at(P, an, 10.0)
                row |= dict(cmd=int(r["cmd"]), plan_exec=float(np.hypot(*(q1 - np.array(p1[:2])))) if p1 else np.nan,
                            plan_off10=project(Lp, SL, q10, s_e - 10, s_e + 80)[1], plan_short10=short)
            dec.append(row)
        roll.append(dict(label=lab, scene=scene, has=True, run=d["run"], rid=d["rid"], score=d["score"], k_trunc=trunc, n_dec=len(ks),
                         has_ctrl=C is not None, **{f: float(d["metrics"].get(f) or 0) for f in ZM.FLAGS},
                         dist_to_gt=d["metrics"].get("dist_to_gt_trajectory")))
    return dict(dec=dec, log=logr, scene=sc, roll=roll, gate=g)


def collect(a):
    import pandas as pd
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    K = KSET[a.set]
    keep = held_scenes() if a.set == "held" else None
    D = run_dirs(a.set)
    with Run("cap_audit", f"collect-{a.set}", config=vars(a)) as run:
        if a.set == "held":
            from jevdrive.data import splits
            run.use_split(splits.load("alpasim_nuplan/onpol-held"))
        P = pick_rollouts(D, keep)
        jobs = {}
        for lab, v in P.items():
            recs = {}
            if lab in OURS:
                for d in D[lab]:
                    if (Path(d) / "driver-logs/drive.jsonl").exists():
                        for s, rr in drive_records(d).items():
                            recs.setdefault((d, s), rr)
            for s, x in v.items():
                jobs.setdefault(s, {})[lab] = dict(x, recs=recs.get((x["run"], s), []))
        cfg = str(Path(D["YR10m10-F-s0"][0]) / "generated-user-config-0.yaml")
        units = [(s, cfg, K, v) for s, v in sorted(jobs.items()) if any(x["rid"] for x in v.values())][: a.limit or None]
        workers = a.workers or min(64, n_cpus())
        run.info(f"{a.set}: {len(units)} scenes x {len(P)} labels, K={K}, {workers} workers, scene config {cfg}")
        res = par.pmap(scene_job, units, workers=workers, run=run, desc="scenes")
        out = Path(run.dir)
        err = [dict(scene=units[i][0], err=res.errors[i].strip().splitlines()[-1][:300]) for i in sorted(res.errors)]
        pd.DataFrame(err, columns=["scene", "err"]).to_csv(out / "errors.csv", index=False)
        vals = [v for v in res.values if v]
        for name in ("dec", "log", "roll"):
            pd.DataFrame([x for v in vals for x in v[name]]).to_parquet(out / f"{name}.parquet", index=False)
        pd.DataFrame([v["scene"] for v in vals]).to_parquet(out / "scene.parquet", index=False)
        from scipy.stats import spearmanr
        R = pd.DataFrame([x for v in vals for x in v["roll"]])
        Lg = pd.DataFrame([x for v in vals for x in v["log"]])
        S = lambda f: sum(v["gate"][f] for v in vals)                                   # noqa: E731
        gp = np.array([p for v in vals for p in v["gate"]["gap_pairs"]])
        nmin = 1470 if a.set == "pub" else 440
        cov = {lab: int(x[x.has & x.has_ctrl.fillna(False).astype(bool)].scene.nunique()) for lab, x in R.groupby("label")}
        G = dict(scenes=len(units), scene_errors=len(err),
                 G_cov=dict(per_label=cov, line=nmin, passed=bool(min(cov.values()) >= nmin)),
                 G_pose=dict(share=S("pose_have") / max(S("pose_need"), 1), passed=bool(S("pose_have") >= 0.99 * S("pose_need"))),
                 G_lat=dict(share=S("lat_ok") / max(S("lat_n"), 1), n=S("lat_n"), passed=bool(S("lat_ok") >= 0.95 * S("lat_n"))),
                 G_gap=dict(spearman=float(spearmanr(gp[:, 0], gp[:, 1])[0]) if len(gp) > 10 else np.nan, n=int(len(gp))),
                 G_dir=dict(same=float((Lg.cls == 1).sum() / max((Lg.cls > 0).sum(), 1)), onc=float((Lg.cls == 2).sum() / max((Lg.cls > 0).sum(), 1))),
                 anchor_match=dict(share=S("anchor_ok") / max(S("anchor_n"), 1), n=S("anchor_n")))
        G["G_gap"]["passed"] = bool(G["G_gap"]["spearman"] >= 0.9)
        G["G_dir"]["passed"] = bool(G["G_dir"]["same"] >= 0.95 and G["G_dir"]["onc"] <= 0.01)
        G["passed"] = all(v["passed"] for k, v in G.items() if k.startswith("G_"))
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.summary.update(scenes=len(units), errors=len(err), passed=G["passed"])
        run.info(json.dumps({k: v for k, v in G.items() if k.startswith("G_")}, default=float))


# ---------------------------------------------------------------- perception column (road_probe's held read, shipped P0 heads)
def percep(a):
    import pandas as pd
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(REPO / "experiments/road_probe/scripts"))
    import rp as RP
    Dh = pd.read_parquet(RP.W / "labels/held.parquet")
    out, prob = RP.shipped("held", Dh)
    eL, eR = np.abs(out[:, 4] - Dh.L20.to_numpy()), np.abs(out[:, 5] - Dh.R20.to_numpy())
    X = pd.DataFrame(dict(scene=Dh.scene.to_numpy(), k=Dh.k.to_numpy(), err20=np.nanmean(np.c_[eL, eR], 1), errL20=eL, errR20=eR,
                          both=(prob[:, 1:3] > 0.5).all(1), turn40=Dh.turn40.to_numpy()))
    X.to_csv(OUT / "percep_held.csv.gz", index=False)
    print(len(X), "decisions,", X.scene.nunique(), "scenes; straight median err20", float(X[X.turn40 < 5].err20.median()))


# ---------------------------------------------------------------- BEV payloads
def bev(a):
    import pickle
    import pandas as pd
    from ao_e1 import drive_records, loader
    C = Path(a.collect)
    picks = json.loads(Path(a.picks).read_text())
    R = pd.read_parquet(C / "roll.parquet")
    cfg = str(Path(R[R.has].run.iloc[0]) / "generated-user-config-0.yaml")
    meta = json.loads((C / "meta.json").read_text())
    K = KSET[meta["config"]["set"]]
    out, recs_of = {}, {}
    for p in picks:
        scene, k, ref_lab = p["scene"], int(p["k"]), p["label"]
        ds = loader(cfg).get_data_source(scene)
        tr = ds.rig.trajectory
        Tg, Pg, Yg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float), np.unwrap(np.asarray(tr.yaws, float))
        vc = ds.rig.vehicle_config
        X = R[(R.scene == scene) & R.has]
        paths, anchor = {}, None
        for _, r in X.iterrows():
            c = ctrl(r.run, r.rid)
            if not c:
                continue
            ts = sorted(t for t in c if (t - T0) % DT == 0 and T0 <= t <= T0 + DT * K)
            paths[r.label] = np.array([c[t][:3] for t in ts])
            if r.label == ref_lab:
                anchor = np.array(c.get(int(T0 + DT * k), c[ts[-1]])[:3])
        t = T0 + DT * k
        if anchor is None:
            anchor = np.array([np.interp(t, Tg, Pg[:, 0]), np.interp(t, Tg, Pg[:, 1]), np.interp(t, Tg, Yg)])
        plan = None
        rr = X[X.label == ref_lab]
        if len(rr) and (Path(rr.run.iloc[0]) / "driver-logs/drive.jsonl").exists():
            key = rr.run.iloc[0]
            recs_of.setdefault(key, drive_records(key))
            r = next((x for x in recs_of[key].get(scene, []) if int((x["t0"] - T0) // DT) == k), None)
            if r is not None:
                plan = ZM.in_ego(anchor, ZM.to_local(np.asarray(r["anchor"], float), np.asarray(r["poses"], float)[:, :2]))
        lanes = ZM.Lanes(ds.map)
        Ls = []
        for ln in ds.map.get_lanes_within(np.array([anchor[0], anchor[1], float(np.interp(t, Tg, Pg[:, 2]))]), 45.0):
            poly, cl = lanes.lane(ln)
            ext = [ZM.in_ego(anchor, np.asarray(gm.exterior.coords)[:, :2]) for gm in getattr(poly, "geoms", [poly]) if hasattr(gm, "exterior")]
            ctr = np.asarray(ln.center.points)[:, :2]
            hd = np.arctan2(*(ctr[-1] - ctr[0])[::-1]) if len(ctr) > 1 else 0.0
            Ls.append(dict(ext=ext, center=ZM.in_ego(anchor, ctr), dpsi=float(np.degrees(ZM.wrap(hd - anchor[2])))))
        st = ZM.obj_state(ZM.objects(ds), t)
        E, dpsi, _ = ZM.rel(st, anchor, float(vc.aabb_x_offset_m + vc.aabb_x_m))
        out[f"{p['cell']}|{scene}|{k}"] = dict(pick=p, lanes=Ls, objs=dict(x=E[:, 0], y=E[:, 1], dpsi=dpsi, L=st["L"], W=st["W"]),
                                              log=ZM.in_ego(anchor, Pg[:, :2]), paths={lab: ZM.in_ego(anchor, v[:, :2]) for lab, v in paths.items()},
                                              plan=plan, ego=(float(vc.aabb_x_offset_m), float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2))
    pickle.dump(out, open(C / "bev.pkl", "wb"), protocol=4)
    print(len(out), "BEV payloads ->", C / "bev.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("tags")
    p = sub.add_parser("collect")
    p.add_argument("--set", choices=("pub", "held"), required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    sub.add_parser("percep")
    p = sub.add_parser("bev")
    p.add_argument("--collect", required=True)
    p.add_argument("--picks", required=True)
    a = ap.parse_args()
    dict(probe=probe, tags=tags, collect=collect, percep=percep, bev=bev)[a.cmd](a)
