#!/usr/bin/env python3
"""ROUND9 ROUTE-CHECK: is the route branch (route memory + command) worth a base retrain for PAI? CPU only, stored reads, no training.

Reads (PAI 281, ACC + GAP, Harmonizer off): R6DAG-L18 s0 / s1 (subject), NTV0-L18-s0, R7J-L18 s0 / s1 (served with JEV_CMD=heading).
Every model here feeds the PROJECTED route memory of experiments/route_mem/lib/rmem.py: 40 slots at route arc -6 .. +72 m (2 m) from the
ego's projection, [valid, x / 20, y / 20, cos h, sin h] in the rear-axle frame; command one-hot in the ego row (shipped rule: the route's
~42 m point |y| > 2 m; R7J: rmem.cmd_heading, 40 m / 20 deg).

  extract (box, repo .venv)    every inference record of the five reads' driver/drive.jsonl -> $O/dec.parquet (one row per decision,
                               map-free scalars: speed, stored / heading-rule command, route coverage ahead, served / model plan 4 s arc,
                               plan vs route lateral, route heading change) + $O/tgt.pkl (full arrays of the target scenes) + $O/meta.pkl
                               (facts, controller start, sim configs)
  events  (box, AlpaSim venv)  target scenes only: every decision against the NuRec lane map (round6 rmc.state_metrics: ego lane offset,
                               route lane offset at 10 / 20 / 40 m, plan 4 s lane offset, human-log station) -> $O/ev.parquet
  band    (box, op-train venv) C: route dependence by distance band on cached pool rows (R6DAG-L18-s0 / s1, CPU forwards; rmem slots of a
                               band masked or shifted 1 m left) -> $O/band_<ckpt>.parquet
  report  (Mac or box, pandas) tables -> experiments/round9/results/route_check/
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
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/route_mem/lib"), str(REPO / "experiments/round6/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/route_check"
RES = REPO / "experiments/round9/results/route_check"
READS = {"R6DAG-s0": ("R6DAG-L18-s0", "round6/dag/pai281/R6DAG-L18-s0-ag"), "R6DAG-s1": ("R6DAG-L18-s1", "round6/dag/pai281/R6DAG-L18-s1-ag"),
         "NTV0": ("NTV0-L18-s0", "defect_fix/pai_accgap/NTV0-L18-s0"), "R7J-s0": ("R7J-L18-s0", "round7/joint/pai281/R7J-L18-s0-ag"),
         "R7J-s1": ("R7J-L18-s1", "round7/joint/pai281/R7J-L18-s1-ag")}
TARGET = ("cor_drift", "off_straight", "off_junc_straight", "cor_lane_change", "cor_straight_at_turn")
PRE = 5.0                                           # seconds before the event


def _arr(x, dt=np.float32):
    return np.asarray(json.loads(x) if isinstance(x, str) else x, dt)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def plan_arc(P):
    """(8, >= 2) ego-frame plan at 0.5 .. 4 s -> cumulative arc (8,) from the origin."""
    P = np.asarray(P, np.float64)[:, :2]
    return np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P], axis=0).T))


def route_scalars(F, ps, pm):
    """F (40, 5) fed memory, ps / pm (8, 3) served / model plan -> dict of map-free scalars."""
    import rmem as RM
    v = F[:, 0] > 0.5
    O_ = RM.OFFS
    r = dict(n_valid=int(v.sum()))
    ahead = v & (O_ >= 0)
    r["cov"] = float(O_[ahead].max()) if ahead.any() else 0.0              # route arc covered ahead of the ego's projection (<= 72)
    r["back"] = float(O_[v].min()) if v.any() else np.nan
    h = np.arctan2(F[:, 4], F[:, 3])
    x, y = F[:, 1] * 20.0, F[:, 2] * 20.0
    i0 = RM.I0
    r["y_rt0"] = float(y[i0]) if v[i0] else np.nan                          # ego's lateral offset to the route at its projection (route frame ~)
    for s in (20, 40, 42, 60, 72):
        i = int(np.flatnonzero(O_ == s)[0])
        r[f"y_rt{s}"] = float(y[i]) if v[i] else np.nan
    if v[i0]:
        dh = np.degrees(wrap(h - h[i0]))
        m40 = ahead & (O_ <= 40)
        r["dth40"] = float(dh[m40][np.argmax(np.abs(dh[m40]))]) if m40.any() else np.nan     # max-abs route heading change 0..40 m
        r["dth_cov"] = float(dh[ahead][np.argmax(np.abs(dh[ahead]))]) if ahead.any() else np.nan
    else:
        r["dth40"] = r["dth_cov"] = np.nan
    r["cmd_h"] = int(np.argmax(RM.cmd_heading(F.astype(np.float16).reshape(-1))))
    for tag, P in (("", ps), ("m", pm)):
        a = plan_arc(P)
        r[f"arc2{tag}"], r[f"arc3{tag}"], r[f"arc4{tag}"] = float(a[3]), float(a[5]), float(a[7])
        r[f"y4{tag}"] = float(P[7, 1])
        # first time the plan's arc runs past the route coverage (s); inf if never within 4 s
        t = 0.5 * np.arange(1, 9)
        r[f"t_exit{tag}"] = float(np.interp(r["cov"], a, t)) if a[-1] > r["cov"] and r["cov"] > a[0] else (0.5 if a[0] >= r["cov"] else np.inf)
        # lateral of the plan point at arc s against the route at the same arc (route normal, + = plan left of route)
        for nm, s in (("4", a[7]), ("cov", min(r["cov"], a[7]))):
            if not (ahead.sum() >= 2 and O_[v][0] <= s <= O_[v][-1]):
                r[f"lat{nm}{tag}"] = np.nan
                continue
            o = O_[v]
            q = np.array([np.interp(s, o, x[v]), np.interp(s, o, y[v])])
            hq = float(np.interp(s, o, np.unwrap(h[v])))
            p = np.array([np.interp(s, a, P[:, 0]), np.interp(s, a, P[:, 1])])
            r[f"lat{nm}{tag}"] = float(-(p - q)[0] * np.sin(hq) + (p - q)[1] * np.cos(hq))
    return r


# ---------------------------------------------------------------- extract (box, .venv)
def _one_log(job):
    f, tgt = job
    rows, full = [], defaultdict(lambda: defaultdict(list))
    for line in open(f):
        if '"infer": true' not in line or '"kind": "drive"' not in line:
            continue
        r = json.loads(line)
        if "rmem" not in r:
            continue
        fx = r.get("fix") or {}
        if isinstance(fx, str):                                                # some records store the dict's repr
            m = re.search(r"'v0': ([-0-9.e]+)", fx)
            fx = {"v0": float(m.group(1))} if m else {}
        F = _arr(r["rmem"]).reshape(40, 5)
        ps = _arr(r["poses"])[:8, :3]
        pm = _arr(r.get("poses_model") or r["poses"])[:8, :3]
        v0 = float(fx.get("v0", _arr(r["ego"])[4] * 10))
        d = dict(log=f, scene=r["scene"], t0=int(r["t0"]), v0=v0, cmd=int(r["cmd"]), **route_scalars(F, ps, pm))
        rows.append(d)
        if r["scene"] in tgt:
            S = full[r["scene"]]
            S["t0"].append(int(r["t0"]))
            S["anchor"].append(np.asarray(r["anchor"], np.float64)[:3])
            S["cmd"].append(int(r["cmd"]))
            S["rm"].append(F)
            S["ps"].append(ps)
            S["pm"].append(pm)
            S["v0"].append(v0)
    return rows, {s: {k: np.asarray(v) for k, v in d.items()} for s, d in full.items()}


def cmd_extract(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    lm = pd.read_csv(REPO / "experiments/round7/results/loss_map.csv")
    tgt = set(lm[lm["R6DAG-s0_primary"].isin(TARGET) | lm["R6DAG-s1_primary"].isin(TARGET)].scene)
    print("target scenes", len(tgt), flush=True)
    allrows, full, meta = [], {}, {}
    jobs = [(k, f) for k, (tag, root) in READS.items() for f in sorted(glob.glob(f"{DATA}/runs/{root}/runs/*/driver/drive.jsonl"))]
    res = par.pmap(_one_log, [(f, tgt) for _, f in jobs], workers=min(len(jobs), a.workers or n_cpus()), desc="logs")
    res.raise_if_failed()
    for k, (tag, root) in READS.items():
        mine = [res.values[i] for i, (kk, _) in enumerate(jobs) if kk == k]
        best = {}                                                                # a scene read twice (retry): keep the log with most decisions
        for rows, fl in mine:
            for s, d in fl.items():
                if s not in full.get(k, {}) or len(d["t0"]) > len(full[k][s]["t0"]):
                    full.setdefault(k, {})[s] = d
            cnt = defaultdict(int)
            for x in rows:
                cnt[x["scene"]] += 1
            for s, n in cnt.items():
                if n > best.get(s, (0, None))[0]:
                    best[s] = (n, rows[0]["log"])
        keep = {(s, lg) for s, (n, lg) in best.items()}
        for rows, _ in mine:
            allrows += [dict(read=k, **x) for x in rows if (x["scene"], x["log"]) in keep]
        F = pd.read_csv(DATA / f"runs/pai_fix/facts/{tag}-281ag.csv")
        tax = pickle.load(open(DATA / f"runs/pai_fix/tax/{tag}-281ag.pkl", "rb"))["rolls"]
        c0 = {o["scene"]: float(o["ctrl"]["t"][0]) for o in tax if "ctrl" in o}
        cfg = {}
        for f in glob.glob(f"{DATA}/runs/{root}/runs/*/sim/generated-user-config-0.yaml"):
            for s in re.findall(r"scene_id: (clipgt-[0-9a-f-]+)", open(f).read()):
                cfg.setdefault(s, f)
        meta[k] = dict(tag=tag, facts=F, c0=c0, cfg=cfg)
        print(k, len(mine), "logs", len(best), "scenes", flush=True)
    for k in full:
        for s, d in full[k].items():
            i = np.argsort(d["t0"], kind="stable")
            full[k][s] = {x: v[i] for x, v in d.items()}
    O.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(allrows).drop(columns="log")
    df.to_parquet(O / "dec.parquet")
    pickle.dump(full, open(O / "tgt.pkl", "wb"), protocol=4)
    pickle.dump(meta, open(O / "meta.pkl", "wb"), protocol=4)
    print("decisions", len(df), df.groupby("read").size().to_dict(), flush=True)


# ---------------------------------------------------------------- events (box, AlpaSim venv)
def _scene(job):
    import dag
    import rmc
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
    L = rmc.Lanes(ds.map, R)
    zz = lambda xy: float(P[int(np.argmin(((P[:, :2] - xy[:2]) ** 2).sum(1))), 2])  # noqa: E731
    lref = dag.Ref(P[:, :2])
    out = []
    for k, d in decs.items():
        for j in range(len(d["t0"])):
            pose = d["anchor"][j]
            r = dict(read=k, scene=scene, t0=int(d["t0"][j]))
            pr = lref.project(np.asarray(pose, np.float64))
            r["st_log"], r["d_log"] = (pr[0], pr[1]) if pr is not None else (np.nan, np.nan)
            m = rmc.state_metrics(L, pose, zz(pose), d["rm"][j].astype(np.float64), d["ps"][j])
            out.append(r | {f"m_{x}": y for x, y in m.items()})
    return out


def cmd_events(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    full, meta = pickle.load(open(O / "tgt.pkl", "rb")), pickle.load(open(O / "meta.pkl", "rb"))
    scenes = sorted({s for k in full for s in full[k]})
    jobs = []
    for s in scenes:
        cfg = next((meta[k]["cfg"][s] for k in READS if s in meta[k]["cfg"]), None)
        if cfg:
            jobs.append((s, cfg, {k: full[k][s] for k in full if s in full[k]}))
    res = par.pmap(_scene, jobs, workers=min(len(jobs), a.workers or n_cpus()), desc="scenes")
    print(res.summary(), flush=True)
    for i, e in list(res.errors.items())[:5]:
        print(jobs[i][0], e.strip().splitlines()[-1][:300], flush=True)
    pd.DataFrame([x for v in res.values if v for x in v]).to_parquet(O / "ev.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("extract", "events"):
        p = sub.add_parser(n)
        p.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    dict(extract=cmd_extract, events=cmd_events)[a.cmd](a)
