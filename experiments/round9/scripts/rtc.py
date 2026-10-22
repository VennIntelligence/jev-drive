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


# ---------------------------------------------------------------- band (box, op-train venv, CPU): C, route dependence by distance band
R6DAG_ARGV = ("--tag {tag} --init NTV0-L18-s0 --allow-init --warmup 1 --eval-every 500 --yr-mass 0.095 --label-bank --agent-lam 10 --agent-on-only "
              "--shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --lr-scale 0.1 --on-k 13 --on-mode self --on on1_JTPHD --on-k 72 "
              "--on-mode mixed --on-lon-all --on-normal --on-wfile {data}/runs/round6/dag/w/R6DAG.npz --on-hold-pct 15 --ego-scale-speed 0.5 "
              "--ego-scale-max 0.2 --lat-onc 0.5 --lat-soft 0.1 --lat-side 0.5 --lat-file {data}/runs/defect_fix/lat/L18NR1v2 --lat-depth 0.0 "
              "--lat-soft-depth 0.0 --lat-buf 0.4 --lat-host --on-arc 0 --diag --warmup 500 --lr-floor 0.1 --on-arc-rows brake --steps 9500 --seed 0 "
              "--batch 136")                                   # R6DAG-L18-s0's train line (runs/round6/dag/stages.txt), rely mode builds the store
BANDS = {"tail": (-7, 0), "b0_20": (0, 20), "b20_40": (20, 40), "b40_72": (40, 73), "all": (-7, 73)}
SPEED_BINS = (0.5, 5, 10, 15, 20, 45)


def cmd_band(a):
    import pandas as pd
    import torch
    import time
    import rmem as RM
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *R6DAG_ARGV.format(tag=a.ckpt, data=DATA).split()]
    exec(compile(src.replace(tail, "_RTC_ARGS = a"), str(rt), "exec"), g)
    ra = g["_RTC_ARGS"]
    ra.rm_from = a.ckpt
    dev = torch.device("cpu")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    S, model = ns.S, ns.model.eval()
    names = S.tab["names"][ns.on0:].astype(str)
    part = np.array([x.split(":")[1] for x in names])
    pt = np.load(DATA / "runs/op_parity/cache/on1_JTPHD/tab.npz")
    assert np.array_equal(pt["names"].astype(str), names), "pool rows are not on1_JTPHD's"
    speed = pt["speed"].astype(np.float64)
    E0 = RM.EGO_DIM
    F = S.ego[ns.on0:, E0:E0 + RM.DIM].float().cpu().numpy().reshape(-1, RM.N_PT, RM.NF)
    v = F[:, :, 0] > 0.5
    h = np.arctan2(F[:, :, 4], F[:, :, 3])
    m40 = v & (RM.OFFS >= 0) & (RM.OFFS <= 40)
    dh = np.abs(wrap(h - h[:, RM.I0:RM.I0 + 1]))
    dth40 = np.degrees(np.where(m40 & v[:, RM.I0:RM.I0 + 1], dh, 0).max(1))
    ok = v[:, RM.I0] & (v.sum(1) >= 10)
    straight = dth40 < 3.0
    rng = np.random.default_rng(a.seed)
    sel = []
    for p_ in a.parts.split(","):
        for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
            for st in (True, False):
                c = np.flatnonzero((part == p_) & ok & (speed >= lo) & (speed < hi) & (straight == st))
                sel += list(rng.choice(c, min(a.per, len(c)), replace=False))
    sel = np.array(sorted(sel))
    print("rows", len(sel), flush=True)
    pi_ = torch.as_tensor(S.pi, device=dev)
    cov = np.where(v & (RM.OFFS >= 0), RM.OFFS, -1).max(1)

    def fwd(r, ego):
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], ego, S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
        return torch.stack([x, y], -1).cpu().numpy()

    variants = ["base"] + [f"{k}:{b}" for k in ("mask", "shift") for b in BANDS] + ["cmd:S", "cmd:L", "cmd:R", "cmd:H"]
    if a.only:
        variants = ["base"] + a.only.split(",")
    out = {k: [] for k in variants}
    t0 = time.time()
    for j in range(0, len(sel), a.bs):
        r = torch.as_tensor(ns.on0 + sel[j:j + a.bs], device=dev)
        e0 = S.ego[r].float().clone()
        for k in variants:
            e = e0.clone()
            if k.startswith(("mask", "shift")):
                kind, b = k.split(":")
                lo, hi = BANDS[b]
                cols = np.flatnonzero((RM.OFFS >= lo) & (RM.OFFS < hi))
                P = e[:, E0:E0 + RM.DIM].view(-1, RM.N_PT, RM.NF)
                if kind == "mask":
                    P[:, cols] = 0.0
                else:                                                            # a.shift m to the left (- = right), valid slots only
                    P[:, cols, 2] += (P[:, cols, 0] > 0.5).float() * a.shift / 20.0
                e[:, E0:E0 + RM.DIM] = P.reshape(-1, RM.DIM)
            elif k.startswith("cmd"):
                c = k.split(":")[1]
                if c == "H":
                    oh = torch.as_tensor(RM.cmd_heading(e[:, E0:E0 + RM.DIM].cpu().numpy().astype(np.float16))[:, :3])
                else:
                    oh = torch.zeros(len(e), 3)
                    oh[:, "LSR".index(c)] = 1.0
                e[:, 1:4] = oh.to(e.dtype)
            out[k].append(fwd(r, e))
        if j == 0:
            print(f"first batch {time.time() - t0:.1f} s", flush=True)
    X = {k: np.concatenate(v_) for k, v_ in out.items()}
    base = X["base"]
    df = pd.DataFrame(dict(name=names[sel], part=part[sel], speed=speed[sel], straight=straight[sel], dth40=dth40[sel], cov=cov[sel],
                           cmd=np.where(S.ego[ns.on0 + sel, 1:4].float().numpy().max(1) > 0.5, S.ego[ns.on0 + sel, 1:4].float().numpy().argmax(1), 3),
                           x4=base[:, -1, 0], y4=base[:, -1, 1],
                           arc4=np.hypot(*np.diff(np.concatenate([np.zeros((len(base), 1, 2)), base], 1), axis=1).transpose(2, 0, 1)).sum(1)))
    for k in variants[1:]:
        d = X[k] - base
        df[f"dy4_{k}"] = d[:, -1, 1]
        df[f"dx4_{k}"] = d[:, -1, 0]
        df[f"dmax_{k}"] = np.hypot(d[..., 0], d[..., 1]).max(1)
    O.mkdir(parents=True, exist_ok=True)
    df.to_parquet(O / f"band_{a.ckpt}{a.suffix}.parquet")
    print(json.dumps(dict(ckpt=a.ckpt, rows=len(df), sec=round(time.time() - t0, 1), horizon=int(base.shape[1]))), flush=True)


# ---------------------------------------------------------------- report (pandas; --src = the dir holding dec / ev / meta / band files)
CMD = np.array(["L", "S", "R", "U"])
BINS = (0.5, 5, 10, 15, 20, 25, 45)


def md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(":--" if df[c].dtype == object else "--:" for c in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return "\n".join(L)


def sbin(v):
    import pandas as pd
    return pd.cut(v, BINS, right=False, labels=[f"{a}-{b}" for a, b in zip(BINS[:-1], BINS[1:])])


def ev_summary(g, te, primary):
    """decisions g of one read on one scene (dec + ev columns), event time te (t0 units) -> window summary with the mechanism label.
    rh = the route's lateral offset from the human log at the ego's station (y_rt0 + d_log, + left); leave = the first decision in the last
    15 s with |d_log| > 1 m (the ego leaves the human's path), else the event; lead-up = the 5 s before leave."""
    w = g[(g.t0 >= te - 15e6) & (g.t0 <= te)]
    if not len(w):
        return None
    rh = (w.y_rt0 + w.d_log).to_numpy()
    off = np.flatnonzero(np.abs(w.d_log.to_numpy()) > 1.0)
    tl = int(w.t0.iloc[off[0]]) if len(off) else te
    lu = w[(w.t0 >= tl - PRE * 1e6) & (w.t0 <= tl)]
    fin = np.isfinite(rh)
    rh0 = float(rh[fin][0]) if fin.any() else np.nan
    rh1 = float(rh[fin][-1]) if fin.any() else np.nan
    last = w.iloc[-1]
    side = float(np.sign(rh0)) if np.isfinite(rh0) and abs(rh0) >= 2.5 else 0.0
    cmd_side = np.where(lu.cmd == 0, 1.0, np.where(lu.cmd == 2, -1.0, 0.0))
    e = dict(fclass_primary=primary, v_leave=float(lu.v0.iloc[-1]) if len(lu) else np.nan, v_lu=float(lu.v0.mean()),
             t_leave=(tl - te) / 1e6, flips_lu=int((np.diff(lu.cmd.to_numpy()) != 0).sum()),
             cmdLR_lu=float((lu.cmd.isin([0, 2])).mean()), cmd_to_route_lu=float((cmd_side == side).mean()) if side else np.nan,
             cmdH_LR_lu=float((lu.cmd_h.isin([0, 2])).mean()), dth40_lu=float(lu.dth40.abs().median()),
             beyond_lu=float((lu.arc4 > lu["cov"]).mean()), arc4_lu=float(lu.arc4.median()),
             rh_start=rh0, rh_end=rh1, rh_absmax=float(np.nanmax(np.abs(rh))) if fin.any() else np.nan,
             y_rt0_end=float(last.y_rt0), d_log_end=float(last.d_log), d_ego_end=float(last.m_d_ego),
             latcov_lu=float(lu.latcov.median()), d_rt20_lu=float(lu.m_d_rt20.median()))
    near_route = np.isfinite(e["y_rt0_end"]) and abs(e["y_rt0_end"]) < 1.5
    if primary == "cor_straight_at_turn":
        m = "route turn not taken"
    elif np.isfinite(rh0) and abs(rh0) >= 2.5 and near_route and abs(e["d_log_end"]) > 2.5:
        m = "route in another lane than the human; ego follows the route"
    elif np.isfinite(rh0) and abs(rh0) >= 2.5 and np.isfinite(rh1) and abs(rh1) < 1.5:
        m = "route in another lane; human merges into it, ego stays"
    elif np.isfinite(rh0) and abs(rh0) < 1.5 and np.isfinite(rh1) and abs(rh1) >= 2.5 and near_route:
        m = "human leaves the route's lane; ego stays on the route"
    elif np.isfinite(e["d_log_end"]) and abs(e["d_log_end"]) <= 1.5:
        m = "on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge"
    elif e["cmdLR_lu"] > 0.2 and e["cmdH_LR_lu"] < 0.2:
        m = "command flip (old rule L/R on a non-turn), route fine"
    else:
        m = "policy drift, route fine"
    e["mech"] = m
    return e


def cmd_report(a):
    import pandas as pd
    src = Path(a.src) if a.src else O
    D = pd.read_parquet(src / "dec.parquet")
    EV = pd.read_parquet(src / "ev.parquet")
    meta = pickle.load(open(src / "meta.pkl", "rb"))
    RES.mkdir(parents=True, exist_ok=True)
    X = D.merge(EV, on=["read", "scene", "t0"], how="inner", validate="1:1")
    FA = {k: meta[k]["facts"].set_index("scene") for k in READS}
    out, L = {}, []

    # ---------------- A: events
    evrows, ctr = [], []
    for k in ("R6DAG-s0", "R6DAG-s1"):
        F = FA[k]
        for sc in sorted(X.scene.unique()):
            if sc not in F.index or F.primary[sc] not in TARGET or not np.isfinite(F.t_ev_s[sc]) or sc not in meta[k]["c0"]:
                continue
            te = meta[k]["c0"][sc] + float(F.t_ev_s[sc]) * 1e6
            g = X[(X.read == k) & (X.scene == sc)].sort_values("t0")
            e = ev_summary(g, te, F.primary[sc])
            if e is None:
                continue
            evrows.append(dict(read=k, scene=sc[7:15], **e))
            w = g[(g.t0 >= te - 15e6) & (g.t0 <= te)]
            st = w.st_log.dropna()
            for o in READS:
                Fo = FA[o]
                h = X[(X.read == o) & (X.scene == sc)].sort_values("t0")
                if len(st):
                    h = h[(h.st_log >= st.min() - 5) & (h.st_log <= st.max())]
                if not len(h):
                    continue
                rh = (h.y_rt0 + h.d_log)
                side = np.sign(evrows[-1]["rh_start"]) if abs(evrows[-1]["rh_start"]) >= 2.5 else 0
                cs = np.where(h.cmd == 0, 1.0, np.where(h.cmd == 2, -1.0, 0.0))
                ctr.append(dict(event_read=k, scene=sc[7:15], read=o, score=float(Fo.score.get(sc, np.nan)), primary=Fo.primary.get(sc, ""),
                                n=len(h), v=float(h.v0.mean()), cmdLR=float(h.cmd.isin([0, 2]).mean()),
                                cmd_to_route=float((cs == side).mean()) if side else np.nan,
                                beyond=float((h.arc4 > h["cov"]).mean()), latcov=float(h.latcov.median()),
                                max_abs_d_log=float(h.d_log.abs().max()), end_y_rt0=float(h.y_rt0.iloc[-1]), end_d_log=float(h.d_log.iloc[-1]),
                                followed=bool(abs(h.y_rt0.iloc[-1]) < 1.5 and abs(h.d_log.iloc[-1]) > 2.5)))
    E, C = pd.DataFrame(evrows), pd.DataFrame(ctr)
    E.to_csv(RES / "events.csv", index=False, float_format="%.3f")
    C.to_csv(RES / "events_contrast.csv", index=False, float_format="%.3f")
    out["mech"] = {k: E[E.read == k].mech.value_counts().to_dict() for k in ("R6DAG-s0", "R6DAG-s1")}
    cols = ["read", "scene", "fclass_primary", "t_leave", "v_leave", "flips_lu", "cmdLR_lu", "cmd_to_route_lu", "cmdH_LR_lu", "dth40_lu",
            "arc4_lu", "beyond_lu", "rh_start", "rh_end", "y_rt0_end", "d_log_end", "latcov_lu", "mech"]
    L += ["## A. The target events (R6DAG-L18 s0 / s1)", "", md(E[cols]), ""]
    piv = C.pivot_table(index=["event_read", "scene"], columns="read", values="score").reset_index()
    L += ["Scores of every read on the same scenes:", "", md(piv), ""]
    cc = C[C.read != C.event_read].pivot_table(index=["event_read", "scene"], columns="read", values="cmd_to_route").reset_index()
    fo = C[C.read != C.event_read].pivot_table(index=["event_read", "scene"], columns="read", values="followed", aggfunc="max").reset_index()
    L += ["Ego ends on the route and > 2.5 m from the human (followed the route into the other lane):", "", md(fo), ""]
    L += ["Share of decisions whose stored command points to the route's side (route in another lane only), same human-log stretch:", "",
          md(cc), ""]

    # ---------------- B: population
    rows = []
    for k in READS:
        d = D[(D.read == k) & (D.v0 >= 0.5)].copy()
        d["bin"] = sbin(d.v0)
        for b, x in d.groupby("bin", observed=True):
            rows.append(dict(read=k, speed=str(b), n=len(x), arc4_p50=x.arc4.median(), arc4_p90=x.arc4.quantile(.9), cov_p50=x["cov"].median(),
                             cov_lt72=100 * (x["cov"] < 72).mean(), no_slot_at_ego=100 * x.y_rt0.isna().mean(),
                             beyond=100 * (x.arc4 > x["cov"]).mean(), beyond10=100 * (x.arc4 > x["cov"] + 10).mean(),
                             beyond_m_p50=(x.arc4 - x["cov"]).clip(lower=0).median(), t_exit_p50=x.t_exit.replace(np.inf, np.nan).median(),
                             arc4_gt72=100 * (x.arc4 > 72).mean()))
    P = pd.DataFrame(rows)
    P.to_csv(RES / "pop_speed.csv", index=False, float_format="%.3f")
    L += ["## B. Plan 4 s arc vs route coverage, every decision (v >= 0.5 m/s)", "", md(P[P.read.str.startswith("R6DAG")]), ""]
    rows = []
    for k in READS:
        d = D[(D.read == k) & (D.v0 >= 0.5)]
        for nm, m in (("straight (|dth40| < 3), v >= 15", (d.dth40.abs() < 3) & (d.v0 >= 15)),
                      ("straight, 10 <= v < 15", (d.dth40.abs() < 3) & (d.v0 >= 10) & (d.v0 < 15)),
                      ("straight, v < 10", (d.dth40.abs() < 3) & (d.v0 < 10)),
                      ("gentle (3-15), v >= 15", (d.dth40.abs().between(3, 15)) & (d.v0 >= 15)),
                      ("no near slot (route gap at the ego)", d.dth40.isna())):
            x = d[m]
            rows.append(dict(read=k, segment=nm, n=len(x), cmd_L=100 * (x.cmd == 0).mean(), cmd_R=100 * (x.cmd == 2).mean(),
                             cmdH_L=100 * (x.cmd_h == 0).mean(), cmdH_R=100 * (x.cmd_h == 2).mean(),
                             flips_per_min=float(np.nan)))
    Q = pd.DataFrame(rows)
    # command flips per minute of straight high-speed driving (stored command), per read
    for i, r in Q.iterrows():
        d = D[D.read == r.read].sort_values(["scene", "t0"])
        ch = (d.cmd.diff().fillna(0) != 0) & (d.scene == d.scene.shift())
        m = {"straight (|dth40| < 3), v >= 15": (d.dth40.abs() < 3) & (d.v0 >= 15)}.get(r.segment)
        if m is not None:
            Q.loc[i, "flips_per_min"] = float(ch[m].sum() / (m.sum() / 600.0))
    Q.to_csv(RES / "pop_cmd.csv", index=False, float_format="%.3f")
    L += ["Command one-hot by segment (stored = what the model was fed; cmdH = the heading rule on the same memory; R7J was fed cmdH):", "",
          md(Q), ""]
    lat = []
    for k in READS:
        d = D[(D.read == k) & (D.v0 >= 0.5)].copy()
        d["bin"] = sbin(d.v0)
        d = d[d.dth40.abs() < 3]
        for b, x in d.groupby("bin", observed=True):
            lat.append(dict(read=k, speed=str(b), n=len(x), latcov_p50=x.latcov.median(), abs_latcov_p90=x.latcov.abs().quantile(.9),
                            abs_lat4_p90_within=x.lat4.abs().quantile(.9)))
        d = D[(D.read == k) & (D.v0 >= 0.5) & (D.dth40.abs() < 3)].copy()
        d["bin"] = sbin(d.v0)
        for b, x in d.groupby("bin", observed=True):
            on = x[x.y_rt0.abs() < 1.0]
            lat[-len(d.bin.cat.categories) + list(d.bin.cat.categories).index(b)].update(
                on_route_n=len(on), on_route_abs_latcov_p90=on.latcov.abs().quantile(.9), on_route_latcov_gt1=100 * (on.latcov.abs() > 1).mean(),
                route_side_ge2p5=100 * (x.y_rt0.abs() >= 2.5).mean()) if b in set(d.bin) else None
    LT = pd.DataFrame(lat)
    LT.to_csv(RES / "pop_lat.csv", index=False, float_format="%.3f")
    L += ["Plan lateral vs the route at the last covered arc (straight route, + = plan left of route):", "", md(LT[LT.read.str.startswith("R6DAG")]), ""]

    # ---------------- C: band
    for f in sorted(src.glob("band_*.parquet")):
        B = pd.read_parquet(f)
        B["bin"] = pd.cut(B.speed, SPEED_BINS, right=False, labels=[f"{a}-{b}" for a, b in zip(SPEED_BINS[:-1], SPEED_BINS[1:])])
        var = [c[4:] for c in B.columns if c.startswith("dy4_")]
        rows = []
        for (b, stt), x in B.groupby(["bin", "straight"], observed=True):
            r = dict(speed=str(b), straight=bool(stt), n=len(x), cov_p50=x["cov"].median(), arc4_p50=x.arc4.median())
            for v_ in var:
                r[f"|dy4| {v_}"] = x[f"dy4_{v_}"].abs().median()
            rows.append(r)
        T_ = pd.DataFrame(rows)
        T_.to_csv(RES / f"{f.stem}.csv", index=False, float_format="%.4f")
        allr = {v_: dict(dy4_abs_p50=float(B[f"dy4_{v_}"].abs().median()), dy4_abs_p90=float(B[f"dy4_{v_}"].abs().quantile(.9)),
                         dx4_abs_p50=float(B[f"dx4_{v_}"].abs().median()), dmax_p90=float(B[f"dmax_{v_}"].quantile(.9)),
                         dy4_signed_p50=float(B[f"dy4_{v_}"].median())) for v_ in var}
        out[f.stem] = allr
        L += [f"## C. {f.stem}: median |plan 4 s lateral change| (m) per variant", "", md(T_, "{:.3f}"), "",
              md(pd.DataFrame(allr).T.reset_index().rename(columns={"index": "variant"}), "{:.3f}"), ""]
    (RES / "tables.md").write_text("\n".join(L) + "\n")
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("extract", "events"):
        p = sub.add_parser(n)
        p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("band")
    p.add_argument("--ckpt", required=True), p.add_argument("--parts", default="pd,ph"), p.add_argument("--per", type=int, default=40)
    p.add_argument("--bs", type=int, default=256), p.add_argument("--threads", type=int, default=0), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift", type=float, default=1.0), p.add_argument("--only", default=""), p.add_argument("--suffix", default="")
    p = sub.add_parser("report")
    p.add_argument("--src", default="")
    a = ap.parse_args()
    dict(extract=cmd_extract, events=cmd_events, band=cmd_band, report=cmd_report)[a.cmd](a)
