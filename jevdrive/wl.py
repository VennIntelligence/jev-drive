"""WL: openpilot proposes, a latent world model imagines, a critic chooses -- the fork data and its readouts
(todos/2026-09-28-wm-loop.md; registered 2026-09-27, criteria written before any WL number).

Generation (CARLA, scripts/wl_gen.sh with scripts/wl_fork_agent.py):
  forks      fork points of P5 v1 BA (pedestrian + cut-in families, x+ / x-) and P6 v0 (main obstacles, x10 / x00),
             seed 0: k1 = first camera tick >= the hazard's first visibility, k2 = k1 + 0.6 s, k3 = 0.4 s before the
             expert's reaction (P5: the x+ / x- ego divergence; P6: the lateral divergence), kept when k3 > k2 and
             v(k) >= 3 m/s; seven actions each; route-grouped eval split (40 routes: P5 pedestrian 12, cut-in 8, P6 20)
             -> runs/wl/{forks.parquet, split.json, jobs.json, forks-<set>.xml}
  d2         D2 random-intervention runs: Bench2Drive 220 routes (minus the eval routes) x TM seeds 0, 1, PDM-Lite
  ids        route ids of a stage (pilot1 / pilot10 / full) still without a done record, for b2d_run --route-ids
  sanity     the registered checklist on the finished fork runs of a stage -> research/results/wl/sanity_<stage>.csv
  prefix     the first check: a fork run's ticks before the fork against its source run (pose, actors, JPEG bytes)
  drops      checklist amendments (a) + (c), 2026-09-28 / 2026-09-29 (user-approved): pose stays a group gate (every
             fork group whose branches are all done is checked once against its source run's ego pose before k, any
             branch > 0.01 m drops the whole group). The old group cosine >= 0.999 gate is a per-run render gate
             instead: a run is dropped from z / training when more than half its pre-fork frames differ from the
             source in front-camera brightness by > 10, or its prefix `temporal` cosine to the source is < 0.95. A
             whole-run brightness scan against the run's own prefix baseline is also computed (no source needed, so
             it would also cover D2 and a failure that starts after the prefix), but it reads ordinary scene drift
             as failure (27 % of the full set, no clean split) and is left out of the gate -- descriptive only in
             drops.json's "whole_scan" until a reference-free method is validated. A fork point leaves the C1/C2
             paired readouts only when its `op`, `hold` or `brake_hard` branch is render-dropped; its other branches
             still train. -> runs/wl/drops.json; --gate exits 3 when the pose-dropped groups exceed 5 % (overall or
             per set) or the render-dropped runs exceed 10 % (overall)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
NAME = os.environ.get("WL_NAME", "wl")                      # "wl2": WL-2 (runs/wl2, processed/wl2_gen, todos/2026-09-29-wl2-prereg.md)
RESULTS = Path(os.environ["WL_RESULTS"]) if os.environ.get("WL_RESULTS") else REPO / "research" / "results" / NAME   # box: outside the tracked tree
TICK, CAM = 0.05, 4
PED_FAM = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian", "ParkingCrossingPedestrian")
CUTIN_FAM = ("StaticCutIn", "ParkingCutIn", "HighwayCutIn")
N_EVAL = {"ped": 12, "cutin": 8, "obstacle": 20}
SPLIT_SEED = 20260928
V_MIN = 3.0
SRC = {"ba": dict(runs="runs/p5v1", gen="gen-ba", proc="carla_p5v1_ba", xml="runs/p5v1/pairs.xml", agent="runs/p5v1/agent-ba.json"),
       "p6": dict(runs="runs/p6", gen="gen", proc="carla_p6", xml="runs/p6/pairs.xml", agent="runs/p6/agent-p6.json")}


def rundir(*p) -> Path:
    d = data_dir() / "runs" / NAME
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def attempt_dir(set_name: str, route_id: str) -> Path:
    s = SRC[set_name]
    rec = json.loads((data_dir() / s["runs"] / s["gen"] / "done" / f"{route_id}.json").read_text())
    return data_dir() / s["runs"] / s["gen"] / "attempts" / str(route_id) / str(rec["attempt"])


def _pose(adir: Path) -> pd.DataFrame:
    p = pd.read_json(adir / "pose.jsonl", lines=True)
    p["tick"] = np.arange(1, len(p) + 1)
    p["v"] = np.hypot(p.vx, p.vy)
    return p.set_index("tick")


def _frames(adir: Path) -> pd.DataFrame:
    return pd.read_json(adir / "frames.jsonl", lines=True).set_index("tick")


def _cam_ceil(k: float) -> int:
    k = int(np.ceil(k))
    return k + (-(k - 1)) % CAM


def _cam_floor(k: float) -> int:
    k = int(np.floor(k))
    return k - (k - 1) % CAM


def _op_plans(set_name: str) -> dict:
    """frame_name -> (20, 2) ego plan, from the stored Cinque plan streams of the set (op_streams_plan/cinque)."""
    d = data_dir() / "processed" / SRC[set_name]["proc"] / "op_streams_plan" / "cinque"
    out = {}
    for f in sorted(d.glob("*.npz")) if d.exists() else []:
        with np.load(f) as z:
            for n, p in zip(z["name"], z["plan"]):
                out[str(n)] = p
    return out


def _sources(seeds=(0,)) -> pd.DataFrame:
    """One row per (base route, seed, world pair): set, class, family, x+ / x- route ids, t_vis, reaction tick."""
    rows = []
    p = pd.read_csv(data_dir() / "processed" / SRC["ba"]["proc"] / "pairs.csv")
    p = p[p.seed.isin(seeds) & p.family.isin(PED_FAM + CUTIN_FAM) & (p.reason == "ok")]
    for r in p.itertuples():
        rows.append({"set": "ba", "cls": "ped" if r.family in PED_FAM else "cutin", "family": r.family, "base_id": str(r.base_id), "seed": int(r.seed),
                     "plus": str(r.plus), "minus": str(r.minus), "t_vis": float(r.t_vis),
                     "t_react": float(r.t_div) if r.t_div <= r.t_last else np.nan})
    c = pd.read_parquet(data_dir() / "processed" / SRC["p6"]["proc"] / "nq3_cases.parquet")
    c = c[c.seed.isin(seeds) & c.main.fillna(False).astype(bool) & (c.reason == "ok")]
    for r in c.itertuples():
        react = r.t_div_lat if pd.notna(r.t_div_lat) else r.t_div
        rows.append({"set": "p6", "cls": "obstacle", "family": r.scenario, "base_id": str(r.base_id), "seed": int(r.seed), "plus": str(r.x10),
                     "minus": str(r.x00), "t_vis": float(r.t_vis), "t_react": float(react) if pd.notna(react) else np.nan})
    return pd.DataFrame(rows)


def split(src: pd.DataFrame) -> dict:
    rng = np.random.RandomState(SPLIT_SEED)
    ev = []
    for cls, n in N_EVAL.items():
        b = np.array(sorted(src[src.cls == cls].base_id.unique()))
        ev += list(b[rng.permutation(len(b))][:n])
    return {"eval": sorted(ev), "seed": SPLIT_SEED, "n_eval": N_EVAL}


def forks(branch_s: float = 3.0, cont_s: float = 20.0) -> dict:
    """The fork table, the eval split, the per-run jobs and one route XML per source set (ids: 9 + 6-digit run number +
    the source id's last digit, so --tm-seed-from-id gives the source's TM seed)."""
    import xml.etree.ElementTree as ET
    from .wl_traj import ACTIONS
    src = _sources()
    sp = split(src)
    plans = {s: _op_plans(s) for s in SRC}
    rows, miss = [], []
    for r in src.itertuples():
        try:
            dp, dm = attempt_dir(r.set, r.plus), attempt_dir(r.set, r.minus)
        except FileNotFoundError:
            miss.append(r.base_id)
            continue
        pp, fp, fm = _pose(dp), _frames(dp), _frames(dm)
        k1 = _cam_ceil(r.t_vis)
        ks = {"k1": k1, "k2": k1 + 12}
        if np.isfinite(r.t_react):
            k3 = _cam_floor(r.t_react - 8)
            if k3 > ks["k2"]:
                ks["k3"] = k3
        for name, k in ks.items():
            if k not in pp.index or pp.v[k] < V_MIN or k not in fp.index or k not in fm.index:
                continue
            for world, rid, fr in (("plus", r.plus, fp), ("minus", r.minus, fm)):
                fn = f"{rid}-{int(fr.frame[k]):07d}"
                rows.append({"set": r.set, "cls": r.cls, "family": r.family, "base_id": r.base_id, "world": world,
                             "src_route": rid, "src_dir": str(attempt_dir(r.set, rid)), "k_name": name, "fork_tick": int(k),
                             "v0": float(pp.v[k]), "src_frame": fn, "has_op": fn in plans[r.set],
                             "split": "eval" if r.base_id in sp["eval"] else "train"})
    f = pd.DataFrame(rows)
    n_no_op = int((~f.has_op).sum())
    f = f[f.has_op].reset_index(drop=True)             # the op-derived branches need openpilot's own plan at the fork
    f["fork_id"] = np.arange(len(f))
    runs = f.loc[f.index.repeat(len(ACTIONS))].reset_index(drop=True)
    runs["action"] = np.tile(ACTIONS, len(f))
    runs["route_id"] = ["9%06d%s" % (i, s[-1]) for i, s in enumerate(runs.src_route)]
    jobs = {}
    for r in runs.itertuples():
        op = plans[r.set].get(r.src_frame)
        jobs[r.route_id] = {"fork_tick": r.fork_tick, "action": r.action, "op_plan": None if op is None else np.round(op, 4).tolist(),
                            "branch_s": branch_s, "cont_s": cont_s, "iv_s": 2.0, "iv_gap": [6.0, 10.0],
                            "seed": int(r.route_id), "src": r.src_dir, "fork_id": int(r.fork_id)}
    for s in SRC:
        root = ET.parse(data_dir() / SRC[s]["xml"]).getroot()
        by_id = {e.get("id"): e for e in root.iter("route")}
        out = ET.Element("routes")
        for r in runs[runs.set == s].itertuples():
            e = ET.fromstring(ET.tostring(by_id[r.src_route]))
            e.set("id", r.route_id)
            e.set("wl_src", r.src_route)
            out.append(e)
        ET.ElementTree(out).write(rundir(f"forks-{s}.xml"))
    runs.to_parquet(rundir("forks.parquet"), index=False)
    rundir("jobs.json").write_text(json.dumps(jobs))
    rundir("split.json").write_text(json.dumps(sp, indent=1))
    info = {"sources": len(src), "missing_source_runs": miss, "fork_points": len(f), "runs": len(runs),
            "by_set_split": f.groupby(["set", "split"]).size().to_dict().__repr__(),
            "by_k": f.k_name.value_counts().to_dict(), "dropped_no_op_plan": n_no_op,
            "routes_by_cls_split": f.groupby(["cls", "split"]).base_id.nunique().to_dict().__repr__()}
    rundir("forks_info.json").write_text(json.dumps(info, indent=1, default=str))
    return info


# ================================================================ WL-2 (todos/2026-09-29-wl2-prereg.md)

TAU_S = (4.0, 3.0, 2.0)                  # P6 fork times: ego-to-obstacle longitudinal distance first <= v * tau
K1B_TICKS = 8                            # "k1 + 0.3 s" on the 5 Hz camera grid: 0.2 or 0.4 s, 0.4 s taken
OCC_AT_S = 2.0                           # P5 training forks: the source run's ego lane is occupied at k + 2 s
WL2_CAPS = {"ba": 150, "p6": 150}        # training fork points (rows, both worlds) per set
WL1_SPLIT = ("runs", "wl", "split.json")


def _p6_tau_ticks(adir: Path, pp: pd.DataFrame) -> dict:
    """{tau: first camera tick where the longitudinal distance from the ego to the nearest scenario obstacle ahead
    (hidden.json entries that are not hidden, |lateral| <= 6 m) is <= v * tau}, from the source run's own ticks."""
    hid = json.loads((adir / "hidden.json").read_text())
    ids = [int(h["id"]) for h in hid if not h.get("hidden")]
    z = np.load(adir / "actors.npz")
    fr0 = int(_frames(adir).frame.iloc[0])
    tick = z["frame"] - fr0 + 1
    m = np.isin(z["id"], ids)
    tk, xy = tick[m], z["xyz"][m][:, :2].astype(float)
    out = {}
    for t in pp.index[(pp.index - 1) % CAM == 0]:
        q = xy[tk == t]
        if not len(q) or pp.v[t] < V_MIN:
            continue
        e = pp.loc[t]
        c, s = np.cos(np.radians(e.yaw)), np.sin(np.radians(e.yaw))
        dx, dy = q[:, 0] - e.x, q[:, 1] - e.y
        x, y = dx * c + dy * s, -dx * s + dy * c
        ok = (x > 0) & (np.abs(y) <= 6.0)
        if not ok.any():
            continue
        d = float(x[ok].min())
        for tau in TAU_S:
            if tau not in out and d <= pp.v[t] * tau:
                out[tau] = int(t)
    return out


def forks2(seeds=(1, 2), branch_s: float = 3.0, seed: int = 20260929) -> dict:
    """WL-2 fork table (main eval + train), 11 candidates, no continuation, from the seed 1 / 2 worlds of the P5 v1 BA /
    P6 v0 sources; eval / train = WL-1's split (its 40 eval base routes never enter training). Times: BA k1, k1 + 0.4 s,
    k2 = k1 + 0.6 s, k3 (0.4 s before the expert's reaction, > k2); P6 the tau = 4 / 3 / 2 s ticks and k3; v >= 3 m/s.
    Training: BA only where the x+ source run's ego lane is occupied at k + 2 s (`occ`), capped per set (deterministic
    group subsample); eval takes every fork the rules give. A group needs the openpilot plan at its fork frame in both
    worlds: the missing streams are listed in plan_keys.txt (p5_openpilot --keys @file) and the call returns without
    writing the job table until they exist. -> runs/wl2/{forks.parquet, jobs.json, forks-<set>.xml, split.json, forks_info.json}"""
    import xml.etree.ElementTree as ET
    from . import nq4_w as W
    from .wl_traj import ACTIONS11
    sp = json.loads(data_dir().joinpath(*WL1_SPLIT).read_text())
    src = _sources(seeds)
    plans = {s: _op_plans(s) for s in SRC}
    rows, miss = [], []
    for r in src.itertuples():
        try:
            dp, dm = attempt_dir(r.set, r.plus), attempt_dir(r.set, r.minus)
        except FileNotFoundError:
            miss.append((r.base_id, r.seed))
            continue
        pp, fp, fm = _pose(dp), _frames(dp), _frames(dm)
        if r.set == "ba":
            k1 = _cam_ceil(r.t_vis)
            ks = {"k1": k1, "k1b": k1 + K1B_TICKS, "k2": k1 + 12}
        else:
            ks = {f"tau{int(t)}": k for t, k in sorted(_p6_tau_ticks(dp, pp).items(), reverse=True)}
        if np.isfinite(r.t_react):
            k3 = _cam_floor(r.t_react - 8)
            if k3 >= max(ks.values(), default=0) + 4:
                ks["k3"] = k3
        seen = set()
        for name, k in ks.items():
            if k in seen or k not in pp.index or pp.v[k] < V_MIN or k not in fp.index or k not in fm.index:
                continue
            seen.add(k)
            for world, rid, fr in (("plus", r.plus, fp), ("minus", r.minus, fm)):
                rows.append({"set": r.set, "cls": r.cls, "family": r.family, "base_id": r.base_id, "seed": r.seed, "world": world,
                             "src_route": rid, "src_dir": str(attempt_dir(r.set, rid)), "k_name": name, "fork_tick": int(k),
                             "v0": float(pp.v[k]), "src_frame": f"{rid}-{int(fr.frame[k]):07d}",
                             "split": "eval" if r.base_id in sp["eval"] else "train"})
    f = pd.DataFrame(rows)
    f["group"] = f.set + "/" + f.base_id + "/" + f.seed.astype(str) + "/" + f.k_name
    # training selection: BA groups whose x+ source lane is occupied at k + 2 s, then the per-set cap
    tr = f[f.split == "train"]
    occ = {}
    for (adir, world), g in tr[(tr.set == "ba") & (tr.world == "plus")].groupby(["src_dir", "world"]):
        frs = _frames(Path(adir))
        want = {int(k): int(frs.frame[k + int(OCC_AT_S / TICK)]) for k in g.fork_tick if k + int(OCC_AT_S / TICK) in frs.index}
        if want:
            res = W._run_rows((adir, sorted(want.values())))[1]
            fo = {r_["frame"]: bool(r_["occ"]) for r_ in res}
            for k, fr_ in want.items():
                occ[(adir, k)] = fo.get(fr_, False)
    keep = set(f[f.split == "eval"].group)
    rng = np.random.RandomState(seed)
    for s_ in SRC:
        g = tr[tr.set == s_]
        if s_ == "ba":
            gp = g[g.world == "plus"]
            grp = sorted(gp[[occ.get((a, k), False) for a, k in zip(gp.src_dir, gp.fork_tick)]].group.unique())
        else:
            grp = sorted(g.group.unique())
        n_row = g.groupby("group").size()
        order = [grp[i] for i in rng.permutation(len(grp))]
        tot = 0
        for gname in order:
            if tot + n_row[gname] > WL2_CAPS[s_]:
                continue
            keep.add(gname)
            tot += n_row[gname]
    f = f[f.group.isin(keep)].copy()
    f["has_op"] = [fn in plans[s_] for s_, fn in zip(f.set, f.src_frame)]
    have_run = {s_: {k.rsplit("-", 1)[0] for k in plans[s_]} for s_ in SRC}          # runs with a stored plan stream
    gone = set(f[~f.has_op].group)
    need = f[f.group.isin(gone) & ~f.has_op & (f.set == "ba")]
    keys = sorted({"p5_" + str(r_) for r_ in need.src_route if str(r_) not in have_run["ba"]})
    rundir("plan_keys.txt").write_text("\n".join(keys))
    if keys:
        return {"status": "openpilot plans missing", "streams": len(keys), "groups": len(gone),
                "run": "scripts/p5_openpilot.py --models cinque --arrays plan temporal --out-sub op_streams_plan --keys @runs/wl2/plan_keys.txt"}
    n_no_op = len(gone)                                    # groups whose stream exists but not the fork frame: dropped
    f = f[~f.group.isin(gone)]
    f = f.sort_values(["set", "split", "seed", "base_id", "fork_tick", "world"], key=lambda c: c.map({"eval": 0, "train": 1}) if c.name == "split" else c
                      ).reset_index(drop=True)
    f["fork_id"] = np.arange(len(f))
    runs = f.loc[f.index.repeat(len(ACTIONS11))].reset_index(drop=True)
    runs["action"] = np.tile(ACTIONS11, len(f))
    runs["route_id"] = ["7%06d%s" % (i, s_[-1]) for i, s_ in enumerate(runs.src_route)]
    jobs = {}
    for r in runs.itertuples():
        op = plans[r.set].get(r.src_frame)
        jobs[r.route_id] = {"fork_tick": r.fork_tick, "action": r.action, "op_plan": np.round(op, 4).tolist(), "branch_s": branch_s,
                            "cont_s": 0.0, "iv_s": 2.0, "iv_gap": [6.0, 10.0], "seed": int(r.route_id), "src": r.src_dir,
                            "fork_id": int(r.fork_id)}
    for s_ in SRC:
        root = ET.parse(data_dir() / SRC[s_]["xml"]).getroot()
        by_id = {e.get("id"): e for e in root.iter("route")}
        out = ET.Element("routes")
        for r in runs[runs.set == s_].itertuples():
            e = ET.fromstring(ET.tostring(by_id[r.src_route]))
            e.set("id", r.route_id)
            e.set("wl_src", r.src_route)
            out.append(e)
        ET.ElementTree(out).write(rundir(f"forks-{s_}.xml"))
    runs.to_parquet(rundir("forks.parquet"), index=False)
    rundir("jobs.json").write_text(json.dumps(jobs))
    rundir("split.json").write_text(json.dumps(sp, indent=1))
    info = {"sources": len(src), "missing_source_runs": miss, "fork_points": len(f), "runs": len(runs), "dropped_no_op_groups": n_no_op,
            "by_set_split": {f"{a}/{b}": int(n) for (a, b), n in f.groupby(["set", "split"]).size().items()},
            "by_k": f.k_name.value_counts().to_dict(),
            "groups_by_set_split": {f"{a}/{b}": int(n) for (a, b), n in f.groupby(["set", "split"]).group.nunique().items()},
            "routes_by_cls_split": {f"{a}/{b}": int(n) for (a, b), n in f.groupby(["cls", "split"]).base_id.nunique().items()}}
    rundir("forks_info.json").write_text(json.dumps(info, indent=1, default=str))
    return info


# ================================================================ WL-2 pilot checklist (todos/2026-09-29-wl2-prereg.md, "分级启动")

def _check_run(args) -> dict:
    """One finished fork run: outcome labels, retries, wall time, and the ego-lane `occ` label at fork + 2 s."""
    from . import nq4_w as W
    rid, adir, out_set, k = args
    a = Path(adir)
    rec = {"route_id": rid}
    try:
        rec.update(outcome(a, k))
        fr = _frames(a)
        t2 = k + int(OCC_AT_S / TICK)
        if t2 in fr.index:
            rec["occ2s"] = bool(W._run_rows((str(a), [int(fr.frame[t2])]))[1][0]["occ"])
        d = json.loads((a.parent.parent.parent / "done" / f"{rid}.json").read_text())
        pr = d.get("profile", {})
        loop_ms = sum(pr.get(f"{m}_ms_mean", 0.0) for m in ("world_tick", "agent", "provider"))
        rec.update(attempt=int(d["attempt"]), wall_s=float(d["wall_s"]), ticks_used=int(pr.get("ticks_used", 0)),
                   setup_s=float(d["wall_s"]) - pr.get("ticks_used", 0) * loop_ms / 1e3,
                   n_attempts=len([x for x in a.parent.iterdir() if x.is_dir()]))
    except Exception as e:                                   # a harness failure of this run
        rec["error"] = repr(e)
    return rec


def check2(out: str, stage: str, workers: int = 24) -> dict:
    """The WL-2 pilot checklist on the finished runs of a stage (items 1-8 of the prereg's table). Description +
    pass / fail per item -> RESULTS/check2_<stage>.json, runs_<stage>.csv."""
    from multiprocessing import Pool
    runs = stage_ids(stage)
    todo = [(r.route_id, str(a), r.set, int(r.fork_tick)) for r in runs.itertuples()
            if (a := _fork_attempt(Path(out) / r.set, r.route_id)) is not None]
    with Pool(workers) as p:
        t = pd.DataFrame(p.map(_check_run, todo, chunksize=4))
    t = runs.drop(columns=["src_dir"]).merge(t, on="route_id", how="left")
    t["done"] = t.get("gap_min_m", pd.Series(np.nan, index=t.index)).notna() | t.get("attempt", pd.Series(np.nan, index=t.index)).notna()
    err = t.get("error", pd.Series(np.nan, index=t.index)).notna()
    dr = drops(out, gate_min=DROP_SET_MIN)
    gone = {d["fork_id"] for d in dr["dropped_groups"]}
    fin = t[t.done & ~err & ~t.fork_id.isin(gone)].copy()
    chk, n = {}, len(t)
    chk["runs"] = {"planned": n, "done": int(t.done.sum()), "errors": int(err.sum()), "harness_fail": float(1 - t.done.mean() + err.mean())}
    chk["retries"] = {"attempt_gt1": float((t.attempt > 1).mean()), "mean_attempts": float(t.n_attempts.mean()),
                      "ended_early": float((fin.ticks < fin.fork_tick + 60).mean())}
    chk["timing"] = {"wall_s_mean": float(t.wall_s.mean()), "wall_s_median": float(t.wall_s.median()), "setup_s_median": float(t.setup_s.median()),
                     "worker_h_per_run": float(t.wall_s.mean() / 3600), "worker_h_per_run_incl_retries": float((t.wall_s * t.n_attempts).mean() / 3600),
                     "by_set_wall_s": {k: float(v) for k, v in t.groupby("set").wall_s.mean().items()}}
    chk["prefix"] = {k: dr[k] for k in ("checked", "dropped", "frac", "per_set")} | {
        "runs_ego_le_1cm": float((lambda e: (e <= PREFIX_POS_M).mean())(pd.read_parquet(rundir("prefix_check.parquet")).set_index("route_id").ego_max_dpos_m.reindex(t.route_id).dropna()))}
    chk["render"] = {k: dr[k] for k in ("render_checked", "render_dropped", "render_frac")}
    lab = fin.groupby("action")[["unsafe_cg", "unsafe", "collision"]].mean().round(3)
    chk["labels_by_action"] = lab.to_dict("index")
    chk["labels_finite"] = float(np.isfinite(fin[["gap_min_m", "travel_m", "lateral_right_m"]].replace(np.inf, 1e3)).all(axis=1).mean())
    piv = lambda col: fin.pivot_table(index="fork_id", columns="action", values=col)                    # noqa: E731
    tr = piv("travel_m")
    chk["brake_travel_lt_hold"] = float((tr.brake_hard < tr.hold).mean())
    chk["brake_mild_between_stop_and_hold"] = float(((tr.op_stop <= tr.brake_mild + 0.05) & (tr.brake_mild <= tr.hold + 0.05)).mean())
    chk["brake_mild_le_op"] = float((tr.brake_mild <= tr.op + 0.05).mean())      # descriptive: fails where op already brakes harder than 2 m/s2
    chk["shift_slow_travel_lt_shift"] = {s_: float((tr[s_ + "_slow"] < tr[s_]).mean()) for s_ in ("shift_L", "shift_R")}
    sh = shift_offsets(out, stage, ("shift_L", "shift_R", "shift_L_slow", "shift_R_slow", "nudge_L"))
    for a_, g in sh.groupby("action"):
        g = g[g.t_eval_s >= 1.0]
        f2 = g[g.t_eval_s >= 2.0]
        chk[f"offset_{a_}"] = {"n": len(g), "sign_ok": float(g.sign_ok.mean()) if len(g) else np.nan,
                               "ge_2m": float((f2.offset_left_m.abs() >= 2.0).mean()) if len(f2) else np.nan,
                               "in_1_2m": float(g.offset_left_m.abs().between(1.0, 2.0).mean()) if len(g) else np.nan}
    xp = fin[fin.world == "plus"]
    u = xp.pivot_table(index="fork_id", columns="action", values="unsafe_cg", aggfunc="max").astype(float)
    solv = (u.op == 1) & (u.drop(columns="op").min(axis=1) == 0)
    fk = xp.groupby("fork_id")[["set", "base_id", "seed", "cls", "split"]].first()
    chk["headroom"] = {"x_plus_fork_points": len(u), "op_unsafe_cg": float((u.op == 1).mean()), "oracle_unsafe_cg": float(u.min(axis=1).mean()),
                       "solvable_share": float(solv.mean()), "solvable_n": int(solv.sum()),
                       "solvable_routes": int(fk[solv.reindex(fk.index).fillna(False)].base_id.nunique())}
    allx = pd.read_parquet(rundir("forks.parquet")).query("world == 'plus' and split == 'eval'").drop_duplicates("fork_id")
    chk["headroom"]["projected_solvable_n_eval_full"] = float(solv.mean() * len(allx))
    o = fin.pivot_table(index="fork_id", columns="action", values="occ2s", aggfunc="max")
    S = [(f_, a_) for f_ in o.index[(o.get("op", pd.Series(dtype=float)) == 1) & o.index.isin(fk.index[fk.split == "eval"])] for a_ in ("shift_L", "shift_R")
         if a_ in o and pd.notna(o.loc[f_, a_])]
    chk["c1c_S"] = {"pairs": len(S), "still_occupied": int(sum(bool(o.loc[f_, a_]) for f_, a_ in S)),
                    "still_occupied_share": float(np.mean([bool(o.loc[f_, a_]) for f_, a_ in S])) if S else np.nan}
    fp = pd.read_parquet(rundir("forks.parquet")).query("set == 'p6' and world == 'plus'").drop_duplicates("fork_id")
    src = fp.groupby(["base_id", "seed"]).size()
    chk["p6_tau_forks_per_source"] = {"sources": len(src), "ge2_share": float((src >= 2).mean()) if len(src) else np.nan}
    def ok(v, lo=None, hi=None):
        return bool(pd.notna(v) and (lo is None or v >= lo) and (hi is None or v <= hi))
    ver = {"1 pose group drops <= 5%": not dr["pose_gate_fail"], "2 render run drops <= 10%": not dr["render_gate_fail"],
           "3 brake_hard < hold": chk["brake_travel_lt_hold"] == 1.0,
           "3 shift sign >= 95%, >= 2 m >= 80%": all(ok(chk.get(f"offset_{a_}", {}).get("sign_ok"), 0.95) and ok(chk.get(f"offset_{a_}", {}).get("ge_2m"), 0.80)
                                                       for a_ in ("shift_L", "shift_R")),
           "4 harness failure <= 5%": chk["runs"]["harness_fail"] <= 0.05,
           "5 brake_mild between op_stop and hold >= 95%": ok(chk["brake_mild_between_stop_and_hold"], 0.95),
           "5 slow shifts sign >= 95%, >= 2 m >= 80%, travel < shift >= 95%": all(
               ok(chk.get(f"offset_{a_}_slow", {}).get("sign_ok"), 0.95) and ok(chk.get(f"offset_{a_}_slow", {}).get("ge_2m"), 0.80)
               and ok(chk["shift_slow_travel_lt_shift"][a_], 0.95) for a_ in ("shift_L", "shift_R")),
           "5 nudge_L offset in 1-2 m >= 95%": ok(chk.get("offset_nudge_L", {}).get("in_1_2m"), 0.95),
           "6 P6 sources with >= 2 x+ forks >= 70%": ok(chk["p6_tau_forks_per_source"]["ge2_share"], 0.70),
           "7 solvable share of x+ >= 25%": ok(chk["headroom"]["solvable_share"], 0.25),
           "8 C1c still-occupied share >= 10%": ok(chk["c1c_S"]["still_occupied_share"], 0.10)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / f"runs_{stage}.csv", index=False, float_format="%.4f")
    (RESULTS / f"check2_{stage}.json").write_text(json.dumps({"checks": chk, "verdict": ver}, indent=1, default=str))
    return {"checks": chk, "verdict": ver}


def d2(seeds=(0, 1), max_s: float = 70.0) -> dict:
    """D2: Bench2Drive 220 routes whose base is not an eval route, TM seeds 0 and 1, PDM-Lite with random windows from
    10 s on (ids 8 + 5-digit base + seed). Appends to jobs.json; writes d2.parquet and forks-d2.xml."""
    import xml.etree.ElementTree as ET
    sp = json.loads(rundir("split.json").read_text())
    b2d = Path(os.environ.get("BENCH2DRIVE_ROOT", data_dir() / "third_party/Bench2Drive")) / "leaderboard/data/bench2drive220.xml"
    root = ET.parse(b2d).getroot()
    out, rows = ET.Element("routes"), []
    jobs = json.loads(rundir("jobs.json").read_text())
    for e in root.iter("route"):
        base = e.get("id")
        if base in sp["eval"]:
            continue
        for sd in seeds:
            rid = "8%05d%d" % (int(base), sd)
            x = ET.fromstring(ET.tostring(e))
            x.set("id", rid)
            out.append(x)
            rows.append({"set": "d2", "route_id": rid, "base_id": base, "seed": sd, "split": "train"})
            jobs[rid] = {"fork_tick": None, "action": None, "op_plan": None, "iv_from_s": 10.0, "max_s": max_s, "iv_s": 2.0,
                         "iv_gap": [6.0, 10.0], "seed": int(rid), "pre_cams": 100000}
    ET.ElementTree(out).write(rundir("forks-d2.xml"))
    pd.DataFrame(rows).to_parquet(rundir("d2.parquet"), index=False)
    rundir("jobs.json").write_text(json.dumps(jobs))
    return {"d2_runs": len(rows), "routes": len(rows) // len(seeds)}


# ================================================================ stages

def stage_ids(stage: str) -> pd.DataFrame:
    """pilot1: one P5 pedestrian x+ fork point with fork tick >= 200 x 7 actions; pilot10: every fork point of 10 routes
    (P5 pedestrian 4, cut-in 2, P6 4, training split); full: everything."""
    runs = pd.read_parquet(rundir("forks.parquet"))
    if stage == "full":
        return runs
    if NAME != "wl":
        return _stage_ids2(runs, stage)
    rng = np.random.RandomState(SPLIT_SEED + 1)
    if stage == "pilot1":
        # a P5 pedestrian x+ fork point late in its run (fork tick >= 200, the earliest such), so that a
        # no-rendering prefix has room before the first saved frame
        c = runs[(runs.cls == "ped") & (runs.world == "plus") & runs.has_op & (runs.fork_tick >= 200)]
        return runs[runs.fork_id == c.sort_values(["fork_tick", "fork_id"]).fork_id.iloc[0]]
    pick = []
    for cls, n in (("ped", 4), ("cutin", 2), ("obstacle", 4)):
        b = np.array(sorted(runs[(runs.cls == cls) & (runs.split == "train") & runs.has_op].base_id.unique()))
        pick += list(b[rng.permutation(len(b))][:n])
    return runs[runs.base_id.isin(pick)]


def _stage_ids2(runs: pd.DataFrame, stage: str) -> pd.DataFrame:
    """WL-2 pilots (both on the eval side, so their runs count towards the full set): pilot1 = the first BA pedestrian
    x+ fork point with fork tick >= 100 x 11 branches; pilot10 = every fork point of 10 sources (BA pedestrian 4,
    cut-in 2, P6 4; a source = base route x seed) x 11 branches."""
    ev = runs[runs.split == "eval"]
    if stage == "pilot1":
        c = ev[(ev.cls == "ped") & (ev.world == "plus") & (ev.fork_tick >= 100)]
        return runs[runs.fork_id == c.fork_id.min()]
    rng = np.random.RandomState(SPLIT_SEED + 2)
    pick = []
    for cls, n in (("ped", 4), ("cutin", 2), ("obstacle", 4)):
        b = sorted(set(zip(ev[ev.cls == cls].base_id, ev[ev.cls == cls].seed)))
        pick += [b[i] for i in rng.permutation(len(b))[:n]]
    return runs[np.array([(b, sd) in pick for b, sd in zip(runs.base_id, runs.seed)]) & (runs.split == "eval").to_numpy()]


def ids(stage: str, set_name: str, out: str) -> str:
    if set_name == "d2":
        r = pd.read_parquet(rundir("d2.parquet")) if stage == "full" else pd.DataFrame({"route_id": []})
    else:
        r = stage_ids(stage)
        r = r[r.set == set_name]
    done = {p.stem for p in (Path(out) / "done").glob("*.json")} if (Path(out) / "done").exists() else set()
    return ",".join(x for x in r.route_id if x not in done)


# ================================================================ checks

def _fork_attempt(out: Path, rid: str) -> Path | None:
    f = out / "done" / f"{rid}.json"
    if not f.exists():
        return None
    return out / "attempts" / rid / str(json.loads(f.read_text())["attempt"])


def _actors_by_tick(adir: Path, t_max: int) -> pd.DataFrame:
    z = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    fr0 = int(_frames(adir).frame.iloc[0])
    t = pd.DataFrame({"tick": z["frame"] - fr0 + 1, "id": z["id"], "x": z["xyz"][:, 0], "y": z["xyz"][:, 1]})
    t["type"] = t.id.map(lambda i: kinds.get(str(i), ["?"])[0])
    return t[t.tick < t_max]


def _match_actors(a: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Server actor ids differ between runs: map each fork-run actor to the source actor of the same type nearest to it
    at the fork-run actor's first tick, then join all ticks on (tick, mapped id)."""
    first = a.sort_values("tick").drop_duplicates("id")
    mp = {}
    for r in first.itertuples():
        c = s[(s.tick == r.tick) & (s.type == r.type)]
        if len(c):
            j = int(np.argmin(np.hypot(c.x - r.x, c.y - r.y)))
            mp[r.id] = int(c.id.iloc[j])
    a = a.assign(sid=a.id.map(mp))
    return a.dropna(subset=["sid"]).astype({"sid": int}).merge(s, left_on=["tick", "sid"], right_on=["tick", "id"],
                                                                 suffixes=("_a", "_s"))


def prefix(out: str, stage: str = "pilot1") -> pd.DataFrame:
    """Every finished fork run of the stage against its source run on the ticks before the fork: max ego position
    difference, max actor position difference (actors matched by type and position, ids differ between servers),
    and the saved JPEGs before the fork against the source's (byte-identical count, mean |pixel difference|)."""
    from PIL import Image
    runs = stage_ids(stage)
    rows = []
    for r in runs.itertuples():
        a = _fork_attempt(Path(out) / r.set, r.route_id)
        if a is None:
            continue
        s = Path(r.src_dir)
        pa, ps = _pose(a), _pose(s)
        k = min(r.fork_tick - 1, pa.index.max(), ps.index.max())
        dpos = float(np.hypot(pa.x[:k].to_numpy() - ps.x[:k].to_numpy(), pa.y[:k].to_numpy() - ps.y[:k].to_numpy()).max())
        m = _match_actors(_actors_by_tick(a, r.fork_tick), _actors_by_tick(s, r.fork_tick))
        n_ids = _actors_by_tick(a, r.fork_tick).id.nunique()
        dact = float(np.hypot(m.x_a - m.x_s, m.y_a - m.y_s).max()) if len(m) else np.nan
        fa, fs = _frames(a), _frames(s)
        same, n_img, mad = 0, 0, []
        for t in fa.index[fa.index < r.fork_tick]:
            files = fa.files[t]
            if not files or t not in fs.index:
                continue
            for cam, rel in files.items():
                n_img += 1
                ba, bs = (a / rel).read_bytes(), (s / fs.files[t][cam]).read_bytes()
                same += ba == bs
                if n_img <= 12 or ba != bs:
                    ia = np.asarray(Image.open(a / rel), np.int16)
                    ib = np.asarray(Image.open(s / fs.files[t][cam]), np.int16)
                    mad.append(float(np.abs(ia - ib).mean()))
        rows.append({"route_id": r.route_id, "action": r.action, "fork_tick": r.fork_tick, "ego_max_dpos_m": dpos,
                     "actor_max_dpos_m": dact, "actors_matched": int(m.id_a.nunique()) if len(m) else 0, "actors": n_ids,
                     "images_compared": n_img, "images_identical": same,
                     "pixel_mad_mean": float(np.mean(mad)) if mad else np.nan, "pixel_mad_max": float(np.max(mad)) if mad else np.nan})
    t = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / f"prefix_{stage}.csv", index=False, float_format="%.4f")
    return t


def prefix_feats(out: str, stage: str = "pilot1", step: str = "spec") -> pd.DataFrame | dict:
    """The registered prefix item: Cinque `temporal` cosine >= 0.999 on every pre-fork frame, plus V-JEPA 2 `mean` cosine
    (description). step "spec": processed/wl_check_<out dir name>/op_plan.json, one stream per fork run = the source run's
    frames before the run's first saved frame + the run's own frames up to the fork (then run
    P5_SET=wl_check_<name> scripts/p5_openpilot.py --models cinque --arrays temporal --out-sub op_streams_vis);
    step "read": both cosines against the source run's stored streams / features at the same ticks."""
    from . import nq4_w as W
    from .p5_openpilot import carla_calib
    name = "wl_check_" + Path(out).name
    d = data_dir() / "processed" / name
    d.mkdir(parents=True, exist_ok=True)
    runs = stage_ids(stage)
    cams = ("front", "front_left", "front_right")
    if step == "spec":
        streams = []
        for r in runs.itertuples():
            a = _fork_attempt(Path(out) / r.set, r.route_id)
            if a is None:
                continue
            fa, fs = _frames(a), _frames(Path(r.src_dir))
            fa = fa[fa.files.map(bool) & (fa.index < r.fork_tick)]
            src = fs[fs.index < fa.index.min()]
            names = [f"{r.src_route}-{f:07d}" for f in src.frame] + [f"{r.route_id}-{f:07d}" for f in fa.frame]
            files = [[f"{r.src_dir}/{x[c]}" for c in cams] for x in src.files] + [[f"{a}/{x[c]}" for c in cams] for x in fa.files]
            streams.append({"key": f"wl_{r.route_id}", "names": names, "targets": list(range(len(src), len(names))),
                            "files": files, "gaps": 0})
        (d / "op_plan.json").write_text(json.dumps({"calib": carla_calib(), "streams": streams}))
        return {"streams": len(streams), "dir": str(d)}
    rows = []
    for r in runs.itertuples():
        src_op = data_dir() / "processed" / SRC[r.set]["proc"] / "op_streams_plan" / "cinque"
        pre = "p5" if r.set == "ba" else r.set
        a = _fork_attempt(Path(out) / r.set, r.route_id)
        f = d / "op_streams_vis" / "cinque" / f"wl_{r.route_id}.npz"
        if a is None or not f.exists():
            continue
        fa, fs = _frames(a), _frames(Path(r.src_dir))
        tick_of = {f"{r.route_id}-{fr:07d}": t for t, fr in zip(fa.index, fa.frame)}
        name_src = {t: f"{r.src_route}-{fr:07d}" for t, fr in zip(fs.index, fs.frame)}
        q = np.load(f)
        so = np.load(src_op / f"{pre}_{r.src_route}.npz") if (src_op / f"{pre}_{r.src_route}.npz").exists() else None
        ref = dict(zip(so["name"], so["temporal"])) if so is not None else {}
        cos = []
        for n, v in zip(q["name"], q["temporal"]):
            u = ref.get(name_src.get(tick_of.get(str(n))))
            if u is not None:
                cos.append(float(v @ u / np.linalg.norm(v) / np.linalg.norm(u)))
        rows.append({"route_id": r.route_id, "action": r.action, "op_frames_compared": len(cos),
                     "op_cos_min": min(cos) if cos else np.nan, "op_cos_mean": float(np.mean(cos)) if cos else np.nan})
    t = pd.DataFrame(rows)
    # V-JEPA 2 on the pre-fork frames with 3 predecessors inside the run, against the same ticks of the source run
    from . import features as F
    fx = F.VJepaFeatures(frames=4)
    vrows = []
    for r in runs.itertuples():
        a = _fork_attempt(Path(out) / r.set, r.route_id)
        if a is None:
            continue
        fa, fs = _frames(a), _frames(Path(r.src_dir))
        ta = [t for t in fa.index if fa.files[t] and t < r.fork_tick]
        ta = [t for t in ta if all(t - 4 * j in ta for j in range(4)) and t in fs.index and all(t - 4 * j in fs.index for j in range(4))]
        if not ta:
            continue
        mk = lambda root, fr, t: sum(([f"{root}/{fr.files[t - 4 * (3 - j)][c]}" for j in range(4)] for c in cams), [])
        tab = pd.DataFrame({"files": [mk(a, fa, t) for t in ta] + [mk(r.src_dir, fs, t) for t in ta]})
        z = W._extract_rows(tab, fx, 16, 4).astype(np.float32)
        za, zs = z[: len(ta)], z[len(ta):]
        c = (za * zs).sum(1) / np.linalg.norm(za, axis=1) / np.linalg.norm(zs, axis=1)
        vrows.append({"route_id": r.route_id, "vjepa_frames_compared": len(ta), "vjepa_cos_min": float(c.min()),
                      "vjepa_cos_mean": float(c.mean())})
    t = t.merge(pd.DataFrame(vrows), on="route_id", how="outer") if vrows else t
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / f"prefix_feats_{stage}.csv", index=False, float_format="%.6f")
    return t


def _gap_front(adir: Path, t0: int, t1: int) -> dict:
    """Min longitudinal gap to any road actor in the ego lane (|y| <= 1.75 m, ego frame) over ticks [t0, t1], and the
    ego's travel, and the min time to collision with those actors (gap / closing speed along the ego heading). Actor rows
    come from actors.npz, tick-aligned through frames.jsonl's first frame."""
    p = _pose(adir)
    fr0 = int(_frames(adir).frame.iloc[0])
    z = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    road = {int(k) for k, v in kinds.items() if v[0].startswith(("walker.", "vehicle.", "static.prop."))}
    tick = z["frame"] - fr0 + 1
    gap, ttc = np.inf, np.inf
    for t in range(t0, min(t1, p.index.max()) + 1):
        e = p.loc[t]
        m = (tick == t) & np.isin(z["id"], list(road))
        if not m.any():
            continue
        q = z["xyz"][m].astype(float)
        dx, dy = q[:, 0] - e.x, q[:, 1] - e.y
        c, s = np.cos(np.radians(e.yaw)), np.sin(np.radians(e.yaw))
        x, y = dx * c + dy * s, -dx * s + dy * c
        ok = (x > 0) & (np.abs(y) <= 1.75) & (np.abs(q[:, 2] - e.z) <= 5) & (np.hypot(dx, dy) > 2.0)
        if ok.any():
            g = x[ok] - 2.4                                   # front bumper to the actor's reference point
            gap = min(gap, float(g.min()))
            va = z["v"][m][ok].astype(float)                  # actor velocity along the ego heading
            closing = (e.vx * c + e.vy * s) - (va[:, 0] * c + va[:, 1] * s)
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = np.where(closing > 0.1, np.maximum(g, 0.0) / closing, np.inf)
            ttc = min(ttc, float(tt.min()))
    k1 = min(t1, p.index.max())
    travel = float(np.hypot(np.diff(p.x[t0:k1]), np.diff(p.y[t0:k1])).sum())
    return {"gap_min_m": gap, "ttc_min_s": ttc, "travel_m": travel}


def outcome(adir: Path, fork_tick: int, branch_s: float = 3.0) -> dict:
    """Branch outcome: collision in the branch (collisions.jsonl, tick-aligned), min in-lane gap, ego travel, lateral
    offset from the fork pose's heading line at the branch end; unsafe = collision or gap < 2 m or TTC < 1.0 s."""
    t1 = fork_tick + int(round(branch_s / TICK))
    fr0 = int(_frames(adir).frame.iloc[0])
    col = [json.loads(l) for l in (adir / "collisions.jsonl").read_text().splitlines()] if (adir / "collisions.jsonl").exists() else []
    hit = [c for c in col if fork_tick <= c["frame"] - fr0 + 1 <= t1]
    g = _gap_front(adir, fork_tick, t1)
    p = _pose(adir)
    e0, e1 = p.loc[fork_tick], p.loc[min(t1, p.index.max())]
    c, s = np.cos(np.radians(e0.yaw)), np.sin(np.radians(e0.yaw))
    lat = float(-(e1.x - e0.x) * s + (e1.y - e0.y) * c)       # CARLA y right: positive = right of the start heading
    first_hit = min((c["frame"] - fr0 + 1 for c in hit), default=None)
    road_hit = [h for h in hit if h["other_type"].startswith(("vehicle.", "walker."))]
    return {"collision": bool(hit), "collision_road": bool(road_hit), "first_collision_tick": first_hit,
            "collision_types": sorted({h["other_type"] for h in hit}), **g, "lateral_right_m": lat,
            "unsafe": bool(hit) or g["gap_min_m"] < 2.0 or g["ttc_min_s"] < 1.0,
            "unsafe_cg": bool(hit) or g["gap_min_m"] < 2.0,           # WL-2's main label: collision or in-lane gap < 2 m (no TTC term)
            "ticks": int(p.index.max())}


def shift_offsets(out: str, stage: str, actions=("shift_L", "shift_R")) -> pd.DataFrame:
    """Checklist item "shift": the executed shift branch's rear-axle position against the op candidate's path (the path it
    shifts), as a signed left offset along that path's local normal, at t = min(3 s, first collision - 1 tick): after a
    collision the car is held by the obstacle and says nothing about the shift. Sign correct = left for shift_L."""
    from .wl_traj import T as TT, world_to_ego
    runs = stage_ids(stage)
    rows = []
    for r in runs[runs.action.isin(list(actions))].itertuples():
        a = _fork_attempt(Path(out) / r.set, r.route_id)
        if a is None or not (a / "wl.json").exists():
            continue
        c = json.loads((a / "wl.json").read_text())["cands"][0]
        tk = pd.read_json(a / "wl_ticks.jsonl", lines=True).set_index("tick")
        o = outcome(a, r.fork_tick)
        t_end = r.fork_tick + 60 if o["first_collision_tick"] is None else min(r.fork_tick + 60, o["first_collision_tick"] - 1)
        t_end = max(t for t in tk.index if t <= t_end) if (tk.index <= t_end).any() else None
        if t_end is None:
            continue
        p = world_to_ego(np.array([tk.rear_xy[t_end]]), np.array(c["rear_xy"]), c["yaw"])[0]
        op = np.vstack([[0.0, 0.0], np.array(c["cands"]["op"])])
        i = int(np.argmin(np.hypot(*(op - p).T)))
        j = min(i + 1, len(op) - 1)
        d = op[j] - op[max(j - 1, 0)]
        n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
        off = float((p - op[i]) @ n)
        rows.append({"route_id": r.route_id, "action": r.action, "t_eval_s": (t_end - r.fork_tick) * TICK,
                     "offset_left_m": off, "sign_ok": off > 0 if "_L" in r.action else off < 0})
    return pd.DataFrame(rows)


# DROP_SET_MIN: the per-set drop fraction is judged once a set has this many checked groups (one drop in stage 3's
# seven P6 groups is 14 %); the overall fraction always, and the full set at its end with no minimum (scripts/wl_full.sh).
PREFIX_POS_M, DROP_MAX, DROP_SET_MIN = 0.01, 0.05, 20
# Checklist amendment (c), 2026-09-29 (user-approved, todos/2026-09-28-wm-loop.md): the per-run render gate replacing
# the old group cosine >= 0.999 gate. RENDER_LUMA_DIFF / RENDER_LUMA_FRAC: a run is render-bad when more than
# RENDER_LUMA_FRAC of its checked frames differ in front-camera brightness by more than RENDER_LUMA_DIFF (0-255).
# RENDER_COS: prefix `temporal` cosine floor (a fault-free run's own p1 measured at 0.9545). RENDER_DROP_MAX: the new
# stop line on render-dropped runs. PAIR_ACTIONS: a fork point leaves the C1/C2 paired readouts only when one of
# these branches is render-dropped; its other branches still train (wl_data.finished drops by run, not by group).
RENDER_COS, RENDER_LUMA_DIFF, RENDER_LUMA_FRAC, RENDER_DROP_MAX = 0.95, 10.0, 0.5, 0.10
PAIR_ACTIONS = ("op", "hold", "brake_hard")


def _prefix_pose(args) -> dict:
    rid, adir, src, k = args
    try:
        pa, ps = _pose(Path(adir)), _pose(Path(src))
        n = min(int(k) - 1, pa.index.max(), ps.index.max())
        d = np.hypot(pa.x[:n].to_numpy() - ps.x[:n].to_numpy(), pa.y[:n].to_numpy() - ps.y[:n].to_numpy())
        return {"route_id": rid, "ego_max_dpos_m": float(d.max())}
    except Exception as e:
        return {"route_id": rid, "ego_max_dpos_m": np.inf, "error": repr(e)}


def _prefix_cos(args) -> dict:
    """Min Cinque `temporal` cosine of the run's own pre-fork frames against the source run's stored stream at the same
    ticks (processed/wl_gen/op_streams_vis from the wl_data op step; NaN until that stream exists)."""
    rid, adir, src, src_route, set_name, k = args
    f = data_dir() / "processed" / "wl_gen" / "op_streams_vis" / "cinque" / f"wl_{rid}.npz"
    pre = "p5" if set_name == "ba" else set_name
    g = data_dir() / "processed" / SRC[set_name]["proc"] / "op_streams_plan" / "cinque" / f"{pre}_{src_route}.npz"
    if not f.exists() or not g.exists():
        return {"route_id": rid, "op_cos_min": np.nan}
    fa, fs = _frames(Path(adir)), _frames(Path(src))
    tick_of = {f"{rid}-{fr:07d}": t for t, fr in zip(fa.index, fa.frame)}
    name_src = {t: f"{src_route}-{fr:07d}" for t, fr in zip(fs.index, fs.frame)}
    with np.load(g) as z:
        ref = dict(zip(z["name"], z["temporal"]))
    cos = []
    with np.load(f) as q:
        for n, v in zip(q["name"], q["temporal"]):
            t = tick_of.get(str(n))
            u = ref.get(name_src.get(t)) if t is not None and t < k else None
            if u is not None:
                cos.append(float(v @ u / np.linalg.norm(v) / np.linalg.norm(u)))
    return {"route_id": rid, "op_cos_min": min(cos) if cos else np.nan, "op_frames": len(cos)}


def _luma(path: Path) -> float:
    """Mean grayscale value (0-255) of a saved camera JPEG."""
    from PIL import Image
    return float(np.asarray(Image.open(path).convert("L"), dtype=np.float32).mean())


def _prefix_luma(args) -> dict:
    """Amendment (c) item 2: front-camera mean brightness of every pre-fork frame against the source run at the same
    tick. Paired with _prefix_cos into the per-run render gate (fork branches only; a D2 run has no source)."""
    rid, adir, src, k = args
    a, s = Path(adir), Path(src)
    fa, fs = _frames(a), _frames(s)
    diffs = []
    for t in fa.index[fa.index < k]:
        fa_f, fs_f = fa.files.get(t), (fs.files.get(t) if t in fs.index else None)
        if not fa_f or not fs_f or "front" not in fa_f or "front" not in fs_f:
            continue
        diffs.append(abs(_luma(a / fa_f["front"]) - _luma(s / fs_f["front"])))
    return {"route_id": rid, "luma_bad_frac": float(np.mean([d > RENDER_LUMA_DIFF for d in diffs])) if diffs else np.nan,
            "luma_frames": len(diffs)}


def _run_luma(args) -> dict:
    """Amendment (c) item 5, the whole-run brightness scan: front-camera mean brightness of every saved frame of a run
    against its own baseline (median over its pre-fork frames for a fork branch, its first 100 ticks / 5 s for a D2
    run, which has neither a fork nor a source). Needs no source run, so it catches a render failure whose onset is
    after the checked prefix (fork branches) and a D2 run entirely (380 of them, invisible to _prefix_luma / _prefix_cos)."""
    rid, adir, k = args
    fa = _frames(Path(adir))
    lum = {t: _luma(Path(adir) / f["front"]) for t, f in fa.files.items() if f and "front" in f}
    if not lum:
        return {"route_id": rid, "whole_bad_frac": np.nan, "whole_frames": 0}
    has_fork = k is not None and not (isinstance(k, float) and np.isnan(k))
    base_ticks = [t for t in lum if t < k] if has_fork else [t for t in lum if t <= 100]
    base = float(np.median([lum[t] for t in (base_ticks or lum)]))
    bad = [abs(v - base) > RENDER_LUMA_DIFF for v in lum.values()]
    return {"route_id": rid, "whole_bad_frac": float(np.mean(bad)), "whole_frames": len(bad)}


def drops(out: str | None = None, workers: int = 24, gate_min: int = 0) -> dict:
    """Checklist amendments (a) + (c) (todo, 2026-09-28 13:30 / 2026-09-29, user-approved). Pose stays a group gate:
    per fork group, once all its branches are done, any branch > PREFIX_POS_M ego pose drift against its source
    drops the whole group. The render gate is per run instead, over every finished fork branch (D2 has no source, so
    it never enters this part): a run is dropped when its prefix front-camera brightness differs from the source by
    > RENDER_LUMA_DIFF on more than RENDER_LUMA_FRAC of its pre-fork frames, or its prefix `temporal` cosine to the
    source is < RENDER_COS. The whole-run brightness scan (_run_luma, no source needed) is also computed and cached
    for every fork + D2 run, but left OUT of the gate: on the full set its self-baseline (a run's own prefix median)
    flags 27 % of runs with a wide graded distribution, not the clean split of a render failure -- it is reading
    ordinary scene drift as the branches drive into different scenery, not a fault. Its numbers are in the returned
    "whole_scan" (descriptive) until a reference-free method is validated; res()["gate_fail"] does not depend on it.
    Results are cached per run in runs/wl/prefix_check.parquet, each metric filled once. --gate (main) fails when the
    pose-dropped groups exceed DROP_MAX (overall, or per set once it has >= gate_min checked groups) or the
    render-dropped runs (luma / cosine only) exceed RENDER_DROP_MAX (overall)."""
    from multiprocessing import Pool
    out = Path(out or rundir("gen"))
    forks_t = pd.read_parquet(rundir("forks.parquet"))
    forks_t["adir"] = [_fork_attempt(out / s_, r) for s_, r in zip(forks_t.set, forks_t.route_id)]
    size = forks_t.groupby("fork_id").size()
    ok_n = forks_t[forks_t.adir.notna()].groupby("fork_id").size()
    full = set(ok_n[ok_n == size.reindex(ok_n.index)].index)
    fd = forks_t[forks_t.adir.notna()].copy()                        # render universe (fork side): any finished branch
    fd["adir"] = fd.adir.astype(str)
    fr = fd[fd.fork_id.isin(full)]                                    # pose universe: the whole 7-branch group is done

    d2f = rundir("d2.parquet")
    if d2f.exists():
        d2 = pd.read_parquet(d2f)
        d2["adir"] = [_fork_attempt(out / s_, r) for s_, r in zip(d2.set, d2.route_id)]
        d2 = d2[d2.adir.notna()].copy()
        d2["adir"] = d2.adir.astype(str)
    else:
        d2 = pd.DataFrame(columns=["route_id", "adir"])

    cache_f = rundir("prefix_check.parquet")
    metrics = ("ego_max_dpos_m", "op_cos_min", "luma_bad_frac", "whole_bad_frac")
    cache = pd.read_parquet(cache_f) if cache_f.exists() else pd.DataFrame(
        {"route_id": pd.Series(dtype=str), "adir": pd.Series(dtype=str), **{m: pd.Series(dtype=float) for m in metrics}})
    for m in metrics:                                        # schema migration: a cache from before amendment (c)
        if m not in cache.columns:
            cache[m] = np.nan
    universe = pd.concat([fd[["route_id", "adir"]], d2[["route_id", "adir"]]], ignore_index=True)
    # right join on (route_id, adir): a checked run keeps its cached metrics, a new run (or a re-run attempt, whose
    # adir changed) gets NaN in every metric and is (re)computed below.
    cache = cache.merge(universe, on=["route_id", "adir"], how="right")

    with Pool(workers) as p:
        need_pose = cache[cache.route_id.isin(fr.route_id) & cache.ego_max_dpos_m.isna()].merge(
            fr[["route_id", "src_dir", "fork_tick"]], on="route_id")
        if len(need_pose):
            new = pd.DataFrame(p.map(_prefix_pose, list(zip(need_pose.route_id, need_pose.adir, need_pose.src_dir, need_pose.fork_tick)), chunksize=8))
            cache = cache.set_index("route_id")
            cache.loc[new.route_id, "ego_max_dpos_m"] = new.set_index("route_id").ego_max_dpos_m
            cache = cache.reset_index()

        # cosine and luma are independent per-run reads (Cinque stream vs. JPEGs); kept as separate need-sets so a
        # cache from before amendment (c) reuses its already-computed cosine and only backfills luma.
        need_cos = cache[cache.route_id.isin(fd.route_id) & cache.op_cos_min.isna()].merge(
            fd[["route_id", "src_dir", "src_route", "set", "fork_tick"]], on="route_id")
        if len(need_cos):
            cos = pd.DataFrame(p.map(_prefix_cos, list(zip(need_cos.route_id, need_cos.adir, need_cos.src_dir,
                                                            need_cos.src_route, need_cos.set, need_cos.fork_tick)), chunksize=8))
            cache = cache.set_index("route_id")
            cache.loc[cos.route_id, "op_cos_min"] = cos.set_index("route_id").op_cos_min
            cache = cache.reset_index()

        need_luma = cache[cache.route_id.isin(fd.route_id) & cache.luma_bad_frac.isna()].merge(
            fd[["route_id", "src_dir", "fork_tick"]], on="route_id")
        if len(need_luma):
            lum = pd.DataFrame(p.map(_prefix_luma, list(zip(need_luma.route_id, need_luma.adir, need_luma.src_dir, need_luma.fork_tick)), chunksize=8))
            cache = cache.set_index("route_id")
            cache.loc[lum.route_id, "luma_bad_frac"] = lum.set_index("route_id").luma_bad_frac
            cache = cache.reset_index()

        ft = pd.concat([fd[["route_id", "fork_tick"]], d2[["route_id"]].assign(fork_tick=np.nan)], ignore_index=True)
        need_whole = cache[cache.whole_bad_frac.isna()].merge(ft, on="route_id")
        if len(need_whole):
            whole = pd.DataFrame(p.map(_run_luma, list(zip(need_whole.route_id, need_whole.adir, need_whole.fork_tick)), chunksize=8))
            cache = cache.set_index("route_id")
            cache.loc[whole.route_id, "whole_bad_frac"] = whole.set_index("route_id").whole_bad_frac
            cache = cache.reset_index()
    cache.to_parquet(cache_f, index=False)

    # ---- pose (group) gate, amendment (a), unchanged
    t = fr[["route_id", "fork_id", "set", "cls", "family", "base_id", "k_name", "world"]].merge(
        cache[["route_id", "ego_max_dpos_m"]], on="route_id")
    t["bad"] = t.ego_max_dpos_m > PREFIX_POS_M
    g = t.groupby("fork_id").agg(set=("set", "first"), bad=("bad", "any"), pos=("ego_max_dpos_m", "max"),
                                 family=("family", "first"), base_id=("base_id", "first"),
                                 k_name=("k_name", "first"), world=("world", "first"))
    per = {s_: {"checked": int(len(x)), "dropped": int(x.bad.sum()), "frac": float(x.bad.mean())} for s_, x in g.groupby("set")}
    pose_gate_fail = bool((float(g.bad.mean()) if len(g) else 0.0) > DROP_MAX
                          or any(v["frac"] > DROP_MAX for v in per.values() if v["checked"] >= gate_min))

    # ---- render (per-run) gate, amendment (c)
    # whole_bad_frac is computed and cached (below) but left OUT of the gate for now: on the full set it flags 27 %
    # of runs (vs. 6 % for luma_bad_frac / op_cos_min alone, matching the diagnosis agent's ~5.5 % estimate), a wide
    # graded distribution (median 0.12, p90 0.84) with no clean split -- a run's own prefix-window brightness isn't a
    # valid whole-run baseline once the branches drive into different scenery, so this reads as normal scene drift,
    # not render failure. whole_scan_* below is descriptive only until main picks a real reference-free method.
    ru = cache[cache.route_id.isin(universe.route_id)].copy()
    ru["checked"] = ru[["luma_bad_frac", "op_cos_min"]].notna().any(axis=1)
    ru["bad"] = (ru.luma_bad_frac > RENDER_LUMA_FRAC) | (ru.op_cos_min < RENDER_COS)
    ruc = ru[ru.checked]
    render_frac = float(ruc.bad.mean()) if len(ruc) else 0.0
    render_gate_fail = render_frac > RENDER_DROP_MAX
    whole_checked = cache[cache.whole_bad_frac.notna()]
    whole_scan = {"checked": int(len(whole_checked)),
                  "flagged_gt_50pct": int((whole_checked.whole_bad_frac > RENDER_LUMA_FRAC).sum()),
                  "median_bad_frac": float(whole_checked.whole_bad_frac.median()) if len(whole_checked) else np.nan}

    # ---- amendment (c) item 3: fork points that leave the C1/C2 paired readouts (their other branches still train)
    fa = fd[["route_id", "fork_id", "action"]].merge(ru[["route_id", "bad"]], on="route_id")
    pair_dropped = sorted(int(i) for i in fa[fa.action.isin(PAIR_ACTIONS) & fa.bad.fillna(False)].fork_id.unique())

    res = {"groups_total": int(size.size), "checked": int(len(g)), "dropped": int(g.bad.sum()),
           "frac": float(g.bad.mean()) if len(g) else 0.0, "per_set": per,
           "cos_checked_runs": int(cache.op_cos_min.notna().sum()),
           "dropped_groups": [{"fork_id": int(i), **{k: (v if isinstance(v, str) else float(v)) for k, v in x.items() if k != "bad"}}
                              for i, x in g[g.bad].iterrows()],
           "render_checked": int(len(ruc)), "render_dropped": int(ruc.bad.sum()), "render_frac": render_frac,
           "dropped_route_ids": sorted(ruc[ruc.bad].route_id.tolist()), "pair_dropped_fork_ids": pair_dropped,
           "whole_scan": whole_scan, "pose_gate_fail": pose_gate_fail, "render_gate_fail": render_gate_fail}
    res["gate_fail"] = bool(pose_gate_fail or render_gate_fail)
    rundir("drops.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def dropped_forks() -> set:
    """Fork ids dropped by amendment (a), the pose group gate (runs/wl/drops.json); empty before the first check."""
    f = rundir("drops.json")
    return {d["fork_id"] for d in json.loads(f.read_text())["dropped_groups"]} if f.exists() else set()


def dropped_runs() -> set:
    """Route ids dropped by amendment (c), the per-run render gate (runs/wl/drops.json); empty before the first check."""
    f = rundir("drops.json")
    return set(json.loads(f.read_text())["dropped_route_ids"]) if f.exists() else set()


def pair_dropped_forks() -> set:
    """Fork ids whose `op` / `hold` / `brake_hard` branch was render-dropped (amendment (c) item 3): they leave the
    C1/C2 paired readouts; their other branches still train."""
    f = rundir("drops.json")
    return set(json.loads(f.read_text())["pair_dropped_fork_ids"]) if f.exists() else set()


def nonped_both(t: pd.DataFrame) -> set:
    """Checklist amendment (b): pedestrian fork points whose `hold` or `op` branch hits a non-pedestrian actor in BOTH
    worlds (x+ and x- of the same base route and k) -> both fork ids. The hazard there is not the pedestrian's; the
    fork points stay in the data and are listed separately in C3. `t` needs fork_id, base_id, k_name, cls, world,
    action, collision_types (list or JSON string)."""
    x = t[(t.cls == "ped") & t.action.isin(["hold", "op"])].copy()
    ty = x.collision_types.map(lambda v: json.loads(v) if isinstance(v, str) else (list(v) if v is not None and not
                                                                                     (isinstance(v, float) and np.isnan(v)) else []))
    x["nonped"] = ty.map(lambda v: any(not str(c).startswith("walker.") for c in v))
    w = x.groupby(["base_id", "k_name", "world"]).nonped.any().unstack("world")
    if not {"plus", "minus"} <= set(w):
        return set()
    both = w[w.plus.fillna(False).astype(bool) & w.minus.fillna(False).astype(bool)].index
    return set(x[x.set_index(["base_id", "k_name"]).index.isin(both)].fork_id.unique())


def sanity(out: str, stage: str) -> dict:
    """The registered checklist (todo, "分叉数据的 sanity checklist"); every item -> pass / fail."""
    runs = stage_ids(stage)
    rows = []
    for r in runs.itertuples():
        a = _fork_attempt(Path(out) / r.set, r.route_id)
        rec = {"route_id": r.route_id, "fork_id": r.fork_id, "base_id": r.base_id, "set": r.set, "cls": r.cls, "family": r.family, "world": r.world,
               "k_name": r.k_name, "action": r.action, "done": a is not None}
        if a is not None:
            try:
                rec.update(outcome(a, r.fork_tick))
            except Exception as e:                          # a harness failure of this run
                rec["error"] = repr(e)
        rows.append(rec)
    t = pd.DataFrame(rows)
    pre = prefix(out, stage) if stage != "full" else pd.DataFrame()   # the full set: pose via drops() only
    dr = drops(out, gate_min=DROP_SET_MIN)
    gone = {d["fork_id"] for d in dr["dropped_groups"]}
    fin = t[t.done & t.get("error", pd.Series(np.nan, index=t.index)).isna() & ~t.fork_id.isin(gone)]
    npb = nonped_both(fin)
    piv = fin.pivot_table(index="fork_id", columns="action", values="travel_m")
    lat = fin.pivot_table(index="fork_id", columns="action", values="lateral_right_m")
    cls = fin.groupby("fork_id")[["cls", "world"]].first()
    unsafe = fin.pivot_table(index="fork_id", columns="action", values="unsafe", aggfunc="max")
    fin = fin.assign(hit_or_gap=fin.collision.astype(bool) | (fin.gap_min_m < 2.0))
    hog = fin.pivot_table(index="fork_id", columns="action", values="hit_or_gap", aggfunc="max")
    col = fin.pivot_table(index="fork_id", columns="action", values="collision", aggfunc="max")
    chk = {}
    chk["prefix_ego_le_1cm_runs"] = float((pre.ego_max_dpos_m <= 0.01).mean()) if len(pre) else np.nan
    chk["prefix_dropped"] = {k: dr[k] for k in ("checked", "dropped", "frac", "per_set")}
    chk["render_dropped"] = {k: dr[k] for k in ("render_checked", "render_dropped", "render_frac")}
    chk["ped_nonped_both_fork_points"] = sorted(int(i) for i in npb)
    chk["brake_travel_lt_hold"] = float((piv.brake_hard < piv.hold).mean()) if {"brake_hard", "hold"} <= set(piv) else np.nan
    sh = shift_offsets(out, stage)
    if len(sh):
        ok = sh[sh.t_eval_s >= 1.0]
        chk["shift_n_evaluated"], chk["shift_n_censored"] = len(ok), int((sh.t_eval_s < 1.0).sum())
        chk["shift_sign_ok"] = float(ok.sign_ok.mean()) if len(ok) else np.nan
        full = ok[ok.t_eval_s >= 2.0]
        chk["shift_ge_2m"] = float((full.offset_left_m.abs() >= 2.0).mean()) if len(full) else np.nan
    # Registered wording: pedestrian items = collision or min in-lane gap < 2 m; P6 items = collision. The `unsafe`
    # label (adds TTC < 1 s) is the model's target and is reported alongside as a description.
    for name, tab in (("hit_or_gap", hog), ("unsafe", unsafe)):
        u = tab.join(cls)
        if {"hold", "op"} <= set(u):
            hold_op = u[["hold", "op"]].astype(float).max(axis=1)
            for c_, w_ in (("ped", "plus"), ("ped", "minus"), ("cutin", "plus"), ("cutin", "minus")):
                m = (u.cls == c_) & (u.world == w_)
                chk[f"{name}_hold_or_op_{c_}_{w_}"] = float(hold_op[m].mean()) if m.any() else np.nan
            m = (u.cls == "ped") & (u.world == "minus") & ~u.index.isin(npb)     # amendment (b): the registered x- item
            chk[f"{name}_hold_or_op_ped_minus_attributable"] = float(hold_op[m].mean()) if m.any() else np.nan
    for name, tab in (("collision", col), ("unsafe", unsafe)):
        u = tab.join(cls)
        m = (u.cls == "obstacle") & (u.world == "plus")
        if m.any() and {"hold", "shift_L", "shift_R"} <= set(u):
            chk[f"obstacle_plus_hold_{name}"] = float(u.hold[m].astype(float).mean())
            chk[f"obstacle_plus_a_shift_no_{name}"] = float((~(u.shift_L[m].astype(bool) & u.shift_R[m].astype(bool))).mean())
    chk["n_fork_points"] = {f"{c_}_{w_}": int(((cls.cls == c_) & (cls.world == w_)).sum())
                            for c_, w_ in cls.drop_duplicates().itertuples(index=False)}
    chk["harness_fail"] = float(1 - t.done.mean() + (t.get("error", pd.Series(np.nan, index=t.index)).notna().mean()))
    verdict = {
        "prefix": not dr["gate_fail"],
        "brake": chk.get("brake_travel_lt_hold") == 1.0,
        "shift": chk.get("shift_sign_ok", 0) >= 0.95 and chk.get("shift_ge_2m", 0) >= 0.80,
        "outcomes": (chk.get("hit_or_gap_hold_or_op_ped_plus", 0) >= 0.20
                     and chk.get("hit_or_gap_hold_or_op_ped_minus_attributable", 1) <= 0.05
                     and chk.get("obstacle_plus_hold_collision", 0) >= 0.30
                     and chk.get("obstacle_plus_a_shift_no_collision", 0) >= 0.50),
        "harness": chk["harness_fail"] <= 0.05}
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / f"runs_{stage}.csv", index=False, float_format="%.4f")
    (RESULTS / f"sanity_{stage}.json").write_text(json.dumps({"checks": chk, "verdict": verdict}, indent=1, default=str))
    return {"checks": chk, "verdict": verdict}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("forks", "forks2", "check2", "d2", "ids", "prefix", "prefix_spec", "prefix_read", "sanity", "drops"))
    ap.add_argument("--gate", action="store_true", help="drops: exit 3 when the dropped fraction exceeds 5 %%")
    ap.add_argument("--gate-min", type=int, default=0, help="drops: per-set gate only for sets with this many checked groups")
    ap.add_argument("--stage", default="pilot1", choices=("pilot1", "pilot10", "full"))
    ap.add_argument("--set", default="ba", choices=tuple(SRC) + ("d2",))
    ap.add_argument("--out", default=str(data_dir() / "runs" / NAME / "gen"), help="generation root (per-set subdirs)")
    a = ap.parse_args()
    if a.step == "forks":
        print(json.dumps(forks(), indent=1, default=str))
    elif a.step == "check2":
        print(json.dumps(check2(a.out, a.stage), indent=1, default=str))
    elif a.step == "forks2":
        print(json.dumps(forks2(), indent=1, default=str))
    elif a.step == "d2":
        print(json.dumps(d2(), indent=1))
    elif a.step == "ids":
        print(ids(a.stage, a.set, os.path.join(a.out, a.set)))
    elif a.step == "prefix":
        print(prefix(a.out, a.stage).to_string(index=False))
    elif a.step in ("prefix_spec", "prefix_read"):
        r = prefix_feats(a.out, a.stage, a.step.split("_")[1])
        print(r.to_string(index=False) if isinstance(r, pd.DataFrame) else json.dumps(r))
    elif a.step == "drops":
        r = drops(a.out, gate_min=a.gate_min)
        compact = {k: v for k, v in r.items() if k not in ("dropped_groups", "dropped_route_ids")} | \
                  {"dropped_fork_ids": [d["fork_id"] for d in r["dropped_groups"]], "dropped_route_ids_n": len(r["dropped_route_ids"])}
        print(json.dumps(compact))
        if a.gate and r["gate_fail"]:
            raise SystemExit(3)
    else:
        print(json.dumps(sanity(a.out, a.stage), indent=1, default=str))


if __name__ == "__main__":
    main()
