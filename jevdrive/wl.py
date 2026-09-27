"""WL: openpilot proposes, a latent world model imagines, a critic chooses -- the fork data and its readouts
(todos/2026-09-28-wm-loop.md; registered 2026-09-27, criteria written before any WL number).

Generation (CARLA, scripts/wl_gen.sh with scripts/wl_fork_agent.py):
  forks      fork points of P5 v1 BA (pedestrian + cut-in families, x+ / x-) and P6 v0 (main obstacles, x10 / x00),
             seed 0: k1 = first camera tick >= the hazard's first visibility, k2 = k1 + 0.6 s, k3 = 0.4 s before the
             expert's reaction (P5: the x+ / x- ego divergence; P6: the lateral divergence), kept when k3 > k2 and
             v(k) >= 3 m/s; seven actions each; route-grouped eval split (40 routes: P5 pedestrian 12, cut-in 8, P6 20)
             -> runs/wl/{forks.parquet, split.json, jobs.json, forks-<set>.xml}
  d2         D2 random-intervention runs: Bench2Drive 220 base routes (minus the eval routes' bases) x TM seeds 0, 1
  ids        route ids of a stage (pilot1 / pilot10 / full) still without a done record, for b2d_run --route-ids
  sanity     the registered checklist on the finished fork runs of a stage -> research/results/wl/sanity_<stage>.csv
  prefix     the first check: a fork run's ticks before the fork against its source run (pose, actors, JPEG bytes)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "wl"
TICK, CAM = 0.05, 4
PED_FAM = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian", "ParkingCrossingPedestrian")
CUTIN_FAM = ("StaticCutIn", "ParkingCutIn", "HighwayCutIn")
N_EVAL = {"ped": 12, "cutin": 8, "obstacle": 20}
SPLIT_SEED = 20260928
V_MIN = 3.0
SRC = {"ba": dict(runs="runs/p5v1", gen="gen-ba", proc="carla_p5v1_ba", xml="runs/p5v1/pairs.xml", agent="runs/p5v1/agent-ba.json"),
       "p6": dict(runs="runs/p6", gen="gen", proc="carla_p6", xml="runs/p6/pairs.xml", agent="runs/p6/agent-p6.json")}


def rundir(*p) -> Path:
    d = data_dir() / "runs" / "wl"
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


def _sources() -> pd.DataFrame:
    """One row per (base route, world pair) of seed 0: set, class, family, x+ / x- route ids, t_vis, reaction tick."""
    rows = []
    p = pd.read_csv(data_dir() / "processed" / SRC["ba"]["proc"] / "pairs.csv")
    p = p[(p.seed == 0) & p.family.isin(PED_FAM + CUTIN_FAM) & (p.reason == "ok")]
    for r in p.itertuples():
        rows.append({"set": "ba", "cls": "ped" if r.family in PED_FAM else "cutin", "family": r.family, "base_id": str(r.base_id),
                     "plus": str(r.plus), "minus": str(r.minus), "t_vis": float(r.t_vis),
                     "t_react": float(r.t_div) if r.t_div <= r.t_last else np.nan})
    c = pd.read_parquet(data_dir() / "processed" / SRC["p6"]["proc"] / "nq3_cases.parquet")
    c = c[(c.seed == 0) & c.main.fillna(False).astype(bool) & (c.reason == "ok")]
    for r in c.itertuples():
        react = r.t_div_lat if pd.notna(r.t_div_lat) else r.t_div
        rows.append({"set": "p6", "cls": "obstacle", "family": r.scenario, "base_id": str(r.base_id), "plus": str(r.x10),
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
            "by_k": f.k_name.value_counts().to_dict(), "op_plan_missing": int((~f.has_op).sum()),
            "routes_by_cls_split": f.groupby(["cls", "split"]).base_id.nunique().to_dict().__repr__()}
    rundir("forks_info.json").write_text(json.dumps(info, indent=1, default=str))
    return info


# ================================================================ stages

def stage_ids(stage: str) -> pd.DataFrame:
    """pilot1: one P5 pedestrian fork point (k1, x+, an eval route) x 7 actions; pilot10: every fork point of 10 routes
    (P5 pedestrian 4, cut-in 2, P6 4, training split); full: everything."""
    runs = pd.read_parquet(rundir("forks.parquet"))
    if stage == "full":
        return runs
    rng = np.random.RandomState(SPLIT_SEED + 1)
    if stage == "pilot1":
        c = runs[(runs.cls == "ped") & (runs.k_name == "k1") & (runs.world == "plus") & runs.has_op]
        return runs[runs.fork_id == c.fork_id.iloc[0]]
    pick = []
    for cls, n in (("ped", 4), ("cutin", 2), ("obstacle", 4)):
        b = np.array(sorted(runs[(runs.cls == cls) & (runs.split == "train") & runs.has_op].base_id.unique()))
        pick += list(b[rng.permutation(len(b))][:n])
    return runs[runs.base_id.isin(pick)]


def ids(stage: str, set_name: str, out: str) -> str:
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


def prefix(out: str, stage: str = "pilot1") -> pd.DataFrame:
    """Every finished fork run of the stage against its source run on the ticks before the fork: max ego position
    difference, max actor position difference (same ids), and whether the saved JPEGs before the fork are
    byte-identical to the source's."""
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
        fa, fs = _frames(a), _frames(s)
        za, zs = np.load(a / "actors.npz"), np.load(s / "actors.npz")
        ta = pd.DataFrame({"frame": za["frame"], "id": za["id"], "x": za["xyz"][:, 0], "y": za["xyz"][:, 1]})
        tsrc = pd.DataFrame({"frame": zs["frame"], "id": zs["id"], "x": zs["xyz"][:, 0], "y": zs["xyz"][:, 1]})
        # frames are server counters: map both to ticks through the recorders' own first frame
        ta["tick"] = ta.frame - int(fa.frame.iloc[0]) + 1
        tsrc["tick"] = tsrc.frame - int(fs.frame.iloc[0]) + 1
        m = ta[ta.tick < r.fork_tick].merge(tsrc[tsrc.tick < r.fork_tick], on=["tick", "id"], suffixes=("_a", "_s"))
        dact = float(np.hypot(m.x_a - m.x_s, m.y_a - m.y_s).max()) if len(m) else np.nan
        same, n_img = 0, 0
        for t in fa.index[fa.index < r.fork_tick]:
            files = fa.files[t]
            if not files or t not in fs.index:
                continue
            for cam, rel in files.items():
                n_img += 1
                same += (a / rel).read_bytes() == (s / fs.files[t][cam]).read_bytes()
        rows.append({"route_id": r.route_id, "action": r.action, "fork_tick": r.fork_tick, "ego_max_dpos_m": dpos,
                     "actor_max_dpos_m": dact, "actor_rows_matched": len(m), "images_compared": n_img, "images_identical": same})
    t = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / f"prefix_{stage}.csv", index=False, float_format="%.4f")
    return t


def _gap_front(adir: Path, t0: int, t1: int) -> dict:
    """Min longitudinal gap to any road actor in the ego lane (|y| <= 1.75 m, ego frame) over ticks [t0, t1], and the
    ego's travel. Actor rows come from actors.npz, tick-aligned through frames.jsonl's first frame."""
    p = _pose(adir)
    fr0 = int(_frames(adir).frame.iloc[0])
    z = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    road = {int(k) for k, v in kinds.items() if v[0].startswith(("walker.", "vehicle.", "static.prop."))}
    tick = z["frame"] - fr0 + 1
    gap = np.inf
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
            gap = min(gap, float(x[ok].min()) - 2.4)          # front bumper to the actor's reference point
    k1 = min(t1, p.index.max())
    travel = float(np.hypot(np.diff(p.x[t0:k1]), np.diff(p.y[t0:k1])).sum())
    return {"gap_min_m": gap, "travel_m": travel}


def outcome(adir: Path, fork_tick: int, branch_s: float = 3.0) -> dict:
    """Branch outcome: collision in the branch (collisions.jsonl, tick-aligned), min in-lane gap, ego travel, lateral
    offset from the fork pose's heading line at the branch end; unsafe = collision or gap < 2 m."""
    t1 = fork_tick + int(round(branch_s / TICK))
    fr0 = int(_frames(adir).frame.iloc[0])
    col = [json.loads(l) for l in (adir / "collisions.jsonl").read_text().splitlines()] if (adir / "collisions.jsonl").exists() else []
    hit = [c for c in col if fork_tick <= c["frame"] - fr0 + 1 <= t1]
    g = _gap_front(adir, fork_tick, t1)
    p = _pose(adir)
    e0, e1 = p.loc[fork_tick], p.loc[min(t1, p.index.max())]
    c, s = np.cos(np.radians(e0.yaw)), np.sin(np.radians(e0.yaw))
    lat = float(-(e1.x - e0.x) * s + (e1.y - e0.y) * c)       # CARLA y right: positive = right of the start heading
    return {"collision": bool(hit), "collision_types": sorted({h["other_type"] for h in hit}), **g, "lateral_right_m": lat,
            "unsafe": bool(hit) or g["gap_min_m"] < 2.0, "ticks": int(p.index.max())}


def sanity(out: str, stage: str) -> dict:
    """The registered checklist (todo, "分叉数据的 sanity checklist"); every item -> pass / fail."""
    runs = stage_ids(stage)
    rows = []
    for r in runs.itertuples():
        a = _fork_attempt(Path(out) / r.set, r.route_id)
        rec = {"route_id": r.route_id, "fork_id": r.fork_id, "set": r.set, "cls": r.cls, "family": r.family, "world": r.world,
               "k_name": r.k_name, "action": r.action, "done": a is not None}
        if a is not None:
            try:
                rec.update(outcome(a, r.fork_tick))
            except Exception as e:                          # a harness failure of this run
                rec["error"] = repr(e)
        rows.append(rec)
    t = pd.DataFrame(rows)
    pre = prefix(out, stage)
    fin = t[t.done & t.get("error", pd.Series(np.nan, index=t.index)).isna()]
    piv = fin.pivot_table(index="fork_id", columns="action", values="travel_m")
    lat = fin.pivot_table(index="fork_id", columns="action", values="lateral_right_m")
    cls = fin.groupby("fork_id")[["cls", "world"]].first()
    unsafe = fin.pivot_table(index="fork_id", columns="action", values="unsafe", aggfunc="max")
    chk = {}
    chk["prefix_ego_le_1cm"] = float((pre.ego_max_dpos_m <= 0.01).mean()) if len(pre) else np.nan
    chk["brake_travel_lt_hold"] = float((piv.brake_hard < piv.hold).mean()) if {"brake_hard", "hold"} <= set(piv) else np.nan
    if {"shift_L", "shift_R"} <= set(lat):
        chk["shift_sign_ok"] = float(pd.concat([lat.shift_L < 0, lat.shift_R > 0]).mean())     # left = negative right
        chk["shift_ge_2m"] = float(pd.concat([lat.shift_L <= -2, lat.shift_R >= 2]).mean())
    u = unsafe.join(cls)
    hold_op = u[["hold", "op"]].max(axis=1) if {"hold", "op"} <= set(u) else None
    if hold_op is not None:
        for c_, w_ in (("ped", "plus"), ("ped", "minus"), ("obstacle", "plus")):
            m = (u.cls == c_) & (u.world == w_)
            chk[f"unsafe_hold_or_op_{c_}_{w_}"] = float(hold_op[m].mean()) if m.any() else np.nan
        m = (u.cls == "obstacle") & (u.world == "plus")
        if m.any() and {"shift_L", "shift_R"} <= set(u):
            chk["obstacle_plus_hold_unsafe"] = float(u.hold[m].mean())
            chk["obstacle_plus_a_shift_safe"] = float((~(u.shift_L[m].astype(bool) & u.shift_R[m].astype(bool))).mean())
    chk["harness_fail"] = float(1 - t.done.mean() + (t.get("error", pd.Series(np.nan, index=t.index)).notna().mean()))
    verdict = {
        "prefix": chk["prefix_ego_le_1cm"] == 1.0,
        "brake": chk.get("brake_travel_lt_hold") == 1.0,
        "shift": chk.get("shift_sign_ok", 0) >= 0.95 and chk.get("shift_ge_2m", 0) >= 0.80,
        "outcomes": (chk.get("unsafe_hold_or_op_ped_plus", 0) >= 0.20 and chk.get("unsafe_hold_or_op_ped_minus", 1) <= 0.05
                     and chk.get("obstacle_plus_hold_unsafe", 0) >= 0.30 and chk.get("obstacle_plus_a_shift_safe", 0) >= 0.50),
        "harness": chk["harness_fail"] <= 0.05}
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / f"runs_{stage}.csv", index=False, float_format="%.4f")
    (RESULTS / f"sanity_{stage}.json").write_text(json.dumps({"checks": chk, "verdict": verdict}, indent=1, default=str))
    return {"checks": chk, "verdict": verdict}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("forks", "ids", "prefix", "sanity"))
    ap.add_argument("--stage", default="pilot1", choices=("pilot1", "pilot10", "full"))
    ap.add_argument("--set", default="ba", choices=tuple(SRC))
    ap.add_argument("--out", default=str(data_dir() / "runs" / "wl" / "gen"), help="generation root (per-set subdirs)")
    a = ap.parse_args()
    if a.step == "forks":
        print(json.dumps(forks(), indent=1, default=str))
    elif a.step == "ids":
        print(ids(a.stage, a.set, os.path.join(a.out, a.set)))
    elif a.step == "prefix":
        print(prefix(a.out, a.stage).to_string(index=False))
    else:
        print(json.dumps(sanity(a.out, a.stage), indent=1, default=str))


if __name__ == "__main__":
    main()
