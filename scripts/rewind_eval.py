#!/usr/bin/env python
"""CARLA rewind fidelity check (todos/2026-09-29-carla-rewind.md): jobs, per-branch cutting, readouts.

  prep   runs/rewind/{jobs.json, routes-<set>.xml, agent-<set>.json, plan.parquet}: one rewind run per (fork, method)
         playing all 7 actions (order rotated per fork), plus from-scratch reruns of every action of the floor forks
         under their WL route ids (same job, same XML, same seed)
  cut    every finished rewind run -> one virtual attempt dir per branch (runs/rewind/branches/<route>/<branch>/):
         the prefix (ticks < fork_tick) plus that branch's frames renumbered as if it had run from the prefix, so
         jevdrive.wl.outcome / _match_actors read it like a from-scratch fork run
  eval   per branch against the WL from-scratch run of the same fork and action (truth): ego / hazard / other-actor
         differences at 0, 0.5, 1, 2, 3 s, labels, walk-start tick -> runs/rewind/eval/{branches.csv, actors.csv}
  opspec openpilot stream plan for the branches (source frames + prefix + branch) -> processed/rewind/op_plan.json;
         then P5_SET=rewind scripts/p5_openpilot.py --models cinque --arrays temporal --out-sub op_streams
  opcos  temporal cosine per branch against the truth's stored stream -> runs/rewind/eval/opcos.csv
  cost   wall time per fork: rewind run vs the 7 truth runs vs the floor reruns -> runs/rewind/eval/cost.csv
Run with the repo's .venv (PYTHONPATH=.) on the box.
"""
import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import wl as WL  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.wl_traj import ACTIONS  # noqa: E402

R = data_dir() / "runs" / "rewind"
# pre-registered fork points (todo "分叉点"), in table order: the action order of fork j starts at ACTIONS[j % 7]
FORKS = (78, 96, 60, 42, 212, 310, 131, 226, 148, 328, 378)
FLOOR_FORKS = (78, 148)
REUSE_SCRATCH_FORKS = (78, 148, 96, 60)      # map-reuse arm U: every action from scratch, same-map runs back to back
METHODS = ("poc", "teleport", "tree", "respawn")
TICK, CAM = 0.05, 4
HORIZONS_S = (0.0, 0.5, 1.0, 2.0, 3.0)


def rid_of(fork_id: int, m: int, seed_digit: str) -> str:
    return "7%04d%02d%s" % (fork_id, m, seed_digit)


def prep(methods=METHODS) -> dict:
    import xml.etree.ElementTree as ET
    f = pd.read_parquet(WL.rundir("forks.parquet"))
    jobs_wl = json.loads(WL.rundir("jobs.json").read_text())
    R.mkdir(parents=True, exist_ok=True)
    jobs, plan = {}, []
    for j, fid in enumerate(FORKS):
        x = f[f.fork_id == fid].set_index("action")
        r0 = x.loc["op"]
        order = [ACTIONS[(j + i) % len(ACTIONS)] for i in range(len(ACTIONS))]
        for m in methods:
            rid = rid_of(fid, METHODS.index(m), r0.route_id[-1])
            jobs[rid] = dict(jobs_wl[r0.route_id], rewind=m, actions=order, action=order[0])
            plan.append({"route_id": rid, "kind": "rewind", "method": m, "fork_id": fid, "set": r0.set,
                         "xml_src": r0.route_id, "order": ",".join(order), "gen": "gen"})
            if m == "tree":                    # arm UR: the same rewind run on a reused map
                plan.append(dict(plan[-1], gen="gen_reuse"))
        if fid in FLOOR_FORKS:
            for a, r in x.iterrows():
                jobs[r.route_id] = jobs_wl[r.route_id]
                plan.append({"route_id": r.route_id, "kind": "floor", "method": "scratch", "fork_id": fid, "set": r.set,
                             "xml_src": r.route_id, "order": a, "gen": "gen"})
        if fid in REUSE_SCRATCH_FORKS:
            for a, r in x.iterrows():
                jobs[r.route_id] = jobs_wl[r.route_id]
                plan.append({"route_id": r.route_id, "kind": "floor", "method": "reuse", "fork_id": fid, "set": r.set,
                             "xml_src": r.route_id, "order": a, "gen": "gen_reuse"})
    p = pd.DataFrame(plan)
    (R / "jobs.json").write_text(json.dumps(jobs))
    for s, g in p.drop_duplicates(["route_id", "set"]).groupby("set"):
        root = ET.parse(WL.rundir(f"forks-{s}.xml")).getroot()
        by_id = {e.get("id"): e for e in root.iter("route")}
        out = ET.Element("routes")
        for r in g.itertuples():
            e = ET.fromstring(ET.tostring(by_id[r.xml_src]))
            e.set("id", r.route_id)
            out.append(e)
        ET.ElementTree(out).write(R / f"routes-{s}.xml")
        c = json.loads((WL.rundir("gen") / f"agent-{s}.json").read_text())
        c["wl_jobs"] = str(R / "jobs.json")
        (R / f"agent-{s}.json").write_text(json.dumps(c))
    p.to_parquet(R / "plan.parquet", index=False)
    return p.groupby(["set", "kind"]).size().to_dict()


def attempt(rid: str, set_name: str, gen: str = "gen") -> Path | None:
    return WL._fork_attempt(R / gen / set_name, rid)


# ---------------------------------------------------------------------------------------------------------- cut


def _cut_one(args):
    rid, set_name, fork_tick, gen = args
    a = attempt(rid, set_name, gen)
    if a is None:
        return []
    ev = [json.loads(l) for l in (a / "rewind.jsonl").read_text().splitlines()]
    starts = {e["branch"]: e for e in ev if e["kind"] == "start"}
    ends = {e["branch"]: e for e in ev if e["kind"] == "end"}
    pose = pd.read_json(a / "pose.jsonl", lines=True)
    frames = pd.read_json(a / "frames.jsonl", lines=True)
    z = dict(np.load(a / "actors.npz"))
    col = [json.loads(l) for l in (a / "collisions.jsonl").read_text().splitlines()] if (a / "collisions.jsonl").exists() else []
    fr0 = int(frames.frame.iloc[0])
    pre_last = fr0 + fork_tick - 2                  # frame of tick k - 1 (tick = frame - fr0 + 1 before the first rewind)
    out = []
    for b, st in sorted(starts.items()):
        f_k = int(st["frame"])                      # frame of the branch's tick k
        f_end = int(ends[b]["frame"]) if b in ends else f_k + 60
        shift = f_k - (pre_last + 1)                # branch frame -> the frame a from-scratch run would have had
        keep = lambda fr: (fr <= pre_last) | ((fr >= f_k) & (fr <= f_end + 1))  # noqa: E731
        d = R / "branches" / gen / rid / str(b)
        d.mkdir(parents=True, exist_ok=True)
        p = pose[keep(pose.frame)].copy()
        p.loc[p.frame >= f_k, "frame"] -= shift
        p.to_json(d / "pose.jsonl", orient="records", lines=True)
        fr = frames[keep(frames.frame)].copy()
        fr.loc[fr.frame >= f_k, "frame"] -= shift
        fr.loc[fr.frame >= pre_last + 1, "tick"] = fr.frame - fr0 + 1
        fr.to_json(d / "frames.jsonl", orient="records", lines=True)
        m = keep(z["frame"])
        zz = {k: v[m] for k, v in z.items()}
        zz["frame"] = np.where(zz["frame"] >= f_k, zz["frame"] - shift, zz["frame"])
        np.savez(d / "actors.npz", **zz)
        with open(d / "collisions.jsonl", "w") as fh:
            for c in col:
                if f_k <= c["frame"] <= f_end + 1:
                    fh.write(json.dumps(dict(c, frame=c["frame"] - shift)) + "\n")
        for name in ("actor_kinds.json", "hidden.json", "meta.json"):
            if (a / name).exists():
                shutil.copy(a / name, d / name)
        out.append({"route_id": rid, "gen": gen, "branch": b, "action": st["action"], "frame_k": f_k, "shift": shift,
                    "vdir": str(d), "n_pose": len(p)})
    return out


def cut() -> pd.DataFrame:
    p = pd.read_parquet(R / "plan.parquet")
    f = pd.read_parquet(WL.rundir("forks.parquet")).drop_duplicates("fork_id").set_index("fork_id")
    rw = p[p.kind == "rewind"]
    with Pool(24) as pool:
        rows = pool.map(_cut_one, [(r.route_id, r.set, int(f.fork_tick[r.fork_id]), r.gen) for r in rw.itertuples()])
    t = pd.DataFrame([x for rr in rows for x in rr]).merge(rw, on=["route_id", "gen"])
    t.to_csv(R / "branches.csv", index=False)
    return t


# ---------------------------------------------------------------------------------------------------------- eval


def _hazard_ids(adir: Path) -> list:
    h = adir / "hidden.json"
    return [x["id"] for x in json.loads(h.read_text())] if h.exists() else []


def _actors(adir: Path, t0: int, t1: int) -> pd.DataFrame:
    z = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    fr0 = int(WL._frames(adir).frame.iloc[0])
    t = pd.DataFrame({"tick": z["frame"] - fr0 + 1, "id": z["id"], "x": z["xyz"][:, 0], "y": z["xyz"][:, 1],
                      "yaw": z["yaw"], "v": np.hypot(z["v"][:, 0], z["v"][:, 1])})
    t["type"] = t.id.map(lambda i: kinds.get(str(i), ["?"])[0])
    return t[(t.tick >= t0) & (t.tick <= t1)]


def _walk_start(t: pd.DataFrame, ids, k: int):
    w = t[t.id.isin(ids) & (t.tick >= k) & (t.v > 0.5)]
    return int(w.tick.min()) if len(w) else None


def _yawd(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def compare(test: Path, truth: Path, k: int) -> tuple[dict, list]:
    """Branch `test` against `truth`, both tick-aligned fork runs with the fork at tick k."""
    t1 = k + 60
    pa, ps = WL._pose(test), WL._pose(truth)
    row = {}
    for h in HORIZONS_S:
        t = k + int(round(h / TICK))
        if t in pa.index and t in ps.index:
            row[f"ego_dpos_{h}"] = float(np.hypot(pa.x[t] - ps.x[t], pa.y[t] - ps.y[t]))
            row[f"ego_dv_{h}"] = float(abs(pa.v[t] - ps.v[t]))
            row[f"ego_dyaw_{h}"] = float(_yawd(pa.yaw[t], ps.yaw[t]))
    oa, os_ = WL.outcome(test, k), WL.outcome(truth, k)
    for key in ("collision", "collision_road", "unsafe"):
        row[f"{key}_test"], row[f"{key}_truth"] = oa[key], os_[key]
    row["ped_crit_test"] = oa["collision"] or oa["gap_min_m"] < 2.0
    row["ped_crit_truth"] = os_["collision"] or os_["gap_min_m"] < 2.0
    for key in ("gap_min_m", "ttc_min_s", "travel_m", "first_collision_tick"):
        row[f"{key}_test"], row[f"{key}_truth"] = oa[key], os_[key]
    aa, as_ = _actors(test, k, t1), _actors(truth, k, t1)
    m = WL._match_actors(aa, as_)            # columns <name>_a (test) / <name>_s (truth)
    hz_t = set(_hazard_ids(truth))
    e0 = ps.loc[k]
    arows = []
    for r in m.itertuples():
        h = (r.tick - k) * TICK
        if min(abs(h - q) for q in HORIZONS_S) > 1e-6:
            continue
        arows.append({"h": round(h, 2), "sid": r.sid, "type": r.type_s, "hazard": r.sid in hz_t,
                      "dist0": float(np.hypot(r.x_s - e0.x, r.y_s - e0.y)),
                      "dpos": float(np.hypot(r.x_a - r.x_s, r.y_a - r.y_s)), "dv": float(abs(r.v_a - r.v_s)),
                      "dyaw": float(_yawd(r.yaw_a, r.yaw_s))})
    # hazard walk start (tick the hazard's speed first exceeds 0.5 m/s at or after k); the test run's hazard is the
    # actor matched to the truth's (a respawned hazard has a new id)
    hz_a = set(m.loc[m.sid.isin(hz_t), "id_a"])
    row["walk_test"], row["walk_truth"] = _walk_start(aa, hz_a, k), _walk_start(as_, hz_t, k)
    row["n_hazard_matched"] = len(hz_a)
    return row, arows


def _eval_one(args):
    kind, key, test, truth, k = args
    try:
        row, arows = compare(Path(test), Path(truth), k)
    except Exception as e:  # noqa: BLE001
        return [dict(key, error=repr(e))], []
    return [dict(key, **row)], [dict(key, **a) for a in arows]


def evaluate() -> dict:
    p = pd.read_parquet(R / "plan.parquet")
    br = pd.read_csv(R / "branches.csv", dtype={"route_id": str})
    f = pd.read_parquet(WL.rundir("forks.parquet"))
    truth = {(r.fork_id, r.action): (r.route_id, r.set, int(r.fork_tick)) for r in f.itertuples()}
    tasks = []
    for r in br.itertuples():
        trid, s, k = truth[(r.fork_id, r.action)]
        tasks.append(("rewind", {"route_id": r.route_id, "gen": r.gen, "method": r.method, "fork_id": r.fork_id,
                                 "branch": r.branch, "action": r.action, "set": s}, r.vdir, str(WL._fork_attempt(WL.rundir("gen") / s, trid)), k))
    for r in p[p.kind == "floor"].itertuples():
        a = attempt(r.route_id, r.set, r.gen)
        if a is None:
            continue
        trid, s, k = truth[(r.fork_id, r.order)]
        tasks.append(("floor", {"route_id": r.route_id, "gen": r.gen, "method": r.method, "fork_id": r.fork_id, "branch": 0,
                                "action": r.order, "set": s}, str(a), str(WL._fork_attempt(WL.rundir("gen") / s, trid)), k))
    with Pool(24) as pool:
        res = pool.map(_eval_one, tasks, chunksize=2)
    out = R / "eval"
    out.mkdir(exist_ok=True)
    b = pd.DataFrame([x for r in res for x in r[0]])
    a = pd.DataFrame([x for r in res for x in r[1]])
    b.to_csv(out / "branches.csv", index=False)
    a.to_csv(out / "actors.csv", index=False)
    return {"branches": len(b), "actor_rows": len(a), "errors": int(b.get("error", pd.Series(dtype=object)).notna().sum())}


# ---------------------------------------------------------------------------------------------------------- openpilot


def opspec() -> dict:
    from jevdrive.p5_openpilot import carla_calib
    br = pd.read_csv(R / "branches.csv", dtype={"route_id": str})
    f = pd.read_parquet(WL.rundir("forks.parquet")).drop_duplicates("fork_id").set_index("fork_id")
    p = pd.read_parquet(R / "plan.parquet")
    streams = []
    cams = ("front", "front_left", "front_right")

    def add(key, run_dir, src, rid_names):
        fr = pd.read_json(Path(run_dir) / "frames.jsonl", lines=True)
        fr = fr[fr.files.map(bool)]
        sf = pd.read_json(Path(src) / "frames.jsonl", lines=True)
        sf = sf[sf.tick < fr.tick.min()]
        names = [f"{Path(src).parent.name}-{x:07d}" for x in sf.frame] + [f"{rid_names}-{x:07d}" for x in fr.frame]
        files = [[f"{src}/{r[c]}" for c in cams] for r in sf.files] + [[f"{run_dir}/{r[c]}" for c in cams] for r in fr.files]
        streams.append({"key": key, "names": names, "targets": list(range(len(sf), len(names))), "files": files, "gaps": 0})

    for r in br.itertuples():
        # a branch dir keeps the run's JPEG paths (relative to the run's attempt dir) with renumbered frames
        a = attempt(r.route_id, f.set[r.fork_id], r.gen)
        vd = Path(r.vdir)
        fr = pd.read_json(vd / "frames.jsonl", lines=True)
        fr = fr[fr.files.map(bool)]
        sf = pd.read_json(Path(f.src_dir[r.fork_id]) / "frames.jsonl", lines=True)
        sf = sf[sf.tick < fr.tick.min()]
        src = f.src_dir[r.fork_id]
        names = [f"{Path(src).parent.name}-{x:07d}" for x in sf.frame] + [f"rw{r.gen}{r.route_id}b{r.branch}-{x:07d}" for x in fr.frame]
        files = [[f"{src}/{q[c]}" for c in cams] for q in sf.files] + [[f"{a}/{q[c]}" for c in cams] for q in fr.files]
        streams.append({"key": f"rw_{r.gen}_{r.route_id}_{r.branch}", "names": names, "targets": list(range(len(sf), len(names))),
                        "files": files, "gaps": 0})
    for r in p[p.kind == "floor"].itertuples():
        a = attempt(r.route_id, r.set, r.gen)
        if a is not None:
            add(f"fl_{r.gen}_{r.route_id}", a, f.src_dir[r.fork_id], f"fl{r.gen}{r.route_id}")
    d = data_dir() / "processed" / "rewind"
    d.mkdir(parents=True, exist_ok=True)
    (d / "op_plan.json").write_text(json.dumps({"calib": carla_calib(), "streams": streams}))
    return {"streams": len(streams), "frames": sum(len(s["names"]) for s in streams)}


def opcos() -> dict:
    """Per branch: cosine of Cinque `temporal` against the truth run's stored stream at the same fork-relative tick."""
    br = pd.read_csv(R / "branches.csv", dtype={"route_id": str})
    p = pd.read_parquet(R / "plan.parquet")
    f = pd.read_parquet(WL.rundir("forks.parquet"))
    truth = {(r.fork_id, r.action): (r.route_id, r.set, int(r.fork_tick)) for r in f.itertuples()}
    mine = data_dir() / "processed" / "rewind" / "op_streams" / "cinque"
    stored = data_dir() / "processed" / "wl_gen" / "op_streams_vis" / "cinque"

    def by_tick(npz, run_dir, prefix):
        q = np.load(npz)
        fr = pd.read_json(Path(run_dir) / "frames.jsonl", lines=True)
        fr0 = int(fr.frame.iloc[0])
        out = {}
        for n, v in zip(q["name"].astype(str), q["temporal"]):
            if n.startswith(prefix):
                out[int(n.rsplit("-", 1)[1]) - fr0 + 1] = v.astype(np.float32)
        return out

    rows = []
    items = [(f"rw_{r.gen}_{r.route_id}_{r.branch}", r.vdir, f"rw{r.gen}{r.route_id}b{r.branch}", r.fork_id, r.action,
              {"route_id": r.route_id, "gen": r.gen, "method": r.method, "branch": r.branch}) for r in br.itertuples()]
    items += [(f"fl_{r.gen}_{r.route_id}", str(attempt(r.route_id, r.set, r.gen)), f"fl{r.gen}{r.route_id}", r.fork_id,
               r.order, {"route_id": r.route_id, "gen": r.gen, "method": r.method, "branch": 0})
              for r in p[p.kind == "floor"].itertuples() if attempt(r.route_id, r.set, r.gen) is not None]
    for key, run_dir, prefix, fid, act, meta in items:
        fa = mine / f"{key}.npz"
        trid, s, k = truth[(fid, act)]
        fs = stored / f"wl_{trid}.npz"
        if not fa.exists() or not fs.exists():
            continue
        A = by_tick(fa, run_dir, prefix)
        S = by_tick(fs, WL._fork_attempt(WL.rundir("gen") / s, trid), trid)
        ts = [t for t in sorted(A) if k <= t <= k + 60 and t in S]
        if not ts:
            continue
        cos = [float(A[t] @ S[t] / (np.linalg.norm(A[t]) * np.linalg.norm(S[t]) + 1e-12)) for t in ts]
        pre = [t for t in sorted(A) if t < k and t in S]
        cpre = [float(A[t] @ S[t] / (np.linalg.norm(A[t]) * np.linalg.norm(S[t]) + 1e-12)) for t in pre]
        rows.append(dict(meta, fork_id=fid, action=act, n=len(ts), cos_min=min(cos), cos_mean=float(np.mean(cos)),
                         cos_last=cos[-1], pre_cos_min=min(cpre) if cpre else np.nan))
    t = pd.DataFrame(rows)
    (R / "eval").mkdir(exist_ok=True)
    t.to_csv(R / "eval" / "opcos.csv", index=False)
    return {"rows": len(t)}


# ---------------------------------------------------------------------------------------------------------- cost


def cost() -> pd.DataFrame:
    p = pd.read_parquet(R / "plan.parquet")
    f = pd.read_parquet(WL.rundir("forks.parquet"))
    rows = []
    for r in p.itertuples():
        a = attempt(r.route_id, r.set, r.gen)
        if a is None:
            continue
        d = json.loads((R / r.gen / r.set / "done" / f"{r.route_id}.json").read_text())
        ph = json.loads((a / "phases.json").read_text()) if (a / "phases.json").exists() else {}
        row = {"route_id": r.route_id, "gen": r.gen, "reused": ph.get("reused"), "load_s": ph.get("load_s"),
               "import_s": ph.get("t_load_call", np.nan) - ph.get("proc_start", np.nan),
               "scenario_build_s": ph.get("scenario_build_s"), "server_age_routes": d.get("server_age_routes"),
               "kind": r.kind, "method": r.method, "fork_id": r.fork_id, "set": r.set,
               "wall_s": d["wall_s"], "ticks": d.get("ticks"), "tick_ms": (d.get("profile") or {}).get("total_ms_mean")}
        if r.kind == "rewind":
            ev = [json.loads(l) for l in (a / "rewind.jsonl").read_text().splitlines()]
            st = [e for e in ev if e["kind"] == "start"]
            en = [e for e in ev if e["kind"] == "end"]
            rw = [e for e in ev if e["kind"] == "rewind"]
            row.update(n_branches=len(en), branch_wall_s=float(np.mean([e["wall"] - s["wall"] for s, e in zip(st, en)])),
                       restore_ms=float(np.mean([e["restore_ms"] for e in rw])) if rw else np.nan,
                       dead_tick_ms=float(np.mean([e["dead_tick_ms"] for e in rw])) if rw else np.nan,
                       snapshot_ms=next((e["snapshot_ms"] for e in ev if e["kind"] == "snapshot"), np.nan),
                       respawned=sum(e.get("respawned", 0) for e in rw), destroyed_new=sum(e.get("destroyed_new", 0) for e in rw))
        rows.append(row)
    t = pd.DataFrame(rows)
    truth = []
    for fid in p.fork_id.unique():
        x = f[f.fork_id == fid]
        w = [json.loads((WL.rundir("gen") / r.set / "done" / f"{r.route_id}.json").read_text())["wall_s"] for r in x.itertuples()]
        truth.append({"fork_id": fid, "truth7_wall_s": float(np.sum(w)), "truth_run_wall_med_s": float(np.median(w))})
    t = t.merge(pd.DataFrame(truth), on="fork_id", how="left")
    (R / "eval").mkdir(exist_ok=True)
    t.to_csv(R / "eval" / "cost.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "cut", "eval", "opspec", "opcos", "cost"))
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    if a.step == "prep":
        print(prep(tuple(a.methods.split(","))))
    elif a.step == "cut":
        print(len(cut()), "branches")
    elif a.step == "eval":
        print(evaluate())
    elif a.step == "opspec":
        print(opspec())
    elif a.step == "opcos":
        print(opcos())
    else:
        print(cost().to_string())


if __name__ == "__main__":
    main()
