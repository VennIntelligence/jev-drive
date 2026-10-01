"""WL data pipeline (fc65452:todos/2026-09-28-wm-loop.md): finished fork / D2 runs -> the world model's universe.

  index    CPU. Every saved camera frame of every finished run (runs/wl/gen/<set>/done): labels and logged actions with
           nq4_w's own code (_run_rows), the per-step action source (1 = inside an intervention window, from
           wl_ticks.jsonl) and the commanded (a, omega) of the window's candidate (experiments.world_model.lib.wl_traj.command_actions)
           -> processed/wl_gen/{index.parquet, windows.parquet}
  opspec   the openpilot stream spec: per run, the source run's frames before the first saved tick (identical prefix,
           checked by experiments.world_model.lib.wl prefix) + the run's own frames -> processed/wl_gen/op_plan.json; then
           P5_SET=wl_gen scripts/p5_openpilot.py --models cinque --arrays temporal --out-sub op_streams_vis
  vjepa    GPU. V-JEPA 2 `mean` (4-frame clip per camera, nq4_w's extractor) for every index row, resumable chunks;
           default batch / workers measured 2026-09-29 on the wm-loop row's GPU 5 + 24 cores (184-207): the clip
           DataLoader is CPU/IO-bound (peak VRAM stayed < 5 GB even at batch 256), workers 12-14 gave ~50 % more
           rows/s than the old workers=7, batch 128+ was slower than 64 (less GPU/CPU overlap), so batch stays 64
  z        temporal | vjepa per index row -> processed/wl_gen/z.npy (float16, nq4_w's layout)
  prune    delete the JPEGs of runs whose z rows exist (pilot runs kept)
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.world_model.archive import nq4_w as W
from experiments.world_model.lib import wl as WL
from jevdrive.common import data_dir, get_logger
from experiments.world_model.lib.wl_traj import command_actions

log = get_logger(__name__)
CAMS = ("front", "front_left", "front_right")


def pdir(*p) -> Path:
    d = data_dir() / "processed" / f"{WL.NAME}_gen"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def finished(out: Path) -> pd.DataFrame:
    """Every run of forks.parquet / d2.parquet with a done record, and its attempt dir; fork groups dropped by checklist
    amendment (a) (runs/wl/drops.json, `python -m experiments.world_model.lib.wl drops`) are left out whole, and individual runs dropped
    by amendment (c)'s per-run render gate are left out one at a time (their group's other branches stay)."""
    f = pd.read_parquet(WL.rundir("forks.parquet"))
    runs = [f[~f.fork_id.isin(WL.dropped_forks())]]
    if WL.rundir("d2.parquet").exists():
        runs.append(pd.read_parquet(WL.rundir("d2.parquet")))
    r = pd.concat(runs, ignore_index=True)
    r = r[~r.route_id.isin(WL.dropped_runs())]
    r["adir"] = [WL._fork_attempt(out / s, rid) for s, rid in zip(r.set, r.route_id)]
    r = r[r.adir.notna()].copy()
    r["adir"] = r.adir.astype(str)
    return r.reset_index(drop=True)


def _one(args):
    """Index rows of one run: saved frames, nq4_w labels / actions, action source and commanded actions."""
    rid, adir = args
    a = Path(adir)
    fr = pd.read_json(a / "frames.jsonl", lines=True)
    fr = fr[fr.files.map(bool)].reset_index(drop=True)
    _, rows = W._run_rows((adir, fr.frame.astype(int).tolist()))
    lab = pd.DataFrame(rows)
    wl = json.loads((a / "wl.json").read_text())
    ticks = pd.read_json(a / "wl_ticks.jsonl", lines=True) if (a / "wl_ticks.jsonl").stat().st_size else pd.DataFrame(
        {"tick": [], "action": []})
    inside = set(ticks.tick.astype(int))
    src = np.array([int(all(t + d in inside for d in range(W.TICKS))) for t in fr.tick])
    a_cmd, w_cmd = lab.a_next.to_numpy(np.float32).copy(), lab.w_next.to_numpy(np.float32).copy()
    act = np.array([None] * len(fr), object)
    wins = []
    pos = {int(t): i for i, t in enumerate(fr.tick)}
    for c in wl["cands"]:
        n = int(round(next(w["len_s"] for w in wl["windows"] if w["tick"] == c["tick"]) / W.DT)) if any(
            w["tick"] == c["tick"] for w in wl["windows"]) else 10
        n = max(n, 10)
        cmd = command_actions(c["op_plan"], np.asarray(c["route_ego"]), c["v0"], n=n)[c["action"]]
        for j in range(n):
            i = pos.get(int(c["tick"]) + W.TICKS * j)
            if i is not None and src[i]:
                a_cmd[i], w_cmd[i], act[i] = cmd[j, 0], cmd[j, 1], c["action"]
        wins.append({"route_id": rid, "tick": int(c["tick"]), "kind": c["kind"], "action": c["action"], "steps": n,
                     "v0": c["v0"], "cands": json.dumps(c["cands"]), "op_plan": json.dumps(c["op_plan"]),
                     "route_ego": json.dumps(c["route_ego"])})
    files = [[f"{adir}/{fr.files[i][c]}" for c in CAMS] for i in range(len(fr))]
    t = pd.DataFrame({"route_id": rid, "adir": adir, "tick": fr.tick.astype(int), "frame": fr.frame.astype(int),
                      "frame_name": [f"{rid}-{f:07d}" for f in fr.frame], "src": src, "a_cmd": a_cmd, "w_cmd": w_cmd,
                      "action": act, "cam_files": files})
    t = pd.concat([t, lab.drop(columns="frame")], axis=1)
    return t, wins


def index(out: str, workers: int = 16) -> dict:
    r = finished(Path(out))
    with Pool(workers) as p:
        res = p.map(_one, list(zip(r.route_id, r.adir)), chunksize=4)
    t = pd.concat([x[0] for x in res], ignore_index=True)
    t = t.merge(r.drop(columns="adir"), on="route_id", how="left")
    # V-JEPA's 4-frame clip: this frame and the 3 before it at 5 Hz, the earliest repeated where the run starts
    t = t.sort_values(["route_id", "tick"]).reset_index(drop=True)
    g = t.groupby("route_id").cumcount().to_numpy()
    cf = t.cam_files.to_list()
    t["files"] = [sum(([cf[i - min(g[i], 3 - q)][c] for q in range(4)] for c in range(3)), []) for i in range(len(t))]
    t.drop(columns="cam_files").to_parquet(pdir("index.parquet"), index=False)
    wins = pd.DataFrame([w for x in res for w in x[1]])
    wins.to_parquet(pdir("windows.parquet"), index=False)
    return {"runs": len(r), "rows": len(t), "intervention_rows": int(t.src.sum()), "windows": len(wins)}


def opspec() -> dict:
    """processed/wl_gen/op_plan.json: per run, source-run frames before the first saved tick, then the run's frames."""
    from jevdrive.p5_openpilot import carla_calib
    t = pd.read_parquet(pdir("index.parquet"))
    streams = []
    for rid, g in t.groupby("route_id", sort=True):
        g = g.sort_values("tick")
        names, files = [], []
        src = g.src_dir.iloc[0] if "src_dir" in g and isinstance(g.src_dir.iloc[0], str) else None
        if src:
            sf = pd.read_json(Path(src) / "frames.jsonl", lines=True)
            sf = sf[sf.tick < g.tick.min()]
            sroute = Path(src).parent.name
            names += [f"{sroute}-{f:07d}" for f in sf.frame]
            files += [[f"{src}/{r[c]}" for c in CAMS] for r in sf.files]
        n0 = len(names)
        names += g.frame_name.tolist()
        files += [[f[4 * c + 3] for c in range(3)] for f in g.files]
        streams.append({"key": f"wl_{rid}", "names": names, "targets": list(range(n0, len(names))), "files": files,
                        "gaps": 0})
    (pdir("op_plan.json")).write_text(json.dumps({"calib": carla_calib(), "streams": streams}))
    return {"streams": len(streams), "frames": sum(len(s["names"]) for s in streams)}


def vjepa(batch: int = 64, workers: int = 12, chunk: int = 3000, rl=None) -> dict:
    from jevdrive import features as F
    t = pd.read_parquet(pdir("index.parquet"))
    d = pdir("vjepa")
    d.mkdir(exist_ok=True)
    done = set()
    for c in d.glob("c*.npz"):
        with np.load(c) as z:
            done |= set(map(str, z["frame_name"]))
    miss = t[~t.frame_name.isin(done)].reset_index(drop=True)
    fx = None
    start = len(list(d.glob("c*.npz")))
    for ci, lo in enumerate(range(0, len(miss), chunk)):
        fx = fx or F.VJepaFeatures(frames=4)
        part = miss.iloc[lo: lo + chunk]
        z = W._extract_rows(part, fx, batch, workers, rl)
        tmp = d / f"c{start + ci:04d}.tmp.npz"
        np.savez(tmp, frame_name=part.frame_name.to_numpy().astype(str), mean=z)
        tmp.replace(d / f"c{start + ci:04d}.npz")
    return {"rows": len(t), "extracted": len(miss)}


def zmat() -> dict:
    t = pd.read_parquet(pdir("index.parquet"))
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    z = np.full((len(t), W.D_OP + W.D_VJ), np.nan, np.float16)
    for c in sorted(pdir("vjepa").glob("c*.npz")):
        with np.load(c) as q:
            i = pos.reindex(q["frame_name"].astype(str)).to_numpy()
            ok = ~np.isnan(i)
            z[i[ok].astype(int), W.D_OP:] = q["mean"][ok]
    for f in sorted(pdir("op_streams_vis", "cinque").glob("wl_*.npz")):
        with np.load(f) as q:
            i = pos.reindex(q["name"].astype(str)).to_numpy()
            ok = ~np.isnan(i)
            z[i[ok].astype(int), :W.D_OP] = q["temporal"][ok]
    have = ~np.isnan(z.astype(np.float32)).any(1)
    np.save(pdir("z.npy"), z)
    np.save(pdir("z_ok.npy"), have)
    return {"rows": len(t), "complete": int(have.sum())}


def prune(keep_stages=("pilot1", "pilot10")) -> dict:
    """Delete cams/ of runs whose every index row has a complete z (pilot runs kept for inspection)."""
    import shutil
    t = pd.read_parquet(pdir("index.parquet"))
    ok = np.load(pdir("z_ok.npy"))
    keep = set()
    for s in keep_stages:
        keep |= set(WL.stage_ids(s).route_id)
    full = pd.Series(ok).groupby(t.route_id.to_numpy()).all()
    gone = 0
    for rid in full[full].index:
        if rid in keep:
            continue
        cams = Path(t.adir[t.route_id == rid].iloc[0]) / "cams"
        if cams.exists():
            shutil.rmtree(cams)
            gone += 1
    return {"pruned_runs": gone}


def main():
    import argparse
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("index", "opspec", "vjepa", "z", "prune"))
    ap.add_argument("--out", default=str(data_dir() / "runs" / WL.NAME / "gen"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.step == "vjepa":
        rl = RunLog("wl", "vjepa")
        r = vjepa(rl=rl)
        rl.close()
    else:
        r = {"index": lambda: index(a.out, a.workers), "opspec": opspec, "z": zmat, "prune": prune}[a.step]()
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
