#!/usr/bin/env python3
"""STOPLINE-PROBE step 1 (Tokyo, inside alpasim-base:0.89.0 with /mnt/nre-data and /repo/src mounted): per-scene facts for the
behaviour labels of red / green, and the index of the PAI_DUMP frames. CPU only; run dirs and scene data are read, never written.

  slp_facts.py --runs <run dir> [<run dir> ...] --out facts.pkl [--workers N]

A run dir is one AlpaSim chunk (scenes of a chunk without dumps get scene facts only) (<run>/driver/dump/s??_k???.npz, <run>/driver/drive.jsonl, <run>/sim/generated-user-config-0.yaml).
Output pickle:
  dumps   list of dict(file, run, scene, now, t0, pose (x, y, yaw at t0, rollout local frame), v (ego speed m/s over the last 0.5 s), log_follow)
  scenes  {scene: dict(T, P, Y (logged rig path, local frame), wait (list of dict id, kind, pts), signs (list of dict type, pos),
                       objs {id: dict T, P (0.5 s grid), cls, static}, tl_status (map.traffic_light_status is not None))}
Only objects that come within 40 m of the logged path are kept.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

_L = {}


def loader(cfg):
    if cfg not in _L:
        from alpasim_runtime.config import UserSimulatorConfig
        from alpasim_runtime.scene_loader import build_scene_loader
        from omegaconf import OmegaConf
        _L[cfg] = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(cfg)))
    return _L[cfg]


def near(P, Q, r):
    return len(Q) and len(P) and np.hypot(*(P[::5, None] - Q[None]).transpose(2, 0, 1)).min() < r


def scene_job(job):
    scene, cfg = job
    from trajdata.maps.vec_map_elements import MapElementType as M
    ds = loader(cfg).get_data_source(scene)
    tr = ds.rig.trajectory
    T, P, Y = np.asarray(tr.timestamps_us, np.int64), np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
    m = ds.map
    wait = [dict(id=str(w.id), kind=str(w.wait_line_type), pts=np.asarray(w.polyline.points, float)[:, :2])
            for w in m.elements[M.WAIT_LINE].values() if near(P, np.asarray(w.polyline.points, float)[:, :2], 30)]
    signs = [dict(type=str(s.sign_type), pos=np.asarray(s.position, float)[:2]) for s in m.elements[M.TRAFFIC_SIGN].values()
             if near(P, np.asarray(s.position, float)[None, :2], 40)]
    objs = {}
    for oid in ds.traffic_objects.keys():
        o = ds.traffic_objects[oid]
        t = o.trajectory
        To = np.asarray(t.timestamps_us, np.int64)
        if len(To) < 1:
            continue
        Po = np.asarray(t.positions, float)[:, :2]
        if not near(P, Po, 40):
            continue
        g = np.arange(To[0], To[-1] + 1, 500_000) if len(To) > 1 else To
        objs[str(oid)] = dict(T=g, P=np.c_[np.interp(g, To, Po[:, 0]), np.interp(g, To, Po[:, 1])].astype(np.float32),
                              cls=str(o.label_class), static=bool(o.is_static))
    return scene, dict(T=T, P=P, Y=Y, wait=wait, signs=signs, objs=objs, tl_status=m.traffic_light_status is not None)


def dump_index(run):
    """Dump frames of one run dir, joined to the drive.jsonl record of the same (scene, now) for the ego pose and speed."""
    rec, f = {}, Path(run) / "driver/drive.jsonl"
    if not f.exists():
        return []
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") == "drive" and r.get("infer"):
            rec[(r["scene"], int(r["now"]))] = r
    lf = "log" in Path(run).name                               # onp_log_driver renders: the ego follows the logged path
    out = []
    for f in sorted(glob.glob(f"{run}/driver/dump/s*_k*.npz")):
        z = np.load(f)
        sc, now = str(z["scene"]), int(z["now"])
        r = rec.get((sc, now))
        if r is None:
            continue
        H = np.asarray(r["hist"], float)                       # poses at t0 - 1.5 / 1.0 / 0.5 / 0 s in the t0 frame
        out.append(dict(file=f, run=str(run), scene=sc, now=now, t0=int(z["t0"]), pose=np.asarray(r["anchor"], float),
                        v=float(np.hypot(*(H[-1, :2] - H[-2, :2])) / 0.5), log_follow=lf))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        dumps = sum(ex.map(dump_index, a.runs), [])
    jobs = {}
    for d in dumps:
        jobs.setdefault(d["scene"], str(Path(d["run"]) / "sim/generated-user-config-0.yaml"))
    import yaml
    for run in a.runs:                                         # scenes of runs without dumps too (scene facts only)
        cfg = Path(run) / "sim/generated-user-config-0.yaml"
        if cfg.exists():
            for x in yaml.safe_load(cfg.read_text()).get("scenes", []):
                jobs.setdefault(x["scene_id"], str(cfg))
    print(len(dumps), "dump frames,", len(jobs), "scenes", flush=True)
    scenes, err = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(scene_job, j): j[0] for j in sorted(jobs.items())}
        for f in futs:
            try:
                s, v = f.result()
                scenes[s] = v
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:300]))
    pickle.dump(dict(dumps=dumps, scenes=scenes, errors=err), open(a.out, "wb"), protocol=4)
    print(len(scenes), "scenes,", len(err), "errors ->", a.out, flush=True)
    for e in err[:5]:
        print(e)


if __name__ == "__main__":
    main()
