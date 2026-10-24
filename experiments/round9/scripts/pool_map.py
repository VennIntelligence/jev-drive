#!/usr/bin/env python3
"""ROUND9 POOL-DESIGN map extract (plans/2026-10-23-pool-design.md): behavior_map.py's per-scene map record for the PAI training scenes
(the 719 curated-train scenes the DAgger rollouts used), plus the OpenDRIVE geoReference longitude that behavior_stats.scene_facts needs
for its km/h rule. Box, AlpaSim venv, CPU only; nothing is simulated.

  pool_map.py --bm <dir with behavior_map.py> --cfgs <generated-user-config-0.yaml>... --out <pkl> [--workers N]

Output = behavior_map's pickle layout (scenes: {scene: record}, errors) with one extra field per record: `lon` (geoReference lon_0, NaN
if absent).
"""
import argparse
import os
import pickle
import re
import sys
import zipfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

LON = re.compile(r"lon_0=([-+0-9.eE]+)")


def job(u):
    import behavior_map as BM
    import rf_tax as RT
    scene, cfg = u
    o = BM.scene_job(u)
    lon = np.nan
    try:
        z = zipfile.ZipFile(RT.loader(cfg).get_data_source(scene).source)
        if "map.xodr" in z.namelist():
            m = LON.search(z.read("map.xodr")[:20000].decode(errors="ignore"))
            lon = float(m.group(1)) if m else np.nan
    except Exception:  # noqa: BLE001
        pass
    o["lon"] = lon
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", required=True)
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    sys.path.insert(0, a.bm)
    import yaml
    from concurrent.futures import ProcessPoolExecutor
    jobs = {}
    for c in a.cfgs:
        for s in yaml.safe_load(open(c))["scenes"]:
            jobs.setdefault(s["scene_id"], c)
    units = sorted(jobs.items())
    print(len(units), "scenes", flush=True)
    res, err = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, u): u[0] for u in units}
        for i, f in enumerate(futs):
            try:
                o = f.result()
                res[o["scene"]] = o
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:300]))
            if i % 100 == 0:
                print(i, "done", len(err), "errors", flush=True)
    pickle.dump(dict(scenes=res, errors=err), open(a.out, "wb"), protocol=4)
    print(len(res), "scenes,", len(err), "errors, lon missing", sum(not np.isfinite(o["lon"]) for o in res.values()), "->", a.out, flush=True)


if __name__ == "__main__":
    main()
