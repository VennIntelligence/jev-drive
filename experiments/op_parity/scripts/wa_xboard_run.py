"""WA-JEPA inference for wa-xboard (plans/2026-10-08-wa-xboard-prereg.md). wajepa env, cwd ~/data/third_party/wajepa, one card from the pool.

  python wa_xboard_run.py --work DIR --jobs TAG:REQ:CACHE[:SEED] ...    -> DIR/out_<TAG>.npz (keys, traj (n, 8, 3))

The released checkpoint through top10_t2/wajepa_run.py (the repo's NAVSIM feature builder, fp32 = its NAVSIM path, batch 1, 4-step flow). Only the images come from a
uint8 memmap cache (DIR/<CACHE>/{paths,cache}.npy); CACHE '-' = no images (all black). SEED overrides the flow noise seed (default: the config's 1).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/top10/lib/top10_t2"))
import wajepa_run as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--jobs", nargs="+", required=True)
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
work = Path(a.work)
agent = R.build_agent()
default_seed = agent.model.flow_inference_seed
print("default flow seed", default_seed, flush=True)
for job in a.jobs:
    tag, req, cache, *seed = job.split(":")
    out = work / f"out_{tag}.npz"
    if out.exists():
        print("skip", tag)
        continue
    agent.model.flow_inference_seed = int(seed[0]) if seed else default_seed
    with np.load(work / req) as f:
        z = {k: f[k] for k in f.files}
    c = None if cache == "-" else (np.load(work / cache / "paths.npy"), np.load(work / cache / "cache.npy", mmap_mode="r"))
    t0 = time.time()
    traj = R.run(agent.model, z, np.arange(len(z["keys"])), False, a.workers, c)
    assert np.isfinite(traj).all(), tag
    np.savez_compressed(str(out) + ".tmp.npz", keys=z["keys"], traj=traj)
    Path(str(out) + ".tmp.npz").replace(out)
    print(f"{tag}: {len(traj)} plans in {time.time() - t0:.0f} s, seed {agent.model.flow_inference_seed}", flush=True)
