"""CARLA half of the image-command test, step 4 (openpilot env, CPU): packed openpilot model frames for every sample of
geom_<run>.pkl (img_carla_geom.py) and the final carla.pkl.

The P4 rig's JPEG triplets are rendered into the road / wide model frames exactly as op_common_cause's carla domain did
(scripts/p5_openpilot.render with jevdrive.p5_openpilot.carla_calib: rotation-only reprojection, model-frame axes =
vehicle axes, origin = the front camera, 1.519 / 0.026 / 1.806 m from the rear axle on the ground). Per sample one npz
$DATA_DIR/runs/op_img_cmd/carla/frames/<token>.npz: frames (k, 2, 6, 128, 256) uint8 [road, wide] of the 5 Hz frames
t0 - 1.6 s ... t0 (oldest first), frame_t (k,) s. carla.pkl = geom_<run>.pkl + `frames` (that path).

  $DATA_DIR/envs/openpilot/bin/python experiments/op_img_cmd/scripts/img_carla_frames.py [--run rec] [--workers 24]
"""
import argparse, json, os, pickle, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_img_cmd" / "carla"


def _init():
    import wod_zeroshot_openpilot as WZ
    from jevdrive import p5_openpilot as P
    WZ._init({}, {"carla": P.carla_calib()}, ".")


def job(s):
    import p5_openpilot as PO
    dst = ROOT / "frames" / f"{s['token']}.npz"
    if not dst.exists():
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, frames=PO.render(s["files"], "carla"), frame_t=s["frame_t"])
        tmp.replace(dst)
    return str(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rec")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    G = pickle.load(open(ROOT / f"geom_{a.run}.pkl", "rb"))
    (ROOT / "frames").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for s, f in zip(G, ex.map(job, G, chunksize=2)):
            s["frames"] = f
    name = "carla" if a.run == "rec" else f"carla_{a.run}"
    pickle.dump(G, open(ROOT / f"{name}.pkl", "wb"))
    summ = dict(samples=len(G), routes=len({s["log"] for s in G}), by_tag=Counter(s["tag"] for s in G),
                by_taken=Counter(s["taken"] for s in G), by_cmd=Counter(s["cmd"] for s in G),
                by_speed=Counter("stop" if s["v"] < 0.5 else "low" if s["v"] < 3 else "moving" for s in G),
                by_town=Counter(s["town"] for s in G))
    (ROOT / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
