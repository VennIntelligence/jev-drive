"""Night queue 2, N2 desire executor check (todos/2026-09-26-night-queue-2.md, N2 and its [A-N2] entry): openpilot's
native plan with a laneChangeLeft / laneChangeRight desire against the same stream without one, open loop on P5 v1
straight frames. Targets and warm-up spans from `python -m jevdrive.night2_n2 targets --set <set>`.

Each target: three arms (none, left = 3, right = 4), each from a zero state 8 s (40 frames) before the target; the
one-hot is held from the target frame on and OPModel turns it into modeld's rising-edge pulse. Reads the plan at the
target frame and the configured frames after it; y at 3 s, left positive (openpilot's plan y points right).
Output: processed/<set>/night2_desire_out.jsonl, one line per (target, model, read point). Resumable per target.

  CUDA_VISIBLE_DEVICES=2 $DATA_DIR/envs/openpilot/bin/python scripts/night2_desire.py --set carla_p5v1_ba
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wod_zeroshot_openpilot as WZ  # noqa: E402
import p5_openpilot as R  # noqa: E402
from drive_backbones_openpilot import bounded_map  # noqa: E402
from jevdrive import p5_openpilot as P  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

ARMS = {"none": 0, "left": 3, "right": 4}


def job(t):
    return t, R.render(t["files"])


def y_left_3s(m, out, v_ego):
    from jevdrive.openpilot.model import T_IDXS, decode
    d = decode(out, m.slices, v_ego, WZ.ACTION_T)
    return float(-np.interp(3.0, T_IDXS, d["plan_pos"][:, 1])), float(np.interp(3.0, T_IDXS, d["plan_pos"][:, 0]))


def run_arm(m, frames, pulse, reads, desire_idx, v_ego):
    m.reset()
    eye = np.eye(8, dtype=np.float32)
    want, got = {pulse + r: r for r in reads}, {}
    for j in range(max(want) + 1):
        des = eye[desire_idx] if (desire_idx and j >= pulse) else eye[0]
        for _ in range(1 if m.skip == 1 else R.HOLD):
            out = m.step(frames[j], desire=des, action_t=WZ.ACTION_T)
        if j in want:
            got[want[j]] = y_left_3s(m, out, v_ego)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="carla_p5v1_ba")
    ap.add_argument("--models", nargs="+", default=list(P.MODELS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel
    root = data_dir() / "processed" / a.set
    plan = json.loads((root / "night2_desire_targets.json").read_text())
    outf = root / "night2_desire_out.jsonl"
    done = {json.loads(x)["key"] for x in open(outf)} if outf.exists() else set()
    items = [t for t in plan["targets"] if t["key"] not in done][: a.limit or None]
    log = RunLog("night2_n2", f"desire-{a.set}")
    log.event("start", args=vars(a), targets=len(items), done=len(done))
    calib = {R.SEQ: P.carla_calib()}
    WZ._init({}, calib, ".")
    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=WZ._init, initargs=({}, calib, ".")) as ex, open(outf, "a") as fh:
        list(ex.map(int, range(a.workers)))      # fork before the TensorRT sessions exist
        models = {k: OPModel(k, WZ.MODELS[k], context_rate=(k == "lebowski")) for k in a.models}
        for t, frames in bounded_map(ex, job, items, 2 * a.workers):
            lines = []
            for k, m in models.items():
                arm = {name: run_arm(m, frames, t["pulse"], t["read"], idx, t["v_ego"]) for name, idx in ARMS.items()}
                for r in t["read"]:
                    y0 = arm["none"][r][0]
                    lines.append({"key": t["key"], "bin": t["bin"], "v_ego": t["v_ego"], "model": k, "read": r,
                                  "y_none": y0, "x_none": arm["none"][r][1], "d_left": arm["left"][r][0] - y0,
                                  "d_right": arm["right"][r][0] - y0})
            fh.write("".join(json.dumps(x) + "\n" for x in lines))
            fh.flush()
            n += 1
            if n % 25 == 0 or n == len(items):
                el = time.time() - t0
                log.info(f"[{n}/{len(items)}] {el:.0f} s, ETA {(len(items) - n) * el / n / 60:.1f} min")
                log.event("progress", n=n, wall_s=el)
    log.event("end", targets=n, wall_s=time.time() - t0)


if __name__ == "__main__":
    main()
