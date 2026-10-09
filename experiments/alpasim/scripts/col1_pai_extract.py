"""COL1, PAI track: compact per-scene records and the driver-side messages out of a finished pai_run.sh run. Reads only.
Runs on the Tokyo box inside the trusted image (alpasim_utils reads the .asl):

  docker run --rm -v /data:/data -e PYTHONPATH=/data/third_party/alpasim/src/utils alpasim-base:0.89.0 bash -c \
    'cd /repo && uv run python <this file> --run /data/runs/alpasim/<lane>/runs/<name> --out <dir> [--cam camera_front_wide_120fov]'

Writes <out>/logs.pkl (c1_extract.one_log per scene: simulated ego / actor poses, logged object trajectories, driver requests and
returns, ego observations, routes, metric series, controller trace, summary row, the driver's drive.jsonl records and start record) and
<out>/msgs/<scene>.pkl (the serialized driver-side messages in arrival order, images of --cam only: the input of col1_pai_replay.py).
Simulator state in these files is a label for analysis, never a driver input.
"""
import argparse
import asyncio
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_extract as X  # noqa: E402

KIND = {"driver_session_request", "driver_camera_image", "driver_ego_trajectory", "route_request", "driver_request"}


def one_msgs(a) -> str:
    run, scene, out, cam = a

    async def go():
        m_ = []
        async for k, m in X._read(X.asl_of(run, scene), KIND):
            if k != "driver_camera_image" or m.camera_image.logical_id == cam:
                m_.append((k, m.SerializeToString()))
        return m_
    pickle.dump(asyncio.run(go()), open(f"{out}/{scene}.pkl", "wb"), protocol=4)
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="camera_front_wide_120fov"), ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    sim = f"{a.run.rstrip('/')}/sim"
    os.makedirs(f"{a.out}/msgs", exist_ok=True)
    S = json.loads(Path(f"{sim}/aggregate/results-summary.json").read_text())["rollouts"]
    scenes = sorted(r["clipgt_id"] for r in S if os.path.isdir(f"{sim}/rollouts/{r['clipgt_id']}"))
    with ProcessPoolExecutor(a.jobs) as ex:
        out = dict(ex.map(X.one_log, [(sim, s) for s in scenes]))
        n = len(list(ex.map(one_msgs, [(sim, s, f"{a.out}/msgs", a.cam) for s in scenes])))
    for r in S:
        if r["clipgt_id"] in out:
            out[r["clipgt_id"]]["summary"] = r
    for line in open(f"{a.run}/driver/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "drive" and r["scene"] in out:
            r.pop("ms", None)
            out[r["scene"]].setdefault("rec", []).append(r)
    for line in open(f"{a.run}/driver/start.jsonl"):
        r = json.loads(line)
        if r["scene"] in out:
            out[r["scene"]]["start"] = r
    pickle.dump(out, open(f"{a.out}/logs.pkl", "wb"), protocol=4)
    print("logs", len(out), "scenes, msgs", n, "->", a.out, flush=True)


if __name__ == "__main__":
    main()
