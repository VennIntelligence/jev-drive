"""C1 offline replay: feed the logged driver-side messages of a finished rollout back through the real driver class (no simulator) and
read its plans again, optionally with the command forced. Open loop: the state at every decision is the logged one, so a forced command
changes that one plan only. Also returns the model's input frames for chosen scenes.

  c1_replay.py --driver sh30|ap2 --msgs <dir of c1_extract.py msgs> --spec <json> --out <pkl>      (box, envs/op-train, one GPU)

spec: {scene: {"ks": [decisions replayed with every command], "frames": bool}}. Per scene and decision the output holds the plan as
run ("run", checked against the logged plan by c1_review) and with the command forced to left / straight / right ("L", "S", "R").
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
FORCED = {"L": [1, 0, 0, 0], "S": [0, 1, 0, 0], "R": [0, 0, 1, 0]}


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, choices=["sh30", "ap2"]), ap.add_argument("--msgs", required=True)
    ap.add_argument("--spec", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sh30_driver as D
    if a.driver == "sh30":
        core = D.C.Core(os.environ.get("SH30_TAG", "SH30-F-s0"), "cuda", os.environ.get("SH30_COLD", "backwarp"))
        drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
    else:
        import ap2_driver as AD
        core = AD.AC.Core(os.environ.get("AP2_TAG", "AP2-AB-s0"), "cuda", "")
        drv = AD.Driver(core, Path(tempfile.mkdtemp()), 0, False)
    pb = D.egodriver_pb2
    last = {}
    orig = core.plan

    def plan(*x, **k):
        last["o"] = orig(*x, **k)
        return last["o"]
    core.plan = plan
    spec, ctx, out = json.loads(Path(a.spec).read_text()), Ctx(), {}
    for i, (scene, sp) in enumerate(sorted(spec.items())):
        f = Path(a.msgs) / f"{scene}.pkl"
        if not f.exists():
            continue
        rec, k, uuid = dict(plans={}, frames={}, cmd={}), 0, None
        for kind, raw in pickle.load(open(f, "rb")):
            if kind == "driver_session_request":
                req = pb.DriveSessionRequest.FromString(raw)
                uuid = req.session_uuid
                drv.start_session(req, ctx)
            elif kind == "driver_camera_image":
                drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
            elif kind == "driver_ego_trajectory":
                drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
            elif kind == "route_request":
                drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
            elif kind == "driver_request":
                req, s = pb.DriveRequest.FromString(raw), drv.sessions[uuid]
                keep = np.asarray(s.cmd, np.float32).copy()
                if k in sp.get("ks", []) or sp.get("frames") or sp.get("all"):
                    drv.drive(req, ctx)
                    rec["plans"][k] = {"run": last["o"]["poses"].copy()}
                    rec["cmd"][k] = int(np.argmax(keep))
                    if sp.get("frames"):
                        rec["frames"][k] = dict(cur=last["o"]["cur"][[0, 4, 7]].copy(), valid=last["o"]["valid"].copy())
                    if k in sp.get("ks", []):
                        for name, c in FORCED.items():
                            s.cmd = np.array(c, np.float32)
                            drv.drive(req, ctx)
                            rec["plans"][k][name] = last["o"]["poses"].copy()
                        s.cmd = keep
                k += 1
        drv.sessions.pop(uuid, None)
        out[scene] = rec
        if i % 20 == 0:
            print(i, len(spec), scene, flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print("replay", a.driver, len(out), "scenes ->", a.out, os.path.getsize(a.out) >> 20, "MB", flush=True)


if __name__ == "__main__":
    main()
