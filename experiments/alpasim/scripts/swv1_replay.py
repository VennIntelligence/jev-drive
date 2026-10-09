#!/usr/bin/env python3
"""Lane SWV1 (plans/2026-10-10-swv1-prereg.md), nuPlan half: COL1's offline replay (col1_lead.py replay: the logged driver-side messages
through the real driver class, plans reproduce) keeping every geometric head the serving path discards. One GPU, a pool job:

  python -m jevdrive.cl submit --name swv1-replay --vram 8 --cpu 8 --log-dir $DATA_DIR/runs/alpasim/swv1/pool -- \
    env ALPASIM_SRC=$DATA_DIR/third_party/alpasim $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/swv1_replay.py

Reads $DATA_DIR/runs/alpasim/col1/lead/{spec.json, msgs/<group>/<scene>.pkl}; writes $DATA_DIR/runs/alpasim/swv1/replay/<group>.pkl:
{scene: [per decision: now, poses (served plan, (8, 3) rear axle), poses_p0 (the shipped policy weights without the adapter on the same
vision tokens), cmd, cam (camera x, y in the ego frame), n_slots, FT / P0: {head: float16 raw slice} for every output head of at most
1100 values (plan with its spread, lead, lead_prob, lane_lines, lane_lines_prob, road_edges, meta, desire_state, pose ...)]},
and slices.json. Simulator state is not read here.
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
LD = DATA / "runs/alpasim/col1/lead"
O = DATA / "runs/alpasim/swv1"
GROUPS = ("P2H10-F-s0", "P2H10-F-s1", "APY10m10-AB-s0", "APY10m10-AB-s1", "ctrl")


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(GROUPS)), ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    import sh30_driver as D
    import pp_train as T
    I, Z = D.C.I, D.C.Z
    spec = json.loads((LD / "spec.json").read_text())
    (O / "replay").mkdir(parents=True, exist_ok=True)
    p0 = T.load_pmodel("P0", torch.device("cuda"))
    pb, ctx = D.egodriver_pb2, Ctx()
    for g in a.groups:
        sp = spec[g]
        if sp["driver"] == "sh30":
            core = D.C.Core(sp["tag"], "cuda", "backwarp")
            drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        else:
            import ap2_driver as AD
            core = AD.AC.Core(sp["tag"], "cuda", "")
            drv = AD.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        sl, cap, last = core.model.net.slices, {}, {}
        keep = {k: v for k, v in sl.items() if v.stop - v.start <= 1100}
        (O / "slices.json").write_text(json.dumps({k: [int(v.start), int(v.stop)] for k, v in sl.items()}))
        core.model.register_forward_hook(lambda m, args, out: cap.update(H=args[0], ego=args[1], tc=args[2], out=out))
        orig = core.plan

        def plan(*x, _o=orig, **k):
            last["o"] = _o(*x, **k)
            return last["o"]
        core.plan = plan
        out = {}
        scenes = sp["scenes"][:a.limit] if a.limit else sp["scenes"]
        for i, scene in enumerate(scenes):
            rec, uuid = [], None
            for kind, raw in pickle.load(open(LD / "msgs" / g / f"{scene}.pkl", "rb")):
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
                    req = pb.DriveRequest.FromString(raw)
                    drv.drive(req, ctx)
                    with torch.no_grad():
                        o0 = p0(cap["H"], cap["ego"], cap["tc"]).float()[0]
                    ft = cap["out"].float()[0]
                    cam = np.asarray(drv.sessions[uuid].cam["t"], np.float64)
                    m0 = o0[core.pi].reshape(33, 15).cpu().numpy()
                    r = dict(now=int(req.time_now_us), poses=last["o"]["poses"].copy(), n_slots=int(last["o"]["valid"].sum()),
                             poses_p0=I.to_rear(m0[:, 0:3], m0[:, 11], I.T_IDXS, cam[:2], Z.T_OUT, "lever"),
                             cmd=int(np.argmax(drv.sessions[uuid].cmd)), cam=cam[:2].copy())
                    for name, v in (("FT", ft), ("P0", o0)):
                        r[name] = {k: v[s].cpu().numpy().astype(np.float16) for k, s in keep.items()}
                    rec.append(r)
            drv.sessions.pop(uuid, None)
            out[scene] = rec
            if i % 25 == 0:
                print(g, i, len(scenes), scene, flush=True)
        pickle.dump(out, open(O / "replay" / f"{g}.pkl", "wb"), protocol=4)
        print("replay", g, len(out), "scenes", flush=True)
        del core, drv
        torch.cuda.empty_cache()
    (O / "REPLAY_DONE").touch()


if __name__ == "__main__":
    main()
