"""COL1, PAI track: offline replay of the logged driver-side messages (col1_pai_extract.py msgs) through the real PAI driver class, and
at every decision, on the same vision tokens, the outputs the serving path throws away. No simulator; open loop (the state at every
decision is the logged one). Runs on the Tokyo box inside the driver image with lib/pai_{core,driver}.py mounted:

  docker run --rm --gpus device=<g> -v <lib>/pai_core.py:/app/jev-drive/experiments/alpasim/lib/pai_core.py:ro (same for pai_driver.py)
    -v /data/runs/alpasim/col1:/col1 -e SH30_TAG=P2H10-F-s0 <image> python /col1/code/scripts/col1_pai_replay.py --msgs <dir> --out <dir>

Per scene -> <out>/<scene>.npz, one row per decision:
  now, t0, cmd            as fed
  run                     the plan the driver class returned in the replay (8, 3): compared with the run's logged plan by the analysis
  ft / ftS / p0 poses     rear-axle plan of the served checkpoint with the fed command, with the command forced to straight, and of the
                          shipped policy weights without the adapter (pp_train.load_pmodel("P0")) on the same tokens
  mu_ft, mu_p0            the raw plan means (33, 15), float16
  lead_*, lp_*            the lead head (144: 72 means = 3 hypotheses x 6 times x (x, y, v, a), then spreads) and lead_prob logits (3)
  frames, frame_k         the newest slot's road + wide model frames (2, 6, 128, 256) uint8 of every --frame-every-th decision
"""
import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/jev-drive/experiments/alpasim/lib")


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="P2H10-F-s0"), ap.add_argument("--frame-every", type=int, default=2)
    a = ap.parse_args()
    import pai_driver as D
    import pp_train as T
    import torch
    C, PC, I = D.C, D.PC, D.I
    PA, Z, A = C.PA, C.Z, None
    core = C.Core(a.tag, "cuda")
    A = core.A
    drv = D.Driver(core, Path(tempfile.mkdtemp()), "backwarp", 1, 0, False)
    p0 = T.load_pmodel("P0", core.dev)
    sl, pb = core.model.net.slices, D.egodriver_pb2
    cap, orig = {}, PC.plan

    def plan(core_, cur, valid, P, V, acc, cmd, cam_t, lht=False):
        cap.update(cur=cur, valid=valid, P=P, V=V, acc=acc, cmd=cmd, cam_t=cam_t)
        cap["o"] = orig(core_, cur, valid, P, V, acc, cmd, cam_t, lht)
        return cap["o"]
    PC.plan = plan

    def extra():
        cur, valid, cam_t = cap["cur"], cap["valid"], np.asarray(cap["cam_t"], np.float64)
        prev = np.concatenate([np.zeros((1,) + C.FRAME, np.uint8), cur[:-1]])
        out = {}
        with torch.no_grad():
            p, c = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
            H = core.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *A.H_SHAPE)
            tc = torch.tensor([[1.0, 0.0]], device=core.dev)
            for name, model, cmd in (("ft", core.model, cap["cmd"]), ("ftS", core.model, [0, 1, 0, 0]), ("p0", p0, cap["cmd"])):
                ego = PA.ego_features(cap["P"], cap["V"], np.tile(np.asarray(cap["acc"], np.float32), (4, 1)), np.asarray(cmd, np.float32))
                o = model(H, torch.from_numpy(ego[None]).to(core.dev), tc).float()[0].cpu().numpy()
                mu = o[sl["plan"].start:sl["plan"].start + 495].reshape(33, 15)
                out[name] = dict(mu=mu, poses=I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever"),
                                 lead=o[sl["lead"]], lp=o[sl["lead_prob"]])
        return out

    Path(a.out).mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    for f in sorted(Path(a.msgs).glob("*.pkl")):
        if (Path(a.out) / f"{f.stem}.npz").exists():
            continue
        R, k, uuid = {n: [] for n in ("now", "t0", "cmd", "run", "ft", "ftS", "p0", "mu_ft", "mu_p0", "lead_ft", "lp_ft", "lead_p0", "lp_p0",
                                      "frames", "frame_k")}, 0, None
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
                req = pb.DriveRequest.FromString(raw)
                cap.clear()
                try:
                    drv.drive(req, ctx)
                except RuntimeError as e:
                    print(f.stem, k, "drive failed:", e, flush=True)
                if "o" in cap:
                    x = extra()
                    R["now"].append(int(req.time_now_us)), R["t0"].append(int(drv.sessions[uuid].anchor.timestamp_us))
                    R["cmd"].append(int(np.argmax(cap["cmd"]))), R["run"].append(cap["o"]["poses"])
                    for n in ("ft", "ftS", "p0"):
                        R[n].append(x[n]["poses"])
                    for n in ("ft", "p0"):
                        R[f"mu_{n}"].append(x[n]["mu"].astype(np.float16)), R[f"lead_{n}"].append(x[n]["lead"]), R[f"lp_{n}"].append(x[n]["lp"])
                    if k % a.frame_every == 0:
                        R["frames"].append(cap["cur"][-1].copy()), R["frame_k"].append(k)
                k += 1
        drv.sessions.pop(uuid, None)
        np.savez_compressed(Path(a.out) / f"{f.stem}.npz", **{n: np.array(v) for n, v in R.items()})
        print(f.stem, k, "calls,", len(R["now"]), "decisions", flush=True)


if __name__ == "__main__":
    main()
