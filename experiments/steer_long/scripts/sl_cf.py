#!/usr/bin/env python3
"""STEER-LONG speed-history counterfactual (research/next-round/steer-long.md amendment 2, item 1). The stored driver messages of the held-452
zero rollouts (eyes_vs_head msgs, six labels) are replayed in arrival order through the real driver class (sh30_driver.Driver, the label's
checkpoint, as eyes_vs_head replay does); at every decision the policy is called three times on the same frames:
  F  as served (the gate: equal to the stored plan)
  A  body velocities rescaled to the log's position-aligned speed at each history key (direction kept), longitudinal acceleration = the log's
  B  A, and the history key positions scaled radially by (log mean speed / ego mean speed) between the key and t0 (clipped to 0.5 .. 3)
Keys whose decision index is < 0 (cold start) keep their values. Log speed from zeros_mech per_decision (s_e, s_log_t), as sl_long.py.

  (box, envs/op-train, one GPU, a pool job)  sl_cf.py [--labels ...] [--limit N]
  -> $DATA_DIR/runs/steer_long/cf/<label>.pkl  {scene: [ {now, k, F, A, B (8, 3 poses), v_log_keys} ]}
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/anchor_offset/scripts"), str(REPO / "experiments/alpasim/scripts"),
                str(REPO / "experiments/alpasim/lib"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/eyes_vs_head/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
MSGS = DATA / "runs/eyes_vs_head/msgs"
ZM = DATA / "runs/zeros_mech/collect/20261012-155215/per_decision.parquet"
OUT = DATA / "runs/steer_long/cf"
LABELS = ["YR10m10-F-s0", "YR10m10-F-s1", "YS-L18-s0", "YS-L18-s1", "YS2G-L18-s0", "YS2G-L18-s1"]
T0, DT = 17_000, 500_000


def log_speed(g):
    """per_decision rows of one (label, scene) -> v_log at each decision's station (position-aligned)."""
    t = g.t_us.to_numpy(float)
    s = np.maximum.accumulate(g.s_log_t.to_numpy(float))
    v = np.gradient(s, t / 1e6)
    su, iu = np.unique(s, return_index=True)
    tq = np.interp(g.s_e.to_numpy(float), su, t[iu])
    return dict(zip(g.k.astype(int), np.interp(tq, t, v)))


def main():
    import pandas as pd
    import torch
    import sh30_driver as D
    from ev import Ctx
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=LABELS)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    PD = pd.read_parquet(ZM)
    VL = {k: log_speed(g.sort_values("k")) for k, g in PD.groupby(["label", "scene"])}
    OUT.mkdir(parents=True, exist_ok=True)
    pb, ctx = D.egodriver_pb2, Ctx()
    with Run("steer_long", "cf", config=vars(a)) as run:
        for lab in a.labels:
            core = D.C.Core(lab, "cuda", "backwarp", 1.0, "gpu")
            drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
            cur, stash = {}, {}
            orig = core.plan

            def plan(frames, pose, vel, acc, cmd, cam_t, _o=orig, **k):
                o = _o(frames, pose, vel, acc, cmd, cam_t, **k)
                vl = VL.get((lab, cur["scene"]), {})
                kd = int(round((cur["now"] - T0) / DT))
                m = len(pose)
                keys = [kd - (m - 1 - i) for i in range(m)]
                v_log = np.array([vl.get(q, np.nan) for q in keys], float)
                V = np.asarray(vel, np.float64).copy()
                sp = np.hypot(V[:, 0], V[:, 1])
                ok = np.isfinite(v_log)
                VA = V.copy()
                for i in np.flatnonzero(ok):
                    VA[i] = V[i] * v_log[i] / sp[i] if sp[i] > 0.3 else [v_log[i], 0.0]
                AA = np.asarray(acc, np.float64).copy()
                if ok[-1] and m > 1 and ok[-2]:
                    AA[0] = (v_log[-1] - v_log[-2]) / 0.5
                oA = _o(frames, pose, VA.astype(np.asarray(vel).dtype), AA.astype(np.asarray(acc).dtype), cmd, cam_t, **k)
                P = np.asarray(pose, np.float64).copy()
                for i in range(m - 1):
                    if ok[i:].all() and sp[i:].mean() > 0.3:
                        P[i, :2] *= float(np.clip(v_log[i:].mean() / sp[i:].mean(), 0.5, 3.0))
                oB = _o(frames, P.astype(np.asarray(pose).dtype), VA.astype(np.asarray(vel).dtype), AA.astype(np.asarray(acc).dtype), cmd, cam_t, **k)
                stash.update(F=np.asarray(o["poses"], np.float32).copy(), A=np.asarray(oA["poses"], np.float32).copy(),
                             B=np.asarray(oB["poses"], np.float32).copy(), v_log_keys=v_log, v_ego_keys=sp, keys=keys)
                return o
            core.plan = plan
            out, md = {}, MSGS / lab
            scenes = sorted(p.stem for p in md.glob("*.pkl"))[: a.limit or None]
            for scene in run.tqdm(scenes, desc=lab):
                rec, uuid = [], None
                cur["scene"] = scene
                for kind, raw in pickle.load(open(md / f"{scene}.pkl", "rb")):
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
                        cur["now"] = int(req.time_now_us)
                        stash.clear()
                        try:
                            drv.drive(req, ctx)
                        except RuntimeError as e:
                            run.info(f"{lab} {scene} t {req.time_now_us}: {e}")
                            continue
                        rec.append(dict(now=cur["now"], **stash))
                drv.sessions.pop(uuid, None)
                out[scene] = rec
            pickle.dump(out, open(OUT / f"{lab}.pkl", "wb"), protocol=4)
            run.info(f"{lab}: {len(out)} scenes, {sum(map(len, out.values()))} decisions")
            del core, drv
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
