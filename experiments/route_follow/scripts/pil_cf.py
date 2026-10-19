#!/usr/bin/env python3
"""PAI-INPUT-LOOP item 2c (plans/2026-10-19-pai-input-loop-prereg.md): the speed-history counterfactual on PAI. A read's stored driver
messages (pai_input_loop.sh msgs: <msgs>/<scene>.pkl, arrival order) are replayed through the real pai_driver.Driver (the checkpoint, the
read's serve env); at every inference the policy is called on the same slot frames three times (steer_long/scripts/sl_cf.py's arms):
  F  as served (gate: equal to the stored `poses_model`, else `poses`)
  A  history velocities rescaled to the log speed at the ego's station at each history key (direction kept), longitudinal acceleration from it
  B  A, and the history key positions scaled radially by (log mean speed / ego mean speed) from the key to t0 (clip 0.5 .. 3)
Log speed: rf_tax.Roll of the same rollout (the ego's rear axle on the controller trace at the key time, projected on the logged path).

  (box, envs/op-train, one card, a pool job)  pil_cf.py --msgs DIR --tax N0.pkl --drive 'glob of drive.jsonl' --tag NTV0-L18-s0 --out cf/s0.pkl
      [--serve 'JEV_VCONT=1.0 JEV_LEAD=1'] [--limit N] [--shard i/n]
  -> cf.pkl {scene: [ {now, t0, F, A, B, stored, v_log_keys, v_ego_keys, v_trace_t0} ]}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2] / "alpasim/lib")]


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs", required=True), ap.add_argument("--tax", required=True), ap.add_argument("--drive", required=True)
    ap.add_argument("--tag", default="NTV0-L18-s0"), ap.add_argument("--out", required=True)
    ap.add_argument("--serve", default="JEV_VCONT=1.0 JEV_LEAD=1"), ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th scene from the i-th")
    a = ap.parse_args()
    for kv in a.serve.split():
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.environ["SH30_TAG"] = a.tag
    import rf_tax as TX
    import pai_driver as D
    PC = D.PC
    rolls = {o["scene"]: TX.Roll(o) for o in pickle.load(open(a.tax, "rb"))["rolls"]}
    stored = {}
    for f in glob.glob(a.drive):
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "drive" and r.get("infer"):
                stored[(r["scene"], r["now"])] = np.asarray(r.get("poses_model") or r["poses"], np.float32)
    # the core as pai_driver.main builds it (route memory, ACC route, reflex, lead output)
    D.RMEM = D.load_branch(a.tag)
    core = D.C.Core(a.tag, "cuda")
    if D.RMEM:
        core.rm, core.rm_dim = True, D.RMEM.dim_of(a.tag)
        D.RMEM.SERVE_ORDER, D.RMEM.SERVE_ANCHOR = D.RMEM.head_order(a.tag), D.RMEM.head_anchor(a.tag)
    if D.FX.ACC:
        sys.path.insert(0, str(HERE.parents[2] / "route_mem/lib"))
        import rmem
        D.ACCR = rmem
    if getattr(core.model, "reflex", False):
        import reflex
        import reflex_mem
        D.RXM, core.rx, core.RX = reflex_mem, True, reflex
    core.lead_out, core.base_out = D.FX.NEED_LEAD or D.VH.NEED_LEAD, D.FX.BASE > 0
    drv = D.Driver(core, Path(tempfile.mkdtemp()), os.environ.get("PAI_COLD", "backwarp"), 1, 0, False, os.environ.get("PAI_KEYWARP") == "1")
    pb, ctx, cur, stash = D.egodriver_pb2, Ctx(), {}, {}
    o_slots, o_plan = PC.slots, PC.plan
    TK = np.asarray(PC.T_KEY_US, np.int64)

    def slots(frames, t0, *x, **k):
        cur["t0"] = int(t0)
        return o_slots(frames, t0, *x, **k)

    def plan(core_, fr, valid, P, V, acc, *x, **k):
        o = o_plan(core_, fr, valid, P, V, acc, *x, **k)
        R, t0 = cur["roll"], cur["t0"]
        tk = t0 + TK
        st = R.log_station(np.array([R.pose(t)[:2] for t in tk]))
        v_log = np.array([R.vlog_at(s) for s in st])
        Vd = np.asarray(V, np.float64)
        sp = np.hypot(Vd[:, 0], Vd[:, 1])
        VA = Vd.copy()
        for i in range(len(VA)):
            VA[i] = Vd[i] * v_log[i] / sp[i] if sp[i] > 0.3 else [v_log[i], 0.0]
        AA = np.asarray(acc, np.float64).copy()
        AA[0] = (v_log[-1] - v_log[-2]) / 0.5
        dt = lambda z, ref: z.astype(np.asarray(ref).dtype)  # noqa: E731
        oA = o_plan(core_, fr, valid, P, dt(VA, V), dt(AA, acc), *x, **k)
        Pd = np.asarray(P, np.float64).copy()
        for i in range(len(Pd) - 1):
            if sp[i:].mean() > 0.3:
                Pd[i, :2] *= float(np.clip(v_log[i:].mean() / sp[i:].mean(), 0.5, 3.0))
        oB = o_plan(core_, fr, valid, dt(Pd, P), dt(VA, V), dt(AA, acc), *x, **k)
        stash.update(t0=t0, F=np.asarray(o["poses"], np.float32).copy(), A=np.asarray(oA["poses"], np.float32).copy(),
                     B=np.asarray(oB["poses"], np.float32).copy(), v_log_keys=v_log, v_ego_keys=sp, v_trace_t0=R.speed(t0))
        return o

    PC.slots, PC.plan = slots, plan
    out, files = {}, sorted(Path(a.msgs).glob("*.pkl"))[: a.limit or None]
    si, sn = map(int, a.shard.split("/"))
    files = files[si::sn]
    t_start = time.time()
    for n, f in enumerate(files):
        scene = f.stem
        if scene not in rolls:
            print(scene, "no rf_tax roll, skipped", flush=True)
            continue
        cur["roll"], rec, uuid, fail = rolls[scene], [], None, 0
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
                stash.clear()
                try:
                    drv.drive(req, ctx)
                except RuntimeError as e:
                    fail += 1
                    continue
                if stash:
                    now = int(req.time_now_us)
                    rec.append(dict(now=now, stored=stored.get((scene, now)), **stash))
        drv.sessions.pop(uuid, None)
        out[scene] = rec
        g = [float(np.abs(r["F"][:, :2] - r["stored"][:, :2]).max()) for r in rec if r["stored"] is not None]
        print(f"{n + 1}/{len(files)} {scene}: {len(rec)} inferences, {fail} failed drives, gate max |F - stored| "
              f"{max(g) if g else float('nan'):.4f} m, {time.time() - t_start:.0f} s", flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "scenes,", sum(map(len, out.values())), "decisions ->", a.out, flush=True)


if __name__ == "__main__":
    main()
