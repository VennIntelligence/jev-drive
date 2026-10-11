#!/usr/bin/env python
"""Resident server of the stateless training protocol for HUGSIM (agent opt `stateless`, lib/parity_hugsim.Stateless;
experiments/hugsim/plans/2026-10-11-stateless-cl-prereg.md, results/stateless_cl.md).

One sh30_core.Core (the checkpoint's torch port, the path of training, of jevdrive.bench navtest and of the AlpaSim nuPlan driver) serves
every scenario connection of a bench worker job; Core.plan is called unchanged, one decision per request, under one lock (its slot warps
run on a thread pool of their own). The reply has the fields of experiments/hugsim/archive/hugsim_zs_server.Openpilot.plan, decoded by the
same jevdrive.openpilot.model.decode from the policy's full output vector, so the agent's plan-to-control path is the reference's. Core.plan
returns the plan only; the action head comes from one more policy pass on the same tokens and ego features (as frame_ab_torch.py), and its
plan must equal Core.plan's.

Protocol (scripts/zeroshot_wire.py): {"cmd": "reset"} -> {"ok", "server"}; {"cmd": "plan", "cam_t": [x, y, z], "yaw_rate", "lht", "speed"}
with arrays keys (m, 2, 6, 128, 256) uint8, pose (m, 3), vel (m, 2), acc (2,), cmd (4,) -> info + {pos, vel, yaw, yaw_rate, t}.

  $DATA_DIR/envs/op-train/bin/python experiments/hugsim/scripts/stateless_server.py --tag P2H10-F-s0 --socket S [--ready-file F]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/alpasim/lib")]
import argparse, json, socket, threading, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import zeroshot_wire as wire  # noqa: E402


def lead_xv(lead) -> dict:
    """hugsim_zs_server.lead_xv: selection 0 at t = 0 of the lead head."""
    m = np.asarray(lead, np.float64).reshape(3, 6, 4)
    return {"lead_x": float(m[0, 0, 0]), "lead_v": float(m[0, 0, 2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ready-file", default="")
    ap.add_argument("--synth", default="cpu", choices=["cpu", "gpu"], help="sh30_core lattice synthesis; cpu = the reference the checkpoints were trained on")
    a = ap.parse_args()
    import torch
    import sh30_core as C
    from jevdrive.openpilot import model as M
    core = C.Core(a.tag, synth=a.synth)
    net, sl = core.model, core.model.net.slices
    t_idxs = M.T_IDXS.astype(np.float32)
    info = {"server": "stateless", "tag": a.tag, "arm": net.arm, "cold": core.cold, "synth": core.synth, "slots": len(C.SLOT_T), "clock": "real"}
    gpu = threading.Lock()

    def plan(meta, arrays):
        t0 = time.perf_counter()
        o = core.plan(list(arrays["keys"]), arrays["pose"], arrays["vel"], arrays["acc"], arrays["cmd"], cam_t=np.asarray(meta["cam_t"], np.float64),
                      yaw_rate=float(meta.get("yaw_rate", 0.0)), lht=bool(meta.get("lht", False)))
        t1 = time.perf_counter()
        with torch.no_grad():                                  # the full output of the same pass: action head, lead, meta
            tc = torch.tensor([[0.0, 1.0] if meta.get("lht") else [1.0, 0.0]], device=core.dev)
            raw = net(o["tokens"][None], torch.from_numpy(o["ego"][None]).to(core.dev), tc).float()[0].cpu().numpy()
        d = M.decode(raw, sl, float(meta.get("speed", 0.0)))
        dev = float(np.abs(d["plan_pos"] - o["mu"][:, 0:3]).max())
        if not dev < 1e-3:
            raise RuntimeError(f"second policy pass differs from Core.plan by {dev} m")
        pl = raw[sl["plan"]]
        lat_std = np.exp(np.minimum(pl[pl.size // 2:], 11)).reshape(33, 15)[:, 1]
        ms = dict(o["ms"], action=1e3 * (time.perf_counter() - t1))
        rep = {"infer_ms": 1e3 * (time.perf_counter() - t0), "ms": {k: round(float(v), 2) for k, v in ms.items()}, "curvature": d["curvature"],
               "curvature_plan": d["curvature_plan"], "curvature_smooth": d["curvature_smooth"], "accel": d["accel"],
               "lat_std4": float(lat_std[t_idxs <= 4.0 + 1e-6].sum()), "engaged": d["engaged"], "lead_prob": float(np.ravel(d["lead_prob"])[0]),
               "valid_slots": int(o["valid"].sum()), **lead_xv(d["lead"])}
        out = {"pos": d["plan_pos"].astype(np.float32), "vel": d["plan_vel"][:, 0].astype(np.float32), "yaw": d["plan_yaw"].astype(np.float32),
               "yaw_rate": d["plan_yaw_rate"].astype(np.float32), "t": t_idxs, "act": np.asarray(raw[sl["action"]][:2], np.float32)}
        return rep, out

    with gpu:                                                  # warm-up with the real shapes (cold start and warm)
        z = dict(pose=np.zeros((1, 3)), vel=np.array([[1.0, 0.0]]), acc=np.zeros(2), cmd=np.array([0, 1, 0, 0], np.float32))
        for m in (1, 4):
            plan({"cam_t": [1.5, 0.0, 1.5], "speed": 1.0}, dict(z, keys=np.zeros((m, 2, 6, 128, 256), np.uint8),
                                                               pose=np.c_[np.linspace(-0.5 * (m - 1), 0, m), np.zeros(m), np.zeros(m)],
                                                               vel=np.tile([1.0, 0.0], (m, 1))))
    stats = {"calls": 0, "busy_s": 0.0, "t0": time.time()}

    def conn_loop(conn):
        try:
            while True:
                meta, arrays = wire.recv(conn)
                if meta["cmd"] == "reset":
                    wire.send(conn, {"ok": True, "server": info}, {})
                    continue
                t = time.perf_counter()
                with gpu:
                    tl = time.perf_counter()
                    rep, out = plan(meta, arrays)
                    stats["busy_s"] += time.perf_counter() - tl
                stats["calls"] += 1
                rep.update(queue_ms=1e3 * (tl - t), server_ms=1e3 * (time.perf_counter() - t))
                wire.send(conn, rep, out)
                if stats["calls"] % 200 == 0:
                    up = time.time() - stats["t0"]
                    print("calls %d, lock busy %.0f%% of %.0f s" % (stats["calls"], 100 * stats["busy_s"] / up, up), flush=True)
        except ConnectionError:
            pass
        finally:
            conn.close()

    sp = Path(a.socket)
    sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.exists():
        sp.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sp))
    srv.listen(64)
    print("stateless server ready:", json.dumps(info), flush=True)
    if a.ready_file:
        Path(a.ready_file).write_text(json.dumps(info))
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=conn_loop, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
