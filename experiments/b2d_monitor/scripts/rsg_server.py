#!/usr/bin/env python
"""B2D policy server for an SH30-family checkpoint with the ROUTE-MEM branch (RSG-L18-s0, decision 286): the PAI serving path
(experiments/alpasim/lib/pai_core.plan on sh30_core.Core, rmem.load_pmodel via Core's route_mem.json sidecar) behind the zeroshot_wire
socket protocol the B2D agent already speaks (experiments/b2d_monitor/lib/op_arb_viz_agent.py, config key "rsg"). envs/op-train, one card.

Per request ({"cmd": "plan"}, sent by the agent every 0.2 s of sim time):
  frames   the CARLA openpilot road (focal 2648) and wide (focal 567) renders (BGRA, 1928 x 1208, level, straight, at the rig camera) ->
           the openpilot road (focal 910) and wide (focal 455) 512 x 256 model frames, nearest model pixel (the same index maps as
           op_arb_server / zeroshot_policy_server), converted as navsim_zs.OpenpilotMaps packs NAVSIM CAM_F0 for this family: BT.601
           full-range YCbCr (JPEG's), chroma = 2 x 2 mean of the model-resolution samples rounded half to even, [road, wide]. Not
           reproduced: the JPEG round trip of the training images.
  slots    the 8 policy slots t0 - 1.4 .. t0 at 0.2 s are the frames the agent sent at those sim times (all real after 1.4 s; missing
           ones are invalid = the policy's stream-start state, pai_core cold rule "zero")
  ego      P (4, 3), V (4, 2) at T_KEY = -1.5 / -1.0 / -0.5 / 0 s in the t0 rear-axle frame (x forward, y left, yaw left positive), acc (2,),
           cmd (4,) one-hot [L, S, R, unknown]: computed by the agent
  route    rmem features (200,) computed by the agent (rmem.memory of the leaderboard route in the current rear-axle frame); `rmem_off`
           in the request (or env SH30_RMEM=0) feeds the empty memory instead (the G-rely style control)
Reply arrays: poses (8, 3) rear axle at 0.5 .. 4 s, mu (33, 15) camera-frame plan, t (33,) T_IDXS, lead (144,), lead_prob (3,) logits,
ego (20,), rmem (200,) as fed; info: stage ms, valid slot count.

    CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/op-train/bin/python experiments/b2d_monitor/scripts/rsg_server.py --tag RSG-L18-s0 --socket S
    (launcher: scripts/rsg_server.sh)
"""
import argparse
import os
import pathlib
import socket
import sys
import threading
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / d) for d in ("", "scripts", "lib", "experiments/alpasim/lib", "experiments/route_mem/lib", "experiments/b2d_monitor/lib")]
import zeroshot_rigs as rigs  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from jevdrive.openpilot import frames as opf  # noqa: E402

SLOT_T = np.round(np.arange(-7, 1) * 0.2, 3)
TOL = 0.03
CHECK_EVERY = int(os.environ.get("RSG_CHECK_EVERY", "50"))   # every Nth full-slot decision: cached vs full re-encode (pai_core.plan)


class Encoder:
    """Per-slot hidden states, cached across decisions. A slot's tokens depend only on its image pair (the previous slot's frame or a
    zero image, this slot's frame), so with slots and decisions both at 0.2 s each decision encodes two new pairs: the newest, and the
    oldest slot (whose pair starts from a zero image, pai_core.plan's rule). Keyed by (t_prev or None, t_cur)."""

    def __init__(self, core):
        self.core, self.cache = core, {}

    def __call__(self, frames, keys):
        torch, A, core = self.core.torch, self.core.A, self.core
        new = [k for k in keys if k not in self.cache]
        if new:
            z = np.zeros(frames[keys[-1][1]].shape, np.uint8)
            p = torch.from_numpy(np.stack([frames[a] if a is not None else z for a, _ in new])).to(core.dev)
            c = torch.from_numpy(np.stack([frames[b] for _, b in new])).to(core.dev)
            with torch.no_grad():
                H = core._enc(p, c) if core._enc else core.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"]
            H = H.reshape(len(new), *A.H_SHAPE)
            for k, h in zip(new, H):
                self.cache[k] = h
        oldest = min(k[1] for k in keys)
        for k in [k for k in self.cache if k[1] < oldest - 1e-6]:
            del self.cache[k]
        return torch.stack([self.cache[k] for k in keys])[None], len(new)


def decide(core, enc, frames, t, P, V, acc, cmd, cam_t, rf):
    """pai_core.plan with the per-slot cache: same slots, same pairs, same policy / export."""
    import sh30_core as C
    from sh30_core import I, PA, Z
    torch = core.torch
    t0 = time.perf_counter()
    have = np.array(sorted(frames))
    slot_t = []
    for dt in SLOT_T:
        i = int(np.argmin(np.abs(have - (t + dt))))
        slot_t.append(float(have[i]) if abs(have[i] - (t + dt)) < TOL else None)
    keys = [((slot_t[j - 1] if j and slot_t[j - 1] is not None else None), slot_t[j]) for j in range(8) if slot_t[j] is not None]
    H, n_new = enc(frames, keys)
    t1 = core._sync()
    ego = PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(cmd, np.float32))
    with torch.no_grad():
        tc = torch.tensor([[1.0, 0.0]], device=core.dev)
        ego_t = torch.from_numpy(np.r_[ego, rf][None].astype(np.float32)).to(core.dev)
        out = (core._pol or core.model)(H, ego_t, tc).float()
        t2 = core._sync()
        mu = out[0, core.pi].reshape(33, 15).cpu().numpy()
        sl = core.model.net.slices
        ld = {k: out[0, sl[k]].cpu().numpy() for k in ("lead", "lead_prob")}
    poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
    t3 = time.perf_counter()
    return {"poses": poses, "mu": mu, "ego": ego, **ld, "valid": len(keys), "new": n_new, "slot_t": slot_t,
            "ms": {"encode": 1e3 * (t1 - t0), "policy": 1e3 * (t2 - t1), "export": 1e3 * (t3 - t2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="RSG-L18-s0")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ready-file", default="")
    ap.add_argument("--cam", default="1.59,0.0,1.86", help="camera position in the rear-axle frame (the agent's op_mount)")
    a = ap.parse_args()
    try:
        import cv2  # noqa: F401
    except ImportError:                                  # envs/op-train has no OpenCV: sh30_core only calls cv2.setNumThreads at init; the cv2
        import types                                     # warps (op_interp synthesis) are not on this path (real slots, cold "zero")
        sys.modules["cv2"] = types.SimpleNamespace(setNumThreads=lambda n: None)
    import sh30_core as C
    import pai_core as PC
    core = C.Core(a.tag, "cuda", cold="zero")
    core.lead_out = True
    compiled = os.environ.get("SH30_COMPILE", "1") == "1"
    if compiled:
        core.compile()
    assert core.rm, "%s has no ROUTE-MEM sidecar" % a.tag
    cam_t = np.array([float(x) for x in a.cam.split(",")])
    import rsg_pack
    pack, lock = rsg_pack.Packer(), threading.Lock()
    w, h = rigs.OP_CAMERA_WH                             # the opf index maps (op_arb_server's): rsg_pack must match them pixel for pixel
    for k, K in (("road", opf.MEDMODEL_K), ("wide", opf.SBIGMODEL_K)):
        M = opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, rigs.OP_FOCAL[k]), k == "wide", model_K=K)
        ref = opf._nn_index(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
        print("index map %s: %d of %d pixels differ from jevdrive.openpilot.frames" % (k, int((ref != pack.idx[k]).sum()), ref.size), flush=True)
    rm_env = os.environ.get("SH30_RMEM", "1") != "0"
    T_IDXS = np.asarray(C.I.T_IDXS, np.float32)
    # warm the model passes
    rng = np.random.default_rng(0)
    tw = time.perf_counter()
    for n in range(12):                                  # warm-up: every batch size the cache produces (1, 2 new pairs; 1..8 slots) compiled
        fr = {round(0.2 * i, 2): rng.integers(0, 255, C.FRAME, dtype=np.uint8) for i in range(n + 1)}
        e = Encoder(core)
        for i in range(n + 1):
            decide(core, e, {k: v for k, v in fr.items() if k <= 0.2 * i + 1e-6}, 0.2 * i, np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2),
                   np.array([0, 1, 0, 0], np.float32), cam_t, np.zeros(core.rm_dim, np.float32))
    print("rsg server ready: tag %s, rm_dim %d, SH30_RMEM %d, compiled %s, net dtype %s, warm-up %.0f s" % (
        a.tag, core.rm_dim, rm_env, compiled, core.model.net.dtype, time.perf_counter() - tw), flush=True)

    def serve(conn):
        frames, enc, n_full = {}, Encoder(core), 0
        try:
            while True:
                meta, arrs = wire.recv(conn)
                if meta.get("cmd") == "reset":
                    frames.clear()
                    enc.cache.clear()
                    wire.send(conn, {"server": {"model": a.tag, "rsg": True, "route_mem": True}})
                    continue
                t = round(float(meta["t"]), 2)
                tp = time.perf_counter()
                frames[t] = np.asarray(arrs["FRAME"], np.uint8) if "FRAME" in arrs else pack(arrs["OP_ROAD"], arrs["OP_WIDE"])
                for k in [k for k in frames if k < t - 2.0]:
                    del frames[k]
                ms_pack = 1e3 * (time.perf_counter() - tp)
                rf = np.asarray(meta["rmem"], np.float32) if rm_env and not meta.get("rmem_off") else np.zeros(core.rm_dim, np.float32)
                P, V, acc = (np.asarray(meta[k], np.float64) for k in ("P", "V", "acc"))
                cmd = np.asarray(meta["cmd4"], np.float32)
                with lock:
                    o = decide(core, enc, frames, t, P, V, acc, cmd, cam_t, rf)
                    chk = None
                    if o["valid"] == 8:
                        n_full += 1
                        if CHECK_EVERY and n_full % CHECK_EVERY == 1:     # exactness: the same decision through pai_core.plan, full re-encode
                            cur = np.stack([frames[x] for x in o["slot_t"]])
                            f = PC.plan(core, cur, np.ones(8, bool), P, V, acc, cmd, cam_t, route=rf)
                            chk = {"mu_max_abs": float(np.max(np.abs(f["mu"] - o["mu"]))), "poses_max_abs": float(np.max(np.abs(f["poses"] - o["poses"])))}
                            print("exactness check (decision %d): %s" % (n_full, chk), flush=True)
                info = {"ms": {"pack": round(ms_pack, 1), **{k: round(v, 1) for k, v in o["ms"].items()}}, "valid": o["valid"], "new": o["new"],
                        "rmem_on": bool(rf.any()), **({"check": chk} if chk else {})}
                wire.send(conn, info, {"poses": np.asarray(o["poses"], np.float32), "mu": np.asarray(o["mu"], np.float32), "t": T_IDXS,
                                       "lead": np.asarray(o["lead"], np.float32), "lead_prob": np.asarray(o["lead_prob"], np.float32),
                                       "ego": np.asarray(o["ego"], np.float32), "rmem": rf})
        except (ConnectionError, EOFError, OSError):
            pass
        finally:
            conn.close()

    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(8)
    if a.ready_file:
        pathlib.Path(a.ready_file).write_text("ready\n")
    while True:
        c, _ = srv.accept()
        threading.Thread(target=serve, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
