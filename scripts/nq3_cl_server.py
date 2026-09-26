#!/usr/bin/env python
"""Closed-loop head server for night queue 3, lane B (todos/2026-09-26-night-queue-3.md, CL3 / CL4 / CL6 / CL5).
envs/openpilot. Agent side: scripts/b2d_zeroshot_agent.py with "model": "head".

Per connection (one CARLA worker, one route) the server keeps its own Cinque session and the last four camera sets.
A plan request carries the P4 Waymo rig's three JPEGs of one 5 Hz camera frame (the recorder's bytes: remap + JPEG q95,
scripts/p5_pair_agent.py), the route desire and the ego input (jevdrive.nq3_cl.ego_input, built by the agent from its
own pose track). The server then does exactly what the P5 offline path does:

  1. model frames from the JPEGs (scripts/p5_openpilot.render_blobs, the P5 calibration record), Cinque stepped four
     times on them (the 5 Hz frame held on the 20 Hz clock, p5_openpilot.run_stream) with the desire, `temporal` tap;
  2. arm "mc": Qwen `L18_last` of the 4-frame x 3-camera clip from the Qwen server; arm "student_b": the 672-d token
     row from the YOLO server; both requested before step 1 and collected after it, so they overlap the Cinque steps;
  3. the head (jevdrive.nq3_cl.Heads): prior + Delta -> (20, 2) rear-axle path at 0.25 ... 5 s.
  Arms "q2" (CL5) and "q2d" (CL5d) use lane C's head (jevdrive.nq3_head.Head, runs/nq3/q2/closed_loop_head) on the same
  `temporal` and ego input: "q2" drives on its trajectory; "q2d" drives on openpilot's own plan from this same stream
  (camera origin = the rig's FRONT camera, rear = d + p - R(psi) d as wod_zeroshot.openpilot_to_wod), and while the
  mode head's last output is bypass-L / -R the desire is laneChangeLeft / -Right instead of the route desire (OPModel turns
  a held desire into a rising-edge pulse, as modeld does).

Night queue 4: meta "heads" (a heads.npz) / "q2_dir" (a Head dir) select one cross-fitted fold per request; absent,
the full-data heads above. With meta "dump" = <path>.npz, the inputs and every intermediate are written there for the equivalence check
(jevdrive.nq3_cl check, rule 8). Protocol: scripts/zeroshot_wire.py.

    CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/openpilot/bin/python scripts/nq3_cl_server.py --socket S --qwen Q --yolo Y
"""
import argparse
import os
import socket
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import p5_openpilot as P5  # noqa: E402
import wod_zeroshot_openpilot as WZ  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from jevdrive import drive_backbones as D  # noqa: E402
from jevdrive import nq3_cl as CL  # noqa: E402
from jevdrive import nq4_k as NK  # noqa: E402
from jevdrive.p5_openpilot import RIG, carla_calib  # noqa: E402

HOLD = P5.HOLD
MODEL = "cinque"


class FeatClient:
    """One connection to a feature server (per head-server connection, so requests never interleave)."""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)

    def __call__(self, blobs):
        wire.send(self.sock, {}, {"jpg%d" % i: b for i, b in enumerate(blobs)})
        return wire.recv(self.sock)

    def close(self):
        self.sock.close()


def openpilot_to_rear(plan_pos, plan_yaw, t_idx, dev_xy):
    """jevdrive.wod_zeroshot.openpilot_to_wod (numpy only here): the plan's camera track -> rear-axle track at 0.25 ... 5 s."""
    p = np.stack([plan_pos[:, 0], -plan_pos[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(plan_yaw, np.float64)
    d = np.asarray(dev_xy, np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    rear = d + p - Rd
    tq = np.arange(1, 21) * 0.25
    return np.stack([np.interp(tq, t_idx, rear[:, k]) for k in range(2)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--qwen", default="", help="Qwen feature server socket (arm mc)")
    ap.add_argument("--yolo", default="", help="YOLO feature server socket (arm student_b)")
    ap.add_argument("--pool", type=int, default=6, help="Cinque sessions built up front, one per CARLA worker")
    ap.add_argument("--ready-file", default="")
    a = ap.parse_args()
    from jevdrive.openpilot.model import OPModel
    t0 = time.time()
    WZ._init({}, {P5.SEQ: carla_calib()}, ".")
    heads = CL.Heads()
    q2, fold_heads = {}, {}

    def q2head(d=None):
        """Lane C's head, or (night queue 4, meta "q2_dir") one cross-fitted fold of it."""
        d = d or str(Path(os.environ["DATA_DIR"]) / "runs" / "nq3" / "q2" / "closed_loop_head")
        if d not in q2:
            from jevdrive.nq3_head import Head
            q2[d] = Head(d)
        return q2[d]

    khead = {}

    def kheads():
        if "h" not in khead:
            khead["h"] = NK.KHead()
        return khead["h"]

    def heads_for(p=None):
        """The full-data heads, or (night queue 4, meta "heads") one cross-fitted fold's heads.npz."""
        if not p:
            return heads
        if p not in fold_heads:
            fold_heads[p] = CL.Heads(Path(p))
        return fold_heads[p]
    from jevdrive.openpilot.model import T_IDXS, decode
    front_xy = np.array(RIG[0][1:3], float)                  # the rig's FRONT camera (x, y) on the rear-axle frame
    taps = D.OP_TAPS[MODEL]
    make = lambda: OPModel(MODEL, WZ.MODELS[MODEL], context_rate=False, taps=list(taps.values()))  # noqa: E731
    free = [make() for _ in range(a.pool)]
    lock = threading.Lock()
    print("head server ready after %.0f s (%d Cinque sessions)" % (time.time() - t0, len(free)), flush=True)
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    Path(a.socket).parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(64)
    if a.ready_file:
        Path(a.ready_file).write_text("ready")
    stats = {"calls": 0, "t0": time.time(), "ms": {}}

    def serve(conn):
        with lock:
            m = free.pop() if free else make()
        m.reset()
        sets = deque(maxlen=4)
        last_mode = [None]
        fq = FeatClient(a.qwen) if a.qwen else None
        fy = FeatClient(a.yolo) if a.yolo else None
        ex = ThreadPoolExecutor(2)
        try:
            while True:
                meta, arrays = wire.recv(conn)
                if meta["cmd"] == "reset":
                    m.reset()
                    sets.clear()
                    last_mode[0] = None
                    wire.send(conn, {"ok": True, "server": {"model": MODEL, "taps": taps}}, {})
                    continue
                t = time.perf_counter()
                arm = meta["arm"]
                jpg = [np.array(arrays["jpg%d" % i]) for i in range(3)]
                sets.append(jpg)
                full = len(sets) == 4
                fut_q = ex.submit(fq, [sets[j][c] for c in range(3) for j in range(4)]) \
                    if arm in ("mc", "mc_real0") and full else None
                fut_y = ex.submit(fy, jpg) if arm == "student_b" else None
                img2 = P5.render_blobs([jpg])[0]
                t_r = time.perf_counter()
                d_idx = int(meta.get("desire", 0))
                if arm == "q2d" and last_mode[0] in (2, 3):          # bypass_L / bypass_R -> laneChangeLeft / Right
                    d_idx = 3 if last_mode[0] == 2 else 4
                desire = np.zeros(8, np.float32)
                desire[d_idx] = 1
                for _ in range(HOLD):
                    raw = m.step(img2, desire=desire, action_t=WZ.ACTION_T)
                op = m.tap_values[taps["temporal"]].copy()
                t_o = time.perf_counter()
                q = fut_q.result()[1]["q"] if fut_q else None
                tok = fut_y.result()[1]["tok"] if fut_y else None
                t_f = time.perf_counter()
                ego = np.asarray(arrays["ego"], np.float32)
                mode, kinfo = None, None
                if arm in ("q2", "q2d"):
                    traj, md = q2head(meta.get("q2_dir"))(op, ego)
                    mode = int(md[0]) if md is not None else None
                    last_mode[0] = mode
                    if arm == "q2":
                        path = np.asarray(traj[0], np.float64)
                    else:
                        dd = decode(raw, m.slices, float(meta.get("speed", 0.0)))
                        path = np.asarray(openpilot_to_rear(dd["plan_pos"], dd["plan_yaw"], T_IDXS, front_xy), np.float64)
                elif arm in NK.ARMS:                                      # night queue 4, K-prep: the fold is the agent's
                    path, kinfo = kheads().predict(arm, meta["kfold"], ego, op, raw[m.slices["lead"]], raw[m.slices["lead_prob"]])
                elif arm in ("mc", "mc_real0") and q is None:          # the first three camera sets of a route: no 4-frame clip yet
                    path = heads_for(meta.get("heads")).predict("ridge_late", ego, op)
                else:
                    path = heads_for(meta.get("heads")).predict(arm, ego, op, q=q, tok=tok)
                t_h = time.perf_counter()
                ms = {"render_ms": 1e3 * (t_r - t), "op_ms": 1e3 * (t_o - t_r), "feat_wait_ms": 1e3 * (t_f - t_o),
                      "head_ms": 1e3 * (t_h - t_f), "server_ms": 1e3 * (t_h - t)}
                info = {k: round(v, 2) for k, v in ms.items()}
                info["full_clip"] = full
                if mode is not None:
                    info.update(mode=mode, desire_used=d_idx)
                if kinfo:
                    info.update({k: float(v) for k, v in kinfo.items()})
                wire.send(conn, info, {"path": path.astype(np.float64)})
                if meta.get("dump"):
                    extra = {"q": q} if q is not None else {}
                    if tok is not None:
                        extra["tok"] = tok
                    if arm in NK.ARMS:
                        extra.update(kfold=str(meta["kfold"]), lead=raw[m.slices["lead"]], lead_prob=raw[m.slices["lead_prob"]])
                    np.savez(meta["dump"], jpg0=jpg[0], jpg1=jpg[1], jpg2=jpg[2], desire=np.int64(meta.get("desire", 0)), ego=ego, img2=img2, op=op, path=path, arm=arm,
                             frame=np.int64(meta.get("frame", -1)), **extra)
                stats["calls"] += 1
                for k, v in ms.items():
                    stats["ms"][k] = stats["ms"].get(k, 0.0) + v
                if stats["calls"] % 1000 == 0:
                    print("calls %d, mean ms %s" % (stats["calls"], {k: round(v / stats["calls"], 1)
                                                                      for k, v in stats["ms"].items()}), flush=True)
        except ConnectionError:
            pass
        finally:
            conn.close()
            ex.shutdown(wait=False)
            for c in (fq, fy):
                if c is not None:
                    c.close()
            with lock:
                free.append(m)

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
