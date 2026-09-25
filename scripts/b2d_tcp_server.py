#!/usr/bin/env python
"""Network forward of the official Bench2DriveZoo TCP model for the shared-control partner (scripts/b2d_partner.py).

The partner runs the shipped TCPAgent inside the CARLA route process (envs/b2d-tcp, Python 3.8, torch 2.2 without
sm_120 kernels, so its network cannot run on this box's GPUs). Only `self.net(rgb, state, target_point)` comes here;
everything else (preprocessing, route planner, PID, throttle cap) stays in the shipped code. The model is built and
loaded exactly as the shipped setup() does (TCP(GlobalConfig()), keys with "model." stripped, strict=False), in fp32
with TF32 off. The forward is stateless (seq_len 1), so connections share one model under a lock.

    CUDA_VISIBLE_DEVICES=1 $DATA_DIR/envs/jevdrive/bin/python scripts/b2d_tcp_server.py --ckpt <tcp_b2d.ckpt> \
        --socket <path> --ready-file <path> [--zoo <Bench2DriveZoo>]
"""
import argparse
import os
import socket
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeroshot_wire as wire  # noqa: E402

OUT_KEYS = ("pred_wp", "action_index", "pred_speed")


def load(ckpt, zoo):
    sys.path.insert(0, zoo)
    from TCP.config import GlobalConfig
    from TCP.model import TCP
    net = TCP(GlobalConfig())
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)["state_dict"]
    res = net.load_state_dict(OrderedDict((k.replace("model.", ""), v) for k, v in sd.items()), strict=False)
    print("checkpoint: %d tensors, missing %d, unexpected %d" % (len(sd), len(res.missing_keys), len(res.unexpected_keys)),
          flush=True)
    return net.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--zoo", default=str(Path(os.environ.get("DATA_DIR", "")) / "third_party/Bench2DriveZoo"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, lock = load(a.ckpt, a.zoo), threading.Lock()
    with torch.no_grad():   # first call builds the cuDNN plans
        net(torch.zeros(1, 3, 256, 900, device="cuda"), torch.zeros(1, 9, device="cuda"), torch.zeros(1, 2, device="cuda"))

    def serve(conn):
        n, t0 = 0, time.time()
        try:
            while True:
                meta, arr = wire.recv(conn)
                with lock, torch.no_grad():
                    out = net(*(torch.from_numpy(np.array(arr[k])).cuda() for k in ("img", "state", "target_point")))
                    res = {k: out[k].float().cpu().numpy() for k in OUT_KEYS}
                wire.send(conn, {"n": n}, res)
                n += 1
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()
            print("%s connection closed after %d forwards (%.0f s)" % (time.strftime("%F %T"), n, time.time() - t0),
                  flush=True)

    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(64)
    Path(a.ready_file).write_text("tcp %d\n" % os.getpid())
    print("tcp server ready", flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
