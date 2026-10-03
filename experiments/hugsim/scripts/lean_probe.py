"""Launch lean, small-rate gain and loop replay of Cinque on logged HUGSIM frames, offline (no simulator).
Plan: experiments/hugsim/plans/2026-10-04-launch-lean-prereg.md (parts 1b, 2, 3).

Frames come from a run's video.mp4 (3 front cameras), the request schedule is zs_agent.py's (warm-up 100 model steps on frame 0,
then 4 per simulator step, desire from the log, the scene's traffic flag). Models: O = shipped Cinque (TensorRT), any other name =
$DATA_DIR/runs/op_adapt_H/onnx/<name>.onnx (the serving ONNX of an op_adapt_h checkpoint, same as h_hugsim.sh).
Readouts per plan, all + = left: lean10 = direction of the 10 s plan point, phi1 = direction of the HUGSIM plan point at 1 s
(what the controller tracks; no forward_only / straight_stop), psi3 = plan heading at 3 s (model time), y3 = lateral at 3 s (m).

  lean    steps 0-2 of each run under input variants (mirror, traffic flag, history, desire, half masks, cameras, yaw)
  rate    step K (default 6) with the history re-rendered at the step's heading (decision 96's de-rotation) plus a fake yaw rate
          w (deg / model-s, left +) on every history frame: G(w) = (x(+w) - x(-w)) / 2 on HUGSIM frames
  replay  the logged frame sequence step by step, normal stepping and the de-rotated replay at every step (decision 96 rule)

    CUDA_VISIBLE_DEVICES=2 $DATA_DIR/envs/openpilot/bin/python experiments/hugsim/scripts/lean_probe.py lean <jobs.json> --out <out.json> \
        [--models O pilot-s0 it_dw3-s0] [--variants base,mirror_tc]
jobs.json: [{"scenario", "run_dir", "dataset"}]; run_dir holds video.mp4, infos.pkl, zs_steps.jsonl.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive import hugsim_zs as Z  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DIL, WARM, PER, CTX = 1.25, 100, 4, 25
CAMYAML = str(data_dir() / "third_party" / "HUGSIM" / "configs" / "sim" / "{}_camera.yaml")
RATES = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
CAM_H = 1.5                                      # camera height for the rolling-start warp (road plane / 60 m sphere)
LEAN_VARS = ("base", "mirror", "mirror_tc", "tc", "single", "warm1", "roll", "desire_l", "desire_r", "mask_L", "mask_R",
             "front", "yaw+1", "yaw-1")


def rss_guard(limit_gb=40.0):
    rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    if rss > limit_gb:
        sys.exit(f"RSS {rss:.1f} GB > {limit_gb} GB")


def load_models(names):
    from jevdrive.openpilot.model import OPModel
    on = data_dir() / "runs" / "op_adapt_H" / "onnx"
    return {n: OPModel("cinque" if n == "O" else str(on / f"{n}.onnx"), "trt") for n in names}


class LoggedRun:
    def __init__(self, job, n_frames):
        import cv2
        d = Path(job["run_dir"])
        self.name, self.d = job["scenario"], d
        infos = pickle.load(open(d / "infos.pkl", "rb"))
        recs = [json.loads(x) for x in open(d / "zs_steps.jsonl")]
        self.tc = list(recs[0].get("opts", {}).get("traffic", [1, 0]))
        self.recs = recs[1:]
        self.cal = Z.calibs(infos[0]["cam_params"], Z.rect_matrix(CAMYAML.format(job["dataset"])))
        self.op = Z.OpenpilotFrames(self.cal)
        cap, self.rgb = cv2.VideoCapture(str(d / "video.mp4")), []
        while len(self.rgb) < min(n_frames, len(self.recs)):
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w = f.shape[0] // 2, f.shape[1] // 3
            self.rgb.append({"CAM_FRONT_LEFT": f[:h, :w], "CAM_FRONT": f[:h, w:2 * w], "CAM_FRONT_RIGHT": f[:h, 2 * w:]})
        self.n = len(self.rgb)
        self.th = np.unwrap([r["theta"] for r in self.recs[:self.n]])          # + = right
        self.v = np.array([r["v"] for r in self.recs[:self.n]])
        self.des = [int(r.get("desire", 0)) for r in self.recs[:self.n]]
        self._front = None

    def frame(self, k, yaw=0.0, front=False):
        if front:
            if self._front is None:
                self._front = Z.OpenpilotFrames(self.cal, cams=("CAM_FRONT",))
            op = self._front
        else:
            op = self.op
        return op.pack(self.rgb[k], op.rot_index(yaw) if round(yaw, 1) != 0.0 else None)


# ---------------------------------------------------------------- image transforms on packed frames (2, 6, 128, 256)
def mirror(img2):
    out = np.empty_like(img2)
    for v in range(2):
        Y, U, V = I.unpack(img2[v])
        out[v] = I.pack(np.ascontiguousarray(Y[:, ::-1]), np.ascontiguousarray(U[:, ::-1]), np.ascontiguousarray(V[:, ::-1]))
    return out


def mask(img2, side):
    out = np.empty_like(img2)
    for v in range(2):
        Y, U, V = (x.copy() for x in I.unpack(img2[v]))
        sl = slice(0, Y.shape[1] // 2) if side == "L" else slice(Y.shape[1] // 2, None)
        sc = slice(0, U.shape[1] // 2) if side == "L" else slice(U.shape[1] // 2, None)
        Y[:, sl], U[:, sc], V[:, sc] = 126, 128, 128
        out[v] = I.pack(Y, U, V)
    return out


def roll_frame(img2, d):
    """img2 re-rendered from d metres behind (rolling-start history)."""
    return I.warp_frame(img2, np.array([0.0, 0.0, CAM_H]), np.array([-d, 0.0, 0.0]), np.zeros(3))


# ---------------------------------------------------------------- model
def metrics(raw, m):
    from jevdrive.openpilot.model import T_IDXS, decode
    d = decode(raw, m.slices, 0.0)
    pos, yaw = d["plan_pos"], d["plan_yaw"]
    p = Z.openpilot_to_plan(pos, T_IDXS, DIL)
    return dict(lean10=float(np.degrees(np.arctan2(-pos[32, 1], max(pos[32, 0], 1e-3)))),
                phi1=float(-np.degrees(np.arctan2(p[1, 0], max(p[1, 1], 1e-3)))),
                psi3=float(-np.degrees(np.interp(3.0, T_IDXS, yaw))), y3=float(-np.interp(3.0, T_IDXS, pos[:, 1])),
                x10=float(pos[32, 0]), pos=np.round(pos[[4, 8, 12, 16, 20, 24, 32], :2], 3).tolist())


def feed(m, seq, tc):
    """seq: [(img2, reps, desire, read)] -> metrics at every read point."""
    m.reset()
    out = []
    for img, reps, des, read in seq:
        dv = np.zeros(8, np.float32)
        dv[int(des)] = 1
        for _ in range(reps):
            raw = m.step(img, desire=dv, traffic=tuple(tc))
        if read:
            out.append(metrics(raw, m))
    return out


# ---------------------------------------------------------------- lean
def lean_seq(R, var, steps=3):
    tc = R.tc[::-1] if var in ("tc", "mirror_tc") else R.tc
    tf = mirror if var.startswith("mirror") else (lambda x: mask(x, var[-1])) if var.startswith("mask") else (lambda x: x)
    yaw = {"yaw+1": 1.0, "yaw-1": -1.0}.get(var, 0.0)
    fr = [tf(R.frame(k, yaw, front=var == "front")) for k in range(steps)]
    des = [R.des[k] for k in range(steps)]
    if var in ("desire_l", "desire_r"):
        des = [des[0]] + [1 if var == "desire_l" else 2] * (steps - 1)
    warm = {"single": PER, "warm1": 5 * PER}.get(var, WARM)
    seq = []
    if var == "roll":                                     # 5 s of history as if rolling at 1 m/s (x DIL model clock)
        seq = [(roll_frame(fr[0], 1.0 * DIL * 0.2 * j), PER, des[0], False) for j in range(CTX, 0, -1)]
        warm = PER
    seq += [(fr[0], warm, des[0], True)] + [(fr[k], PER, des[k], True) for k in range(1, steps)]
    return seq, tc


def cmd_lean(a, jobs, models):
    res = {}
    vars_ = a.variants.split(",") if a.variants else LEAN_VARS
    for job in jobs:
        R = LoggedRun(job, 3)
        r = res[R.name] = {"tc": R.tc, "theta": R.th.tolist(), "v": R.v.tolist(), "logged": [x["model_pos"] for x in R.recs[:3]]}
        for var in vars_:
            seq, tc = lean_seq(R, var)
            for k, m in models.items():
                r[f"{k}|{var}"] = feed(m, seq, tc)
        rss_guard()
        print(f"lean {R.name}: done", flush=True)
    return res


# ---------------------------------------------------------------- rate
def cmd_rate(a, jobs, models):
    res = {}
    K = a.k
    for job in jobs:
        R = LoggedRun(job, K + 1)
        r = res[R.name] = {"tc": R.tc, "theta": R.th.tolist(), "v": R.v.tolist(), "k": K}
        for w in (0.0,) + tuple(s * x for x in RATES for s in (1.0, -1.0)):
            seq = []
            for j in range(K + 1):
                t = (j - K) * 0.2
                yaw = float(np.degrees(R.th[j] - R.th[K])) + w * t
                seq.append((R.frame(j, yaw), WARM if j == 0 else PER, R.des[j], j == K))
            for k, m in models.items():
                r[f"{k}|{w:+g}"] = feed(m, seq, R.tc)[0]
        rss_guard()
        print(f"rate {R.name}: done", flush=True)
    return res


# ---------------------------------------------------------------- replay
def cmd_replay(a, jobs, models):
    res = {}
    for job in jobs:
        R = LoggedRun(job, a.steps)
        r = res[job.get("key", R.name)] = {"tc": R.tc, "theta": R.th.tolist(), "v": R.v.tolist(),
                                           "logged": [x["model_pos"] for x in R.recs[:R.n]]}
        fr = [R.frame(k) for k in range(R.n)]
        for k, m in models.items():
            r[f"{k}|normal"] = feed(m, [(fr[j], WARM if j == 0 else PER, R.des[j], True) for j in range(R.n)], R.tc)
            der = [None]
            for s in range(1, R.n):
                j0 = max(0, s - CTX)
                seq = [(R.frame(j, float(np.degrees(R.th[j] - R.th[s]))), WARM if j == j0 else PER, R.des[j], j == s)
                       for j in range(j0, s + 1)]
                der.append(feed(m, seq, R.tc)[0])
            der[0] = r[f"{k}|normal"][0]
            r[f"{k}|derot"] = der
            rss_guard()
        print(f"replay {job.get('key', R.name)}: {R.n} steps done", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("lean", "rate", "replay"))
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["O", "pilot-s0", "it_dw3-s0"])
    ap.add_argument("--variants", default="")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    t0 = time.time()
    models = load_models(a.models)
    res = {"lean": cmd_lean, "rate": cmd_rate, "replay": cmd_replay}[a.mode](a, jobs, models)
    res["_meta"] = dict(mode=a.mode, models=a.models, wall_s=time.time() - t0, k=a.k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    sys.stdout.flush()
    os._exit(0)                                   # TensorRT teardown can hang


if __name__ == "__main__":
    main()
