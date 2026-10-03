"""Lane EDGE intervention (plan: experiments/skill_pack/plans/2026-10-04-roadedge-diagnosis-plan.md, addendum 2): show
openpilot a virtual camera at a lower height and check that its metric outputs scale back.

envs/openpilot, one GPU for a few minutes, CPU pool for rendering. For a seed-0 subset of navtest tokens and each ratio r:
  keys: the 4 CAM_F0 keyframes rendered into the road / wide model frames (as jevdrive.navsim_zs.OpenpilotMaps) with the
        ground-plane homography of a camera lowered from h_true = CAM_F0 z + 0.35 to h_true / r: a model ray below the
        horizon (device frame x fwd, z down) samples CAM_F0 along (x, y, r z); rays above the horizon are unchanged.
  synth: the 6 context-rate frames by the CPU ego-motion warp (jevdrive.op_interp.synth_cpu "warp", road plane at the
        virtual height), the same for every r so arms differ only by r (the shipped run used GIMM frames).
  rollout: Cinque, desire none, zero state, the op_lb step schedule; plan + all heads -> <out>/hgt_r<r>.npz in the
        op_lb plan-file format (names, plan_pos, plan_yaw, heads, info.heads_slices).
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
ZG = 0.35
_G = {}


def render(args):
    e, r = args
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.openpilot.frames import MEDMODEL_K, SBIGMODEL_K, VIEW_FROM_DEVICE, MODEL_W, MODEL_H
    import op_lb
    cam = e["cams"][-1]["CAM_F0"]
    m = Z.OpenpilotMaps.__new__(Z.OpenpilotMaps)
    uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
    m.idx, m.coverage = [], []
    w, h = Z.NUPLAN_WH
    for Km in (MEDMODEL_K, SBIGMODEL_K):
        ray = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(Km @ VIEW_FROM_DEVICE).T    # x fwd, y right, z down
        ray[..., 2] = np.where(ray[..., 2] > 0, ray[..., 2] * r, ray[..., 2])
        uv, ok, _ = Z.project_nuplan(ray * np.array([1., -1., -1.]), cam, 1)
        xi = np.clip(np.rint(uv[..., 0]), 0, w - 1).astype(np.int64)
        yi = np.clip(np.rint(uv[..., 1]), 0, h - 1).astype(np.int64)
        m.idx.append((yi * w + xi).ravel())
        m.coverage.append(float(ok.mean()))
    keys = np.stack([m(m.decode(e["cams"][f]["CAM_F0"]["path"])) for f in range(4)])
    camv = np.asarray(cam["t"], float).copy()
    camv[2] = (camv[2] + ZG) / r                      # the warp's road plane: ground at the virtual camera height
    syn = I.synth_cpu(keys, "warp", op_lb.SYN_T, I.track_navsim(e["pose"], e["vel"]), camv)
    return keys, syn, m.coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--ratios", nargs="+", type=float, default=[1.0, 1.44])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default=os.path.expandvars("$DATA_DIR/runs/skill_pack/edge_diag/height"))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from concurrent.futures import ProcessPoolExecutor
    import op_lb
    from jevdrive.common import data_dir
    from jevdrive.openpilot.model import OPModel, decode
    idx = pickle.load(open(data_dir() / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))
    sel = np.sort(np.random.default_rng(0).choice(len(idx), a.n, replace=False))
    sub = [idx[k] for k in sel]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = OPModel("cinque", "trt", cache=data_dir() / "runs" / "op_interp" / "trt_cache" / "cinque-trt", context_rate=False)
    ts, src = op_lb._steps(0.0, False)
    heads = sorted(((q, s) for q, s in m.slices.items() if q not in ("hidden_state", "pad")), key=lambda x: x[1].start)
    keep = np.concatenate([np.arange(s.start, s.stop) for _, s in heads])
    hs = dict(zip([q for q, _ in heads], np.cumsum([0] + [s.stop - s.start for _, s in heads]).tolist()))
    for r in a.ratios:
        P = {"plan_pos": [], "plan_vel": [], "plan_yaw": [], "heads": [], "cov": []}
        with ProcessPoolExecutor(a.workers) as ex:
            for e, (kf, sf, cov) in zip(sub, ex.map(render, [(e, r) for e in sub], chunksize=4)):
                frames = [np.ascontiguousarray(kf[j] if s == "k" else sf[j]) for s, j in src]
                tc = (0, 1) if e["map"] in op_lb.LHT_MAPS else (1, 0)
                m.reset()
                for f in frames:
                    raw = m.step(f, desire=np.zeros(8, np.float32), traffic=tc, action_t=op_lb.ACTION_T)
                d = decode(raw, m.slices, float(np.linalg.norm(e["vel"][-1])), op_lb.ACTION_T)
                for q in ("plan_pos", "plan_vel", "plan_yaw"):
                    P[q].append(d[q])
                P["heads"].append(raw[keep])
                P["cov"].append(cov)
        info = json.dumps({"model": "cinque", "frames": "warp", "ratio": r, "zg": ZG, "heads_slices": hs, "step_times": ts.tolist()})
        np.savez(out / f"hgt_r{r:g}.npz", names=np.array([e["token"] for e in sub]), info=info,
                 **{k: np.asarray(v, np.float32) for k, v in P.items()})
        print(f"r {r}: {len(sub)} tokens, coverage road / wide {np.mean(P['cov'], 0).round(3).tolist()}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
