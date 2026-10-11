#!/usr/bin/env python
"""Frame A/B, stage 1: the served queued ONNX replayed offline on the frames a HUGSIM run recorded (agent opt `trace_frames`).
experiments/hugsim/plans/2026-10-11-frame-ab-prereg.md, results/frame_ab.md.

Per scenario the recorded stream (packed frame, parity bias, desire, model steps per decision) is stepped through the same ONNX on the
serving backend, three times from a zero state:
  off   as served                                   gate (i): must reproduce the plan the rollout logged
  nod   desire zero (the training rows have no desire pulse)
  zb    bias zero                                    the wrong-model control of gate (i)
Writes <out>/onnx/<scenario>.npz (logged and replayed plans, yaw, action head) and <out>/gate_i.json.

  $DATA_DIR/envs/openpilot/bin/python experiments/hugsim/scripts/frame_ab_onnx.py --run-dir <bench run dir> --out <dir>
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts")]
import argparse, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

TOL, T_MAX = 0.03, 3.2          # m; model seconds (= 4 s simulated under the 1.25 dilation)
ARMS = ("off", "nod", "zb")


def load(d: Path) -> dict:
    """A scenario's zs_frames -> stacked arrays (decisions in order) and the setup record."""
    fs = sorted(d.glob("[0-9]*.npz"))
    z = [np.load(f) for f in fs]
    S = {k: np.stack([x[k] for x in z]) for k in ("img2", "bias", "ego", "state", "pos", "yaw", "vel", "act")}
    assert (np.arange(len(fs)) == [int(f.stem) for f in fs]).all(), d
    return dict(S, setup=json.loads((d / "setup.json").read_text()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--onnx", default=str(data_dir() / "runs/bench/hugsim/_onnx/pp-P2H10-F-s0.onnx"))
    ap.add_argument("--backend", default="trt")
    ap.add_argument("--out", required=True)
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    with Run("hugsim_frame_ab", "onnx", seed=a.seed, config=vars(a), resume=a.resume) as run:
        from jevdrive.openpilot import model as M          # onnxruntime with TensorRT: envs/openpilot only
        m = M.OPModel(a.onnx, a.backend)
        near = M.T_IDXS <= T_MAX + 1e-9
        rows = []
        for d in run.tqdm(sorted(Path(a.run_dir).glob("bench/zs/*/zs_frames")), desc="scenarios"):
            S, name = load(d), d.parent.name
            N, res = len(S["img2"]), {}
            for arm in ARMS:
                m.reset()
                pos, yaw, act = np.zeros((N, 33, 3), np.float32), np.zeros((N, 33), np.float32), np.zeros((N, 4))
                for n in range(N):
                    st = S["state"][n]
                    des = np.zeros(8, np.float32)
                    des[0 if arm == "nod" else int(st[7])] = 1                      # as hugsim_zs_server.Openpilot.plan
                    b = np.zeros((1, 32, 512), np.float16) if arm == "zb" else np.asarray(S["bias"][n], np.float16).reshape(1, 32, 512)
                    m.extra = {"intent_bias": b}
                    for _ in range(int(st[8])):
                        raw = m.step(S["img2"][n], desire=des, traffic=tuple(S["setup"]["traffic"]))
                    o = M.decode(raw, m.slices, float(st[9]))
                    pos[n], yaw[n] = o["plan_pos"], o["plan_yaw"]
                    act[n] = o["curvature"], o["curvature_plan"], o["curvature_smooth"], o["accel"]
                res.update({f"pos_{arm}": pos, f"yaw_{arm}": yaw, f"act_{arm}": act})
                dmax = np.abs(pos[:, near, :2] - S["pos"][:, near, :2]).max((1, 2))
                rows.append(dict(scenario=name, arm=arm, n=N, over=int((dmax > TOL).sum()), max_m=float(dmax.max()), median_m=float(np.median(dmax)),
                                 kappa_max=float(np.abs(act[:, 0] - S["act"][:, 0]).max())))
            np.savez(out / "onnx" / f"{name}.npz", pos_log=S["pos"], yaw_log=S["yaw"], act_log=S["act"], state=S["state"], **res)
            run.info("%s: %s", name, json.dumps([r for r in rows if r["scenario"] == name]))
        tot = {arm: dict(n=sum(r["n"] for r in rows if r["arm"] == arm), over=sum(r["over"] for r in rows if r["arm"] == arm),
                         max_m=max(r["max_m"] for r in rows if r["arm"] == arm)) for arm in ARMS}
        for t in tot.values():
            t["share_over"] = t["over"] / max(t["n"], 1)
        ok = tot["off"]["share_over"] <= 0.01 and tot["zb"]["share_over"] >= 0.90
        (out / "gate_i.json").write_text(json.dumps(dict(tol_m=TOL, t_max_model_s=T_MAX, backend=a.backend, onnx=a.onnx, passed=bool(ok), total=tot, rows=rows), indent=1))
        run.summary.update(gate_i=bool(ok), **{f"{k}_share_over": v["share_over"] for k, v in tot.items()})
        run.info("gate (i) %s: %s", "PASS" if ok else "FAIL", json.dumps(tot))


if __name__ == "__main__":
    main()
