"""Lane EDGE fix arms (plan: experiments/skill_pack/plans/2026-10-04-roadedge-diagnosis-plan.md, addendum 1): undo the
camera-height scale of openpilot's metric world on NAVSIM by rescaling the native Cinque plan.

navsim2 env, CPU. Per token (reference-free):
  h_model = median of the model's lane-line z (height of the road below the camera, calib frame) over the two ego lines
            and the knots with 5 <= x <= 30 m;  h_true = CAM_F0 z + ZG (ZG = 0.35 m: the NAVSIM ego origin is the rear axle
            at axle height, measured from nearby vehicle boxes);  k_own = clip(h_true / h_model, 1, 2).
  U(k)  plan positions (camera-origin calib frame) times k, yaw unchanged (uniform scale: path and speed);
  P(k)  path geometry times k, each knot keeps its arc length (speed profile unchanged), yaw of the scaled path there.
Then the same lever-arm conversion to the 8 rear-axle poses as the shipped predictions (checked: base reproduces them).
Writes $DATA_DIR/runs/op_lb/<data>/preds/gimm-cinque_edge-<arm>__base.npz and k_own stats.

  edge_rescale.py --data lb_navtrain lb_navtest lb_navhard --arms U-own P-own U1.15 U1.3 U1.45 P1.15 P1.3 P1.45
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

ZG = 0.35
X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
IDX = {"lb_navtrain": "navtrain_slim", "lb_navtest": "navtest_slim", "lb_navhard": "navhard_two_stage_slim"}


def h_model(heads, sl):
    ll = heads[:, sl["lane_lines"]:sl["lane_lines"] + 264].reshape(-1, 4, 33, 2)
    m = (X_IDXS >= 5) & (X_IDXS <= 30)
    return np.median(ll[:, 1:3, m, 1].reshape(len(heads), -1), 1)


def scale_plan(pos, yaw, k, mode):
    """pos (33, 3) calib frame (camera origin), yaw (33,) -> scaled (pos, yaw)."""
    if mode == "U":
        return pos * np.array([k, k, 1.0]), yaw
    xy = pos[:, :2]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if s[-1] < 1e-3:
        return pos, yaw
    s = np.maximum.accumulate(s)
    so = s / k                                         # original-path arc length whose scaled image has arc length s
    new = np.stack([k * np.interp(so, s, xy[:, 0]), k * np.interp(so, s, xy[:, 1])], 1)
    return np.c_[new, pos[:, 2]], np.interp(so, s, yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=list(IDX))
    ap.add_argument("--arms", nargs="+", default=["U-own", "P-own", "U1.15", "U1.3", "U1.45", "P1.15", "P1.3", "P1.45"])
    a = ap.parse_args()
    stats = {}
    for data in a.data:
        z = np.load(L.D / f"runs/op_lb/{data}/plans/gimm@cinque.npz")
        sl = json.loads(str(z["info"]))["heads_slices"]
        names = z["names"].tolist()
        cam = {e["token"]: np.asarray(e["cams"][-1]["CAM_F0"]["t"], float) for e in pickle.load(open(L.D / f"runs/navsim_zs/index/{IDX[data]}.pkl", "rb"))}
        hm = h_model(z["heads"], sl)
        ht = np.array([cam[t][2] + ZG for t in names])
        kown = np.clip(ht / hm, 1.0, 2.0)
        stats[data] = dict(n=len(names), h_model_median=float(np.median(hm)), h_true_median=float(np.median(ht)),
                           k_own_q=np.percentile(kown, [5, 25, 50, 75, 95]).round(3).tolist())
        base = L.poses_by_token(L.D / f"runs/op_lb/{data}/preds/gimm-cinque__base.npz")
        conv = lambda p, y, t: I.to_rear(p, y, I.T_IDXS, cam[t][:2], L.T_POSE, "lever", "linear")  # noqa: E731
        err = max(float(np.abs(conv(z["plan_pos"][i], z["plan_yaw"][i], t) - base[t]).max()) for i, t in enumerate(names[:200]))
        stats[data]["base_reproduction_max_abs"] = err
        assert err < 1e-3, f"{data}: base poses not reproduced ({err})"
        for arm in a.arms:
            mode = arm[0]
            P = np.stack([conv(*scale_plan(z["plan_pos"][i], z["plan_yaw"][i], kown[i] if arm.endswith("own") else float(arm[1:]), mode), t)
                          for i, t in enumerate(names)]).astype(np.float32)
            out = L.D / f"runs/op_lb/{data}/preds/gimm-cinque_edge-{arm}__base.npz"
            np.savez(out, tokens=np.array(names), poses=P)
            print(data, arm, out, flush=True)
    out = L.D / "runs/skill_pack/edge_diag/rescale_stats.json"
    out.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
