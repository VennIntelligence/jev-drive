"""Open-loop tables for the turn-calibration analysis: openpilot's plan-derived curvature vs the logged human future (WOD-E2E rater frames, NAVSIM navtest).

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/turn_calibration_openloop_extract.py

Chord curvature 2 y / (x^2 + y^2) of the point a metres along each path (origin = ego at t0), made right-positive like the B2D tables
(openpilot's plan frame is y-right, the logged futures are y-left). Sources (plan files of runs/op_interp, rendered per the contract fixes of decision 36 / 37):
  wod_real    WOD-E2E 479 rater frames, real 10 Hz history, Cinque           (logged future 5 s @ 4 Hz)
  nav_warp    NAVSIM navtest 12 146 tokens, 2 Hz exam history interpolated (warp), Cinque (logged future 4 s @ 2 Hz)
  nav_native  NAVSIM navtest, the exam's native 2 Hz protocol, Cinque (runs/navsim_zs/openpilot/navtest/cinque_none.npz)
Output: $DATA_DIR/runs/op_closed_loop/turn_calibration/openloop.npz (arrays <src>_<field>).
"""
import sys
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402

D = data_dir() / "runs"
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
ARCS = (10.0, 15.0, 25.0)


def chord(path, a):
    """path (n,2) incl. the origin; chord curvature (left-positive) at arc a; NaN when the path is shorter than a."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < a:
        return np.nan
    x, y = np.interp(a, s, path[:, 0]), np.interp(a, s, path[:, 1])
    return 2 * y / (x * x + y * y)


def end_heading(path):
    d = path[-1] - path[-3]
    return float(np.arctan2(d[1], d[0]))


def table(plans, humans, v_h, cluster, side):
    """plans (n,k,2), humans (n,m,2) with the origin NOT included; returns dict of arrays."""
    n = len(plans)
    out = {"v": np.asarray(v_h, float), "cluster": np.asarray(cluster), "side": np.asarray(side)}
    for a in ARCS:
        out["mod%d" % a] = np.array([chord(np.r_[[[0, 0]], plans[i]], a) for i in range(n)])      # right-positive
        out["req%d" % a] = np.array([-chord(np.r_[[[0, 0]], humans[i]], a) for i in range(n)])
    out["ang"] = np.degrees(np.array([-end_heading(np.r_[[[0, 0]], humans[i]]) for i in range(n)]))   # right-positive total turn of the logged future
    out["mang"] = np.degrees(np.array([end_heading(np.r_[[[0, 0]], plans[i][:25]]) for i in range(n)]))
    return out


def main():
    res = {}
    # ---- WOD
    r = np.load(D / "wod_zeroshot/score/20260924-174107/per_frame.npz", allow_pickle=True)
    p = np.load(D / "op_interp/wod/plans/real@cinque.npz", allow_pickle=True)
    assert list(r["names"]) == list(p["names"])
    logged = r["logged"]
    v = np.linalg.norm(logged[:, 3] - logged[:, 0], axis=1) / 0.75
    clus = np.array([str(x).rsplit("-", 1)[0] for x in r["names"]])
    plans = p["plan_pos"][:, :, :2]
    intent = r["intent"]
    t = table(plans, logged, v, clus, np.where(intent == 2, -1, np.where(intent == 3, 1, 0)))     # routing intent 2 left / 3 right
    res.update({"wod_real_" + k: x for k, x in t.items()})
    # ---- NAVSIM
    idx = pickle.load(open(D / "navsim_zs/index/navtest_slim.pkl", "rb"))
    fut = np.load(D / "navsim_zs/index/navtest_future.npz", allow_pickle=True)
    assert list(fut["tokens"]) == [e["token"] for e in idx]
    hum = fut["poses"][:, :, :2]
    v = np.array([np.linalg.norm(e["vel"][-1]) for e in idx])
    clus = np.array([e["log_name"] for e in idx])
    cmd = np.array([int(np.argmax(e["cmd"][-1])) for e in idx])       # 0 left, 1 straight, 2 right (NAVSIM driving command)
    side = np.where(cmd == 0, -1, np.where(cmd == 2, 1, 0))
    nav = np.load(D / "op_interp/navfull/plans/warp@cinque.npz", allow_pickle=True)
    assert list(nav["names"]) == list(fut["tokens"])
    t = table(nav["plan_pos"][:, :, :2], hum, v, clus, side)
    res.update({"nav_warp_" + k: x for k, x in t.items()})
    nat = np.load(D / "navsim_zs/openpilot/navtest/cinque_none.npz", allow_pickle=True)
    assert list(nat["tokens"]) == list(fut["tokens"])
    t = table(nat["plans"][:, :, :2], hum, v, clus, side)
    res.update({"nav_native_" + k: x for k, x in t.items()})
    out = D / "op_closed_loop/turn_calibration/openloop.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print("->", out, {k: v.shape for k, v in list(res.items())[:3]})


if __name__ == "__main__":
    main()
