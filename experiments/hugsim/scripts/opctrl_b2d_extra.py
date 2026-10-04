"""Extra readouts of the B2D openpilot-lateral-path lane (plans/2026-10-04-op-control-stack-b2d-prereg.md), from ticks.jsonl / plans.jsonl.

  $DATA_DIR/envs/jevdrive/bin/python experiments/hugsim/scripts/opctrl_b2d_extra.py   (on the box)
  -> experiments/hugsim/results/opctrl_b2d/extra.md

1. Realised low-speed c, drive vs opc (lowspeed_c_measure.b2d_log / slope_ci: lat == op, non-warm, non-zone steps).
2. Who steers: share of non-warm ticks owned by the openpilot path (lat == op), by the route (zone / div), and the heading change each owner delivered
   (sum |d yaw| over its ticks), per arm. Junction turns are zone-owned in both arms; the opc path acts only on op-owned ticks.
3. opc path trace on op-owned ticks (`opc` field): how often modeld's hold / inactive, jerk / lateral-acceleration / |kappa| clipping bind
   (clipped = des != act), the delay's effect (|real - des|), and the peak |kappa| the model asked for vs the car realised.
4. Turning in op-owned bends: ticks with |commanded kappa| > 0.01 (R < 100 m): realised / commanded kappa, and the lag-free steer from the shipped path is not logged, so
   drive's own steer is compared through the steer->kappa geometry (steer log) at the same speed bins.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import lowspeed_c_measure as cm  # noqa: E402
from vlm_arb_common import DATA  # noqa: E402

ARMS = DATA / "runs/vlm_arb/arms"
OUT = REPO / "experiments/hugsim/results/opctrl_b2d"
BINS = [(0, 1), (1, 2), (2, 3), (0, 3)]


def runs(arm):
    return sorted(ARMS.glob("v2-%s-s[23]-q*/attempts/*/1/ticks.jsonl" % arm))


def owners(arm):
    """Per-arm tick accounting; returns dict."""
    n = dict(op=0, zone=0, div=0, other=0); dyaw = dict(op=0.0, zone=0.0, div=0.0, other=0.0)
    for f in runs(arm):
        d = f.parent
        tk = [json.loads(x) for x in open(d / "ticks.jsonl")]
        pl = {p["frame"]: p for p in map(json.loads, open(d / "plans.jsonl"))}
        yaw = np.unwrap(np.array([t["truth"][2] for t in tk]))
        for i in range(1, len(tk)):
            p = pl.get(tk[i]["frame"])
            if p is None or p.get("warm"):
                continue
            k = "op" if p.get("lat") == "op" else "zone" if p.get("lat_why") == "zone" else "div" if p.get("lat_why") == "div" else "other"
            n[k] += 1
            dyaw[k] += abs(math.degrees(yaw[i] - yaw[i - 1]))
    return n, dyaw


def trace(arm="opc"):
    rows = []
    for f in runs(arm):
        for x in open(f):
            r = json.loads(x)
            if "opc" in r:
                o = r["opc"]
                rows.append((r["v"], o["cmd"], o["act"], o["des"], o["real"], o["active"]))
    return np.array(rows, float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# B2D openpilot lateral path: extra readouts (drive = shipped, opc = OP_CTRL)", ""]
    L += ["## 1. Realised c (heading change over 5 ticks per degree of the plan's 1 s direction; lat == op, non-warm, non-zone; magnitude)", "",
          "| v (m/s) | drive | opc |", "|---|---|---|"]
    res = {}
    for arm in ("drive", "opc"):
        rows = cm.b2d_log((str(ARMS), "v2-%s-s[23]-q*/attempts/*/1/ticks.jsonl" % arm))
        res[arm] = {b: cm.slope_ci(rows, b) for b in BINS}
    f = lambda t: "n/a" if t is None else "%.3f [%.3f, %.3f] (n %d, runs %d)" % (abs(t[0]), *sorted([abs(t[1]), abs(t[2])]), t[3], t[4])  # noqa: E731
    for b in BINS:
        L.append("| %d-%d | %s | %s |" % (b[0], b[1], f(res["drive"][b]), f(res["opc"][b])))
    L += ["", "## 2. Who steers (non-warm ticks) and the heading change each owner delivered", "", "| arm | owner | ticks | share | sum abs dyaw (deg) | share of heading |", "|---|---|---|---|---|---|"]
    for arm in ("drive", "opc"):
        n, dy = owners(arm)
        for k in n:
            L.append("| %s | %s | %d | %.1f%% | %.0f | %.1f%% |" % (arm, k, n[k], 100 * n[k] / max(sum(n.values()), 1), dy[k], 100 * dy[k] / max(sum(dy.values()), 1e-9)))
    T = trace()
    if len(T):
        v, cmd, act, des, real, active = T.T
        mov = v > 0.3
        L += ["", "## 3. The path on op-owned ticks (opc)", "",
              "- op-owned ticks with a trace: %d; v <= 0.3 (modeld hold / controlsd inactive): %.1f%%" % (len(T), 100 * (~mov).mean()),
              "- clip_curvature binds (des != act, moving): %.2f%% of moving ticks" % (100 * (np.abs(des - act)[mov] > 1e-7).mean()),
              "- |kappa_cmd| quantiles on moving ticks (1/m) 50 / 90 / 99 / max: %s" % " / ".join("%.4f" % q for q in np.r_[np.percentile(np.abs(cmd[mov]), [50, 90, 99]), np.abs(cmd[mov]).max()]),
              "- |kappa_real| same: %s" % " / ".join("%.4f" % q for q in np.r_[np.percentile(np.abs(real[mov]), [50, 90, 99]), np.abs(real[mov]).max()]),
              "- lateral-acceleration cap binds where |kappa_cmd| v^2 > 3: %.3f%% of moving ticks" % (100 * (np.abs(cmd) * v ** 2 > 3.0)[mov].mean())]
        bend = mov & (np.abs(cmd) > 0.01)
        if bend.sum() > 20:
            sl = float((cmd[bend] * real[bend]).sum() / (cmd[bend] ** 2).sum())
            L += ["- bends (|kappa_cmd| > 0.01, R < 100 m; %d ticks): realised / commanded kappa slope %.3f (delay-lagged but not attenuated if ~1)" % (bend.sum(), sl)]
    (OUT / "extra.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
