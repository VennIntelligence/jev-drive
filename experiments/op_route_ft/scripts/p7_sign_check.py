"""Sign check of the p7 plan-position execution path on the d133 B2D 25-turn units (results/p7_sign_check.md).

Per plan record (20 Hz) where openpilot owns lateral and its plan path is executed (p7; hyb above its switch speed; curv =
the action-curvature control, where the plan is only logged), correlate the plan's lateral offset with the measured ego
motion over the next second:
  y1      plan lateral at t + 1 s, rear-axle frame x forward / y LEFT (op_xy[0], after rigs.openpilot_plan_to_rig)
  yexec   lateral of the path actually handed to P7, place([0,0] + plan, s_fin) at the 2 s arc of the arbitrated profile
          (reconstructed from the logged 1/2/3/5 s plan points; exact up to that coarse polyline)
  r_left  ego yaw rate, left positive, over (t, t + 1 s] = -d(CARLA yaw)/dt from the truth pose (CARLA yaw is clockwise)
  s_left  mean steer command over (t, t + 0.5 s], left positive (= -CARLA steer)
Also: plan arc at 5 s vs the arbitrated arc at 5 s (how far place() extrapolates the plan's last segment).
Runs on the box (numpy): python experiments/op_route_ft/scripts/p7_sign_check.py [--out json]."""
import argparse
import glob
import json
import math
import os

import numpy as np

DATA = os.path.join(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"), "runs/op_route_ft")
UNITS = {  # (arm, executor) -> unit glob
    ("shipped", "p7"): "plan_track/shipped/b2d/turns-s2-k*", ("rc-ctl-s0", "p7"): "plan_track/rc-ctl-s0/b2d/turns-s2-k*",
    ("rc-bear-s0", "p7"): "plan_track/rc-bear-s0/b2d/turns-s2-k*", ("rc-poly-s0", "p7"): "plan_track/rc-poly-s0/b2d/turns-s2-k*",
    ("shipped", "hyb"): "plan_track/shipped-hyb/b2d/turns-s2-k*", ("shipped", "curv"): "desire_off/shipped/b2d/turns-s2-k*",
    ("rc-ctl-s0", "curv"): "desire_off/rc-ctl-s0/b2d/turns-s2-k*", ("rc-bear-s0", "curv"): "desire_off/rc-bear-s0/b2d/turns-s2-k*"}
HYB_V = 3.0


def arc(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def place(path, s):
    """lib/op_arb_agent.place (copied: that module imports CARLA)."""
    a = arc(path)
    if a[-1] < 1e-3:
        return np.repeat(path[:1], len(s), 0)
    out = np.stack([np.interp(s, a, path[:, k]) for k in range(2)], -1)
    over = s > a[-1]
    if over.any():
        d = path[-1] - path[max(len(path) - 2, 0)]
        d = d / max(np.linalg.norm(d), 1e-6)
        out[over] = path[-1] + (s[over] - a[-1])[:, None] * d
    return out


def load(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def attempt_rows(adir, cond):
    plans, ticks = load(os.path.join(adir, "plans.jsonl")), load(os.path.join(adir, "ticks.jsonl"))
    ticks = [r for r in ticks if "truth" in r]
    if len(ticks) < 40:
        return []
    tt = np.array([r["t"] for r in ticks])
    yaw = np.unwrap(np.array([r["truth"][2] for r in ticks]))
    steer = np.array([r["steer"] for r in ticks])
    rows, hyb = [], False
    for r in plans:
        v = float(r["v"])
        hyb = v >= HYB_V or (hyb and v >= HYB_V - 0.5)
        plan_exec = cond == "p7" or (cond == "hyb" and hyb)
        if r.get("warm") or r.get("lat") != "op" or (cond != "curv" and not plan_exec):
            continue
        t = float(r["t"])
        if t + 1.0 > tt[-1]:
            continue
        op = np.array(r["op_xy"], float)                     # t = 1, 2, 3, 5 s
        geom = np.r_[[[0.0, 0.0]], op]
        s5 = min(r["s"].values())
        s2 = min(r["s2"].values())
        yexec = float(place(geom, np.array([max(s2, 0.0)]))[0, 1])
        y_aim = float(place(geom, np.array([max(3.0, 0.5 * v)]))[0, 1])   # P7 lookahead max(3 m, 0.5 s * v)
        i0, i1 = np.searchsorted(tt, [t, t + 1.0])
        ih = np.searchsorted(tt, t + 0.5)
        rows.append(dict(t=t, v=v, y1=float(op[0, 1]), x1=float(op[0, 0]), yexec=yexec, yaim=y_aim, L5=float(arc(geom)[-1]), s5=s5,
                         r_left=-float(yaw[min(i1, len(tt) - 1)] - yaw[i0]) / max(tt[min(i1, len(tt) - 1)] - tt[i0], 1e-3),
                         s_left=-float(steer[i0 + 1:ih + 1].mean()) if ih > i0 else 0.0,
                         k_left=-float(r["act_k"]), base_y=float(r["base_xy"][1][1]),
                         k_plan=2 * float(op[2, 1]) / float(op[2] @ op[2]) if float(op[2] @ op[2]) > 1.0 else float("nan")))
    return rows


def stat(x, y, n_min=20):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < n_min:
        return dict(n=int(len(x)))
    agree = float(np.mean(np.sign(x) == np.sign(y)))
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return dict(n=int(len(x)), agree=round(agree, 3), pearson=round(float(np.corrcoef(x, y)[0, 1]), 3),
                spearman=round(float(np.corrcoef(rx, ry)[0, 1]), 3))


def summarize(rows):
    g = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    if not rows:
        return {}
    mov = (g["v"] > 0.5) & (np.abs(g["y1"]) > 0.05)
    lat = np.abs(g["yexec"]) > 0.05
    ext = g["s5"] > g["L5"] + 1.0
    out = {"n_frames": len(rows),
           "y1_vs_yawrate (v>0.5, |y1|>0.05)": stat(g["y1"][mov], g["r_left"][mov]),
           "yexec_vs_yawrate (v>0.5, |yexec|>0.05)": stat(g["yexec"][lat & (g["v"] > 0.5)], g["r_left"][lat & (g["v"] > 0.5)]),
           "yaim_vs_steer (|yaim|>0.05)": stat(g["yaim"][np.abs(g["yaim"]) > 0.05], g["s_left"][np.abs(g["yaim"]) > 0.05]),
           "y1_vs_steer (|y1|>0.05)": stat(g["y1"][np.abs(g["y1"]) > 0.05], g["s_left"][np.abs(g["y1"]) > 0.05]),
           "y1_vs_actk (|y1|>0.05)": stat(g["y1"][np.abs(g["y1"]) > 0.05], g["k_left"][np.abs(g["y1"]) > 0.05]),
           "y1_vs_route_y10 (|y1|>0.05, |route|>0.3)": stat(g["y1"][(np.abs(g["y1"]) > 0.05) & (np.abs(g["base_y"]) > 0.3)],
                                                         g["base_y"][(np.abs(g["y1"]) > 0.05) & (np.abs(g["base_y"]) > 0.3)]),
           "share_extrapolated (s5 > L5 + 1 m)": round(float(ext.mean()), 3),
           "share_plan_short (L5 < 2 m)": round(float(np.mean(g["L5"] < 2.0)), 3),
           "share_v<1": round(float(np.mean(g["v"] < 1.0)), 3)}
    for name, m in (("extrapolated", ext), ("not_extrapolated", ~ext)):
        mm = m & (np.abs(g["yaim"]) > 0.05)
        out["yaim_vs_steer | " + name] = stat(g["yaim"][mm], g["s_left"][mm])
        mm2 = m & (np.abs(g["y1"]) > 0.05)
        out["y1_vs_steer | " + name] = stat(g["y1"][mm2], g["s_left"][mm2])
    for lo, hi in ((0.0, 0.3), (0.3, 1.5), (1.5, 3.0), (3.0, 99.0)):    # speed bins: plan curvature (3 s point) vs measured
        m = (g["v"] >= lo) & (g["v"] < hi)
        if m.sum() < 20:
            continue
        km = np.abs(g["r_left"][m]) / np.maximum(g["v"][m], 0.3)
        kp = np.abs(g["k_plan"][m])
        mm = m & (np.abs(g["k_left"]) > 1e-3) & (np.abs(g["r_left"]) > 0.02)
        out["v[%g,%g)" % (lo, hi)] = dict(n=int(m.sum()), plan_k3s_med=round(float(np.nanmedian(kp)), 3),
                                          plan_k_gt_0p1=round(float(np.nanmean(kp > 0.1)), 2), meas_k_med=round(float(np.median(km)), 3),
                                          meas_k_gt_0p1=round(float(np.mean(km > 0.1)), 2), act_k_med=round(float(np.median(np.abs(g["k_left"][m]))), 4),
                                          yawrate_vs_y1=stat(g["y1"][mm], g["r_left"][mm]), yawrate_vs_actk=stat(g["k_left"][mm], g["r_left"][mm]))
    for lo, hi in ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 1e9)):      # plan length bins: does the executed aim follow the model's intent?
        m = (g["L5"] >= lo) & (g["L5"] < hi) & (np.abs(g["yaim"]) > 0.05) & (np.abs(g["k_left"]) > 1e-3)
        out["L5[%g,%g) aim_vs_actk" % (lo, hi)] = dict(stat(g["yaim"][m], g["k_left"][m]), share=round(float(np.mean((g["L5"] >= lo) & (g["L5"] < hi))), 3),
                                                     v_med=round(float(np.median(g["v"][m])), 2) if m.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    res, pooled = {}, {"p7": [], "hyb": [], "curv": []}
    for (arm, cond), pat in UNITS.items():
        rows = []
        for adir in sorted(glob.glob(os.path.join(a.data, pat, "attempts", "*", "*"))):
            if os.path.exists(os.path.join(adir, "plans.jsonl")) and os.path.exists(os.path.join(adir, "ticks.jsonl")):
                rows += attempt_rows(adir, cond)
        res["%s|%s" % (arm, cond)] = summarize(rows)
        pooled[cond] += rows
    for cond, rows in pooled.items():
        res["POOLED|" + cond] = summarize(rows)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
