#!/usr/bin/env python
"""Controller transfer c at launch speeds: heading realised per 0.25 s step per degree of the plan's 1 s direction.

Three readings, each by speed bin (decision 111's c is the HUGSIM PR#57 controller pooled over v < 3 m/s: 0.19):
  hugsim-syn   PR #57 iLQR (HUGSIM tree) + the env's kinematic bicycle on a synthetic constant-lean plan (1 s direction
               psi in the car frame, plan speed = max(v, 1) m/s, speed held, steer state carried over steps), c_k = dtheta_k / psi
  hugsim-log   regression through the origin of logged dtheta on the plan's 1 s direction (cinque-fixed exam logs)
  b2d-syn      scripts/b2d_controller.py (pursuit P7 config of the `drive` arm) at 20 Hz + bicycle with CARLA's steering curve,
               plan refreshed at 5 Hz, same constant-lean plan
  b2d-log      same regression on the `drive` arm's ticks.jsonl (truth yaw) and plans.jsonl (op_xy 1 s point), lat == "op" steps
  --rule tau   repeats the synthetic readings with the low-speed steering low-pass (lib/lowspeed_ctrl.py)

    cd $DATA_DIR/third_party/HUGSIM-zs/fixed && $DATA_DIR/envs/hugsim/bin/python <this file> --out <dir> [--part all]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts"), str(REPO / "experiments/hugsim/archive")]
sys.path.insert(0, os.getcwd())
D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
DT, L = 0.25, 2.7
SPEEDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (0.0, 3.0)]


def hugsim_syn(psi_deg=2.0, n=6, rule=None):
    import ctrl_offline as C
    import lowspeed_ctrl as LC
    heading, dt, cap = C.VARIANTS["fixed"]
    sol = C.solver(dt, cap)
    out = {}
    psi = math.radians(psi_deg)
    for v in SPEEDS:
        steer, th, cs = 0.0, 0.0, []
        for k in range(n):
            vp = max(v, 1.0)
            d = vp * 0.5 * np.arange(1, 7)
            plan = np.stack([d * math.sin(psi), d * math.cos(psi)], -1)       # (x right, y forward), constant lean
            s = sol.solve(np.array([0.0, 0.0, 0.0, v, steer]), C.reference(plan, heading, dt))
            _, sr = s[-1].input_trajectory[0]
            if rule is not None:
                sr = LC.hugsim_rate(rule, v, steer, sr, DT)
            steer += sr * DT
            dth = v * math.tan(steer) / L * DT
            cs.append(math.degrees(dth) / psi_deg)
        out[v] = cs
    return out


def hugsim_loop(v, s_gain, rule=None, K=30, kick=0.5):
    """Virtual launch loop at held speed v: plan lean psi_k = s_gain * (theta_k - theta_{k-5}) (6-step window kernel of decision
    100), controller + bicycle, a kick of `kick` deg on theta at step 1. Returns (growth per step z over steps 8..K, theta series)."""
    import ctrl_offline as C
    import lowspeed_ctrl as LC
    heading, dt, cap = C.VARIANTS["fixed"]
    sol = C.solver(dt, cap)
    th, steer, hist = [0.0] * 6, 0.0, []
    for k in range(K):
        psi = math.radians(s_gain * (th[-1] - th[-6]))
        d = max(v, 1.0) * 0.5 * np.arange(1, 7)
        plan = np.stack([d * math.sin(psi), d * math.cos(psi)], -1)
        s = sol.solve(np.array([0.0, 0.0, 0.0, v, steer]), C.reference(plan, heading, dt))
        sr = s[-1].input_trajectory[0][1]
        if rule is not None:
            sr = LC.hugsim_rate(rule, v, steer, sr, DT)
        steer += sr * DT
        th.append(th[-1] + math.degrees(v * math.tan(steer) / L * DT) + (kick if k == 0 else 0.0))
    e = np.abs(np.array(th[6:]))
    lo, hi = 8, K
    z = float((e[hi - 1] / e[lo - 1]) ** (1.0 / (hi - lo))) if e[lo - 1] > 1e-9 else float("nan")
    return z, [round(x, 3) for x in th[6:]]


def b2d_loop(v, s_gain, rule=None, K=30, kick=0.5, cfg=None):
    import b2d_controller as BC
    import lowspeed_ctrl as LC
    cfg = cfg or str(REPO / "experiments/b2d_tfv6/results/tfv6-controller/controller-eval/P7.json")
    c = BC.pursuit_from_config(cfg)
    c.reset()
    filt = LC.B2DFilter(rule) if rule is not None else None
    th, t, rate = [0.0] * 6, 0.0, 0.0
    for k in range(K):
        psi = math.radians(s_gain * (th[-1] - th[-6]))                     # left +
        sp = max(v, 1.0) * 0.25 * np.arange(1, 21)
        c.update(np.stack([sp * math.cos(psi), sp * math.sin(psi)], -1), t)
        x = 0.0
        for _ in range(5):
            _, st, _ = c.step(t, v, rate)
            if filt is not None:
                st = filt(st, v, 0.05)
            sc = float(np.interp(v * 3.6, c.steering_curve[:, 0], c.steering_curve[:, 1]))
            kap = math.tan(-st * c.max_steer_rad * sc) / c.wheelbase
            rate = v * kap
            x += math.degrees(rate * 0.05)
            t += 0.05
        th.append(th[-1] + x + (kick if k == 0 else 0.0))
    e = np.abs(np.array(th[6:]))
    lo, hi = 8, K
    z = float((e[hi - 1] / e[lo - 1]) ** (1.0 / (hi - lo))) if e[lo - 1] > 1e-9 else float("nan")
    return z, [round(x, 3) for x in th[6:]]


def hugsim_log(root):
    X = {b: [] for b in BINS}
    runs = sorted((Path(root) / "scored-op" / "cinque-fixed" / "zs").iterdir())
    rows = []
    for d in runs:
        f = d / "zs_steps.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(x) for x in open(f)][1:]
        for k in range(len(recs) - 1):
            r, nx = recs[k], recs[k + 1]
            plan = np.array(r["plan"], float)
            if len(plan) < 2 or np.linalg.norm(plan[1]) < 0.3:
                continue
            rows.append((d.name, float(r["v"]), math.degrees(math.atan2(plan[1, 0], plan[1, 1])),
                         math.degrees(nx["theta"] - r["theta"])))
    return rows


def b2d_syn(psi_deg=2.0, n=6, rule=None, cfg=None):
    import b2d_controller as BC
    import lowspeed_ctrl as LC
    cfg = cfg or str(REPO / "experiments/b2d_tfv6/results/tfv6-controller/controller-eval/P7.json")
    out = {}
    psi = math.radians(psi_deg)
    for v in SPEEDS:
        c = BC.pursuit_from_config(cfg)
        c.reset()
        wb = c.wheelbase
        x = y = th = 0.0
        steer_cmd, hist, cs = 0.0, [], []
        t, th0 = 0.0, 0.0
        filt = LC.B2DFilter(rule) if rule is not None else None
        for k in range(n):
            vp = max(v, 1.0)
            # plan in the rear-axle frame (x forward, y left), 20 points at 0.25 s, constant lean psi (left +)
            sp = vp * 0.25 * np.arange(1, 21)
            traj = np.stack([sp * math.cos(psi), sp * math.sin(psi)], -1)
            # first point must be consistent with motion from the origin: the controller reads the spacing as speed
            c.update(traj, t)
            for _ in range(5):                                               # 0.25 s at 20 Hz
                thr, st, br = c.step(t, v, -(0.0 if not hist else hist[-1]))   # yaw rate: positive left
                if filt is not None:
                    st = filt(st, v, 0.05)
                scale = float(np.interp(v * 3.6, c.steering_curve[:, 0], c.steering_curve[:, 1]))
                delta = -st * c.max_steer_rad * scale                       # CARLA steer positive right -> left-positive angle
                kappa = math.tan(delta) / wb
                dth = v * kappa * 0.05
                th += dth
                hist.append(v * kappa)
                t += 0.05
            cs.append(math.degrees(th - th0) / psi_deg)
            th0 = th
            # plan is regenerated in the new body frame with the same lean (persistent lean)
        out[v] = cs
    return out


def b2d_log(arms_glob, max_runs=100):
    rows = []
    for f in sorted(Path(arms_glob[0]).glob(arms_glob[1]))[:max_runs]:
        d = f.parent
        tk = [json.loads(x) for x in open(d / "ticks.jsonl")]
        pl = {p["frame"]: p for p in map(json.loads, open(d / "plans.jsonl"))}
        yaw = np.unwrap(np.array([t["truth"][2] for t in tk]))
        fr = [t["frame"] for t in tk]
        for i in range(0, len(tk) - 5, 5):
            p = pl.get(fr[i])
            if p is None or p.get("lat") != "op" or p.get("warm") or p.get("zone"):
                continue
            x1, y1 = p["op_xy"][0]
            if x1 < 0.3:
                continue
            rows.append((d.parent.parent.name + "/" + d.parent.name, float(tk[i]["v"]), math.degrees(math.atan2(y1, x1)),
                         math.degrees(yaw[i + 5] - yaw[i])))
    return rows


def slope_ci(rows, b, n=2000, seed=0):
    """Through-origin slope of dtheta on psi in speed bin b, run-cluster bootstrap CI; rows (run, v, psi, dth)."""
    sel = [r for r in rows if b[0] <= r[1] < b[1] and abs(r[2]) < 15]
    if len(sel) < 20:
        return None
    runs = sorted({r[0] for r in sel})
    g = {k: np.array([[r[2], r[3]] for r in sel if r[0] == k]) for k in runs}
    def sl(ks):
        a = np.concatenate([g[k] for k in ks])
        return float((a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum())
    rng = np.random.default_rng(seed)
    bs = [sl(list(rng.choice(runs, len(runs)))) for _ in range(n)]
    return sl(runs), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(sel), len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all")
    ap.add_argument("--rule", default=None, help="JSON of lowspeed_ctrl rule parameters (synthetic parts only)")
    ap.add_argument("--b2d-arms", default=str(D / "runs/vlm_arb/arms"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rule = json.loads(a.rule) if a.rule else None
    res = {}
    if a.part in ("all", "syn"):
        for psi in (1.0, 2.0, 5.0):
            res[f"hugsim_syn_psi{psi}"] = hugsim_syn(psi, rule=rule)
            res[f"b2d_syn_psi{psi}"] = b2d_syn(psi, rule=rule)
    if a.part == "loop":
        for sg in (4.3, 9.0):
            for v in SPEEDS[1::2] + [3.0]:
                res[f"loop_s{sg}_v{v}"] = {"hugsim": hugsim_loop(v, sg, rule)[0], "b2d": b2d_loop(v, sg, rule)[0]}
    if a.part in ("all", "log"):
        rows = hugsim_log(D / "runs/hugsim-exam")
        res["hugsim_log"] = {f"{b[0]}-{b[1]}": slope_ci(rows, b) for b in BINS}
        rows = b2d_log((a.b2d_arms, "v2-drive-s[23]-q*/attempts/*/1/ticks.jsonl"))
        res["b2d_log"] = {f"{b[0]}-{b[1]}": slope_ci(rows, b) for b in BINS}
    (out / ("c_measure" + ("_rule" if a.rule is not None else "") + ".json")).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k)
        for kk, vv in v.items():
            print("   ", kk, [round(x, 3) for x in vv] if isinstance(vv, list) else vv)


if __name__ == "__main__":
    main()
