"""Spin-out analysis of the HUGSIM zero-shot exam (64 scenarios x {cinque, lebowski, ltf, cv} x {official, fixed}).

For every run: executed heading error against the recorded route (infos.pkl), commanded direction of the plan
(zs_steps.jsonl for openpilot, data.pkl planned_traj for LTF / cv), and the first step at which the yaw diverges.
    python spin_analysis.py <runs_root> <routes.json> <out_dir>
runs_root holds scored-op/{cinque,lebowski}-{official,fixed}/zs/<run>/ and scored-base/{ltf,cv}-{official,fixed}/*/<run>/
with results.csv in each of scored-op, scored-base; routes.json comes from spin_export_routes.py.
Conventions: theta positive = turning right; plan axes x right, y forward (HUGSIM); phi = atan2(x, y) of the plan
point at 1.0 s, positive = to the right of the ego heading.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SPIN_DEG, ONSET_DEG = 60.0, 20.0


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def load_run(d: Path, agent: str):
    infos = pickle.load(open(d / "infos.pkl", "rb"))
    pos = np.array([[i["ego_pos"][0], i["ego_pos"][2]] for i in infos])
    th = []
    for i in infos:
        R = Rotation.from_euler("XYZ", i["ego_rot"]).as_matrix()
        th.append(np.arctan2(R[0, 2], R[2, 2]))
    th = np.unwrap(th)
    v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in infos])
    steer = np.array([float(np.ravel(i["ego_steer"])[0]) for i in infos])
    plans = None
    zf = d / "zs_steps.jsonl"
    if zf.exists():
        recs = [json.loads(x) for x in open(zf)][1:]
        plans = np.array([r["plan"] for r in recs])[:, :, :]            # (n, 6, 2) x right, y forward, as sent
        raw = [r.get("raw_plan") for r in recs]
    else:
        fr = pickle.load(open(d / "data.pkl", "rb"))[0]["frames"]
        plans = []
        for f in fr:
            ex, ey, _, _, _, _, eyaw = f["ego_box"]
            g = np.array([p[:2] for p in f["planned_traj"]["traj"]]) - [ex, ey]
            c, s = np.cos(eyaw), np.sin(eyaw)
            fwd, left = g[:, 0] * c + g[:, 1] * s, -g[:, 0] * s + g[:, 1] * c
            plans.append(np.stack([-left, fwd], -1))
        plans = np.array(plans)
        raw = None
    n = min(len(pos), len(plans))
    return pos[:n], th[:n], v[:n], steer[:n], plans[:n]


def analyse(pos, th, v, steer, plans, route):
    rxz, ryaw = np.asarray(route["xz"]), np.unwrap(route["yaw"])
    idx = np.array([np.argmin(((rxz - p) ** 2).sum(1)) for p in pos])
    look = np.minimum(idx + 6, len(rxz) - 1)                           # ~5 m ahead on the densified route
    e = wrap(th - ryaw[idx])                                           # heading error, + = pointing right of the route
    # direction of the route a few metres ahead, relative to the current heading (what a good plan should point at)
    d = rxz[look] - pos
    des = wrap(np.arctan2(d[:, 0], d[:, 1]) - th)
    phi = np.arctan2(plans[:, 1, 0], plans[:, 1, 1])                   # plan point at 1.0 s
    phi3 = np.arctan2(plans[:, -1, 0], np.maximum(plans[:, -1, 1], 1e-3))
    plen = np.linalg.norm(plans[:, -1], axis=1)
    moving = plen > 1.0                                                # a stop plan has no direction
    dth = np.diff(th, append=th[-1])                                   # executed yaw change over the step
    out = dict(max_abs_e=float(np.degrees(np.abs(e).max())), n=len(pos))
    out["spin"] = out["max_abs_e"] >= SPIN_DEG
    on = np.where(np.abs(e) >= np.radians(ONSET_DEG))[0]
    out["onset"] = int(on[0]) if len(on) else -1
    # step where the yaw starts to diverge: last step before onset at which |e| was still below 5 deg
    if out["onset"] >= 0:
        k0 = out["onset"]
        while k0 > 0 and abs(e[k0 - 1]) >= np.radians(5):
            k0 -= 1
        out["start"] = k0
        w = slice(max(0, k0 - 1), k0 + 3)                              # the plans that produced the divergence
        pe = (phi - des)[w][moving[w]]
        out["plan_err_deg"] = float(np.degrees(np.mean(np.abs(pe)))) if len(pe) else float("nan")
        sgn = np.sign(e[out["onset"]])
        out["plan_err_signed_deg"] = float(np.degrees(np.mean(sgn * (phi - des)[w]))) if len(pe) else float("nan")
        out["v_start"], out["v_onset"] = float(v[k0]), float(v[out["onset"]])
        out["moving_at_start"] = bool(moving[k0])
        out["first_dir"] = "right" if sgn > 0 else "left"
        out["sign_ok_steps"] = float(np.mean(np.sign(dth[:-1][moving[:-1]]) == np.sign(phi[:-1][moving[:-1]]))) if moving[:-1].any() else float("nan")
    # temporal order: first step the plan points >= 10 deg away from the route direction (same side as the later spin)
    # against the first step the heading error reaches 5 deg. Plan first -> the model asked for the turn;
    # heading first -> the vehicle yawed while the plan still pointed along the route (controller / conversion).
    ty = np.where(np.abs(e) >= np.radians(5))[0]
    out["t_yaw5"] = int(ty[0]) if len(ty) else -1
    if len(ty):
        sg = np.sign(e[ty[0]]) if abs(e[ty[0]]) > 0 else 1.0
        tp = np.where(moving & (sg * wrap(phi - des) >= np.radians(10)))[0]
        out["t_plan10"] = int(tp[0]) if len(tp) else -1
        out["plan_first"] = bool(len(tp) and tp[0] <= ty[0])
        pre = slice(0, ty[0] + 1)
        out["plan_err_pre_deg"] = float(np.degrees(np.mean(np.abs(wrap(phi - des))[pre][moving[pre]]))) if moving[pre].any() else float("nan")
    m = moving & (np.abs(phi) > np.radians(3))
    m[-1] = False
    out["track_sign"] = float(np.mean(np.sign(dth[m]) == np.sign(phi[m]))) if m.any() else float("nan")
    out["plan_turn_frac"] = float(np.mean(np.abs(phi - des)[moving] > np.radians(15))) if moving.any() else 0.0
    return out, dict(e=e, phi=phi, des=des, dth=dth, v=v, moving=moving)


def main(root, routes, out):
    root, out = Path(root), Path(out)
    routes = json.load(open(routes))
    rows = []
    for grp in ("scored-op", "scored-base"):
        for r in csv.DictReader(open(root / grp / "results.csv")):
            tag = r["tag"]
            d = Path(r["run_dir"])
            d = root / grp / tag / d.parent.name / d.name
            if not (d / "infos.pkl").exists():
                continue
            pos, th, v, steer, plans = load_run(d, r["agent"])
            res, ser = analyse(pos, th, v, steer, plans, routes[r["scene"]])
            res.update(scenario=r["scenario"], dataset=r["dataset"], difficulty=r["difficulty"], agent=r["agent"],
                       controller=r["controller"], end=r["end"], hdscore=float(r["hdscore"]), rc=float(r["rc"]),
                       steps=int(r["steps"]))
            rows.append(res)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "spin_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "runs ->", out / "spin_episodes.csv")


if __name__ == "__main__" and len(sys.argv) <= 4:
    main(*sys.argv[1:4])


def hazard(root, routes, out):
    """Per-step hazard of leaving the route heading, by speed bin: among steps with |heading error| < 5 deg (still aligned),
    the share whose next step has |e| >= 5 deg, and the share whose plan already points >= 10 deg off the route direction.
    Written to <out>/spin_hazard.csv (agent, controller, speed bin, steps, p_diverge, p_plan_off)."""
    root, out = Path(root), Path(out)
    routes = json.load(open(routes))
    acc = {}
    for grp in ("scored-op", "scored-base"):
        for r in csv.DictReader(open(root / grp / "results.csv")):
            d = Path(r["run_dir"])
            d = root / grp / r["tag"] / d.parent.name / d.name
            if not (d / "infos.pkl").exists():
                continue
            pos, th, v, steer, plans = load_run(d, r["agent"])
            _, s = analyse(pos, th, v, steer, plans, routes[r["scene"]])
            e, phi, des, mv = s["e"], s["phi"], s["des"], s["moving"]
            for k in range(len(e) - 1):
                if abs(e[k]) < np.radians(5) and mv[k]:
                    b = "0-2" if v[k] < 2 else "2-4" if v[k] < 4 else "4-6" if v[k] < 6 else "6-9" if v[k] < 9 else "9+"
                    a = acc.setdefault((r["agent"], r["controller"], b), [0, 0, 0])
                    a[0] += 1
                    a[1] += abs(e[k + 1]) >= np.radians(5)
                    a[2] += abs(wrap(phi[k] - des[k])) >= np.radians(10)
    with open(out / "spin_hazard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "controller", "speed_bin", "steps", "p_diverge_next", "p_plan_off10"])
        for (a, c, b), (n, d1, d2) in sorted(acc.items()):
            w.writerow([a, c, b, n, round(d1 / n, 4), round(d2 / n, 4)])


if __name__ == "__main__" and len(sys.argv) > 4 and sys.argv[4] == "hazard":
    hazard(*sys.argv[1:4])
