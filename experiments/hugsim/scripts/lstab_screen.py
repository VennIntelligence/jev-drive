"""Offline screen of the launch-stabilisation rule (plans/2026-10-04-launch-stab-prereg.md, part A): a virtual closed loop on
logged HUGSIM frames. Positions follow the log; the heading is simulated: every logged frame is re-rendered (rotation only, exact
for any depth) at the virtual heading, the model plans, and the heading advances by c(v) x the plan direction at 1 s (decision
100's controller transfer: 0.06 / 0.18 / 0.23 deg per 0.25 s step per deg below 1 / 1-2 / 2-3 m/s, 0.23 above). Arms:
native (shipped Cinque) and lstab (lib/launch_stab.py: forked second session on frames at the launch heading, lateral plan
rotated into the car frame). Each arm runs with a +1 / -1 / 0 deg heading kick at step 1; loop gain per step =
(e(K) / e(1)) ** (1 / (K - 1)) with e = (theta(+1) - theta(-1)) / 2.

    CUDA_VISIBLE_DEVICES=0 $DATA_DIR/envs/openpilot/bin/python experiments/hugsim/scripts/lstab_screen.py <jobs.json> --out <out.json>
jobs.json: [{"scenario", "run_dir", "dataset"}] (run_dir with video.mp4, infos.pkl, zs_steps.jsonl), as lean_probe.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "lib"), str(Path(__file__).resolve().parent)]
import launch_stab as LS  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402
from jevdrive.openpilot.model import T_IDXS, decode  # noqa: E402
from lean_probe import LoggedRun, load_models, metrics, rss_guard, WARM, PER  # noqa: E402

KICKS = (1.0, -1.0, 0.0)


def ctrl(v):
    return 0.06 if v < 1 else 0.18 if v < 2 else 0.23


def loop(R, m, ms, arm, kick, steps):
    """Virtual closed loop; returns per-step virtual heading (deg, + right) and plan direction at 1 s (deg, + left)."""
    th = float(R.th[0])                                  # rad, + right
    th_v, phis, modes = [], [], []
    gate = LS.LaunchGate() if arm == "lstab" else None
    m.reset()
    if ms is not None:
        ms.reset()
    for k in range(steps):
        if k == 1:
            th += np.radians(kick)
        th_v.append(float(np.degrees(th)))
        reps = WARM if k == 0 else PER
        dv = np.zeros(8, np.float32)
        dv[R.des[k]] = 1
        mode = "off"
        if gate is not None:
            mode = gate.step(0.25 * k, float(R.v[k]), float(th), R.des[k])
            if mode == "fork":
                LS.fork_state(ms, m)
        img = R.frame(k, float(np.degrees(R.th[k] - th)))         # logged frame seen from the virtual heading
        for _ in range(reps):
            raw = m.step(img, desire=dv, traffic=tuple(R.tc))
        out = metrics(raw, m)
        phi = out["phi1"]
        if mode != "off":
            imgs = R.frame(k, float(np.degrees(R.th[k] - gate.ref)))   # ... and from the launch heading
            for _ in range(reps):
                sraw = ms.step(imgs, desire=dv, traffic=tuple(R.tc))
            d, dn = decode(sraw, ms.slices, 0.0), decode(raw, m.slices, 0.0)
            sp, _ = LS.to_car(d["plan_pos"], None, gate.delta(float(th)))
            pos = LS.merge_lateral(dn["plan_pos"], sp)
            p = Z.openpilot_to_plan(pos, T_IDXS, 1.25)
            phi = float(-np.degrees(np.arctan2(p[1, 0], max(p[1, 1], 1e-3))))
        phis.append(phi)
        modes.append(mode)
        th -= np.radians(ctrl(float(R.v[k])) * phi)                  # phi + left, theta + right
    return th_v, phis, modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    t0 = time.time()
    mods = load_models(["O"])
    m = mods["O"]
    ms = load_models(["O"])["O"]
    res = {}
    for job in jobs:
        R = LoggedRun(job, a.steps)
        n = min(a.steps, R.n)
        r = res[job["scenario"]] = {"v": R.v[:n].tolist(), "theta_log": np.degrees(R.th[:n]).tolist()}
        for arm in ("native", "lstab"):
            for kick in KICKS:
                th_v, phis, modes = loop(R, m, ms, arm, kick, n)
                r[f"{arm}|{kick:+g}"] = {"theta": th_v, "phi1": phis, "mode": modes}
        rss_guard()
        print(f"{job['scenario']}: {n} steps", flush=True)
    res["_meta"] = dict(steps=a.steps, wall_s=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
