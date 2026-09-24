"""Isolate HUGSIM's actuation (traj2control -> iLQR -> kinematic bicycle) and compare the official reference-heading
computation against upstream PR #57 (hyzhou404/HUGSIM, unmerged). No rendering: the env's own step equations.

Two drives, both with plans in the frame every shipped client uses (x right, y forward, 0.5 s spacing):
  straight  the same dead-straight 5 m/s plan every step, starting at 5 m/s
  route     scripts/hugsim/agent_client.py's privileged route follower on a recorded route (ground_param.pkl)

  cd $DATA_DIR/third_party/HUGSIM && $DATA_DIR/envs/hugsim/bin/python \
      $DATA_DIR/jev-drive/scripts/hugsim/lqr_heading_check.py --route <scene dir with ground_param.pkl> --out <dir>
Writes traj.csv (one row per step and variant) and, with --fig <dir>, hugsim-lqr-heading.{pdf,png}.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())  # HUGSIM repo root
from sim.ilqr.lqr import plan2control  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_client import RoutePolicy  # noqa: E402

DT, L = 0.25, 2.7  # configs/sim/kinematic.yaml: dt, Lr + Lf


def traj2control(plan, v, steer, fixed):
    """sim/utils/sim_utils.traj2control, with PR #57 as the only difference when fixed=True."""
    stats = np.zeros((plan.shape[0] + 1, 5))
    stats[1:, :2] = plan[:, [1, 0]]
    pa, pb = 0.0, 0.0
    for i, row in enumerate(plan):
        a, b = (row[1], row[0]) if fixed else (row[0], row[1])
        rot = np.arctan2(b - pb, a - pa)
        rot = np.where(rot > np.pi / 2, rot - np.pi, rot)
        rot = np.where(rot < -np.pi / 2, rot + np.pi, rot)
        stats[i + 1, 2] = rot
        pa, pb = a, b
    return plan2control(stats, np.array([0.0, 0.0, 0.0, v, steer]))


def drive(policy, fixed, v0, steps):
    x = z = th = st = 0.0
    v, rows = v0, []
    for k in range(steps):
        info = {'ego_pos': [x, 0.0, z], 'ego_rot': [0.0, th, 0.0], 'ego_velo': v, 'ego_steer': st}
        acc, sr = traj2control(policy(None, info), v, st, fixed)
        rows.append((k * DT, x, z, th, v, st, acc, sr))
        v += acc * DT                      # hug_sim.HUGSimEnv.step, verbatim
        st += sr * DT
        x += v * math.sin(th) * DT
        z += v * math.cos(th) * DT
        th += v * math.tan(st) / L * DT
    return pd.DataFrame(rows, columns=['t', 'x', 'z', 'yaw', 'v', 'steer', 'acc', 'steer_rate'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fig')
    a = ap.parse_args()
    t = 0.5 * np.arange(1, 7)
    straight = lambda obs, info: np.stack([0 * t, 5.0 * t], 1)  # noqa: E731
    route = RoutePolicy(a.route, v_max=6.0, a_max=1.5)
    dfs = []
    for name, pol, v0, n in (('straight', straight, 5.0, 40), ('route', route, 1.0, 60)):
        for fixed in (False, True):
            d = drive(pol, fixed, v0, n)
            d['drive'], d['controller'] = name, 'PR #57 fix' if fixed else 'official'
            if name == 'route':  # distance to the recorded route (the env terminates at 10 m)
                d['off_route'] = [np.min(np.linalg.norm(route.xz - p, axis=1)) for p in d[['x', 'z']].to_numpy()]
            dfs.append(d)
    df = pd.concat(dfs)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / 'traj.csv', index=False)
    for (dr, c), g in df.groupby(['drive', 'controller'], sort=False):
        last = g.iloc[-1]
        extra = f", max off-route {g.off_route.max():.2f} m" if 'off_route' in g and g.off_route.notna().any() else ''
        print(f"{dr:8s} {c:10s} after {last.t + DT:.1f} s: x {last.x:6.2f} z {last.z:6.2f} "
              f"yaw {math.degrees(last.yaw):6.1f} deg{extra}")
    if a.fig:
        plot(df, route.xz, Path(a.fig))


def plot(df, route_xz, out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jevdrive.plots import COLOR, PAGE, STYLE, plt, save
    plt.rcParams.update(STYLE)
    col = {'official': COLOR['qwen_last'], 'PR #57 fix': COLOR['qwen_mean']}
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.3))
    for ax, dr in zip(axes, ('straight', 'route')):
        if dr == 'route':
            ax.plot(route_xz[:, 0], route_xz[:, 1], color=COLOR['baseline'], lw=3, alpha=0.4, label='recorded route')
        else:
            ax.plot([0, 0], [0, 50], color=COLOR['baseline'], lw=3, alpha=0.4, label='planned path')
        for c, g in df[df.drive == dr].groupby('controller', sort=False):
            ax.plot(g.x, g.z, color=col[c], label=f"executed, {c} controller")
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x (right) [m]')
        ax.set_ylabel('z (forward) [m]')
    h, l = axes[1].get_legend_handles_labels()
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0[:1] + h, l0[:1] + l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out, 'hugsim-lqr-heading')


if __name__ == '__main__':
    main()
