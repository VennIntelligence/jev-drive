"""Minimal HUGSIM agent client speaking the official closed_loop.py FIFO protocol.

Protocol (see docs/hugsim.md): the simulator writes pickle((obs, info)) to <output>/obs_pipe, the agent
answers with pickle(waypoints) on <output>/plan_pipe, where waypoints is an (N, 2) float array in the
HUGSIM "lidar" ego frame (x right, y forward, metres), one point every 0.5 s starting at t = +0.5 s.
Sending None ends the episode as an agent crash; the simulator sends the string 'Done' at the end.

Policies (no learned model; for smoke tests and as floor / ceiling references):
  cv     constant velocity straight ahead (the floor: ignores the road).
  route  privileged: follows the scene's recorded ego route (ground_param.pkl) at a capped speed.
         It reads the route from the scene directory, which a real agent never sees.
"""
import argparse
import os
import pickle
import time

import numpy as np
from scipy.spatial.transform import Rotation

DT_PLAN, N_PLAN = 0.5, 6  # 3 s horizon at 2 Hz, the rate score_calculator assumes


def world_to_ego(info, pts_xz):
    """World (x, z) points on the ground -> ego frame (x right, y forward)."""
    r = Rotation.from_euler('XYZ', info['ego_rot']).as_matrix()
    p = np.zeros((len(pts_xz), 3))
    p[:, [0, 2]] = pts_xz
    p[:, 1] = info['ego_pos'][1]
    loc = (p - np.asarray(info['ego_pos'])) @ r  # R^T (p - t), row-vector form
    return loc[:, [0, 2]]


class RoutePolicy:
    def __init__(self, scene_dir, v_max, a_max):
        with open(os.path.join(scene_dir, 'ground_param.pkl'), 'rb') as f:
            cam_poses, _, _ = pickle.load(f)
        xz = cam_poses[:, [0, 2], 3]
        keep = np.r_[True, np.linalg.norm(np.diff(xz, axis=0), axis=1) > 1e-3]
        self.xz = xz[keep]
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.xz, axis=0), axis=1))]
        self.v_max, self.a_max = v_max, a_max

    def __call__(self, obs, info):
        ego = np.array([info['ego_pos'][0], info['ego_pos'][2]])
        i = int(np.argmin(np.linalg.norm(self.xz - ego, axis=1)))
        v = float(info['ego_velo'])
        t = DT_PLAN * np.arange(1, N_PLAN + 1)
        # accelerate towards v_max, distance travelled at each waypoint time
        vt = np.minimum(v + self.a_max * t, self.v_max)
        d = np.cumsum(np.r_[vt[0] + v, vt[1:] + vt[:-1]] * 0.5 * DT_PLAN)
        s = np.clip(self.s[i] + d, 0, self.s[-1] + 1e-6)
        pts = np.stack([np.interp(s, self.s, self.xz[:, k]) for k in (0, 1)], axis=1)
        if self.s[i] + d[-1] > self.s[-1]:  # past the end of the route: extrapolate straight
            tail = self.xz[-1] - self.xz[-2]
            tail /= np.linalg.norm(tail) + 1e-9
            over = np.maximum(self.s[i] + d - self.s[-1], 0)
            pts += over[:, None] * tail
        return world_to_ego(info, pts)


def cv_policy(obs, info, v_min=1.0):
    v = max(float(info['ego_velo']), v_min)
    t = DT_PLAN * np.arange(1, N_PLAN + 1)
    return np.stack([np.zeros_like(t), v * t], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--policy', choices=['cv', 'route'], default='route')
    ap.add_argument('--scene_dir', default=os.environ.get('HUGSIM_SCENE_DIR'))
    ap.add_argument('--v_max', type=float, default=6.0)
    ap.add_argument('--a_max', type=float, default=1.5)
    args = ap.parse_args()

    policy = cv_policy if args.policy == 'cv' else RoutePolicy(args.scene_dir, args.v_max, args.a_max)
    obs_pipe, plan_pipe = (os.path.join(args.output, n) for n in ('obs_pipe', 'plan_pipe'))
    os.makedirs(args.output, exist_ok=True)
    for p in (obs_pipe, plan_pipe):
        if not os.path.exists(p):
            os.mkfifo(p)
    print(f'agent ready: policy={args.policy}', flush=True)
    step, t_wait = 0, 0.0
    while True:
        t0 = time.perf_counter()
        with open(obs_pipe, 'rb') as f:
            msg = pickle.loads(f.read())
        t_wait += time.perf_counter() - t0
        if isinstance(msg, str) and msg == 'Done':
            print(f'done after {step} steps, {t_wait:.1f} s waiting for observations', flush=True)
            return
        obs, info = msg
        wp = np.asarray(policy(obs, info), dtype=np.float64)
        with open(plan_pipe, 'wb') as f:
            f.write(pickle.dumps(wp))
        if step % 20 == 0:
            print(f'step {step} t={info["timestamp"]:.2f} v={info["ego_velo"]:.2f} cmd={info["command"]} '
                  f'wp_last={wp[-1].round(2).tolist()}', flush=True)
        step += 1


if __name__ == '__main__':
    main()
