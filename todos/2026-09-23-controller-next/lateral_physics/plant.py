"""Planar MKZ plant with PhysX-style tyre semantics, and its open-loop replay fit against logged runs.

Frame: x forward, y right, yaw right-positive (CARLA's left-handed world is the mirror image of a
right-handed one, so the usual planar equations hold with y meaning "right").

Tyres follow PxVehicleComputeTireForceDefault in its linear range: per axle F = -C * v_lat / (|v_long| + 1 m/s)
(gMinLatSpeedForTireModel = 1 * tolerance length = 1 m/s in UE's cm world), with C proportional to the
static axle load (C = c * m g l_other / L). Front steering: CARLA steer * max_steer * steering_curve(v) is the
inner-wheel angle; PhysX Ackermann (accuracy 1) is represented by the bicycle-centre angle.

    python plant.py fit --logs logs.json      # grid-fit c, yaw-inertia ratio and steer lag on logged windows
"""
import argparse
import json
import math

import numpy as np

MASS, WHEELBASE, TRACK = 1696., 2.8604714913890885, 1.5929
MAX_STEER = math.radians(69.99999237060547)
CURVE = np.array([[0., 1.], [20., .9], [60., .8], [120., .7]])
REAR_FROM_ORIGIN, COM_FROM_ORIGIN = -1.388633220199954, .3  # CoM x from calibration-physics.json (see note)
L_R = COM_FROM_ORIGIN - REAR_FROM_ORIGIN
L_F = WHEELBASE - L_R
G = 9.81


def centre_angle(steer, speed):
    """Bicycle-centre road-wheel angle for a CARLA steer command (right positive)."""
    nominal = steer * MAX_STEER * np.interp(abs(speed) * 3.6, CURVE[:, 0], CURVE[:, 1])
    a = abs(nominal)
    if a < 1e-9:
        return 0.
    return math.copysign(math.atan(1. / (1. / math.tan(a) + TRACK / (2 * WHEELBASE))), nominal)


class Plant:
    def __init__(self, c=11.1, inertia_ratio=1., lag_s=0., delay_ticks=0, dt=.01):
        self.c, self.lag, self.delay, self.dt = c, lag_s, delay_ticks, dt
        self.iz = inertia_ratio * MASS * L_F * L_R
        self.cf, self.cr = c * MASS * G * L_R / WHEELBASE, c * MASS * G * L_F / WHEELBASE
        self.reset(np.zeros(2), 0., 0., 0., 0.)

    def reset(self, rear_xy, yaw, vx, vy_rear, w, steer=0.):
        self.rear, self.yaw, self.vx = np.array(rear_xy, float), float(yaw), float(vx)
        self.vy, self.w = float(vy_rear) + L_R * float(w), float(w)  # CoM lateral velocity
        self.delta = centre_angle(steer, vx)
        self.queue = [steer] * self.delay

    def step(self, steer, vx, duration=.05):
        """Advance by one control tick with a zero-order-hold steer command (plus optional delay/lag)."""
        if self.delay:
            self.queue.append(steer)
            steer = self.queue.pop(0)
        self.vx = float(vx)
        for _ in range(int(round(duration / self.dt))):
            target = centre_angle(steer, self.vx)
            self.delta = target if self.lag <= 0 else self.delta + (target - self.delta) * min(1., self.dt / self.lag)
            d = self.delta
            vf = self.vy + L_F * self.w
            long_f, lat_f = math.cos(d) * self.vx + math.sin(d) * vf, -math.sin(d) * self.vx + math.cos(d) * vf
            fyf = -self.cf * lat_f / (abs(long_f) + 1.)
            fyr = -self.cr * (self.vy - L_R * self.w) / (abs(self.vx) + 1.)
            dvy = (fyf * math.cos(d) + fyr) / MASS - self.vx * self.w
            dw = (L_F * fyf * math.cos(d) - L_R * fyr) / self.iz
            vy_rear = self.vy - L_R * self.w
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.rear += self.dt * np.array([self.vx * c - vy_rear * s, self.vx * s + vy_rear * c])
            self.yaw += self.dt * self.w
            self.vy += self.dt * dvy
            self.w += self.dt * dw
        return self

    @property
    def vy_rear(self):
        return self.vy - L_R * self.w


def replay(frames, plant):
    """Open-loop: logged applied steer and speed in, simulated yaw rate / rear lateral velocity out."""
    a = {k: np.array([np.nan if v is None else v for v in frames[k]], float) for k in frames}
    ok = np.isfinite(a['w']) & np.isfinite(a['vy']) & np.isfinite(a['applied']) & np.isfinite(a['vx'])
    idx = np.where(ok)[0]
    i0 = idx[0]
    plant.reset(np.zeros(2), 0., a['vx'][i0], a['vy'][i0], a['w'][i0], a['applied'][i0])
    w, vy = [plant.w], [plant.vy_rear]
    for i in range(i0 + 1, idx[-1] + 1):
        plant.step(a['applied'][i], a['vx'][i])
        w.append(plant.w)
        vy.append(plant.vy_rear)
    sl = slice(i0, idx[-1] + 1)
    return a['w'][sl], np.array(w), a['vy'][sl], np.array(vy)


def fit(logs):
    keys = [k for k in logs['frames'] if k.startswith('pose-g2-v1/')]
    best = []
    for c in (9., 10., 11., 12., 13., 14.07):
        for ratio in (.7, 1., 1.3):
            for lag in (0., .03, .06):
                for delay in (0, 1):
                    err_w = err_vy = n = 0.
                    for k in keys:
                        mw, sw, mv, sv = replay(logs['frames'][k], Plant(c, ratio, lag, delay))
                        m = np.isfinite(mw) & np.isfinite(mv)
                        err_w += np.sum((mw[m] - sw[m]) ** 2)
                        err_vy += np.sum((mv[m] - sv[m]) ** 2)
                        n += m.sum()
                    best.append((math.sqrt(err_w / n), math.sqrt(err_vy / n), c, ratio, lag, delay))
    best.sort()
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['fit'])
    p.add_argument('--logs', required=True)
    args = p.parse_args()
    logs = json.load(open(args.logs))
    rows = fit(logs)
    print('rmse_w_rps rmse_vy_rear_mps c inertia_ratio lag_s delay_ticks')
    for r in rows[:12]:
        print('%.4f %.4f %.2f %.1f %.2f %d' % r)
    for r in rows:
        if r[2] == 14.07 and r[3] == 1. and r[4] == 0. and r[5] == 0:
            print('physx-nominal c=14.07:', '%.4f %.4f' % r[:2])


if __name__ == '__main__':
    main()
