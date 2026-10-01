"""Lateral error decomposition and rear-axle slip measurements from closed-loop controller logs.

Python 3.8 + NumPy only (runs with /data/envs/carla/bin/python on the Tokyo box).
Reads control.jsonl / validation_trace.json / route_reference.json of each case and prints one JSON
document to stdout: per-window statistics plus per-frame arrays for the four fixed windows.

Conventions: CARLA world (x east, y south), yaw and yaw rate right-positive, CTE left-positive.
Truth is used for scoring only; nothing here feeds back into any controller.

    ssh ujs@100.108.238.8 /data/envs/carla/bin/python - < analyze_logs.py > logs.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RUNS = [('pose-g2-v1', Path('/data/runs/b2d/controller/lateral-followup/pose-g2-v1'),
         ('baseline-zero', 'candidate-fixed-k')),
        ('turns-v1', Path('/data/runs/b2d/controller/turns-v1'), ('baseline-max', 'short-max'))]
# route, name, core start/end, window start/end (m), frozen in lateral-evidence-v1/turn-windows.csv
WINDOWS = [('26966', 'right-sharp', 29., 46., 24., 51.), ('24240', 'left', 23., 59., 18., 64.),
           ('17563', 'S1', 32.5, 43.5, 27.5, 48.5), ('17563', 'S2', 78.5, 90., 73.5, 95.)]
WHEELBASE, TRACK, MAX_STEER = 2.8604714913890885, 1.5929, math.radians(69.99999237060547)
CURVE = np.array([[0., 1.], [20., .9], [60., .8], [120., .7]])
STEER_LIMIT, STEER_RATE, DT = .8, 2., .05


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class Polyline:
    def __init__(self, xy):
        xy = np.asarray(xy, float)
        self.xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
        self.d = np.diff(self.xy, axis=0)
        self.len = np.linalg.norm(self.d, axis=1)
        self.s = np.r_[0., np.cumsum(self.len)]

    def project(self, p):
        """Global nearest segment, identical to the frozen analyzer. Returns station, left-positive CTE."""
        p = np.asarray(p, float)[:, None, :]
        u = np.clip(np.sum((p - self.xy[:-1]) * self.d, axis=2) / self.len ** 2, 0, 1)
        foot = self.xy[:-1] + u[..., None] * self.d
        i = np.argmin(np.sum((foot - p) ** 2, axis=2), axis=1)
        k = np.arange(len(i))
        delta = p[:, 0] - foot[k, i]
        cte = (self.d[i, 1] * delta[:, 0] - self.d[i, 0] * delta[:, 1]) / self.len[i]
        return self.s[i] + u[k, i] * self.len[i], cte

    def at(self, s):
        s = np.clip(s, 0, self.s[-1])
        return np.column_stack([np.interp(s, self.s, self.xy[:, j]) for j in (0, 1)])

    def curvature(self, s, h=2.5):
        """Signed circumcircle curvature through s-h, s, s+h; right turn positive (CARLA yaw)."""
        a, b, c = self.at(s - h), self.at(s), self.at(s + h)
        u, v = b - a, c - b
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * np.linalg.norm(c - a, axis=1)
        return 2 * cross / np.maximum(den, 1e-12)


def load_case(directory):
    rec = [json.loads(l) for l in (directory / 'control.jsonl').read_text().splitlines() if l]
    rec = [r for r in rec if r.get('truth_xy') is not None and r.get('pose_xy') is not None]
    trace = {r['frame']: r for r in json.loads((directory / 'validation_trace.json').read_text())}
    ref = Polyline(json.loads((directory / 'route_reference.json').read_text())['world_xy'])
    g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rec], float)
    f = dict(frame=g('frame'), t=g('sim_time'), truth=g('truth_xy'), yaw=g('truth_yaw'), pose=g('pose_xy'),
             pose_yaw=g('pose_yaw'), steer=g('steer'), raw=g('raw_steer'), speed=g('speed_mps'),
             gyro_left=g('yaw_rate_rps'), own_cte=g('cross_track_m'), look=g('lookahead_m'),
             aim=np.array([r.get('aim_xy') or [np.nan, np.nan] for r in rec], float),
             traj=g('trajectory_frame'))
    f['applied'] = np.array([trace.get(int(fr), {}).get('applied_control', {}).get('steer', np.nan)
                             for fr in f['frame']])
    f['k_lat'] = np.array([r['pose_status'].get('lateral_coefficient_s2_per_m', 0.) for r in rec])
    return f, ref


def kinematics(f):
    t, xy, yaw = f['t'], f['truth'], f['yaw']
    n = len(t)
    vel = np.full((n, 2), np.nan)
    vel[1:-1] = (xy[2:] - xy[:-2]) / (t[2:] - t[:-2])[:, None]
    w = np.full(n, np.nan)
    w[1:-1] = wrap(yaw[2:] - yaw[:-2]) / (t[2:] - t[:-2])
    fwd = np.column_stack((np.cos(yaw), np.sin(yaw)))
    right = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    vx, vy = np.sum(vel * fwd, axis=1), np.sum(vel * right, axis=1)  # rear-axle velocity, body frame
    return vx, vy, w


def steer_to_angles(steer, speed):
    """CARLA/PhysX: nominal angle = steer * max_steer * curve(speed); Ackermann (accuracy 1) makes
    it the inner-wheel angle. Returns nominal and bicycle-centre angles, sign of steer (right +)."""
    scale = np.interp(np.abs(speed) * 3.6, CURVE[:, 0], CURVE[:, 1])
    nominal = steer * MAX_STEER * scale
    a = np.abs(nominal)
    with np.errstate(divide='ignore'):
        centre = np.where(a > 1e-6, np.arctan(1 / (1 / np.tan(np.maximum(a, 1e-6)) + TRACK / (2 * WHEELBASE))), 0.)
    return nominal, np.sign(nominal) * centre


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(n=int(len(x)), rms=float(np.sqrt(np.mean(x ** 2))), mean=float(np.mean(x)),
                p95_abs=float(np.percentile(np.abs(x), 95)), max_abs=float(np.max(np.abs(x))))


def analyse(f, ref):
    s, cte = ref.project(f['truth'])
    s_est, cte_est = ref.project(f['pose'])
    loc = cte - cte_est  # localization error on the path normal (truth left of estimate positive)
    vx, vy, w = kinematics(f)
    kappa_ref = ref.curvature(s)
    # Aim point in world from the estimated pose; its route CTE measures how much of the vehicle's own
    # route offset the 5 Hz rejoin path still carries at the lookahead.
    c, sn = np.cos(f['pose_yaw']), np.sin(f['pose_yaw'])
    ax, ay = f['aim'][:, 0], -f['aim'][:, 1]  # local x forward / y left -> world (right-handed flip)
    aim_world = f['pose'] + np.column_stack((ax * c - ay * sn, ax * sn + ay * c))
    _, aim_cte = ref.project(aim_world)
    d2 = np.sum(f['aim'] ** 2, axis=1)
    kappa_cmd = -2 * f['aim'][:, 1] / np.maximum(d2, 1e-9)  # right positive, from pure-pursuit geometry
    nominal, centre = steer_to_angles(f['applied'], vx)
    kappa_nom = np.tan(nominal) / WHEELBASE  # what the controller's steer map assumes
    kappa_ack = np.tan(centre) / WHEELBASE  # Ackermann-centre kinematic curvature
    kappa_act = w / np.maximum(vx, .5)  # rear-axle body yaw curvature
    speed = np.maximum(vx, .5)
    # Slip angles in the force sense (positive = force toward the turn centre of a right turn).
    # alpha_r also equals body yaw minus rear-axle course, i.e. the audited body-course offset.
    alpha_r = -np.arctan2(vy, speed)
    alpha_r_physx = -np.arctan(vy / (speed + 1.))  # PhysX: atan(v_lat / (|v_long| + 1 m/s))
    alpha_f = centre - np.arctan2(vy + WHEELBASE * w, speed)
    rate = np.r_[np.nan, np.diff(f['steer']) / np.diff(f['t'])]
    slew = np.r_[False, np.abs(np.diff(f['steer'])) >= STEER_RATE * np.diff(f['t']) - 1e-6]
    limited = np.abs(f['raw'] - f['steer']) > 1e-8
    updated = np.r_[True, np.diff(f['traj']) != 0]
    return dict(s=s, cte=cte, cte_est=cte_est, loc=loc, own=f['own_cte'], aim_cte=aim_cte, vx=vx, vy=vy,
                w=w, ay=vx * w, kappa_ref=kappa_ref, kappa_cmd=kappa_cmd, kappa_nom=kappa_nom,
                kappa_ack=kappa_ack, kappa_act=kappa_act, alpha_r=alpha_r, alpha_r_physx=alpha_r_physx,
                alpha_f=alpha_f, steer=f['steer'], raw=f['raw'], applied=f['applied'], rate=rate,
                slew=slew, limited=limited, saturated=np.abs(f['steer']) >= STEER_LIMIT - 1e-6,
                look=f['look'], updated=updated, speed=f['speed'], frame=f['frame'], t=f['t'],
                k_lat=f['k_lat'])


def summarise(a, lo, hi):
    m = (a['s'] >= lo) & (a['s'] <= hi)
    corr = float(np.corrcoef(a['cte_est'][m], a['loc'][m])[0, 1]) if m.sum() > 2 else None
    turning = m & np.isfinite(a['w']) & (np.abs(a['w']) > .05)
    x = a['vx'] ** 2 * a['w']
    kfit = float(-np.sum(x[turning] * a['vy'][turning]) / np.sum(x[turning] ** 2)) if turning.sum() > 2 else None
    ratio = lambda num, den: float(np.sum(num[turning] * den[turning]) / np.sum(den[turning] ** 2)) if turning.sum() > 2 else None
    return dict(
        n=int(m.sum()), truth_cte=stats(a['cte'][m]), est_cte=stats(a['cte_est'][m]), loc_lat=stats(a['loc'][m]),
        own_path_cte=stats(a['own'][m]), aim_route_cte=stats(a['aim_cte'][m]), corr_est_loc=corr,
        cov_share=dict(var_truth=float(np.var(a['cte'][m])), var_est=float(np.var(a['cte_est'][m])),
                       var_loc=float(np.var(a['loc'][m])),
                       two_cov=float(2 * np.cov(a['cte_est'][m], a['loc'][m], bias=True)[0, 1])),
        aim_retained_ratio=float(np.nansum(a['aim_cte'][m] * a['cte_est'][m]) / np.nansum(a['cte_est'][m] ** 2)),
        saturated_fraction=float(np.mean(a['saturated'][m])), slew_fraction=float(np.mean(a['slew'][m])),
        limited_fraction=float(np.mean(a['limited'][m])), steer_abs_max=float(np.max(np.abs(a['steer'][m]))),
        steer_rate=stats(a['rate'][m]), lookahead_mean=float(np.mean(a['look'][m])),
        kappa_ref_abs_max=float(np.nanmax(np.abs(a['kappa_ref'][m]))),
        kappa_ref_abs_mean=float(np.nanmean(np.abs(a['kappa_ref'][m]))),
        k_fit_truth=kfit, ay_abs_mean=float(np.nanmean(np.abs(a['ay'][turning]))) if turning.any() else None,
        ay_abs_p95=float(np.nanpercentile(np.abs(a['ay'][turning]), 95)) if turning.any() else None,
        beta_r_deg=stats(np.degrees(a['alpha_r'][m])),  # body minus course, right turn positive
        kappa_act_over_nom=ratio(a['kappa_act'], a['kappa_nom']),
        kappa_act_over_ack=ratio(a['kappa_act'], a['kappa_ack']),
        kappa_act_over_ref=ratio(a['kappa_act'], a['kappa_ref']),
        kappa_cmd_over_ref=ratio(a['kappa_cmd'], a['kappa_ref']),
        alpha_f_minus_r_deg=stats(np.degrees(a['alpha_f'][turning] - a['alpha_r'][turning])) if turning.any() else None,
        update_fraction=float(np.mean(a['updated'][m])))


def main():
    out = dict(cases={}, frames={})
    for run, root, variants in RUNS:
        for route in ('26966', '24240', '17563'):
            for variant in variants:
                d = root / route / variant / 'pursuit'
                if not (d / 'control.jsonl').is_file():
                    continue
                a = analyse(*load_case(d))
                key = '%s/%s/%s' % (run, route, variant)
                out['cases'][key] = {}
                for r, name, c0, c1, w0, w1 in WINDOWS:
                    if r != route:
                        continue
                    out['cases'][key][name] = dict(window=summarise(a, w0, w1), core=summarise(a, c0, c1))
                    m = (a['s'] >= w0 - 5) & (a['s'] <= w1 + 10)
                    out['frames']['%s/%s' % (key, name)] = {
                        k: [None if not np.isfinite(v) else float(v) for v in np.asarray(a[k][m], float)]
                        for k in ('frame', 't', 's', 'cte', 'cte_est', 'loc', 'own', 'aim_cte', 'vx', 'vy', 'w',
                                  'ay', 'kappa_ref', 'kappa_cmd', 'kappa_nom', 'kappa_ack', 'kappa_act',
                                  'alpha_r', 'alpha_r_physx', 'alpha_f', 'steer', 'raw', 'applied', 'rate',
                                  'look', 'updated', 'k_lat')}
    json.dump(out, sys.stdout, allow_nan=False)


if __name__ == '__main__':
    main()
