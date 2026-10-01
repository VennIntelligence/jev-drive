"""Closed-loop screen of lateral-controller candidates on a fitted MKZ plant (no CARLA).

Runs the production Controller (pursuit, linear aim, max(3 m, .5 s*v)) and RouteAdapter (5 Hz rejoin) on
the plant from plant.py (c fitted on logs, open-loop replay RMSE: yaw rate .01 rad/s, rear lateral .016 m/s).
Candidate laws only re-map the controller's own aim point to a steer command; its rate/amplitude limits stay.
Localization is either truth or truth plus the logged path-normal pose error of a real arm, indexed by station,
so the screen can be validated against the logged CARLA CTE before it ranks anything.

    python closed_loop.py --src ../../../scripts --refs <dir of ref_<route>.json> --logs logs.json --out closed_loop.json
"""
import argparse
import json
import math
import sys

import numpy as np

from plant import Plant, G, WHEELBASE, TRACK, MAX_STEER, CURVE

# route: cruise m/s, sim start/stop station, [(window, start, end, core start, core end)] (turn-windows.csv)
ROUTES = {'26966': (8., 10., 70., [('right-sharp', 24., 51., 29., 46.)]),
          '24240': (8., 5., 75., [('left', 18., 64., 23., 59.)]),
          '17563': (6., 15., 105., [('S1', 27.5, 48.5, 32.5, 43.5), ('S2', 73.5, 95., 78.5, 90.)])}


class Polyline:
    def __init__(self, xy):
        xy = np.asarray(xy, float)
        self.xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
        self.d = np.diff(self.xy, axis=0)
        self.len = np.linalg.norm(self.d, axis=1)
        self.s = np.r_[0., np.cumsum(self.len)]

    def project(self, p):
        u = np.clip(np.sum((p - self.xy[:-1]) * self.d, axis=1) / self.len ** 2, 0, 1)
        foot = self.xy[:-1] + u[:, None] * self.d
        i = int(np.argmin(np.sum((foot - p) ** 2, axis=1)))
        delta = p - foot[i]
        cte = (self.d[i, 1] * delta[0] - self.d[i, 0] * delta[1]) / self.len[i]
        return self.s[i] + u[i] * self.len[i], cte

    def at(self, s):
        s = np.clip(s, 0, self.s[-1])
        return np.array([np.interp(s, self.s, self.xy[:, j]) for j in (0, 1)])

    def heading(self, s, h=2.5):
        d = self.at(s + h) - self.at(s - h)
        return math.atan2(d[1], d[0])


def k_of_v(v, c):
    """Rear-axle lateral propagation coefficient implied by the PhysX tyre model: (1 + 1/v) / (c g)."""
    return (1. + 1. / max(abs(v), .5)) / (c * G)


def steer_from_curvature(kappa, speed, ackermann):
    """Invert curvature to CARLA steer. Production: nominal angle = atan(L kappa). Ackermann-aware: the
    commanded nominal angle is PhysX's inner-wheel angle, so ask for cot(inner) = cot(centre) - w/(2L)."""
    a = abs(kappa) * WHEELBASE
    if ackermann and a > 1e-9:
        angle = math.atan(1. / max(1. / a - TRACK / (2 * WHEELBASE), 1e-3))
    else:
        angle = math.atan(a)
    scale = float(np.interp(speed * 3.6, CURVE[:, 0], CURVE[:, 1]))
    return -math.copysign(angle, kappa) / (MAX_STEER * scale)  # kappa left positive -> CARLA right positive


def run(mods, ref_xy, loc_profile, c, route='26966'):
    import b2d_controller as bc
    from b2d_controller_adapter import RouteAdapter, world_to_local
    seed_speed, s0, s1, _ = ROUTES[route]
    ref = Polyline(ref_xy)
    config = dict(wheelbase=WHEELBASE, max_steer_deg=math.degrees(MAX_STEER), steering_curve=CURVE.tolist(),
                  longitudinal_mode='pi', lookahead='max', pi_kp=.5, pi_ki=.25,
                  max_lookahead_time_s=mods.get('lookahead_time', .5), aim_interpolation='linear')
    original_advance = bc.advance_pose
    if mods.get('odometry_slip'):
        def advance(pose, speed, yaw_rate, dt):  # controller odometry with the same rear-slip term
            out = original_advance(pose, speed, yaw_rate, dt)
            lateral = -k_of_v(speed, c) * speed * speed * yaw_rate * dt  # left positive
            mid = pose[2] + yaw_rate * dt / 2
            out[:2] += lateral * np.array([-math.sin(mid), math.cos(mid)])
            return out
        bc.advance_pose = advance
    try:
        ctrl = bc.Controller('pursuit', **config)
        adapter = RouteAdapter(ref_xy, seed_speed)
        if mods.get('route_direct'):
            def direct(xy, yaw, self=adapter):  # track the route itself from the projected station
                arc = self.progress + np.arange(1, 21) * .25 * self.cruise
                return world_to_local(np.array([ref.at(s) for s in arc]), xy, yaw)
            adapter.trajectory = direct
        plant = Plant(c=c)
        start = ref.at(s0)
        # CARLA world is y-south / yaw right-positive; the plant uses the same convention.
        plant.reset(start, ref.heading(s0), seed_speed, 0., 0.)
        rows, steer, t, tick = [], 0., 0., 0
        while True:
            s_true, cte = ref.project(plant.rear)
            if s_true >= s1 or t > 30:
                break
            n_left = np.array([math.sin(ref.heading(s_true)), -math.cos(ref.heading(s_true))])
            bias = float(np.interp(s_true, *loc_profile, left=0., right=0.)) if loc_profile is not None else 0.
            pose_xy, pose_yaw = plant.rear - bias * n_left, plant.yaw  # loc = truth CTE - estimated CTE
            speed, yaw_rate_left = plant.vx, -plant.w
            adapter.project(pose_xy, pose_yaw, speed, t)
            if tick % 4 == 0:
                ctrl.update(adapter.trajectory(pose_xy, pose_yaw), t)
            _, raw_command, _ = ctrl.step(t, speed, yaw_rate_left)
            diag = ctrl.diagnostics
            new = raw_command
            if (mods.get('ackermann') or mods.get('slip_frame')) and diag.get('aim_xy') is not None:
                aim = np.array(diag['aim_xy'])
                if mods.get('slip_frame'):  # pursue along the rear-axle velocity, not the body heading
                    gamma = math.atan(-k_of_v(speed, c) * speed * yaw_rate_left)  # left positive slip direction
                    cg, sg = math.cos(gamma), math.sin(gamma)
                    aim = np.array([cg * aim[0] + sg * aim[1], -sg * aim[0] + cg * aim[1]])
                kappa = 2 * aim[1] / max(float(aim @ aim), 1e-8)
                raw = steer_from_curvature(kappa, speed, mods.get('ackermann', False))
                new = float(np.clip(raw, steer - ctrl.steer_rate * .05, steer + ctrl.steer_rate * .05))
                new = float(np.clip(new, -ctrl.max_steer, ctrl.max_steer))
                ctrl._last_steer = new
            steer = new
            vy = plant.vy_rear
            heading_ref = ref.heading(s_true)
            body_err = math.degrees((plant.yaw - heading_ref + math.pi) % (2 * math.pi) - math.pi)
            course_err = body_err + math.degrees(math.atan2(vy, plant.vx))
            rows.append((s_true, cte, body_err, course_err, steer, plant.w, bias))
            plant.step(steer, seed_speed)
            t += .05
            tick += 1
    finally:
        bc.advance_pose = original_advance
    return np.array(rows)


def metrics(rows, lo, hi):
    m = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
    r = rows[m]
    rate = np.diff(rows[:, 4]) / .05
    return dict(n=int(m.sum()), cte_rms=float(np.sqrt(np.mean(r[:, 1] ** 2))),
                cte_p95=float(np.percentile(np.abs(r[:, 1]), 95)), cte_mean=float(np.mean(r[:, 1])),
                body_p95=float(np.percentile(np.abs(r[:, 2]), 95)), course_p95=float(np.percentile(np.abs(r[:, 3]), 95)),
                steer_max=float(np.max(np.abs(r[:, 4]))),
                rate_p95=float(np.percentile(np.abs(rate[m[1:]]), 95)))


VARIANTS = {'production': {}, 'ackermann': dict(ackermann=True), 'slip_frame': dict(slip_frame=True),
            'ackermann+slip': dict(ackermann=True, slip_frame=True), 'odometry_slip': dict(odometry_slip=True),
            'route_direct': dict(route_direct=True), 'short_lookahead': dict(lookahead_time=.375),
            'ackermann+slip+odometry': dict(ackermann=True, slip_frame=True, odometry_slip=True),
            'all+route_direct': dict(ackermann=True, slip_frame=True, odometry_slip=True, route_direct=True)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', required=True, help='directory holding b2d_controller.py and b2d_controller_adapter.py')
    p.add_argument('--refs', required=True, help='directory holding ref_<route>.json (route_reference.json copies)')
    p.add_argument('--logs', required=True, help='output of analyze_logs.py')
    p.add_argument('--c', type=float, default=11.)
    p.add_argument('--out')
    args = p.parse_args()
    sys.path.insert(0, args.src)
    logs = json.load(open(args.logs))
    arms = ('baseline-zero', 'candidate-fixed-k')
    result = dict(c=args.c, logged={}, runs={})
    for route, (_, _, _, windows) in ROUTES.items():
        ref_xy = np.asarray(json.load(open('%s/ref_%s.json' % (args.refs, route)))['world_xy'], float)
        profiles = {'truth': None}
        for arm in arms:
            s, loc = [], []
            for name, *_ in windows:
                f = logs['frames']['pose-g2-v1/%s/%s/%s' % (route, arm, name)]
                s += f['s']
                loc += f['loc']
                result['logged']['%s|%s|%s' % (route, name, arm)] = logs['cases']['pose-g2-v1/%s/%s' % (route, arm)][name]['window']
            s, loc = np.array(s, float), np.array(loc, float)
            o = np.argsort(s)
            profiles['loc:' + arm] = (s[o], loc[o])
        for pose, profile in profiles.items():
            for variant, mods in VARIANTS.items():
                if pose != 'truth' and variant not in ('production', 'ackermann', 'slip_frame', 'ackermann+slip',
                                                       'ackermann+slip+odometry', 'all+route_direct'):
                    continue
                rows = run(mods, ref_xy, profile, args.c, route)
                for name, w0, w1, c0, c1 in windows:
                    result['runs']['%s|%s|%s|%s' % (route, name, pose, variant)] = dict(
                        window=metrics(rows, w0, w1), core=metrics(rows, c0, c1))
    for key, r in result['runs'].items():
        w, c = r['window'], r['core']
        print('%-58s win CTE rms %.3f p95 %.3f mean %+.3f | body P95 %.2f course P95 %.2f | core rms %.3f | steer max %.3f rate P95 %.2f'
              % (key, w['cte_rms'], w['cte_p95'], w['cte_mean'], w['body_p95'], w['course_p95'], c['cte_rms'], w['steer_max'], w['rate_p95']))
    for key, w in result['logged'].items():
        print('logged %-40s CTE rms %.3f p95 %.3f' % (key, w['truth_cte']['rms'], w['truth_cte']['p95_abs']))
    if args.out:
        json.dump(result, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
