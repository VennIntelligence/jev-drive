"""Isolate polyline aim sampling on noiseless prescribed circular motion."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from b2d_controller import Controller


def hermite_aim(points, arc, station):
    lengths = np.diff(arc)
    if np.any(lengths <= 1e-8):
        return np.array([np.interp(station, arc, points[:, i]) for i in range(2)])
    slopes = np.diff(points, axis=0) / lengths[:, None]
    derivatives = np.vstack((slopes[0], (slopes[:-1] * lengths[1:, None] + slopes[1:] * lengths[:-1, None]) / (lengths[:-1] + lengths[1:])[:, None], slopes[-1]))
    i = min(max(0, int(np.searchsorted(arc, station, side='right')) - 1), len(lengths) - 1)
    h = lengths[i]
    u = np.clip((station - arc[i]) / h, 0., 1.)
    return ((2*u**3 - 3*u**2 + 1) * points[i] + (u**3 - 2*u**2 + u) * h * derivatives[i]
            + (-2*u**3 + 3*u**2) * points[i+1] + (u**3 - u**2) * h * derivatives[i+1])


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    rows, summaries = [], []
    for radius in (8., 20.):
        for speed in (6., 8.):
            for sign in (-1., 1.):
                for coefficient in (.5, .375):
                    ctl = Controller(preset='pursuit', lookahead='max', max_lookahead_time_s=coefficient,
                                     longitudinal_mode='pi', pi_kp=.5, pi_ki=.25)
                    w = sign * speed / radius
                    future = np.arange(1, 21)*.25
                    trajectory = np.column_stack((radius*np.sin(speed*future/radius), sign*radius*(1-np.cos(speed*future/radius))))
                    case = []
                    previous = None
                    for tick in range(200):
                        t = tick*.05
                        if tick % 4 == 0:
                            ctl.update(trajectory, t)
                        ctl.step(t, speed, w)
                        points, geometry = ctl._geometry(t-ctl._source_time)
                        station = min(geometry[0] + max(3., coefficient*speed), ctl._arc[-1])
                        aim = hermite_aim(points, ctl._arc, station)
                        kappa = 2*aim[1]/float(aim@aim)
                        scale = float(np.interp(speed*3.6, ctl.steering_curve[:, 0], ctl.steering_curve[:, 1]))
                        candidate = -math.atan(ctl.wheelbase*kappa)/(ctl.max_steer_rad*scale)
                        reference = -math.atan(ctl.wheelbase*sign/radius)/(ctl.max_steer_rad*scale)
                        linear = ctl.diagnostics['raw_steer']
                        row = dict(radius=radius,speed=speed,sign=sign,coefficient=coefficient,tick=tick,phase=tick%4,
                                   linear=linear,hermite=candidate,analytic=reference)
                        if previous is not None:
                            row.update(linear_rate=(linear-previous['linear'])/.05,hermite_rate=(candidate-previous['hermite'])/.05)
                        rows.append(row)
                        if tick>=20:case.append(row)
                        previous=row
                    summary={k:case[0][k] for k in ('radius','speed','sign','coefficient')}
                    for kind in ('linear','hermite'):
                        summary[kind+'_rate_abs_p95']=float(np.percentile([abs(r[kind+'_rate']) for r in case],95))
                        summary[kind+'_analytic_rms']=float(np.sqrt(np.mean([(r[kind]-r['analytic'])**2 for r in case])))
                    summaries.append(summary)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'frames.csv').open('x') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    sources=[Path(__file__),REPO/'scripts/b2d_controller.py']
    (out/'manifest.json').write_text(json.dumps(dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Prescribed exact circular motion, no plant feedback/no rejoin/no localization noise; diagnostic interpolation only, not CARLA improvement.'),indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
