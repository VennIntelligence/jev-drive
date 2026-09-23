"""Predict the rear-axle lateral propagation coefficient k from PhysX 4 / CARLA 0.9.15 tyre semantics.

PxVehicleComputeTireForceDefault (PhysX 3.4 and 4.1 identical here):
    latStiff = restTireLoad * mLatStiffY * f1(normalisedLoad * 3 / mLatStiffX),  f1(K) = min(1, K - K^2/3 + K^3/27)
    latSlip  = atan(v_lat / (|v_long| + gMinLatSpeedForTireModel)),  gMinLatSpeed = 1 * tolerance length = 1 m/s
    F_lat    = latStiff * tan(latSlip) * (1 - K/3 + K^2/27),  K = latStiff |tan latSlip| / (mu * tireLoad)
CARLA maps mLatStiffX = lat_stiff_max_load (3), mLatStiffY = lat_stiff_value (20) per wheel.
Steady turn: F_yr = m a_y l_f / L and restTireLoad(rear axle) = m g l_f / L, so the CoM cancels:
    v_y,rear = -(1 + v0/v) v^2 w / (c g),  c = mLatStiffY * f1(3 / mLatStiffX),  v0 = 1 m/s.

    python physics_k.py --logs logs.json
"""
import argparse
import json
import math

import numpy as np

G, K_FROZEN = 9.81, .010659832
MASS, WHEELBASE, TRACK = 1696., 2.8604714913890885, 1.5929
L_R = .3 + 1.388633220199954  # CoM x (calibration-physics.json, taken as metres ahead of the actor origin)
L_F = WHEELBASE - L_R


def f1(k):
    return min(1., k - k * k / 3 + k ** 3 / 27)


def k_of(c, v, v0=1.):
    return (1 + v0 / v) / (c * G)


def axle_factor(ay, mu, height=.55, rear_roll_share=.5):
    """Rear-axle effective stiffness relative to the linear, no-transfer value at lateral accel ay:
    lateral load transfer through the concave f1 plus the friction smoothing 1 - K/3 + K^2/27."""
    dn = ay / G * 2 * height / TRACK * rear_roll_share * (MASS / (2 * MASS * L_F / WHEELBASE))
    alpha = ay / (20 * f1(1) * G)  # first guess, then fixed-point passes
    for _ in range(3):
        force = 0.
        for n in (1 + dn, 1 - dn):
            stiff = 20 * f1(n)
            kk = stiff * math.tan(alpha) / (mu * n)
            force += stiff * math.tan(alpha) * (1 - kk / 3 + kk * kk / 27) / 2
        alpha *= (ay / G) / force
    return (ay / G) / (20 * f1(1) * math.tan(alpha))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs', required=True)
    args = p.parse_args()
    logs = json.load(open(args.logs))
    c_nom = 20 * f1(3 / 3)
    filtered = .2308 + 1 * (2 - .2308) / 2  # UE4 default tyre-load filter, if active (unverified)
    print('f1(1) = %.4f, c_nominal = %.3f /rad; with UE default load filter c = %.3f /rad' % (f1(1), c_nom, 20 * f1(filtered)))
    print('CoM-free check: l_f=%.3f l_r=%.3f; k = m l_f/(L C_r) = 1/(c g) for any l_f' % (L_F, L_R))
    for c in (c_nom, 11.0):
        print('c=%.2f  k(6 m/s)=%.5f k(8 m/s)=%.5f  k(8, no +1 m/s)=%.5f' % (c, k_of(c, 6), k_of(c, 8), k_of(c, 8, 0)))
    print('frozen k=%.6f implies c=%.2f at 8 m/s, %.2f at 6 m/s' % (K_FROZEN, 1.125 / (K_FROZEN * G), (7 / 6) / (K_FROZEN * G)))
    for mu in (3.5, 2.45, 1.5):
        print('mu=%.2f  rear stiffness factor at a_y=1.5: %.3f, 5: %.3f, 8.3 m/s^2: %.3f' % (
            mu, axle_factor(1.5, mu), axle_factor(5, mu), axle_factor(8.3, mu)))
    rows = []
    for key, fr in logs['frames'].items():
        if not key.startswith('pose-g2-v1/'):
            continue
        route, arm, name = key.split('/')[1:]
        lo, hi = {'right-sharp': (24, 51), 'left': (18, 64), 'S1': (27.5, 48.5), 'S2': (73.5, 95)}[name]
        a = {k: np.array([np.nan if v is None else v for v in fr[k]], float) for k in ('s', 'vx', 'w', 'alpha_r')}
        m = (a['s'] >= lo) & (a['s'] <= hi) & np.isfinite(a['w'])
        v, w = a['vx'][m], a['w'][m]
        beta = lambda c: np.degrees(np.arctan(k_of(c, np.maximum(v, .5)) * v * w))
        meas = np.degrees(a['alpha_r'][m])
        rows.append((name, arm, meas.mean(), beta(c_nom).mean(), beta(11.).mean(),
                     np.degrees(np.arctan(K_FROZEN * v * w)).mean(),
                     np.percentile(np.abs(meas), 95), np.percentile(np.abs(beta(11.)), 95)))
    print('window arm | beta_r mean deg: measured, c=14.07, c=11, frozen k | |beta| P95 measured, c=11')
    for r in rows:
        print('%-11s %-18s %6.2f %6.2f %6.2f %6.2f | %5.2f %5.2f' % r)


if __name__ == '__main__':
    main()
