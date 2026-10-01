"""Which vehicle point is Bench2Drive's anno['x','y'] (the `pos` TCP labels are built around)?

H1: GNSS sensor point (actor x=-1.4 m, lat/lon noise sigma 5e-6 deg ~ 0.56 m).
H2: CARLA actor location (no noise).

Per frame: d = R(-yaw) (pos_xy - actor_xy) in the CARLA body frame (x forward, y right).
Also rebuilds TCP waypoint labels exactly as Zoo/TCP/data.py:161-175 and compares them with the
actor's true future displacement, to check the lateral sign and to estimate the origin offset from turns:
label - gt_actor = d_fwd * (cos dpsi - 1, sin dpsi) + noise.

Run on the box:  ssh autodl '~/data/envs/jevdrive/bin/python - <root>' < check_pos_origin.py
"""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '~/data/datasets/bench2drive-mini').expanduser()
STEPS = (5, 10, 15, 20)  # gen_tcp_data.py: frames i+5..i+20 at 10 Hz


def load_clip(clip: Path):
    files = sorted((clip / 'anno').glob('*.json.gz'))[:-1]  # gen_tcp_data drops the last frame
    rows = []
    for f in files:
        a = json.load(gzip.open(f, 'rt'))
        ego = next(b for b in a['bounding_boxes'] if b['class'] == 'ego_vehicle')
        rows.append((a['x'], a['y'], a['theta'], ego['location'][0], ego['location'][1],
                     np.deg2rad(ego['rotation'][2]), ego['speed']))
    return clip.name, np.array(rows, dtype=np.float64)


def rot(yaw):  # world -> CARLA body (x fwd, y right); same matrix as data.py with theta = compass - pi/2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def analyse(name, r):
    px, py, th, ax, ay, yaw, v = r.T
    d = np.einsum('nij,nj->ni', rot(yaw), np.stack([px - ax, py - ay], -1))
    # Same fit as the paired-v2 log check: pos - actor = c + d_f h(yaw) + d_r r(yaw)
    h, rt = np.stack([np.cos(yaw), np.sin(yaw)], -1), np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    A = np.zeros((2 * len(r), 4))
    A[0::2, 0], A[1::2, 1] = 1, 1
    A[0::2, 2], A[1::2, 2], A[0::2, 3], A[1::2, 3] = h[:, 0], h[:, 1], rt[:, 0], rt[:, 1]
    b = np.stack([px - ax, py - ay], -1).reshape(-1)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    lag1 = np.corrcoef(d[1:, 0], d[:-1, 0])[0, 1]
    compass_err = np.rad2deg(np.abs(wrap(th - np.pi / 2 - yaw))).max()

    # TCP labels (data.py) vs actor ground truth, all anchors gen_tcp_data keeps
    n = len(r)
    idx = np.arange(0, n - 20 - 5)
    R0 = rot(th[idx] - np.pi / 2)
    lab, gta, dpsi = [], [], []
    for k in STEPS:
        lab.append(np.einsum('nij,nj->ni', R0, np.stack([px[idx + k] - px[idx], py[idx + k] - py[idx]], -1)))
        gta.append(np.einsum('nij,nj->ni', rot(yaw[idx]), np.stack([ax[idx + k] - ax[idx], ay[idx + k] - ay[idx]], -1)))
        dpsi.append(wrap(yaw[idx + k] - yaw[idx]))
    lab, gta, dpsi = np.stack(lab, 1), np.stack(gta, 1), np.stack(dpsi, 1)  # (N,4,2), (N,4)
    return dict(name=name, n=n, d=d, coef=coef, lag1=lag1, compass_err=compass_err, yaw_span=np.rad2deg(np.ptp(yaw)),
                lab=lab, gta=gta, dpsi=dpsi, moving=v[idx] > 1.0)


def main():
    clips = sorted(p for p in ROOT.iterdir() if (p / 'anno').is_dir())
    with ProcessPoolExecutor(len(clips)) as ex:
        res = [analyse(*c) for c in ex.map(load_clip, clips)]

    print('== pos - actor in body frame (x fwd, y right) ==')
    for s in res:
        d = s['d']
        print(f"{s['name']}: n={s['n']} mean=({d[:, 0].mean():+.3f},{d[:, 1].mean():+.3f}) "
              f"std=({d[:, 0].std():.3f},{d[:, 1].std():.3f}) lag1_corr={s['lag1']:+.2f} yaw_span={s['yaw_span']:.0f}deg "
              f"fit c=({s['coef'][0]:+.2f},{s['coef'][1]:+.2f}) d_fwd={s['coef'][2]:+.3f} d_right={s['coef'][3]:+.3f} "
              f"max|compass-pi/2-yaw|={s['compass_err']:.4f}deg")
    d = np.concatenate([s['d'] for s in res])
    print(f"ALL: n={len(d)} mean=({d[:, 0].mean():+.3f},{d[:, 1].mean():+.3f}) std=({d[:, 0].std():.3f},{d[:, 1].std():.3f}) "
          f"sem=({d[:, 0].std() / np.sqrt(len(d)):.3f},{d[:, 1].std() / np.sqrt(len(d)):.3f})")

    print('== TCP label vs actor future displacement (moving anchors) ==')
    lab = np.concatenate([s['lab'][s['moving']] for s in res])
    gta = np.concatenate([s['gta'][s['moving']] for s in res])
    dpsi = np.concatenate([s['dpsi'][s['moving']] for s in res])
    for k, t in enumerate((0.5, 1.0, 1.5, 2.0)):
        corr = np.corrcoef(lab[:, k, 1], gta[:, k, 1])[0, 1]
        sl = np.polyfit(gta[:, k, 1], lab[:, k, 1], 1)[0]
        print(f"+{t}s: n={len(lab)} corr(label_y, gt_right)={corr:+.3f} slope={sl:+.3f} gt_right range=[{gta[:, k, 1].min():+.1f},{gta[:, k, 1].max():+.1f}]")
    # Origin from turns: lateral residual = d_fwd * sin(dpsi) + noise; longitudinal = d_fwd * (cos dpsi - 1)
    res_lat, s_ = (lab[..., 1] - gta[..., 1]).ravel(), np.sin(dpsi).ravel()
    turn = np.abs(s_) > np.sin(np.deg2rad(10))
    X = np.stack([s_, np.ones_like(s_)], -1)
    coef, *_ = np.linalg.lstsq(X, res_lat, rcond=None)
    resid = res_lat - X @ coef
    se = np.sqrt(resid.var() / ((s_ - s_.mean()) ** 2).sum())
    print(f"turn fit: lateral residual = d*sin(dpsi) + c -> d={coef[0]:+.3f}+-{se:.3f} m, c={coef[1]:+.3f}, "
          f"n={len(s_)} ({turn.sum()} with |dpsi|>10deg, max |dpsi|={np.rad2deg(np.abs(dpsi).max()):.0f}deg)")
    # Direct turn check: in a right turn (dpsi>0 in CARLA) the path bends right, gt_right>0; label_y must follow
    tr = turn.reshape(dpsi.shape)[:, 3]
    print(f"turn anchors @2s: n={tr.sum()} sign agreement label_y vs gt_right = "
          f"{(np.sign(lab[tr, 3, 1]) == np.sign(gta[tr, 3, 1])).mean():.3f}, sign(dpsi)==sign(gt_right): "
          f"{(np.sign(dpsi[tr, 3]) == np.sign(gta[tr, 3, 1])).mean():.3f}")


if __name__ == '__main__':
    main()
