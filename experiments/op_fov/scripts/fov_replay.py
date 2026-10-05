#!/usr/bin/env python
"""Wider field of view, zero-shot: shipped Cinque on real comma1M turns with the model frames cut at smaller focals.

comma1M is the only local real data from openpilot's own rig: the comma 3/3X wide camera sees ~119 deg (pinhole focal 567 px
at 1928 px), and openpilot crops it to 58.7 deg (model-frame focal 455). Here the same perspective warp (modeld's
get_warp_matrix + nearest-neighbour gather, jevdrive/openpilot/frames.py) is run with a smaller model-frame focal, so the
512x256 frame holds a wider scene; principal point (and so the horizon row) is unchanged. Arms (plan: ../plans/):

  N       road 910 / wide 455   (shipped: 31.4 / 58.7 deg)
  W90     wide 256              (90 deg)
  W116    wide 160              (116 deg, the ecam's full width)
  R40     road 720 (39.2 deg), rows the road camera does not cover (bottom ~12%) filled from the wide camera

  scan <out>                    turn + straight events over every comma1M segment -> <out>/events.json
  run  --windows F [--arms ..]  replay each window x arm (pool job; one CUDA session per worker) -> <cache>/<win>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ROOT = Path.home() / "data/datasets/comma1M"
WARM = 120                        # 6 s warm-up (> the 5 s context) before the scored part
PRE, POST = 80, 200               # scored part: 4 s before turn onset .. 10 s after
STRAIGHT_SCORE = 200
F_ROAD, F_WIDE = 2648.0, 567.0    # comma 3/3X AR0231 / OX03C10 at 1928x1208 (frames.CAMERAS)
ARMS = {"N": (910.0, 455.0), "W90": (910.0, 256.0), "W116": (910.0, 160.0), "R40": (720.0, 455.0)}
CACHE_VERSION = "fov-v1"


def hfov(f):
    return float(np.degrees(2 * np.arctan(256.0 / f)))


# ---------------------------------------------------------------- geometry
def model_K(f, wide):
    from jevdrive.openpilot.frames import MEDMODEL_CY, MODEL_H, MODEL_W
    cy = 0.5 * (MODEL_H + MEDMODEL_CY) if wide else MEDMODEL_CY
    return np.array([[f, 0, MODEL_W / 2], [0, f, cy], [0, 0, 1.0]])


def warp_matrix(rpy, cam_f, f_model, wide, wh=(1928, 1208)):
    """camera pixel <- model pixel, modeld's get_warp_matrix with the model-frame focal as a parameter."""
    from jevdrive.openpilot.frames import VIEW_FROM_DEVICE, intrinsics, rot_from_euler
    return intrinsics(*wh, cam_f) @ VIEW_FROM_DEVICE @ rot_from_euler(rpy) @ np.linalg.inv(model_K(f_model, wide) @ VIEW_FROM_DEVICE)


def _proj(M, w, h):
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    M = M.astype(np.float32)
    sx, sy, sw = (M[i, 0] * x + M[i, 1] * y + M[i, 2] for i in range(3))
    return sx / sw, sy / sw


def gather_index(rpy, f_model, wide, fill=False, wh=(1928, 1208)):
    """Flat gather indices into concat(road plane, wide plane) for the Y plane (256x512) and the UV planes (128x256), and the
    share of model pixels that no camera sees (clamped to the image edge, as modeld's warp does). fill: road-frame pixels the
    road camera does not see come from the wide camera (R40 arm); otherwise the road frame is the road camera alone."""
    from jevdrive.openpilot.frames import MODEL_H, MODEL_W
    uv_scale = np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]])
    srcs = [(F_WIDE, 1)] if wide else [(F_ROAD, 0), (F_WIDE, 1)] if fill else [(F_ROAD, 0)]
    out, miss = [], 0.0
    for sub, (W, H) in ((1, (MODEL_W, MODEL_H)), (2, (MODEL_W // 2, MODEL_H // 2))):
        cw, ch = wh[0] // sub, wh[1] // sub
        S = uv_scale if sub == 2 else np.ones((3, 3))
        idx = np.full(H * W, -1, np.int64)
        for j, (cf, cam) in enumerate(srcs):
            sx, sy = _proj(warp_matrix(rpy, cf, f_model, wide, wh) * S, W, H)
            xi, yi = np.rint(sx).ravel(), np.rint(sy).ravel()
            inside = (xi >= 0) & (xi <= cw - 1) & (yi >= 0) & (yi <= ch - 1)
            last = j == len(srcs) - 1
            take = (idx < 0) & (inside | last)
            if last and sub == 1:
                miss = float((take & ~inside).mean())
            xi, yi = np.clip(xi, 0, cw - 1).astype(np.int64), np.clip(yi, 0, ch - 1).astype(np.int64)
            idx[take] = cam * cw * ch + yi[take] * cw + xi[take]
        out.append(idx)
    return out[0], out[1], miss


class ArmWarper:
    """Both model frames of one arm from one decoded (road, wide) frame pair: (2, 6, 128, 256) uint8, [road, wide]."""

    def __init__(self, rpy, f_road, f_wide):
        self.road = gather_index(rpy, f_road, False, fill=f_road < 900)
        self.wide = gather_index(rpy, f_wide, True)
        self.miss = dict(road=self.road[2], wide=self.wide[2])

    @staticmethod
    def _one(planes, g, out):
        Y = planes[0][g[0]].reshape(256, 512)
        out[0], out[1], out[2], out[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
        out[4] = planes[1][g[1]].reshape(128, 256)
        out[5] = planes[2][g[1]].reshape(128, 256)

    def __call__(self, road, wide, out):
        planes = [np.concatenate([a.ravel(), b.ravel()]) for a, b in zip(road, wide)]
        self._one(planes, self.road, out[0])
        self._one(planes, self.wide, out[1])
        return out


# ---------------------------------------------------------------- events
def motion(meta):
    from jevdrive.openpilot.frames import rot_from_euler
    om = meta["omega_dev"] @ rot_from_euler(meta["rpy_calib"])        # (calib_from_device @ omega)^T
    return np.linalg.norm(meta["vel"], axis=1), om[:, 2]                # yaw rate, calib frame z down: + = right


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def scan_segment(seg):
    """Turn events: maximal runs of |yaw rate| > 0.08 rad/s (0.5 s smoothing, one sign, gaps < 0.5 s merged) whose heading change
    is >= 45 deg within <= 10 s at median speed 2-10 m/s. Straight windows: 10 s with |yaw rate| < 0.03, heading change < 3 deg,
    speed > 8 m/s throughout."""
    from jevdrive.openpilot.frames import load_segment_meta
    d = ROOT / seg
    if not ((d / "fcamera.hevc").exists() and (d / "ecamera.hevc").exists()):
        return []
    meta = load_segment_meta(d)
    if meta["fcam_wh"] != (1928, 1208) or not meta["has_ecam"]:
        return []
    v, yr = motion(meta)
    t = meta["t_loc"]
    n = min(len(v), len(meta["t_f"]), len(meta["t_e"]))
    v, yr, t = v[:n], yr[:n], t[:n]
    yrs = np.convolve(yr, np.ones(10) / 10, mode="same")
    psi = np.r_[0, np.cumsum(0.5 * (yrs[1:] + yrs[:-1]) * np.diff(t))]
    ev = []
    for sgn in (1, -1):
        runs = _runs(sgn * yrs > 0.08)
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] < 10:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for a, b in merged:
            dpsi = psi[b - 1] - psi[a]
            if abs(np.degrees(dpsi)) < 45 or (b - a) > 200 or not 2 <= np.median(v[a:b]) <= 10:
                continue
            start, hi = a - PRE - WARM, a + POST
            if start < 0 or b + 20 > n:
                continue
            ev.append(dict(kind="turn", seg=seg, start=int(start), onset=int(a), end=int(b), hi=int(min(hi, n)),
                           dir="right" if sgn > 0 else "left", dpsi_deg=float(np.degrees(dpsi)), dur_s=float((b - a) / 20),
                           v_med=float(np.median(v[a:b])), v_onset=float(v[a])))
    for s in range(0, n - WARM - STRAIGHT_SCORE - 100, 100):
        sc = slice(s + WARM, s + WARM + STRAIGHT_SCORE)
        if (np.abs(yrs[sc]).max() < 0.03 and abs(np.degrees(psi[sc.stop - 1] - psi[sc.start])) < 3 and v[sc].min() > 8):
            ev.append(dict(kind="straight", seg=seg, start=int(s), onset=int(s + WARM), end=int(sc.stop), hi=int(sc.stop),
                           v_med=float(np.median(v[sc]))))
            break                                                       # one straight window per segment
    return ev


def cmd_scan(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("op_fov", "scan", config=vars(a)) as run:
        segs = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
        res = par.pmap(scan_segment, segs, run=run, desc="segments")
        ev = [e for v in res.values if v for e in v]
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        json.dump(ev, open(out / "events.json", "w"), indent=0)
        k = [e for e in ev if e["kind"] == "turn"]
        run.summary.update(segments=len(segs), turns=len(k), turn_segments=len({e["seg"] for e in k}),
                           straight=sum(e["kind"] == "straight" for e in ev), failed=len(res.errors))
        run.info("%s", run.summary)


# ---------------------------------------------------------------- replay
_M = None


def wname(w):
    return f"{w['seg']}_{w['start']}"


def replay(job):
    """One window x all arms -> npz: per arm action (raw 4), plan mu (33, 15), lane lines mu (4, 33, 2) + prob, at every frame."""
    global _M
    from jevdrive import cache
    from jevdrive.openpilot.frames import decode_hevc, load_segment_meta
    from jevdrive.openpilot.model import OPModel, mdn_mu, sigmoid
    w, arms, out, backend = job
    f = Path(out) / f"{wname(w)}.npz"
    key = window_key(w, arms)
    if cache.done(f, key):
        return str(f)
    meta = load_segment_meta(ROOT / w["seg"])
    lo, hi = w["start"], w["hi"]
    warps = {k: ArmWarper(meta["rpy_calib"], *ARMS[k]) for k in arms}
    fr = np.zeros((len(arms), hi - lo, 2, 6, 128, 256), np.uint8)
    for n, (pr, pw) in enumerate(zip(decode_hevc(ROOT / w["seg"] / "fcamera.hevc"), decode_hevc(ROOT / w["seg"] / "ecamera.hevc"))):
        if n >= hi:
            break
        if n >= lo:
            for j, k in enumerate(arms):
                warps[k](pr, pw, fr[j, n - lo])
    if _M is None:
        _M = OPModel("cinque", backend)
    sl = _M.slices
    res = {}
    for j, k in enumerate(arms):
        _M.reset()
        act, plan, ll, lp = [], [], [], []
        for x in fr[j]:
            raw = _M.step(x, desire=np.zeros(8), traffic=(1, 0), action_t=(0.275, 0.525))
            act.append(raw[sl["action"]])
            plan.append(mdn_mu(raw[sl["plan"]], (33, 15)))
            ll.append(mdn_mu(raw[sl["lane_lines"]], (4, 33, 2)))
            lp.append(sigmoid(raw[sl["lane_lines_prob"]])[1::2])
        res |= {f"{k}_act": np.array(act, np.float32), f"{k}_plan": np.array(plan, np.float32),
                f"{k}_ll": np.array(ll, np.float32), f"{k}_lp": np.array(lp, np.float32), f"{k}_miss": np.array([warps[k].miss["road"], warps[k].miss["wide"]])}
    v, yr = motion(meta)
    cache.cached(f, key, lambda: dict(res, t=meta["t_loc"], v=v, yr=yr, lo=lo, hi=hi, pos=meta["pos"], R=meta["R"],
                                      rpy_calib=meta["rpy_calib"], ev=json.dumps(w)))
    return str(f)


def window_key(w, arms):
    from jevdrive import cache
    return cache.key(params=dict(w=w, arms={k: ARMS[k] for k in arms}, warm=WARM), code=[replay, gather_index], version=CACHE_VERSION)


def cmd_run(a):
    from jevdrive import cache, par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_fov", a.tag, config=vars(a)) as run:
        sp = splits.load(a.split)
        run.use_split(sp)
        ev = {wname(e): e for e in json.load(open(a.events))}
        wins = [ev[m] for m in sp.members]
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        arms = a.arms.split(",")
        jobs = [(w, arms, str(out), a.backend) for w in wins]
        res = par.pmap(replay, jobs, workers=a.workers, run=run, desc="windows",
                       skip=lambda j: cache.done(Path(j[2]) / f"{wname(j[0])}.npz", window_key(j[0], j[1])))
        run.summary.update(windows=len(wins), ok=res.ok, skipped=len(res.skipped), failed=len(res.errors), out=str(out))
        res.raise_if_failed()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--events")
    ap.add_argument("--split")
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", default="cuda-iob")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)
