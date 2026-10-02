"""Common-cause factorial runner (openpilot env, one GPU). Plan: ../plans/2026-10-03-common-cause-prereg.md.

Frozen Cinque (op_lb's TensorRT backend and action_t), zero state per run, 20 Hz steps, the last step's plan is read.
Per sample of samples/<domain>.json (cc_prep.py) every variant of `variants(sample)` is run; history variants change the
frames, desire variants the desire input. Output: $DATA_DIR/runs/op_common_cause/raw/<domain>/<shard>.npz with
plan_pos (n, V, 33, 3), plan_yaw (n, V, 33), hidden (n, V, H) fp16, in_diff (n, V) (mean |frame - normal frame| over the
steps they share), pulses (n, V) (steps whose desire input is a rising edge), NaN / -1 where a variant was not run.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. $DATA_DIR/envs/openpilot/bin/python experiments/op_common_cause/scripts/cc_run.py \
      --domain carla --shard 0/3 [--limit 20] [--check]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_common_cause"
HIST = ("normal", "long", "repeat", "single", "rotL", "rotR")
DES = ("off", "pulse", "sustained", "lc", "wrong")
VARIANTS = [f"{h}|{d}" for h in HIST for d in DES]
RATE = np.radians(10.0)          # fake yaw rate of rotL / rotR (rad/s)
T_ON = -1.0                      # desire onset (s before t0)
ACTION_T = (0.275, 0.525)


def wanted(s, domain):
    """Variants run for this sample (prereg 'cross')."""
    hs = [h for h in HIST if h != "long" or (domain != "nav" and s.get("long"))]
    v = [f"{h}|off" for h in hs]
    if s["cmd"] != 0:
        v += [f"{h}|{d}" for h in hs for d in ("pulse", "sustained")] + ["normal|lc", "normal|wrong"]
    return v


def step_times(window):
    return np.round(np.arange(-round(window / 0.05), 1) * 0.05, 3)


# ---------------------------------------------------------------- frames per domain

_W = {}


def _init_stream(domain):
    import wod_zeroshot_openpilot as WZ
    if domain in ("wod", "wodt"):
        from jevdrive import wod_zeroshot as Z
        spans, _ = Z.load_spans()
        calib = json.loads((Z.root() / "op_calib.json").read_text())
        if domain == "wodt":
            from jevdrive import drive_backbones as DBK
            spans = json.loads((DBK.root() / "op_plan_trainval.json").read_text())["spans"]
            calib |= json.loads((DBK.root() / "op_calib_trainval.json").read_text())
        WZ._init(spans, calib, str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    else:
        from jevdrive import p5_openpilot as P
        WZ._init({}, {"carla": P.carla_calib()}, ".")
    _W["domain"] = domain


def render_sample(s):
    """5 Hz model frames (k, 2, 6, 128, 256), oldest first, the last at t0."""
    if _W["domain"] in ("wod", "wodt"):
        import drive_backbones_openpilot as DB
        return DB.render(s["hist"])
    import p5_openpilot as PO
    return PO.render(s["files"], "carla")


class NavFrames:
    """op_lb's exact navtrain inputs: keys + GIMM frames on the op_lb step schedule (31 steps, -1.5 ... 0 s)."""

    def __init__(self):
        import op_lb as B
        self.B = B
        self.keys = B.Keys("lb_navtrain")
        self.syn = np.load(B.root("lb_navtrain") / "gimm.npy", mmap_mode="r")
        self.ts, self.src = B._steps(0.0, False)
        sparse = np.r_[I.T_KEY, B.SYN_T]
        self.src_t = np.array([sparse[j] if k == "k" else sparse[4 + j] for k, j in self.src])

    def steps(self, row):
        kf, sf = self.keys[row], np.asarray(self.syn[row])
        fr = [np.ascontiguousarray(kf[j] if k == "k" else sf[j]) for k, j in self.src]
        return fr, self.src_t.copy()


def stream_steps(frames, window):
    """Step frames and their source times for a 5 Hz stream (last frame at t0) over `window` seconds."""
    ts = step_times(window)
    tau = -0.2 * np.arange(len(frames) - 1, -1, -1)
    k = np.searchsorted(tau, ts + 1e-6, side="right") - 1
    assert (k >= 0).all(), "stream shorter than the window"
    return [frames[j] for j in k], tau[k]


_MAPS = {}


def warp_rot(f, cam, d):
    """op_interp.warp_frame(f, cam, (0, 0, d), 0) with the remap tables cached per (camera, yaw): same arithmetic."""
    import cv2
    key = (tuple(np.round(cam, 2)), round(d, 6))
    if key not in _MAPS:
        if len(_MAPS) > 120:          # ~2.6 MB per entry; WOD has one calibration per sequence
            _MAPS.clear()
        c = np.round(cam, 2)
        mp = []
        for view in ("road", "wide"):
            mx, my = I.warp_map(view, c, np.array([0.0, 0.0, d]), np.zeros(3))
            mp.append((mx, my, (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25))
        _MAPS[key] = mp
    out = np.empty_like(f)
    for k, (mx, my, hx, hy) in enumerate(_MAPS[key]):
        Y, U, V = I.unpack(f[k])
        Yw = cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Uw = cv2.remap(np.ascontiguousarray(U), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Vw = cv2.remap(np.ascontiguousarray(V), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[k] = I.pack(Yw, Uw, Vw)
    return out


def rotate(fr, src_t, cam, sign):
    """Each step frame re-rendered as if the ego had yawed at sign * RATE up to t0 (source time t -> extra yaw sign*RATE*t)."""
    cache, out = {}, []
    for f, t in zip(fr, src_t):
        key = (id(f), float(t))
        if key not in cache:
            d = sign * RATE * float(t)
            cache[key] = f if abs(d) < 1e-9 else warp_rot(f, cam, d)
        out.append(cache[key])
    return out


# ---------------------------------------------------------------- one sample

def history(name, base, base_t, cam, long_fr=None):
    if name == "normal":
        return base, base_t
    if name == "long":
        return long_fr
    if name == "repeat":
        return [base[-1]] * len(base), np.zeros(len(base))
    if name == "single":
        return [base[-1]] * 4, np.zeros(4)
    return rotate(base, base_t, cam, 1.0 if name == "rotL" else -1.0), base_t


def desire_steps(d, cmd, n):
    ts = np.round(np.arange(-n + 1, 1) * 0.05, 3)
    if d == "off" or cmd == 0:
        return np.zeros(n, int), ts
    left = cmd < 0
    if d == "wrong":
        left = not left
    k = (3 if left else 4) if d == "lc" else (1 if left else 2)
    return np.where(ts >= T_ON - 1e-9, k, 0), ts


def run_one(m, fr, des, sustained, tc, decode, v):
    from op_lb import desire_arr
    m.reset()
    D = desire_arr(des, len(fr))
    prev, pulses = np.zeros(8, np.float32), 0
    for f, d in zip(fr, D):
        if sustained and d.any():
            m.prev_desire[:] = 0
            prev[:] = 0
        pulses += int(((d - prev) > .99).any())
        prev = d
        raw = m.step(f, desire=d, traffic=tc, action_t=ACTION_T)
    dd = decode(raw, m.slices, max(v, 0.0), ACTION_T)
    return dd["plan_pos"].copy(), dd["plan_yaw"].copy(), raw[m.slices["hidden_state"]].astype(np.float16), pulses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=("nav", "wod", "wodt", "carla"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="nav: normal|off vs op_lb's cached plans; rotation sign")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    from jevdrive.openpilot.model import OPModel, decode
    from jevdrive.run import Run
    import op_lb as B
    S = json.loads((ROOT / "samples" / f"{a.domain}.json").read_text())
    S = [S[k] for k in np.random.default_rng(0).permutation(len(S))][si::sn][: a.limit or None]   # any prefix is a random subset
    out = ROOT / "raw" / a.domain
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.domain}-{si}of{sn}" + ("-check" if a.check or a.limit else "")
    with Run("op_common_cause", tag, config=vars(a)) as run:
        ex = None
        if a.domain != "nav":     # fork the render workers before the TensorRT session exists (drive_backbones_openpilot)
            ex = ProcessPoolExecutor(a.workers, initializer=_init_stream, initargs=(a.domain,))
            list(ex.map(int, range(a.workers)))
        m = OPModel("cinque", B.BACKENDS["cinque"], cache=data_dir() / "runs" / "op_interp" / "trt_cache" / f"cinque-{B.BACKENDS['cinque']}",
                    context_rate=False)
        nH = m.slices["hidden_state"].stop - m.slices["hidden_state"].start
        n, V = len(S), len(VARIANTS)
        R = dict(plan_pos=np.full((n, V, 33, 3), np.nan, np.float32), plan_yaw=np.full((n, V, 33), np.nan, np.float32),
                 hidden=np.zeros((n, V, nH), np.float16), in_diff=np.full((n, V), np.nan, np.float32),
                 pulses=np.full((n, V), -1, np.int16))
        nav = NavFrames() if a.domain == "nav" else None
        if nav is None:
            from drive_backbones_openpilot import bounded_map
            src = bounded_map(ex, render_sample, S, 2 * a.workers)
        else:
            src = iter([None] * n)
        t0, steps, tm = time.time(), 0, {"hist": 0.0, "model": 0.0}
        for i, (s, frames) in enumerate(zip(S, run.tqdm(src, total=n, desc=tag))):
            cam = np.asarray(s["cam"], float)
            if nav is not None:
                base, base_t = nav.steps(s["row"])
                tc = (0, 1) if s["lht"] else (1, 0)
                long_fr = None
            else:
                frames = list(frames)
                base, base_t = stream_steps(frames, 1.5)
                long_fr = stream_steps(frames, 4.8) if "long|off" in wanted(s, a.domain) else None
                tc = (1, 0)
            hist = {}
            for vn in wanted(s, a.domain):
                h, d = vn.split("|")
                if h not in hist:
                    tq = time.perf_counter()
                    hist[h] = history(h, base, base_t, cam, long_fr)
                    tm["hist"] += time.perf_counter() - tq
                fr, _ = hist[h]
                des, _ = desire_steps(d, s["cmd"], len(fr))
                tq = time.perf_counter()
                pp, py, hid, pul = run_one(m, fr, des, d == "sustained", tc, decode, s["v"])
                tm["model"] += time.perf_counter() - tq
                j = VARIANTS.index(vn)
                R["plan_pos"][i, j], R["plan_yaw"][i, j], R["hidden"][i, j], R["pulses"][i, j] = pp, py, hid, pul
                k = min(len(fr), len(base))
                R["in_diff"][i, j] = float(np.mean([np.abs(fr[-q].astype(np.int16) - base[-q].astype(np.int16)).mean()
                                                    for q in range(1, k + 1, 4)]))
                steps += len(fr)
            if (i + 1) % 100 == 0:          # checkpoint: a stopped shard still leaves a random subset
                np.savez(out / f"{tag}.part.tmp.npz", ids=np.array([s["id"] for s in S[: i + 1]]), variants=np.array(VARIANTS),
                         **{k: v[: i + 1] for k, v in R.items()})
                os.replace(out / f"{tag}.part.tmp.npz", out / f"{tag}.part.npz")
            if i == 0 or (i + 1) % 25 == 0:
                run.info(f"[{i + 1}/{n}] {(time.time() - t0) / (i + 1):.2f} s/sample, {steps / (time.time() - t0):.0f} steps/s, " + ", ".join(f"{k} {v / (i + 1):.2f} s" for k, v in tm.items()))
        if nav is None:
            ex.shutdown()
        np.savez(out / f"{tag}.tmp.npz", ids=np.array([s["id"] for s in S]), variants=np.array(VARIANTS), **R)
        os.replace(out / f"{tag}.tmp.npz", out / f"{tag}.npz")
        (out / f"{tag}.part.npz").unlink(missing_ok=True)
        if a.domain == "nav" and (a.check or a.limit):
            c = np.load(B.root("lb_navtrain") / "plans" / "gimm@cinque.npz")
            cp = dict(zip(c["names"].tolist(), c["plan_pos"]))
            err = max(float(np.abs(R["plan_pos"][i, VARIANTS.index("normal|off")] - cp[s["id"]]).max()) for i, s in enumerate(S))
            run.info(f"nav normal|off vs cached op_lb plan: max |d plan_pos| = {err:.2e} m")
            run.summary["nav_repro_max_m"] = err
        run.summary.update(n=n, steps=steps, loop_s=time.time() - t0)
    sys.stdout.flush()
    os._exit(0)          # TensorRT teardown can hang


if __name__ == "__main__":
    main()
