"""PAI driver slot switch (experiments/alpasim/lib/pai_core.py, PAI_KEYWARP): with the switch off `slots` is the function it was before
the switch existed, bit for bit; with it on the slots are sh30_core's lattice of the four 2 Hz keyframes and nothing else of the stream.
CPU only (torch + cv2); python -m unittest tests.test_pai_keywarp -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/alpasim/lib"))
try:
    import torch  # noqa: F401
    import pai_core as PC
    from pai_core import C, I
except Exception as e:                                    # the serving stack is not importable here
    PC, WHY = None, repr(e)


def slots_before(frames, t0, P, V, cam_t, cold, tol_us=30_000):
    """pai_core.slots as of ef2f829d (before the switch), verbatim: the frozen reference of the off path."""
    T_SLOT_US = np.round(C.SLOT_T * 1e6).astype(np.int64)
    ts = np.array(sorted(frames), np.int64)
    near = [ts[np.abs(ts - (t0 + d)).argmin()] for d in T_SLOT_US]
    real = np.array([abs(int(n) - (t0 + int(d))) <= tol_us for n, d in zip(near, T_SLOT_US)])
    cur = np.zeros((len(T_SLOT_US),) + C.FRAME, np.uint8)
    for j in np.flatnonzero(real):
        cur[j] = frames[int(near[j])]
    valid = real.copy()
    if cold == "backwarp" and not real.all():
        track, ri = I.track_navsim(P, V), np.flatnonzero(real)
        for j in np.flatnonzero(~real):
            s = ri[np.abs(ri - j).argmin()]
            cur[j] = I.warp_frame(cur[s], np.asarray(cam_t, np.float64), track(C.SLOT_T[j]), track(C.SLOT_T[s]))
        valid[:] = True
    return cur, valid, real


def stream(rng, n: int, drop=(), jitter: int = 4_000, t_end: int = 7_300_000):
    """n frames at 10 Hz ending at t_end (us) with timestamp jitter, some dropped; smooth images so that a warp is not noise."""
    y, x = np.mgrid[0:128, 0:256]
    out = {}
    for i in range(n):
        if i in drop:
            continue
        f = (96 + 60 * np.sin(x / (9 + rng.random(12)[:, None, None] * 9) + y / 14 + i / 3) + rng.normal(0, 6, (12, 128, 256))).clip(0, 255)
        out[t_end - 100_000 * i + (int(rng.integers(-jitter, jitter + 1)) if i else 0)] = f.astype(np.uint8).reshape(C.FRAME)
    return out, t_end


def history(rng):
    """A turning ego history (4, 3), (4, 2) at op_interp.T_KEY in the t0 frame."""
    v, w = 4 + 8 * rng.random(), 0.4 * (rng.random() - 0.5)
    P = np.array([[v * t * np.cos(w * t / 2), v * t * np.sin(w * t / 2), w * t] for t in I.T_KEY])
    return P, np.tile([v, 0.0], (4, 1))


CAM = [1.7, 0.02, 1.45]


@unittest.skipIf(PC is None, "pai_core not importable: " + (WHY if PC is None else ""))
class TestPaiKeywarp(unittest.TestCase):
    def test_off_is_the_function_before(self):
        rng = np.random.default_rng(0)
        cases = [(16, ()), (16, (3, 4)), (16, (0 + 2, 9, 14)), (1, ()), (3, ()), (6, (2,)), (9, ()), (12, (11,))]
        for n, drop in cases:
            fr, t0 = stream(rng, n, drop)
            P, V = history(rng)
            for cold in C.COLD:
                ref = slots_before(fr, t0, P, V, CAM, cold)
                for got in (PC.slots(fr, t0, P, V, CAM, cold), PC.slots(fr, t0, P, V, CAM, cold, keywarp=False, dev="cpu")):
                    for a, b in zip(ref, got):
                        self.assertIsInstance(b, np.ndarray)
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertTrue(np.array_equal(a, b), (n, drop, cold))

    def test_on_is_the_lattice_of_the_keys(self):
        rng = np.random.default_rng(1)
        for n, drop, e in ((16, (), 0), (16, (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14), 0), (12, (), 1), (8, (), 2), (3, (), 3), (16, (10,), 2)):
            fr, t0 = stream(rng, n, drop)
            P, V = history(rng)
            keys = [fr[min(fr, key=lambda t: abs(t - (t0 + d)))] for d in PC.T_KEY_US[e:]]
            for cold in C.COLD:
                cur, valid, real = PC.slots(fr, t0, P, V, CAM, cold, keywarp=True, dev="cpu")
                ref, rvalid = C.lattice(C.stack_keys(keys), e, I.track_navsim(P, V), np.asarray(CAM, np.float64), cold)   # the CPU reference path
                self.assertTrue(np.array_equal(valid, rvalid) and np.array_equal(real, C.SLOT_T >= I.T_KEY[e] - 1e-9), (n, e, cold))
                cur = cur.numpy()
                self.assertEqual(cur.shape, (8,) + C.FRAME)
                for j, t in enumerate(C.SLOT_T):                                   # a slot that is a keyframe is that frame, untouched
                    k = np.flatnonzero(np.isclose(I.T_KEY, t))
                    if len(k) and k[0] >= e:
                        self.assertTrue(np.array_equal(cur[j], keys[k[0] - e]))
                d = cur[valid].astype(int) - ref[valid]                            # warp_gpu = warp_frame up to 1/32 px rounding ties
                self.assertLess((d != 0).mean(), 1e-4, (n, e, cold))
                self.assertLessEqual(abs(d).max(), 2 if d.any() else 0)

    def test_on_reads_only_the_keys(self):
        rng = np.random.default_rng(2)
        fr, t0 = stream(rng, 16)
        P, V = history(rng)
        a = PC.slots(fr, t0, P, V, CAM, "backwarp", keywarp=True, dev="cpu")[0]
        for t in list(fr):
            if min(abs(t - (t0 + d)) for d in PC.T_KEY_US) > 30_000:
                fr[t] = rng.integers(0, 256, C.FRAME, dtype=np.uint8)
        self.assertEqual(sum(min(abs(t - (t0 + d)) for d in PC.T_KEY_US) > 30_000 for t in fr), 12)
        self.assertTrue(np.array_equal(a.numpy(), PC.slots(fr, t0, P, V, CAM, "backwarp", keywarp=True, dev="cpu")[0].numpy()))


if __name__ == "__main__":
    unittest.main()
