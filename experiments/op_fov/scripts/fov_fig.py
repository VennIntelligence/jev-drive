#!/usr/bin/env python
"""Figure: the model's actual road / wide input frames per FOV arm on one comma1M turn, 1 s before onset and 2 s after.

  python fov_fig.py <events.json> <window name> <out.png>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[3])]
import fov_replay as F  # noqa: E402


def main(events, win, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jevdrive.openpilot.frames import decode_hevc, load_segment_meta, unpack_luma
    w = {F.wname(e): e for e in json.load(open(events))}[win]
    meta = load_segment_meta(F.ROOT / w["seg"])
    want = {w["onset"] - 20: "1 s before onset", w["onset"] + 40: "2 s after onset"}
    arms = [k for k in F.ARMS if k not in F.FROZEN_WIDE]
    warps = {k: F.ArmWarper(meta["rpy_calib"], *F.ARMS[k]) for k in arms}
    got = {}
    for n, (pr, pw) in enumerate(zip(decode_hevc(F.ROOT / w["seg"] / "fcamera.hevc"), decode_hevc(F.ROOT / w["seg"] / "ecamera.hevc"))):
        if n in want:
            for k, wp in warps.items():
                o = wp(pr, pw, np.zeros((2, 6, 128, 256), np.uint8))
                got[(n, k)] = (unpack_luma(o[0]), unpack_luma(o[1]))
        if n >= max(want):
            break
    fig, ax = plt.subplots(len(arms), 4, figsize=(16, 2.1 * len(arms)))
    for r, k in enumerate(arms):
        fr, fw = F.ARMS[k]
        for c, (n, lab) in enumerate(want.items()):
            for j, cam in enumerate(("road", "wide")):
                a = ax[r, 2 * c + j]
                a.imshow(got[(n, k)][j], cmap="gray", vmin=0, vmax=255)
                a.axhline(47.6 if cam == "road" else 151.8, color="lime", lw=0.6, alpha=0.6)
                a.set_xticks([]), a.set_yticks([])
                f = fr if cam == "road" else fw
                fx = np.atleast_1d(f)[0]
                a.set_title(f"{k} {cam} f {f if np.isscalar(f) else '%g/%g' % tuple(f)} ({F.hfov(fx):.0f} deg), {lab}", fontsize=8)
    fig.suptitle(f"comma1M {w['seg'][:8]} {w['dir']} turn {w['dpsi_deg']:.0f} deg at {w['v_med']:.1f} m/s: openpilot's 512x256 inputs per arm "
                 "(green = horizon row)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
