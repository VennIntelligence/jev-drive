#!/usr/bin/env python
"""Stitch the per-plan model-input dumps of one zero-shot Bench2Drive attempt (frames/*.jpg, written by
scripts/zeroshot_policy_server.py) into an H.264 video. Runs in any env with PyAV (envs/openpilot on the box).

    ~/data/envs/openpilot/bin/python scripts/zeroshot_b2d_video.py <attempt dir> --fps 10
"""
import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("attempt")
    p.add_argument("--fps", type=int, default=10, help="2 Hz plans at 10 fps = 5x real time")
    p.add_argument("--out", default="")
    a = p.parse_args()
    frames = sorted(Path(a.attempt, "frames").glob("*.jpg"))
    out = Path(a.out or Path(a.attempt) / "route.mp4")
    first = Image.open(frames[0])
    w, h = (first.width // 2) * 2, (first.height // 2) * 2
    with av.open(str(out), "w") as c:
        s = c.add_stream("libx264", rate=a.fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "26", "preset": "medium"}
        for f in frames:
            img = np.asarray(Image.open(f).convert("RGB"))[:h, :w]
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    print("%d frames -> %s (%.1f MB)" % (len(frames), out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
