"""GIF of one arrow turn: the chase (third-person) camera next to the openpilot model input frames with the sky arrow as the model sees it
(envs/openpilot python: av, PIL, numpy). Input: an attempt of the jfa gif stage (chase_raw.mp4 + video_frames.jsonl) and its IMG_CL_DUMP
directory (every overlaid request, file name = <n>_t<sim time>_<command>_<distance>.png, road view above wide view).

  junction_forced_gif.py <attempt_dir> <dump_dir> <out.gif> --label "NA route 28180" --route-turn 0 [--before 40 --after 25] [--fps 10] [--speedup 2]

The window is the sim time the car is [before] m of route arc before the turn start until [after] m past its end (turn = the --route-turn-th
maneuver >= 25 deg of the route polyline, as junction_cl_report); the path position of the car is its nearest dense-route point.
"""
import argparse
import bisect
import json
import re
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "lib"), str(REPO / "experiments/vlm_arb/scripts"), str(Path(__file__).resolve().parent)]
from route_poly import maneuvers  # noqa: E402
import turn_calibration_lib as L  # noqa: E402
from vlm_arb_gif import Attempt, FONT  # noqa: E402


def window(att, mi, before, after):
    xy = np.array(json.load(open(att.d / "route.json"))["xy"])
    P, g = L.resample(xy)
    ms = [m for m in maneuvers(P) if abs(m["angle"]) >= 25]
    m = ms[mi]
    s0, s1 = g[m["i0"]] - before, g[m["i1"]] + after
    tk = [t for t in att.ticks if "truth" in t]
    T = np.array([t["t"] for t in tk])
    near = np.array([int(np.argmin(np.linalg.norm(P - np.array(t["truth"][:2]), axis=1))) for t in tk])
    sc = g[near]
    ok = np.where((sc >= s0) & (sc <= s1))[0]
    return float(T[ok[0]]), float(T[ok[-1]]), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt")
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--route-turn", type=int, default=0)
    ap.add_argument("--before", type=float, default=40.0)
    ap.add_argument("--after", type=float, default=25.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--speedup", type=int, default=2)
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--max-mb", type=float, default=5.8)
    n = ap.parse_args()
    att = Attempt(n.attempt)
    t0, t1, m = window(att, n.route_turn, n.before, n.after)
    files = sorted(Path(n.dump).glob("*.png"))
    ft = np.array([float(re.search(r"_t([\d.]+)_", f.name).group(1)) for f in files])
    cut = np.flatnonzero(np.diff(ft) < -1.0) + 1                  # the server outlives a retried attempt: sim time restarts at 0; keep the last block (= the attempt dir given)
    if len(cut):
        files, ft = files[cut[-1]:], ft[cut[-1]:]
    ncmd = [re.search(r"_t[\d.]+_(\w+?)_([\w.]+)\.png", f.name).groups() for f in files]
    h = int(round(n.width * 9 / 16 / 2) * 2)
    targets = np.arange(t0, t1, n.speedup / n.fps)
    idx = {}
    for j, tt in enumerate(targets):
        idx.setdefault(int(np.abs(att.t - tt).argmin()), []).append(j)
    font = ImageFont.truetype(FONT, 12)
    frames = [None] * len(targets)
    src = av.open(str(att.d / "chase_raw.mp4"))
    for i, fr in enumerate(src.decode(video=0)):
        if i > max(idx):
            break
        if i not in idx:
            continue
        chase = Image.fromarray(fr.to_ndarray(format="rgb24")).resize((n.width, h), Image.LANCZOS)
        for j in idx[i]:
            k = int(np.abs(ft - targets[j]).argmin())
            model = Image.open(files[k]).convert("RGB").resize((n.width // 2 + 20, n.width // 2 + 20), Image.LANCZOS)   # both model views, 1:1 aspect
            canvas = Image.new("RGB", (n.width + model.width, max(h, model.height)), (12, 16, 24))
            canvas.paste(chase, (0, 0))
            canvas.paste(model, (n.width, 0))
            d = ImageDraw.Draw(canvas)
            x = att.at(att.ticks, "t", att.t[i])
            cmd, dist = ncmd[k]
            d.rectangle((0, 0, n.width, 16), fill=(12, 16, 24))
            d.text((4, 2), "%s  t=%.1f s  v=%.1f km/h  x%d" % (n.label, att.t[i], x["v"] * 3.6, n.speedup), font=font, fill="white")
            d.text((n.width + 4, model.height - 15), "model input: arrow = %s %s m" % (cmd, dist), font=font, fill=(110, 220, 255))
            frames[j] = canvas
    src.close()
    frames = [f for f in frames if f is not None]
    for c, scale in ((48, 1.0), (32, 1.0), (24, 0.9), (24, 0.8)):
        fr = [f.resize((int(f.width * scale) // 2 * 2, int(f.height * scale) // 2 * 2), Image.LANCZOS) if scale < 1 else f for f in frames]
        q = [f.quantize(colors=c, method=Image.MEDIANCUT, dither=Image.NONE) for f in fr]
        q[0].save(n.out, save_all=True, append_images=q[1:], duration=int(1000 / n.fps), loop=0, optimize=True, disposal=1)
        if Path(n.out).stat().st_size <= n.max_mb * 1e6:
            break
    print("%s: %d frames (sim %.1f..%.1f s), %.2f MB, turn %+.0f deg" % (n.out, len(q), t0, t1, Path(n.out).stat().st_size / 1e6, m["angle"]))


if __name__ == "__main__":
    main()
