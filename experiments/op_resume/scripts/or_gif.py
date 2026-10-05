#!/usr/bin/env python
"""GIF of one HUGSIM run around the resume rule's first firing: the simulator's three front renders (video.mp4, top row) above the
two frames Cinque actually got (road + wide, from zs_dump img2: the packed YUV after the HUGSIM rig warp), with time, speed, the
rule state, the lead head and the plan's 3 s distance. Needs a run with agent opt dump_every 1.

  python experiments/op_resume/scripts/or_gif.py <run_dir> <out.gif> [--before 40] [--after 80] [--fps 8] [--step 1] [--scale 1]
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw


def yuv_to_rgb(p):
    """(6, 128, 256) packed openpilot frame -> (256, 512, 3) uint8 RGB (BT.601 limited range, the inverse of HUGSIM's pack)."""
    Y = np.empty((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = p[:4]
    U = np.repeat(np.repeat(p[4].astype(np.float32), 2, 0), 2, 1) - 128
    V = np.repeat(np.repeat(p[5].astype(np.float32), 2, 0), 2, 1) - 128
    C = 1.164 * (Y - 16)
    rgb = np.stack([C + 1.596 * V, C - 0.392 * U - 0.813 * V, C + 2.017 * U], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--before", type=int, default=40)
    ap.add_argument("--after", type=int, default=80)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--step", type=int, default=1, help="every k-th simulator step")
    ap.add_argument("--scale", type=float, default=1.0, help="output size factor (repo GIFs: keep under ~8 MB)")
    a = ap.parse_args()
    run = Path(a.run)
    L = [json.loads(x) for x in open(run / "zs_steps.jsonl") if '"setup"' not in x[:12]]
    fire = next((k for k, r in enumerate(L) if r.get("rr") == "fire"), None)
    if fire is None:
        raise SystemExit("the rule never fired in %s" % run)
    k0, k1 = max(0, fire - a.before), min(len(L), fire + a.after)
    vid = iio.get_reader(str(run / "video.mp4"))
    frames = []
    for k in range(k0, k1, a.step):
        npz = run / "zs_dump" / ("%04d.npz" % k)
        if not npz.exists():
            continue
        img2 = np.load(npz)["img2"]
        try:
            sim = vid.get_data(k)
        except IndexError:
            break
        top = Image.fromarray(sim[: sim.shape[0] // 2]).resize((1024, 192))
        road, wide = (Image.fromarray(yuv_to_rgb(img2[i])).resize((512, 256)) for i in (0, 1))
        canvas = Image.new("RGB", (1024, 192 + 256 + 44), (20, 20, 20))
        canvas.paste(top, (0, 0))
        canvas.paste(road, (0, 192))
        canvas.paste(wide, (512, 192))
        r = L[k]
        d = ImageDraw.Draw(canvas)
        state = r.get("rr") or "-"
        s3 = float(np.hypot(*r["plan"][5])) if len(r.get("plan", [])) >= 6 else 0.0
        txt = "t %5.2f s  v %4.2f m/s  rule: %-10s  lead p %.2f x %4.1f m  plan 3 s: %4.1f m" % (
            r["t"], r["v"], state, r.get("lead_prob") or 0, r.get("lead_x") or 0, s3)
        col = (255, 200, 0) if state in ("fire", "launch") else (230, 230, 230)
        d.text((8, 192 + 256 + 6), txt, fill=col)
        d.text((8, 192 + 256 + 24), "top: simulator front renders (left / front / right)   bottom: Cinque's input frames, road | wide",
               fill=(160, 160, 160))
        d.text((6, 4), "simulator", fill=(255, 255, 255))
        d.text((6, 196), "model road", fill=(255, 255, 0))
        d.text((518, 196), "model wide", fill=(255, 255, 0))
        if a.scale != 1.0:
            canvas = canvas.resize((int(canvas.width * a.scale), int(canvas.height * a.scale)), Image.LANCZOS)
        frames.append(np.asarray(canvas))
    iio.mimsave(a.out, frames, duration=1.0 / a.fps, loop=0)
    print("%d frames (steps %d-%d, first firing at step %d) -> %s" % (len(frames), k0, k1 - 1, fire, a.out))


if __name__ == "__main__":
    main()
