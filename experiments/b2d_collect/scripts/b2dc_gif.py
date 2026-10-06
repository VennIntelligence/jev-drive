#!/usr/bin/env python
"""Captioned GIFs of collected B2D clips (runs on the Mac: the box has no CJK font). Input: the panel files scripts/b2dc_check.py writes
(<data>/check/panels/<route>.npz, copied to tmp/), each frame = chase view | road | wide model frames as Cinque gets them, with the nominal
horizon rows (dashed yellow), the logged future (green) and P2's plan (red) projected on the ground. Captions are Chinese.

  rsync -a autodl:$DATA_DIR/runs/b2d_collect/data/smoke/check/panels/ tmp/b2dc_panels/smoke/
  .venv/bin/python experiments/b2d_collect/scripts/b2dc_gif.py tmp/b2dc_panels/smoke --out experiments/b2d_collect/figs --tag smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
CMD = {0: "左转", 1: "直行", 2: "右转"}


def caption(c, r):
    td = c["turn_dist"]
    turn = f"，距下一个转弯 {td:.0f} m" if np.isfinite(td) and td < 200 else ""
    return (f"路线 {r['route_id']}（{r['town']}，场景 {r.get('type', '')}）  t = {c['t']:.1f} s  车速 {c['speed']:.1f} m/s  "
            f"路线指令 {CMD[c['cmd']]}{turn}",
            f"左：第三人称（模型看不到）  中：road 帧  右：wide 帧（均为 Cinque 实际输入）  "
            f"黄虚线 = 标称地平线行，绿 = 专家实际未来 4 s，红 = P2 规划")


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 15)
    small = ImageFont.truetype(FONT, 13)
    made = []
    for f in sorted(Path(a.panels).glob("*.npz")):
        z = np.load(f)
        frames, caps, r = z["frames"], json.loads(str(z["caps"])), json.loads(str(z["route"]))
        imgs = []
        for fr, c in list(zip(frames, caps))[: a.max_frames]:
            im = Image.fromarray(fr)
            W, H = im.size
            canvas = Image.new("RGB", (W, H + 46), (20, 20, 20))
            canvas.paste(im, (0, 46))
            d = ImageDraw.Draw(canvas)
            l1, l2 = caption(c, r)
            d.text((6, 3), l1, font=font, fill=(255, 255, 255))
            d.text((6, 25), l2, font=small, fill=(200, 200, 200))
            if a.width and W > a.width:
                canvas = canvas.resize((a.width, int(round(canvas.size[1] * a.width / W))), Image.BILINEAR)
            imgs.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=192))
        if not imgs:
            continue
        p = out / f"{a.tag}_{r['route_id']}.gif"
        imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
        made.append((str(p), len(imgs), round(p.stat().st_size / 1e6, 2)))
    for m in made:
        print(*m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("panels")
    ap.add_argument("--out", default="experiments/b2d_collect/figs")
    ap.add_argument("--tag", default="clip")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=75)
    ap.add_argument("--width", type=int, default=1100)
    main(ap.parse_args())
