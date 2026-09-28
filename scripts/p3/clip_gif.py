"""Small looping review clips for the docs, made from the full-size P3 review MP4s (scripts/p3/ds.py clip).

The 2 x 2 panel area is downscaled to --width and every --step-th frame is kept (10 Hz -> 5 fps at the default); the
caption bar is redrawn at the small size from the exam-item filter labels (jevdrive.nq4_p3_filter scenes ->
clip_labels/<key>.json), because the full-size caption is not legible after downscaling. Animated WebP by default
(GitHub renders it inline; a GIF of the same frames is 3-4x larger), GIF with --fmt gif.

  $DATA_DIR/envs/drivestudio/bin/python scripts/p3/clip_gif.py --scenes 0 1 2 4 6 --out <dir>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HZ, PRE_S = 10, 4.0
F = Path(os.environ["DATA_DIR"]) / "runs/nq4/p3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fmt", choices=("webp", "gif"), default="webp")
    ap.add_argument("--quality", type=int, default=70)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ttf = Path(sys.executable).parents[1] / "lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
    head_font, font = ImageFont.truetype(str(ttf), 15), ImageFont.truetype(str(ttf), 12)
    for k in a.scenes:
        key = f"p3_{k:03d}"
        sel = json.loads((F / "targets" / f"{k:03d}.json").read_text())
        lab = json.loads((F / "filter/clip_labels" / f"{key}.json").read_text())
        names = {w: chr(65 + i) for i, w in enumerate(sel["delete_tracks"])}
        f0 = int(sel["f0"])
        t_start = max(f0 - int(PRE_S * HZ), 0)
        rd = imageio.get_reader(F / "filter/clips" / f"{key}.mp4")
        frames = []
        for i, im in enumerate(rd):
            if i % a.step:
                continue
            t = t_start + i
            H2 = im.shape[0] - 112                       # the full-size caption bar is 112 px high
            W = a.width
            h = round(H2 * W / im.shape[1])
            fl = lab["frames"].get(str(t), {})
            react = any(v.get("react") for v in fl.values())
            lines = [(f"{key}   t - f0 = {(t - f0) / HZ:+.1f} s   ego {lab['v'][t]:.1f} m/s   "
                      + ("REACT (filter: must-react frame)" if react else "no react"), head_font, (255, 90, 90) if react else (235, 235, 235))]
            parts = []
            for w in sel["delete_tracks"]:
                v = fl.get(w)
                parts.append(f"{names[w]}: -" if v is None else
                             f"{names[w]}: d {v['d']:.0f} m, lat {v['L']:+.1f} m, ttr {v['ttr']:.1f} s, "
                             f"{'in lane' if v['in_lane'] else 'off lane'}, {'LEAD' if v['lead'] else 'no lead'}, "
                             + (f"label {v['lab_h']:.0f} px" if v["lab_h"] is not None else "no label"))
            lines += [("   |   ".join(parts[j:j + 2]), font, (215, 215, 215)) for j in range(0, len(parts), 2)]
            bar = 10 + 22 + 17 * (len(lines) - 1)
            canvas = Image.new("RGB", (W, h + bar), (20, 20, 20))
            canvas.paste(Image.fromarray(im[:H2]).resize((W, h), Image.LANCZOS), (0, 0))
            d = ImageDraw.Draw(canvas)
            y = h + 6
            for text, fnt, col in lines:
                d.text((8, y), text, fill=col, font=fnt)
                y += 22 if fnt is head_font else 17
            frames.append(canvas)
        out = a.out / f"{key}.{a.fmt}"
        dur = int(1000 * a.step / HZ)
        if a.fmt == "webp":
            frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0, quality=a.quality, method=6)
        else:
            pal = [f.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
            pal[0].save(out, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
        print(json.dumps({"scene": key, "frames": len(frames), "size": [frames[0].width, frames[0].height],
                          "bytes": out.stat().st_size, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
