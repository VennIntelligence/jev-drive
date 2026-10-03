"""Review sheet of the Q3 overlays (op-train venv, CPU): what the model sees at t0 with the sky arrow and with sky + green line.
Rows: CARLA junction samples (test split) and nav eval junction samples; columns: road / wide frame for `none`, then each exit class
for sky and sg (luma only; the arrow is magenta, the line green, both visible as chroma to the model).
Output experiments/op_img_cmd/figs/sky_sheet.png.

  $DATA_DIR/envs/op-train/bin/python experiments/op_img_cmd/scripts/img3_sheet.py
"""
import os, sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_adapt_h" / "scripts"), str(Path(__file__).resolve().parent)]
import img2_bank as QB  # noqa: E402
import img3_bank as B3  # noqa: E402


def rgb(packed):
    """packed (6, 128, 256) -> (256, 512, 3) uint8 RGB from the YUV planes."""
    from jevdrive import op_interp as I
    Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
    U, V = (np.repeat(np.repeat(z, 2, 0), 2, 1) - 128 for z in (U, V))
    out = np.stack([Y + 1.402 * V, Y - 0.344 * U - 0.714 * V, Y + 1.772 * U], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    from PIL import Image, ImageDraw
    sp = QB.carla_split()
    G = [s for s in QB.carla_samples() if sp[s["token"]] == "test" and s["tag"] == "d10"][:3]
    G = B3.with_lanes("sgcarla", G, [s["cam"] for s in G])
    rows = []
    for s in G:
        tiles = []
        for name, fn in (("sky", lambda s: B3.sky_variants({k: v for k, v in s.items() if k != "lanes"})), ("sg", B3.sg_variants)):
            ss = s if name == "sg" else {k: v for k, v in s.items() if k != "lanes"}
            vs, im = B3.carla_render((ss, "test"))
            for v, (fam, c) in enumerate(vs):
                if fam == "none" and name == "sg":
                    continue
                t = np.concatenate([rgb(im[v, 9, 0]), rgb(im[v, 9, 1])], 0)
                img = Image.fromarray(t)
                ImageDraw.Draw(img).text((6, 6), f"{s['token']} {fam} {c} (taken {s['taken']}, d {s['dist']:.0f} m)", fill=(255, 255, 0))
                tiles.append(np.asarray(img))
        rows.append(np.concatenate(tiles, 1))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    sheet = Image.fromarray(np.concatenate(rows, 0))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2))
    out = REPO / "experiments" / "op_img_cmd" / "figs" / "sky_sheet.png"
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
