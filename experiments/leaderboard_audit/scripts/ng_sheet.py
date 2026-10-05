"""Night-gap audit step 3: contact sheet of WOD-E2E val target frames in openpilot's model view (road + wide frame as the shipped model gets them,
rendered by the exam harness scripts/wod_zeroshot_openpilot.py model_frames; last history frame = the target frame).
Rows: 6 night, 2 dusk, 4 day frames (fixed seed, spread over sequences) with sequence luma.  Needs wod_frames.csv from ng_analyze.py.
  $DATA_DIR/envs/jevdrive/bin/python experiments/leaderboard_audit/scripts/ng_sheet.py  -> $DATA_DIR/runs/night_gap/sheet.png"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import wod_zeroshot_openpilot as R  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402

D = Path(os.environ["DATA_DIR"])
OUT = D / "runs/night_gap"


def to_rgb(p):
    """(6,128,256) packed YCbCr planes -> (256,512,3) uint8 RGB (BT.601 full range)."""
    Y = np.empty((256, 512), np.float32)
    for k, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        Y[i::2, j::2] = p[k]
    cb = np.repeat(np.repeat(p[4].astype(np.float32), 2, 0), 2, 1) - 128
    cr = np.repeat(np.repeat(p[5].astype(np.float32), 2, 0), 2, 1) - 128
    rgb = np.stack([Y + 1.402 * cr, Y - 0.344136 * cb - 0.714136 * cr, Y + 1.772 * cb], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    w = pd.read_csv(OUT / "wod_frames.csv")
    pick = pd.concat([w[w.lab == l].drop_duplicates("seq").sample(n, random_state=1) for l, n in (("night", 6), ("dusk", 2), ("day", 4))])
    spans, _ = Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    R._init(spans, op_calib, str(D / "datasets/waymo_e2e/front3"))
    tiles = []
    for _, r in pick.iterrows():
        _, _, fr = R.model_frames(r["name"])
        a, b = to_rgb(fr[-1, 0]), to_rgb(fr[-1, 1])
        t = np.concatenate([a, b], 1)
        im = Image.fromarray(t); ImageDraw.Draw(im).text((6, 6), f'{r["lab"]} luma {r.luma:.0f}  {r["cluster"]}  ADE3 {r.op_ade3:.2f} m', fill=(255, 255, 0))
        tiles.append(np.asarray(im))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    Image.fromarray(np.concatenate(rows, 0)).save(OUT / "sheet.png")
    pick[["name", "lab", "luma", "cluster", "op_ade3"]].to_csv(OUT / "sheet_frames.csv", index=False)


if __name__ == "__main__":
    main()
