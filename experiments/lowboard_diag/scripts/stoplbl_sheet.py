"""stoplbl: contact sheets of 20 sampled navtrain frames per situation, front image + the label (CPU, box).
  $DATA_DIR/envs/navsim2/bin/python experiments/lowboard_diag/scripts/stoplbl_sheet.py [OUT_DIR]
"""
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

D = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else D / "runs/lowboard_diag/stoplbl/sheets")
OUT.mkdir(parents=True, exist_ok=True)
L = pd.read_parquet(D / "runs/lowboard_diag/stoplbl/navtrain_s300.parquet")
L = L[L.full | (L.s_end >= 50)]
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.data import splits
L = L[splits.load("navsim/navtrain").mask(L.token)]          # only navtrain tokens have images
have = set(os.listdir(D / "datasets/navsim/sensor_blobs/trainval"))
L = L[L.log.isin(have)]
print("frames with images:", len(L), "logs", L.log.nunique())
rng = np.random.default_rng(1)


def pick(df, n):
    df = df.drop_duplicates("log") if df.log.nunique() >= n else df
    return df.sample(min(n, len(df)), random_state=int(rng.integers(1 << 30)))


SIT = {
    "red": (L[(L.tl_red_d >= 6) & (L.tl_red_d <= 40)], lambda r: f"RED light (log state), stop line {r.tl_red_d:.0f} m ahead; v0 {r.v0:.1f}, driver min v {r.tl_red_vmin:.1f}"),
    "stopsign": (L[(L.ss_d >= 6) & (L.ss_d <= 40)], lambda r: f"STOP SIGN line {r.ss_d:.0f} m ahead; v0 {r.v0:.1f}, driver min v {r.ss_vmin:.1f}"),
    "yield": (L[((L.ts_d >= 6) & (L.ts_d <= 40) & L.ss_d.isna() & L.tl_line_d.isna()) | ((L.yl_d >= 6) & (L.yl_d <= 40))],
              lambda r: f"YIELD/TURN-STOP line {np.nanmin([r.ts_d, r.yl_d]):.0f} m ahead; v0 {r.v0:.1f}, driver min v {np.nanmin([r.ts_vmin, r.yl_vmin if 'yl_vmin' in r else 99]):.1f}"),
    "turn": (L[(L.turn_deg >= 45) & (L.turn_s >= 5) & (L.turn_s <= 40)],
             lambda r: f"TURN {r.turn_deg:.0f} deg starts {r.turn_s:.0f} m ahead; v0 {r.v0:.1f}, driver entry {r.turn_vin:.1f}, v_curv(2 m/s2) {r.turn_vc:.1f}"),
}
cache = {}
for name, (df, txt) in SIT.items():
    sel = pick(df, 20)
    tiles = []
    meta = []
    for _, r in sel.iterrows():
        if r.log not in cache:
            cache.clear()
            cache[r.log] = pickle.load(open(D / "datasets/navsim/navsim_logs/trainval" / f"{r.log}.pkl", "rb"))
        f = cache[r.log][int(r.idx)]
        img = cv2.imread(str(D / "datasets/navsim/sensor_blobs/trainval" / f["cams"]["CAM_F0"]["data_path"]))
        img = cv2.resize(img, (800, 450))
        cv2.rectangle(img, (0, 0), (800, 56), (0, 0, 0), -1)
        s = txt(r)
        cv2.putText(img, s[:70], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, s[70:140], (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"#{len(tiles)} {r.token[:6]}", (6, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
        meta.append(dict(i=len(tiles) - 1, token=r.token, log=r.log, label=s))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    for part in range(0, len(tiles), 4):          # 4 tiles per image (2 x 2) so each is readable
        t = tiles[part:part + 4]
        while len(t) < 4:
            t.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / f"{name}_{part // 4}.jpg"), np.vstack([np.hstack(t[:2]), np.hstack(t[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    pd.DataFrame(meta).to_csv(OUT / f"{name}_meta.csv", index=False)
print("done")
