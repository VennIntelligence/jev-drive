"""Review sheet: what the model sees. t0 road frame (and optionally the wide one) of a few junction samples, every
overlay family, one command per column block. Any env with numpy / cv2 / PIL / scipy.

  python experiments/op_img_cmd/scripts/img_sheet.py --n 4 --out figs/overlay_sheet.png [--view wide] [--tokens a b]
"""
import argparse, json, os, pickle, sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import img_overlay as O  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def rgb(packed):
    from jevdrive import op_interp as I
    Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
    U = U.repeat(2, 0).repeat(2, 1) - 128
    V = V.repeat(2, 0).repeat(2, 1) - 128
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tokens", nargs="*")
    ap.add_argument("--view", default="road", choices=("road", "wide"))
    ap.add_argument("--fams", nargs="*", default=["none", "band", "lines", "arrow_road", "sign", "cones", "barrier", "wall", "fill_grass", "combo"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt-only", action="store_true", help="one column per family: command = a branch the driver did not take")
    ap.add_argument("--width", type=int, default=4096)
    a = ap.parse_args()
    import cv2
    G = [s for s in pickle.load(open(data_dir() / "runs" / "op_img_cmd" / "geom" / "nav.pkl", "rb")) if s["kind"] == "junction" and O.valid(s)]
    if a.tokens:
        G = [s for s in G if s["token"] in a.tokens]
    else:
        rng = np.random.default_rng(1)
        G = [G[k] for k in rng.choice(len(G), a.n, replace=False)]
    root = data_dir() / "runs" / "op_lb" / "lb_navtrain"
    keys = np.load(root / "keys.npy", mmap_mode="r")
    cam = json.loads((root / "meta.json").read_text())["cam"]
    v = 0 if a.view == "road" else 1
    rows = []
    for s in G:
        f = np.asarray(keys[s["row"], 3])
        cls = sorted({b["cls"] for b in s["branches"]} - {"uturn"})[:2]
        if a.alt_only:
            cls = [c for c in cls if c != s["taken"]][:1] or cls[:1]
        tiles = []
        for fam in a.fams:
            for c in (cls if fam != "none" else cls[:1]):
                im = rgb(O.draw(f, O.primitives(s, fam, c), (0.0, 0.0, 0.0), np.asarray(cam[s["row"]]))[v])
                im = np.ascontiguousarray(im)
                cv2.putText(im, f"{fam}" + ("" if fam == "none" else f" -> {c}"), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                tiles.append(im)
        cv2.putText(tiles[0], f"v {s['v']:.1f} m/s, taken {s['taken']}, dist {s['dist']:.1f} m", (6, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        rows.append(tiles)
    ncol = max(len(r) for r in rows)
    blank = np.zeros_like(rows[0][0])
    grid = np.concatenate([np.concatenate(r + [blank] * (ncol - len(r)), 1) for r in rows], 0)
    from PIL import Image
    im = Image.fromarray(grid)
    im.thumbnail((a.width, a.width))
    im.save(a.out)
    print(a.out, grid.shape)


if __name__ == "__main__":
    main()
