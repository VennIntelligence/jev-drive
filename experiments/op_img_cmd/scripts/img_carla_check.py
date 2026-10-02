"""CARLA geometry check sheet: the approach + taken-branch band and the taken branch's boundary lines (img_overlay
families band / lines) drawn into the t0 road and wide model frames of a few carla.pkl samples, next to the raw road
frame. If the geometry, the frame conventions and the model-frame camera are right, the band lies on the ego lane and
the lines on the lane edges through the junction. Red: the approach lane's boundaries (should sit on the painted
lane markings before the junction). Columns: t0 road raw, t0 road drawn, t0 wide drawn, oldest history frame (t0 - 1.6 s)
road drawn with its own pose (the paint must stay on the same road spot).

  $DATA_DIR/envs/openpilot/bin/python experiments/op_img_cmd/scripts/img_carla_check.py [--pkl carla] [--n 6]
      -> experiments/op_img_cmd/figs/carla_geom_check.png
"""
import argparse, os, pickle, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
import img_overlay as O  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def rgb(packed):
    """(2, 6, 128, 256) packed -> [road, wide] uint8 RGB (256, 512, 3) (full-range BT.601, chroma replicated)."""
    out = []
    for p in packed:
        Y, U, V = (z.astype(np.float32) for z in I.unpack(p))
        U, V = (np.repeat(np.repeat(z, 2, 0), 2, 1) - 128 for z in (U, V))
        out.append(np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8))
    return out


def main():
    import cv2
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="carla")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tokens", default="", help="comma list instead of the automatic pick")
    ap.add_argument("--scale", type=float, default=0.75, help="tile size / 512 x 256")
    ap.add_argument("--cam-dz", type=float, default=0.0, help="debug: shift the camera origin up (m)")
    ap.add_argument("--out", default=str(REPO / "experiments" / "op_img_cmd" / "figs" / "carla_geom_check.png"))
    a = ap.parse_args()
    G = [s for s in pickle.load(open(data_dir() / "runs" / "op_img_cmd" / "carla" / f"{a.pkl}.pkl", "rb")) if O.valid(s)]
    rng = np.random.default_rng(0)
    pick = []
    for tag in ("d10", "d5", "d20", "d1", "stop", "d10"):        # spread over distances and taken classes
        c = [s for s in G if s["tag"] == tag and s not in pick and s["taken"] not in {p["taken"] for p in pick[-2:]}] or \
            [s for s in G if s["tag"] == tag and s not in pick]
        if c:
            pick.append(c[rng.integers(len(c))])
    pick = pick + [s for s in G if s not in pick][: a.n - len(pick)]
    pick = [s for s in G if s["token"] in a.tokens.split(",")] if a.tokens else pick[: a.n]
    rows = []
    for s in pick:
        fr = np.load(s["frames"])["frames"]
        cam = np.asarray(s["cam"], float) + (0, 0, a.cam_dz)
        edges = [O.ribbon(O.chain(s["approach"], k), 0.06) for k in ("l", "r")]
        lay = O.primitives(s, "band", s["taken"]) + O.primitives(s, "lines", s["taken"]) + [(edges[0] + edges[1], O.RED, 1.0, 0)]
        d, d0 = (O.draw(fr[k], lay, s["pose"][k], cam) for k in (-1, 0))
        tiles = [rgb(fr[-1])[0], rgb(d)[0], rgb(d)[1], rgb(d0)[0]]
        row = np.concatenate([cv2.resize(t, (int(512 * a.scale), int(256 * a.scale)), interpolation=cv2.INTER_AREA) for t in tiles], 1)
        txt = f"{s['token']} {s['town']} taken {s['taken']} cmd {s['cmd']} v {s['v']:.1f} m/s dist {s['dist']:.1f} m"
        cv2.putText(row, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        rows.append(row)
    img = Image.fromarray(np.concatenate(rows, 0)).quantize(64)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    img.save(a.out, optimize=True)
    print(a.out, os.path.getsize(a.out) // 1024, "KB;", [s["token"] for s in pick])


if __name__ == "__main__":
    main()
