"""WOD-E2E clip: what shipped openpilot sees and plans on a real Waymo log, as the WOD harness feeds it (scripts/wod_zeroshot_openpilot.py:
FRONT + FRONT_LEFT + FRONT_RIGHT stitched into the road / wide model frames, nearest neighbour, each 10 Hz frame fed twice, zero state 10 s
before the first shown frame). Open loop: nothing is executed, the car drives the log.

Per shown 10 Hz frame, three panels:
  left    the WOD FRONT camera (source; WOD has no third-person camera)
  middle  BEV in the current rear-axle frame (x up, left to the left): logged past 4 s (grey), logged future 5 s (green), openpilot's plan
          converted to WOD rear-axle waypoints as scored (cyan, jevdrive.wod_zeroshot.openpilot_to_wod)
  right   openpilot's road (top) and wide (bottom) model frames, RGB of the exact YCbCr pixels the model gets, plan drawn on a flat road
          at the camera height

  $DATA_DIR/envs/op-train/bin/python experiments/leaderboard_audit/scripts/lbx_wod_clip.py pick [--n 10]
  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/leaderboard_audit/scripts/lbx_wod_clip.py make <out.gif> [--seq S --frame F]
pick lists val sequences with a launch from standstill into a turn (the event frame); make without --seq takes the first one.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import waymo as W  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

SHARDS = data_dir() / "datasets" / "waymo_e2e" / "front3"
CAM_H = 1.86                     # true front camera height above the road (decision 108)
PPM, BW, BH = 5.0, 300, 512


def candidates():
    df = W.load_index()
    past, fut = W.load_ego()
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    val = (df.split.astype(str).to_numpy() == "val") & df.sequence.astype(str).isin(set(calib)).to_numpy()
    v0 = np.linalg.norm(past[:, -1, 2:4], axis=1)
    d3 = np.linalg.norm(fut[:, 11, :2], axis=1)                     # displacement at 3 s
    ch = fut[:, -1, :2] - fut[:, -5, :2]
    turn = np.degrees(np.abs(np.arctan2(ch[:, 1], ch[:, 0])))        # heading of the last 1 s of the future against t0
    ok = val & (v0 < 0.2) & (d3 > 3.0) & (turn > 45) & (np.linalg.norm(ch, axis=1) > 2.0) & (df.frame.to_numpy() >= 40) & (df.frame.to_numpy() <= 120)
    rows = np.flatnonzero(ok)
    seen, out = set(), []
    for r in rows:
        s = str(df.sequence.iloc[r])
        if s in seen:
            continue
        seen.add(s)
        out.append((s, int(df.frame.iloc[r]), float(turn[r]), float(d3[r])))
    out.sort(key=lambda c: -c[2])                                       # sharpest turn first
    return df, past, fut, calib, out


def gather_maps(cal):
    cal = {int(c): d for c, d in cal.items()}
    cal = {c: cal[c] for c in Z.OP_SRC}
    sizes = [(cal[c]["width"], cal[c]["height"]) for c in Z.OP_SRC]
    idx = {}
    for k in ("road", "wide"):
        src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), cal)
        idx[k] = G.nn_gather_index(src, U, V, sizes).ravel()
    return idx


def pack(ycc):
    Y = ycc[..., 0]
    uv = np.rint(ycc[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))).astype(np.uint8)
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])


def read_frame(row, idx):
    """(packed (2, 6, 128, 256), road / wide RGB, FRONT RGB small) of one index row."""
    planes, front = [], None
    with open(SHARDS / row.shard, "rb") as f:
        for off, ln in ((row.front_off, row.front_len), (row.front_left_off, row.front_left_len), (row.front_right_off, row.front_right_len)):
            f.seek(int(off))
            b = f.read(int(ln))
            im = Image.open(io.BytesIO(b))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
            if front is None:
                fr = Image.open(io.BytesIO(b))
                fr.draft("RGB", (fr.size[0] // 2, fr.size[1] // 2))
                front = np.asarray(fr.convert("RGB").resize((768, 512)))
    cat = np.concatenate(planes)
    packed, rgb = np.empty((2, 6, 128, 256), np.uint8), {}
    for m, k in enumerate(("road", "wide")):
        ycc = np.ascontiguousarray(cat[idx[k]].reshape(G.OP_H, G.OP_W, 3))
        packed[m] = pack(ycc)
        rgb[k] = np.asarray(Image.fromarray(ycc, "YCbCr").convert("RGB"))
    return packed, rgb, front


def put(img, txt, xy, scale=0.5, col=(255, 255, 255), bg=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = xy
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 4), bg, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def bev(past, fut, wod_plan):
    img = np.full((BH, BW, 3), 245, np.uint8)
    q = lambda p: np.stack([BW / 2 - p[:, 1] * PPM, 0.75 * BH - p[:, 0] * PPM], 1).round().astype(np.int32)  # noqa: E731
    for r in range(10, 80, 10):
        cv2.circle(img, (BW // 2, int(0.75 * BH)), int(r * PPM), (225, 225, 225), 1)
    cv2.polylines(img, [q(past[:, :2])], False, (140, 140, 140), 3, cv2.LINE_AA)
    cv2.polylines(img, [q(np.r_[np.zeros((1, 2)), fut[:, :2]])], False, (40, 160, 40), 3, cv2.LINE_AA)
    cv2.polylines(img, [q(np.r_[np.zeros((1, 2)), wod_plan])], False, (0, 170, 220), 2, cv2.LINE_AA)
    car = np.array([[3.8, 0.95], [3.8, -0.95], [-1.0, -0.95], [-1.0, 0.95]])
    cv2.polylines(img, [q(car)], True, (200, 80, 0), 2, cv2.LINE_AA)
    put(img, "BEV, rear-axle frame", (6, 16), 0.42)
    put(img, "grey past  green logged future", (6, 36), 0.4, (0, 0, 0), (245, 245, 245))
    put(img, "cyan openpilot plan (scored)", (6, 54), 0.4, (0, 120, 170), (245, 245, 245))
    cv2.line(img, (8, BH - 12), (8 + int(10 * PPM), BH - 12), (0, 0, 0), 2)
    put(img, "10 m", (8, BH - 18), 0.4, (0, 0, 0), (245, 245, 245))
    return img


def draw_plan(im, plan_pos, m):
    p = plan_pos[plan_pos[:, 0] > 1.0]
    if len(p) < 2:
        return
    xyz = np.stack([p[:, 1], np.full(len(p), CAM_H), p[:, 0]], 1)      # device frame (x fwd, y right) -> OpenCV view
    uv = xyz @ G.OP_K[m].T
    uv = (uv[:, :2] / uv[:, 2:]).round().astype(np.int32)
    cv2.polylines(im, [uv], False, (0, 220, 255), 2, cv2.LINE_AA)


def make(a):
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    df, past, fut, calib, cands = candidates()
    seq, ev = (a.seq, a.frame) if a.seq else cands[0][:2]
    rows = df[df.sequence.astype(str) == seq].sort_values("frame")
    pos = {int(f): i for i, f in zip(rows.index, rows.frame)}
    f0, f1 = max(int(rows.frame.min()), ev - a.before), min(int(rows.frame.max()), ev + a.after)
    fw = max(int(rows.frame.min()), f0 - 100)
    idx = gather_maps(calib[seq])
    dev = np.array(calib[seq]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
    m = OPModel("cinque", a.backend)
    m.reset()
    frames, log = [], []
    for f in range(fw, f1 + 1):
        if f not in pos:
            raise SystemExit(f"{seq}: frame {f} missing in the index")
        r = df.loc[pos[f]]
        packed, rgb, front = read_frame(r, idx)
        for _ in range(2):
            raw = m.step(packed, action_t=(0.275, 0.525))
        if f < f0:
            continue
        i = df.index.get_loc(pos[f])
        v = float(np.linalg.norm(past[i, -1, 2:4]))
        d = decode(raw, m.slices, v, (0.275, 0.525))
        wp = Z.openpilot_to_wod(d["plan_pos"], d["plan_yaw"], T_IDXS, dev)
        road, wide = rgb["road"].copy(), rgb["wide"].copy()
        for im, k in ((road, "road"), (wide, "wide")):
            draw_plan(im, d["plan_pos"], k)
            put(im, f"openpilot {k} input", (6, 18), 0.45)
        fr = front.copy()
        put(fr, "WOD FRONT camera (source; no third-person view in WOD)", (6, 20), 0.5)
        put(fr, f"t {(f - ev) / 10:+5.1f} s from launch   v {v:4.1f} m/s", (6, 44), 0.5)
        put(fr, f"action: curvature {d['curvature']:+.3f} 1/m  accel {d['accel']:+.2f} m/s2", (6, 68), 0.5)
        put(fr, f"plan v(+1 s) {float(np.interp(1.0, T_IDXS, np.linalg.norm(d['plan_vel'][:, :2], axis=1))):4.1f} m/s", (6, 92), 0.5)
        panel = np.concatenate([fr, bev(past[i], fut[i], wp), np.concatenate([road, wide], 0)], 1)
        top = np.full((30, panel.shape[1], 3), 30, np.uint8)
        put(top, f"WOD-E2E val {seq} frames {f0}-{f1} | shipped Cinque, open loop (plan scored, nothing executed) | 10 Hz frame fed twice, "
                 f"zero state {(f0 - fw) / 10:.0f} s before", (8, 21), 0.5, (255, 255, 255), (30, 30, 30))
        frames.append(np.concatenate([top, panel], 0))
        log.append(dict(frame=f, v=v, curvature=d["curvature"], accel=d["accel"], plan_y5=float(wp[-1, 1]), fut_y5=float(fut[i, -1, 1])))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out, a.max_mb, 100)
    json.dump(dict(seq=seq, event=ev, f0=f0, f1=f1, warm_from=fw, backend=a.backend, steps=log), open(out.with_suffix(".json"), "w"), indent=1)
    print(out, out.stat().st_size, len(frames))


def save_gif(frames, out, max_mb, ms):
    for scale, ncol in ((0.6, 128), (0.52, 96), (0.45, 80), (0.4, 64), (0.34, 64)):
        ims = [Image.fromarray(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)) for f in frames]
        pal = ims[len(ims) // 2].quantize(colors=ncol, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        q[0].save(out, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=True)
        if out.stat().st_size < max_mb * 1e6:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "make"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--seq", default="")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--before", type=int, default=30, help="shown frames before the event")
    ap.add_argument("--after", type=int, default=80)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--max-mb", type=float, default=4.5)
    a = ap.parse_args()
    if a.cmd == "pick":
        for c in candidates()[-1][: a.n]:
            print(*c)
    else:
        make(a)


if __name__ == "__main__":
    main()
