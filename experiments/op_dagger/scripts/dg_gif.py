"""GIF of one reprojection rollout (dg_common engine): shipped vs dg1 after the same closed-loop swerve on a held-out WOD-E2E clip
(decision 132's cswerve arm: heading +beta for 3 steps, the ego ends 0.5 m left of the logged path, then openpilot's lateral path steers).

Per 5 Hz step, one row per model:
  left   BEV in the t0 frame (x up, left to the left): logged path (green), the ego's rollout trace (blue), the model's plan of this step
         (cyan); WOD has no third-person camera, the BEV stands in for it
  right  the road (top) and wide (bottom) model frames the model gets: logged history up to t0, then the logged frame of that time
         re-projected to the ego's offset (jevdrive.op_interp.warp_frame), plan drawn on a flat road at the camera height
The clip is picked automatically: among held-out mid / cruise clips where the swerve is feasible, the one whose shipped - dg1 offset at
step K is closest to the median over those clips (typical, not extreme).

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/op_dagger/scripts/dg_gif.py <out.gif> [--models shipped dg1]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_common as C  # noqa: E402
import dg_roll as DR  # noqa: E402
from jevdrive import camgeom as G  # noqa: E402

PPM, BW, BH = 30.0, 300, 512
GREEN, BLUE, CYAN = (40, 160, 40), (30, 90, 230), (0, 190, 230)     # RGB (frames are RGB)
OFF = 0.5


def put(img, txt, xy, scale=0.45, col=(255, 255, 255), bg=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = xy
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 4), bg, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def rgb(packed):
    """(6, 128, 256) packed YUV420 -> (256, 512, 3) uint8 RGB (BT.601 full range, chroma replicated)."""
    from jevdrive import op_interp as I
    Y, U, V = (x.astype(np.float32) for x in I.unpack(np.asarray(packed)))
    U = U.repeat(2, 0).repeat(2, 1) - 128
    V = V.repeat(2, 0).repeat(2, 1) - 128
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def run(R, S, c, exo):
    """Closed rollout of clip c with exo; per step j = 0..K: frame (2, 6, 128, 256), plan (33, 15), kappa, offset."""
    torch = R.torch
    T9 = R.history(S, [c])
    tc = S.t["tc"][[c]].astype(np.float32)
    ego = C.Ego(S.t["pose"][c], S.t["v"][c], "closed", exo, float(S.t["k0"][c]))
    prev = np.asarray(S.imgs[[c], C.T0])
    steps = []

    def out(T9, j):
        with torch.no_grad():
            o = R.m(T9, torch.ones(1, 9, dtype=torch.bool, device=R.dev), torch.from_numpy(tc).to(R.dev).to(R.dtype))["outputs"]
        o = o.float().cpu().numpy()
        h = R.H(o, S.t["v"][[c], C.T0 + j])
        return o[0, R.H.pi].reshape(33, 15), float(h["kappa"][0])

    plan, k = out(T9, 0)
    steps.append(dict(frame=prev[0], plan=plan, kappa=k, off=(0.0, 0.0, 0.0)))
    for j in range(1, C.K + 1):
        off = ego.advance(j, k)
        cur = C.warp(S.imgs[c, C.T0 + j], S.t["cam"][c], off)[None]
        T9 = torch.cat([T9[:, 1:], R.trunk(prev, cur)[:, None]], 1)
        prev = cur
        plan, k = out(T9, j)
        steps.append(dict(frame=cur[0], plan=plan, kappa=k, off=off))
    return steps


def world(P, j, off):
    """Ego pose (x, y, yaw) in the t0 frame from the logged pose j and the offset against it."""
    c, s = np.cos(P[j, 2]), np.sin(P[j, 2])
    return np.array([P[j, 0] + c * off[0] - s * off[1], P[j, 1] + s * off[0] + c * off[1], P[j, 2] + off[2]])


def bev(P, fut, trace, plan, cam, j, label):
    img = np.full((BH, BW, 3), 245, np.uint8)
    x0 = P[j, 0]                                           # scroll with the log so the ego stays near the bottom
    q = lambda p: np.stack([BW / 2 - p[:, 1] * PPM, 0.8 * BH - (p[:, 0] - x0) * PPM], 1).round().astype(np.int32)  # noqa: E731
    cv2.polylines(img, [q(np.r_[np.zeros((1, 2)), fut])], False, GREEN, 3, cv2.LINE_AA)
    if len(trace) > 1:
        cv2.polylines(img, [q(np.array(trace)[:, :2])], False, BLUE, 3, cv2.LINE_AA)
    e = trace[-1]
    cy, sy = np.cos(e[2]), np.sin(e[2])
    p = plan[plan[:, 0] < 30.0]
    pw = np.stack([e[0] + cy * (cam[0] + p[:, 0]) + sy * p[:, 1], e[1] + sy * (cam[0] + p[:, 0]) - cy * p[:, 1]], 1)   # plan y right
    cv2.polylines(img, [q(pw)], False, CYAN, 2, cv2.LINE_AA)
    car = np.array([[3.8, 0.95], [3.8, -0.95], [-1.0, -0.95], [-1.0, 0.95]])
    cw = np.stack([e[0] + cy * car[:, 0] - sy * car[:, 1], e[1] + sy * car[:, 0] + cy * car[:, 1]], 1)
    cv2.polylines(img, [q(cw)], True, BLUE, 2, cv2.LINE_AA)
    put(img, label, (6, 18), 0.5, (255, 255, 0))
    put(img, "green logged path  blue ego", (6, 40), 0.4, (0, 0, 0), (245, 245, 245))
    put(img, "cyan plan", (6, 58), 0.4, (0, 120, 170), (245, 245, 245))
    cv2.line(img, (8, BH - 12), (8 + int(2 * PPM), BH - 12), (0, 0, 0), 2)
    put(img, "2 m", (8, BH - 18), 0.4, (0, 0, 0), (245, 245, 245))
    return img


def model_panel(frame, plan, cam_h):
    ims = []
    for m, k in enumerate(("road", "wide")):
        im = rgb(frame[m]).copy()
        p = plan[plan[:, 0] > 1.0]
        if len(p) > 1:
            xyz = np.stack([p[:, 1], np.full(len(p), cam_h), p[:, 0]], 1)
            uv = xyz @ G.OP_K[k].T
            uv = (uv[:, :2] / uv[:, 2:]).round().astype(np.int32)
            cv2.polylines(im, [uv], False, CYAN, 2, cv2.LINE_AA)
        put(im, f"openpilot {k} input", (6, 18))
        ims.append(im)
    return np.concatenate(ims, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--models", nargs=2, default=["shipped", "dg1"])
    ap.add_argument("--clip", type=int, default=-1)
    ap.add_argument("--max-mb", type=float, default=4.5)
    a = ap.parse_args()
    S = C.Clips("heldout")
    Rs = [DR.Runner(m, "cuda", 8) for m in a.models]
    cand = [c for c in range(S.n) if S.t["cat"][c] in ("mid", "cruise") and C.swerve_exo(OFF, S.t["v"][c]) is not None]
    if a.clip >= 0:
        c = a.clip
    else:
        d = {}
        for c in cand:
            e = C.swerve_exo(OFF, S.t["v"][c])
            d[c] = abs(run(Rs[0], S, c, e)[-1]["off"][1]) - abs(run(Rs[1], S, c, e)[-1]["off"][1])
        med = float(np.median(list(d.values())))
        c = min(d, key=lambda k: abs(d[k] - med))
        print("clips", len(d), "median shipped - dg1 |dy| at K", round(med, 3), "-> clip", c, S.t["id"][c], flush=True)
    exo = C.swerve_exo(OFF, S.t["v"][c])
    P, cam, fut = S.t["pose"][c], S.t["cam"][c], S.t["fut20"][c][0]
    rolls = [run(R, S, c, exo) for R in Rs]
    frames = []
    for j in range(-C.T0, C.K + 1):                        # logged history (model frames only) then the rollout
        rows = []
        for m, st in zip(a.models, rolls):
            if j < 0:
                fr = np.asarray(S.imgs[c, C.T0 + j])
                trace = [world(P, 0, (0, 0, 0))]
                b = bev(P, fut, trace, st[0]["plan"], cam, 0, f"{m}")
                put(b, f"logged history t {j * C.DT:+.1f} s", (6, 80), 0.45, (255, 255, 255), (90, 90, 90))
                mp = model_panel(fr, np.zeros((0, 15)), cam[2])
            else:
                s = st[j]
                trace = [world(P, i, st[i]["off"]) for i in range(j + 1)]
                b = bev(P, fut, trace, s["plan"], cam, j, f"{m}")
                put(b, f"t +{j * C.DT:.1f} s  dy {s['off'][1]:+.2f} m  dpsi {np.degrees(s['off'][2]):+.1f} deg", (6, 80), 0.42)
                put(b, f"desired curvature {s['kappa']:+.4f} 1/m", (6, 100), 0.42)
                if 1 <= j <= 3:
                    put(b, "swerve: exogenous heading kick", (6, 122), 0.42, (255, 255, 255), (0, 0, 200))
                mp = model_panel(s["frame"], s["plan"], cam[2])
            rows.append(np.concatenate([b, mp], 1))
        top = np.full((30, rows[0].shape[1], 3), 30, np.uint8)
        put(top, f"WOD-E2E held-out {S.t['id'][c][:8]} ({S.t['cat'][c]}) | reprojection, swerve +{OFF} m, op lateral path | 5 Hz",
            (8, 21), 0.48, (255, 255, 255), (30, 30, 30))
        sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
        frames.append(np.concatenate([top, rows[0], sep, rows[1]], 0))
    frames += [frames[-1]] * 5
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for scale, ncol in ((0.7, 128), (0.6, 96), (0.5, 80), (0.42, 64)):
        ims = [Image.fromarray(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)) for f in frames]
        pal = ims[len(ims) // 2].quantize(colors=ncol, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        q[0].save(out, save_all=True, append_images=q[1:], duration=400, loop=0, optimize=True)
        if out.stat().st_size < a.max_mb * 1e6:
            break
    json.dump(dict(clip=int(c), id=str(S.t["id"][c]), cat=str(S.t["cat"][c]), models=a.models,
                   off_K={m: [float(x) for x in st[-1]["off"]] for m, st in zip(a.models, rolls)}), open(out.with_suffix(".json"), "w"), indent=1)
    print(out, out.stat().st_size, len(frames), flush=True)
    DR.bye()


if __name__ == "__main__":
    main()
