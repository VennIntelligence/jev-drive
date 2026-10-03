"""B2D loss-budget example clips: third-person view (chase camera, or a BEV from privileged.jsonl when the run has no chase
recording) next to openpilot's input frames (road + wide after modeld's warp) with the model's plan and lead overlaid.

The model frames are reconstructed, not dumped: the VLM channel saved the raw CARLA road / wide images of every light request
(VLM_SAVE_FRAMES=1, <attempt>/vlm_frames/<t>_{road,wide}.jpg, RGB JPEG, ~2 Hz while a request is open); they are gathered at the
same nearest-neighbour warp indices modeld / op_arb_server use (jevdrive.openpilot.frames, calibration 0, MEDMODEL_K for road,
SBIGMODEL_K for wide) and shown in RGB (the tensor is YUV of the same pixels; JPEG loss included). Between saved frames the last one
is held and its age is printed. Runs without vlm_frames (drive, pbyp*) get a "model view not saved" bar.

  envs/openpilot python:
  lbx_b2d_clips.py <attempt_dir> <out.gif|out.png> --label L --seg t0:t1[:speedup] ... [--bev] [--width 480] [--fps 8] [--max-mb 3]
  (--seg t:t repeated with a .png output writes a contact sheet of single frames, 3 per row)
"""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/vlm_arb/scripts")]
import vlm_arb_gif as G  # noqa: E402
from jevdrive.openpilot import frames as opf  # noqa: E402

CAM_WH, FOCAL, MOUNT = (1928, 1208), {"road": 2648.0, "wide": 567.0}, (1.779, 0.0, 1.433)   # scripts/zeroshot_rigs.py
KM = {"road": opf.MEDMODEL_K, "wide": opf.SBIGMODEL_K}
IDX = {n: opf._nn_index(opf.get_warp_matrix(np.zeros(3), opf.intrinsics(*CAM_WH, f), n == "wide"), (opf.MODEL_W, opf.MODEL_H), CAM_WH)
       for n, f in FOCAL.items()}
FONT = ImageFont.truetype(G.FONT, 12)
SMALL = ImageFont.truetype(G.FONT, 10)


class Run(G.Attempt):
    def __init__(self, d):
        super().__init__(d)
        fd = self.d / "vlm_frames"
        self.ft = sorted({float(p.name.split("_")[0]) for p in fd.glob("*_road.jpg")}) if fd.is_dir() else []
        self.priv = G.rows(self.d / "privileged.jsonl")
        self.priv_t = [r["t"] for r in self.priv]
        self.route_xy = np.array(json.loads((self.d / "route.json").read_text())["xy"])
        f0 = self.ticks[0]["frame"] - self.ticks[0]["t"] / 0.05       # contacts.jsonl t is the world clock: map frames to scenario t
        self.hits = {}
        for c in self.contacts:
            self.hits.setdefault(c["id"], (c["frame"] - f0) * 0.05)
        self.cache = {}

    def model_frames(self, t):
        i = bisect.bisect_right(self.ft, t + 1e-6) - 1
        if i < 0:
            return None, None
        tq = self.ft[i]
        if tq not in self.cache:
            out = {}
            for n in ("road", "wide"):
                px = np.asarray(Image.open(self.d / "vlm_frames" / ("%08.2f_%s.jpg" % (tq, n))).convert("RGB")).reshape(-1, 3)
                out[n] = px[IDX[n]].reshape(opf.MODEL_H, opf.MODEL_W, 3)
            self.cache = {tq: out}
        return tq, self.cache[tq]


def project(name, X, Y):
    """camera-frame ground point (X forward, Y right, Z = camera height down) -> model-frame pixel"""
    p = KM[name] @ np.array([Y, MOUNT[2], X])
    return p[:2] / p[2]


def model_panel(run, t, name, w):
    tq, fr = run.model_frames(t)
    h = w // 2
    if fr is None:
        im = Image.new("RGB", (w, h), (40, 40, 40))
        ImageDraw.Draw(im).text((6, h // 2 - 6), "%s: no saved frame yet" % name, font=SMALL, fill="white")
        return im
    im = Image.fromarray(fr[name])
    d = ImageDraw.Draw(im)
    p = run.at(run.plans, "t", tq)
    if p:
        pts = [(2.5, 0.0)] + [(x - MOUNT[0], -y) for x, y in p["op_xy"]]          # rig rear axle, y left -> camera X, Y right
        uv = [tuple(project(name, X, Y)) for X, Y in pts if X > 0.5]
        d.line(uv, fill=(0, 255, 90), width=3)
        lead, lp = p.get("lead"), p.get("lp")
        if lead and lp and lp[0] > 0.5 and lead[0][0] > 1:
            x, y = lead[0][0], lead[0][1]
            (u0, v0), (u1, _) = project(name, x, y - 0.9), project(name, x, y + 0.9)
            ht = (u1 - u0) / 1.8 * 1.5
            d.rectangle((u0, v0 - ht, u1, v0), outline=(255, 200, 0), width=2)
            d.text((u0, v0 - ht - 12), "lead %.0fm p%.2f" % (x, lp[0]), font=SMALL, fill=(255, 200, 0))
    im = im.resize((w, h), Image.LANCZOS)
    if t - tq > 1.5:                                  # stale: no request (no saved frame) for a while; the model saw newer frames
        im = Image.blend(im, Image.new("RGB", im.size, (0, 0, 0)), 0.55)
        ImageDraw.Draw(im).text((4, h // 2 - 6), "STALE: no frame saved since %.1fs" % tq, font=SMALL, fill=(255, 90, 90))
    ImageDraw.Draw(im).text((4, 2), "%s input @%.1fs%s" % (name, tq, " (held %.1fs)" % (t - tq) if t - tq > 0.3 else ""),
                            font=SMALL, fill="white", stroke_width=1, stroke_fill="black")
    return im


def bev_panel(run, t, size, span=50.0):
    i = max(bisect.bisect_right(run.priv_t, t + 1e-6) - 1, 0)
    r = run.priv[i]
    ego = r["ego"]
    ex, ey, yaw = ego["xyz"][0], ego["xyz"][1], ego["yaw"]
    s = size / span
    cx, cy = size / 2, size * 0.68

    def to_px(P):
        P = np.asarray(P, float).reshape(-1, 2) - [ex, ey]
        lon = P[:, 0] * math.cos(yaw) + P[:, 1] * math.sin(yaw)
        lat = -P[:, 0] * math.sin(yaw) + P[:, 1] * math.cos(yaw)                  # CARLA: y right, yaw clockwise
        return list(zip(cx + lat * s, cy - lon * s))

    def box(x, y, a, ext):
        c = np.array([[ext[0], ext[1]], [ext[0], -ext[1]], [-ext[0], -ext[1]], [-ext[0], ext[1]]])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return to_px(c @ R.T + [x, y])
    im = Image.new("RGB", (size, size), (28, 30, 34))
    d = ImageDraw.Draw(im)
    near = run.route_xy[np.linalg.norm(run.route_xy - [ex, ey], axis=1) < span]
    if len(near) > 1:
        d.line(to_px(near), fill=(80, 90, 110), width=max(int(3.5 * s), 2))
    tr = np.array([x["truth"][:2] for x in run.ticks if x["t"] <= t])
    if len(tr) > 1:
        d.line(to_px(tr[-400:]), fill=(70, 130, 220), width=1)
    for a in r["actors"]:
        hit = run.hits.get(a["id"])
        col = (255, 60, 60) if hit is not None else (200, 200, 200)
        d.polygon(box(a["xyz"][0], a["xyz"][1], a["yaw"], a["extent"]), outline=col, fill=None if hit is None else (120, 30, 30))
        if hit is not None:
            u, v = to_px([a["xyz"][:2]])[0]
            d.text((u + 4, v), "hit %.1fs" % hit, font=SMALL, fill=col)
    d.polygon(box(ex, ey, yaw, ego["extent"]), fill=(60, 140, 255))
    pc = r.get("pc") or {}
    d.text((4, 2), "BEV @%.1fs, %d m" % (r["t"], span), font=SMALL, fill="white")
    if pc.get("bypass"):
        d.text((4, size - 14), "bypass on%s" % (", gap_open" if pc.get("gap_open") else ""), font=SMALL, fill=(255, 200, 0))
    return im


def compose(run, t, k, label, w, chase_img, bev):
    l1, l2 = G.overlay_text(run, t, k)
    hw = w // 2
    if run.ft:
        mv = Image.new("RGB", (w, hw // 2))
        mv.paste(model_panel(run, t, "road", hw), (0, 0))
        mv.paste(model_panel(run, t, "wide", hw), (hw, 0))
    else:
        mv = Image.new("RGB", (w, 18), (60, 30, 30))
        ImageDraw.Draw(mv).text((6, 3), "model view not saved in this run (third-person only)", font=SMALL, fill="white")
    if chase_img is not None:
        top = chase_img
    else:
        top = Image.new("RGB", (w, hw), (0, 0, 0))
        top.paste(bev_panel(run, t, hw), (0, 0))
        if run.ft:                                   # BEV left, road / wide stacked right
            top.paste(model_panel(run, t, "road", hw), (hw, 0))
            top.paste(model_panel(run, t, "wide", hw), (hw, hw // 2))
            mv = None
    hdr = 34
    im = Image.new("RGB", (w, hdr + top.height + (mv.height if mv else 0)), (12, 16, 24))
    d = ImageDraw.Draw(im)
    d.text((6, 3), label + "  " + l1, font=FONT, fill="white")
    d.text((6, 18), l2, font=FONT, fill=(110, 220, 255))
    im.paste(top, (0, hdr))
    if mv:
        im.paste(mv, (0, hdr + top.height))
    return im


def render(run, out, label, segs, w, fps, bev, max_mb):
    targets = []
    for sg in segs:
        p = sg.split(":")
        t0, t1, k = float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1
        targets += [(t0 + i * k / fps, k) for i in range(int((t1 - t0) * fps / k) + 1)]
    frames = [None] * len(targets)
    if not bev and (run.d / "chase_raw.mp4").exists():
        import av
        idx = {}
        for j, (tt, k) in enumerate(targets):
            idx.setdefault(int(np.abs(run.t - tt).argmin()), []).append(j)
        src = av.open(str(run.d / "chase_raw.mp4"))
        for i, fr in enumerate(src.decode(video=0)):
            if i > max(idx):
                break
            if i in idx:
                img = Image.fromarray(fr.to_ndarray(format="rgb24")).resize((w, int(round(w * 9 / 32)) * 2), Image.LANCZOS)
                for j in idx[i]:
                    frames[j] = compose(run, float(run.t[i]), targets[j][1], label, w, img, False)
        src.close()
    else:
        frames = [compose(run, tt, k, label, w, None, True) for tt, k in targets]
    if out.endswith(".png"):
        fw, fh = frames[0].size
        n = (len(frames) + 2) // 3
        sheet = Image.new("RGB", (3 * fw, n * fh))
        for j, f in enumerate(frames):
            sheet.paste(f, ((j % 3) * fw, (j // 3) * fh))
        sheet.save(out, optimize=True)
        print(out, sheet.size)
        return
    for c, scale in ((96, 1.0), (64, 1.0), (48, 1.0), (48, 0.9), (32, 0.85), (32, 0.75), (24, 0.7)):
        fr = [f.resize((int(f.width * scale) // 2 * 2, int(f.height * scale) // 2 * 2), Image.LANCZOS) if scale < 1 else f for f in frames]
        m = fr[::max(len(fr) // 16, 1)]                         # one global palette from a montage of sampled frames
        mont = Image.new("RGB", (fr[0].width, fr[0].height * len(m)))
        for j, f in enumerate(m):
            mont.paste(f, (0, j * f.height))
        pal = mont.quantize(colors=c, method=Image.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.NONE) for f in fr]
        q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=False, disposal=1)
        if Path(out).stat().st_size <= max_mb * 1e6:
            break
    print("%s: %d frames, %d colors, %dx%d, %.2f MB" % (out, len(q), c, q[0].width, q[0].height, Path(out).stat().st_size / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--seg", action="append", default=[])
    ap.add_argument("--bev", action="store_true")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--max-mb", type=float, default=3.0)
    a = ap.parse_args()
    run = Run(a.attempt)
    for i, th in run.hits.items():
        x = run.at(run.ticks, "t", th)
        print("first contact id=%d at scenario t=%.2f s, ego v=%.1f m/s" % (i, th, x["v"]))
    render(run, a.out, a.label, a.seg, a.width, a.fps, a.bev, a.max_mb)
