"""B2D loss-budget example clips from the reruns of lbx_b2d_rerun.py: third-person chase view | openpilot road input | openpilot wide input.

The two model panels are the frames the network actually received (dumped by lbx_dump_server.py after modeld's warp, YUV -> RGB, 512 x 256 each),
with the model's own outputs drawn on them (no ground truth): green = plan (pos over the next 10 s, x fwd / y right), cyan = lane lines (drawn when lane prob > 0.5), red = road edges, yellow box = first lead (prob > 0.5). Ground-plane projection with the model calibration (zero), camera height 1.433 m.

Dump records carry no clock (meta has no "t"): record k of the route session (the cid with most records) is plan request 2k + off of ticks.jsonl;
off in 0..3 is chosen by the speed match (mean |v_dump - v_tick|, printed).

  envs/openpilot python:
  lbx_b2d_view.py <attempt_dir> <dump_dir> <out.gif|out.jpg> --label L --seg t0:t1[:speedup] ... [--key t] [--width 512] [--fps 8] [--max-mb 3]
  (--key t writes one JPG frame at scenario time t instead of a GIF)
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lbx_b2d_clips as C  # noqa: E402  (Run: attempt + contacts, project(), KM, MOUNT)

G = C.G
X_IDXS = 192.0 * (np.arange(33) / 32.0) ** 2          # openpilot lane line / road edge distance grid (m)


class View(C.Run):
    def __init__(self, d, dump):
        super().__init__(d)
        dump = Path(dump)
        recs = [json.loads(l) for l in open(dump / "dump.jsonl")]
        cids = {}
        for r in recs:
            cids[r["cid"]] = cids.get(r["cid"], 0) + 1
        cid = max(cids, key=cids.get)
        self.recs = sorted((r for r in recs if r["cid"] == cid), key=lambda r: r["seq"])
        self.dump = dump
        tv = np.array([x["v"] for x in self.ticks])
        dv = np.array([r["v"] for r in self.recs])
        err = {}
        for off in range(4):
            n = min(len(dv), (len(tv) - off) // 2)
            err[off] = float(np.abs(dv[:n] - tv[off:off + 2 * n:2]).mean())
        self.off = min(err, key=err.get)
        print("dump alignment: %d records (cid %d), off=%d, mean |dv|=%.3f m/s (%s)" % (
            len(self.recs), cid, self.off, err[self.off], {k: round(v, 3) for k, v in err.items()}))
        self.rt = np.array([self.ticks[min(2 * k + self.off, len(self.ticks) - 1)]["t"] for k in range(len(self.recs))])
        self.img = {}

    def rec_at(self, t):
        return max(int(np.searchsorted(self.rt, t + 1e-6) - 1), 0)

    def frames(self, k):
        if k not in self.img:
            im = np.asarray(Image.open(self.dump / ("%07d.jpg" % self.recs[k]["seq"])).convert("RGB"))
            self.img = {k: dict(road=im[:, :512], wide=im[:, 512:])}
        return self.img[k]


def poly(d, name, X, Y, col, w, alpha=1.0):
    if alpha < 0.5:
        return
    pts = [tuple(C.project(name, x, y)) for x, y in zip(X, Y) if x > 1.0]
    pts = [p for p in pts if abs(p[0]) < 4000 and abs(p[1]) < 4000]
    if len(pts) > 1:
        d.line(pts, fill=tuple(int(c * alpha) for c in col), width=w)


def panel(run, k, name, w):
    r = run.recs[k]
    im = Image.fromarray(run.frames(k)[name].copy())
    d = ImageDraw.Draw(im)
    for i, ll in enumerate(r["lane_lines"] or []):
        poly(d, name, X_IDXS, ll, (0, 230, 255), 2, r["lane_prob"][i])
    for e in r["road_edges"] or []:
        poly(d, name, X_IDXS, e, (255, 60, 60), 2)
    pos = np.array(r["pos"])
    poly(d, name, pos[:, 0], pos[:, 1], (0, 255, 90), 4)
    lead = r["lead"]
    if r["lp"] > 0.5 and lead[0] > 1:
        (u0, v0), (u1, _) = C.project(name, lead[0], lead[1] - 0.9), C.project(name, lead[0], lead[1] + 0.9)
        ht = (u1 - u0) / 1.8 * 1.5
        d.rectangle((u0, v0 - ht, u1, v0), outline=(255, 200, 0), width=2)
        d.text((u0, v0 - ht - 12), "lead %.0fm p%.2f" % (lead[0], r["lp"]), font=C.SMALL, fill=(255, 200, 0))
    im = im.resize((w, w // 2), Image.LANCZOS)
    ImageDraw.Draw(im).text((4, 2), "%s input @%.1fs" % (name, run.rt[k]), font=C.SMALL, fill="white", stroke_width=1, stroke_fill="black")
    return im


def compose(run, t, sp, label, ch, chase_img):
    l1, l2 = G.overlay_text(run, t, sp)
    k = run.rec_at(t)
    pw = chase_img.height // 2 * 2
    pw = pw if pw % 2 == 0 else pw + 1
    road, wide = panel(run, k, "road", pw), panel(run, k, "wide", pw)
    hdr = 34
    im = Image.new("RGB", (chase_img.width + pw, hdr + chase_img.height), (12, 16, 24))
    d = ImageDraw.Draw(im)
    d.text((6, 3), label + "  " + l1, font=C.FONT, fill="white")
    d.text((6, 18), l2, font=C.FONT, fill=(110, 220, 255))
    im.paste(chase_img, (0, hdr))
    im.paste(road, (chase_img.width, hdr))
    im.paste(wide, (chase_img.width, hdr + road.height))
    return im


def render(run, out, label, segs, w, fps, max_mb, key=None):
    import av
    targets = []
    if key is not None:
        targets = [(key, 1)]
    for sg in segs:
        p = sg.split(":")
        t0, t1, s = float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1
        targets += [(t0 + i * s / fps, s) for i in range(int((t1 - t0) * fps / s) + 1)]
    ch_h = int(round(w * 9 / 16 / 2) * 2)
    frames = [None] * len(targets)
    idx = {}
    for j, (tt, s) in enumerate(targets):
        idx.setdefault(int(np.abs(run.t - tt).argmin()), []).append(j)
    src = av.open(str(run.d / "chase_raw.mp4"))
    for i, fr in enumerate(src.decode(video=0)):
        if i > max(idx):
            break
        if i in idx:
            img = Image.fromarray(fr.to_ndarray(format="rgb24")).resize((w, ch_h), Image.LANCZOS)
            for j in idx[i]:
                frames[j] = compose(run, float(run.t[i]), targets[j][1], label, ch_h, img)
    src.close()
    if key is not None:
        frames[0].save(out, quality=90)
        print(out, frames[0].size, "%.2f MB" % (Path(out).stat().st_size / 1e6))
        return
    for c, scale in ((96, 1.0), (64, 1.0), (48, 1.0), (48, 0.9), (32, 0.85), (32, 0.75), (24, 0.7)):
        fr = [f.resize((int(f.width * scale) // 2 * 2, int(f.height * scale) // 2 * 2), Image.LANCZOS) if scale < 1 else f for f in frames]
        m = fr[::max(len(fr) // 16, 1)]
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
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--seg", action="append", default=[])
    ap.add_argument("--key", type=float)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max-mb", type=float, default=3.0)
    a = ap.parse_args()
    run = View(a.attempt, a.dump)
    for i, th in run.hits.items():
        print("first contact id=%d at scenario t=%.2f s, ego v=%.1f m/s" % (i, th, run.at(run.ticks, "t", th)["v"]))
    render(run, a.out, a.label, a.seg, a.width, a.fps, a.max_mb, a.key)
