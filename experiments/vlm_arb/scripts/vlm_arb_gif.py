"""Event timelines and GIF clips from the chase recordings of the vlm_arb GIF reruns (envs/openpilot python: needs av, PIL, numpy).

  vlm_arb_gif.py events <attempt_dir>
      official status and infractions, collisions with time / speeds / other actor, ego-light transitions, stops, VLM answer changes
  vlm_arb_gif.py cut <attempt_dir> <out.gif> --label "<arm> r<route> s<seed>" --seg t0:t1[:speedup] [--seg ...] [--width 560] [--fps 8] [--colors 96]
      a clip of the sim-time segments (speedup > 1 skips frames and is printed on the overlay as xN); overlay: sim time, ego speed, true ego light,
      VLM answer in force (answers are used from t_eff), arbitration rows (VLM arms) or bypass state (privileged arms)
"""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_MB = 5.8
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LIGHT = {0: "GREEN", 1: "YELLOW", 2: "RED"}
ANS = {"red_or_yellow_for_ego": "RED/YEL", "green_for_ego": "GREEN", "no_light": "no light", "light_for_other_lane": "other lane", "na": "-"}


def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []


class Attempt:
    def __init__(self, d):
        self.d = Path(d)
        self.ticks, self.vf = rows(self.d / "ticks.jsonl"), rows(self.d / "video_frames.jsonl")
        self.tick_frames = [r["frame"] for r in self.ticks]
        self.t = np.array([self.tick(f)["t"] for f in (r["frame"] for r in self.vf)]) if self.vf else np.zeros(0)
        dec = rows(self.d / "vlm_decisions.jsonl")
        self.ans = [r for r in dec if r.get("k") == "a"]
        self.tab = [r for r in dec if r.get("k") == "s"]
        self.plans = rows(self.d / "plans.jsonl")
        self.plan_t = [r["t"] for r in self.plans]
        self.rec = json.loads((self.d / "results.json").read_text())["_checkpoint"]["records"][0]
        self.contacts = rows(self.d / "contacts.jsonl")
        self.scene = rows(self.d / "scene.jsonl")
        self.cfg = json.loads((self.d / "route_result.json").read_text()).get("config", {}) if (self.d / "route_result.json").exists() else {}
        self.route = self.d.parent.name

    def tick(self, frame):
        return self.ticks[max(bisect.bisect_right(self.tick_frames, frame) - 1, 0)]

    def at(self, lst, key, t):
        ts = [r[key] for r in lst]
        i = bisect.bisect_right(ts, t + 1e-6) - 1
        return lst[i] if i >= 0 else None


def events(a):
    r = a.rec
    print("status:", r["status"], "scores:", r["scores"])
    T = np.array([x["t"] for x in a.ticks])
    xy = np.array([x["truth"][:2] for x in a.ticks])
    for kind, msgs in r["infractions"].items():
        if kind in ("min_speed_infractions",) or not msgs:
            continue
        for m in msgs:
            import re
            g = re.search(r"x=([-\d.]+), y=([-\d.]+)", m)
            if g:
                d = np.linalg.norm(xy - np.array([float(g.group(1)), float(g.group(2))]), axis=1)
                i = int(np.flatnonzero(d <= d.min() + 0.5)[0])
                x = a.ticks[i]
                print("INFRACTION %s: nearest-arrival t=%.1f s v=%.1f m/s ctx=%s | %s" % (kind, x["t"], x["v"], x.get("ctx"), m))
            else:
                print("INFRACTION %s: %s" % (kind, m))
    seen = set()
    for c in a.contacts:
        key = (c["id"], round(c["t"] / 2))
        if key in seen:
            continue
        seen.add(key)
        x = a.tick(c["frame"])
        extra = ""
        sc = next((s for s in a.scene if s["frame"] >= c["frame"]), None)
        if sc:
            hero = next(o for o in sc["actors"] if o["id"] == sc["hero_id"])
            other = next((o for o in sc["actors"] if o["id"] == c["id"]), None)
            if other:
                v = math.hypot(*other["velocity"][:2])
                dx, dy = np.array(other["location"][:2]) - np.array(hero["location"][:2])
                yaw = math.radians(hero["rotation"][2])
                lon, lat = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
                dyaw = (other["rotation"][2] - hero["rotation"][2] + 180) % 360 - 180
                extra = " other v=%.1f m/s, rel pos lon=%.1f lat=%.1f m, yaw diff=%.0f deg" % (v, lon, lat, dyaw)
        print("CONTACT t=%.1f s %s id=%d ego v=%.1f m/s impulse=%.0f%s" % (c["t"], c["type"], c["id"], x["v"], c["impulse"], extra))
    prev = None
    for x in a.ticks:
        ctx = x.get("ctx", {})
        s = (ctx.get("tl_id"), ctx.get("tl")) if "tl" in ctx else None
        if s != prev:
            print("LIGHT t=%.1f s id/state=%s tl_dist=%s v=%.1f" % (x["t"], s, ctx.get("tl_dist"), x["v"]))
            prev = s
    start = None
    for x in a.ticks + [dict(t=1e9, v=9)]:
        if x["v"] < 0.3 and start is None:
            start = x["t"]
        elif x["v"] >= 0.3 and start is not None:
            if x["t"] - start >= 2:
                print("STOP t=%.1f..%.1f s (%.1f s)" % (start, min(x["t"], a.ticks[-1]["t"]), min(x["t"], a.ticks[-1]["t"]) - start))
            start = None
    prev = None
    for x in a.ans:
        s = x["ans"].get("Q_light")
        if s != prev:
            print("VLM t_eff=%.1f s answer=%s (truth tl=%s dist=%s)" % (x["t_eff"], s, x["gt"].get("tl"), x["gt"].get("tl_dist")))
            prev = s
    prev = None
    for x in a.tab:
        s = (tuple(x.get("rules", [])), x.get("release"), x.get("r5"))
        if s != prev:
            print("ROWS t=%.1f s rules=%s release=%s r5=%s v=%.1f" % (x["t"], list(s[0]), s[1], s[2], x["v"]))
            prev = s
    prev = None
    for x in a.plans:
        pc = x.get("pc") or {}
        s = (pc.get("bypass"), pc.get("gap_open"), pc.get("borrow"), pc.get("hold"))
        if s != prev and "bypass" in pc:
            print("PC t=%.1f s bypass=%s gap_open=%s borrow=%s hold=%s v=%.1f" % (x["t"], *s, x["v"]))
            prev = s
    print("duration %.1f s, video frames %d" % (a.ticks[-1]["t"], len(a.vf)))


def overlay_text(a, t, speed):
    x = a.at(a.ticks, "t", t)
    ctx = x.get("ctx", {})
    light = "ego light %s %.0f m" % (LIGHT.get(ctx.get("tl"), "?"), ctx["tl_dist"]) if "tl" in ctx and ctx.get("tl_dist") is not None else "ego light --"
    l1 = "t=%.1f s  v=%.1f km/h%s" % (t, x["v"] * 3.6, "  (x%d speed)" % speed if speed > 1 else "")
    if a.tab:
        s = a.at(a.tab, "t", t)
        ans = a.at(a.ans, "t_eff", t)
        q = ANS.get(ans["ans"].get("Q_light"), "-") if ans else ("none" if not a.ans else "-")
        rows_ = ("+".join(s.get("rules", [])) or "none") + (" release" if s and s.get("release") else "") + (" R5" if s and s.get("r5") else "") if s else "-"
        l2 = "%s | VLM: %s | rows: %s" % (light, q, rows_)
    else:
        p = a.at([dict(t=r["t"], pc=r.get("pc") or {}) for r in a.plans], "t", t)
        pc = p["pc"] if p else {}
        l2 = "%s | bypass %s gap_open %s hold %s" % (light, "on" if pc.get("bypass") else "off", "yes" if pc.get("gap_open") else "no", "yes" if pc.get("hold") else "no")
    return l1, l2


def cut(a, out, label, segs, width, fps, colors):
    h = int(round(width * 9 / 16 / 2) * 2)
    targets = []
    for s in segs:
        p = s.split(":")
        t0, t1, k = float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1
        n = int((t1 - t0) * fps / k) + 1
        targets += [(t0 + i * k / fps, k) for i in range(n)]
    idx = {}
    for j, (tt, k) in enumerate(targets):
        i = int(np.abs(a.t - tt).argmin())
        idx.setdefault(i, []).append(j)
    font = ImageFont.truetype(FONT, 12)
    frames = [None] * len(targets)
    src = av.open(str(a.d / "chase_raw.mp4"))
    last = max(idx)
    for i, fr in enumerate(src.decode(video=0)):
        if i > last:
            break
        if i not in idx:
            continue
        img = Image.fromarray(fr.to_ndarray(format="rgb24")).resize((width, h), Image.LANCZOS)
        for j in idx[i]:
            tt, k = targets[j]
            im = img.copy()
            d = ImageDraw.Draw(im)
            l1, l2 = overlay_text(a, a.t[i], k)
            d.rectangle((0, 0, width, 34), fill=(12, 16, 24))
            d.text((6, 3), label + "  " + l1, font=font, fill="white")
            d.text((6, 18), l2, font=font, fill=(110, 220, 255))
            frames[j] = im
    src.close()
    frames = [f for f in frames if f is not None]
    if out.endswith('.png'):                       # contact sheet of the sampled frames, 3 columns (one --seg t:t per frame)
        rows_n = (len(frames) + 2) // 3
        sheet = Image.new('RGB', (3 * width, rows_n * h))
        for k, f in enumerate(frames):
            sheet.paste(f, ((k % 3) * width, (k // 3) * h))
        sheet.save(out)
        print(out)
        return
    # size cap: fewer colors, then a smaller frame, until the GIF is under MAX_MB
    for c, scale in ((colors, 1.0), (32, 1.0), (24, 1.0), (32, 0.9), (24, 0.85)):
        fr = [f.resize((int(width * scale) // 2 * 2, int(h * scale) // 2 * 2), Image.LANCZOS) if scale < 1 else f for f in frames]
        q = [f.quantize(colors=c, method=Image.MEDIANCUT, dither=Image.NONE) for f in fr]
        q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=1)
        if Path(out).stat().st_size <= MAX_MB * 1e6:
            break
    print("%s: %d frames, %d colors, %dx%d, %.2f MB" % (out, len(q), c, q[0].width, q[0].height, Path(out).stat().st_size / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["events", "cut"])
    ap.add_argument("attempt")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--label", default="")
    ap.add_argument("--seg", action="append", default=[])
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--colors", type=int, default=48)
    n = ap.parse_args()
    att = Attempt(n.attempt)
    events(att) if n.cmd == "events" else cut(att, n.out, n.label, n.seg, n.width, n.fps, n.colors)
