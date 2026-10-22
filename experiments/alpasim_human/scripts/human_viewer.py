#!/usr/bin/env python
"""Window for the human AlpaSim driver ("alpasim human drive (phone wheel)", DISPLAY=:0): the front camera as the driver receives it
with the route projected onto the road (blue, dots every 10 m, the nearest visible dot labelled; the phone's 4 s plan orange) through
the session's front-camera calibration (<live>/camera.json from the driver; for older sessions the scene's start.jsonl of a pai_driver /
human run), a turn banner from the route's shape, a small top-down inset (rig frame), side cameras when rendered, a compact strip
under the image (speed, applied steering / acceleration, phone bars, scene clock), and "DRIVER: HUMAN (phone)". Reads
<live>/state.json and *.jpg (lib/human_driver.py). A left click anywhere writes <live>/start: the held scene starts. Heartbeat
<live>/viewer.json (with the shadow toggle). Q / Esc closes, F11 toggles fullscreen. While the window has focus the keyboard drives too
(arrows / WASD, Space handbrake; experiments/phone_wheel/lib/keyboard_wheel.py, merged with the phone in the wheel daemon).
"HARMONIZER ON" shows when the stack renders with the diffusion harmonizer.
Model mode (README "Model mode"; state.json "driver" = a tag): the tag reads "DRIVER: MODEL <tag>" (teal) and the image area shows the
model's own view by default: the widest frame the policy gets, the openpilot wide model frame at the decision time t0 as the model driver
fed it (<live>/model.jpg, model.json from lib/model_view.py), scaled into the same image area as the human view with the same overlays
(route with its 10 m dots, the served trajectory orange, the shadow) plus the model's own plan before any serving switch (yellow).
M toggles back to the raw front camera (the human's view).

Layout: derived from the window's aspect (canvas 1120 units high). The image takes the full height left by the top bar, the strip and a
one-line legend; the right column is the scene gallery: every finished scene of the panel run (<run>/done/*.json, run from
<live>/loop.json), newest first, as the last front frame before the scene closed (<run>/stacks/<stack>/driver/thumbs/<rollout>.jpg,
written by the driver) with score and why; the mouse wheel scrolls it.
Shadow (display only; G, or the panel's "Shadow: human log", the last change wins): the recorded human log's path ahead coloured by
the log speed (violet slow, magenta, yellow fast), a wireframe box where the log ego is at the current sim time, and a readout "LOG x
km/h  YOU y km/h  n m / n s behind (ahead)  n m left (right) of the log path", in the front view and the inset (state.json "shadow").

    /data/envs/carla/bin/python experiments/alpasim_human/scripts/human_viewer.py --dir /data/runs/alpasim_human/live
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../phone_wheel/lib"))
from keyboard_wheel import KeyboardWheel, pygame_keys  # noqa: E402

TITLE = "alpasim human drive (phone wheel)"
BLUE, WHITE, GREY, ORANGE, RED, GREEN = (70, 170, 250), (230, 235, 240), (150, 150, 150), (255, 150, 60), (240, 70, 60), (80, 220, 120)
GHOST, LBL, TEAL, YEL = (235, 215, 255), (150, 200, 255), (40, 210, 190), (255, 225, 80)
SPEED = ((0, (150, 60, 220)), (25, (225, 70, 200)), (50, (255, 215, 90)), (70, (255, 245, 170)))   # km/h -> shadow colour
CH, GW_MIN, INFO, LEG = 1120, 280, 118, 30
CALIB_GLOBS = ("/data/runs/alpasim_human/*/driver/start.jsonl", "/data/runs/alpasim/pai1/runs/*/driver/start.jsonl",
               "/data/runs/alpasim/*/runs/*/driver/start.jsonl")


class Calib:
    """scene -> front camera calibration (spec, R, cam_t); <live>/camera.json first, start.jsonl files as the fallback."""
    def __init__(self, live):
        self.live, self.db, self.m = live, None, None

    def get(self, scene):
        try:
            c = json.load(open(os.path.join(self.live, "camera.json")))
            if c.get("scene") == scene and "spec" in c:
                return c
        except (OSError, ValueError):
            pass
        if self.db is None:
            self.db = {}
            for f in sorted(set(x for g in CALIB_GLOBS for x in glob.glob(g))):
                try:
                    for line in open(f):
                        r = json.loads(line)
                        if "spec" in r and r.get("scene"):
                            self.db[r["scene"]] = r
                except (OSError, ValueError):
                    pass
        return self.db.get(scene)


def uvs(pts, cal):
    """Rig-frame points (n, 3), rig origin on the ground -> pixels of the native image (n, 2), NaN where not in view."""
    P = np.asarray(pts, float).reshape(-1, 3)
    pc = (P - np.asarray(cal["cam_t"])) @ np.asarray(cal["R"])          # R^T (p - t): camera frame x right, y down, z forward
    sp = cal["spec"]
    cx, cy = sp["c"]
    with np.errstate(all="ignore"):
        if sp.get("model") == "ftheta":
            rho = np.maximum(np.hypot(pc[:, 0], pc[:, 1]), 1e-9)
            th = np.arctan2(rho, pc[:, 2])
            r = np.polyval(sp["fw"][::-1], th)
            uv, bad = np.stack([cx + r * pc[:, 0] / rho, cy + r * pc[:, 1] / rho], -1), th >= sp.get("max_angle", 1.5)
        else:
            fx, fy = sp["f"]
            uv, bad = np.stack([cx + fx * pc[:, 0] / pc[:, 2], cy + fy * pc[:, 1] / pc[:, 2]], -1), np.zeros(len(P), bool)
        w, h = sp["native"]
        bad |= (pc[:, 2] <= 1.0) | ~((uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h))
    uv[bad] = np.nan
    return uv


def dense(xy, step=1.0):
    """Resample a polyline (n, >= 2 columns, arc length from the first two) every `step` m; the other columns are interpolated."""
    xy = np.asarray(xy, float)
    xy = xy.reshape(-1, xy.shape[-1] if xy.ndim == 2 else 2)
    if len(xy) < 2:
        return xy
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy[:, :2], axis=0).T))]
    t = np.arange(0, s[-1] + 1e-6, step)
    return np.stack([np.interp(t, s, xy[:, c]) for c in range(xy.shape[1])], -1)


def maneuver(route):
    """(code 1 left / 2 right / 3 straight, text, distance m) from the route's shape in the rig frame."""
    r = dense(route, 1.0)
    if len(r) < 6:
        return 3, "FOLLOW THE ROAD", None
    hd = np.unwrap(np.arctan2(*np.diff(r, axis=0)[:, ::-1].T))
    turn = hd[-1] - hd[0]
    if abs(turn) < math.radians(25):
        lat = r[-1, 1]
        return (3, "STRAIGHT", None) if abs(lat) < 3 else (1, "KEEP LEFT", None) if lat > 0 else (2, "KEEP RIGHT", None)
    i = int(np.argmax(np.abs(hd - hd[0]) > math.radians(12)))
    return (1, "TURN LEFT", float(np.hypot(*r[i]))) if turn > 0 else (2, "TURN RIGHT", float(np.hypot(*r[i])))


def arrow(surface, code, center, s, color):
    x, y = center
    left = [(x - s, y), (x - s // 4, y - s * 2 // 3), (x - s // 4, y - s // 4), (x + s, y - s // 4), (x + s, y + s // 4),
            (x - s // 4, y + s // 4), (x - s // 4, y + s * 2 // 3)]
    if code == 1:
        pts = left
    elif code == 2:
        pts = [(2 * x - px, py) for px, py in left]
    else:
        pts = [(x, y - s), (x + s * 2 // 3, y - s // 4), (x + s // 4, y - s // 4), (x + s // 4, y + s), (x - s // 4, y + s),
               (x - s // 4, y - s // 4), (x - s * 2 // 3, y - s // 4)]
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (0, 0, 0), pts, 2)


def speed_col(kmh):
    for (a, ca), (b, cb) in zip(SPEED[:-1], SPEED[1:]):
        if kmh <= b:
            f = max(0.0, (kmh - a) / (b - a))
            return tuple(int(x + f * (y - x)) for x, y in zip(ca, cb))
    return SPEED[-1][1]


def ghost_corners(g, z=True):
    """The log ego's box (rear axle origin, 5.2 x 1.9 m, 1.5 m high) at ghost (x, y, z, yaw): (8, 3) bottom then top, or (4, 2)."""
    x, y, gz, yaw = g
    c, s = math.cos(yaw), math.sin(yaw)
    fp = [(-1.1, -0.95), (4.1, -0.95), (4.1, 0.95), (-1.1, 0.95)]
    if not z:
        return [(x + c * a - s * b, y + s * a + c * b) for a, b in fp]
    return [[x + c * a - s * b, y + s * a + c * b, gz + hz] for hz in (0.0, 1.5) for a, b in fp]


def overlay(img, st, uvf, sc, font, shadow_on, lw=None, extra=()):
    """Route (blue line, 10 m dots, the nearest visible one labelled), the 4 s plan (orange), `extra` (xy, colour, width) polylines and the
    shadow drawn into img. uvf: rig-frame points (n, 3) -> pixels of the source image (NaN where not in view); sc: source px -> img px;
    lw: line width scale (default sc: widths are for a 1920 px wide image)."""
    lw = sc if lw is None else lw
    W, H = img.get_size()
    seen = lambda P: (lambda uv: uv[np.isfinite(uv).all(1)])(uvf(np.asarray(P, float).reshape(-1, 3)))  # noqa: E731
    for xy, col, wdt in ((st.get("plan") or [], ORANGE, 6), (st.get("route") or [], BLUE, 9)) + tuple(extra):
        d = dense(xy, 0.5)
        if len(d) < 2:
            continue
        uv = seen(np.c_[d[:, :2], np.zeros(len(d))]) * sc
        if len(uv) > 1:
            pygame.draw.lines(img, col, False, [tuple(p) for p in uv], max(2, int(wdt * lw)))
    sh = st.get("shadow") if shadow_on else None
    if sh and len(sh.get("path") or []) > 1:
        P = dense(sh["path"], 0.5)
        uv = uvf(P[:, :3]) * sc
        for a, b, kmh in zip(uv[:-1], uv[1:], P[:-1, 3]):
            if np.isfinite(a).all() and np.isfinite(b).all():
                pygame.draw.line(img, speed_col(kmh), tuple(a), tuple(b), max(2, int(5 * lw)))
    if sh and sh.get("ghost"):
        uv = uvf(np.asarray(ghost_corners(sh["ghost"]), float)) * sc
        for i, j in [(i, (i + 1) % 4) for i in range(4)] + [(4 + i, 4 + (i + 1) % 4) for i in range(4)] + [(i, i + 4) for i in range(4)]:
            if np.isfinite(uv[i]).all() and np.isfinite(uv[j]).all():
                pygame.draw.line(img, GHOST, tuple(uv[i]), tuple(uv[j]), max(2, int(4 * lw)))
        top = uv[4:][np.isfinite(uv[4:]).all(1)]
        if len(top):
            t = font.render("LOG", True, GHOST)
            img.blit(t, (int(top[:, 0].mean()) - t.get_width() // 2, int(top[:, 1].min()) - t.get_height() - 4))
    d = dense(st.get("route") or [], 0.5)
    if len(d):
        dist, labelled = np.hypot(d[:, 0], d[:, 1]), False
        for m in range(10, 130, 10):
            k = int(np.argmin(np.abs(dist - m)))
            if abs(dist[k] - m) < 0.6:
                uv = seen([[d[k, 0], d[k, 1], 0.0]]) * sc
                if len(uv):
                    u, v = map(int, uv[0])
                    pygame.draw.circle(img, BLUE, (u, v), max(4, int(14 * lw)))
                    pygame.draw.circle(img, WHITE, (u, v), max(4, int(14 * lw)), 2)
                    if not labelled and 0 <= u < W and 0 <= v < H:
                        img.blit(font.render("%d m" % m, True, WHITE), (u + int(18 * lw), v - 12))
                        labelled = True


def model_uv(pts, mj, size=(512, 256)):
    """Rig-frame points (n, 3) -> pixels (n, 2) of the 512 x 256 wide model frame (model_view.py's convention), NaN behind 1 m or far
    outside the frame (as uvs)."""
    P = np.asarray(pts, float).reshape(-1, 3)
    d = (P - np.asarray(mj["cam_t"], float)) * [1.0, -1.0, -1.0]                  # rig (x fwd, y left, z up) -> device (x fwd, y right, z down)
    q = d @ (np.asarray(mj["K_wide"]) @ np.asarray(mj["view_from_device"])).T
    w, h = size
    with np.errstate(all="ignore"):
        uv = q[:, :2] / q[:, 2:]
        uv[(q[:, 2] <= 1.0) | ~np.isfinite(uv).all(1) | ~((uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h))] = np.nan
    return uv


def draw_model(c, R, st, mj, img, big, small, shadow_on):
    """Model mode: the wide model frame at t0 (img, 512 x 256) fitted into the image area R, with the human view's overlays (route,
    served trajectory, shadow) and the model's own plan (yellow)."""
    f = fit(img, R.w, R.h)
    own = [[0.0, 0.0]] + [[p[0], p[1]] for p in mj.get("poses") or []]
    overlay(f, st, lambda P: model_uv(P, mj, img.get_size()), f.get_width() / img.get_width(), big, shadow_on,
            lw=f.get_width() / 1920.0, extra=((own, YEL, 4),))
    x0, y0 = R.x + (R.w - f.get_width()) // 2, R.y + (R.h - f.get_height()) // 2
    c.blit(f, (x0, y0))
    c.blit(small.render("model wide frame at t0 (512 x 256, as fed)", True, TEAL), (x0 + 8, y0 + 6))
    age = time.time() - mj.get("t", 0)
    c.blit(small.render("yellow: the model's own plan   orange: served trajectory   decision %d, %.1f s ago" % (mj.get("n", 0), age),
                        True, TEAL if age < 1.5 else RED), (x0 + 8, y0 + f.get_height() - 26))


def bar(c, x, y, w, h, v, col, centered=False):
    pygame.draw.rect(c, (45, 53, 63), (x, y, w, h))
    if centered:
        v = max(-1.0, min(1.0, v))
        a, b = sorted((x + w // 2, int(x + w // 2 + v * w / 2)))
        pygame.draw.rect(c, col, (a, y, max(b - a, 2), h))
        pygame.draw.line(c, WHITE, (x + w // 2, y - 3), (x + w // 2, y + h + 3), 2)
    else:
        pygame.draw.rect(c, col, (x, y, int(w * max(0.0, min(1.0, v))), h))


def inset(c, st, x0, y0, size, font, shadow_on):
    """Top-down, rig frame (x forward = up, y left = left), ego at the bottom centre, 80 m ahead."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    s.fill((10, 14, 20, 210))
    k, ox, oy = (size - 30) / 80.0, size / 2, size - 20
    P = lambda p: (int(ox - p[1] * k), int(oy - p[0] * k))  # noqa: E731
    for d in (20, 40, 60, 80):
        pygame.draw.circle(s, (50, 60, 72), (int(ox), int(oy)), int(d * k), 1)
    for p in st.get("route") or []:
        pygame.draw.circle(s, BLUE, P(p), 4)
    sh = st.get("shadow") if shadow_on else None
    if sh:
        for p in sh.get("path") or []:
            pygame.draw.circle(s, speed_col(p[3]), P(p), 2)
        if sh.get("ghost"):
            pygame.draw.polygon(s, GHOST, [P(q) for q in ghost_corners(sh["ghost"], z=False)], 2)
    plan = st.get("plan") or []
    if len(plan) > 1:
        pygame.draw.lines(s, ORANGE, False, [P(p) for p in plan], 3)
    pygame.draw.polygon(s, WHITE, [(ox, oy - 10), (ox - 6, oy + 6), (ox + 6, oy + 6)])
    s.blit(font.render("top view, rings 20 m", True, GREY), (6, 4))
    c.blit(s, (x0, y0))


def load(path, cache):
    try:
        m = os.stat(path).st_mtime_ns
    except OSError:
        return None
    if cache.get(path, (None,))[0] != m:
        try:
            cache[path] = (m, pygame.image.load(path).convert())
        except (pygame.error, OSError):
            return cache.get(path, (None, None))[1]
    return cache[path][1]


def fit(img, w, h):
    r = min(w / img.get_width(), h / img.get_height())
    return pygame.transform.smoothscale(img, (int(img.get_width() * r), int(img.get_height() * r)))


def layout(W, H, aspect):
    """(image, strip, gallery) rects of a W x H canvas: the image as large as the strip, the legend and a GW_MIN gallery allow."""
    ih = int(min(H - 52 - INFO - LEG - 8, (W - 56 - GW_MIN) / aspect))
    img = pygame.Rect(20, 52, int(ih * aspect), ih)
    return img, pygame.Rect(20, img.bottom + 6, img.w, INFO), pygame.Rect(img.right + 16, 52, W - img.right - 36, H - 52 - LEG - 6)


class Gallery:
    """Finished scenes of the panel run (<live>/loop.json run -> done/*.json), newest first, with the driver's last front frame."""
    def __init__(self, live):
        self.live, self.t, self.items, self.run, self.scroll, self.cache, self.total = live, 0.0, [], None, 0, {}, 0

    def update(self):
        if time.time() - self.t < 2:
            return
        self.t = time.time()
        try:
            run = json.load(open(os.path.join(self.live, "loop.json"))).get("run")
        except (OSError, ValueError):
            run = None
        out = []
        for f in glob.glob(os.path.join(run or "/nonexistent", "done", "*.json")):
            try:
                d = json.load(open(f))
            except (OSError, ValueError):
                continue
            d["thumb"] = os.path.join(run, "stacks", d.get("stack", ""), "driver", "thumbs", "%s.jpg" % d.get("rollout"))
            out.append(d)
        self.run, self.items = run, sorted(out, key=lambda d: -d.get("t", 0))

    def thumb(self, path, w, h):
        try:
            key = (path, os.stat(path).st_mtime_ns, w)
        except OSError:
            return None
        if key not in self.cache:
            try:
                self.cache[key] = fit(pygame.image.load(path).convert(), w, h)
            except (pygame.error, OSError):
                self.cache[key] = None
        return self.cache[key]

    def draw(self, c, r, font, small):
        pygame.draw.rect(c, (24, 29, 38), r, border_radius=8)
        c.blit(small.render("FINISHED SCENES (%d), newest first" % len(self.items), True, LBL), (r.x + 8, r.y + 6))
        if not self.items:
            c.blit(small.render("no finished scene in this run yet" if self.run else "no panel run", True, GREY), (r.x + 8, r.y + 36))
            return
        tw = r.w - 16
        th, step = tw * 9 // 16, tw * 9 // 16 + 62
        self.total = step * len(self.items)
        self.scroll = max(0, min(self.scroll, self.total - (r.h - 34)))
        c.set_clip(pygame.Rect(r.x, r.y + 30, r.w, r.h - 30))
        y = r.y + 34 - self.scroll
        for d in self.items:
            if r.y < y + step and y < r.bottom:
                im = self.thumb(d["thumb"], tw, th)
                if im is not None:
                    c.blit(im, (r.x + 8, y))
                else:
                    pygame.draw.rect(c, (40, 46, 56), (r.x + 8, y, tw, th))
                    c.blit(small.render("no frame saved", True, GREY), (r.x + 16, y + th // 2 - 10))
                sc = d.get("score") or 0.0
                t = font.render("%.2f %s" % (sc, d.get("why", "")), True, GREEN if sc >= 1 else RED if sc <= 0 else ORANGE)
                c.blit(t if t.get_width() <= tw else small.render("%.2f %s" % (sc, d.get("why", "")), True, GREEN if sc >= 1 else RED if sc <= 0 else ORANGE),
                       (r.x + 8, y + th + 4))
                extra = [d.get("scene", "")[-12:]] + (["MODEL %s" % d["driver"]] if d.get("driver", "human") != "human" else []) + \
                        (["attempt %d" % d["attempt"]] if d.get("attempt", 1) > 1 else []) + \
                        (["harmonized"] if d.get("harmonizer") else []) + (["shadow %.0f s" % d["shadow_s"]] if d.get("shadow_s") else [])
                c.blit(small.render("  ".join(extra), True, GREY), (r.x + 8, y + th + 32))
            y += step
        c.set_clip(None)


def draw(c, st, live, imgs, F, status, cal=None, kb="", shadow_on=False, gal=None, model_view=True, mj=None):
    font, big, tag, huge, small = F
    CW, H = c.get_size()
    front = imgs.get("front")
    R, S, G = layout(CW, H, front.get_width() / front.get_height() if front is not None else 16 / 9.0)
    lab = lambda m, p, col=WHITE, f=None: c.blit((f or font).render(m, True, col), p)  # noqa: E731
    model = (st.get("driver") or "human") != "human" and status == "LIVE"
    mview = model and model_view and mj is not None and imgs.get("model") is not None and mj.get("tag") == st["driver"]
    t = tag.render("DRIVER: MODEL %s" % st["driver"] if model else "DRIVER: HUMAN (phone)", True, (0, 0, 0))
    pygame.draw.rect(c, TEAL if model else ORANGE, (14, 3, t.get_width() + 20, 42), border_radius=8)
    c.blit(t, (24, 24 - t.get_height() // 2))
    xr = CW - 14
    for on, text, colr in ((st.get("harmonizer"), "HARMONIZER ON", GREEN), (shadow_on, "SHADOW ON (G)", GHOST)):
        if on:
            h = tag.render(text, True, (0, 0, 0))
            xr -= h.get_width() + 20
            pygame.draw.rect(c, colr, (xr, 3, h.get_width() + 20, 42), border_radius=8)
            c.blit(h, (xr + 10, 24 - h.get_height() // 2))
            xr -= 10
    waiting = st.get("waiting") and status == "LIVE"
    try:                                                          # the loop's progress (scripts/human_loop.py): state, msg, t0, batch
        L = json.load(open(os.path.join(live, "launcher.json")))
    except (OSError, ValueError):
        L = {}
    fresh = time.time() - L.get("t", 0) < 10
    loading = fresh and L.get("state") == "loading" and status != "LIVE"
    el = " %.0f s" % (time.time() - L["t0"]) if loading and L.get("t0") else ""
    if status == "LIVE":
        line, col = ("SKIPPING", GREY) if st.get("aborted") else ("PAUSED", ORANGE) if st.get("paused") else ("DRIVING", GREEN) if not waiting else \
            ("MODEL STARTING", ORANGE) if model else ("WAITING FOR PHONE", RED) if st.get("stale") else \
            ("PHONE OK, CLICK TO START" if st.get("start_mode") == "click" else "PHONE OK, PRESS THROTTLE TO START", ORANGE)
    elif L.get("state") and L.get("msg"):
        line, col = "%s%s: %s" % (L["state"].upper(), el, L["msg"]), {"refused": RED, "error": RED, "finished": GREEN}.get(L["state"], ORANGE)
        if status == "BETWEEN" and not loading and not L.get("batch"):   # a one-shot terminal stack (no loop progress)
            i, n = st.get("scene_i") or 0, st.get("n_scenes") or 0
            line, col = (("LOADING NEXT SCENE (%d/%d)" % (i + 1, n)) if i < n else "ALL %d SCENES DRIVEN, SCORING" % n), ORANGE
    else:
        line, col = "no run: start experiments/alpasim_human/scripts/human_pai.sh or human_nuplan.sh", GREY
    count = "batch: %d of %d driven" % tuple(L["batch"]) if fresh and L.get("batch") else \
        "scene %s of %s" % (st.get("scene_i", "-"), st.get("n_scenes", "-"))
    lab("%s   %s   %s   %s" % (line, (st.get("track") or "").upper(), count, st.get("scene", "")[-24:] if status == "LIVE" else ""),
        (t.get_width() + 50, 12), col)
    pygame.draw.rect(c, (0, 0, 0), R)
    if mview:
        try:
            draw_model(c, R, st, mj, imgs["model"], big, small, shadow_on and status == "LIVE")
        except Exception:  # noqa: BLE001  (a malformed frame must not take the window down)
            pass
    elif front is not None:
        f = front.copy()
        try:
            if cal is not None:
                overlay(f, st, lambda P: uvs(P, cal), f.get_width() / cal["spec"]["native"][0], big, shadow_on and status == "LIVE")
        except Exception:  # noqa: BLE001  (a calibration surprise must not take the window down)
            pass
        f = fit(f, R.w, R.h)
        c.blit(f, (R.x + (R.w - f.get_width()) // 2, R.y + (R.h - f.get_height()) // 2))
    if model and st.get("harmonizer"):                            # served harmonizer settings: slower than real time (fine for watching)
        h = font.render("model + harmonizer at shipped settings: about 4 wall s per sim s (slow motion)", True, ORANGE)
        pygame.draw.rect(c, (0, 0, 0), (R.x + 4, R.bottom - 66, h.get_width() + 16, 34))
        c.blit(h, (R.x + 12, R.bottom - 62))
    if st.get("route") and status == "LIVE":
        code, text, dist = maneuver(st["route"])
        head = big.render(text + ("" if dist is None else "  in %d m" % dist), True, WHITE)
        bw, cx = head.get_width() + 140, R.centerx
        pygame.draw.rect(c, (0, 0, 0), (cx - bw // 2, R.y + 36, bw, 84), border_radius=10)
        arrow(c, code, (cx - bw // 2 + 60, R.y + 78), 34, WHITE if code in (1, 2) else GREY)
        c.blit(head, (cx - bw // 2 + 115, R.y + 78 - head.get_height() // 2))
    for i, name in enumerate(("left", "right")):
        im = imgs.get(name)
        if im is not None and name in (st.get("frames") or []) and not mview:
            f = fit(im, 320, 180)
            c.blit(f, (R.x + 10 if i == 0 else R.right - 10 - f.get_width(), R.bottom - 10 - f.get_height()))
    isz = max(200, min(300, R.w // 6))
    inset(c, st, R.right - isz - 10, R.y + 100, isz, font, shadow_on and status == "LIVE")
    sh = st.get("shadow") if shadow_on and status == "LIVE" else None
    if sh:
        gap = "log ended" if sh.get("ended") else "past the end of the log" if sh.get("gap_s") is None else \
            "%.0f m / %.1f s %s" % (abs(sh["gap_m"]), abs(sh["gap_s"]), "behind" if sh["gap_m"] > 0 else "ahead")
        text = "LOG %.0f km/h   YOU %.0f km/h   %s   %.1f m %s of the log path" % (
            3.6 * (sh.get("v_log") or 0), 3.6 * (st.get("speed") or 0), gap, abs(sh.get("off_m") or 0),
            "left" if (sh.get("off_m") or 0) > 0 else "right")
        m = tag.render(text, True, GHOST)
        m = m if m.get_width() + 250 < R.w else font.render(text, True, GHOST)
        bw = m.get_width() + 250
        x0, y0 = R.centerx - bw // 2, R.bottom - 64
        pygame.draw.rect(c, (0, 0, 0), (x0, y0, bw, 52), border_radius=10)
        c.blit(m, (x0 + 14, y0 + 26 - m.get_height() // 2))
        for k in range(20):                                       # speed legend: 0 .. 60 km/h
            pygame.draw.rect(c, speed_col(3 * k), (x0 + bw - 226 + 7 * k, y0 + 10, 7, 14))
        lab("0", (x0 + bw - 226, y0 + 26), GREY, small)
        lab("60 km/h", (x0 + bw - 226 + 72, y0 + 26), GREY, small)
    msg = None
    if status == "LIVE" and st.get("aborted"):
        msg = ("SKIPPING THIS SCENE (Next item)", GREY)
    elif status == "LIVE" and st.get("paused"):
        msg = ("PAUSED: resume on the panel", ORANGE)
    elif waiting and model:
        msg = ("MODEL STARTING", ORANGE)
    elif waiting:
        msg = ("WAITING FOR PHONE", RED) if st.get("stale") else \
            ("CLICK TO START" if st.get("start_mode") == "click" else "PRESS THROTTLE TO START", ORANGE)
    elif status == "LIVE" and st.get("stale") and st.get("started") and not model:
        msg = ("PHONE SILENT: BRAKING", RED)
    elif loading:
        msg = ("LOADING" + el, ORANGE)
    elif status != "LIVE":
        msg = (line[:60], col)
    if msg:
        m = (huge if waiting else big).render(msg[0], True, msg[1])
        pygame.draw.rect(c, (0, 0, 0), (R.centerx - m.get_width() // 2 - 20, R.centery - m.get_height() // 2 - 10,
                                        m.get_width() + 40, m.get_height() + 20))
        c.blit(m, (R.centerx - m.get_width() // 2, R.centery - m.get_height() // 2))
        if loading:                                               # what is loading, under the big LOADING n s
            h = font.render(L["msg"][:110], True, ORANGE)
            c.blit(h, (R.centerx - h.get_width() // 2, R.centery + 60))
        if waiting and not model and not st.get("paused") and not st.get("aborted"):
            h = font.render("held %.0f s; the scene clock starts on your %s" % (st.get("wait_s", 0), "click (left mouse button, anywhere in "
                            "this window)" if st.get("start_mode") == "click" else "first throttle press"), True, ORANGE)
            c.blit(h, (R.centerx - h.get_width() // 2, R.centery + 60))
    x1, x2, x3, y0 = S.x, S.x + int(S.w * 0.34), S.x + int(S.w * 0.67), S.y      # the strip: ego | phone | scene clock
    lab("EGO", (x1, y0), LBL)
    lab("%.0f km/h" % (3.6 * (st.get("speed") or 0)), (x1, y0 + 26), WHITE, big)
    if model:
        lab("plan: %+.1f m/s in 1 s" % (st.get("accel") or 0), (x1 + 210, y0 + 30))
        lab("model drive %s ms" % ("-" if st.get("model_ms") is None else "%.0f" % st["model_ms"]), (x1 + 210, y0 + 60))
        lab("MODEL", (x2, y0), LBL)
        lab(st["driver"], (x2, y0 + 26), TEAL, big)
        v = (mj or {}).get("valid") or []
        lab("view: %s (M toggles)   slots valid %d of %d" % ("model wide frame" if mview else "raw camera", sum(v), len(v)), (x2, y0 + 80), GREY, small)
    else:
        lab("wheel %+.1f deg (+ left)" % (st.get("delta_deg") or 0), (x1 + 210, y0 + 30))
        lab("accel %+.1f m/s^2" % (st.get("accel") or 0), (x1 + 210, y0 + 60))
        lab("PHONE", (x2, y0), LBL)
    bw = x3 - x2 - 130
    for i, (n, k, colr, cen) in enumerate((("steer", "steer_in", BLUE, True), ("throttle", "throttle_in", GREEN, False),
                                           ("brake", "brake_in", RED, False)) if not model else ()):
        lab(n, (x2, y0 + 24 + 23 * i), WHITE, small)
        bar(c, x2 + 90, y0 + 28 + 23 * i, bw, 14, st.get(k) or 0, colr, cen)
    age = st.get("age_ms")
    model or lab("input age %s   %s   source %s   %s" % ("-" if age is None or age > 5e4 else "%.0f ms" % age, "STALE" if st.get("stale") else "ok",
                                                st.get("source") or "-", kb), (x2, y0 + 94), RED if st.get("stale") else GREY, small)
    lab("SCENE / CLOCK", (x3, y0), LBL)
    tot = st.get("sim_total") or 0
    lab("sim %.1f s of ~%.1f s" % (st.get("sim_t") or 0, tot), (x3, y0 + 24), WHITE, small)
    bar(c, x3, y0 + 50, S.right - x3, 14, (st.get("sim_t") or 0) / tot if tot else 0, BLUE)
    w = st.get("wall_per_sim")
    lab("wall s per sim s %s (slow motion %s)" % ("-" if w is None else "%.2f" % w, "-" if w is None else "%.1fx" % w),
        (x3, y0 + 70), ORANGE if w and w > 1.3 else WHITE, small)
    u = st.get("unpaced_per_sim")
    lab("paced %sx   unpaced %s   late %s   start: %s" % (st.get("realtime", "-"), "-" if u is None else "%.2f" % u, st.get("late", 0),
                                                         st.get("start_mode", "-")), (x3, y0 + 94), GREY, small)
    lab(("blue: route   orange: the model's served trajectory   yellow: its own plan   M: model view / raw camera   G: shadow (%s)   F11: "
         "fullscreen" if model else "blue: route on the road (dots every 10 m; sent 40-80 m ahead)   orange: your inputs' next 4 s   G: shadow "
         "of the human log, path by log speed + LOG box (%s)   F11: fullscreen") % ("on" if shadow_on else "off"), (20, H - LEG + 4), (200, 200, 200), small)
    if gal is not None:
        gal.draw(c, G, font, small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--window-size", nargs=2, type=int, default=(1600, 1120))
    ap.add_argument("--snapshot", help="save one rendered frame of a live scene and exit")
    a = ap.parse_args()
    pygame.display.init()
    pygame.font.init()
    win = tuple(a.window_size)
    screen = pygame.display.set_mode(win, pygame.RESIZABLE)
    pygame.display.set_caption(TITLE)
    canvas = lambda s: pygame.Surface((max(1200, int(round(CH * s.get_width() / max(1, s.get_height())))), CH))  # noqa: E731
    c = canvas(screen)
    F = tuple(pygame.font.SysFont("DejaVu Sans", s, bold=b) for s, b in ((22, False), (44, True), (30, True), (90, True), (19, False)))
    clock, cache, st, st_m, beat, calib, kb = pygame.time.Clock(), {}, {}, None, 0.0, Calib(a.dir), KeyboardWheel()
    gal, shadow, t_ctl, mview, mj, mj_m = Gallery(a.dir), {"on": False, "panel": None}, 0.0, True, None, None
    os.makedirs(a.dir, exist_ok=True)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                mview = not mview
            if e.type == pygame.KEYDOWN and e.key == pygame.K_g:
                shadow["on"], beat = not shadow["on"], 0.0
            if e.type == pygame.KEYDOWN and e.key == pygame.K_F11:
                full = not (screen.get_flags() & pygame.FULLSCREEN)
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if full else pygame.display.set_mode(win, pygame.RESIZABLE)
                c = canvas(screen)
            if e.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                c = canvas(screen)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button in (4, 5):
                gal.scroll += 120 if e.button == 5 else -120
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                with open(os.path.join(a.dir, "start"), "w") as fh:
                    fh.write(json.dumps({"t": time.time()}))
        kb.update(pygame_keys(pygame))
        if time.time() - t_ctl > 0.5:                             # the panel's option: a change there sets the toggle
            t_ctl = time.time()
            try:
                opt = bool(json.load(open(os.path.join(a.dir, "control.json"))).get("options", {}).get("shadow"))
            except (OSError, ValueError, AttributeError):
                opt = shadow["panel"]
            if opt is not None and opt != shadow["panel"]:
                shadow["panel"], shadow["on"], beat = opt, opt, 0.0
        if time.time() - beat > 1:
            beat = time.time()
            tmp = os.path.join(a.dir, ".viewer.%d.tmp" % os.getpid())
            with open(tmp, "w") as fh:
                json.dump({"t": beat, "pid": os.getpid(), "shadow": shadow["on"]}, fh)
            os.replace(tmp, os.path.join(a.dir, "viewer.json"))
        p = os.path.join(a.dir, "state.json")
        try:
            m = os.stat(p).st_mtime_ns
            if m != st_m:
                st, st_m = json.load(open(p)), m
        except (OSError, ValueError):
            pass
        age = time.time() - st.get("t", 0)
        status = "BETWEEN" if st.get("closed") and age < 120 else ("LIVE" if st and age < 10 and not st.get("closed") else "IDLE")
        imgs = {k: load(os.path.join(a.dir, k + ".jpg"), cache) for k in ("front", "left", "right")} if st else {}
        if (st.get("driver") or "human") != "human":                # model mode: the model's own frames (lib/model_view.py)
            imgs["model"] = load(os.path.join(a.dir, "model.jpg"), cache)
            try:
                m = os.stat(os.path.join(a.dir, "model.json")).st_mtime_ns
                if m != mj_m:
                    mj, mj_m = json.load(open(os.path.join(a.dir, "model.json"))), m
            except (OSError, ValueError):
                pass
        gal.update()
        c.fill((17, 21, 29))
        draw(c, st, a.dir, imgs, F, status, calib.get(st.get("scene")) if st.get("scene") else None, kb.text(), shadow["on"], gal, mview, mj)
        screen.fill((0, 0, 0))
        r = min(screen.get_width() / c.get_width(), screen.get_height() / c.get_height())
        screen.blit(pygame.transform.smoothscale(c, (max(1, int(c.get_width() * r)), max(1, int(c.get_height() * r)))), (0, 0))
        pygame.display.flip()
        if a.snapshot and status == "LIVE" and (not os.environ.get("SNAPSHOT_SIM_T") or (st.get("sim_t") or 0) >= float(os.environ["SNAPSHOT_SIM_T"])):
            pygame.image.save(c, a.snapshot)
            pygame.quit()
            return
        clock.tick(20)


if __name__ == "__main__":
    main()
