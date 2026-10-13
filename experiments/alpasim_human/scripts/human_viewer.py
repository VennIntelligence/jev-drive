#!/usr/bin/env python
"""Window for the human AlpaSim driver ("alpasim human drive (phone wheel)", DISPLAY=:0): the front camera as the driver receives it
with the route projected onto the road (blue, distance dots; the phone's 4 s plan orange) through the session's front-camera
calibration (<live>/camera.json from the driver; for older sessions the scene's start.jsonl of a pai_driver / human run), a turn
banner from the route's shape, a small top-down inset (rig frame), side cameras when rendered, speed, applied steering /
acceleration, the phone panel, scene progress, sim vs wall time, and "DRIVER: HUMAN (phone)". Reads <live>/state.json and *.jpg
(lib/human_driver.py). A left click anywhere writes <live>/start: the held scene starts. Heartbeat <live>/viewer.json. Q / Esc closes.

    /data/envs/carla/bin/python experiments/alpasim_human/scripts/human_viewer.py --dir /data/runs/alpasim_human/live
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pygame

TITLE = "alpasim human drive (phone wheel)"
BLUE, WHITE, GREY, ORANGE, RED, GREEN = (70, 170, 250), (230, 235, 240), (150, 150, 150), (255, 150, 60), (240, 70, 60), (80, 220, 120)
CW, CH, IW, IH = 1600, 1120, 1560, 780
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


def project(pts, cal):
    """Rig-frame points (n, 3), rig origin on the ground -> pixels of the native image (n, 2) for the points in view, else []."""
    P = np.asarray(pts, float).reshape(-1, 3)
    if cal is None or not len(P):
        return np.zeros((0, 2))
    pc = (P - np.asarray(cal["cam_t"])) @ np.asarray(cal["R"])          # R^T (p - t): camera frame x right, y down, z forward
    pc = pc[pc[:, 2] > 1.0]
    sp = cal["spec"]
    cx, cy = sp["c"]
    if sp.get("model") == "ftheta":
        rho = np.hypot(pc[:, 0], pc[:, 1])
        th = np.arctan2(rho, pc[:, 2])
        k = th < sp.get("max_angle", 1.5)
        pc, rho, th = pc[k], np.maximum(rho[k], 1e-9), th[k]
        r = np.polyval(sp["fw"][::-1], th)
        uv = np.stack([cx + r * pc[:, 0] / rho, cy + r * pc[:, 1] / rho], -1)
    else:
        fx, fy = sp["f"]
        uv = np.stack([cx + fx * pc[:, 0] / pc[:, 2], cy + fy * pc[:, 1] / pc[:, 2]], -1)
    w, h = sp["native"]
    return uv[(uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)]


def dense(xy, step=1.0):
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return xy
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    t = np.arange(0, s[-1] + 1e-6, step)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], -1)


def maneuver(route):
    """(code 1 left / 2 right / 3 straight, text, distance m) from the route's shape in the rig frame."""
    r = dense(route, 1.0)
    if len(r) < 6:
        return 3, "FOLLOW THE ROAD", None
    hd = np.unwrap(np.arctan2(*np.diff(r, axis=0)[:, ::-1].T))
    turn = hd[-1] - hd[0]
    if abs(turn) < math.radians(25):
        lat = r[-1, 1]
        return 3, "STRAIGHT" if abs(lat) < 3 else ("KEEP LEFT" if lat > 0 else "KEEP RIGHT"), None
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


def overlay(img, st, cal, font):
    """Route (blue line, 10 m dots) and the 4 s plan (orange) drawn on the road in the native image."""
    if cal is None:
        return
    sc = img.get_width() / cal["spec"]["native"][0]
    for xy, col, wdt in ((st.get("plan") or [], ORANGE, 6), (st.get("route") or [], BLUE, 9)):
        d = dense(xy, 0.5)
        if len(d) < 2:
            continue
        uv = project(np.c_[d, np.zeros(len(d))], cal) * sc
        if len(uv) > 1:
            pygame.draw.lines(img, col, False, [tuple(p) for p in uv], max(2, int(wdt * sc)))
    d = dense(st.get("route") or [], 0.5)
    if len(d):
        dist = np.hypot(d[:, 0], d[:, 1])
        for m in range(10, 130, 10):
            k = int(np.argmin(np.abs(dist - m)))
            if abs(dist[k] - m) < 0.6:
                uv = project([[d[k, 0], d[k, 1], 0.0]], cal) * sc
                if len(uv):
                    u, v = map(int, uv[0])
                    pygame.draw.circle(img, BLUE, (u, v), max(4, int(14 * sc)))
                    pygame.draw.circle(img, WHITE, (u, v), max(4, int(14 * sc)), 2)
                    img.blit(font.render("%d m" % m, True, WHITE), (u + int(18 * sc), v - 12))


def bar(c, x, y, w, h, v, col, centered=False):
    pygame.draw.rect(c, (45, 53, 63), (x, y, w, h))
    if centered:
        v = max(-1.0, min(1.0, v))
        a, b = sorted((x + w // 2, int(x + w // 2 + v * w / 2)))
        pygame.draw.rect(c, col, (a, y, max(b - a, 2), h))
        pygame.draw.line(c, WHITE, (x + w // 2, y - 3), (x + w // 2, y + h + 3), 2)
    else:
        pygame.draw.rect(c, col, (x, y, int(w * max(0.0, min(1.0, v))), h))


def inset(c, st, x0, y0, size, font):
    """Top-down, rig frame (x forward = up, y left = left), ego at the bottom centre, 80 m ahead."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    s.fill((10, 14, 20, 210))
    k, ox, oy = (size - 30) / 80.0, size / 2, size - 20
    P = lambda p: (int(ox - p[1] * k), int(oy - p[0] * k))  # noqa: E731
    for d in (20, 40, 60, 80):
        pygame.draw.circle(s, (50, 60, 72), (int(ox), int(oy)), int(d * k), 1)
    for p in st.get("route") or []:
        pygame.draw.circle(s, BLUE, P(p), 4)
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


def draw(c, st, live, imgs, F, status, cal=None):
    font, big, tag, huge = F
    lab = lambda m, p, col=WHITE, f=None: c.blit((f or font).render(m, True, col), p)  # noqa: E731
    t = tag.render("DRIVER: HUMAN (phone)", True, (0, 0, 0))
    pygame.draw.rect(c, ORANGE, (14, 3, t.get_width() + 20, 42), border_radius=8)
    c.blit(t, (24, 24 - t.get_height() // 2))
    waiting = st.get("waiting") and status == "LIVE"
    if status == "LIVE":
        line, col = ("DRIVING", GREEN) if not waiting else ("WAITING FOR PHONE", RED) if st.get("stale") else \
            ("PHONE OK, CLICK TO START" if st.get("start_mode") == "click" else "PHONE OK, PRESS THROTTLE TO START", ORANGE)
    else:
        try:
            L = json.load(open(os.path.join(live, "launcher.json")))
            line, col = "%s: %s" % (L["state"].upper(), L["msg"]), {"refused": RED, "error": RED, "finished": GREEN}.get(L["state"], ORANGE)
        except (OSError, ValueError, KeyError):
            line, col = "no run: start experiments/alpasim_human/scripts/human_pai.sh or human_nuplan.sh", GREY
        if status == "BETWEEN":
            i, n = st.get("scene_i") or 0, st.get("n_scenes") or 0
            line, col = (("LOADING NEXT SCENE (%d/%d)" % (i + 1, n)) if i < n else "ALL %d SCENES DRIVEN, SCORING" % n), ORANGE
    lab("%s   %s   scene %s of %s   %s" % (line, (st.get("track") or "").upper(), st.get("scene_i", "-"), st.get("n_scenes", "-"),
                                          st.get("scene", "")[-24:]), (t.get_width() + 50, 12), col)
    pygame.draw.rect(c, (0, 0, 0), (20, 52, IW, IH))
    front = imgs.get("front")
    if front is not None:
        f = front.copy()
        try:
            overlay(f, st, cal, big)
        except Exception:  # noqa: BLE001  (a calibration surprise must not take the window down)
            pass
        f = fit(f, IW, IH)
        c.blit(f, (20 + (IW - f.get_width()) // 2, 52 + (IH - f.get_height()) // 2))
    if st.get("route") and status == "LIVE":
        code, text, dist = maneuver(st["route"])
        head = big.render(text + ("" if dist is None else "  in %d m" % dist), True, WHITE)
        bw = head.get_width() + 140
        pygame.draw.rect(c, (0, 0, 0), (20 + (IW - bw) // 2, 58, bw, 84), border_radius=10)
        arrow(c, code, (20 + (IW - bw) // 2 + 60, 100), 34, WHITE if code in (1, 2) else GREY)
        c.blit(head, (20 + (IW - bw) // 2 + 115, 100 - head.get_height() // 2))
    for i, name in enumerate(("left", "right")):
        im = imgs.get(name)
        if im is not None and name in (st.get("frames") or []):
            f = fit(im, 320, 180)
            c.blit(f, (30 if i == 0 else 20 + IW - 10 - f.get_width(), 52 + IH - 10 - f.get_height()))
    inset(c, st, 20 + IW - 250, 152, 240, font)
    msg = None
    if waiting:
        msg = ("WAITING FOR PHONE", RED) if st.get("stale") else \
            ("CLICK TO START" if st.get("start_mode") == "click" else "PRESS THROTTLE TO START", ORANGE)
    elif status == "LIVE" and st.get("stale") and st.get("started"):
        msg = ("PHONE SILENT: BRAKING", RED)
    elif status != "LIVE":
        msg = (line[:60], col)
    if msg:
        m = (huge if waiting else big).render(msg[0], True, msg[1])
        pygame.draw.rect(c, (0, 0, 0), (20 + (IW - m.get_width()) // 2 - 20, 52 + IH // 2 - m.get_height() // 2 - 10,
                                        m.get_width() + 40, m.get_height() + 20))
        c.blit(m, (20 + (IW - m.get_width()) // 2, 52 + IH // 2 - m.get_height() // 2))
        if waiting:
            lab("held %.0f s; the scene clock starts on your click (left mouse button, anywhere in this window)" % st.get("wait_s", 0),
                (20 + (IW - 900) // 2, 52 + IH // 2 + 60), ORANGE)
    y0, x1, x2, x3 = 850, 20, 560, 1100
    lab("EGO", (x1, y0), (150, 200, 255))
    lab("%.0f km/h" % (3.6 * (st.get("speed") or 0)), (x1, y0 + 30), WHITE, big)
    lab("applied: wheel angle %+.1f deg (+ = left)" % (st.get("delta_deg") or 0), (x1, y0 + 90))
    lab("applied: accel %+.1f m/s^2" % (st.get("accel") or 0), (x1, y0 + 120))
    lab("PHONE", (x2, y0), (150, 200, 255))
    for i, (n, k, colr, cen) in enumerate((("steer", "steer_in", BLUE, True), ("throttle", "throttle_in", GREEN, False),
                                           ("brake", "brake_in", RED, False))):
        lab(n, (x2, y0 + 34 + 30 * i))
        bar(c, x2 + 100, y0 + 38 + 30 * i, 400, 18, st.get(k) or 0, colr, cen)
    age = st.get("age_ms")
    lab("input age %s   %s" % ("-" if age is None or age > 5e4 else "%.0f ms" % age, "STALE" if st.get("stale") else "ok"),
        (x2, y0 + 128), RED if st.get("stale") else WHITE)
    lab("SCENE / CLOCK", (x3, y0), (150, 200, 255))
    tot = st.get("sim_total") or 0
    lab("sim %.1f s of ~%.1f s" % (st.get("sim_t") or 0, tot), (x3, y0 + 34))
    bar(c, x3, y0 + 66, 460, 18, (st.get("sim_t") or 0) / tot if tot else 0, BLUE)
    w = st.get("wall_per_sim")
    lab("wall s per sim s %s (slow motion %s)" % ("-" if w is None else "%.2f" % w, "-" if w is None else "%.1fx" % w),
        (x3, y0 + 96), ORANGE if w and w > 1.3 else WHITE)
    u = st.get("unpaced_per_sim")
    lab("paced to %sx real time   unpaced %s   late %s   start: %s" % (st.get("realtime", "-"), "-" if u is None else "%.2f" % u,
                                                                     st.get("late", 0), st.get("start_mode", "-")), (x3 - 40, y0 + 126), GREY)
    lab("blue: route on the road (dots every 10 m; the simulator sends it 40-80 m ahead)   orange: where your inputs take you in 4 s",
        (20, 1088), (200, 200, 200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--window-size", nargs=2, type=int, default=(1600, 1120))
    ap.add_argument("--snapshot", help="save one rendered frame of a live scene and exit")
    a = ap.parse_args()
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(tuple(a.window_size), pygame.RESIZABLE)
    pygame.display.set_caption(TITLE)
    c = pygame.Surface((CW, CH))
    F = tuple(pygame.font.SysFont("DejaVu Sans", s, bold=b) for s, b in ((22, False), (44, True), (30, True), (90, True)))
    clock, cache, st, st_m, beat, calib = pygame.time.Clock(), {}, {}, None, 0.0, Calib(a.dir)
    os.makedirs(a.dir, exist_ok=True)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit()
                return
            if e.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                with open(os.path.join(a.dir, "start"), "w") as fh:
                    fh.write(json.dumps({"t": time.time()}))
        if time.time() - beat > 1:
            beat = time.time()
            tmp = os.path.join(a.dir, "viewer.tmp")
            with open(tmp, "w") as fh:
                json.dump({"t": beat, "pid": os.getpid()}, fh)
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
        c.fill((17, 21, 29))
        draw(c, st, a.dir, imgs, F, status, calib.get(st.get("scene")) if st.get("scene") else None)
        r = min(screen.get_width() / CW, screen.get_height() / CH)
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.smoothscale(c, (max(1, int(CW * r)), max(1, int(CH * r)))), (0, 0))
        pygame.display.flip()
        if a.snapshot and status == "LIVE":
            pygame.image.save(c, a.snapshot)
            pygame.quit()
            return
        clock.tick(20)


if __name__ == "__main__":
    main()
