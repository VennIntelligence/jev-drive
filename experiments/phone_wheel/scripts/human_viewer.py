#!/usr/bin/env python
"""Window for the human driver: openpilot's wide model frame (as lib/human_agent.py publishes it through
scripts/drive_runtime/preview) with the route command banner, the benchmark route (blue, 10/20/40/80 m dots), ego state
and the phone-wheel block. Read-only, no CARLA connection; Q / Escape closes only this window. Python 3.8, envs/carla.

    PYTHONPATH=scripts python experiments/phone_wheel/scripts/human_viewer.py --dir /data/runs/human_drive/live
"""
import argparse
import os
import sys
import time

import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                                "scripts"))
from drive_runtime.preview import PreviewReader  # noqa: E402

TITLE = "human drive (phone wheel)"
BLUE, WHITE, GREY, ORANGE, RED, GREEN = (70, 170, 250), (230, 235, 240), (150, 150, 150), (255, 150, 60), (240, 70, 60), (80, 220, 120)
CW, CH = 1600, 1120


def arrow(surface, code, center, s, color):
    x, y = center
    left = [(x - s, y), (x - s // 4, y - s * 2 // 3), (x - s // 4, y - s // 4), (x + s, y - s // 4), (x + s, y + s // 4),
            (x - s // 4, y + s // 4), (x - s // 4, y + s * 2 // 3)]
    if code in (1, 5):
        pts = left
    elif code in (2, 6):
        pts = [(2 * x - px, py) for px, py in left]
    else:
        pts = [(x, y - s), (x + s * 2 // 3, y - s // 4), (x + s // 4, y - s // 4), (x + s // 4, y + s), (x - s // 4, y + s),
               (x - s // 4, y - s // 4), (x - s * 2 // 3, y - s // 4)]
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (0, 0, 0), pts, 2)


def bar(canvas, x, y, w, h, value, color, centered=False):
    pygame.draw.rect(canvas, (45, 53, 63), (x, y, w, h))
    if centered:
        v = max(-1.0, min(1.0, value))
        a, b = sorted((x + w // 2, int(x + w // 2 + v * w / 2)))
        pygame.draw.rect(canvas, color, (a, y, max(b - a, 2), h))
        pygame.draw.line(canvas, WHITE, (x + w // 2, y - 3), (x + w // 2, y + h + 3), 2)
    else:
        pygame.draw.rect(canvas, color, (x, y, int(w * max(0.0, min(1.0, value))), h))


def draw(canvas, state, images, font, big, status, tag):
    def label(msg, pos, color=WHITE, f=None):
        canvas.blit((f or font).render(msg, True, color), pos)
    hu = state.get("human") or {}
    t = tag.render("DRIVER: HUMAN (phone)", True, (0, 0, 0))     # unmistakable next to the model's viewer window
    pygame.draw.rect(canvas, ORANGE, (14, 3, t.get_width() + 20, 42), border_radius=8)
    canvas.blit(t, (24, 24 - t.get_height() // 2))
    label("%s   route %s   step %s   sim %.1f s" % (status, state.get("route", "-"), state.get("step", "-"), state.get("sim_t", 0)),
          (t.get_width() + 50, 12))
    W, H = 1560, 780
    if "wide" in images:
        img = images["wide"].copy()
        pts = [tuple(p) for p in state.get("route_px", [])]
        if len(pts) > 1:
            pygame.draw.lines(img, BLUE, False, pts, 4)
        for dm, u, v in state.get("route_marks", []):
            pygame.draw.circle(img, BLUE, (int(u), int(v)), 6)
            img.blit(font.render("%d m" % dm, True, BLUE), (int(u) + 8, int(v) - 12))
        canvas.blit(pygame.transform.smoothscale(img, (W, H)), (20, 48))
    code, dist = state.get("cmd_code", 0), state.get("cmd_dist")
    head = big.render("%s%s" % (state.get("cmd", "-").upper(), "" if dist is None else "  in %d m" % dist), True, WHITE)
    pygame.draw.rect(canvas, (0, 0, 0), (30, 58, max(720, head.get_width() + 150), 90))
    arrow(canvas, code, (95, 103), 40, WHITE if code in (1, 2, 5, 6) else GREY)
    canvas.blit(head, (160, 75))
    if state.get("waiting"):
        msg = big.render("PRESS THROTTLE TO START", True, ORANGE)
        canvas.blit(msg, (20 + (W - msg.get_width()) // 2, 48 + H // 2 - 30))
    if hu.get("stale") and status == "LIVE" and not state.get("waiting"):
        msg = big.render("PHONE SILENT: BRAKING", True, RED)
        canvas.blit(msg, (20 + (W - msg.get_width()) // 2, 48 + H // 2 - 30))
    y0, x1, x2, x3 = 840, 20, 560, 1100
    v = state.get("speed", 0)
    label("EGO", (x1, y0), (150, 200, 255))
    label("%.0f km/h" % (3.6 * v), (x1, y0 + 30), WHITE, big)
    label("yaw rate %.1f deg/s   accel %.1f m/s^2" % (state.get("yaw_rate", 0), state.get("accel", 0)), (x1, y0 + 90))
    label("applied: steer %+.2f  thr %.2f  brk %.2f" % (state.get("steer", 0), state.get("throttle", 0), state.get("brake", 0)),
          (x1, y0 + 120))
    flags = " ".join(f for f, on in (("REVERSE", hu.get("reverse")), ("HANDBRAKE", hu.get("handbrake"))) if on)
    label(flags, (x1, y0 + 150), ORANGE)
    label("PHONE", (x2, y0), (150, 200, 255))
    label("steer", (x2, y0 + 34))
    bar(canvas, x2 + 90, y0 + 38, 400, 18, state.get("steer", 0), BLUE, centered=True)
    label("throttle", (x2, y0 + 64))
    bar(canvas, x2 + 90, y0 + 68, 400, 18, state.get("throttle", 0), GREEN)
    label("brake", (x2, y0 + 94))
    bar(canvas, x2 + 90, y0 + 98, 400, 18, state.get("brake", 0), RED)
    age = hu.get("age_ms")
    label("input age %s   %s" % ("-" if age is None or age > 5e4 else "%.0f ms" % age, "STALE" if hu.get("stale") else "ok"),
          (x2, y0 + 128), RED if hu.get("stale") else WHITE)
    label("raw steer %+.2f  gain %s expo %s   throttle ^%s x%s" % (hu.get("steer_in", 0), hu.get("gain"), hu.get("expo"),
                                                                  hu.get("thr_gamma"), hu.get("thr_gain")), (x2, y0 + 158), GREY)
    label("ROUTE / SIM", (x3, y0), (150, 200, 255))
    p = hu.get("progress", 0) or 0
    label("progress %.0f %% of %s m" % (100 * p, hu.get("route_m")), (x3, y0 + 34))
    bar(canvas, x3, y0 + 66, 460, 18, p, BLUE)
    wps, raw = hu.get("wall_per_sim"), hu.get("raw_per_sim")
    label("wall s per sim s %s (paced %sx)" % ("-" if wps is None else "%.2f" % wps, hu.get("realtime")), (x3, y0 + 96),
          ORANGE if wps and wps > 1.3 else WHITE)
    label("unpaced %s   late %s" % ("-" if raw is None else "%.2f" % raw, hu.get("late", 0)), (x3, y0 + 126), GREY)
    label("blue: benchmark route ahead (flat ground, dots 10/20/40/80 m)   image: openpilot wide camera as the model sees it",
          (20, 1088), (200, 200, 200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--window-size", nargs=2, type=int, default=(1600, 1120), metavar=("WIDTH", "HEIGHT"))
    ap.add_argument("--snapshot", help="save one rendered frame (after step 20) and exit")
    a = ap.parse_args()
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(tuple(a.window_size), pygame.RESIZABLE)
    pygame.display.set_caption(TITLE)
    canvas = pygame.Surface((CW, CH))
    font = pygame.font.SysFont("DejaVu Sans", 22)
    big = pygame.font.SysFont("DejaVu Sans", 44, bold=True)
    tag = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
    clock = pygame.time.Clock()
    reader, packet, state, images, last_draw = None, None, {}, {}, 0.0
    try:
        while True:
            dirty = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return
                if e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                    dirty = True
            if reader is None:
                try:
                    reader = PreviewReader(a.dir)
                except (OSError, ValueError):
                    pass
            fresh = reader.read() if reader else None
            if fresh:
                packet, state, images = fresh, fresh["state"], {}
                for name, spec in packet["images"].items():
                    h, w, _ = spec["shape"]
                    images[name] = pygame.image.frombuffer(memoryview(packet["pixels"])[spec["offset"]:spec["offset"] + spec["size"]],
                                                           (w, h), "RGB")
                dirty = True
            if dirty or time.monotonic() - last_draw > 0.5:
                canvas.fill((17, 21, 29))
                age = time.time() - state.get("t", 0)
                status = ("LIVE" if age < 3 else "STALE") if state.get("running") else ("ENDED" if state else "WAITING FOR A ROUTE")
                draw(canvas, state, images, font, big, status, tag)
                r = min(screen.get_width() / CW, screen.get_height() / CH)
                screen.fill((0, 0, 0))
                screen.blit(pygame.transform.smoothscale(canvas, (max(1, int(CW * r)), max(1, int(CH * r)))), (0, 0))
                pygame.display.flip()
                last_draw = time.monotonic()
                if a.snapshot and state.get("step", 0) > 20:
                    pygame.image.save(canvas, a.snapshot)
                    return
            clock.tick(60)
    finally:
        if reader:
            reader.close()
        pygame.quit()


if __name__ == "__main__":
    main()
