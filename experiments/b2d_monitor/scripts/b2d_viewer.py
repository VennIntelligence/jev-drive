#!/usr/bin/env python
"""One input mosaic + chase camera, with model predictions projected onto the road; extended for the B2D monitor (experiments/b2d_monitor):
the openpilot layout (the wide model frame with route / plan / executed path / route-memory overlays, the top bar with model, config and
`phase`) and the route picker that writes <live>/request.json for scripts/viz_loop.py. scripts/b2d_viewer.py is the original.

Read-only separate process (it writes only the picker's request.json and thumbnails); Q/Escape closes only this window. No CARLA
connection. Launcher: experiments/b2d_monitor/scripts/viewer.sh.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import sys

import numpy as np
import pygame

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from drive_runtime.preview import PreviewReader  # noqa: E402


def anchored_waypoints(state):
    # Explicit fixed reference, not a smoothed/recentered model prediction.
    return [[0.0, 0.0]] + state.get('pred_wp', [])


def project_waypoints(state, width, height):
    inverse = np.asarray(state.get('camera_inverse', np.eye(4)))
    focal = width / (2 * math.tan(math.radians(state.get('camera_fov', 90)) / 2))
    points = []
    origin = state.get('prediction_origin', [-1.4, 0.0])
    for forward, right in anchored_waypoints(state):
        forward += origin[0]
        right += origin[1]
        x, y, z, _ = inverse @ np.array([forward, right, 0.15, 1.0])
        if x > 0.1:
            points.append((int(width / 2 + focal * y / x), int(height / 2 - focal * z / x)))
    return points


def draw_bev(state):
    surface = pygame.Surface((440, 480))
    surface.fill((23, 29, 37))
    scale = 14
    def pixel(p):
        return (int(220 + p[1] * scale), int(395 - p[0] * scale))
    for meters in range(-10, 31, 5):
        pygame.draw.line(surface, (45, 53, 63), (0, pixel((meters, 0))[1]), (440, pixel((meters, 0))[1]))
    for meters in range(-15, 16, 5):
        pygame.draw.line(surface, (45, 53, 63), (pixel((0, meters))[0], 0), (pixel((0, meters))[0], 480))
    route = [pixel(p) for p in state.get('route_local', [])]
    if len(route) > 1:
        pygame.draw.lines(surface, (70, 170, 250), False, route, 2)
    pygame.draw.rect(surface, (80, 200, 135), (206, 367, 28, 56), 2)
    pygame.draw.line(surface, (80, 200, 135), (220, 395), (220, 360), 2)
    origin = state.get('prediction_origin', [-1.4, 0.0])
    points = [pixel((p[0] + origin[0], p[1] + origin[1])) for p in anchored_waypoints(state)]
    if len(points) > 1:
        pygame.draw.lines(surface, (255, 220, 40), False, points, 2)
    for point in points[1:]:
        pygame.draw.circle(surface, (255, 220, 40), point, 4)
    pygame.draw.circle(surface, (255, 255, 255), points[0], 5, 2)
    return surface


def chase_waypoints(state, width, height):
    """Never project the current prediction onto a different camera frame."""
    if 'chase_frame' not in state:
        return project_waypoints(state, width, height)
    aligned = state.get('chase_state')
    if not aligned or aligned.get('frame') != state['chase_frame']:
        return []
    return project_waypoints(aligned, width, height)


ROUTE_INFO = {}             # route id -> (town, first scenario type), from --routes-xml


def load_routes(path):
    import xml.etree.ElementTree as ET
    for r in ET.parse(path).getroot().iter('route'):
        sc = [x.get('type') for x in r.iter('scenario')]
        ROUTE_INFO[r.get('id')] = (r.get('town'), sc[0] if sc else '-')
        ROUTE_ORDER.append(r.get('id'))


# ---- openpilot layout (experiments/b2d_monitor/lib/op_arb_viz_agent.py): the wide camera as warped for the model, with what the model is fed
OP_COLORS = {'route': (70, 170, 250), 'plan': (255, 220, 40), 'drive': (80, 220, 120), 'rmem': (235, 80, 235)}


def draw_arrow(surface, code, center, size, color):
    x, y = center
    s = size
    if code in (1, 5):      # left turn / lane change left
        pts = [(x - s, y), (x - s // 4, y - s * 2 // 3), (x - s // 4, y - s // 4), (x + s, y - s // 4),
               (x + s, y + s // 4), (x - s // 4, y + s // 4), (x - s // 4, y + s * 2 // 3)]
    elif code in (2, 6):    # right
        pts = [(2 * x - px, py) for px, py in [(x - s, y), (x - s // 4, y - s * 2 // 3), (x - s // 4, y - s // 4),
               (x + s, y - s // 4), (x + s, y + s // 4), (x - s // 4, y + s // 4), (x - s // 4, y + s * 2 // 3)]]
    else:                   # straight / follow lane
        pts = [(x, y - s), (x + s * 2 // 3, y - s // 4), (x + s // 4, y - s // 4), (x + s // 4, y + s),
               (x - s // 4, y + s), (x - s // 4, y - s // 4), (x - s * 2 // 3, y - s // 4)]
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (0, 0, 0), pts, 2)


BG, FG, HEAD = (17, 21, 29), (230, 235, 240), (150, 200, 255)
PHASE_COLORS = {'LOADING': (255, 210, 90), 'WARM-UP': (255, 150, 60), 'DRIVING': (80, 220, 120), 'STOPPED': (240, 80, 80)}


class Text:
    """Fonts at window scale r (layout units are the old 1600x1120 canvas) and a cache of rendered strings."""
    def __init__(self, r):
        self.r = r
        self.fonts = {False: pygame.font.SysFont('DejaVu Sans', max(8, round(22 * r))),
                      True: pygame.font.SysFont('DejaVu Sans', max(12, round(44 * r)), bold=True)}
        self.img_font = pygame.font.SysFont('DejaVu Sans', 22)      # marks drawn on the 1024x512 frame
        self.cache, self.boxes = {}, {}

    def get(self, msg, color=FG, big=False, width=None):
        key = (msg, color, big, width)
        surf = self.cache.get(key)
        if surf is None:
            if len(self.cache) > 800:
                self.cache.clear()
            font = self.fonts[big]
            surf = font.render(msg, True, color)
            if width is not None and surf.get_width() > width:   # cut to the space left with '...', never scale <= 0
                width = max(1, int(width))
                lo, hi = 0, len(msg)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if font.size(msg[:mid] + '...')[0] <= width:
                        lo = mid
                    else:
                        hi = mid - 1
                surf = font.render(msg[:lo] + '...', True, color) if lo else pygame.Surface((1, max(1, surf.get_height())), pygame.SRCALPHA)
                if surf.get_width() > width:
                    surf = pygame.transform.smoothscale(surf, (width, max(1, surf.get_height())))
            self.cache[key] = surf
        return surf

    def box(self, size, alpha=200):
        if size not in self.boxes:
            self.boxes[size] = pygame.Surface(size, pygame.SRCALPHA)
            self.boxes[size].fill((0, 0, 0, alpha))
        return self.boxes[size]


def phase_of(state):
    """(word, full text) of the agent's phase field (LOADING / WARM-UP <s> / DRIVING / STOPPED <reason>), or None."""
    p = state.get('phase')
    if not p:
        return None
    p = str(p)
    return p.split()[0].upper(), p


def op_regions(r, x0):
    S = lambda v: int(round(v * r))     # noqa: E731
    return {'top': pygame.Rect(0, 0, x0, S(48)), 'image': pygame.Rect(S(20), S(48), S(1560), S(780)),
            'panels': pygame.Rect(0, S(828), x0, S(1120) - S(828))}


def draw_openpilot(dst, state, images, T, status, x0, banner):
    """Straight onto the window at scale T.r: the frame is scaled once into its rectangle, text comes from T's cache.
    banner: None or (title, subtitle, colour) drawn large across the image. Returns the rectangles it changed."""
    r = T.r
    S = lambda v: int(round(v * r))     # noqa: E731
    R = op_regions(r, x0)

    def label(message, position, color=FG, big=False, width=None):
        dst.blit(T.get(message, color, big, width), (S(position[0]), S(position[1])))
    rid = str(state.get('route', '-'))
    town, kind = ROUTE_INFO.get(rid, ('?', '?'))
    dst.set_clip(pygame.Rect(0, 0, x0, dst.get_height()))     # long rows never spill under the picker
    # top bar: DRIVER tag, phase tag, the rest of the line
    dst.fill(BG, R['top'])
    tag = T.get(' DRIVER: %s ' % (state.get('model') or state.get('agent') or '?'), (20, 20, 20), True, width=S(760))
    pygame.draw.rect(dst, (255, 120, 30), (S(14), S(2), tag.get_width() + S(8), S(44)), border_radius=6)
    dst.blit(tag, (S(18), S(-2)))
    x = S(14) + tag.get_width() + S(20)
    ph = phase_of(state)
    if ph:
        pt = T.get(' %s ' % ph[1], (20, 20, 20), width=S(420))
        pygame.draw.rect(dst, PHASE_COLORS.get(ph[0], (200, 200, 200)), (x, S(6), pt.get_width(), S(36)), border_radius=5)
        dst.blit(pt, (x, S(10)))
        x += pt.get_width() + S(16)
    bar = T.get('%s  |  %s %s %s  |  %d %%  |  %s  sim %.0f s%s' % (
        state.get('config') or 'op-drive', rid, town, kind, round(100 * state.get('progress', 0)),
        status, state.get('sim_t', 0), '  WARM-UP' if state.get('warm') and not ph else ''), width=max(1, S(1560) - x))
    dst.blit(bar, (x, S(12)))
    # the frame with its overlays, drawn at 1024x512 and scaled once
    img_rect = R['image']
    if 'wide' in images:
        img = images['wide'].convert()      # display format, own copy: overlays never touch the shared bytes
        for key, width in (('route', 4), ('drive', 3), ('plan', 4)):
            pts = state.get(key + '_px') or []
            if len(pts) > 1:
                pygame.draw.lines(img, OP_COLORS[key], False, pts, width)
        rm, mask = state.get('rmem_px') or [], state.get('rmem_mask')
        if len(rm) > 1:     # route memory: dim over its whole length, bright where the mask is on
            pygame.draw.lines(img, (110, 50, 110), False, rm, 2)
            on = mask if isinstance(mask, list) and len(mask) == len(rm) else [1] * len(rm)
            for a, b, ma, mb in zip(rm, rm[1:], on, on[1:]):
                if ma and mb:
                    pygame.draw.line(img, OP_COLORS['rmem'], a, b, 5)
        for dm, u, v in state.get('route_marks', []):
            pygame.draw.circle(img, OP_COLORS['route'], (int(u), int(v)), 6)
            img.blit(T.img_font.render('%d m' % dm, True, OP_COLORS['route']), (int(u) + 8, int(v) - 12))
        if img_rect.w > 0 and img_rect.h > 0:
            pygame.transform.scale(img, img_rect.size, dst.subsurface(img_rect.clip(dst.get_rect())))
    else:
        dst.fill((40, 44, 52), img_rect)
    # route command banner (top left of the image)
    code, dist = state.get('cmd_code', 0), state.get('cmd_dist')
    head = T.get('%s%s' % (state.get('cmd', '-').upper(), '' if dist is None else '  in %d m' % dist), big=True, width=S(1380))
    pygame.draw.rect(dst, (0, 0, 0), (S(30), S(58), max(S(720), head.get_width() + S(150)), S(118)))
    draw_arrow(dst, code, (S(95), S(117)), S(46), (255, 255, 255) if code in (1, 2, 5, 6) else (150, 150, 150))
    dst.blit(head, (S(160), S(70)))
    d = state.get('desire', 0)
    label('desire fed to model: %d = %s' % (d, state.get('desire_name', '?')), (160, 130),
          (255, 150, 60) if d else (200, 200, 200))
    if banner:              # LOADING / WARM-UP / STOPPED across the image
        title, sub, color = banner
        dst.blit(T.box((S(1560), S(220))), (S(20), S(330)))
        label(title, (60, 360), color, True, S(1500))
        label(sub, (60, 440), FG, False, S(1500))
    # panels below the image
    dst.fill(BG, R['panels'])
    y0, x1, x2, x3 = 840, 20, 560, 1100
    label('EGO (agent pose filter / speedometer)', (x1, y0), HEAD)
    rows = ['speed %.2f m/s  (%.1f km/h)' % (state.get('speed', 0), 3.6 * state.get('speed', 0)),
            'yaw rate %.2f deg/s' % state.get('yaw_rate', 0), 'accel %.2f m/s^2' % state.get('accel', 0),
            'steer %.3f  throttle %.2f  brake %.2f' % (state.get('steer', 0), state.get('throttle', 0), state.get('brake', 0))]
    lon = state.get('lon')
    if lon:                 # LonSmooth (plan-speed tracking): what the longitudinal controller asks and sends
        rows += ['lon [%s]: target %.1f m/s (%s)  plan v(1 s) %.1f  lead cap %.1f' % (lon.get('mode'), lon.get('v_tgt', 0), lon.get('src'),
                                                                                lon.get('vp1', 0), lon.get('v_lead', 0)),
                 'requested accel %s m/s^2' % ('-' if lon.get('a_req') is None else '%.2f' % lon['a_req']),
                 'pedal: throttle %.2f brake %.2f   %s' % (lon.get('thr', 0), lon.get('brk', 0),
                                                       'STOP HOLD %.1f s' % lon.get('hold_t', 0) if lon.get('hold') else 'rolling')]
    for i, row in enumerate(rows):
        label(row, (x1, y0 + 32 + 30 * i), (255, 150, 60) if lon and i == 6 and lon.get('hold') else FG)
    label('SENT TO MODEL (besides road + wide frames)', (x2, y0), HEAD)
    sent = state.get('sent', {})
    rows = ['%s = %s' % (k, sent[k]) for k in sorted(sent) if k != 't']
    pe = state.get('parity_ego') or []
    if pe:                  # op_parity arm: the 20 ego features its bias server turns into intent_bias
        rows = rows[:4] + ['parity ego (%d):' % len(pe), ' '.join('%g' % x for x in pe[:10]), ' '.join('%g' % x for x in pe[10:])]
    for i, row in enumerate(rows[:7]):
        label(row, (x2, y0 + 32 + 30 * i))
    label('MODEL OUTPUT / ARBITRATION', (x3, y0), HEAD)
    s5 = state.get('s5') or {}
    rows = ['lateral: %s%s' % (state.get('lat'), ' (%s)' % state['lat_why'] if state.get('lat_why') else ''),
            'speed source: %s%s' % (state.get('lon_src'), '  LATCHED STOP' if state.get('latch') else ''),
            'lead %.1f m  v %.1f m/s  p %.2f' % (state.get('lead_x', 0), state.get('lead_v', 0), state.get('lead_p', 0)),
            'plan v(0,1,2,3,5 s) %s' % (state.get('vplan'),),
            'curv %s  accel %s' % (state.get('act_k'), state.get('act_a')),
            'vision speed %s  engaged p %s' % (state.get('pose_v'), state.get('eng')),
            '5 s distance: ' + ' '.join('%s %.0f' % (k, v) for k, v in s5.items())]
    for i, row in enumerate(rows):
        label(row, (x3, y0 + 32 + 30 * i))
    label('blue: benchmark route ahead (flat ground, dots 10/20/40/80 m)   yellow: openpilot plan   green: executed path'
          + ('   magenta: route memory (bright = mask on%s)' % ('' if not isinstance(state.get('rmem_mask'), (int, float))
                                                            else ', extent %g' % state['rmem_mask']) if state.get('rmem_px') else ''),
          (20, 1088), (200, 200, 200))
    dst.set_clip(None)
    return list(R.values())


def banner_for(state, loop, loading):
    """The loop's LOADING (the preview still shows the previous route) first, else the agent's phase if not DRIVING."""
    if loading:
        rid = str(loop.get('route', '?'))
        town, kind = ROUTE_INFO.get(rid, ('?', '?'))
        return ('LOADING %s  %s' % (rid, town), '%s   (%s; CARLA map load, agent start)   %.0f s' % (
            kind, loop.get('why', ''), time.time() - loop.get('t', time.time())), PHASE_COLORS['LOADING'])
    ph = phase_of(state)
    if ph and ph[0] != 'DRIVING':
        rid = str(state.get('route', '?'))
        town, kind = ROUTE_INFO.get(rid, ('?', '?'))
        return (ph[1], '%s %s  %s' % (rid, town, kind), PHASE_COLORS.get(ph[0], FG))
    return None


# ---- route picker (right column, drawn at screen resolution): click = next route, double click = abort and run it now.
# Protocol with scripts/viz_loop.py: <live>/request.json {route, mode: next|now, ts} (consumed by the
# loop), <live>/loop.json {route, state, why, next, ids, t} (written by the loop when it starts a route).
ROUTE_ORDER = []            # val XML order
LOOP_IDS = '24721,24944,27005,24416,27870,24519,26023,334,24948,28147,27297'.split(',')   # until loop.json names them


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


class RoutePicker:
    ROW, HEAD, TOP = 66, 34, 96

    def __init__(self, live, xml):
        self.live = Path(live)
        self.runs = self.live.parent / 'runs'      # the loop's output dirs (DS of the last finished attempt per route)
        self.thumb_dir = self.live / 'thumbs'
        self.font = pygame.font.SysFont('DejaVu Sans', 19)
        self.small = pygame.font.SysFont('DejaVu Sans', 15)
        self.bold = pygame.font.SysFont('DejaVu Sans', 19, bold=True)
        self.scroll, self.hits, self.expanded = 0, [], {'loop routes'}
        self.ds, self.thumb_files, self.thumbs, self.saved = {}, set(), {}, set()
        self.version, self.key = 0, None    # the list is redrawn only when what it shows changes
        self.lists = {}             # list name -> ids (registered op_arb lists)
        self.loop, self.last_click = {}, (None, 0.0)
        r = read_json(self.live / 'request.json') or {}       # the viewer is the only writer of request.json
        self.queue = list(r.get('queue', [])) if 'queue' in r else []
        self.written = bool(self.queue)
        self.xml = xml
        threading.Thread(target=self._registered, daemon=True).start()
        threading.Thread(target=self._scan, daemon=True).start()

    def _registered(self):
        try:
            text = subprocess.run(['bash', str(REPO / 'experiments/op_closed_loop/archive/op_arb.sh'),
                                   'routes'], env=dict(os.environ, OP_ARB_DIR='/tmp/op_arb_routes'),
                                  capture_output=True, text=True, timeout=20).stdout
            self.lists = {ln.split()[0]: ln.split()[1].split(',') for ln in text.splitlines() if len(ln.split()) == 2}
        except Exception:   # noqa: BLE001 - the picker still has the loop routes
            pass

    def _scan(self):            # last finished DS per route from the newest run dirs; thumbnails present
        while True:
            ds = {}
            for f in sorted(self.runs.glob('loop-*/attempts/*/*/attempt.json'), key=lambda p: p.stat().st_mtime):
                a = read_json(f)
                r = read_json(f.parent / 'results.json')
                if a and r and a.get('status') == 'finished':
                    try:
                        ds[f.parent.parent.name] = r['_checkpoint']['global_record']['scores_mean']['score_composed']
                    except (KeyError, TypeError):
                        pass
            files = set(p.stem for p in self.thumb_dir.glob('*.png'))
            if ds != self.ds or files != self.thumb_files:
                self.ds, self.thumb_files = ds, files
                self.version += 1
            time.sleep(15)

    def poll(self):             # once a second, small files only
        self.loop = read_json(self.live / 'loop.json') or {}
        done = set((read_json(self.live / 'consumed.json') or {}).get('ts', []))
        q = [e for e in self.queue if e['ts'] not in done]
        if self.written and q and not (self.live / 'request.json').exists():
            q = q[1:]           # a loop started before 2026-10-13 19:00 deletes the file when it takes the head
        if q != self.queue:
            self.queue = q
            self.write_queue()

    def save_thumb(self, rid, img):
        if rid in self.saved or rid in self.thumb_files:
            return
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.thumb_dir / ('.%s.png' % rid)
        pygame.image.save(pygame.transform.smoothscale(img, (256, 128)), str(tmp))
        os.replace(str(tmp), str(self.thumb_dir / ('%s.png' % rid)))
        self.saved.add(rid)
        self.thumb_files.add(rid)
        self.version += 1

    def thumb(self, rid, size):
        if rid not in self.thumb_files:
            return None
        if rid not in self.thumbs:
            try:
                self.thumbs[rid] = pygame.transform.smoothscale(pygame.image.load(str(self.thumb_dir / ('%s.png' % rid))), size)
            except (pygame.error, OSError):
                return None
        return self.thumbs[rid]

    def groups(self):
        loop_ids = self.loop.get('ids') or LOOP_IDS
        tags = {}
        for name, ids in [('loop', loop_ids)] + sorted(self.lists.items()):
            for rid in ids:
                tags.setdefault(rid, []).append(name)
        by_type = {}
        for rid in tags:
            by_type.setdefault(ROUTE_INFO.get(rid, ('?', '?'))[1], []).append(rid)
        out = [('loop routes', loop_ids)] + [(k, by_type[k]) for k in sorted(by_type)]
        return out + [('all val routes', sorted(ROUTE_ORDER, key=lambda r: (ROUTE_INFO[r][1], int(r))))], tags

    def write_queue(self):
        path = self.live / 'request.json'
        if not self.queue:
            if path.exists():
                path.unlink()
            self.written = False
            return
        head = self.queue[0]    # route / mode / ts of the head: what loops started before the queue protocol read
        tmp = self.live / '.request.json'
        tmp.write_text(json.dumps(dict(queue=self.queue, route=head['route'], mode=head['mode'], ts=head['ts'])))
        os.replace(str(tmp), str(path))
        self.written = True

    def request_route(self, rid, mode):
        """next: queue it, or un-queue it if already queued; now: run it at once (the loop aborts the current route)."""
        queued = [e for e in self.queue if e['route'] == rid]
        self.queue = [e for e in self.queue if e['route'] != rid]
        if mode == 'now':
            self.queue.insert(0, {'route': rid, 'mode': 'now', 'ts': time.time()})
        elif not queued:
            self.queue.append({'route': rid, 'mode': 'next', 'ts': time.time()})
        self.write_queue()

    def event(self, ev, x0):
        if ev.type == pygame.MOUSEWHEEL and pygame.mouse.get_pos()[0] >= x0 and self.y0 <= pygame.mouse.get_pos()[1] < self.y1:
            self.scroll = max(0, self.scroll - ev.y * self.ROW)
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[0] >= x0 and self.y0 <= ev.pos[1] < self.y1:
            for rect, kind, key in self.hits:
                if rect.collidepoint(ev.pos):
                    if kind == 'head':
                        self.expanded ^= {key}
                    else:
                        last, t = self.last_click
                        now = time.monotonic()
                        if last == key and now - t < 0.45:
                            self.request_route(key, 'now')
                            now = 0.0       # a third click starts a new pair
                        else:
                            self.request_route(key, 'next')
                        self.last_click = (key, now)
                    return True
        return False

    y0, y1 = 0, 1

    def draw(self, screen, x0, y0, y1, state, loading):
        """Fixed box x0..right edge, y0..y1 (the image band); the list scrolls inside it."""
        cur = str(self.loop.get('route', state.get('route', ''))) if loading else str(state.get('route', ''))
        key = (screen.get_size(), x0, y0, y1, self.scroll, cur, loading, self.version, tuple(sorted(self.expanded)),
               tuple((e['route'], e['mode']) for e in self.queue), self.loop.get('next'), tuple(self.loop.get('ids') or ()),
               len(self.lists))
        if key == self.key:
            return None
        self.key = key
        self.y0, self.y1 = y0, y1
        full = screen
        screen = full.subsurface((0, y0, full.get_width(), y1 - y0))
        w, h = screen.get_width() - x0, screen.get_height()
        pygame.draw.rect(screen, (24, 29, 38), (x0, 0, w, h))
        pygame.draw.line(screen, (60, 70, 85), (x0, 0), (x0, h), 2)
        qpos = dict((e['route'], ('NOW>' if e['mode'] == 'now' else 'Q%d' % (k + 1))) for k, e in enumerate(self.queue))
        groups, tags = self.groups()
        self.hits, y = [], self.TOP - self.scroll
        surf = pygame.Surface((w, h - self.TOP))
        surf.fill((24, 29, 38))
        tw, th = 112, 56
        for name, ids in groups:
            closed = name not in self.expanded
            if -self.HEAD < y - self.TOP < h:
                pygame.draw.rect(surf, (38, 46, 60), (0, y - self.TOP, w, self.HEAD - 4))
                surf.blit(self.bold.render('%s %s (%d)' % ('+' if closed else '-', name, len(ids)), True, (150, 200, 255)),
                          (8, y - self.TOP + 5))
                self.hits.append((pygame.Rect(x0, y0 + max(y, self.TOP), w, self.HEAD - 4), 'head', name))
            y += self.HEAD
            if closed:
                continue
            for rid in ids:
                if -self.ROW < y - self.TOP < h:
                    ry = y - self.TOP
                    bg = ((40, 80, 50) if rid == cur else (90, 70, 20) if rid in qpos else None)
                    if bg:
                        pygame.draw.rect(surf, bg, (2, ry, w - 4, self.ROW - 4))
                    t = self.thumb(rid, (tw, th))
                    if t:
                        surf.blit(t, (6, ry + 3))
                    else:           # placeholder of the same size until live/thumbs/<id>.png exists
                        pygame.draw.rect(surf, (70, 76, 86), (6, ry + 3, tw, th))
                        surf.blit(self.small.render('no frame yet', True, (150, 155, 165)), (14, ry + 22))
                    town, kind = ROUTE_INFO.get(rid, ('?', '?'))
                    ds = self.ds.get(rid)
                    surf.blit(self.font.render('%s  %s' % (rid, town.replace('Town', 'T')), True, (235, 238, 242)), (tw + 14, ry + 2))
                    dtxt = 'DS %.0f' % ds if ds is not None else 'DS -'
                    ds_s = self.font.render(dtxt, True, (120, 220, 140) if ds is not None and ds >= 50 else (240, 170, 90) if ds is not None else (130, 135, 145))
                    surf.blit(ds_s, (w - ds_s.get_width() - 10, ry + 2))
                    mark = ('NOW' if rid == cur else qpos.get(rid, ''))
                    sub = ' '.join(tags.get(rid, [])) or kind[:26]
                    surf.blit(self.small.render(('%s  %s' % (mark, sub)).strip(), True, (170, 175, 185)), (tw + 14, ry + 32))
                    self.hits.append((pygame.Rect(x0, y0 + y, w, self.ROW - 4).clip(pygame.Rect(x0, y0 + self.TOP, w, h - self.TOP)), 'row', rid))
                y += self.ROW
        total = y + self.scroll - self.TOP
        self.scroll = min(self.scroll, max(0, total - (h - self.TOP)))
        screen.blit(surf, (x0, self.TOP))
        # header: what runs, what is next
        screen.blit(self.bold.render('ROUTES  click next | dbl-click now', True, (230, 235, 240)), (x0 + 10, 10))
        nxt = ' '.join(e['route'] for e in self.queue[:4]) or self.loop.get('next', '-')
        line = '%s %s   next %s%s' % ('loading' if loading else 'now', cur or '-', nxt,
                                       ' (queue)' if self.queue else ' (list)')
        screen.blit(self.font.render(line, True, (255, 210, 90) if loading else (200, 205, 215)), (x0 + 10, 40))
        screen.blit(self.small.render('click a queued row again to unqueue', True, (130, 135, 145)), (x0 + 10, 68))
        return pygame.Rect(x0, y0, w, y1 - y0)


class Rates:
    """Sim ticks per wall second from the preview (step, t), the viewer's own redraw rate, ratio to real time (20 Hz)."""
    def __init__(self):
        from collections import deque
        self.sim, self.draws = deque(maxlen=200), deque(maxlen=200)

    def state(self, state):
        if 'step' in state and 't' in state and (not self.sim or self.sim[-1] != (state['t'], state['step'])):
            if self.sim and state['step'] < self.sim[-1][1]:
                self.sim.clear()                # new route
            self.sim.append((state['t'], state['step']))

    def draw(self, screen, x0, h, font, small, render_ms=None):
        now = time.monotonic()
        self.draws.append(now)
        fps = sum(1 for t in self.draws if now - t < 2) / 2
        win = [p for p in self.sim if self.sim[-1][0] - p[0] < 3]
        tps = ((win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0])) if len(win) > 1 and win[-1][0] > win[0][0] else None
        stale = not self.sim or time.time() - self.sim[-1][0] > 3
        pygame.draw.rect(screen, (17, 21, 29), (x0, 0, screen.get_width() - x0, h))
        line = 'sim -' if tps is None or stale else 'sim %.1f ticks/s  =  %.2fx real time' % (tps, tps / 20)
        per = ((win[-1][1] - win[0][1]) / (len(win) - 1)) if tps and not stale and len(win) > 1 else None
        screen.blit(font.render(line, True, (120, 220, 140) if tps and not stale and tps >= 18 else (240, 190, 90)),
                    (x0 + 10, 4))
        # the agent publishes a preview frame every `per` ticks: the viewer cannot show more frames than that
        screen.blit(small.render('viewer %.0f fps%s%s' % (fps, '' if per is None else ', preview every %.0f tick%s' % (
            per, '' if round(per) == 1 else 's'), '' if render_ms is None else ', draw %.0f ms' % render_ms),
            True, (170, 175, 185)), (x0 + 10, 4 + font.get_height()))
        return pygame.Rect(x0, 0, screen.get_width() - x0, h)


def pick_w(screen):
    # Fixed column: the wide image (canvas 20..1580) keeps >= 85 % of its old 2418 px at a 2480 px window.
    return max(300, int(0.157 * screen.get_width()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--snapshot', help='Save one rendered dashboard without grabbing X11')
    parser.add_argument('--follow-runs', help='Follow newest child run/live in a profiling series')
    parser.add_argument('--window-size', nargs=2, type=int, default=(1600, 1120),
                        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--routes-xml', help='Bench2Drive route XML: town and scenario type of the route in the top line')
    args = parser.parse_args()
    if args.routes_xml:
        load_routes(args.routes_xml)
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(tuple(args.window_size), pygame.RESIZABLE)
    pygame.display.set_caption('Bench2Drive TCP | model input + chase trajectory')
    canvas = pygame.Surface((1600, 1120))
    font = pygame.font.SysFont('DejaVu Sans', 22)
    big = pygame.font.SysFont('DejaVu Sans', 44, bold=True)
    clock = pygame.time.Clock()
    picker = RoutePicker(args.dir, args.routes_xml)
    rates = Rates()
    T, last_size, render_ms = None, None, None
    last_poll = 0
    reader, packet = None, None
    state, images = {}, {}
    last_draw = 0
    saved = False
    last_follow = 0

    def label(message, position, color=(230, 235, 240)):
        canvas.blit(font.render(message, True, color), position)

    try:
        running = True
        while running:
            dirty = False
            if args.follow_runs and time.monotonic() - last_follow > 1:
                feeds = list(Path(args.follow_runs).glob('*/live/status.json'))
                if feeds:
                    directory = str(max(feeds, key=lambda p: p.stat().st_mtime).parent)
                    if directory != args.dir:
                        if reader:
                            reader.close()
                        reader = None
                        args.dir = directory
                last_follow = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                                event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    dirty = True
                elif picker.event(event, screen.get_width() - pick_w(screen)):
                    dirty = True
            if time.monotonic() - last_poll > 1:
                picker.poll()
                last_poll = time.monotonic()
            if reader is None:
                try:
                    reader = PreviewReader(args.dir)
                except (OSError, ValueError):
                    pass
            fresh = reader.read() if reader else None
            if fresh:
                packet = fresh  # Own bytes until all frombuffer surfaces have been replaced.
                state = packet['state']
                rates.state(state)
                app = {'TFv6': 'TFv6', 'openpilot': 'openpilot wide (model frame)'}.get(state.get('agent'), 'Bench2Drive TCP')
                pygame.display.set_caption(app + ' | model input + trajectory')
                images = {}
                for name, spec in packet['images'].items():
                    h, w, _ = spec['shape']
                    pixels = memoryview(packet['pixels'])[spec['offset']:spec['offset'] + spec['size']]
                    images[name] = pygame.image.frombuffer(pixels, (w, h), 'RGB')
                dirty = True
            if dirty or time.monotonic() - last_draw > 1:
                t_render = time.perf_counter()
                age = time.time() - state.get('t', 0)
                status = ('LIVE' if age < 3 else 'STALE') if state.get('running') else ('ENDED' if state else 'WAITING')
                lp = picker.loop
                loading = lp.get('state') == 'loading' and (str(state.get('route')) != str(lp.get('route')) or
                                                            state.get('t', 0) < lp.get('t', 0))
                x0 = screen.get_width() - pick_w(screen)
                ratio = min(x0 / 1580, screen.get_height() / 1120)     # uniform: the image stays 2:1
                full = T is None or T.r != ratio or screen.get_size() != last_size
                if full:
                    T, last_size, picker.key = Text(ratio), screen.get_size(), None
                    screen.fill(BG)
                if state.get('agent') == 'openpilot':
                    rects = draw_openpilot(screen, state, images, T, status, x0, banner_for(state, lp, loading))
                    rid = state.get('route')
                    if rid and not loading and 'wide' in images and state.get('progress', 1) < 0.05 and age < 3:
                        picker.save_thumb(str(rid), images['wide'])
                else:               # TFv6 / TCP layouts: the old full-canvas path
                    canvas.fill(BG)
                    if state.get('agent') == 'TFv6':
                        label('TFv6 / ' + status + '   route ' + str(state.get('route', 'Waiting for agent')), (20, 12))
                        label('MODEL INPUT (three-camera mosaic)', (20, 48))
                        if 'input' in images:
                            canvas.blit(pygame.transform.scale(images['input'], (880, 293)), (20, 80))
                        canvas.blit(draw_bev(state), (920, 80))
                        label('MODEL OUTPUT / local actor frame', (920, 48))
                        label('Speed %.2f m/s' % state.get('speed_mps', 0), (20, 410))
                        label('Target %.2f m/s' % (state.get('target_speed_mps') or 0), (20, 450))
                        for i, key in enumerate(('throttle', 'brake', 'steer')):
                            label('%s: %.2f' % (key, state.get(key, 0)), (20, 500 + i * 36))
                        label('Step %s   ensemble %s' % (state.get('step', '-'), state.get('ensemble_size', '-')), (20, 625))
                        inference_ms = state.get('inference_ms')
                        label('Inference %.1f ms' % inference_ms if inference_ms is not None else 'Inference warming up', (20, 665))
                        label('Blue: predicted route / yellow: temporal waypoints', (20, 1060))
                        label('Actor-origin visualization; no controller or model output is modified.', (20, 1090))
                    else:
                        label('TCP / ' + status + '   ' + str(state.get('route', 'Waiting for agent'))[:95], (20, 12))
                        label('LEFT (model crop)', (20, 48))
                        label('FRONT (model crop)', (566, 48))
                        label('RIGHT (model crop)', (1034, 48))
                        if 'input' in images:
                            # Preserve the existing contiguous 315/270/315 mosaic and its aspect ratio.
                            canvas.blit(pygame.transform.scale(images['input'], (1560, 444)), (20, 80))
                        label('CHASE / yellow: prediction; white: fixed origin (flat-road projection)', (20, 540))
                        if 'chase' in images:
                            chase = images['chase'].copy()  # Overlay never modifies shared model/sensor bytes.
                            points = chase_waypoints(state, chase.get_width(), chase.get_height())
                            if len(points) > 1:
                                pygame.draw.lines(chase, (255, 220, 40), False, points, 3)
                            for point in points:
                                pygame.draw.circle(chase, (255, 220, 40), point, 4)
                            if points:
                                pygame.draw.circle(chase, (255, 255, 255), points[0], 5, 2)
                            canvas.blit(pygame.transform.scale(chase, (880, 495)), (20, 580))
                        canvas.blit(draw_bev(state), (920, 580))
                        label('BEV / 5 m grid', (920, 540))
                        label('%.2f m/s' % state.get('speed_mps', 0), (1380, 590))
                        for i, key in enumerate(('throttle', 'brake', 'steer')):
                            label('%s: %.2f' % (key, state.get(key, 0)), (1380, 630 + i * 32))
                        label('Step %s' % state.get('step', '-'), (1380, 740))
                        if 'chase_frame' in state:
                            lag = state.get('frame', 0) - state['chase_frame']
                            label('View lag %d' % lag if state['chase_frame'] >= 0 else 'View waiting', (1380, 770))
                        label('Timing (ms)', (1380, 790))
                        for i, (key, value) in enumerate(state.get('phases', {}).items()):
                            short = {'sensor_wait_ms': 'Sensors', 'preprocess_ms': 'Preprocess',
                                     'gpu_ms': 'GPU', 'policy_ms': 'Policy', 'preview_ms': 'Preview'}.get(key, key)
                            label('%s %.1f' % (short, value), (1380, 830 + 32 * i))
                        label('Green: ego outline / yellow: prediction / blue: supplied route (when available)', (20, 1086))
                    size = (max(1, int(1600 * ratio)), max(1, int(1120 * ratio)))
                    screen.blit(pygame.transform.scale(canvas, size), (0, 0))
                    rects, full, picker.key, T = [], True, None, None
                pr = picker.draw(screen, x0, int(48 * ratio), int(828 * ratio), state, loading)
                if pr:
                    rects.append(pr)
                rects.append(rates.draw(screen, x0, int(48 * ratio), picker.font, picker.small, render_ms))
                if full:
                    pygame.display.flip()
                else:
                    pygame.display.update(rects)
                if args.snapshot and state.get('step', 0) > 50 and not saved:
                    pygame.image.save(screen, args.snapshot)
                    saved = True
                last_draw = time.monotonic()
                ms = 1000 * (time.perf_counter() - t_render)
                render_ms = ms if render_ms is None else 0.9 * render_ms + 0.1 * ms
            clock.tick(30)
    finally:
        if reader:
            reader.close()
        pygame.quit()


if __name__ == '__main__':
    main()
