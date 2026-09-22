#!/usr/bin/env python
"""One input mosaic + chase camera, with model predictions projected onto the road.

Read-only separate process; Q/Escape closes only this window. No CARLA connection.
"""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import pygame
from drive_runtime.preview import PreviewReader


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--snapshot', help='Save one rendered dashboard without grabbing X11')
    parser.add_argument('--follow-runs', help='Follow newest child run/live in a profiling series')
    args = parser.parse_args()
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1600, 1120), pygame.RESIZABLE)
    pygame.display.set_caption('Bench2Drive TCP | model input + chase trajectory')
    canvas = pygame.Surface((1600, 1120))
    font = pygame.font.SysFont('DejaVu Sans', 22)
    clock = pygame.time.Clock()
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
            if reader is None:
                try:
                    reader = PreviewReader(args.dir)
                except (OSError, ValueError):
                    pass
            fresh = reader.read() if reader else None
            if fresh:
                packet = fresh  # Own bytes until all frombuffer surfaces have been replaced.
                state = packet['state']
                images = {}
                for name, spec in packet['images'].items():
                    h, w, _ = spec['shape']
                    pixels = memoryview(packet['pixels'])[spec['offset']:spec['offset'] + spec['size']]
                    images[name] = pygame.image.frombuffer(pixels, (w, h), 'RGB')
                dirty = True
            if dirty or time.monotonic() - last_draw > 1:
                canvas.fill((17, 21, 29))
                age = time.time() - state.get('t', 0)
                status = ('LIVE' if age < 3 else 'STALE') if state.get('running') else ('ENDED' if state else 'WAITING')
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
                ratio = min(screen.get_width() / 1600, screen.get_height() / 1120)
                size = (max(1, int(1600 * ratio)), max(1, int(1120 * ratio)))
                screen.fill((0, 0, 0))
                screen.blit(pygame.transform.scale(canvas, size), (0, 0))
                pygame.display.flip()
                if args.snapshot and state.get('step', 0) > 50 and not saved:
                    pygame.image.save(canvas, args.snapshot)
                    saved = True
                last_draw = time.monotonic()
            clock.tick(30)
    finally:
        if reader:
            reader.close()
        pygame.quit()


if __name__ == '__main__':
    main()
