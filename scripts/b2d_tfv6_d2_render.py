"""Replay approved D2 CARLA recorders into chase videos and annotated contact sheets.

Run only after D2 driving ends, against an owned offscreen CARLA server. The
camera and all overlays live in the replay process, never in the control loop.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import subprocess
import time

import carla
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/data/runs/b2d/tfv6-w2/d2')
REPO = Path(__file__).resolve().parents[1]
SHEETS = REPO / 'todos/2026-09-23-tfv6-controller/results/diagnosis/d2'
GROUPS = [('2', '27529', 0, 'B', 'stop', 17.3), ('2', '27529', 0, 'C', 'stop', 17.3),
          ('2', '27529', 0, 'B', 'deviation', 22.8), ('2', '27529', 0, 'C', 'deviation', 22.8),
          ('2', '28154', 0, 'B', 'collision', 11.8), ('2', '28154', 0, 'C', 'collision', 11.8),
          ('1', '26405', 1, 'A', 'collision', 13.0), ('1', '26405', 1, 'B', 'collision', 13.0),
          ('1', '3514', 1, 'B', 'static', 1.5), ('1', '3514', 1, 'C', 'static', 1.5)]
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)


def case_dir(level, route, seed, arm, repeat=1):
    return ROOT / f'repeat-{repeat}' / 'cases' / level / f'route-{route}' / f'seed-{seed}' / arm


def load_case(level, route, seed, arm, repeat=1):
    path = case_dir(level, route, seed, arm, repeat)
    done = json.loads((path / 'done.json').read_text())
    run = Path(done['run_dir'])
    frames = [json.loads(line) for line in (run.parent / 'frames.jsonl').open()]
    frames = [f for f in frames if f.get('truth')]
    recorder = next((run / 'attempts' / route / '1' / 'recorder').glob('*.log'))
    return frames, recorder


def _camera_frames(client, recorder, frames, center, raw_dir):
    # Beginning-to-window replay avoids relying on an undocumented offset
    # between recorder start and the first agent tick.
    raw_dir.mkdir(parents=True, exist_ok=True)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    world.apply_settings(settings)
    duration = center + 4.0
    client.set_replayer_time_factor(1.0)
    info = client.replay_file(str(recorder), 0.0, duration, 0, False)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    world.apply_settings(settings)
    camera = None
    hero_id = None
    locations = {}
    image_paths = {}
    image_queue = queue.Queue()
    total_ticks = int((duration + 5.0) / .05)
    for tick in range(total_ticks):
        try:
            world.tick(15.0)
        except RuntimeError:
            break
        snapshot = world.get_snapshot()
        if hero_id is None:
            vehicles = list(world.get_actors().filter('vehicle.*'))
            candidates = [a for a in vehicles if a.attributes.get('role_name') == 'hero']
            if not candidates and tick > 20 and vehicles:
                first_xy = np.asarray(frames[0]['truth']['location'][:2])
                candidates = [min(vehicles, key=lambda a: np.linalg.norm(
                    np.asarray([a.get_location().x, a.get_location().y]) - first_xy))]
            for actor in candidates:
                hero_id = actor.id
                bp = world.get_blueprint_library().find('sensor.camera.rgb')
                bp.set_attribute('image_size_x', '960')
                bp.set_attribute('image_size_y', '540')
                bp.set_attribute('fov', '80')
                bp.set_attribute('sensor_tick', '0.1')
                tf = carla.Transform(carla.Location(x=-7.5, z=4.2), carla.Rotation(pitch=-20))
                camera = world.spawn_actor(bp, tf, attach_to=actor)
                camera.listen(image_queue.put)
                break
        if hero_id is not None:
            actor = snapshot.find(hero_id)
            if actor is not None:
                loc = actor.get_transform().location
                locations[snapshot.frame] = (loc.x, loc.y)
        # Keep the callback queue short; save all camera images for later alignment.
        while not image_queue.empty():
            img = image_queue.get_nowait()
            path = raw_dir / f'{img.frame:08d}.jpg'
            img.save_to_disk(str(path))
            image_paths[img.frame] = path
    time.sleep(.3)
    while not image_queue.empty():
        img = image_queue.get_nowait()
        path = raw_dir / f'{img.frame:08d}.jpg'
        img.save_to_disk(str(path))
        image_paths[img.frame] = path
    if camera is not None:
        camera.stop()
        camera.destroy()
    client.stop_replayer(False)
    if not image_paths or not locations:
        raise RuntimeError(f'No replay camera images or ego positions: {recorder}; info={info}')
    return locations, image_paths


def _align(locations, frames):
    # Find the monotonic frame-index shift minimizing motion disagreement.
    replay_ids = np.array(sorted(locations), dtype=int)
    replay_xy = np.array([locations[k] for k in replay_ids], dtype=float)
    truth_xy = np.array([f['truth']['location'][:2] for f in frames], dtype=float)
    # CARLA recorder replay may start a few frames before the first agent tick.
    best = (float('inf'), None)
    for shift in range(-500, 501):
        idx = np.arange(len(replay_ids)) - shift
        good = (idx >= 0) & (idx < len(frames))
        if good.sum() < 15:
            continue
        residual = np.linalg.norm(replay_xy[good] - truth_xy[idx[good]], axis=1)
        score = float(np.percentile(residual, 60))
        if score < best[0]:
            best = score, shift
    if best[1] is None:
        raise RuntimeError('Could not align replay and telemetry')
    if best[0] > 2.0:
        raise RuntimeError(f'Replay-to-telemetry alignment residual {best[0]:.2f} m is too large')
    # Frame IDs may include pre-replay ticks; only their ordered positions matter.
    return {int(frame_id): int(j - best[1]) for j, frame_id in enumerate(replay_ids)}, best


def _overlay(img, frame, title):
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle((0, 0, w, 31), fill=(10, 14, 20))
    draw.text((12, 7), title, font=FONT, fill=(255, 255, 255))
    # Ego-local bird's-eye inset: x forward = up, y right = right.
    x0, y0, panel = w - 255, 38, 242
    draw.rectangle((x0, y0, x0 + panel, y0 + panel), fill=(15, 23, 31), outline=(220, 220, 220), width=1)
    draw.text((x0 + 8, y0 + 7), 'TFv6 plan (ego frame)', font=SMALL, fill=(240, 240, 240))
    origin = (x0 + panel / 2, y0 + panel - 22)
    scale = 8.0
    def proj(p):
        return (origin[0] + float(p[1]) * scale, origin[1] - float(p[0]) * scale)
    def path(points, color):
        if not points:
            return
        pts = [proj(p) for p in points if p is not None and len(p) >= 2]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        for x, y in pts:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    path(frame.get('route_prediction'), (0, 158, 115))
    path(frame.get('waypoint'), (55, 175, 235))
    draw.polygon([(origin[0], origin[1]-10), (origin[0]-7, origin[1]+7),
                  (origin[0]+7, origin[1]+7)], fill=(255, 255, 255))
    draw.text((x0+8, y0+panel-18), 'route=green  wp=blue', font=SMALL, fill=(220, 220, 220))
    ctrl = frame.get('executed_control') or {}
    speed = frame.get('truth', {}).get('forward_speed_mps', 0)
    labels = [('throttle', (0, 158, 115), 1.0), ('brake', (215, 60, 55), 1.0),
              ('steer', (55, 175, 235), 1.0)]
    draw.rectangle((8, h - 102, 312, h - 8), fill=(10, 14, 20))
    draw.text((16, h - 98), f'v={speed:.1f} m/s  target={frame.get("target_speed") or 0:.1f}',
              font=SMALL, fill=(240, 240, 240))
    for row, (name, color, maximum) in enumerate(labels):
        y = h - 77 + row*22
        value = float(ctrl.get(name, 0.0))
        draw.text((16, y), f'{name[:3]} {value:+.2f}', font=SMALL, fill=(240, 240, 240))
        draw.rectangle((110, y+3, 290, y+13), outline=(170, 170, 170), width=1)
        if name == 'steer':
            mid = 200
            target = mid + int(max(-1, min(1, value))*90)
            draw.rectangle((min(mid,target), y+3, max(mid,target), y+13), fill=color)
        else:
            draw.rectangle((110, y+3, 110+int(max(0,min(1,value))*180), y+13), fill=color)
    return img


def render_one(client, level, route, seed, arm, window, center, repeat=1):
    frames, recorder = load_case(level, route, seed, arm, repeat)
    root = ROOT / 'render' / f'{level}-{route}-{seed}-{arm}-r{repeat}-{window}'
    raw = root / 'raw'
    overlay = root / 'overlay'
    overlay.mkdir(parents=True, exist_ok=True)
    locations, image_paths = _camera_frames(client, recorder, frames, center, raw)
    mapped, fit = _align(locations, frames)
    selected = []
    for image_id, path in sorted(image_paths.items()):
        index = mapped.get(image_id)
        if index is None or not 0 <= index < len(frames):
            continue
        frame = frames[index]
        t = frame['sim_time']
        if center - 2.2 <= t <= center + 2.2:
            selected.append((t, path, frame))
    if len(selected) < 5:
        raise RuntimeError(f'Too few aligned images around {center}s: {len(selected)}; fit={fit}')
    for n, (t, path, frame) in enumerate(selected):
        with Image.open(path) as src:
            img = _overlay(src, frame, f'L{level} route {route} seed {seed} {arm} repeat {repeat}   t={t:.2f}s')
            img.save(overlay / f'{n:06d}.jpg', quality=86, optimize=True)
    mp4 = root / 'window.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', '10',
                    '-i', str(overlay / '%06d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '24', str(mp4)], check=True)
    result = {'level': level, 'route': route, 'seed': seed, 'arm': arm, 'repeat': repeat,
              'window': window,
              'center_s': center, 'frames': len(selected), 'alignment_residual_m': fit[0],
              'alignment_shift_ticks': fit[1], 'mp4': str(mp4), 'recorder': str(recorder),
              'frame_times_s': [x[0] for x in selected]}
    (root / 'render.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def contact_sheet(rows, out):
    # Two arms x three moments; reduce palette to keep each PNG below 500 KiB.
    tiles = []
    for row in rows:
        root = Path(row['mp4']).parent
        paths = sorted((root / 'overlay').glob('*.jpg'))
        times = row['frame_times_s']
        for target in [row['center_s'] - 1, row['center_s'], row['center_s'] + 1]:
            index = min(range(len(times)), key=lambda i: abs(times[i] - target))
            with Image.open(paths[index]) as source:
                tiles.append(source.resize((480, 270)))
    sheet = Image.new('RGB', (1440, 540), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3)*480, (i // 3)*270))
    out.parent.mkdir(parents=True, exist_ok=True)
    for colors in (128, 96, 64, 48, 32):
        palette = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        palette.save(out, optimize=True, compress_level=9)
        if out.stat().st_size < 500*1024:
            return out.stat().st_size
    raise RuntimeError(f'Contact sheet exceeds 500 KiB: {out.stat().st_size}')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=6600)
    parser.add_argument('--route')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    client = carla.Client('127.0.0.1', args.port)
    client.set_timeout(120.0)
    print('connected', client.get_server_version(), flush=True)
    for route, window in sorted(set((x[1], x[4]) for x in GROUPS)):
        if args.route and route != args.route:
            continue
        entries = [x for x in GROUPS if x[1] == route and x[4] == window]
        rows = []
        for level, rid, seed, arm, label, center in entries:
            print('replay', level, rid, seed, arm, label, flush=True)
            rows.append(render_one(client, level, rid, seed, arm, label, center, args.repeat))
            print(rows[-1], flush=True)
        sheet = SHEETS / f'{route}-{window}-contact-sheet.png'
        size = contact_sheet(rows, sheet)
        print('sheet', sheet, size, flush=True)


if __name__ == '__main__':
    main()
