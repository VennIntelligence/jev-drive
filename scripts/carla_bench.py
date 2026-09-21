#!/usr/bin/env python
"""Headless CARLA smoke test and throughput benchmark. See docs/carla.md.

Runs against an already-started server (scripts/carla_server.sh start <i>):
  python scripts/carla_bench.py --port 2000 --town Town05 --cameras 1 --width 1600 --height 900

Checks that frames actually contain an image (a server that renders nothing still answers RPCs
and hands out all-black buffers), then reports simulation FPS over a fixed number of ticks.
Prints one JSON object on the last line so a caller can parse it.
"""
import argparse, json, statistics, sys, time

import numpy as np

import carla

# Bench2Drive's agents drive at 20 Hz with a synchronous server.
DELTA = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town05")
    p.add_argument("--cameras", type=int, default=1)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--traffic", type=int, default=0, help="background vehicles to spawn")
    p.add_argument("--ticks", type=int, default=400)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--tag", default="")
    p.add_argument("--tm-port", type=int, default=0,
                   help="traffic manager port; 0 derives it from --port. A TM port lingers after "
                        "its server dies, so consecutive runs on one rpc port need distinct values")
    p.add_argument("--spawn-lift", type=float, default=0.0,
                   help="raise every spawn transform by this many metres and let the actor settle; "
                        "on large maps (Town12/13) spawning too low segfaults the server")
    return p.parse_args()


def main():
    a = parse_args()
    client = carla.Client("127.0.0.1", a.port)
    client.set_timeout(300.0)   # loading a world under N-way contention is slow
    print(f"client {client.get_client_version()} / server {client.get_server_version()}", flush=True)

    world = client.load_world(a.town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DELTA
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()

    def lifted(tf):
        if not a.spawn_lift:
            return tf
        return carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + a.spawn_lift),
            tf.rotation)

    # try_spawn_actor returns None for a placement the server rejects; spawn_actor raises, and on a
    # large map a bad placement can take the server down instead.
    ego = None
    for sp in spawns:
        ego = world.try_spawn_actor(bp.filter("vehicle.lincoln.mkz_2017")[0], lifted(sp))
        if ego is not None:
            break
    if ego is None:
        print("FAIL: no spawn point accepted the ego"); return 1
    actors = [ego]

    tm = client.get_trafficmanager(a.tm_port or (8000 + (a.port - 2000)))
    tm.set_synchronous_mode(True)
    ego.set_autopilot(True, tm.get_port())
    for sp in spawns[1:1 + a.traffic]:
        v = world.try_spawn_actor(bp.filter("vehicle.*")[0], lifted(sp))
        if v:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)

    # Bench2Drive-style camera rig: front, front-left, front-right, back, back-left, back-right.
    yaws = [0.0, -55.0, 55.0, 180.0, -110.0, 110.0]
    last, cams = {}, []
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(a.width))
    cam_bp.set_attribute("image_size_y", str(a.height))
    cam_bp.set_attribute("fov", "70")
    for i in range(a.cameras):
        tf = carla.Transform(carla.Location(x=0.8, z=1.6), carla.Rotation(yaw=yaws[i % len(yaws)]))
        cam = world.spawn_actor(cam_bp, tf, attach_to=ego)
        cam.listen(lambda img, k=i: last.__setitem__(k, img))
        cams.append(cam)
        actors.append(cam)

    for _ in range(a.warmup):
        world.tick()

    # Pixel check: a server that fails to render returns a uniform buffer.
    stats = []
    if a.cameras:
        for _ in range(10):
            world.tick()
        for k, img in sorted(last.items()):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
            stats.append({"cam": k, "mean": float(arr.mean()), "std": float(arr.std()),
                          "unique": int(len(np.unique(arr[::16, ::16])))})
        for s in stats:
            print(f"cam {s['cam']}: mean {s['mean']:.1f} std {s['std']:.1f} unique {s['unique']}", flush=True)

    dts = []
    t0 = time.perf_counter()
    for _ in range(a.ticks):
        t = time.perf_counter()
        world.tick()
        dts.append(time.perf_counter() - t)
    wall = time.perf_counter() - t0

    black = bool(stats) and all(s["std"] < 1.0 for s in stats)
    out = {
        "tag": a.tag, "port": a.port, "town": a.town, "cameras": a.cameras,
        "resolution": f"{a.width}x{a.height}", "traffic": a.traffic, "ticks": a.ticks,
        "fps": round(a.ticks / wall, 2),
        "ms_per_tick_median": round(1e3 * statistics.median(dts), 2),
        "ms_per_tick_p95": round(1e3 * sorted(dts)[int(0.95 * len(dts))], 2),
        "realtime_factor": round(a.ticks * DELTA / wall, 2),
        "pixels": stats, "black_frames": black,
    }
    print(json.dumps(out), flush=True)

    # Tear down only after reporting. Sensors must stop listening first: a callback that fires
    # on a destroyed sensor throws in a CARLA worker thread, which aborts the whole process.
    for cam in cams:
        cam.stop()
    world.tick()
    client.apply_batch_sync([carla.command.DestroyActor(x) for x in reversed(actors)], True)
    settings.synchronous_mode = False
    world.apply_settings(settings)
    return 1 if black else 0


if __name__ == "__main__":
    sys.exit(main())
