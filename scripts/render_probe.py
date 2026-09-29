#!/usr/bin/env python
"""Leaderboard-free probe of CARLA render state (todos/2026-09-28-wm-loop.md, "渲染故障的根因").

Starts one fresh CARLA server, then runs the same episode EPISODES times on it (load_world each time, as the
leaderboard does between routes). An episode replays the leaderboard's order of calls against one static camera
placed at a route's start (the WL front camera: 1088 x 1560, fov 52), logging the camera's mean luma every tick:
  A  set_weather(route weather), tick, set_day_night_cycle(False) (RouteLightsBehavior.__init__), spawn the camera,
     10 ticks (AgentWrapper.setup_sensors)
  B  set_weather(same weather) again (RouteWeatherBehavior's first update), 10 ticks
  C  RouteLightsBehavior's first update: turn on the lights within 100 m and off the others, by the client's is_on,
     10 ticks
  E  the same rule applied from a fresh client (the server's true on/off state), 10 ticks
  D  set_day_night_cycle(True) (RouteLightsBehavior.terminate), 5 ticks, destroy the camera
Out: <out>/probe.jsonl, one line per tick: episode, phase, tick, luma, server lights on.

    $DATA_DIR/envs/carla/bin/python scripts/render_probe.py --routes <xml> --route-id <id> --out <dir> [--gpu 6]
"""
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import carla
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_run import Server  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routes", required=True)
    p.add_argument("--route-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--index", type=int, default=480)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--cam-attrs", default="{}")
    a = p.parse_args()
    route = next(r for r in ET.parse(a.routes).getroot().iter("route") if r.get("id") == a.route_id)
    wx = route.find("weathers")[0].attrib
    weather = carla.WeatherParameters()
    for k, v in wx.items():
        if k != "route_percentage" and hasattr(weather, k):
            setattr(weather, k, float(v))
    wp = [route.find("waypoints")[i].attrib for i in (0, 1)]
    x0, y0, z0, x1, y1 = (float(wp[i][k]) for i, k in ((0, "x"), (0, "y"), (0, "z"), (1, "x"), (1, "y")))
    yaw = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "probe.jsonl", "w", buffering=1)
    server = Server(a.index, out / "servers", "Epic", a.gpu)
    server.start()
    try:
        client = carla.Client("localhost", server.port)
        client.set_timeout(120.0)
        for ep in range(a.episodes):
            world = client.load_world(route.get("town"), reset_settings=False)
            world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=0.05))
            world.tick()
            lm = world.get_lightmanager()
            img = {}
            state = {"tick": 0}

            def log(phase, n):
                for _ in range(n):
                    f = world.tick()
                    t0 = time.time()
                    while f not in img and time.time() - t0 < 20:
                        time.sleep(0.005)
                    im = img.pop(f, None)
                    state["tick"] += 1
                    probe = carla.Client("localhost", server.port)     # fresh: LightManager.is_on is a client cache
                    probe_lm = probe.get_world().get_lightmanager()
                    srv = sum(l.is_on for l in probe_lm.get_all_lights())
                    fh.write(json.dumps({"episode": ep, "phase": phase, "tick": state["tick"], "srv_on": srv,
                                         "luma": None if im is None else round(float(im.mean()), 2)}) + "\n")

            world.set_weather(weather)
            world.tick()
            lm.set_day_night_cycle(False)
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            for k, v in dict({"image_size_x": 1088, "image_size_y": 1560, "fov": 52.0756},
                             **json.loads(a.cam_attrs)).items():
                bp.set_attribute(k, str(v))
            cam = world.spawn_actor(bp, carla.Transform(carla.Location(x0, y0, z0 + 1.8), carla.Rotation(yaw=yaw)))
            cam.listen(lambda m: img.__setitem__(m.frame, np.frombuffer(m.raw_data, np.uint8).reshape(
                m.height, m.width, 4)[::4, ::4, :3] @ np.array([0.114, 0.587, 0.299])))
            log("A", 10)
            world.set_weather(weather)
            log("B", 10)
            here = carla.Location(x0, y0, z0)
            lights = lm.get_all_lights()
            lm.turn_on([l for l in lights if l.location.distance(here) <= 100 and not l.is_on])
            lm.turn_off([l for l in lights if l.location.distance(here) > 100 and l.is_on])
            log("C", 10)
            probe = carla.Client("localhost", server.port)          # E: the same rule on the server's true state
            plm = probe.get_world().get_lightmanager()
            truth = plm.get_all_lights()
            plm.turn_off([l for l in truth if l.location.distance(here) > 100 and l.is_on])
            plm.turn_on([l for l in truth if l.location.distance(here) <= 100 and not l.is_on])
            log("E", 10)
            lm.set_day_night_cycle(True)
            log("D", 5)
            cam.stop()
            cam.destroy()
            world.tick()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
