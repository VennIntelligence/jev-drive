#!/usr/bin/env python
"""Vehicle-physics probe for the CARLA rewind (todos/2026-09-29-carla-rewind.md): after set_transform +
set_target_velocity the car loses speed for about a second (wheel spin / drivetrain state is not restored). This
drives one car to a fork, runs a truth branch, then rewinds with several restore variants and replays the same branch
controls, and prints the speed / position error of each variant against the truth.

    $DATA_DIR/envs/carla/bin/python scripts/rewind_physics_probe.py --port 24100 --tm-port 30100 [--town Town04]
Variants (N = warm-up ticks replayed from the recorded prefix):
  poc        teleport to the fork state (k-1) + linear / angular velocity, then the branch
  vel{N}     teleport to the state at k-1-N, then N ticks replaying the recorded controls while forcing the recorded
             velocity each tick (no further teleport), then the branch
  tp{N}      as vel{N}, with a teleport to the recorded transform every warm-up tick
  velfix{N}  vel{N}, and a final teleport + velocity to the exact k-1 state before the branch
"""
import argparse
import json
import math

import carla
import numpy as np


def st(a):
    t, v, w = a.get_transform(), a.get_velocity(), a.get_angular_velocity()
    return (carla.Transform(carla.Location(t.location.x, t.location.y, t.location.z),
                            carla.Rotation(t.rotation.pitch, t.rotation.yaw, t.rotation.roll)),
            carla.Vector3D(v.x, v.y, v.z), carla.Vector3D(w.x, w.y, w.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=24100)
    ap.add_argument("--tm-port", type=int, default=30100)
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    client = carla.Client("127.0.0.1", a.port)
    client.set_timeout(600)
    world = client.load_world(a.town)
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
    rng = np.random.RandomState(0)
    # prefix: throttle ramps, mild steering noise, 140 ticks (~7 s)
    pre_ctl = [carla.VehicleControl(throttle=float(np.clip(0.35 + 0.3 * math.sin(i / 17), 0, 1)),
                                    steer=float(0.02 * rng.randn())) for i in range(140)]
    branches = {"hold": [carla.VehicleControl(throttle=0.5)] * 60,
                "brake": [carla.VehicleControl(brake=0.6)] * 60,
                "steer": [carla.VehicleControl(throttle=0.4, steer=0.3)] * 60}
    box = {}

    def fresh():
        """Reload the world and drive the prefix; returns the per-tick states and the fork state."""
        nonlocal_world = client.reload_world(False)
        box["w"] = nonlocal_world
        nonlocal_world.apply_settings(s)
        car = nonlocal_world.spawn_actor(bp, nonlocal_world.get_map().get_spawn_points()[0])
        box["car"] = car
        for _ in range(20):
            nonlocal_world.tick()
        hist = []
        for c in pre_ctl:
            hist.append(st(car))
            car.apply_control(c)
            nonlocal_world.tick()
        return hist, st(car)

    def run(ctls):
        car, w = box["car"], box["w"]
        tr = []
        for c in ctls:
            car.apply_control(c)
            w.tick()
            t, v, _ = st(car)
            tr.append((t.location.x, t.location.y, math.hypot(v.x, v.y), t.rotation.yaw))
        return np.array(tr)

    def set_state(state, vel=True, tp=True):
        t, v, w = state
        car = box["car"]
        if tp:
            car.set_transform(t)
        if vel:
            car.set_target_velocity(v)
            car.set_target_angular_velocity(w)

    res = {}
    variants = ["poc", "vel5", "vel10", "vel20", "tp10", "velfix10"]
    for name, ctls in branches.items():
        hist, fork = fresh()
        res[(name, "direct")] = run(ctls)
        for var in variants:
            if var == "poc":
                set_state(fork)
            else:
                n = int(var.lstrip("abcdefghijklmnopqrstuvwxyz"))
                set_state(hist[-n])
                for i in range(n):
                    box["car"].apply_control(pre_ctl[-n + i])
                    box["w"].tick()
                    if i + 1 < n:
                        set_state(hist[-n + i + 1], tp=var.startswith("tp"))
                    else:
                        set_state(fork, tp=var.startswith("velfix") or var.startswith("tp"))
            res[(name, var)] = run(ctls)
        hist2, fork2 = fresh()                      # the floor: the same prefix and branch from a reloaded world
        res[(name, "floor")] = run(ctls)
    variants = ["floor"] + variants
    out = {}
    for name in branches:
        ref = res[(name, "direct")]
        for var in variants:
            x = res[(name, var)]
            d = np.hypot(x[:, 0] - ref[:, 0], x[:, 1] - ref[:, 1])
            dv = np.abs(x[:, 2] - ref[:, 2])
            out[f"{name}/{var}"] = {"dpos_0.5": float(d[9]), "dpos_1": float(d[19]), "dpos_3": float(d[59]),
                                    "dv_max": float(dv.max()), "dv_1": float(dv[19]), "v_ref_0": float(ref[0, 2]), "v_0": float(x[0, 2])}
            print(f"{name:6s} {var:9s} v0 {x[0, 2]:6.2f} (ref {ref[0, 2]:6.2f})  dpos 0.5/1/3 s {d[9]:6.3f} {d[19]:6.3f} {d[59]:6.3f}"
                  f"  |dv| max {dv.max():5.2f}", flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=1))
    box["car"].destroy()
    s.synchronous_mode = False
    box["w"].apply_settings(s)


if __name__ == "__main__":
    main()
