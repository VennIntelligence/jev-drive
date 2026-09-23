#!/usr/bin/env python
"""Record stock MKZ physics and bounded steering/braking response on an owned CARLA server.

Run in tmux. This diagnostic uses privileged state explicitly, never supplies it to a policy,
and writes all measurements before shutting down only the server it started. Python 3.8.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import carla
import numpy as np

from b2d_run import Server


def serial(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [serial(x) for x in value]
    result = {}
    for key in dir(value):
        if key.startswith('_'):
            continue
        item = getattr(value, key)
        if not callable(item):
            try:
                result[key] = serial(item)
            except (TypeError, RecursionError):
                result[key] = str(item)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--server-index', type=int, default=60)
    p.add_argument('--town', default='Town10HD')
    p.add_argument('--physics-only', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'servers').mkdir(exist_ok=True)
    log = (out / 'log.txt').open('a')
    events = (out / 'events.jsonl').open('a')

    def event(kind, **fields):
        row = dict(t=time.time(), kind=kind, **fields)
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + '\n'); log.flush()
        events.write(line + '\n'); events.flush()

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    server = Server(a.server_index, out / 'servers', 'Epic', gpu_rank=0)
    actors = []
    event('start', configuration=vars(a), pid=os.getpid())
    try:
        server.start()
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                       '--format=csv'], text=True)
        (out / 'gpu.txt').write_text(gpu)
        event('gpu', processes=gpu)
        client = carla.Client('127.0.0.1', server.port)
        client.set_timeout(90)
        world = client.load_world(a.town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        settings.substepping = True
        settings.max_substep_delta_time = .01
        settings.max_substeps = 10
        world.apply_settings(settings)
        blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        # Pick a long nearly straight segment using static map geometry for calibration only.
        choices = []
        for start in world.get_map().get_spawn_points():
            wp = world.get_map().get_waypoint(start.location)
            path = [wp]
            for _ in range(100):
                nxt = path[-1].next(2)
                if not nxt:
                    break
                path.append(min(nxt, key=lambda v: abs((v.transform.rotation.yaw - start.rotation.yaw + 180) % 360 - 180)))
                angle = abs((path[-1].transform.rotation.yaw - start.rotation.yaw + 180) % 360 - 180)
                if angle > 8 or path[-1].is_junction:
                    break
            choices.append((len(path), start))
        spawn = max(choices, key=lambda x: x[0])[1]
        vehicle = world.spawn_actor(blueprint, spawn)
        actors.append(vehicle)
        for _ in range(40):
            vehicle.apply_control(carla.VehicleControl(brake=1))
            world.tick()
        physics = vehicle.get_physics_control()
        tf = vehicle.get_transform()
        # CARLA 0.9.15 wheel positions are world coordinates in centimeters.
        positions = np.array([[w.position.x, w.position.y, w.position.z] for w in physics.wheels]) / 100
        front, rear = positions[:2].mean(axis=0), positions[2:].mean(axis=0)
        forward = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y, tf.get_forward_vector().z])
        origin = np.array([tf.location.x, tf.location.y, tf.location.z])
        measured = dict(wheelbase=float(np.linalg.norm(front - rear)),
                        rear_axle_offset_m=float((rear - origin).dot(forward)),
                        max_steer_deg=float(np.mean([w.max_steer_angle for w in physics.wheels[:2]])),
                        steering_curve=[[v.x, v.y] for v in physics.steering_curve])
        # Validate wheel world/centimeter interpretation by independently translating the actor.
        shifted = carla.Transform(carla.Location(tf.location.x + 1, tf.location.y, tf.location.z + .1), tf.rotation)
        vehicle.set_transform(shifted)
        world.tick()
        moved = vehicle.get_physics_control()
        translation = [[(b.position.x-a.position.x)/100, (b.position.y-a.position.y)/100,
                        (b.position.z-a.position.z)/100] for a, b in zip(physics.wheels, moved.wheels)]
        vehicle.set_transform(tf)
        world.tick()
        result = dict(client_version=client.get_client_version(), server_version=client.get_server_version(),
                      town=world.get_map().name, server_port=server.port, vehicle=vehicle.type_id,
                      transform=serial(tf), physics=serial(physics), constants=measured,
                      wheel_translation_check_m=translation, wheel_position_units='world centimeters',
                      gpu=gpu, dt=.05, timestamp=time.time())
        (out / 'physics.json').write_text(json.dumps(result, indent=2))
        event('physics', constants=measured, wheel_translation_check_m=translation)
        if not 2 < measured['wheelbase'] < 4 or not -3 < measured['rear_axle_offset_m'] < 0:
            raise RuntimeError('Unexpected wheel coordinates: inspect physics.json before using defaults')
        (out / 'controller_config.json').write_text(json.dumps(measured, indent=2))
        if not a.physics_only:
            # A deliberately steep diagnostic curve separates km/h from m/s without fitting
            # tire dynamics. Restore stock physics before the response measurements.
            probe_physics = vehicle.get_physics_control()
            probe_physics.steering_curve = [carla.Vector2D(0, 1), carla.Vector2D(10, .1)]
            vehicle.apply_physics_control(probe_physics)
            unit_rows = []
            for tick in range(40):
                direction = vehicle.get_transform().get_forward_vector()
                vehicle.set_target_velocity(direction * 2.)
                vehicle.apply_control(carla.VehicleControl(steer=.05))
                world.tick()
                if tick >= 20:
                    unit_rows.append(dict(speed_mps=vehicle.get_velocity().dot(vehicle.get_transform().get_forward_vector()),
                                          left_deg=vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel),
                                          right_deg=vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)))
            (out / 'steering_unit_probe.json').write_text(json.dumps(dict(
                diagnostic_curve=[[0, 1], [10, .1]], steer=.05, samples=unit_rows), indent=2))
            event('steering_unit_probe', speed_mean=float(np.mean([r['speed_mps'] for r in unit_rows])),
                  wheel_mean_deg=float(np.mean([(r['left_deg']+r['right_deg'])/2 for r in unit_rows])))
            vehicle.apply_physics_control(physics)
            vehicle.set_transform(tf)
            imu_samples = []
            imu_bp = world.get_blueprint_library().find('sensor.other.imu')
            imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
            imu.listen(lambda data: imu_samples.append(dict(frame=data.frame, gyro_z=data.gyroscope.z,
                                                          compass=data.compass)))
            actors.append(imu)
            rows = []
            for desired_speed in (2., 6., 10.):
                for steer in (-.15, .15):
                    vehicle.set_transform(tf)
                    vehicle.set_target_velocity(carla.Vector3D())
                    vehicle.set_target_angular_velocity(carla.Vector3D())
                    for tick in range(140):
                        vel = vehicle.get_velocity()
                        now_tf = vehicle.get_transform()
                        speed = vel.dot(now_tf.get_forward_vector())
                        control = carla.VehicleControl(throttle=float(np.clip((desired_speed-speed)*.5, 0, .75)),
                                                       brake=float(np.clip((speed-desired_speed)*.5, 0, 1)),
                                                       steer=steer if tick >= 80 else 0.)
                        vehicle.apply_control(control)
                        world.tick()
                        rows.append(dict(frame=world.get_snapshot().frame, target_speed=desired_speed, command_steer=control.steer, tick=tick,
                                         speed=vehicle.get_velocity().dot(vehicle.get_transform().get_forward_vector()),
                                         yaw_rate_rps=math.radians(vehicle.get_angular_velocity().z),
                                         wheel_angle_deg=vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel),
                                         location=serial(vehicle.get_location())))
                    event('response', target_speed=desired_speed, steer=steer)
            with (out / 'response.jsonl').open('w') as fh:
                for row in rows:
                    fh.write(json.dumps(row)+'\n')
            (out / 'imu_samples.json').write_text(json.dumps(imu_samples))
            summaries = []
            for v in (2., 6., 10.):
                for command in (-.15, .15):
                    r = [x for x in rows if x['target_speed'] == v and abs(x['command_steer'] - command) < 1e-6 and x['tick'] >= 120]
                    summaries.append(dict(target_speed=v, steer=command, speed_mean=float(np.mean([x['speed'] for x in r])),
                                          yaw_rate_mean=float(np.mean([x['yaw_rate_rps'] for x in r])),
                                          wheel_angle_deg_mean=float(np.mean([x['wheel_angle_deg'] for x in r]))))
            (out / 'response_summary.json').write_text(json.dumps(summaries, indent=2))
        event('end', status='completed')
    except BaseException as exc:
        event('end', status='failed', error=repr(exc))
        raise
    finally:
        for actor in reversed(actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        server.stop()
        log.close(); events.close()


if __name__ == '__main__':
    main()
