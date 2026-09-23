#!/usr/bin/env python
"""Sensor-driven map-route validation without scenario or background actors.

Privileged route geometry and truth are diagnostic inputs. Truth measurement never
changes agent state. This tool produces controller validation, not leaderboard scores.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np


class TruthProjection:
    """Independent evaluator projection; never reads estimated route progress."""
    def __init__(self, points):
        self.points = np.array(points, dtype=float, copy=True)
        self.lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        self.progress = None

    def measure(self, xy, yaw, speed):
        # Bounded true progress avoids jumping branches at crossings. Initial
        # search covers 20m; subsequent range tolerates reverse and 20Hz motion.
        low = 0. if self.progress is None else max(0., self.progress - 3.)
        high = min(self.arc[-1], 20. if self.progress is None else self.progress + max(3., abs(speed) * .2))
        best = None
        for i, length in enumerate(self.lengths):
            if length <= 1e-9 or self.arc[i] > high or self.arc[i + 1] < low:
                continue
            tangent = (self.points[i + 1] - self.points[i]) / length
            along = float(np.dot(xy - self.points[i], tangent))
            along = min(max(along, max(0., low - self.arc[i])), min(length, high - self.arc[i]))
            residual = xy - (self.points[i] + along * tangent)
            distance = float(np.linalg.norm(residual))
            heading = math.atan2(tangent[1], tangent[0])
            score = distance ** 2 + 4. * (1. - math.cos(heading - yaw))
            if best is None or score < best[0]:
                # Left positive in CARLA world x-forward/y-right convention.
                cross = float(tangent[1] * residual[0] - tangent[0] * residual[1])
                best = (score, self.arc[i] + along, cross, distance, heading)
        if best is None:
            raise ValueError('independent truth projection has no valid segment')
        self.progress = float(best[1])
        return dict(progress_m=self.progress, cross_track_m=best[2],
                    projection_distance_m=best[3], path_heading_rad=best[4],
                    remaining_along_m=max(0., float(self.arc[-1] - self.progress)))


def _finite(values):
    return [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x)]


def _stats(values):
    values = _finite(values)
    if not values:
        return dict(count=0, rms=None, median=None, p90=None, p95=None, max=None)
    array = np.asarray(values)
    return dict(count=len(values), rms=float(np.sqrt(np.mean(array ** 2))),
                median=float(np.median(np.abs(array))), p90=float(np.percentile(np.abs(array), 90)),
                p95=float(np.percentile(np.abs(array), 95)), max=float(np.max(np.abs(array))))


def summarize(rows, records, cruise, stop_deceleration, status, collisions, hold_start):
    """Recomputable G2 metrics with explicit missing-data and startup handling."""
    steady = [row for row in rows if row['elapsed_s'] >= 2.]
    cruise_rows = [row for row in rows if row['reference_speed_mps'] >= cruise - .1]
    cruising = [row for row in cruise_rows if row['elapsed_s'] >= 5.]
    lateral_full = _stats(row.get('cross_track_m') for row in rows)
    lateral_steady = _stats(row.get('cross_track_m') for row in steady)
    distance_full = _stats(row.get('projection_distance_m') for row in rows)
    speed_all_cruise = _stats(row['speed'] - cruise for row in cruise_rows)
    speed_cruise = _stats(row['speed'] - cruise for row in cruising)
    speed_reference = _stats(row['speed'] - row['reference_speed_mps'] for row in rows)
    command_errors = []
    for record in records:
        speed, target = record.get('speed_mps'), record.get('target_speed_mps')
        if len(_finite([speed, target])) == 2:
            command_errors.append(speed - target)
    pose = _stats(row.get('pose_error_m') for row in records)
    raw_pose = _stats(row.get('raw_pose_error_m') for row in records)
    heading = _stats(math.degrees(x['pose_heading_error_rad']) for x in records
                     if len(_finite([x.get('pose_heading_error_rad')])) == 1)
    hold = rows[hold_start:] if hold_start is not None else []
    if hold:
        positions = np.asarray([row['truth_xy'] for row in hold])
        # Diameter captures drift between any two positions, including a return
        # past the original hold point. N is only ~101 in a successful case.
        displacement = float(max(np.max(np.linalg.norm(positions - p, axis=1)) for p in positions))
        hold_duration = hold[-1]['sim_time'] - hold[0]['sim_time']
        hold_max_speed = max(abs(row['speed']) for row in hold)
    else:
        displacement = hold_duration = hold_max_speed = None
    aligned = [r for r in records if r.get('truth_frame') == r.get('frame') and 'truth_error' not in r]
    frame_ids = [r.get('frame') for r in records]
    valid_frames = bool(frame_ids) and all(isinstance(f, int) for f in frame_ids)
    monotonic = valid_frames and all(a < b for a, b in zip(frame_ids, frame_ids[1:]))
    summary = dict(status=status, ticks=len(rows), collisions=collisions,
                   cross_track_rms_m=lateral_full['rms'], cross_track_p95_m=lateral_full['p95'],
                   cross_track_full=lateral_full, cross_track_steady=lateral_steady,
                   projection_distance_full=distance_full, speed_rms_mps=speed_cruise['rms'],
                   cruise_speed_after_5s=speed_cruise, cruise_speed_all=speed_all_cruise,
                   reference_speed_all=speed_reference, command_speed_all=_stats(command_errors),
                   pose_p90_m=pose['p90'], fused_pose=pose, raw_pose=raw_pose,
                   pose_heading_deg=heading, endpoint_error_m=rows[-1]['endpoint_error_m'] if rows else None,
                   stop_hold_s=hold_duration, stop_hold_displacement_m=displacement,
                   stop_hold_max_speed_mps=hold_max_speed,
                   hold_pitch_deg=_stats(row['pitch_deg'] for row in hold),
                   final_pitch_deg=rows[-1]['pitch_deg'] if rows else None,
                   telemetry_count=len(records), truth_aligned_count=len(aligned),
                   telemetry_frames_monotonic=bool(monotonic),
                   metric_protocol=dict(lateral_gate='full_route_independent_truth_projection',
                                        lateral_steady_warmup_s=2., cruise_startup_allowance_s=5.,
                                        cruise_reference='truth_remaining_distance_constant_deceleration',
                                        stop_deceleration_mps2=stop_deceleration,
                                        stop_hold_displacement='maximum_pairwise_rear_axle_distance'))
    def at_most(value, limit):
        return len(_finite([value])) == 1 and value <= limit
    gates = dict(completed=status == 'completed', no_collision=not collisions,
                 full_lateral_rms=at_most(lateral_full['rms'], .5),
                 full_lateral_p95=at_most(lateral_full['p95'], 1.),
                 cruise_speed=at_most(speed_cruise['rms'], .5),
                 pose=at_most(pose['p90'], .5), heading=at_most(heading['p90'], 1.),
                 endpoint=at_most(summary['endpoint_error_m'], 1.),
                 hold_duration=hold_duration is not None and hold_duration >= 5. - 1e-6,
                 hold_displacement=at_most(displacement, .1),
                 hold_speed=hold_max_speed is not None and hold_max_speed < .1,
                 telemetry_complete=bool(rows) and len(rows) == len(records) == len(aligned) and bool(monotonic)
                 and frame_ids == [row.get('frame') for row in rows])
    summary.update(gates=gates, gate_pass=all(gates.values()))
    return summary



def load_matrix(controller_config, presets, variants_path=None, route_cruises_path=None,
                cruise_mps=8.):
    """Read/validate every case input once, before starting any simulator process.

    Mapping insertion order is the declared variant order; each variant's preset
    order is retained. The caller repeats this list inside each route loop.
    """
    if variants_path:
        variants = json.loads(Path(variants_path).read_text())
        if not isinstance(variants, dict) or not variants:
            raise ValueError('--variants must be a nonempty label-to-config object')
    else:
        if not controller_config:
            raise ValueError('--controller-config is required unless --variants is supplied')
        variants = {'default': {'controller_config': controller_config,
                                'presets': presets.split(',')}}
    cases = []
    for label, variant in variants.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label):
            raise ValueError('Variant label must be a safe directory name: %r' % label)
        if not isinstance(variant, dict) or set(variant) != {'controller_config', 'presets'}:
            raise ValueError('Each variant needs controller_config and presets only')
        raw_path = variant['controller_config']
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError('controller_config must be a path string')
        if variants_path and not Path(raw_path).is_absolute():
            raise ValueError('Variant controller_config must be an absolute path')
        config_path = Path(raw_path).resolve()
        config_bytes = config_path.read_bytes()
        parameters = json.loads(config_bytes)
        if not isinstance(parameters, dict):
            raise ValueError('Controller config must be an object')
        adapter = dict(parameters.get('adapter', {}))
        adapter.update({k: v for k, v in parameters.items() if k in
                        ('rear_axle_offset_m', 'route_stop_deceleration')})
        rear = float(adapter['rear_axle_offset_m'])
        deceleration = float(adapter.get('route_stop_deceleration', 2.))
        if not math.isfinite(rear) or not math.isfinite(deceleration) or deceleration <= 0:
            raise ValueError('Invalid rear axle or stop deceleration configuration')
        selected = variant['presets']
        if (not isinstance(selected, list) or not selected
                or any(x not in ('carla', 'tcp', 'pursuit') for x in selected)
                or len(selected) != len(set(selected))):
            raise ValueError('presets must be a nonempty unique list of carla/tcp/pursuit')
        for preset in selected:
            cases.append(dict(variant=label, preset=preset, controller_config=str(config_path),
                              controller_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                              config_bytes=config_bytes, rear_axle_offset_m=rear,
                              stop_deceleration_mps2=deceleration))
    cruises = json.loads(Path(route_cruises_path).read_text()) if route_cruises_path else {}
    if not isinstance(cruises, dict):
        raise ValueError('--route-cruises must map route IDs (or default) to positive speeds')
    cruises.setdefault('default', cruise_mps)
    for key, value in cruises.items():
        if (not isinstance(key, str) or not key or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError('Route cruise speeds must be finite and positive')
    return cases, cruises


def archive_matrix(out, routes_path, cases, cruises, variants_path=None, route_cruises_path=None):
    """Run against archived config bytes, so later source edits cannot alter a case."""
    inputs = out / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    routes_bytes = routes_path.read_bytes()
    (inputs / 'routes.xml').write_bytes(routes_bytes)
    for name, source in [('variants.json', variants_path), ('route-cruises.json', route_cruises_path)]:
        if source:
            (inputs / name).write_bytes(Path(source).read_bytes())
    manifest_cases = []
    for case in cases:
        archived = inputs / ('controller-%s.json' % case['variant'])
        archived.write_bytes(case['config_bytes'])
        case['archived_controller_config'] = str(archived.resolve())
        manifest_cases.append({k: v for k, v in case.items() if k != 'config_bytes'})
    manifest = dict(routes_source=str(routes_path), routes_sha256=hashlib.sha256(routes_bytes).hexdigest(),
                    ordering='route_then_variant_then_preset', cases=manifest_cases,
                    route_cruises_mps=cruises, variant_directories=bool(variants_path))
    (inputs / 'matrix.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def case_metadata(case, route, cruise):
    return dict(route_id=route.get('id'), town=route.get('town'), preset=case['preset'],
                variant=case['variant'], controller_config=case['controller_config'],
                archived_controller_config=case['archived_controller_config'],
                controller_config_sha256=case['controller_config_sha256'], cruise_mps=cruise,
                rear_axle_offset_m=case['rear_axle_offset_m'],
                stop_deceleration_mps2=case['stop_deceleration_mps2'])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--routes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--controller-config')
    p.add_argument('--variants', help='JSON label -> {controller_config: absolute path, presets: list}')
    p.add_argument('--route-cruises', help='JSON route ID -> cruise m/s, optional default key')
    p.add_argument('--presets', default='carla,tcp,pursuit')
    p.add_argument('--server-index', type=int, default=64)
    p.add_argument('--cruise-mps', type=float, default=8)
    p.add_argument('--max-ticks', type=int, default=1800)
    p.add_argument('--rig', default='none')
    a = p.parse_args()
    routes_path, out = [Path(x).resolve() for x in (a.routes, a.out)]
    cases, cruises = load_matrix(a.controller_config, a.presets, a.variants,
                                 a.route_cruises, a.cruise_mps)
    root = ET.parse(str(routes_path)).getroot()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'servers').mkdir(exist_ok=True)
    matrix_manifest = archive_matrix(out, routes_path, cases, cruises, a.variants, a.route_cruises)
    from b2d_run import Server
    from b2d_route import add_bench2drive_to_path
    add_bench2drive_to_path('/data/third_party/Bench2Drive')
    import carla
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import GameTime
    from leaderboard.utils.route_manipulation import interpolate_trajectory
    from leaderboard.autoagents.agent_wrapper import AgentWrapper
    from b2d_agent import StubAgent
    import b2d_hooks
    b2d_hooks._patch_sensor_tick()
    server = Server(a.server_index, out / 'servers', 'Epic', gpu_rank=0, windowed=True)
    log = (out / 'log.txt').open('a')
    events = (out / 'events.jsonl').open('a')
    results = []

    def event(kind, **fields):
        row = dict(t=time.time(), kind=kind, **fields)
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        for fh in (log, events):
            fh.write(line + '\n')
            fh.flush()

    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupt)
    event('start', configuration=vars(a), matrix=matrix_manifest, pid=os.getpid(), protocol='no_background_diagnostic')
    try:
        server.start()
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'], text=True)
        (out / 'gpu.txt').write_text(gpu)
        event('gpu', processes=gpu)
        client = carla.Client('localhost', server.port)
        client.set_timeout(90)
        for route in root.findall('route'):
            cruise = float(cruises.get(route.get('id'), cruises['default']))
            try:
                world = client.get_world()
                if world.get_map().name.split('/')[-1] != route.get('town'):
                    world = client.load_world(route.get('town'))
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = .05
                settings.substepping = True
                settings.max_substep_delta_time = .01
                settings.max_substeps = 10
                settings.actor_active_distance = 2000
                settings.tile_stream_distance = 3000
                world.apply_settings(settings)
                world.set_weather(carla.WeatherParameters.ClearNoon)
                CarlaDataProvider.set_client(client)
                CarlaDataProvider.set_world(world)
                locations = [carla.Location(float(x.get('x')), float(x.get('y')), float(x.get('z')))
                             for x in route.findall('./waypoints/position')]
                gps, dense = interpolate_trajectory(locations)
            except Exception as exc:
                # A failed map/interpolation setup still represents every planned
                # preset on this route; save those missing cases, then continue.
                setup_traceback = traceback.format_exc()
                for case in cases:
                    preset = case['preset']
                    metadata = case_metadata(case, route, cruise)
                    run = out / route.get('id')
                    if a.variants:
                        run = run / case['variant']
                    run = run / preset
                    run.mkdir(parents=True, exist_ok=True)
                    summary = summarize([], [], cruise, case['stop_deceleration_mps2'], 'setup_error', [], None)
                    summary.update(**metadata,
                                   exception=repr(exc), wall_s=0., cleanup_errors=[], telemetry_parse_errors=[])
                    (run / 'exception.txt').write_text(setup_traceback)
                    (run / 'validation.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
                    results.append(summary)
                    event('route_end', **summary)
                (out / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
                continue
            for case in cases:
                preset = case['preset']
                rear = case['rear_axle_offset_m']
                stop_deceleration = case['stop_deceleration_mps2']
                metadata = case_metadata(case, route, cruise)
                run = out / route.get('id')
                if a.variants:
                    run = run / case['variant']
                run = run / preset
                run.mkdir(parents=True, exist_ok=True)
                cfg = dict(rig=a.rig, width=800, height=450, policy='none', drive='controller', decimate=4,
                           controller_preset=preset, controller_config=case['archived_controller_config'], out=str(run), cruise_mps=cruise)
                cfg_path = run / 'agent_config.json'
                cfg_path.write_text(json.dumps(cfg, indent=2))
                actor = agent = wrapper = collision_sensor = None
                started = time.perf_counter()
                rows, collisions = [], []
                status, hold_start, case_error = 'tick_cap', None, None
                event('route_start', **metadata)
                try:
                    GameTime.restart()
                    spawn = carla.Transform(dense[0][0].location + carla.Location(z=.5), dense[0][0].rotation)
                    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
                    bp.set_attribute('role_name', 'hero')
                    actor = world.spawn_actor(bp, spawn)
                    world.get_spectator().set_transform(carla.Transform(spawn.location + carla.Location(z=30), carla.Rotation(pitch=-90)))
                    for _ in range(40):
                        actor.apply_control(carla.VehicleControl(brake=1))
                        world.tick()
                    agent = StubAgent('localhost', server.port)
                    agent.set_global_plan(gps, dense)
                    agent.setup(str(cfg_path))
                    truth_projection = TruthProjection(agent._route_adapter.points)
                    wrapper = AgentWrapper(agent)
                    wrapper.setup_sensors(actor)
                    collision_sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=actor)
                    collision_sensor.listen(lambda x: collisions.append(dict(frame=x.frame, other=x.other_actor.type_id)))
                    first_time = None
                    for tick in range(a.max_ticks):
                        world.tick()
                        snapshot = world.get_snapshot()
                        GameTime.on_carla_tick(snapshot.timestamp)
                        CarlaDataProvider.on_carla_tick()
                        # Read-only plant diagnostics before applying this tick's
                        # new command. Report the API field without assuming it
                        # alone proves an internal transmission state transition.
                        applied = actor.get_control()
                        control = agent()
                        actor.apply_control(control)
                        actor_snapshot = snapshot.find(actor.id)
                        tf = actor_snapshot.get_transform()
                        velocity = actor_snapshot.get_velocity()
                        yaw = math.radians(tf.rotation.yaw)
                        # Full forward-vector offset includes pitch, important for
                        # evaluating rear-axle location during the slope hold test.
                        forward = tf.get_forward_vector()
                        right = tf.get_right_vector()
                        acceleration = actor_snapshot.get_acceleration()
                        angular_velocity = actor_snapshot.get_angular_velocity()
                        truth = np.array([tf.location.x + rear * forward.x, tf.location.y + rear * forward.y])
                        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                        sim_time = float(snapshot.timestamp.elapsed_seconds)
                        if first_time is None:
                            first_time = sim_time
                        projection = truth_projection.measure(truth, yaw, speed)
                        remaining = float(np.linalg.norm(truth - truth_projection.points[-1]))
                        reference_speed = min(cruise, math.sqrt(2. * stop_deceleration * projection['remaining_along_m']))
                        rows.append(dict(tick=tick, frame=snapshot.frame, sim_time=sim_time, elapsed_s=sim_time - first_time,
                                         speed=speed, signed_speed=velocity.dot(forward), truth_xy=truth.tolist(),
                                         pitch_deg=tf.rotation.pitch, endpoint_error_m=remaining,
                                         applied_control=dict(throttle=applied.throttle, brake=applied.brake,
                                                              steer=applied.steer, gear=applied.gear,
                                                              manual_gear_shift=applied.manual_gear_shift),
                                         plant_kinematics=dict(
                                             acceleration_mps2=[acceleration.x, acceleration.y, acceleration.z],
                                             angular_velocity_deg_s=[angular_velocity.x, angular_velocity.y, angular_velocity.z],
                                             forward_vector=[forward.x, forward.y, forward.z],
                                             right_vector=[right.x, right.y, right.z],
                                             location_m=[tf.location.x, tf.location.y, tf.location.z],
                                             rotation_deg=[tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw]),
                                         reference_speed_mps=reference_speed, **projection))
                        if tick % 100 == 0:
                            event('tick', route_id=route.get('id'), variant=case['variant'], preset=preset, ticks=tick + 1, speed=speed, progress_m=projection['progress_m'])
                        if remaining < 1. and speed < .1:
                            if hold_start is None:
                                hold_start = tick
                            if sim_time - rows[hold_start]['sim_time'] >= 5. - 1e-6:
                                status = 'completed'
                                break
                        else:
                            hold_start = None
                        if collisions:
                            status = 'collision'
                            break
                        # Intentional endpoint parking needs a complete five-second hold.
                        # One hundred samples span only 99 intervals, so the blocked detector
                        # must not preempt the hold one tick before it qualifies.
                        if hold_start is None and tick > 400 and max(x['speed'] for x in rows[-100:]) < .1:
                            status = 'blocked'
                            break
                except BaseException as exc:
                    status = 'cancelled' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'error'
                    case_error = repr(exc)
                    (run / 'exception.txt').write_text(traceback.format_exc())
                finally:
                    cleanup_errors = []
                    for label, cleanup in [('collision_stop', lambda: collision_sensor.stop() if collision_sensor else None),
                                           ('collision_destroy', lambda: collision_sensor.destroy() if collision_sensor else None),
                                           ('wrapper_cleanup', lambda: wrapper.cleanup() if wrapper else None),
                                           ('agent_destroy', lambda: agent.destroy() if agent else None),
                                           ('actor_destroy', lambda: actor.destroy() if actor else None)]:
                        try:
                            cleanup()
                        except Exception as exc:
                            cleanup_errors.append('%s: %r' % (label, exc))
                    (run / 'validation_trace.json').write_text(json.dumps(rows, allow_nan=False))
                    records, parse_errors = [], []
                    telemetry = run / 'control.jsonl'
                    if telemetry.exists():
                        for index, line in enumerate(telemetry.read_text().splitlines()):
                            try:
                                record = json.loads(line)
                                if not isinstance(record, dict):
                                    raise ValueError('telemetry row is not an object')
                                records.append(record)
                            except (TypeError, ValueError) as exc:
                                parse_errors.append(dict(line=index + 1, error=repr(exc)))
                    summary = summarize(rows, records, cruise, stop_deceleration, status, collisions, hold_start)
                    summary.update(**metadata,
                                   wall_s=time.perf_counter() - started, exception=case_error,
                                   cleanup_errors=cleanup_errors, telemetry_parse_errors=parse_errors)
                    if parse_errors or cleanup_errors:
                        summary['gate_pass'] = False
                    results.append(summary)
                    (run / 'validation.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
                    (out / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
                    event('route_end', **summary)
                if status == 'cancelled':
                    raise KeyboardInterrupt()
                world.tick()
        event('end', status='completed', cases=len(results), gate_pass=all(x['gate_pass'] for x in results))
    except BaseException as exc:
        event('end', status='failed', error=repr(exc))
        raise
    finally:
        server.stop()
        log.close()
        events.close()


if __name__ == '__main__':
    main()
