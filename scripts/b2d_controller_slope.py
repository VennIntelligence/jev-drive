#!/usr/bin/env python
"""Optional privileged slope plant diagnostic on a caller-owned CARLA server.

run(client, config, out) never starts or stops a server. It may load Town12 when
the current map has no suitable slope, after confirming no existing vehicles
would be destroyed. Call after evaluation actors have been cleaned up. Only the
helper's own MKZ is destroyed. World settings are restored on the final map.

This measures brake holding with stationary trajectories; it is not a sensor
localization, stopping-distance, planner, or leaderboard-score test.
"""
import argparse
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
from b2d_controller import Controller


def _pitch(angle):
    return (float(angle) + 180.) % 360. - 180.


def _select_candidates(world_map, minimum_pitch=3., maximum_pitch=15., limit=8):
    """Use only static driving-lane geometry, with spatially distinct sites."""
    choices = []
    for waypoint in world_map.generate_waypoints(10.):
        pitch = _pitch(waypoint.transform.rotation.pitch)
        if (waypoint.is_junction or waypoint.lane_width < 3. or
                not minimum_pitch <= abs(pitch) <= maximum_pitch):
            continue
        # generate_waypoints normally returns driving lanes; check explicitly
        # without importing CARLA so static selection can be tested offline.
        if 'Driving' not in str(waypoint.lane_type):
            continue
        choices.append(waypoint)
    choices.sort(key=lambda wp: (-abs(_pitch(wp.transform.rotation.pitch)),
                                wp.road_id, wp.lane_id, wp.s))
    selected = []
    for waypoint in choices:
        p = waypoint.transform.location
        if all(math.hypot(p.x - w.transform.location.x, p.y - w.transform.location.y) >= 20.
               for w in selected):
            selected.append(waypoint)
        if len(selected) >= limit:
            break
    return selected


def hold_summary(rows, minimum_pitch=3.):
    """Evaluate raw signed motion and 3D axle positions; never clamp reverse."""
    if len(rows) < 2:
        return dict(status='untested', reason='insufficient_hold_samples', gate_pass=False,
                    samples=len(rows), duration_s=None, displacement_m=None)
    positions = np.asarray([row['rear_xyz'] for row in rows], dtype=float)
    finite = np.isfinite(positions).all() and all(np.isfinite([
        row['sim_time'], row['signed_speed_mps'], row['speed_mps'], row['pitch_deg']]).all() for row in rows)
    if not finite:
        return dict(status='error', reason='nonfinite_truth', gate_pass=False, samples=len(rows))
    diameter = float(max(np.max(np.linalg.norm(positions - point, axis=1)) for point in positions))
    duration = float(rows[-1]['sim_time'] - rows[0]['sim_time'])
    pitch = np.asarray([row['pitch_deg'] for row in rows])
    signed = [row['signed_speed_mps'] for row in rows]
    max_speed = max(row['speed_mps'] for row in rows)
    minimum_grade_pitch = float(np.min(np.abs(pitch)))
    gates = dict(duration=duration >= 5. - 1e-6, displacement=diameter <= .1,
                 speed=max_speed < .1, measured_slope=minimum_grade_pitch >= minimum_pitch)
    return dict(status='passed' if all(gates.values()) else 'failed', gate_pass=all(gates.values()),
                gates=gates, samples=len(rows), duration_s=duration, displacement_m=diameter,
                max_speed_mps=float(max_speed), min_signed_speed_mps=float(min(signed)),
                max_signed_speed_mps=float(max(signed)), reverse_samples=sum(v < 0. for v in signed),
                pitch_median_deg=float(np.median(pitch)), min_abs_pitch_deg=minimum_grade_pitch,
                grade_percent=float(100. * math.tan(math.radians(float(np.median(pitch))))),
                displacement_definition='maximum_pairwise_3D_rear_axle_distance',
                endpoint_displacement_m=float(np.linalg.norm(positions[-1] - positions[0])))


def run(client, config, out):
    """Run at most eight static slope sites and return a JSON-compatible result.

config accepts a controller-parameter JSON path or dict. Optional config['slope']
contains allow_town12 (default True), minimum_pitch_deg (default 3), and max_sites
(default 8, upper bound 8). Existing caller vehicles suppress map replacement.
A failed measured hold ends the test; sites are retried only for unavailable
spawn positions, insufficient actual slope, or inability to settle.
"""
    import carla
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
    settings_cfg = configuration.get('slope', {})
    minimum_pitch = float(settings_cfg.get('minimum_pitch_deg', 3.))
    max_sites = min(8, int(settings_cfg.get('max_sites', 8)))
    if not math.isfinite(minimum_pitch) or minimum_pitch < 3. or max_sites < 1:
        raise ValueError('minimum_pitch_deg must be >=3 and max_sites >=1')
    allowed = set(inspect.signature(Controller).parameters)
    parameters = {key: value for key, value in configuration.items() if key in allowed}
    parameters.setdefault('preset', 'pursuit')
    parameters['dt'] = .05
    adapter = configuration.get('adapter', {})
    rear_offset = float(configuration.get('rear_axle_offset_m', adapter.get('rear_axle_offset_m', -1.388633220199954)))
    event_file = (out / 'events.jsonl').open('a')
    trace_file = (out / 'slope.jsonl').open('a')
    log_file = (out / 'log.txt').open('a')
    started = time.perf_counter()
    attempts = []
    vehicle = None
    restore_settings = None
    world = None
    result = dict(status='untested', reason='no_suitable_slope', gate_pass=False)

    def event(kind, **fields):
        record = dict(t=time.time(), kind=kind, **fields)
        line = json.dumps(record, allow_nan=False)
        event_file.write(line + '\n')
        event_file.flush()
        log_file.write(line + '\n')
        log_file.flush()

    try:
        world = client.get_world()
        initial_map = world.get_map().name
        event('start', map=initial_map, parameters=parameters, minimum_pitch_deg=minimum_pitch,
              protocol='privileged_stationary_trajectory_plant_diagnostic')
        candidates = _select_candidates(world.get_map(), minimum_pitch, limit=max_sites)
        if not candidates and settings_cfg.get('allow_town12', True):
            vehicles = list(world.get_actors().filter('vehicle.*'))
            if vehicles:
                result['reason'] = 'no_current_map_slope_and_existing_vehicles_prevent_map_replacement'
            elif initial_map.split('/')[-1] != 'Town12':
                available = [name for name in client.get_available_maps() if name.split('/')[-1] == 'Town12']
                if available:
                    event('map_load', town='Town12')
                    world = client.load_world('Town12')
                    candidates = _select_candidates(world.get_map(), minimum_pitch, limit=max_sites)
                else:
                    result['reason'] = 'no_current_map_slope_and_town12_unavailable'
        result.update(initial_map=initial_map, map=world.get_map().name,
                      client_version=client.get_client_version(), server_version=client.get_server_version())
        if candidates:
            restore_settings = world.get_settings()
            synchronous = world.get_settings()
            synchronous.synchronous_mode = True
            synchronous.fixed_delta_seconds = .05
            synchronous.substepping = True
            synchronous.max_substep_delta_time = .01
            synchronous.max_substeps = 10
            world.apply_settings(synchronous)
        blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020') if candidates else None
        if blueprint is not None:
            blueprint.set_attribute('role_name', 'slope_diagnostic')
        for candidate in candidates:
            location, rotation = candidate.transform.location, candidate.transform.rotation
            site = dict(road_id=candidate.road_id, lane_id=candidate.lane_id, s=candidate.s,
                        map_pitch_deg=_pitch(rotation.pitch), spawn_xyz=[location.x, location.y, location.z])
            attempts.append(site)
            spawn = carla.Transform(carla.Location(location.x, location.y, location.z + .35), rotation)
            vehicle = world.try_spawn_actor(blueprint, spawn)
            if vehicle is None:
                site['status'] = 'spawn_unavailable'
                event('site_end', **site)
                continue
            controller = Controller(**parameters)
            zero = np.zeros((20, 2), dtype=float)
            # The initial full brake avoids an uncontrolled tick while the newly
            # spawned vehicle falls the small clearance distance onto the road.
            vehicle.apply_control(carla.VehicleControl(brake=1.))
            stable_ticks = 0
            hold_rows = []
            hold_start = None
            site['status'] = 'did_not_settle'
            for tick in range(241):
                world.tick()
                snapshot = world.get_snapshot()
                actor = snapshot.find(vehicle.id)
                if actor is None:
                    raise RuntimeError('slope diagnostic actor missing from current snapshot')
                transform, velocity = actor.get_transform(), actor.get_velocity()
                forward = transform.get_forward_vector()
                position = transform.location
                signed_speed = velocity.x * forward.x + velocity.y * forward.y + velocity.z * forward.z
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                angular = actor.get_angular_velocity()
                stamp = float(snapshot.timestamp.elapsed_seconds)
                if tick % 4 == 0:
                    controller.update(zero, stamp)
                # Input magnitude follows Controller's nonnegative speed API.
                # Signed velocity is separately logged/evaluated without a
                # deadband or clipping, including arbitrarily small reverse.
                throttle, steer, brake = controller.step(stamp, speed, -math.radians(angular.z))
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                row = dict(site=len(attempts) - 1, tick=tick, frame=snapshot.frame, sim_time=stamp,
                           phase='hold' if hold_start is not None else 'settle',
                           rear_xyz=[position.x + rear_offset * forward.x,
                                     position.y + rear_offset * forward.y,
                                     position.z + rear_offset * forward.z],
                           signed_speed_mps=signed_speed, speed_mps=speed,
                           pitch_deg=_pitch(transform.rotation.pitch), throttle=throttle, steer=steer,
                           brake=brake, reason=controller.diagnostics['reason'])
                stable_ticks = stable_ticks + 1 if speed < .1 else 0
                if hold_start is None and tick >= 40 and stable_ticks >= 20:
                    if abs(row['pitch_deg']) < minimum_pitch:
                        site.update(status='actual_slope_below_threshold', actual_pitch_deg=row['pitch_deg'])
                        trace_file.write(json.dumps(row, allow_nan=False) + '\n')
                        break
                    hold_start = stamp
                    row['phase'] = 'hold'
                trace_file.write(json.dumps(row, allow_nan=False) + '\n')
                trace_file.flush()
                if hold_start is not None:
                    hold_rows.append(row)
                    if stamp - hold_start >= 5. - 1e-6:
                        hold_result = hold_summary(hold_rows, minimum_pitch)
                        site.update(status=hold_result['status'], hold=hold_result)
                        result.update(hold_result)
                        result['reason'] = None if hold_result['gate_pass'] else 'hold_gate_failed'
                        result['site'] = site
                        break
                elif tick >= 120:
                    break
            vehicle.destroy()
            vehicle = None
            event('site_end', **site)
            # A measured hold failure is evidence, not a reason to select a
            # better outcome at another site. Only unsuitable setups retry.
            if hold_start is not None:
                break
        if attempts and result['status'] == 'untested':
            result['reason'] = 'no_site_produced_a_qualified_five_second_hold'
    except BaseException as exc:
        result.update(status='cancelled' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'error',
                      reason=repr(exc), gate_pass=False)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        cleanup_errors = []
        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        if world is not None and restore_settings is not None:
            try:
                world.apply_settings(restore_settings)
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        result.update(attempts=attempts, wall_s=time.perf_counter() - started,
                      controller_parameters=parameters, rear_axle_offset_m=rear_offset,
                      protocol='privileged_stationary_trajectory_plant_diagnostic',
                      server_owned=False, cleanup_errors=cleanup_errors,
                      limitations=['Fullzero trajectories test brake holding after settling, not approach braking.',
                                   'True state enters this plant diagnostic only; it is not a sensor-score test.'])
        if cleanup_errors:
            result['gate_pass'] = False
        (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        event('end', **result)
        event_file.close()
        trace_file.close()
        log_file.close()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True, help='Existing caller-owned CARLA server port')
    parser.add_argument('--controller-config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.)
    result = run(client, args.controller_config, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['gate_pass'] else (2 if result['status'] == 'untested' else 1)


if __name__ == '__main__':
    raise SystemExit(main())
