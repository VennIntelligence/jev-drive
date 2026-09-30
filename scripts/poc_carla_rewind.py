#!/usr/bin/env python3
"""
POC CARLA Rewind & Branching Simulation Optimization
Evaluates CARLA 0.9.15 Replayer vs. In-Memory State Restore (Teleport & Velocity Injection)
for multi-branch trajectory generation at fork points.
"""

import argparse
import json
import logging
import math
import os
import queue
import sys
import time
from pathlib import Path

try:
    import carla
except ImportError:
    # Try common local or egg paths
    carla_eggs = list(Path('/root/autodl-tmp/ujs/third_party/carla/CARLA_0.9.15/PythonAPI/carla/dist').glob('*.egg'))
    if carla_eggs:
        sys.path.append(str(carla_eggs[0]))
    import carla

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("poc_carla_rewind")


def get_speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_actor_snapshot(actor):
    return {
        "id": actor.id,
        "type_id": actor.type_id,
        "transform": actor.get_transform(),
        "velocity": actor.get_velocity(),
        "angular_velocity": actor.get_angular_velocity(),
    }


def restore_actor_snapshot(actor, snap):
    actor.set_transform(snap["transform"])
    actor.set_target_velocity(snap["velocity"])
    actor.set_target_angular_velocity(snap["angular_velocity"])


def run_experiment(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    # Settings
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)

    results = {
        "metadata": {
            "carla_version": client.get_server_version(),
            "map": world.get_map().name,
            "fps": int(1.0 / settings.fixed_delta_seconds),
            "fork_time_s": 5.0,
            "branch_time_s": 3.0,
        },
        "baseline_rerun": {},
        "method_a_replayer": {},
        "method_b_state_restore": {},
    }

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    def clean_actors():
        for s in world.get_actors().filter("sensor.*"):
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        for v in world.get_actors().filter("vehicle.*"):
            try:
                v.destroy()
            except Exception:
                pass
        world.tick()

    clean_actors()
    logger.info("Cleared existing actors.")

    rec_filename = "poc_rewind_test.log"

    # =========================================================================
    # Phase 1: Golden Run (Record from t=0.0s to t=5.0s, save state snapshot)
    # =========================================================================
    logger.info("=== Phase 1: Recording Golden Run (0.0s -> 5.0s, 100 ticks) ===")
    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(ego_bp, spawn_points[0])

    # Spawn 3 NPC vehicles nearby
    npcs = []
    for i in range(1, 4):
        npc_bp = bp_lib.filter("vehicle.*")[i % 10]
        npc = world.try_spawn_actor(npc_bp, spawn_points[i * 2])
        if npc:
            npc.set_autopilot(True, args.tm_port)
            npcs.append(npc)

    # Attach RGB camera to ego
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "360")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=ego)
    cam_queue = queue.Queue()
    cam.listen(cam_queue.put)

    # Attach Collision Sensor
    col_bp = bp_lib.find("sensor.other.collision")
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
    col_events = []
    col_sensor.listen(lambda event: col_events.append(event))

    world.tick()
    while not cam_queue.empty():
        cam_queue.get_nowait()

    # Start CARLA Recorder
    client.start_recorder(rec_filename, True)
    logger.info("Recorder started on: %s", rec_filename)

    # Drive straight for 100 ticks (5.0s)
    ego.apply_control(carla.VehicleControl(throttle=0.75, steer=0.0))

    t_golden_start = time.time()
    for tick in range(100):
        world.tick()
        cam_queue.get(timeout=2.0)

    t_golden_end = time.time()
    golden_wall_s = t_golden_end - t_golden_start
    logger.info("Golden run 0->5.0s completed in %.3f s (%.1f FPS)", golden_wall_s, 100.0 / golden_wall_s)

    # Capture snapshot at t=5.0s (fork point)
    fork_ego_state = get_actor_snapshot(ego)
    fork_npcs_state = [get_actor_snapshot(npc) for npc in npcs if npc.is_alive]
    fork_ego_speed = get_speed_kmh(ego)
    fork_ego_loc = ego.get_location()

    logger.info("Fork state at t=5.0s: Pos=(%.2f, %.2f), Speed=%.2f km/h, NPCs=%d",
                fork_ego_loc.x, fork_ego_loc.y, fork_ego_speed, len(fork_npcs_state))

    client.stop_recorder()
    logger.info("Stopped recorder.")

    # =========================================================================
    # Phase 2: Baseline Approach (Rerun from tick 0 for every branch)
    # =========================================================================
    logger.info("=== Phase 2: Evaluating Baseline Approach (Rerun tick 0 -> 100 + 60 ticks branch) ===")
    # To run Branch 1 & Branch 2 in baseline:
    # Each branch requires: clean/respawn or reset -> rerun 100 ticks -> run 60 ticks branch.
    baseline_branch_records = {}

    for branch_name, branch_ctrl in [
        ("branch_1_hard_brake", carla.VehicleControl(throttle=0.0, brake=1.0)),
        ("branch_2_sharp_steer", carla.VehicleControl(throttle=0.75, steer=0.8)),
    ]:
        t_b_start = time.time()
        # Clean & Respawn to simulate restarting from route start
        clean_actors()
        b_ego = world.spawn_actor(ego_bp, spawn_points[0])
        b_npcs = []
        for i in range(1, 4):
            npc = world.try_spawn_actor(bp_lib.filter("vehicle.*")[i % 10], spawn_points[i * 2])
            if npc:
                npc.set_autopilot(True, args.tm_port)
                b_npcs.append(npc)
        b_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=b_ego)
        b_q = queue.Queue()
        b_cam.listen(b_q.put)
        world.tick()
        while not b_q.empty():
            b_q.get_nowait()

        # Step 1: Lead-in run (100 ticks)
        b_ego.apply_control(carla.VehicleControl(throttle=0.75, steer=0.0))
        t_lead_start = time.time()
        for _ in range(100):
            world.tick()
            b_q.get(timeout=2.0)
        t_lead_end = time.time()
        lead_s = t_lead_end - t_lead_start

        # Step 2: Branch execution (60 ticks = 3.0s)
        b_ego.apply_control(branch_ctrl)
        t_exec_start = time.time()
        branch_speeds = []
        branch_yaws = []
        for _ in range(60):
            world.tick()
            b_q.get(timeout=2.0)
            branch_speeds.append(get_speed_kmh(b_ego))
            branch_yaws.append(b_ego.get_transform().rotation.yaw)
        t_exec_end = time.time()
        exec_s = t_exec_end - t_exec_start
        total_branch_s = time.time() - t_b_start

        baseline_branch_records[branch_name] = {
            "lead_wall_s": lead_s,
            "exec_wall_s": exec_s,
            "total_wall_s": total_branch_s,
            "final_speed_kmh": branch_speeds[-1],
            "final_yaw_deg": branch_yaws[-1],
        }
        logger.info("Baseline %s: Lead=%.2fs, Exec=%.2fs, Total=%.2fs, FinalSpeed=%.1f km/h",
                    branch_name, lead_s, exec_s, total_branch_s, branch_speeds[-1])
        b_cam.stop()
        b_cam.destroy()
        b_ego.destroy()
        for n in b_npcs:
            n.destroy()
        world.tick()

    results["baseline_rerun"] = baseline_branch_records

    # =========================================================================
    # Phase 3: Method A - CARLA Official Replayer (replay_file + stop_replayer)
    # =========================================================================
    logger.info("=== Phase 3: Evaluating Method A (CARLA Replayer Handover) ===")
    method_a_records = {}

    for branch_name, branch_ctrl in [
        ("branch_1_hard_brake", carla.VehicleControl(throttle=0.0, brake=1.0)),
        ("branch_2_sharp_steer", carla.VehicleControl(throttle=0.75, steer=0.8)),
    ]:
        clean_actors()
        t_rep_start = time.time()

        # Call replay_file starting directly at 5.0s
        # Note: replay_file(name, time_start, duration, follow_id)
        rep_msg = client.replay_file(rec_filename, 5.0, 0.0, 0)
        world.tick()
        t_rep_loaded = time.time()
        restore_wall_s = t_rep_loaded - t_rep_start

        # Find replayed ego actor
        vehs = [v for v in world.get_actors().filter("vehicle.*") if v.attributes.get("role_name") == "hero"]
        if not vehs:
            vehs = list(world.get_actors().filter("vehicle.*"))
        replayed_ego = vehs[0]

        # Stop replayer keeping actors
        client.stop_replayer(True)
        world.tick()

        # Re-attach camera
        r_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=replayed_ego)
        r_q = queue.Queue()
        r_cam.listen(r_q.put)
        world.tick()
        while not r_q.empty():
            r_q.get_nowait()

        init_speed = get_speed_kmh(replayed_ego)
        # Apply branch control
        replayed_ego.apply_control(branch_ctrl)

        t_exec_start = time.time()
        r_speeds = []
        r_yaws = []
        frames_received = 0
        for _ in range(60):
            world.tick()
            try:
                img = r_q.get(timeout=2.0)
                frames_received += 1
            except queue.Empty:
                pass
            r_speeds.append(get_speed_kmh(replayed_ego))
            r_yaws.append(replayed_ego.get_transform().rotation.yaw)
        t_exec_end = time.time()
        exec_s = t_exec_end - t_exec_start
        total_s = time.time() - t_rep_start

        method_a_records[branch_name] = {
            "restore_wall_s": restore_wall_s,
            "exec_wall_s": exec_s,
            "total_wall_s": total_s,
            "speed_at_handover_kmh": init_speed,
            "final_speed_kmh": r_speeds[-1],
            "final_yaw_deg": r_yaws[-1],
            "frames_received": frames_received,
        }
        logger.info("Method A %s: Restore=%.3fs, Exec=%.2fs, Total=%.2fs, Speed@Handover=%.1f km/h, FinalSpeed=%.1f km/h",
                    branch_name, restore_wall_s, exec_s, total_s, init_speed, r_speeds[-1])
        r_cam.stop()
        r_cam.destroy()
        clean_actors()

    results["method_a_replayer"] = method_a_records

    # =========================================================================
    # Phase 4: Method B - In-Memory State Restore (Teleport & Velocity Injection)
    # =========================================================================
    logger.info("=== Phase 4: Evaluating Method B (In-Memory State Snapshot & Dynamic Restore) ===")
    # Method B runs the baseline lead-in ONCE, keeps all actors and sensors alive in memory,
    # and restores state instantaneously before each branch!
    clean_actors()

    # Spawn once
    b_ego = world.spawn_actor(ego_bp, spawn_points[0])
    b_npcs = []
    for i in range(1, 4):
        npc = world.try_spawn_actor(bp_lib.filter("vehicle.*")[i % 10], spawn_points[i * 2])
        if npc:
            npc.set_autopilot(True, args.tm_port)
            b_npcs.append(npc)
    b_cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=b_ego)
    b_col = world.spawn_actor(col_bp, carla.Transform(), attach_to=b_ego)
    b_q = queue.Queue()
    b_cam.listen(b_q.put)
    b_cols = []
    b_col.listen(lambda e: b_cols.append(e))

    world.tick()
    while not b_q.empty():
        b_q.get_nowait()

    # Run lead-in 100 ticks to reach fork point
    b_ego.apply_control(carla.VehicleControl(throttle=0.75, steer=0.0))
    for _ in range(100):
        world.tick()
        b_q.get(timeout=2.0)

    # Capture live in-memory snapshot
    snap_ego = get_actor_snapshot(b_ego)
    snap_npcs = [get_actor_snapshot(n) for n in b_npcs if n.is_alive]
    logger.info("Method B baseline lead-in reached. Snap ego speed: %.1f km/h", get_speed_kmh(b_ego))

    method_b_records = {}

    for branch_name, branch_ctrl in [
        ("branch_1_hard_brake", carla.VehicleControl(throttle=0.0, brake=1.0)),
        ("branch_2_sharp_steer", carla.VehicleControl(throttle=0.75, steer=0.8)),
    ]:
        t_restore_start = time.time()
        # Teleport and re-inject velocity
        restore_actor_snapshot(b_ego, snap_ego)
        for n, snap_n in zip(b_npcs, snap_npcs):
            if n.is_alive:
                restore_actor_snapshot(n, snap_n)
        world.tick()
        b_q.get(timeout=2.0)
        t_restore_end = time.time()
        restore_wall_s = t_restore_end - t_restore_start

        speed_at_restore = get_speed_kmh(b_ego)

        # Apply branch control
        b_ego.apply_control(branch_ctrl)
        t_exec_start = time.time()
        b_speeds = []
        b_yaws = []
        frames_received = 0
        for _ in range(60):
            world.tick()
            try:
                img = b_q.get(timeout=2.0)
                frames_received += 1
            except queue.Empty:
                pass
            b_speeds.append(get_speed_kmh(b_ego))
            b_yaws.append(b_ego.get_transform().rotation.yaw)
        t_exec_end = time.time()
        exec_s = t_exec_end - t_exec_start
        total_s = restore_wall_s + exec_s

        method_b_records[branch_name] = {
            "restore_wall_s": restore_wall_s,
            "exec_wall_s": exec_s,
            "total_wall_s": total_s,
            "speed_at_restore_kmh": speed_at_restore,
            "final_speed_kmh": b_speeds[-1],
            "final_yaw_deg": b_yaws[-1],
            "frames_received": frames_received,
        }
        logger.info("Method B %s: Restore=%.4fs, Exec=%.2fs, Total=%.2fs, Speed@Restore=%.1f km/h, FinalSpeed=%.1f km/h",
                    branch_name, restore_wall_s, exec_s, total_s, speed_at_restore, b_speeds[-1])

    results["method_b_state_restore"] = method_b_records

    # Clean up
    b_cam.stop()
    b_cam.destroy()
    b_col.stop()
    b_col.destroy()
    clean_actors()
    world.apply_settings(original_settings)

    # Save output json
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20250)
    parser.add_argument("--tm-port", type=int, default=20252)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", default="runs/poc_carla_rewind/results.json")
    args = parser.parse_args()
    run_experiment(args)
