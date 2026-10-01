"""BLUE (SimLingo + 0.11M language gate) single-frame smoke on a real CARLA front-camera frame.

  capture: spawn the Bench2Drive ego (lincoln mkz 2020) with SimLingo's camera rig (1024x512, FOV 110,
           x=-1.5 z=2.0) in a running CARLA 0.9.15 server, drive on TM autopilot among NPCs, grab one frame
           plus GNSS / IMU / speed, then log the ego's next 20 s as ground truth.
  infer:   feed that frame through BLUE's own agent code (team_code/agent_simlingo.py tick() preprocessing
           + DrivingModelGate forward), report gate decision, waypoints, path, control, latency, ADE vs log.

Route: the leaderboard gives SimLingo a sparse GPS route; here the dense route is the ego's own logged future
path (1 m resampled), sparsified with the leaderboard's downsample_route(50) and fed to SimLingo's RoutePlanner.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

BLUE = Path(os.environ.get('BLUE_ROOT', Path.home() / 'data/third_party/blue'))


def capture(a):
    import carla, queue
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    client = carla.Client('127.0.0.1', a.port); client.set_timeout(120)
    world = client.get_world() if a.town is None else client.load_world(a.town)
    orig = world.get_settings(); s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    tm = client.get_trafficmanager(a.tm_port); tm.set_synchronous_mode(True); tm.set_random_device_seed(a.seed)
    bp = world.get_blueprint_library(); rng = np.random.default_rng(a.seed)
    spawns = world.get_map().get_spawn_points()
    ego = world.spawn_actor(bp.find('vehicle.lincoln.mkz_2020'), spawns[a.spawn % len(spawns)])
    npcs = []
    vbps = [b for b in bp.filter('vehicle.*') if int(b.get_attribute('number_of_wheels')) == 4]
    for i in rng.permutation(len(spawns))[:a.npcs]:
        v = world.try_spawn_actor(vbps[rng.integers(len(vbps))], spawns[i])
        if v: v.set_autopilot(True, a.tm_port); npcs.append(v)
    ego.set_autopilot(True, a.tm_port)
    cam_bp = bp.find('sensor.camera.rgb')
    for k, v in dict(image_size_x=1024, image_size_y=512, fov=110).items(): cam_bp.set_attribute(k, str(v))
    q = {k: queue.Queue() for k in ('rgb', 'gps', 'imu')}
    sens = [world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-1.5, z=2.0)), attach_to=ego),
            world.spawn_actor(bp.find('sensor.other.gnss'), carla.Transform(), attach_to=ego),
            world.spawn_actor(bp.find('sensor.other.imu'), carla.Transform(), attach_to=ego)]
    for sn, k in zip(sens, q): sn.listen(q[k].put)
    get = lambda k, f: next(m for m in iter(lambda: q[k].get(timeout=30), None) if m.frame == f)
    fwd_speed = lambda: float(np.dot([ego.get_velocity().x, ego.get_velocity().y, ego.get_velocity().z],
                                     [ego.get_transform().get_forward_vector().x, ego.get_transform().get_forward_vector().y,
                                      ego.get_transform().get_forward_vector().z]))
    try:
        t = 0
        while True:  # warm up, then wait until the ego is actually moving
            f = world.tick(); t += 1
            img, gnss, imu = get('rgb', f), get('gps', f), get('imu', f)
            if t >= a.warm and fwd_speed() > 3.0 or t > a.warm + 600: break
        tr = ego.get_transform()
        bgra = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4).copy()
        import cv2; cv2.imwrite(str(out / 'front.png'), bgra[:, :, :3])
        geo = world.get_map().transform_to_geolocation(carla.Location(0, 0, 0))
        meta = dict(town=world.get_map().name, frame=f, tick=t, spawn=a.spawn, seed=a.seed, npcs=len(npcs),
                    speed=fwd_speed(), gnss=[gnss.latitude, gnss.longitude, gnss.altitude],
                    imu=[imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z,
                         imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z, imu.compass],
                    ego=[tr.location.x, tr.location.y, tr.location.z, tr.rotation.yaw],
                    lat_ref=geo.latitude, lon_ref=geo.longitude)
        fut = []
        for _ in range(a.future):
            world.tick(); l = ego.get_transform().location; fut.append([l.x, l.y, l.z])
        np.save(out / 'future_world.npy', np.array(fut))  # 20 Hz, starting one tick after the frame
        (out / 'capture.json').write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta))
    finally:
        for sn in sens: sn.stop()
        world.tick()
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in sens + npcs + [ego]])
        tm.set_synchronous_mode(False); world.apply_settings(orig)


def infer(a):
    import torch, carla
    sys.path[:0] = [str(BLUE), str(BLUE / 'team_code'), str(BLUE / 'Bench2Drive/leaderboard'),
                    str(BLUE / 'Bench2Drive/scenario_runner'), os.environ['CARLA_ROOT'] + '/PythonAPI/carla']
    os.chdir(BLUE)  # the agent resolves ./pretrained/InternVL2-1B relative to cwd
    os.environ.update(BLUE_MODE='trained_gate', BLUE_GATE_CKPT=a.gate, BLUE_GATE_THRESHOLD=str(a.threshold),
                      SAVE_PATH=a.out, ROUTES='')
    # torch>=2.6 loads weights_only by default; the gate .pt stores numpy scalars in its config (torch 2.2 era).
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, *[type(np.dtype(t)) for t in 'fdil?']])
    import agent_simlingo as A
    from agents.navigation.local_planner import RoadOption
    from leaderboard.utils.route_manipulation import downsample_route
    from team_code.nav_planner import RoutePlanner
    import team_code.transfuser_utils as t_u
    cap = Path(a.cap); meta = json.loads((cap / 'capture.json').read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    agent = A.LingoAgent.__new__(A.LingoAgent)
    t0 = time.perf_counter(); agent.setup(a.ckpt + '+smoke'); t_load = time.perf_counter() - t0
    miss = agent.model.load_state_dict(torch.load(a.ckpt), strict=False)  # re-check what strict=False hides
    model = agent.model.eval()

    # Route: logged future path -> 1 m dense -> leaderboard downsample(50) -> SimLingo RoutePlanner (as _init does).
    ego = np.array(meta['ego']); fut = np.load(cap / 'future_world.npy')
    path = np.vstack([ego[None, :3], fut]); seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    s = np.arange(0, seg[-1], 1.0)
    dense = np.stack([np.interp(s, seg, path[:, i]) for i in range(3)], 1)
    route = [(carla.Transform(carla.Location(*map(float, p))), RoadOption.LANEFOLLOW) for p in dense]
    plan = [route[i] for i in downsample_route(route, 50)]
    agent._route_planner = RoutePlanner(agent.route_planner_min_distance, agent.route_planner_max_distance,
                                        meta['lat_ref'], meta['lon_ref'])
    agent._route_planner.set_route(plan, False)
    agent.initialized, agent.metric_info, agent.step = True, {}, 10**4
    agent.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

    import cv2
    bgr = cv2.imread(str(cap / 'front.png')); bgra = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
    f = meta['frame']
    inp = {'rgb_0': (f, bgra), 'gps': (f, np.array(meta['gnss'])), 'imu': (f, np.array(meta['imu'])),
           'speed': (f, {'speed': meta['speed']})}
    t0 = time.perf_counter(); tick = agent.tick(inp); t_tick = time.perf_counter() - t0
    gps_err = float(np.linalg.norm(agent._route_planner.convert_gps_to_carla(np.array(meta['gnss']))[:2] - ego[:2]))
    di = A.DrivingInput(**agent.DrivingInput)

    def timed(n_warm=5, n=20):
        ts = []
        for i in range(n_warm + n):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad(): r = model(di)
            torch.cuda.synchronize(); (i >= n_warm) and ts.append((time.perf_counter() - t) * 1e3)
        ts = np.array(ts)
        return r, dict(mean=ts.mean(), p50=np.median(ts), p95=np.percentile(ts, 95), n=n, warmup=n_warm)

    torch.cuda.reset_peak_memory_stats()
    (wps, rte, lang), lat = timed()
    dec, score = model.gate_decisions[0], model.gate_scores[0]
    peak = torch.cuda.max_memory_allocated() / 2**30
    steer, thr, brk = agent.control_pid(rte.float(), tick['speed'], wps.float())
    wps, rte = wps.float()[0].cpu().numpy(), rte.float()[0].cpu().numpy()

    # Forced branches for context: gate always-direct (threshold>1) and always-language (threshold<0).
    forced = {}
    for name, th in (('direct', 2.0), ('language', -1.0)):
        model.gate.threshold = th
        (w2, r2, l2), forced[name] = timed(3, 10)
        forced[name]['wps'] = w2.float()[0].cpu().numpy().round(2).tolist(); forced[name]['language'] = l2
    model.gate.threshold = a.threshold

    # Ground truth in the ego frame at the capture (vehicle origin, x forward, y right; same as SimLingo labels).
    yaw = math.radians(ego[3])
    gt = np.array([t_u.inverse_conversion_2d(p[:2], ego[:2], yaw) for p in fut])
    k = np.arange(1, len(wps) + 1) * 5 - 1  # waypoint i <-> t = 0.25 s * (i + 1) at 20 Hz
    gtw = gt[k[k < len(gt)]]
    ade = float(np.linalg.norm(wps[:len(gtw)] - gtw, axis=1).mean()); fde = float(np.linalg.norm(wps[len(gtw) - 1] - gtw[-1]))
    gpath = dense[:21]; gpath = np.array([t_u.inverse_conversion_2d(p[:2], ego[:2], yaw) for p in gpath])[1:]
    path_ade = float(np.linalg.norm(rte[:len(gpath)] - gpath[:len(rte)], axis=1).mean())

    summ = dict(model='BLUE (SimLingo epoch=013 + blue_simlingo_gate.pt)', capture=meta,
                gpu=torch.cuda.get_device_name(0), load_s=t_load, tick_preproc_ms=t_tick * 1e3,
                missing_keys=miss.missing_keys, unexpected_keys=miss.unexpected_keys,
                gate=dict(decision=int(dec), language_on=bool(dec), score=float(score), threshold=a.threshold),
                language=lang, prompt=agent.prompt, target_points_ego=np.array(agent.target_points).round(2).tolist(),
                gps_to_carla_err_m=gps_err,
                input_shapes=dict(camera_images=list(di.camera_images.shape), prompt_tokens=int(di.prompt.phrase_ids.shape[1])),
                latency_ms=lat, latency_span='model(DrivingInput): LLM prefill + gate MLP + (direct: 2nd forward with driving '
                'tokens | language: greedy decode <=100 tokens + driving forward); excludes tick() preprocessing',
                forced_branch_latency_ms=forced, peak_vram_gb=peak,
                waypoints_ego=wps.round(3).tolist(), waypoints_dt_s=0.25, path_ego_1m=rte.round(3).tolist(),
                control=dict(steer=float(steer), throttle=float(thr), brake=bool(brk)),
                gt_waypoints_ego=gtw.round(3).tolist(), ade_m=ade, fde_m=fde, path_ade_m=path_ade)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: summ[k] for k in ('gate', 'language', 'prompt', 'latency_ms', 'peak_vram_gb', 'waypoints_ego',
                                           'gt_waypoints_ego', 'ade_m', 'fde_m', 'path_ade_m', 'control', 'missing_keys',
                                           'unexpected_keys', 'gps_to_carla_err_m', 'input_shapes', 'tick_preproc_ms')},
                     indent=1, default=float))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture'); c.add_argument('--out', required=True); c.add_argument('--port', type=int, default=3850)
    c.add_argument('--tm-port', type=int, default=9850); c.add_argument('--town', default=None)
    c.add_argument('--spawn', type=int, default=0); c.add_argument('--seed', type=int, default=1)
    c.add_argument('--npcs', type=int, default=40); c.add_argument('--warm', type=int, default=200)
    c.add_argument('--future', type=int, default=400)
    i = sub.add_parser('infer'); i.add_argument('--cap', required=True); i.add_argument('--out', required=True)
    i.add_argument('--ckpt', required=True); i.add_argument('--gate', required=True)
    i.add_argument('--threshold', type=float, default=0.66)
    a = p.parse_args(); capture(a) if a.cmd == 'capture' else infer(a)
