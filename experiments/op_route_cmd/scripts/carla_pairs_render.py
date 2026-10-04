"""CARLA counterfactual route pairs, stage 3 step 2: render the openpilot model frames of every pose of a plan (envs/carla, one CARLA
server + this client per process; the server is started and stopped here, owned by pid).

Rig: free (unattached) RGB cameras teleported along the ego lane, so no vehicle body, hood or tinted windshield is in view. Camera
1.22 m above the road surface (openpilot device height), level (pitch = road pitch, roll 0, calibration rpy = 0), 1.519 m ahead of the
rear axle (the P4 / Waymo FRONT camera position the rotation-only warp assumes). The model frames are cut from the render with the SAME
rays as the real-data pipeline (jevdrive.camgeom: OP_K road f 910, cy 47.6 and wide f 455, cy 151.8 in a 512 x 256 frame, so the
horizon rows are the nominal ones) and packed like wod_zeroshot_openpilot._pack. Both model frames lie within +-32 deg of the optical
axis, so one pinhole front camera covers them: this is geometrically what the three-camera (front + two 45 deg side) stitch of
p5_openpilot.render delivers, minus the side cameras (`--compare3` renders both and prints the difference).

Per pose: `frames` (10, 2, 6, 128, 256) uint8 [road, wide] at 5 Hz, t0 - 1.8 s ... t0 (oldest first), plus frame_t, the rig poses and
the settle statistics. History is synthetic (lane-centred kinematics from the plan, no traffic). Exposure adapts fast
(`exposure_speed_*`), and after a teleport the first frame is only taken once consecutive renders agree (tile streaming, exposure).

  $DATA_DIR/envs/carla/bin/python carla_pairs_render.py --plan <poses.pkl> --out <dir> --gpu 2 --idx 240 --cpus 102-103,206-207 \
      [--shard 0/2] [--limit N] [--hero] [--compare3 3]
  -> <dir>/frames/<town>/<pose id>.npz, <dir>/render_<shard>.jsonl (one line per pose: timings, settle ticks), DONE / ERROR
"""
import argparse, json, math, os, pickle, queue, signal, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import p5_openpilot as P5  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data")))
CAM_XYZ = (1.519, 0.026, 1.22)            # rear axle -> front camera (x fwd, y left, height above the road), m
FOCAL = 1113.5                            # px; any value works, the model frames are cut from rays
MARGIN_PX = 3
SETTLE_MIN, SETTLE_MAX, SETTLE_EPS = 2, 60, 0.35
LARGE = ("Town11", "Town12", "Town13", "Town15")
RIG3 = P5.RIG                              # the P4 three-camera rig (front, front_left, front_right), for --compare3


# ---------------------------------------------------------------- model-frame maps
def calib(cams, W, H, f=FOCAL):
    out = {}
    for i, (_, x, y, z, yaw) in enumerate(cams, 1):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        E = np.eye(4)
        E[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        E[:3, 3] = x, y, z
        out[i] = {"intrinsic": [f, f, W / 2, H / 2, 0, 0, 0, 0, 0], "extrinsic": E.ravel().tolist(), "width": W, "height": H}
    return out


def build_maps(cams):
    """Per camera render size and, for road / wide, per camera (mapx, mapy, mask) cv2.remap maps into the (256, 512) model frame."""
    big = 8000
    ext = np.zeros((len(cams), 2))
    for k in ("road", "wide"):
        src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), calib(cams, big, big))
        for i in range(len(cams)):
            m = src == i + 0
            if m.any():
                ext[i] = np.maximum(ext[i], [np.abs(U[m] - big / 2).max(), np.abs(V[m] - big / 2).max()])
    # keys of calib() are 1-based; choose_sources returns the index into list(calibs)
    sizes = [(int(2 * math.ceil(e[0] + MARGIN_PX)), int(2 * math.ceil(e[1] + MARGIN_PX))) for e in ext]
    sizes = [(w, h) if w > 0 else (16, 16) for w, h in sizes]
    W, H = max(s[0] for s in sizes), max(s[1] for s in sizes)
    cal = calib(cams, W, H)
    maps = {}
    for k in ("road", "wide"):
        src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), cal)
        assert (src >= 0).all(), f"{k}: model frame not covered by the render"
        maps[k] = [((U - 0.5).astype(np.float32), (V - 0.5).astype(np.float32), src == i) for i in range(len(cams))]
    return (W, H), maps


def model_frame(imgs, maps_k):
    """imgs: list of BGRA (H, W, 4) renders -> (256, 512, 3) uint8 YCbCr (full range, as libjpeg) of the model frame."""
    out = np.zeros((G.OP_H, G.OP_W, 3), np.uint8)
    for im, (mx, my, mask) in zip(imgs, maps_k):
        if not mask.any():
            continue
        r = cv2.remap(im[:, :, :3], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[mask] = r[mask]
    ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    return ycc[..., [0, 2, 1]]


def pack(ycc):
    """(256, 512, 3) YCbCr -> (6, 128, 256), wod_zeroshot_openpilot._pack."""
    Y = ycc[..., 0]
    uv = np.rint(ycc[..., 1:].reshape(128, 2, 256, 2, 2).astype(np.float32).mean((1, 3))).astype(np.uint8)
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, idx, gpu, cpus, log):
        self.port = 2000 + 50 * idx
        root = os.environ.get("CARLA_ROOT", str(DATA / "third_party" / "carla" / "CARLA_0.9.15"))
        icd = "/etc/vulkan/icd.d/nvidia_icd.json"
        env = dict(os.environ, VK_ICD_FILENAMES=icd)
        env.pop("SDL_VIDEODRIVER", None)
        cpuset = set()
        for part in cpus.split(","):
            a, _, b = part.partition("-")
            cpuset |= set(range(int(a), int(b or a) + 1))
        cmd = [f"{root}/CarlaUE4.sh", "-RenderOffScreen", "-nosound", f"-carla-rpc-port={self.port}", f"-graphicsadapter={gpu}",
               "-quality-level=Epic", "-RPCThreads=4", "-StreamingThreads=4", "-SecondaryThreads=4"]
        self.p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
                                  preexec_fn=lambda: os.sched_setaffinity(0, cpuset))
        self.pgid = self.p.pid

    def stop(self):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                return
            time.sleep(3)


def connect(port, tries=60):
    import carla
    for _ in range(tries):
        try:
            c = carla.Client("127.0.0.1", port, worker_threads=8)
            c.set_timeout(120.0)
            c.get_server_version()
            return c
        except RuntimeError:
            time.sleep(3)
    raise RuntimeError("no CARLA server on port %d" % port)


# ---------------------------------------------------------------- world / camera
class Rig:
    def __init__(self, client, sizes, cams, hero):
        import carla
        self.carla, self.client, self.sizes, self.cams, self.use_hero = carla, client, sizes, cams, hero
        self.world, self.town, self.sensors, self.q, self.hero = None, None, [], None, None

    def load(self, town):
        carla = self.carla
        self.teardown()
        self.world = self.client.load_world(town)
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds, s.no_rendering_mode = True, 0.05, False
        if town in LARGE:
            s.tile_stream_distance, s.actor_active_distance = 650.0, 650.0
        self.world.apply_settings(s)
        self.town = town
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.q = queue.Queue()
        self.sensors = []
        for i, ((W, H), _) in enumerate(zip(self.sizes, self.cams)):
            b = self.world.get_blueprint_library().find("sensor.camera.rgb")
            b.set_attribute("image_size_x", str(W))
            b.set_attribute("image_size_y", str(H))
            b.set_attribute("fov", str(math.degrees(2 * math.atan(W / 2 / FOCAL))))
            for k, v in (("exposure_speed_up", "100.0"), ("exposure_speed_down", "100.0")):
                if b.has_attribute(k):
                    b.set_attribute(k, v)
            cam = self.world.spawn_actor(b, carla.Transform(carla.Location(0, 0, -200)))
            cam.listen(lambda im, i=i: self.q.put((i, im.frame, im)))
            self.sensors.append(cam)
        self.hero = None
        if self.use_hero:
            vb = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
            vb.set_attribute("role_name", "hero")
            self.hero = self.world.spawn_actor(vb, carla.Transform(carla.Location(0, 0, -300)))
            self.hero.set_simulate_physics(False)

    def teardown(self):
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except RuntimeError:
                pass
        if self.hero is not None:
            try:
                self.hero.destroy()
            except RuntimeError:
                pass
        self.sensors, self.hero = [], None

    def set_pose(self, x, y, z, yaw, pitch):
        """Rig pose = rear axle on the road (CARLA coordinates, yaw deg clockwise)."""
        carla = self.carla
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for sensor, (_, (cx, cy, cz, cyaw)) in zip(self.sensors, zip(self.sizes, self.cams)):
            # cams: (x fwd, y left, z, yaw deg left+) in the vehicle frame; CARLA y is right, yaw clockwise
            lx, ly = cx, -cy
            loc = carla.Location(x + c * lx - s * ly, y + s * lx + c * ly, z + cz)
            sensor.set_transform(carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw - cyaw, roll=0.0)))
        if self.hero is not None:
            self.hero.set_transform(carla.Transform(carla.Location(x - 60 * c, y - 60 * s, z - 2.0), carla.Rotation(yaw=yaw)))

    def grab(self, timeout=20.0):
        """tick once, return the BGRA renders of the cameras for that frame."""
        f = self.world.tick()
        got, deadline = {}, time.time() + timeout
        while len(got) < len(self.sensors):
            try:
                i, fr, im = self.q.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                raise TimeoutError("camera frame %d missing" % f)
            if fr == f:
                got[i] = np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4)
        return [got[i] for i in range(len(self.sensors))]

    def weather(self, name):
        self.world.set_weather(getattr(self.carla.WeatherParameters, name))


def small(im):
    return cv2.resize(im[:, :, :3], (64, 40), interpolation=cv2.INTER_AREA).astype(np.float32)


# ---------------------------------------------------------------- one pose
def render_pose(rig, maps, pose, stats):
    h = pose["hist"]
    n = len(h["x"])
    stopped = pose["profile"] == "stopped"
    order = [n - 1] if stopped else list(range(n))
    rig.weather(pose["weather"])
    t0 = time.perf_counter()
    rig.set_pose(h["x"][order[0]], h["y"][order[0]], h["z"][order[0]], h["yaw"][order[0]], h["pitch"][order[0]])
    prev, ticks = None, 0
    while True:                                   # settle: teleport, tile streaming, exposure
        imgs = rig.grab()
        ticks += 1
        cur = small(imgs[0])
        if prev is not None and ticks >= SETTLE_MIN and np.abs(cur - prev).mean() < SETTLE_EPS:
            break
        prev = cur
        if ticks >= SETTLE_MAX:
            stats["unsettled"] = True
            break
    stats["settle_ticks"] = ticks
    t_settle = time.perf_counter()
    frames = []
    for j, k in enumerate(order):
        if j:
            rig.set_pose(h["x"][k], h["y"][k], h["z"][k], h["yaw"][k], h["pitch"][k])
            imgs = rig.grab()
        frames.append(np.stack([pack(model_frame(imgs, maps[m])) for m in ("road", "wide")]))
    if stopped:
        frames = frames * n
    t1 = time.perf_counter()
    stats.update(renders=ticks + len(order) - 1, wall_settle=t_settle - t0, wall_frames=t1 - t_settle, mean_y=float(frames[-1][0][:4].mean()))
    return np.stack(frames)


def compare3(rig_front, client, port, poses, a):
    """Render a few poses with the front camera alone and with the three-camera rig; print the model-frame difference."""
    sizes3, maps3 = build_maps([(n, x, y, z, yw) for n, x, y, z, yw in [(r[0], r[1], r[2], 1.22, r[4]) for r in RIG3]])
    cams3 = [(r[1], r[2], 1.22, r[4]) for r in RIG3]
    r3 = Rig(client, [sizes3] * 3, cams3, a.hero)
    r3.load(poses[0]["town"])
    for p in poses:
        h = p["hist"]
        for rig in (r3,):
            rig.weather(p["weather"])
            rig.set_pose(h["x"][-1], h["y"][-1], h["z"][-1], h["yaw"][-1], h["pitch"][-1])
            for _ in range(8):
                im3 = rig.grab()
        d = {}
        for m in ("road", "wide"):
            d[m] = model_frame(im3, maps3[m])
        yield p, d


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--cpus", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hero", action="store_true", help="spawn a hero vehicle 60 m behind the camera (Large Map tile streaming)")
    ap.add_argument("--compare3", type=int, default=0, help="N poses: also render the 3-camera rig and print the model-frame difference")
    ap.add_argument("--towns", default="")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    k, n = map(int, a.shard.split("/"))
    poses = pickle.load(open(a.plan, "rb"))
    if a.towns:
        poses = [p for p in poses if p["town"] in a.towns.split(",")]
    poses = [p for i, p in enumerate(poses) if i % n == k]
    if a.limit:
        poses = poses[: a.limit]
    todo = [p for p in poses if not (out / "frames" / p["town"] / (p["id"] + ".npz")).exists()]
    print("poses", len(poses), "todo", len(todo), flush=True)
    log = open(out / f"render_{k}of{n}.jsonl", "a")
    srv = Server(a.idx, a.gpu, a.cpus, out / f"server_{k}of{n}.log")
    rig = None
    try:
        time.sleep(25)
        client = connect(srv.port)
        cams = [(CAM_XYZ[0], CAM_XYZ[1], CAM_XYZ[2], 0.0)]
        sizes, maps = build_maps([("front", *CAM_XYZ[:2], CAM_XYZ[2], 0.0)])
        print("render size", sizes, flush=True)
        rig = Rig(client, [sizes], cams, a.hero)
        t_start, n_done, n_render, town = time.time(), 0, 0, None
        for p in todo:
            if p["town"] != town:
                t = time.time()
                rig.load(p["town"])
                town = p["town"]
                print("loaded %s in %.1f s" % (town, time.time() - t), flush=True)
                log.write(json.dumps(dict(event="load", town=town, wall=time.time() - t)) + "\n")
            st = dict(id=p["id"], town=p["town"], profile=p["profile"], weather=p["weather"])
            try:
                fr = render_pose(rig, maps, p, st)
            except TimeoutError as e:
                st["error"] = str(e)
                log.write(json.dumps(st) + "\n")
                log.flush()
                raise
            dst = out / "frames" / p["town"] / (p["id"] + ".npz")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".tmp.npz")
            np.savez(tmp, frames=fr, frame_t=np.asarray(p["hist"]["t"], np.float32), x=p["hist"]["x"], y=p["hist"]["y"], z=p["hist"]["z"],
                     yaw=p["hist"]["yaw"], pitch=p["hist"]["pitch"])
            tmp.replace(dst)
            n_done += 1
            n_render += st["renders"]
            st["t"] = round(time.time() - t_start, 1)
            log.write(json.dumps(st) + "\n")
            log.flush()
            if n_done % 20 == 0 or n_done == len(todo):
                el = time.time() - t_start
                print("%d/%d poses  %.2f poses/s  %.1f ticks/s" % (n_done, len(todo), n_done / el, n_render / el), flush=True)
        if a.compare3:
            ref = poses[: a.compare3]
            rig.teardown()
            for p, d in compare3(rig, client, srv.port, ref, a):
                f1 = np.load(out / "frames" / p["town"] / (p["id"] + ".npz"))["frames"][-1]
                for mi, m in enumerate(("road", "wide")):
                    p1 = np.stack([pack(d[m])]) if False else pack(d[m])
                    print("compare3", p["id"], m, "mean |front-only - 3cam| (Y, 0..255):", float(np.abs(f1[mi][:4].astype(float) - p1[:4].astype(float)).mean()),
                          flush=True)
        (out / f"DONE_{k}of{n}").write_text("ok\n")
    except BaseException as e:  # noqa: BLE001
        (out / f"ERROR_{k}of{n}").write_text(repr(e) + "\n")
        raise
    finally:
        if rig is not None:
            try:
                rig.teardown()
            except Exception:  # noqa: BLE001
                pass
        srv.stop()


if __name__ == "__main__":
    main()
