"""CARLA counterfactual route pairs, re-render at the open-loop-aligned rig (2026-10-05). Same poses / history as carla_pairs_render.py
(plan = the same poses.pkl), different camera:

  * Mount = interface.B2D_SPEC_MOUNT (x 1.59 m ahead of the rear axle, y 0, z 1.86 m above the road, level): the B2D `spec` camera, which is the
    viewpoint of the NAVSIM / WOD open-loop boards. No per-board camera.
  * The model frames are built exactly as the B2D closed-loop harness builds them (scripts/zeroshot_rigs.openpilot_sensor_specs + the
    OpenpilotModel.pack of scripts/zeroshot_policy_server.py): TWO separate pinhole sensors at the same mount, 1928 x 1208, road f 2648 px
    (40.0 deg) and wide f 567 px (118.9 deg, one pinhole, not a three-camera stitch), each warped to the 512 x 256 model frame by modeld's
    nearest-neighbour warp (opf._nn_index, rpy 0), BT.601 limited-range YUV (Y 16..235), 2 x 2 chroma means, packed to (6, 128, 256).
    (The 1.22 m set used one 1260 x 750 pinhole cut twice with bilinear sampling and full-range YCbCr: superseded.)
  * Extra: a wide raw capture of the t0 pose, three level pinhole cameras at yaw -60 / 0 / +60 deg, 960 x 540, f 480 px (90 deg each, ~210 deg
    in total, 30 deg overlap), same mount, JPEG q90 in <out>/pano/<town>/<pose id>.npz (jpeg_l, jpeg_c, jpeg_r as uint8 byte arrays, BGR). A wider
    model input can be cut from them later (rays from the f / yaw above) without re-rendering. The pano sensors only render for the t0 grab.

  $DATA_DIR/envs/carla/bin/python carla_pairs_render_ol.py --plan <poses.pkl> --out <dir> --gpu 0 --idx 160 --cpus 0-3 [--shard 0/6] [--limit N] [--hero]
  -> <dir>/frames/<town>/<id>.npz (frames (10, 2, 6, 128, 256) uint8, same layout as before), <dir>/pano/<town>/<id>.npz, <dir>/render_<shard>.jsonl, DONE_/ERROR_
"""
import argparse, json, math, os, pickle, queue, sys, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
import carla_pairs_render as R0  # noqa: E402  (Server, connect, LARGE, small, SETTLE_*: the unchanged parts)
from jevdrive.openpilot import interface as IF  # noqa: E402

MOUNT = IF.B2D_SPEC_MOUNT                                   # (x ahead of the rear axle, y left, z above the road), m
SENSOR_WH = (1928, 1208)                                    # comma 3X native, scripts/zeroshot_rigs.OP_CAMERA_WH
FOCAL = {"road": 2648.0, "wide": 567.0}                     # scripts/zeroshot_rigs.OP_FOCAL
PANO_YAW, PANO_WH, PANO_F, PANO_Q = (60.0, 0.0, -60.0), (960, 540), 480.0, 90   # left, centre, right (yaw left +); 90 deg HFOV each
MODEL_W, MODEL_H, MEDMODEL_CY = 512, 256, 47.6              # jevdrive/openpilot/frames.py
VIEW_FROM_DEVICE = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
MODEL_K = {"road": np.array([[910.0, 0, MODEL_W / 2], [0, 910.0, MEDMODEL_CY], [0, 0, 1]]),
           "wide": np.array([[455.0, 0, MODEL_W / 2], [0, 455.0, 0.5 * (256 + MEDMODEL_CY)], [0, 0, 1]])}


def nn_index(M, dst_wh, src_wh):
    """frames._nn_index verbatim."""
    w, h = dst_wh
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    M = M.astype(np.float32)
    sx, sy, sw = (M[i, 0] * x + M[i, 1] * y + M[i, 2] for i in range(3))
    xi = np.clip(np.rint(sx / sw), 0, src_wh[0] - 1).astype(np.int64)
    yi = np.clip(np.rint(sy / sw), 0, src_wh[1] - 1).astype(np.int64)
    return (yi * src_wh[0] + xi).ravel()


def build_idx():
    """zeroshot_policy_server.OpenpilotModel.__init__: name -> (luma gather index, 2 x 2 chroma block indices); rpy 0."""
    w, h = SENSOR_WH
    out = {}
    for name, f in FOCAL.items():
        cam_K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]])
        M = cam_K @ VIEW_FROM_DEVICE @ np.linalg.inv(MODEL_K[name] @ VIEW_FROM_DEVICE)      # frames.get_warp_matrix(zeros)
        y = nn_index(M, (MODEL_W, MODEL_H), (w, h))
        uv = nn_index(M * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], np.float32), (MODEL_W // 2, MODEL_H // 2), (w // 2, h // 2))
        r, c = np.divmod(uv, w // 2)
        out[name] = (y, np.stack([(2 * r + i) * w + 2 * c + j for i in (0, 1) for j in (0, 1)]))
    return out


def pack(bgra, idx):
    """OpenpilotModel.pack verbatim: CARLA BGRA -> (6, 128, 256), BT.601 limited-range YUV at the warp indices."""
    y_idx, quad = idx
    px = bgra.reshape(-1, 4)
    b, g, r = (px[y_idx, k].astype(np.float32) for k in range(3))
    Y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).reshape(MODEL_H, MODEL_W)
    q = px[quad].astype(np.float32).mean(0)
    b, g, r = q[:, 0], q[:, 1], q[:, 2]
    U = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    V = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    Y, U, V = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (Y, U, V))
    out = np.empty((6, MODEL_H // 2, MODEL_W // 2), np.uint8)
    out[0], out[1], out[2], out[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
    out[4] = U.reshape(MODEL_H // 2, MODEL_W // 2)
    out[5] = V.reshape(MODEL_H // 2, MODEL_W // 2)
    return out


# ---------------------------------------------------------------- world / cameras
class Rig:
    """Sensors: [road, wide] (always listening) + 3 pano cameras (listening only for the t0 grab). All at MOUNT, rear-axle frame."""

    def __init__(self, client, hero):
        import carla
        self.carla, self.client, self.use_hero = carla, client, hero
        self.world, self.town, self.sensors, self.q, self.hero, self.on = None, None, [], None, None, set()
        x, y, z = MOUNT
        self.cams = [(x, y, z, 0.0)] * 2 + [(x, y, z, yw) for yw in PANO_YAW]              # (x fwd, y left, z, yaw left +)
        self.specs = [(SENSOR_WH, FOCAL["road"]), (SENSOR_WH, FOCAL["wide"])] + [(PANO_WH, PANO_F)] * 3

    def load(self, town, at):
        carla = self.carla
        self.teardown()
        self.world = self.client.load_world(town)
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds, s.no_rendering_mode = True, 0.05, False
        if town in R0.LARGE:
            s.tile_stream_distance, s.actor_active_distance = 650.0, 650.0
        self.world.apply_settings(s)
        self.town = town
        self.q = queue.Queue()
        self.sensors, self.hero, self.on = [], None, set()
        if self.use_hero:
            vb = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
            vb.set_attribute("role_name", "hero")
            for dx, dz in ((0, 0.6), (0, 1.5), (0, 3.0), (6, 0.6), (-6, 0.6), (12, 1.5), (-12, 1.5)):
                self.hero = self.world.try_spawn_actor(vb, carla.Transform(carla.Location(at[0] + dx, at[1], at[2] + dz)))
                if self.hero is not None:
                    break
            if self.hero is None:
                raise RuntimeError("hero spawn failed at %s" % (at,))
            self.hero.set_simulate_physics(False)
            self.world.tick()
        for i, ((W, H), f) in enumerate(self.specs):
            b = self.world.get_blueprint_library().find("sensor.camera.rgb")
            b.set_attribute("image_size_x", str(W))
            b.set_attribute("image_size_y", str(H))
            b.set_attribute("fov", str(math.degrees(2 * math.atan(W / 2 / f))))
            for k, v in (("exposure_speed_up", "100.0"), ("exposure_speed_down", "100.0")):
                if b.has_attribute(k):
                    b.set_attribute(k, v)
            cam = self.world.spawn_actor(b, carla.Transform(carla.Location(at[0], at[1], at[2] + 2.0)))
            self.sensors.append(cam)
        self.listen(range(2))

    def listen(self, which):
        which = set(which)
        for i, s in enumerate(self.sensors):
            if i in which and i not in self.on:
                s.listen(lambda im, i=i: self.q.put((i, im.frame, im)))
            elif i not in which and i in self.on:
                s.stop()
        self.on = which
        self.active = sorted(which)
        while not self.q.empty():
            self.q.get_nowait()

    def teardown(self):
        for i, s in enumerate(self.sensors):
            try:
                if i in self.on:
                    s.stop()
                s.destroy()
            except RuntimeError:
                pass
        if self.hero is not None:
            try:
                self.hero.destroy()
            except RuntimeError:
                pass
        self.sensors, self.hero, self.on = [], None, set()

    def set_pose(self, x, y, z, yaw, pitch):
        """Rig pose = rear axle on the road (CARLA coordinates, yaw deg clockwise)."""
        carla = self.carla
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for sensor, (cx, cy, cz, cyaw) in zip(self.sensors, self.cams):
            lx, ly = cx, -cy                            # the pose is the rear axle; CARLA y is right
            loc = carla.Location(x + c * lx - s * ly, y + s * lx + c * ly, z + cz)
            sensor.set_transform(carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw - cyaw, roll=0.0)))
        if self.hero is not None:
            self.hero.set_transform(carla.Transform(carla.Location(x - 60 * c, y - 60 * s, z - 2.0), carla.Rotation(yaw=yaw)))

    def grab(self, timeout=30.0):
        f = self.world.tick()
        got, deadline = {}, time.time() + timeout
        while len(got) < len(self.active):
            try:
                i, fr, im = self.q.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                raise TimeoutError("camera frame %d missing" % f)
            if fr == f:
                got[i] = np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4)
        return got

    def weather(self, name):
        self.world.set_weather(getattr(self.carla.WeatherParameters, name))


def render_pose(rig, idx, pose, stats):
    h = pose["hist"]
    n = len(h["x"])
    stopped = pose["profile"] == "stopped"
    order = [n - 1] if stopped else list(range(n))
    rig.weather(pose["weather"])
    t0 = time.perf_counter()
    rig.set_pose(h["x"][order[0]], h["y"][order[0]], h["z"][order[0]], h["yaw"][order[0]], h["pitch"][order[0]])
    prev, ticks = None, 0
    while True:                                   # settle: teleport, tile streaming, exposure
        got = rig.grab()
        ticks += 1
        cur = R0.small(got[1])
        if prev is not None and ticks >= R0.SETTLE_MIN and np.abs(cur - prev).mean() < R0.SETTLE_EPS:
            break
        prev = cur
        if ticks >= R0.SETTLE_MAX:
            stats["unsettled"] = True
            break
    stats["settle_ticks"] = ticks
    t_settle = time.perf_counter()
    frames = []
    for j, k in enumerate(order):
        if j:
            rig.set_pose(h["x"][k], h["y"][k], h["z"][k], h["yaw"][k], h["pitch"][k])
            got = rig.grab()
        frames.append(np.stack([pack(got[0], idx["road"]), pack(got[1], idx["wide"])]))
    if stopped:
        frames = frames * n
    t1 = time.perf_counter()
    rig.listen(range(5))                          # pano at t0: the pose is unmoved; exposure of the fresh sensors needs a few ticks
    for _ in range(3):
        got = rig.grab()
    pano = [cv2.imencode(".jpg", got[2 + i][:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, PANO_Q])[1] for i in range(3)]
    rig.listen(range(2))
    t2 = time.perf_counter()
    stats.update(renders=ticks + len(order) - 1 + 3, wall_settle=t_settle - t0, wall_frames=t1 - t_settle, wall_pano=t2 - t1,
                 mean_y=float(frames[-1][0][:4].mean()), pano_kb=round(sum(len(p) for p in pano) / 1024, 1))
    return np.stack(frames), pano


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
    ap.add_argument("--ids", default="", help="comma list of pose ids (test)")
    ap.add_argument("--hero", action="store_true")
    ap.add_argument("--towns", default="")
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("frames", "pano"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    k, n = map(int, a.shard.split("/"))
    poses = pickle.load(open(a.plan, "rb"))
    if a.towns:
        poses = [p for p in poses if p["town"] in a.towns.split(",")]
    cost = np.cumsum([1.0 if q["town"] in R0.LARGE else 0.4 for q in poses])
    lo, hi = np.searchsorted(cost, cost[-1] * k / n, "left"), np.searchsorted(cost, cost[-1] * (k + 1) / n, "left")
    poses = poses[lo if k else 0:hi if k + 1 < n else len(poses)]
    if a.ids:
        poses = [p for p in poses if p["id"] in a.ids.split(",")]
    if a.limit:
        poses = poses[: a.limit]
    todo = [p for p in poses if not (out / "frames" / p["town"] / (p["id"] + ".npz")).exists()]
    print("poses", len(poses), "todo", len(todo), "mount", MOUNT, flush=True)
    log = open(out / f"render_{k}of{n}.jsonl", "a")
    idx = build_idx()
    (out / "rig.json").write_text(json.dumps(dict(mount=MOUNT, sensors={m: dict(wh=SENSOR_WH, f=FOCAL[m]) for m in FOCAL},
                                                  pano=dict(yaw_deg_left_positive=PANO_YAW, wh=PANO_WH, f=PANO_F, jpeg_q=PANO_Q, order="left centre right"),
                                                  model="frames.py nearest warp, BT.601 limited-range YUV, as OpenpilotModel.pack")))
    state = dict(srv=None, rig=None, client=None, restarts=0)

    def start():
        stop()
        state["srv"] = R0.Server(a.idx, a.gpu, a.cpus, out / f"server_{k}of{n}.log")
        time.sleep(25)
        state["client"] = R0.connect(state["srv"].port)
        state["rig"] = Rig(state["client"], a.hero)

    def stop():                                   # the server dies with its actors: no per-actor destroy (120 s time-out each on a dead server)
        if state["srv"] is not None:
            state["srv"].stop()
        state["srv"] = state["rig"] = None

    try:
        start()
        t_start, n_done, n_render, town, tries, i = time.time(), 0, 0, None, 0, 0
        while i < len(todo):
            p = todo[i]
            try:
                if p["town"] != town:
                    t = time.time()
                    state["rig"].load(p["town"], (p["hist"]["x"][0], p["hist"]["y"][0], p["hist"]["z"][0]))
                    town = p["town"]
                    print("loaded %s in %.1f s" % (town, time.time() - t), flush=True)
                    log.write(json.dumps(dict(event="load", town=town, wall=time.time() - t)) + "\n")
                st = dict(id=p["id"], town=p["town"], profile=p["profile"], weather=p["weather"])
                fr, pano = render_pose(state["rig"], idx, p, st)
            except (RuntimeError, TimeoutError) as e:
                tries += 1
                state["restarts"] += 1
                log.write(json.dumps(dict(event="restart", id=p["id"], error=str(e)[:200], tries=tries)) + "\n")
                log.flush()
                print("RESTART (%d) at %s: %s" % (state["restarts"], p["id"], str(e)[:120]), flush=True)
                if state["restarts"] > 12:
                    raise
                if tries > 3:
                    log.write(json.dumps(dict(event="skip", id=p["id"])) + "\n")
                    i, tries = i + 1, 0
                start()
                town = None
                continue
            tries = 0
            i += 1
            for sub, payload in (("pano", dict(jpeg_l=pano[0], jpeg_c=pano[1], jpeg_r=pano[2])),
                                 ("frames", dict(frames=fr, frame_t=np.asarray(p["hist"]["t"], np.float32), x=p["hist"]["x"], y=p["hist"]["y"],
                                                 z=p["hist"]["z"], yaw=p["hist"]["yaw"], pitch=p["hist"]["pitch"]))):
                dst = out / sub / p["town"] / (p["id"] + ".npz")
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.stem + ".tmp.npz")
                np.savez(tmp, **payload)
                tmp.replace(dst)                  # frames last: its existence marks the pose done
            n_done += 1
            n_render += st["renders"]
            st["t"] = round(time.time() - t_start, 1)
            log.write(json.dumps(st) + "\n")
            log.flush()
            if n_done % 20 == 0 or n_done == len(todo):
                el = time.time() - t_start
                print("%d/%d poses  %.2f poses/s  %.1f ticks/s" % (n_done, len(todo), n_done / el, n_render / el), flush=True)
        (out / f"DONE_{k}of{n}").write_text("ok\n")
    except BaseException as e:  # noqa: BLE001
        (out / f"ERROR_{k}of{n}").write_text(repr(e) + "\n")
        raise
    finally:
        stop()


if __name__ == "__main__":
    main()
