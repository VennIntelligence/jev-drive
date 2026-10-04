"""Record a chase camera alongside the unchanged op-drive control loop (Python 3.8)."""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("lib",)]  # the stale experiments/op_closed_loop/lib copy was removed 2026-10-05 (3fc2eece:experiments/op_closed_loop/lib/op_arb_agent.py)
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

import carla

from op_arb_agent import OpArbAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def get_entry_point():
    return "RecordedOpDrive"


class RecordedOpDrive(OpArbAgent):
    def setup(self, config):
        super().setup(config)
        self.chase = None
        self.video_error = None
        self.video_dropped = 0
        self.video_queue = queue.Queue(maxsize=64)
        self.video_thread = None
        self.video_proc = None
        self.video_frames = 0
        self.record_dir = Path(os.environ["B2D_ATTEMPT_OUT"])
        self.scene_log = (self.record_dir / "scene.jsonl").open("w")
        self.last_scene_frame = -1

    def _start_camera(self):
        hero = CarlaDataProvider.get_hero_actor()
        world = CarlaDataProvider.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for key, value in {"image_size_x": "1280", "image_size_y": "720", "fov": "90",
                           "sensor_tick": "0.1"}.items():
            bp.set_attribute(key, value)
        transform = carla.Transform(carla.Location(x=-8.0, z=4.5), carla.Rotation(pitch=-20.0))
        self.chase = world.spawn_actor(bp, transform, attach_to=hero)
        encoder = Path(__file__).parent / "op_drive_record_video.py"
        python = Path(os.environ["DATA_DIR"]) / "envs/openpilot/bin/python"
        self.video_proc = subprocess.Popen(
            [str(python), str(encoder), "encode", str(self.record_dir / "chase_raw.mp4")],
            stdin=subprocess.PIPE)
        self.video_thread = threading.Thread(target=self._encode, daemon=True)
        self.video_thread.start()
        self.chase.listen(self._image)

    def _image(self, image):
        try:
            self.video_queue.put_nowait((image.frame, image.timestamp, bytes(image.raw_data)))
        except queue.Full:
            self.video_dropped += 1

    def _encode(self):
        try:
            with (self.record_dir / "video_frames.jsonl").open("w") as log:
                while True:
                    item = self.video_queue.get()
                    if item is None:
                        break
                    frame, timestamp, raw = item
                    self.video_proc.stdin.write(raw)
                    log.write(json.dumps({"frame": frame, "timestamp": timestamp}) + "\n")
                    log.flush()
                    self.video_frames += 1
        except Exception as exc:
            self.video_error = repr(exc)

    def __call__(self):
        control = super().__call__()
        if self.chase is None:
            self._start_camera()
        frame = int(GameTime.get_frame())
        if frame - self.last_scene_frame >= 2:
            hero = CarlaDataProvider.get_hero_actor()
            origin = hero.get_location()
            actors = []
            for actor in CarlaDataProvider.get_world().get_actors():
                if not actor.type_id.startswith(("vehicle.", "walker.", "static.prop.", "traffic.traffic_light")):
                    continue
                tf = actor.get_transform()
                if tf.location.distance(origin) > 80:
                    continue
                velocity = actor.get_velocity()
                box = getattr(actor, "bounding_box", carla.BoundingBox())
                row = {"id": actor.id, "type": actor.type_id,
                       "location": [tf.location.x, tf.location.y, tf.location.z],
                       "rotation": [tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw],
                       "velocity": [velocity.x, velocity.y, velocity.z],
                       "extent": [box.extent.x, box.extent.y, box.extent.z],
                       "box_location": [box.location.x, box.location.y, box.location.z],
                       "box_rotation": [box.rotation.roll, box.rotation.pitch, box.rotation.yaw]}
                if actor.type_id.startswith("traffic.traffic_light"):
                    row["light"] = str(actor.get_state())
                actors.append(row)
            self.scene_log.write(json.dumps({"frame": frame, "t": float(GameTime.get_time()),
                                            "hero_id": hero.id, "actors": actors}) + "\n")
            self.last_scene_frame = frame
        if self.video_error:
            raise RuntimeError("Chase recording failed: " + self.video_error)
        return control

    def destroy(self):
        try:
            if getattr(self, "chase", None) is not None:
                self.chase.stop()
                self.chase.destroy()
                self.chase = None
            if getattr(self, "video_thread", None) is not None:
                if self.video_thread.is_alive():
                    self.video_queue.put(None, timeout=30)
                self.video_thread.join(timeout=60)
                if self.video_thread.is_alive():
                    self.video_error = "Encoder thread did not finish"
            if getattr(self, "video_proc", None) is not None:
                self.video_proc.stdin.close()
                if self.video_proc.wait(timeout=60) != 0:
                    self.video_error = "Video encoder exited unsuccessfully"
            if getattr(self, "scene_log", None) is not None:
                self.scene_log.close()
            if hasattr(self, "record_dir"):
                (self.record_dir / "video_summary.json").write_text(json.dumps(
                    {"frames": self.video_frames, "dropped": self.video_dropped,
                     "error": self.video_error}, indent=2))
        finally:
            super().destroy()
