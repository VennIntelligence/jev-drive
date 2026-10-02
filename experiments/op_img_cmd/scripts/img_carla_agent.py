"""Recording agent for the CARLA half of the image-command test: scripts/p4_carla_agent.py (privileged BehaviorAgent,
Waymo-calibrated 3-camera rig at 5 Hz, pose truth at 20 Hz) unchanged, plus a stop rule so a route ends shortly after
the ego has crossed the junction of its traversal: stop once the ego has been within PASS_M of the junction exit for
AFTER_EXIT_S and AFTER_ENTRY_S has passed since it was at the entry (the 4 s future of the last t0 is then logged).
Entry / exit points per route: plan/stops.json (img_carla_prep.py), path in the agent config key "stops".

  b2d_run.py ... --agent experiments/op_img_cmd/scripts/img_carla_agent.py --agent-config <cfg.json>  (B2D_SENSOR_TICK=1)
"""
import json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import p4_carla_agent as p4  # noqa: E402
from srunner.scenariomanager.timer import GameTime  # noqa: E402

PASS_M, AFTER_EXIT_S, AFTER_ENTRY_S = 3.5, 2.0, 5.5


def get_entry_point():
    return "ImgCarlaAgent"


class ImgCarlaAgent(p4.P4Agent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        rid = os.environ["BENCHMARK_ROUTE_ID"]
        st = json.loads(Path(self.cfg["stops"]).read_text())[rid]
        self._pts = {k: st[k] for k in ("entry", "exit")}
        self._seen = {}
        (self.out / "stop_points.json").write_text(json.dumps(st))

    def __call__(self):
        control = super().__call__()
        loc, t = self._hero.get_location(), GameTime.get_time()
        for k, (x, y) in self._pts.items():
            if k not in self._seen and math.hypot(loc.x - x, loc.y - y) < PASS_M:
                self._seen[k] = t
        if "exit" in self._seen and t - self._seen["exit"] >= AFTER_EXIT_S and t - self._seen.get("entry", self._seen["exit"]) >= AFTER_ENTRY_S:
            p4.STOP.update(flag=True, why="junction_done")
        return control

    def destroy(self):
        if hasattr(self, "_seen"):
            (self.out / "junction_times.json").write_text(json.dumps(self._seen))
        super().destroy()
