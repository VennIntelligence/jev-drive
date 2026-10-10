#!/usr/bin/env python
"""BODY1 stage 2, P0-E: submit the input-port probe arms to the GPU pool (prereg section 4). One pool job of 6 CARLA workers per arm, the
exact submit of the design's cost probe ($DATA_DIR/runs/body1/s2p/submit_probe.py: P2H10-F-s0, spec_plan_smooth, zones off, seed
cllib.TURN_SEED, 6 junction routes) plus the agent's top-level "port" key; arms are chained so at most 6 workers run at once.

  python experiments/body1/scripts/s2p0_submit.py A0 A1        # A0 as shipped, A1 both fixes; A2 fix 1 only, A3 fix 2 only
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts")]
import cllib as C  # noqa: E402
from jevdrive.cl import pool as P  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

IDS = "10255 5423 28008 15102 28147 334".split()
ARMS = {"A0": dict(log=True), "A1": dict(ego=True, cmd=True, log=True), "A2": dict(ego=True, log=True), "A3": dict(cmd=True, log=True)}

if __name__ == "__main__":
    root, prev = data_dir() / "runs/body1/s2p0", []
    for arm in sys.argv[1:]:
        out = root / arm
        u = C.b2d_unit("b1s2p0e-" + arm, IDS, out / "arm", C.TURN_SEED, True, str(data_dir() / "runs/op_parity/hugsim/onnx/pp-P2H10-F-s0.onnx"),
                       arm="spec_plan_smooth", env={"PARITY_TAG": "P2H10-F-s0", "TOP_ARGS": '"port": %s' % json.dumps(ARMS[arm])})
        (out / "job").mkdir(parents=True, exist_ok=True)
        jid = P.submit(u["cmd"], name=u["name"], owner="body1", vram_gb=u["vram_gb"], carla=u["carla"], cpu=18, ram_gb=70, tries=1, timeout_h=0.9,
                       env=u["env"], log_dir=str(out / "job"), cwd=str(REPO), after=list(prev))
        (out / "job" / "pool_id").write_text(jid + "\n")
        print(arm, jid)
        prev = [jid]
