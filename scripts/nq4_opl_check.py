#!/usr/bin/env python
"""Rule-8 equivalence check of the op_native_launch arms (night queue 4 OPL; todos/2026-09-26-night-queue-3.md, CL
section, [OPL] entries): the openpilot model sees and returns exactly what the offline path computes, with the TCP
launch partner in the loop.

The agent ("raw_dump": n) saves the first n requests of a route from the policy server's reset on: the raw road / wide
BGRA renders, the desire and speed it sent, and the plan / curvature / accel it got back. Here a fresh process builds the
same model (zeroshot_policy_server.OpenpilotModel: modeld's warp indices, BT.601 packing, OPModel, decode), starts a
fresh recurrent state and steps it through the dumped requests in order; every output must be bit-identical. Also checked
from the same dumps and the route's plans.jsonl: the requests are contiguous from the reset, consecutive requests are
exactly one model step apart (Cinque 1 tick, Lebowski 4 ticks), and every plan handed to P7 has road and wide from the
same frame, 4 ticks after the previous one (>= 99 %, the D3 A3 criterion).

    $DATA_DIR/envs/openpilot/bin/python scripts/nq4_opl_check.py cinque <attempt dir> ... --out check.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

STEP_TICKS = {"cinque": 1, "small": 1, "lebowski": 4}


def check_attempt(pol, model, adir):
    fs = sorted(Path(adir, "frames").glob("raw_*.npz"))
    idx = [int(f.stem[4:]) for f in fs]
    r = {"attempt": str(adir), "requests": len(fs), "contiguous": idx == list(range(len(fs)))}
    state = pol.new_state({})
    n_same, max_abs, frames = 0, 0.0, []
    try:
        for f in fs:
            d = np.load(f)
            frames.append(int(d["frame"]))
            meta = {"desire": int(d["desire"]), "speed": float(d["speed"])}
            info, out = pol.plan(state, meta, pol.prepare(meta, {"OP_ROAD": d["road"], "OP_WIDE": d["wide"]}))
            ok = all("out_" + k in d.files and out[k].dtype == d["out_" + k].dtype
                     and np.array_equal(out[k], d["out_" + k]) for k in out)
            ok = ok and all(float(info[k]) == float(d["info_" + k]) for k in ("curvature", "accel", "engaged")
                            if "info_" + k in d)
            n_same += bool(ok)
            max_abs = max(max_abs, float(np.abs(out["pos"].astype(np.float64) - d["out_pos"]).max()))
    finally:
        if hasattr(pol, "release"):
            pol.release(state)
    gaps = np.diff(frames)
    r.update(bit_identical=n_same, pos_max_abs=max_abs,
             request_step_ok=float(np.mean(gaps == STEP_TICKS[model])) if len(gaps) else None)
    plans = [json.loads(line) for line in open(Path(adir, "plans.jsonl"))]
    acc = [p for p in plans if p.get("accepted") is not None]
    same = [p["cam_frames"]["OP_ROAD"] == p["cam_frames"]["OP_WIDE"] for p in acc]
    pg = np.diff([p["frame"] for p in acc])
    r.update(plans=len(acc), road_wide_same=float(np.mean(same)) if same else None,
             plan_gap4=float(np.mean(pg == 4)) if len(pg) else None)
    r["pass"] = bool(r["contiguous"] and len(fs) > 0 and n_same == len(fs) and (r["request_step_ok"] or 0) >= 0.99
                     and (r["road_wide_same"] or 0) >= 0.99 and (r["plan_gap4"] or 0) >= 0.99)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(STEP_TICKS))
    p.add_argument("attempts", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="trt")
    a = p.parse_args()
    from zeroshot_policy_server import OpenpilotModel
    pol = OpenpilotModel(argparse.Namespace(model=a.model, backend=a.backend, pool=1))
    res = []
    for adir in a.attempts:
        res.append(check_attempt(pol, a.model, adir))
        print(json.dumps(res[-1]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if res and all(r["pass"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
