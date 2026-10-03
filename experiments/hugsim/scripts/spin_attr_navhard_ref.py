"""Early reference heading change per navhard token (plan: experiments/hugsim/plans/2026-10-04-spin-attribution-prereg.md, Q4).
navsim2 env, CPU.   python spin_attr_navhard_ref.py <out.csv> [procs]
Per token (metric cache of v2_navhard_two_stage): heading (deg, ego frame at t0, + left) of the PDM reference (`mc.trajectory`) and the
human future (`mc.human_trajectory`) at 1 s and 2 s, the lateral offset at 2 s, the reference speed at 0 s, and the driving command.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "skill_pack" / "scripts")]
import offroad_lib as L  # noqa: E402


def one(item):
    token, path = item
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    mc = L.load_cache(path)
    samp = TrajectorySampling(num_poses=40, interval_length=0.1)
    out = {"token": token}
    for name, tr in (("pdm", mc.trajectory), ("hum", mc.human_trajectory)):
        try:
            st = get_trajectory_as_array(tr, samp, mc.ego_state.time_point)
            e = L.to_ego(mc, st[:, :3])
            out[f"{name}_h1"] = float(np.degrees(e[10, 2]))
            out[f"{name}_h2"] = float(np.degrees(e[20, 2]))
            out[f"{name}_y2"] = float(e[20, 1])
            out[f"{name}_x2"] = float(e[20, 0])
        except Exception as ex:  # noqa: BLE001
            out[f"{name}_err"] = repr(ex)[:80]
    out["v0"] = float(np.hypot(mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.x, mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.y))
    return out


def main():
    import pandas as pd
    paths = L.cache_paths()
    items = list(paths.items())
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 32) as ex:
        rows = list(ex.map(one, items, chunksize=16))
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
    print(len(rows), "tokens ->", sys.argv[1])


if __name__ == "__main__":
    main()
