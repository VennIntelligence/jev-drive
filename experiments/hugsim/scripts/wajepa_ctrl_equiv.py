"""WA-JEPA's in-process controller fix (close_loop/run_fixed_controller.py traj2control_fixed) against our tree HUGSIM-zs/fixed
(official + patches/hugsim/optional/lqr-heading-fix.patch = upstream PR #57): same control on the same inputs?
Inputs: random / straight / jittering / zero plans and, if given, the plans of a finished run's plan_log.pkl with the ego state of its infos.pkl.
    $DATA_DIR/envs/hugsim/bin/python experiments/hugsim/scripts/wajepa_ctrl_equiv.py [run_dir ...]
"""
import importlib.util
import os
import pickle
import sys

import numpy as np

D = os.environ["DATA_DIR"]
sys.path.insert(0, f"{D}/third_party/HUGSIM-zs/fixed")
import sim.utils.sim_utils as su                      # noqa: E402  (our patched tree)
from sim.ilqr.lqr import plan2control                 # noqa: E402

ours = su.traj2control
src = open(f"{D}/third_party/wajepa/close_loop/run_fixed_controller.py").read()
ns = {"np": np, "plan2control": plan2control}
exec(src[src.index("def traj2control_fixed"):src.index("su.traj2control =")], ns)   # their function, verbatim
theirs = ns["traj2control_fixed"]

rng = np.random.default_rng(0)
cases = [("zero", np.zeros((8, 2))), ("straight", np.stack([np.zeros(8), np.arange(1, 9) * 2.0], 1))]
cases += [("jitter", np.stack([rng.normal(0, 0.2, 8), np.arange(1, 9) * 2.0], 1)) for _ in range(20)]
cases += [("random", np.cumsum(rng.normal(0, 2, (8, 2)), 0)) for _ in range(40)]
for d in sys.argv[1:]:
    log = pickle.load(open(f"{d}/plan_log.pkl", "rb"))
    cases += [(f"run{d[-12:]}:{r['step']}", np.asarray(r["traj_lidar"])) for r in log]
worst, n = 0.0, 0
for name, plan in cases:
    info = {"ego_velo": float(rng.uniform(0, 8)), "ego_steer": float(rng.uniform(-0.3, 0.3))}
    a, b = ours(plan.copy(), dict(info)), theirs(plan.copy(), dict(info))
    a, b = [np.asarray(x, float) for x in (a if isinstance(a, tuple) else (a,))], [np.asarray(x, float) for x in (b if isinstance(b, tuple) else (b,))]
    worst = max(worst, max(float(np.abs(x - y).max()) for x, y in zip(a, b)))
    n += 1
print(f"{n} plans, max |ours - theirs| over the controller outputs: {worst:.3e}")
