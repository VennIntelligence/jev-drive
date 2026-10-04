"""What would openpilot's longitudinal path do in the stuck phases? Reads the CPU replay (opctrl_replay.py, records acc_act = the action head's acceleration in
model units) of the stuck opctrl runs and of two base3 runs, and per run reports, over the steps where the logged car is standing (v < 0.24 = 0.3 / 1.25):
the share with raw action accel >= 0.1 (the should_stop exit), the median / max raw accel, and the emulated openpilot state and realised accel when the
logged speed is fed to lib/op_ctrl.py OpLongitudinal (open loop: the car's logged speed, not the emulated one).
    python opctrl_long_diag_acc.py <replay_dir>/op <replay_dir>/base [out.json]   (box, envs/hugsim python)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
import op_ctrl as O  # noqa: E402

out = {}
for d in sys.argv[1:3]:
    for f in sorted(Path(d).glob("*.json")):
        R = json.load(open(f))
        v, a = np.array(R["v"]), np.array(R["acc_act"])
        S = O.OpLongitudinal()
        st, ar = [], []
        for vi, ai in zip(v, a):
            x, l = O.hugsim_acc(S, float(ai), float(vi), 0.25)
            st.append(l["state"])
            ar.append(x)
        still = v < 0.24
        moved = np.where(v > 1.0)[0]
        k0 = int(moved[0]) if len(moved) else 0
        s = still & (np.arange(len(v)) >= k0)
        out[f"{Path(d).name}/{R['scenario']}"] = dict(
            n=len(v), n_still=int(s.sum()), frac_acc_ge_0p1=float((a[s] >= 0.1).mean()) if s.any() else None,
            acc_med=float(np.median(a[s])) if s.any() else None, acc_max=float(a[s].max()) if s.any() else None,
            plan_acc0_med=float(np.median(np.array(R["plan_acc0"])[s])) if s.any() else None,
            frac_pid_still=float(np.mean([x == "pid" for x, m in zip(st, s) if m])) if s.any() else None,
            max_emulated_a_sim_still=float(np.max(np.array(ar)[s])) if s.any() else None,
            acc_at_launch=a[:6].round(2).tolist())
json.dump(out, open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout, indent=1)
