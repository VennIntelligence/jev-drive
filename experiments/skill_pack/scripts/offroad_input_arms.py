"""Does the lag depend on how the 2 Hz history is turned into frames? (Q2, 'frame-rate interpolation of the inputs')

navsim2 env, CPU. Three cached native-Cinque arms on navhard (same model, same output adapter, different history synthesis:
GIMM-VFI on the full 10 Hz grid + context grid, GIMM on the 0.2 s context grid only, ego-motion warp): early heading response vs the
PDM reference and stage-2 DAC from their official per-token CSVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

ARMS = {"gimm (lb_navhard, the native arm)": (L.NATIVE_POSES, L.NATIVE_CSV),
        "gimm g0.2 (context grid only)": (L.D / "runs/op_interp/navhard/preds/gimm_g0.2-cinque__base.npz", "v2_navhard_two_stage_opi_navhard_gimm_g0.2-cinque__base"),
        "warp (ego-motion warp)": (L.D / "runs/op_interp/navhard/preds/warp-cinque__base.npz", "v2_navhard_two_stage_opi_navhard_warp-cinque__base")}
tab = pd.read_pickle(L.OUT / "token_table.pkl").set_index("token")
rows = []
for name, (p, csv) in ARMS.items():
    P = L.poses_by_token(p)
    c, _ = L.eval_csv(csv)
    c = c.set_index("token")
    toks = [t for t in tab.index if tab.loc[t, "stage"] == "two" and t in P]
    g = tab.loc[toks]
    psi1 = np.array([P[t][1, 2] for t in toks])      # pose 2 = 1.0 s
    y1 = np.array([P[t][1, 1] for t in toks])
    mv = (g.speed.to_numpy() > 3) & (np.abs(g.ref_yaw1.to_numpy()) > 0.05)
    ratio = psi1[mv] * np.sign(g.ref_yaw1.to_numpy()[mv]) / np.abs(g.ref_yaw1.to_numpy()[mv])
    dac = c.loc[toks, "drivable_area_compliance_stage_two"]
    rows.append(dict(arm=name, n=len(toks), dac_fail=float((dac == 0).mean()), psi1_over_ref_median=float(np.median(ratio)), n_turning=int(mv.sum()),
                     frac_lt_half=float((ratio < .5).mean()), abs_y1_median=float(np.median(np.abs(y1[g.speed.to_numpy() > 3])))))
out = pd.DataFrame(rows)
out.to_csv(REPO / "experiments/skill_pack/results/navhard-offroad/tables/q2_input_arms.csv", index=False)
print(out.round(3).to_string(index=False))
