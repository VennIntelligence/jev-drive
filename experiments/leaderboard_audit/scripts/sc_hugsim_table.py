"""HUGSIM 64-scene exam split by source dataset: spins / HD of the shipped Cinque (native) and it_dw3 against camera height, plus the
sharpness and plan-speed ratio of sc_hugsim_probe.py (pass its json). Mac: .venv/bin/python experiments/leaderboard_audit/scripts/sc_hugsim_table.py probe.json"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[2]
I = E / "leaderboard_audit/results/loss_budget/hugsim_inputs"
H_EFF = {"nuscenes": 1.2, "pandaset": 1.5, "kitti360": 1.5, "waymo": 1.8}      # board_views.md: recorded minus cam_rect lowering
rng = np.random.default_rng(0)
nat = pd.read_csv(I / "derot_runs.csv").query("arm == 'base'").set_index("scenario")
dw3 = pd.read_csv(I / "it_dw3-s0_all64.csv").set_index("scenario")
dw3["spin"] = pd.read_csv(E / "op_adapt_h/results/one_driver/hugsim64/it_dw3-s0_all64/spins.csv").set_index("scenario").spin.reindex(dw3.index)
pr = pd.DataFrame(json.load(open(sys.argv[1]))).set_index("scenario")
assert len(nat) == len(dw3) == 64, (len(nat), len(dw3))


def ci(x, B=5000):
    x = np.asarray(x, float)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return f"{x.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"


rows = []
for ds in ("nuscenes", "pandaset", "kitti360", "waymo"):
    s = nat.index[nat.dataset == ds]
    p = pr.loc[pr.dataset == ds]
    sr = p.speed_ratio.dropna()
    rows.append(dict(dataset=ds, h_eff=H_EFF[ds], n=len(s), nat_spins=int(nat.loc[s, "spin"].sum()), nat_HD=ci(nat.loc[s, "hdscore"]), nat_RC=ci(nat.loc[s, "rc"]),
                     dw3_spins=int(dw3.loc[s, "spin"].sum()), dw3_HD=ci(dw3.loc[s, "hdscore"]), pred_scale=round(1.22 / H_EFF[ds], 2),
                     speed_ratio=f"{sr.median():.2f} (n={len(sr)} scenes)", lap_native=round(p.lap_native.median()), lap_road=round(p.lap_road.median()), hf_native=round(float(p.hf_native.median()), 4), hf_road=round(float(p.hf_road.median()), 4),
                     road_over_native=round(float((p.lap_road / p.lap_native).median()), 2), focal=round(float(p.f.iloc[0])), size="x".join(map(str, p["size"].iloc[0]))))
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(E / "leaderboard_audit/results/scale_check_hugsim.csv", index=False)
