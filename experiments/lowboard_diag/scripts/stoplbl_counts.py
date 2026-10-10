"""stoplbl: counts of situation frames on a sample of navtrain logs (labels from stoplbl_label.py), cluster bootstrap over logs.
  .venv/bin/python experiments/lowboard_diag/scripts/stoplbl_counts.py tmp/stoplbl/navtrain_s300.parquet experiments/lowboard_diag/results/stoplbl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.data import splits  # noqa: E402

L = pd.read_parquet(sys.argv[1])
OUT = Path(sys.argv[2])
sp = splits.load("navsim/navtrain")
L = L[sp.mask(L.token)]
L = L[L.full | (L.s_end >= 50)].reset_index(drop=True)
N_ALL = 103288
rng = np.random.default_rng(0)
logs = L.log.values
ul = np.unique(logs)
idx = {u: np.where(logs == u)[0] for u in ul}
S = {}
S["red50"] = L.tl_red_d <= 50
S["red50_driver_stopped(vmin<1)"] = S["red50"] & (L.tl_red_vmin < 1.0)
S["red50_driver_decel(v_line<0.6 v0, not stopped)"] = S["red50"] & (L.tl_red_vmin >= 1.0) & (L.tl_red_vline < 0.6 * L.v0)
S["red50_driver_passed(vmin>=3)"] = S["red50"] & (L.tl_red_vmin >= 3.0)
S["tl_line50_any_state"] = L.tl_line_d <= 50
S["tl_line50_green_or_other"] = (L.tl_line_d <= 50) & ~S["red50"]
S["stop_sign50"] = L.ss_d <= 50
S["stop_sign50_driver_stopped(vmin<1)"] = S["stop_sign50"] & (L.ss_vmin < 1.0)
S["stop_sign50_driver_decel"] = S["stop_sign50"] & (L.ss_vmin >= 1.0) & (L.ss_vline < 0.6 * L.v0)
S["yield_turnstop50"] = (L.ts_d <= 50) | (L.yl_d <= 50)
S["turnstop50_driver_stopped(vmin<1)"] = (L.ts_d <= 50) & (L.ts_vmin < 1.0)
S["ped_crosswalk_with_ped50"] = L.cwped_d <= 50
S["ped_cw50_driver_stopped(vmin<1)"] = S["ped_crosswalk_with_ped50"] & (L.cwped_vmin < 1.0)
S["any_stop_target50 (red|sign|yield|ped)"] = S["red50"] | S["stop_sign50"] | S["yield_turnstop50"] | S["ped_crosswalk_with_ped50"]
S["turn>=45deg within 50 m"] = (L.turn_deg >= 45) & (L.turn_s <= 50)
S["turn>=45deg within 30 m"] = (L.turn_deg >= 45) & (L.turn_s <= 30)
S["turn>=45deg within 50 m, v0>3"] = S["turn>=45deg within 50 m"] & (L.v0 > 3)
rows = []
for k, m in S.items():
    m = m.fillna(False).values
    b = []
    for _ in range(500):
        pick = np.concatenate([idx[u] for u in rng.choice(ul, len(ul))])
        b.append(m[pick].mean())
    lo, hi = np.percentile(b, [2.5, 97.5])
    rows.append(dict(situation=k, frames_sampled=int(m.sum()), frac=m.mean(), lo=lo, hi=hi, extrap_navtrain=round(m.mean() * N_ALL), extrap_lo=round(lo * N_ALL), extrap_hi=round(hi * N_ALL)))
df = pd.DataFrame(rows)
df.to_csv(OUT / "navtrain_counts.csv", index=False)
print(f"sample: {len(ul)} logs, {len(L)} navtrain tokens with a complete 50 m route of {N_ALL}")
print(df.round(4).to_string())
# turn entry speed distribution of the log driver vs the curvature-limited speed (a_lat 2 m/s^2)
T = L[S["turn>=45deg within 50 m"].fillna(False)]
q = lambda x: x.quantile([.1, .25, .5, .75, .9]).round(2).tolist()
t = pd.DataFrame({"turn_vin": q(T.turn_vin), "turn_vc(a_lat=2)": q(T.turn_vc), "v0": q(T.v0), "ratio vin/vc": q(T.turn_vin / T.turn_vc)}, index=["p10", "p25", "p50", "p75", "p90"])
t.to_csv(OUT / "navtrain_turn_entry_speed.csv")
print(t)
print("turn entry speed bins (m/s):", pd.cut(T.turn_vin, [-1, 2, 4, 6, 8, 10, 40]).value_counts(normalize=True).sort_index().round(3).to_dict())
print("share with vin > vc:", float((T.turn_vin > T.turn_vc).mean()), " vin > 1.5 vc:", float((T.turn_vin > 1.5 * T.turn_vc).mean()))
# computability of the stop-distance target: frames whose route is complete (>=50 m) / all navtrain tokens in the sampled logs
