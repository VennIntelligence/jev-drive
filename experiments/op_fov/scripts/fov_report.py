#!/usr/bin/env python
"""Metrics of the op_fov replays (fov_replay.py run): per window x arm, then paired arm - N with cluster bootstrap over segments.

  python fov_report.py metrics <npz dir> <out.csv>        (box or Mac; numpy only)
  python fov_report.py table <metrics.csv> <split> <out stem>   (Mac: jevdrive.stats tables + decision lines)

Definitions are the pre-registration's (../plans/2026-10-05-fov-prereg.md). Sign: openpilot curvature / yaw, + = right.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
X_IDXS = 192.0 * (np.arange(33) / 32.0) ** 2
LAT_T = 0.275
ONSET_FRAC = 0.3          # action / measured curvature reaches 0.3 x the measured peak


def curvature_from_plan(yaws, yaw_rates, v_ego, action_t):
    """jevdrive.openpilot.model.curvature_from_plan (copied: that module loads onnxruntime at import)."""
    psi = np.interp(action_t, T_IDXS, yaws) if action_t >= 0.3 else action_t / 0.3 * np.interp(0.3, T_IDXS, yaws)
    v = max(v_ego, 1.0)
    return 2 * psi / (v * action_t) - yaw_rates[0] / v


def calib_future(z, i, ts):
    """Logged positions at t_i + ts in the calibrated frame of frame i (x fwd, y right, z down); NaN past the segment end."""
    from scipy.spatial.transform import Rotation
    t, pos = z["t"], z["pos"]
    tq = t[i] + np.asarray(ts)
    p = np.stack([np.interp(tq, t, pos[:, k]) for k in range(3)], 1) - pos[i]
    cfd = Rotation.from_euler("xyz", z["rpy_calib"]).as_matrix().T
    out = (cfd @ z["R"][i].T @ p.T).T
    out[tq > t[-1]] = np.nan
    return out


def window_metrics(f):
    z = np.load(f, allow_pickle=True)
    ev = json.loads(str(z["ev"]))
    arms = sorted({k.split("_")[0] for k in z.files if k.endswith("_act")})
    lo = int(z["lo"])
    t, v, yr = z["t"], z["v"], z["yr"]
    yr_s = np.convolve(yr, np.ones(5) / 5, mode="same")
    k_all = yr_s / np.maximum(v, 1.0)
    psi = np.r_[0, np.cumsum(0.5 * (yr_s[1:] + yr_s[:-1]) * np.diff(t))]
    n = len(z[f"{arms[0]}_act"])
    gi = lo + np.arange(n)                                   # segment frame index of each replayed frame
    scored = np.arange(n) >= 120
    vv = v[gi]
    k_meas = np.interp(t[gi] + LAT_T, t, k_all, right=np.nan)
    k_now = k_all[gi]
    dpsi2 = np.interp(t[gi] + 2.0, t, psi, right=np.nan) - psi[gi]
    rows = []
    if ev["kind"] == "turn":
        sel = scored & (gi >= ev["onset"] - 20) & (gi <= ev["end"] + 20) & (vv > 2) & np.isfinite(k_meas)
        sgn = 1.0 if ev["dir"] == "right" else -1.0
        peak = np.nanmax(sgn * k_now[scored])
        tt = t[gi] - t[ev["onset"]]
        on = scored & (sgn * k_now >= ONSET_FRAC * peak)
        tau_meas = tt[on][0] if on.any() else np.nan
        fut = np.array([calib_future(z, i, [2.0, 3.0]) for i in gi[sel]])          # (m, 2, 3)
    else:
        sel = scored & (vv > 5)
        fut5 = np.array([calib_future(z, i, T_IDXS[T_IDXS <= 5.0][1:]) for i in gi[sel]])
    for a in arms:
        act, plan, ll, lp = z[f"{a}_act"], z[f"{a}_plan"], z[f"{a}_ll"], z[f"{a}_lp"]
        k_act = act[:, 0] / np.maximum(1.0, vv) ** 2
        vplan = plan[:, 0, 3]
        r = dict(win=f.stem, seg=ev["seg"], kind=ev["kind"], arm=a, miss_wide=float(z[f"{a}_miss"][1]),
                 v_ratio=float(np.median(vplan[sel] / vv[sel])) if sel.any() else np.nan)
        lw = ll[:, 2, :, 0] - ll[:, 1, :, 0]
        lw10 = np.array([np.interp(10.0, X_IDXS, w) for w in lw])
        okl = sel & (lp[:, 1] > 0.5) & (lp[:, 2] > 0.5)
        r["lane_w10"] = float(np.median(lw10[okl])) if okl.sum() >= 10 else np.nan
        if ev["kind"] == "turn":
            km, ka = k_meas[sel], k_act[sel]
            r["A_act"] = float((ka * km).sum() / (km * km).sum())
            kp = np.array([curvature_from_plan(p[:, 11], p[:, 14], max(x, 0.0), LAT_T) for p, x in zip(plan[sel], vv[sel])])
            r["A_plan"] = float((kp * km).sum() / (km * km).sum())
            h2 = np.array([np.interp(2.0, T_IDXS, p[:, 11]) for p in plan[sel]])
            d2 = dpsi2[sel]
            okh = np.isfinite(d2)
            r["H2"] = float((h2[okh] * d2[okh]).sum() / (d2[okh] ** 2).sum())
            y2 = np.array([np.interp(2.0, T_IDXS, p[:, 1]) for p in plan[sel]])
            r["Y2"] = float(np.nansum(y2 * fut[:, 0, 1]) / np.nansum(fut[:, 0, 1] ** 2))
            p3 = np.array([[np.interp(3.0, T_IDXS, p[:, c]) for c in (0, 1)] for p in plan[sel]])
            r["FDE3"] = float(np.nanmean(np.linalg.norm(p3 - fut[:, 1, :2], axis=1)))
            ona = scored & (sgn * k_act >= ONSET_FRAC * peak)
            r["tau_act"] = float(tt[ona][0]) if ona.any() else np.nan
            r["tau_meas"] = float(tau_meas)
            r["lead"] = r["tau_meas"] - r["tau_act"]
            r["peak_ratio"] = float(np.nanmax(sgn * k_act[sel]) / peak)
        else:
            ts = T_IDXS[T_IDXS <= 5.0][1:]
            pp = plan[sel][:, 1:len(ts) + 1, :2]
            r["ADE5"] = float(np.nanmean(np.linalg.norm(pp - fut5[:, :, :2], axis=2)))
        rows.append(r)
    return rows


def cmd_metrics(src, out):
    import csv
    rows = [r for f in sorted(Path(src).glob("*.npz")) for r in window_metrics(f)]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("win", "seg", "kind", "arm"), k))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", out)


def cmd_table(src, split, stem):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    sp = splits.load(split)
    d = pd.read_csv(src)
    d = d[d.win.isin(sp.members)]
    base = d[d.arm == "N"].set_index("win")
    rows = []
    diff_m = ("A_act", "A_plan", "H2", "Y2", "peak_ratio", "lead", "FDE3")
    ratio_m = ("lane_w10", "v_ratio", "ADE5")
    for arm in [a for a in ("W90", "W116", "W90h", "R40") if a in set(d.arm)]:
        x = d[d.arm == arm].set_index("win")
        for kind, ms in (("turn", diff_m + ratio_m[:2]), ("straight", ratio_m)):
            w = base.index[(base.kind == kind)].intersection(x.index)
            for m in ms:
                if m not in x or x.loc[w, m].isna().all():
                    continue
                a, b = x.loc[w, m].to_numpy(float), base.loc[w, m].to_numpy(float)
                if m in ratio_m:
                    r = stats.bootstrap(a / b, groups=base.loc[w, "seg"].to_numpy())
                    r.update(mean_a=float(np.nanmean(a)), mean_b=float(np.nanmean(b)))
                    op = "ratio"
                else:
                    r = stats.paired(a, b, groups=base.loc[w, "seg"].to_numpy())
                    op = "diff"
                rows.append(dict(arm=arm, set=kind, metric=m, op=op, **r))
    stats.write_table(rows, Path(stem), note=f"split {sp.id}; diff = arm - N, ratio = arm / N per window; cluster bootstrap over segments")
    print(Path(str(stem) + ".md").read_text())


if __name__ == "__main__":
    {"metrics": cmd_metrics, "table": cmd_table}[sys.argv[1]](*sys.argv[2:])
