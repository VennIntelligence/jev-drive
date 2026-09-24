"""Frame-by-frame diagnosis of openpilot in Bench2Drive from the zero-shot agent's logs (plans.jsonl with plan_pos /
plan_yaw and ticks.jsonl with the logged truth pose; b2d_zeroshot_agent.py after the smoke).

Per plan: the plan (camera frame, x forward / y right) against the camera's actual future track from the truth
pose at 1 / 2 / 4 s (the comma1M replay's metric, scripts/openpilot_replay.py), desired curvature against the
actual curvature 0.275 s later, and whether the controller was fed. In shadow mode (drive = oracle) the route
oracle drives, so this is the open-loop error on CARLA frames; in model mode it is the closed loop.

  python scripts/zeroshot_b2d_openpilot_diag.py RUN_DIR [RUN_DIR ...] --out table.csv
"""
import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
CAM_X = 1.779          # openpilot camera ahead of the rear axle (scripts/zeroshot_rigs.OP_MOUNT_RIG)
HORIZONS = (1.0, 2.0, 4.0)
LAT_DELAY = 0.275


def load(attempt):
    plans = [json.loads(l) for l in open(attempt / "plans.jsonl")]
    ticks = [json.loads(l) for l in open(attempt / "ticks.jsonl")]
    return plans, ticks


def attempt_rows(attempt):
    plans, ticks = load(attempt)
    cfg = json.loads((attempt / "agent_config.json").read_text()) if (attempt / "agent_config.json").exists() else {}
    cam_x = float(cfg.get("op_mount", [CAM_X])[0])
    tt = [t for t in ticks if "truth" in t]
    if not tt or not plans or "plan_pos" not in plans[0]:
        return []
    t = np.array([x["t"] for x in tt])
    tr = np.array([x["truth"] for x in tt])                  # rear axle, CARLA world (y right, yaw clockwise)
    yaw = np.unwrap(tr[:, 2])
    cam = tr[:, :2] + cam_x * np.stack([np.cos(yaw), np.sin(yaw)], -1)
    speed = np.array([x["v"] for x in tt])
    yaw_rate = np.gradient(yaw, t)
    curv = yaw_rate / np.maximum(speed, 1.0)                  # right-positive, as openpilot's decoded curvature
    rows = []
    for p in plans:
        t0 = p["t"]
        if t0 < t[0] or t0 > t[-1]:
            continue
        c0 = np.array([np.interp(t0, t, cam[:, k]) for k in range(2)])
        y0 = np.interp(t0, t, yaw)
        fwd, right = np.array([math.cos(y0), math.sin(y0)]), np.array([-math.sin(y0), math.cos(y0)])
        pp = np.asarray(p["plan_pos"])
        row = dict(frame=p["frame"], t=t0, speed=p["speed"], warmup=p.get("warmup", False), desire=p.get("desire"),
                   curvature=p.get("curvature"), accepted=p.get("accepted"),
                   curv_truth=float(np.interp(t0 + LAT_DELAY, t, curv)))
        for h in HORIZONS:
            if t0 + h > t[-1]:
                row[f"lat@{h:g}s"] = row[f"lon@{h:g}s"] = row[f"plan_y@{h:g}s"] = row[f"lonb@{h:g}s"] = np.nan
                continue
            c = np.array([np.interp(t0 + h, t, cam[:, k]) for k in range(2)]) - c0
            gx, gy = c @ fwd, c @ right
            px, py = (np.interp(h, T_IDXS, pp[:, k]) for k in range(2))
            row[f"lat@{h:g}s"], row[f"lon@{h:g}s"], row[f"plan_y@{h:g}s"] = abs(py - gy), abs(px - gx), py
            row[f"lonb@{h:g}s"] = px - gx                      # signed: + means the plan runs ahead of the car
        rows.append(row)
    return rows


def summarize(rows, label):
    r = [x for x in rows if not x["warmup"] and x["speed"] > 1.0 and np.isfinite(x["lat@2s"])]
    if not r:
        return dict(label=label, n=0)
    k = np.array([x["curvature"] for x in r])
    kt = np.array([x["curv_truth"] for x in r])
    out = dict(label=label, n=len(r))
    for h in HORIZONS:
        out[f"lat@{h:g}s"] = float(np.nanmean([x[f"lat@{h:g}s"] for x in r]))
        out[f"lon@{h:g}s"] = float(np.nanmean([x[f"lon@{h:g}s"] for x in r]))
        out[f"lonb@{h:g}s"] = float(np.nanmean([x[f"lonb@{h:g}s"] for x in r]))
    out["curv_r"] = float(np.corrcoef(k, kt)[0, 1]) if len(r) > 2 and k.std() > 0 and kt.std() > 0 else np.nan
    out["curv_mae_per_km"] = float(np.abs(k - kt).mean() * 1e3)
    y2 = np.array([x["plan_y@2s"] for x in r])
    out["flip_frac"] = float(np.mean(np.sign(y2[1:]) != np.sign(y2[:-1]))) if len(y2) > 2 else np.nan
    out["dy2_mean"] = float(np.mean(np.abs(np.diff(y2)))) if len(y2) > 2 else np.nan
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    table, pooled = [], {}
    for run in a.runs:
        for att in sorted(run.glob("attempts/*/*")):
            rows = attempt_rows(att)
            if not rows:
                continue
            s = summarize(rows, f"{run.name}/{att.parent.name}")
            table.append(s)
            pooled.setdefault(run.name, []).extend(rows)
    for name, rows in pooled.items():
        table.append(summarize(rows, f"{name}/ALL"))
    keys = ["label", "n", "lat@1s", "lat@2s", "lat@4s", "lon@1s", "lon@2s", "lon@4s", "lonb@2s", "lonb@4s", "curv_r", "curv_mae_per_km",
            "flip_frac", "dy2_mean"]
    w = csv.DictWriter(a.out.open("w") if a.out else sys.stdout, keys, extrasaction="ignore")
    w.writeheader()
    for s in table:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
