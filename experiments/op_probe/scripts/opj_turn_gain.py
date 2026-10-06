"""Turn gain: is P2's sharp-turn deficit proportional shrinkage of turning, or a cap (model output ceiling / openpilot lateral clip)?  CPU only, stored outputs.

  navtest  (box)  per-token table: logged future vs the plans of P0 (W frames), P2 s0/s1, P2+hinge s0/s1, WA-JEPA -> $OUT/navtest_turn.csv.gz
  hugsim   (box)  per-step table of the HUGSIM 64 runs (spec + exam presets, P2 / P2+hinge, 2 seeds): requested curvature (kappa), openpilot lateral
                  chain trace (act / des / real from sim.log `op_ctrl` lines), speed, steer, pose -> $OUT/hugsim_steps.csv.gz
  report   (Mac)  statistics + figures + the markdown from the two tables (copied to results/turn-gain/data/)

  box:  .venv/bin/python experiments/op_probe/scripts/opj_turn_gain.py navtest|hugsim [--out DIR]
  Mac:  .venv/bin/python experiments/op_probe/scripts/opj_turn_gain.py report
"""
import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results/turn-gain"
OUTB = D / "runs/op_probe/turn_gain"
PRED = D / "runs/op_lb/lb_navtest/preds"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
WA_TRAJ = D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl"
HS = D / "runs/op_parity/hugsim"
MODELS = {"P0": ["warp-cinque_PPP0__base.npz"], "P2": ["warp-cinque_PPP2-F-s0__base.npz", "warp-cinque_PPP2-F-s1__base.npz"],
          "P2H": ["warp-cinque_PPP2H10-F-s0__base.npz", "warp-cinque_PPP2H10-F-s1__base.npz"]}
EDGES = [0, 5, 20, 45, 400]
LABELS = {"P0": "P0 shipped (W)", "P2": "P2", "P2H": "P2 + hinge", "WA": "WA-JEPA"}
COL = {"P0": "#7f7f7f", "P2": "#d95f02", "P2H": "#7570b3", "WA": "#1b9e77"}
B, SEED = 4000, 0


# ---------------------------------------------------------------- plan descriptors (t0 rear-axle frame, x forward, y left, 8 poses at 0.5 s)
def desc(P):
    """P (n, 8, 3) -> dict of arrays. yaw/y at 2 s (idx 3) and 4 s (idx 7), arc length, mean curvature over [0, T], peak segment lateral accel / curvature."""
    P = np.asarray(P, float)
    n = len(P)
    P0 = np.concatenate([np.zeros((n, 1, 3)), P], 1)
    yaw = np.unwrap(P0[:, :, 2], axis=1)
    seg = np.linalg.norm(np.diff(P0[:, :, :2], axis=1), axis=2)                 # (n, 8)
    dyaw = np.diff(yaw, axis=1)
    v = seg / 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        kseg = np.where(seg > 0.5, dyaw / seg, np.nan)                          # curvature of segments longer than 0.5 m
    alat = dyaw / 0.5 * v                                                       # signed, v^2 * kappa
    s2, s4 = seg[:, :4].sum(1), seg.sum(1)
    out = dict(yaw2=np.degrees(yaw[:, 4]), yaw4=np.degrees(yaw[:, 8]), y2=P0[:, 4, 1], y4=P0[:, 8, 1], s2=s2, s4=s4,
               k2=np.where(s2 > 3, yaw[:, 4] / np.maximum(s2, 1e-6), np.nan), k4=np.where(s4 > 6, yaw[:, 8] / np.maximum(s4, 1e-6), np.nan),
               v_mean=s4 / 4.0)
    sgn = np.sign(yaw[:, 8])                                                    # side of the turn: peaks are taken in the turn direction
    out["kpk"] = np.nanmax(np.where(np.isfinite(kseg), kseg * sgn[:, None], -np.inf), 1)
    out["kpk"] = np.where(np.isfinite(out["kpk"]), out["kpk"], np.nan)
    out["apk"] = np.max(alat * sgn[:, None], 1)
    out["amax_abs"] = np.max(np.abs(alat), 1)
    return out


def cmd_navtest(a):
    z = np.load(TAB)
    names, log, fut, v0 = z["names"], z["log"], z["fut"], z["speed"]
    ok = ~np.isnan(fut[:, 0, 0])
    df = pd.DataFrame({"token": names, "log": log, "v0": v0, "ok": ok})
    for k, v in desc(np.nan_to_num(fut)).items():
        df[f"L_{k}"] = np.where(ok, v, np.nan)
    plans = {}
    for m, fs in MODELS.items():
        for i, f in enumerate(fs):
            q = np.load(PRED / f)
            pos = {t: j for j, t in enumerate(q["tokens"].tolist())}
            ix = np.array([pos.get(t, -1) for t in names])
            plans[f"{m}s{i}" if len(fs) > 1 else m] = np.where((ix >= 0)[:, None, None], q["poses"][np.maximum(ix, 0)], np.nan)
    w = pickle.load(open(WA_TRAJ, "rb"))["trajectories"]
    plans["WA"] = np.stack([np.asarray(w[t], float) if t in w else np.full((8, 3), np.nan) for t in names])
    for m, P in plans.items():
        have = ~np.isnan(P[:, 0, 0])
        for k, v in desc(np.nan_to_num(P)).items():
            df[f"{m}_{k}"] = np.where(have, v, np.nan)
    out = Path(a.out or OUTB)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "navtest_turn.csv.gz", index=False, float_format="%.5g")
    print(len(df), "tokens;", {m: int((~np.isnan(P[:, 0, 0])).sum()) for m, P in plans.items()})


# ---------------------------------------------------------------- hugsim per-step table
def cmd_hugsim(a):
    res = pd.read_csv(HS / "results.csv")
    rows = []
    for preset, pre in (("spec", "pp-spec-"), ("exam", "pp-")):
        for arm in ("P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1"):
            tag = pre + arm
            r = res[res.tag == tag].drop_duplicates("scenario", keep="last")
            for _, x in r.iterrows():
                d = Path(x.run_dir)
                if not (d / "zs_steps.jsonl").exists():
                    continue
                st = [json.loads(l) for l in open(d / "zs_steps.jsonl")]
                oc = []
                if (d / "sim.log").exists():
                    oc = [json.loads(m.group(1)) for m in (re.match(r"op_ctrl (\{.*\})", l) for l in open(d / "sim.log")) if m]
                for i, s in enumerate(st):
                    o = oc[i] if i < len(oc) else {}
                    rows.append(dict(preset=preset, arm=arm, scenario=x.scenario, scene=x.scene, step=s["step"], t=s["t"], x=s["pos"][0], z=s["pos"][1],
                                     theta=s["theta"], v=s["v"], steer=s["steer"], kappa=s.get("kappa"), act=o.get("act"), des=o.get("des"),
                                     real=o.get("real"), kmean=o.get("kmean"), n_steps=int(x.steps), end=x.end, hd=x.hdscore, dac=x.dac, nc=x.nc))
    out = Path(a.out or OUTB)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "hugsim_steps.csv.gz", index=False, float_format="%.6g")
    print(len(rows), "steps")


# ---------------------------------------------------------------- report helpers
def cboot(fn, g, B=B, seed=SEED, **cols):
    """Cluster bootstrap over groups g of a statistic fn(**sliced cols) -> (est, lo, hi). Resamples groups; concatenates their rows."""
    g = np.asarray(g)
    codes, uniq = pd.factorize(g)
    order = np.argsort(codes, kind="stable")
    cuts = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    cols = {k: np.asarray(v)[order] for k, v in cols.items()}
    est = fn(**cols)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([np.arange(cuts[p], cuts[p + 1]) for p in pick])
        vals.append(fn(**{k: v[idx] for k, v in cols.items()}))
    return est, float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def slope0(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float((x[m] * y[m]).sum() / (x[m] ** 2).sum()) if m.any() else np.nan


def ratio_of_sums(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(y[m].sum() / x[m].sum()) if m.any() and x[m].sum() != 0 else np.nan


def fmt(r, k=2):
    return f"{r[0]:.{k}f} [{r[1]:.{k}f}, {r[2]:.{k}f}]"
