"""Lateral gain calibration of the native plan, fitted on navtrain (offroad_amplitude.py -> gains.json), scored with the official scorers.
NOT a privileged counterfactual: the gain uses no reference, no label and no scene information; every token is modified.

navsim2 env, CPU.
  make          write the navtest pose files (<run>/gain_<variant>_navtest.npz) for the official v1 scorer
  navhard       in-process devkit scoring of both stages (as offroad_replay_cf.py), combined / stage DAC, group bootstrap CI
  navtest-report  paired bootstrap from the official per-token CSVs (after navsim_zs_score.sh)
Variants: gain_single (one lateral gain), gain_horizon (gain per horizon 1, 2, 3, 4 s, linear in between).
Path edit: y'(t) = g(t) y(t) on the 0.1 s dense plan, heading corrected by the change of the path tangent (as in offroad_replay_cf.v_scale).
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

OUT = REPO / "experiments/skill_pack/results/navhard-offroad"
VARIANTS = ["gain_single", "gain_horizon"]
_W = {}


def gains():
    g = json.load(open(OUT / "gains.json"))
    return dict(gain_single=lambda t: np.full_like(t, g["single"]),
                gain_horizon=lambda t: np.interp(t, [0, 1, 2, 3, 4], [g["horizon"]["1"], g["horizon"]["1"], g["horizon"]["2"], g["horizon"]["3"], g["horizon"]["4"]]))


def apply_gain(p8, gf):
    d = L.dense_from_poses(p8)
    y2 = d[:, 1] * gf(L.T_DENSE)
    dx, dy, dy2 = np.gradient(d[:, 0], 0.1), np.gradient(d[:, 1], 0.1), np.gradient(y2, 0.1)
    o = d.copy()
    o[:, 1] = y2
    o[:, 2] = d[:, 2] + np.where(np.hypot(dx, dy) > 1.0, np.arctan2(dy2, dx) - np.arctan2(dy, dx), 0.0)
    return L.poses_from_dense(o)


def cmd_make(a):
    p = np.load(L.D / "runs/op_lb/lb_navtest/preds/gimm-cinque__base.npz")
    G = gains()
    for v in VARIANTS:
        poses = np.stack([apply_gain(x.astype(np.float64), G[v]) for x in p["poses"]]).astype(np.float32)
        f = L.OUT / f"{v}_navtest.npz"
        np.savez(f, tokens=p["tokens"], poses=poses)
        print(f, poses.shape, "mean |y4| base / new", float(np.abs(p["poses"][:, -1, 1]).mean()), float(np.abs(poses[:, -1, 1]).mean()))


def _init():
    import offroad_replay_cf as R
    argsns = argparse.Namespace(variants=[])
    R._init(argsns)
    _W.update(R=R, G=gains())


def work(token):
    R = _W["R"]
    mc = L.load_cache(R._W["cp"][token])
    p8 = R._W["P"]["native"][token]
    out = {"token": token, "stage": R._W["stage"][token], "rows": {}}
    row, _ = R._score(mc, p8, token)
    out["rows"]["base"] = row
    for v in VARIANTS:
        row, _ = R._score(mc, apply_gain(p8, _W["G"][v]), token)
        out["rows"][v] = row
    return out


def group_scores(df, mapping):
    """Per mapping entry: mean of the two group scores (stage-1 score x weighted stage-2 mean), as calculate_individual_mapping_scores."""
    d = df.set_index("token")
    out = []
    for (o, p), pairs in mapping.items():
        gs = []
        for t1, t2s in ((o, [x[0] for x in pairs]), (p, [x[1] for x in pairs])):
            w = d.loc[t2s, "weight"].to_numpy()
            gs.append(d.loc[t1, "score"] * (d.loc[t2s, "score"].to_numpy() * w).sum() / w.sum())
        out.append(np.mean(gs))
    return np.array(out)


def cmd_navhard(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import offroad_replay_cf as R
    idx = L.index()
    tokens = [e["token"] for e in idx]
    t0 = time.time()
    res = {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tokens, chunksize=4)):
            res[r["token"]] = r
            if i % 1000 == 0:
                print(i, time.time() - t0, flush=True)
    _, _, _, mapping, samp = L.setup_scoring()
    summ, grp, tok = {}, {}, {}
    for v in ["base"] + VARIANTS:
        rows = [res[t]["rows"][v] for t in tokens]
        comb, s1, s2, df = R.aggregate(rows, mapping, samp)
        d = df.set_index("token")
        summ[v] = dict(combined=100 * float(comb["score"]), s1=100 * float(s1["score"]), s2=100 * float(s2["score"]),
                       dac_s1=100 * float(s1["drivable_area_compliance"]), dac_s2=100 * float(s2["drivable_area_compliance"]),
                       ep_s2=100 * float(s2["ego_progress"]), ddc_s2=100 * float(s2["driving_direction_compliance"]),
                       dac_s2_uniform=100 * float(d.loc[[t for t in tokens if res[t]["stage"] == "two"], "drivable_area_compliance"].mean()))
        grp[v] = group_scores(df, mapping)
        tok[v] = d["score"]
        print(v, summ[v], flush=True)
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(grp["base"]), (5000, len(grp["base"])))
    for v in VARIANTS:
        dg = grp[v] - grp["base"]
        bs = 100 * dg[B].mean(1)
        summ[v]["combined_delta"] = 100 * float(dg.mean())
        summ[v]["combined_delta_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    (OUT / "gain_navhard.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_navtest(a):
    import glob
    base = pd.read_csv(sorted(glob.glob(str(L.D / "runs/navsim/eval/v1_navtest_opi_lb_navtest_gimm-cinque__base/*/*.csv")))[-1]).set_index("token")
    base = base[base.valid.astype(bool) & (base.index != "average")]       # the v1 CSV ends with an "average" summary row
    out = {}
    rng = np.random.default_rng(0)
    out["base"] = dict(pdms=100 * float(base.score.mean()), dac=100 * float(base.drivable_area_compliance.mean()), n=len(base))
    for v in VARIANTS:
        fs = sorted(glob.glob(str(L.D / f"runs/navsim/eval/v1_navtest_{v}_navtest/*/*.csv")))
        if not fs:
            continue
        d = pd.read_csv(fs[-1]).set_index("token")
        d = d[d.valid.astype(bool) & (d.index != "average")]
        idx = base.index.intersection(d.index)
        ds = (d.loc[idx, "score"] - base.loc[idx, "score"]).to_numpy()
        dd = (d.loc[idx, "drivable_area_compliance"] - base.loc[idx, "drivable_area_compliance"]).to_numpy()
        B = rng.integers(0, len(idx), (5000, len(idx)))
        out[v] = dict(pdms=100 * float(d.loc[idx, "score"].mean()), dac=100 * float(d.loc[idx, "drivable_area_compliance"].mean()), n=len(idx),
                      pdms_delta=100 * float(ds.mean()), pdms_delta_ci95=[100 * float(np.percentile(ds[B].mean(1), 2.5)), 100 * float(np.percentile(ds[B].mean(1), 97.5))],
                      dac_delta=100 * float(dd.mean()), dac_delta_ci95=[100 * float(np.percentile(dd[B].mean(1), 2.5)), 100 * float(np.percentile(dd[B].mean(1), 97.5))],
                      **{f"{c}_delta": 100 * float((d.loc[idx, c] - base.loc[idx, c]).mean()) for c in ("no_at_fault_collisions", "ego_progress", "time_to_collision_within_bound", "comfort")})
    (OUT / "gain_navtest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "navhard", "navtest-report"])
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    {"make": cmd_make, "navhard": cmd_navhard, "navtest-report": cmd_navtest}[a.cmd](a)
