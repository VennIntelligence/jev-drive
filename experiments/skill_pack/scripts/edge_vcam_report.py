"""Lane EDGE-FIX readouts (plan: experiments/skill_pack/plans/2026-10-04-roadedge-diagnosis-plan.md, addendum 3).

navsim2 env, CPU. Arms are op_lb pred stems (runs/op_lb/<data>/preds/<stem>__base.npz, official CSVs
v1_<split>_opi_<data>_<stem>__base / v2_navhard_two_stage_opi_lb_navhard_<stem>__base).
  pick     --kind height | trk: navtrain v1 PDMS of each candidate against a base stem -> PICK <arg> (largest delta, kept even
           if negative, as pre-registered) and results/roadedge/vcam/navtrain_<kind>.json
  test     navtest per-token paired deltas from the official v1 CSVs; navhard per-token devkit pdm_score of every arm (the
           harness of hist_align_report.py, checked against the official combined EPDMS), devkit aggregation, paired bootstrap
           over the scene-mapping groups; contrasts b - a and difference-in-differences (b1 - a1) - (b0 - a0) with the same
           bootstrap draws. --contrasts name=a:b, --did name=a0:a1:b0:b1 (stems or arm letters of --arms k=stem).
  scale    navtest lane / road width ratio and plan speed over the logged speed of plan files (edge_height_eval.model_cols)
"""
import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

OUT = REPO / "experiments/skill_pack/results/roadedge/vcam"
V1 = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
V2 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
      "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
SPLIT = {"lb_navtrain": "navtrain", "lb_navtest": "navtest"}


def ci(x, B):
    b = x[B].mean(1)
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def v1_rows(data, stem):
    fs = sorted(glob.glob(str(L.D / "runs/navsim/eval" / f"v1_{SPLIT[data]}_opi_{data}_{stem}__base" / "*/*.csv")))
    if not fs:
        raise FileNotFoundError(f"no official CSV for {data} {stem}")
    d = pd.read_csv(fs[-1]).set_index("token")
    return d[d.valid.astype(bool) & (d.index != "average")]


def cmd_pick(a):
    base = v1_rows("lb_navtrain", a.base)
    out = {"base": dict(stem=a.base, pdms=round(100 * float(base.score.mean()), 3), n=len(base))}
    rng = np.random.default_rng(0)
    for arg, stem in (c.split("=", 1) for c in a.cands):
        d = v1_rows("lb_navtrain", stem)
        i = base.index.intersection(d.index)
        ds = 100 * (d.loc[i, "score"] - base.loc[i, "score"]).to_numpy()
        out[arg] = dict(stem=stem, n=len(i), pdms=round(100 * float(d.loc[i, "score"].mean()), 3), delta=round(float(ds.mean()), 3),
                        delta_ci95=ci(ds, rng.integers(0, len(i), (5000, len(i)))),
                        **{f"{c}_delta": round(100 * float((d.loc[i, c] - base.loc[i, c]).mean()), 3) for c in V1})
    best = max((k for k in out if k != "base"), key=lambda k: out[k]["delta"])
    out["pick"] = best
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"navtrain_{a.kind}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("PICK", best)


_W = {}


def _init(stems):
    import offroad_replay_cf as R
    R._init(argparse.Namespace(variants=[]))
    _W.update(R=R, P={s: L.poses_by_token(L.D / f"runs/op_lb/lb_navhard/preds/{s}__base.npz") for s in stems})


def _work(token):
    R = _W["R"]
    mc = L.load_cache(R._W["cp"][token])
    return token, {s: R._score(mc, P[token], token)[0] for s, P in _W["P"].items()}


def official_combined(stem):
    fs = sorted(glob.glob(str(L.D / "runs/navsim/eval" / f"v2_navhard_two_stage_opi_lb_navhard_{stem}__base" / "*/*.csv")))
    return None if not fs else round(100 * float(pd.read_csv(fs[-1]).set_index("token").loc["extended_pdm_score_combined", "score"]), 3)


def cmd_test(a):
    arms = dict(x.split("=", 1) for x in a.arms)
    con = [(n, *v.split(":")) for n, v in (c.split("=", 1) for c in a.contrasts)]
    did = [(n, *v.split(":")) for n, v in (c.split("=", 1) for c in a.did)]
    res = {"arms": arms}
    # navtest: per-token official v1 rows
    T = {k: v1_rows("lb_navtest", s) for k, s in arms.items()}
    idx = sorted(set.intersection(*(set(d.index) for d in T.values())))
    sc = {k: 100 * d.loc[idx, "score"].to_numpy() for k, d in T.items()}
    B = np.random.default_rng(0).integers(0, len(idx), (5000, len(idx)))
    nt = {"n": len(idx), "pdms": {k: round(float(v.mean()), 3) for k, v in sc.items()},
          "dac": {k: round(100 * float(T[k].loc[idx, "drivable_area_compliance"].mean()), 3) for k in arms}}
    for n, x, y in con:
        d = sc[y] - sc[x]
        nt[n] = dict(a=x, b=y, delta=round(float(d.mean()), 3), ci95=ci(d, B),
                     **{f"{c}_delta": round(100 * float((T[y].loc[idx, c] - T[x].loc[idx, c]).mean()), 3) for c in V1})
    for n, a0, a1, b0, b1 in did:
        d = (sc[b1] - sc[b0]) - (sc[a1] - sc[a0])
        nt[n] = dict(did=f"({b1}-{b0})-({a1}-{a0})", delta=round(float(d.mean()), 3), ci95=ci(d, B))
    res["navtest"] = nt
    print(json.dumps(nt, indent=1), flush=True)
    # navhard: per-token devkit scores, two-stage aggregation, group bootstrap
    import offroad_replay_cf as R
    from offroad_gain import group_scores
    tokens = [e["token"] for e in L.index()]
    stems = sorted(set(arms.values()))
    t0, rows = time.time(), {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(stems,)) as pool:
        for i, (t, r) in enumerate(pool.imap_unordered(_work, tokens, chunksize=4)):
            rows[t] = r
            if i % 1000 == 0:
                print("navhard", i, f"{time.time() - t0:.0f} s", flush=True)
    _, _, _, mapping, samp = L.setup_scoring()
    nh, grp = {"n_tokens": len(tokens)}, {}
    for k, s in arms.items():
        comb, s1, s2, df = R.aggregate([rows[t][s] for t in tokens], mapping, samp)
        grp[k] = group_scores(df, mapping)
        nh[k] = dict(stem=s, combined=round(100 * float(comb["score"]), 3), stage1=round(100 * float(s1["score"]), 3),
                     stage2=round(100 * float(s2["score"]), 3), official_combined=official_combined(s),
                     dac_s1=round(100 * float(s1["drivable_area_compliance"]), 3), dac_s2=round(100 * float(s2["drivable_area_compliance"]), 3))
    G = np.random.default_rng(0).integers(0, len(next(iter(grp.values()))), (5000, len(next(iter(grp.values())))))
    nh["n_groups"] = int(G.shape[1])
    for n, x, y in con:
        d = 100 * (grp[y] - grp[x])
        nh[n] = dict(a=x, b=y, delta=round(float(d.mean()), 3), ci95=ci(d, G))
    for n, a0, a1, b0, b1 in did:
        d = 100 * ((grp[b1] - grp[b0]) - (grp[a1] - grp[a0]))
        nh[n] = dict(did=f"({b1}-{b0})-({a1}-{a0})", delta=round(float(d.mean()), 3), ci95=ci(d, G))
    res["navhard"] = nh
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(nh, indent=1))


def cmd_scale(a):
    import pickle
    from edge_height_eval import model_cols
    S = pd.read_pickle(L.D / "runs/skill_pack/edge_diag/sections_navtest.pkl")
    idx = {e["token"]: e for e in pickle.load(open(L.D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
    res = {}
    for arm in a.plans:
        M = model_cols(L.D / f"runs/op_lb/lb_navtest/plans/{arm}.npz").merge(
            S[["token", "x", "inter", "ego0L", "ego0R", "hum_y", "scL", "scR"]], on=["token", "x"])
        M["v"] = M.token.map(lambda t: float(np.linalg.norm(idx[t]["vel"][-1])))
        lv = (~M.inter) & M.ego0L.notna() & (M.p1 > .5) & (M.p2 > .5) & ~(M.hum_y.abs() > 1)
        ev = (~M.inter) & M.scL.notna()
        sp = M.drop_duplicates("token")
        sp = sp[sp.v > 3]
        res[arm] = dict(n_tokens=int(M.token.nunique()), n_lane_sections=int(lv.sum()),
                        lane_ratio=round(float((M.ow / (M.ego0L - M.ego0R))[lv].median()), 4),
                        road_ratio=round(float((M.rw / (M.scL - M.scR))[ev].median()), 4),
                        speed_ratio=round(float((sp.pv0 / sp.v).median()), 4), h_model=round(float(M[lv].hz.median()), 4))
        print(arm, res[arm], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scale_navtest.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "test", "scale"])
    ap.add_argument("--kind", default="height")
    ap.add_argument("--base", default="vh187-cinque")
    ap.add_argument("--cands", nargs="+", default=[], help="arg=stem")
    ap.add_argument("--arms", nargs="+", default=[], help="key=stem")
    ap.add_argument("--contrasts", nargs="+", default=[], help="name=a:b (b - a)")
    ap.add_argument("--did", nargs="+", default=[], help="name=a0:a1:b0:b1")
    ap.add_argument("--plans", nargs="+", default=[], help="scale: lb_navtest plan stems")
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    {"pick": cmd_pick, "test": cmd_test, "scale": cmd_scale}[a.cmd](a)
