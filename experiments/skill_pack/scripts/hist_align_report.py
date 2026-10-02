"""History alignment rule (plans/2026-10-04-history-align-plan.md): paired official-scorer comparison against the native arm.

navsim2 env, CPU.
  navhard   per-token devkit pdm_score of every arm's pose file (offroad_replay_cf._score, the harness that reproduces the
            official two-stage numbers), devkit aggregation, paired bootstrap over the scene-mapping groups; the combined
            EPDMS of every arm is checked against its official CSV (op_interp_score.sh) when that exists. Plus the
            wrong-direction class: stage-2 opposite-side rate, the 312 baseline opposite-side tokens, DAC-failure classes.
  navtest   paired per-token PDMS difference from the official v1 CSVs.
Output: results/history-align/{navhard,navtest}.json, tables/*.csv.

  $DATA_DIR/envs/navsim2/bin/python experiments/skill_pack/scripts/hist_align_report.py navhard --arms rot0 straight --procs 28
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

OUT = REPO / "experiments/skill_pack/results/history-align"
POSES = lambda data, arm: L.D / f"runs/op_lb/{data}/preds/gimm-cinque{'' if arm == 'base' else '_al-' + arm}__base.npz"  # noqa: E731
CSV = lambda ver, split, data, arm: f"{ver}_{split}_opi_{data}_gimm-cinque{'' if arm == 'base' else '_al-' + arm}__base"  # noqa: E731
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
_W = {}


def _init(arms):
    import offroad_replay_cf as R
    R._init(argparse.Namespace(variants=[]))
    _W.update(R=R, P={a: L.poses_by_token(POSES("lb_navhard", a)) for a in arms})


def work(token):
    R = _W["R"]
    mc = L.load_cache(R._W["cp"][token])
    return token, {a: R._score(mc, P[token], token)[0] for a, P in _W["P"].items()}


def ci(x, B):
    b = x[B].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def official_combined(name):
    fs = sorted(glob.glob(str(L.D / "runs/navsim/eval" / name / "*/*.csv")))
    if not fs:
        return None
    d = pd.read_csv(fs[-1]).set_index("token")
    return 100 * float(d.loc["extended_pdm_score_combined", "score"])


def cmd_navhard(a):
    import offroad_analysis as A
    import offroad_replay_cf as R
    arms = ["base"] + a.arms
    tokens = [e["token"] for e in L.index()]
    stage = {e["token"]: e["stage"] for e in L.index()}
    t0, rows = time.time(), {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(arms,)) as pool:
        for i, (t, r) in enumerate(pool.imap_unordered(work, tokens, chunksize=4)):
            rows[t] = r
            if i % 1000 == 0:
                print(i, f"{time.time() - t0:.0f} s", flush=True)
    _, _, _, mapping, samp = L.setup_scoring()
    from offroad_gain import group_scores
    summ, grp, dfs = {}, {}, {}
    for v in arms:
        comb, s1, s2, df = R.aggregate([rows[t][v] for t in tokens], mapping, samp)
        dfs[v] = df.set_index("token")
        summ[v] = dict(combined=100 * float(comb["score"]), stage1=100 * float(s1["score"]), stage2=100 * float(s2["score"]),
                       official_combined=official_combined(CSV("v2", "navhard_two_stage", "lb_navhard", v)),
                       **{f"{c}_s1": 100 * float(s1[c]) for c in SUBS}, **{f"{c}_s2": 100 * float(s2[c]) for c in SUBS})
        grp[v] = group_scores(df, mapping)
        print(v, {k: round(x, 2) if isinstance(x, float) else x for k, x in summ[v].items() if k in ("combined", "stage1", "stage2", "official_combined")}, flush=True)
    B = np.random.default_rng(0).integers(0, len(grp["base"]), (5000, len(grp["base"])))
    for v in a.arms:
        dg = 100 * (grp[v] - grp["base"])
        summ[v]["combined_delta"] = float(dg.mean())
        summ[v]["combined_delta_ci95"] = ci(dg, B)
        summ[v]["n_groups"] = len(dg)
    # wrong-direction class and DAC-failure classes (definitions of the diagnosis plan; classes from the baseline table)
    tab = pd.read_pickle(L.OUT / "token_table.pkl").set_index("token")
    cls = A.classify(tab.reset_index(), "native")
    cls.index = tab.index
    two = [t for t in tokens if stage[t] == "two"]
    ref = tab.loc[two, "ref_end_y"].to_numpy()
    PB = {v: L.poses_by_token(POSES("lb_navhard", v)) for v in arms}
    end = {v: np.array([PB[v][t][-1, 1] for t in two]) for v in arms}
    opp = {v: (end[v] * ref < 0) & (np.abs(end[v]) > 1) & (np.abs(ref) > 1) & (np.abs(end[v] - ref) > 2) for v in arms}
    dac = {v: dfs[v].loc[two, "drivable_area_compliance"].to_numpy() for v in arms}
    sc = {v: dfs[v].loc[two, "score"].to_numpy() for v in arms}
    o0 = opp["base"]
    wd = dict(n_stage2=len(two), n_base_opposite=int(o0.sum()))
    for v in arms:
        wd[v] = dict(opposite_rate_stage2=float(opp[v].mean()), opposite_n=int(opp[v].sum()), still_opposite_on_base_set=float(opp[v][o0].mean()),
                     dac_on_base_set=float(dac[v][o0].mean()), score_on_base_set=float(sc[v][o0].mean()),
                     dac_stage2=float(dac[v].mean()), dac_off_base_set=float(dac[v][~o0].mean()), score_off_base_set=float(sc[v][~o0].mean()))
    c2 = cls.loc[two].replace("", "pass").to_numpy()
    crow = []
    for c in ["pass", "wrong_direction", "early_clip", "junction", "late_undershoot", "other"]:
        k = c2 == c
        r = dict(base_class=c, n=int(k.sum()))
        for v in arms:
            r[f"dac_{v}"] = float(dac[v][k].mean())
            r[f"score_{v}"] = float(sc[v][k].mean())
        crow.append(r)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(crow).to_csv(OUT / "tables/navhard_classes.csv", index=False)
    # per-command split of the stage-2 score change (where the gain / loss sits)
    cmd = tab.loc[two, "cmd"].to_numpy()
    cmr = [dict(cmd=c, n=int((cmd == c).sum()), **{f"score_{v}": float(sc[v][cmd == c].mean()) for v in arms}, **{f"dac_{v}": float(dac[v][cmd == c].mean()) for v in arms})
           for c in ["left", "straight", "right", "unknown"] if (cmd == c).any()]
    pd.DataFrame(cmr).to_csv(OUT / "tables/navhard_stage2_by_cmd.csv", index=False)
    (OUT / "navhard.json").write_text(json.dumps(dict(arms=summ, wrong_direction=wd), indent=1))
    print(json.dumps(dict(arms=summ, wrong_direction=wd), indent=1))
    print(pd.DataFrame(crow).round(3).to_string(index=False))
    print(pd.DataFrame(cmr).round(3).to_string(index=False))


def cmd_navtest(a):
    def load(arm):
        fs = sorted(glob.glob(str(L.D / "runs/navsim/eval" / CSV("v1", "navtest", "lb_navtest", arm) / "*/*.csv")))
        if not fs:
            return None
        d = pd.read_csv(fs[-1]).set_index("token")
        return d[d.valid.astype(bool) & (d.index != "average")]
    base = load("base")
    out = {"base": dict(pdms=100 * float(base.score.mean()), n=len(base))}
    rng = np.random.default_rng(0)
    subs = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
    for v in a.arms:
        d = load(v)
        if d is None:
            continue
        idx = base.index.intersection(d.index)
        ds = 100 * (d.loc[idx, "score"] - base.loc[idx, "score"]).to_numpy()
        B = rng.integers(0, len(idx), (5000, len(idx)))
        out[v] = dict(pdms=100 * float(d.loc[idx, "score"].mean()), n=len(idx), pdms_delta=float(ds.mean()), pdms_delta_ci95=ci(ds, B),
                      **{f"{c}_delta": 100 * float((d.loc[idx, c] - base.loc[idx, c]).mean()) for c in subs})
    (OUT / "navtest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["navhard", "navtest"])
    ap.add_argument("--arms", nargs="+", default=["rot0", "straight", "straight_keys"])
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"navhard": cmd_navhard, "navtest": cmd_navtest}[a.cmd](a)
