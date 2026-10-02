"""Tracker-lag pre-compensation (plans/2026-10-04-tracker-precomp-plan.md): readouts against the uncompensated arms.

navsim2 env, CPU.
  navtrain  official v1 PDMS of every alpha on the 3 000-token navtrain subset -> alpha* (results/tracker-precomp/navtrain.json)
  tracking  tracker response and plan realisation from trk_precomp.py diagnostics (all splits, both models)
  navhard   in-process devkit two-stage EPDMS (harness of offroad_replay_cf, reproduces the official CSVs) of base / compensated /
            ideal-tracker arms for native and N4; paired bootstrap over the scene-mapping groups; per-metric stage 1 / 2;
            the compensated arms are checked against their official CSVs
  navtest   official v1 CSVs: paired per-token bootstrap of PDMS and every sub-metric
"ideal" = the plan's own interpolated poses scored as if tracked exactly (dynamic fields by finite differences; comfort
terms are not meaningful for it and are not used).
"""
import argparse
import glob
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

OUT = REPO / "experiments/skill_pack/results/tracker-precomp"
RUN = L.D / "runs/skill_pack/trk"
BASE_POSES = {"native": {"lb_navhard": L.NATIVE_POSES, "lb_navtest": L.D / "runs/op_lb/lb_navtest/preds/gimm-cinque__base.npz",
                         "lb_navtrain": L.D / "runs/op_lb/lb_navtrain/preds/gimm-cinque__base.npz"},
              "n4": {"lb_navhard": L.N4_POSES, "lb_navtest": L.D / "runs/skill_pack/raise/n4/navtest_n4.npz"}}
BASE_CSV = {("native", "navtest"): "v1_navtest_opi_lb_navtest_gimm-cinque__base", ("n4", "navtest"): "v1_navtest_sp_n4_navtest",
            ("native", "navhard"): L.NATIVE_CSV, ("n4", "navhard"): L.N4_CSV, ("native", "navtrain"): "v1_navtrain_opi_lb_navtrain_gimm-cinque__base",
            ("native", "gain"): "v1_navtest_gain_single_navtest"}
V1 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
V2 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
      "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
comp_csv = lambda ver, split, m, al, mode="full": f"{ver}_{split}_trk_{m}_{mode}_a{al:g}"  # noqa: E731
sfx = lambda mode: "" if mode == "full" else f"_{mode}"  # noqa: E731
_W = {}


def ci(x, B):
    b = x[B].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def load_csv(name):
    fs = sorted(glob.glob(str(L.D / "runs/navsim/eval" / name / "*/*.csv")))
    if not fs:
        return None
    d = pd.read_csv(fs[-1]).set_index("token")
    return d


def v1_rows(name):
    d = load_csv(name)
    return None if d is None else d[d.valid.astype(bool) & (d.index != "average")]


# ---------------------------------------------------------------- navtrain: alpha*

def cmd_navtrain(a):
    base = v1_rows(BASE_CSV["native", "navtrain"])
    out = {"base": dict(pdms=100 * float(base.score.mean()), n=len(base))}
    rng = np.random.default_rng(0)
    for al in a.alphas:
        d = v1_rows(comp_csv("v1", "navtrain", "native", al, a.mode))
        idx = base.index.intersection(d.index)
        ds = 100 * (d.loc[idx, "score"] - base.loc[idx, "score"]).to_numpy()
        out[f"a{al:g}"] = dict(alpha=al, pdms=100 * float(d.loc[idx, "score"].mean()), n=len(idx), delta=float(ds.mean()),
                               delta_ci95=ci(ds, rng.integers(0, len(idx), (5000, len(idx)))),
                               **{f"{c}_delta": 100 * float((d.loc[idx, c] - base.loc[idx, c]).mean()) for c in V1})
    best = max(a.alphas, key=lambda x: (out[f"a{x:g}"]["delta"], -x))
    out["alpha_star"] = best if out[f"a{best:g}"]["delta"] > 0 else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"navtrain{sfx(a.mode)}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("ALPHA_STAR", out["alpha_star"] if out["alpha_star"] is not None else "none")


# ---------------------------------------------------------------- tracking: response and realisation

def cmd_tracking(a):
    rows = []
    for data in ("lb_navtrain", "lb_navtest", "lb_navhard"):
        f = RUN / f"diag_{data}{sfx(a.mode)}.pkl"
        if not f.exists():
            continue
        res = pickle.load(open(f, "rb"))
        stage = {e["token"]: e["stage"] for e in L.index()} if data == "lb_navhard" else {}
        for t, r in res.items():
            for k, v in r.items():
                if not isinstance(k, tuple):
                    continue
                for arm in ("base", "comp"):
                    s = v[arm]
                    rows.append(dict(data=data, stage=stage.get(t, ""), model=k[0], arm=arm, token=t, v0=r["v0"], pos_mean=s["pos_mean"],
                                     lat_mean=s["lat_mean"], path_d_mean=s.get("path_d_mean", np.nan),
                                     **{f"path_d{h}": (s["path_d"][i] if "path_d" in s else np.nan) for i, h in enumerate((1, 2, 4))}, **{f"pos{h}": s["pos"][i] for i, h in enumerate((1, 2, 4))},
                                     **{f"lat{h}": s["lat"][i] for i, h in enumerate((1, 2, 4))}, **{f"lon{h}": s["lon"][i] for i, h in enumerate((1, 2, 4))},
                                     **{f"sim_y{h}": s["sim_y"][i] for i, h in enumerate((1, 2, 3, 4))},
                                     **{f"plan_y{h}": s["plan_y"][i] for i, h in enumerate((1, 2, 3, 4))},
                                     dev=v["dev"], dev_y1=v["dev_y"][0], dev_y4=v["dev_y"][3], iters=len(v["cost"]) - 1,
                                     cost0=v["cost"][0], cost1=v["cost"][-1]))
    df = pd.DataFrame(rows)
    df.to_parquet(RUN / f"tracking{sfx(a.mode)}.parquet")
    summ = []
    for (data, model, arm), g in df.groupby(["data", "model", "arm"]):
        r = dict(data=data, model=model, arm=arm, n=len(g))
        for h in (1, 2, 4):
            py, sy = g[f"plan_y{h}"].to_numpy(), g[f"sim_y{h}"].to_numpy()
            k = np.abs(py) > 0.5
            r[f"realised_slope{h}"] = float((py[k] * sy[k]).sum() / (py[k] ** 2).sum())     # through-origin slope, |plan y| > 0.5 m
            r[f"realised_median{h}"] = float(np.median(sy[k] / py[k]))
            r[f"n_lat{h}"] = int(k.sum())
            r[f"pos{h}_mean"] = float(g[f"pos{h}"].mean())
            r[f"lat{h}_abs_mean"] = float(g[f"lat{h}"].abs().mean())
            r[f"lon{h}_mean"] = float(g[f"lon{h}"].mean())
        r["pos_mean"] = float(g.pos_mean.mean())
        r["path_d_abs_mean"] = float(g.path_d_mean.mean())
        for h in (1, 2, 4):
            r[f"path_d{h}_abs_mean"] = float(g[f"path_d{h}"].abs().mean())
        r["submitted_dev_max_median"] = float(g.dev.median())
        r["submitted_dev_max_p90"] = float(g.dev.quantile(0.9))
        r["dev_y4_abs_median"] = float(g.dev_y4.abs().median())
        summ.append(r)
    s = pd.DataFrame(summ)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / f"tracking{sfx(a.mode)}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(s.round(3).to_string(index=False))


# ---------------------------------------------------------------- navhard: in-process scoring incl. ideal tracker

class IdealSim:
    """Returns the submitted proposal (index 1 of the [PDM, prediction] batch) as its own 'simulated' states (exact tracking);
    dynamic fields by finite differences."""

    def __init__(self, sim):
        self.sim, self.proposal_sampling = sim, sim.proposal_sampling

    def simulate_proposals(self, states, ego):
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex as S
        out = self.sim.simulate_proposals(states, ego)
        if len(states) != 2:            # the human-penalty filter's own call (human trajectory alone): tracked as usual
            return out
        p, o, dt = states[1], out[1], 0.1
        h = np.unwrap(p[:, S.HEADING])
        v = np.gradient(p[:, :2], dt, axis=0)
        vx = v[:, 0] * np.cos(h) + v[:, 1] * np.sin(h)
        vx[0] = o[0, S.VELOCITY_X]
        w = np.gradient(h, dt)
        o[1:, :3] = p[1:, :3]
        o[1:, S.VELOCITY_X], o[1:, S.VELOCITY_Y] = vx[1:], 0.0
        o[1:, S.ACCELERATION_X], o[1:, S.ACCELERATION_Y] = np.gradient(vx, dt)[1:], 0.0
        o[1:, S.ANGULAR_VELOCITY], o[1:, S.ANGULAR_ACCELERATION] = w[1:], np.gradient(w, dt)[1:]
        return out


def _init_nh(arms):
    import offroad_replay_cf as R
    R._init(argparse.Namespace(variants=[]))
    _W.update(R=R, sim=R._W["sim"], ideal=IdealSim(R._W["sim"]), P={k: L.poses_by_token(f) for k, (f, _) in arms.items()},
              kind={k: kd for k, (_, kd) in arms.items()})


def work_nh(token):
    R = _W["R"]
    mc = L.load_cache(R._W["cp"][token])
    out = {}
    for k, P in _W["P"].items():
        R._W["sim"] = _W["ideal"] if _W["kind"][k] == "ideal" else _W["sim"]
        out[k] = R._score(mc, P[token], token)[0]
    R._W["sim"] = _W["sim"]
    return token, out


def cmd_navhard(a):
    import offroad_replay_cf as R
    from offroad_gain import group_scores
    arms = {}
    for m in ("native", "n4"):
        arms[f"{m}/base"] = (BASE_POSES[m]["lb_navhard"], "real")
        arms[f"{m}/ideal"] = (BASE_POSES[m]["lb_navhard"], "ideal")
        for al in a.alphas:
            arms[f"{m}/a{al:g}"] = (RUN / f"lb_navhard/{m}_{a.mode}_a{al:g}.npz", "real")
    tokens = [e["token"] for e in L.index()]
    stage = {e["token"]: e["stage"] for e in L.index()}
    t0, rows = time.time(), {}
    with mp.get_context("fork").Pool(a.procs, initializer=_init_nh, initargs=(arms,)) as pool:
        for i, (t, r) in enumerate(pool.imap_unordered(work_nh, tokens, chunksize=4)):
            rows[t] = r
            if i % 1000 == 0:
                print(i, f"{time.time() - t0:.0f} s", flush=True)
    _, _, _, mapping, samp = L.setup_scoring()
    summ, grp, dfs = {}, {}, {}
    for v in arms:
        comb, s1, s2, df = R.aggregate([rows[t][v] for t in tokens], mapping, samp)
        dfs[v] = df.set_index("token")
        m, arm = v.split("/")
        off = None
        if arm == "base":
            off = load_csv(BASE_CSV[m, "navhard"])
        elif arm != "ideal":
            off = load_csv(comp_csv("v2", "navhard_two_stage", m, float(arm[1:]), a.mode))
        summ[v] = dict(combined=100 * float(comb["score"]), stage1=100 * float(s1["score"]), stage2=100 * float(s2["score"]),
                       official_combined=None if off is None else 100 * float(off.loc["extended_pdm_score_combined", "score"]),
                       **{f"{c}_s1": 100 * float(s1[c]) for c in V2}, **{f"{c}_s2": 100 * float(s2[c]) for c in V2})
        grp[v] = group_scores(df, mapping)
        print(v, {k: round(x, 2) for k, x in summ[v].items() if k in ("combined", "stage1", "stage2", "official_combined") and x is not None}, flush=True)
    B = np.random.default_rng(0).integers(0, len(grp["native/base"]), (5000, len(grp["native/base"])))
    two = [t for t in tokens if stage[t] == "two"]
    one = [t for t in tokens if stage[t] == "one"]
    for v in arms:
        m, arm = v.split("/")
        if arm == "base":
            continue
        dg = 100 * (grp[v] - grp[f"{m}/base"])
        summ[v]["combined_delta"], summ[v]["combined_delta_ci95"], summ[v]["n_groups"] = float(dg.mean()), ci(dg, B), len(dg)
        for st, tk in (("s1", one), ("s2", two)):          # uniform per-token sub-metric deltas with token bootstrap
            Bt = np.random.default_rng(1).integers(0, len(tk), (2000, len(tk)))
            for c in V2 + ["score"]:
                x = 100 * (dfs[v].loc[tk, c].to_numpy(float) - dfs[f"{m}/base"].loc[tk, c].to_numpy(float))
                x = np.nan_to_num(x)
                summ[v][f"{c}_{st}_tok_delta"] = float(x.mean())
                summ[v][f"{c}_{st}_tok_delta_ci95"] = ci(x, Bt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"navhard{sfx(a.mode)}.json").write_text(json.dumps(summ, indent=1))
    keep = ["score", "weight"] + V2
    pd.concat({v: d[[c for c in keep if c in d.columns]] for v, d in dfs.items()}).to_parquet(RUN / f"navhard_tokens{sfx(a.mode)}.parquet")
    print(json.dumps(summ, indent=1))


# ---------------------------------------------------------------- navtest: official CSVs

def cmd_navtest(a):
    out = {}
    rng = np.random.default_rng(0)
    for m in ("native", "n4"):
        base = v1_rows(BASE_CSV[m, "navtest"])
        out[f"{m}/base"] = dict(pdms=100 * float(base.score.mean()), n=len(base), **{c: 100 * float(base[c].mean()) for c in V1})
        names = {f"{m}/a{al:g}": comp_csv("v1", "navtest", m, al, a.mode) for al in a.alphas}
        if m == "native":
            names["native/gain_single"] = BASE_CSV["native", "gain"]
        for v, nm in names.items():
            d = v1_rows(nm)
            if d is None:
                continue
            idx = base.index.intersection(d.index)
            B = rng.integers(0, len(idx), (5000, len(idx)))
            r = dict(pdms=100 * float(d.loc[idx, "score"].mean()), n=len(idx))
            for c in ["score"] + V1:
                x = 100 * (d.loc[idx, c].to_numpy(float) - base.loc[idx, c].to_numpy(float))
                r[f"{c}_delta"], r[f"{c}_delta_ci95"] = float(x.mean()), ci(x, B)
            r["changed_tokens"] = int((d.loc[idx, "score"] != base.loc[idx, "score"]).sum())
            out[v] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"navtest{sfx(a.mode)}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["navtrain", "tracking", "navhard", "navtest"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--mode", default="full", choices=["full", "path"])
    a = ap.parse_args()
    {"navtrain": cmd_navtrain, "tracking": cmd_tracking, "navhard": cmd_navhard, "navtest": cmd_navtest}[a.cmd](a)
