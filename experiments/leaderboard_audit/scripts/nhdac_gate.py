"""Gated tracker-lag compensation (follow-up to decisions 97 / 112). Pre-registration: plans/2026-10-04-navhard-dac-gated-comp-prereg.md.

Compensation (trk_precomp.py) is per token and independent, so a gated arm = per token the compensated arm where the gate fires, the
uncompensated arm elsewhere. Gates use only the plan and ego_status (reference-free, map-free):
  gE  mean distance of the scorer's LQR replay of the unmodified plan to the plan polyline (predicted tracking error)
  gK  |plan heading at 4 s| in the ego frame (curvature proxy)
Oracle (not deployable): compensate only the tokens whose primary class is L (needs the scorer polygon).

  select   navtrain grid over mode x alpha x gate x fraction (official per-token v1 CSVs) -> gate_config.json
  navtest  gated / ungated / base official CSVs for native and best (prints MISSING <csv> if a compensated arm is not scored yet)
  navhard  in-process two-stage EPDMS incl. oracle L, paired CI over mapping groups; also gate-quality diagnostics
navsim2 env, CPU:  python nhdac_gate.py {select,navtest,navhard} [--procs 100]
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/skill_pack/scripts")]
import offroad_lib as L  # noqa: E402
import trk_report as T  # noqa: E402

RUN = L.D / "runs/skill_pack/trk"
O = L.D / "runs/leaderboard_audit/navhard_dac/gated"
OUT = REPO / "experiments/leaderboard_audit/results/navhard_dac"
MODES, ALPHAS, GATES, FRACS = ["full", "path"], [0.25, 0.5, 0.75, 1.0], ["gE", "gK"], [5, 10, 20, 35, 50, 100]
BEST = "gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base.npz"
P = L.D / "runs/op_lb"
BASE_NPZ = {("native", d): P / d / "preds/gimm-cinque__base.npz" for d in ("lb_navhard", "lb_navtest", "lb_navtrain")}
BASE_NPZ.update({("best", d): P / d / "preds" / BEST for d in ("lb_navhard", "lb_navtest")})
BEST_NAVTEST_CSV = "v1_navtest_opi_lb_navtest_gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base"


def gate_stats(data, model):
    """gE, gK per token (Series). gE from the trk diagnostics (uncompensated replay), gK from the plan."""
    f = (O / f"diag_{data}_full-path.pkl") if model == "best" else (RUN / f"diag_{data}_path.pkl")
    res = pickle.load(open(f, "rb"))
    gE = {t: r[model, "path"]["base"]["path_d_mean"] for t, r in res.items()}
    z = np.load(BASE_NPZ[model, data])
    h = z["poses"][:, -1, 2].astype(float)
    gK = dict(zip(z["tokens"].tolist(), np.abs((h + np.pi) % (2 * np.pi) - np.pi)))
    return pd.DataFrame(dict(gE=pd.Series(gE), gK=pd.Series(gK))).dropna()


def comp_csv(split, model, mode, al):
    if model == "native":
        return T.comp_csv("v1", split, "native", al, mode)
    return f"v1_{split}_gate_best_{mode}_a{al:g}"


def best_dir(d):
    return O / d


def cmd_select(a):
    base = T.v1_rows(T.BASE_CSV["native", "navtrain"])
    G = gate_stats("lb_navtrain", "native")
    rng = np.random.default_rng(0)
    rows, comp = [], {}
    for m in MODES:
        for al in ALPHAS:
            d = T.v1_rows(comp_csv("navtrain", "native", m, al))
            idx = base.index.intersection(d.index).intersection(G.index)
            comp[m, al] = d
            for g in GATES:
                for fr in FRACS:
                    thr = -np.inf if fr == 100 else float(np.quantile(G.loc[idx, g], 1 - fr / 100))
                    on = (G.loc[idx, g] >= thr).to_numpy()
                    ds = 100 * np.where(on, d.loc[idx, "score"] - base.loc[idx, "score"], 0.0)
                    rows.append(dict(mode=m, alpha=al, gate=g, frac=fr, thr=thr, delta=float(ds.mean()), n_on=int(on.sum()), n=len(idx), ds=ds, idx=idx))
    tab = pd.DataFrame(rows)
    out = {}
    for nm, pool in (("registered", rows), ("posthoc_gated", [r for r in rows if r["frac"] < 100])):
        best = max(pool, key=lambda r: (round(r["delta"], 9), -r["alpha"], -r["frac"]))
        B = rng.integers(0, best["n"], (5000, best["n"]))
        cfg = {k: v for k, v in best.items() if k not in ("ds", "idx")}
        cfg.update(ci=T.ci(best["ds"], B), base_pdms=100 * float(base.score.mean()), selected=bool(best["delta"] > 0))
        cfg["thr"] = None if cfg["thr"] == -np.inf else cfg["thr"]
        out[nm] = cfg
    O.mkdir(parents=True, exist_ok=True)
    (O / "gate_config.json").write_text(json.dumps(out, indent=1))
    cfg = out
    t = tab.drop(columns=["ds", "idx"])
    t.to_csv(OUT / "gated_navtrain_grid.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(t.pivot_table(index=["mode", "alpha"], columns=["gate", "frac"], values="delta").round(2).to_string())
    print(json.dumps(cfg, indent=1))


def load_cfg():
    return json.loads((O / "gate_config.json").read_text())


def fire(data, model, cfg):
    G = gate_stats(data, model)
    return G[cfg["gate"]] >= (-np.inf if cfg["thr"] is None else cfg["thr"])


def cmd_navtest(a):
    cfgs = load_cfg()
    out, rng, miss = {}, np.random.default_rng(0), []
    for cn, cfg in cfgs.items():
        m, al = cfg["mode"], cfg["alpha"]
        for model in ("native", "best"):
            base = T.v1_rows(T.BASE_CSV["native", "navtest"] if model == "native" else BEST_NAVTEST_CSV)
            comp = T.v1_rows(comp_csv("navtest", model, m, al))
            if comp is None:
                miss.append(f"{comp_csv('navtest', model, m, al)} <- {O}/lb_navtest/best_{m}_a{al:g}.npz")
                continue
            on = fire("lb_navtest", model, cfg)
            idx = base.index.intersection(comp.index).intersection(on.index)
            o = on.loc[idx].to_numpy()
            B = rng.integers(0, len(idx), (5000, len(idx)))
            r = dict(cfg=cfg, base_pdms=100 * float(base.loc[idx, "score"].mean()), n=len(idx), n_on=int(o.sum()))
            for nm, sel in (("gated", o), ("ungated", np.ones_like(o))):
                sc = np.where(sel, comp.loc[idx, "score"], base.loc[idx, "score"])
                x = 100 * (sc - base.loc[idx, "score"].to_numpy())
                r[nm] = dict(pdms=100 * float(sc.mean()), delta=float(x.mean()), ci=T.ci(x, B))
                for c in T.V1:
                    y = 100 * (np.where(sel, comp.loc[idx, c], base.loc[idx, c]) - base.loc[idx, c].to_numpy())
                    r[nm][f"{c}_delta"] = float(y.mean())
            out[f"{cn}/{model}"] = r
    if miss:
        print("MISSING", *sorted(set(miss)), sep="\n")
    (OUT / "gated_navtest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def auc(score, pos):
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_navhard(a):
    import offroad_replay_cf as R
    from offroad_gain import group_scores
    cfgs = load_cfg()
    fl = pd.read_csv(OUT / "nhdac_failures.csv")
    tokens = [e["token"] for e in L.index()]
    stage = {e["token"]: e["stage"] for e in L.index()}
    W = O / "mixed"
    W.mkdir(parents=True, exist_ok=True)
    arms, info, diag = {}, {}, {}

    def comp_npz(model, mode, alpha):
        return (RUN / f"lb_navhard/native_{mode}_a{alpha:g}.npz") if model == "native" else (O / f"lb_navhard/best_{mode}_a{alpha:g}.npz")

    def mix(name, model, mode, alpha, sel):
        b, c = np.load(BASE_NPZ[model, "lb_navhard"]), np.load(comp_npz(model, mode, alpha))
        assert (b["tokens"] == c["tokens"]).all()
        s = np.array([sel.get(t, False) for t in b["tokens"].tolist()])
        poses = np.where(s[:, None, None], c["poses"], b["poses"]).astype(np.float32)
        np.savez(W / f"{name.replace('/', '_')}.npz", tokens=b["tokens"], poses=poses)
        arms[name] = (W / f"{name.replace('/', '_')}.npz", "real")
        info[name] = dict(model=model, n_on=int(s.sum()), n_on_s1=int(sum(sel.get(t, False) for t in tokens if stage[t] == "one")),
                          n_on_s2=int(sum(sel.get(t, False) for t in tokens if stage[t] == "two")))

    for model in ("native", "best"):
        G = gate_stats("lb_navhard", model)
        Lset = set(fl[(fl.arm == model) & (fl.primary == "tracker_lag")].token)
        arms[f"{model}/base"] = (BASE_NPZ[model, "lb_navhard"], "real")
        diag[model] = {}
        for cn, cfg in cfgs.items():
            m, al = cfg["mode"], cfg["alpha"]
            on = fire("lb_navhard", model, cfg).to_dict()
            mix(f"{model}/{cn}_gated", model, m, al, on)
            diag[model][cn] = dict(gate_L_captured=float(np.mean([on.get(t, False) for t in Lset])),
                                   gate_s1=float(np.mean([on.get(t, False) for t in tokens if stage[t] == "one"])),
                                   gate_s2=float(np.mean([on.get(t, False) for t in tokens if stage[t] == "two"])))
        mix(f"{model}/ungated_path_a0.25", model, "path", 0.25, {t: True for t in tokens})
        for mm, aa in (("full", 1.0), ("path", 1.0), ("path", 0.25)):
            mix(f"{model}/oracleL_{mm}_a{aa:g}", model, mm, aa, {t: t in Lset for t in tokens})
        yL = np.array([t in Lset for t in G.index])
        diag[model].update({g: dict(auc_L=auc(G[g].to_numpy(), yL)) for g in GATES}, n_L=int(yL.sum()))
    rows = {}
    with mp.get_context("fork").Pool(a.procs, initializer=T._init_nh, initargs=(arms,)) as pool:
        for i, (t, r) in enumerate(pool.imap_unordered(T.work_nh, tokens, chunksize=4)):
            rows[t] = r
    pickle.dump(rows, open(O / "navhard_rows.pkl", "wb"), protocol=4)
    _, _, _, mapping, samp = L.setup_scoring()
    summ, grp, dfs = {}, {}, {}
    for v in arms:
        comb, s1, s2, df = R.aggregate([rows[t][v] for t in tokens], mapping, samp)
        dfs[v] = df.set_index("token")
        summ[v] = dict(combined=100 * float(comb["score"]), stage1=100 * float(s1["score"]), stage2=100 * float(s2["score"]), **info.get(v, {}))
        grp[v] = group_scores(df, mapping)
    off = {"native/base": T.load_csv(L.NATIVE_CSV)}
    for k, d in off.items():
        summ[k]["official_combined"] = 100 * float(d.loc["extended_pdm_score_combined", "score"])
    B = np.random.default_rng(0).integers(0, len(grp["native/base"]), (5000, len(grp["native/base"])))
    for v in arms:
        model, arm = v.split("/")
        if arm == "base":
            continue
        dg = 100 * (grp[v] - grp[f"{model}/base"])
        summ[v]["delta"], summ[v]["ci"] = float(dg.mean()), T.ci(dg, B)
        for st in ("one", "two"):
            tk = [t for t in tokens if stage[t] == st]
            summ[v][f"dac_{st}"] = 100 * float((dfs[v].loc[tk, "drivable_area_compliance"] - dfs[f"{model}/base"].loc[tk, "drivable_area_compliance"]).mean())
    (OUT / "gated_navhard.json").write_text(json.dumps(dict(cfg=cfgs, arms=summ, gate_diag=diag), indent=1))
    print(json.dumps(dict(arms=summ, gate_diag=diag), indent=1))


def cmd_capture(a):
    """Share of class-L tokens (and of all tokens) that each gate fires on, thresholds = navtrain quantiles."""
    fl = pd.read_csv(OUT / "nhdac_failures.csv")
    tr = gate_stats("lb_navtrain", "native")
    out = {}
    for model in ("native", "best"):
        G = gate_stats("lb_navhard", model)
        Lset = set(fl[(fl.arm == model) & (fl.primary == "tracker_lag")].token)
        y = G.index.isin(Lset)
        for g in GATES:
            for fr in FRACS[:-1]:
                on = (G[g] >= np.quantile(tr[g], 1 - fr / 100)).to_numpy()
                out[f"{model}/{g}/{fr}"] = dict(L_captured=float(on[y].mean()), tokens_fired=float(on.mean()), precision_L=float(y[on].mean()))
    (OUT / "gated_capture.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "navtest", "navhard", "capture"])
    ap.add_argument("--procs", type=int, default=100)
    a = ap.parse_args()
    {"select": cmd_select, "navtest": cmd_navtest, "navhard": cmd_navhard, "capture": cmd_capture}[a.cmd](a)
