"""P5 v0 exam on the CARLA counterfactual pairs (todos/2026-09-24-p5-carla-pairs-v0.md).

  heads   `ridge ego` and `ridge_late` (L18_last / L18_mean) trained on CARLA expert frames outside every pair's
          observation window, 5 folds grouped by base route; per-frame predicted speed at 2 s on every
          observation frame (pair and null, both worlds)
  probes  linear probes for the factor (hazard present-and-approaching, red vs green) on the same features, same
          folds, trained on non-observation P5 frames; AUC of x+ against x- on the pair frames
  exam    label validity, directional flip rates, false flips, route-bootstrap CIs; tables into the run dir
  figs    flip rate per family per examinee; expert delta against the null

The frames, labels and TFv6 readouts come from jevdrive/p5_pairs.py (index -> processed/carla_p5).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import p5_pairs as P
from .common import get_logger

log = get_logger(__name__)
K_FOLDS = 5
TAU_EXP_MIN = 0.5                     # m/s; pre-registered floor under the null's 95th percentile
MIN_PAIRS_WITH_REACTIVE = 5           # a family enters the pooled numbers only with this many
N_BOOT = 2000
LAMS_PROBE = (1e-4, 1e-3, 1e-2, 1e-1)
TFV6 = {"TFv6 target speed": "ts", "TFv6 waypoint speed 2 s": "wp2", "TFv6 target speed (decoded scalar)": "ts_scalar"}
HEADS = ("ridge ego", "ridge_late L18_last", "ridge_late L18_mean")
PRIMARY = ("TFv6 target speed", "ridge_late L18_last")


def load():
    d = P.processed()
    t = pd.read_parquet(d / "index.parquet")
    past, fut = np.load(d / "past.npy"), np.load(d / "future.npy")
    obs = pd.read_parquet(d / "obs.parquet")
    null = pd.read_parquet(d / "null.parquet")
    pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str})
    return t, past, fut, obs, null, pairs


def folds(t: pd.DataFrame, pairs: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Fold per row: P5 base routes are dealt into K folds; a P4 route with the same id goes with it; every other
    P4 route is always training (-1)."""
    bases = np.array(sorted(pairs.base_id.unique()))
    f = dict(zip(np.random.default_rng(seed).permutation(bases), np.arange(len(bases)) % K_FOLDS))
    return t.base_id.map(f).fillna(-1).astype(int).to_numpy()


def ego_input(t: pd.DataFrame, past: np.ndarray) -> np.ndarray:
    from . import waymo
    return np.concatenate([waymo.ego_state(past), np.eye(len(waymo.INTENTS), dtype=np.float32)[t.intent.to_numpy()]],
                          1).astype(np.float32)


def heads(t, past, fut, X, fold, rl) -> dict:
    """Out-of-fold predictions (n, 20, 2) on the observation rows for each head; NaN elsewhere."""
    from . import planner, waymo_stage_a as sa
    n = len(t)
    F = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    E = torch.as_tensor(ego_input(t, past), device="cuda")
    Xs = {k: torch.as_tensor(v, device="cuda") for k, v in X.items()}
    seq = t.base_id.to_numpy()
    role = t.role.to_numpy()
    out = {h: np.full((n, 20, 2), np.nan, np.float32) for h in HEADS}
    for f in range(K_FOLDS):
        tr = np.flatnonzero((role == "train") & (fold != f))
        ev = np.flatnonzero((role == "obs") & (fold == f))
        if not len(ev):
            continue
        sp = SimpleNamespace(train=tr, val=ev, seq=seq)
        Xe = planner.standardize(E, tr)
        pv, st, We = sa.ridge_cv(Xe, F, sp, fut)
        out["ridge ego"][ev] = pv[:, 0]
        base = planner.linear_apply(We, Xe, np.arange(n))[0]
        R = F - base
        res = R.reshape(n, 20, 2).cpu().numpy()
        rl.log.info("fold %d: %d train rows (%d routes), %d obs rows; ridge ego lambda %g", f, len(tr),
                    len(np.unique(seq[tr])), len(ev), st["lam"])
        for tap, Xt in Xs.items():
            Xi = planner.standardize(Xt, tr)
            pr, stv, _ = sa.ridge_cv(Xi, R, sp, res)
            out[f"ridge_late {tap}"][ev] = pr[:, 0] + base[ev].reshape(-1, 20, 2).cpu().numpy()
            rl.log.info("fold %d ridge_late %s: lambda %g", f, tap, stv["lam"])
            rl.event("head_fold", fold=f, tap=tap, lam=stv["lam"], lam_ego=st["lam"], n_train=len(tr), n_obs=len(ev))
            del Xi
        torch.cuda.empty_cache()
    return out


def probes(t, X, fold, rl) -> tuple[dict, pd.DataFrame]:
    """Out-of-fold probe scores on the observation rows, and each probe's AUC on held-out training rows."""
    from sklearn.model_selection import GroupShuffleSplit
    from .p4_carla import _std, auc, logreg
    n = len(t)
    role, seq = t.role.to_numpy(), t.base_id.to_numpy()
    scores, rows = {}, []
    for task in ("hazard", "light"):
        y_all = t[task].to_numpy(dtype=float)
        for tap, Xa in X.items():
            Xt = torch.as_tensor(Xa, device="cuda")
            s = np.full(n, np.nan)
            for f in range(K_FOLDS):
                tr = np.flatnonzero((role == "train") & (fold != f) & ~np.isnan(y_all))
                ho = np.flatnonzero((role == "train") & (fold == f) & ~np.isnan(y_all))
                ev = np.flatnonzero((role == "obs") & (fold == f))
                if len(np.unique(y_all[tr])) < 2 or not len(ev):
                    continue
                mu, sd = _std(Xt, tr)
                Z = (Xt - mu) / sd
                y = torch.as_tensor(np.nan_to_num(y_all).astype(np.int64), device="cuda")
                sel = []
                if len(np.unique(seq[tr])) >= 5:
                    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tr, groups=seq[tr]))
                    fi, si = tr[a], tr[b]
                    if len(np.unique(y_all[fi])) == 2 and len(np.unique(y_all[si])) == 2:
                        for lam in LAMS_PROBE:
                            Wb = logreg(Z[fi], y[fi], lam)
                            sel.append(auc(y_all[si], (Z[si] @ Wb[0] + Wb[1]).softmax(1)[:, 1].cpu().numpy()))
                lam = LAMS_PROBE[int(np.argmax(sel))] if sel else 1e-2
                Wb = logreg(Z[tr], y[tr], lam)
                pr = (Z @ Wb[0] + Wb[1]).softmax(1)[:, 1].cpu().numpy()
                s[ev] = pr[ev]
                r = {"probe": task, "tap": tap, "fold": f, "lam": lam, "n_train": len(tr),
                     "pos_train": int(y_all[tr].sum()), "n_heldout": len(ho),
                     "auc_heldout": auc(y_all[ho], pr[ho]) if len(np.unique(y_all[ho])) == 2 else np.nan}
                rows.append(r)
                rl.event("probe_fold", **r)
            scores[(task, tap)] = s
            del Xt
            torch.cuda.empty_cache()
    return scores, pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics

def boot_ratio(num: np.ndarray, den: np.ndarray, groups: np.ndarray, b: int = N_BOOT, seed: int = 0):
    """sum(num) / sum(den) and its percentile CI, resampling whole groups (base routes)."""
    codes, uniq = pd.factorize(groups)
    sn, sd = np.bincount(codes, num, len(uniq)), np.bincount(codes, den, len(uniq))
    idx = np.random.default_rng(seed).integers(len(uniq), size=(b, len(uniq)))
    tot = sd[idx].sum(1)
    r = sn[idx].sum(1)[tot > 0] / tot[tot > 0]
    point = float(sn.sum() / sd.sum()) if sd.sum() else np.nan
    return point, float(np.quantile(r, 0.025)) if len(r) else np.nan, float(np.quantile(r, 0.975)) if len(r) else np.nan


def deltas(obs: pd.DataFrame, null: pd.DataFrame, t: pd.DataFrame, preds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delta_model per examinee on pair frames (plus - minus) and null frames (plus - null)."""
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    obs, null = obs.copy(), null.copy()
    for name, col in TFV6.items():
        obs[name] = obs[f"tf_{col}_plus"] - obs[f"tf_{col}_minus"]
        null[name] = null[f"tf_{col}_plus"] - null[f"tf_{col}_null"]
    for h, pr in preds.items():
        v = P.v2(pr)
        obs[h] = v[pos[obs.fn_plus].to_numpy()] - v[pos[obs.fn_minus].to_numpy()]
        null[h] = v[pos[null.fn_plus].to_numpy()] - v[pos[null.fn_null].to_numpy()]
    return obs, null


def _moved(x, tau):
    """|Delta_model| at or above the examinee's noise floor, and not exactly zero (`ridge ego` has tau = 0)."""
    a = np.abs(x)
    return (a >= tau) & (a > 0)


def exam(obs: pd.DataFrame, null: pd.DataFrame, pairs: pd.DataFrame, examinees) -> dict:
    """Every pre-registered number: label validity, tau, flip rates, false flips, per family and pooled."""
    tau_exp = max(float(np.quantile(np.abs(null.d_expert), 0.95)) if len(null) else 0.0, TAU_EXP_MIN)
    obs = obs.assign(reactive=np.abs(obs.d_expert) > tau_exp, dstop=obs.stop_plus.astype(int) - obs.stop_minus.astype(int))
    pair_key = obs.base_id + "/" + obs.seed.astype(str)
    reactive_pairs = obs[obs.reactive].assign(pk=pair_key[obs.reactive]).groupby("family").pk.nunique()
    fams = sorted(obs.family.unique())
    pooled_fams = [f for f in fams if reactive_pairs.get(f, 0) >= MIN_PAIRS_WITH_REACTIVE]

    validity = []
    for fam in list(pairs.family.drop_duplicates().sort_values()):
        pp = pairs[pairs.family == fam]
        o = obs[obs.family == fam]
        validity.append({"family": fam, "pairs": len(pp), "runs_ok": int((pp.reason != "missing_run").sum()),
                         "deterministic_to_visibility": int((pp.reason == "ok").sum()),
                         **{f"drop_{r}": int((pp.reason == r).sum()) for r in
                            ("never_visible", "background_drift", "expert_reacted_before_visible", "missing_run")},
                         "obs_frames": len(o), "reactive_frames": int(o.reactive.sum()),
                         "nonreactive_frames": int((~o.reactive).sum()),
                         "pairs_with_reactive": int(reactive_pairs.get(fam, 0)),
                         "stop_flip_frames": int((o.dstop != 0).sum()),
                         "d_expert_median": float(o.d_expert.median()) if len(o) else np.nan,
                         "d_expert_reactive_median": float(o.d_expert[o.reactive].median()) if o.reactive.any() else np.nan,
                         "impure_frames": int((o.impure_visible > 0).sum()), "in_pooled": fam in pooled_fams})
    validity = pd.DataFrame(validity)

    taus, rows = {}, []
    halves = {}
    nb = np.array(sorted(null.base_id.unique()))
    perm = np.random.default_rng(0).permutation(nb)
    halves = {0: set(perm[: len(nb) // 2]), 1: set(perm[len(nb) // 2:])}
    for ex in examinees:
        nv = null[ex].dropna()
        tau = float(np.quantile(np.abs(nv), 0.95)) if len(nv) else np.nan
        taus[ex] = tau
        oos = []
        for h in (0, 1):
            a = null[null.base_id.isin(halves[h])][ex].dropna()
            b = null[~null.base_id.isin(halves[h])][ex].dropna()
            if len(a) and len(b):
                oos.append(float(_moved(b, np.quantile(np.abs(a), 0.95)).mean()))
        for scope, sub in [("pooled", obs[obs.family.isin(pooled_fams)])] + [(f, obs[obs.family == f]) for f in fams]:
            s = sub[sub[ex].notna()]
            r = s[s.reactive]
            flip = ((np.sign(r[ex]) == np.sign(r.d_expert)) & _moved(r[ex], tau)).astype(float)
            anyflip = _moved(r[ex], tau).astype(float)
            nr = s[~s.reactive]
            ff = _moved(nr[ex], tau).astype(float)
            fr, lo, hi = boot_ratio(flip.to_numpy(), np.ones(len(r)), r.base_id.to_numpy()) if len(r) else (np.nan,) * 3
            row = {"examinee": ex, "scope": scope, "tau_model": tau, "n_reactive": len(r),
                   "routes_reactive": r.base_id.nunique(), "pairs_reactive": (r.base_id + "/" + r.seed.astype(str)).nunique(),
                   "flip_rate": fr, "flip_lo": lo, "flip_hi": hi,
                   "any_direction_rate": float(anyflip.mean()) if len(r) else np.nan,
                   "wrong_direction_rate": float(((np.sign(r[ex]) != np.sign(r.d_expert)) & _moved(r[ex], tau)).mean())
                   if len(r) else np.nan,
                   "n_nonreactive": len(nr), "false_flip_nonreactive": float(ff.mean()) if len(nr) else np.nan}
            if scope == "pooled":
                n_all = null[ex].dropna()
                row.update(false_flip_null_insample=float(_moved(n_all, tau).mean()) if len(n_all) else np.nan,
                           false_flip_null_oos=float(np.mean(oos)) if oos else np.nan, n_null=len(n_all))
                # family-equal pooled flip rate
                fam_rates = [((np.sign(g[ex]) == np.sign(g.d_expert)) & _moved(g[ex], tau)).mean()
                             for _, g in r.groupby("family")]
                row["flip_rate_family_equal"] = float(np.mean(fam_rates)) if fam_rates else np.nan
            rows.append(row)
    return {"tau_exp": tau_exp, "validity": validity, "flips": pd.DataFrame(rows), "taus": taus,
            "pooled_families": pooled_fams, "obs": obs}


def probe_auc(obs: pd.DataFrame, t: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """AUC of separating x+ from x- pair frames with the factor probe (hazard families: hazard probe; Light: light)."""
    from .p4_carla import auc
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    rows = []
    for (task, tap), s in scores.items():
        for scope in ["pooled"] + sorted(obs.family.unique()):
            fam_ok = obs.family == "Light" if task == "light" else obs.family != "Light"
            o = obs[fam_ok] if scope == "pooled" else obs[(obs.family == scope) & fam_ok]
            if not len(o):
                continue
            sp, sm = s[pos[o.fn_plus].to_numpy()], s[pos[o.fn_minus].to_numpy()]
            ok = ~np.isnan(sp) & ~np.isnan(sm)
            if ok.sum() < 5:
                continue
            y = np.r_[np.ones(ok.sum()), np.zeros(ok.sum())]
            sc = np.r_[sp[ok], sm[ok]]
            g = np.r_[o.base_id.to_numpy()[ok], o.base_id.to_numpy()[ok]]
            a = auc(y, sc)
            # route bootstrap of the AUC
            codes, uniq = pd.factorize(g)
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(500):
                pick = rng.integers(len(uniq), size=len(uniq))
                m = np.concatenate([np.flatnonzero(codes == c) for c in pick])
                if len(np.unique(y[m])) == 2:
                    bs.append(auc(y[m], sc[m]))
            rows.append({"probe": task, "tap": tap, "scope": scope, "n_frames": int(ok.sum()), "routes": len(uniq),
                         "auc_plus_vs_minus": a, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                         "mean_score_plus": float(sp[ok].mean()), "mean_score_minus": float(sm[ok].mean())})
    return pd.DataFrame(rows)


def run(rl):
    t, past, fut, obs, null, pairs = load()
    X = P.load_features(t)
    fold = folds(t, pairs)
    rl.log.info("%d frames (%s), %d pair frames, %d null frames, %d cases", len(t),
                t.groupby(["source", "role"]).size().to_dict(), len(obs), len(null), len(pairs))
    preds = heads(t, past, fut, X, fold, rl)
    scores, probe_folds = probes(t, X, fold, rl)
    o, n = deltas(obs, null, t, preds)
    ex = list(TFV6) + list(HEADS)
    res = exam(o, n, pairs, ex)
    pa = probe_auc(res["obs"], t, scores)
    d = rl.dir
    res["validity"].to_csv(d / "label_validity.csv", index=False)
    res["flips"].to_csv(d / "flip_rates.csv", index=False)
    pa.to_csv(d / "probe_auc.csv", index=False)
    probe_folds.to_csv(d / "probe_folds.csv", index=False)
    res["obs"].to_parquet(d / "obs_scored.parquet", index=False)
    n.to_parquet(d / "null_scored.parquet", index=False)
    pairs.to_csv(d / "pairs.csv", index=False)
    pd.Series({"tau_exp": res["tau_exp"], "pooled_families": res["pooled_families"], **{f"tau {k}": v for k, v in res["taus"].items()}}
              ).to_json(d / "summary.json")
    rl.log.info("tau_exp %.3f m/s; pooled families %s", res["tau_exp"], res["pooled_families"])
    rl.log.info("label validity\n%s", res["validity"].to_markdown(index=False))
    rl.log.info("flip rates (pooled)\n%s", res["flips"][res["flips"].scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("probe AUC (pooled)\n%s", pa[pa.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    return res, pa


# ---------------------------------------------------------------- figures

COLORS = {"TFv6 target speed": "#0072B2", "TFv6 waypoint speed 2 s": "#56B4E9", "ridge_late L18_last": "#D55E00",
          "ridge_late L18_mean": "#E69F00"}
SHORT = {"PedestrianCrossing": "PedCrossing", "DynamicObjectCrossing": "DynObjCrossing",
         "VehicleTurningRoutePedestrian": "TurnPedestrian", "ParkingCrossingPedestrian": "ParkingPedestrian",
         "OppositeVehicleRunningRedLight": "OppRedLightRunner", "HardBreakRoute": "HardBrake", "StaticCutIn": "StaticCutIn",
         "ParkingCutIn": "ParkingCutIn", "HighwayCutIn": "HighwayCutIn", "Light": "Red vs green", "pooled": "Pooled"}


def figs(run, out):
    """Two figures from a finished exam run dir, into `out` (research/figs)."""
    import matplotlib.pyplot as plt
    from .p4_carla import _style
    ps = _style()
    fl = pd.read_csv(run / "flip_rates.csv")
    val = pd.read_csv(run / "label_validity.csv")
    obs = pd.read_parquet(run / "obs_scored.parquet")
    null = pd.read_parquet(run / "null_scored.parquet")
    tau = pd.read_json(run / "summary.json", typ="series")["tau_exp"]

    # (1) directional flip rate per family and examinee, with route-bootstrap CIs
    ex = [e for e in COLORS if e in set(fl.examinee)]
    scopes = ["pooled"] + [f for f in val.family if f in set(fl.scope)]
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 2.7))
    w = 0.8 / len(ex)
    for j, e in enumerate(ex):
        d = fl[fl.examinee == e].set_index("scope").reindex(scopes)
        x = np.arange(len(scopes)) + (j - (len(ex) - 1) / 2) * w
        ok = d.n_reactive.fillna(0).to_numpy() > 0
        ax.bar(x[ok], d.flip_rate[ok], w, color=COLORS[e], label=e, linewidth=0)
        ax.errorbar(x[ok], d.flip_rate[ok], yerr=[d.flip_rate[ok] - d.flip_lo[ok], d.flip_hi[ok] - d.flip_rate[ok]],
                    fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2)
    n = fl[fl.examinee == ex[0]].set_index("scope").reindex(scopes).n_reactive.fillna(0).astype(int)
    for j, e in enumerate(ex):                                  # a zero bar is a result, not a missing one
        d = fl[fl.examinee == e].set_index("scope").reindex(scopes)
        x = np.arange(len(scopes)) + (j - (len(ex) - 1) / 2) * w
        for xi, fr, k in zip(x, d.flip_rate, d.n_reactive.fillna(0)):
            if k > 0 and fr == 0:
                ax.plot(xi, 0.012, marker="_", color=COLORS[e], markersize=4, mew=1.5)
    ax.set_xticks(np.arange(len(scopes)))
    ax.set_xticklabels([f"{SHORT.get(s, s)} ($n$={k})" for s, k in zip(scopes, n)], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Directional flip rate")
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.axhline(0.5, color=ps.BASELINE, linewidth=0.5, linestyle="--")
    ax.axhline(0.2, color=ps.BASELINE, linewidth=0.5, linestyle=":")
    ps.bars(ax)
    ax.legend(ncol=len(ex), loc="upper center", fontsize=7)
    fig.tight_layout(pad=0.3)
    ps.save(fig, out / "p5-flip-rates")
    plt.close(fig)

    # (2) expert delta on pair frames against the weather-only null
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.3), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axs[0]
    for vals, lab, c in ((np.abs(obs.d_expert), "pair frames", "#D55E00"), (np.abs(null.d_expert), "null frames", ps.BASELINE)):
        v = np.sort(vals.to_numpy())
        if len(v):
            ax.step(v, 1 - np.arange(len(v)) / len(v), where="post", color=c, label=f"{lab} ($n$={len(v)})")
    ax.axvline(tau, color="#333333", linewidth=0.6, linestyle="--")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(0, 10)
    ax.set_xlabel(r"$|\Delta_\mathrm{expert}|$ = |speed at 2 s, x$^+$ $-$ x$^-$| (m/s)")
    ax.set_ylabel("Fraction of frames $\\geq$ x")
    ax.legend(fontsize=7)
    ps.panel(ax, "(a)")
    ax = axs[1]
    fams = [f for f in val.family if (obs.family == f).any()]
    rng = np.random.default_rng(0)
    for i, f in enumerate(fams):
        v = obs.d_expert[obs.family == f].to_numpy()
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.25, 0.25, len(v)), v, s=2, color="#D55E00", alpha=0.35,
                   linewidths=0)
    ax.axhspan(-tau, tau, color=ps.BASELINE, alpha=0.15, linewidth=0)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([SHORT.get(f, f) for f in fams], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel(r"$\Delta_\mathrm{expert}$ (m/s)")
    ps.zero_line(ax)
    ps.panel(ax, "(b)")
    fig.tight_layout(pad=0.3)
    ps.save(fig, out / "p5-expert-delta")
    plt.close(fig)


def main():
    import argparse
    from .runlog import RunLog
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "figs"])
    p.add_argument("--run-dir", default="")
    a = p.parse_args()
    if a.cmd == "run":
        rl = RunLog("p5_pairs", "exam")
        run(rl)
        rl.close()
    else:
        from pathlib import Path
        from .common import data_dir
        out = data_dir() / "runs" / "p5_pairs" / "figs"
        out.mkdir(parents=True, exist_ok=True)
        figs(Path(a.run_dir), out)


if __name__ == "__main__":
    main()
