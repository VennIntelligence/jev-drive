"""E4c of the elicitation program (todos/2026-09-26-elicitation-program.md, deviation-log entry [E4c] 01:11):
reaction-latency curves on both P5 v1 sets and the human onset anchor on WOD-E2E rater frames.

  curve   per pair (>= 1 reactive frame), cumulative share of pairs with a directional flip (|Delta_model| >=
          tau_model, sign = the pair's expert direction) on any observation frame with t - t_vis <= x,
          x = 0 .. 10 s in 0.2 s steps; null cases give the floor; area = mean of C(x) over x in [L, 10]
  onset   first-flip time against each expert's onset (first reactive frame), pairs matched across the two sets
  human   WOD rater frames of Cut_ins / Pedestrian / Cyclist: first time the speed profile of the log and of
          rater_best drops 0.5 m/s below the constant-acceleration extrapolation from the past

Stored per-frame deltas only (elicit_e4.load); nothing is refitted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import elicit_e4 as E4, p5_exam as E
from .common import data_dir, get_logger
from .p5_pairs import TICK

log = get_logger(__name__)
GRID = np.round(np.arange(0, 10.0 + 1e-9, 0.2), 1)
N_BOOT = 2000
CLUSTERS = ("Cut_ins", "Pedestrian", "Cyclist")
DROP = 0.5                                   # m/s below the extrapolation


def _cummin_time(df: pd.DataFrame, key: str, hit: np.ndarray) -> pd.Series:
    """Earliest t (s after visibility) with hit per key; inf when never."""
    t = np.where(hit, df.dt.to_numpy(), np.inf)
    return pd.Series(t, index=df.index).groupby(df[key]).min()


def _curve(first: np.ndarray) -> np.ndarray:
    """(n_units,) first-hit times -> (len(GRID),) cumulative share."""
    return (first[:, None] <= GRID[None, :] + 1e-9).mean(0) if len(first) else np.full(len(GRID), np.nan)


def _area(c: np.ndarray, lo: float, hi: float = 10.0) -> float:
    m = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
    return float(c[m].mean())


def curves(obs, null, taus, pooled, examinees, rl, set_):
    """Per examinee x scope: curve rows, area rows, per-pair first flips (for the onset comparison)."""
    obs = obs.assign(pk=obs.base_id + "/" + obs.seed.astype(str), dt=(obs.k - obs.t_vis) * TICK)
    null = null.assign(dt=(null.k - null.t_vis) * TICK)
    scopes = {"pooled": obs.family.isin(pooled), "pedestrian": obs.family.isin(E.PED_FAMILIES),
              "cut-in": obs.family.isin(E4.CUTIN)}
    r = obs[obs.reactive]
    direction = np.sign(r.groupby("pk").d_expert.sum())
    onset = r.groupby("pk").dt.min()
    # null direction: the seed-0 pair of the same base, braking (-1) otherwise
    d0 = {pk.split("/")[0]: d for pk, d in direction.items() if pk.endswith("/0")}
    null_dir = null.base_id.map(d0).fillna(-1.0).to_numpy()
    rng = np.random.default_rng(0)
    crv, area, first_rows = [], [], []
    for scope, msk in scopes.items():
        o = obs[msk & obs.pk.isin(direction.index)]
        units = np.array(sorted(o.pk.unique()))
        if not len(units):
            continue
        base_of = pd.Series([u.split("/")[0] for u in units], index=units)
        n_fam = null[null.family.isin(obs[msk].family.unique())]
        nd = null_dir[np.flatnonzero(null.family.isin(obs[msk].family.unique()))]
        nbases = np.array(sorted(n_fam.base_id.unique()))
        bases = np.array(sorted(set(base_of) | set(nbases)))
        draws = rng.integers(len(bases), size=(N_BOOT, len(bases)))
        M = np.stack([np.bincount(dr, minlength=len(bases)) for dr in draws]).astype(float)   # base multiplicities
        bpos = pd.Series(np.arange(len(bases)), index=bases)
        for ex in examinees:
            tau, L = taus[ex], E4.latency(ex)
            ok = o[ex].notna()
            oo = o[ok]
            d = direction.reindex(oo.pk).to_numpy()
            moved = E._moved(oo[ex], tau).to_numpy()
            hit = moved & (np.sign(oo[ex].to_numpy()) == d)
            f_all = _cummin_time(oo, "pk", hit).reindex(units).fillna(np.inf)
            f_rea = _cummin_time(oo, "pk", hit & oo.reactive.to_numpy()).reindex(units).fillna(np.inf)
            nn = n_fam[n_fam[ex].notna()]
            ndir = nd[n_fam[ex].notna().to_numpy()]
            nhit = E._moved(nn[ex], tau).to_numpy() & (np.sign(nn[ex].to_numpy()) == ndir)
            f_null = _cummin_time(nn, "base_id", nhit).reindex(nbases).fillna(np.inf)
            c_all, c_rea, c_null = _curve(f_all.to_numpy()), _curve(f_rea.to_numpy()), _curve(f_null.to_numpy())
            for x, a, b, c in zip(GRID, c_all, c_rea, c_null):
                crv.append({"set": set_, "examinee": ex, "scope": scope, "x": x, "all_frames": a,
                            "reactive_only": b, "null": c})
            # bootstrap areas: resample bases; pairs and null cases follow their base (vectorised as base counts)
            def per_base(first_t, bidx):
                H = np.zeros((len(bases), len(GRID)))
                np.add.at(H, bidx, (first_t[:, None] <= GRID[None, :] + 1e-9).astype(float))
                return H, np.bincount(bidx, minlength=len(bases)).astype(float)
            Hp, npb = per_base(f_all.to_numpy(), bpos[base_of.to_numpy()].to_numpy())
            Hn, nnb = per_base(f_null.to_numpy(), bpos[nbases].to_numpy())
            with np.errstate(invalid="ignore", divide="ignore"):
                cp, cn = (M @ Hp) / (M @ npb)[:, None], (M @ Hn) / (M @ nnb)[:, None]
            good = np.isfinite(cp[:, 0]) & np.isfinite(cn[:, 0])
            cp, cn = cp[good], cn[good]
            mL = GRID >= L - 1e-9
            m3 = mL & (GRID <= 3.0 + 1e-9)            # A3: co-reported readout, decision [决定] 01:48
            boot = {"A": cp[:, mL].mean(1), "A_null": cn[:, mL].mean(1), "A3": cp[:, m3].mean(1),
                    "A3_null": cn[:, m3].mean(1)}
            A, An = np.array(boot["A"]), np.array(boot["A_null"])
            q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
            area.append({"set": set_, "examinee": ex, "scope": scope, "L_s": L, "pairs": len(units),
                         "null_cases": len(nbases),
                         "area_L_10": _area(c_all, L), "area_lo": q(A)[0], "area_hi": q(A)[1],
                         "null_area_L_10": _area(c_null, L), "null_lo": q(An)[0], "null_hi": q(An)[1],
                         "area_minus_null": _area(c_all, L) - _area(c_null, L),
                         "diff_lo": q(A - An)[0], "diff_hi": q(A - An)[1],
                         "area_L_3_descriptive": _area(c_all, L, 3.0) if L <= 3 else np.nan,
                         "A3": _area(c_all, L, 3.0), "A3_lo": q(boot["A3"])[0], "A3_hi": q(boot["A3"])[1],
                         "A3_null": _area(c_null, L, 3.0), "A3_null_lo": q(boot["A3_null"])[0],
                         "A3_null_hi": q(boot["A3_null"])[1],
                         "A3_minus_null": _area(c_all, L, 3.0) - _area(c_null, L, 3.0),
                         "A3_diff_lo": q(boot["A3"] - boot["A3_null"])[0], "A3_diff_hi": q(boot["A3"] - boot["A3_null"])[1],
                         "area_reactive_only_L_10": _area(c_rea, L), "end_all": c_all[-1], "end_reactive_only": c_rea[-1],
                         "null_end": c_null[-1]})
            first_rows.append(pd.DataFrame({"set": set_, "examinee": ex, "scope": scope, "pk": units,
                                            "t_first_flip": f_all.to_numpy(), "onset_own": onset.reindex(units).to_numpy()}))
        rl.log.info("%s %s: %d pairs, %d null cases", set_, scope, len(units), len(nbases))
    return pd.DataFrame(crv), pd.DataFrame(area), pd.concat(first_rows), onset


def onset_compare(first: pd.DataFrame, onsets: dict) -> pd.DataFrame:
    """Median first-flip minus each expert's onset, on the pairs where the examinee flips."""
    f = first[np.isfinite(first.t_first_flip)].copy()
    rows = []
    for e, on in onsets.items():
        f[f"onset_{e}"] = on.reindex(f.pk).to_numpy()
    for (s, ex, sc), g in f.groupby(["set", "examinee", "scope"], sort=False):
        row = {"set": s, "examinee": ex, "scope": sc, "pairs_flipped": len(g),
               "t_first_flip_median": float(g.t_first_flip.median())}
        for e in onsets:
            dlt = (g.t_first_flip - g[f"onset_{e}"]).dropna()
            row |= {f"n_{e}": len(dlt), f"flip_minus_onset_{e}_median": float(dlt.median()) if len(dlt) else np.nan,
                    f"share_before_onset_{e}": float((dlt < 0).mean()) if len(dlt) else np.nan}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- WOD human onset

def human_onset():
    """Onset of the log and of rater_best on the Cut_ins / Pedestrian / Cyclist rater frames (s after the frame)."""
    from . import waymo
    D = data_dir() / "processed" / "waymo_e2e"
    idx = pd.read_parquet(D / "index.parquet", columns=["cluster", "split"])
    r = pd.read_parquet(D / "rater.parquet")
    r = r[np.isin(idx.cluster.to_numpy()[r.row.to_numpy()], CLUSTERS)]
    best = r.sort_values(["row", "score", "traj"], ascending=[True, False, True]).groupby("row").head(1)
    rows = best.row.to_numpy()
    past = np.load(D / "past.npy", mmap_mode="r")[np.sort(rows)]
    fut = np.load(D / "future.npy", mmap_mode="r")[np.sort(rows)][..., :2]
    best = best.set_index("row").loc[np.sort(rows)]
    rb = np.stack([np.concatenate([np.stack([x, y], -1)[:20], np.repeat(np.stack([x, y], -1)[-1:], max(0, 20 - len(x)), 0)])
                   for x, y in zip(best.pos_x, best.pos_y)]).astype(np.float64)
    kin = waymo.past_kinematics(np.asarray(past, np.float64), k=4)
    tm = (np.arange(20) + 0.5) * waymo.DT                               # interval midpoints 0.125 .. 4.875 s
    vc = np.maximum(kin["v"][:, None] + kin["a"][:, None] * tm[None, :], 0)
    out = pd.DataFrame({"row": np.sort(rows), "cluster": idx.cluster.to_numpy()[np.sort(rows)],
                        "v0": kin["v"], "a0": kin["a"], "rater_best_score": best.score.to_numpy()})
    for name, tr in (("log", np.asarray(fut, np.float64)), ("rater_best", rb)):
        pts = np.concatenate([np.zeros((len(tr), 1, 2)), tr], 1)
        v = np.linalg.norm(np.diff(pts, axis=1), axis=-1) / waymo.DT
        below = v < vc - DROP
        out[f"onset_{name}"] = np.where(below.any(1), tm[below.argmax(1)], np.nan)
        # post-hoc (deviation log [E4c] 01:14): the rater trajectories' first waypoint sits at ~0.18 s, not 0.25 s
        # (first-interval speed 0.71 v0 against 0.98-1.0 v0 afterwards), so the registered search puts a spurious
        # onset at 0.125 s; this variant starts the search at the second interval for both trajectories
        b2 = below[:, 1:]
        out[f"onset_{name}_from2"] = np.where(b2.any(1), tm[1:][b2.argmax(1)], np.nan)
    return out


def human_summary(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cl, g in [("all three", h)] + [(c, h[h.cluster == c]) for c in CLUSTERS]:
        for name in ("log", "rater_best", "log_from2", "rater_best_from2"):
            v = g[f"onset_{name}"].dropna()
            rows.append({"cluster": cl, "trajectory": name, "frames": len(g), "with_onset": len(v),
                         "share_with_onset": len(v) / len(g) if len(g) else np.nan,
                         **{f"p{int(q * 100)}": float(np.quantile(v, q)) if len(v) else np.nan
                            for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                         "share_onset_1_3s": float(((v >= 1) & (v <= 3)).mean()) if len(v) else np.nan})
        for suf in ("", "_from2"):
            both = g.dropna(subset=[f"onset_log{suf}", f"onset_rater_best{suf}"])
            rows.append({"cluster": cl, "trajectory": f"rater_best{suf} - log{suf} (paired)", "frames": len(g),
                         "with_onset": len(both),
                         "p50": float((both[f"onset_rater_best{suf}"] - both[f"onset_log{suf}"]).median()) if len(both) else np.nan})
    return pd.DataFrame(rows)


def run(rl):
    crv, area, first, onsets = [], [], [], {}
    for s in ("ba", "pdm"):
        obs, null, taus, pooled, ex = E4.load(s)
        c, a, f, on = curves(obs, null, taus, pooled, ex, rl, s)
        crv.append(c), area.append(a), first.append(f)
        onsets[s] = on
    crv, area, first = pd.concat(crv), pd.concat(area), pd.concat(first)
    oc = onset_compare(first, onsets)
    eo = pd.concat([pd.DataFrame({"expert": e, "pk": v.index, "onset": v.to_numpy()}) for e, v in onsets.items()])
    scope_of = first[first.scope != "pooled"].drop_duplicates(["set", "pk"])[["set", "pk", "scope"]]
    eo = eo.merge(scope_of.rename(columns={"set": "expert"}), on=["expert", "pk"], how="left").fillna({"scope": "other"})
    eo_sum = eo.groupby(["expert", "scope"]).onset.describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()
    h = human_onset()
    hs = human_summary(h)
    d = rl.dir
    crv.to_csv(d / "curves.csv", index=False)
    area.to_csv(d / "areas.csv", index=False)
    oc.to_csv(d / "first_flip_vs_onset.csv", index=False)
    eo.to_csv(d / "expert_onsets.csv", index=False)
    eo_sum.to_csv(d / "expert_onset_summary.csv", index=False)
    h.to_csv(d / "human_onset_frames.csv", index=False)
    hs.to_csv(d / "human_onset_summary.csv", index=False)
    rl.log.info("expert onsets (s after visibility)\n%s", eo_sum.to_markdown(index=False, floatfmt=".2f"))
    rl.log.info("human onset (s after the rater frame)\n%s", hs.to_markdown(index=False, floatfmt=".2f"))
    rl.log.info("areas\n%s", area[area.scope != "cut-in"][["set", "examinee", "scope", "L_s", "pairs", "area_L_10", "area_lo",
                                                          "area_hi", "null_area_L_10", "area_minus_null", "diff_lo",
                                                          "diff_hi", "area_L_3_descriptive", "end_all"]]
                .to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("first flip vs onset\n%s", oc.to_markdown(index=False, floatfmt=".2f"))


# ---------------------------------------------------------------- figure

def fig(run_dir: Path, out: Path):
    """(a,b) cumulative directional flip curves for the main examinees on the pedestrian pairs of both sets, with
    the null floor and the expert onset CDF; (c) human onset distribution on WOD."""
    import matplotlib.pyplot as plt
    from .p4_carla import _style
    ps = _style()
    c = pd.read_csv(run_dir / "curves.csv")
    eo = pd.read_csv(run_dir / "expert_onsets.csv")
    h = pd.read_csv(run_dir / "human_onset_frames.csv")
    main = {"M-C pair [cinque]": ("M-C pair", "#D55E00"), "ridge_late op-cinque temporal": ("openpilot", "#0072B2"),
            "TFv6 waypoint speed 2 s": ("TFv6 wp", "#009E73"), "gate any": ("Rule gate (GT)", "#CC79A7")}
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.2))
    for ax, s, name in ((axs[0], "ba", "(a) BehaviorAgent, pedestrian"), (axs[1], "pdm", "(b) PDM-Lite, pedestrian")):
        for ex, (lab, col) in main.items():
            g = c[(c.set == s) & (c.examinee == ex) & (c.scope == "pedestrian")]
            ax.plot(g.x, g.all_frames, color=col, label=lab, linewidth=1.0)
            if ex == "M-C pair [cinque]":
                ax.plot(g.x, g.null, color=col, linestyle=":", linewidth=0.8, label="M-C null floor")
        v = np.sort(eo[(eo.expert == s) & (eo.scope == "pedestrian")].onset.to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=ps.BASELINE, linewidth=0.8, label="Expert onset CDF")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$t - t_\mathrm{vis}$ (s)")
        ps.panel(ax, name)
    axs[0].set_ylabel("Share of pairs flipped")
    axs[0].legend(fontsize=5.5, loc="upper left")
    ax = axs[2]
    for name, col in (("log", "#0072B2"), ("rater_best", "#D55E00")):
        v = np.sort(h[f"onset_{name}_from2"].dropna().to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(h), where="post", color=col, label=name.replace("_", " "))
    ax.axvspan(1, 3, color=ps.BASELINE, alpha=0.15, linewidth=0)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Onset after rater frame (s)")   # post-hoc from2 variant, see the deviation log
    ax.set_ylabel("Share of frames")
    ax.legend(fontsize=6, loc="upper left")
    ps.panel(ax, "(c) WOD human onset")
    fig.tight_layout(pad=0.3)
    ps.save(fig, out / "elicit-e4c-latency")
    plt.close(fig)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "human", "fig"])
    ap.add_argument("--run-dir", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        rl = RunLog("elicitation", "e4c")
        run(rl)
        rl.close()
    elif a.cmd == "human":                                  # rerun only the WOD part into an existing run dir
        h = human_onset()
        h.to_csv(Path(a.run_dir) / "human_onset_frames.csv", index=False)
        human_summary(h).to_csv(Path(a.run_dir) / "human_onset_summary.csv", index=False)
        log.info("human onset\n%s", human_summary(h).to_markdown(index=False, floatfmt=".2f"))
    else:
        out = data_dir() / "runs" / "elicitation" / "figs"
        out.mkdir(parents=True, exist_ok=True)
        fig(Path(a.run_dir), out)


if __name__ == "__main__":
    main()
