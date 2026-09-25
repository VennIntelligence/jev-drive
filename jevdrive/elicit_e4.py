"""E4 of the elicitation program (todos/2026-09-26-elicitation-program.md, deviation-log entry [E4] 00:25): scoring
windows for the P5 v1 PDM-Lite set, where the expert reacts ~0.4 s after the factor becomes visible.

  (a) visibility gate  keep an observation frame only if k >= t_vis + L / TICK, L = the examinee's perception +
                       decision latency; tau_model unchanged (official run, all null frames), null false flips
                       recomputed on the null frames under the same gate (in-sample and route-half out-of-sample)
  (b) per pair         the whole observation window; a (base, seed) pair with >= 1 reactive frame passes when any of
                       them flips in the expert's direction; a null case false-flips when any of its frames moves

Every number comes from stored per-frame deltas (p5_exam / reactivity_mc / fusion_diag.q6gt run dirs); nothing is
refitted. Both expert sets are scored; the PDM-Lite set is the one the criterion reads.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import p5_exam as E
from .common import data_dir, get_logger
from .p5_pairs import TICK

log = get_logger(__name__)
CUTIN = ("HighwayCutIn", "StaticCutIn", "ParkingCutIn")
RUNS = {  # expert set -> (p5_exam D0 run, M-C run, Q6 GT run), relative to runs/
    "pdm": ("p5_pairs/exam-d0-carla_p5v1_pdm/20260925-230423", "reactivity/mc-carla_p5v1_pdm/20260925-230424",
            "fusion_diag/q6gt_carla_p5v1_pdm/20260925-233007"),
    "ba": ("p5_pairs/exam-d0-carla_p5v1_ba/20260925-233124", "reactivity/mc-carla_p5v1_ba/20260925-233126",
           "fusion_diag/q6gt_carla_p5v1_ba/20260925-233032"),
}
Q6_GATES = ("gate any", "gate pedestrian", "gate ttc", "gate intrusion")
SAM_PROXY = "gate any (L = 0.6 s)"     # descriptive: GT rule gate under SAM's latency


def latency(ex: str) -> float:
    """Perception + decision latency L (s) per examinee, as registered."""
    if ex.startswith("TFv6"):
        return 0.1
    if ex == "ridge ego" or ex in Q6_GATES:
        return 0.0
    if ex == SAM_PROXY:
        return 0.6
    if ex.startswith("ridge_late op-") or ex.startswith("prior [") or ex.startswith("M-C pair op"):
        return 0.1
    if ex.startswith("ridge_late L18") or ex.startswith("M-C "):     # Qwen stream, dual-stream arms
        return 0.3
    raise KeyError(ex)


def load(set_: str):
    """Scored pair and null frames of one expert set with every examinee as a column, taus, pairs, tau_exp."""
    R = data_dir() / "runs"
    d0, mc, q6 = (R / p for p in RUNS[set_])
    key = ["base_id", "family", "seed", "k", "fn_plus"]
    obs = pd.read_parquet(d0 / "obs_scored.parquet")
    null = pd.read_parquet(d0 / "null_scored.parquet")
    taus = dict(pd.read_json(d0 / "summary.json", typ="series").filter(like="tau ").rename(lambda s: s[4:]))
    tau_exp = float(pd.read_json(d0 / "summary.json", typ="series")["tau_exp"])
    pooled = list(pd.read_json(d0 / "summary.json", typ="series")["pooled_families"])
    mo, mn = pd.read_parquet(mc / "obs_scored.parquet"), pd.read_parquet(mc / "null_scored.parquet")
    arms = [c for c in mo.columns if c.startswith(("prior [", "M-C "))]
    fl = pd.read_csv(mc / "flip_rates.csv")
    taus |= fl[fl.scope == "pooled"].set_index("examinee").tau_model.loc[arms].to_dict()
    qo, qn = pd.read_parquet(q6 / "obs.parquet"), pd.read_parquet(q6 / "null.parquet")
    for df in (obs, null, mo, mn, qo, qn):
        df["base_id"] = df.base_id.astype(str)
    nkey = ["base_id", "family", "seed", "k", "fn_plus", "fn_null"]
    obs = obs.merge(mo[key + arms], on=key, how="left", validate="1:1").merge(qo[key + list(Q6_GATES)], on=key,
                                                                             how="left", validate="1:1")
    null = null.merge(mn[nkey + arms], on=nkey, how="left", validate="1:1").merge(qn[nkey + list(Q6_GATES)],
                                                                                 on=nkey, how="left", validate="1:1")
    for g in Q6_GATES:
        taus[g] = 0.0
    obs[SAM_PROXY], null[SAM_PROXY], taus[SAM_PROXY] = obs["gate any"], null["gate any"], 0.0
    assert obs[arms + list(Q6_GATES)].notna().all().all() and null[arms + list(Q6_GATES)].notna().all().all()
    # the official reactive label, unchanged (tau_exp of the official run)
    assert (obs.reactive == (np.abs(obs.d_expert) > tau_exp)).all()
    pairs = pd.read_csv(d0 / "pairs.csv", dtype={"base_id": str})
    tv = pairs.set_index(["base_id", "seed"]).t_vis
    obs["t_vis"] = tv.reindex(pd.MultiIndex.from_frame(obs[["base_id", "seed"]])).to_numpy()
    s0 = pairs[pairs.seed == 0].set_index("base_id")
    null["t_vis"] = s0.t_vis.fillna(s0.t_trig).reindex(null.base_id).to_numpy()   # p5_pairs' null-window origin
    assert obs.t_vis.notna().all() and null.t_vis.notna().all() and (obs.k >= obs.t_vis).all()
    examinees = list(E.TFV6) + [c for c in obs.columns if c.startswith(("ridge ego", "ridge_late"))] + arms + list(Q6_GATES) + [SAM_PROXY]
    return obs, null, taus, pooled, examinees


def _halves(null: pd.DataFrame):
    nb = np.array(sorted(null.base_id.unique()))
    perm = np.random.default_rng(0).permutation(nb)       # p5_exam.exam's split
    return set(perm[: len(nb) // 2]), set(perm[len(nb) // 2:])


def _null_ff(null, ex, tau, halves, per_case: bool):
    """In-sample and out-of-sample null false flips; per_case: a case counts once if any of its frames moves."""
    nv = null[["base_id", ex]].dropna()

    def rate(sub, t):
        m = E._moved(sub[ex], t)
        return float(m.groupby(sub.base_id).any().mean() if per_case else m.mean()) if len(sub) else np.nan

    oos = []
    for h in halves:
        a, b = nv[nv.base_id.isin(h)], nv[~nv.base_id.isin(h)]
        a_all = null[null.base_id.isin(h)][ex].dropna()         # tau from the half's *ungated* null, as officially
        if len(a_all) and len(b):
            oos.append(rate(b, np.quantile(np.abs(a_all), 0.95)))
    return rate(nv, tau), float(np.mean(oos)) if oos else np.nan


def score(obs, null, taus, pooled, examinees, halves_null) -> pd.DataFrame:
    """Rows: examinee x window (official per frame, (a) gated, (b) per pair) x scope."""
    scopes = {"pooled": obs.family.isin(pooled), "pedestrian": obs.family.isin(E.PED_FAMILIES),
              "cut-in": obs.family.isin(CUTIN)}
    halves = _halves(halves_null)
    rows = []
    for ex in examinees:
        tau, L = taus[ex], latency(ex)
        gate_o = obs.k >= obs.t_vis + round(L / TICK)
        gate_n = null.k >= null.t_vis + round(L / TICK)
        for window in ("per frame", "(a) gated", "(b) per pair"):
            o = obs[gate_o] if window == "(a) gated" else obs
            n = null[gate_n] if window == "(a) gated" else null
            ff_in, ff_oos = _null_ff(n, ex, tau, halves, window == "(b) per pair")
            for scope, m in scopes.items():
                s = o[m.loc[o.index] & o[ex].notna()]
                r = s[s.reactive]
                ok = ((np.sign(r[ex]) == np.sign(r.d_expert)) & E._moved(r[ex], tau)).astype(float)
                nr = s[~s.reactive]
                row = {"examinee": ex, "L_s": L, "window": window, "scope": scope, "tau_model": tau,
                       "null_ff": ff_in, "null_ff_oos": ff_oos,
                       "false_flip_nonreactive": float(E._moved(nr[ex], tau).mean()) if len(nr) else np.nan}
                if window == "(b) per pair":
                    pk = r.base_id + "/" + r.seed.astype(str)
                    g = ok.groupby(pk).max()
                    base = r.groupby(pk).base_id.first()
                    fr, lo, hi = E.boot_ratio(g.to_numpy(), np.ones(len(g)), base.loc[g.index].to_numpy()) \
                        if len(g) else (np.nan,) * 3
                    row |= {"n": len(g), "unit": "pair"}
                else:
                    fr, lo, hi = E.boot_ratio(ok.to_numpy(), np.ones(len(r)), r.base_id.to_numpy()) \
                        if len(r) else (np.nan,) * 3
                    row |= {"n": len(r), "unit": "frame"}
                rows.append(row | {"routes": r.base_id.nunique(), "flip": fr, "lo": lo, "hi": hi,
                                   # descriptive: direction agreement and size relative to tau, ignoring tau
                                   "sign_agree": float((np.sign(r[ex]) == np.sign(r.d_expert)).mean()) if len(r) else np.nan,
                                   "abs_over_tau_median": float((np.abs(r[ex]) / tau).median()) if len(r) and tau > 0 else np.nan})
    return pd.DataFrame(rows)


def lead_times(obs) -> pd.DataFrame:
    """Where the reactive frames sit relative to visibility (s), per family group."""
    r = obs[obs.reactive]
    dt = (r.k - r.t_vis) * TICK
    grp = np.where(r.family.isin(E.PED_FAMILIES), "pedestrian", np.where(r.family.isin(CUTIN), "cut-in", "other"))
    return pd.DataFrame({"group": grp, "dt": dt.to_numpy()}).groupby("group").dt.describe(
        percentiles=[.1, .25, .5, .75, .9]).reset_index()


def criterion(res: dict) -> pd.DataFrame:
    """Registered E4 rule on the PDM-Lite set: M-C dual-stream pedestrian flip under (a) within +-15 pp of the BA
    set's official per-frame number, and (a)'s out-of-sample null false flip <= 7%."""
    rows = []
    for m in ("cinque", "lebowski"):
        ex = f"M-C pair [{m}]"
        p, b = res["pdm"], res["ba"]
        a = p[(p.examinee == ex) & (p.window == "(a) gated") & (p.scope == "pedestrian")].iloc[0]
        ref = b[(b.examinee == ex) & (b.window == "per frame") & (b.scope == "pedestrian")].iloc[0]
        d = a.flip - ref.flip
        rows.append({"examinee": ex, "pdm_a_ped_flip": a.flip, "pdm_a_n": a.n, "ba_official_ped_flip": ref.flip,
                     "diff_pp": 100 * d, "pdm_a_null_ff_oos": a.null_ff_oos,
                     "pass": bool(abs(d) <= 0.15 and a.null_ff_oos <= 0.07)})
    return pd.DataFrame(rows)


def run(rl):
    res, lt = {}, []
    for s in ("pdm", "ba"):
        obs, null, taus, pooled, ex = load(s)
        rl.log.info("%s: %d pair frames (%d reactive), %d null frames, %d examinees, pooled %s", s, len(obs),
                    int(obs.reactive.sum()), len(null), len(ex), pooled)
        res[s] = score(obs, null, taus, pooled, ex, null)
        res[s].insert(0, "set", s)
        res[s].to_csv(rl.dir / f"e4_{s}.csv", index=False)
        lt.append(lead_times(obs).assign(set=s))
        # the per-frame window must reproduce the official numbers (sanity: same exam, same frames)
        chk = res[s][(res[s].window == "per frame") & (res[s].scope == "pooled")].set_index("examinee").flip
        R = data_dir() / "runs"
        off = pd.concat([pd.read_csv(R / RUNS[s][i] / f) for i, f in ((0, "flip_rates.csv"), (1, "flip_rates.csv"),
                                                                      (2, "q6_flip_rates_gt.csv"))])
        off = off[off.scope == "pooled"].drop_duplicates("examinee").set_index("examinee").flip_rate
        common = chk.index.intersection(off.index)
        dmax = float((chk[common] - off[common]).abs().max())
        rl.event("reproduce_official", set=s, n=len(common), max_abs_diff=dmax)
        rl.log.info("%s: per-frame pooled flips vs the official runs, %d examinees, max |diff| %.2e", s, len(common), dmax)
        assert dmax < 1e-9
    lt = pd.concat(lt)
    lt.to_csv(rl.dir / "reactive_lead_times.csv", index=False)
    crit = criterion(res)
    crit.to_csv(rl.dir / "criterion.csv", index=False)
    allr = pd.concat(res.values())
    allr.to_csv(rl.dir / "e4_all.csv", index=False)
    rl.log.info("reactive frames after visibility (s)\n%s", lt.to_markdown(index=False, floatfmt=".2f"))
    for s in res:
        t = res[s][res[s].scope.isin(["pooled", "pedestrian"])].pivot_table(
            index="examinee", columns=["scope", "window"], values="flip", sort=False)
        rl.log.info("%s flips\n%s", s, t.to_markdown(floatfmt=".3f"))
    rl.log.info("criterion\n%s", crit.to_markdown(index=False, floatfmt=".3f"))
    return allr, crit


def fig(run_dir: Path, out: Path):
    """Pedestrian and cut-in flip rate of the main examinees, PDM-Lite official / (a) / (b) next to BA official."""
    import matplotlib.pyplot as plt
    from .p4_carla import _style
    ps = _style()
    d = pd.read_csv(run_dir / "e4_all.csv")
    ex = ["TFv6 waypoint speed 2 s", "ridge_late op-cinque temporal", "ridge_late L18_last", "M-C pair [cinque]",
          "M-C hard [cinque]", "gate any"]
    lab = {"TFv6 waypoint speed 2 s": "TFv6 wp", "ridge_late op-cinque temporal": "openpilot",
           "ridge_late L18_last": "Qwen", "M-C pair [cinque]": "M-C pair", "M-C hard [cinque]": "M-C hard",
           "gate any": "Rule gate (GT)"}
    cols = [("ba", "per frame", "BA, per frame", ps.BASELINE), ("pdm", "per frame", "PDM-Lite, per frame", "#56B4E9"),
            ("pdm", "(a) gated", "PDM-Lite, (a) gated", "#0072B2"), ("pdm", "(b) per pair", "PDM-Lite, (b) per pair", "#D55E00")]
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.3), sharey=True)
    w = 0.8 / len(cols)
    for ax, scope, name in zip(axs, ("pedestrian", "cut-in"), ("(a) Pedestrian", "(b) Cut-in")):
        for j, (s, win, lb, c) in enumerate(cols):
            v = d[(d.set == s) & (d.window == win) & (d.scope == scope)].set_index("examinee").reindex(ex)
            x = np.arange(len(ex)) + (j - (len(cols) - 1) / 2) * w
            ax.bar(x, v.flip, w, color=c, label=lb, linewidth=0)
            ax.errorbar(x, v.flip, yerr=[v.flip - v.lo, v.hi - v.flip], fmt="none", ecolor="#333333", elinewidth=0.5,
                        capsize=1.0)
        ax.set_xticks(np.arange(len(ex)))
        ax.set_xticklabels([lab[e] for e in ex], rotation=30, ha="right", fontsize=6.5)
        ax.axhline(0.2, color=ps.BASELINE, linewidth=0.5, linestyle=":")
        ax.set_ylim(0, 1.0)
        ps.bars(ax)
        ps.panel(ax, name)
    axs[0].set_ylabel("Directional flip rate")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(pad=0.3)
    ps.save(fig, out / "elicit-e4-windows")
    plt.close(fig)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "fig"])
    ap.add_argument("--run-dir", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        rl = RunLog("elicitation", "e4")
        run(rl)
        rl.close()
    else:
        out = data_dir() / "runs" / "elicitation" / "figs"
        out.mkdir(parents=True, exist_ok=True)
        fig(Path(a.run_dir), out)


if __name__ == "__main__":
    main()
