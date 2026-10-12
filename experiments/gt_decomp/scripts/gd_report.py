"""GT-DECOMP read (plans/2026-10-12-gt-decomp-prereg.md): tables and figures from gd.py collect. Mac, project venv; called as gd.py report.

Inputs: --pub / --held = copies of gt_decomp collect dirs (roll.parquet, step.parquet, gates.json); --cap-pub / --cap-held = cap_audit
collect copies (scene.parquet: v0, dpsi_log, s_onc for the groups); --tags = cap_audit results/tags.csv (box only, not in git).
CIs: log-clustered bootstrap (ca_report.Boot: default_rng(0), 10 000 draws of logs, ratio of sums), one draw for every driver of a set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/cap_audit/scripts"), str(REPO / "research")]
import ca_report as CR  # noqa: E402

OUT, FIGS = REPO / "experiments/gt_decomp/results", REPO / "experiments/gt_decomp/figs"
OURS = {"YR10m10-F-s0": ["YR10m10-F-s0"], "YS2G": ["YS2G-L18-s0", "YS2G-L18-s1"], "YS-L18": ["YS-L18-s0", "YS-L18-s1"]}
REFS = ["WA-JEPA", "LTF", "DD", "G0014"]
ROWS = list(OURS) + ["APY10m10-AB-s0"] + REFS
NAME = {"DD": "DiffusionDrive", "G0014": "garage 0014", "YR10m10-F-s0": "YR10m10-F-s0 (#2)", "APY10m10-AB-s0": "APY10m10-AB-s0 (#1)"}
PRIVATE_GT, PRIVATE_LABEL = 1.733, "YR10m10-F-s0"


def per_scene(d):
    """-> one row per (label, scene): rollout values + per-scene means of the per-step split on kept steps, with seed-mean labels."""
    R = pd.read_parquet(Path(d) / "roll.parquet")
    S = pd.read_parquet(Path(d) / "step.parquet")
    k = S[S.kept].copy()
    k["aalong"], k["across"], k["a2"], k["c2"] = k.along.abs(), k.cross.abs(), k.along ** 2, k.cross ** 2
    g = k.groupby(["label", "scene"])
    M = g[["along", "aalong", "across", "a2", "c2", "e_geo"]].mean()
    last = k.loc[g.k.idxmax()].set_index(["label", "scene"])
    M["along_last"] = last.along
    R = R.set_index(["label", "scene"]).join(M).reset_index()
    R["score"] = R.score.astype(float)
    R["gt_over_d"] = R.gt_cf - R["gt"]
    R["gt_over_dlb"] = R.gt_cf_lb - R["gt"]
    R["gt2"], R["lat2"] = R["gt"] ** 2, R.gt_lat ** 2
    R["nz"] = (R.score > 0).astype(float)
    R["prog_short"] = ((R.score > 0) & (R.score < 1 - 1e-9)).astype(float)
    R["prog_cost"] = np.where(R.score > 0, 1 - R.score, 0.0)
    R["prog_cost_all"] = 1 - R.prog_score                                                  # progress shortfall incl. zero scenes
    R["clip4"] = (R["gt"] >= 4.0).astype(float)
    num = [c for c in R.columns if c not in ("label", "scene", "run") and pd.api.types.is_numeric_dtype(R[c])]
    extra = []
    for name, labs in OURS.items():
        if len(labs) > 1:
            x = R[R.label.isin(labs)].groupby("scene")[num].mean().reset_index()
            x["label"] = name
            extra.append(x)
    return pd.concat([R] + extra, ignore_index=True), S


def groups_of(cap, tags):
    Sc = pd.read_parquet(Path(cap) / "scene.parquet")
    T = pd.read_csv(tags)
    Sc, G = CR.groups(Sc, T)
    return Sc, G


def drv_table(P, logs, boot, cols, members=None):
    rows = []
    for lab in ROWS:
        x = P[P.label == lab].set_index("scene")
        if members is not None:
            x = x.reindex(sorted(members & set(x.index)))
        r = dict(driver=NAME.get(lab, lab), n=len(x))
        for c in cols:
            pt, bd, _, _ = boot.draws(x[c], logs.reindex(x.index))
            lo, hi = CR.ci(bd)
            r[c], r[c + "_lo"], r[c + "_hi"] = pt, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def paired(P, logs, boot, cols, members=None, ours=OURS):
    rows = []
    W = {lab: x.set_index("scene") for lab, x in P.groupby("label")}
    for o in ours:
        for r in REFS:
            sc = sorted(set(W[o].index) & set(W[r].index) & (members if members is not None else set(W[o].index)))
            row = dict(ours=o, ref=NAME.get(r, r), n=len(sc))
            for c in cols:
                dx = W[o][c].reindex(sc) - W[r][c].reindex(sc)
                pt, bd, _, nl = boot.draws(dx, logs.reindex(sc))
                lo, hi = CR.ci(bd)
                row[c], row[c + "_lo"], row[c + "_hi"] = pt, lo, hi
            row["logs"] = nl
            rows.append(row)
    return pd.DataFrame(rows)


def f3(v, lo=None, hi=None, p=3):
    if v is None or not np.isfinite(v):
        return "-"
    s = f"{v:+.{p}f}" if lo is not None and (lo < 0 < hi or lo < 0 or v < 0) else f"{v:.{p}f}"
    return s + (f" [{lo:+.{p}f}, {hi:+.{p}f}]" if lo is not None and np.isfinite(lo) else "")


def md(df, cols, heads, ci=()):
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join([":--"] + ["--:"] * (len(heads) - 1)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ci:
                cells.append(f3(v, r[c + "_lo"], r[c + "_hi"]))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def read_set(st, d, cap, tags):
    P, S = per_scene(d)
    Sc, G = groups_of(cap, tags)
    logs = Sc.set_index("scene").log
    P = P[P.scene.isin(logs.index)]
    boot = CR.Boot(logs.unique())
    return P, S, Sc, G, logs, boot


def lag_curve(S, K):
    k = S[S.kept]
    rows = []
    for lab in ROWS:
        labs = OURS.get(lab, [lab])
        x = k[k.label.isin(labs)].groupby("k")
        rows.append(pd.DataFrame(dict(label=lab, k=x.along.mean().index, along=x.along.mean().values, across=x.cross.apply(lambda v: v.abs().mean()).values,
                                      n=x.size().values)))
    return pd.concat(rows, ignore_index=True)


def figs(Pp, Lc):
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    col = {"YR10m10-F-s0": ps.PALETTE["blue"], "YS2G": ps.PALETTE["green"], "YS-L18": ps.PALETTE["yellow"], "APY10m10-AB-s0": ps.PALETTE["black"],
           "WA-JEPA": ps.PALETTE["purple"], "LTF": ps.PALETTE["orange"], "DD": ps.PALETTE["vermillion"], "G0014": ps.PALETTE["sky_blue"]}
    short = {"YR10m10-F-s0": "YR (#2)", "YS2G": "YS2G", "YS-L18": "YS-L18", "APY10m10-AB-s0": "APY (#1)", "WA-JEPA": "WA-JEPA", "LTF": "LTF",
             "DD": "DiffDrive", "G0014": "garage"}
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.3))
    x = np.arange(len(ROWS))
    m = Pp.groupby("label")[["gt", "gt_lat", "gt_over", "a2", "c2"]].mean().reindex(ROWS)
    ax = axs[0]
    ax.bar(x, m.gt_lat, color=ps.PALETTE["blue"], width=0.65, label="cross-track (lateral to logged path)")
    ax.bar(x, m.gt_over, bottom=m.gt_lat, color=ps.PALETTE["orange"], width=0.65, label="along-track (past the path end)")
    ax.set_xticks(x, [short[r] for r in ROWS], rotation=35, ha="right")
    ax.set_ylabel("leaderboard GT distance (m)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, 2.5)
    ps.bars(ax)
    ps.panel(ax, "(a)")
    ax = axs[1]
    ax.bar(x, m.c2, color=ps.PALETTE["blue"], width=0.65, label="cross-track$^2$")
    ax.bar(x, m.a2, bottom=m.c2, color=ps.PALETTE["orange"], width=0.65, label="along-track$^2$")
    ax.set_xticks(x, [short[r] for r in ROWS], rotation=35, ha="right")
    ax.set_ylabel("time-aligned error$^2$ per decision (m$^2$)")
    ax.legend(loc="upper left", fontsize=7)
    ps.bars(ax)
    ps.panel(ax, "(b)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.92, wspace=0.28)
    s1 = ps.save(fig, FIGS / "gd_split")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.3))
    for lab in ROWS:
        c = Lc[Lc.label == lab]
        ax.plot(c.k * 0.5, c.along, color=col[lab], lw=1.6 if lab in OURS else 0.9, ls="-" if lab in OURS or lab in REFS else "--", label=short[lab],
                marker="o" if lab in OURS else None, ms=2)
    ps.zero_line(ax)
    ax.set_xlabel("time since start (s)")
    ax.set_ylabel("along-track vs human (m, + ahead)")
    ax.set_ylim(-3.2, 1.9)
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.6)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.97)
    s2 = ps.save(fig, FIGS / "gd_lag")
    plt.close(fig)
    return s1, s2


def report(a):
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    L = []
    w = L.append
    J = {}
    for st, d, cap in (("pub", a.pub, a.cap_pub), ("held", a.held, a.cap_held)):
        P, S, Sc, G, logs, boot = read_set(st, d, cap, a.tags)
        gates = json.loads((Path(d) / "gates.json").read_text())
        K = 10 if st == "pub" else 18
        J[st] = dict(gates_passed=gates["passed"], g1a_run_max=gates["G1a_run"]["max_abs"],
                     g1a_min=min(min(v[m] for m in ("gt", "lat", "loc", "prog", "score")) for v in gates["G1a_rollout"]["per_label"].values()),
                     g1b_min=min(min(v["d"], v["e"]) for v in gates["G1b_geometry"]["per_label"].values()), gt_geo_match=gates["gt_geo_match"])
        cA = ["gt", "gt_lat", "gt_over", "over_any", "loc", "clip4"]
        TA = drv_table(P, logs, boot, cA)
        sh = P.groupby("label").apply(lambda x: 1 - x.lat2.sum() / x.gt2.sum() if x.gt2.sum() else np.nan).reindex(ROWS)
        TA["over_sq_share"] = sh.values
        TA["over_share"] = (TA.gt_over / TA["gt"]).values
        cB = ["e_geo", "along", "aalong", "across", "along_last"]
        TB = drv_table(P, logs, boot, cB)
        k = S[S.kept]
        TB["along_sq_share"] = [float((lambda x: (x.along ** 2).sum() / ((x.along ** 2).sum() + (x.cross ** 2).sum()))(k[k.label.isin(OURS.get(r, [r]))]))
                                for r in ROWS]
        PA = paired(P, logs, boot, ["gt", "gt_lat", "gt_over", "aalong", "along", "across", "along_last"])
        PA["over_of_gt_excess"] = PA.gt_over / PA["gt"]
        CF = drv_table(P, logs, boot, ["gt", "gt_cf", "gt_over_d", "gt_over_dlb"])
        CF["cf_ext_share"] = [float(P[P.label == r].cf_ext.sum() / P[P.label == r].cf_n.sum()) for r in ROWS]
        PCF = paired(P, logs, boot, ["gt_cf"])
        PR = drv_table(P, logs, boot, ["score", "nz", "prog_short", "prog_cost", "prog_cost_all"])
        PR["n_short"] = [int(P[P.label == r].prog_short.sum()) if r not in OURS or len(OURS[r]) == 1 else float(P[P.label == r].prog_short.sum()) for r in ROWS]
        PPR = paired(P, logs, boot, ["prog_cost", "score"])
        gr = []
        for gname, mem in G.items():
            if gname == "all" or len(mem) < 30:
                continue
            t = drv_table(P, logs, boot, ["gt", "gt_over", "along_last", "prog_cost"], members=mem).set_index("driver")
            p = paired(P, logs, boot, ["gt", "gt_over", "along_last", "prog_cost"], members=mem, ours={"YR10m10-F-s0": 0, "YS2G": 0})
            for o in ("YR10m10-F-s0", "YS2G"):
                q = p[p.ours == o].set_index("ref")
                gr.append(dict(group=gname, ours=o, n=int(q.n.min()), logs=int(q.logs.min()), gt=t.loc[NAME.get(o, o), "gt"],
                               gt_refs=f"{t.loc[[NAME.get(r, r) for r in REFS], 'gt'].min():.2f}-{t.loc[[NAME.get(r, r) for r in REFS], 'gt'].max():.2f}",
                               d_gt_min=q["gt"].min(), d_gt_max=q["gt"].max(), d_over_max=q.gt_over.abs().max(),
                               lag=t.loc[NAME.get(o, o), "along_last"],
                               lag_refs=f"{t.loc[[NAME.get(r, r) for r in REFS], 'along_last'].min():+.2f} to {t.loc[[NAME.get(r, r) for r in REFS], 'along_last'].max():+.2f}",
                               d_lag_lo=q.along_last.max(), cost=t.loc[NAME.get(o, o), "prog_cost"], d_cost_min=q.prog_cost.min(), d_cost_max=q.prog_cost.max()))
        GR = pd.DataFrame(gr)
        Lc = lag_curve(S, K)
        for nm, df in (("drivers", TA), ("timealigned", TB), ("paired", PA), ("cf", CF), ("cf_paired", PCF), ("progress", PR), ("progress_paired", PPR),
                       ("groups", GR), ("lag", Lc)):
            df.to_csv(OUT / f"{st}_{nm}.csv", index=False)
        # arithmetic (c), our submission on this set
        y = P[P.label == PRIVATE_LABEL]
        lo, hi = y["gt"][y["gt"] < 4].mean(), y["gt"][y["gt"] >= 4].mean()
        J[st] |= dict(yr_gt=float(y["gt"].mean()), yr_lo=float(lo), yr_hi=float(hi), yr_clip4=float((y["gt"] >= 4).mean()),
                      need_clip4=float((PRIVATE_GT - lo) / (hi - lo)), yr_over_any=int(y.over_any.sum()), n=len(y),
                      yr_lat=float(y.gt_lat.mean()), yr_over=float(y.gt_over.mean()), yr_over_per=float(y.gt_over[y.over_any > 0].mean()))
        if st == "pub":
            Pp, Lp = P, Lc
        # ---- markdown
        tag = "10-decision public (1479 scenes, 136 logs)" if st == "pub" else "18-decision held (onpol-held@v1, 452 scenes)"
        w(f"## {tag}\n")
        w(f"Gates: G1a rollout re-aggregation min share within 1e-3 m = {J[st]['g1a_min']:.4f} (line 0.999); run-level max |diff| = "
          f"{J[st]['g1a_run_max']:.1e} m; G1b geometry min share = {J[st]['g1b_min']:.4f} (line 0.99); rollout GT from geometry = evaluator "
          f"{J[st]['gt_geo_match']:.4f}; passed = {J[st]['gates_passed']}.\n")
        w("### A. Leaderboard GT distance, split at the arg-max step (m; mean over scenes [95% CI, logs])\n")
        w(md(TA, ["driver", "n", "gt", "gt_lat", "gt_over", "over_share", "over_sq_share", "over_any", "clip4", "loc"],
             ["driver", "n", "GT (leaderboard)", "cross-track part", "along-track part (past path end)", "along share", "along share of GT^2",
              "share of scenes with any past-end step", "share >= 4 m", "time-aligned max (dist_to_gt_location)"], ci=("gt", "gt_lat", "gt_over")))
        w("\nalong-track part = dist_to_gt_trajectory - lateral_dist_to_gt_trajectory at the arg-max step: non-zero only when the ego has driven past the "
          "end of the logged path. Being behind the human contributes exactly 0 to the leaderboard GT.\n")
        w("### B. Time-aligned error per decision, in the human path frame (kept steps; per-scene mean, then mean over scenes)\n")
        w(md(TB, ["driver", "n", "e_geo", "along", "aalong", "across", "along_last", "along_sq_share"],
             ["driver", "n", "time-aligned error", "along (+ ahead)", "abs along", "abs cross", "along at last kept step", "along share of error^2"],
             ci=("e_geo", "along", "aalong", "across", "along_last")))
        w("\n### Paired, ours minus each reference (per scene; log-clustered CI)\n")
        w(md(PA, ["ours", "ref", "n", "gt", "gt_lat", "gt_over", "over_of_gt_excess", "along_last", "across"],
             ["ours", "ref", "n", "GT", "cross part", "along part", "along part / GT excess", "along at last step", "abs cross (time-aligned)"],
             ci=("gt", "gt_lat", "gt_over", "along_last", "across")))
        w("\n### Counterfactual (a)/(b): same path, the human's timing (m)\n")
        w(md(CF, ["driver", "gt", "gt_cf", "gt_over_d", "gt_over_dlb", "cf_ext_share"],
             ["driver", "GT actual", "GT, human timing", "change", "change, reached stations only (lower bound)", "share of steps extrapolated"],
             ci=("gt_over_d", "gt_over_dlb")))
        w("\nPaired GT under human timing, ours minus reference:\n")
        w(md(PCF, ["ours", "ref", "n", "gt_cf"], ["ours", "ref", "n", "GT difference, human timing"], ci=("gt_cf",)))
        w("\n### Progress in the scene score\n")
        w(md(PR, ["driver", "n", "score", "nz", "n_short", "prog_short", "prog_cost", "prog_cost_all"],
             ["driver", "n", "mean score", "non-zero share", "non-zero scenes with score < 1", "share", "mean score lost to progress (non-zero scenes) / N",
              "progress shortfall incl. zeros"], ci=("score", "prog_cost")))
        w("\nPaired, ours minus reference:\n")
        w(md(PPR, ["ours", "ref", "n", "prog_cost", "score"], ["ours", "ref", "n", "progress cost", "score"], ci=("prog_cost", "score")))
        w("\n### By scenario group (ours vs the four references; d = ours minus reference, range over the four)\n")
        w(md(GR, ["group", "ours", "n", "gt", "gt_refs", "d_gt_min", "d_gt_max", "d_over_max", "lag", "lag_refs", "cost", "d_cost_min", "d_cost_max"],
             ["group", "ours", "n", "GT", "GT refs", "d GT min", "d GT max", "max abs d along part", "along at last step", "refs", "progress cost",
              "d cost min", "d cost max"]))
        w("\n### Along-track lag by decision (m, + ahead of the human; kept steps)\n")
        pv = Lc.pivot(index="k", columns="label", values="along")[ROWS]
        w(md(pv.reset_index().rename(columns=lambda c: NAME.get(c, c)), ["k"] + [NAME.get(c, c) for c in ROWS], ["k"] + [NAME.get(c, c) for c in ROWS]))
        # ---- exploratory (not pre-registered): where the along-track (past-end) part comes from
        Lg = pd.read_parquet(Path(cap) / "log.parquet")
        vend = Lg[Lg.k == K].set_index("scene").v
        Lpath = pd.read_parquet(Path(d) / "pathinfo.parquet").set_index("scene").L
        v0 = Sc.set_index("scene").v0
        ex = []
        for lab in ROWS:
            x = P[P.label == lab].set_index("scene")
            ov = x.over_any > 0.25                                                          # seed-mean rows: past the end in either seed
            n = len(x)
            ex.append(dict(driver=NAME.get(lab, lab), n_over=int(ov.sum()), share_over=float(ov.mean()),
                           cf_over=float(x.gt_over_d[ov].sum() / n), cf_rest=float(x.gt_over_d[~ov].sum() / n), cf_rest_mean=float(x.gt_over_d[~ov].mean()),
                           ge4=int((x["gt"] >= 4).sum()), ge4_over_only=int(((x["gt"] >= 4) & (x.gt_lat < 4)).sum()),
                           gt_over_on_over=float(x.gt_over[ov].mean()), score_over=float(x.score[ov].mean()), score_rest=float(x.score[~ov].mean()),
                           v0_over=float(v0.reindex(x.index)[ov].median()), vend_over=float(vend.reindex(x.index)[ov].median()),
                           vend_rest=float(vend.reindex(x.index)[~ov].median()), L_over=float(Lpath.reindex(x.index)[ov].median()),
                           L_rest=float(Lpath.reindex(x.index)[~ov].median()), lead_over=float(x.along_last[ov].median())))
        EX = pd.DataFrame(ex)
        EX.to_csv(OUT / f"{st}_exploratory.csv", index=False)
        J[st]["exploratory"] = EX.set_index("driver").to_dict("index")
        w("\n### Exploratory (not pre-registered): where the along-track part comes from\n")
        w(md(EX, ["driver", "n_over", "share_over", "gt_over_on_over", "cf_over", "cf_rest", "ge4", "ge4_over_only", "score_over", "score_rest", "lead_over",
                  "v0_over", "vend_over", "vend_rest", "L_over", "L_rest"],
             ["driver", "scenes past the path end", "share", "along part on those", "CF change from those (per all scenes)", "CF change from the rest",
              "clips >= 4 m", "of which cross < 4 m", "score on past-end scenes", "score on the rest", "median along at last step there (m)",
              "median human v0 there", "median human v at end there", "rest", "median logged path (m) there", "rest"]))
        w("\nPast-end scenes are scenes where the driver ends ahead of the human (median along at the last step above). The human-speed and logged-path "
          "columns say which human behaviour they come from on this set. 'CF change from the rest' = what the human timing does to GT on every other scene "
          "(this is where cruise lag lives).\n")
        w(f"\n### Arithmetic (c), {PRIVATE_LABEL}\n")
        w(f"Local GT {J[st]['yr_gt']:.3f} m; clips < 4 m average {lo:.3f} m, clips >= 4 m average {hi:.3f} m, {100 * J[st]['yr_clip4']:.1f}% of clips >= 4 m. "
          f"Leaderboard private {PRIVATE_GT} m with the same per-clip structure needs {100 * J[st]['need_clip4']:.1f}% of clips to leave the 4 m corridor "
          f"(arithmetic only). Lagging the human cannot add to this metric (nearest-point distance); {J[st]['yr_over_any']} of {J[st]['n']} clips have "
          "any step past the end of the logged path. "
          f"Along-track route (arithmetic only): with the cross part fixed at {J[st]['yr_lat']:.3f} m, the along (past-end) part would have to grow from "
          f"{J[st]['yr_over']:.3f} to {PRIVATE_GT - J[st]['yr_lat']:.3f} m, i.e. {(PRIVATE_GT - J[st]['yr_lat']) / J[st]['yr_over_per']:.0%} of clips "
          f"past the end at the local {J[st]['yr_over_per']:.2f} m per such clip (local {J[st]['yr_over_any'] / J[st]['n']:.1%}).\n")
    s1, s2 = figs(Pp, Lp)
    J["figs"] = dict(split=s1, lag=s2)
    head = ["# GT-DECOMP: the leaderboard GT distance, along-track vs cross-track", "",
            "Pre-registration: [../plans/2026-10-12-gt-decomp-prereg.md](../plans/2026-10-12-gt-decomp-prereg.md). Code: `scripts/gd.py` (collect, box), "
            "`scripts/gd_report.py` (this file). Box: `$DATA_DIR/runs/gt_decomp/collect-{pub,held}/`. CSVs: `results/{pub,held}_*.csv`. Figures: "
            "`figs/gd_split.png`, `figs/gd_lag.png`.", "",
            "Definition (AlpaSim 0bb4c4b `src/eval/src/eval/scorers/ground_truth.py` 59-60, 106-108): per step, distance from the ego box centroid to the "
            "nearest point of the whole logged path (not time-aligned); MAX over the kept steps; mean over clips. See the plan, section 1.", ""]
    (OUT / "gt_decomp.md").write_text("\n".join(head + L) + "\n")
    (OUT / "gd.json").write_text(json.dumps(J, indent=1, default=float))
    print(json.dumps(J, indent=1, default=float))
