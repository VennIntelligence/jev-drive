"""spec_plan run: P2 / P2+hinge under the HUGSIM `spec_plan` preset (lateral curvature from the model's own plan) vs the stored `spec` and `exam` runs.

  extract  (box, envs/hugsim)  per-run behaviour of the 12 (arm, preset) sets on the 64 scenarios -> results/hugsim_specplan/extract.csv,
                               traces.json.gz (per-step heading rate, speed, requested curvature)
  report   (Mac)               tables.md, the CSVs behind them, figures figs/*.png
Harness, scoring and classes as results/hugsim_full.md (spin = heading error >= 60 deg vs the recorded route; class = spin, else the runner's end).
Oscillation: heading rate w_i = (theta_{i+1} - theta_i) in deg per 0.25 s step; a flip is a sign change between consecutive moving steps
(v > 1 m/s) whose |w| both exceed FLIP_DEG; a run oscillates when it has >= 8 flips and >= 15% of its moving steps are flips.
Turning scenario = recorded route yaw range >= 30 deg (as experiments/op_probe/scripts/opj_turn_gain.py: 23 scenarios).
"""
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/op_parity/results/hugsim_specplan"
ARMS = ["P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1"]
PRE = {"exam": "pp-", "spec": "pp-spec-", "specplan": "pp-specplan-", "smooth": "pp-specplansmooth-", "mpc": "pp-specplanmpc-"}
SHORT = ["exam", "spec", "spec_plan", "smooth", "MPC"]
FLIP_DEG, MOVING = 0.3, 1.0
CLS = ["complete", "spin", "bg_coll", "fg_coll", "off_route", "stuck"]


def osc(w, v):
    m = (v[:-1] > MOVING) & (v[1:] > MOVING) if len(v) > 1 else np.zeros(0, bool)
    w = np.asarray(w)
    sig = np.abs(w) > FLIP_DEG
    pair = m[: len(w) - 1] & sig[:-1] & sig[1:] if len(w) > 1 else np.zeros(0, bool)
    flips = int((np.sign(w[:-1]) * np.sign(w[1:]) < 0)[pair].sum()) if len(w) > 1 else 0
    nm = int(max(1, (v > MOVING).sum()))
    return flips, flips / nm


def extract():
    sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
    import spin_analysis as SA
    D = Path(os.environ["DATA_DIR"])
    OUT.mkdir(parents=True, exist_ok=True)
    scen = {Path(p).stem for p in open(REPO / "experiments/hugsim/scripts/derot_all64.txt").read().split()}
    routes = json.load(open(D / "runs/op_parity/hugsim/routes.json"))
    turn = {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in routes.items()}
    want = {pfx + a: (pr, a) for a in ARMS for pr, pfx in PRE.items()}
    last = {}
    for r in csv.DictReader(open(D / "runs/op_parity/hugsim/results.csv")):
        if r["tag"] in want and r["scenario"] in scen and r["end"] != "crash":
            last[(r["tag"], r["scenario"])] = r
    rows, traces = [], {}
    for (tag, sc), r in sorted(last.items()):
        d = Path(r["run_dir"])
        pos, th, v, steer, plans = SA.load_run(d, r["agent"])
        res, _ = SA.analyse(pos, th, v, steer, plans, routes[r["scene"]])
        w = np.degrees(np.diff(th))
        fl, fr = osc(w, v)
        kap = []
        zf = d / "zs_steps.jsonl"
        if zf.exists():
            kap = [x.get("kappa") for x in (json.loads(l) for l in open(zf)) if "kappa" in x]
        rows.append(dict(r, preset=want[tag][0], arm=want[tag][1], max_abs_e=res["max_abs_e"], spin=bool(res["spin"]), v_max40=float(v[:40].max()),
                         standing=float((v < 0.3).mean()), n=len(v), flips=fl, flip_rate=fr, w_rms=float(np.sqrt(np.mean(w ** 2))), w_max=float(np.abs(w).max()),
                         oscillates=bool(fl >= 8 and fr >= 0.15), route_turn=turn[r["scene"]], turning=turn[r["scene"]] >= 30))
        traces[f"{want[tag][0]}|{want[tag][1]}|{sc}"] = dict(w=[round(float(x), 2) for x in w], v=[round(float(x), 2) for x in v],
                                                           kappa=[round(float(x), 4) for x in kap if x is not None])
    with open(OUT / "extract.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with gzip.open(OUT / "traces.json.gz", "wt") as f:
        json.dump(traces, f)
    print(len(rows), "rows ->", OUT)


# ------------------------------------------------------------------------------------------------------------------------- report
HR = REPO / "experiments/hugsim/results"
LAB = {"exam": "exam (iLQR tracks plan)", "spec": "spec (action curvature)", "specplan": "spec_plan (point plan curvature)", "smooth": "spec_plan_smooth (0.5-1.5 s)", "mpc": "spec_plan_mpc (legacy lateral MPC)"}
COL = {"exam": "#999999", "spec": "#0072B2", "specplan": "#D55E00", "smooth": "#E69F00", "mpc": "#009E73"}
GROUPS = {"P2": ["P2-F-s0", "P2-F-s1"], "P2+hinge": ["P2H10-F-s0", "P2H10-F-s1"], "all 4": ARMS}


def cls_of(df):
    c = df["end"].astype(str).replace({"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"})
    c[df["spin"].astype(bool)] = "spin"
    return c


def boot(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fmt(t, d=3):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def report():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "research"))
    import plot_style
    plot_style.apply()
    ex = pd.read_csv(OUT / "extract.csv")
    ex["cls"] = cls_of(ex)
    figs = OUT / "figs"
    figs.mkdir(exist_ok=True)
    wa = pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'")
    wx = pd.read_csv(HR / "wajepa_ref/wajepa_extract.csv")
    wx = wx[wx.tag == "wajepa"].set_index("scenario")
    wa = wa.drop_duplicates("scenario", keep="last").set_index("scenario")
    turn = ex.drop_duplicates("scenario").set_index("scenario")["turning"]
    wa["turning"] = turn.reindex(wa.index)
    wa["spin"] = wx["spin"].reindex(wa.index).astype(bool)
    wa["cls"] = cls_of(wa)
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))   # noqa: E731
    sc_all = sorted(ex.scenario.unique())
    tur = sorted(turn[turn].index)
    stra = sorted(turn[~turn].index)
    P(f"## T0 coverage: scenarios per (arm, preset); turning routes {len(tur)}, straight {len(stra)}\n")
    cov = ex.groupby(["arm", "preset"]).size().unstack()
    P(cov.to_markdown() + "\n")
    P("## T1 per (arm, preset), 64 scenarios, one run each\n")
    cols = ["arm", "preset", "n", "HD", "RC", "complete", "spin", "bg", "fg", "off_route", "stuck", "osc runs", "flips/100 moving steps (mean)", "w_rms deg/step (mean)"]
    rows = []

    def summ(d, name, preset):
        c = d.cls.value_counts()
        return [name, preset, len(d), d.hdscore.mean(), d.rc.mean(), c.get("complete", 0), c.get("spin", 0), c.get("bg_coll", 0), c.get("fg_coll", 0),
                c.get("off_route", 0), c.get("stuck", 0), int(d.oscillates.sum()) if "oscillates" in d else "-",
                100 * d.flip_rate.mean() if "flip_rate" in d else "-", d.w_rms.mean() if "w_rms" in d else "-"]
    for a in ARMS:
        for pr in PRE:
            rows.append(summ(ex[(ex.arm == a) & (ex.preset == pr)], a, pr))
    rows.append(summ(wa, "WA-JEPA", "exam"))
    T1 = pd.DataFrame(rows, columns=cols)
    T1.to_csv(OUT / "T1_arms.csv", index=False)
    P(T1.to_markdown(index=False, floatfmt=".3f") + "\n")

    def seedmean(df, arms, pr, scs):          # per-scenario mean over the group's arms
        d = df[(df.preset == pr) & df.arm.isin(arms) & df.scenario.isin(scs)]
        return d.groupby("scenario").hdscore.mean().reindex(scs)
    strata = {"all 64": sc_all, f"turning {len(tur)}": tur, f"straight {len(stra)}": stra}
    P("## T2 HD by group x preset x stratum (per-scenario mean over the group's seeds / arms; bootstrap 95% CI over scenarios)\n")
    rows = []
    for g, arms in GROUPS.items():
        for sn, scs in strata.items():
            r = [g, sn]
            for pr in PRE:
                m = boot(seedmean(ex, arms, pr, scs).values)
                r.append(f"{m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]")
            rows.append(r)
    for sn, scs in strata.items():
        rows.append(["WA-JEPA", sn, "-", "-", "%.3f [%.3f, %.3f]" % boot(wa.hdscore.reindex(scs).values)])
    T2 = pd.DataFrame(rows, columns=["group", "stratum"] + list(PRE))
    T2.to_csv(OUT / "T2_hd_strata.csv", index=False)
    P(T2.to_markdown(index=False) + "\n")

    P("## T3 end classes by stratum (mean count per arm, over the group's arms; spin = heading error >= 60 deg)\n")
    rows = []
    for sn, scs in list(strata.items())[1:]:
        for pr in PRE:
            d = ex[(ex.preset == pr) & ex.scenario.isin(scs)]
            c = d.cls.value_counts() / d.arm.nunique()
            rows.append([sn, pr] + [round(c.get(k, 0), 2) for k in CLS] + [round(d.hdscore.mean(), 3), round(d.rc.mean(), 3), round(d.oscillates.mean() * len(scs), 2),
                                                                         round(100 * d.flip_rate.mean(), 1)])
        c = wa[wa.index.isin(scs)].cls.value_counts()
        rows.append([sn, "WA-JEPA"] + [c.get(k, 0) for k in CLS] + [round(wa.hdscore.reindex(scs).mean(), 3), round(wa.rc.reindex(scs).mean(), 3), "-", "-"])
    T3 = pd.DataFrame(rows, columns=["stratum", "preset"] + CLS + ["HD", "RC", "oscillating runs", "flips/100 moving"])
    T3.to_csv(OUT / "T3_classes.csv", index=False)
    P(T3.to_markdown(index=False) + "\n")

    P("## T4 paired HD difference (scenario = unit, seed / arm mean within the group), bootstrap 95% CI; wins / losses / ties at |d| < 0.02\n")
    rows = []
    for g, arms in GROUPS.items():
        for sn, scs in strata.items():
            for new in ("specplan", "smooth", "mpc"):
                sp = seedmean(ex, arms, new, scs)
                for ref in [r for r in ("spec", "exam", "specplan") if r != new]:
                    d = (sp - seedmean(ex, arms, ref, scs)).values
                    rows.append([g, sn, f"{new} - {ref}", fmt(boot(d)), int((d > 0.02).sum()), int((d < -0.02).sum()), int((abs(d) <= 0.02).sum())])
                d = (sp - wa.hdscore.reindex(scs)).values
                rows.append([g, sn, f"{new} - WA-JEPA", fmt(boot(d)), int((d > 0.02).sum()), int((d < -0.02).sum()), int((abs(d) <= 0.02).sum())])
    for a in ARMS:
        for new in ("specplan", "smooth", "mpc"):
            sp = seedmean(ex, [a], new, sc_all)
            for ref in ("spec", "exam"):
                d = (sp - seedmean(ex, [a], ref, sc_all)).values
                rows.append([a, "all 64", f"{new} - {ref}", fmt(boot(d)), int((d > 0.02).sum()), int((d < -0.02).sum()), int((abs(d) <= 0.02).sum())])
    T4 = pd.DataFrame(rows, columns=["group", "stratum", "comparison", "mean diff [CI]", "wins", "losses", "ties"])
    T4.to_csv(OUT / "T4_paired.csv", index=False)
    P(T4.to_markdown(index=False) + "\n")

    # per-scenario table of the turning routes
    P("## T5 the turning routes: HD per preset (mean over the 4 arms), end classes of the 4 arms\n")
    rows = []
    for sc in tur:
        r = [sc, round(turn_deg(ex, sc), 0)]
        for pr in PRE:
            r.append(round(seedmean(ex, ARMS, pr, [sc]).iloc[0], 3))
        for pr in ("specplan", "smooth", "mpc"):
            d = ex[(ex.preset == pr) & (ex.scenario == sc)]
            r.append(" ".join(f"{k}" for k in d.sort_values("arm").cls))
        rows.append(r)
    T5 = pd.DataFrame(rows, columns=["scenario", "route turn deg"] + list(PRE) + [f"{k} classes (P2 s0, s1, hinge s0, s1)" for k in ("specplan", "smooth", "mpc")])
    T5.to_csv(OUT / "T5_turning.csv", index=False)
    P(T5.to_markdown(index=False) + "\n")
    tr = json.load(gzip.open(OUT / "traces.json.gz", "rt"))
    kr = []
    for key, t in tr.items():
        pr, a, sc = key.split("|")
        if len(t["kappa"]) > 2:
            k = np.asarray(t["kappa"])
            kr.append(dict(preset=pr, arm=a, scenario=sc, turning=sc in set(tur), k_abs=np.abs(k).mean(), k_step=np.abs(np.diff(k)).mean(),
                           k_sign_flips=int((np.sign(k[:-1]) * np.sign(k[1:]) < 0).sum()) / len(k)))
    KR = pd.DataFrame(kr)
    KR.to_csv(OUT / "kappa_runs.csv", index=False)
    P("## T6 requested curvature (what the lateral path is fed), mean per run then over runs: |kappa| (1/m), mean |kappa_t - kappa_t-1| (1/m per step), sign flips of kappa per step\n")
    T6 = KR.groupby(["turning", "preset"])[["k_abs", "k_step", "k_sign_flips"]].mean().round(4).reset_index()
    T6.to_csv(OUT / "T6_kappa.csv", index=False)
    P(T6.to_markdown(index=False) + "\n")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figures(ex, wa, tur, stra, seedmean, plt)


def turn_deg(ex, sc):
    return float(ex[ex.scenario == sc].route_turn.iloc[0])


def figures(ex, wa, tur, stra, seedmean, plt):
    import pandas as pd
    figs = OUT / "figs"
    sc_all = sorted(ex.scenario.unique())
    # fig1: HD by preset
    fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.4), sharey=True)
    for ax, (sn, scs) in zip(axs, [("all 64", sc_all), ("turning %d" % len(tur), tur), ("straight %d" % len(stra), stra)]):
        for gi, (g, arms) in enumerate(GROUPS.items()):
            if g == "all 4":
                continue
            for pi, pr in enumerate(PRE):
                m = boot(seedmean(ex, arms, pr, scs).values)
                ax.bar(gi * 6 + pi, m[0], color=COL[pr], yerr=[[m[0] - m[1]], [m[2] - m[0]]], capsize=1.5, error_kw=dict(lw=.6))
        ax.axhline(wa.hdscore.reindex(scs).mean(), color="k", ls="--", lw=.7)
        ax.set_xticks([2, 8]); ax.set_xticklabels(["P2", "P2+hinge"]); ax.set_title(sn)
    axs[0].set_ylabel("HD-Score (seed mean)")
    axs[0].text(0.02, wa.hdscore.mean() + .01, "WA-JEPA", fontsize=6.5, transform=axs[0].get_yaxis_transform())
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COL[p]) for p in PRE], [LAB[p] for p in PRE], loc="lower center", ncol=3, fontsize=6)
    fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(figs / "fig1_hd.png"); plt.close(fig)
    # fig2: paired scatter
    fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.6))
    for ax, new in zip(axs, ("specplan", "smooth", "mpc")):
        ref = "spec"
        x = seedmean(ex, ARMS, ref, sc_all); y = seedmean(ex, ARMS, new, sc_all)
        t = np.array([s in tur for s in sc_all])
        ax.plot([0, 1], [0, 1], color="#999", lw=.6)
        ax.scatter(x[~t], y[~t], s=9, color="#777", label="straight")
        ax.scatter(x[t], y[t], s=12, color=COL["specplan"], label="turning")
        ax.set_xlabel("HD under spec"); ax.set_ylabel(f"HD under {new}"); ax.set_title("4-arm mean per scenario", fontsize=7)
    axs[0].legend(); fig.tight_layout(); fig.savefig(figs / "fig2_paired.png"); plt.close(fig)
    # fig3: classes
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 2.5), sharey=True)
    cc = dict(zip(CLS, ["#009E73", "#CC79A7", "#E69F00", "#D55E00", "#56B4E9", "#999999"]))
    for ax, (sn, scs) in zip(axs, [("turning %d" % len(tur), tur), ("straight %d" % len(stra), stra)]):
        names = list(PRE) + ["WA-JEPA"]
        bot = np.zeros(len(names))
        for k in CLS:
            v = []
            for pr in names:
                if pr == "WA-JEPA":
                    v.append((wa[wa.index.isin(scs)].cls == k).sum())
                else:
                    d = ex[(ex.preset == pr) & ex.scenario.isin(scs)]
                    v.append((d.cls == k).sum() / d.arm.nunique())
            ax.bar(range(len(names)), v, bottom=bot, color=cc[k], label=k, width=.7); bot += v
        ax.set_xticks(range(len(names))); ax.set_xticklabels(SHORT + ["WA-JEPA"], fontsize=6); ax.set_title(sn)
    axs[0].set_ylabel("scenarios (mean per arm)"); axs[1].legend(fontsize=6, loc="upper right", ncol=2); fig.tight_layout(); fig.savefig(figs / "fig3_classes.png"); plt.close(fig)
    # fig4: oscillation
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 2.5))
    for ax, (key, yl) in zip(axs, [("flip_rate", "heading-rate sign flips per moving step"), ("w_rms", "heading-rate RMS (deg per 0.25 s)")]):
        for pi, pr in enumerate(PRE):
            d = ex[(ex.preset == pr) & ex.scenario.isin(tur)][key].values
            ax.scatter(pi + np.random.default_rng(0).uniform(-.18, .18, len(d)), d, s=5, color=COL[pr], alpha=.6)
            ax.hlines(np.mean(d), pi - .3, pi + .3, color="k", lw=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, fontsize=6.5); ax.set_ylabel(yl); ax.set_title("turning routes, 4 arms")
    fig.tight_layout(); fig.savefig(figs / "fig4_oscillation.png"); plt.close(fig)
    # fig6: requested curvature jitter
    KR = pd.read_csv(OUT / "kappa_runs.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 2.5))
    for ax, (key, yl) in zip(axs, [("k_abs", "mean |requested curvature| (1/m)"), ("k_step", "mean step-to-step change (1/m)")]):
        for pi, pr in enumerate(PRE):
            d = KR[(KR.preset == pr) & KR.turning][key].values
            ax.scatter(pi + np.random.default_rng(1).uniform(-.18, .18, len(d)), d, s=5, color=COL[pr], alpha=.6)
            ax.hlines(np.mean(d), pi - .3, pi + .3, color="k", lw=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, fontsize=6.5); ax.set_ylabel(yl); ax.set_title("turning routes, 4 arms")
    fig.tight_layout(); fig.savefig(figs / "fig6_kappa.png"); plt.close(fig)
    # fig5: example traces (largest specplan - spec swing among turning scenarios, arm P2-F-s0)
    tr = json.load(gzip.open(OUT / "traces.json.gz", "rt"))
    d = (seedmean(ex, ["P2-F-s0"], "specplan", tur) - seedmean(ex, ["P2-F-s0"], "spec", tur)).dropna()
    picks = [d.idxmax(), d.idxmin()]
    fig, axs = plt.subplots(2, 2, figsize=(6.875, 3.9), sharex="col")
    for j, sc in enumerate(picks):
        for pr in ("spec", "specplan", "smooth", "mpc"):
            t = tr.get(f"{pr}|P2-F-s0|{sc}")
            if not t:
                continue
            axs[0, j].plot(t["kappa"], color=COL[pr], lw=.9, label=LAB[pr]); axs[1, j].plot(t["w"], color=COL[pr], lw=.9)
        axs[0, j].set_title(f"{sc} (dHD {d[sc]:+.2f})", fontsize=7); axs[0, j].set_ylabel("requested curvature (1/m)"); axs[1, j].set_ylabel("heading rate (deg / step)"); axs[1, j].set_xlabel("step")
    axs[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(figs / "fig5_traces.png"); plt.close(fig)


if __name__ == "__main__":
    {"extract": extract, "report": report}[sys.argv[1]]()
