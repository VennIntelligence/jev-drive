"""Tables and figure for results/wod_launch_gain.md.  python wod_launch_gain_report.py <jsonl_dir> <out_dir> <png>
Reads <jsonl_dir>/<model>.s*.jsonl (wod_launch_gain.py) and the HUGSIM replay tables of spin_attribution (q2_replay_steps.csv, q2_log_inference.csv)."""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

jd, out, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
HS = Path(__file__).resolve().parents[1] / "results" / "spin_attribution"
W, C = 2.0, 0.19


def zwin(cs):
    """6-step window kernel growth per step (decision 100 / spin_attr_q2.zwin) for c * s."""
    r = np.roots(np.r_[1.0, -(1 + cs), np.zeros(5), cs])
    r = r[np.abs(r - 1) > 1e-6]
    return float(np.max(np.abs(r)))


def s_star():
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if zwin(C * mid) < 1 else (lo, mid)
    return lo


def boot(vals, groups, f=np.median, n=4000, seed=0):
    """cluster bootstrap (resample groups) of f over the values."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    u, inv = np.unique(groups, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(u))]
    rng = np.random.default_rng(seed)
    b = [f(vals[np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])]) for _ in range(n)]
    return float(f(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load(model):
    rows = []
    for f in sorted(glob.glob(str(jd / f"{model}.s*.jsonl"))):
        rows += [json.loads(x) for x in open(f)]
    if not rows:
        return pd.DataFrame()
    D = pd.DataFrame([dict(row=r["row"], id=r["id"], m=r["m"], cluster=r["cluster"], split=r["split"], v0=r["v0"],
                           phi=r["+0"]["phi1"], gain=(r["+2"]["phi1"] - r["-2"]["phi1"]) / 2 / (1.2 * W)) for r in rows])
    D["ev"] = [f"{i.rsplit('-', 1)[0]}|{int(i.rsplit('-', 1)[1]) - 2 * (m - 1)}" for i, m in zip(D.id, D.m)]   # onset frame identifies the event
    return D


H = pd.read_csv(HS / "q2_replay_steps.csv")
H["grp"] = np.where(H.group == "spin", "spin", "non-spin")
HL = pd.read_csv(HS / "q2_log_inference.csv")
res, lines = {}, []
ss = s_star()
lines.append(f"threshold: z(c * s) = 1 at s* = {ss:.3f} deg/deg (c = {C}), i.e. c * s* = {C * ss:.3f}")
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
cols = {"cinque": "#1f77b4", "it_dw3-s0": "#2ca02c"}
for g, c in (("spin", "#d62728"), ("non-spin", "#7f7f7f")):
    d = H[(H.grp == g) & (H.step <= 9)].groupby("step").s_local
    ax[0].plot(d.median().index, d.median().values, "-o", color=c, ms=3, label=f"HUGSIM {g} (median, 6 / 12 logs)")
    ax[0].fill_between(d.median().index, d.quantile(0.25), d.quantile(0.75), color=c, alpha=0.12)
tab = []
for model in ("cinque", "it_dw3-s0"):
    D = load(model)
    if D.empty:
        continue
    D.to_csv(out / f"rows_{model}.csv", index=False)
    for sub, d in (("all", D), ("dev", D[D.split == "dev"])):
        for m in (1, 2, 3):
            x = d[d.m == m]
            if len(x) < 5:
                continue
            md, lo, hi = boot(x.gain, x.cluster)
            mn, mlo, mhi = boot(x.gain, x.cluster, np.mean)
            tab.append(dict(model=model, subset=sub, step=m, n=len(x), n_cluster=x.cluster.nunique(), gain_median=md, lo=lo, hi=hi,
                            gain_mean=mn, mean_lo=mlo, mean_hi=mhi, ct=C * md, z=zwin(C * md), z_lo=zwin(max(C * lo, 0)), z_hi=zwin(max(C * hi, 0)),
                            frac_above_s_star=float((x.gain > ss).mean())))
    T = pd.DataFrame(tab)
    t = T[(T.model == model) & (T.subset == "all")]
    ax[0].errorbar(t.step, t.gain_median, yerr=[t.gain_median - t.lo, t.hi - t.gain_median], fmt="-s", color=cols[model], capsize=3, label=f"WOD real, {model} (median, 95% cluster CI, n {int(t.n.iloc[0])})")
    # launch lean per event: mean |phi1| over steps 1-2
    e = D[D.m.isin([1, 2])].groupby("ev").agg(cluster=("cluster", "first"), split=("split", "first"), lean=("phi", lambda p: float(np.mean(np.abs(p)))), k=("m", "size"))
    e = e[e.k == 2]
    md, lo, hi = boot(e.lean, e.cluster)
    mn, mlo, mhi = boot(e.lean, e.cluster, np.mean)
    p_sp = mannwhitneyu(e.lean, HL[HL.spin == 1].phi12).pvalue
    p_ns = mannwhitneyu(e.lean, HL[HL.spin == 0].phi12).pvalue
    lines.append(f"{model}: launch lean |phi1| steps 1-2 per event n={len(e)}: median {md:.2f} [{lo:.2f}, {hi:.2f}], mean {mn:.2f} [{mlo:.2f}, {mhi:.2f}], "
                 f"p90 {np.percentile(e.lean, 90):.2f}; frac >= 1 deg {np.mean(e.lean >= 1):.3f}; MWU vs HUGSIM native spin p={p_sp:.3g}, non-spin p={p_ns:.3g}")
    xs = np.sort(e.lean)
    ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=cols[model], label=f"WOD real, {model}")
    if model == "cinque":
        lean_c = e
for g, c in ((1, "#d62728"), (0, "#7f7f7f")):
    v = np.sort(HL[HL.spin == g].phi12)
    ax[1].plot(v, np.arange(1, len(v) + 1) / len(v), "--", color=c, label=f"HUGSIM native logs, {'spin' if g else 'non-spin'} (n {len(v)})")
ax[1].set_xlim(0, 4)
ax[1].set_xlabel("launch lean |phi1| at steps 1-2 (deg)")
ax[1].set_ylabel("ECDF")
ax[1].legend(fontsize=7)
ax[1].set_title("(b) launch lean")
T = pd.DataFrame(tab)
T.to_csv(out / "gain_table.csv", index=False)
# growth panel: z with CI
yy, labs = [], []
def zrow(label, md, lo, hi, color):
    y = len(labs)
    ax[2].errorbar([zwin(C * md)], [y], xerr=[[zwin(C * md) - zwin(C * max(lo, 0))], [zwin(C * hi) - zwin(C * md)]], fmt="o", color=color, capsize=3)
    labs.append(label)
for g, c in (("spin", "#d62728"), ("non-spin", "#7f7f7f")):
    x = H[(H.grp == g) & (H.step == 1)].s_local
    zrow(f"HUGSIM {g} (n {len(x)})", *boot(x, np.arange(len(x))), c)
for model in ("cinque", "it_dw3-s0"):
    t = T[(T.model == model) & (T.subset == "all") & (T.step == 1)]
    if len(t):
        r = t.iloc[0]
        zrow(f"WOD real {model} (n {int(r.n)})", r.gain_median, r.lo, r.hi, cols[model])
ax[2].axvline(1, color="k", lw=0.8)
ax[2].set_yticks(range(len(labs)))
ax[2].set_yticklabels(labs, fontsize=8)
ax[2].set_xlabel("loop growth per step z (window kernel, c = 0.19), step-1 gain median, 95% CI")
ax[2].set_title("(c) loop growth")
ax[0].set_xlabel("launch step (HUGSIM step = WOD m)")
ax[0].set_ylabel("local gain (deg of 1 s plan direction per deg of added history yaw)")
ax[0].set_xlim(0.7, 9.3)
ax[0].set_title("(a) local gain by step")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(png, dpi=130)
for _, r in T.iterrows():
    lines.append(f"{r.model} {r.subset} step {int(r.step)}: n={int(r.n)} ({int(r.n_cluster)} segments) gain median {r.gain_median:.2f} [{r.lo:.2f}, {r.hi:.2f}], mean {r.gain_mean:.2f} [{r.mean_lo:.2f}, {r.mean_hi:.2f}]; "
                 f"c*gain {r.ct:.3f}; z {r.z:.2f} [{r.z_lo:.2f}, {r.z_hi:.2f}]; frac events above s* {r.frac_above_s_star:.2f}")
x = H[H.step == 1].groupby("grp").s_local.agg(["median", "size"])
for g in ("spin", "non-spin"):
    lines.append(f"HUGSIM {g} step 1 (CPU replay): median gain {x.loc[g, 'median']:.2f} (n {int(x.loc[g, 'size'])}), z {zwin(C * x.loc[g, 'median']):.2f}")
(out / "summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
