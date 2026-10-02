"""Tables and figures of the stop-position probe from the out-of-fold predictions (plan 2026-10-03-stoppos-probe.md). Project venv.

  python stoppos_report.py --run RUN_DIR --out experiments/vlm_arb/results/stoppos
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoppos_data as D  # noqa: E402
from jevdrive import stats  # noqa: E402

A_BRAKE = 3.0           # m/s^2, the base governor's braking limit (plan, Heads and readouts 5)
MARGINS = (2.0, 4.0, 6.0)


def load_oof(run_dir):
    z = np.load(Path(run_dir) / "oof.npz")
    return {n.replace("|", "/"): z[n] for n in z.files}


def softmax(lg, T=1.0):
    z = lg / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ question 2
def reg_tables(frames, oof, taps):
    rows = []
    sets = (("all", None), ("cl", "cl"), ("p4", "p4"))
    for task, mcol, truth in (("stop", "m_stop", "d_stop"), ("junc", "m_junc", "d_junc")):
        base = frames[(frames[mcol]) & (frames[truth] >= 0)]
        models = [(t, m, f"{t}/{m}/{task}") for t in taps for m in ("ridge", "mlp", "ridgeclk")]
        models += [("base", "route", f"base/route/{task}"), ("base", "clockridge", f"base/clockridge/{task}"), ("base", "clockgbm", f"base/clockgbm/{task}")]
        for tap, model, name in models:
            if name not in oof:
                continue
            pred = oof[name][base.index]
            d = base.assign(pred=pred)
            d = d[d.pred.notna()]
            d["err"] = d.pred - d[truth]
            for sname, src in sets:
                dd = d if src is None else d[d.src == src]
                if len(dd):
                    rows += D.bin_rows(dd, truth, "err", dict(task=task, tap=tap, model=model, set=sname))
    return pd.DataFrame(rows)


def clf_tables(frames, oof, taps):
    rows = []
    for task, mcol, ycol in (("stop40", "m_cls", "y_stop40"), ("light", "m_ls", "y_light"), ("sign", "m_ls", "y_sign")):
        base = frames[frames[mcol]]
        names = [(t, m, f"{t}/{m}/{task}") for t in taps for m in ("logit", "mlp", "logitclk")]
        names += [("base", "route", f"base/route/{task}"), ("base", "clocklogit", f"base/clocklogit/{task}"), ("base", "clockgbm", f"base/clockgbm/{task}")]
        for tap, model, name in names:
            if name not in oof:
                continue
            d = base.assign(s=oof[name][base.index])
            d = d[d.s.notna()]
            for sname, src in (("all", None), ("cl", "cl"), ("p4", "p4")):
                dd = d if src is None else d[d.src == src]
                if dd[ycol].nunique() < 2:
                    continue
                w = D.group_weights(dd)
                y = dd[ycol].to_numpy().astype(bool)
                thr = 0.0 if model != "route" else None
                r = dict(task=task, tap=tap, model=model, set=sname, frames=len(dd), routes=dd.id.nunique(), pos_routes=dd[y].id.nunique(), auc=D.auc(y, dd.s, w))
                # route-clustered CI of the AUC: bootstrap over routes of per-route-weighted AUC is not defined for single-class routes; use frame-cluster resample
                if thr is not None:
                    pred = dd.s.to_numpy() > thr
                    tpr = (w * (pred & y)).sum() / max((w * y).sum(), 1e-9)
                    tnr = (w * (~pred & ~y)).sum() / max((w * ~y).sum(), 1e-9)
                    r.update(acc=float((w * (pred == y)).sum() / w.sum()), tpr=float(tpr), tnr=float(tnr), bal_acc=float((tpr + tnr) / 2))
                rows.append(r)
    t = pd.DataFrame(rows)
    # route-clustered AUC interval for the headline rows (cluster bootstrap over routes)
    return t


def auc_ci(frames, oof, name, mcol, ycol, n_boot=300, seed=0):
    d = frames[frames[mcol]].assign(s=oof[name][frames[frames[mcol]].index])
    d = d[d.s.notna()]
    ids = d.id.unique()
    rng = np.random.default_rng(seed)
    by = {i: g for i, g in d.groupby("id")}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids))
        g = pd.concat([by[i].assign(cl=j) for j, i in enumerate(pick)])
        w = (1.0 / g.groupby("cl").key.transform("nunique") / g.groupby(["cl", "key"]).key.transform("size")).to_numpy()
        out.append(D.auc(g[ycol].to_numpy().astype(bool), g.s.to_numpy(), w))
    out = np.array(out)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def paired_vs(frames, oof, a, b, task="stop", bins=((0, 10),)):
    """Route-clustered paired difference of the macro MAE (a - b) of two regression outputs on the same frames."""
    mcol, truth = ("m_stop", "d_stop") if task == "stop" else ("m_junc", "d_junc")
    rows = []
    base = frames[frames[mcol] & (frames[truth] >= 0)]
    for lo, hi in bins:
        d = base[(base[truth] >= lo) & (base[truth] < hi)]
        d = d.assign(ea=(oof[a][d.index] - d[truth]).abs(), eb=(oof[b][d.index] - d[truth]).abs())
        d = d[d.ea.notna() & d.eb.notna()]
        ra, rb = D.route_means(d, "ea"), D.route_means(d, "eb")
        idx = ra.index.intersection(rb.index)
        r = stats.paired(ra[idx].to_numpy(), rb[idx].to_numpy())
        rows.append(dict(task=task, a=a, b=b, bin=f"{lo}-{hi} m", routes=len(idx), diff=r["mean"], lo=r["lo"], hi=r["hi"], mae_a=r["mean_a"], mae_b=r["mean_b"]))
    return rows


# ------------------------------------------------------------------------------------------------ question 3
def ece(p, correct, w, nb=10):
    """Weighted expected calibration error with equal-mass bins of the confidence p."""
    o = np.argsort(p)
    p, correct, w = p[o], correct[o], w[o]
    c = np.cumsum(w) / w.sum()
    b = np.minimum((c * nb).astype(int), nb - 1)
    tot, rel = 0.0, []
    for i in range(nb):
        m = b == i
        if not m.any():
            continue
        ww = w[m].sum() / w.sum()
        conf, acc = np.average(p[m], weights=w[m]), np.average(correct[m], weights=w[m])
        tot += ww * abs(conf - acc)
        rel.append((i, float(conf), float(acc), float(ww)))
    return tot, rel


def cls4_tables(frames, oof, tap):
    base = frames[frames.m_cls]
    L = np.stack([oof[f"{tap}/logit/cls4_{c}"][base.index] for c in range(4)], 1)
    T = oof[f"{tap}/logit/cls4_T"][base.index]
    ok = ~np.isnan(L).any(1)
    base, L, T = base[ok], L[ok], T[ok]
    y = base.y_cls4.to_numpy().astype(int)
    w = D.group_weights(base)
    out, rel_rows = [], []
    for nm, P in (("uncalibrated", softmax(L)), ("temperature-scaled", softmax(L, T[:, None]))):
        pred = P.argmax(1)
        conf = P.max(1)
        e, rel = ece(conf, (pred == y).astype(float), w)
        onehot = np.eye(4)[y]
        brier = float((w[:, None] * (P - onehot) ** 2).sum() / w.sum())
        ece_c = []
        for c in range(4):
            ec, rl = ece(P[:, c], (y == c).astype(float), w)
            ece_c.append(ec)
            rel_rows += [dict(probs=nm, cls=D.CLASSES[c], bin=i, conf=a, acc=b, weight=ww) for i, a, b, ww in rl]
        rel_rows += [dict(probs=nm, cls="top-label", bin=i, conf=a, acc=b, weight=ww) for i, a, b, ww in rel]
        out.append(dict(probs=nm, tap=tap, frames=len(base), routes=base.id.nunique(), acc=float((w * (pred == y)).sum() / w.sum()),
                        ece_top=e, ece_macro_ovr=float(np.mean(ece_c)), brier=brier, **{f"ece_{D.CLASSES[c]}": ece_c[c] for c in range(4)},
                        T_median=float(np.median(T))))
    P = softmax(L, T[:, None])
    pred = P.argmax(1)
    cm = np.zeros((4, 4))
    for t_, p_, w_ in zip(y, pred, w):
        cm[t_, p_] += w_
    cm = cm / cm.sum(1, keepdims=True)
    conf = pd.DataFrame(cm, index=[f"true {c}" for c in D.CLASSES], columns=[f"pred {c}" for c in D.CLASSES])
    # route-command baseline: classes from d_cmd - delta with the same thresholds
    return pd.DataFrame(out), pd.DataFrame(rel_rows), conf


def cls4_baselines(frames, oof):
    base = frames[frames.m_cls]
    y = base.y_cls4.to_numpy().astype(int)
    rows = []
    d = oof["base/route/cls4_d"][base.index]
    ok = ~np.isnan(d)
    pred = np.select([d <= 1.5, d <= 6, d <= 20], [3, 2, 1], 0)
    # route-command rule applies only to frames of routes with a command; frames without one are predicted free
    w = D.group_weights(base[ok])
    rows.append(dict(model="route command (d_cmd - delta)", acc=float((w * (pred[ok] == y[ok])).sum() / w.sum()), frames=int(ok.sum())))
    if "base/clockgbm/cls4_0" in oof:
        L = np.stack([oof[f"base/clockgbm/cls4_{c}"][base.index] for c in range(4)], 1)
        ok2 = ~np.isnan(L).any(1)
        w2 = D.group_weights(base[ok2])
        rows.append(dict(model="speed / time / odometer (GBM)", acc=float((w2 * (L[ok2].argmax(1) == y[ok2])).sum() / w2.sum()), frames=int(ok2.sum())))
    prior = np.bincount(y, weights=D.group_weights(base), minlength=4)
    rows.append(dict(model="majority class", acc=float(prior.max() / prior.sum()), frames=len(base)))
    return pd.DataFrame(rows)


def interval_tables(frames, oof, tap):
    base = frames[frames.m_stop & (frames.d_stop >= 0)]
    rows = []
    for nm, lo, hi in (("raw 10-90%", f"{tap}/q/lo_raw", f"{tap}/q/hi_raw"), ("conformalised 80%", f"{tap}/q/lo", f"{tap}/q/hi")):
        d = base.assign(lo=oof[lo][base.index], hi=oof[hi][base.index])
        d = d[d.lo.notna()]
        d["cov"] = ((d.d_stop >= d.lo) & (d.d_stop <= d.hi)).astype(float)
        d["width"] = d.hi - d.lo
        for label, m in [("0-40 m", (d.d_stop >= 0))] + [(n, (d.d_stop >= a) & (d.d_stop < b if b < 40 else d.d_stop <= b)) for (a, b), n in zip(D.BINS, D.BIN_NAMES)]:
            for sname, src in (("all", None), ("cl", "cl"), ("p4", "p4")):
                dd = d[m & ((d.src == src) if src else True)]
                if dd.empty:
                    continue
                c = D.route_means(dd, "cov")
                r = stats.bootstrap(c.to_numpy())
                rows.append(dict(interval=nm, tap=tap, bin=label, set=sname, routes=len(c), frames=len(dd), coverage=r["mean"], lo=r["lo"], hi=r["hi"],
                                 width=float(D.route_means(dd, "width").mean()), pooled_coverage=float(dd["cov"].mean())))
    return pd.DataFrame(rows)


def replay(frames, oof, tap, hp):
    """Stopping rules on held-out light approaches along the logged trajectories (open loop)."""
    fold = D.fold_of(frames)
    delta = {h["fold"]: h["delta"] for h in hp if h.get("tag") == "route"}
    f = frames.assign(fold=fold, gate=oof[f"{tap}/logit/stop40"], lo=oof[f"{tap}/q/lo"], hi=oof[f"{tap}/q/hi"])
    f["route_est"] = f.d_cmd_c - f.fold.map(delta)
    att = f[f.m_stop].key.unique()
    light_att = f[f.key.isin(att)]
    rows = []
    for key, g in light_att.groupby("key", sort=False):
        g = g[g.m_cls & g.gate.notna()].sort_values("t")
        if g.empty:
            continue
        for rule, sig in (("U", g.hi), ("L", g.lo), ("R", g.route_est)):
            for m in MARGINS:
                hit = (sig <= m) & ((g.gate > 0) if rule != "R" else True)
                d0 = g.d_stop.dropna()
                r = dict(key=key, id=g.id.iloc[0], src=g.src.iloc[0], rule=rule, margin=m, d0=float(d0.iloc[0]) if len(d0) else np.nan)
                if hit.any():
                    h = g[hit].iloc[0]
                    v = 0.0 if np.isnan(h.v) else h.v
                    brk = v * v / (2 * A_BRAKE)
                    r.update(fired=True, d_stop=h.d_stop, d_junc=h.d_junc, v=v, x_line=h.d_stop - brk, x_entr=h.d_junc - brk, t_fire=h.t)
                else:
                    r.update(fired=False)
                rows.append(r)
    res = pd.DataFrame(rows)
    # false fires: approaches without a light line within 40 m (junctions without a light, far frames of light routes)
    neg = f[f.m_cls & ~f.m_stop & f.gate.notna()]
    frows = []
    for key, g in neg.groupby("key", sort=False):
        g = g.sort_values("t")
        for rule, sig in (("U", g.hi), ("L", g.lo)):
            for m in MARGINS:
                frows.append(dict(key=key, id=g.id.iloc[0], src=g.src.iloc[0], kind=g.kind.iloc[0], rule=rule, margin=m, fired=bool(((sig <= m) & (g.gate > 0)).any()),
                                  has_junction=bool(g.d_junc.between(0, 40).any())))
    return res, pd.DataFrame(frows)


def replay_summary(res):
    rows = []
    parts = []
    for sname, ssel in (("cl", res.src == "cl"), ("p4", res.src == "p4"), ("all", res.src != "")):
        for aname, asel in (("all approaches", res.d0 > -99), ("start >= 12 m", res.d0 >= 12.0)):
            sub = res[ssel & asel]
            if len(sub):
                parts.append(sub.assign(src=f"{sname} / {aname}"))
    allr = pd.concat(parts)
    for (src, rule, m), g in allr.groupby([allr.src, "rule", "margin"]):
        n, fired = len(g), g[g.fired.astype(bool)]
        q = lambda c: [float(x) for x in np.quantile(fired[c], [0, 0.1, 0.5, 0.9, 1])] if len(fired) else [np.nan] * 5  # noqa: E731
        qx, qe = q("x_line"), q("x_entr")
        rows.append(dict(set=src, rule=f"{g.rule.iloc[0]}({m:g})", attempts=n, routes=g.id.nunique(), fired=len(fired) / n,
                         x_line_min=qx[0], x_line_p10=qx[1], x_line_med=qx[2], x_line_p90=qx[3], x_line_max=qx[4],
                         past_line=float((fired.x_line < 0).mean()) if len(fired) else np.nan,
                         short_gt3=float((fired.x_line > 3).mean()) if len(fired) else np.nan,
                         x_entr_min=qe[0], x_entr_med=qe[2], x_entr_max=qe[4], past_entr=float((fired.x_entr < 0).mean()) if len(fired) else np.nan,
                         d_fire_med=float(fired.d_stop.median()) if len(fired) else np.nan, v_fire_med=float(fired.v.median()) if len(fired) else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures
def figures(out_dir, reg, rel, iv, res, taps, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    names = D.BIN_NAMES
    series = [(t, "ridge", t) for t in taps] + [("base", "route", "route command"), ("base", "clockgbm", "speed / time / odometer")]
    cols = {"temporal": "#1f77b4", "vision": "#ff7f0e", "hidden": "#2ca02c", "native": "#9467bd", "route command": "#d62728", "speed / time / odometer": "#7f7f7f"}
    for a, task in zip(ax, ("stop", "junc")):
        for tap, model, label in series:
            d = reg[(reg.task == task) & (reg.tap == tap) & (reg.model == model) & (reg.set == "all")].set_index("bin")
            if d.empty:
                continue
            x = np.arange(len(names))
            y = [d.mae.get(n, np.nan) for n in names]
            lo = [d.lo.get(n, np.nan) for n in names]
            hi = [d.hi.get(n, np.nan) for n in names]
            ls = "--" if tap == "base" else "-"
            a.errorbar(x + (list(cols).index(label) - 2.5) * 0.04, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], marker="o", ms=4, capsize=2, ls=ls,
                       label=label, color=cols[label])
        a.set_xticks(range(len(names)), names)
        a.set_xlabel("true distance of the bumper to the " + ("stop line" if task == "stop" else "junction entrance"))
        a.set_ylabel("mean absolute error (m), route-macro, 95% CI over routes")
        a.set_title(("Stop line" if task == "stop" else "Junction entrance") + ": linear probe by true distance")
        a.set_yscale("log")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.savefig(out_dir / "stoppos_mae_by_bin.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    r = rel[(rel.cls == "top-label")]
    for nm, mk in (("uncalibrated", "o"), ("temperature-scaled", "s")):
        d = r[r.probs == nm]
        ax[0].plot(d.conf, d.acc, marker=mk, label=nm)
    ax[0].plot([0, 1], [0, 1], "k:")
    ax[0].set_xlabel("top-label confidence")
    ax[0].set_ylabel("accuracy")
    ax[0].set_title(f"Reliability of the 4-class head ({primary})")
    ax[0].legend()
    d = iv[(iv.set == "all") & (iv.bin != "0-40 m")]
    for nm, mk in (("raw 10-90%", "o"), ("conformalised 80%", "s")):
        dd = d[d.interval == nm]
        ax[1].errorbar(range(len(dd)), dd.coverage, yerr=[dd.coverage - dd.lo, dd.hi - dd.coverage], marker=mk, capsize=3, label=nm)
    ax[1].axhline(0.8, color="k", ls=":")
    ax[1].set_xticks(range(len(D.BIN_NAMES)), D.BIN_NAMES)
    ax[1].set_ylabel("coverage of the true distance")
    ax[1].set_title("Interval coverage by true distance")
    ax[1].legend()
    sub = res[(res.fired.astype(bool)) & (res.margin == 4.0)]
    for rule, c in (("U", "#1f77b4"), ("L", "#2ca02c"), ("R", "#d62728")):
        x = sub[sub.rule == rule].x_line
        if len(x):
            ax[2].hist(x.clip(-8, 20), bins=np.arange(-8, 21, 1.0), alpha=0.45, color=c, label=f"{rule}(4 m), n={len(x)}")
    ax[2].axvline(0, color="k")
    ax[2].set_xlabel("implied stop point relative to the stop line (m; < 0 is past the line)")
    ax[2].set_title("Stopping rules replayed, held-out approaches")
    ax[2].legend()
    fig.savefig(out_dir / "stoppos_calibration_replay.png", dpi=130)
    plt.close(fig)


def md(df, f=".2f"):
    return df.to_markdown(index=False, floatfmt=f)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = D.load_frames()
    oof = load_oof(a.run)
    hp = json.load(open(Path(a.run) / "hparams.json"))
    taps = [t for t in D.TAPS if f"{t}/ridge/stop" in oof]
    reg = reg_tables(frames, oof, taps)
    reg.to_csv(out / "reg_by_bin.csv", index=False)
    clf = clf_tables(frames, oof, taps)
    clf.to_csv(out / "clf.csv", index=False)
    # primary tap: lowest pooled macro MAE of d_stop in 0-10 m (mean of the two bins)
    r = reg[(reg.task == "stop") & (reg.model == "ridge") & (reg.set == "all") & reg.tap.isin(taps) & reg.bin.isin(D.BIN_NAMES[:2])]
    primary = r.groupby("tap").mae.mean().idxmin()
    (out / "primary_tap.txt").write_text(primary + "\n")
    c4, rel, conf = cls4_tables(frames, oof, primary)
    c4all = pd.concat([cls4_tables(frames, oof, t)[0] for t in taps if f"{t}/logit/cls4_0" in oof])
    c4all.to_csv(out / "cls4_calibration.csv", index=False)
    rel.to_csv(out / "cls4_reliability.csv", index=False)
    conf.to_csv(out / "cls4_confusion.csv")
    cls4_baselines(frames, oof).to_csv(out / "cls4_baselines.csv", index=False)
    iv = pd.concat([interval_tables(frames, oof, t) for t in taps if f"{t}/q/lo" in oof])
    iv.to_csv(out / "interval_coverage.csv", index=False)
    res, fal = replay(frames, oof, primary, hp)
    res.to_csv(out / "replay_attempts.csv", index=False)
    fal.to_csv(out / "replay_false_fires.csv", index=False)
    rs = replay_summary(res)
    rs.to_csv(out / "replay_summary.csv", index=False)
    fs = fal.groupby(["src", "kind", "rule", "margin"]).agg(attempts=("fired", "size"), fire_rate=("fired", "mean")).reset_index()
    fs.to_csv(out / "replay_false_fire_rates.csv", index=False)
    pv = []
    for tap in taps:
        pv += paired_vs(frames, oof, f"{tap}/ridge/stop", "base/route/stop", "stop", bins=((0, 5), (5, 10), (0, 10), (10, 40)))
        pv += paired_vs(frames, oof, f"{tap}/ridge/stop", "base/clockgbm/stop", "stop", bins=((0, 5), (5, 10), (0, 10), (10, 40)))
        pv += paired_vs(frames, oof, f"{tap}/ridge/junc", "base/route/junc", "junc", bins=((0, 10), (10, 40)))
        pv += paired_vs(frames, oof, f"{tap}/ridge/junc", "base/clockgbm/junc", "junc", bins=((0, 10), (10, 40)))
        pv += paired_vs(frames, oof, f"{tap}/ridgeclk/stop", "base/clockgbm/stop", "stop", bins=((0, 10), (10, 40)))
    pd.DataFrame(pv).to_csv(out / "paired_vs_baselines.csv", index=False)
    auc_rows = []
    for tap in taps:
        for task, mcol, ycol in (("stop40", "m_cls", "y_stop40"), ("light", "m_ls", "y_light"), ("sign", "m_ls", "y_sign")):
            nm = f"{tap}/logit/{task}"
            if nm in oof:
                lo, hi = auc_ci(frames, oof, nm, mcol, ycol, n_boot=200)
                auc_rows.append(dict(task=task, tap=tap, auc_lo=lo, auc_hi=hi))
    pd.DataFrame(auc_rows).to_csv(out / "clf_auc_ci.csv", index=False)
    figures(out, reg, rel, iv, res, taps, primary)
    print("primary tap", primary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
