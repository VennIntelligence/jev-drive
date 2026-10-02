"""Markdown tables of the stop-position probe from the result CSVs (plan 2026-10-03-stoppos-probe.md). Project venv.

  python stoppos_tables.py DIR      writes DIR/tables.md (the narrative of results/stoppos.md is written by hand around these tables)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BINS = ["0-5 m", "5-10 m", "10-20 m", "20-40 m"]
NAME = {"ridge": "linear", "mlp": "MLP", "ridgeclk": "linear + clock", "route": "route command", "clockridge": "clock (ridge)", "clockgbm": "clock (GBM)"}


def md(df, f=".2f"):
    return df.to_markdown(index=False, floatfmt=f)


def cell(r):
    if not np.isfinite(r.get("mae", np.nan)):
        return "-"
    return f"{r.mae:.2f} [{r.lo:.2f}, {r.hi:.2f}] ({r.bias:+.2f})"


def reg_pivot(reg, task, sset, models=("ridge", "mlp", "ridgeclk")):
    d = reg[(reg.task == task) & (reg.set == sset)]
    rows = []
    for (tap, model), g in d.groupby(["tap", "model"], sort=False):
        if tap != "base" and model not in models:
            continue
        r = {"input": f"{tap} / {NAME[model]}" if tap != "base" else NAME[model]}
        for b in BINS:
            x = g[g.bin == b]
            r[b] = cell(x.iloc[0]) if len(x) else "-"
        x = g[g.bin == BINS[0]]
        r["routes in bin 0-5 / 20-40"] = f"{int(x.routes.iloc[0]) if len(x) else 0} / {int(g[g.bin == BINS[3]].routes.iloc[0]) if len(g[g.bin == BINS[3]]) else 0}"
        rows.append(r)
    return pd.DataFrame(rows)


def main(d):
    d = Path(d)
    out = []
    P = lambda name: pd.read_csv(d / name)  # noqa: E731
    exists = lambda name: (d / name).exists()  # noqa: E731
    if exists("data_counts.csv"):
        c = pd.read_csv(d / "data_counts.csv", header=None, index_col=0)
        c.columns = ["cl", "p4"]
        c.index.name = "quantity"
        out += ["## Data counts", md(c.drop(index="set").reset_index(), ".0f"), ""]
    if exists("reg_by_bin.csv"):
        reg = P("reg_by_bin.csv")
        for task, title in (("stop", "stop line (light approaches with a known stop line)"), ("junc", "junction entrance")):
            for sset in ("all", "cl", "p4"):
                out += [f"## Probe error, {title}, set = {sset}",
                        "Macro mean absolute error in m over routes with a 95% route-bootstrap interval, and the signed bias (prediction minus truth) in parentheses, by true distance of the bumper.", "",
                        md(reg_pivot(reg, task, sset)), ""]
    if exists("clf.csv"):
        c = P("clf.csv")
        ci = P("clf_auc_ci.csv") if exists("clf_auc_ci.csv") else None
        c = c.copy()
        c["input"] = np.where(c.tap == "base", c.model, c.tap + " / " + c.model)
        if ci is not None:
            c = c.merge(ci.assign(model="logit"), on=["task", "tap", "model"], how="left")
        cols = [x for x in ("task", "set", "input", "frames", "routes", "pos_routes", "auc", "auc_lo", "auc_hi", "acc", "tpr", "tnr", "bal_acc") if x in c]
        out += ["## Classifiers (out-of-fold)", md(c[cols], ".3f"), ""]
    for name, title in (("cls4_calibration.csv", "4-class action head: calibration"), ("cls4_baselines.csv", "4-class action head: baselines"),
                        ("cls4_confusion.csv", "4-class action head: confusion matrix of the primary tap (row-normalised, temperature-scaled)")):
        if exists(name):
            t = P(name)
            out += [f"## {title}", md(t, ".3f"), ""]
    if exists("interval_coverage.csv"):
        t = P("interval_coverage.csv")
        t = t[t.set == "all"]
        out += ["## Interval coverage on held-out routes (all sets)", md(t, ".3f"), ""]
    for name, title in (("replay_summary.csv", "Stopping rules replayed on held-out light approaches"),
                        ("replay_false_fire_rates.csv", "False fires of the rules on approaches without a light line within 40 m")):
        if exists(name):
            out += [f"## {title}", md(P(name), ".2f"), ""]
    if exists("paired_vs_baselines.csv"):
        out += ["## Paired differences of the macro MAE (a - b; negative = a better), route-bootstrap", md(P("paired_vs_baselines.csv"), ".2f"), ""]
    for name in sorted(d.glob("scaling_*.csv")):
        if "draws" not in name.name:
            out += [f"## Data size: {name.stem}", md(pd.read_csv(name), ".3f"), ""]
    for name, title in (("transfer_reg.csv", "Domain transfer, regression"), ("transfer_clf.csv", "Domain transfer, stop40")):
        if exists(name):
            t = P(name)
            out += [f"## {title}", md(t, ".2f"), ""]
    for name, title in (("native_by_distance.csv", "Native outputs by distance (medians over routes, v >= 3 m/s)"), ("native_spearman.csv", "Native outputs: within-route Spearman correlation with the distance"),
                        ("native_discrimination.csv", "Native outputs: red/yellow approaches, 0-10 m vs 20-40 m"), ("fidelity.csv", "Replay fidelity (CL, 2 Hz hold-10 replay vs 20 Hz logged)")):
        if exists(name):
            out += [f"## {title}", md(P(name), ".3f"), ""]
    if exists("native_red_vs_green.csv"):
        t = P("native_red_vs_green.csv")
        out += ["## Native outputs: AUC of red/yellow vs green at the same distance (v >= 3 m/s; controls: v, d_stop)",
                md(t.pivot(index="metric", columns="bin", values="auc_high_red")[BINS].reset_index(), ".2f"), ""]
    (d / "tables.md").write_text("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
