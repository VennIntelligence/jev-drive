#!/usr/bin/env python
"""NAVSIM zero-shot exam: collect the devkit score CSVs into result tables and figures
(todos/2026-09-24-zeroshot-exam/navsim.md). Runs on the box in envs/navsim2 (pandas + matplotlib), reads
$DATA_DIR/runs/navsim/eval/<v>_<split>_<name>/<ts>/*.csv (latest per name) and writes to
$DATA_DIR/runs/navsim_zs/report/: results.csv / results.md, per-command and failure tables, per-token scores
(scores.csv.gz), ADE vs the logged future, and figures via research/plot_style.py.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "research")]
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

EVAL = data_dir() / "runs" / "navsim" / "eval"
OUT = Z.root("report")
V2 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
      "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
V1 = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
         "traffic_light_compliance": "TLC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
         "lane_keeping": "LK", "history_comfort": "HC", "two_frame_extended_comfort": "EC", "comfort": "C",
         "extended_comfort": "EC"}
# row order and display names (model, variant); files are <v>_<split>_<name>
AGENTS = [("human", "human (log)", ""), ("cv", "constant velocity", ""),
          ("alpamayo_nav", "Alpamayo 1.5", "nav"), ("alpamayo_nonav", "Alpamayo 1.5", "no-nav"),
          ("lebowski_none", "openpilot Lebowski", "none"), ("lebowski_cmd", "openpilot Lebowski", "cmd"),
          ("cinque_none", "openpilot Cinque v3", "none"), ("cinque_cmd", "openpilot Cinque v3", "cmd"),
          ("small_none", "openpilot small", "none"), ("small_cmd", "openpilot small", "cmd")]


def latest_csv(ver, split, name):
    fs = sorted(glob.glob(str(EVAL / f"{ver}_{split}_{name}" / "*" / "*.csv")))
    return pd.read_csv(fs[-1]) if fs else None


def per_token(df):
    return df[~df["token"].str.contains("_")].copy() if df is not None else None   # drop the summary rows


def boot(x, y=None, B=2000, seed=0):
    """95% bootstrap CI (tokens resampled) of 100 * mean(x), or of 100 * mean(x - y) for paired arrays."""
    d = np.asarray(x, float) - (0 if y is None else np.asarray(y, float))
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return tuple(100 * np.percentile(m, [2.5, 97.5]))


def paired_table(sc):
    """Paired differences on common tokens: Alpamayo nav - no-nav (the 3000-token subset), nav - openpilot."""
    idx = Z.load_index("navtest")
    cmd = {e["token"]: ["left", "straight", "right", "unknown"][int(np.argmax(e["cmd"][-1]))] for e in idx}
    rows = []
    for metric in ("PDMS", "EPDMS"):
        d = sc[sc.metric == metric].pivot_table(index="token", columns="agent", values="score")
        for a, b in (("alpamayo_nav", "alpamayo_nonav"), ("alpamayo_nav", "lebowski_none"), ("lebowski_cmd", "lebowski_none"),
                     ("cinque_cmd", "cinque_none"), ("small_cmd", "small_none"), ("alpamayo_nav", "cv")):
            if a not in d or b not in d:
                continue
            dd = d[[a, b]].dropna()
            for c in ("all", "left", "straight", "right"):
                g = dd if c == "all" else dd[dd.index.map(cmd) == c]
                lo, hi = boot(g[a], g[b])
                rows.append({"metric": metric, "a": a, "b": b, "command": c, "n": len(g), "a_mean": 100 * g[a].mean(),
                             "b_mean": 100 * g[b].mean(), "diff": 100 * (g[a] - g[b]).mean(), "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "paired_navtest.csv", index=False, float_format="%.2f")
    return df


def navtest_tables():
    idx = Z.load_index("navtest")
    cmd = {e["token"]: ["left", "straight", "right", "unknown"][int(np.argmax(e["cmd"][-1]))] for e in idx}
    rows, per, allscores = [], [], []
    for ver, cols, metric in (("v1", V1, "PDMS"), ("v2", V2, "EPDMS")):
        for name, model, var in AGENTS:
            df = per_token(latest_csv(ver, "navtest", name))
            if df is None:
                continue
            ok = df[df["valid"]]
            lo, hi = boot(ok["score"].to_numpy())
            r = {"metric": metric, "model": model, "variant": var, "n": len(df), "n_valid": len(ok),
                 "score": 100 * ok["score"].mean(), "ci_lo": lo, "ci_hi": hi}
            r.update({SHORT[c]: 100 * ok[c].mean() for c in cols if c in ok})
            rows.append(r)
            ok = ok.assign(command=ok["token"].map(cmd))
            for c, g in ok.groupby("command"):
                per.append({"metric": metric, "model": model, "variant": var, "command": c, "n": len(g),
                            "score": 100 * g["score"].mean(), "EP": 100 * g["ego_progress"].mean(),
                            "DAC": 100 * g["drivable_area_compliance"].mean(),
                            "NC": 100 * g["no_at_fault_collisions"].mean()})
            allscores.append(ok[["token", "score"] + [c for c in cols if c in ok]].assign(metric=metric, agent=name))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_navtest.csv", index=False, float_format="%.2f")
    pd.DataFrame(per).to_csv(OUT / "results_navtest_by_command.csv", index=False, float_format="%.2f")
    pd.concat(allscores).to_csv(OUT / "scores_navtest.csv.gz", index=False, float_format="%.4g")
    return res, pd.DataFrame(per)


def navhard_table():
    rows = []
    for name, model, var in AGENTS:
        df = latest_csv("v2", "navhard_two_stage", name)
        if df is None:
            continue
        summ = df[df["token"].str.startswith("extended_pdm_score")].set_index("token")
        r = {"model": model, "variant": var, "n_tokens": int((~df["token"].str.contains("_")).sum())}
        for key, lab in (("extended_pdm_score_stage_one", "stage1"), ("extended_pdm_score_stage_two", "stage2"),
                         ("extended_pdm_score_combined", "EPDMS")):
            if key in summ.index:
                r[lab] = 100 * float(summ.loc[key, "score"])
        comb = summ.loc["extended_pdm_score_combined"] if "extended_pdm_score_combined" in summ.index else None
        if comb is not None:
            for c in V2:
                for st in ("stage_one", "stage_two"):
                    k = f"{c}_{st}"
                    if k in comb and pd.notna(comb[k]):
                        r[f"{SHORT[c]}_{st[6:]}"] = 100 * float(comb[k])
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_navhard.csv", index=False, float_format="%.2f")
    return res


def ade_table():
    """ADE / FDE of every prediction file against the logged future (navtest), for the failure analysis."""
    fut = np.load(Z.root("index") / "navtest_future.npz")
    gt = dict(zip(fut["tokens"].tolist(), fut["poses"]))
    rows = []
    files = sorted(Z.root("preds", "navtest").glob("*.npz")) + sorted(Z.root("openpilot", "navtest").glob("*_*.npz"))
    for f in files:
        z = np.load(f)
        p = z["poses"]
        g = np.stack([gt[t] for t in z["tokens"].tolist()])
        e = np.linalg.norm(p[..., :2] - g[..., :2], axis=-1)
        rows.append({"pred": f.stem, "n": len(p), "ADE4": e.mean(), "FDE4": e[:, -1].mean(),
                     "median_ADE4": np.median(e.mean(1)), "progress_ratio": np.median(p[:, -1, 0] / np.maximum(g[:, -1, 0], 1))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ade_navtest.csv", index=False, float_format="%.3f")
    return df


def md(df: pd.DataFrame, fmt: str = "{:.1f}") -> str:
    """Markdown table without the tabulate dependency."""
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("" if pd.isna(v) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    res, per = navtest_tables()
    pair = paired_table(pd.read_csv(OUT / "scores_navtest.csv.gz"))
    hard = navhard_table()
    ade = ade_table()
    (OUT / "results.md").write_text("## navtest\n\n" + md(res) + "\n\n## navtest by command\n\n" + md(per)
                                    + "\n\n## paired differences\n\n" + md(pair) + "\n\n## navhard two-stage\n\n" + md(hard) + "\n\n## ADE vs log (navtest)\n\n"
                                    + md(ade, "{:.3f}") + "\n")
    print((OUT / "results.md").read_text())
