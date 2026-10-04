"""The navtest token subset of the guard's `subset` mode: a log x driving-command stratified draw, checked against every arm that
already has a full official navtest v1 score of the same pipeline (op_lb gimm frames, `__base` export).

    $DATA_DIR/envs/jevdrive/bin/python experiments/op_guard/scripts/nav_subset.py study            # fractions x seeds -> tracking table
    $DATA_DIR/envs/jevdrive/bin/python experiments/op_guard/scripts/nav_subset.py freeze --frac 0.25 # seed 0 -> splits navsim/op-guard-navtest-sub

Draw: stratum = (log, driving command at t0); within a stratum a seeded permutation and k = floor(frac * n + u), u ~ U(0, 1) from the same
generator (randomised rounding: every stratum keeps its expected share, small strata are not forced in). The frozen subset is seed 0 of
the chosen fraction (not the best-tracking seed: picking the seed on these arms would fit the subset to them).
Tracking readout per fraction over seeds: |subset mean - full mean| of shipped, |subset delta - full delta| of every arm vs shipped
(the guard rule's quantity), and the log-cluster paired CI half-width (jevdrive.stats) of the subset delta.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

EVAL = data_dir() / "runs/navsim/eval"
PFX = "v1_navtest_opi_lb_navtest_"
SHIPPED = "gimm-cinque__base"
OUT = REPO / "experiments/op_guard/results"
SPLIT = ("navsim", "op-guard-navtest-sub")


def strata() -> pd.DataFrame:
    from jevdrive import navsim_zs as Z
    idx = Z.load_index("navtest", slim=True)
    return pd.DataFrame(dict(token=[e["token"] for e in idx], log=[e["log_name"] for e in idx],
                             cmd=[int(np.argmax(e["cmd"][-1])) for e in idx])).set_index("token")


def draw(st: pd.DataFrame, frac: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for _, g in st.groupby(["log", "cmd"], sort=True):
        t = g.index.to_numpy()[rng.permutation(len(g))]
        out += t[: int(np.floor(frac * len(t) + rng.random()))].tolist()
    return sorted(out)


def scores() -> pd.DataFrame:
    """token x arm PDMS (0-100) from the latest official CSV of every scored arm."""
    cols = {}
    for d in sorted(glob.glob(str(EVAL / f"{PFX}*__base"))):
        fs = sorted(glob.glob(d + "/*/*.csv"))
        if not fs:
            continue
        x = pd.read_csv(fs[-1])
        x = x[x.valid.astype(bool) & (x.token != "average")].set_index("token")
        cols[Path(d).name[len(PFX):]] = 100 * x.score
    return pd.DataFrame(cols)


def cmd_study(a):
    st, S = strata(), scores()
    S = S.loc[st.index.intersection(S.index)]
    arms = [c for c in S if c != SHIPPED and S[c].notna().all()]
    full = {c: float((S[c] - S[SHIPPED]).mean()) for c in arms}
    print(f"{len(arms)} arms with full scores, shipped full PDMS {S[SHIPPED].mean():.3f}")
    rows = []
    for f in a.fracs:
        for s in range(a.seeds):
            sub = draw(st, f, s)
            x = S.loc[sub]
            err = np.array([float((x[c] - x[SHIPPED]).mean()) - full[c] for c in arms])
            r = dict(frac=f, seed=s, n=len(sub), shipped_err=float(x[SHIPPED].mean() - S[SHIPPED].mean()),
                     delta_abs_err_mean=float(np.abs(err).mean()), delta_abs_err_max=float(np.abs(err).max()))
            if s < a.ci_seeds:          # cluster CI half-width of the subset delta, median over arms
                hw = [(lambda q: (q["hi"] - q["lo"]) / 2)(stats.paired(x[c].to_numpy(), x[SHIPPED].to_numpy(), groups=st.loc[sub, "log"].to_numpy(), n_boot=2000))
                      for c in arms]
                r["ci_halfwidth_median"] = float(np.median(hw))
            rows.append(r)
    R = pd.DataFrame(rows)
    agg = R.groupby("frac").agg(n=("n", "mean"), shipped_abs_err_mean=("shipped_err", lambda v: np.abs(v).mean()),
                                shipped_abs_err_max=("shipped_err", lambda v: np.abs(v).max()),
                                delta_abs_err_mean=("delta_abs_err_mean", "mean"), delta_abs_err_p95=("delta_abs_err_max", lambda v: np.percentile(v, 95)),
                                delta_abs_err_max=("delta_abs_err_max", "max"), ci_halfwidth_median=("ci_halfwidth_median", "mean")).reset_index()
    seed0 = R[R.seed == 0].set_index("frac")[["n", "shipped_err", "delta_abs_err_mean", "delta_abs_err_max"]].add_prefix("seed0_").reset_index()
    agg = agg.merge(seed0, on="frac")
    stats.write_table(agg, OUT / "navtest_subset_study", floatfmt=".3f",
                      note=f"{a.seeds} seeds per fraction; {len(arms)} arms with full official navtest v1 scores ({PFX}*__base) vs {SHIPPED}; "
                           "errors in PDMS points; delta_abs_err_max = worst arm per seed (p95 / max over seeds); CI = log-cluster paired bootstrap.")
    print(agg.round(3).to_string(index=False))


def cmd_freeze(a):
    from jevdrive.data import splits
    st, S = strata(), scores()
    sub = draw(st, a.frac, 0)
    x = S.loc[sub]
    arms = [c for c in S if c != SHIPPED and S[c].notna().all()]
    per_arm = {c: dict(full=float((S[c] - S[SHIPPED]).mean()), subset=float((x[c] - x[SHIPPED]).mean())) for c in arms}
    sp = splits.define(*SPLIT, sub, unit="token", status="frozen",
                       origin=f"navsim/navtest@v1, stratum (log, driving command at t0), frac {a.frac}, seed 0, randomised rounding "
                              "(experiments/op_guard/scripts/nav_subset.py draw)",
                       used_by=["experiments/op_guard (navtest line, subset mode)"],
                       notes=f"shipped PDMS subset {x[SHIPPED].mean():.3f} vs full {S[SHIPPED].mean():.3f}; study: experiments/op_guard/results/navtest_subset_study.md")
    (OUT / "navtest_subset_freeze.json").write_text(json.dumps(dict(split_id=sp.id, n=len(sub), frac=a.frac, seed=0,
                                                                  shipped_subset=float(x[SHIPPED].mean()), shipped_full=float(S[SHIPPED].mean()),
                                                                  arms=per_arm), indent=1) + "\n")
    print(sp.id, len(sub), f"shipped {x[SHIPPED].mean():.3f} vs {S[SHIPPED].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("study")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.25, 0.33, 0.5])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--ci-seeds", type=int, default=3)
    p = sp.add_parser("freeze")
    p.add_argument("--frac", type=float, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"study": cmd_study, "freeze": cmd_freeze}[a.cmd](a)
