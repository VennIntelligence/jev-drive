"""One definition of bootstrap CIs and of result tables, so numbers from different experiments compare (docs/lib.md).

    r = paired(score_a, score_b, groups=log_ids)       # dict: n, units, mean, lo, hi, n_boot, seed, alpha
    write_table([dict(arm="A-B", **r)], run.path("results"))   # results.csv + results.md

Default = the repo's dominant convention (jevdrive/nq3_cl_report.py boot_mean, tfv6_rules.boot_paired): percentile
CI of the mean over B = 10000 resamples of units, `default_rng(0).integers(0, n, (B, n))`, linear interpolation. With
`groups` it is the cluster bootstrap of jevdrive/traj.py boot_ci (resample clusters, ratio of sums), bit-compatible
with it for the same B / seed. Resample independent units: scenes / logs / routes, not frames of one sequence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_BOOT, SEED, ALPHA = 10_000, 0, 0.05


def bootstrap(x, groups=None, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Mean of x with a (1 - alpha) percentile bootstrap CI. Non-finite values are dropped (n counts the rest).
    groups=None: x holds one value per independent unit. groups: unit label per value; clusters are resampled and
    the statistic is sum / count over the drawn clusters (the frame-weighted mean)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    g = None if groups is None else np.asarray(groups)[ok]
    x = x[ok]
    out = dict(n=len(x), units=len(x), mean=np.nan, lo=np.nan, hi=np.nan, n_boot=n_boot, seed=seed, alpha=alpha)
    if not len(x):
        return out
    rng = np.random.default_rng(seed)
    if g is None:
        b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    else:
        import pandas as pd
        codes, uniq = pd.factorize(g)
        s, c = np.bincount(codes, x, len(uniq)), np.bincount(codes, minlength=len(uniq)).astype(float)
        idx = rng.integers(len(uniq), size=(n_boot, len(uniq)))
        b = s[idx].sum(1) / c[idx].sum(1)
        out["units"] = len(uniq)
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    out.update(mean=float(x.mean()), lo=float(lo), hi=float(hi))
    return out


def paired(a, b, groups=None, **kw) -> dict:
    """CI of mean(a - b) with one resample applied to both arms (index-aligned pairs; pairs with a non-finite side
    are dropped). Also returns the arm means over the kept pairs as mean_a / mean_b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms differ in shape: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    r = bootstrap(a[ok] - b[ok], None if groups is None else np.asarray(groups)[ok], **kw)
    r.update(mean_a=float(a[ok].mean()) if ok.any() else np.nan, mean_b=float(b[ok].mean()) if ok.any() else np.nan)
    return r


def fmt(r: dict, f: str = ".3f", scale: float = 1.0) -> str:
    """'mean [lo, hi]' text for prose and tables."""
    return f"{r['mean'] * scale:{f}} [{r['lo'] * scale:{f}}, {r['hi'] * scale:{f}}]"


def write_table(rows, stem, floatfmt: str = ".3f", note: str = ""):
    """Write <stem>.csv (all columns, full precision) and <stem>.md (same rows; columns that are constant over the
    rows, e.g. n_boot / seed / alpha, move to a footer line). Rows: dicts or a DataFrame. Returns the DataFrame."""
    import pandas as pd
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    const = [c for c in ("n_boot", "seed", "alpha") if c in df and df[c].nunique(dropna=False) == 1 and len(df) > 1]
    md = df.drop(columns=const).to_markdown(index=False, floatfmt=floatfmt)
    foot = ", ".join(f"{c} = {df[c].iloc[0]}" for c in const)
    if "lo" in df:
        foot = "CI: percentile bootstrap" + (f", {foot}" if foot else "")
    text = md + "\n" + "".join(f"\n{s}\n" for s in (foot, note) if s)
    stem.with_suffix(".md").write_text(text)
    return df
