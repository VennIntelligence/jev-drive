"""Data, targets, folds and evaluation helpers of the stop-position probe (plan 2026-10-03-stoppos-probe.md). Project venv."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402
from jevdrive.data import splits  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
PROC = DATA / "processed/vlm_arb_stoppos"
FEATS = DATA / "runs/vlm_arb_stoppos/extract/feats"
TAPS = ("temporal", "vision", "hidden", "native")
BINS = ((0, 5), (5, 10), (10, 20), (20, 40))
BIN_NAMES = [f"{a}-{b} m" for a, b in BINS]
CLASSES = ("free", "slow", "creep", "at_line")
CLOCK = ("v", "tdrive", "dist")
PAST_STOP, PAST_JUNC = -2.0, -3.8
K = 5


def load_frames() -> pd.DataFrame:
    f = pd.concat([pd.read_parquet(PROC / "frames_cl.parquet"), pd.read_parquet(PROC / "frames_p4.parquet")], ignore_index=True)
    f["id"] = f.src + "-" + f.route
    f["stop_ok"] = f.stop_ok.fillna(False).astype(bool)
    f = f.sort_values(["key", "k"], kind="stable").reset_index(drop=True)
    f["kind"] = f.kind.fillna("none")
    f["past"] = (f.d_stop < PAST_STOP) | (f.d_junc < PAST_JUNC)
    f["has_label"] = f.v.notna()
    ok = f.has_label
    f["m_junc"] = ok & f.d_junc.between(PAST_JUNC, 40)
    f["y_junc"] = f.d_junc.clip(0, 40)
    f["m_stop"] = ok & f.stop_ok & f.d_stop.between(PAST_STOP, 40)
    f["y_stop"] = f.d_stop
    f["m_cls"] = ok & f.stop_ok & ~f.past
    f["y_stop40"] = f.m_stop.astype(int)
    d = f.d_stop
    f["y_cls4"] = np.select([d <= 1.5, d <= 6, d <= 20], [3, 2, 1], 0)
    f.loc[~f.m_stop, "y_cls4"] = 0
    f["m_ls"] = ok & f.d_junc.between(0, 40) & f.has_light.notna()
    f["y_light"] = (f.has_light == 1).astype(int)
    f["y_sign"] = (f.has_sign == 1).astype(int)
    f["d_cmd_c"] = f.d_cmd.where(f.d_cmd.notna(), np.inf)
    f["clock_ok"] = f[list(CLOCK)].notna().all(axis=1)
    return f


def load_feats(frames: pd.DataFrame, tap: str, keys=None) -> np.ndarray:
    """(N, D) array aligned to `frames` (rows of a stream in k order); frames of streams without features are NaN."""
    dims = dict(temporal=512, vision=512, hidden=16384, native=2066)
    dt = np.float16 if tap == "hidden" else np.float32
    out = np.full((len(frames), dims[tap]), np.nan, dt)
    start = 0
    for key, g in frames.groupby("key", sort=False):
        n = len(g)
        p = FEATS / (key.replace("/", "__") + ".npz")
        if p.exists() and (keys is None or key in keys):
            with np.load(p) as z:
                a = z[tap]
            assert len(a) == n, (key, len(a), n)
            out[g.index.to_numpy()] = a
        start += n
    return out


def have_features(frames: pd.DataFrame) -> np.ndarray:
    ok = {k for k in frames.key.unique() if (FEATS / (k.replace("/", "__") + ".npz")).exists()}
    return frames.key.isin(ok).to_numpy()


def fold_ids(k: int) -> set:
    return set(splits.load(f"b2d/stoppos-cv5-fold{k}").members)


def fold_of(frames: pd.DataFrame) -> np.ndarray:
    fo = {}
    for k in range(K):
        for r in fold_ids(k):
            fo[r] = k
    return frames.id.map(fo).to_numpy()


def inner_cal(routes: pd.DataFrame, train_ids, seed: int, frac: float = 0.25) -> set:
    """25% of the training routes (stratified by set and kind, seeded) as the calibration set."""
    rng = np.random.default_rng(seed)
    r = routes[routes.id.isin(train_ids)]
    cal = []
    for _, g in r.groupby(["src", "kind"]):
        ids = np.array(sorted(g.id))
        rng.shuffle(ids)
        cal += list(ids[:max(1, int(round(frac * len(ids))))] if len(ids) > 3 else ids[:0])
    return set(cal)


def routes_table() -> pd.DataFrame:
    return pd.read_csv(REPO / "experiments/vlm_arb/results/stoppos_routes.csv")


def attempt_weights(frames: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """1 / (frames of the attempt inside the mask): every attempt has the same total weight (the evaluation's weighting)."""
    w = np.zeros(len(frames))
    g = frames.key[mask]
    n = g.map(g.value_counts())
    w[mask] = 1.0 / n.to_numpy()
    return w


# ------------------------------------------------------------------------------------------------ evaluation
def group_weights(df: pd.DataFrame) -> np.ndarray:
    """Weights of the frames of df (columns id, key) such that every route sums to 1 and every attempt of a route gets the same share."""
    n_att = df.groupby("id").key.transform("nunique")
    n_fr = df.groupby("key").key.transform("size")
    return (1.0 / n_att / n_fr).to_numpy()


def route_means(df: pd.DataFrame, col: str) -> pd.Series:
    """Per route: mean over attempts of the attempt's mean of `col`."""
    a = df.groupby(["id", "key"])[col].mean()
    return a.groupby(level=0).mean()


def wquantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


def bin_rows(df: pd.DataFrame, truth: str, err: str, label: dict, bins=BINS) -> list:
    """One row per true-distance bin: macro MAE with a route-clustered bootstrap CI, bias, weighted median / p90 of |err|."""
    rows = []
    for (lo, hi), name in zip(bins, BIN_NAMES):
        m = df[(df[truth] >= lo) & (df[truth] < hi if hi < 40 else df[truth] <= hi)]
        m = m.assign(ae=m[err].abs())
        if m.empty:
            rows.append(dict(**label, bin=name, routes=0, frames=0))
            continue
        mae = route_means(m, "ae")
        bias = route_means(m, err)
        w = group_weights(m)
        r = stats.bootstrap(mae.to_numpy())
        rows.append(dict(**label, bin=name, routes=len(mae), attempts=m.key.nunique(), frames=len(m), mae=r["mean"], lo=r["lo"], hi=r["hi"],
                         bias=float(bias.mean()), med=wquantile(m.ae, w, 0.5), p90=wquantile(m.ae, w, 0.9)))
    return rows


def auc(y, s, w=None):
    """Weighted ROC AUC (ties count half)."""
    y, s = np.asarray(y).astype(bool), np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if y.all() or (~y).all():
        return np.nan
    o = np.argsort(s, kind="mergesort")
    y, s, w = y[o], s[o], w[o]
    # unique score groups
    idx = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    wp, wn = np.add.reduceat(w * y, idx), np.add.reduceat(w * ~y, idx)
    cn = np.cumsum(wn) - wn
    return float(((wp * (cn + 0.5 * wn)).sum()) / (w[y].sum() * w[~y].sum()))
