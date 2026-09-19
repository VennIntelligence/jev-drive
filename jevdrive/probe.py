"""Steps 3-4: linear probes on every (backbone, layer, pooling) plus baselines, and the results table.

Every feature set gets the same probe: standardize -> multinomial logistic regression, with C picked by
scene-grouped inner CV on the training scenes (neg log-loss). Two protocols:
  val   official split: fit on train scenes, evaluate on val scenes
  loso  leave-one-scene-out over all scenes, predictions pooled (more eval samples, still few scenes)
Baselines: majority (train class prior, Laplace-smoothed), ego (same probe on past ego state) and ego_rule
(no training: extrapolate the current yaw rate over the horizon and apply the label thresholds).
Metrics on all eval samples and on the hard subset (|current yaw rate| small, see labels.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import CLASSES, get_logger, processed_dir
from .labels import EGO_COLS, HORIZON, N, THRESH_DEG

log = get_logger(__name__)
K = len(CLASSES)
CS = np.logspace(-4, 2, 13)
EPS = 1e-4


def load_features(version: str, tokens: pd.Series) -> dict[str, np.ndarray]:
    """All stored feature arrays, row-aligned to `tokens`: {'<backbone>/<name>': (n, d) float32}."""
    feats = {}
    for d in sorted((processed_dir(version) / "features").glob("*/")):
        pos = pd.Index(pd.read_parquet(d / "index.parquet").sample_token).get_indexer(tokens)
        if (pos < 0).any():
            log.warning("%s misses %d labeled samples, skipped", d.name, (pos < 0).sum())
            continue
        for f in sorted(d.glob("*.npy")):
            feats[f"{d.name}/{f.stem}"] = np.load(f, mmap_mode="r")[pos].astype(np.float32)
    return feats


def ego_rule(yaw_rate_now: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """One-hot (eps-smoothed) turn class from constant-yaw-rate extrapolation; its NLL only reflects eps."""
    dyaw = yaw_rate_now * HORIZON
    return np.eye(K)[np.select([dyaw > THRESH_DEG, dyaw < -THRESH_DEG], [0, 2], 1)] * (1 - K * eps) + eps


def fit_predict(X: np.ndarray | None, y: np.ndarray, groups: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Class probabilities (len(te), K) from a probe fit on `tr`. X=None is the class-prior baseline."""
    if X is None:
        p = np.bincount(y[tr], minlength=K) + 1.0
        return np.tile(p / p.sum(), (len(te), 1))
    if X.ndim == 1:  # 1-D input = current yaw rate for the ego_rule baseline
        return ego_rule(X[te])
    inner = list(GroupKFold(n_splits=min(4, len(np.unique(groups[tr])))).split(tr, groups=groups[tr]))
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=CS, cv=inner, scoring="neg_log_loss", l1_ratios=(0.0,), max_iter=3000, use_legacy_attributes=False))
    clf.fit(X[tr], y[tr])
    p = np.full((len(te), K), EPS)
    p[:, clf.classes_] = clf.predict_proba(X[te])
    return p / p.sum(1, keepdims=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred, turn = p.argmax(1), y != 1
    return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, labels=range(K), average="macro", zero_division=0),
            "nll": log_loss(y, p, labels=range(K)), "brier": ((p - np.eye(K)[y]) ** 2).sum(1).mean(),
            "turn_recall": (pred[turn] == y[turn]).mean() if turn.any() else np.nan, "n": len(y), "n_turn": int(turn.sum())}


def run(version: str, out_dir: Path, n_jobs: int = -1) -> pd.DataFrame:
    lab = pd.read_parquet(processed_dir(version) / "labels.parquet")
    y, groups, hard = lab.label.to_numpy(), lab.scene.to_numpy(), lab.hard.to_numpy()
    ego = lab[EGO_COLS].to_numpy(np.float32)
    feats = load_features(version, lab.sample_token)
    sets = {"majority": None, "ego_rule": lab[f"yaw_rate_{N - 1}"].to_numpy(), "ego": ego, **feats,
            **{f"{k}+ego": np.hstack([v, ego]) for k, v in feats.items() if k.startswith("qwen/")}}

    folds = {"val": [(np.flatnonzero(lab.split == "train"), np.flatnonzero(lab.split == "val"))]}
    folds["loso"] = [(np.flatnonzero(groups != s), np.flatnonzero(groups == s)) for s in np.unique(groups)]
    tasks = [(name, proto, tr, te) for name in sets for proto, fs in folds.items() for tr, te in fs]
    log.info("probing %d feature sets x %s folds = %d fits", len(sets), {k: len(v) for k, v in folds.items()}, len(tasks))
    probs = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(sets[n], y, groups, tr, te) for n, _, tr, te in tasks)

    rows, preds = [], {}
    for (name, proto, _, te), p in zip(tasks, probs):
        preds.setdefault((name, proto), []).append((te, p))
    for (name, proto), parts in preds.items():
        te, p = np.concatenate([t for t, _ in parts]), np.concatenate([q for _, q in parts])
        for subset, m in (("all", slice(None)), ("hard", hard[te])):
            rows.append({"set": name, "protocol": proto, "subset": subset, **metrics(y[te][m], p[m])})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.md").write_text(to_markdown(res))
    log.info("results -> %s\n%s", out_dir, to_markdown(res))
    return res


def to_markdown(res: pd.DataFrame) -> str:
    out = []
    order = list(dict.fromkeys(res.set))
    for proto in ("val", "loso"):
        r = res[res.protocol == proto]
        a = r[r.subset == "all"].set_index("set").loc[order]
        h = r[r.subset == "hard"].set_index("set").loc[order]
        t = pd.DataFrame({"acc": a.acc, "macro_f1": a.macro_f1, "nll": a.nll, "brier": a.brier, "turn_rec": a.turn_recall,
                          "hard_acc": h.acc, "hard_nll": h.nll, "hard_turn_rec": h.turn_recall})
        n = a.iloc[0]
        out.append(f"### {proto}: n={n.n} (turns {n.n_turn}), hard n={h.iloc[0].n} (turns {h.iloc[0].n_turn})\n\n"
                   + t.to_markdown(floatfmt=".3f") + "\n")
    return "\n".join(out)
