"""P0 on Waymo E2E: decisions 3d and 20 again, with the head fitted on the train split.

Decisions 1, 3d and 20 all end on the same qualifier -- the head has only ever been trained on 239 val
sequences, and the whole thing has to be redone once train lands. Train landed (263/263 shards, 415663
frames, 2037 sequences), so this redoes it. Nothing about the protocol changes except the training data:
the same feature set and layer, the same late fusion, the same sequence-grouped CV for lambda, the same
paired sequence bootstrap, the same subset definitions, the same 0.05 m threshold. The half-val split of
`waymo_stage_a.val_halves` is replaced by the real one -- fit on every train frame, evaluate on every val
frame -- and `waymo_stage_a.Halves` takes those masks unchanged, so the inner selection split is the same
shape it was.

Arms, all on top of the same frozen `ridge ego` base (decisions 20 already judged the rest of the B family:
sq and top25 damage every subset at once, so only the mildest weighting is carried over):

  A  uniform `ridge_late`, the quantity decisions 3d is about
  B  `ego:a1`, the weighting the fit half chose there, now chosen by the same rule inside train
  C  `ridge_late` fitted on train's pre-onset frames only -- about 10x the frames arm C had on half-val
  D  the MLP head, uniform and under the same weighting
  plus `cls ego` and `cls_late vision+ego`, which carry the RFS row and the classifier's own DiD

Everything runs on the CPU: the GPU is extracting features for another experiment. Two consequences are
worth knowing. `ce_solve` has no `mem_get_info` there (planner.CPU_SOLVE_BUDGET stands in), and its backward
pass multiplies by softmax outputs that go denormal once the logits grow, which costs a factor of 70 on x86
until flush-to-zero is enabled -- see `use_cpu`.
"""
import json

import numpy as np
import pandas as pd
import torch

from . import planner, probe, traj, waymo, waymo_l0 as l0, waymo_stage_a as sa
from .common import get_logger, n_cpus

log = get_logger(__name__)
SUBSETS = sa.SUBSETS
BASE = l0.BASE
# The arms decisions 20 says are worth repeating with ten times the data, plus the classifier heads that
# carry RFS. The rest of the B family and both control families are settled and not re-run.
WEIGHT_SCHEME, WEIGHT_FAMILY = "ego:a1", {"ego": ("a1",)}
K_REF = 1024


def use_cpu() -> None:
    """Point every solver at the CPU and switch on flush-to-zero.

    The arms are linear algebra on at most 2560 dimensions, so the CPU is a fine place for them, and the GPU
    is busy. Flush-to-zero is not a numerical liberty: `ce_solve` multiplies the design matrix by
    `softmax - target`, whose entries fall below 1e-38 for every anchor the model is confident about, and an
    x86 core takes a microcode trap on each of those. Measured on the box, one backward GEMM of the
    classifier (20000 x 2560 x 1024) takes 3.12 s with denormals and 0.043 s without, for the same result to
    float32 precision.
    """
    if torch.cuda.is_available():
        raise SystemExit("run this with CUDA_VISIBLE_DEVICES= : the GPU belongs to another experiment")
    planner.DEV = probe.DEV = traj.DEV = sa.DEV = l0.DEV = "cpu"
    torch.set_num_threads(n_cpus())
    torch.set_flush_denormal(True)
    log.info("CPU mode: %d threads, flush-to-zero on", n_cpus())


def load_all(set_name: str = "qwen_front3", layer: str = sa.LAYER):
    """`waymo_stage_a.load_all` over train and val instead of val alone.

    Same join on `frame_name` (a row position is a position into the index as it stood when that shard was
    extracted, so it is not a key), same filter on a usable future and an extracted feature row.
    """
    df = waymo.load_index()
    past, future = waymo.load_ego()
    idx, arrs = waymo.load_features(set_name, [layer])
    at = pd.Series(np.arange(len(idx)), index=idx.frame_name.to_numpy())
    pos = at.reindex(waymo.frame_names(df)).to_numpy()
    keep = df.split.isin(("train", "val")).to_numpy() & df.has_future.to_numpy() & ~np.isnan(pos)
    rows = np.flatnonzero(keep)
    order = pos[rows].astype(int)
    sub = {k: v[rows] for k, v in waymo.subsets(df, past, future).items()}
    ego = np.concatenate([waymo.ego_state(past[rows]), waymo.intent_onehot(df.iloc[rows])], 1)
    split = df.split.to_numpy()[rows]
    log.info("rows: %d frames with a future and a feature (%d indexed); train %d / %d sequences, val %d / %d",
             len(rows), len(df), int((split == "train").sum()), df.sequence[rows[split == "train"]].nunique(),
             int((split == "val").sum()), df.sequence[rows[split == "val"]].nunique())
    log.info("subsets on val: %s", {k: int((sub[k] & (split == "val")).sum()) for k in SUBSETS})
    return df, rows, waymo.future_xy(future[rows]), ego.astype(np.float32), arrs[layer][order], sub, past, split


def split_of(df: pd.DataFrame, rows: np.ndarray, split: np.ndarray, seed: int = 0) -> sa.Halves:
    """The real split as a `Halves`: fit on every train frame, evaluate on every val frame."""
    seq = df.sequence.to_numpy()[rows]
    sp = sa.Halves(df, seq, split == "train", split == "val", seed)
    log.info("train fit %d frames / %d sequences (inner sel %d), val eval %d / %d", len(sp.train),
             len(np.unique(seq[sp.train])), len(sp.sel), len(sp.val), len(np.unique(seq[sp.val])))
    return sp


def run_arms(set_name: str, layer: str, seed: int, folds: int, steps: str):
    """Fit the regression arms and return (predictions, context, the weighting table).

    The context carries the standardised inputs and the ego residual, because `fit_cls` runs afterwards on
    the same ones and neither standardising nor refitting the base twice would be free at this size.
    """
    df, rows, fut, ego, img, sub, past, split = load_all(set_name, layer)
    sp = split_of(df, rows, split, seed)
    n, T = len(rows), fut.shape[1]
    pre = sub["pre_onset"]

    s = {"ego": l0.ego_surprise(ego, fut, sp, folds)}
    F = torch.as_tensor(fut.reshape(n, -1))
    Xe = planner.standardize(torch.as_tensor(ego), sp.train)
    Xi = planner.standardize(torch.from_numpy(np.asarray(img)).float(), sp.train)

    preds = {}
    p_ego, st_ego, W_ego = sa.ridge_cv(Xe, F, sp, fut)
    preds[BASE] = (p_ego, st_ego)
    base = planner.linear_apply(W_ego, Xe, np.arange(n))[0]
    R = F - base
    res_fut = R.reshape(-1, T, 2).cpu().numpy()
    off = base[sp.val].reshape(-1, 1, T, 2).cpu().numpy()
    log.info("%s: lambda %.3g, val ADE %.4f", BASE, st_ego["lam"], l0.ade(p_ego[:, 0], fut[sp.val]).mean())

    w1 = torch.ones(n)
    ws = {"uniform": w1,
          WEIGHT_SCHEME: torch.as_tensor(l0.weight_schemes(s, sp.train, WEIGHT_FAMILY)[WEIGHT_SCHEME])}
    brows = []
    for name, wv in ws.items():
        p, st, _ = l0.wridge_cv(Xi, R, sp, res_fut, wv, pre, folds)
        arm = "A ridge_late uniform" if name == "uniform" else f"B {name}"
        preds[arm] = (p + off, st)
        brows.append({"scheme": name, "arm": arm, "eff_n": float(wv[sp.train].sum().item() ** 2
                      / wv[sp.train].square().sum().item()), **st})
        log.info("%-22s lam %.3g  weighted sel ADE %.4f  train CV pre_onset ADE %.4f (n_fit %d)",
                 arm, st["lam"], st["sel_ade"], st["sel_pre_ade"], st["n_fit"])
    p, st, _ = l0.wridge_cv(Xi, R, sp, res_fut, w1, pre, folds, rows=np.flatnonzero(pre))
    preds["C pre_onset fit"] = (p + off, st)
    log.info("%-22s lam %.3g  fitted on %d pre_onset frames", "C pre_onset fit", st["lam"], st["n_fit"])
    if "mlp" in steps:
        for arm, wv in (("D mlp uniform", w1), (f"D mlp {WEIGHT_SCHEME}", ws[WEIGHT_SCHEME])):
            p, st = l0.mlp_late(Xi, sp, res_fut, R, wv, seed)
            preds[arm] = (p + off, st)
            log.info("%-22s %d epochs, inner-val ADE %.4f", arm, st["epochs"], st["sel_ade"])

    ctx = {"df": df, "rows": rows, "fut": fut, "sub": sub, "seq": df.sequence.to_numpy()[rows], "sp": sp,
           "past": past, "direction": -1, "s": s, "best_scheme": WEIGHT_SCHEME, "F": F, "Xe": Xe, "Xi": Xi}
    return preds, ctx, pd.DataFrame(brows)


def fit_cls(preds: dict, ctx: dict, seed: int = 0, k: int = K_REF) -> torch.Tensor:
    """Add `cls ego` and `cls_late vision+ego` to `preds` and return the vocabulary they score.

    These are the two heads the RFS row of decisions 3d is reported on, and they are what makes a
    multi-modal judge possible at all in P1, since a regression head has one mode. They are also by far the
    most expensive thing here -- a batched L-BFGS over 1024 anchors and 414k frames on the CPU -- which is
    why they run after the regression tables are already on disk.
    """
    sp, F, fut = ctx["sp"], ctx["F"], ctx["fut"]
    vocab = traj.kmeans(F[sp.train], k, seed=seed)
    ids, _ = traj.nearest(F, vocab, planner.SOFT_M)
    hard = np.concatenate([np.ones((len(F), 1)), np.zeros((len(F), planner.SOFT_M - 1))], 1).astype(np.float32)
    he = planner.Heads(ctx["Xe"], sp, fut, F, waymo.RFS_FREQ)
    preds["cls ego"] = he.cls((ids, hard), vocab, keep=True)
    log.info("cls ego: lambda %.3g, inner-val ADE %.4f", *(preds["cls ego"][1][q] for q in ("lam", "sel_ade")))
    preds["cls_late vision+ego"] = planner.Heads(ctx["Xi"], sp, fut, F, waymo.RFS_FREQ).cls(
        (ids, hard), vocab, offset=he.scores)
    log.info("cls_late: lambda %.3g, inner-val ADE %.4f",
             *(preds["cls_late vision+ego"][1][q] for q in ("lam", "sel_ade")))
    return vocab


def save_preds(preds: dict, ctx: dict, out_dir, vocab=None) -> None:
    """Every arm's per-frame prediction on the evaluation split, so any judge can be applied without refitting.

    P1 is exactly that: it changes the judge, not the model, and refitting a 2560-dimensional head to change
    a metric would be both slow and a chance to accidentally change the model too. The classifier heads keep
    all of their top-k anchors, because one of the candidate judges is a minADE over them; the regression
    heads have a single mode and store one trajectory.
    """
    sp, rows = ctx["sp"], ctx["rows"]
    extra = {f"topk_{name}": p.astype(np.float32) for name, (p, _) in preds.items() if p.shape[1] > 1}
    if vocab is not None:
        extra["vocab"] = vocab.reshape(len(vocab), -1, 2).cpu().numpy().astype(np.float32)
    np.savez_compressed(
        out_dir / "preds.npz", frame_name=waymo.frame_names(ctx["df"])[rows[sp.val]],
        sequence=ctx["seq"][sp.val], s_ego=ctx["s"]["ego"][sp.val], fut=ctx["fut"][sp.val],
        speed=waymo.init_speed(ctx["past"][rows[sp.val]]),
        cluster=ctx["df"].cluster.astype(str).to_numpy()[rows[sp.val]],
        **{f"sub_{k}": ctx["sub"][k][sp.val] for k in SUBSETS},
        **{f"pred_{name}": p[:, 0].astype(np.float32) for name, (p, _) in preds.items()}, **extra)
    log.info("per-frame predictions of %d arms on %d evaluation frames -> %s", len(preds), len(sp.val),
             out_dir / "preds.npz")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="mlp,cls", help="comma list of mlp,cls; the ridge arms always run")
    ap.add_argument("--layer", default=sa.LAYER)
    ap.add_argument("--feature-set", default="qwen_front3")
    ap.add_argument("--folds", type=int, default=l0.FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    use_cpu()
    rl = RunLog("waymo_p0", "train_split")
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    preds, ctx, schemes = run_arms(a.feature_set, a.layer, a.seed, a.folds, a.steps)
    vocab = None

    def report():
        """Write every table and the per-frame predictions as they stand. Run once before the classifier
        heads start, so a four-hour L-BFGS cannot hold the numbers decisions 3d is waiting for."""
        res, pairs, dids, dec = l0.evaluate(preds, ctx)
        save_preds(preds, ctx, rl.dir, vocab)
        l0._write(rl, (("arms", [res]), ("paired", [pairs]), ("did", [dids]), ("deciles", [dec]),
                       ("schemes", [schemes])))
        try:   # a broken figure must not lose the tables
            from . import plots
            plots.l0_decile_curve(dec, rl.dir, [a for a in plots.ARM_LABEL if a in preds],
                                  ((-1, "-"),), "p0-decile-relative-gain")
        except Exception as e:  # noqa: BLE001
            log.warning("figure failed: %s", e)

    report()
    if "cls" in a.steps:
        vocab = fit_cls(preds, ctx, a.seed)
        report()
    (rl.dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
