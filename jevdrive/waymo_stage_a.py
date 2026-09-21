"""Stage A on Waymo E2E: the half-val dress rehearsal (todos/2026-09-20-waymo-stage-a-dryrun.md).

Waymo's train split is not downloaded, so the rehearsal splits val itself: one half of the sequences trains
the head, the other is evaluated. That is a rehearsal and not the paper's number -- the paper trains on the
train split -- but it is the only way to reach the measurement decisions 3d pre-registers before train lands.

  val_halves   sequence-disjoint halves of val, stratified by scenario cluster
  vocab_sweep  the vocabulary table: oracle minADE/minFDE against the logged future, and the fraction of
               rater-scored frames no anchor in the vocabulary can bring inside any rater trust region

The coverage column is the point. On nuScenes the same sweep read one way in ADE (64 anchors look plenty)
and the opposite way in trust-region terms (64 anchors forfeit 16 % of frames outright), and here it is
computed against the real RFS geometry rather than a stand-in, because the rater trajectories are local.
Neither needs a single image feature, so this table runs while everything else waits on the download.
"""
import json

import numpy as np
import pandas as pd
import torch

from . import planner, traj, waymo
from .common import get_logger

log = get_logger(__name__)
DEV = "cuda"
LAYER = "L18_mean"   # the extracted layer closest to the L19-L22 band decisions 5 fixed on nuScenes
SUBSETS = ("all", "pre_onset", "straight_yaw", "turn_yaw")


def val_halves(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Sequence -> 0 (fit) or 1 (eval), stratified by scenario cluster, alternating within each cluster.

    Balanced on sequence count, which is also rater-frame count because every val sequence carries exactly
    one rater-scored frame. Deliberately *not* balanced on pre-onset frame count: that count depends on how
    many shards have landed, so balancing on it would silently move the split every time the download
    advances. The cluster map covers all 479 sequences from the start, so this split is final now. The
    realised pre-onset counts per half are reported instead of being forced.

    Stratifying by cluster is not cosmetic: RFS averages per cluster and then over the clusters present, so
    a half missing a cluster is not measuring the same thing as the other half.
    """
    seq = df[df.split == "val"].drop_duplicates("sequence")[["sequence", "cluster"]]
    rng = np.random.default_rng(seed)
    half = {}
    for c, g in seq.sort_values("sequence").groupby("cluster", sort=True):
        s = g.sequence.to_numpy()
        half |= {q: i % 2 for i, q in enumerate(s[rng.permutation(len(s))])}
    out = pd.Series(half, name="half").sort_index()
    log.info("val halves: %d sequences over %d clusters -> %s", len(out), seq.cluster.nunique(),
             out.value_counts().sort_index().to_dict())
    return out


def half_report(df: pd.DataFrame, past: np.ndarray, future: np.ndarray, halves: pd.Series) -> pd.DataFrame:
    """Per half: sequences, indexed frames, rater frames and the manoeuvre subsets, plus the cluster spread."""
    h = df.sequence.map(halves).to_numpy()
    val = (df.split == "val").to_numpy() & df.has_future.to_numpy()
    sub = waymo.subsets(df, past, future)
    rater = np.zeros(len(df), bool)
    rater[waymo.load_rater()[0]] = True
    rows = []
    for i in (0, 1):
        m = val & (h == i)
        rows.append({"half": i, "role": ("fit", "eval")[i], "sequences": df.sequence[m].nunique(),
                     "clusters": df.cluster[m].nunique(), "frames": int(m.sum()),
                     "rater_frames": int((m & rater).sum()),
                     **{k: int((m & sub[k]).sum()) for k in ("pre_onset", "straight_yaw", "turn_yaw")}})
    return pd.DataFrame(rows)


def vocab_coverage_rfs(anchors: torch.Tensor, rows: np.ndarray, chunk: int = 8) -> np.ndarray:
    """Per rater-scored frame: 1.0 when no anchor in the whole vocabulary lands inside any rater's trust
    region, i.e. the frame is floored whatever the scorer picks. This is the coverage floor in the metric's
    own terms and it is what the oracle minADE cannot see."""
    r, rtraj, scores = waymo.load_rater()
    keep = np.isin(r, rows)
    r, rtraj, scores = r[keep], rtraj[keep], scores[keep]
    if not len(r):
        return np.zeros(0)
    past, _ = waymo.load_ego()
    speed = waymo.init_speed(past[r])
    A = anchors.reshape(len(anchors), -1, 2).cpu().numpy().astype(np.float32)
    out = []
    for a in range(0, len(r), chunk):  # (frames, K anchors, 20, 2) against 3 rater trajectories each
        pred = np.broadcast_to(A, (len(r[a:a + chunk]), *A.shape))
        _, inside = waymo.rater_feedback_score(pred, rtraj[a:a + chunk], scores[a:a + chunk],
                                               speed[a:a + chunk], details=True)
        out.append(~inside.any(1))
    return np.concatenate(out).astype(float)


def vocab_sweep(ks=(64, 256, 1024, 4096, 8192), seed: int = 0, out_dir=None, rl=None) -> pd.DataFrame:
    """The vocabulary table. Anchors are k-means over the fit half's logged futures; everything is measured
    on the eval half, so nothing the table reports has seen its own training trajectories."""
    df = waymo.load_index()
    past, future = waymo.load_ego()
    halves = val_halves(df, seed)
    h = df.sequence.map(halves).to_numpy()
    val = (df.split == "val").to_numpy() & df.has_future.to_numpy()
    fit, ev = np.flatnonzero(val & (h == 0)), np.flatnonzero(val & (h == 1))
    gt = waymo.future_xy(future[ev])
    seqs, sub = df.sequence.to_numpy()[ev], waymo.subsets(df, past, future)
    log.info("vocabulary: fit on %d frames / %d sequences, evaluate on %d frames / %d sequences",
             len(fit), df.sequence[fit].nunique(), len(ev), df.sequence[ev].nunique())

    F = torch.as_tensor(waymo.future_xy(future[fit]).reshape(len(fit), -1), device=DEV)
    rows = []
    for k in ks:
        C = traj.kmeans(F, k, seed=seed)
        o = traj.oracle_metrics(C, gt)
        cov = vocab_coverage_rfs(C, ev)
        r = {"K": k, "n_eval": len(ev), "n_rater": len(cov),
             "oracle_minade": o["oracle_ade"].mean(), "oracle_minfde": o["oracle_fde"].mean(),
             "rfs_uncoverable": cov.mean() if len(cov) else np.nan}
        r["oracle_minade_lo"], r["oracle_minade_hi"] = traj.boot_ci(o["oracle_ade"], seqs)
        for name in ("pre_onset", "straight_yaw"):
            m = sub[name][ev]
            r[f"oracle_minade_{name}"] = o["oracle_ade"][m].mean() if m.any() else np.nan
            r[f"n_{name}"] = int(m.sum())
        rows.append(r)
        log.info("K=%-5d oracle minADE %.3f minFDE %.3f | no anchor inside any rater region: %.3f (n=%d)",
                 k, r["oracle_minade"], r["oracle_minfde"], r["rfs_uncoverable"], r["n_rater"])
        if out_dir is not None:
            np.save(out_dir / f"vocab_K{k}.npy", C.reshape(k, -1, 2).cpu().numpy())
        if rl is not None:
            rl.event("vocab", **r)
    res = pd.DataFrame(rows)
    if out_dir is not None:
        res.to_csv(out_dir / "vocab_sweep.csv", index=False)
        (out_dir / "vocab_sweep.md").write_text(to_markdown(res))
    log.info("\n%s", to_markdown(res))
    return res


class Halves:
    """One direction of the half-val rehearsal: which rows fit the head, which evaluate it, and the inner
    split of the fit half that picks each head's regularisation. Everything is grouped by sequence."""

    def __init__(self, df, seq, fit_mask, eval_mask, seed=0, inner=0.2):
        self.seq = seq
        self.train, self.val = np.flatnonzero(fit_mask), np.flatnonzero(eval_mask)
        from sklearn.model_selection import GroupShuffleSplit
        f, v = next(GroupShuffleSplit(1, test_size=inner, random_state=seed)
                    .split(self.train, groups=seq[self.train]))
        self.fit, self.sel = self.train[f], self.train[v]
        self.scenes = seq  # planner.Heads calls it scenes; on Waymo the group is the sequence


def load_all(set_name: str = "qwen_front3", layer: str = LAYER, seed: int = 0):
    """Index, targets, ego input, image features and the manoeuvre subsets, all restricted to val frames
    that have a future and an extracted feature row."""
    df = waymo.load_index()
    past, future = waymo.load_ego()
    idx, arrs = waymo.load_features(set_name, [layer])
    # join on frame_name: `row` is a position into the index as it was when the shard was extracted
    fname = (df.sequence.astype(str) + "-" + df.frame.map("{:03d}".format)).to_numpy()
    at = pd.Series(np.arange(len(idx)), index=idx.frame_name.to_numpy())
    pos = at.reindex(fname).to_numpy()
    keep = (df.split == "val").to_numpy() & df.has_future.to_numpy() & ~np.isnan(pos)
    rows = np.flatnonzero(keep)
    order = pos[rows].astype(int)
    sub = {k: v[rows] for k, v in waymo.subsets(df, past, future).items()}
    ego = np.concatenate([waymo.ego_state(past[rows]), waymo.intent_onehot(df.iloc[rows])], 1)
    log.info("rows: %d val frames with a future and a feature (%d indexed); subsets %s", len(rows), len(df),
             {k: int(sub[k].sum()) for k in SUBSETS})
    return df, rows, waymo.future_xy(future[rows]), ego.astype(np.float32), arrs[layer][order], sub, past


def rfs_rows(df, rows):
    """(positions into `rows`, rater trajectories, scores) for the rater-scored frames among them."""
    r, rtraj, scores = waymo.load_rater()
    pos = pd.Series(np.arange(len(rows)), index=rows).reindex(r).to_numpy()
    ok = ~np.isnan(pos)
    return pos[ok].astype(int), rtraj[ok], scores[ok]


def ridge_cv(X, Y, sp, fut, folds: int = 4, seed: int = 0):
    """Ridge with the L2 strength chosen by `folds`-fold sequence-grouped CV inside the fit half.

    A single 20 % holdout is what planner.Heads.ridge does and it is enough on nuScenes' 700 training
    scenes. Here the fit half is 241 sequences, one holdout is 48 of them, and that was not representative:
    direction 1 picked the bottom of the grid on an inner-val ADE of 1.80 and then scored 3.54 on the
    evaluation half, while direction 0 picked a middling lambda and scored 1.88. Averaging the selection
    over folds costs one extra eigendecomposition per fold and removes that lottery."""
    from sklearn.model_selection import GroupKFold
    d, T = X.shape[1], fut.shape[1]
    score = np.zeros(len(planner.LAM_RIDGE))
    for tr, te in GroupKFold(folds).split(sp.train, groups=sp.seq[sp.train]):
        W = planner.ridge_solve(X, Y, sp.train[tr], planner.LAM_RIDGE)
        p = planner.linear_apply(W, X, sp.train[te]).reshape(len(planner.LAM_RIDGE), -1, T, 2).cpu().numpy()
        score += [np.linalg.norm(p[i] - fut[sp.train[te]], axis=-1).mean() for i in range(len(score))]
    best = planner._pick(score / folds, planner.LAM_RIDGE, "ridge (grouped CV)")
    W = planner.ridge_solve(X, Y, sp.train, [planner.LAM_RIDGE[best]])
    pv = planner.linear_apply(W, X, sp.val).reshape(-1, T, 2).cpu().numpy()
    return pv[:, None], {"lam": float(planner.LAM_RIDGE[best]), "sel_ade": float(score[best] / folds)}, W


def heads(Xi, Xe, sp, fut, F, k_ref, vocab, tgt, rate=waymo.RFS_FREQ):
    """ego-only and vision-on-top-of-ego, as regression and as vocabulary classification.

    Vision enters by late fusion in both: the ego head is fitted first and its output -- the predicted
    waypoints for ridge, the anchor logits for the classifier -- is frozen as an offset the image head only
    corrects. planner v0 showed the PCA concatenation does not work for the classifier, and a shared L2
    strength cannot regularise 2560 image dimensions and 100 ego dimensions at once (decisions 10)."""
    T = fut.shape[1]
    out = {}
    he = planner.Heads(Xe, sp, fut, F, rate)
    p_ego, st_ego, W_ego = ridge_cv(Xe, F, sp, fut)
    out["ridge ego"] = (p_ego, st_ego)
    base = planner.linear_apply(W_ego, Xe, np.arange(len(Xe)))[0]
    res_fut = (F - base).reshape(-1, T, 2).cpu().numpy()
    p_lat, st_lat, _ = ridge_cv(Xi, F - base, sp, res_fut)
    out["ridge_late vision+ego"] = (p_lat + base[sp.val].reshape(-1, 1, T, 2).cpu().numpy(), st_lat)

    p_ec, st_ec = he.cls(tgt, vocab, keep=True)
    out["cls ego"] = (p_ec, st_ec)
    p_cl, st_cl = planner.Heads(Xi, sp, fut, F, rate).cls(tgt, vocab, offset=he.scores)
    out["cls_late vision+ego"] = (p_cl, st_cl)
    return out


def measure(direction: int, set_name: str = "qwen_front3", layer: str = LAYER, k_ref: int = 1024,
            seed: int = 0, out_dir=None, rl=None):
    """One direction of the rehearsal. Returns (per-head val predictions, the rows, the context)."""
    df, rows, fut, ego, img, sub, past = load_all(set_name, layer, seed)
    halves = val_halves(df, seed)
    h = df.sequence.map(halves).to_numpy()[rows]
    fit_mask, eval_mask = h == direction, h == (1 - direction)
    seq = df.sequence.to_numpy()[rows]
    sp = Halves(df, seq, fit_mask, eval_mask, seed)
    log.info("direction %d: fit %d frames / %d sequences, eval %d / %d", direction, len(sp.train),
             len(np.unique(seq[sp.train])), len(sp.val), len(np.unique(seq[sp.val])))

    n, T = len(rows), fut.shape[1]
    F = torch.as_tensor(fut.reshape(n, -1), device=DEV)
    Xe = planner.standardize(torch.as_tensor(ego, device=DEV), sp.train)
    Xi = planner.standardize(torch.from_numpy(np.asarray(img)).to(DEV).float(), sp.train)
    vocab = traj.kmeans(F[sp.train], k_ref, seed=seed)
    ids, err = traj.nearest(F, vocab, planner.SOFT_M)
    tgt = (ids, np.concatenate([np.ones((n, 1)), np.zeros((n, planner.SOFT_M - 1))], 1).astype(np.float32))
    preds = heads(Xi, Xe, sp, fut, F, k_ref, vocab, tgt)
    ctx = {"df": df, "rows": rows, "fut": fut, "sub": sub, "seq": seq, "sp": sp, "past": past,
           "vocab": vocab, "k_ref": k_ref, "direction": direction}
    return preds, ctx


def evaluate(preds, ctx, out_dir=None, rl=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three tables: per head by subset (ADE, and RFS where the raters are), paired deltas, and the DiD.

    ADE on the subsets is against the logged future, which is all a non-rater frame has. RFS and the ADE
    beside it are on the rater-scored frames of the evaluation half, against the raters, as the metric
    defines them."""
    sp, sub, seq, fut = ctx["sp"], ctx["sub"], ctx["seq"], ctx["fut"]
    v, gt, sq = sp.val, fut[sp.val], seq[sp.val]
    masks = {k: sub[k][v] for k in SUBSETS}
    pos, rtraj, scores = rfs_rows(ctx["df"], ctx["rows"][v])
    speed = waymo.init_speed(ctx["past"][ctx["rows"][v]])
    cluster = ctx["df"].cluster.astype(str).to_numpy()[ctx["rows"][v]]
    per, rows_out = {}, []
    for name, (p, st) in preds.items():
        e = np.linalg.norm(p[:, 0] - gt, axis=-1)
        m = {"ade": e.mean(1), "ade3": e[:, :3 * waymo.RFS_FREQ].mean(1), "fde": e[:, -1]}
        if len(pos):
            sc = waymo.rater_feedback_score(p[pos, 0], rtraj, scores, speed[pos])
            best = rtraj[np.arange(len(pos)), scores.argmax(1)]
            m["rfs"] = np.full(len(gt), np.nan)
            m["rfs"][pos] = sc
            m["ade_rater"] = np.full(len(gt), np.nan)
            m["ade_rater"][pos] = np.linalg.norm(p[pos, 0] - best, axis=-1).mean(1)
        per[name] = m
        r = {"head": name, "direction": ctx["direction"], **{f"fit_{k}": w for k, w in st.items()}}
        for sn, mask in masks.items():
            r[f"ade_{sn}"] = m["ade"][mask].mean()
            r[f"n_{sn}"] = int(mask.sum())
        if len(pos):
            r["rfs"], r["n_rater"] = sc.mean(), len(pos)
            r["rfs_cluster"] = waymo.rfs_by_cluster(sc, cluster[pos])[0]
            r["rfs_lo"], r["rfs_hi"] = traj.boot_ci(sc, sq[pos])
            r["ade_rater"] = m["ade_rater"][pos].mean()
            r["floored"] = float((sc <= waymo.RFS_FLOOR + 1e-9).mean())
        rows_out.append(r)
        log.info("dir %d %-24s RFS %.3f  ADE %.3f | pre_onset %.3f (n=%d)  straight %.3f (n=%d)",
                 ctx["direction"], name, r.get("rfs", np.nan), r["ade_all"], r["ade_pre_onset"],
                 r["n_pre_onset"], r["ade_straight_yaw"], r["n_straight_yaw"])

    pairs, dids = [], []
    for a, b in (("ridge ego", "ridge_late vision+ego"), ("cls ego", "cls_late vision+ego")):
        d = per[b]["ade"] - per[a]["ade"]
        for sn, mask in masks.items():
            lo, hi = traj.boot_ci(d[mask], sq[mask])
            pairs.append({"direction": ctx["direction"], "from": a, "to": b, "subset": sn,
                          "n": int(mask.sum()), "ade_from": per[a]["ade"][mask].mean(),
                          "ade_to": per[b]["ade"][mask].mean(), "dade": d[mask].mean(),
                          "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2})
        hi_m, lo_m = masks["pre_onset"], masks["straight_yaw"]
        point, cl, ch = traj.boot_did(d, sq, hi_m, lo_m)
        dids.append({"direction": ctx["direction"], "from": a, "to": b, "hi": "pre_onset",
                     "lo": "straight_yaw", "n_hi": int(hi_m.sum()), "n_lo": int(lo_m.sum()),
                     "dade_hi": d[hi_m].mean(), "dade_lo": d[lo_m].mean(), "did": point,
                     "did_lo": cl, "did_hi": ch, "halfwidth": (ch - cl) / 2})
        log.info("dir %d DiD %s: %+.4f [%+.4f, %+.4f] half-width %.4f (pre_onset %+.4f n=%d, "
                 "straight %+.4f n=%d)", ctx["direction"], b, point, cl, ch, (ch - cl) / 2,
                 d[hi_m].mean(), int(hi_m.sum()), d[lo_m].mean(), int(lo_m.sum()))
    return pd.DataFrame(rows_out), pd.DataFrame(pairs), pd.DataFrame(dids)


def to_markdown(res: pd.DataFrame) -> str:
    t = res.set_index("K")[["n_eval", "oracle_minade", "oracle_minfde", "n_rater", "rfs_uncoverable",
                            "oracle_minade_pre_onset", "oracle_minade_straight_yaw"]]
    return ("Anchors from k-means on the fit half's logged futures; everything measured on the eval half.\n"
            "`rfs_uncoverable` is the fraction of rater-scored frames where **no** anchor lands inside "
            "**any** rater trust region, so the frame is floored whatever the scorer picks -- the coverage "
            "floor in the metric's own terms, which the oracle minADE cannot see.\n\n"
            + t.to_markdown(floatfmt=".4f") + "\n")


def feature_status(name: str = "qwen_front3", split: str = "val") -> dict:
    """What the pipeline can see, not just what it did: shards sitting on disk, shards the index knows
    about, shards whose features are built. A pass that reports "0 to extract, 29 done" looks healthy when
    the truth is "42 shards sitting there unseen", so the loop prints all three every time."""
    on_disk = len([p for p in waymo.shard_dir().glob(f"{split}_*.tfrecord-*") if p.is_file()])
    try:
        df = waymo.load_index()
        df = df[df.split == split]
        indexed, frames = df.shard.nunique(), len(df)
    except (FileNotFoundError, OSError):
        indexed, frames = 0, 0
    root = waymo.out_dir("features", name)
    built = len([d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()]) if root.exists() else 0
    return {"on_disk": on_disk, "indexed": indexed, "built": built, "frames_indexed": frames}


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="split,vocab", help="comma list of split,vocab,measure,status")
    ap.add_argument("--k-ref", type=int, default=1024)
    ap.add_argument("--layer", default=LAYER)
    ap.add_argument("--feature-set", default="qwen_front3")
    ap.add_argument("--directions", default="0,1", help="which half trains the head; both are reported")
    ap.add_argument("--ks", default="64,256,1024,4096,8192")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", default="qwen_front3", help="status: the feature set to count")
    a = ap.parse_args()
    if a.steps == "status":  # no RunLog: this is polled every few minutes by the extraction loop
        st = feature_status(a.set)
        print(" ".join(str(st[k]) for k in ("on_disk", "indexed", "built", "frames_indexed")))
        return
    rl = RunLog("waymo_stage_a", "half_val")
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    df = waymo.load_index()
    past, future = waymo.load_ego()
    if "split" in a.steps:
        rep = half_report(df, past, future, val_halves(df, a.seed))
        rep.to_csv(rl.dir / "halves.csv", index=False)
        rl.log.info("halves\n%s", rep.to_markdown(index=False))
        rl.event("halves", rows=rep.to_dict("records"))
    if "vocab" in a.steps:
        vocab_sweep(tuple(int(k) for k in a.ks.split(",")), a.seed, rl.dir, rl)
    if "measure" in a.steps:
        res, pairs, dids = [], [], []
        for d in (int(x) for x in a.directions.split(",")):
            preds, ctx = measure(d, a.feature_set, a.layer, a.k_ref, a.seed, rl.dir, rl)
            r, p_, dd = evaluate(preds, ctx, rl.dir, rl)
            res.append(r), pairs.append(p_), dids.append(dd)
        res, pairs, dids = (pd.concat(x, ignore_index=True) for x in (res, pairs, dids))
        for name, t in (("heads", res), ("paired", pairs), ("did", dids)):
            t.to_csv(rl.dir / f"{name}.csv", index=False)
            rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
            rl.event(name, rows=t.to_dict("records"))
    (rl.dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
