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

from . import traj, waymo
from .common import get_logger

log = get_logger(__name__)
DEV = "cuda"


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


def to_markdown(res: pd.DataFrame) -> str:
    t = res.set_index("K")[["n_eval", "oracle_minade", "oracle_minfde", "n_rater", "rfs_uncoverable",
                            "oracle_minade_pre_onset", "oracle_minade_straight_yaw"]]
    return ("Anchors from k-means on the fit half's logged futures; everything measured on the eval half.\n"
            "`rfs_uncoverable` is the fraction of rater-scored frames where **no** anchor lands inside "
            "**any** rater trust region, so the frame is floored whatever the scorer picks -- the coverage "
            "floor in the metric's own terms, which the oracle minADE cannot see.\n\n"
            + t.to_markdown(floatfmt=".4f") + "\n")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="split,vocab", help="comma list of split,vocab")
    ap.add_argument("--ks", default="64,256,1024,4096,8192")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
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
    (rl.dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
