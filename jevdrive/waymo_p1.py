"""P1 on Waymo E2E: which judge should the "critical moments" comparisons be read under?

Decision 20 ended on a question it could not answer. At the top of the s_ego scale the logged future -- the
trajectory the human driver actually took -- is itself not endorsed by the raters (RFS 5.92, 41.7 % floored,
7.8 m from the best-rated rater trajectory), so ADE against the log is a contested target exactly where the
interesting frames are. Before any representation is swapped, the metric that would judge the swap has to be
settled, or the swap cannot be read.

Nothing is fitted here. P0 wrote every arm's per-frame prediction on the whole of val to `preds.npz`, and
this module applies four candidate judges to those same predictions:

  (i)   ADE restricted to frames whose logged future is credible, operationalised two ways: s_ego deciles
        1-9 (circular -- s_ego is the ego model's own residual, so the base being compared picks the exam),
        and the rater frames whose logged future scores RFS >= 7 or is not floored (not circular, but small)
  (ii)  minADE over the top-k anchors, k = 1, 3, 6. The classifier head supplies its own top-k; a regression
        head has one mode, so its candidates are built by snapping its prediction to its k nearest anchors in
        the same vocabulary -- a construction, not the model's own confidence ordering, and labelled as such
  (iii) RFS on the rater frames, overall and per s_ego quintile
  (iv)  ADE to the best-rated rater trajectory, on the rater frames

For every judge: the ranking of the arms, whether it agrees with the ADE-against-log ranking, the sample
size, and the half-width of the paired interval on pre-onset -- which is where the disagreement would have
to be resolved and where, for the rater-based judges, decision 3b already says there are six frames.
"""
import json

import numpy as np
import pandas as pd

from . import traj, waymo
from .common import get_logger

log = get_logger(__name__)
BASE = "ridge ego"
REFERENCE = "ADE vs log (all)"     # the judge everything else is compared against
TOPK = (1, 3, 6)
CREDIBLE_RFS = 7.0                 # decision 20's own cut: the deciles whose logged future scores above this


def load_run(run_dir) -> dict:
    """The per-frame predictions P0 wrote, plus the rater arrays joined onto the same frames."""
    # allow_pickle because a run made before `save_preds` cast its string columns stored them as object
    # arrays; the file is one of our own run directories, not foreign input
    z = np.load(run_dir / "preds.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    arms = [k[len("pred_"):] for k in z.files if k.startswith("pred_")]
    df = waymo.load_index()
    r, rtraj, scores = waymo.load_rater(df)
    at = pd.Series(np.arange(len(d["frame_name"])), index=d["frame_name"])
    pos = at.reindex(waymo.frame_names(df)[r]).to_numpy()
    ok = ~np.isnan(pos)
    d |= {"arms": arms, "rater_pos": pos[ok].astype(int), "rater_traj": rtraj[ok],
          "rater_scores": scores[ok]}
    log.info("%d evaluation frames, %d arms (%s), %d rater frames", len(d["frame_name"]), len(arms),
             ", ".join(arms), int(ok.sum()))
    return d


def ade(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p - g, axis=-1).mean(-1)


def min_ade(cand: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
    """minADE over the first `k` candidates of (n, kmax, T, 2), which are ordered by score."""
    return np.linalg.norm(cand[:, :k] - g[:, None], axis=-1).mean(-1).min(1)


def snap(pred: np.ndarray, vocab: np.ndarray, k: int) -> np.ndarray:
    """The `k` anchors of `vocab` closest to each single-mode prediction, best first.

    This is what makes judge (ii) applicable to a regression head at all. It is a construction and not a
    model output: the k anchors are near the one trajectory the head predicted, so they describe the
    vocabulary's resolution around that point, not a set of manoeuvres the head thinks are plausible. A
    regression arm whose minADE_6 falls has not become multi-modal.
    """
    import torch
    ids, _ = traj.nearest(torch.from_numpy(pred.reshape(len(pred), -1)),
                          torch.from_numpy(vocab.reshape(len(vocab), -1)), k)
    return vocab[ids]


def paired(v: np.ndarray, base: np.ndarray, seq: np.ndarray, m: np.ndarray) -> dict:
    """Mean of a judge on the subset `m`, and the paired delta against the base with a sequence bootstrap."""
    lo, hi = traj.boot_ci((v - base)[m], seq[m]) if m.any() else (np.nan, np.nan)
    return {"n": int(m.sum()), "value": float(v[m].mean()) if m.any() else np.nan,
            "delta": float((v - base)[m].mean()) if m.any() else np.nan,
            "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2}


def judges(d: dict, arms: list[str] | None = None) -> pd.DataFrame:
    """Every judge x arm x scope, as a long table of means and paired intervals against `ridge ego`."""
    arms = arms or d["arms"]
    gt, seq, s, pos = d["fut"], d["sequence"], d["s_ego"], d["rater_pos"]
    rtraj, rscore = d["rater_traj"], d["rater_scores"]
    best = rtraj[np.arange(len(pos)), rscore.argmax(1)]
    speed, n = d["speed"], len(gt)
    log_rfs = waymo.rater_feedback_score(gt[pos], rtraj, rscore, speed[pos])
    log_floored = log_rfs <= waymo.RFS_FLOOR + 1e-9
    dec = np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11))[1:-1], s, "right"), 0, 9)
    quint = np.clip(np.searchsorted(np.quantile(s[pos], np.linspace(0, 1, 6))[1:-1], s[pos], "right"), 0, 4)

    rater = np.zeros(n, bool)
    rater[pos] = True
    credible_b = np.zeros(n, bool)
    credible_b[pos] = (log_rfs >= CREDIBLE_RFS) | ~log_floored
    # One scope list for every judge. RFS and the rater-best ADE are NaN off the rater frames, so a scope
    # intersects itself with them automatically and "dec10" under RFS is decision 20's own top-decile row.
    scopes = {"all": np.ones(n, bool), "pre_onset": d["sub_pre_onset"],
              "straight_yaw": d["sub_straight_yaw"], "dec1-9": dec < 9, "dec10": dec == 9,
              "dec1-9 pre_onset": (dec < 9) & d["sub_pre_onset"],
              "rater": rater, "rater credible": credible_b, "rater pre_onset": rater & d["sub_pre_onset"]}

    per = {}     # (judge, arm) -> per-frame value over all n frames, NaN where the judge does not apply
    for a in arms:
        p = d[f"pred_{a}"]
        per[("ADE vs log", a)] = ade(p, gt)
        if "vocab" in d:
            cand = d[f"topk_{a}"] if f"topk_{a}" in d else snap(p, d["vocab"], max(TOPK))
            for k in TOPK:
                per[(f"minADE{k}", a)] = min_ade(cand, gt, k)
        for name, v in (("RFS", waymo.rater_feedback_score(p[pos], rtraj, rscore, speed[pos])),
                        ("ADE vs rater_best", ade(p[pos], best))):
            full = np.full(n, np.nan)
            full[pos] = v
            per[(name, a)] = full

    rows = []
    for (judge, a), v in per.items():
        base = per[(judge, BASE)]
        for sn, m in scopes.items():
            m = m & ~np.isnan(v)
            rows.append({"judge": judge, "arm": a, "scope": sn, **paired(v, base, seq, m)})
        if judge == "RFS":
            for q in range(5):
                m = rater.copy()
                m[pos] = quint == q
                rows.append({"judge": "RFS", "arm": a, "scope": f"rater q{q + 1}",
                             **paired(v, base, seq, m & ~np.isnan(v))})
    ref = pd.DataFrame([{"judge": "logged future", "arm": "logged future", "scope": "rater",
                         "n": len(pos), "value": float(log_rfs.mean()),
                         "delta": np.nan, "lo": np.nan, "hi": np.nan, "halfwidth": np.nan},
                        {"judge": "logged future floored", "arm": "logged future", "scope": "rater",
                         "n": len(pos), "value": float(log_floored.mean()), "delta": np.nan,
                         "lo": np.nan, "hi": np.nan, "halfwidth": np.nan}])
    return pd.concat([pd.DataFrame(rows), ref], ignore_index=True)


def ranks(t: pd.DataFrame, arms: list[str], pairs=None) -> pd.DataFrame:
    """One row per (judge, scope): the arms best first (`a > b` reads "a is judged better than b", whichever
    direction that judge's numbers run), and whether that order matches the reference judge, ADE against the
    log on every evaluation frame.

    RFS is the one judge where larger is better, so it is ordered the other way; everything else is a
    displacement in metres. `pre_onset_halfwidth` is the widest paired interval any arm has on the
    pre-onset frames under that judge -- for the rater-based judges that is six frames (decision 3b), which
    is the whole reason they cannot carry the mechanism.
    """
    out = []
    for judge, scope in (pairs or SCOPE_PAIRS):
        q = t[(t.judge == judge) & (t.scope == scope) & t.arm.isin(arms)].dropna(subset=["value"])
        if not len(q):
            continue
        # the pre-onset row of this judge's own domain: every frame for a displacement judge, the six rater
        # frames decision 3b counted for a rater judge
        pre = t[(t.judge == judge) & t.scope.isin(("pre_onset", "rater pre_onset")) & (t.arm != BASE)
                & t.arm.isin(arms)].dropna(subset=["halfwidth"])
        pre = pre[pre.n == pre.n.max()] if len(pre) else pre
        out.append({"judge": f"{judge} ({scope})", "n": int(q.n.iloc[0]),
                    "ranking": " > ".join(q.sort_values("value", ascending=judge != "RFS").arm),
                    "pre_onset_n": int(pre.n.iloc[0]) if len(pre) else 0,
                    "pre_onset_halfwidth": float(pre.halfwidth.max()) if len(pre) else np.nan})
    res = pd.DataFrame(out)
    if len(res) and (res.judge == REFERENCE).any():
        ref = res.loc[res.judge == REFERENCE, "ranking"].iloc[0]
        res["agrees_with_reference"] = res.ranking == ref
    return res


SCOPE_PAIRS = (("ADE vs log", "all"), ("ADE vs log", "dec1-9"), ("ADE vs log", "dec10"),
               ("ADE vs log", "pre_onset"), ("ADE vs log", "dec1-9 pre_onset"),
               ("ADE vs log", "rater credible"),
               ("minADE1", "all"), ("minADE3", "all"), ("minADE6", "all"),
               ("minADE3", "pre_onset"), ("minADE6", "pre_onset"),
               ("RFS", "rater"), ("RFS", "rater credible"), ("RFS", "dec10"),
               ("ADE vs rater_best", "rater"), ("ADE vs rater_best", "rater credible"),
               ("ADE vs rater_best", "dec10"))


def main():
    import argparse
    from pathlib import Path
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0-run", required=True, help="the P0 run directory holding preds.npz")
    ap.add_argument("--arms", default="", help="comma list; default every arm in preds.npz")
    a = ap.parse_args()
    from .waymo_p0 import use_cpu
    use_cpu()
    rl = RunLog("waymo_p1", "judge")
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    d = load_run(Path(a.p0_run))
    arms = [x for x in a.arms.split(",") if x] or d["arms"]
    t = judges(d, arms)
    r = ranks(t, arms)
    for name, x in (("judges", t), ("ranks", r)):
        x.to_csv(rl.dir / f"{name}.csv", index=False)
        rl.log.info("%s\n%s", name, x.to_markdown(index=False, floatfmt=".4f"))
        rl.event(name, rows=x.to_dict("records"))
    (rl.dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
