"""L0 on Waymo E2E: is the intervention signal linearly present in the frozen features, or absent from them?

Decision 3d measured a *positive* difference-of-differences: vision buys less on pre-manoeuvre-onset frames
than on straight ones. Two readings of that point at opposite next experiments, and decisions 20 pre-registers
the split between them. Either the information is linearly readable but the uniform mean-squared objective
drowns 1510 onset frames in 90k straight ones (a readout problem: reweight the objective, or use a stronger
head, and the gain appears), or the frozen features carry no linear direction for it at all (a representation
problem: no head will help). The arms, all on top of the same frozen `ridge ego` base and inside decision 3d's
sequence-grouped half-val split:

  A  uniform `ridge_late`, which must reproduce 3d before anything else is read
  B  surprise-weighted `ridge_late`: weighted least squares over the kinematic surprise s_i
  C  `ridge_late` fitted on the fit half's pre-onset frames only -- the upper bound of a linear readout there
  D  the MLP head of jevdrive/planner.py, uniform and under the best B weighting

The kinematic surprise is s_i = ADE(logged future, constant-velocity extrapolation) over the 5 s horizon,
using the same `waymo.baselines()["cv"]` arc the baseline tables use. An attention-pooling arm is not possible
here: the cached features are per-frame pooled vectors, not spatial tokens.
"""
import numpy as np
import pandas as pd
import torch

from . import planner, traj, waymo, waymo_stage_a as sa
from .common import get_logger

log = get_logger(__name__)
DEV = "cuda"
SUBSETS = sa.SUBSETS
BASE = "ridge ego"
FOLDS = 4


# ---------------------------------------------------------------- kinematic surprise

def surprise(past: np.ndarray, future: np.ndarray) -> np.ndarray:
    """s_i: ADE over the 5 s horizon between the logged future and the constant-velocity extrapolation.

    `cv` is the constant-speed, zero-yaw-rate arc, i.e. a straight line at the speed estimated from the last
    second of ego history -- the same baseline decisions 3c's table is built on. The error it leaves is
    dominated by longitudinal events (braking, launching) rather than by turning; decisions 20 records that.
    """
    return np.linalg.norm(waymo.baselines(past)["cv"] - waymo.future_xy(future), axis=-1).mean(1)


def surprise_report(df: pd.DataFrame, past: np.ndarray, future: np.ndarray, split: str = "val") -> pd.DataFrame:
    """Per subset: n, the s quantiles, the longitudinal/lateral split of the CV error, and the share of the
    top-decile-s frames the subset holds against its own base rate in the split."""
    m0 = (df.split == split).to_numpy() & df.has_future.to_numpy()
    err = waymo.baselines(past)["cv"] - waymo.future_xy(future)
    s, lon, lat = surprise(past, future), np.abs(err[..., 0]).mean(1), np.abs(err[..., 1]).mean(1)
    top = m0 & (s >= np.quantile(s[m0], 0.9))
    qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    sub, rows = waymo.subsets(df, past, future), []
    for k in SUBSETS:
        m = m0 & sub[k]
        rows.append({"subset": k, "n": int(m.sum()), "s_mean": s[m].mean(),
                     **{f"s_q{int(q * 100)}": v for q, v in zip(qs, np.quantile(s[m], qs))},
                     "cv_lon": lon[m].mean(), "cv_lat": lat[m].mean(), "lat_share": lat[m].mean() / s[m].mean(),
                     "base_rate": m.sum() / m0.sum(), "share_of_top_decile": (top & m).sum() / top.sum(),
                     "recall_of_subset": (top & m).sum() / max(m.sum(), 1)})
    return pd.DataFrame(rows)


def noise_check(df: pd.DataFrame, past: np.ndarray, future: np.ndarray, n: int = 50,
                split: str = "val") -> pd.DataFrame:
    """The `n` highest-surprise frames with the diagnostics that would expose an artifact rather than an event.

    An artifact would show up as a discontinuous ego history (a speed jump between consecutive 0.25 s
    intervals far outside the split's own distribution, or repeated positions from padding) or as a future
    that is not physically reachable. Decision 13's exact-hit history rule does not apply: `past_states` is
    read per frame from the tfrecord, not assembled across frames, so no window can be padded here.
    """
    m0 = (df.split == split).to_numpy() & df.has_future.to_numpy()
    s, kin = surprise(past, future), waymo.past_kinematics(past)
    step = np.linalg.norm(np.diff(past[:, :, :2], axis=1), axis=-1)
    jump = np.abs(np.diff(step / waymo.DT, axis=1)).max(1)
    gt = waymo.future_xy(future)
    fspd = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(gt), 1, 2), np.float32), gt], 1), axis=1),
                          axis=-1) / waymo.DT
    rated = np.zeros(len(df), bool)
    rated[waymo.load_rater()[0]] = True
    sub = waymo.subsets(df, past, future)
    o = np.flatnonzero(m0)[np.argsort(-s[m0])[:n]]
    lim = float(np.quantile(jump[m0], 0.999))
    out = pd.DataFrame({"sequence": df.sequence.to_numpy()[o], "frame": df.frame.to_numpy()[o], "s": s[o],
                        "v0": kin["v"][o], "a0": kin["a"][o], "yaw_rate_deg": np.degrees(kin["w"][o]),
                        "past_speed_jump": jump[o], "past_dup_steps": (step[o] < 1e-6).sum(1),
                        "future_speed_max": fspd[o].max(1), "rater": rated[o],
                        **{k: sub[k][o] for k in ("pre_onset", "straight_yaw", "turn_yaw")},
                        "cluster": df.cluster.to_numpy()[o]})
    out["artifact"] = (out.past_speed_jump > lim) | (out.past_dup_steps > 0) | ~np.isfinite(out.s)
    log.info("noise check: %d/%d of the top-s frames look like artifacts (past speed jump above the split's "
             "99.9th percentile %.2f m/s, or padded history); %d distinct sequences, %d rater frames",
             int(out.artifact.sum()), n, lim, out.sequence.nunique(), int(out.rater.sum()))
    return out


# ---------------------------------------------------------------- weighted ridge

def weight_schemes(s: np.ndarray, fit: np.ndarray) -> dict[str, np.ndarray]:
    """The weightings of decisions 20, each normalised to mean 1 over the fit half.

    `s` is scaled by its own fit-half mean first, so every scheme is a function of the relative surprise and
    none of them depends on the metre scale of the split.
    """
    r = s / s[fit].mean()
    out = {"uniform": np.ones_like(r), "lin": r, "sq": r ** 2, "a1": 1 + r, "a4": 1 + 4 * r,
           "top50": (s >= np.quantile(s[fit], 0.5)).astype(np.float64),
           "top25": (s >= np.quantile(s[fit], 0.75)).astype(np.float64)}
    return {k: (w / w[fit].mean()).astype(np.float32) for k, w in out.items()}


def wridge_solve(X: torch.Tensor, Y: torch.Tensor, rows: np.ndarray, w: torch.Tensor, lams) -> torch.Tensor:
    """planner.ridge_solve with per-sample weights: W minimises the *weighted* mean squared error + lam |W|^2.

    Scaling the rows of X and Y by sqrt(w) is only correct once the centring is weighted too, otherwise the
    intercept is fitted to the unweighted mean and the weighting leaks into it. Weights are normalised to sum
    to len(rows) so that `lam` keeps the same meaning as in the uniform solve.
    """
    a, b, u = X[rows], Y[rows], w[rows]
    u = (u / u.mean()).unsqueeze(1)
    mx, my = (u * a).mean(0), (u * b).mean(0)
    q = u.sqrt()
    a, b = q * (a - mx), q * (b - my)
    ev, V = planner.gram_eigh(a)
    z = V.T @ (a.T @ b)
    W = torch.stack([V @ (z / (ev[:, None] + lam * len(rows))) for lam in lams])
    return torch.cat([W, (my - mx @ W).unsqueeze(1)], 1)


def wridge_cv(X, Y, sp, fut, w: torch.Tensor, pre: np.ndarray, folds: int = FOLDS, rows=None):
    """Weighted ridge whose L2 strength is chosen by sequence-grouped CV inside the fit half.

    The selection score is the *weighted* ADE on each held-out fold, because the weighting is the arm's
    objective and selecting on the uniform score would half-undo it. The unweighted pre-onset ADE over the
    same held-out folds is returned alongside as `sel_pre_ade`: that is the fit-half-only quantity decisions
    20 pre-registers for choosing which B scheme arm D inherits, and it never sees the evaluation half.

    `rows` restricts the fit to a subset of the fit half (arm C), leaving selection grouped by sequence
    inside that subset.
    """
    from sklearn.model_selection import GroupKFold
    T, lams = fut.shape[1], planner.LAM_RIDGE
    tr = sp.train if rows is None else np.intersect1d(sp.train, rows)
    score, pre_num, pre_den = np.zeros(len(lams)), np.zeros(len(lams)), 0.0
    wn = w.cpu().numpy()
    for a, b in GroupKFold(folds).split(tr, groups=sp.seq[tr]):
        W = wridge_solve(X, Y, tr[a], w, lams)
        te = tr[b]
        p = planner.linear_apply(W, X, te).reshape(len(lams), -1, T, 2).cpu().numpy()
        e = np.linalg.norm(p - fut[te], axis=-1).mean(-1)            # (P, len(te))
        score += (e * wn[te]).sum(1) / wn[te].sum()
        if pre[te].any():
            pre_num += e[:, pre[te]].sum(1)
            pre_den += int(pre[te].sum())
    best = planner._pick(score / folds, lams, "wridge (grouped CV)")
    W = wridge_solve(X, Y, tr, w, [lams[best]])
    pv = planner.linear_apply(W, X, sp.val).reshape(-1, T, 2).cpu().numpy()
    st = {"lam": float(lams[best]), "sel_ade": float(score[best] / folds), "n_fit": len(tr),
          "sel_pre_ade": float(pre_num[best] / pre_den) if pre_den else float("nan")}
    return pv[:, None], st, W


def wmlp_solve(X, y, rows, out, epochs, w: torch.Tensor, sel=None, seed: int = 0):
    """planner.mlp_solve with per-sample weights on the squared error. Regression only: arm D is the
    continuous head, so no vocabulary and no cross-entropy."""
    torch.manual_seed(seed)
    net = planner._mlp(X.shape[1], out)
    opt = torch.optim.AdamW(net.parameters(), lr=planner.MLP_LR, weight_decay=planner.MLP_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    r = torch.as_tensor(rows, device=DEV)
    best = (np.inf, epochs)
    for ep in range(epochs):
        net.train()
        for b in r[torch.randperm(len(r), device=DEV)].split(planner.MLP_BS):
            opt.zero_grad(set_to_none=True)
            ((net(X[b]) - y[b]).pow(2).mean(1) * w[b]).mean().backward()
            opt.step()
        sched.step()
        if sel is not None:
            best = min(best, (sel[1](planner.mlp_predict(net, X, sel[0])), ep + 1))
    return net, best


def mlp_late(Xi, sp, res_fut, R, w: torch.Tensor, seed: int = 0):
    """The MLP head on the image features, predicting the residual over the ego ridge, early-stopped on the
    weighted inner-val score and refitted on the whole fit half for that many epochs (planner.Heads.mlp)."""
    T = res_fut.shape[1]
    wn = w.cpu().numpy()

    def score(o):
        e = np.linalg.norm(o.reshape(-1, T, 2).cpu().numpy() - res_fut[sp.sel], axis=-1).mean(1)
        return float((e * wn[sp.sel]).sum() / wn[sp.sel].sum())

    _, (s, ep) = wmlp_solve(Xi, R, sp.fit, 2 * T, planner.MLP_EPOCHS, w, (sp.sel, score), seed)
    net, _ = wmlp_solve(Xi, R, sp.train, 2 * T, ep, w, None, seed)
    p = planner.mlp_predict(net, Xi, sp.val).reshape(-1, T, 2).cpu().numpy()
    return p[:, None], {"epochs": ep, "sel_ade": s}


# ---------------------------------------------------------------- the arms

def run_direction(direction: int, set_name: str = "qwen_front3", layer: str = sa.LAYER, seed: int = 0,
                  folds: int = FOLDS, rl=None) -> tuple[dict, dict, pd.DataFrame]:
    """All of L0's arms for one direction of the half-val split. Returns (predictions, context, B table)."""
    df, rows, fut, ego, img, sub, past = sa.load_all(set_name, layer, seed)
    halves = sa.val_halves(df, seed)
    h = df.sequence.map(halves).to_numpy()[rows]
    seq = df.sequence.to_numpy()[rows]
    sp = sa.Halves(df, seq, h == direction, h == (1 - direction), seed)
    n, T = len(rows), fut.shape[1]
    log.info("direction %d: fit %d frames / %d sequences, eval %d / %d", direction, len(sp.train),
             len(np.unique(seq[sp.train])), len(sp.val), len(np.unique(seq[sp.val])))

    s = surprise(past[rows], fut)
    pre = sub["pre_onset"]
    F = torch.as_tensor(fut.reshape(n, -1), device=DEV)
    Xe = planner.standardize(torch.as_tensor(ego, device=DEV), sp.train)
    Xi = planner.standardize(torch.from_numpy(np.asarray(img)).to(DEV).float(), sp.train)

    preds = {}
    p_ego, st_ego, W_ego = sa.ridge_cv(Xe, F, sp, fut)
    preds[BASE] = (p_ego, st_ego)
    base = planner.linear_apply(W_ego, Xe, np.arange(n))[0]
    R = F - base
    res_fut = R.reshape(-1, T, 2).cpu().numpy()
    off = base[sp.val].reshape(-1, 1, T, 2).cpu().numpy()

    ws = weight_schemes(s, sp.train)
    brows = []
    for name, wv in ws.items():
        w = torch.as_tensor(wv, device=DEV)
        p, st, _ = wridge_cv(Xi, R, sp, res_fut, w, pre, folds)
        arm = "A ridge_late uniform" if name == "uniform" else f"B {name}"
        preds[arm] = (p + off, st)
        brows.append({"direction": direction, "scheme": name, "arm": arm, "eff_n": float(wv[sp.train].sum() ** 2
                      / (wv[sp.train] ** 2).sum()), **st})
        log.info("dir %d %-22s lam %.3g  weighted sel ADE %.4f  fit-half CV pre_onset ADE %.4f (n_fit %d)",
                 direction, arm, st["lam"], st["sel_ade"], st["sel_pre_ade"], st["n_fit"])
        if rl is not None:
            rl.event("scheme", **brows[-1])
    b = pd.DataFrame(brows)
    best = b[b.scheme != "uniform"].sort_values("sel_pre_ade").scheme.iloc[0]
    log.info("dir %d: B scheme chosen on the fit half alone: %s", direction, best)

    w1 = torch.ones(n, device=DEV)
    p, st, _ = wridge_cv(Xi, R, sp, res_fut, w1, pre, folds, rows=np.flatnonzero(pre))
    preds["C pre_onset fit"] = (p + off, st)
    for arm, wv in (("D mlp uniform", w1), (f"D mlp {best}", torch.as_tensor(ws[best], device=DEV))):
        p, st = mlp_late(Xi, sp, res_fut, R, wv, seed)
        preds[arm] = (p + off, st)

    ctx = {"df": df, "rows": rows, "fut": fut, "sub": sub, "seq": seq, "sp": sp, "past": past,
           "direction": direction, "s": s, "best_scheme": best}
    return preds, ctx, b


def evaluate(preds: dict, ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per arm by subset (ADE, RFS where the raters are), the paired delta against `ridge ego`, and the DiD.

    Every arm is paired against the same base, so the deltas across arms are comparable frame by frame. RFS
    is reported overall only: decision 3b settled that the pre-onset subset holds 6 rater frames in the whole
    of val and cannot carry the metric.
    """
    sp, sub, seq, fut = ctx["sp"], ctx["sub"], ctx["seq"], ctx["fut"]
    v, gt, sq = sp.val, fut[sp.val], seq[sp.val]
    masks = {k: sub[k][v] for k in SUBSETS}
    pos, rtraj, scores = sa.rfs_rows(ctx["df"], ctx["rows"][v])
    speed = waymo.init_speed(ctx["past"][ctx["rows"][v]])
    cluster = ctx["df"].cluster.astype(str).to_numpy()[ctx["rows"][v]]
    ade, rows_out = {}, []
    for name, (p, st) in preds.items():
        e = np.linalg.norm(p[:, 0] - gt, axis=-1).mean(1)
        ade[name] = e
        r = {"arm": name, "direction": ctx["direction"], **{f"fit_{k}": q for k, q in st.items()}}
        for sn, m in masks.items():
            r[f"ade_{sn}"], r[f"n_{sn}"] = e[m].mean(), int(m.sum())
        if len(pos):
            sc = waymo.rater_feedback_score(p[pos, 0], rtraj, scores, speed[pos])
            r["rfs"], r["n_rater"] = sc.mean(), len(pos)
            r["rfs_cluster"] = waymo.rfs_by_cluster(sc, cluster[pos])[0]
            r["rfs_lo"], r["rfs_hi"] = traj.boot_ci(sc, sq[pos])
            r["floored"] = float((sc <= waymo.RFS_FLOOR + 1e-9).mean())
        rows_out.append(r)
        log.info("dir %d %-22s RFS %.3f  ADE %.3f | pre_onset %.3f (n=%d)  straight %.3f (n=%d)",
                 ctx["direction"], name, r.get("rfs", np.nan), r["ade_all"], r["ade_pre_onset"],
                 r["n_pre_onset"], r["ade_straight_yaw"], r["n_straight_yaw"])

    pairs, dids = [], []
    for name in preds:
        if name == BASE:
            continue
        d = ade[name] - ade[BASE]
        for sn, m in masks.items():
            lo, hi = traj.boot_ci(d[m], sq[m])
            pairs.append({"direction": ctx["direction"], "arm": name, "subset": sn, "n": int(m.sum()),
                          "ade_base": ade[BASE][m].mean(), "ade_arm": ade[name][m].mean(),
                          "dade": d[m].mean(), "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2})
        hi_m, lo_m = masks["pre_onset"], masks["straight_yaw"]
        point, cl, ch = traj.boot_did(d, sq, hi_m, lo_m)
        dids.append({"direction": ctx["direction"], "arm": name, "n_hi": int(hi_m.sum()),
                     "n_lo": int(lo_m.sum()), "dade_pre_onset": d[hi_m].mean(),
                     "dade_straight": d[lo_m].mean(), "did": point, "did_lo": cl, "did_hi": ch,
                     "halfwidth": (ch - cl) / 2})
        log.info("dir %d %-22s pre_onset %+.4f  straight %+.4f  DiD %+.4f [%+.4f, %+.4f]", ctx["direction"],
                 name, d[hi_m].mean(), d[lo_m].mean(), point, cl, ch)
    return pd.DataFrame(rows_out), pd.DataFrame(pairs), pd.DataFrame(dids)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="surprise,arms", help="comma list of surprise,arms")
    ap.add_argument("--layer", default=sa.LAYER)
    ap.add_argument("--feature-set", default="qwen_front3")
    ap.add_argument("--directions", default="0,1")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rl = RunLog("waymo_l0", "surprise_weighting")
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    if "surprise" in a.steps:
        df = waymo.load_index()
        past, future = waymo.load_ego()
        rep = surprise_report(df, past, future)
        nz = noise_check(df, past, future)
        rep.to_csv(rl.dir / "surprise.csv", index=False)
        nz.to_csv(rl.dir / "noise_check.csv", index=False)
        rl.log.info("surprise\n%s", rep.to_markdown(index=False, floatfmt=".3f"))
        rl.log.info("noise check\n%s", nz.to_markdown(index=False, floatfmt=".2f"))
        rl.event("surprise", rows=rep.to_dict("records"), artifacts=int(nz.artifact.sum()), n=len(nz))
        del df, past, future
    if "arms" in a.steps:
        res, pairs, dids, bs = [], [], [], []
        for d in (int(x) for x in a.directions.split(",")):
            preds, ctx, b = run_direction(d, a.feature_set, a.layer, a.seed, a.folds, rl)
            r, p_, dd = evaluate(preds, ctx)
            res.append(r), pairs.append(p_), dids.append(dd), bs.append(b)
            del preds
            torch.cuda.empty_cache()
        for name, t in (("arms", res), ("paired", pairs), ("did", dids), ("schemes", bs)):
            t = pd.concat(t, ignore_index=True)
            t.to_csv(rl.dir / f"{name}.csv", index=False)
            rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
            rl.event(name, rows=t.to_dict("records"))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
