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

Surprise means "what the prior cannot predict", so which prior it is measured against decides what the
weighting actually emphasises. Three definitions are carried, all as ADE over the 5 s horizon:

  s_ego   against the `ridge ego` prediction itself, out-of-fold on the fit half -- the primary definition,
          because it is the residual of exactly the prior every arm is measured against
  s_ctrv  against the constant-turn-rate constant-speed arc, the best zero-parameter prior available
  s_cv    against the constant-speed zero-yaw-rate arc: a control, and a deliberately poor one -- it calls an
          ongoing turn surprising although the yaw-rate history predicts it, and its magnitude is dominated
          by longitudinal braking (decisions 20)

An attention-pooling arm is not possible here: the cached features are per-frame pooled vectors, not spatial
tokens.
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


# ---------------------------------------------------------------- surprise

S_KINDS = ("ego", "ctrv", "cv")          # primary first; the two kinematic ones are controls
PRIMARY = "ego"
FAMILY_SCHEMES = {"ego": ("lin", "sq", "a1", "a4", "top25"),      # primary family, full set bar the median cut
                  "ctrv": ("lin", "top25"), "cv": ("lin", "top25")}  # controls: linear and the hard quartile


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame ADE over the whole horizon between two (n, T, 2) trajectory sets."""
    return np.linalg.norm(pred - gt, axis=-1).mean(1)


def ridge_np(X: np.ndarray, Y: np.ndarray, rows: np.ndarray, lams) -> np.ndarray:
    """planner.ridge_solve on the CPU, for the 100-dimensional ego input only.

    The ego head is small enough (d = 100) that the whole selection runs in numpy in under a second, which is
    what makes s_ego computable without touching the GPU. The arithmetic is the same: centre, eigendecompose
    the gram, and read every lambda off one decomposition.
    """
    a, b = X[rows].astype(np.float64), Y[rows].astype(np.float64)
    mx, my = a.mean(0), b.mean(0)
    a, b = a - mx, b - my
    ev, V = np.linalg.eigh(a.T @ a)
    z = V.T @ (a.T @ b)
    W = np.stack([V @ (z / (ev[:, None] + lam * len(rows))) for lam in lams])
    return np.concatenate([W, (my - mx @ W)[:, None]], 1)


def ego_surprise(ego: np.ndarray, fut: np.ndarray, sp, folds: int = FOLDS) -> np.ndarray:
    """s_ego: how far the logged future falls from what the *ego readout itself* predicts.

    This is the definition the experiment is actually about -- the residual of the prior the arms are measured
    against -- so it has to be honest on both halves: the fit half gets out-of-fold predictions from the same
    sequence-grouped folds the lambda selection uses, and the evaluation half gets the model fitted on the
    whole fit half. Using in-sample predictions on the fit half would make the best-fitted frames look
    unsurprising and quietly bias every weighting towards whatever the ridge happened to miss.

    Lambda is picked exactly as jevdrive/waymo_stage_a.ridge_cv picks it: mean ADE over the held-out folds.
    """
    from sklearn.model_selection import GroupKFold
    T, lams = fut.shape[1], planner.LAM_RIDGE
    mu, sd = ego[sp.train].mean(0), ego[sp.train].std(0)
    X = (ego - mu) / np.where(sd > 1e-6, sd, 1)
    Y = fut.reshape(len(fut), -1)
    oof = np.zeros((len(lams), len(fut), T, 2), np.float32)
    score = np.zeros(len(lams))
    for a, b in GroupKFold(folds).split(sp.train, groups=sp.seq[sp.train]):
        W = ridge_np(X, Y, sp.train[a], lams)
        te = sp.train[b]
        p = (X[te] @ W[:, :-1] + W[:, -1, None]).reshape(len(lams), -1, T, 2)
        oof[:, te] = p
        score += [ade(p[i], fut[te]).mean() for i in range(len(lams))]
    best = planner._pick(score / folds, lams, "ego surprise (grouped CV)")
    W = ridge_np(X, Y, sp.train, [lams[best]])
    oof[best, sp.val] = (X[sp.val] @ W[0, :-1] + W[0, -1]).reshape(-1, T, 2)
    log.info("s_ego: lambda %.3g, held-out ADE %.3f over %d fit frames, %d eval frames",
             lams[best], score[best] / folds, len(sp.train), len(sp.val))
    return ade(oof[best], fut)


def surprise_set(past: np.ndarray, fut: np.ndarray, ego: np.ndarray, sp, folds: int = FOLDS):
    """The three surprise definitions and the error field each one is the ADE of.

    s_cv is the original control: the constant-speed, zero-yaw-rate arc, which calls an ongoing turn
    surprising even though the yaw-rate history predicts it perfectly. s_ctrv uses the constant-turn-rate
    constant-speed arc instead -- the best zero-parameter prior available -- and s_ego uses the experiment's
    own ego ridge. Only s_ego is "what the prior the arms are measured against cannot predict".
    """
    b = waymo.baselines(past)
    err = {"cv": b["cv"] - fut, "ctrv": b["ctrv"] - fut}
    s = {k: ade(b[k], fut) for k in ("cv", "ctrv")}
    s["ego"] = ego_surprise(ego, fut, sp, folds)
    return {k: s[k] for k in S_KINDS}, err


def surprise_report(s: dict, err: dict, sub: dict, direction: int) -> pd.DataFrame:
    """Per surprise definition and subset: n, the s quantiles, the longitudinal/lateral decomposition of the
    prior's error, and how much the top decile of s enriches the subset over its own base rate."""
    qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    rows = []
    for kind, sv in s.items():
        top = sv >= np.quantile(sv, 0.9)
        for k in SUBSETS:
            m = sub[k]
            r = {"direction": direction, "kind": kind, "subset": k, "n": int(m.sum()), "s_mean": sv[m].mean(),
                 **{f"s_q{int(q * 100)}": v for q, v in zip(qs, np.quantile(sv[m], qs))},
                 "base_rate": m.mean(), "share_of_top_decile": (top & m).sum() / top.sum(),
                 "enrichment": ((top & m).sum() / top.sum()) / max(m.mean(), 1e-9),
                 "recall_of_subset": (top & m).sum() / max(m.sum(), 1)}
            if kind in err:   # the ego prior's error field is not a closed form, so only the arcs decompose
                lon, lat = np.abs(err[kind][..., 0]).mean(1), np.abs(err[kind][..., 1]).mean(1)
                r |= {"lon": lon[m].mean(), "lat": lat[m].mean(), "lat_share": lat[m].mean() / sv[m].mean()}
            rows.append(r)
    return pd.DataFrame(rows)


def surprise_corr(s: dict, past: np.ndarray, rows: np.ndarray, direction: int) -> pd.DataFrame:
    """Correlation of each surprise definition with the two kinematic quantities that would explain it away:
    |longitudinal acceleration| (braking and launching) and speed."""
    kin = waymo.past_kinematics(past[rows])
    return pd.DataFrame([{"direction": direction, "kind": k,
                          "corr_abs_a0": float(np.corrcoef(v, np.abs(kin["a"]))[0, 1]),
                          "corr_v0": float(np.corrcoef(v, kin["v"])[0, 1])} for k, v in s.items()])


def noise_check(df: pd.DataFrame, rows: np.ndarray, past: np.ndarray, fut: np.ndarray, s: dict, sub: dict,
                direction: int, n: int = 50) -> pd.DataFrame:
    """The `n` highest-surprise frames of each definition, with the diagnostics that would expose an artifact
    rather than an event.

    An artifact would show up as a discontinuous ego history (a speed jump between consecutive 0.25 s
    intervals far outside the split's own distribution, or repeated positions from padding) or as a future
    that is not physically reachable. Decision 13's exact-hit history rule does not apply: `past_states` is
    read per frame from the tfrecord, not assembled across frames, so no window here can be padded.
    """
    p, kin = past[rows], waymo.past_kinematics(past[rows])
    step = np.linalg.norm(np.diff(p[:, :, :2], axis=1), axis=-1)
    jump = np.abs(np.diff(step / waymo.DT, axis=1)).max(1)
    fspd = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(fut), 1, 2), np.float32), fut], 1), axis=1),
                          axis=-1) / waymo.DT
    rated = np.zeros(len(df), bool)
    rated[waymo.load_rater(df)[0]] = True
    rated = rated[rows]
    lim = float(np.quantile(jump, 0.999))
    out = []
    for kind, sv in s.items():
        o = np.argsort(-sv)[:n]
        t = pd.DataFrame({"direction": direction, "kind": kind, "sequence": df.sequence.to_numpy()[rows][o],
                          "frame": df.frame.to_numpy()[rows][o], "s": sv[o], "v0": kin["v"][o],
                          "a0": kin["a"][o], "yaw_rate_deg": np.degrees(kin["w"][o]),
                          "past_speed_jump": jump[o], "past_dup_steps": (step[o] < 1e-6).sum(1),
                          "future_speed_max": fspd[o].max(1), "rater": rated[o],
                          **{k: sub[k][o] for k in ("pre_onset", "straight_yaw", "turn_yaw")},
                          "cluster": df.cluster.to_numpy()[rows][o]})
        t["artifact"] = (t.past_speed_jump > lim) | (t.past_dup_steps > 0) | ~np.isfinite(t.s)
        log.info("noise check %-4s: %d/%d artifacts (past speed jump over the 99.9th percentile %.2f m/s, or "
                 "padded history); %d sequences, %d rater frames, %d pre_onset, %d turn_yaw", kind,
                 int(t.artifact.sum()), n, lim, t.sequence.nunique(), int(t.rater.sum()),
                 int(t.pre_onset.sum()), int(t.turn_yaw.sum()))
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- weighted ridge

def weight_schemes(s: dict, fit: np.ndarray, families: dict | None = None) -> dict[str, np.ndarray]:
    """The weightings of decisions 20, one family per surprise definition, all normalised to mean 1 on the
    fit half.

    Each family scales its own s by its own fit-half mean first, so a scheme is a function of the relative
    surprise and never of the metre scale of that definition. The primary family is `ego`; `ctrv` and `cv`
    are carried as controls with the reduced set of schemes, which is what keeps the wall time in range.
    `families` narrows that to the schemes a caller actually wants (P0 repeats only the one the fit half
    chose), and it then needs only those definitions of s.
    """
    out = {"uniform": np.ones(len(s[PRIMARY]))}
    for kind, names in (families or FAMILY_SCHEMES).items():
        r = s[kind] / s[kind][fit].mean()
        cand = {"lin": r, "sq": r ** 2, "a1": 1 + r, "a4": 1 + 4 * r,
                "top50": (s[kind] >= np.quantile(s[kind][fit], 0.5)).astype(np.float64),
                "top25": (s[kind] >= np.quantile(s[kind][fit], 0.75)).astype(np.float64)}
        out |= {f"{kind}:{nm}": cand[nm] for nm in names}
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
    """All of L0's arms for one direction of the half-val split.

    Returns (predictions, context, the weighting table, the surprise report, the noise check)."""
    df, rows, fut, ego, img, sub, past = sa.load_all(set_name, layer, seed)
    halves = sa.val_halves(df, seed)
    h = df.sequence.map(halves).to_numpy()[rows]
    seq = df.sequence.to_numpy()[rows]
    sp = sa.Halves(df, seq, h == direction, h == (1 - direction), seed)
    n, T = len(rows), fut.shape[1]
    log.info("direction %d: fit %d frames / %d sequences, eval %d / %d", direction, len(sp.train),
             len(np.unique(seq[sp.train])), len(sp.val), len(np.unique(seq[sp.val])))

    s, err = surprise_set(past[rows], fut, ego, sp, folds)
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
    prim = b[b.scheme.str.startswith(f"{PRIMARY}:")]
    best = prim.sort_values("sel_pre_ade").scheme.iloc[0]
    log.info("dir %d: B scheme chosen on the fit half alone, primary family %s: %s", direction, PRIMARY, best)

    w1 = torch.ones(n, device=DEV)
    p, st, _ = wridge_cv(Xi, R, sp, res_fut, w1, pre, folds, rows=np.flatnonzero(pre))
    preds["C pre_onset fit"] = (p + off, st)
    for arm, wv in (("D mlp uniform", w1), (f"D mlp {best}", torch.as_tensor(ws[best], device=DEV))):
        p, st = mlp_late(Xi, sp, res_fut, R, wv, seed)
        preds[arm] = (p + off, st)

    rep = surprise_report(s, err, sub, direction)
    rep = rep.merge(surprise_corr(s, past, rows, direction), on=["direction", "kind"], how="left")
    nz = noise_check(df, rows, past, fut, s, sub, direction)
    ctx = {"df": df, "rows": rows, "fut": fut, "sub": sub, "seq": seq, "sp": sp, "past": past,
           "direction": direction, "s": s, "best_scheme": best}
    return preds, ctx, b, rep, nz


def evaluate(preds: dict, ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per arm by subset (ADE, RFS where the raters are), the paired delta against `ridge ego`, the DiD, and
    the same deltas stratified by the evaluation half's s_ego decile.

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
    per, rows_out = {}, []
    for name, (p, st) in preds.items():
        e = np.linalg.norm(p[:, 0] - gt, axis=-1).mean(1)
        per[name] = e
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
        d = per[name] - per[BASE]
        for sn, m in masks.items():
            lo, hi = traj.boot_ci(d[m], sq[m])
            pairs.append({"direction": ctx["direction"], "arm": name, "subset": sn, "n": int(m.sum()),
                          "ade_base": per[BASE][m].mean(), "ade_arm": per[name][m].mean(),
                          "dade": d[m].mean(), "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2})
        hi_m, lo_m = masks["pre_onset"], masks["straight_yaw"]
        point, cl, ch = traj.boot_did(d, sq, hi_m, lo_m)
        dids.append({"direction": ctx["direction"], "arm": name, "n_hi": int(hi_m.sum()),
                     "n_lo": int(lo_m.sum()), "dade_pre_onset": d[hi_m].mean(),
                     "dade_straight": d[lo_m].mean(), "did": point, "did_lo": cl, "did_hi": ch,
                     "halfwidth": (ch - cl) / 2})
        log.info("dir %d %-22s pre_onset %+.4f  straight %+.4f  DiD %+.4f [%+.4f, %+.4f]", ctx["direction"],
                 name, d[hi_m].mean(), d[lo_m].mean(), point, cl, ch)
    dec = decile_report(per, ctx)
    for name in dec.arm.unique():
        g = dec[(dec.arm == name) & (dec.scope == "all")].sort_values("decile")
        log.info("dir %d %-22s relative gain by s_ego decile: %s", ctx["direction"], name,
                 " ".join(f"{x:+.3f}" for x in g.rel_gain))
    return pd.DataFrame(rows_out), pd.DataFrame(pairs), pd.DataFrame(dids), dec


def _write(rl, tables):
    """Concatenate each table over the directions, write it as CSV next to the log, and log it."""
    for name, parts in tables:
        t = pd.concat(parts, ignore_index=True)
        t.to_csv(rl.dir / f"{name}.csv", index=False)
        rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
        rl.event(name, rows=t.to_dict("records"))


def decile_report(per: dict, ctx: dict, scopes=("all", "straight_yaw")) -> pd.DataFrame:
    """Every arm against the base, stratified by the evaluation half's own s_ego decile.

    The pre-onset subset turned out to be a poor stand-in for "where the ego prior fails": s_ego enriches it
    only 1.47x, and the prior's largest residuals are braking and launching, not turning (decisions 20). The
    decile cut puts the whole intervention mass on the axis instead -- about 5 000 evaluation frames in the
    top decile against 750 pre-onset ones -- and the s_ego used here comes from the model fitted on the fit
    half, so it is out-of-sample on every frame it stratifies.

    The quantity to read is the *relative* gain, not the absolute one: a high-s_ego frame mechanically has
    more error available to remove, so an absolute delta that grows with the decile says nothing on its own.
    The bootstrap resamples sequences, exactly as everywhere else; the relative interval is the absolute one
    divided by that decile's ego ADE, which is a constant within the decile.
    """
    sp, seq, s = ctx["sp"], ctx["seq"], ctx["s"][PRIMARY]
    v, sq, sv = sp.val, seq[sp.val], s[sp.val]
    q = np.quantile(sv, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q[1:-1], sv, "right"), 0, 9)
    rows = []
    for scope in scopes:
        sm = ctx["sub"][scope][v]
        for name, e in per.items():
            if name == BASE:
                continue
            d = e - per[BASE]
            for i in range(10):
                m = sm & (dec == i)
                if not m.any():
                    continue
                lo, hi = traj.boot_ci(d[m], sq[m])
                b = per[BASE][m].mean()
                rows.append({"direction": ctx["direction"], "arm": name, "scope": scope, "decile": i,
                             "n": int(m.sum()), "s_ego_lo": float(sv[m].min()), "s_ego_hi": float(sv[m].max()),
                             "ade_base": b, "ade_arm": e[m].mean(), "dade": d[m].mean(), "lo": lo, "hi": hi,
                             "rel_gain": d[m].mean() / b, "rel_lo": lo / b, "rel_hi": hi / b})
    return pd.DataFrame(rows)


def rater_bins(set_name: str = "qwen_front3", layer: str = sa.LAYER, seed: int = 0, folds: int = FOLDS,
               q: int = 5, q_pooled: int = 10):
    """Is the top s_ego bin unpredictable, or only unpredicted? Ask the raters, who fitted nothing.

    Stratifying the evaluation half by its *realised* s_ego puts the frames with the most irreducible error
    into the top bin by construction, so the inverted U of `decile_report` has a selection caveat: "vision
    does not buy the large departures" could be "nothing observable at t = 0 buys them". The logged future
    and the rater scores depend on no fitted model, so they separate the two. If the raters disagree with
    what the driver actually did in the top bin, that bin is multi-modal and ADE against the log is the wrong
    target there; if the log stays highly rated, the departures are legible to humans and the miss is ours.

    Everything here is CPU: the ego head is 100-dimensional, and the image head is refitted with numpy at the
    lambda the GPU run selected, which the caller checks against that run's RFS before reading the bins.
    """
    df, rows, fut, ego, img, sub, past = sa.load_all(set_name, layer, seed)
    halves = sa.val_halves(df, seed)
    h = df.sequence.map(halves).to_numpy()[rows]
    seq = df.sequence.to_numpy()[rows]
    pos, rtraj, scores = sa.rfs_rows(df, rows)
    speed = waymo.init_speed(past[rows])
    best = rtraj[np.arange(len(pos)), scores.argmax(1)]
    log_rfs = waymo.rater_feedback_score(fut[pos], rtraj, scores, speed[pos])
    log_ade = ade(fut[pos], best)
    out, pooled_s = [], np.full(len(rows), np.nan)
    for d in (0, 1):
        sp = sa.Halves(df, seq, h == d, h == (1 - d), seed)
        s = ego_surprise(ego, fut, sp, folds)
        pooled_s[sp.val] = s[sp.val]                       # every frame scored by the half that did not fit it
        ev = np.isin(pos, sp.val)                          # rater frames of this direction's evaluation half
        preds = _cpu_heads(ego, img, fut, sp, seed)
        for name, p in preds.items():
            r = waymo.rater_feedback_score(p[pos[ev]], rtraj[ev], scores[ev], speed[pos[ev]])
            out += _bin_rows(s[pos[ev]], q, d, name, rfs=r, floored=(r <= waymo.RFS_FLOOR + 1e-9),
                             ade_rater=ade(p[pos[ev]], best[ev]))
        out += _bin_rows(s[pos[ev]], q, d, "logged_future", rfs=log_rfs[ev],
                         floored=(log_rfs[ev] <= waymo.RFS_FLOOR + 1e-9), ade_rater=log_ade[ev])
    out += _bin_rows(pooled_s[pos], q_pooled, -1, "logged_future", rfs=log_rfs,
                     floored=(log_rfs <= waymo.RFS_FLOOR + 1e-9), ade_rater=log_ade)
    return pd.DataFrame(out)


def _bin_rows(s: np.ndarray, q: int, direction: int, name: str, **cols) -> list:
    """One row per quantile bin of `s`, with the mean of every column and the bin's edges and count."""
    edge = np.quantile(s, np.linspace(0, 1, q + 1))
    b = np.clip(np.searchsorted(edge[1:-1], s, "right"), 0, q - 1)
    return [{"direction": direction, "series": name, "bins": q, "bin": i, "n": int((b == i).sum()),
             "s_lo": float(s[b == i].min()), "s_hi": float(s[b == i].max()),
             **{k: float(v[b == i].mean()) for k, v in cols.items()}} for i in range(q) if (b == i).any()]


def _cpu_heads(ego: np.ndarray, img, fut: np.ndarray, sp, seed: int = 0) -> dict:
    """`ridge ego` and arm A refitted in numpy, at the lambda each picked by grouped CV on the fit half."""
    from sklearn.model_selection import GroupKFold
    T, lams = fut.shape[1], planner.LAM_RIDGE
    Y = fut.reshape(len(fut), -1)
    out = {}
    for name, X0 in (("ridge ego", np.asarray(ego, np.float32)),
                     ("A ridge_late uniform", np.asarray(img, np.float32))):
        mu, sd = X0[sp.train].mean(0), X0[sp.train].std(0)
        X = (X0 - mu) / np.where(sd > 1e-6, sd, 1)
        tgt = Y if name == "ridge ego" else Y - out["ridge ego"].reshape(len(Y), -1)
        score = np.zeros(len(lams))
        for a, b in GroupKFold(4).split(sp.train, groups=sp.seq[sp.train]):
            W = ridge_np(X, tgt, sp.train[a], lams)
            te = sp.train[b]
            p = (X[te] @ W[:, :-1] + W[:, -1, None]).reshape(len(lams), -1, T, 2)
            score += [ade(p[i], tgt[te].reshape(-1, T, 2)).mean() for i in range(len(lams))]
        i = planner._pick(score / 4, lams, f"{name} (cpu)")
        W = ridge_np(X, tgt, sp.train, [lams[i]])
        p = (X @ W[0, :-1] + W[0, -1]).reshape(-1, T, 2)
        out[name] = p if name == "ridge ego" else p + out["ridge ego"]
        log.info("%s (cpu refit): lambda %.3g, eval ADE %.4f", name, lams[i], ade(out[name][sp.val],
                                                                                 fut[sp.val]).mean())
    return out


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="surprise,arms", help="comma list of surprise,arms,rater_bins")
    ap.add_argument("--layer", default=sa.LAYER)
    ap.add_argument("--feature-set", default="qwen_front3")
    ap.add_argument("--directions", default="0,1")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rl = RunLog("waymo_l0", "surprise_weighting")
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    if "surprise" in a.steps and "arms" not in a.steps:   # the surprise tables without fitting any head
        rep, nz = [], []
        df, rows, fut, ego, _, sub, past = sa.load_all(a.feature_set, "vit_mean", a.seed)
        h0 = df.sequence.map(sa.val_halves(df, a.seed)).to_numpy()[rows]
        for d in (int(x) for x in a.directions.split(",")):
            h = h0
            sp = sa.Halves(df, df.sequence.to_numpy()[rows], h == d, h == (1 - d), a.seed)
            s, err = surprise_set(past[rows], fut, ego, sp, a.folds)
            rep.append(surprise_report(s, err, sub, d).merge(surprise_corr(s, past, rows, d),
                                                             on=["direction", "kind"], how="left"))
            nz.append(noise_check(df, rows, past, fut, s, sub, d))
        _write(rl, (("surprise", rep), ("noise_check", nz)))
    if "arms" in a.steps:
        res, pairs, dids, bs, rep, nz, dcs = [], [], [], [], [], [], []
        for d in (int(x) for x in a.directions.split(",")):
            preds, ctx, b, rp, nzd = run_direction(d, a.feature_set, a.layer, a.seed, a.folds, rl)
            r, p_, dd, dec = evaluate(preds, ctx)
            for lst, x in ((res, r), (pairs, p_), (dids, dd), (bs, b), (rep, rp), (nz, nzd), (dcs, dec)):
                lst.append(x)
            del preds
            torch.cuda.empty_cache()
        _write(rl, (("arms", res), ("paired", pairs), ("did", dids), ("schemes", bs),
                    ("surprise", rep), ("noise_check", nz), ("deciles", dcs)))
    if "rater_bins" in a.steps:
        _write(rl, (("rater_bins", [rater_bins(a.feature_set, a.layer, a.seed, a.folds)]),))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
