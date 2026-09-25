"""M-C: dual-stream reaction head on the P5 counterfactual pairs (todos/2026-09-25-reactivity-program.md, M-C and
deviation-log item 4, written before any fit).

  pred(x) = prior(x) + Delta(x)
  prior   `ridge_late` on an openpilot `temporal` tap, exactly experiment 1's examinee (p5_exam.heads, route folds,
          fitted on role == train frames only), so every pair frame is out of sample for it
  Delta   linear, W z(x) + b; z = per-stream standardised [Qwen `L18_last` | openpilot `temporal`], each stream scaled by
          1 / sqrt(d_stream) so both carry the same total variance

Arms (all share the prior, the folds and the exam):
  pair       paired-difference loss on the training folds' pair frames (x+/x- and x+/x_null):
             sum |W (z+ - z-) - [(y+ - y-) - (p+ - p-)]|^2 + mu sum_train |W (z - zbar)|^2 + lam |W|^2,
             mu = n_pair / n_train (fixed), lam by 3-fold route-grouped CV on the paired MSE
  hard       hard-example reweighting control: weighted ridge on the residual y - p over the training frames plus the
             training folds' pair frames as single frames, w = s_ego / mean(s_ego), s_ego = per-frame ADE of `ridge ego`
  uniform    the same with w = 1 (reference only)
  pair qwen / pair op   single-stream controls of `pair`
The exam is p5_exam.exam unchanged; the criteria (pedestrian flips, cut-in paired delta, null false flips) are in
`criteria`.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import p5_exam as E, p5_openpilot, p5_pairs as P, planner, waymo_stage_a as sa
from .common import get_logger

log = get_logger(__name__)
LAMS = 10.0 ** np.arange(-1, 6)            # pre-registered grid (sum-of-squares loss)
INNER = 3
CUTIN = ("HighwayCutIn", "StaticCutIn", "ParkingCutIn")
NULL_FF_MAX = 0.07
PED_FLIP_MIN = 0.20


def _std(X: torch.Tensor, rows) -> torch.Tensor:
    return planner.standardize(X, rows) / np.sqrt(X.shape[1])


def _solve_pair(D, R, Zc, mu, lams):
    """W for every lam of min |D W - R|^2 + mu |Zc W|^2 + lam |W|^2 (one eigh)."""
    ev, V = torch.linalg.eigh((D.T @ D + mu * (Zc.T @ Zc)).double().cpu())
    ev, V = ev.float().to(D.device), V.float().to(D.device)
    B = V.T @ (D.T @ R)
    return [V @ (B / (ev[:, None] + lam)) for lam in lams]


def _solve_weighted(Z, Y, w, lams):
    """(W, b) for every lam of min sum_i w_i |Z_i W + b - Y_i|^2 + lam |W|^2."""
    w = w / w.sum() * len(w)
    mz, my = (w[:, None] * Z).sum(0) / w.sum(), (w[:, None] * Y).sum(0) / w.sum()
    A, B = (Z - mz) * w.sqrt()[:, None], (Y - my) * w.sqrt()[:, None]
    ev, V = torch.linalg.eigh((A.T @ A).double().cpu())
    ev, V = ev.float().to(Z.device), V.float().to(Z.device)
    C = V.T @ (A.T @ B)
    out = []
    for lam in lams:
        W = V @ (C / (ev[:, None] + lam))
        out.append((W, my - mz @ W))
    return out


def _inner_splits(groups: np.ndarray, k: int = INNER):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(k).split(groups, groups=groups))


def fit_fold(f, fold, t, F, Ego, Xop, Q, pr_ip, pr_im, pr_group, rl, tag, pr_fam=None) -> dict:
    """Predictions (n, 40) on every row for every arm, fitted without fold f."""
    n = len(t)
    role, seq = t.role.to_numpy(), t.base_id.to_numpy()
    tr = np.flatnonzero((role == "train") & (fold != f))
    ev = np.flatnonzero((role == "obs") & (fold == f))
    sp = SimpleNamespace(train=tr, val=ev, seq=seq)
    fut = F.reshape(n, 20, 2).cpu().numpy()
    # prior: p5_exam.heads for one tap, but kept on every row
    Xe = planner.standardize(Ego, tr)
    _, st_e, We = sa.ridge_cv(Xe, F, sp, fut)
    base = planner.linear_apply(We, Xe, np.arange(n))[0]
    Xi = planner.standardize(Xop, tr)
    R0 = F - base
    _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
    prior = base + planner.linear_apply(Wp, Xi, np.arange(n))[0]
    s_ego = (base - F).reshape(n, 20, 2).norm(dim=-1).mean(1)
    out = {"prior": prior}
    # pair rows of the training folds
    keep = fold[pr_ip] != f
    ip, im, grp = pr_ip[keep], pr_im[keep], pr_group[keep]
    Rp = (F[ip] - F[im]) - (prior[ip] - prior[im])
    inner = _inner_splits(grp)
    streams = {"pair": (Q, Xop), "pair qwen": (Q,), "pair op": (Xop,)}
    for arm, parts in streams.items():
        Z = torch.cat([_std(x, tr) for x in parts], 1)
        Zc = Z[tr] - Z[tr].mean(0)
        mu = len(ip) / len(tr)
        D = Z[ip] - Z[im]
        score = np.zeros(len(LAMS))
        for a, b in inner:
            Ws = _solve_pair(D[a], Rp[a], Zc, mu, LAMS)
            score += [float(((D[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
        best = int(np.argmin(score))
        W = _solve_pair(D, Rp, Zc, mu, [LAMS[best]])[0]
        out[f"M-C {arm}"] = prior + (Z - Z[tr].mean(0)) @ W
        if pr_fam is not None:      # diagnostic: in-sample fit of the paired target at 2 s speed, per family group
            fam = pr_fam[keep]
            v = lambda y: np.linalg.norm((y[:, 14:16] - y[:, 12:14]).cpu().numpy(), axis=1) / 0.25  # noqa: E731
            fit = prior + (Z - Z[tr].mean(0)) @ W
            dt = v(F[ip]) - v(F[im])
            for grp_name, msk in (("pedestrian", np.isin(fam, E.PED_FAMILIES)), ("cut-in", np.isin(fam, CUTIN))):
                big = msk & (np.abs(dt) > 0.5)
                dm = v(fit[ip]) - v(fit[im])
                dp = v(prior[ip]) - v(prior[im])
                rl.event("mc_insample", fold=f, arm=arm, model=tag, group=grp_name, n=int(big.sum()),
                         slope_fit=float(np.polyfit(dt[big], dm[big], 1)[0]) if big.sum() > 2 else None,
                         slope_prior=float(np.polyfit(dt[big], dp[big], 1)[0]) if big.sum() > 2 else None)
        if arm == "pair":           # sensitivity only: no zero constraint
            W0 = _solve_pair(D, Rp, Zc, 0.0, [LAMS[best]])[0]
            out["M-C pair (mu=0)"] = prior + (Z - Z[tr].mean(0)) @ W0
        rl.event("mc_fold", fold=f, arm=arm, model=tag, lam=float(LAMS[best]), lam_edge=best in (0, len(LAMS) - 1),
                 n_pair=len(ip), n_train=len(tr), mu=mu, inner_mse=(score / len(ip)).tolist())
        rl.log.info("fold %d %s %s: lam %g%s, %d pair rows, %d train rows", f, tag, arm, LAMS[best],
                    " (grid edge)" if best in (0, len(LAMS) - 1) else "", len(ip), len(tr))
    # hard-example reweighting control and uniform imitation, same dual-stream z
    Z = torch.cat([_std(Q, tr), _std(Xop, tr)], 1)
    rows = np.unique(np.r_[tr, ip, im])
    Y = F[rows] - prior[rows]
    g = seq[rows].astype(str)
    inner_r = _inner_splits(g)
    for arm, w in (("hard", s_ego[rows] / s_ego[rows].mean()), ("uniform", torch.ones(len(rows), device=F.device))):
        score = np.zeros(len(LAMS))
        for a, b in inner_r:
            fits = _solve_weighted(Z[rows[a]], Y[a], w[a], LAMS)
            score += [float((w[b] * ((Z[rows[b]] @ W + c - Y[b]) ** 2).sum(1)).sum()) for W, c in fits]
        best = int(np.argmin(score))
        W, c = _solve_weighted(Z[rows], Y, w, [LAMS[best]])[0]
        out[f"M-C {arm}"] = prior + Z @ W + c
        rl.event("mc_fold", fold=f, arm=arm, model=tag, lam=float(LAMS[best]), lam_edge=best in (0, len(LAMS) - 1),
                 n_rows=len(rows))
        rl.log.info("fold %d %s %s: lam %g%s, %d rows", f, tag, arm, LAMS[best],
                    " (grid edge)" if best in (0, len(LAMS) - 1) else "", len(rows))
    rl.log.info("fold %d %s: ridge ego lam %g, prior lam %g", f, tag, st_e["lam"], st_p["lam"])
    return out


def criteria(res: dict, arms, prior: str) -> pd.DataFrame:
    """The three pre-registered checks per arm (deviation-log item 4)."""
    o, taus, fl = res["obs"], res["taus"], res["flips"]
    r = o[o.reactive]

    def flips(sub, ex):
        return ((np.sign(sub[ex]) == np.sign(sub.d_expert)) & E._moved(sub[ex], taus[ex])).astype(float).to_numpy()

    ped, cut = r[r.family.isin(E.PED_FAMILIES)], r[r.family.isin(CUTIN)]
    rows = []
    for ex in arms:
        pf, plo, phi = E.boot_ratio(flips(ped, ex), np.ones(len(ped)), ped.base_id.to_numpy())
        ff = float(fl[(fl.examinee == ex) & (fl.scope == "pooled")].false_flip_null_oos.iloc[0])
        d = flips(cut, ex) - flips(cut, prior)
        cd, clo, chi = E.boot_ratio(d, np.ones(len(cut)), cut.base_id.to_numpy())
        cr = float(flips(cut, ex).mean())
        a, b, c = pf >= PED_FLIP_MIN and plo > ff, chi >= 0, ff <= NULL_FF_MAX
        rows.append({"arm": ex, "ped_reactive": len(ped), "ped_flip": pf, "ped_lo": plo, "ped_hi": phi,
                     "null_ff_oos": ff, "cutin_flip": cr, "cutin_delta_vs_prior": cd, "cutin_lo": clo, "cutin_hi": chi,
                     "a_ped": a, "b_cutin": b, "c_null": c, "pass": a and b and c})
    return pd.DataFrame(rows)


def run(rl, models=("cinque", "lebowski")):
    t, past, fut, obs, null, pairs = E.load()
    n = len(t)
    Q = torch.as_tensor(P.load_features(t, ("L18_last",))["L18_last"], device="cuda")
    op = p5_openpilot.load(t, models)
    fold = E.folds(t, pairs)
    F = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    Ego = torch.as_tensor(E.ego_input(t, past), device="cuda")
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    pr_fam = np.r_[obs.family.to_numpy(), np.full(len(null), "null")].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    preds = {}
    for m in models:
        Xop = torch.as_tensor(op[f"op-{m} temporal"], device="cuda")
        for f in range(E.K_FOLDS):
            ev = obs_rows[fold[obs_rows] == f]
            if not len(ev):
                continue
            o = fit_fold(f, fold, t, F, Ego, Xop, Q, pr_ip, pr_im, pr_group, rl, m, pr_fam)
            for arm, v in o.items():
                key = f"{arm} [{m}]"
                preds.setdefault(key, np.full((n, 20, 2), np.nan, np.float32))[ev] = v[ev].reshape(-1, 20, 2).cpu().numpy()
        del Xop
        torch.cuda.empty_cache()
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(preds))
    crit = pd.concat([criteria(res, [k for k in preds if k.endswith(f"[{m}]")], f"prior [{m}]") for m in models])
    d = rl.dir
    res["flips"].to_csv(d / "flip_rates.csv", index=False)
    crit.to_csv(d / "criteria.csv", index=False)
    res["obs"].to_parquet(d / "obs_scored.parquet", index=False)
    nn.to_parquet(d / "null_scored.parquet", index=False)
    np.savez_compressed(d / "preds_obs.npz", rows=obs_rows, **{k: v[obs_rows] for k, v in preds.items()})
    fl = res["flips"]
    rl.log.info("pooled flips\n%s", fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("criteria\n%s", crit.to_markdown(index=False, floatfmt=".3f"))
    return res, crit


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cinque,lebowski")
    ap.add_argument("--lams", default="", help="log10 range lo,hi of the lam grid; default the pre-registered -1,5 "
                                                 "(wider grids are the post-hoc sensitivity, deviation-log item 5)")
    a = ap.parse_args()
    global LAMS
    if a.lams:
        lo, hi = map(int, a.lams.split(","))
        LAMS = 10.0 ** np.arange(lo, hi + 1)
    rl = RunLog("reactivity", "mc" + (f"-lams{lo}_{hi}" if a.lams else ""))
    run(rl, tuple(a.models.split(",")))
    rl.close()


if __name__ == "__main__":
    main()
