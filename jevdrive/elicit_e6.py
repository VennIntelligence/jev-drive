"""Elicitation E6 (b): a Hydra-MDP-style scoring head on frozen openpilot `temporal` for NAVSIM
(todos/2026-09-26-elicitation-program.md, E6 and deviation [E6] 00:34, written before any fit or score).

  prep   the G2 / G3 K = 1024 vocabulary (recomputed, checked against G3's cls_late outputs), the 20 000-token navtrain
         subset, the anchors file for scripts/elicit_e6_score.py and a hydra train_test_split for the v1.1 metric cache
  fit    per-sub-score linear heads (NC, DAC, EP, TTC, C) over the anchors, weights of the log-score aggregation on
         held-out logs, and the selected anchors on navtest / navhard_two_stage as replay-agent predictions

    python -m jevdrive.elicit_e6 prep [--model cinque]
    python -m jevdrive.elicit_e6 fit <prep run dir> <score dir> [--model cinque]
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import navsim_heads as H, planner, traj
from .common import get_logger

log = get_logger(__name__)
G3 = "runs/navsim_zs/heads/20260925-232810"
N_SUB, SEED_SUB, SEED_HOLD = 20_000, 0, 1
SUBS = ("NC", "DAC", "EP", "TTC", "C")
LAMS = (1e-5, 1e-4, 1e-3, 1e-2)
W_IM, W_MUL, W_ONE = (0, 0.1, 0.5, 1), (1, 2, 5, 10), (0, 0.5, 1, 2, 5)
PDMS_W = np.array([5.0, 5.0, 2.0])              # v1.1 weights of EP, TTC, C (driving direction has weight 0)


def _navtrain(feats: bool) -> dict:
    tr = H.load("navtrain", feats)
    keep = (tr["stage"] == "one") & ~np.isnan(tr["fut"]).any((1, 2))
    return {k: v[keep] for k, v in tr.items()}


def prep(rl, model: str):
    from .common import data_dir
    tr = _navtrain(False)
    fut = tr["fut"]
    # k-means on the CPU: traj.kmeans on the GPU is not reproducible run to run (index_add_ uses float atomics), so
    # G3's own vocabulary cannot be recomputed (deviation [E6] 00:37); this one is fixed once and saved
    F = torch.as_tensor(fut[..., :2].reshape(len(fut), -1))
    A = traj.kmeans(F, H.K, seed=0)
    ids = traj.nearest(F, A, 1)[0][:, 0]
    s_, c_ = np.zeros((H.K, 8)), np.zeros((H.K, 8))
    np.add.at(s_, ids, np.sin(fut[..., 2]))
    np.add.at(c_, ids, np.cos(fut[..., 2]))
    anchors = np.concatenate([A.reshape(H.K, 8, 2).numpy(), np.arctan2(s_, c_)[..., None]], -1).astype(np.float32)
    g3 = np.load(data_dir() / G3 / f"navtest_cls_late_{model}_temporal.npz")["poses"]
    d = np.abs(g3[:, None, :, :2] - anchors[None, :, :, :2]).max((2, 3)).min(1)
    rl.log.info(f"G3 cls_late navtest outputs vs this vocabulary: nearest-anchor (x, y) diff median {np.median(d):.3f}, "
                f"p90 {np.quantile(d, 0.9):.3f}, share <= 1e-4 m {(d <= 1e-4).mean():.3f}; oracle ADE "
                f"{np.linalg.norm(A.numpy()[ids].reshape(-1, 8, 2) - fut[..., :2], axis=-1).mean():.3f} m")
    rng = np.random.default_rng(SEED_SUB)
    sub = np.sort(rng.choice(len(tr["tokens"]), N_SUB, replace=False))
    toks, logs = tr["tokens"][sub], tr["log"][sub]
    np.savez(rl.dir / "anchors.npz", anchors=anchors.astype(np.float32), ids=ids)
    (rl.dir / "tokens.txt").write_text("\n".join(toks) + "\n")
    # hydra split for the v1.1 metric caching: navtrain's scene filter restricted to the subset
    sp = rl.dir / "hydra" / "train_test_split"
    sp.mkdir(parents=True)
    q = lambda xs: "\n".join(f"    - '{x}'" for x in xs)  # noqa: E731
    sp.joinpath("e6sub.yaml").write_text(
        "data_split: trainval\nscene_filter:\n  _target_: navsim.common.dataclasses.SceneFilter\n  _convert_: 'all'\n"
        "  num_history_frames: 4\n  num_future_frames: 10\n  frame_interval: 1\n  has_route: true\n  max_scenes: null\n"
        f"  log_names:\n{q(sorted(set(logs)))}\n  tokens:\n{q(toks)}\n")
    rl.log.info(f"subset: {len(toks)} tokens over {len(set(logs))} logs -> {rl.dir}")


# ---------------------------------------------------------------- fit

def _bce_fit(X, T, rows, lam, iters=100):
    """Linear logits X W + b fitted to soft targets T (n, K) in [0, 1]: mean BCE + lam/2 |W|^2, full-batch L-BFGS."""
    Xr, Tr = X[rows], T[rows]
    W = torch.zeros(X.shape[1], T.shape[1], device=X.device, requires_grad=True)
    b = torch.logit(Tr.mean(0).clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xr @ W + b, Tr) + lam / 2 * W.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def _bce(X, T, rows, W, b) -> float:
    return float(torch.nn.functional.binary_cross_entropy_with_logits(X[rows] @ W + b, T[rows]))


def _cls_logits(Xe_tr, Xf_tr, ids, logs, rows_fit, Xe_ev: dict, Xf_ev: dict, lams: tuple) -> dict:
    """`cls ego` then `cls_late` (late fusion on cls ego's out-of-fold logits) fitted on rows_fit at G3's lambdas;
    logits (n, K) on every evaluation matrix. rows_fit = all rows reproduces G3's head."""
    le, ll = lams
    folds = H._group_folds(logs[rows_fit], H.FOLDS)
    off = torch.zeros(len(rows_fit), H.K, device=H.DEV)
    Xe_f, Xf_f = Xe_tr[rows_fit], Xf_tr[rows_fit]
    tg_f = (ids[rows_fit][:, None], np.ones((len(rows_fit), 1), np.float32))
    for f in range(H.FOLDS):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        W, _ = planner.ce_solve(Xe_f, tg_f, a, [le], H.K)
        off[b] = planner.linear_apply(W, Xe_f, b)[0]
    We, _ = planner.ce_solve(Xe_f, tg_f, np.arange(len(rows_fit)), [le], H.K)
    Wl, _ = planner.ce_solve(Xf_f, tg_f, np.arange(len(rows_fit)), [ll], H.K, offset=off)
    return {s: planner.linear_apply(We, Xe_ev[s], np.arange(len(Xe_ev[s])))[0] +
            planner.linear_apply(Wl, Xf_ev[s], np.arange(len(Xf_ev[s])))[0] for s in Xe_ev}


def _select(im: torch.Tensor, heads: dict, w) -> torch.Tensor:
    """argmax_k of the weighted log-score sum (Hydra-MDP's inference rule)."""
    w_im, w_mul, w_ttc, w_ep, w_c = w
    s = w_im * torch.log_softmax(im, 1)
    s = s + w_mul * (torch.nn.functional.logsigmoid(heads["NC"]) + torch.nn.functional.logsigmoid(heads["DAC"]))
    s = s + w_ttc * torch.nn.functional.logsigmoid(heads["TTC"]) + w_ep * torch.nn.functional.logsigmoid(heads["EP"])
    s = s + w_c * torch.nn.functional.logsigmoid(heads["C"])
    return s.argmax(1)


def fit(rl, prep_dir: Path, score_dir: Path, model: str):
    from .common import data_dir
    g3 = json.loads((data_dir() / G3 / "stats.json").read_text())
    lams = (g3["cls ego K1024"]["lam"], g3[f"cls_late {model} temporal"]["lam"])
    tr = _navtrain(True)
    ev = {s: H.load(s, True) for s in H.EVAL}
    an = np.load(prep_dir / "anchors.npz")
    anchors, ids = an["anchors"], an["ids"]
    # per-anchor scores of the subset
    toks, sub, pdms = [], [], []
    for p in sorted(score_dir.glob("chunk_*.npz")):
        z = np.load(p)
        toks.append(z["tokens"]), sub.append(z["sub"]), pdms.append(z["pdms"])
    toks, sub, pdms = np.concatenate(toks), np.concatenate(sub), np.concatenate(pdms)
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    rows = np.array([pos[t] for t in toks])
    rl.log.info(f"scored subset tokens: {len(rows)}; vocabulary oracle PDMS {pdms.max(1).mean():.4f}")
    # features: standardised ego (+) temporal on all of navtrain, as G3
    Xe, *Xe_ev = H._std(tr["ego"], *(ev[s]["ego"] for s in H.EVAL))
    Xf, *Xf_ev = H._std(tr[model], *(ev[s][model] for s in H.EVAL))
    Xe_ev, Xf_ev = dict(zip(H.EVAL, Xe_ev)), dict(zip(H.EVAL, Xf_ev))
    X = torch.cat([Xe[rows], Xf[rows]], 1)
    Xev = {s: torch.cat([Xe_ev[s], Xf_ev[s]], 1) for s in H.EVAL}
    T = {m: torch.as_tensor(sub[:, :, i], device=H.DEV) for i, m in enumerate(SUBS)}
    hold = H._group_folds(tr["log"][rows], 5, seed=SEED_HOLD) == 0
    fit_r, hold_r = np.flatnonzero(~hold), np.flatnonzero(hold)
    rl.log.info(f"inner split: {len(fit_r)} fit / {len(hold_r)} held-out tokens")
    # (4) lambda per sub-score head on the held-out logs, then refit on all subset rows
    lam_sel, heads_hold, heads_all = {}, {}, {}
    for m in SUBS:
        sc = []
        for lam in LAMS:
            W, b = _bce_fit(X, T[m], fit_r, lam)
            sc.append(_bce(X, T[m], hold_r, W, b))
        k = int(np.argmin(sc))
        lam_sel[m] = LAMS[k]
        W, b = _bce_fit(X, T[m], fit_r, LAMS[k])
        heads_hold[m] = X[hold_r] @ W + b
        W, b = _bce_fit(X, T[m], np.arange(len(rows)), LAMS[k])
        heads_all[m] = {s: Xev[s] @ W + b for s in H.EVAL}
        rl.event("e6_head", sub=m, lam=LAMS[k], bce_hold=sc, edge=k in (0, len(LAMS) - 1))
        rl.log.info(f"{m}: lam {LAMS[k]:g} held-out BCE {sc}")
    # (5) imitation logits on the held-out tokens out of sample: cls ego / cls_late without the held-out logs
    hold_logs = set(tr["log"][rows[hold_r]])
    fit_all = np.flatnonzero(~np.isin(tr["log"], list(hold_logs)))
    im_hold = _cls_logits(Xe, Xf, ids, tr["log"], fit_all, {"h": Xe[rows[hold_r]]}, {"h": Xf[rows[hold_r]]}, lams)["h"]
    P_hold = torch.as_tensor(pdms[hold_r], device=H.DEV)
    ar = torch.arange(len(hold_r), device=H.DEV)
    best, grid = None, []
    for w in itertools.product(W_IM, W_MUL, W_ONE, W_ONE, W_ONE):
        v = float(P_hold[ar, _select(im_hold, heads_hold, w)].mean())
        grid.append((*w, v))
        if best is None or v > best[1]:
            best = (w, v)
    v_im = float(P_hold[ar, im_hold.argmax(1)].mean())
    rl.log.info(f"held-out PDMS: oracle {pdms[hold_r].max(1).mean():.4f}, imitation only {v_im:.4f}, "
                f"best weights {best[0]} -> {best[1]:.4f}")
    np.savetxt(rl.dir / "weight_grid.csv", np.array(grid), delimiter=",", header="w_im,w_mul,w_ttc,w_ep,w_c,pdms_hold",
               comments="")
    # final: G3-equivalent imitation logits on navtest / navhard, check against G3's outputs
    im_ev = _cls_logits(Xe, Xf, ids, tr["log"], np.arange(len(tr["tokens"])), Xe_ev, Xf_ev, lams)
    out = {}
    for s in H.EVAL:
        g3p = np.load(data_dir() / G3 / f"{s}_cls_late_{model}_temporal.npz")["poses"]
        im_sel = im_ev[s].argmax(1).cpu().numpy()
        agree = float((np.abs(anchors[im_sel, :, :2] - g3p[..., :2]).max((1, 2)) < 0.5).mean())
        rl.log.info(f"{s}: refitted cls_late (a') within 0.5 m of G3's output on {agree:.4f} of tokens")
        np.savez(rl.dir / f"{s}_clsref_{model}_temporal.npz", tokens=ev[s]["tokens"], poses=anchors[im_sel])
        sel = _select(im_ev[s], {m: heads_all[m][s] for m in SUBS}, best[0]).cpu().numpy()
        out[s] = sel
        np.savez(rl.dir / f"{s}_hydra_{model}_temporal.npz", tokens=ev[s]["tokens"], poses=anchors[sel])
        rl.log.info(f"{s}: {float((sel == im_sel).mean()):.3f} of tokens keep the imitation argmax")
    stats = {"lam": lam_sel, "weights": best[0], "pdms_hold": best[1], "pdms_hold_imitation": v_im,
             "pdms_hold_oracle": float(pdms[hold_r].max(1).mean()), "n_scored": len(rows), "n_hold": len(hold_r),
             "cls_lams": lams}
    (rl.dir / "stats.json").write_text(json.dumps(stats, indent=1))
    rl.log.info(json.dumps(stats))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "fit"))
    ap.add_argument("prep_dir", nargs="?")
    ap.add_argument("score_dir", nargs="?")
    ap.add_argument("--model", default="cinque")
    a = ap.parse_args()
    rl = RunLog("elicitation", f"e6-{a.step}")
    if a.step == "prep":
        prep(rl, a.model)
    else:
        fit(rl, Path(a.prep_dir), Path(a.score_dir), a.model)
    rl.close()


if __name__ == "__main__":
    main()
