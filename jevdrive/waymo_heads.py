"""P3e: multimodal heads on the Qwen3-VL-4B native-video features, next to the ridge readout.

P3(d'') found `qwenvid_p3` to be the only representation whose pre-onset CI excludes zero in both half-val
directions, but it was read by `ridge_late`, a single-mode linear head. This module swaps the head and keeps
everything else of `waymo_ladder` -- the frozen subset, the two directions, the `ridge ego` base, the judge:

  cls   fixed-vocabulary classification over K = 1024 anchors (decision 8), `planner.Heads.cls` exactly as
        P0 used it: a linear softmax solved by L-BFGS, lambda picked on the fit half's inner split, and for the
        vision arm the ego classifier's logits frozen as an offset (late fusion, decisions 9 / 10).
  diff  a DiffusionDrive-style truncated diffusion head: M = 20 anchors noised to t = 50 of a 1000-step DDPM
        schedule, two DDIM steps back to t = 0, a small transformer over the M mode tokens plus one condition
        token (ego and feature projected separately, so 100 ego dims are not drowned by 2560 feature dims),
        predicting each mode's offset and score. The positive mode is the anchor nearest the logged future.

Both vocabularies are k-means over every train-split logged future: train and val sequences are disjoint, so
this leaks nothing, and 415k trajectories cover the space far better than the ~10k of one fit half.

Every head reports its top-1 (highest-score) mode through the ladder's judge, so it lands on the same ruler as
ridge. The multimodal quantities -- oracle minADE over the vocabulary or the modes, minADE@k over the ranked
modes, the rater trust-region coverage gap -- are diagnostics and are written to their own table.
"""
import math

import numpy as np
import pandas as pd
import torch

from . import planner, traj, waymo, waymo_ladder as lad, waymo_p1, waymo_stage_a as sa
from .common import get_logger

log = get_logger(__name__)
DEV = "cuda"
TAG = "p3e"
TAPS = ("L18_last", "L18_mean")
VOCAB_K, DIFF_M, KMAX = 1024, 20, 20
N_STEPS, T_TRUNC, DDIM_STEPS = 1000, 50, (50, 25)
BETAS = torch.linspace(1e-4, 0.02, N_STEPS, dtype=torch.float64)
ABAR = torch.cumprod(1 - BETAS, 0).float()
DM, LAYERS, HEADS, DROP = 256, 2, 4, 0.1
EPOCHS, BS, LR, WD = 60, 256, 3e-4, 1e-2
VRAM_BUDGET = 30e9                    # the heads stream's share of the card (RESOURCE_LEDGER.md)


# ---------------------------------------------------------------- vocabularies

def train_futures() -> np.ndarray:
    """(n, 20, 2) logged futures of every train-split frame that has one."""
    df = waymo.load_index()
    _, future = waymo.load_ego()
    m = (df.split == "train").to_numpy() & df.has_future.to_numpy()
    return waymo.future_xy(future[m]).astype(np.float32)


def vocabularies(ks=(VOCAB_K, DIFF_M), seed: int = 0) -> tuple[dict, np.ndarray]:
    """k-means anchors per K over the train split, and the (centre, scale) that maps metres to about [-1, 1].

    The scale follows DiffusionDrive: one affine map per coordinate from the 0.5-99.5 % range of all train
    waypoints, not a per-waypoint z-score, so the truncated noise is a fixed number of metres everywhere.
    """
    fut = train_futures()
    F = torch.as_tensor(fut.reshape(len(fut), -1), device=DEV)
    voc = {k: traj.kmeans(F, k, seed=seed) for k in ks}
    lo, hi = np.percentile(fut.reshape(-1, 2), [0.5, 99.5], axis=0)
    norm = np.stack([(hi + lo) / 2, (hi - lo) / 2]).astype(np.float32)          # (2: centre / half-range, 2: x / y)
    log.info("vocabularies from %d train futures: K %s; normalisation centre %s half-range %s",
             len(fut), list(ks), norm[0].round(2), norm[1].round(2))
    return voc, norm


# ---------------------------------------------------------------- classification arm

def _cross_fit(scores: torch.Tensor, Xe: torch.Tensor, tgt, sp, lam: float, K: int, folds: int = 4) -> torch.Tensor:
    """The ego classifier's logits with the fit-half rows replaced by out-of-fold ones.

    P0 froze the ego logits as fitted on every training row, which is harmless at 414k rows. On one fit half
    (~10k rows) the ego classifier picks the smallest lambda and all but memorises its rows, so an in-sample
    offset is near-perfect exactly where the vision head is trained and the head learns nothing: in the
    smoke run `cls_late` was indistinguishable from `cls ego`. Out-of-fold offsets (sequence-grouped, at the
    lambda the ego head chose) are what the vision head will actually face on the evaluation half.
    """
    from sklearn.model_selection import GroupKFold
    out = scores.clone()
    for tr, te in GroupKFold(folds).split(sp.train, groups=sp.seq[sp.train]):
        W, _ = planner.ce_solve(Xe, tgt, sp.train[tr], [lam], K)
        out[sp.train[te]] = planner.linear_apply(W, Xe, sp.train[te])[0]
    return out


def cls_arm(ctx: dict, X, vocab: torch.Tensor, side: dict, name: str):
    """`planner.Heads.cls` on the ladder's rows: ego-only when `X` is None, else late fusion on the ego logits.

    The ego classifier is fitted once per direction and cached in `side`, so every vision arm of that
    direction corrects the same frozen offset.
    """
    def fit(sel, sp, R, res_fut, seed, pre=None):
        fut = ctx["fut"][sel]
        F = torch.as_tensor(fut.reshape(len(sel), -1), device=DEV)
        ids, _ = traj.nearest(F, vocab, 1)
        tgt = (ids, np.ones(ids.shape, np.float32))
        key = ("cls ego", id(sp))
        if key not in side:
            Xe = lad.standardize_np(ctx["ego"][sel], sp.train)
            he = planner.Heads(Xe, sp, fut, F, waymo.RFS_FREQ)
            pv, st = he.cls(tgt, vocab, KMAX, keep=True)
            side[key] = (_cross_fit(he.scores, Xe, tgt, sp, st["lam"], len(vocab)), pv, st)
        scores, pv, st = side[key]
        if X is not None:
            Xi = lad.standardize_np(X[sel], sp.train)
            pv, st = planner.Heads(Xi, sp, fut, F, waymo.RFS_FREQ).cls(tgt, vocab, KMAX, offset=scores)
            del Xi
        torch.cuda.empty_cache()
        side["modes"][name] = pv.astype(np.float32)                   # (n_val, KMAX, T, 2), best first
        side["pool"][name] = ("vocab K1024", vocab)
        return pv[:, 0] - (fut - res_fut)[sp.val], {"K": len(vocab), **st}
    return fit


# ---------------------------------------------------------------- truncated diffusion arm

def _t_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f
    return torch.cat([a.sin(), a.cos()], 1)


class TruncDiffusion(torch.nn.Module):
    """Denoiser over M anchor-initialised modes: x_t (b, M, 2T) -> (x0 estimate, score) per mode."""

    def __init__(self, anchors: torch.Tensor, d_ego: int, d_feat: int = 0, dm: int = DM, layers: int = LAYERS,
                 heads: int = HEADS, drop: float = DROP, feat_drop: float | None = None):
        super().__init__()
        M, T2 = anchors.shape
        self.register_buffer("anchors", anchors)
        self.ego = torch.nn.Sequential(torch.nn.Linear(d_ego, dm), torch.nn.LayerNorm(dm))
        self.feat = torch.nn.Sequential(torch.nn.Dropout(drop if feat_drop is None else feat_drop),
                                        torch.nn.Linear(d_feat, dm),
                                        torch.nn.LayerNorm(dm)) if d_feat else None
        self.inp, self.mode = torch.nn.Linear(T2, dm), torch.nn.Parameter(torch.randn(M, dm) * 0.02)
        self.temb = torch.nn.Sequential(torch.nn.Linear(dm, dm), torch.nn.SiLU(), torch.nn.Linear(dm, dm))
        enc = torch.nn.TransformerEncoderLayer(dm, heads, 4 * dm, drop, batch_first=True, norm_first=True)
        self.enc, self.norm = torch.nn.TransformerEncoder(enc, layers, enable_nested_tensor=False), torch.nn.LayerNorm(dm)
        self.off, self.score = torch.nn.Linear(dm, T2), torch.nn.Linear(dm, 1)
        torch.nn.init.zeros_(self.off.weight)        # start at the anchors: refine, don't replace
        torch.nn.init.zeros_(self.off.bias)
        self.dm = dm

    def forward(self, xt, t, e, f=None):
        c = self.ego(e) + (self.feat(f) if self.feat is not None else 0)
        h = self.inp(xt) + self.mode + self.temb(_t_embed(t, self.dm))[:, None]
        z = self.norm(self.enc(torch.cat([c[:, None], h], 1))[:, 1:])
        # x0 is the mode's anchor plus a learned offset; x_t only conditions it. The first version returned
        # x_t + offset, which asks the net to cancel the truncated noise itself (at t = 50 that is ~0.17 of the
        # half-range, ~8.7 m in x) through a 256-d bottleneck: on the fit half's inner split its positive-mode
        # x0 stayed at 6.1 m against a raw-anchor oracle of 1.7 m. Anchored, the same net refines to 1.15 m.
        return self.anchors + self.off(z), self.score(z).squeeze(-1)


def _noised(anchors: torch.Tensor, t: torch.Tensor, gen=None) -> torch.Tensor:
    ab = ABAR.to(anchors.device)[t - 1][:, None, None]
    eps = torch.randn((len(t), *anchors.shape), device=anchors.device, generator=gen)
    return ab.sqrt() * anchors + (1 - ab).sqrt() * eps


@torch.inference_mode()
def ddim_sample(net: TruncDiffusion, e, f=None, gen=None):
    """Two deterministic DDIM steps (eta = 0) from the anchors noised to T_TRUNC; returns (x0, score)."""
    b, ab = len(e), ABAR.to(e.device)
    x = _noised(net.anchors, torch.full((b,), DDIM_STEPS[0], device=e.device), gen)
    for i, t in enumerate(DDIM_STEPS):
        x0, s = net(x, torch.full((b,), t, device=e.device), e, f)
        if i + 1 < len(DDIM_STEPS):
            tn = DDIM_STEPS[i + 1]
            eps = (x - ab[t - 1].sqrt() * x0) / (1 - ab[t - 1]).sqrt()
            x = ab[tn - 1].sqrt() * x0 + (1 - ab[tn - 1]).sqrt() * eps
    return x0, s


def diff_arm(ctx: dict, X, anchors: torch.Tensor, norm: np.ndarray, side: dict, name: str,
             epochs: int = EPOCHS, pca: int | None = None, feat_drop: float | None = None):
    """The truncated diffusion head, conditioned on the ego state and, when `X` is given, the pooled feature.

    `pca` projects the standardised feature onto its top principal components of the fit half first, and
    `feat_drop` sets the dropout on the feature input: the two remedies for a 2560-dim condition that
    overfits ~10k rows (Stage A2 in the todo).

    Same discipline as `waymo_ladder._train`: train on the fit half's inner-fit rows, pick the epoch count by
    top-1 ADE on its sequence-grouped inner-val rows, refit on the whole fit half for that many epochs, read
    the evaluation half once. Inference noise is seeded, so the predictions are reproducible.
    """
    c, s = (torch.as_tensor(norm[i], device=DEV) for i in (0, 1))
    A = ((anchors.view(len(anchors), -1, 2) - c) / s).reshape(len(anchors), -1)

    def fit(sel, sp, R, res_fut, seed, pre=None):
        fut = ctx["fut"][sel]
        T = fut.shape[1]
        G = torch.as_tensor(fut, device=DEV)
        Gn = ((G - c) / s).reshape(len(sel), -1)
        pos = torch.as_tensor(traj.nearest(G.reshape(len(sel), -1), anchors, 1)[0][:, 0], device=DEV)
        E = lad.standardize_np(ctx["ego"][sel], sp.train)
        Z = lad.standardize_np(X[sel], sp.train) if X is not None else None
        if Z is not None and pca:
            Z = planner.standardize(planner.pca(Z, sp.train, pca), sp.train)
        make = lambda: TruncDiffusion(A, E.shape[1], 0 if Z is None else Z.shape[1],  # noqa: E731
                                      feat_drop=feat_drop).to(DEV)

        def predict(net, rows):
            net.eval()
            gen = torch.Generator(device=DEV).manual_seed(seed)
            out = [ddim_sample(net, E[b], None if Z is None else Z[b], gen)
                   for b in torch.as_tensor(rows, device=DEV).split(4096)]
            x0, sc = torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])
            order = sc.argsort(1, descending=True)
            x0 = x0.gather(1, order[..., None].expand_as(x0)).view(len(rows), -1, T, 2) * s + c
            return x0.cpu().numpy(), sc.gather(1, order).softmax(1).cpu().numpy()

        def run(rows, n_ep, sel_rows):
            torch.manual_seed(seed)
            net = make()
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
            r = torch.as_tensor(rows, device=DEV)
            best = (np.inf, n_ep)
            for ep in range(n_ep):
                net.train()
                for b in r[torch.randperm(len(r), device=DEV)].split(BS):
                    t = torch.randint(1, T_TRUNC + 1, (len(b),), device=DEV)
                    x0, sc = net(_noised(net.anchors, t), t, E[b], None if Z is None else Z[b])
                    p = pos[b]
                    loss = ((x0[torch.arange(len(b), device=DEV), p] - Gn[b]).abs().mean()
                            + torch.nn.functional.cross_entropy(sc, p))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sched.step()
                if sel_rows is not None:
                    m, _ = predict(net, sel_rows)
                    best = min(best, (float(np.linalg.norm(m[:, 0] - fut[sel_rows], axis=-1).mean()), ep + 1))
            return net, best

        _, (sel_ade, ep) = run(sp.fit, epochs, sp.sel)
        net, _ = run(sp.train, ep, None)
        modes, prob = predict(net, sp.val)
        st = {"epochs": ep, "sel_ade": sel_ade, "params": sum(q.numel() for q in net.parameters()), "M": len(A)}
        del net
        torch.cuda.empty_cache()
        side["modes"][name] = modes.astype(np.float32)
        side["pool"][name] = (f"anchors M{len(A)}", anchors)
        return modes[:, 0] - (fut - res_fut)[sp.val], st
    return fit


# ---------------------------------------------------------------- diagnostics (never compared with ridge)

def _dec19(s: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11))[1:-1], s, "right"), 0, 9) <= 8


def _per_mode_rfs(modes, rtraj, scores, speed) -> tuple[np.ndarray, np.ndarray]:
    """(n, M) RFS of every mode on its own, and (n, M) whether it sits inside some rater's trust region."""
    out = [waymo.rater_feedback_score(modes[:, i], rtraj, scores, speed, details=True) for i in range(modes.shape[1])]
    return np.stack([o[0] for o in out], 1), np.stack([o[1][:, 0] for o in out], 1)


def diagnostics(side: dict, sctx: dict) -> pd.DataFrame:
    """Oracle minADE, minADE@k over the ranked modes and the rater trust-region coverage gap, per arm.

    Two pools per arm: the modes the head actually emits (the top-KMAX anchors of the classifier, the M
    denoised modes of the diffusion head) and the fixed set it chooses from (the whole K=1024 vocabulary, the
    M raw anchors). For the diffusion head the gap between the two is what the denoiser refines.
    """
    sp = sctx["sp"]
    v = sp.val
    gt, pre = sctx["fut"][v], sctx["sub"]["pre_onset"][v]
    straight = sctx["sub"]["straight_yaw"][v]
    k19 = _dec19(sctx["s"][v])
    pos, rtraj, rscore = sa.rfs_rows(sctx["df"], sctx["rows"][v])
    speed = waymo.init_speed(sctx["past"][sctx["rows"][v]])[pos]
    rows = []
    for name, modes in side["modes"].items():
        pool_name, pool = side["pool"][name]
        e = np.linalg.norm(modes - gt[:, None], axis=-1).mean(-1)                  # (n, M) ranked
        o = traj.oracle_metrics(pool, gt)["oracle_ade"]
        rfs_m, ins_m = _per_mode_rfs(modes[pos], rtraj, rscore, speed)
        uncov_pool = sa.vocab_coverage_rfs(pool, sctx["rows"][v][pos], sctx["df"]) if len(pos) else np.nan
        r = {"direction": sctx["direction"], "arm": name, "n_modes": modes.shape[1], "pool": pool_name,
             "n_rater": len(pos)}
        for scope, m in (("all", np.ones(len(v), bool)), ("pre_onset dec1-9", pre & k19),
                         ("straight dec1-9", straight & k19)):
            r |= {f"minade1 [{scope}]": e[m, 0].mean(), f"minade6 [{scope}]": e[m, :6].min(1).mean(),
                  f"minade{modes.shape[1]} [{scope}]": e[m].min(1).mean(),
                  f"oracle_pool [{scope}]": o[m].mean()}
        r |= {"rfs_top1": rfs_m[:, 0].mean(), "rfs_oracle_modes": rfs_m.max(1).mean(),
              "uncovered_modes": float((~ins_m.any(1)).mean()), "uncovered_pool": float(np.mean(uncov_pool))}
        r["oracle_modes - oracle_pool [all]"] = e.min(1).mean() - o.mean()   # diffusion: < 0 = the denoiser refines
        rows.append(r)
        log.info("dir %d %-26s top1 ADE %.3f minADE6 %.3f minADE%d %.3f | pool %s oracle %.3f (pre %.3f / "
                 "straight %.3f) | rater uncovered: modes %.3f pool %.3f", sctx["direction"], name,
                 r["minade1 [all]"], r["minade6 [all]"], modes.shape[1], e.min(1).mean(), pool_name,
                 r["oracle_pool [all]"], r["oracle_pool [pre_onset dec1-9]"], r["oracle_pool [straight dec1-9]"],
                 r["uncovered_modes"], r["uncovered_pool"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- head vs ridge on the same tap

def head_vs_ridge(run_dir, tag: str = TAG) -> pd.DataFrame:
    """Decision 22's headline for each head's top-1 against `ridge_late` on the same tap, and the vision
    increment inside each head family against ridge's own (vision arm minus that family's ego arm)."""
    df = waymo.load_index()
    past, future = waymo.load_ego()
    sub = waymo.subsets(df, past, future)
    at = pd.Series(np.arange(len(df)), index=waymo.frame_names(df))
    r, rtraj, rscore = waymo.load_rater(df)
    rname = waymo.frame_names(df)[r]
    out = []
    for f in lad._preds_files(run_dir, tag):
        z = np.load(f, allow_pickle=True)
        d = int(f.stem.rsplit("dir", 1)[1])
        fn = z["frame_name"].astype(str)
        idx = at.reindex(fn).to_numpy().astype(int)
        seq, gt = df.sequence.to_numpy()[idx], z["fut"]
        k19 = _dec19(z["s_ego"])
        masks = {"pre_onset dec1-9": sub["pre_onset"][idx] & k19, "straight dec1-9": sub["straight_yaw"][idx] & k19}
        rp = pd.Series(np.arange(len(fn)), index=fn).reindex(rname).to_numpy()
        ok = ~np.isnan(rp)
        rp = rp[ok].astype(int)
        P = {k[5:]: z[k] for k in z.files if k.startswith("pred_")}
        ade = {k: waymo_p1.ade(p, gt) for k, p in P.items()}
        rfs = {k: waymo.rater_feedback_score(p[rp], rtraj[ok], rscore[ok], z["speed"][rp]) for k, p in P.items()}
        ego = {"ridge": lad.BASE, "cls": "cls ego K1024", "diff": f"diff ego M{DIFF_M}"}
        for tap in TAPS:
            ridge = f"A ridge_late qwenvid {tap}"
            fams = [("cls", f"cls_late qwenvid {tap}")] + [("diff", k) for k in P if k.startswith(f"diff qwenvid {tap}")]
            for fam, arm in fams:
                if arm not in P or ridge not in P:
                    continue
                for sn, m in masks.items():
                    out.append({"direction": d, "tap": tap, "family": fam, "arm": arm, "compare": "head top-1 - ridge_late",
                                "subset": sn, **waymo_p1.paired(ade[arm], ade[ridge], seq, m)})
                    if ego[fam] in P:
                        inc_h, inc_r = ade[arm] - ade[ego[fam]], ade[ridge] - ade[lad.BASE]
                        out.append({"direction": d, "tap": tap, "family": fam, "arm": arm,
                                    "compare": "vision increment: head family - ridge", "subset": sn,
                                    **waymo_p1.paired(inc_h, inc_r, seq, m)})
                out.append({"direction": d, "tap": tap, "family": fam, "arm": arm, "compare": "RFS head top-1 - ridge_late",
                            "subset": "rater frames", **waymo_p1.paired(rfs[arm], rfs[ridge], seq[rp],
                                                                         np.ones(len(rp), bool))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- driver

DIFF_VARIANTS = {"pca16": {"pca": 16}, "pca64": {"pca": 64}, "drop0.5": {"feat_drop": 0.5}}


def build_arms(ctx, feats, voc, norm, side, which, epochs, variants=()):
    arms = {}
    for tap in TAPS:
        arms[f"A ridge_late qwenvid {tap}"] = lad.ridge_arm(feats[tap])
    if "cls" in which:
        arms["cls ego K1024"] = cls_arm(ctx, None, voc[VOCAB_K], side, "cls ego K1024")
        for tap in TAPS:
            n = f"cls_late qwenvid {tap}"
            arms[n] = cls_arm(ctx, feats[tap], voc[VOCAB_K], side, n)
    if "diff" in which:
        n = f"diff ego M{DIFF_M}"
        arms[n] = diff_arm(ctx, None, voc[DIFF_M], norm, side, n, epochs)
        for tap in TAPS:
            n = f"diff qwenvid {tap}"
            arms[n] = diff_arm(ctx, feats[tap], voc[DIFF_M], norm, side, n, epochs)
            for v in variants:
                arms[f"{n} {v}"] = diff_arm(ctx, feats[tap], voc[DIFF_M], norm, side, f"{n} {v}", epochs,
                                            **DIFF_VARIANTS[v])
    return arms


def run(rl, directions=(0, 1), which=("cls", "diff"), epochs: int = EPOCHS, seed: int = 0, variants=()):
    ctx = lad.base_context(seed)
    keep = lad.load_subset(ctx)
    a = lad.align(ctx, lad.QWENVID_SET, list(TAPS))
    keep &= a["covered"]
    feats = {t: a[t] for t in TAPS}
    voc, norm = vocabularies(seed=seed)
    for k, C in voc.items():
        np.save(rl.dir / f"vocab_K{k}.npy", C.view(k, -1, 2).cpu().numpy())
    np.save(rl.dir / "diff_norm.npy", norm)
    tables = {k: [] for k in ("arms", "paired", "did", "deciles", "fits", "diagnostics")}
    for d in directions:
        side = {"modes": {}, "pool": {}}
        s = lad.s_ego_full(ctx, d)
        preds, sctx, fits = lad.run_direction(ctx, keep, build_arms(ctx, feats, voc, norm, side, which, epochs, variants),
                                              d, s, rl)
        for name, t in zip(("arms", "paired", "did", "deciles"), lad.judge(preds, sctx)):
            tables[name].append(t)
        tables["fits"].append(fits)
        tables["diagnostics"].append(diagnostics(side, sctx))
        lad.save_preds(rl, preds, sctx, TAG)
        np.savez_compressed(rl.dir / f"{TAG}_modes_dir{d}.npz", frame_name=sctx["fname"][sctx["sp"].val].astype(str),
                            **{f"modes_{k}": v for k, v in side["modes"].items()})
        del preds, side
        torch.cuda.empty_cache()
    lad.write_tables(rl, tables, TAG)
    for name, t in (lad.rejudge(rl.dir, TAG, seed) | {"head_vs_ridge": head_vs_ridge(rl.dir)}).items():
        t.to_csv(rl.dir / f"{name}.csv", index=False)
        rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
        rl.event(name, rows=t.to_dict("records"))
    rl.log.info("qualifier: %s", lad.CIRCULAR)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--directions", default="0,1")
    ap.add_argument("--heads", default="cls,diff", help="comma list of cls,diff")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="diffusion: max epochs for early stopping")
    ap.add_argument("--tag", default="p3e-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diff-variants", default="", help=f"comma list of {list(DIFF_VARIANTS)}")
    a = ap.parse_args()
    torch.set_num_threads(8)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_BUDGET / total))
    rl = RunLog("waymo_heads", a.tag)
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    run(rl, tuple(int(x) for x in a.directions.split(",")), tuple(a.heads.split(",")), a.epochs, a.seed,
        tuple(v for v in a.diff_variants.split(",") if v))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
