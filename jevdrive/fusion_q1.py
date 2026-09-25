"""Q1 of the fusion diagnostics (todos/2026-09-25-fusion-diagnostics.md): the complementarity matrix.

Do openpilot `temporal` and the Qwen3-VL-4B video tokens together beat the better of the two? Every arm is read by
the unchanged heads and judges; this module only adds arms, a best-single selection made on training rows, and
descriptive readouts.

  arms     singles   openpilot Cinque / Lebowski `temporal`, Qwen `L18_last` / `L18_mean`, V-JEPA 2 `mean` (WOD only)
           concat    Cinque (Lebowski) `temporal` + Qwen `L18_last`: each stream z-scored on the training rows and
                     scaled by 1 / sqrt(d_stream), so both carry the same total variance (M-C's `reactivity_mc._std`);
                     no further per-column standardisation on top
           late      equal-weight average of two singles' out-of-sample predictions (the same two streams as the
                     concat); ridge averages the trajectories, cls takes the top-1 anchor of the summed full logits
  heads    `ridge_late` (`waymo_ladder.ridge_arm`'s fit) and `cls_late` (`waymo_heads.cls_arm`: K = 1024 vocabulary,
           linear softmax by L-BFGS on the frozen out-of-fold `cls ego` logits); P5 is ridge only (`p5_exam.heads`)
  data     wod_a   the p2p3_v1 subset, half/half cross-fit in both directions (entry 40)
           wod_b   fit on the Qwen-thinned train rows, evaluate on the val rows Qwen covers (one direction)
           p5      P5 v0, route 5-fold, `p5_exam.exam`
  readout  per arm: `waymo_ladder.rejudge` (per direction) and `drive_backbones.crossfit` (pooled), plus DiD, RFS
           frame / cluster mean and RFS per WOD cluster; concat (and late) minus the best single, paired, with the
           best single chosen on training rows only (wod_a: the fit half, read through the other direction's
           out-of-sample predictions; wod_b: the inner selection split of the train rows; p5: the other folds)

The deviation log [Q1] in the todo fixes every choice above before any number was read.
"""
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .common import get_logger

log = get_logger(__name__)
DEV = "cuda"
B_DECIDE = 500                      # bootstrap draws of the pre-registered decision quantities (todo: "500 次")
SINGLES = ("op-cinque temporal", "op-lebowski temporal", "qwenvid L18_last", "qwenvid L18_mean", "vjepa2 mean")
PAIRS = (("op-cinque temporal", "qwenvid L18_last"), ("op-lebowski temporal", "qwenvid L18_last"))
WOD_A_SETS = {"op-cinque temporal": ("op_cinque_p3", "temporal", True),
              "op-lebowski temporal": ("op_lebowski_p3", "temporal", True),
              "qwenvid L18_last": ("qwenvid_p3", "L18_last", True), "qwenvid L18_mean": ("qwenvid_p3", "L18_mean", True),
              "vjepa2 mean": ("vjepa2_p3", "mean", True)}
WOD_B_SETS = {"op-cinque temporal": ("op_cinque_p3_trainval", "temporal", True),
              "op-lebowski temporal": ("op_lebowski_p3_trainval", "temporal", True),
              "qwenvid L18_last": ("qwenvid_train_t4", "L18_last", False),
              "qwenvid L18_mean": ("qwenvid_train_t4", "L18_mean", False),
              "vjepa2 mean": ("vjepa2_p3_trainval", "mean", True)}
TAG = {"wod_a": "q1a", "wod_b": "q1b"}
CLS_EGO = "cls ego K1024"


def concat_name(a, b):
    return f"concat {a} + {b}"


def late_name(a, b):
    return f"late {a} + {b}"


# ---------------------------------------------------------------- standardisers

def std_cols(X):
    """The heads' own rule (`waymo_ladder.standardize_np`): every column z-scored on the training rows."""
    from . import waymo_ladder as L
    return lambda sel, rows: L.standardize_np(X[sel], rows)


def std_two(parts):
    """Per-stream z-score on the training rows, each stream times 1 / sqrt(d): equal total variance per stream."""
    from . import waymo_ladder as L
    return lambda sel, rows: torch.cat([L.standardize_np(p[sel], rows) / math.sqrt(p.shape[1]) for p in parts], 1)


# ---------------------------------------------------------------- WOD arms

def _sel_ctx(ctx, sel, sp, side):
    """wod_b's selection split: the ego ridge fitted on sp.fit alone, its residual as the target on sp.fit and its
    per-frame ADE on sp.sel as that split's own s_ego. Cached once per direction."""
    from . import planner, waymo_stage_a as sa
    key = ("selctx", id(sp))
    if key not in side:
        fut = ctx["fut"][sel]
        n, T = len(sel), fut.shape[1]
        F = torch.as_tensor(fut.reshape(n, -1), device=DEV)
        spf = SimpleNamespace(train=sp.fit, val=sp.sel, seq=sp.seq)
        Xe = planner.standardize(torch.as_tensor(ctx["ego"][sel], device=DEV), sp.fit)
        pe, st, We = sa.ridge_cv(Xe, F, spf, fut)
        base = planner.linear_apply(We, Xe, np.arange(n))[0]
        R = F - base
        side[key] = {"spf": spf, "R": R, "res": R.reshape(-1, T, 2).cpu().numpy(),
                     "base_sel": base[sp.sel].reshape(-1, T, 2).cpu().numpy()}
        side["sel"] = {"fname": ctx["fname"][sel][sp.sel], "fut": fut[sp.sel], "ridge ego": pe[:, 0],
                       "pre_onset": ctx["sub"]["pre_onset"][sel][sp.sel],
                       "s_ego": np.linalg.norm(pe[:, 0] - fut[sp.sel], axis=-1).mean(1), "lam_ego": st["lam"]}
    return side[key]


def ridge_fit(get, side, name, ctx=None):
    """`waymo_ladder.ridge_arm`'s fit with the standardiser as a parameter (std_cols(X) is ridge_arm exactly).
    With `ctx`, also the selection-split prediction (fit on sp.fit, read on sp.sel) for the best-single choice."""
    from . import waymo_stage_a as sa

    def fit(sel, sp, R, res_fut, seed, pre=None):
        Xi = get(sel, sp.train)
        p, st, _ = sa.ridge_cv(Xi, R, sp, res_fut)
        d = Xi.shape[1]
        del Xi
        side[(name, id(sp))] = p[:, 0]
        if ctx is not None:
            sc = _sel_ctx(ctx, sel, sp, side)
            Xf = get(sel, sp.fit)
            ps, st_s, _ = sa.ridge_cv(Xf, sc["R"], sc["spf"], sc["res"])
            side["sel"][name] = ps[:, 0] + sc["base_sel"]
            st["sel_lam"] = st_s["lam"]
            del Xf
        torch.cuda.empty_cache()
        return p[:, 0], {"d": d, **st}
    return fit


def cls_fit(ctx, get, vocab, side, name, keep_sel: bool = False):
    """`waymo_heads.cls_arm` for a feature arm, with `planner.Heads.cls` written out line by line so that the full
    evaluation logits (offset included) and the selection-split top-1 at the chosen lambda are kept. The numbers are
    the same calls in the same order; `check_cls` verifies it against cls_arm itself."""
    from . import planner, traj, waymo_heads as H

    def fit(sel, sp, R, res_fut, seed, pre=None):
        fut = ctx["fut"][sel]
        F = torch.as_tensor(fut.reshape(len(sel), -1), device=DEV)
        ids, _ = traj.nearest(F, vocab, 1)
        tgt = (ids, np.ones(ids.shape, np.float32))
        scores = side[("cls ego", id(sp))][0]          # the ego arm (cls_arm, unchanged) runs first and caches it
        Xi = get(sel, sp.train)
        K = len(vocab)
        A = vocab.reshape(K, -1, 2).cpu().numpy()
        W, st = planner.ce_solve(Xi, tgt, sp.fit, planner.LAM_CLS, K, offset=scores)
        top = planner.cls_topk(W, Xi, sp.sel, 1, offset=scores)
        s = [float(np.linalg.norm(A[top[:, i, 0]] - fut[sp.sel], axis=-1).mean()) for i in range(len(planner.LAM_CLS))]
        best = planner._pick(s, planner.LAM_CLS, "cls")
        del W
        W, st2 = planner.ce_solve(Xi, tgt, sp.train, [planner.LAM_CLS[best]], K, offset=scores)
        pv = A[planner.cls_topk(W, Xi, sp.val, H.KMAX, offset=scores)[:, 0]]
        logits = torch.cat([planner.linear_apply(W, Xi, c)[0] + scores[c] for c in np.array_split(sp.val, 8)])
        del W, Xi
        torch.cuda.empty_cache()
        base = (fut - res_fut)[sp.val]
        side[(name, id(sp))] = {"logits": logits, "A": A, "base": base}
        if keep_sel:
            _sel_ctx(ctx, sel, sp, side)
            side["sel"][name] = A[top[:, best, 0]]
        st = {"K": K, "lam": float(planner.LAM_CLS[best]), "sel_ade": s[best],
              **{f"sel_{k}": v for k, v in st.items()}, **st2}
        return pv[:, 0] - base, st
    return fit


def late_ridge(side, a, b):
    def fit(sel, sp, R, res_fut, seed, pre=None):
        return (side[(a, id(sp))] + side[(b, id(sp))]) / 2, {"late": f"{a} | {b}"}
    return fit


def late_cls(side, a, b):
    def fit(sel, sp, R, res_fut, seed, pre=None):
        x, y = side[(a, id(sp))], side[(b, id(sp))]
        top = (x["logits"] + y["logits"]).argmax(1).cpu().numpy()
        agree = float((x["logits"].argmax(1) == y["logits"].argmax(1)).float().mean())
        return x["A"][top] - x["base"], {"late": f"{a} | {b}", "top1_agree": agree}
    return fit


def build_arms(ctx, feats, vocab, side, sel_split: bool, heads=("ridge", "cls")):
    from . import waymo_heads as H
    arms = {}
    c = ctx if sel_split else None
    if "ridge" in heads:
        for n in SINGLES:
            arms[f"ridge_late {n}"] = ridge_fit(std_cols(feats[n]), side, f"ridge_late {n}", c)
        for a, b in PAIRS:
            arms[f"ridge_late {concat_name(a, b)}"] = ridge_fit(std_two([feats[a], feats[b]]), side,
                                                                f"ridge_late {concat_name(a, b)}")
            arms[f"ridge_late {late_name(a, b)}"] = late_ridge(side, f"ridge_late {a}", f"ridge_late {b}")
    if "cls" in heads:
        arms[CLS_EGO] = H.cls_arm(ctx, None, vocab, side, CLS_EGO)
        for n in SINGLES:
            arms[f"cls_late {n}"] = cls_fit(ctx, std_cols(feats[n]), vocab, side, f"cls_late {n}", sel_split)
        for a, b in PAIRS:
            arms[f"cls_late {concat_name(a, b)}"] = cls_fit(ctx, std_two([feats[a], feats[b]]), vocab, side,
                                                            f"cls_late {concat_name(a, b)}")
            arms[f"cls_late {late_name(a, b)}"] = late_cls(side, f"cls_late {a}", f"cls_late {b}")
    return arms


def wod_context(protocol: str):
    from . import drive_backbones as D, waymo_ladder as L
    if protocol == "wod_a":
        ctx = L.base_context()
        keep, sets = L.load_subset(ctx), WOD_A_SETS
    else:
        ctx = L.train_context(p0_run=D.P0_RUN)
        keep, sets = np.ones(len(ctx["fname"]), bool), WOD_B_SETS
    feats, cache = {}, {}
    for n, (st, arr, flat) in sets.items():
        if st not in cache:
            want = [a for (s2, a, _) in sets.values() if s2 == st]
            cache[st] = L.align(ctx, st, want, flat=flat)
        keep &= cache[st]["covered"]
        feats[n] = cache[st][arr]
    return ctx, keep, feats


def run_wod(rl, protocol: str, heads=("ridge", "cls")):
    """One protocol end to end: fit every arm (`waymo_ladder.run_ladder`, unchanged), then the unchanged judges."""
    from . import drive_backbones as D, waymo_heads as H, waymo_ladder as L
    t0 = time.time()
    ctx, keep, feats = wod_context(protocol)
    tag = TAG[protocol]
    voc = H.vocabularies((H.VOCAB_K,))[0][H.VOCAB_K] if "cls" in heads else None
    side = {"modes": {}, "pool": {}}
    b = protocol == "wod_b"
    arms = build_arms(ctx, feats, voc, side, b, heads)
    rl.event("q1_rows", protocol=protocol, rows=int(keep.sum()), evaluated=int((keep & (ctx["half"] == 1)).sum()),
             fit=int((keep & (ctx["half"] == 0)).sum()), arms=list(arms))
    log.info("%s: %d arms over %d rows (half 0: %d, half 1: %d)", protocol, len(arms), int(keep.sum()),
             int((keep & (ctx["half"] == 0)).sum()), int((keep & (ctx["half"] == 1)).sum()))
    L.run_ladder(lambda k: arms, tag, keep, ctx, directions=(0,) if b else (0, 1), rl=rl)
    if b and "sel" in side:
        s = side["sel"]
        np.savez_compressed(rl.dir / f"{tag}_selsplit.npz", **{k if k in ("fname", "fut", "pre_onset", "s_ego", "lam_ego")
                                                              else f"pred_{k}": v for k, v in s.items()})
    for name, t in L.rejudge(rl.dir, tag).items():
        t.to_csv(rl.dir / f"{name}_{tag}.csv", index=False)
    if not b:
        for name, t in D.crossfit(rl.dir, tag).items():
            if not name.startswith("_"):
                t.to_csv(rl.dir / f"{name}_{tag}.csv", index=False)
    rl.event("q1_fit_done", protocol=protocol, seconds=time.time() - t0)
    return tag


# ---------------------------------------------------------------- WOD readouts

def _choose(scores: dict, higher: bool) -> str:
    ok = {k: v for k, v in scores.items() if np.isfinite(v)}
    return (max if higher else min)(ok, key=ok.get)


def wod_readout(run_dir, protocol: str) -> dict:
    """Per-arm readouts on the pooled out-of-sample predictions and concat / late minus the best single."""
    from . import drive_backbones as D, traj, waymo as W, waymo_p1 as P1
    from .waymo_ladder import BASE
    tag = TAG[protocol]
    d = D._pooled(run_dir, tag)
    preds = {k[2:]: v for k, v in d.items() if k.startswith("p:")}
    natives = D.native_preds(d["fname"])
    gt, seq, pos = d["gt"], d["seq"], d["rpos"]
    rseq, cl = seq[pos], d["cluster"][pos]
    allp = preds | natives
    e = {k: np.linalg.norm(p - gt, axis=-1).mean(1) for k, p in allp.items()}
    rfs = {k: W.rater_feedback_score(p[pos], d["rtraj"], d["rscore"], d["speed"][pos]) for k, p in allp.items()}
    k19 = d["dec"] <= 8
    pre, straight = d["sub"]["pre_onset"] & k19, d["sub"]["straight_yaw"] & k19
    ones = np.ones(len(pos), bool)

    rows = []
    for a in allp:
        for ref in [BASE] + ([CLS_EGO] if a.startswith("cls_late") else []):
            if a == ref:
                continue
            dd = e[a] - e[ref]
            did, dlo, dhi = traj.boot_did(dd, seq, pre, straight)
            rp = P1.paired(e[a], e[ref], seq, pre)
            ra = P1.paired(e[a], e[ref], seq, k19)
            rs = P1.paired(rfs[a], rfs[ref], rseq, ones)
            cm, cm_ref = W.rfs_by_cluster(rfs[a], cl)[0], W.rfs_by_cluster(rfs[ref], cl)[0]
            rows.append({"protocol": protocol, "arm": a, "vs": ref, "n_pre19": rp["n"], "pre19_delta": rp["delta"],
                         "pre19_lo": rp["lo"], "pre19_hi": rp["hi"], "all19_delta": ra["delta"], "all19_lo": ra["lo"],
                         "all19_hi": ra["hi"], "straight19_delta": float(dd[straight].mean()), "did": did,
                         "did_lo": dlo, "did_hi": dhi, "n_rater": len(pos), "rfs_frame": float(rfs[a].mean()),
                         "rfs_frame_delta": rs["delta"], "rfs_lo": rs["lo"], "rfs_hi": rs["hi"], "rfs_cluster": cm,
                         "rfs_cluster_delta": cm - cm_ref})
    per_arm = pd.DataFrame(rows)
    by_cluster = pd.DataFrame({k: W.rfs_by_cluster(v, cl)[1] for k, v in rfs.items()}).T
    by_cluster.insert(0, "cluster_mean", by_cluster.mean(1))
    by_cluster.loc["n frames"] = pd.Series(cl).value_counts()
    by_cluster = by_cluster.reset_index(names="arm")

    # best single per direction, chosen on rows this direction never evaluates
    sel_scores = []
    fam = {"ridge": "ridge_late", "cls": "cls_late"}
    best = {}                                                  # (family, readout, direction) -> arm
    if protocol == "wod_a":
        for f, pre_ in fam.items():
            for dirn in (0, 1):
                s_rows = d["dir"] == 1 - dirn                  # the fit half of `dirn`, as the other direction read it
                sc_pre = {n: float(e[f"{pre_} {n}"][s_rows & pre].mean()) for n in SINGLES}
                rs_rows = s_rows[pos]
                sc_rfs = {n: float(rfs[f"{pre_} {n}"][rs_rows].mean()) for n in SINGLES}
                best[(f, "pre", dirn)], best[(f, "rfs", dirn)] = _choose(sc_pre, False), _choose(sc_rfs, True)
                for n in SINGLES:
                    sel_scores.append({"family": f, "direction": dirn, "single": n, "sel_rows": "fit half (other direction's oos)",
                                       "n_pre19": int((s_rows & pre).sum()), "pre19_ade": sc_pre[n],
                                       "n_rater": int(rs_rows.sum()), "rfs_frame": sc_rfs[n]})
    else:
        z = np.load(Path(run_dir) / f"{tag}_selsplit.npz", allow_pickle=True)
        s = z["s_ego"]
        dec = np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11))[1:-1], s, "right"), 0, 9)
        m = z["pre_onset"] & (dec <= 8)
        for f, pre_ in fam.items():
            sc = {n: float(np.linalg.norm(z[f"pred_{pre_} {n}"] - z["fut"], axis=-1).mean(1)[m].mean())
                  for n in SINGLES if f"pred_{pre_} {n}" in z.files}
            best[(f, "pre", 0)] = best[(f, "rfs", 0)] = _choose(sc, False)   # train rows carry no rater scores
            for n, v in sc.items():
                sel_scores.append({"family": f, "direction": 0, "single": n, "sel_rows": "train inner selection split",
                                   "n_pre19": int(m.sum()), "pre19_ade": v, "n_rater": 0, "rfs_frame": np.nan})
    sel_scores = pd.DataFrame(sel_scores)

    dirs = sorted(np.unique(d["dir"]))
    comp = []
    for f, pre_ in fam.items():
        if f"{pre_} {SINGLES[0]}" not in preds:
            continue
        for ro in ("pre", "rfs"):
            chosen = {dirn: best[(f, ro, dirn)] for dirn in dirs}
            pb = np.empty_like(gt)
            for dirn in dirs:
                mm = d["dir"] == dirn
                pb[mm] = preds[f"{pre_} {chosen[dirn]}"][mm]
            eb = np.linalg.norm(pb - gt, axis=-1).mean(1)
            rb = W.rater_feedback_score(pb[pos], d["rtraj"], d["rscore"], d["speed"][pos])
            label = "best single [" + ", ".join(f"dir{k}: {v}" for k, v in chosen.items()) + "]"
            for a, b in PAIRS:
                for kind, arm in (("concat", f"{pre_} {concat_name(a, b)}"), ("late", f"{pre_} {late_name(a, b)}")):
                    for vs_name, ve, vr in ((label, eb, rb), (f"{pre_} {a}", e[f"{pre_} {a}"], rfs[f"{pre_} {a}"]),
                                            (f"{pre_} {b}", e[f"{pre_} {b}"], rfs[f"{pre_} {b}"])):
                        if ro == "pre":
                            dv = (e[arm] - ve)[pre]
                            lo, hi = traj.boot_ci(dv, seq[pre], b=B_DECIDE)
                            extra = {}
                        else:
                            dv = rfs[arm] - vr
                            lo, hi = traj.boot_ci(dv, rseq, b=B_DECIDE)
                            extra = {"cluster_mean_delta": W.rfs_by_cluster(rfs[arm], cl)[0] - W.rfs_by_cluster(vr, cl)[0]}
                        comp.append({"protocol": protocol, "family": f, "readout": "pre-onset dec1-9 ADE" if ro == "pre"
                                     else "RFS frame", "kind": kind, "arm": arm, "vs": vs_name,
                                     "is_best_single": vs_name == label, "n": len(dv), "delta": float(dv.mean()),
                                     "lo": lo, "hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0), **extra})
    return {"per_arm": per_arm, "rfs_by_cluster": by_cluster, "best_single_selection": sel_scores,
            "concat_vs_best": pd.DataFrame(comp)}


# ---------------------------------------------------------------- P5

def _p5_heads_std(t, past, fut, X: dict, fold, rl, std) -> dict:
    """`p5_exam.heads` with the feature standardiser as a parameter (the loop, folds and solver are the same)."""
    from . import p5_exam as E, planner, waymo_stage_a as sa
    n = len(t)
    F = torch.as_tensor(fut.reshape(n, -1), device=DEV)
    Ego = torch.as_tensor(E.ego_input(t, past), device=DEV)
    seq, role = t.base_id.to_numpy(), t.role.to_numpy()
    out = {h: np.full((n, 20, 2), np.nan, np.float32) for h in X}
    for f in range(E.K_FOLDS):
        tr = np.flatnonzero((role == "train") & (fold != f))
        ev = np.flatnonzero((role == "obs") & (fold == f))
        if not len(ev):
            continue
        sp = SimpleNamespace(train=tr, val=ev, seq=seq)
        Xe = planner.standardize(Ego, tr)
        _, _, We = sa.ridge_cv(Xe, F, sp, fut)
        base = planner.linear_apply(We, Xe, np.arange(n))[0]
        R = F - base
        res = R.reshape(n, 20, 2).cpu().numpy()
        for h, parts in X.items():
            Xi = std(parts, tr)
            pr, stv, _ = sa.ridge_cv(Xi, R, sp, res)
            out[h][ev] = pr[:, 0] + base[ev].reshape(-1, 20, 2).cpu().numpy()
            rl.event("head_fold", fold=f, head=h, lam=stv["lam"], n_train=len(tr), n_obs=len(ev))
            del Xi
        torch.cuda.empty_cache()
    return out


def _std_p5_cols(parts, tr):
    from . import planner
    return planner.standardize(torch.as_tensor(parts[0], device=DEV), tr)


def _std_p5_two(parts, tr):
    from . import planner
    return torch.cat([planner.standardize(torch.as_tensor(p, device=DEV), tr) / math.sqrt(p.shape[1]) for p in parts], 1)


def _flip(o: pd.DataFrame, ex: str, tau: float) -> np.ndarray:
    from . import p5_exam as E
    return ((np.sign(o[ex]) == np.sign(o.d_expert)) & E._moved(o[ex], tau)).astype(float).to_numpy()


def run_p5(rl, op_sub: str = "op_streams") -> dict:
    from . import p5_exam as E, p5_openpilot, p5_pairs as P
    t0 = time.time()
    t, past, fut, obs, null, pairs = E.load()
    X = P.load_features(t) | p5_openpilot.load(t, ("cinque", "lebowski"), ("temporal",), op_sub)
    X = {"qwenvid L18_last": X["L18_last"], "qwenvid L18_mean": X["L18_mean"],
         "op-cinque temporal": X["op-cinque temporal"], "op-lebowski temporal": X["op-lebowski temporal"]}
    singles = [s for s in SINGLES if s in X]
    fold = E.folds(t, pairs)
    rl.log.info("P5 set %s (op %s): %d frames, %d pair frames, %d null frames; singles %s", P.processed(), op_sub,
                len(t), len(obs), len(null), singles)
    heads = E.heads(t, past, fut, X, fold, rl)                           # unchanged: ridge ego + one ridge_late per tap
    preds = {"ridge ego": heads["ridge ego"]} | {f"ridge_late {k}": heads[f"ridge_late {k}"] for k in singles}
    # equivalence: the parametrised loop with the heads' own standardiser reproduces p5_exam.heads
    chk = _p5_heads_std(t, past, fut, {"qwenvid L18_last": [X["qwenvid L18_last"]]}, fold, rl, _std_p5_cols)
    ev = ~np.isnan(chk["qwenvid L18_last"][:, 0, 0])
    diff = float(np.abs(chk["qwenvid L18_last"][ev] - preds["ridge_late qwenvid L18_last"][ev]).max())
    rl.event("p5_equivalence", max_abs_diff=diff, rows=int(ev.sum()))
    rl.log.info("P5 equivalence: parametrised heads vs p5_exam.heads on L18_last, max |diff| %.3g over %d rows", diff, ev.sum())
    cc = _p5_heads_std(t, past, fut, {concat_name(a, b): [X[a], X[b]] for a, b in PAIRS}, fold, rl, _std_p5_two)
    for a, b in PAIRS:
        preds[f"ridge_late {concat_name(a, b)}"] = cc[concat_name(a, b)]
        preds[f"ridge_late {late_name(a, b)}"] = (preds[f"ridge_late {a}"] + preds[f"ridge_late {b}"]) / 2

    # best single per evaluation fold, chosen on the other four folds' frames
    o0, n0 = E.deltas(obs, null, t, {k: preds[k] for k in [f"ridge_late {s}" for s in singles]})
    r0 = E.exam(o0, n0, pairs, [f"ridge_late {s}" for s in singles])
    ob = r0["obs"]
    fmap = pd.Series(fold, index=t.base_id.to_numpy()).groupby(level=0).first()
    ob_f, nu_f = ob.base_id.map(fmap).to_numpy(), n0.base_id.map(fmap).to_numpy()
    pooled_fams = r0["pooled_families"]
    rows, choice = [], {}
    for f in range(E.K_FOLDS):
        o_s = ob[(ob_f != f) & ob.reactive.to_numpy()]
        sc = {}
        for s in singles:
            ex = f"ridge_late {s}"
            tau = float(np.quantile(np.abs(n0[nu_f != f][ex].dropna()), 0.95))
            fp = _flip(o_s[o_s.family.isin(pooled_fams)], ex, tau).mean()
            fped = _flip(o_s[o_s.family.isin(E.PED_FAMILIES)], ex, tau).mean()
            sc[s] = (fp, fped)
            rows.append({"fold": f, "single": s, "tau_other_folds": tau, "flip_pooled_other_folds": fp,
                         "flip_ped_other_folds": fped})
        order = {s: i for i, s in enumerate(singles)}
        choice[("pooled", f)] = max(singles, key=lambda s: (sc[s][0], -order[s]))
        choice[("ped", f)] = max(singles, key=lambda s: (sc[s][1], sc[s][0], -order[s]))
    sel_scores = pd.DataFrame(rows)
    fold_row = fold
    for ro in ("pooled", "ped"):
        p = np.full_like(preds["ridge ego"], np.nan)
        for f in range(E.K_FOLDS):
            m = fold_row == f
            p[m] = preds[f"ridge_late {choice[(ro, f)]}"][m]
        preds[f"best single ({ro} readout)"] = p
    sel_scores["chosen_pooled"] = sel_scores.fold.map(lambda f: choice[("pooled", f)])
    sel_scores["chosen_ped"] = sel_scores.fold.map(lambda f: choice[("ped", f)])

    o, n = E.deltas(obs, null, t, preds)
    ex = list(E.TFV6) + list(preds)
    res = E.exam(o, n, pairs, ex)                                          # unchanged
    r = res["obs"][res["obs"].reactive]
    taus, fl = res["taus"], res["flips"]
    comp = []
    for ro, scope_rows in (("pooled", r[r.family.isin(res["pooled_families"])]), ("ped", r[r.family.isin(E.PED_FAMILIES)])):
        bname = f"best single ({ro} readout)"
        for a, b in PAIRS:
            for kind, arm in (("concat", f"ridge_late {concat_name(a, b)}"), ("late", f"ridge_late {late_name(a, b)}")):
                for vs in (bname, f"ridge_late {a}", f"ridge_late {b}"):
                    dv = _flip(scope_rows, arm, taus[arm]) - _flip(scope_rows, vs, taus[vs])
                    dlt, lo, hi = E.boot_ratio(dv, np.ones(len(dv)), scope_rows.base_id.to_numpy(), b=B_DECIDE)
                    comp.append({"protocol": "p5", "family": "ridge", "readout": f"flip rate ({ro})", "kind": kind,
                                 "arm": arm, "vs": vs, "is_best_single": vs == bname, "n": len(dv),
                                 "routes": scope_rows.base_id.nunique(), "delta": dlt, "lo": lo, "hi": hi,
                                 "ci_excludes_0": bool(lo > 0 or hi < 0),
                                 "arm_rate": float(_flip(scope_rows, arm, taus[arm]).mean()),
                                 "vs_rate": float(_flip(scope_rows, vs, taus[vs]).mean())})
    d = rl.dir
    fl.to_csv(d / "p5_flip_rates.csv", index=False)
    res["validity"].to_csv(d / "p5_label_validity.csv", index=False)
    res["obs"].to_parquet(d / "p5_obs_scored.parquet", index=False)
    n.to_parquet(d / "p5_null_scored.parquet", index=False)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    np.savez_compressed(d / "p5_preds_obs.npz", rows=obs_rows, **{k: v[obs_rows] for k, v in preds.items()})
    rl.event("q1_p5_done", seconds=time.time() - t0, equivalence_max_abs_diff=diff)
    return {"p5_flips": fl, "p5_best_single_selection": sel_scores, "p5_concat_vs_best": pd.DataFrame(comp)}


# ---------------------------------------------------------------- checks

def check_cls(rl, arm: str = "qwenvid L18_last", direction: int = 0) -> dict:
    """cls_fit reproduces waymo_heads.cls_arm: same rows (wod_a, one direction), same ego offset, top-1 compared."""
    from . import waymo_heads as H, waymo_ladder as L
    ctx, keep, feats = wod_context("wod_a")
    voc = H.vocabularies((H.VOCAB_K,))[0][H.VOCAB_K]
    side = {"modes": {}, "pool": {}}
    arms = {CLS_EGO: H.cls_arm(ctx, None, voc, side, CLS_EGO),
            "cls_arm (waymo_heads)": H.cls_arm(ctx, feats[arm], voc, side, "cls_arm (waymo_heads)"),
            "cls_fit (fusion_q1)": cls_fit(ctx, std_cols(feats[arm]), voc, side, "cls_fit (fusion_q1)"),
            "late self": late_cls(side, "cls_fit (fusion_q1)", "cls_fit (fusion_q1)")}
    t0 = time.time()
    preds, sctx, fits = L.run_direction(ctx, keep, arms, direction, L.s_ego_full(ctx, direction), rl)
    a, b, c = preds["cls_arm (waymo_heads)"], preds["cls_fit (fusion_q1)"], preds["late self"]
    out = {"rows": len(a), "top1_identical": float((np.abs(a - b).max((1, 2)) == 0).mean()),
           "late_self_identical": float((np.abs(c - b).max((1, 2)) == 0).mean()),
           "max_abs_diff": float(np.abs(a - b).max()), "seconds": time.time() - t0,
           "lam": fits.set_index("arm").lam.to_dict()}
    rl.event("check_cls", **out)
    rl.log.info("check_cls: %s", out)
    return out


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="p5,wod_a,wod_b", help="comma list of check_cls,p5,wod_a,wod_b,readout")
    ap.add_argument("--heads", default="ridge,cls")
    ap.add_argument("--run", default=None, help="readout: the run directory to read (default: this run)")
    ap.add_argument("--vram-gb", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--op-sub", default="op_streams", help="p5: openpilot stream dir (P5 v1: op_streams_vis)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / total))
    import os
    p5_set = os.environ.get("P5_SET", "carla_p5")
    rl = RunLog("fusion_diag", "q1" if p5_set == "carla_p5" else f"q1-{p5_set.removeprefix('carla_')}")
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    heads = tuple(a.heads.split(","))
    steps = a.steps.split(",")
    run_dir = Path(a.run) if a.run else rl.dir

    def dump(tables: dict, prefix: str):
        for name, t in tables.items():
            t.to_csv(rl.dir / f"{prefix}{name}.csv", index=False)
            rl.log.info("%s%s\n%s", prefix, name, t.to_markdown(index=False, floatfmt=".4f"))

    for step in steps:
        t0 = time.time()
        rl.event("step_start", step=step)
        if step == "check_cls":
            check_cls(rl)
        elif step == "p5":
            dump(run_p5(rl, a.op_sub), "")
        elif step in ("wod_a", "wod_b"):
            run_wod(rl, step, heads)
            dump(wod_readout(rl.dir, step), f"{TAG[step]}_")
        elif step == "readout":
            for p in ("wod_a", "wod_b"):
                if list(run_dir.glob(f"{TAG[p]}_preds_dir*.npz")):
                    dump(wod_readout(run_dir, p), f"{TAG[p]}_")
        rl.event("step_end", step=step, seconds=time.time() - t0)
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
