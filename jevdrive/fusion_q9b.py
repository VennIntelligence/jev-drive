"""Fusion diagnostics Q9b: can a spatial-token readout rescue Qwen's pedestrians on P5?
Pre-registration: todos/2026-09-25-fusion-diagnostics.md (Q9b, and the [Q9b] lines of the deviation log).

  profile   the grid extraction on the first obs rows in two numerical configurations (the stored P5 one, eager
            batch 2, and qwenvid_train_t4's, compile batch 8): ms/frame, peak VRAM, and the pooled taps against the
            stored P5 features on the same rows (the equivalence check)
  extract   `L18_grid` (3 cameras x 4 x 4 tokens x 2560) on the 9 919 obs rows, resumable chunks under
            processed/carla_p5/features_grid/
  fit       prior = M-C's (openpilot `ridge_late`, role == train rows); Delta = waymo_ladder.AttnPool over the 48
            grid tokens; arms: paired-difference loss (seeds 0-2; seed 0 is the reading), hard-example reweighting,
            and a linear pooled-`L18_last` paired arm under the same loss (comparison only); p5_exam.exam unchanged,
            reactivity_mc.criteria unchanged
"""
import time

import numpy as np
import pandas as pd
import torch

from . import p5_exam as E, p5_pairs as P, planner
from .common import get_logger

log = get_logger(__name__)
GRID_HW = (4, 4)
N_TOK = 3 * GRID_HW[0] * GRID_HW[1]
CHUNK = 1000
TAPS = ("L18_mean", "L18_last")
SEEDS = (0, 1, 2)
FIXED_EPOCHS = 0                           # post-hoc sensitivity only: train this many epochs, no early stopping
HARD_SEED = 0                              # overnight queue [SEEDS]: the hard-example arm's _train seed


def grid_root(*p):
    d = P.processed("features_grid", *p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def obs_index() -> pd.DataFrame:
    t = pd.read_parquet(P.processed() / "index.parquet")
    return t[t.role == "obs"].reset_index(drop=True)


# ================================================================ extraction

def _compare(dst, names) -> dict:
    t = pd.DataFrame({"frame_name": names})
    ref = P.load_features(t, TAPS)
    out = {}
    for a in TAPS:
        x, y = np.load(dst / f"{a}.npy").astype(np.float32), ref[a]
        d = x - y
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        out[a] = {"rows": len(x), "identical_rows": int((d == 0).all(1).sum()), "max_abs": float(np.abs(d).max()),
                  "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(y)), "min_cos": float(cos.min())}
    return out


def profile(rl, n: int, configs, workers: int, ref_chunk: str = ""):
    """`ref_chunk`: take the rows in a stored P5 chunk's own order, so every batch pairs the same clips as the
    stored extraction did (the bit-for-bit test of the recipe); otherwise the first obs rows."""
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    o = obs_index()
    if ref_chunk:
        t = pd.read_parquet(P.processed() / "index.parquet").set_index("frame_name")
        names = pd.read_parquet(P.processed("features", ref_chunk) / "index.parquet").frame_name
        o = t.loc[names].reset_index()
    items, names = o.files.map(list).tolist()[:n], o.frame_name.to_numpy()[:n]
    rows = []
    for comp, b in configs:
        fx = qv.make_fx(compile=comp, grid_hw=GRID_HW)
        tag = f"b{b}-{'compile' if comp else 'eager'}" + (f"-{ref_chunk}" if ref_chunk else "")
        dst = grid_root("profile", rl.dir.name, tag)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        st = F.extract(fx, items, b, workers, dst, rl, f"profile/{tag}", dataset=p4.ClipFiles)
        eq = _compare(dst, names)
        r = {"config": tag, "rows": n, "wall_s": time.perf_counter() - t0, "ms_per_frame": st["ms_per_frame"],
             "peak_vram_gb": st["peak_vram_gb"], "peak_reserved_gb": st["peak_vram_reserved_gb"],
             **{f"{a}_{k}": v for a, e in eq.items() for k, v in e.items() if k != "rows"}}
        rows.append(r)
        rl.event("profile", **r)
        rl.log.info("%s: %.1f ms/frame, peak %.1f GB | %s", tag, r["ms_per_frame"], r["peak_vram_gb"], eq)
        del fx
        F.free_gpu()
    t = pd.DataFrame(rows)
    t.to_csv(rl.dir / "profile.csv", index=False)
    rl.log.info("profile\n%s", t.to_markdown(index=False, floatfmt=".4g"))


def _file_keys(files: pd.Series) -> pd.Series:
    from .p5_qwen import _file_keys as fk
    return fk(files)


def reuse(rl, src: str = "carla_p5") -> pd.DataFrame:
    """P5 v1: obs rows whose 12 JPEGs are the same files (directories resolved) as a v0 obs row take v0's grid row
    as is (features_grid/r000); the rest is left to `extract`. The v0 grid is the same recipe (compile, batch 4)."""
    import json
    import os
    cur = os.environ.get("P5_SET", "carla_p5")
    o = obs_index()
    dst = grid_root("r000")
    if cur == src:
        return o.iloc[:0]
    os.environ["P5_SET"] = src
    try:
        v0 = obs_index()
        names0, parts0 = [], []
        for d in sorted(grid_root().glob("c*")):
            names0.append(pd.read_parquet(d / "index.parquet").frame_name)
            parts0.append((d, len(names0[-1])))
    finally:
        os.environ["P5_SET"] = cur
    k0 = pd.Series(v0.frame_name.to_numpy(), index=_file_keys(v0.files).to_numpy())
    k0 = k0[~k0.index.duplicated()]
    src_name = pd.Series(_file_keys(o.files).map(k0).to_numpy(), index=o.frame_name.to_numpy()).dropna()
    rl.log.info("%s: %d obs rows, %d with the same 12 files as a %s obs row", cur, len(o), len(src_name), src)
    if (dst / "meta.json").exists() and pd.read_parquet(dst / "index.parquet").frame_name.tolist() == src_name.index.tolist():
        return src_name
    pos0 = pd.Series(np.arange(sum(map(len, names0))), index=pd.concat(names0).to_numpy())
    G0 = np.concatenate([np.load(d / "L18_grid.npy", mmap_mode="r") for d, _ in parts0])
    rows = pos0.reindex(src_name.to_numpy()).to_numpy()
    assert not np.isnan(rows).any()
    np.save(dst / "L18_grid.npy", G0[rows.astype(int)])
    pd.DataFrame({"frame_name": src_name.index, "src_frame": src_name.to_numpy()}).to_parquet(dst / "index.parquet", index=False)
    (dst / "meta.json").write_text(json.dumps({"reused_rows": len(src_name), "from": src}))
    rl.event("reuse", rows=len(src_name), of=len(o), src=src)
    return src_name


def reuse_check(rl, n: int, batch: int, compile: bool, workers: int):
    """Recompute the grid of `n` reused rows (spread over r000) and compare with the copied v0 rows."""
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    idx = pd.read_parquet(grid_root("r000") / "index.parquet")
    sel = np.linspace(0, len(idx) - 1, n).astype(int)
    t = obs_index().set_index("frame_name")
    names = idx.frame_name.to_numpy()[sel]
    fx = qv.make_fx(compile=compile, grid_hw=GRID_HW)
    dst = grid_root("reuse_check", rl.dir.name)
    st = F.extract(fx, t.loc[names].files.map(list).tolist(), batch, workers, dst, rl, "reuse_check", dataset=p4.ClipFiles)
    x = np.load(dst / "L18_grid.npy").astype(np.float32)
    y = np.load(grid_root("r000") / "L18_grid.npy", mmap_mode="r")[sel].astype(np.float32)
    cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    r = {"rows": n, "identical_rows": int(((x - y) == 0).all(1).sum()), "max_abs": float(np.abs(x - y).max()),
         "rel_l2": float(np.linalg.norm(x - y) / np.linalg.norm(y)), "min_cos": float(cos.min()),
         "pooled_vs_stored": _compare(dst, names), "ms_per_frame": st["ms_per_frame"]}
    rl.event("reuse_check", **r)
    rl.log.info("reuse check (recomputed vs copied v0 grid rows): %s", r)


def extract(rl, batch: int, compile: bool, workers: int):
    from . import features as F, p4_carla as p4, waymo_qwenvid as qv
    o = obs_index()
    reused = reuse(rl)
    o = o[~o.frame_name.isin(set(reused.index))].reset_index(drop=True)
    chunks = [o.iloc[i:i + CHUNK] for i in range(0, len(o), CHUNK)]
    left = [(i, c) for i, c in enumerate(chunks) if not (grid_root(f"c{i:03d}") / "meta.json").exists()]
    rl.log.info("%d obs rows to extract in %d chunks, %d to do (batch %d, compile %s, workers %d)", len(o), len(chunks),
                len(left), batch, compile, workers)
    if not left:
        return
    fx = qv.make_fx(compile=compile, grid_hw=GRID_HW)
    t0, done = time.perf_counter(), 0
    import os
    for i, c in left:
        dst = grid_root(f"c{i:03d}")
        if (dst / "meta.json").exists():
            continue
        try:                                   # several processes (one per card) claim chunks through a lock file
            os.close(os.open(grid_root() / f"c{i:03d}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        st = F.extract(fx, c.files.map(list).tolist(), batch, workers, dst, rl, f"grid/c{i:03d}", dataset=p4.ClipFiles)
        c[["frame_name"]].to_parquet(dst / "index.parquet", index=False)
        eq = _compare(dst, c.frame_name.to_numpy())
        (dst / "meta.json").write_text(pd.Series({"recipe": "P3(d'') qwenvid + L18_grid 4x4", "compile": compile,
                                                  **st, "equiv": eq}).to_json())
        done += len(c)
        el = time.perf_counter() - t0
        rl.log.info("chunk %d: %d clips, %.1f ms/frame; L18_mean rel %.2e min cos %.6f; ETA %.0f min", i, len(c),
                    st["ms_per_frame"], eq["L18_mean"]["rel_l2"], eq["L18_mean"]["min_cos"],
                    el / done * (sum(len(x) for _, x in left) - done) / 60)
        rl.event("chunk_done", chunk=i, n=len(c), ms_per_frame=st["ms_per_frame"], equiv=eq)


def load_grid(names) -> np.ndarray:
    """(len(names), N_TOK * 2560) float16, aligned to `names`."""
    parts, idx = [], []
    for d in sorted(grid_root().glob("[cr][0-9]*")):
        if (d / "meta.json").exists():
            idx.append(pd.read_parquet(d / "index.parquet").frame_name)
            parts.append(np.load(d / "L18_grid.npy", mmap_mode="r"))
    pos = pd.Series(np.arange(sum(map(len, idx))), index=pd.concat(idx).to_numpy())
    at = pd.Series(names).map(pos)
    assert at.notna().all(), f"{int(at.isna().sum())} rows without a grid"
    return np.concatenate(parts)[at.astype(int).to_numpy()]


# ================================================================ heads

def prior_fold(f, fold, t, F, Ego, Xop):
    """reactivity_mc.fit_fold's prior, line for line: ridge ego + ridge_late residual, role == train & fold != f."""
    from types import SimpleNamespace
    from . import waymo_stage_a as sa
    n = len(t)
    role, seq = t.role.to_numpy(), t.base_id.to_numpy()
    tr = np.flatnonzero((role == "train") & (fold != f))
    ev = np.flatnonzero((role == "obs") & (fold == f))
    sp = SimpleNamespace(train=tr, val=ev, seq=seq)
    fut = F.reshape(n, 20, 2).cpu().numpy()
    Xe = planner.standardize(Ego, tr)
    _, st_e, We = sa.ridge_cv(Xe, F, sp, fut)
    base = planner.linear_apply(We, Xe, np.arange(n))[0]
    Xi = planner.standardize(Xop, tr)
    R0 = F - base
    _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(n, 20, 2).cpu().numpy())
    prior = base + planner.linear_apply(Wp, Xi, np.arange(n))[0]
    s_ego = (base - F).reshape(n, 20, 2).norm(dim=-1).mean(1)
    return prior, s_ego, {"lam_ego": st_e["lam"], "lam_prior": st_p["lam"]}


def _train(make, loss_fn, items: np.ndarray, groups: np.ndarray, seed: int):
    """waymo_ladder._train's recipe on an arbitrary loss: early stop on a 20 % route-grouped inner split of the
    training items, then refit on all of them for the chosen number of epochs."""
    from sklearn.model_selection import GroupShuffleSplit
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(items, groups=groups))

    def run(rows, n_ep, sel):
        torch.manual_seed(seed)
        net = make().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=planner.MLP_LR, weight_decay=planner.MLP_WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
        r = torch.as_tensor(rows, device="cuda")
        best, curve = (np.inf, n_ep), []
        for ep in range(n_ep):
            net.train()
            for bb in r[torch.randperm(len(r), device="cuda")].split(planner.MLP_BS):
                opt.zero_grad(set_to_none=True)
                loss_fn(net, bb).backward()
                opt.step()
            sched.step()
            if sel is not None:
                net.eval()
                with torch.inference_mode():
                    s = sum(float(loss_fn(net, c)) * len(c) for c in torch.as_tensor(sel, device="cuda").split(2048))
                curve.append(s / len(sel))
                best = min(best, (curve[-1], ep + 1))
        return net, best, curve

    if FIXED_EPOCHS:
        net, _, _ = run(items, FIXED_EPOCHS, None)
        net.eval()
        return net, {"epochs": FIXED_EPOCHS, "sel_loss": float("nan"), "curve": []}
    _, (sel_loss, ep), curve = run(items[a], planner.MLP_EPOCHS, items[b])
    net, _, _ = run(items, ep, None)
    net.eval()
    return net, {"epochs": ep, "sel_loss": sel_loss, "curve": curve}


def _v2(y):                                   # speed at 2 s from flattened (., 40) trajectories, as reactivity_mc
    return torch.linalg.norm(y[:, 14:16] - y[:, 12:14], dim=1).cpu().numpy() / 0.25


def fit_fold(f, fold, t, F, prior, s_ego, G, gpos, Q, pr_ip, pr_im, pr_group, pr_fam, rl, tag) -> dict:
    from .reactivity_mc import CUTIN, LAMS, _inner_splits, _solve_pair
    from .waymo_ladder import AttnPool, _chan_stats
    keep = fold[pr_ip] != f
    ip, im, grp, fam = pr_ip[keep], pr_im[keep], pr_group[keep], pr_fam[keep]
    frames = np.unique(np.r_[ip, im])                              # training-pair frames (global rows)
    mu, sd = _chan_stats(G, gpos[frames])
    z = lambda rows: (G[gpos[rows]].float() - mu) / sd              # noqa: E731
    Rp = (F[ip] - F[im]) - (prior[ip] - prior[im])
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    out = {"prior": prior}

    def apply(net, rows):
        with torch.inference_mode():
            return torch.cat([net(z(c)) for c in np.array_split(rows, max(1, len(rows) // 1024))])

    def slopes(fit, arm):
        dt = _v2(F[ip]) - _v2(F[im])
        dm, dp = _v2(fit[ip]) - _v2(fit[im]), _v2(prior[ip]) - _v2(prior[im])
        for g, msk in (("pedestrian", np.isin(fam, E.PED_FAMILIES)), ("cut-in", np.isin(fam, CUTIN))):
            big = msk & (np.abs(dt) > 0.5)
            ok = big.sum() > 2
            rl.event("q9b_insample", fold=f, arm=arm, model=tag, group=g, n=int(big.sum()),
                     slope_fit=float(np.polyfit(dt[big], dm[big], 1)[0]) if ok else None,
                     slope_prior=float(np.polyfit(dt[big], dp[big], 1)[0]) if ok else None)

    # paired-difference arm on the grid, several seeds
    pairs = np.arange(len(ip))
    lossp = lambda net, b: (net(z(ip[b.cpu().numpy()])) - net(z(im[b.cpu().numpy()])) - Rp[b]).pow(2).mean()  # noqa: E731
    for seed in SEEDS:
        net, st = _train(lambda: AttnPool(G.shape[-1], 40), lossp, pairs, grp, seed)
        d = apply(net, obs_rows)
        dm = apply(net, frames).mean(0)
        full = prior.clone()
        full[obs_rows] = prior[obs_rows] + d - dm
        arm = f"Q9b pair grid s{seed}"
        tr_fit = prior.clone()
        tr_fit[frames] = prior[frames] + apply(net, frames) - dm
        slopes(tr_fit, arm)
        out[arm] = full
        rl.event("q9b_fold", fold=f, arm=arm, model=tag, n_pairs=len(ip), **{k: v for k, v in st.items() if k != "curve"},
                 curve=st["curve"])
        rl.log.info("fold %d %s %s: %d epochs, inner paired MSE %.4f", f, tag, arm, st["epochs"], st["sel_loss"])
        del net
    # hard-example reweighting control: the same head on single frames, residual y - p, w = s_ego / mean
    Y = F - prior
    w = s_ego / s_ego[frames].mean()
    lossh = lambda net, b: (w[b] * (net(z(b.cpu().numpy())) - Y[b]).pow(2).mean(1)).mean()  # noqa: E731
    net, st = _train(lambda: AttnPool(G.shape[-1], 40), lossh, frames, t.base_id.to_numpy()[frames].astype(str), HARD_SEED)
    full = prior.clone()
    full[obs_rows] = prior[obs_rows] + apply(net, obs_rows)
    tr_fit = prior.clone()
    tr_fit[frames] = prior[frames] + apply(net, frames)
    slopes(tr_fit, "Q9b hard grid")
    out["Q9b hard grid"] = full
    rl.event("q9b_fold", fold=f, arm="Q9b hard grid", model=tag, n_frames=len(frames),
             **{k: v for k, v in st.items() if k != "curve"}, curve=st["curve"])
    rl.log.info("fold %d %s hard: %d epochs, inner weighted MSE %.4f", f, tag, st["epochs"], st["sel_loss"])
    del net
    # comparison only: the linear paired arm on pooled L18_last under the same loss (no role == train mu term)
    Z = planner.standardize(Q, frames) / np.sqrt(Q.shape[1])
    zbar = Z[frames].mean(0)
    D = Z[ip] - Z[im]
    Zc = torch.zeros(1, Z.shape[1], device=Z.device)
    score = np.zeros(len(LAMS))
    for a, b in _inner_splits(grp):
        Ws = _solve_pair(D[a], Rp[a], Zc, 0.0, LAMS)
        score += [float(((D[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
    best = int(np.argmin(score))
    W = _solve_pair(D, Rp, Zc, 0.0, [LAMS[best]])[0]
    out["Q9b pair pooled-linear"] = prior + (Z - zbar) @ W
    slopes(out["Q9b pair pooled-linear"], "Q9b pair pooled-linear")
    rl.event("q9b_fold", fold=f, arm="Q9b pair pooled-linear", model=tag, lam=float(LAMS[best]),
             lam_edge=best in (0, len(LAMS) - 1))
    return out


def fit(rl, models=("cinque", "lebowski"), op_sub: str = "op_streams", out_name: str = "q9b"):
    from . import p5_openpilot
    from .reactivity_mc import criteria
    from .fusion_diag import RESULTS
    t, past, fut, obs, null, pairs = E.load()
    n = len(t)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    G = torch.as_tensor(load_grid(t.frame_name.to_numpy()[obs_rows]), device="cuda").view(len(obs_rows), N_TOK, -1)
    gpos = np.full(n, -1)
    gpos[obs_rows] = np.arange(len(obs_rows))
    Q = torch.as_tensor(P.load_features(t, ("L18_last",))["L18_last"], device="cuda")
    op = p5_openpilot.load(t, models, sub=op_sub)
    fold = E.folds(t, pairs)
    F = torch.as_tensor(fut.reshape(n, -1), device="cuda")
    Ego = torch.as_tensor(E.ego_input(t, past), device="cuda")
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    pr_fam = np.r_[obs.family.to_numpy(), np.full(len(null), "null")].astype(str)
    preds = {}
    for m in models:
        Xop = torch.as_tensor(op[f"op-{m} temporal"], device="cuda")
        for f in range(E.K_FOLDS):
            ev = obs_rows[fold[obs_rows] == f]
            if not len(ev):
                continue
            prior, s_ego, st = prior_fold(f, fold, t, F, Ego, Xop)
            rl.log.info("fold %d %s: prior %s", f, m, st)
            o = fit_fold(f, fold, t, F, prior, s_ego, G, gpos, Q, pr_ip, pr_im, pr_group, pr_fam, rl, m)
            for arm, v in o.items():
                preds.setdefault(f"{arm} [{m}]", np.full((n, 20, 2), np.nan, np.float32))[ev] = \
                    v[ev].reshape(-1, 20, 2).cpu().numpy()
            torch.cuda.empty_cache()
        del Xop
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(preds))
    crit = pd.concat([criteria(res, [k for k in preds if k.endswith(f"[{m}]")], f"prior [{m}]") for m in models])
    ins = pd.read_json(rl.dir / "events.jsonl", lines=True)
    ins = ins[ins.kind == "q9b_insample"].dropna(axis=1, how="all")
    d = rl.dir
    fl = res["flips"]
    fl.to_csv(d / "flip_rates.csv", index=False)
    crit.to_csv(d / "criteria.csv", index=False)
    res["obs"].to_parquet(d / "obs_scored.parquet", index=False)
    nn.to_parquet(d / "null_scored.parquet", index=False)
    np.savez_compressed(d / "preds_obs.npz", rows=obs_rows, **{k: v[obs_rows] for k, v in preds.items()})
    out = RESULTS / (out_name + (f"-fixed{FIXED_EPOCHS}" if FIXED_EPOCHS else ""))
    out.mkdir(parents=True, exist_ok=True)
    crit.to_csv(out / "q9b_criteria.csv", index=False, float_format="%.4g")
    fl[fl.scope != "pooled"][["examinee", "scope", "n_reactive", "flip_rate", "flip_lo", "flip_hi", "tau_model"]].to_csv(
        out / "q9b_flips_by_family.csv", index=False, float_format="%.4g")
    fl[fl.scope == "pooled"].to_csv(out / "q9b_flips_pooled.csv", index=False, float_format="%.4g")
    ins.to_csv(out / "q9b_insample_slopes.csv", index=False, float_format="%.4g")
    rl.log.info("in-sample slopes (median over folds)\n%s", ins.groupby(["model", "arm", "group"])[["slope_fit", "slope_prior"]]
                .median().reset_index().to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("pooled flips\n%s", fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("criteria\n%s", crit.to_markdown(index=False, floatfmt=".3f"))
    return res, crit


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("profile", "extract", "reuse-check", "fit"))
    ap.add_argument("--op-sub", default="op_streams", help="fit: openpilot stream set (v1: op_streams_vis)")
    ap.add_argument("--out", default="q9b", help="fit: results path under research/results/fusion-diagnostics")
    ap.add_argument("--tag", default="", help="run-dir tag suffix (e.g. v1-pdm)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--configs", default="eager:2,compile:8", help="profile: comma list of eager|compile:batch")
    ap.add_argument("--ref-chunk", default="", help="profile: stored P5 chunk whose row order to reuse (e.g. c010)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vram-gb", type=float, default=12.0)
    ap.add_argument("--fixed-epochs", type=int, default=0, help="fit: post-hoc sensitivity, no early stopping")
    ap.add_argument("--hard-seed", type=int, default=None, help="fit: seed of the hard-example arm ([SEEDS]); given -> own run dir")
    a = ap.parse_args()
    global FIXED_EPOCHS, HARD_SEED
    FIXED_EPOCHS = a.fixed_epochs
    HARD_SEED = a.hard_seed or 0
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / total))
    rl = RunLog("fusion_diag", f"q9b-{a.step}" + (f"-{a.tag}" if a.tag else "") +
                (f"-fixed{a.fixed_epochs}" if a.fixed_epochs else "") + (f"-hardseed{a.hard_seed}" if a.hard_seed is not None else ""))
    rl.event("start", args=vars(a))
    if a.step == "profile":
        profile(rl, a.n, [(c == "compile", int(b)) for c, b in (x.split(":") for x in a.configs.split(","))], a.workers,
                a.ref_chunk)
    elif a.step == "extract":
        extract(rl, a.batch, a.compile, a.workers)
    elif a.step == "reuse-check":
        reuse_check(rl, a.n, a.batch, a.compile, a.workers)
    else:
        fit(rl, op_sub=a.op_sub, out_name=a.out)
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
