"""WL-2 representation dry run on WL-1 data (fc65452:todos/2026-09-29-wl-dryrun.md): the ego lateral channel and the spatial-token arm.

Descriptive only: WL-1's eval fork points were already seen, so nothing here is a judged WL-2 result.

Arms (all with WL-1's recipe: W.CFG, 10% inner-val routes, seed = init + inner-val draw):
  B   WL-1 `main` universe, z + 5-dim ego state s = [e_y, e_psi, v, a, omega] (e_y / e_psi against the run's route.json polyline)
  Bh  B without windows whose future touches shift_R / op_slow (C2(b))
  Bi  B on the wl_gen index rows only (no W logged frames): the same-data control of T
  T   spatial tokens: [openpilot temporal 512 | 24 V-JEPA grid tokens PCA-128 | s], a token Transformer (26 tokens / step)

  prep-ego  CPU. e_y / e_psi of every W-meta row and every wl_gen index row -> processed/wl_gen/dryrun/ey_{wm,idx}.npy
  prep-tok  GPU small. shared PCA 1024 -> 128 over the 24 tokens of processed/wl_gen/tok.npy -> dryrun/tokpca.npy
  train     one arm x one seed -> runs/wl/dryrun/<arm>/seed<s>/<stamp>/{preds.npz, forks.parquet, curve.json, model.pt}
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.world_model.archive import nq4_w as W
from experiments.world_model.lib import wl as WL
from experiments.world_model.archive import wl_model as M
from jevdrive.common import data_dir, get_logger, n_cpus
from experiments.world_model.lib.wl_traj import ACTIONS as WL_ACTIONS

log = get_logger(__name__)
ARMS = ("B", "Bh", "Bi", "T")
S_DIM, N_TOK, D_TOK = 5, 24, 128
T_STEPS = 4000                                       # T: 26x longer sequences; half of WL-1's 8 000 steps (batch 128), see the todo
CAP_GB = 15.0                                        # VRAM budget of this lane (GPU 5 is shared with the Cosmos run)


def ddir(*p) -> Path:
    d = data_dir() / "processed" / "wl_gen" / "dryrun"
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


# ================================================================ ego channel

def _ego_group(args):
    """e_y (left positive, m) and e_psi (rad) of the ego pose at `frames` against the run's route polyline."""
    from scipy.spatial import cKDTree
    adir, frames = args
    a = Path(adir)
    R = np.array([[p["x"], p["y"]] for p in json.loads((a / "route.json").read_text())], float)
    R = R[np.r_[True, np.hypot(*np.diff(R, axis=0).T) > 1e-3]]
    p = pd.read_json(a / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame").reindex(frames)
    X, yaw = p[["x", "y"]].to_numpy(float), np.radians(p.yaw.to_numpy(float))
    ok = ~np.isnan(X).any(1)
    out = np.full((len(frames), 2), np.nan, np.float32)
    if len(R) < 2 or not ok.any():
        return out
    i = cKDTree(R).query(X[ok])[1]
    Xo = X[ok]

    def seg(j):
        j = np.clip(j, 0, len(R) - 2)
        A, d = R[j], R[j + 1] - R[j]
        t = np.clip(((Xo - A) * d).sum(1) / np.maximum((d ** 2).sum(1), 1e-9), 0, 1)
        P = A + t[:, None] * d
        return P, d, np.hypot(*(Xo - P).T)
    (P0, d0, e0), (P1, d1, e1) = seg(i - 1), seg(i)
    u = (e1 <= e0)[:, None]
    P, d = np.where(u, P1, P0), np.where(u, d1, d0)
    th = np.arctan2(d[:, 1], d[:, 0])
    left = np.c_[np.sin(th), -np.cos(th)]                  # CARLA: y right, yaw clockwise -> left of the heading is (sin, -cos)
    out[ok, 0] = ((Xo - P) * left).sum(1)
    out[ok, 1] = (yaw[ok] - th + np.pi) % (2 * np.pi) - np.pi
    return out


def prep_ego(workers: int = 24) -> dict:
    from multiprocessing import Pool
    wm = pd.read_parquet(W.wdir("meta.parquet"), columns=["adir", "frame"])
    ix = pd.read_parquet(WL_INDEX, columns=["adir", "frame"])
    tab = pd.concat([wm.assign(src="wm", pos=np.arange(len(wm))), ix.assign(src="idx", pos=np.arange(len(ix)))], ignore_index=True)
    tab["frame"] = tab.frame.astype(int)
    groups = [(a, g.frame.to_numpy(), g.src.to_numpy(), g.pos.to_numpy()) for a, g in tab.groupby("adir", sort=False)]
    res = {"wm": np.full((len(wm), 2), np.nan, np.float32), "idx": np.full((len(ix), 2), np.nan, np.float32)}
    with Pool(workers) as pool:
        for (a, fr, src, pos), o in zip(groups, pool.imap(_ego_group, [(g[0], g[1]) for g in groups], chunksize=8)):
            for k in ("wm", "idx"):
                s = src == k
                res[k][pos[s]] = o[s]
    stats = {}
    for k, v in res.items():
        stats[k] = {"rows": len(v), "nan": int(np.isnan(v[:, 0]).sum()), "ey_abs_median": float(np.nanmedian(np.abs(v[:, 0]))),
                    "ey_abs_p95": float(np.nanpercentile(np.abs(v[:, 0]), 95)), "epsi_abs_median": float(np.nanmedian(np.abs(v[:, 1])))}
        np.save(ddir(f"ey_{k}.npy"), v)
    (ddir("ego_stats.json")).write_text(json.dumps(stats, indent=1))
    return stats


# ================================================================ spatial tokens: shared PCA

def prep_tok(n_fit: int = 30000, chunk: int = 4096) -> dict:
    import torch
    tok = np.load(ddir("..", "tok.npy"), mmap_mode="r")
    N = len(tok)
    rows = np.sort(np.random.RandomState(0).choice(N, n_fit, replace=False))
    s1 = torch.zeros(1024, dtype=torch.float64, device="cuda")
    s2 = torch.zeros(1024, 1024, dtype=torch.float64, device="cuda")
    n = 0
    for lo in range(0, n_fit, 1000):
        x = torch.as_tensor(np.asarray(tok[rows[lo: lo + 1000]]), device="cuda").float().reshape(-1, 1024)
        s1 += x.sum(0).double()
        s2 += (x.T @ x).double()
        n += len(x)
    mu = s1 / n
    e, V = torch.linalg.eigh(s2 / n - torch.outer(mu, mu))
    e, V = e.flip(0), V.flip(1)
    P = V[:, :D_TOK].float()
    var = float(e[:D_TOK].sum() / e.sum())
    out = np.lib.format.open_memmap(ddir("tokpca.npy"), mode="w+", dtype=np.float16, shape=(N, N_TOK * D_TOK))
    mu32 = mu.float()
    for lo in range(0, N, chunk):
        x = torch.as_tensor(np.asarray(tok[lo: lo + chunk]), device="cuda").float()
        out[lo: lo + len(x)] = (((x - mu32) @ P).reshape(len(x), -1)).half().cpu().numpy()
    out.flush()
    info = {"rows": N, "fit_tokens": n, "dim": D_TOK, "variance_retained": var}
    ddir("tokpca.json").write_text(json.dumps(info, indent=1))
    return info


# ================================================================ universe and data

WL_INDEX = data_dir() / "processed" / "wl_gen" / "index.parquet"


def ego_cols(m: pd.DataFrame, ey: np.ndarray) -> np.ndarray:
    return np.c_[np.nan_to_num(ey, nan=0.0), m.v.to_numpy(), m.a_prev.to_numpy(), m.w_prev.to_numpy()].astype(np.float32)


def universe_ext(arm: str):
    """(meta, z + s) of the arm; T replaces the V-JEPA mean by the PCA'd grid tokens (index rows only)."""
    m, z = M.universe("main")
    ev = set(json.loads(WL.rundir("split.json").read_text())["eval"])
    wm = pd.read_parquet(W.wdir("meta.parquet"), columns=["base_id"])
    keep = ~wm.base_id.astype(str).isin(ev).to_numpy()
    ok = np.load(M.pdir("z_ok.npy"))
    ey = np.concatenate([np.load(ddir("ey_wm.npy"))[keep], np.load(ddir("ey_idx.npy"))[ok]])
    n1 = int(keep.sum())
    S = ego_cols(m, ey)
    if arm in ("Bi", "T"):
        m, z, S = m.iloc[n1:].reset_index(drop=True), z[n1:], S[n1:]
    if arm == "T":
        z = np.concatenate([z[:, :W.D_OP], np.load(ddir("tokpca.npy"), mmap_mode="r")[np.flatnonzero(ok)]], 1)
    return m, np.concatenate([z, S.astype(np.float16)], 1)


class Data(M.Data):
    """M.Data with the standardisation moments accumulated in chunks (the fp32 copy of all training rows does not fit the VRAM budget)."""

    def __init__(self, meta, z, train_rows, dev="cuda", chunk=16384):
        import torch
        tr = np.asarray(train_rows)
        s1 = torch.zeros(z.shape[1], dtype=torch.float64, device=dev)
        s2 = torch.zeros_like(s1)
        for lo in range(0, len(tr), chunk):
            x = torch.as_tensor(z[tr[lo: lo + chunk]], device=dev).double()
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
        n = len(tr)
        mu = s1 / n
        self.mu = mu.float()
        self.sd = ((s2 / n - mu * mu).clamp_min(0) * n / (n - 1)).sqrt().float().clamp_min(1e-4)
        self.Z = torch.empty((len(z), z.shape[1]), device=dev, dtype=torch.bfloat16)
        for lo in range(0, len(z), 8192):
            self.Z[lo: lo + 8192] = ((torch.as_tensor(z[lo: lo + 8192], device=dev).float() - self.mu) / self.sd).bfloat16()
        self.hact = torch.as_tensor(np.c_[meta[["a_prev", "w_prev"]].to_numpy(np.float32) / W.ACT_SCALE,
                                          meta.v.to_numpy(np.float32)[:, None] / W.V_SCALE], device=dev)
        self.fact = torch.as_tensor(meta[["a_next", "w_next"]].to_numpy(np.float32) / W.ACT_SCALE, device=dev)
        self.dev = dev
        self.off = torch.arange(W.WIN, device=dev)
        self.S = torch.as_tensor(meta.src.to_numpy(np.int64), device=dev)


def block_mse3(pred, tgt):
    """Equal weight to the openpilot block, the V-JEPA (or token) block and the s block."""
    e = (pred - tgt) ** 2
    return (e[..., :W.D_OP].mean() + e[..., W.D_OP: -S_DIM].mean() + e[..., -S_DIM:].mean()) / 3


# ================================================================ token predictor

def build_tok(cfg=W.CFG):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    d, T = cfg["d"], N_TOK + 2

    class TokM(nn.Module):
        def __init__(s):
            super().__init__()
            s.op_in, s.tok_in, s.s_in = nn.Linear(W.D_OP, d), nn.Linear(D_TOK, d), nn.Linear(S_DIM, d)
            s.hin = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            s.qin = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
            s.src = nn.Embedding(2, d)
            nn.init.normal_(s.src.weight, std=0.02)
            s.kind, s.qkind = nn.Parameter(torch.randn(T, d) * 0.02), nn.Parameter(torch.randn(T, d) * 0.02)
            s.pos = nn.Parameter(torch.randn(W.WIN, d) * 0.02)
            s.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, cfg["heads"], cfg["ff"], cfg["drop"], activation="gelu",
                                                                 batch_first=True, norm_first=True) for _ in range(cfg["layers"])])
            s.norm = nn.LayerNorm(d)
            s.op_out, s.tok_out, s.s_out = nn.Linear(d, W.D_OP), nn.Linear(d, D_TOK), nn.Linear(d, S_DIM)
            for o in (s.op_out, s.tok_out, s.s_out):
                nn.init.zeros_(o.weight)
                nn.init.zeros_(o.bias)

        def embed(s, z):                                        # (B, T, dz) -> (B, T, 26, d)
            op, tk, st = z[..., :W.D_OP], z[..., W.D_OP: W.D_OP + N_TOK * D_TOK], z[..., -S_DIM:]
            return torch.cat([s.op_in(op)[:, :, None], s.tok_in(tk.unflatten(-1, (N_TOK, D_TOK))), s.s_in(st)[:, :, None]], 2)

        def forward(s, zh, hact, fact, sh, sf):
            B = zh.shape[0]
            h = s.embed(zh) + s.kind + (s.hin(hact) + s.src(sh))[:, :, None]
            q = s.qkind[None, None].expand(B, W.FUT, -1, -1) + (s.qin(fact) + s.src(sf))[:, :, None]
            x = (torch.cat([h, q], 1) + s.pos[None, :, None]).flatten(1, 2)
            for layer in s.layers:
                x = checkpoint(layer, x, use_reentrant=False) if (s.training and torch.is_grad_enabled()) else layer(x)
            x = s.norm(x).unflatten(1, (W.WIN, T))[:, W.HIST:]
            out = torch.cat([s.op_out(x[:, :, 0]), s.tok_out(x[:, :, 1: 1 + N_TOK]).flatten(2), s.s_out(x[:, :, -1])], -1)
            return zh[:, -1:] + out
    return TokM()


# ================================================================ training (WL-1's loop, block loss over three blocks)

def _train(model, data, tr, va, seed, rl, tag, cfg, batch, compile_):
    import torch
    steps = cfg["steps"]
    fwd = torch.compile(model, mode="reduce-overhead") if compile_ else model
    g = torch.Generator(device="cuda").manual_seed(seed)
    tr, va = torch.as_tensor(tr, device="cuda"), torch.as_tensor(va, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    best, state, curve, t0 = np.inf, None, [], time.time()
    for step in range(1, steps + 1):
        model.train()
        zh, ha, fa, zf, sh, sf = data.batch(tr[torch.randint(len(tr), (batch,), device="cuda", generator=g)])
        with torch.autocast("cuda", torch.bfloat16):
            pred = fwd(zh, ha, fa, sh, sf)
        loss = block_mse3(pred.float(), zf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for lo in range(0, len(va), 512):
                    b = data.batch(va[lo: lo + 512])
                    tot += float(block_mse3(model(b[0], b[1], b[2], b[4], b[5]).float(), b[3])) * len(b[0])
                    n += len(b[0])
            vl = tot / max(n, 1)
            curve.append({"step": step, "train": float(loss), "val": vl, "wall_s": time.time() - t0})
            rl.scalar(f"{tag}/train_loss", float(loss), step)
            rl.scalar(f"{tag}/val_loss", vl, step)
            rl.info(f"{tag} step {step}: train {float(loss):.4f}, inner-val {vl:.4f}, {step / (time.time() - t0):.1f} steps/s, "
                    f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
            if vl < best:
                best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return curve


def _logreg(Z, rows, y, C=1.0, chunk=32768):
    """N2's probe objective (sum BCE + ||w||^2 / 2C, L-BFGS), chunked over rows."""
    import torch
    rows, y = torch.as_tensor(rows, device="cuda"), torch.as_tensor(y, device="cuda")
    w = torch.zeros(Z.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, tolerance_grad=1e-6, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tot = torch.zeros((), device="cuda")
        for lo in range(0, len(rows), chunk):
            x = Z[rows[lo: lo + chunk]].float()
            l = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y[lo: lo + chunk], reduction="sum")
            l.backward()
            tot += l.detach()
        r = (w * w).sum() / (2 * C)
        r.backward()
        return tot + r.detach()
    opt.step(closure)
    return w.detach(), b.detach()


def _ridges(Z, tr_rows, va_rows, targets: dict, chunk=32768):
    """W.fit_ridge for several targets sharing one Gram matrix (chunked, fp32 blocks accumulated in fp64)."""
    import torch
    dz = Z.shape[1]
    G = torch.zeros(dz, dz, dtype=torch.float64, device="cuda")
    xty = {k: torch.zeros(dz, dtype=torch.float64, device="cuda") for k in targets}
    mu = {k: float(np.mean(t[0])) for k, t in targets.items()}
    for lo in range(0, len(tr_rows), chunk):
        x = Z[torch.as_tensor(tr_rows[lo: lo + chunk], device="cuda")].float()
        G += (x.T @ x).double()
        for k, t in targets.items():
            xty[k] += (x.T @ torch.as_tensor(t[0][lo: lo + chunk] - mu[k], device="cuda").float()).double()
    e, V = torch.linalg.eigh(G)
    Xv = Z[torch.as_tensor(va_rows, device="cuda")].float()
    out = {}
    for k, t in targets.items():
        r = V.T @ xty[k]
        yv = torch.as_tensor(t[1], device="cuda").float()
        best = None
        for lam in W.RIDGE_LAMS:
            w = (V @ (r / (e + lam))).float()
            err = float(((Xv @ w + mu[k] - yv) ** 2).mean())
            if best is None or err < best[0]:
                best = (err, lam, w)
        out[k] = (best[2], torch.as_tensor(mu[k], device="cuda"), best[1])
    return out


def fit_probes(data, meta, tr_rows, va_rows) -> dict:
    """W.probes (+ the WL `v` ridge) with chunked fits: same objectives, bounded VRAM."""
    P = {}
    for p in W.PROBES_CLS:
        y = meta[p].to_numpy(np.float32)[tr_rows]
        ok = ~np.isnan(y)
        if ok.sum() < 100 or y[ok].min() == y[ok].max():
            continue
        P[p] = ("logit", *_logreg(data.Z, tr_rows[ok], y[ok]))
    tg = {k: (meta[c].to_numpy(np.float32)[tr_rows], meta[c].to_numpy(np.float32)[va_rows]) for k, c in (("d_front", "d_front"), ("v", "v"))}
    for k, (w, mu, lam) in _ridges(data.Z, tr_rows, va_rows, tg).items():
        P[k] = ("ridge", w, mu, lam if k == "d_front" else 0)
    return P


def train(arm: str, seed: int, rl, steps: int | None = None) -> dict:
    import torch
    torch.cuda.set_per_process_memory_fraction(CAP_GB * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(seed)
    m, z = universe_ext(arm)
    st = M.train_windows(m, "holdout" if arm == "Bh" else "main")
    routes = np.array(sorted(m.base_id.astype(str).unique()))
    inner = set(routes[np.random.RandomState(seed).rand(len(routes)) < 0.10])
    is_in = m.base_id.astype(str).isin(inner).to_numpy()
    anc = st + W.HIST - 1
    tr_st, va_st = st[~is_in[anc]], st[is_in[anc]]
    tr_rows = np.flatnonzero(~is_in & (m.split != "eval").to_numpy())
    va_rows = np.flatnonzero(is_in & (m.split != "eval").to_numpy())
    data = Data(m, z, tr_rows)
    del z
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    dz = data.Z.shape[1]
    model = (build_tok() if arm == "T" else M.build(dz)).cuda()
    rl.info(f"{arm} seed {seed}: {len(m)} rows (dz {dz}), {len(tr_st)} training windows ({int(m.src.sum())} intervention rows), {len(va_st)} inner-val windows")
    t0 = time.time()
    cfg = {**W.CFG, "steps": steps or (T_STEPS if arm == "T" else W.CFG["steps"])}
    curve = _train(model, data, tr_st, va_st, seed, rl, f"{arm}/seed{seed}", cfg, batch=128 if arm == "T" else cfg["batch"], compile_=arm != "T")
    rl.info(f"{arm}/seed{seed}: predictor fit {time.time() - t0:.0f} s")
    P = fit_probes(data, M.meta_for_probes(m), tr_rows, va_rows)
    fa = M.fork_anchors(m)
    starts = torch.as_tensor(fa.anchor.to_numpy() - (W.HIST - 1), device="cuda")
    n, A = len(fa), len(WL_ACTIONS)
    cmds = torch.as_tensor(np.stack(fa.cmds.to_list()), device="cuda").reshape(n * A, W.FUT, 2)
    pred = M.predict(model, data, starts.repeat_interleave(A), cmds, bs=256 if arm == "T" else 1024).reshape(n, A, W.FUT, -1)
    z0 = data.Z[starts + W.HIST - 1].float()
    rd = {k: W.apply_probes({k: P[k]}, pred)[k] for k in ("d_front", "occ", "ped", "v")}
    d = rl.dir
    np.savez_compressed(d / "preds.npz", fork_id=fa.fork_id.to_numpy(), z0=z0.half().cpu().numpy(), z5=pred[:, :, 4].half().cpu().numpy(),
                        z10=pred[:, :, 9].half().cpu().numpy(), cmds=np.stack(fa.cmds.to_list()), **{f"p_{k}": v for k, v in rd.items()})
    fa.drop(columns="cmds").to_parquet(d / "forks.parquet", index=False)
    (d / "curve.json").write_text(json.dumps(curve))
    torch.save({"state": model.state_dict(), "mu": data.mu.cpu(), "sd": data.sd.cpu(), "arm": arm, "seed": seed,
                "probes": {k: (v[0], v[1].cpu(), float(v[2]) if not hasattr(v[2], "cpu") else v[2].cpu()) for k, v in P.items()},
                "steps": cfg["steps"], "n_params": sum(p.numel() for p in model.parameters())}, d / "model.pt")
    return {"arm": arm, "seed": seed, "fork_points": n, "dir": str(d), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "n_params": sum(p.numel() for p in model.parameters())}


def main():
    import argparse
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prep-ego", "prep-tok", "train"))
    ap.add_argument("--arm", default="B", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="smoke run: fewer steps, written under runs/wl/dryrun_smoke")
    a = ap.parse_args()
    if a.step == "prep-ego":
        r = prep_ego(min(24, n_cpus()))
    elif a.step == "prep-tok":
        r = prep_tok()
    else:
        rl = RunLog("wl", "dryrun_smoke" if a.steps else "dryrun", a.arm, f"seed{a.seed}")
        r = train(a.arm, a.seed, rl, a.steps or None)
        rl.info(json.dumps(r))
        rl.close()
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
