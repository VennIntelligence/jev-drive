"""WL-2 training (fc65452:todos/2026-09-29-wl2-prereg.md, "模型与臂"): the arms, the k-fold seeds, the fork-point predictions.

Universe (one table, rows in this order): W's logged frames of every base route outside the WL eval routes (ds 0, source 0),
WL-1's index rows (ds 1: processed/wl_gen) and WL-2's index rows (ds 2: processed/wl2_gen), all with z = openpilot `temporal`
512 | V-JEPA `mean` 3 072 (fp16) and the ego channel s = [e_y, e_psi, v, a_prev, omega_prev] (experiments.world_model.archive.wl_dryrun.ego_cols).
Every arm reads the same tables and differs only in which rows / windows it trains on and what its input is:

  A     z, WL-1 structure (source embedding, two-block loss), all rows                      seeds 0-4
  B     z + s, three-block loss (openpilot / V-JEPA / s), all rows                          seeds 0-4   (the judged arm)
  Bh    B without windows whose future touches shift_R / op_slow (C2(b))                    seeds 0-4
  Bhc   B without windows whose future touches shift_L_slow / shift_R_slow                  seeds 0-2
  W     W original: logged frames only, no s, source 0 (the known-reversed control)         seeds 0-4
  Bs    B on the index rows only (ds 1, 2): the same rows as T (W's frames have no token)   seeds 0-2
  T     spatial tokens: [openpilot 512 | 24 grid tokens PCA-256 | s], token Transformer     seeds 0-2
  Ax/Bx cross-fit (descriptive): A / B trained on every base route except those of fold `seed` (eval routes included)

Seed i = route fold i as inner-val (base routes of the training universe, stratified by class, 5 folds) + initialisation i.
The eval routes (WL-1 split.json, 40 base routes) never enter training or the inner-val, in any seed of the arms above.

  python -m experiments.world_model.archive.wl2_model prep-outcomes | prep-tok
  python -m experiments.world_model.archive.wl2_model train --arms B A --seeds 0 1 2 [--steps N]      (universe loaded once; resumable per arm x seed)
  python -m experiments.world_model.archive.wl2_model vrep --seeds 0 1 2       WL-1's frozen `main` predictors on the WL-2 (and WL-1) fork points
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("WL_NAME", "wl2")
from experiments.world_model.archive import nq4_w as W  # noqa: E402
from experiments.world_model.lib import wl as WL  # noqa: E402
from experiments.world_model.archive import wl_dryrun as D  # noqa: E402
from experiments.world_model.archive import wl_model as M  # noqa: E402
from jevdrive.common import data_dir, get_logger, n_cpus  # noqa: E402
from experiments.world_model.lib.wl_traj import ACTIONS11, command_actions  # noqa: E402

log = get_logger(__name__)
DD = data_dir()
P1, P2 = DD / "processed" / "wl_gen", DD / "processed" / "wl2_gen"
R1, R2 = DD / "runs" / "wl", DD / "runs" / "wl2"
S_DIM, N_TOK, D_TOK = D.S_DIM, D.N_TOK, 256
HOLD = {"Bh": ("shift_R", "op_slow"), "Bhc": ("shift_L_slow", "shift_R_slow")}
SEEDS = {"A": 5, "B": 5, "Bh": 5, "Bhc": 3, "W": 5, "Bs": 3, "T": 3, "Ax": 5, "Bx": 5}
WITH_S = {"B", "Bh", "Bhc", "Bs", "T", "Bx"}
GID = 100000                                            # global fork id = ds * GID + fork_id
FORK_COLS = ("set", "cls", "family", "base_id", "world", "k_name", "split", "fork_tick")


def pdir2(*p) -> Path:
    return P2.joinpath(*p)


# ================================================================ CPU preparation

def _outcome(a):
    rid, adir, k = a
    try:
        return {"route_id": rid, **WL.outcome(Path(adir), int(k))}
    except Exception as e:
        return {"route_id": rid, "error": repr(e)}


def _offset(a):
    """The executed shift of one branch (experiments.world_model.lib.wl.shift_offsets' definition), signed left offset from the op path."""
    from experiments.world_model.lib.wl_traj import world_to_ego
    rid, adir, k, action = a
    try:
        adir = Path(adir)
        c = json.loads((adir / "wl.json").read_text())["cands"][0]
        tk = pd.read_json(adir / "wl_ticks.jsonl", lines=True).set_index("tick")
        o = WL.outcome(adir, k)
        t_end = k + 60 if o["first_collision_tick"] is None else min(k + 60, o["first_collision_tick"] - 1)
        t_end = max(t for t in tk.index if t <= t_end)
        p = world_to_ego(np.array([tk.rear_xy[t_end]]), np.array(c["rear_xy"]), c["yaw"])[0]
        op = np.vstack([[0.0, 0.0], np.array(c["cands"]["op"])])
        i = int(np.argmin(np.hypot(*(op - p).T)))
        j = min(i + 1, len(op) - 1)
        d = op[j] - op[max(j - 1, 0)]
        n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
        return {"route_id": rid, "action": action, "t_eval_s": (t_end - k) * WL.TICK, "offset_left_m": float((p - op[i]) @ n)}
    except Exception as e:
        return {"route_id": rid, "action": action, "error": repr(e)}


def prep_outcomes(workers: int = 24) -> dict:
    """WL-2: outcome of every finished fork run -> processed/wl2_gen/outcomes.parquet, and the executed offset of every
    shift-type branch -> offsets.parquet (the covariate of the slow-shift analysis)."""
    from multiprocessing import Pool
    from experiments.world_model.archive.wl_data import finished
    r = finished(R2 / "gen")
    with Pool(workers) as p:
        o = pd.DataFrame(p.map(_outcome, list(zip(r.route_id, r.adir, r.fork_tick)), chunksize=4))
        o["collision_types"] = o.get("collision_types", pd.Series(dtype=object)).map(lambda x: json.dumps(x) if isinstance(x, list) else x)
        o.to_parquet(pdir2("outcomes.parquet"), index=False)
        sh = r[r.action.str.startswith(("shift_", "nudge_"))]
        off = pd.DataFrame(p.map(_offset, list(zip(sh.route_id, sh.adir, sh.fork_tick, sh.action)), chunksize=4))
    off.to_parquet(pdir2("offsets.parquet"), index=False)
    return {"runs": len(o), "outcome_errors": int(o.get("error", pd.Series(dtype=object)).notna().sum()), "unsafe": float(o.unsafe.mean()),
            "unsafe_cg": float(o.unsafe_cg.mean()), "offset_rows": len(off), "offset_errors": int(off.get("error", pd.Series(dtype=object)).notna().sum())}


def prep_tok(n_fit: int = 30000, chunk: int = 4096) -> dict:
    """One PCA 1 024 -> 256 shared over the 24 grid tokens, fitted on n_fit random rows of each of WL-1 and WL-2, applied to
    both -> processed/wl2_gen/pca/tokpca256_w{1,2}.npy (rows aligned with each index.parquet)."""
    import torch
    (P2 / "pca").mkdir(exist_ok=True)
    toks = {1: np.load(P1 / "tok.npy", mmap_mode="r"), 2: np.load(P2 / "tok.npy", mmap_mode="r")}
    s1 = torch.zeros(1024, dtype=torch.float64, device="cuda")
    s2 = torch.zeros(1024, 1024, dtype=torch.float64, device="cuda")
    n = 0
    for ds, tok in toks.items():
        rows = np.sort(np.random.RandomState(ds).choice(len(tok), n_fit, replace=False))
        for lo in range(0, n_fit, 1000):
            x = torch.as_tensor(np.asarray(tok[rows[lo: lo + 1000]]), device="cuda").float().reshape(-1, 1024)
            s1 += x.sum(0).double()
            s2 += (x.T @ x).double()
            n += len(x)
    mu = s1 / n
    e, V = torch.linalg.eigh(s2 / n - torch.outer(mu, mu))
    e, V = e.flip(0), V.flip(1)
    Pm, mu32 = V[:, :D_TOK].float(), mu.float()
    for ds, tok in toks.items():
        out = np.lib.format.open_memmap(pdir2("pca", f"tokpca256_w{ds}.tmp.npy"), mode="w+", dtype=np.float16, shape=(len(tok), N_TOK * D_TOK))
        for lo in range(0, len(tok), chunk):
            x = torch.as_tensor(np.asarray(tok[lo: lo + chunk]), device="cuda").float()
            out[lo: lo + len(x)] = ((x - mu32) @ Pm).reshape(len(x), -1).half().cpu().numpy()
        out.flush()
        del out
        pdir2("pca", f"tokpca256_w{ds}.tmp.npy").replace(pdir2("pca", f"tokpca256_w{ds}.npy"))
    info = {"fit_tokens": n, "dim": D_TOK, "variance_retained": float(e[:D_TOK].sum() / e.sum()), "rows_w1": len(toks[1]), "rows_w2": len(toks[2])}
    pdir2("pca", "tokpca256.json").write_text(json.dumps(info, indent=1))
    return info


# ================================================================ universe

class Universe:
    """m (meta), z (N, 3584 fp16), S (N, 5 fp32), the fork table and, on demand, the PCA'd tokens of the index rows."""

    def __init__(self):
        ev = set(json.loads(WL.rundir("split.json").read_text())["eval"])
        wm = pd.read_parquet(W.wdir("meta.parquet"))
        wz = np.load(W.wdir("z.npy"), mmap_mode="r")
        keep = ~wm.base_id.astype(str).isin(ev).to_numpy()
        m1 = wm[keep].assign(src=0, a_cmd=wm.a_next[keep], w_cmd=wm.w_next[keep], kind="log", split="train", route_id=wm.adir[keep],
                             tick=wm.frame[keep], action=None, fork_id=-1, ds=0)
        m1["win_start"] = wm.win_start[keep].to_numpy()
        parts, zs, eys, self.tok_rows = [m1], [np.asarray(wz[np.flatnonzero(keep)])], [np.load(P1 / "dryrun" / "ey_wm.npy")[keep]], {}
        for ds, P in ((1, P1), (2, P2)):
            t = pd.read_parquet(P / "index.parquet")
            ok = np.load(P / "z_ok.npy")
            ey = np.load(P / "dryrun" / "ey_idx.npy") if ds == 1 else np.load(P / "ey.npy")
            self.tok_rows[ds] = np.flatnonzero(ok)
            t = t[ok].copy()
            t["action"] = t.action_x
            t["kind"] = np.where(t.set == "d2", "d2", "fork")
            t["base_id"] = t.base_id.astype(str)
            t["win_start"] = M._segments(t, "route_id").to_numpy()
            t["ds"] = ds
            for c in ("a", "b", "c"):
                t[c] = np.nan
            parts.append(t)
            zs.append(np.load(P / "z.npy", mmap_mode="r")[np.flatnonzero(ok)])
            eys.append(ey[ok])
        cols = ["ds", "set", "base_id", "route_id", "tick", "kind", "split", "world", "cls", "fork_id", "action", "src", "v", "a_prev", "w_prev",
                "a_next", "w_next", "a_cmd", "w_cmd", "ped", "occ", "d_front", "a", "b", "c", "win_start"]
        m = pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)
        m["a_next"], m["w_next"] = m.a_cmd.fillna(m.a_next), m.w_cmd.fillna(m.w_next)
        for c in ("a_next", "w_next", "a_prev", "w_prev"):
            m[c] = m[c].fillna(0.0).clip(-15, 15)
        m["v"] = m.v.fillna(0.0)
        m["src"] = m.src.fillna(0).astype(int)
        m["base_id"] = m.base_id.astype(str)
        self.m, self.z = m, np.concatenate(zs)
        self.S = D.ego_cols(m, np.concatenate(eys))
        self.ev = ev
        self.n_w = int(len(m1))
        self._tokp = None
        self.fold5 = route_folds(m, ev)
        self.fa = None

    def tokp(self) -> np.ndarray:
        """(N, 24 x 256) PCA tokens per universe row (zeros for W's frames, which have none)."""
        if self._tokp is None:
            out = np.zeros((len(self.m), N_TOK * D_TOK), np.float16)
            o = self.n_w
            for ds in (1, 2):
                q = np.load(pdir2("pca", f"tokpca256_w{ds}.npy"), mmap_mode="r")
                n = len(self.tok_rows[ds])
                out[o: o + n] = q[self.tok_rows[ds]]
                o += n
            self._tokp = out
        return self._tokp

    def forks(self) -> pd.DataFrame:
        """Per fork point of WL-1 and WL-2 (every split): anchor row and the 11 candidates' commanded actions (10 steps)."""
        if self.fa is not None:
            return self.fa
        m = self.m
        row = pd.Series(np.arange(len(m)), index=m.ds.astype(str) + "|" + m.route_id.astype(str) + "|" + m.tick.astype(str))
        ok_hist = m.win_start.to_numpy()
        out = []
        for ds, R, P in ((1, R1, P1), (2, R2, P2)):
            runs = pd.read_parquet(R / "forks.parquet")
            wins = pd.read_parquet(P / "windows.parquet")
            wins = wins[wins.kind == "fork"].drop_duplicates("route_id").set_index("route_id")
            for fid, g in runs.groupby("fork_id"):
                for r in g.itertuples():
                    i = row.get(f"{ds}|{r.route_id}|{r.fork_tick}")
                    if i is None or i < W.HIST - 1 or r.route_id not in wins.index or not ok_hist[i - (W.HIST - 1)]:
                        continue
                    w = wins.loc[r.route_id]
                    cmds = command_actions(json.loads(w.op_plan), np.asarray(json.loads(w.route_ego)), float(w.v0))
                    out.append({"gid": ds * GID + int(fid), "ds": ds, "fork_id": int(fid), "anchor": int(i), "v0": float(w.v0),
                                "cmds": np.stack([cmds[a] for a in ACTIONS11]), **{k: getattr(r, k) for k in FORK_COLS}})
                    break
        self.fa = pd.DataFrame(out)
        return self.fa


def route_folds(m: pd.DataFrame, ev: set, k: int = 5, seed: int = 20260930, include_eval: bool = False) -> dict:
    """base route -> fold, stratified by class (ped / cutin / obstacle from the fork rows, 'other' for the rest): the routes of
    each class in a seeded permutation, round-robin into k folds. Eval routes are left out unless include_eval."""
    cls = m.groupby("base_id").cls.agg(lambda s: s.dropna().iloc[0] if s.notna().any() else "other")
    rng = np.random.RandomState(seed)
    out, n = {}, 0
    for c in sorted(cls.unique()):                       # the round-robin counter runs on across classes: folds stay balanced
        rs = sorted(b for b in cls.index[cls == c] if include_eval or b not in ev)
        for j in rng.permutation(len(rs)):
            out[rs[j]] = n % k
            n += 1
    return out


def rows_windows(m: pd.DataFrame, arm: str, seed: int, ev: set, fold5: dict, foldx: dict):
    """(train windows, inner-val windows, train rows, inner-val rows) of an arm x seed, and the leakage checks."""
    ds = m.ds.to_numpy()
    base = m.base_id.to_numpy()
    if arm in ("Ax", "Bx"):
        held = np.array([foldx.get(b, -1) == seed for b in base])
        allowed = ~held
        rng = np.random.RandomState(seed)
        routes = np.array(sorted(set(base[allowed])))
        inner_set = set(routes[rng.rand(len(routes)) < 0.10])
        is_in = np.isin(base, list(inner_set))
    else:
        allowed = ~np.isin(base, list(ev))
        allowed &= (ds == 0) if arm == "W" else ((ds > 0) if arm in ("Bs", "T") else True)
        is_in = np.array([fold5.get(b, -1) == seed for b in base])
    st = np.flatnonzero(m.win_start.to_numpy() & allowed)
    fut = st[:, None] + W.HIST - 1 + np.arange(W.FUT)
    if arm in HOLD:
        bad = m.action.isin(HOLD[arm]).to_numpy() & (m.src.to_numpy() == 1)
        st = st[~bad[fut].any(1)]
    anc = st + W.HIST - 1
    tr_st, va_st = st[~is_in[anc]], st[is_in[anc]]
    tr_rows = np.flatnonzero(~is_in & allowed)
    va_rows = np.flatnonzero(is_in & allowed)
    # leakage checks: no window of the training / inner-val sets touches a row outside `allowed`; train and inner-val share no route
    for s_, nm in ((tr_st, "train"), (va_st, "inner-val")):
        w = s_[:, None] + np.arange(W.WIN)
        assert allowed[w].all(), f"{arm} seed {seed}: a {nm} window reaches a row outside the arm's universe"
        assert (ds[w] == ds[s_][:, None]).all(), f"{arm}: a window spans two datasets"
    assert not (set(base[tr_rows]) & set(base[va_rows])), f"{arm} seed {seed}: train and inner-val share a base route"
    if arm not in ("Ax", "Bx"):
        assert not (set(base[tr_rows]) | set(base[va_rows])) & ev, f"{arm}: an eval route in the training universe"
        assert (m.split.to_numpy()[st] != "eval").all()
    if arm in HOLD:
        w = tr_st[:, None] + W.HIST - 1 + np.arange(W.FUT)
        assert not (m.action.isin(HOLD[arm]).to_numpy()[w] & (m.src.to_numpy()[w] == 1)).any(), f"{arm}: a held-out action is in a training window"
    return tr_st, va_st, tr_rows, va_rows


# ================================================================ data on the GPU and the predictors

class Data2(D.Data):
    """D.Data over several host blocks (concatenated on the GPU per chunk); moments over the training rows unless given."""

    def __init__(self, meta, blocks, train_rows, dev="cuda", chunk=16384, moments=None):
        import torch

        def cat(rows):
            return torch.cat([torch.as_tensor(b[rows], device=dev).float() for b in blocks], 1)
        dz = sum(b.shape[1] for b in blocks)
        if moments is None:
            tr = np.sort(np.asarray(train_rows))
            s1 = torch.zeros(dz, dtype=torch.float64, device=dev)
            s2 = torch.zeros_like(s1)
            for lo in range(0, len(tr), chunk):
                x = cat(tr[lo: lo + chunk]).double()
                s1 += x.sum(0)
                s2 += (x * x).sum(0)
            n = len(tr)
            mu = s1 / n
            self.mu = mu.float()
            self.sd = ((s2 / n - mu * mu).clamp_min(0) * n / (n - 1)).sqrt().float().clamp_min(1e-4)
        else:
            self.mu, self.sd = moments[0].to(dev).float(), moments[1].to(dev).float()
        N = len(meta)
        self.Z = torch.empty((N, dz), device=dev, dtype=torch.bfloat16)
        for lo in range(0, N, 8192):
            self.Z[lo: lo + 8192] = ((cat(np.arange(lo, min(lo + 8192, N))) - self.mu) / self.sd).bfloat16()
        self.hact = torch.as_tensor(np.c_[meta[["a_prev", "w_prev"]].to_numpy(np.float32) / W.ACT_SCALE,
                                          meta.v.to_numpy(np.float32)[:, None] / W.V_SCALE], device=dev)
        self.fact = torch.as_tensor(meta[["a_next", "w_next"]].to_numpy(np.float32) / W.ACT_SCALE, device=dev)
        self.dev = dev
        self.off = torch.arange(W.WIN, device=dev)
        self.S = torch.as_tensor(meta.src.to_numpy(np.int64), device=dev)


def blocks_of(U: Universe, arm: str) -> list:
    if arm == "T":
        return [U.z[:, :W.D_OP], U.tokp(), U.S]
    return [U.z, U.S] if arm in WITH_S else [U.z]


def build_tok(cfg=W.CFG):
    """The token Transformer of the T arm: D.build_tok with D_TOK = 256 (24 tokens x 256 after the shared PCA)."""
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

        def embed(s, z):
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


def loss_fn(arm: str):
    return D.block_mse3 if arm in WITH_S else W.block_mse


def fit(model, data, tr, va, seed, rl, tag, cfg, batch, compile_, lossf, ckpt: Path | None = None):
    """WL-1's loop (AdamW, warm-up + cosine, best inner-val state); resumes from `ckpt` (model, optimiser, sampler state, best)."""
    import torch
    steps = cfg["steps"]
    fwd = torch.compile(model, mode="reduce-overhead") if compile_ else model
    g = torch.Generator(device="cuda").manual_seed(seed)
    tr, va = torch.as_tensor(tr, device="cuda"), torch.as_tensor(va, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    best, state, curve, t0, step0 = np.inf, None, [], time.time(), 1
    if ckpt is not None and ckpt.exists():
        c = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        g.set_state(c["gen"].cpu())
        best, state, curve, step0 = c["best"], c["state"], c["curve"], c["step"] + 1
        rl.info(f"{tag}: resumed at step {step0}")
    for step in range(step0, steps + 1):
        model.train()
        zh, ha, fa, zf, sh, sf = data.batch(tr[torch.randint(len(tr), (batch,), device="cuda", generator=g)])
        with torch.autocast("cuda", torch.bfloat16):
            pred = fwd(zh, ha, fa, sh, sf)
        loss = lossf(pred.float(), zf)
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
                    tot += float(lossf(model(b[0], b[1], b[2], b[4], b[5]).float(), b[3])) * len(b[0])
                    n += len(b[0])
            vl = tot / max(n, 1)
            curve.append({"step": step, "train": float(loss), "val": vl, "wall_s": time.time() - t0})
            rl.scalar(f"{tag}/train_loss", float(loss), step)
            rl.scalar(f"{tag}/val_loss", vl, step)
            rl.info(f"{tag} step {step}: train {float(loss):.4f}, inner-val {vl:.4f}, {(step - step0 + 1) / (time.time() - t0):.1f} steps/s, "
                    f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
            if vl < best:
                best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt is not None and step % (2 * cfg["eval_every"]) == 0 and step < steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": g.get_state(),
                            "best": best, "state": state, "curve": curve, "step": step}, ckpt)
    model.load_state_dict(state)
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    return curve


def probe_dict(P):
    return {k: (v[0], v[1].cpu(), float(v[2]) if not hasattr(v[2], "cpu") else v[2].cpu()) for k, v in P.items()}


def predict_forks(model, data, fa: pd.DataFrame, P: dict, src: int, bs: int, d: Path):
    """The 11 candidates of every fork point rolled out 10 steps -> probe readouts; preds.npz + forks.parquet in `d`."""
    import torch
    starts = torch.as_tensor(fa.anchor.to_numpy() - (W.HIST - 1), device="cuda")
    n, A = len(fa), len(ACTIONS11)
    cmds = torch.as_tensor(np.stack(fa.cmds.to_list()), device="cuda").reshape(n * A, W.FUT, 2)
    pred = M.predict(model, data, starts.repeat_interleave(A), cmds, bs=bs, src=src).reshape(n, A, W.FUT, -1)
    z0 = data.Z[starts + W.HIST - 1].float()
    rd = {k: W.apply_probes({k: P[k]}, pred)[k] for k in ("d_front", "occ", "ped", "v")}
    np.savez_compressed(d / "preds.npz", gid=fa.gid.to_numpy(), z0=z0.half().cpu().numpy(), z5=pred[:, :, 4].half().cpu().numpy(),
                        z10=pred[:, :, 9].half().cpu().numpy(), cmds=np.stack(fa.cmds.to_list()), **{f"p_{k}": v for k, v in rd.items()})
    fa.drop(columns="cmds").to_parquet(d / "forks.parquet", index=False)
    return pred


def train_arm(U: Universe, arm: str, seed: int, rl, steps: int | None = None, resume: bool = True) -> dict:
    import torch
    torch.manual_seed(seed)
    m = U.m
    foldx = None
    if arm in ("Ax", "Bx"):
        if not hasattr(U, "foldx"):
            U.foldx = route_folds(m, U.ev, seed=20260931, include_eval=True)
        foldx = U.foldx
        (R2 / "xfit_folds.json").write_text(json.dumps(foldx))
    tr_st, va_st, tr_rows, va_rows = rows_windows(m, arm, seed, U.ev, U.fold5, foldx)
    data = Data2(m, blocks_of(U, arm), tr_rows)
    dz = data.Z.shape[1]
    torch.manual_seed(seed)
    model = (build_tok() if arm == "T" else M.build(dz)).cuda()
    rl.info(f"{arm} seed {seed}: {len(m)} rows (dz {dz}), {len(tr_st)} training windows, {len(va_st)} inner-val windows, "
            f"{len(tr_rows)} train rows / {len(va_rows)} inner-val rows, {int(m.src.iloc[tr_rows].sum())} intervention train rows")
    cfg = {**W.CFG, **({"steps": steps} if steps else {})}
    t0 = time.time()
    ck = (R2 / "ckpt" / f"{arm}_seed{seed}.pt") if resume and arm == "T" and not steps else None
    if ck is not None:
        ck.parent.mkdir(exist_ok=True)
    curve = fit(model, data, tr_st, va_st, seed, rl, f"{arm}/seed{seed}", cfg, cfg["batch"], arm != "T", loss_fn(arm), ck)
    fit_s = time.time() - t0
    rl.info(f"{arm}/seed{seed}: predictor fit {fit_s:.0f} s")
    P = D.fit_probes(data, M.meta_for_probes(m), tr_rows, va_rows)
    fa = U.forks()
    pred = predict_forks(model, data, fa, P, 0 if arm == "W" else 1, 256 if arm == "T" else 1024, rl.dir)
    del pred
    (rl.dir / "curve.json").write_text(json.dumps(curve))
    torch.save({"state": model.state_dict(), "mu": data.mu.cpu(), "sd": data.sd.cpu(), "arm": arm, "seed": seed, "probes": probe_dict(P),
                "steps": cfg["steps"], "n_params": sum(p.numel() for p in model.parameters()), "fit_s": fit_s}, rl.dir / "model.pt")
    return {"arm": arm, "seed": seed, "forks": len(fa), "dir": str(rl.dir), "fit_s": fit_s, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "n_train_windows": len(tr_st), "n_inner_windows": len(va_st), "best_val": min(c["val"] for c in curve)}


def vrep(U: Universe, seed: int, rl) -> dict:
    """WL-1's frozen `main` predictor (runs/wl/model/main/seed<seed>) on the WL-2 (and WL-1) fork points, 11 candidates asked
    (the 7 WL-1 ones are the readout); the critic is refitted on top by the report."""
    import torch
    d1 = sorted((R1 / "model" / "main" / f"seed{seed}").glob("*/model.pt"))[-1]
    ck = torch.load(d1, map_location="cpu")
    assert "probes" in ck
    data = Data2(U.m, [U.z], np.arange(0), moments=(ck["mu"], ck["sd"]))
    model = M.build(data.Z.shape[1]).cuda()
    model.load_state_dict({k: v.cuda() for k, v in ck["state"].items()})
    P = {k: (v[0], v[1].cuda(), v[2].cuda() if hasattr(v[2], "cuda") else v[2]) for k, v in ck["probes"].items()}
    predict_forks(model, data, U.forks(), P, 1, 1024, rl.dir)
    (rl.dir / "src.json").write_text(json.dumps({"wl1_checkpoint": str(d1)}))
    return {"arm": "vrep", "seed": seed, "src": str(d1)}


def latest(arm: str, seed: int) -> Path | None:
    fs = sorted((R2 / "model" / arm / f"seed{seed}").glob("*/preds.npz" if arm == "vrep" else "*/model.pt"))
    return fs[-1].parent if fs else None


def main():
    import argparse
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prep-outcomes", "prep-tok", "train", "vrep"))
    ap.add_argument("--arms", nargs="+", default=["B"], choices=tuple(SEEDS) + ("vrep",))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=0, help="smoke run: fewer steps, written under runs/wl2/smoke")
    a = ap.parse_args()
    if a.arms == ["vrep"]:
        a.step = "vrep"
    if a.step == "prep-outcomes":
        print(json.dumps(prep_outcomes(min(24, n_cpus())), indent=1))
    elif a.step == "prep-tok":
        print(json.dumps(prep_tok(), indent=1))
    else:
        U = Universe()
        log.info("universe: %d rows (W %d), %d fork points", len(U.m), U.n_w, len(U.forks()))
        for arm in (["vrep"] if a.step == "vrep" else a.arms):
            for s in a.seeds:
                if a.step == "train" and s >= SEEDS[arm]:
                    continue
                if not a.steps and latest(arm, s) is not None:
                    log.info("skip %s seed %d (done)", arm, s)
                    continue
                rl = RunLog("wl2", "smoke" if a.steps else "model", arm, f"seed{s}")
                r = vrep(U, s, rl) if a.step == "vrep" else train_arm(U, arm, s, rl, a.steps or None)
                rl.info(json.dumps(r, default=float))
                rl.close()
                print(json.dumps(r, default=float))


if __name__ == "__main__":
    main()
