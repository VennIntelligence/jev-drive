"""WL world model, critics and the registered readouts C1-C3 (todos/2026-09-28-wm-loop.md, "模型" / "判据").

The predictor is W's (jevdrive/nq4_w.py: 8 history steps -> 10 future Delta z, 6-layer Transformer, CFG unchanged) plus one
learned 2-way "action source" embedding on every history and query token (0 = policy: the logged expert action;
1 = intervention: the commanded (a, omega) of a WL window's candidate). All readouts feed source 1.

Universe: W's logged frames (processed/nq4_w, source 0) of every base route outside the WL eval split, plus the WL index
(processed/wl_gen: fork runs of the training split and D2). Arms:
  main     everything
  intonly  only windows whose 10 future steps are all intervention steps
  holdout  main without any window whose future touches a shift_R or op_slow step (C2(b))
  worig    W's logged frames only (the known-reversed control of C1)

  outcomes  CPU. jevdrive.wl.outcome for every finished fork run -> processed/wl_gen/outcomes.parquet
  train     one arm x one seed: predictor, probes on real z, predictions of every fork point's 7 candidates
            -> runs/wl/model/<arm>/seed<s>/<stamp>/{preds.npz, probes.npz, curve.json}
  report    C1-C3 from the stored predictions (critics fitted here, CPU / GPU small) -> research/results/wl/
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import nq4_w as W
from . import wl as WL
from .common import data_dir, get_logger
from .wl_data import pdir
from .wl_traj import ACTIONS, command_actions

log = get_logger(__name__)
ARMS = ("main", "intonly", "holdout", "worig")
HOLD_OUT = ("shift_R", "op_slow")
SEEDS = (0, 1, 2)
UNSAFE_P = 0.2
MLP = dict(hidden=256, drop=0.1, lr=1e-3, wd=1e-2, steps=3000, batch=512)


# ================================================================ outcomes (CPU)

def _outcome(args):
    rid, adir, k = args
    try:
        return {"route_id": rid, **WL.outcome(Path(adir), int(k))}
    except Exception as e:
        return {"route_id": rid, "error": repr(e)}


def outcomes(workers: int = 16) -> dict:
    from multiprocessing import Pool
    from .wl_data import finished
    r = finished(Path(data_dir() / "runs" / "wl" / "gen"))
    r = r[r.set != "d2"]
    with Pool(workers) as p:
        o = pd.DataFrame(p.map(_outcome, list(zip(r.route_id, r.adir, r.fork_tick)), chunksize=4))
    o["collision_types"] = o.get("collision_types", pd.Series(dtype=object)).map(lambda x: json.dumps(x) if isinstance(x, list) else x)
    o.to_parquet(pdir("outcomes.parquet"), index=False)
    return {"runs": len(o), "errors": int(o.get("error", pd.Series(dtype=object)).notna().sum()), "unsafe": float(o.unsafe.mean())}


# ================================================================ universe

def _segments(m: pd.DataFrame, key: str) -> pd.Series:
    seg = (m[key] != m[key].shift()) | (m.tick.diff() != W.TICKS)
    s = seg.cumsum()
    pos, n = m.groupby(s).cumcount(), m.groupby(s)[key].transform("size")
    return (n - pos) >= W.WIN


def universe(arm: str):
    """(meta, z) of the arm's rows: W's logged frames outside the eval routes (+ the WL index unless worig)."""
    ev = set(json.loads(WL.rundir("split.json").read_text())["eval"])
    wm = pd.read_parquet(W.wdir("meta.parquet"))
    wz = np.load(W.wdir("z.npy"), mmap_mode="r")
    keep = ~wm.base_id.astype(str).isin(ev).to_numpy()
    m1 = wm[keep].assign(src=0, a_cmd=wm.a_next[keep], w_cmd=wm.w_next[keep], kind="log", split="train", route_id=wm.adir[keep],
                         tick=wm.frame[keep], action=None, fork_id=-1)
    m1["win_start"] = wm.win_start[keep].to_numpy()
    parts, zs = [m1], [np.asarray(wz[np.flatnonzero(keep)])]
    if arm != "worig":
        t = pd.read_parquet(pdir("index.parquet"))
        ok = np.load(pdir("z_ok.npy"))
        t = t[ok].copy()
        t["action"] = t.action_x                            # per-step window candidate (the merge with forks.parquet renamed it)
        t["kind"] = np.where(t.set == "d2", "d2", "fork")
        t["base_id"] = t.base_id.astype(str)
        t["win_start"] = _segments(t, "route_id").to_numpy()
        for c in ("a", "b", "c"):
            t[c] = np.nan
        parts.append(t)
        zs.append(np.load(pdir("z.npy"), mmap_mode="r")[np.flatnonzero(ok)])
    cols = ["set", "base_id", "route_id", "tick", "kind", "split", "world", "cls", "fork_id", "action", "src", "v", "a_prev",
            "w_prev", "a_next", "w_next", "a_cmd", "w_cmd", "ped", "occ", "d_front", "a", "b", "c", "win_start"]
    m = pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)
    m["a_next"], m["w_next"] = m.a_cmd.fillna(m.a_next), m.w_cmd.fillna(m.w_next)        # the model's future actions
    for c in ("a_next", "w_next", "a_prev", "w_prev"):
        m[c] = m[c].fillna(0.0).clip(-15, 15)
    m["v"] = m.v.fillna(0.0)
    m["src"] = m.src.fillna(0).astype(int)
    return m, np.concatenate(zs)


def train_windows(m: pd.DataFrame, arm: str) -> np.ndarray:
    st = np.flatnonzero(m.win_start.to_numpy() & (m.split != "eval").to_numpy())
    fut = st[:, None] + W.HIST - 1 + np.arange(W.FUT)                  # rows whose step is a future action
    src = m.src.to_numpy()[fut]
    if arm == "intonly":
        st = st[src.all(1)]
    elif arm == "holdout":
        bad = m.action.isin(HOLD_OUT).to_numpy() & (m.src.to_numpy() == 1)
        st = st[~bad[fut].any(1)]
    return st


# ================================================================ model

def build(dz: int = W.D_OP + W.D_VJ, cfg=W.CFG):
    import torch
    from torch import nn
    base = W.build_model(cfg, dz)

    class WLM(nn.Module):
        def __init__(s):
            super().__init__()
            s.w = base
            s.src = nn.Embedding(2, cfg["d"])
            nn.init.normal_(s.src.weight, std=0.02)

        def forward(s, zh, hact, fact, sh, sf):
            w = s.w
            x = torch.cat([w.zin(zh) + w.hin(hact) + s.src(sh), w.qin(fact) + s.src(sf)], 1) + w.pos
            return zh[:, -1:] + w.out(w.norm(w.enc(x)[:, W.HIST:]))
    return WLM()


class Data(W.Data):
    def __init__(self, meta, z, train_rows, dev="cuda"):
        import torch
        super().__init__(meta, z, train_rows, dev)
        self.S = torch.as_tensor(meta.src.to_numpy(np.int64), device=dev)

    def batch(self, starts, fact=None, act_starts=None, src_future=None):
        zh, ha, fa, zf = super().batch(starts, fact, act_starts)
        r = starts[:, None] + self.off
        sh = self.S[r[:, :W.HIST]]
        sf = self.S[r[:, W.HIST - 1: W.WIN - 1]] if src_future is None else src_future.expand(len(starts), W.FUT)
        return zh, ha, fa, zf, sh, sf


def _train(model, data, tr, va, seed, rl, tag, cfg=W.CFG):
    import torch
    steps = cfg["steps"]
    fwd = torch.compile(model, mode="reduce-overhead")
    g = torch.Generator(device="cuda").manual_seed(seed)
    tr, va = torch.as_tensor(tr, device="cuda"), torch.as_tensor(va, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) *
                                              0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    best, state, curve, t0 = np.inf, None, [], time.time()
    for step in range(1, steps + 1):
        model.train()
        zh, ha, fa, zf, sh, sf = data.batch(tr[torch.randint(len(tr), (cfg["batch"],), device="cuda", generator=g)])
        with torch.autocast("cuda", torch.bfloat16):
            pred = fwd(zh, ha, fa, sh, sf)
        loss = W.block_mse(pred.float(), zf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for lo in range(0, len(va), 1024):
                    b = data.batch(va[lo: lo + 1024])
                    tot += float(W.block_mse(model(b[0], b[1], b[2], b[4], b[5]).float(), b[3])) * len(b[0])
                    n += len(b[0])
            vl = tot / max(n, 1)
            curve.append({"step": step, "train": float(loss), "val": vl, "wall_s": time.time() - t0})
            rl.scalar(f"{tag}/train_loss", float(loss), step)
            rl.scalar(f"{tag}/val_loss", vl, step)
            rl.info(f"{tag} step {step}: train {float(loss):.4f}, inner-val {vl:.4f}, {step / (time.time() - t0):.1f} steps/s")
            if vl < best:
                best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return curve


def predict(model, data, starts, fact, bs: int = 1024, src: int = 1):
    """(n, 10, dz) predictions with the given future actions (physical units) and action source (1 = intervention;
    the worig arm never saw source 1, so it is read with 0)."""
    import torch
    model.eval()
    one = torch.full((1, 1), src, dtype=torch.long, device="cuda")
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for lo in range(0, len(starts), bs):
            zh, ha, fa, _, sh, sf = data.batch(starts[lo: lo + bs], fact[lo: lo + bs], src_future=one)
            out.append(model(zh, ha, fa, sh, sf).float())
    return torch.cat(out)


# ================================================================ fork points

def fork_anchors(m: pd.DataFrame) -> pd.DataFrame:
    """Per fork point (every split): the anchor row (the fork tick of any of its branch runs present in m, with a full
    history), and the 7 candidates' commanded actions (10 steps) from that run's window record."""
    runs = pd.read_parquet(WL.rundir("forks.parquet"))
    wins = pd.read_parquet(pdir("windows.parquet"))
    wins = wins[wins.kind == "fork"].set_index("route_id")
    row = pd.Series(np.arange(len(m)), index=m.route_id.astype(str) + "|" + m.tick.astype(str))
    ok_hist = m.win_start.to_numpy()
    out = []
    for fid, g in runs.groupby("fork_id"):
        for r in g.itertuples():
            i = row.get(f"{r.route_id}|{r.fork_tick}")
            if i is None or r.route_id not in wins.index or not ok_hist[i - (W.HIST - 1)]:
                continue
            w = wins.loc[r.route_id]
            cmds = command_actions(json.loads(w.op_plan), np.asarray(json.loads(w.route_ego)), float(w.v0))
            out.append({"fork_id": fid, "anchor": int(i), "cmds": np.stack([cmds[a] for a in ACTIONS]),
                        **{k: getattr(r, k) for k in ("set", "cls", "family", "base_id", "world", "k_name", "split", "fork_tick")}})
            break
    return pd.DataFrame(out)


def train(arm: str, seed: int, rl, steps: int | None = None) -> dict:
    import torch
    torch.manual_seed(seed)
    m, z = universe(arm)
    st = train_windows(m, arm)
    routes = np.array(sorted(m.base_id.astype(str).unique()))
    rng = np.random.RandomState(seed)
    inner = set(routes[rng.rand(len(routes)) < 0.10])
    is_in = m.base_id.astype(str).isin(inner).to_numpy()
    anc = st + W.HIST - 1
    tr_st, va_st = st[~is_in[anc]], st[is_in[anc]]
    tr_rows = np.flatnonzero(~is_in & (m.split != "eval").to_numpy())
    va_rows = np.flatnonzero(is_in & (m.split != "eval").to_numpy())
    data = Data(m, z, tr_rows)
    torch.manual_seed(seed)
    model = build(z.shape[1]).cuda()
    rl.info(f"{arm} seed {seed}: {len(m)} rows, {len(tr_st)} training windows ({int(m.src.sum())} intervention rows), "
            f"{len(va_st)} inner-val windows")
    t_fit = time.time()
    curve = _train(model, data, tr_st, va_st, seed, rl, f"{arm}/seed{seed}", cfg={**W.CFG, **({"steps": steps} if steps else {})})
    rl.info(f"{arm}/seed{seed}: predictor fit {time.time() - t_fit:.0f} s")
    P = W.probes(data, meta_for_probes(m), tr_rows, va_rows)
    Xt, Xv = data.Z[torch.as_tensor(tr_rows, device="cuda")].float(), data.Z[torch.as_tensor(va_rows, device="cuda")].float()
    wv, muv, _ = W.fit_ridge(Xt, torch.as_tensor(m.v.to_numpy(np.float32)[tr_rows], device="cuda"), Xv,
                             torch.as_tensor(m.v.to_numpy(np.float32)[va_rows], device="cuda"))
    P["v"] = ("ridge", wv, muv, 0)
    # every fork point: anchor z, 7 candidates' predictions -> probe readouts + the critic inputs
    if arm == "worig":                                  # the fork rows are not in this arm's universe: borrow them
        mf, zf = universe("main")
        dataf = Data(mf, zf, np.arange(0))
        dataf.mu, dataf.sd = data.mu, data.sd
        for lo in range(0, len(zf), 8192):
            dataf.Z[lo: lo + 8192] = ((torch.as_tensor(zf[lo: lo + 8192], device="cuda").float() - data.mu) / data.sd).bfloat16()
        dataf.S.zero_()
        fa = fork_anchors(mf)
    else:
        dataf, fa = data, fork_anchors(m)
    starts = torch.as_tensor(fa.anchor.to_numpy() - (W.HIST - 1), device="cuda")
    n, A = len(fa), len(ACTIONS)
    cmds = torch.as_tensor(np.stack(fa.cmds.to_list()), device="cuda").reshape(n * A, W.FUT, 2)
    pred = predict(model, dataf, starts.repeat_interleave(A), cmds, src=0 if arm == "worig" else 1).reshape(n, A, W.FUT, -1)
    z0 = dataf.Z[starts + W.HIST - 1].float()
    rd = {k: W.apply_probes({k: P[k]}, pred)[k] for k in ("d_front", "occ", "ped", "v")}
    d = rl.dir
    np.savez_compressed(d / "preds.npz", fork_id=fa.fork_id.to_numpy(), z0=z0.half().cpu().numpy(),
                        z5=pred[:, :, 4].half().cpu().numpy(), z10=pred[:, :, 9].half().cpu().numpy(),
                        cmds=np.stack(fa.cmds.to_list()), **{f"p_{k}": v for k, v in rd.items()})
    fa.drop(columns="cmds").to_parquet(d / "forks.parquet", index=False)
    (d / "curve.json").write_text(json.dumps(curve))
    torch.save({"state": model.state_dict(), "mu": data.mu.cpu(), "sd": data.sd.cpu(), "arm": arm, "seed": seed,
                "steps": steps or W.CFG["steps"], "n_params": sum(p.numel() for p in model.parameters())}, d / "model.pt")
    return {"arm": arm, "seed": seed, "fork_points": n, "dir": str(d), "n_params": sum(p.numel() for p in model.parameters())}


def meta_for_probes(m):
    return m.assign(a=m.a.astype(float), b=m.b.astype(float), c=m.c.astype(float), ped=m.ped.astype(float),
                    occ=m.occ.astype(float))


# ================================================================ critics and the registered readouts

def _latest(arm, seed) -> Path | None:
    fs = sorted((data_dir() / "runs" / "wl" / "model" / arm / f"seed{seed}").glob("*/model.pt"))
    return fs[-1].parent if fs else None


def _mlp_fit(X, y, seed, X_eval):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    X, Xe = torch.as_tensor(X, device="cuda").float(), torch.as_tensor(X_eval, device="cuda").float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-4)
    X, Xe = (X - mu) / sd, (Xe - mu) / sd
    y = torch.as_tensor(y, device="cuda").float()
    net = nn.Sequential(nn.Linear(X.shape[1], MLP["hidden"]), nn.GELU(), nn.Dropout(MLP["drop"]),
                        nn.Linear(MLP["hidden"], MLP["hidden"]), nn.GELU(), nn.Dropout(MLP["drop"]),
                        nn.Linear(MLP["hidden"], 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=MLP["lr"], weight_decay=MLP["wd"])
    g = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(MLP["steps"]):
        net.train()
        i = torch.randint(len(X), (min(MLP["batch"], len(X)),), device="cuda", generator=g)
        loss = nn.functional.binary_cross_entropy_with_logits(net(X[i])[:, 0], y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xe)[:, 0]).cpu().numpy()


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def _truth() -> pd.DataFrame:
    """Per fork run: outcome, and the branch's own labels at fork tick + 2 s."""
    runs = pd.read_parquet(WL.rundir("forks.parquet"))
    o = pd.read_parquet(pdir("outcomes.parquet"))
    t = pd.read_parquet(pdir("index.parquet"), columns=["route_id", "tick", "d_front", "occ", "ped"])
    t2 = runs[["route_id", "fork_tick"]].assign(tick=runs.fork_tick + W.FUT * W.TICKS).merge(t, on=["route_id", "tick"], how="left")
    t = runs.merge(o, on="route_id", how="left").merge(t2[["route_id", "d_front", "occ", "ped"]], on="route_id", how="left")
    for c in ("occ", "ped", "unsafe"):
        t[c] = t[c].astype(float)
    return t


def _boot(df, fn, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    b = df.base_id.unique()
    by = {k: g for k, g in df.groupby("base_id")}
    xs = [fn(pd.concat([by[k] for k in rng.choice(b, len(b))])) for _ in range(n)]
    return [float(np.nanpercentile(xs, 2.5)), float(np.nanpercentile(xs, 97.5))]


def _boot_seeds(dfs, fn, n=2000, seed=0):
    """Route bootstrap of the seed-mean of a per-seed statistic (the same route resample for every seed)."""
    rng = np.random.RandomState(seed)
    b = dfs[0].base_id.unique()
    by = [{k: g for k, g in df.groupby("base_id")} for df in dfs]
    xs = []
    for _ in range(n):
        pick = rng.choice(b, len(b))
        xs.append(np.nanmean([fn(pd.concat([m[k] for k in pick])) for m in by]))
    return [float(np.nanpercentile(xs, 2.5)), float(np.nanpercentile(xs, 97.5))]


def _long(arm, seed):
    d = _latest(arm, seed)
    if d is None:
        return None
    q = np.load(d / "preds.npz")
    fa = pd.read_parquet(d / "forks.parquet")
    return q, fa


def report() -> dict:
    tr = _truth()
    key = tr.set_index(["fork_id", "action"])
    WL.RESULTS.mkdir(parents=True, exist_ok=True)
    out, rows, per_seed_rows = {}, [], []
    exit_pairs = WL.pair_dropped_forks()                # amendment (c) item 3: op / hold / brake_hard branch render-dropped
    for arm in ARMS:
        per = []
        for s in SEEDS:
            r = _long(arm, s)
            if r is None:
                continue
            q, fa = r
            d = pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), len(ACTIONS)), "action": np.tile(ACTIONS, len(fa)),
                              "d10": q["p_d_front"][:, :, 9].ravel(), "occ10": q["p_occ"][:, :, 9].ravel(),
                              "occmax": q["p_occ"].max(2).ravel(), "pedmax": q["p_ped"].max(2).ravel(),
                              "dmin": q["p_d_front"].min(2).ravel(), "prog": (q["p_v"] * W.DT).sum(2).ravel(), "seed": s})
            per.append(d)
        if not per:
            continue
        d = pd.concat(per).groupby(["fork_id", "action"]).mean(numeric_only=True).reset_index().drop(columns="seed")
        d = d.join(key, on=["fork_id", "action"], rsuffix="_true")
        out[arm] = d
        # ---- C1 on eval fork points, one frame per seed (the seed-mean of the per-seed statistic is the primary read)
        c1s = []
        for dd in per:
            e = dd.join(key, on=["fork_id", "action"], rsuffix="_true")
            e = e[(e.split == "eval") & ~e.fork_id.isin(exit_pairs)]
            pv = lambda col: e.pivot_table(index="fork_id", columns="action", values=col, dropna=False)
            d10, dtrue, occp, occt = pv("d10"), pv("d_front"), pv("occ10"), pv("occ")
            base = e.groupby("fork_id")[["base_id", "cls", "world"]].first()
            c1 = pd.DataFrame({"brake_gt_hold": d10.brake_hard > d10.hold,
                               "err": ((d10.brake_hard - d10.hold) - (dtrue.brake_hard - dtrue.hold)).abs()}).join(base)
            for s_ in ("shift_L", "shift_R"):
                have = occt[s_].notna() & occt.op.notna()
                c1[f"ok_{s_}"] = ((occp[s_] < occp.op) == (occt[s_] < occt.op)).where(have)
                c1[f"occ_op_{s_}"] = (occt.op > 0.5) & have                       # the description subset: op branch lane occupied at 2 s
            c1["seed"] = int(dd.seed.iloc[0])
            c1s.append(c1)
        stat = {
            "brake_gt_hold": lambda x: x.brake_gt_hold.mean(),
            "median_abs_err_m": lambda x: x.err.median(),
            "shift_occ_agreement": lambda x: pd.concat([x.ok_shift_L, x.ok_shift_R]).mean(),
            "shift_occ_agreement_occupied": lambda x: pd.concat([x.ok_shift_L[x.occ_op_shift_L], x.ok_shift_R[x.occ_op_shift_R]]).mean()}
        row = {"arm": arm, "criterion": "C1", "n_seeds": len(c1s), "n_fork_points": len(c1s[0]), "n_pair_exit": len(exit_pairs)}
        for k, f in stat.items():
            v = [f(c) for c in c1s]
            row[k] = float(np.mean(v))
            row[f"{k}_ci"] = _boot_seeds(c1s, f)
            row[f"{k}_per_seed"] = [float(x) for x in v]
        row["pass"] = bool(row["brake_gt_hold"] >= 0.85 and row["median_abs_err_m"] <= 3.0 and row["shift_occ_agreement"] >= 0.75)
        rows.append(row)
        pd.concat(c1s).to_csv(WL.RESULTS / f"c1_points_{arm}.csv", float_format="%.4f")
    res = pd.DataFrame(rows)
    # ---- C2 / C3 on the main (and holdout) arm
    c2 = {}
    for arm in ("main", "holdout"):
        if arm not in out:
            continue
        preds = {s: _long(arm, s) for s in SEEDS}
        probs_l, probs_q = [], []
        for s, r in preds.items():
            if r is None:
                continue
            q, fa = r
            n, A = q["z0"].shape[0], len(ACTIONS)
            z0 = np.repeat(q["z0"].astype(np.float32)[:, None], A, 1)
            Xl = np.concatenate([z0, q["z5"].astype(np.float32), q["z10"].astype(np.float32)], -1).reshape(n * A, -1)
            Xq = np.concatenate([z0, q["cmds"].reshape(n, A, -1).astype(np.float32)], -1).reshape(n * A, -1)
            lab = pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), A), "action": np.tile(ACTIONS, n)}).join(
                key[["unsafe", "split"]], on=["fork_id", "action"])
            trn = (lab.split == "train").to_numpy() & lab.unsafe.notna().to_numpy()
            if arm == "holdout":
                trn &= ~lab.action.isin(HOLD_OUT).to_numpy()
            y = lab.unsafe.fillna(False).astype(float).to_numpy()
            probs_l.append(_mlp_fit(Xl[trn], y[trn], s, Xl))
            probs_q.append(_mlp_fit(Xq[trn], y[trn], s, Xq))
        if not probs_l:
            continue
        lab = lab.assign(p_learn=np.mean(probs_l, 0), p_q=np.mean(probs_q, 0))
        e = lab[(lab.split == "eval") & lab.unsafe.notna() & ~lab.fork_id.isin(exit_pairs)].join(key[["base_id", "cls", "world", "travel_m"]], on=["fork_id", "action"])
        e = e.merge(out[arm][["fork_id", "action", "occmax", "pedmax", "dmin", "prog"]], on=["fork_id", "action"])
        sig = lambda x: 1 / (1 + np.exp(-x))
        e["p_probe"] = np.maximum(np.maximum(sig(e.occmax), sig(e.pedmax)), (e.dmin < 5).astype(float))
        if arm == "holdout":
            h = e[e.action.isin(HOLD_OUT)]
            diff = lambda x: _auc(x.unsafe, x.p_learn) - _auc(x.unsafe, x.p_q)
            c2["holdout"] = {"n": len(h), "auc_learn": _auc(h.unsafe, h.p_learn), "auc_q": _auc(h.unsafe, h.p_q),
                             "diff": diff(h), "diff_ci": _boot(h, diff), "pass_2b": _boot(h, diff)[0] > 0}
            continue

        def pairwise(x):
            ok, tot = 0, 0
            for _, g in x.groupby("fork_id"):
                u, sfe = g[g.unsafe.astype(bool)].p_learn.to_numpy(), g[~g.unsafe.astype(bool)].p_learn.to_numpy()
                ok += (u[:, None] > sfe[None]).sum()
                tot += len(u) * len(sfe)
            return ok / max(tot, 1)
        diff = lambda x: _auc(x.unsafe, x.p_learn) - _auc(x.unsafe, x.p_q)
        c2["main"] = {"n": len(e), "auc_learn": _auc(e.unsafe, e.p_learn), "auc_learn_ci": _boot(e, lambda x: _auc(x.unsafe, x.p_learn)),
                      "auc_q": _auc(e.unsafe, e.p_q), "auc_probe": _auc(e.unsafe, e.p_probe), "pairwise": pairwise(e),
                      "diff_learn_q": diff(e), "diff_ci": _boot(e, diff)}
        c2["main"]["pass_c2"] = bool(c2["main"]["auc_learn"] >= 0.85 and c2["main"]["pairwise"] >= 0.80)
        c2["main"]["pass_2a"] = bool(c2["main"]["diff_ci"][0] >= -0.02)
        # ---- C3: the selection rule on eval fork points (real branch outcomes)
        sel = []
        for fid, g in e.groupby("fork_id"):
            ok = g[g.p_learn < UNSAFE_P]
            pick = ok.loc[ok.prog.idxmax()] if len(ok) else g.loc[g.p_learn.idxmin()]
            op = g[g.action == "op"]
            orc = g.sort_values(["unsafe", "travel_m"], ascending=[True, False]).iloc[0]
            if not len(op):
                continue
            sel.append({"fork_id": fid, "base_id": g.base_id.iloc[0], "world": g.world.iloc[0], "cls": g.cls.iloc[0],
                        "pick": pick.action, "unsafe_pick": float(pick.unsafe), "unsafe_op": float(op.unsafe.iloc[0]),
                        "unsafe_oracle": float(orc.unsafe), "travel_pick": float(pick.travel_m), "travel_op": float(op.travel_m.iloc[0])})
        sel = pd.DataFrame(sel)
        plus, minus = sel[sel.world == "plus"], sel[sel.world == "minus"]
        red = lambda x: 1 - x.unsafe_pick.mean() / max(x.unsafe_op.mean(), 1e-9)
        c3 = {"n_plus": len(plus), "unsafe_pick_plus": plus.unsafe_pick.mean(), "unsafe_op_plus": plus.unsafe_op.mean(),
              "unsafe_oracle_plus": plus.unsafe_oracle.mean(), "rel_reduction": red(plus),
              "diff_ci": _boot(plus, lambda x: x.unsafe_op.mean() - x.unsafe_pick.mean()),
              "n_minus": len(minus), "travel_ratio_minus": float((minus.travel_pick / minus.travel_op.clip(lower=0.1)).mean()),
              "unsafe_pick_minus": minus.unsafe_pick.mean(), "unsafe_op_minus": minus.unsafe_op.mean(),
              "picks": sel.pick.value_counts().to_dict()}
        c3["pass"] = bool(c3["rel_reduction"] >= 0.5 and c3["diff_ci"][0] > 0 and c3["travel_ratio_minus"] >= 0.9
                          and c3["unsafe_pick_minus"] <= c3["unsafe_op_minus"] + 0.02)
        # checklist amendment (b), description only (C3's registered rules above are unchanged): pedestrian fork points
        # whose hold / op branch hits a non-pedestrian actor in both worlds, listed separately
        npb = WL.nonped_both(tr)
        sel["nonped_both"] = sel.fork_id.isin(npb)
        x = sel[sel.nonped_both]
        c3["nonped_both"] = {"n": len(x), "fork_ids": sorted(int(i) for i in x.fork_id),
                             **{f"{c}_{w}": float(x[x.world == w][c].mean()) if (x.world == w).any() else None
                                for c in ("unsafe_pick", "unsafe_op", "unsafe_oracle") for w in ("plus", "minus")}}
        c2["c3"] = c3
        sel.to_csv(WL.RESULTS / "c3_selection.csv", index=False)
    WL.RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(WL.RESULTS / "c1.csv", index=False, float_format="%.4f")
    (WL.RESULTS / "c2_c3.json").write_text(json.dumps(c2, indent=1, default=float))
    return {"c1": res, "c2_c3": c2}


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("outcomes", "train", "report"))
    ap.add_argument("--arm", default="main", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="smoke run: fewer steps, written under runs/wl/model_smoke")
    a = ap.parse_args()
    if a.step == "train":
        rl = RunLog("wl", "model_smoke" if a.steps else "model", a.arm, f"seed{a.seed}")
        r = train(a.arm, a.seed, rl, a.steps or None)
        rl.info(json.dumps(r))
        rl.close()
    elif a.step == "outcomes":
        r = outcomes()
    else:
        r = report()
        print(r["c1"].to_string(index=False))
        r = r["c2_c3"]
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
