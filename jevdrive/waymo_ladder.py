"""P2 and P3: the readout ladder and the backbone ladder, on one frozen stratified subset of Waymo val.

Decision 20 judged branch 2 -- a linear readout of the frozen Qwen3-VL-4B features is exhausted on the
pre-onset subset, so the next move is at the representation. That judgement was made with one input shape
(a single frame, pooled to one 2560-vector) and one head (ridge). P2 loosens the input and the head while
keeping the backbone; P3 changes the backbone while keeping everything else. Both are measured the same way,
so the two ladders share this module:

  the subset   one stratified selection of val frames, chosen once by `choose_subset` and frozen in a file.
               Every pre-onset, rater-scored and turning frame is kept -- those carry the judgement -- and
               the rest of val is subsampled by sequence down to ~20k frames, because re-extracting four
               backbones over all 106 360 is days of GPU, not hours.
  the split    decision 3d's sequence-disjoint halves of val, seed 0, unchanged, so arm A can be recomputed
               on exactly these frames and every comparison is like-for-like.
  the head     ridge_late over ego + feature for the linear arms, and for P2's token heads the same
               late-fusion form with attention pooling in front of it: every arm predicts the residual the
               ego ridge leaves, and every arm's lambda / epoch count is chosen inside the fit half alone.
  the judge    pre-onset paired dADE with a scene-bootstrap CI in both directions, the straight-subset dADE
               beside it, the DiD, the s_ego decile relative-gain curve and RFS on the rater frames.

s_ego, the kinematic surprise that the decile curve stratifies on, is computed on the *whole* half-val as
decision 20 defines it and then restricted to the subset, so the decile axis means the same thing as in that
entry. Each arm's own ego base is fitted on the subset's fit half instead, because the arms are paired
against it frame by frame and it must see the same rows they do.

Every arm's per-frame predictions are written to the run directory, so P1's judge can be applied later
without refitting anything.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from . import planner, traj, waymo, waymo_l0 as l0, waymo_stage_a as sa
from .common import data_dir, get_logger

log = get_logger(__name__)
DEV = "cuda"
LAYER = sa.LAYER                    # L18_mean, the layer decision 5 and stage A settled on
SUBSETS = sa.SUBSETS
BASE = "ridge ego"
SUBSET_FILE = "p2p3_v1"
TARGET_N = 20_000
QUOTA = {"turn_yaw": 4_000, "straight_yaw": 10_000}   # capped strata; pre-onset and rater frames are kept whole


# ---------------------------------------------------------------- context and subset

def base_context(seed: int = 0, feature_set: str = "qwen_front3", layer: str = LAYER) -> dict:
    """Everything the ladder needs about val, on the rows that have a future and a cached pooled feature.

    This is `waymo_stage_a.load_all` plus the pieces every arm shares: the frame names (the join key for any
    other feature set), the half assignment, the rater rows and the sequence codes.
    """
    df, rows, fut, ego, img, sub, past = sa.load_all(feature_set, layer, seed)
    halves = sa.val_halves(df, seed)
    fname = waymo.frame_names(df)[rows]
    rated = np.zeros(len(df), bool)
    rated[waymo.load_rater(df)[0]] = True
    ctx = {"df": df, "rows": rows, "fname": fname, "fut": fut, "ego": ego, "pooled": np.asarray(img),
           "sub": sub, "past": past, "seq": df.sequence.to_numpy()[rows],
           "half": df.sequence.map(halves).to_numpy()[rows], "rater": rated[rows], "seed": seed}
    log.info("context: %d rows, %d sequences, %d rater frames, subsets %s", len(rows),
             len(np.unique(ctx["seq"])), int(ctx["rater"].sum()), {k: int(sub[k].sum()) for k in SUBSETS})
    return ctx


def s_ego_full(ctx: dict, direction: int, folds: int = l0.FOLDS) -> np.ndarray:
    """Decision 20's s_ego over every context row: out-of-fold on the fit half, fit-half model on the other.

    Computed on the whole half-val and not inside the subset on purpose. The decile axis is meant to be the
    same axis entry 20 reports, and a prior refitted on 20k rows is a different prior from the one that
    produced those deciles.
    """
    sp = sa.Halves(ctx["df"], ctx["seq"], ctx["half"] == direction, ctx["half"] == (1 - direction), ctx["seed"])
    return l0.ego_surprise(ctx["ego"], ctx["fut"], sp, folds)


def _take(seq: np.ndarray, pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    """`n` rows drawn from `pool` (positions), the same share out of every sequence.

    Sampling per sequence rather than globally is not cosmetic: the bootstrap resamples sequences and RFS
    averages per scenario cluster, so a draw that empties some sequences costs resolution in both.
    """
    if n >= len(pool):
        return pool
    codes = pd.factorize(seq[pool])[0]
    order = np.lexsort((np.random.default_rng(seed).random(len(pool)), codes))   # random within a sequence
    start = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    rank = np.empty(len(pool), np.int64)
    rank[order] = np.arange(len(pool)) - np.repeat(start, np.diff(np.r_[start, len(pool)]))
    quota = np.ceil(np.bincount(codes) * (n / len(pool))).astype(int)[codes]
    return pool[rank < quota]


def choose_subset(ctx: dict, target: int = TARGET_N, quota: dict = QUOTA, seed: int = 0) -> np.ndarray:
    """Boolean mask over the context rows: the frames P2(c) and all of P3 re-extract features for.

    Kept whole: every pre-onset frame and every rater-scored frame. Those two carry the judgement -- the
    pre-onset paired delta and RFS -- so the subset must not cost them a single frame, and it does not: all
    1510 and all 479 are in.

    Everything else is capped, and the caps are set by what each stratum is *for*. `straight_yaw` is the DiD's
    control arm, and in the full half-val its 23 612 evaluation frames contributed almost no variance; cutting
    it to a few thousand would widen the DiD until it said nothing, so it gets the larger quota. `turn_yaw` is
    only decision 3c's already-turning control and needs enough frames to be read, not all 11 060 of them.
    The remainder fills up to `target` from the frames in no yaw subset, which is what keeps the overall ADE
    column representative of val rather than of its manoeuvres.
    """
    keep = ctx["rater"] | ctx["sub"]["pre_onset"]
    for i, (k, n) in enumerate(quota.items()):
        keep[_take(ctx["seq"], np.flatnonzero(ctx["sub"][k] & ~keep), n, seed + i)] = True
    rest = np.flatnonzero(~keep)
    keep[_take(ctx["seq"], rest, max(target - int(keep.sum()), 0), seed + len(quota))] = True
    log.info("subset: %d frames (%d sequences) of %d; %s", int(keep.sum()),
             len(np.unique(ctx["seq"][keep])), len(keep),
             {k: int((ctx["sub"][k] & keep).sum()) for k in SUBSETS} | {"rater": int((ctx["rater"] & keep).sum())})
    return keep


def subset_path(name: str = SUBSET_FILE):
    p = data_dir() / "processed" / "waymo_e2e" / "subsets"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


def write_subset(ctx: dict, keep: np.ndarray, name: str = SUBSET_FILE) -> dict:
    """Freeze the selection: the frame names, their global index rows and their half, plus a summary whose
    sha256 is what the todo and decisions entry quote. Nothing downstream re-derives the selection."""
    t = pd.DataFrame({"frame_name": ctx["fname"][keep], "row": ctx["rows"][keep],
                      "sequence": ctx["seq"][keep], "half": ctx["half"][keep],
                      "rater": ctx["rater"][keep], **{k: ctx["sub"][k][keep] for k in SUBSETS}})
    t = t.sort_values("frame_name").reset_index(drop=True)
    t.to_parquet(subset_path(name), index=False)
    digest = hashlib.sha256("\n".join(t.frame_name).encode()).hexdigest()
    summary = {"name": name, "frames": len(t), "sequences": int(t.sequence.nunique()),
               "sha256_frame_names": digest,
               **{k: int(t[k].sum()) for k in ("rater", *SUBSETS) if k != "all"},
               **{f"half{i}_frames": int((t.half == i).sum()) for i in (0, 1)},
               **{f"half{i}_sequences": int(t[t.half == i].sequence.nunique()) for i in (0, 1)}}
    log.info("subset frozen at %s: %s", subset_path(name), summary)
    return summary


def load_subset(ctx: dict, name: str = SUBSET_FILE) -> np.ndarray:
    """The frozen subset as a mask over the current context rows, checked by frame name, never by row id."""
    t = pd.read_parquet(subset_path(name))
    keep = np.isin(ctx["fname"], t.frame_name.to_numpy())
    if int(keep.sum()) != len(t):
        raise RuntimeError(f"{name}: {len(t)} frames in the file but {int(keep.sum())} found in the index")
    return keep


class Aligned:
    """Another feature set's array addressed by *context row*, without materialising the full length.

    The obvious implementation -- scatter the set's rows into a zero array as long as the context -- costs
    106 360 x 368 640 x 2 bytes for the token grid, which is 78 GB of RAM for 20 237 rows of real data. This
    keeps the compact array on disk as a memmap and carries the row map, so `X[sel]` reads exactly the rows
    the split asks for and nothing else. Rows the set does not cover map to -1 and must be excluded by the
    caller through `covered` before any arm sees them.
    """

    def __init__(self, arr, pos: np.ndarray):
        self.arr, self.pos = arr, pos
        self.shape = (len(pos), arr.shape[1])

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, sel):
        return np.asarray(self.arr[self.pos[sel]])


class Concat:
    """Several feature matrices side by side, addressed by context row. `X[sel]` concatenates the slices.

    Used by the three-camera V-JEPA arm (one matrix per camera) and by the late-fusion arm (V-JEPA beside
    Qwen). Concatenation, not averaging, is deliberate for the cameras: pre-onset is about *which way* the
    car is going to turn, so the left and right views carry the signal itself and averaging them destroys it.
    """

    def __init__(self, parts):
        self.parts = list(parts)
        self.shape = (len(self.parts[0]), sum(p.shape[1] for p in self.parts))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sel):
        return np.concatenate([np.asarray(p[sel], np.float32) for p in self.parts], 1)


def align(ctx: dict, set_name: str, arrays: list[str] | None = None, flat: bool = True) -> dict:
    """Another feature set's arrays, addressed by the context rows. `covered` says which rows it has."""
    idx, arrs = (waymo.load_flat_features if flat else waymo.load_features)(set_name, arrays)
    arrays = arrays or sorted(arrs)
    at = pd.Series(np.arange(len(idx)), index=idx.frame_name.to_numpy())
    pos = at.reindex(ctx["fname"]).to_numpy()
    covered = ~np.isnan(pos)
    ipos = np.where(covered, np.nan_to_num(pos, nan=0), 0).astype(np.int64)
    out = {"covered": covered} | {a: Aligned(arrs[a], ipos) for a in arrays}
    log.info("%s: %d/%d context rows covered, arrays %s", set_name, int(covered.sum()), len(covered),
             {a: arrs[a].shape[1] for a in arrays})
    return out


# ---------------------------------------------------------------- heads

def standardize_np(X: np.ndarray, rows: np.ndarray) -> torch.Tensor:
    """float32 on the GPU, standardised with the fit rows' statistics only (planner.standardize's rule)."""
    return planner.standardize(torch.from_numpy(np.ascontiguousarray(X)).to(DEV).float(), rows)


def ridge_arm(X: np.ndarray):
    """The linear arm: ridge_late on `X`, lambda by the same grouped CV every other ridge in the project uses.

    `X` is indexed by the whole context; `sel` cuts it down to the rows this run fits and evaluates on, which
    is what `sp` indexes. Slicing here rather than at the call site keeps every arm's feature matrix defined
    once, over all of val, however the ladder restricts the rows.
    """
    def fit(sel, sp, R, res_fut, seed, pre=None):
        Xi = standardize_np(X[sel], sp.train)
        p, st, _ = sa.ridge_cv(Xi, R, sp, res_fut)
        del Xi
        torch.cuda.empty_cache()
        return p[:, 0], {"d": X.shape[1], **st}
    return fit


class AttnPool(torch.nn.Module):
    """One learned query attends over the token grid; the pooled vector goes through one linear map.

    This is the smallest head that can weight *where* it looks, which is the whole point of P2(c): the pooled
    baseline is the special case where the attention is uniform, so anything this buys is bought by the
    pooling and not by the capacity.
    """

    def __init__(self, d: int, out: int, dk: int = 256, d_side: int = 0, drop: float = planner.DROPOUT):
        super().__init__()
        self.proj, self.q = torch.nn.Linear(d, dk), torch.nn.Parameter(torch.randn(dk) * 0.02)
        self.drop, self.out = torch.nn.Dropout(drop), torch.nn.Linear(dk + d_side, out)
        self.dk = dk

    def forward(self, x, side=None):                         # x (b, N, d), side (b, d_side) or None
        k = self.proj(x)
        a = (k @ self.q / self.dk ** 0.5).softmax(-1)
        z = self.drop((a.unsqueeze(-1) * k).sum(1))
        return self.out(z if side is None else torch.cat([z, side], 1))


class TokenTransformer(torch.nn.Module):
    """A two-layer encoder over the token grid with the ego state as one extra token and a CLS read-out.

    The ego state enters the head as well as the base it corrects, which is deliberate: the question is
    whether the tokens become useful once something can condition on the ego state while looking at them.
    """

    def __init__(self, d: int, d_ego: int, out: int, dm: int = 256, layers: int = 2, heads: int = 4,
                 drop: float = planner.DROPOUT):
        super().__init__()
        self.tok, self.ego = torch.nn.Linear(d, dm), torch.nn.Linear(d_ego, dm)
        self.cls = torch.nn.Parameter(torch.randn(dm) * 0.02)
        enc = torch.nn.TransformerEncoderLayer(dm, heads, 4 * dm, drop, batch_first=True, norm_first=True)
        self.enc, self.out = torch.nn.TransformerEncoder(enc, layers), torch.nn.Linear(dm, out)

    def forward(self, x, e):
        h = torch.cat([self.cls.expand(len(x), 1, -1), self.ego(e).unsqueeze(1), self.tok(x)], 1)
        return self.out(self.enc(h)[:, 0])


def _pred(o):
    return o[0] if isinstance(o, tuple) else o


def _train(make_net, forward, sp, R, res_fut, seed: int, epochs: int = planner.MLP_EPOCHS,
           pre: np.ndarray | None = None):
    """planner.Heads.mlp's recipe for any net: early-stop on the sequence-grouped inner split of the fit half,
    then refit on the whole fit half for that many epochs. Nothing here sees the evaluation half.

    `pre` (a pre-onset mask over all rows) adds `sel_pre_ade`: the pre-onset ADE on the inner split at the
    epoch early stopping chose. It selects nothing here -- the stopping rule is the overall inner-val ADE,
    as for every other head -- but arm (e) uses it to pick its L1 strength, which is the quantity decision 20
    pre-registered for choosing among L0's weighting schemes.
    """
    T = res_fut.shape[1]
    pm = None if pre is None else pre[sp.sel]

    def run(rows, n_ep, sel):
        torch.manual_seed(seed)
        net = make_net().to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=planner.MLP_LR, weight_decay=planner.MLP_WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
        r = torch.as_tensor(rows, device=DEV)
        best = (np.inf, n_ep, float("nan"))
        for ep in range(n_ep):
            net.train()
            for b in r[torch.randperm(len(r), device=DEV)].split(planner.MLP_BS):
                opt.zero_grad(set_to_none=True)
                o = forward(net, b)
                p_, pen = o if isinstance(o, tuple) else (o, 0.0)
                ((p_ - R[b]).pow(2).mean() + pen).backward()
                opt.step()
            sched.step()
            if sel is not None:
                net.eval()
                with torch.inference_mode():
                    p = torch.cat([_pred(forward(net, c)) for c in torch.as_tensor(sel, device=DEV).split(2048)])
                d = np.linalg.norm(p.reshape(-1, T, 2).float().cpu().numpy() - res_fut[sel], axis=-1).mean(1)
                best = min(best, (float(d.mean()), ep + 1,
                                  float(d[pm].mean()) if pm is not None and pm.any() else float("nan")))
        return net, best

    _, (s, ep, s_pre) = run(sp.fit, epochs, sp.sel)
    net, _ = run(sp.train, ep, None)
    net.eval()
    out, gates = [], []
    with torch.inference_mode():
        for c in torch.as_tensor(sp.val, device=DEV).split(2048):
            out.append(forward(net, c))
            if hasattr(net, "last_gate"):     # arm (e): the gate is a reported quantity, not a detail
                gates.append(net.last_gate)
    p = torch.cat([_pred(o) for o in out])
    st = {"epochs": ep, "sel_ade": s, "sel_pre_ade": s_pre,
          "params": sum(q.numel() for q in net.parameters())}
    if gates:
        st["gate"] = torch.cat(gates).float().cpu().numpy()
    del net
    torch.cuda.empty_cache()
    return p.reshape(-1, T, 2).float().cpu().numpy(), st


def _chan_stats(X: torch.Tensor, rows: np.ndarray, chunk: int = 1024):
    """Per-channel mean and std over (fit rows, tokens), accumulated in float64 a chunk at a time."""
    r = torch.as_tensor(rows, device=X.device)
    s1 = torch.zeros(X.shape[-1], device=X.device, dtype=torch.float64)
    s2, n = s1.clone(), 0
    for b in r.split(chunk):
        x = X[b].double()
        s1 += x.sum((0, 1))
        s2 += x.square().sum((0, 1))
        n += x.shape[0] * x.shape[1]
    mu = s1 / n
    return mu.float(), (s2 / n - mu.square()).clamp_min(1e-12).sqrt().float().clamp_min(1e-6)


def grid_arm(G: np.ndarray, n_tok: int, kind: str = "attn", side: np.ndarray | None = None):
    """P2(c) and (d): a head that sees the token grid. `G` is (n, n_tok * d) float16 as it is on disk.

    `side` is a second input the head also sees: the ego state for the transformer arm (c-tf), and the
    temporal pooled concatenation for arm (d), which is exactly "(b) and (c) together".
    """
    d = G.shape[1] // n_tok

    def fit(sel, sp, R, res_fut, seed, pre=None):
        X = torch.from_numpy(np.ascontiguousarray(G[sel])).to(DEV).view(-1, n_tok, d)   # (n_sel, N, d) fp16
        mu, sd = _chan_stats(X, sp.train)                     # chunked: a float32 copy of the fit half is 15 GB
        S = standardize_np(side[sel], sp.train) if side is not None else None
        ds = 0 if S is None else S.shape[1]
        if kind == "attn":
            make = lambda: AttnPool(d, R.shape[1], d_side=ds)                                 # noqa: E731
            fwd = lambda net, b: net((X[b].float() - mu) / sd, None if S is None else S[b])   # noqa: E731
        else:
            make = lambda: TokenTransformer(d, ds, R.shape[1])                                # noqa: E731
            fwd = lambda net, b: net((X[b].float() - mu) / sd, S[b])                          # noqa: E731
        p, st = _train(make, fwd, sp, R, res_fut, seed)
        del X, S
        torch.cuda.empty_cache()
        return p, {"d": d, "tokens": n_tok, "head": kind, "d_side": ds, **st}
    return fit


class GatedResidual(torch.nn.Module):
    """Arm (e): output = g(x) * delta(x), with g a scalar gate in [0, 1] and an L1 penalty on it.

    The whole head predicts the correction to the ego prior, exactly as every other arm does, but it is
    forced to factor that correction into "how much to react" and "what the reaction is". Decision 25 wants
    to know whether real data alone teaches the gate *where* to open; the L1 term is what makes the question
    meaningful, because without it the gate can sit at 1 everywhere and the arm is just the ungated head.
    """

    def __init__(self, kind: str, d: int, out: int, l1: float, dk: int = 256, drop: float = planner.DROPOUT):
        super().__init__()
        self.kind, self.l1, self.dk = kind, l1, dk
        if kind == "attn":
            self.proj = torch.nn.Linear(d, dk)
            self.q = torch.nn.Parameter(torch.randn(dk) * 0.02)
        else:
            self.proj = torch.nn.Sequential(torch.nn.Linear(d, planner.HIDDEN), torch.nn.GELU(),
                                            torch.nn.Linear(planner.HIDDEN, dk))
        self.drop = torch.nn.Dropout(drop)
        self.delta, self.gate = torch.nn.Linear(dk, out), torch.nn.Linear(dk, 1)

    def encode(self, x):
        if self.kind != "attn":
            return self.drop(self.proj(x))
        k = self.proj(x)
        a = (k @ self.q / self.dk ** 0.5).softmax(-1)
        return self.drop((a.unsqueeze(-1) * k).sum(1))

    def forward(self, x):
        z = self.encode(x)
        g = torch.sigmoid(self.gate(z))
        self.last_gate = g.detach().squeeze(-1)    # read back by _train; the caller holds the features
        return g * self.delta(z), self.l1 * g.mean()


def gated_arm(X: np.ndarray, kind: str = "mlp", n_tok: int | None = None, l1s=(0.0, 1e-3, 1e-2)):
    """Arm (e), with the L1 strength chosen inside the fit half on pre-onset ADE and never on the eval half.

    The selection quantity is the one decision 20 pre-registered for choosing among L0's B schemes: the
    unweighted pre-onset ADE over the fit half's own grouped folds. Here it is approximated by the inner
    split `sp.sel` restricted to its pre-onset frames, which is the same discipline at a tenth of the cost.
    """
    d = X.shape[1] // n_tok if n_tok else X.shape[1]

    def fit(sel, sp, R, res_fut, seed, pre=None):
        # The token grid stays float16 on the card and is cast one batch at a time: a float32 copy of all
        # 20 237 x 144 x 2560 is 29.8 GB, which is what OOM'd this arm the first time round while the 32B
        # extraction held 64 GB.
        Z = torch.from_numpy(np.ascontiguousarray(X[sel])).to(DEV)
        if n_tok:
            Z = Z.view(-1, n_tok, d)
            mu, sd = _chan_stats(Z, sp.train)
            take = lambda b: (Z[b].float() - mu) / sd        # noqa: E731
        else:
            Z = planner.standardize(Z.float(), sp.train)
            take = lambda b: Z[b]                            # noqa: E731
        best, out = None, None
        for l1 in l1s:
            p, st = _train(lambda: GatedResidual(kind, d, R.shape[1], l1), lambda net, b: net(take(b)),
                           sp, R, res_fut, seed, pre=pre)
            score = st["sel_pre_ade"] if np.isfinite(st["sel_pre_ade"]) else st["sel_ade"]
            log.info("  (e) %s l1=%.0e: inner-val pre-onset ADE %.4f (overall %.4f), mean gate %.3f",
                     kind, l1, score, st["sel_ade"], float(st["gate"].mean()))
            if best is None or score < best:
                best, out = score, (p, {"l1": l1, "head": f"gated-{kind}", "d": d, **st})
        del Z
        torch.cuda.empty_cache()
        return out
    return fit


def mlp_arm(X: np.ndarray):
    """The compute-matched control: planner's MLP on the pooled vector. Same optimiser, same schedule, same
    early stopping as the token heads, so a win for those is a win for attention and not for capacity."""
    def fit(sel, sp, R, res_fut, seed, pre=None):
        Xi = standardize_np(X[sel], sp.train)
        p, st = _train(lambda: planner._mlp(Xi.shape[1], R.shape[1]), lambda net, b: net(Xi[b]),
                       sp, R, res_fut, seed)
        del Xi
        torch.cuda.empty_cache()
        return p, {"d": X.shape[1], "head": "mlp", **st}
    return fit


# ---------------------------------------------------------------- running and judging

def run_direction(ctx: dict, keep: np.ndarray, arms: dict, direction: int, s_full: np.ndarray,
                  rl=None) -> tuple[dict, dict, pd.DataFrame]:
    """Fit the ego base and every arm on one direction's fit half, restricted to `keep`.

    `keep` is a mask over the context rows; every arm is fitted and evaluated on exactly those rows, which is
    what makes the recomputed arm A comparable with the new arms frame by frame.
    """
    sel = np.flatnonzero(keep)
    df, seq = ctx["df"], ctx["seq"][sel]
    h = ctx["half"][sel]
    fut, ego = ctx["fut"][sel], ctx["ego"][sel]
    sp = sa.Halves(df, seq, h == direction, h == (1 - direction), ctx["seed"])
    n, T = len(sel), fut.shape[1]
    log.info("direction %d: fit %d frames / %d sequences, eval %d / %d", direction, len(sp.train),
             len(np.unique(seq[sp.train])), len(sp.val), len(np.unique(seq[sp.val])))

    F = torch.as_tensor(fut.reshape(n, -1), device=DEV)
    Xe = planner.standardize(torch.as_tensor(ego, device=DEV), sp.train)
    p_ego, st_ego, W_ego = sa.ridge_cv(Xe, F, sp, fut)
    base = planner.linear_apply(W_ego, Xe, np.arange(n))[0]
    R = F - base
    res_fut = R.reshape(-1, T, 2).cpu().numpy()
    off = base[sp.val].reshape(-1, T, 2).cpu().numpy()

    preds, stats, gates = {BASE: p_ego[:, 0]}, {BASE: st_ego}, {}
    for name, fit in arms.items():
        p, st = fit(sel, sp, R, res_fut, ctx["seed"], ctx["sub"]["pre_onset"][sel])
        gate = st.pop("gate", None)
        if gate is not None:
            gates[name] = gate
        preds[name], stats[name] = p + off, st
        log.info("dir %d %-28s fitted: %s", direction, name, {k: round(v, 5) if isinstance(v, float) else v
                                                              for k, v in st.items()})
        if rl is not None:
            rl.event("arm_fit", direction=direction, arm=name, **st)
    sctx = {"sel": sel, "sp": sp, "seq": seq, "fut": fut, "direction": direction, "gates": gates,
            "sub": {k: ctx["sub"][k][sel] for k in SUBSETS}, "rows": ctx["rows"][sel],
            "fname": ctx["fname"][sel], "s": s_full[sel], "df": df, "past": ctx["past"]}
    return preds, sctx, pd.DataFrame([{"direction": direction, "arm": k, **v} for k, v in stats.items()])


def judge(preds: dict, sctx: dict) -> tuple[pd.DataFrame, ...]:
    """The pre-registered reading: per-arm ADE and RFS, the paired delta by subset, the DiD, and the decile
    curve. Identical in form to `waymo_l0.evaluate`, so the numbers line up column by column with entry 20."""
    sp, sub, seq, fut = sctx["sp"], sctx["sub"], sctx["seq"], sctx["fut"]
    v, gt, sq = sp.val, fut[sp.val], seq[sp.val]
    masks = {k: sub[k][v] for k in SUBSETS}
    pos, rtraj, scores = sa.rfs_rows(sctx["df"], sctx["rows"][v])
    speed = waymo.init_speed(sctx["past"][sctx["rows"][v]])
    cluster = sctx["df"].cluster.astype(str).to_numpy()[sctx["rows"][v]]
    per, rows_out = {}, []
    for name, p in preds.items():
        e = np.linalg.norm(p - gt, axis=-1).mean(1)
        per[name] = e
        r = {"arm": name, "direction": sctx["direction"]}
        for sn, m in masks.items():
            r[f"ade_{sn}"], r[f"n_{sn}"] = e[m].mean(), int(m.sum())
        if len(pos):
            sc = waymo.rater_feedback_score(p[pos], rtraj, scores, speed[pos])
            r["rfs"], r["n_rater"] = sc.mean(), len(pos)
            r["rfs_cluster"] = waymo.rfs_by_cluster(sc, cluster[pos])[0]
            r["rfs_lo"], r["rfs_hi"] = traj.boot_ci(sc, sq[pos])
            r["floored"] = float((sc <= waymo.RFS_FLOOR + 1e-9).mean())
        rows_out.append(r)
        log.info("dir %d %-28s RFS %.3f  ADE %.3f | pre_onset %.3f (n=%d)  straight %.3f (n=%d)",
                 sctx["direction"], name, r.get("rfs", np.nan), r["ade_all"], r["ade_pre_onset"],
                 r["n_pre_onset"], r["ade_straight_yaw"], r["n_straight_yaw"])

    pairs, dids = [], []
    for name in preds:
        if name == BASE:
            continue
        d = per[name] - per[BASE]
        for sn, m in masks.items():
            lo, hi = traj.boot_ci(d[m], sq[m])
            pairs.append({"direction": sctx["direction"], "arm": name, "subset": sn, "n": int(m.sum()),
                          "ade_base": per[BASE][m].mean(), "ade_arm": per[name][m].mean(),
                          "dade": d[m].mean(), "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2,
                          "meets_threshold": bool(d[m].mean() <= -0.05 and hi < 0)})
        hi_m, lo_m = masks["pre_onset"], masks["straight_yaw"]
        point, cl, ch = traj.boot_did(d, sq, hi_m, lo_m)
        dids.append({"direction": sctx["direction"], "arm": name, "n_hi": int(hi_m.sum()),
                     "n_lo": int(lo_m.sum()), "dade_pre_onset": d[hi_m].mean(),
                     "dade_straight": d[lo_m].mean(), "did": point, "did_lo": cl, "did_hi": ch,
                     "halfwidth": (ch - cl) / 2})
        log.info("dir %d %-28s pre_onset %+.4f [%+.4f, %+.4f]  straight %+.4f  DiD %+.4f",
                 sctx["direction"], name, d[hi_m].mean(), *traj.boot_ci(d[hi_m], sq[hi_m]),
                 d[lo_m].mean(), point)
    return pd.DataFrame(rows_out), pd.DataFrame(pairs), pd.DataFrame(dids), deciles(per, sctx)


def deciles(per: dict, sctx: dict, scopes=("all", "straight_yaw")) -> pd.DataFrame:
    """Relative gain by the evaluation half's own s_ego decile, exactly as `waymo_l0.decile_report` does it.

    Read by shape, not by any single bin: does the relative gain stop falling in the top deciles. Bin 10 does
    not decide anything -- entry 20 showed with the raters that the logged future is itself contested there.
    """
    sp, seq = sctx["sp"], sctx["seq"]
    v, sq, sv = sp.val, seq[sp.val], sctx["s"][sp.val]
    q = np.quantile(sv, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q[1:-1], sv, "right"), 0, 9)
    rows = []
    for scope in scopes:
        sm = sctx["sub"][scope][v]
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
                rows.append({"direction": sctx["direction"], "arm": name, "scope": scope, "decile": i + 1,
                             "n": int(m.sum()), "s_ego_lo": float(sv[m].min()), "s_ego_hi": float(sv[m].max()),
                             "ade_base": b, "ade_arm": e[m].mean(), "dade": d[m].mean(),
                             "rel_gain": 100 * d[m].mean() / b, "rel_lo": 100 * lo / b, "rel_hi": 100 * hi / b})
    return pd.DataFrame(rows)


def gate_report(sctx: dict) -> pd.DataFrame:
    """Arm (e)'s gate: mean g and the fraction above 0.5, by subset and by s_ego decile.

    This is the table decision 25 asks for. The question is not whether the gate is open on average but
    whether it is open *preferentially* where the ego prior is about to be wrong -- pre-onset frames and the
    top s_ego deciles -- against straight frames, when nothing but real data trained it.
    """
    sp, sv = sctx["sp"], sctx["s"][sctx["sp"].val]
    q = np.quantile(sv, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q[1:-1], sv, "right"), 0, 9)
    rows = []
    for name, g in sctx["gates"].items():
        groups = {k: sctx["sub"][k][sp.val] for k in SUBSETS}
        groups |= {f"s_ego decile {i + 1}": dec == i for i in range(10)}
        for k, m in groups.items():
            if m.any():
                rows.append({"direction": sctx["direction"], "arm": name, "group": k, "n": int(m.sum()),
                             "gate_mean": float(g[m].mean()), "gate_open": float((g[m] > 0.5).mean())})
    return pd.DataFrame(rows)


def save_preds(rl, preds: dict, sctx: dict, tag: str = ""):
    """Per-frame predictions of every arm, keyed by frame name, so P1's judge can be applied without refitting."""
    np.savez_compressed(rl.dir / f"{tag}_preds_dir{sctx['direction']}.npz",
                        frame_name=sctx["fname"][sctx["sp"].val].astype(str),
                        s_ego=sctx["s"][sctx["sp"].val].astype(np.float32),
                        gt=sctx["fut"][sctx["sp"].val].astype(np.float32),
                        **{k: v.astype(np.float32) for k, v in preds.items()})


def write_tables(rl, tables: dict[str, list], tag: str = ""):
    out = {}
    for name, parts in tables.items():
        parts = [q for q in parts if len(q)]
        if not parts:
            continue
        t = pd.concat(parts, ignore_index=True)
        t.to_csv(rl.dir / f"{tag}_{name}.csv", index=False)
        rl.log.info("%s %s\n%s", tag, name, t.to_markdown(index=False, floatfmt=".4f"))
        rl.event(name, tag=tag, rows=t.to_dict("records"))
        out[name] = t
    return out


def run_ladder(arms_for: callable, tag: str, keep: np.ndarray | None, ctx: dict, directions=(0, 1), rl=None):
    """Both directions of one ladder. `arms_for(keep_mask)` builds the arm dict once the rows are known."""
    tables = {k: [] for k in ("arms", "paired", "did", "deciles", "fits", "gates")}
    for d in directions:
        s = s_ego_full(ctx, d)
        k = np.ones(len(ctx["fname"]), bool) if keep is None else keep
        preds, sctx, fits = run_direction(ctx, k, arms_for(k), d, s, rl)
        a, p, dd, dec = judge(preds, sctx)
        for name, t in (("arms", a), ("paired", p), ("did", dd), ("deciles", dec), ("fits", fits),
                        ("gates", gate_report(sctx))):
            tables[name].append(t)
        if rl is not None:
            save_preds(rl, preds, sctx, tag)
        del preds
        torch.cuda.empty_cache()
    if rl is None:
        return tables
    out = write_tables(rl, tables, tag)
    if "deciles" in out and len(out["deciles"]):
        from . import plots
        f = plots.ladder_decile_curve(out["deciles"], rl.dir, f"{tag}-decile-relative-gain")
        rl.log.info("figure: %s", f)
    return out


# ---------------------------------------------------------------- P2(b): temporal pooled input

def temporal_arm(ctx: dict, stride: int, n_back: int = 3, layer: str = LAYER):
    """(b): the pooled vector of the current frame concatenated with the same vector `n_back` strides back.

    Waymo's frame index steps 0.1 s (decision 14), so 0.25 / 0.5 / 0.75 s have no integer frame and are not
    interpolated -- interpolating would invent a feature that is not on disk. Two even strides bracket that
    range instead: 2 frames (0.2 s) and 3 frames (0.3 s).

    Returns (feature matrix over the context rows, complete-window mask). Decision 13: a window counts only
    when every slot is an exact hit, and the *whole* comparison table is restricted to those frames, arm A
    included, or the rows are not looking at the same data.
    """
    df, rows = ctx["df"], ctx["rows"]
    hist, _, got = waymo.history_rows(df, n_back, stride, targets=rows)
    full = got.all(1)
    at = pd.Series(np.arange(len(rows)), index=rows)
    src = at.reindex(hist.ravel()).to_numpy().reshape(hist.shape)     # history rows that are context rows
    ok = full & ~np.isnan(src).any(1)
    src = np.where(np.isnan(src), 0, src).astype(int)
    P = ctx["pooled"]
    X = np.concatenate([P[src[:, k]] for k in range(n_back + 1)], 1)
    log.info("temporal stride %d frames (%.1f s), %d slots: %d/%d rows have a complete window",
             stride, stride * waymo.FRAME_DT, n_back + 1, int(ok.sum()), len(ok))
    return X, ok


# ---------------------------------------------------------------- the two ladders

GRID_SET = "qwen_grid_p2"            # P2(c): the token-grid extraction over the frozen subset
GRID_TOKENS = 3 * 6 * 8              # three cameras, each token map pooled to 6 x 8
GRID_ARRAY = LAYER.split("_")[0] + "_grid"
P3_SETS = {                          # arm -> (feature set on disk, the arrays to read)
    "a qwen32b": ("qwen32b_front3_p3", ["L32_mean", "L32_last", "L50_mean", "L50_last"]),
    "b h3": ("h3_dit_p3", None),
    "c wan": ("wan22_dit_p3", None),   # arrays named by noise level and block, discovered on disk
    "d vjepa2": ("vjepa2_p3", ["mean", "last_mean"]),
}


def p2b(ctx: dict, rl, strides=(2, 3), n_back: int = 3):
    """(b): pooled temporal concatenation, over the whole half-val.

    It needs no new extraction, so it is the one rung of the ladder that does not pay the subset's loss of
    power. Every stride's complete-window mask is intersected first and the whole table, arm A included, runs
    on that intersection (decision 13): rows evaluated on different frames are not comparable.
    """
    X, ok = {}, np.ones(len(ctx["fname"]), bool)
    for s in strides:
        X[s], m = temporal_arm(ctx, s, n_back)
        ok &= m
    log.info("(b): %d/%d frames have a complete window at every stride in %s", int(ok.sum()), len(ok), strides)
    arms = {"A ridge_late pooled": ridge_arm(ctx["pooled"]),
            **{f"b temporal s{s} ({n_back + 1}x{s * waymo.FRAME_DT:.1f}s)": ridge_arm(X[s]) for s in strides}}
    return run_ladder(lambda k: arms, "p2b", ok, ctx, rl=rl)


def repro(ctx: dict, rl):
    """Arm A on every context row: this must reproduce decision 3d / L0's arm A before anything else is read.

    L0 recorded lambda 10, inner-val residual ADE 1.7818 (direction 0) and pre-onset dADE -0.0274 against
    straight -0.1254. Anything else means the ladder's plumbing is not the plumbing entry 20 was measured
    with, and no number below it can be compared with that entry.
    """
    return run_ladder(lambda k: {"A ridge_late pooled": ridge_arm(ctx["pooled"])}, "repro", None, ctx, rl=rl)


def p2c(ctx: dict, keep: np.ndarray, rl, stride: int = 2, n_back: int = 3):
    """(c) and (d): the token-grid heads, their compute-matched control and the two combined, on the subset."""
    g = align(ctx, GRID_SET, [GRID_ARRAY])
    Xt, okt = temporal_arm(ctx, stride, n_back)
    keep = keep & g["covered"] & okt
    arms = {"A ridge_late pooled": ridge_arm(ctx["pooled"]),
            "b temporal (subset)": ridge_arm(Xt),
            "c-mlp pooled (compute-matched)": mlp_arm(ctx["pooled"]),
            "c-attn grid": grid_arm(g[GRID_ARRAY], GRID_TOKENS, "attn"),
            "c-tf grid + ego token": grid_arm(g[GRID_ARRAY], GRID_TOKENS, "tf", ctx["ego"]),
            "d temporal + attn grid": grid_arm(g[GRID_ARRAY], GRID_TOKENS, "attn", Xt)}
    return run_ladder(lambda k: arms, "p2c", keep, ctx, rl=rl)


def p2e(ctx: dict, keep: np.ndarray, rl):
    """(e): the gated residual head, in both its trunks, against arm A on the same frames (decision 25)."""
    g = align(ctx, GRID_SET, [GRID_ARRAY])
    keep = keep & g["covered"]
    arms = {"A ridge_late pooled": ridge_arm(ctx["pooled"]),
            "e gated pooled-mlp": gated_arm(ctx["pooled"], "mlp"),
            "e gated attn grid": gated_arm(g[GRID_ARRAY], "attn", GRID_TOKENS)}
    return run_ladder(lambda k: arms, "p2e", keep, ctx, rl=rl)


def p3(ctx: dict, keep: np.ndarray, rl, sets: list[str] | None = None):
    """The backbone ladder: arm A recomputed on the subset, plus one ridge_late arm per extracted set.

    A set that is not on disk is skipped with a line saying so rather than failing the ladder: the point of
    the ordering in the todo is that one model failing to download never blocks the rest.
    """
    arms, keep = {"A ridge_late pooled (qwen4b L18)": ridge_arm(ctx["pooled"])}, keep.copy()
    for name in (sets or list(P3_SETS)):
        st, want = P3_SETS[name]
        try:
            a = align(ctx, st, want)
        except (FileNotFoundError, OSError) as e:
            log.warning("%s: %s not on disk (%s) -- skipping this arm", name, st, e)
            continue
        keep &= a["covered"]
        for w in (want or [k for k in a if k != "covered"]):
            arms[f"{name} {w}"] = ridge_arm(a[w])
    log.info("P3: %d arms over %d frames", len(arms), int(keep.sum()))
    return run_ladder(lambda k: arms, "p3", keep, ctx, rl=rl)


def extract_grid(rl=None, batch_size: int = 4, limit: int | None = None, grid_hw=(6, 8)):
    """P2(c)'s extraction: the same frozen Qwen3-VL-4B, stopped at LAYER, storing the pooled token grid.

    `pooled=True` is kept on so that the run also writes L18_mean through this new code path; comparing it
    bit for bit against the cached `qwen_front3` array is the check that the grid features are the same
    features, only not averaged. Batch stays at 4: entry in the train-features todo, batch size changes the
    kernels and therefore the feature.
    """
    from . import features as F
    ctx = base_context()
    keep = load_subset(ctx)
    rows = ctx["rows"][keep][:limit] if limit else ctx["rows"][keep]
    layer = int(LAYER[1:3])
    fx = F.QwenFeatures(layers=[layer], n_images=3, grid_hw=grid_hw, compile=True)
    return waymo.extract_subset(GRID_SET if not limit else GRID_SET + "_probe", fx, rows,
                                batch_size=batch_size, rl=rl)


def grid_check(name: str = GRID_SET + "_probe") -> dict:
    """Does the grid extraction produce the same features as the cached set, or different ones?

    The grid run stops the decoder at LAYER and pools differently, but `L18_mean` should come out of it
    exactly as it came out of the `qwen_front3` run: same decode, same smart-resize, same batch size, same
    kernels. If it does not, the grid arm is not looking at arm A's features and the two are not comparable,
    which is the one thing that would invalidate all of P2(c). Reported as the max absolute difference and
    the share of rows that are bit-for-bit identical.
    """
    idx, arrs = waymo.load_flat_features(name, ["L18_mean"])
    ref_idx, ref = waymo.load_features("qwen_front3", ["L18_mean"])
    at = pd.Series(np.arange(len(ref_idx)), index=ref_idx.frame_name.to_numpy())
    pos = at.reindex(idx.frame_name.to_numpy()).to_numpy()
    a = np.asarray(arrs["L18_mean"]).astype(np.float32)
    b = np.asarray(ref["L18_mean"])[pos.astype(int)].astype(np.float32)
    d = np.abs(a - b)
    out = {"n": len(a), "identical_rows": int((d == 0).all(1).sum()),
           "max_abs_diff": float(d.max()), "mean_rel_diff": float(d.mean() / np.abs(b).mean())}
    log.info("grid check against qwen_front3: %s", out)
    return out


def extract_qwen32b(rl=None, batch_size: int = 4, limit: int | None = None,
                    layers=(32, 50), model_id: str = "Qwen/Qwen3-VL-32B-Instruct"):
    """P3(a): Qwen3-VL-32B down the same path as `qwen_front3` -- three cameras, the same smart-resize to
    960x1088, the same chat template, batch 4 -- so the only thing that differs from arm A is the backbone.

    Layer 50 of 64 is the one MiniMax H3 uses as its conditioner (decision 21), and layer 32 is the
    half-depth analogue of L18 of 36 in the 4B setting.
    """
    from . import features as F
    ctx = base_context()
    keep = load_subset(ctx)
    rows = ctx["rows"][keep][:limit] if limit else ctx["rows"][keep]
    fx = F.QwenFeatures(layers=list(layers), n_images=3, model_id=model_id, compile=False,
                        device_map="cuda")
    name = P3_SETS["a qwen32b"][0] + ("_probe" if limit else "")
    return waymo.extract_subset(name, fx, rows, batch_size=batch_size, rl=rl)


def extract_wan(rl=None, batch_size: int = 2, limit: int | None = None, **kw):
    """P3(c): Wan2.2-TI2V-5B's DiT, one forward per camera per noise level, over the frozen subset."""
    from .dit_features import WanDiTFeatures
    ctx = base_context()
    keep = load_subset(ctx)
    rows = ctx["rows"][keep][:limit] if limit else ctx["rows"][keep]
    fx = WanDiTFeatures(**kw)
    name = P3_SETS["c wan"][0] + ("_probe" if limit else "")
    return waymo.extract_subset(name, fx, rows, batch_size=batch_size, rl=rl)


VJEPA_VARIANTS = {                      # arm key -> (set name, camera, frames, stride, checkpoint)
    "d vjepa2":        ("vjepa2_p3",      "front",       4,  2, None),
    "d1 vjepa2 fl":    ("vjepa2_p3_fl",   "front_left",  4,  2, None),
    "d1 vjepa2 fr":    ("vjepa2_p3_fr",   "front_right", 4,  2, None),
    "d2 vjepa2 clip8": ("vjepa2_p3_f8",   "front",       8,  2, None),
    "d3 vjepa2 clip16": ("vjepa2_p3_f16", "front",      16,  2, None),
    "d4 vjepa2 vitg":  ("vjepa2g_p3",     "front",       4,  2, "facebook/vjepa2-vitg-fpc64-256"),
}


def extract_vjepa2(rl=None, batch_size: int = 8, limit: int | None = None, frames: int = 4, stride: int = 2,
                   cam: str = "front", name: str | None = None, model_id: str | None = None):
    """P3(d): V-JEPA 2 ViT-L over a short clip ending at the frame -- decision 12's video self-supervised control.

    Two things about this row have to travel with its number, exactly as decision 12 says. The checkpoint is
    `fpc64`, trained on 64-frame clips, and it is fed 4 (two tubelets): a known distribution shift, not a
    neutral setting. And V-JEPA takes one camera where the rest of the ladder takes three, so it sees less of
    the scene. The row is evidence about feeding time at Stage-A cost, not a backbone ranking.
    """
    from . import features as F
    ctx = base_context()
    keep = load_subset(ctx)
    rows = ctx["rows"][keep][:limit] if limit else ctx["rows"][keep]
    items, idx, full = waymo.clip_items(ctx["df"], rows, frames - 1, stride, cam)
    fx = F.VJepaFeatures(frames=frames, **({"model_id": model_id} if model_id else {}))
    name = (name or P3_SETS["d vjepa2"][0]) + ("_probe" if limit else "")
    return waymo.extract_items(name, fx, items, idx, batch_size, rl=rl, cams=[cam], frames_per_clip=frames,
                               clip_stride=stride, complete=int(full.sum()), requested=len(rows))


def p3d(ctx: dict, keep: np.ndarray, rl, variants: list[str] | None = None):
    """The V-JEPA 2 ladder: one variable moved at a time off arm (d), plus late fusion with Qwen.

    Every variant that is on disk joins the table; the rows are the intersection of what all of them cover,
    so the arms are compared on the same frames (decision 13, and the same discipline as P2(b)).
    """
    want = variants or ["d vjepa2", "d1 vjepa2 fl", "d1 vjepa2 fr", "d2 vjepa2 clip8", "d3 vjepa2 clip16",
                        "d4 vjepa2 vitg"]
    got, keep = {}, keep.copy()
    for k in want:
        st = VJEPA_VARIANTS[k][0]
        try:
            a = align(ctx, st, ["mean"])
        except (FileNotFoundError, OSError) as e:
            log.warning("%s: %s not on disk (%s) -- skipping", k, st, e)
            continue
        got[k], keep = a, keep & a["covered"]
    arms = {"A ridge_late pooled (qwen4b L18)": ridge_arm(ctx["pooled"])}
    if "d vjepa2" in got:
        arms["d vjepa2 (front, 4 frames)"] = ridge_arm(got["d vjepa2"]["mean"])
        arms["d5 late fusion vjepa2 + qwen"] = ridge_arm(Concat([got["d vjepa2"]["mean"], ctx["pooled"]]))
    if all(k in got for k in ("d vjepa2", "d1 vjepa2 fl", "d1 vjepa2 fr")):
        arms["d1 vjepa2 3 cameras"] = ridge_arm(Concat([got[k]["mean"] for k in
                                                        ("d vjepa2", "d1 vjepa2 fl", "d1 vjepa2 fr")]))
    for k, label in (("d2 vjepa2 clip8", "d2 vjepa2 clip 8"), ("d3 vjepa2 clip16", "d3 vjepa2 clip 16"),
                     ("d4 vjepa2 vitg", "d4 vjepa2 ViT-g")):
        if k in got:
            arms[label] = ridge_arm(got[k]["mean"])
    log.info("P3(d'): %d arms over %d frames", len(arms), int(keep.sum()))
    return run_ladder(lambda k: arms, "p3d", keep, ctx, rl=rl)


def shape_change(dec: pd.DataFrame, lo: int = 4, tol: float = 2.0) -> pd.DataFrame:
    """The pre-registered shape test: is the top decile's relative gain within `tol` points of the best bin?

    "Best" is the most negative relative gain over deciles `lo`..10, which is where the logged future is a
    target worth fitting (decision 20 disqualified bin 10 from deciding whether an arm buys anything, but it
    is exactly the bin this test is about, so it takes part here).
    """
    rows = []
    for (d, arm, scope), g in dec.groupby(["direction", "arm", "scope"]):
        g = g[g.decile >= lo]
        if g.empty:
            continue
        best = g.rel_gain.min()
        top = g[g.decile == 10].rel_gain
        if top.empty:
            continue
        rows.append({"direction": d, "arm": arm, "scope": scope,
                     "peak_decile": int(g.loc[g.rel_gain.idxmin(), "decile"]), "peak_rel_gain": best,
                     "decile10_rel_gain": float(top.iloc[0]), "gap": float(top.iloc[0]) - best,
                     "shape_change": bool(float(top.iloc[0]) - best <= tol)})
    return pd.DataFrame(rows)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="subset",
                    help="comma list of subset,repro,p2b,p2c,p2e,p3,p3d,grid,gridcheck,qwen32b,vjepa2,wan")
    ap.add_argument("--tag", default=None, help="run directory tag; defaults to the step list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="profiling: extract only this many frames")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sets", default=None, help="p3 / p3d: comma list of arm keys to include")
    ap.add_argument("--variant", default=None, help="vjepa2: a key of VJEPA_VARIANTS to extract")
    a = ap.parse_args()
    rl = RunLog("waymo_ladder", a.tag or a.steps.replace(",", "-"))
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    steps = a.steps.split(",")
    if "grid" in steps:
        rl.event("extract", **extract_grid(rl, a.batch_size, a.limit))
    if "qwen32b" in steps:
        rl.event("extract", **extract_qwen32b(rl, a.batch_size, a.limit))
    if "gridcheck" in steps:
        rl.event("grid_check", **grid_check())
    if "vjepa2" in steps:
        st, cam, fr, sd, mid = VJEPA_VARIANTS[a.variant or "d vjepa2"]
        rl.event("extract", **extract_vjepa2(rl, a.batch_size, a.limit, fr, sd, cam, st, mid))
    if "wan" in steps:
        rl.event("extract", **extract_wan(rl, a.batch_size, a.limit))
    if {"subset", "repro", "p2b", "p2c", "p2e", "p3", "p3d"} & set(steps):
        ctx = base_context(a.seed)
        if "subset" in steps:
            summary = write_subset(ctx, choose_subset(ctx, seed=a.seed))
            (rl.dir / "subset.json").write_text(json.dumps(summary, indent=2))
            rl.event("subset", **summary)
        if "repro" in steps:
            repro(ctx, rl)
        if "p2b" in steps:
            p2b(ctx, rl)
        if "p2c" in steps:
            p2c(ctx, load_subset(ctx), rl)
        if "p2e" in steps:
            p2e(ctx, load_subset(ctx), rl)
        if "p3" in steps:
            p3(ctx, load_subset(ctx), rl, a.sets.split(",") if a.sets else None)
        if "p3d" in steps:
            out = p3d(ctx, load_subset(ctx), rl, a.sets.split(",") if a.sets else None)
            sc = shape_change(out["deciles"])
            sc.to_csv(rl.dir / "p3d_shape.csv", index=False)
            rl.log.info("p3d shape test\n%s", sc.to_markdown(index=False, floatfmt=".2f"))
            rl.event("shape", rows=sc.to_dict("records"))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
