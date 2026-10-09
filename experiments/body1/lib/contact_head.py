"""BODY1 arm S0 (plans/2026-10-10-body1-prereg.md, section 3 + Amendment 1): the contact predictor on frozen Cinque `view_39` tokens.

Input   scene = vision tokens of the 8 policy slots (8, 32, 512) with a slot-validity mask + the 20-dim ego vector (it carries the command);
        query = 8 rear-axle poses at 0.5 .. 4 s in the state's frame.
Output  per query: agent-contact logit, boundary-contact logit, first-contact arc length / time (agent), arc length (boundary), minimum agent
        clearance, minimum drivable margin (OUT); arch `step` also predicts clearance and margin per 0.5 s interval of the sweep.
Archs   step   scene memory (transformer over ego + vision tokens); a query is 9 step tokens (t = 0 and the 8 poses) that attend to each other
               and to the memory; dense per-interval regression targets + a pooled per-query head. The sweep is queried along its own time
               axis, so object motion can be read from the 8 slots at the time the ego is there.
        crit   the same memory, one token per query (a per-query cross-attention critic), no dense targets.
        blind  arch step without vision tokens: ego + query only, the floor every AUC is compared with.
Labels  rows of bd1_rows.py; per-interval clearance / margin from `step_labels` (bd1_train.py steps). Agent boxes, the raster and logged futures
        are labels only. Agent positive = a_hit (a rear-end by a faster object is not a contact; a row whose only contact is one has weight 0
        in training). Boundary positive = margin < -0.20 m with the first contact after t = 0; rows with a margin in [-0.20, 0), rows already
        outside at t = 0 and rows without raster coverage have no boundary label (Amendment 1 item 2).
Classes primary user class with Amendment 1 item 1 (class 3 also holds straight / launch tokens with a side hazard), `classes`.
"""
import hashlib

import numpy as np

import b1 as B
import sweep as SW

OWN = (0, 1)                                    # query slots of the student's own plan (P2H10-F-s0 / -s1)
NT = 9                                          # step tokens: t = 0 and the 8 poses; token k > 0 owns the dense steps of (t_{k-1}, t_k]
A_CLIP, M_CLIP = (-1.0, 8.0), (-2.0, 4.0)       # clip of the clearance / margin targets (m); margin clip = decision 192
B_DEPTH = -0.20
OUT = ("a_logit", "b_logit", "a_s", "a_t", "b_s", "clr", "margin")
S_SCALE, T_SCALE = 10.0, 4.0                    # arc length / time targets are regressed in these units
ROW_KEYS = ("gi", "off", "log", "q", "qok", "par", "a_hit", "a_t", "a_s", "a_cls", "a_rear", "a_t0", "a_clr", "b_hit", "b_t", "b_s", "b_margin", "b_cov", "b_t0")
TYPES = ("agent", "boundary", "none")


def sroot():
    return B.root() / "s0"


def is_val(log: str) -> bool:
    """Validation part of body1-train-logs (model selection only): sha256('bd1val|' + log) % 10 == 0."""
    return int(hashlib.sha256(f"bd1val|{log}".encode()).hexdigest(), 16) % 10 == 0


def frac_logs(logs, frac: float) -> set:
    """Nested log subsets for the learning curve: the first ceil(frac * n) logs in sha256('bd1lc|' + log) order."""
    lg = sorted(set(logs), key=lambda l: hashlib.sha256(f"bd1lc|{l}".encode()).hexdigest())
    return set(lg[: max(1, int(np.ceil(frac * len(lg))))])


def classes(T: dict):
    """tax.npz -> (primary user class per token with Amendment 1 item 1: 0 other, 1, 2, 3; > 45 deg flag)."""
    c3 = T["c3"] | (np.isin(T["man"], (B.MAN.index("straight"), B.MAN.index("launch"))) & T["f_side"])
    return np.where(T["c1"], 1, np.where(T["c2"], 2, np.where(c3, 3, 0))).astype(np.int8), np.abs(T["dyaw"]) > 45


# ---------------------------------------------------------------- per-interval labels (CPU, once)
def step_labels(z: dict, L: dict, chunk: int = 96):
    """Rows unit z, label store L -> (clearance (n, Q, 9), margin (n, Q, 9)) float16: min over the dense steps of each step token's interval of the
    ego-box clearance to any valid object at the matching time / of the corner SDF; 99 where undefined. min over tokens = a_clr / b_margin."""
    n, Q = z["q"].shape[:2]
    A, M = np.empty((n, Q, NT), np.float16), np.empty((n, Q, NT), np.float16)
    win = lambda x: np.concatenate([x[..., :1], x[..., 1:].reshape(x.shape[:-1] + (8, 5)).min(-1)], -1)  # noqa: E731
    for c in range(0, n, chunk):
        sl = slice(c, min(c + chunk, n))
        gi = z["gi"][sl]
        d = SW.dense(z["q"][sl], np.broadcast_to(z["off"][sl][:, None], (len(gi), Q, 2)))
        b, v, _ = SW.dense_boxes(np.asarray(L["box"][gi]), np.asarray(L["valid"][gi]))
        v &= (np.asarray(L["cls"][gi]) >= 0)[:, None]
        ex, ey = SW.ego_centre(d)
        bb = b[:, None]
        cl, _ = SW.clearance(ex[..., None], ey[..., None], d[..., 2:3], bb[..., 0], bb[..., 1], bb[..., 2], bb[..., 3] / 2, bb[..., 4] / 2)
        A[sl] = win(np.where(v[:, None], cl, SW.BIG).min(-1))
        m, ins = SW.corner_margins(d, np.asarray(L["sdf"][gi]))
        M[sl] = win(np.where(ins.all(-1), m.min(-1), SW.BIG))
    return A, M


# ---------------------------------------------------------------- rows
def load_rows(fams, shards, keep, steps=True) -> dict:
    """Concatenated row units of the (family, shard) dirs, restricted to the states where keep(log (n,), hold (n,)) -> bool mask.
    Adds fam (index into `fams`), ego (n, 20), src = [(cache dir, rows in it, first output row)] for `load_tokens`, sa / sm (n, Q, 9) step labels."""
    out, src, o = {k: [] for k in ROW_KEYS + ("fam", "hold", "ego") + (("sa", "sm") if steps else ())}, [], 0
    for fi, f in enumerate(fams):
        for k in shards:
            d = B.cdir(f, k)
            z = np.load(B.root() / "rows" / f"{d}.npz")
            m = np.flatnonzero(keep(z["log"], z["hold"]))
            if not len(m):
                continue
            for key in ROW_KEYS:
                out[key].append(z[key][m])
            out["hold"].append(z["hold"][m]), out["fam"].append(np.full(len(m), fi, np.int8)), out["ego"].append(B.tab(f, k)["ego"][m].astype(np.float32))
            if steps:
                s = np.load(sroot() / "steps" / f"{d}.npz")
                out["sa"].append(s["a"][m]), out["sm"].append(s["m"][m])
            src.append((d, m, o))
            o += len(m)
    R = {k: np.concatenate(v) for k, v in out.items()}
    return R | dict(src=src, fams=list(fams))


def targets(R: dict) -> dict:
    """Row units -> per-(state, query) labels and masks (see the module docstring)."""
    mg = R["b_margin"]
    b_def = mg < 90
    b_pos = b_def & (mg < B_DEPTH) & ~R["b_t0"]
    b_ok = R["qok"] & (b_pos | (b_def & (mg >= 0)))
    rear_only = R["a_rear"] & ~R["a_hit"]
    return dict(a=R["a_hit"], a_ok=R["qok"], b=b_pos, b_ok=b_ok, use=R["qok"] & ~rear_only, typ=np.where(R["a_hit"], 0, np.where(b_pos, 1, 2)).astype(np.int8))


def balance(cls_state, Y, fit) -> np.ndarray:
    """`w4` rule on the given training states: (4 classes x 3 contact types) get equal mass. -> row weights (n, Q), 0 outside `use`, mean 1 over
    the used rows of `fit`."""
    c = np.broadcast_to(cls_state[:, None], Y["typ"].shape)
    cell = c * 3 + Y["typ"]
    n = np.bincount(cell[fit][Y["use"][fit]], minlength=12).astype(float)
    w = np.where(Y["use"], (1.0 / np.maximum(n, 1))[cell], 0.0)
    return (w / w[fit][Y["use"][fit]].mean()).astype(np.float32), n.reshape(4, 3)


def load_tokens(src, n, dev, chunk: int = 1024, threads: int = 8):
    """The (n, 8, 32, 512) fp16 vision tokens of the selected states on the device, gathered chunk-wise from the memory-mapped caches (the host
    holds at most `threads` chunks of 0.27 GB)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    V = torch.empty((n, 8, 32, 512), dtype=torch.float16, device=dev)
    jobs = [(d, m[c:c + chunk], o + c) for d, m, o in src for c in range(0, len(m), chunk)]

    def one(j):
        d, rows, o = j
        f = np.load(B.cache_root() / f"{d}@warp" / "front.npy", mmap_mode="r")
        return o, np.ascontiguousarray(f[rows])
    with ThreadPoolExecutor(threads) as ex:
        for o, x in ex.map(one, jobs):
            V[o:o + len(x)] = torch.from_numpy(x).to(dev)
    return V


# ---------------------------------------------------------------- the head
def build(arch: str = "step", d: int = 256, enc: int = 3, dec: int = 3, drop: float = 0.1):
    import torch
    import torch.nn as nn
    assert arch in ("step", "crit", "blind")
    mlp = lambda i, o: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))  # noqa: E731

    class Dec(nn.Module):
        def __init__(s):
            super().__init__()
            s.n0, s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            s.sa, s.ca = nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True), nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, x, mem, pad, T):
            b, n, _ = x.shape
            if T > 1:                                                           # attention inside one query's step tokens
                y = s.n0(x).view(b * n // T, T, d)
                x = x + s.sa(y, y, y, need_weights=False)[0].view(b, n, d)
            x = x + s.ca(s.n1(x), mem, mem, key_padding_mask=pad, need_weights=False)[0]
            return x + s.ff(s.n2(x))

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            s.arch, s.vision, s.T = arch, arch != "blind", 1 if arch == "crit" else NT
            s.ep = mlp(20, d)
            if s.vision:
                s.vln, s.vp = nn.LayerNorm(512), nn.Linear(512, d)
                s.slot, s.pos = nn.Parameter(torch.randn(8, 1, d) * 0.02), nn.Parameter(torch.randn(1, 32, d) * 0.02)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True), enc, enable_nested_tensor=False)
            s.register_buffer("fr", torch.pi * 2.0 ** torch.arange(8))
            s.sp, s.temb = mlp(38, d), nn.Parameter(torch.randn(NT, d) * 0.02)
            if arch == "crit":
                s.qp = mlp(NT * d, d)
            s.dec = nn.ModuleList([Dec() for _ in range(dec)])
            s.so = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
            s.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(OUT)))

        def memory(s, V, valid, E):
            """V (B, 8, 32, 512), valid (B, 8) bool, E (B, 20) standardised -> memory (B, 1 + 256, d), padding mask."""
            e = s.ep(E)[:, None]
            if not s.vision:
                return s.enc(e), None
            v = (s.vp(s.vln(V.float())) + s.slot + s.pos).flatten(1, 2)
            pad = torch.cat([valid.new_zeros(len(valid), 1), ~valid.repeat_interleave(32, 1)], 1)
            return s.enc(torch.cat([e, v], 1), src_key_padding_mask=pad), pad

        def steps(s, q):
            """q (B, Q, 8, 3) -> step features (B, Q, 9, 38): Fourier of x, y; x, y, sin / cos yaw, step length, yaw step."""
            with torch.autocast("cuda", enabled=False):
                p = torch.cat([torch.zeros_like(q[..., :1, :]), q.float()], -2)
                u = torch.stack([(p[..., 0] - 24.0) / 32.0, p[..., 1] / 24.0], -1)[..., None] * s.fr
                dp = torch.diff(p, dim=-2, prepend=p[..., :1, :])
                return torch.cat([torch.sin(u).flatten(-2), torch.cos(u).flatten(-2), p[..., 0:1] / 32.0, p[..., 1:2] / 24.0, torch.sin(p[..., 2:3]), torch.cos(p[..., 2:3]),
                                  torch.hypot(dp[..., 0:1], dp[..., 1:2]) / 8.0, dp[..., 2:3]], -1)

        def forward(s, V, valid, E, q):
            """-> (per-query outputs (B, Q, len(OUT)) fp32, per-step (clearance, margin) (B, Q, 9, 2) fp32 or None)."""
            B_, Q = q.shape[:2]
            mem, pad = s.memory(V, valid, E)
            x = s.sp(s.steps(q)) + s.temb                                       # (B, Q, 9, d)
            x = s.qp(x.flatten(2))[:, :, None] if s.T == 1 else x
            x = x.flatten(1, 2)
            for l in s.dec:
                x = l(x, mem, pad, s.T)
            x = x.view(B_, Q, s.T, -1)
            return s.out(torch.cat([x.mean(2), x.amax(2)], -1)).float(), (s.so(x).float() if s.T > 1 else None)
    return Head()


def predict(net, V, E, q, idx=None, slots: int = 8, valid=None, batch: int = 512):
    """Batched eval forward of device tensors (rows idx, default all) -> numpy (out (n, Q, len(OUT)), step (n, Q, 9, 2) or None).
    The head sees the `slots` newest slots, or the explicit mask valid (n, 8) aligned with the selected rows."""
    import torch
    net.eval()
    idx = torch.arange(len(q), device=q.device) if idx is None else idx
    O, S = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c in range(0, len(idx), batch):
            i = idx[c:c + batch]
            v = (torch.arange(8, device=q.device)[None] >= 8 - slots).expand(len(i), 8) if valid is None else valid[c:c + batch]
            o, st = net(V[i] if V is not None else None, v, E[i], q[i])
            O.append(o.cpu().numpy()), S.append(st.cpu().numpy() if st is not None else None)
    net.train()
    return np.concatenate(O), (np.concatenate(S) if S[0] is not None else None)


def load_ckpt(path, dev):
    """ckpt.pt of bd1_train.py -> (net in eval mode, ckpt dict with emu / esd / config / val)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    net = build(c["arch"], c["d"], c["enc"], c["dec"], c["drop"]).to(dev)
    net.load_state_dict(ck["model"])
    return net.eval(), ck


# ---------------------------------------------------------------- AUC with a log-clustered bootstrap
def auc(y, s) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def auc_boot(y, s, groups, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """AUC (ties half) with a percentile cluster bootstrap by `groups`, the resampling of jevdrive.stats.bootstrap(groups=...): the same
    `default_rng(seed).integers(G, size=(n_boot, G))` cluster draws; the statistic is recomputed exactly per draw from the (cluster x cluster)
    matrix of positive-over-negative wins. -> dict(auc, lo, hi, n, pos, logs, logs_pos, n_boot, seed, alpha)."""
    import pandas as pd
    y, s = np.asarray(y, bool), np.asarray(s, float)
    codes, uniq = pd.factorize(np.asarray(groups))
    G = len(uniq)
    out = dict(auc=np.nan, lo=np.nan, hi=np.nan, n=len(y), pos=int(y.sum()), logs=G, logs_pos=int(len(np.unique(codes[y]))), n_boot=n_boot, seed=seed, alpha=alpha)
    if not y.any() or y.all():
        return out
    sp, gp, sn, gn = s[y], codes[y], s[~y], codes[~y]
    W = np.zeros((len(sp), G))
    for h in np.unique(gn):
        x = np.sort(sn[gn == h])
        lo, hi = np.searchsorted(x, sp, "left"), np.searchsorted(x, sp, "right")
        W[:, h] = lo + 0.5 * (hi - lo)
    U = np.zeros((G, G))
    np.add.at(U, gp, W)
    n1, n0 = np.bincount(gp, minlength=G).astype(float), np.bincount(gn, minlength=G).astype(float)
    idx = np.random.default_rng(seed).integers(G, size=(n_boot, G))
    c = np.zeros((n_boot, G))
    np.add.at(c, (np.arange(n_boot)[:, None], idx), 1.0)
    den = (c @ n1) * (c @ n0)
    b = np.einsum("bg,gh,bh->b", c, U, c)[den > 0] / den[den > 0]
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    out.update(auc=float(U.sum() / (n1.sum() * n0.sum())), lo=float(lo), hi=float(hi))
    return out
