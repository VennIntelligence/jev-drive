#!/usr/bin/env python3
"""ROUND9 BODY-SCHEMA (research/body-schema/notes.md section 0, registered before any read): on PAI own states (on1_PAID, R5LC-L18-s0's
DAgger states on 718 train scenes), is a plan-conditioned schema head on frozen Cinque tokens accurate (per-waypoint signed clearance to the
road edge and to agents ahead, overshoot past the human's stop point), and is its gradient a safe plan correction? No policy training.

  prep   (op-train, CPU)  folds by scene, stop-point labels from the human log future, PF / FS sets, the forward row selection -> $O/prep.npz
  fwd    (op-train, CPU)  --ckpt TAG --shard i/n: eval-mode plans of TAG on the selected rows (pai_s_gate.cmd_fwd's build, R6DAG's argv,
                          rm_from TAG) -> $O/fwd/<TAG>_<i>.npz
  train  (op-train, GPU)  --fold k: the schema head (vision) and its blind twin (ego + query only) on the other folds' rows, labels on the fly
                          for the served plan + 7 random perturbations; then a small learned refine on the frozen vision head -> $O/f<k>/heads.pt
  eval   (op-train, GPU)  --fold k: (a) predictions / true labels on the fold's rows for the served, NTV0 and PCL1 plans; (b) corrections on
                          the PF / FS rows (schema gradient, blind gradient, true-raster oracle, learned refine) -> $O/f<k>/eval.npz
  report (op-train, CPU)  every line of notes section 0 -> $O/report.json (fetched to experiments/round9/results/body_schema/ by hand)

Labels are the trainer's own geometry with margin 0 (lib/drivable_hinge.Hinge.margins on the road_pai raster of PAI-S-GATE,
lib/agent_hinge.AgentHinge.distances on the replay boxes): the 41 dense steps are grouped into the 8 waypoint intervals ((k-1) x 0.5, k x 0.5] s
and the minimum is taken. Stop point: the human's 4 s log future, the first knot from which every later 0.5 s interval is < 0.5 m/s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("JEV_REPO", HERE.parents[2]))
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round9/scripts"), str(REPO / "experiments/body1/lib"),
                str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/route_follow/scripts"),
                str(REPO / "experiments/op_probe/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/body_schema"
CR = DATA / "runs/op_parity/cache"
POOL = CR / "on1_PAID"
ROAD = DATA / "runs/round9/pai_s_gate/road/road_on1_PAID.npz"
META = DATA / "runs/round9/pai_s_gate/meta.npz"
NF, PF_S, FWD_WIN_S = 4, 2.5, 6.0                   # folds; pre-failure window; forward rows: window before an event
CLIP = ((-2.0, 4.0), (-1.0, 8.0), (-10.0, 10.0))    # road margin / agent clearance / stop overshoot targets (m)
BIG = 1e3
CH = ("road", "agent", "over")
CKPTS = ("NTV0-L18-s0", "PCL1-F-s0")
ETAS, KS = (0.1, 0.3, 1.0, 3.0), (1, 5)
LAMS = (0.2, 1.0)                                    # learned refine: weight of the squared plan change
MARGIN = 0.3                                         # the trainer's hinge margin, used in the correction objective


def fold_of(scene: str) -> int:
    return int(hashlib.sha256(f"bsl|{scene}".encode()).hexdigest(), 16) % NF


# ---------------------------------------------------------------- prep (CPU)
def cmd_prep(a):
    tab = np.load(POOL / "tab.npz")
    M = np.load(META)
    names = tab["names"].astype(str)
    assert np.array_equal(M["names"].astype(str), names)
    n = len(names)
    scene = M["scene"].astype(str)
    fold = np.array([fold_of(s) for s in scene], np.int8)
    t0, eo, ec, er = M["t0"], M["ev_off"], M["ev_col"], M["ev_cor"]
    dto, dtc = (eo - t0) / 1e6, (ec - t0) / 1e6
    pf_off = (eo > 0) & (dto > 0) & (dto <= PF_S)
    pf_col = (ec > 0) & (dtc > 0) & (dtc <= PF_S)
    win = ((eo > 0) & (dto > 0) & (dto <= FWD_WIN_S)) | ((ec > 0) & (dtc > 0) & (dtc <= FWD_WIN_S))
    sc_u, sc_i = np.unique(scene, return_inverse=True)
    clean = np.ones(len(sc_u), bool)
    np.logical_and.at(clean, sc_i, (eo < 0) & (ec < 0) & (er < 0) & (M["score"] >= 0.98))
    fs = clean[sc_i]
    # stop point from the human's log future (row frame): 9 knots log0, log_fut
    p = np.concatenate([M["log0"][:, None, :2], M["log_fut"][..., :2]], 1).astype(np.float64)
    yaw = np.concatenate([M["log0"][:, None, 2], M["log_fut"][..., 2]], 1).astype(np.float64)
    v = np.linalg.norm(np.diff(p, axis=1), axis=-1) / 0.5                         # (n, 8) interval speeds
    ok = M["log_fut_ok"].all(1) & np.isfinite(p).all((1, 2)) & np.isfinite(M["log0"]).all(1)
    rest = np.flip(np.cumprod(np.flip(v < 0.5, 1), 1), 1).astype(bool)          # rest[k]: every interval >= k is slow
    has = rest.any(1) & ok & (M["v0"] >= 0.5)
    k = np.where(has, rest.argmax(1) + 1, 0)                                      # first resting knot (1..8)
    st = np.zeros((n, 4), np.float32)
    st[:, 0], st[:, 1], st[:, 2] = p[np.arange(n), k, 0], p[np.arange(n), k, 1], yaw[np.arange(n), k]
    st[:, 3] = has
    st[~has, :3] = 0
    rng = np.random.default_rng(0)
    sel = win | (fs & (rng.random(n) < 0.25)) | (rng.random(n) < 0.15)
    fsamp = fs & (rng.random(n) < 0.35)                                          # FS rows used in (b)
    O.mkdir(parents=True, exist_ok=True)
    np.savez(O / "prep.npz", names=names, scene=scene, fold=fold, v0=M["v0"], pf_off=pf_off, pf_col=pf_col, fs=fs, fs_b=fsamp, stop=st,
             sel=np.flatnonzero(sel), score=M["score"], fail=M["fail"].astype(str))
    info = dict(rows=n, scenes=len(sc_u), folds=np.bincount(fold, minlength=NF).tolist(), pf_off=int(pf_off.sum()), pf_col=int(pf_col.sum()),
                pf_scenes=int(len(np.unique(scene[pf_off | pf_col]))), fs_rows=int(fs.sum()), fs_scenes=int(clean.sum()), stop_rows=int(has.sum()),
                fwd_rows=int(sel.sum()))
    print(json.dumps(info), flush=True)


# ---------------------------------------------------------------- fwd (CPU): plans of a checkpoint on the selected rows
def cmd_fwd(a):
    import torch
    import pai_s_gate as PSG
    import rtc
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *rtc.R6DAG_ARGV.format(tag=a.ckpt, data=DATA).split()]
    exec(compile(src.replace(tail, "_BSL_ARGS = a"), str(rt), "exec"), g)
    ra = g["_BSL_ARGS"]
    ra.rm_from = a.ckpt
    dev = torch.device("cpu")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    S, model = ns.S, ns.model.eval()
    pn = S.tab["names"][ns.on0:].astype(str)
    names, _, _ = PSG.pool_keys()
    key = {":".join(x.split(":")[-2:]): i for i, x in enumerate(names)}
    pd_rows = np.flatnonzero(np.char.startswith(pn, "on:pd:"))
    assert len(pd_rows) == len(names), (len(pd_rows), len(names))
    pid = np.array([key[":".join(x.split(":")[-2:])] for x in pn[pd_rows]])
    inv = np.full(len(names), -1)
    inv[pid] = np.arange(len(pid))
    want = np.load(O / "prep.npz")["sel"]
    i, k = (int(x) for x in a.shard.split("/"))
    want = want[i::k]
    rows = inv[want]
    assert (rows >= 0).all()
    pi_ = torch.as_tensor(S.pi, device=dev)
    P = []
    t0 = time.time()
    for j in range(0, len(rows), a.bs):
        r = torch.as_tensor(ns.on0 + pd_rows[rows[j:j + a.bs]], device=dev)
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], S.ego[r], S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
        P.append(torch.stack([x, y, psi], -1).numpy())
        if j == 0:
            print(f"first batch {time.time() - t0:.1f} s, rows {len(rows)}", flush=True)
    (O / "fwd").mkdir(parents=True, exist_ok=True)
    np.savez(O / "fwd" / f"{a.ckpt}_{i}.npz", idx=want, poses=np.concatenate(P))
    print(json.dumps(dict(ckpt=a.ckpt, shard=a.shard, rows=len(rows), sec=round(time.time() - t0, 1))), flush=True)


def plans_of(ckpt, n):
    """(n, 8, 3) plans of ckpt on PAID rows from the fwd shards, NaN where not forwarded."""
    P = np.full((n, 8, 3), np.nan, np.float32)
    for f in sorted((O / "fwd").glob(f"{ckpt}_*.npz")):
        z = np.load(f)
        P[z["idx"]] = z["poses"]
    return P


# ---------------------------------------------------------------- geometry: the trainer's terms as per-waypoint labels
class Geo:
    def __init__(self, names, stop, dev):
        import torch
        from agent_hinge import AgentHinge
        from drivable_hinge import Hinge
        t = time.time()
        self.road = Hinge(ROAD, names, dev, 0.0, bank=True)
        self.ag = AgentHinge(POOL / "agent.npz", names, dev, 0.0)
        self.stop = torch.as_tensor(stop, device=dev)
        print(f"labels on {dev} in {time.time() - t:.0f} s: road ok {float(self.road.ok.float().mean()):.4f}, agent ok "
              f"{float(self.ag.ok.float().mean()):.4f}", flush=True)

    def dense(self, q, rows):
        """q (B, 8, 3) rear-axle poses, rows (B,) -> road (B, 41), agent (B, 41) per 0.1 s step (margin 0; BIG where no object counts)."""
        x, y, psi = q.unbind(-1)
        m = self.road.margins(x, y, psi, rows).view(len(q), 41, 4).amin(-1)
        d, cnt, _ = self.ag.distances(x, y, psi, rows)
        return m, d.masked_fill(~cnt, BIG).amin(-1)

    def labels(self, q, rows):
        """q (B, Q, 8, 3), rows (B,) -> Y (B, Q, 8, 3) [road, agent, over] (clipped), ok (B, Q, 8, 3) bool, t0 (B, Q, 2) road / agent at t0."""
        import torch
        B, Q = q.shape[:2]
        rr = rows.repeat_interleave(Q)
        m, d = self.dense(q.reshape(B * Q, 8, 3), rr)
        road = m[:, 1:].reshape(B * Q, 8, 5).amin(-1)
        ag = d[:, 1:].reshape(B * Q, 8, 5).amin(-1)
        st = self.stop[rr]
        h = st[:, 2:3]
        qq = q.reshape(B * Q, 8, 3)
        over = (qq[..., 0] - st[:, :1]) * torch.cos(h) + (qq[..., 1] - st[:, 1:2]) * torch.sin(h)
        Y = torch.stack([road.clamp(*CLIP[0]), ag.clamp(*CLIP[1]), over.clamp(*CLIP[2])], -1).view(B, Q, 8, 3)
        ok = torch.stack([self.road.ok[rr], self.ag.ok[rr], st[:, 3] > 0.5], -1)[:, None].expand(B * Q, 8, 3).reshape(B, Q, 8, 3)
        return Y, ok, torch.stack([m[:, 0], d[:, 0]], -1).view(B, Q, 2)

    def viol(self, q, rows, margin=0.0):
        """Differentiable true violation per plan (B,): sum over the 41 steps of relu(margin - road) + relu(margin - agent) (rows without a label
        contribute 0 for that term)."""
        m, d = self.dense(q, rows)
        okr, oka = self.road.ok[rows].float()[:, None], self.ag.ok[rows].float()[:, None]
        return ((margin - m[:, 1:]).clamp_min(0) * okr + (margin - d[:, 1:]).clamp_min(0) * oka).sum(1)


# ---------------------------------------------------------------- queries: random perturbations of a plan (training only)
def perturb(q, P, g):
    """q (B, 8, 3) -> (B, P, 8, 3): each copy takes a random subset of: time gain (speed x U[0.5, 1.5] in log), rotation about the origin
    (U[-7, 7] deg), lateral ramp (A U[-2, 2] m reached at T U[1, 4] s, smoothstep); half the copies at a quarter of the size. Yaw is
    recomputed from the path tangent where a step moves > 0.3 m."""
    import torch
    B, dev = len(q), q.device
    x = q[:, None].expand(B, P, 8, 3).clone()
    U = lambda *s: torch.rand(*s, generator=g, device=dev)                         # noqa: E731
    sc = torch.where(U(B, P) < 0.5, 0.25, 1.0)
    use = U(B, P, 3) < 0.6
    t = torch.arange(1, 9, device=dev, dtype=torch.float32) * 0.5
    P9 = torch.cat([torch.zeros_like(x[:, :, :1]), x], 2)                           # (B, P, 9, 3)
    gain = torch.exp((U(B, P) * 2 - 1) * np.log(1.5) * sc)
    gain = torch.where(use[..., 0], gain, torch.ones_like(gain))
    tau = t * gain[..., None]                                                        # (B, P, 8)
    j = (tau / 0.5).floor().clamp(0, 7).long()
    f = (tau / 0.5 - j)[..., None]
    a0 = torch.gather(P9, 2, j[..., None].expand(-1, -1, -1, 3))
    a1 = torch.gather(P9, 2, (j + 1)[..., None].expand(-1, -1, -1, 3))
    x = a0 + f * (a1 - a0)                                                           # linear; f > 1 extrapolates the last segment
    th = (U(B, P) * 2 - 1) * np.deg2rad(7.0) * sc * use[..., 1]
    c, s = torch.cos(th)[..., None], torch.sin(th)[..., None]
    x = torch.stack([c * x[..., 0] - s * x[..., 1], s * x[..., 0] + c * x[..., 1], x[..., 2] + th[..., None]], -1)
    A = (U(B, P) * 4 - 2) * sc * use[..., 2]
    T = 1 + 3 * U(B, P)
    u = (t / T[..., None]).clamp(0, 1)
    d = A[..., None] * u * u * (3 - 2 * u)
    x = torch.stack([x[..., 0] - torch.sin(x[..., 2]) * d, x[..., 1] + torch.cos(x[..., 2]) * d, x[..., 2]], -1)
    xy = torch.cat([torch.zeros_like(x[..., :1, :2]), x[..., :2]], 2)
    dxy = xy[..., 1:, :] - xy[..., :-1, :]
    mv = dxy.norm(dim=-1) > 0.3
    yaw = torch.where(mv, torch.atan2(dxy[..., 1], dxy[..., 0]), x[..., 2])
    yaw = q[:, None, :, 2] + torch.atan2(torch.sin(yaw - q[:, None, :, 2]), torch.cos(yaw - q[:, None, :, 2]))   # unwrap near the base yaw
    return torch.stack([x[..., 0], x[..., 1], yaw], -1)


# ---------------------------------------------------------------- the head (BODY1 S0 `step` architecture, 3 per-step channels + stop logit)
def build_head(vision=True, d=256, enc=3, dec=3, drop=0.1):
    import torch
    import torch.nn as nn
    mlp = lambda i, o: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))  # noqa: E731

    class Dec(nn.Module):
        def __init__(s):
            super().__init__()
            s.n0, s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            s.sa, s.ca = nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True), nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, x, mem, pad):
            b, n, _ = x.shape
            y = s.n0(x).reshape(b * n // 9, 9, d)
            x = x + s.sa(y, y, y, need_weights=False)[0].reshape(b, n, d)
            x = x + s.ca(s.n1(x), mem, mem, key_padding_mask=pad, need_weights=False)[0]
            return x + s.ff(s.n2(x))

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            s.vision = vision
            s.ep = mlp(20, d)
            if vision:
                s.vln, s.vp = nn.LayerNorm(512), nn.Linear(512, d)
                s.slot, s.pos = nn.Parameter(torch.randn(8, 1, d) * 0.02), nn.Parameter(torch.randn(1, 32, d) * 0.02)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True), enc,
                                          enable_nested_tensor=False)
            s.register_buffer("fr", torch.pi * 2.0 ** torch.arange(8))
            s.sp, s.temb = mlp(38, d), nn.Parameter(torch.randn(9, d) * 0.02)
            s.dec = nn.ModuleList([Dec() for _ in range(dec)])
            s.so = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
            s.stop = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def memory(s, V, valid, E):
            e = s.ep(E)[:, None]
            if not s.vision:
                return s.enc(e), None
            v = (s.vp(s.vln(V.float())) + s.slot + s.pos).flatten(1, 2)
            pad = torch.cat([valid.new_zeros(len(valid), 1), ~valid.repeat_interleave(32, 1)], 1)
            return s.enc(torch.cat([e, v], 1), src_key_padding_mask=pad), pad

        def steps(s, q):
            p = torch.cat([torch.zeros_like(q[..., :1, :]), q.float()], -2)
            u = torch.stack([(p[..., 0] - 24.0) / 32.0, p[..., 1] / 24.0], -1)[..., None] * s.fr
            dp = torch.diff(p, dim=-2, prepend=p[..., :1, :])
            return torch.cat([torch.sin(u).flatten(-2), torch.cos(u).flatten(-2), p[..., 0:1] / 32.0, p[..., 1:2] / 24.0, torch.sin(p[..., 2:3]),
                              torch.cos(p[..., 2:3]), torch.hypot(dp[..., 0:1], dp[..., 1:2]) / 8.0, dp[..., 2:3]], -1)

        def forward(s, mem, pad, q):
            """mem from memory(); q (B, Q, 8, 3) -> per-waypoint (B, Q, 8, 3) [road, agent, over] in metres."""
            B_, Q = q.shape[:2]
            x = (s.sp(s.steps(q)) + s.temb).flatten(1, 2)
            for l in s.dec:
                x = l(x, mem, pad)
            return s.so(x.reshape(B_, Q, 9, -1)[:, :, 1:]).float() * 2.0

        def stop_logit(s, mem):
            return s.stop(mem[:, 0]).float()[:, 0]
    return Head()


def build_refine(d=256):
    """Learned refine: (plan 24, schema output 24, schema gradient 24, ego 20) -> plan change (8, 3): 2 m tanh on x, y; 0.3 rad on yaw."""
    import torch
    import torch.nn as nn

    class Ref(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(92, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 24))
            nn.init.zeros_(s.f[-1].weight), nn.init.zeros_(s.f[-1].bias)

        def forward(s, q, c, gq, E):
            z = torch.cat([q.flatten(1) / 20.0, c.flatten(1).clamp(-5, 5) / 5.0, gq.flatten(1).clamp(-5, 5), E], 1)
            o = torch.tanh(s.f(z)).view(-1, 8, 3)
            return o * torch.tensor([2.0, 2.0, 0.3], device=q.device)
    return Ref()


# ---------------------------------------------------------------- data on the card
class Data:
    def __init__(self, dev, need_geo=True):
        import torch
        P = np.load(O / "prep.npz")
        self.P = {k: P[k] for k in P.files}
        tab = np.load(POOL / "tab.npz")
        self.names = tab["names"].astype(str)
        self.served = tab["served"].astype(np.float32)
        self.ego = tab["ego"].astype(np.float32)
        self.front = np.load(CR / "on1_PAID@warp" / "front.npy", mmap_mode="r")
        self.dev = dev
        self.geo = Geo(self.names, self.P["stop"], dev) if need_geo else None

    def tokens(self, rows):
        import torch
        r = np.asarray(rows)
        o = np.argsort(r)
        x = np.empty((len(r),) + self.front.shape[1:], np.float16)
        x[o] = self.front[r[o]]
        V = torch.from_numpy(x).to(self.dev, non_blocking=True)
        return V, V.abs().amax((2, 3)) > 0


def ego_std(D, rows):
    m, s = D.ego[rows].mean(0), D.ego[rows].std(0)
    return m, np.maximum(s, 1e-3)


def loss_fn(pred, Y, ok):
    """Weighted Huber on the clipped targets, emphasis near 0 (the decision band)."""
    import torch
    w = (1.0 + 3.0 * torch.exp(-Y.abs())) * ok.float()
    return (torch.nn.functional.huber_loss(pred, Y, reduction="none", delta=1.0) * w).sum() / w.sum().clamp_min(1.0)


# ---------------------------------------------------------------- train (GPU)
def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    dev = torch.device("cuda")
    torch.manual_seed(a.fold)
    D = Data(dev)
    Pp = D.P
    tr = np.flatnonzero((Pp["fold"] != a.fold) & (Pp["v0"] >= 0.5) & D.geo.road.ok.cpu().numpy())
    em, es = ego_std(D, tr)
    Et = torch.as_tensor((D.ego - em) / es, device=dev)
    Sv = torch.as_tensor(D.served, device=dev)
    stop_ok = torch.as_tensor(Pp["stop"][:, 3] > 0.5, device=dev)
    nets = {"vis": build_head(True).to(dev), "blind": build_head(False).to(dev)}
    opt = {k: torch.optim.AdamW(n.parameters(), lr=a.lr, weight_decay=0.01) for k, n in nets.items()}
    sch = {k: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=a.lr, total_steps=a.steps, pct_start=0.05) for k, o in opt.items()}
    print(f"fold {a.fold}: train rows {len(tr)}, params vis {sum(p.numel() for p in nets['vis'].parameters()) / 1e6:.2f} M", flush=True)
    rng = np.random.default_rng(a.fold)
    g = torch.Generator(device=dev)
    g.manual_seed(a.fold)
    ex = ThreadPoolExecutor(2)
    draw = lambda: rng.choice(tr, a.batch, replace=False)                          # noqa: E731
    nxt = [(r, ex.submit(D.tokens, r)) for r in (draw(), draw())]
    t0, log = time.time(), []
    for step in range(a.steps):
        rows, fut = nxt.pop(0)
        nxt.append((r2 := draw(), ex.submit(D.tokens, r2)))
        V, valid = fut.result()
        rt = torch.as_tensor(rows, device=dev)
        q = torch.cat([Sv[rt][:, None], perturb(Sv[rt], a.q - 1, g)], 1)
        with torch.no_grad():
            Y, ok, _ = D.geo.labels(q, rt)
        rec = {}
        for k, net in nets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mem, pad = net.memory(V if k == "vis" else None, valid, Et[rt])
                pred = net(mem, pad, q)
                sl = net.stop_logit(mem)
            L = loss_fn(pred, Y, ok) + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(sl, stop_ok[rt].float())
            opt[k].zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt[k].step(), sch[k].step()
            rec[k] = float(L)
        if step % 200 == 0 or step == a.steps - 1:
            log.append(dict(step=step, **rec, sec=round(time.time() - t0, 1)))
            print(json.dumps(log[-1]), flush=True)
    for n in nets.values():
        n.eval()
    # ---- learned refine on the frozen vision head: loss = true hinge (margin 0.3) of plan + delta + lam |delta|^2 (privileged, train only)
    net = nets["vis"]
    for p in net.parameters():
        p.requires_grad_(False)
    refs = {}
    for lam in LAMS:
        ref = build_refine().to(dev)
        o2 = torch.optim.AdamW(ref.parameters(), lr=3e-4, weight_decay=0.0)
        for step in range(a.ref_steps):
            rows, fut = nxt.pop(0)
            nxt.append((r2 := draw(), ex.submit(D.tokens, r2)))
            V, valid = fut.result()
            rt = torch.as_tensor(rows, device=dev)
            q = torch.cat([Sv[rt][:, None], perturb(Sv[rt], 3, g)], 1).flatten(0, 1)                # (4B, 8, 3)
            r4 = rt.repeat_interleave(4)
            with torch.no_grad():
                mem, pad = net.memory(V, valid, Et[rt])
            c, gq = schema_grad(net, mem.repeat_interleave(4, 0), None if pad is None else pad.repeat_interleave(4, 0), q)
            dq = ref(q, c, gq, Et[r4])
            L = (D.geo.viol(q + dq, r4, MARGIN) + lam * (dq[..., :2] ** 2).sum((1, 2))).mean()
            o2.zero_grad(set_to_none=True)
            L.backward()
            o2.step()
            if step % 200 == 0:
                print(json.dumps(dict(refine=lam, step=step, loss=float(L))), flush=True)
        refs[str(lam)] = ref.state_dict()
    out = O / f"f{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(vis=nets["vis"].state_dict(), blind=nets["blind"].state_dict(), refine=refs, em=em, es=es, log=log, cfg=vars(a)), out / "heads.pt")
    print(f"saved {out / 'heads.pt'} after {time.time() - t0:.0f} s", flush=True)


def schema_grad(net, mem, pad, q, create=False):
    """-> schema output c (n, 8, 3) at q and the gradient of V_hat = sum_k relu(0.3 - road_k) + relu(0.3 - agent_k) wrt q (n, 8, 3)."""
    import torch
    with torch.enable_grad():
        qq = q.detach().requires_grad_(True)
        c = net(mem, pad, qq[:, None])[:, 0]
        v = ((MARGIN - c[..., 0]).clamp_min(0) + (MARGIN - c[..., 1]).clamp_min(0)).sum()
        gq, = torch.autograd.grad(v, qq, create_graph=create)
    return c.detach(), gq.detach()


def clip_step(q0, q):
    import torch
    d = q[..., :2] - q0[..., :2]
    n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    d = d * (2.0 / n).clamp(max=1.0)
    return torch.cat([q0[..., :2] + d, q0[..., 2:3] + (q[..., 2:3] - q0[..., 2:3]).clamp(-0.3, 0.3)], -1)


# ---------------------------------------------------------------- eval (GPU)
def cmd_eval(a):
    import torch
    dev = torch.device("cuda")
    D = Data(dev)
    Pp = D.P
    ck = torch.load(O / f"f{a.fold}" / "heads.pt", map_location="cpu", weights_only=False)
    nets = {}
    for k in ("vis", "blind"):
        nets[k] = build_head(k == "vis").to(dev)
        nets[k].load_state_dict(ck[k])
        nets[k].eval()
    refs = {}
    for lam, sd in ck["refine"].items():
        refs[lam] = build_refine().to(dev)
        refs[lam].load_state_dict(sd)
        refs[lam].eval()
    Et = torch.as_tensor((D.ego - ck["em"]) / ck["es"], device=dev)
    n = len(D.names)
    plans = {"served": D.served, **{c: plans_of(c, n) for c in CKPTS}}
    te = np.flatnonzero((Pp["fold"] == a.fold) & D.geo.road.ok.cpu().numpy())
    if a.limit:
        te = np.sort(np.random.default_rng(0).choice(te, min(a.limit, len(te)), replace=False))
    res = dict(rows=te)
    # ---- (a): predictions and true labels for each plan on the fold's rows
    for pk, PL in plans.items():
        have = np.isfinite(PL[te]).all((1, 2))
        r_all = te[have]
        Yp = {k: np.full((len(r_all), 8, 3), np.nan, np.float32) for k in nets}
        Yt = np.full((len(r_all), 8, 3), np.nan, np.float32)
        T0 = np.full((len(r_all), 2), np.nan, np.float32)
        SL = {k: np.full(len(r_all), np.nan, np.float32) for k in nets}
        for j in range(0, len(r_all), a.bs):
            r = r_all[j:j + a.bs]
            rt = torch.as_tensor(r, device=dev)
            q = torch.as_tensor(PL[r], device=dev)[:, None]
            V, valid = D.tokens(r)
            with torch.no_grad():
                Y, ok, t0 = D.geo.labels(q, rt)
                Y = torch.where(ok, Y, torch.full_like(Y, np.nan))
                Yt[j:j + len(r)], T0[j:j + len(r)] = Y[:, 0].cpu().numpy(), t0[:, 0].cpu().numpy()
                for k, net in nets.items():
                    mem, pad = net.memory(V if k == "vis" else None, valid, Et[rt])
                    Yp[k][j:j + len(r)] = net(mem, pad, q)[:, 0].cpu().numpy()
                    SL[k][j:j + len(r)] = net.stop_logit(mem).cpu().numpy()
        res[f"a_{pk}_rows"] = r_all
        res[f"a_{pk}_true"], res[f"a_{pk}_t0"] = Yt, T0
        for k in nets:
            res[f"a_{pk}_pred_{k}"], res[f"a_{pk}_stop_{k}"] = Yp[k], SL[k]
        print(f"(a) {pk}: {len(r_all)} rows", flush=True)
    # ---- (b): corrections on PF and FS rows
    bset = te[(Pp["pf_off"] | Pp["pf_col"] | Pp["fs_b"])[te]]
    if a.limit:
        bset = bset[:a.limit // 4]
    for pk, PL in plans.items():
        have = np.isfinite(PL[bset]).all((1, 2))
        r_all = bset[have]
        res[f"b_{pk}_rows"] = r_all
        acc = {}
        for j in range(0, len(r_all), a.bs):
            r = r_all[j:j + a.bs]
            rt = torch.as_tensor(r, device=dev)
            q0 = torch.as_tensor(PL[r], device=dev)
            V, valid = D.tokens(r)
            with torch.no_grad():
                mem = {k: net.memory(V if k == "vis" else None, valid, Et[rt]) for k, net in nets.items()}
            outs = {"base": q0}
            for meth in ("vis", "blind", "oracle"):
                for eta in ETAS:
                    for K in KS:
                        q = q0.clone()
                        for _ in range(K):
                            if meth == "oracle":
                                with torch.enable_grad():
                                    qq = q.detach().requires_grad_(True)
                                    v = D.geo.viol(qq, rt, MARGIN).sum()
                                    gq, = torch.autograd.grad(v, qq)
                            else:
                                _, gq = schema_grad(nets[meth], *mem[meth], q)
                            q = clip_step(q0, q - eta * gq)
                        outs[f"{meth}_e{eta}_k{K}"] = q
            c, gq = schema_grad(nets["vis"], *mem["vis"], q0)
            for lam, ref in refs.items():
                with torch.no_grad():
                    outs[f"refine_l{lam}"] = clip_step(q0, q0 + ref(q0, c, gq, Et[rt]))
            vhat = ((MARGIN - c[..., 0]).clamp_min(0) + (MARGIN - c[..., 1]).clamp_min(0)).sum(1)
            acc.setdefault("vhat", []).append(vhat.cpu().numpy())
            with torch.no_grad():
                for key, q in outs.items():
                    m, d = D.geo.dense(q, rt)
                    okr, oka = D.geo.road.ok[rt][:, None], D.geo.ag.ok[rt][:, None]
                    road = torch.where(okr, m[:, 1:], torch.full_like(m[:, 1:], BIG))
                    ag = torch.where(oka, d[:, 1:], torch.full_like(d[:, 1:], BIG))
                    v = road.neg().clamp_min(0).sum(1) + ag.neg().clamp_min(0).sum(1)
                    dd = (q[..., :2] - q0[..., :2]).norm(dim=-1)
                    rec = np.stack([v.cpu().numpy(), road.amin(1).cpu().numpy(), ag.amin(1).clamp(max=BIG).cpu().numpy(),
                                    dd[:, -1].cpu().numpy(), dd.amax(1).cpu().numpy()], 1)
                    acc.setdefault(key, []).append(rec.astype(np.float32))
        for key, v in acc.items():
            res[f"b_{pk}_{key}"] = np.concatenate(v)
        print(f"(b) {pk}: {len(r_all)} rows, {len(acc)} settings", flush=True)
    np.savez(O / f"f{a.fold}" / "eval.npz", **res)
    print(f"saved {O / f'f{a.fold}' / 'eval.npz'}", flush=True)


# ---------------------------------------------------------------- report (CPU)
def cmd_report(a):
    from contact_head import auc_boot
    P = np.load(O / "prep.npz")
    scene, stop = P["scene"], P["stop"]
    E = {k: np.load(O / f"f{k}" / "eval.npz") for k in range(NF) if (O / f"f{k}" / "eval.npz").exists()}
    out = dict(folds=sorted(E))

    def cat(key, folds):
        return np.concatenate([E[f][key] for f in folds])
    # ---- (a), pooled over every fold (no selection in (a))
    fa = sorted(E)
    A = {}
    for pk in ("served",) + CKPTS:
        if f"a_{pk}_rows" not in E[fa[0]].files or not len(E[fa[0]][f"a_{pk}_rows"]):
            continue
        rows = cat(f"a_{pk}_rows", fa)
        Yt, T0 = cat(f"a_{pk}_true", fa), cat(f"a_{pk}_t0", fa)
        clean = (T0[:, 0] >= 0) & (T0[:, 1] >= 0)                       # not off / in contact at t0
        g = scene[rows]
        rec = dict(rows=int(len(rows)), clean=int(clean.sum()))
        for k in ("vis", "blind"):
            Yp = cat(f"a_{pk}_pred_{k}", fa)
            for ci, ch in enumerate(CH[:2]):
                okc = clean & np.isfinite(Yt[:, :, ci]).all(1)
                y = Yt[okc, :, ci].min(1) < 0
                s = -Yp[okc, :, ci].min(1)
                b = auc_boot(y, s, g[okc], n_boot=2000)
                rec[f"{ch}_auc_{k}"] = b
                per = []
                for w in range(8):
                    yw, pw = Yt[okc, w, ci], Yp[okc, w, ci]
                    band = (yw >= -1) & (yw <= 2)
                    from contact_head import auc as auc1
                    per.append(dict(t=0.5 * (w + 1), auc=auc1(yw < 0, -pw), mae_band=float(np.abs(pw - yw)[band].mean()) if band.any() else None,
                                    n_band=int(band.sum()), pos=int((yw < 0).sum())))
                rec[f"{ch}_per_wp_{k}"] = per
            st = stop[rows, 3] > 0.5
            okc = clean & st
            if okc.any():
                yo, po = Yt[okc, :, 2], Yp[okc, :, 2]
                rec[f"over_auc_{k}"] = auc_boot(yo.max(1) > 1.0, po.max(1), g[okc], n_boot=2000)
                rec[f"over_mae_2s_{k}"] = float(np.abs(po[:, 3] - yo[:, 3]).mean())
            sl = cat(f"a_{pk}_stop_{k}", fa)
            from contact_head import auc as auc1
            rec[f"stop_exists_auc_{k}"] = auc1(stop[rows, 3] > 0.5, sl)
        A[pk] = rec
    out["a"] = A
    s = A["served"]
    out["lines_a"] = dict(
        A1=dict(vis=s["road_auc_vis"]["auc"], blind=s["road_auc_blind"]["auc"],
                passed=bool(s["road_auc_vis"]["auc"] >= 0.90 and s["road_auc_vis"]["auc"] >= s["road_auc_blind"]["auc"] + 0.05)),
        A2=dict(vis=s["agent_auc_vis"]["auc"], blind=s["agent_auc_blind"]["auc"],
                passed=bool(s["agent_auc_vis"]["auc"] >= 0.85 and s["agent_auc_vis"]["auc"] >= s["agent_auc_blind"]["auc"] + 0.05)),
        A3=dict(road=s["road_per_wp_vis"][3]["mae_band"], agent=s["agent_per_wp_vis"][3]["mae_band"],
                passed=bool(s["road_per_wp_vis"][3]["mae_band"] <= 0.5 and s["agent_per_wp_vis"][3]["mae_band"] <= 0.5)),
        A4=dict(vis=s.get("over_auc_vis", {}).get("auc"), passed=bool((s.get("over_auc_vis", {}).get("auc") or 0) >= 0.85)))

    # ---- (b): select on fold 0, read folds 1..3
    def bstats(folds, pk, key):
        rows = cat(f"b_{pk}_rows", folds)
        base, r = cat(f"b_{pk}_base", folds), cat(f"b_{pk}_{key}", folds)
        pf = (P["pf_off"] | P["pf_col"])[rows] & (base[:, 0] > 0)
        fs = P["fs_b"][rows] & ~(P["pf_off"] | P["pf_col"])[rows]
        fs_clean = fs & (base[:, 0] == 0)
        red = r[pf, 0] < base[pf, 0] - 1e-6
        clr = r[pf, 0] <= 0
        vh = cat(f"b_{pk}_vhat", folds)
        thr = float(np.quantile(vh[fs_clean], 0.98)) if fs_clean.any() else np.inf
        return dict(flag_thr_fs98=thr, pf_flag_recall=float((vh[pf] > thr).mean()) if pf.any() else None,
                    pf_vhat_pos=float((vh[pf] > 0).mean()) if pf.any() else None, fs_vhat_pos=float((vh[fs_clean] > 0).mean()) if fs_clean.any() else None,
                    pf=int(pf.sum()), pf_scenes=int(len(np.unique(scene[rows][pf]))), reduced=float(red.mean()) if pf.any() else None,
                    cleared=float(clr.mean()) if pf.any() else None, pf_d4_med=float(np.median(r[pf, 3])) if pf.any() else None,
                    pf_off_reduced=float(red[(P["pf_off"][rows])[pf]].mean()) if (P["pf_off"][rows] & pf).any() else None,
                    pf_col_reduced=float(red[(P["pf_col"][rows] & ~P["pf_off"][rows])[pf]].mean()) if (P["pf_col"][rows] & ~P["pf_off"][rows] & pf).any() else None,
                    fs=int(fs.sum()), fs_d4_med=float(np.median(r[fs, 3])), fs_d4_p90=float(np.quantile(r[fs, 3], 0.9)),
                    fs_moved_02=float((r[fs, 4] > 0.2).mean()), fs_new_viol=float((r[fs_clean, 0] > 0).mean()) if fs_clean.any() else None)
    keys = [k[len("b_served_"):] for k in E[fa[0]].files if k.startswith("b_served_") and k not in ("b_served_rows", "b_served_base", "b_served_vhat")]
    B = {}
    for pk in ("served",) + CKPTS:
        if f"b_{pk}_base" not in E[fa[0]].files:
            continue
        sel_f = [0] if 0 in E else fa[:1]
        rd_f = [f for f in fa if f not in sel_f]
        st0 = {k: bstats(sel_f, pk, k) for k in keys}
        ok = [k for k, v in st0.items() if v["fs_d4_med"] <= 0.2 and v["fs_d4_p90"] <= 0.5 and (v["fs_new_viol"] or 0) <= 0.01 and not k.startswith("oracle")]
        best = max(ok, key=lambda k: st0[k]["reduced"] or -1) if ok else None
        rd = {k: bstats(rd_f, pk, k) for k in keys} if rd_f else {}
        orc = [k for k in keys if k.startswith("oracle")]
        best_or = max(orc, key=lambda k: (st0[k]["reduced"] or -1) if st0[k]["fs_d4_med"] <= 0.2 else -1)
        B[pk] = dict(select_folds=sel_f, read_folds=rd_f, selected=best, oracle_selected=best_or, fold0=st0, read=rd)
        if best and rd:
            r_, o_ = rd[best], rd[best_or]
            B[pk]["lines_b"] = dict(B1=dict(reduced=r_["reduced"], cleared=r_["cleared"], passed=bool((r_["reduced"] or 0) >= 0.5 and (r_["cleared"] or 0) >= 0.25)),
                                    B2=dict(d4_med=r_["fs_d4_med"], d4_p90=r_["fs_d4_p90"], new_viol=r_["fs_new_viol"],
                                            passed=bool(r_["fs_d4_med"] <= 0.2 and r_["fs_d4_p90"] <= 0.5 and (r_["fs_new_viol"] or 0) <= 0.01)),
                                    B3=dict(ratio=(r_["reduced"] or 0) / max(o_["reduced"] or 1e-9, 1e-9), passed=bool((r_["reduced"] or 0) >= 0.5 * (o_["reduced"] or 0))))
    out["b"] = B
    (O / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(dict(lines_a=out["lines_a"], b={k: dict(selected=v["selected"], lines=v.get("lines_b")) for k, v in B.items()}), indent=1, default=float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prep")
    f = sp.add_parser("fwd")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--shard", default="0/1")
    f.add_argument("--bs", type=int, default=256)
    f.add_argument("--threads", type=int, default=0)
    t = sp.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--ref-steps", type=int, default=1500)
    t.add_argument("--batch", type=int, default=96)
    t.add_argument("--q", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    e = sp.add_parser("eval")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--bs", type=int, default=128)
    e.add_argument("--limit", type=int, default=0, help="smoke: at most this many fold rows")
    sp.add_parser("report")
    a = p.parse_args()
    dict(prep=cmd_prep, fwd=cmd_fwd, train=cmd_train, eval=cmd_eval, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
