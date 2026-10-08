"""op_parity margin critic (plans/2026-10-08-margin-critic-prereg.md): how well can a head on the frozen representation predict the map-SDF footprint
margin of an arbitrary candidate trajectory when it is trained on exactly that quantity (map geometry as the training label, navtrain scale), and how much
of decision 187's privileged selection gain does the predicted margin recover on the navtest > 20 deg tokens.

  extract   (GPU, pool job per shard) hidden state (select_4, mean) of every navtrain token from the fold model that held its log out -> <tag>/feat/s<i>.npz
  bank      (CPU) 48 training trajectories per token: F33 around the held-out plan + random transforms of the plan and of the logged future -> <tag>/bank/s<i>.npz
  train     (GPU, pool job per arm) implicit SDF field with footprint queries; early stop on navsim/op-parity-full-dev; predicted margins of the 33 navtest
            candidates x 2 SH30 seeds -> <tag>/pred/<arm>.npz (Q, same layout as turn_selinput's margins.npz M); label gate against that M
  select    (CPU) decision 187's rule / ridge / tree heads (turn_selinput.unit2, same folds and repeats) with Q in place of the map margin -> <tag>/select.pkl
  report    readings 1-4, verdict, tables, figure
Map geometry is a training label only; at test time the head sees frozen vision tokens, SH30's hidden state, ego + command and the candidate.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, hashlib, json, pickle, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import turn_ceiling as TC  # noqa: E402

D = TC.D
OUT = D / "runs/op_parity/margin_critic"
RES = _R / "experiments/op_parity/results/margin_critic"
FIG = _R / "experiments/op_parity/figs/margin_critic"
CR = D / "runs/op_parity/cache"
NSH, K = 12, 5
SMOKE_SHARD = 2                                         # a shard that also has WA-Cf features (reference arm)
TIDX = [10, 20, 30, 40]                                 # 1 / 2 / 3 / 4 s on the 0.1 s grid
CLIP, PCLIP = (-2.0, 4.0), (-3.0, 6.0)                  # margin clip (= decision 187), point SDF clip of the training target
NB, NF33, NPLAN = 48, 33, 41                            # bank: c00-c32 = F33 on the held-out plan, 33-40 random on the plan, 41-47 random on the logged future
RNG_O, RNG_K, RNG_V = (-1.5, 1.5), (0.6, 1.6), (0.5, 1.5)
X0, Y0, RS, NH, NW = -8.0, -24.0, 0.5, 128, 96          # label raster (sc_analyze.X0 / Y0 / RES)
ARMS = {                                                # v: c8 = Cinque 8 frames x 32 tokens, c1 = newest frame, wa = WA-Cf tokens, None = no vision
    "MC": dict(v="c8", h=1, frac=1.0, steps=8000, vram=46), "MC-30": dict(v="c8", h=1, frac=0.3, steps=8000, vram=24),
    "MC-10": dict(v="c8", h=1, frac=0.1, steps=4000, vram=14), "MC-3": dict(v="c8", h=1, frac=0.03, steps=4000, vram=12),
    "MC-V": dict(v="c8", h=0, frac=1.0, steps=8000, vram=46), "MC-E": dict(v=None, h=0, frac=1.0, steps=8000, vram=10),
    "R-WA": dict(v="wa", h=0, frac=1.0, steps=6000, vram=12, shards=(2, 3, 4)), "R-C1": dict(v="c1", h=0, frac=1.0, steps=6000, vram=12, shards=(2, 3, 4))}
CURVE = ("MC-3", "MC-10", "MC-30", "MC")
BATCH, NTRAJ, LR, WD, WARM, EVAL = 256, 6, 3e-4, 0.05, 300, 500
REF = dict(map_dac=0.916, curb_dac=0.629, map_rep=0.706, curb_rep=0.546, ceiling=9.55)
THR = dict(half=0.5, auc_margin=0.05, rise_auc=0.02, rise_mae=0.05, floor_auc=0.03, ref_auc=0.05)


def fold_of_log(log):
    return int(hashlib.sha256(f"cf{K}|{log}".encode()).hexdigest(), 16) % K


def shard(i):
    return f"navtrain_full.s{i}of{NSH}"


def tdir(tag, *p):
    return OUT.joinpath(tag, *p)


# ---------------------------------------------------------------- 1. hidden state of the held-out fold model (GPU)
def cmd_extract(a):
    import torch
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    from jevdrive.data import splits
    from jevdrive.run import Run
    import nt_labels as NL
    import tsn_extract as TX
    out = tdir(a.tag, "feat", f"s{a.shard}.npz")
    with Run("op_parity", f"margin_critic/extract-{a.tag}-s{a.shard}", seed=0, config=vars(a)) as run:
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        trs = [splits.load(f"navsim/op-parity-cf{K}f{j}-train") for j in range(K)]
        ntr, nt = splits.load("navsim/navtrain"), splits.load("navsim/navtest")
        for s in (*trs, ntr, nt):
            run.use_split(s)
        data = shard(a.shard)
        S = T.Store([data], dev, need_side=(N.cache_dir(data, "gimm") / "side.npy").exists(), frames="warp")
        n = min(S.n, a.limit) if a.limit else S.n
        names, logs = S.tab["names"][:n], S.tab["log"][:n]
        assert ntr.mask(names).all() and not nt.mask(names).any(), "shard tokens must be navtrain and not navtest"
        fold = np.array([fold_of_log(l) for l in logs])
        h4, hm, diff = np.zeros((n, 512), np.float16), np.zeros((n, 512), np.float16), np.zeros(n)
        speed = S.tb["speed"][:n]
        for j in range(K):
            idx = np.flatnonzero(fold == j)
            if not len(idx):
                continue
            assert not trs[j].mask(logs[idx]).any(), f"fold {j}: a held-out log is in the fold's training split"
            mj, model = TX.load_model(T, N, resolve, j, dev)
            P, h4[idx], hm[idx], _ = TX.run_rows(model, S, idx, run)
            ref = np.load(NL.ol(a.shard, "plans", f"{NL.stem(mj.name)}.npz"))
            assert ref["names"][:n].tolist() == names.tolist()
            diff[idx] = TX.plan_diff(ref["plan_pos"][idx], P)
            run.info("fold %d: %d rows, max |dplan| vs stored %.3g m", j, len(idx), diff[idx][speed[idx] >= 0.5].max(initial=0))
            del model
            torch.cuda.empty_cache()
        ok = speed >= 0.5
        gate = dict(shard=a.shard, n=int(n), max_diff_m=float(diff[ok].max(initial=0)), tol_m=TX.TOL_M)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), tokens=names, log=logs, fold=fold, select_4=h4, mean=hm, gate=np.array(json.dumps(gate)))
        out.with_suffix(".tmp.npz").rename(out)
        run.summary.update(gate)
        assert gate["max_diff_m"] < TX.TOL_M, f"G-leak: extraction differs from the stored fold-model plan by {gate['max_diff_m']} m"


# ---------------------------------------------------------------- 2. trajectory bank (CPU)
def rtransform(P, o, k, v):
    """turn_ceiling.transform with per-row parameters (o, k, v of shape (N,)): curvature gain, lateral offset, speed scaling."""
    Q = np.concatenate([np.zeros((len(P), 1, 3)), np.asarray(P, np.float64)], 1)
    Q[..., 2] = np.unwrap(Q[..., 2], axis=1)
    Q = TC.offset(TC.curv(Q, k[:, None]), o[:, None])
    for i in range(len(Q)):
        Q[i:i + 1] = TC.speed(Q[i:i + 1], float(v[i]))
    Q[..., 2] = np.angle(np.exp(1j * Q[..., 2]))
    return Q[:, 1:]


def _bank(u):
    i, tag, limit = u
    out = tdir(tag, "bank", f"s{i}.npz")
    tab = np.load(CR / shard(i) / "tab.npz")
    n = min(limit, len(tab["names"])) if limit else len(tab["names"])
    names, logs = tab["names"][:n], tab["log"][:n]
    fold = np.array([fold_of_log(l) for l in logs])
    P = np.zeros((n, 8, 3))
    for j in range(K):
        z = np.load(D / f"runs/bench/ol/{shard(i)}/preds/CF{K}f{j}-F-s0-warp__base.npz")
        assert (z["tokens"][:n] == names).all()
        P[fold == j] = z["poses"][:n][fold == j]
    fut = tab["fut"][:n].astype(np.float64)
    has = ~np.isnan(fut[:, 0, 0])
    base_f = np.where(has[:, None, None], np.nan_to_num(fut), P)
    rng = np.random.default_rng(1000 + i)
    traj = np.zeros((n, NB, 8, 3), np.float32)
    for c, (_, o, k, v) in enumerate(TC.candidates()):
        traj[:, c] = TC.transform(P, o, k, v)
    assert np.array_equal(traj[:, 0], P.astype(np.float32))
    for c in range(NF33, NB):
        o, k, v = rng.uniform(*RNG_O, n), np.exp(rng.uniform(*np.log(RNG_K), n)), np.exp(rng.uniform(*np.log(RNG_V), n))
        traj[:, c] = rtransform(P if c < NPLAN else base_f, o, k, v)
    assert np.isfinite(traj).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".tmp.npz"), tokens=names, log=logs, fold=fold, traj=traj, ego=tab["ego"][:n].astype(np.float32), has_fut=has)
    out.with_suffix(".tmp.npz").rename(out)
    return i, n


def cmd_bank(a):
    import multiprocessing as mp
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", f"margin_critic/bank-{a.tag}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        units = [(i, a.tag, a.limit) for i in ([SMOKE_SHARD] if a.tag == "smoke" else range(NSH))]
        r = par.pmap(_bank, units, workers=max(1, min(n_cpus() // 2, len(units))), run=run, desc="bank shards", mp_context=mp.get_context("fork"),
                     skip=lambda u: tdir(u[1], "bank", f"s{u[0]}.npz").exists() and not a.force)
        r.raise_if_failed()
        run.summary.update(n_shards=len(units))


# ---------------------------------------------------------------- 3. the head (GPU)
def _torch_ops():
    import torch
    import torch.nn.functional as F
    import sc_analyze as SC
    assert (SC.X0, SC.Y0, SC.RES) == (X0, Y0, RS)

    def footprint(P):
        """(..., 8, 3) rear-axle poses -> corner points (..., 41, 4, 2); sc_analyze.footprint in torch."""
        IM = torch.as_tensor(SC.interp_matrix(), dtype=P.dtype, device=P.device)
        CO = torch.as_tensor(SC.CORNERS, dtype=P.dtype, device=P.device)
        d = torch.einsum("kj,...jc->...kc", IM, torch.cat([torch.zeros_like(P[..., :1, :]), P], -2))
        c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
        return torch.stack([d[..., 0:1] + c * CO[:, 0] - s * CO[:, 1], d[..., 1:2] + s * CO[:, 0] + c * CO[:, 1]], -1)

    def sample(sdf, pts):
        """sdf (B, 128, 96), pts (B, Q, 2) metres -> (B, Q): bilinear, border clamp (= sc_analyze.sdf_at)."""
        g = torch.stack([2 * (pts[..., 1] - Y0) / (RS * NW) - 1, 2 * (pts[..., 0] - X0) / (RS * NH) - 1], -1)
        return F.grid_sample(sdf[:, None].float(), g[:, :, None].float(), mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]

    def margins(v):
        """point values (..., 41, 4) -> cumulative min footprint margin at 1 / 2 / 3 / 4 s (..., 4), unclipped."""
        return torch.cummin(v.min(-1).values, -1).values[..., TIDX]
    return footprint, sample, margins


def build_model(nv, use_h, d=256):
    import torch
    import torch.nn as nn

    class XL(nn.Module):
        def __init__(s):
            super().__init__()
            s.n1, s.n2, s.att = nn.LayerNorm(d), nn.LayerNorm(d), nn.MultiheadAttention(d, 4, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, q, mem):
            q = q + s.att(s.n1(q), mem, mem, need_weights=False)[0]
            return q + s.ff(s.n2(q))

    class MC(nn.Module):
        def __init__(s):
            super().__init__()
            s.nv, s.use_h = nv, use_h
            if nv:
                s.vln, s.vp, s.vpos = nn.LayerNorm(512), nn.Linear(512, d), nn.Parameter(torch.randn(nv, d) * 0.02)
            if use_h:
                s.hp, s.hpos = nn.Linear(512, d), nn.Parameter(torch.randn(2, d) * 0.02)
            s.ep = nn.Sequential(nn.Linear(20, d), nn.GELU(), nn.Linear(d, d))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, activation="gelu", batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
            s.register_buffer("fr", torch.pi * 2.0 ** torch.arange(8))
            s.qp = nn.Sequential(nn.Linear(32, d), nn.GELU(), nn.Linear(d, d))
            s.x = nn.ModuleList([XL(), XL()])
            s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def memory(s, V, H, E):
            t = [s.ep(E)[:, None]]
            if s.nv:
                t.append(s.vp(s.vln(V.float())) + s.vpos)
            if s.use_h:
                t.append(s.hp(H) + s.hpos)
            return s.enc(torch.cat(t, 1))

        def field(s, mem, pts):
            with torch.autocast("cuda", enabled=False):                                                        # the high frequencies need fp32
                u = torch.stack([(pts[..., 0] - 24.0) / 32.0, pts[..., 1] / 24.0], -1).float()[..., None] * s.fr  # (B, Q, 2, 8)
                ff = torch.cat([torch.sin(u), torch.cos(u)], -1).flatten(-2)
            q = s.qp(ff)
            for l in s.x:
                q = l(q, mem)
            return s.out(q).squeeze(-1).float()
    return MC()


def _vision(kind, data, rows, tokens):
    """(len(rows), T, 512) fp16 vision tokens of the cache rows `rows` of one data dir."""
    if kind in ("c8", "c1"):
        f = np.load(CR / f"{data}@warp" / "front.npy", mmap_mode="r")
        x = np.asarray(f[rows]) if kind == "c8" else np.asarray(f[rows, -1])
        return np.ascontiguousarray(x.reshape(len(rows), -1, 512))
    out, hit = np.zeros((len(rows), 32, 512), np.float16), np.zeros(len(rows), bool)
    pos = {t: i for i, t in enumerate(tokens.tolist())}
    for f in sorted((D / "runs/op_probe/feats/WA").glob(f"{data}.s*of3.npz")):
        z = np.load(f)
        ix = np.array([pos.get(t, -1) for t in z["tokens"].tolist()])
        out[ix[ix >= 0]], hit[ix[ix >= 0]] = z["Cf"][ix >= 0], True
    assert hit.all(), f"WA-Cf features missing for {(~hit).sum()} rows of {data}"
    return out


def frac_logs(logs, frac):
    lg = sorted(set(logs.tolist()), key=lambda l: hashlib.sha256(f"mc|{l}".encode()).hexdigest())
    return set(lg[: max(1, int(np.ceil(frac * len(lg))))])


def cmd_train(a):
    import torch
    import torch.nn.functional as F
    from jevdrive.data import splits
    from jevdrive.run import Run
    import sc_analyze as SC
    import turn_selinput as TS
    spec, smoke = ARMS[a.arm], a.tag == "smoke"
    torch.set_num_threads(4)
    with Run("op_parity", f"margin_critic/train-{a.tag}-{a.arm}", seed=a.seed, config=dict(vars(a), **spec)) as run:
        dev = torch.device("cuda")
        s_tr, s_dv, s_nt, s_ntr = (splits.load(f"navsim/{x}") for x in ("op-parity-full-train", "op-parity-full-dev", "navtest", "navtrain"))
        for s in (s_tr, s_dv, s_nt, s_ntr):
            run.use_split(s)
        splits.check_disjoint(s_tr, s_dv)
        footprint, sample, margins = _torch_ops()
        t0 = time.time()
        # ---- navtrain rows
        zs = np.load(D / "runs/op_probe/labels/navtrain_all.npz")
        sidx = {t: i for i, t in enumerate(zs["tokens"].tolist())}
        sdf_all, ok_all = zs["sdf"], zs["ok"]
        shards = [SMOKE_SHARD] if smoke else list(spec.get("shards", range(NSH)))
        B = [np.load(tdir(a.tag, "bank", f"s{i}.npz")) for i in shards]
        tok, log = np.concatenate([b["tokens"] for b in B]), np.concatenate([b["log"] for b in B])
        assert s_ntr.mask(tok).all() and not s_nt.mask(tok).any()
        si = np.array([sidx[t] for t in tok])
        if smoke:                                                                   # the smoke rows hold too few op-parity-full-dev tokens: a 20% log split of them
            lg = np.unique(log)
            dvl = set(lg[::5].tolist())
            dv = np.array([l in dvl for l in log]) & ok_all[si]
            tr = ~np.array([l in dvl for l in log]) & ok_all[si]
        else:
            tr, dv = s_tr.mask(tok) & ok_all[si], s_dv.mask(tok) & ok_all[si]
            if spec["frac"] < 1:
                keep = frac_logs(log[tr], spec["frac"])
                tr &= np.array([l in keep for l in log])
        assert not set(log[tr].tolist()) & set(log[dv].tolist()), "train and dev share a log"
        use = np.flatnonzero(tr | dv)
        off = np.cumsum([0] + [len(b["tokens"]) for b in B])
        G = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        traj, ego = G(np.concatenate([b["traj"] for b in B])[use]), np.concatenate([b["ego"] for b in B])[use]
        sdf = G(sdf_all[si[use]], torch.float16)
        Hraw = None
        if spec["h"]:
            Fz = [np.load(tdir(a.tag, "feat", f"s{i}.npz")) for i in shards]
            assert all((f["tokens"][:len(b["tokens"])] == b["tokens"]).all() for f, b in zip(Fz, B))
            Hraw = np.concatenate([np.stack([f["select_4"][:len(b["tokens"])], f["mean"][:len(b["tokens"])]], 1) for f, b in zip(Fz, B)])[use].astype(np.float32)
        V = None
        if spec["v"]:
            nvt = 256 if spec["v"] == "c8" else 32
            V = torch.empty((len(use), nvt, 512), dtype=torch.float16, device=dev)
            for k_, i in enumerate(shards):
                m = np.flatnonzero((use >= off[k_]) & (use < off[k_ + 1]))
                rows = use[m] - off[k_]
                for c in range(0, len(m), 2048):
                    V[m[c:c + 2048]] = G(_vision(spec["v"], shard(i), rows[c:c + 2048], B[k_]["tokens"][rows[c:c + 2048]]), torch.float16)
        itr, idv = np.flatnonzero(tr[use]), np.flatnonzero(dv[use])
        emu, esd = ego[itr].mean(0), ego[itr].std(0) + 1e-6
        E = G((ego - emu) / esd)
        if spec["h"]:
            hmu, hsd = Hraw[itr].mean(0), Hraw[itr].std(0) + 1e-6
            H = G((Hraw - hmu) / hsd)
        run.info("rows: train %d (%d logs), dev %d (%d logs); loaded in %.0f s", len(itr), len(set(log[use][itr].tolist())), len(idv), len(set(log[use][idv].tolist())), time.time() - t0)
        # ---- model
        torch.manual_seed(a.seed)
        net = build_model(V.shape[1] if V is not None else 0, bool(spec["h"])).to(dev)
        steps = 300 if smoke else spec["steps"]
        ev = 100 if smoke else EVAL
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
        AC = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

        def predict(Vx, Hx, Ex, tj, chunk=96):
            """tj (m, C, 8, 3) -> predicted margins (m, C, 4), unclipped."""
            net.eval()
            out = []
            with torch.no_grad(), AC():
                for c in range(0, len(tj), chunk):
                    sl = slice(c, c + chunk)
                    pts = footprint(tj[sl])
                    mem = net.memory(Vx[sl] if Vx is not None else None, Hx[sl] if Hx is not None else None, Ex[sl])
                    out.append(margins(net.field(mem, pts.flatten(1, 3)).view(pts.shape[:-1])))
            net.train()
            return torch.cat(out)

        def true_margins(sd, tj, chunk=512):
            out = []
            for c in range(0, len(tj), chunk):
                pts = footprint(tj[c:c + chunk])
                out.append(margins(sample(sd[c:c + chunk], pts.flatten(1, 3)).view(pts.shape[:-1])))
            return torch.cat(out)
        gi = torch.as_tensor(idv, device=dev)
        dv_true = true_margins(sdf[gi], traj[gi, :NF33]).clamp(*CLIP)
        const_mae = float((dv_true[..., 3] - true_margins(sdf[torch.as_tensor(itr[:20000], device=dev)], traj[torch.as_tensor(itr[:20000], device=dev), :NF33]).clamp(*CLIP)[..., 3].median()).abs().mean())
        best, best_mae, curve = None, np.inf, []
        gtr = torch.as_tensor(itr, device=dev)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        t1 = time.time()
        for step in run.tqdm(range(steps), desc=f"train {a.arm}"):
            b = gtr[torch.randint(len(gtr), (BATCH,), device=dev, generator=gen)]
            tj = traj[b[:, None], torch.randint(NB, (BATCH, NTRAJ), device=dev, generator=gen)]
            pts = footprint(tj)                                                                                # (B, T, 41, 4, 2)
            y = sample(sdf[b], pts.flatten(1, 3)).view(pts.shape[:-1])
            with AC():
                p = net.field(net.memory(V[b] if V is not None else None, H[b] if spec["h"] else None, E[b]), pts.flatten(1, 3)).view(pts.shape[:-1])
            lp = F.huber_loss(p, y.clamp(*PCLIP), delta=0.5)
            lm = F.huber_loss(margins(p).clamp(*CLIP), margins(y).clamp(*CLIP), delta=0.5)
            opt.zero_grad(set_to_none=True)
            (lp + lm).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(), sched.step()
            if step % 50 == 0:
                run.scalar("loss/point", float(lp), step), run.scalar("loss/margin", float(lm), step)
            if (step + 1) % ev == 0 or step + 1 == steps:
                q = predict(V[gi] if V is not None else None, H[gi] if spec["h"] else None, E[gi], traj[gi, :NF33]).clamp(*CLIP)
                mae = float((q[..., 3] - dv_true[..., 3]).abs().mean())
                curve.append((step + 1, mae, float(lp), float(lm)))
                run.scalar("dev/mae4s", mae, step + 1)
                run.info("step %d: loss point %.4f margin %.4f, dev 4 s margin MAE %.4f m (constant %.4f), %.2f it/s", step + 1, float(lp), float(lm), mae, const_mae, (step + 1) / (time.time() - t1))
                if mae < best_mae:
                    best_mae, best = mae, {k: v.detach().clone() for k, v in net.state_dict().items()}
        its = steps / (time.time() - t1)
        net.load_state_dict(best)
        # ---- navtest > 20 deg: the 33 candidates x 2 SH30 seeds
        tk, _ = TC.bucket_tokens()
        assert s_nt.mask(tk).all() and not s_ntr.mask(tk).any()
        tab = np.load(CR / "lb_navtest/tab.npz")
        row = {t: i for i, t in enumerate(tab["names"].tolist())}
        r = np.array([row[t] for t in tk])
        o = np.argsort(r)
        Vt = None
        if spec["v"]:
            Vt = torch.empty((len(tk), V.shape[1], 512), dtype=torch.float16, device=dev)
            Vt[torch.as_tensor(o, device=dev)] = G(_vision(spec["v"], "lb_navtest", r[o], tab["names"][r[o]]), torch.float16)
        Et = G((tab["ego"][r] - emu) / esd)
        Z = np.load(TC.OUT / "poses.npz")
        assert np.array_equal(Z["tokens"], tk)
        lab = np.load(TS.LAB)
        lt = {t: i for i, t in enumerate(lab["tokens"].tolist())}
        sdt = G(lab["sdf"][[lt[t] for t in tk]], torch.float16)
        nc = len(TC.candidates())
        Q, Mm = np.zeros((2, nc, len(tk), 4), np.float32), np.zeros((2, nc, len(tk), 4), np.float32)
        fp_err = 0.0
        for s in TC.SEEDS:
            tj = G(np.stack([Z[TC.key(s, c)] for c in range(nc)], 1).astype(np.float64), torch.float64)       # (n, 33, 8, 3)
            Ht = None
            if spec["h"]:
                h = np.load(TS.HID.format(s))
                assert (h["names"] == tab["names"]).all()
                Ht = G((np.stack([h["select_4"][r], h["mean"][r]], 1).astype(np.float32) - hmu) / hsd)
            fp_err = max(fp_err, float(np.abs(footprint(tj[:200, 0]).cpu().numpy() - SC.footprint(Z[TC.key(s, 0)][:200].astype(np.float64))).max()))
            Mm[s] = true_margins(sdt, tj.float()).clamp(*CLIP).permute(1, 0, 2).cpu().numpy()
            Q[s] = predict(Vt, Ht, Et, tj.float()).clamp(*CLIP).permute(1, 0, 2).cpu().numpy()
        M187 = np.load(TS.OUT / "margins.npz")
        assert np.array_equal(M187["tokens"], tk)
        gate = dict(label_max_abs_diff_m=float(np.abs(Mm - M187["M"]).max()), footprint_max_abs_diff_m=fp_err)
        gate["ok"] = bool(gate["label_max_abs_diff_m"] <= 0.02 and fp_err <= 1e-3)
        out = tdir(a.tag, "pred", f"{a.arm}.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), Q=Q, tokens=tk, curve=np.array(curve), meta=np.array(json.dumps(dict(
            arm=a.arm, n_train=int(len(itr)), n_train_logs=len(set(log[use][itr].tolist())), n_dev=int(len(idv)), best_dev_mae=best_mae, const_dev_mae=const_mae,
            it_s=its, steps=steps, load_s=t1 - t0, gate=gate, vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30))))
        out.with_suffix(".tmp.npz").rename(out)
        torch.save(best, tdir(a.tag, "pred", f"{a.arm}.pt"))
        run.summary.update(best_dev_mae=best_mae, const_dev_mae=const_mae, it_s=its, n_train=int(len(itr)), vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **gate)
        run.info("gate %s; dev MAE %.4f (constant %.4f); %.2f it/s; peak VRAM %.1f GB", gate, best_mae, const_mae, its, torch.cuda.max_memory_allocated() / 2 ** 30)
        assert gate["ok"], f"G-label failed: {gate}"
        assert best_mae < const_mae or a.arm == "MC-E", "the head does not beat the constant predictor on dev"


# ---------------------------------------------------------------- 4. decision 187's heads on the predicted margin (CPU)
def hook(TS, TD, F, sources, tag):
    """Register Q1 / Q2 / Q3 arms per margin source in turn_selinput's tables (in memory only; forked workers inherit them)."""
    _D = TS._D
    for q in sources:
        Q = np.clip(np.load(tdir(tag, "pred", f"{q}.npz"))["Q"], *CLIP)
        for f, cv in TS.FKS:
            fk = TD.fkey(f, cv)
            m = np.ascontiguousarray(Q[:, F[f]].transpose(0, 2, 1, 3)).astype(np.float64)
            _D[f"Q@{q}|{fk}"], _D[f"Q@{q}flat|{fk}"] = m, m.reshape(m.shape[0], m.shape[1], -1)
        TS.MARG.update({f"Q1@{q}": f"Q@{q}", f"Q2@{q}": f"Q@{q}", f"Q3@{q}": f"Q@{q}"})
        TS.L_ARMS[f"Q1@{q}"] = ["ego", "plan", f"Q@{q}flat"]
        TS.G_ARMS, TS.R_ARMS = (*TS.G_ARMS, f"Q2@{q}"), (*TS.R_ARMS, f"Q3@{q}")

    def xof(streams, k, fk):
        return np.concatenate([_D[s][..., :k] if s in TS.PCA_STREAMS else _D[f"{s}|{fk}"] if s.endswith("flat") else _D[s] for s in streams], -1)
    TS.xof = xof


def sources_of(tag):
    return [q for q in ARMS if tdir(tag, "pred", f"{q}.npz").exists()]


def cmd_select(a):
    import multiprocessing as mp
    from threadpoolctl import threadpool_limits
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    import turn_dewater as TD
    import turn_selinput as TS
    with Run("op_parity", f"margin_critic/select-{a.tag}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        C, tok, dyaw, log, X, F = TS.prepare()
        src = sources_of(a.tag)
        hook(TS, TD, F, src, a.tag)
        f0 = TS.FK[0]
        rl, rg = (1, 1) if a.tag == "smoke" else (TS.REPS_L, TS.REPS_G)
        fks = lambda q: TS.FK if q in ("MC", "MC-V") else TS.FK[:1]  # noqa: E731
        units = [("oof", f"{h}@{q}", fk, r, 1.0) for q in src for fk in fks(q) for h in ("Q1", "Q3") for r in range(rl)]
        units += [("oof", p, f0, r, 1.0) for p in ("P1", "P3") for r in range(rl)]                              # gate G-P
        trees = [("oof", f"Q2@{q}", fk, r, 1.0) for q in src for fk in fks(q) for r in range(rg)]
        workers = max(1, min(n_cpus() // 4, a.workers))
        run.info("%d ridge / rule units on %d workers, %d tree units; sources %s", len(units), workers, len(trees), src)
        res = par.pmap(TS.unit2, units, workers=workers, run=run, desc="ridge/rule units", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        with threadpool_limits(limits=min(workers, 24), user_api="openmp"):
            res.values += [TS.unit2(u) for u in run.tqdm(trees, desc="tree units")]
        R = dict(res.values)
        with open(TS.OUT / "select2.pkl", "rb") as fh:
            R187 = pickle.load(fh)["res"]
        gp = max(float(np.abs(R[k]["d"] - R187[k]["d"]).max()) for k in R if k[1] in ("P1", "P3"))
        with open(tdir(a.tag, "select.pkl"), "wb") as fh:
            pickle.dump(dict(res=R, tok=tok, sources=src, gate_P_max_abs_diff=gp), fh)
        run.summary.update(n_units=len(units) + len(trees), gate_P_max_abs_diff=gp)
        run.info("G-P: max |d - select2.pkl| over the re-run P1 / P3 units %.3g", gp)
        assert gp == 0.0, f"G-P failed: the re-run P1 / P3 differ from decision 187's by {gp}"


# ---------------------------------------------------------------- 5. report
def cmd_report(a):
    from scipy.stats import rankdata
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    import turn_dewater as TD
    import turn_selinput as TS
    with Run("op_parity", f"margin_critic/report-{a.tag}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        out, figd = _pl.Path(a.out), _pl.Path(a.figs)
        out.mkdir(parents=True, exist_ok=True)
        C, tok, dyaw, log, X, F = TS.prepare()
        f0, f1 = TS.FK
        with open(tdir(a.tag, "select.pkl"), "rb") as fh:
            sel = pickle.load(fh)
        with open(TS.OUT / "select2.pkl", "rb") as fh:
            R = {**pickle.load(fh)["res"], **sel["res"]}
        src = sel["sources"]
        Mz = np.load(TS.OUT / "margins.npz")
        M, Cb = Mz["M"].astype(np.float64), Mz["C"].astype(np.float64)                  # (S, 33, n, 4)
        PQ = {q: np.load(tdir(a.tag, "pred", f"{q}.npz")) for q in src}
        Q = {q: np.clip(PQ[q]["Q"], *CLIP).astype(np.float64) for q in src}
        meta = {q: json.loads(str(PQ[q]["meta"])) for q in src}
        f19 = F["F19"]
        dac_fail = X[:, 0, :, 1] < 1                                                    # (S, n)
        rep_ex = TS._D["Y", f0][:, :, 1:].max(-1) > 1e-9
        lc = TS._D["lcode"]
        ceil = {fk: TS._D["Y", fk].max(-1).mean(0) for fk in TS.FK}
        zero = np.zeros(len(log))
        idx = [np.flatnonzero(lc == g) for g in range(lc.max() + 1)]
        rng = np.random.default_rng(0)
        boot = [np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))]) for _ in range(1000)]

        def auc(sc, y):
            pt, q = TS._auc(sc, y), np.array([TS._auc(sc[:, bi], y[:, bi]) for bi in boot])
            return pt, q

        def aucs(sc, y):
            pt, q = auc(sc, y)
            return f"{pt:.3f} [{np.quantile(q, 0.025):.3f}, {np.quantile(q, 0.975):.3f}]"

        def bs(v, f=".3f"):
            return stats.fmt(stats.bootstrap(v, groups=log), f)

        def spear(q):
            """Mean over seeds of the per-token Spearman of Q vs M (4 s) across the F19 candidates; NaN where the map margins span < 0.2 m."""
            a_, b_ = q[:, f19, :, 3], M[:, f19, :, 3]
            ra, rb = rankdata(a_, axis=1), rankdata(b_, axis=1)
            ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                rho = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
            rho[np.ptp(b_, axis=1) < 0.2] = np.nan
            rho[~np.isfinite(rho) & (np.ptp(b_, axis=1) >= 0.2)] = 0.0                  # constant prediction: no ranking information
            return np.nanmean(rho, 0)
        # ---- reading 1: margin quality
        rows, A = [], {}
        refs = {"map margin (PRIV, decision 187)": M, "SH30 own road edge (decision 187)": Cb}
        for name, q in {**refs, **Q}.items():
            e = np.abs(q - M)
            sp = spear(q)
            A[name, "dac"], A[name, "rep"] = auc(-q[:, 0, :, 3], dac_fail), auc(-q[:, 0, :, 3], rep_ex)
            mt = meta.get(name, {})
            r_ = float(np.corrcoef(q[:, f19, :, 3].ravel(), M[:, f19, :, 3].ravel())[0, 1])
            rows.append({"margin": name, "train tokens": mt.get("n_train", ""), "train logs": mt.get("n_train_logs", ""), "navtrain dev MAE 4 s": mt.get("best_dev_mae", np.nan),
                         "MAE identity 4 s (m)": bs(e[:, 0, :, 3].mean(0)), "MAE F19 4 s (m)": bs(e[:, f19, :, 3].mean((0, 1))),
                         "MAE F19 1 / 2 / 3 s": " / ".join(f"{e[:, f19, :, t].mean():.2f}" for t in range(3)), "Pearson r F19 4 s": r_,
                         "within-token Spearman F19": stats.fmt(stats.bootstrap(sp[np.isfinite(sp)], groups=log[np.isfinite(sp)])) if name != "map margin (PRIV, decision 187)" else "1",
                         "AUC DAC failure of the identity": aucs(-q[:, 0, :, 3], dac_fail), "AUC a repair exists": aucs(-q[:, 0, :, 3], rep_ex),
                         "identity margin < 0, %": 100 * (q[:, 0, :, 3] < 0).mean()})
        stats.write_table(rows, out / "margin_quality", note=f"navtest > 20 deg, {len(tok)} tokens x 2 SH30 seeds; MAE against the map margin (both clipped to [-2, 4] m), per-token mean then log-cluster bootstrap "
                          f"(B 10 000); AUC CIs log-cluster bootstrap B 1 000; DAC failure base rate {100 * dac_fail.mean():.2f}%, repair base rate {100 * rep_ex.mean():.1f}%; "
                          f"within-token Spearman over the 19 F19 candidates on token-seeds whose map margins span >= 0.2 m ({100 * (np.ptp(M[:, f19, :, 3], axis=1) >= 0.2).mean():.1f}%).")
        # ---- reading 2: selection
        B = {k: v for k, v in TD.buckets(dyaw).items() if "left" not in k and "right" not in k}
        HEADS = {"1": "ridge (E + all candidates' margins)", "2": "trees (candidate rows)", "3": "1-parameter rule (margin only)"}

        def dmean(arm, fk):
            v = [r["d"] for k, r in R.items() if k[:3] == ("oof", arm, fk) and k[4] == 1.0]
            return (np.mean(v, 0), len(v)) if v else (None, 0)

        def gci(d, m=slice(None), alpha=0.05):
            return stats.paired(100 * d[m], zero[m], groups=log[m], alpha=alpha)
        rows, Gn = [], {}
        for q in ["PRIV", *src]:
            for fk in TS.FK:
                for h in HEADS:
                    arm = f"P{h}" if q == "PRIV" else f"Q{h}@{q}"
                    d, nrep = dmean(arm, fk)
                    if d is None:
                        continue
                    dp, _ = dmean(f"P{h}", fk)
                    g, gb = gci(d), gci(d, alpha=0.05 / 3)
                    rc, rp = TC.ratio_ci(d, ceil[fk], log), TC.ratio_ci(d, dp, log)
                    pk = np.stack([x["picks"] for k, x in R.items() if k[:3] == ("oof", arm, fk) and k[4] == 1.0])
                    Gn[q, h, fk] = dict(g=g, gb=gb, rp=rp, rc=rc, d=d)
                    rows.append({"margin": "map (PRIV)" if q == "PRIV" else q, "head": HEADS[h], "family x convention": fk, "repeats": nrep, "> 20 deg": TC.cell(g),
                                 "20-45 deg": TC.cell(gci(d, B["20-45 deg"])), "> 45 deg": TC.cell(gci(d, B["> 45 deg"])), "Bonferroni CI (98.33%)": TC.cell(gb),
                                 "recovery of the ceiling": f"{100 * rc['mean']:.1f}% [{100 * rc['lo']:.1f}, {100 * rc['hi']:.1f}]",
                                 "recovery of the privileged arm": "" if q == "PRIV" else f"{100 * rp['mean']:.1f}% [{100 * rp['lo']:.1f}, {100 * rp['hi']:.1f}]",
                                 "minus the privileged arm": "" if q == "PRIV" else TC.cell(gci(d - dp)), "moved %": 100 * (pk != 0).mean()})
        stats.write_table(rows, out / "selection", note="out-of-fold gain over SH30 of the selected candidate, EPDMS x 100 (no EC), seed mean, mean over repeats of 5 folds by log (heads re-fitted inside navtest "
                          "exactly as decision 187, the margin head never saw navtest); paired log-cluster bootstrap, B 10 000. PRIV rows are decision 187's arms (map geometry, upper bound).")
        # ---- verdict
        gh = json.loads((tdir(a.tag, "gate_hidden.json")).read_text()) if tdir(a.tag, "gate_hidden.json").exists() else dict(ok=True)
        main = "MC" if gh["ok"] and "MC" in src else "MC-V"
        vd = dict(main=main, gate_hidden=gh, thresholds=THR, heads={})
        if main in src:
            for h in HEADS:
                x, l9 = Gn[main, h, f0], Gn[main, h, f1]
                vd["heads"][h] = dict(gain=x["g"]["mean"], lo95=x["g"]["lo"], hi95=x["g"]["hi"], lo_bonf=x["gb"]["lo"], rec_priv=x["rp"]["mean"], rec_ceiling=x["rc"]["mean"], l9=l9["g"]["mean"],
                                      a=bool(x["gb"]["lo"] > 0 and x["rp"]["mean"] >= THR["half"] and l9["g"]["mean"] > 0), sig=bool(x["gb"]["lo"] > 0), null=bool(x["g"]["lo"] <= 0))
            dac = A[main, "dac"][0]
            line = REF["curb_dac"] + THR["auc_margin"]
            case = "a" if any(v["a"] for v in vd["heads"].values()) else "c" if all(v["null"] for v in vd["heads"].values()) and dac < line else \
                "b1" if any(v["sig"] for v in vd["heads"].values()) else "b2" if dac >= line else "b"
            vd.update(case=case, dac_auc=dac, dac_line=line, rep_auc=A[main, "rep"][0])
            if "MC-E" in src:
                df = A[main, "dac"][1] - A["MC-E", "dac"][1]
                vd["floor"] = dict(dac_auc_floor=A["MC-E", "dac"][0], diff=dac - A["MC-E", "dac"][0], lo=float(np.quantile(df, 0.025)), hi=float(np.quantile(df, 0.975)),
                                   reads_scene=bool(dac - A["MC-E", "dac"][0] > THR["floor_auc"]))
        # ---- reading 3: learning curve
        cv = [q for q in CURVE if q in src]
        rows = []
        for q in cv:
            e = np.abs(Q[q] - M)
            rows.append({"arm": q, "train tokens": meta[q]["n_train"], "train logs": meta[q]["n_train_logs"], "navtrain dev MAE 4 s": meta[q]["best_dev_mae"],
                         "MAE F19 4 s (m)": bs(e[:, f19, :, 3].mean((0, 1))), "AUC DAC failure": aucs(-Q[q][:, 0, :, 3], dac_fail), "AUC a repair exists": aucs(-Q[q][:, 0, :, 3], rep_ex),
                         **{f"Q{h} F19 x pc": TC.cell(Gn[q, h, f0]["g"]) for h in HEADS}})
        if rows:
            stats.write_table(rows, out / "learning_curve", note="nested subsets of the navtrain training logs (sorted by sha256('mc|' + log)); one training seed per point")
        if len(cv) >= 2 and "MC-30" in src and "MC" in src:
            m30, m100 = np.abs(Q["MC-30"] - M)[:, f19, :, 3].mean(), np.abs(Q["MC"] - M)[:, f19, :, 3].mean()
            da = A["MC", "dac"][0] - A["MC-30", "dac"][0]
            vd["curve"] = dict(dac_auc_30_to_100=da, mae_rel_drop_30_to_100=float(1 - m100 / m30), rising=bool(da >= THR["rise_auc"] or 1 - m100 / m30 >= THR["rise_mae"]))
        # ---- reading 4: reference arm
        if "R-WA" in src and "R-C1" in src:
            df = A["R-WA", "dac"][1] - A["R-C1", "dac"][1]
            dm = stats.paired(np.abs(Q["R-WA"] - M)[:, f19, :, 3].mean((0, 1)), np.abs(Q["R-C1"] - M)[:, f19, :, 3].mean((0, 1)), groups=log)
            rows = [{"contrast R-WA - R-C1": "MAE F19 4 s (m)", "value": stats.fmt(dm)},
                    {"contrast R-WA - R-C1": "AUC DAC failure", "value": f"{A['R-WA', 'dac'][0] - A['R-C1', 'dac'][0]:+.3f} [{np.quantile(df, 0.025):+.3f}, {np.quantile(df, 0.975):+.3f}]"}]
            g2 = None
            for h in HEADS:
                g = gci(Gn["R-WA", h, f0]["d"] - Gn["R-C1", h, f0]["d"])
                g2 = g if h == "2" else g2
                rows.append({"contrast R-WA - R-C1": f"Q{h} gain, F19 x pc", "value": TC.cell(g)})
            stats.write_table(rows, out / "reference", note="same head, same navtrain tokens (shards 2-4), one frame of 32 x 512 tokens + ego: WA-Cf (NAVSIM-trained encoder, in-sample on navtrain) minus frozen Cinque")
            vd["reference"] = dict(dac_auc_diff=A["R-WA", "dac"][0] - A["R-C1", "dac"][0], dac_lo=float(np.quantile(df, 0.025)), dac_hi=float(np.quantile(df, 0.975)), q2_diff=g2,
                                   lost_in_encoder=bool(A["R-WA", "dac"][0] - A["R-C1", "dac"][0] >= THR["ref_auc"] or g2["lo"] > 0))
        vd["gates"] = dict(P_max_abs_diff=sel["gate_P_max_abs_diff"], label={q: meta[q]["gate"] for q in src})
        vd["cost"] = {q: dict(it_s=meta[q]["it_s"], steps=meta[q]["steps"], vram_gb=meta[q]["vram_gb"], load_s=meta[q]["load_s"]) for q in src}
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        run.info("verdict: %s", json.dumps(vd, default=float))
        run.summary.update(case=vd.get("case", ""), main=main)
        fig_report(figd, M, Q, Gn, A, meta, src, main, f19, f0)


def fig_report(FIGD, M, Q, Gn, A, meta, src, main, f19, f0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIGD.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(ps.DOUBLE_COLUMN_IN, 2.5), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.7, 1, 1]))
    ax = axs[0]
    if main in Q:
        ax.hexbin(M[:, f19, :, 3].ravel(), Q[main][:, f19, :, 3].ravel(), gridsize=40, bins="log", cmap="Greys", extent=(*CLIP, *CLIP), linewidths=0)
        ax.plot(CLIP, CLIP, color=ps.PALETTE["vermillion"], lw=0.7)
    ax.set_xlabel("map margin, 4 s (m)"), ax.set_ylabel(f"predicted margin, {main} (m)")
    ax = axs[1]
    grp = [("PRIV", "map\n(PRIV)", ps.PALETTE["vermillion"])] + [(q, q, ps.PALETTE["blue"] if q.startswith("MC") else ps.PALETTE["green"]) for q in src if q not in CURVE[:-1]]
    w = 0.26
    for j, h in enumerate("123"):
        v = np.array([[Gn[q, h, f0]["g"]["mean"], Gn[q, h, f0]["g"]["mean"] - Gn[q, h, f0]["g"]["lo"], Gn[q, h, f0]["g"]["hi"] - Gn[q, h, f0]["g"]["mean"]] for q, _, _ in grp])
        ax.bar(np.arange(len(grp)) + (j - 1) * w, v[:, 0], w, yerr=v[:, 1:].T, color=[c for _, _, c in grp], alpha=(1.0, 0.7, 0.4)[j], error_kw=dict(lw=0.5, capsize=1))
    ax.set_xticks(range(len(grp)), [g[1] for g in grp], fontsize=6)
    ax.set_ylabel("selector - SH30, EPDMS x 100\n(F19 x pc, > 20 deg, 95% CI)"), ps.zero_line(ax), ps.bars(ax)
    ax.text(0.98, 0.95, "bars: ridge / trees / rule", transform=ax.transAxes, fontsize=6, ha="right", va="top")
    cv = [q for q in CURVE if q in src]
    if cv:
        n = [meta[q]["n_train"] for q in cv]
        for ax, key, lab, refs in ((axs[2], "dac", "AUC, DAC failure of the identity", (REF["map_dac"], REF["curb_dac"])), (axs[3], "rep", "AUC, a repair exists", (REF["map_rep"], REF["curb_rep"]))):
            v = np.array([[A[q, key][0], *np.quantile(A[q, key][1], [0.025, 0.975])] for q in cv])
            ax.plot(n, v[:, 0], color=ps.PALETTE["blue"], marker="o", ms=2.5)
            ax.fill_between(n, v[:, 1], v[:, 2], color=ps.PALETTE["blue"], alpha=0.15, lw=0)
            ax.axhline(refs[0], color=ps.PALETTE["vermillion"], ls="--", lw=0.7), ax.axhline(refs[1], color=ps.BASELINE, ls="--", lw=0.7)
            if "MC-E" in src:
                ax.axhline(A["MC-E", key][0], color=ps.PALETTE["black"], ls=":", lw=0.7)
            ax.set_xscale("log"), ax.set_xlabel("navtrain training tokens"), ax.set_ylabel(lab)
        axs[2].text(0.03, 0.5, "red: map (PRIV)\ngrey: own road edge\ndotted: ego-only head", transform=axs[2].transAxes, fontsize=5.5, va="center")
    fig.savefig(FIGD / "margin_critic.png", dpi=300)
    plt.close(fig)


def cmd_hidden(a):
    """G-hidden: per-dimension Pearson correlation of the fold models' hidden state with SH30-F-s0's on the navtest turn tokens (features only)."""
    import turn_selinput as TS
    tk, _ = TC.bucket_tokens()
    f = next((p for p in (D / "runs/op_parity/turn_selnt/feat/full/navtest.npz", D / "runs/op_parity/turn_selnt/feat/smoke/navtest.npz") if p.exists()), None)
    assert f is not None, "no fold-model hidden states on navtest (turn_selnt/feat/*/navtest.npz)"
    z, h = np.load(f), np.load(TS.HID.format(0))
    assert np.array_equal(z["tokens"], tk)
    row = {t: i for i, t in enumerate(h["names"].tolist())}
    r = np.array([row[t] for t in tk])
    a0 = np.concatenate([h["select_4"][r], h["mean"][r]], 1).astype(np.float64)
    med = []
    for j in range(K):
        b = np.concatenate([z[f"select_4_{j}"], z[f"mean_{j}"]], 1).astype(np.float64)
        x, y = a0 - a0.mean(0), b - b.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            med.append(float(np.nanmedian((x * y).sum(0) / np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0)))))
    g = dict(source=str(f), median_corr_per_fold=med, median=float(np.median(med)), ok=bool(np.median(med) >= 0.8))
    tdir(a.tag).mkdir(parents=True, exist_ok=True)
    tdir(a.tag, "gate_hidden.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "bank", "train", "select", "report", "hidden", "arms"):
        p = sp.add_parser(name)
        p.add_argument("--tag", default="full")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--arm", default="MC")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--force", action="store_true")
        p.add_argument("--workers", type=int, default=48)
        p.add_argument("--out", default=str(RES))
        p.add_argument("--figs", default=str(FIG))
    a = ap.parse_args()
    if a.cmd == "arms":
        print("\n".join(f"{k} {v['vram']}" for k, v in ARMS.items()))
        return
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
