#!/usr/bin/env python3
"""HEAD1 trainer (experiments/corridor, plans/2026-10-10-head1-prereg.md): a head on the FROZEN Cinque vision tokens (8 policy slots x 32 x
512, warp protocol) + the 20-dim ego input (it carries the command) that predicts the heading-versus-arc-length profile of the road ahead
(head1_labels.py: L = logged path, M = map lane sequence). The map and the logged future are training labels only; nothing privileged is read
at inference. No WA-JEPA weights or features.

  envs/op-train python head1_train.py --arm L|M|C|blind --fold J [--frac F] --seed S [--smoke]
Arms    L / M: loss on that channel only; C: both channels (multi-task); blind: arm C without vision tokens (ego + command only, the floor).
Folds   decision 191's log-level folds: fits on navsim/op-parity-cf5f{J}-train (the logs fold policy CF5f{J}-F-s0 saw) minus their validation
        part (sha256("head1val|" + log) % 10 == 0, step selection only); predicts the fold's dev logs (held out) and navtest.
Output  <run dir>/pred.npz: names_dev, dev (n, 22, 2), names_test, test (n, 22, 2) [channel 0 = L, 1 = M, rad], grid; ckpt.pt; curve.json.
        A pointer copy goes to $DATA_DIR/runs/corridor/head1/pred/<arm>-f<J>-p<frac%>-s<S>.npz (what head1_read.py reads).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.common import data_dir  # noqa: E402

CR = data_dir() / "runs/op_parity/cache"
H1 = data_dir() / "runs/corridor/head1"
NSH, NG = 12, 22
TRAIN_DIRS = [f"navtrain_full.s{k}of{NSH}" for k in range(NSH)]
TEST_DIR = "lb_navtest"
DELTA = 0.1                                   # Huber delta, rad


def _h(salt, log):
    return int(hashlib.sha256(f"{salt}|{log}".encode()).hexdigest(), 16)


def is_val(log):
    return _h("head1val", log) % 10 == 0


def frac_logs(logs, frac):
    lg = sorted(set(logs), key=lambda l: hashlib.sha256(f"head1lc|{l}".encode()).hexdigest())
    return set(lg[: max(1, int(np.ceil(frac * len(lg))))])


def pred_name(arm, fold, frac, seed):
    return f"{arm}-f{fold}-p{int(round(frac * 1000)):04d}-s{seed}"


def load_tokens(src, n, dev, chunk=1024, threads=8):
    """(n, 8, 32, 512) fp16 on the device from the memory-mapped caches; src = [(cache dir, rows in it, first output row)]."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    V = torch.empty((n, 8, 32, 512), dtype=torch.float16, device=dev)
    jobs = [(d, m[c:c + chunk], o + c) for d, m, o in src for c in range(0, len(m), chunk)]

    def one(j):
        d, rows, o = j
        return o, np.ascontiguousarray(np.load(CR / f"{d}@warp" / "front.npy", mmap_mode="r")[rows])
    with ThreadPoolExecutor(threads) as ex:
        for o, x in ex.map(one, jobs):
            V[o:o + len(x)] = torch.from_numpy(x).to(dev)
    return V


def build(vision=True, d=256, enc=3, dec=2, drop=0.1):
    import torch
    import torch.nn as nn

    class Dec(nn.Module):
        def __init__(s):
            super().__init__()
            s.n0, s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            s.sa, s.ca = nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True), nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, x, mem):
            y = s.n0(x)
            x = x + s.sa(y, y, y, need_weights=False)[0]
            x = x + s.ca(s.n1(x), mem, mem, need_weights=False)[0]
            return x + s.ff(s.n2(x))

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            s.vision = vision
            s.ep = nn.Sequential(nn.Linear(20, d), nn.GELU(), nn.Linear(d, d))
            if vision:
                s.vln, s.vp = nn.LayerNorm(512), nn.Linear(512, d)
                s.slot, s.pos = nn.Parameter(torch.randn(8, 1, d) * 0.02), nn.Parameter(torch.randn(1, 32, d) * 0.02)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True), enc, enable_nested_tensor=False)
            s.q = nn.Parameter(torch.randn(NG, d) * 0.02)
            s.dec = nn.ModuleList([Dec() for _ in range(dec)])
            s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

        def forward(s, V, E):
            """V (B, 8, 32, 512) | None, E (B, 20) standardised -> (B, 22, 2) fp32: heading (rad) at the grid arc lengths, channels L, M."""
            e = s.ep(E)[:, None]
            mem = s.enc(torch.cat([e, (s.vp(s.vln(V.float())) + s.slot + s.pos).flatten(1, 2)], 1) if s.vision else e)
            x = s.q[None].expand(len(E), -1, -1)
            for l in s.dec:
                x = l(x, mem)
            return s.out(x).float()
    return Head()


def predict(net, V, E, idx, batch=512):
    import torch
    net.eval()
    O = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c in range(0, len(idx), batch):
            i = idx[c:c + batch]
            O.append(net(V[i] if V is not None else None, E[i]).cpu().numpy())
    net.train()
    return np.concatenate(O)


def main(a):
    import torch
    import torch.nn.functional as F
    from jevdrive.data import splits
    from jevdrive.run import Run
    vision = a.arm != "blind"
    ch = {"L": [0], "M": [1], "C": [0, 1], "blind": [0, 1]}[a.arm]
    name = pred_name(a.arm, a.fold, a.frac, a.seed) + ("-smoke" if a.smoke else "")
    with Run("corridor", f"head1/train-{name}", seed=a.seed, config=vars(a)) as run:
        dev = torch.device("cuda")
        s_tr, s_dv, s_nt = (splits.load(f"navsim/op-parity-cf5f{a.fold}-train"), splits.load(f"navsim/op-parity-cf5f{a.fold}-dev"), splits.load("navsim/navtest"))
        for s in (s_tr, s_dv, s_nt):
            run.use_split(s)
        splits.check_disjoint(s_tr, s_dv)
        tabs = [np.load(CR / d / "tab.npz") for d in TRAIN_DIRS]
        names, logs, ego = (np.concatenate([t[k] for t in tabs]) for k in ("names", "log", "ego"))
        off = np.cumsum([0] + [len(t["names"]) for t in tabs])
        Lb = np.load(H1 / "labels/navtrain.npz")
        assert (Lb["names"] == names).all(), "labels are not in token-cache order"
        tt = np.load(CR / TEST_DIR / "tab.npz")
        assert s_nt.mask(tt["names"]).all() and not set(tt["log"].tolist()) & set(logs.tolist()), "navtest shares a log with navtrain"
        in_tr, in_dv = s_tr.mask(logs), s_dv.mask(logs)
        val = in_tr & np.array([is_val(l) for l in logs.tolist()])
        fit = in_tr & ~val
        if a.frac < 1:
            keep = frac_logs(logs[fit].tolist(), a.frac)
            fit &= np.array([l in keep for l in logs.tolist()])
        assert not set(logs[fit].tolist()) & set(logs[val].tolist()) and not set(logs[in_tr].tolist()) & set(logs[in_dv].tolist())
        if a.smoke:
            rng = np.random.default_rng(0)
            for m in (fit, val, in_dv):
                m[rng.permutation(np.flatnonzero(m))[2000:]] = False
        use = np.flatnonzero(fit | val | in_dv)                                    # rows of navtrain on the card, then navtest
        n_tr, n_te = len(use), (2000 if a.smoke else len(tt["names"]))
        pos = np.full(len(names), -1)
        pos[use] = np.arange(n_tr)
        i_fit, i_val, i_dv, i_te = pos[fit], pos[val], pos[in_dv], n_tr + np.arange(n_te)
        run.info("fold %d: fit %d tokens / %d logs, val %d / %d, held-out dev %d / %d, navtest %d", a.fold, len(i_fit), len(set(logs[fit].tolist())), len(i_val),
                 len(set(logs[val].tolist())), len(i_dv), len(set(logs[in_dv].tolist())), n_te)
        t0 = time.time()
        V = None
        if vision:
            src = [(TRAIN_DIRS[k], use[(use >= off[k]) & (use < off[k + 1])] - off[k], int(np.searchsorted(use, off[k]))) for k in range(NSH)]
            V = load_tokens(src + [(TEST_DIR, np.arange(n_te), n_tr)], n_tr + n_te, dev)
            run.info("tokens on the card: %.1f GB in %.0f s", V.numel() * 2 / 2 ** 30, time.time() - t0)
        e_all = np.concatenate([ego[use], tt["ego"][:n_te]]).astype(np.float32)
        emu, esd = e_all[i_fit].mean(0), e_all[i_fit].std(0) + 1e-6
        G = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        E = G((e_all - emu) / esd)
        Y = np.stack([Lb["L"][use], Lb["M"][use]], -1)                              # (n_tr, 22, 2), nan = no label
        Y[..., [c for c in (0, 1) if c not in ch]] = np.nan
        Yt, ok = G(np.nan_to_num(Y)), G(np.isfinite(Y))
        w = 1 + np.minimum(np.abs(np.degrees(np.nan_to_num(Lb["dyaw4"][use]))), 90.0) / 30.0
        p = np.zeros(n_tr)
        p[i_fit] = w[i_fit]
        cdf = torch.as_tensor(np.cumsum(p / p.sum()), device=dev)
        Wv = G(w)
        gval = torch.as_tensor(i_val, device=dev)

        def loss_of(o, i, wt=None):
            l = F.huber_loss(o, Yt[i], delta=DELTA, reduction="none") * ok[i]
            if wt is not None:
                l = l * wt[:, None, None]
                return l.sum() / (ok[i] * wt[:, None, None]).sum().clamp_min(1e-6)
            return l.sum() / ok[i].sum().clamp_min(1e-6)

        torch.manual_seed(a.seed)
        net = build(vision).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        steps, warm, ev = (60, 10, 30) if a.smoke else (a.steps, a.warm, a.eval)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        run.info("%s: %.2f M parameters", a.arm, sum(x.numel() for x in net.parameters()) / 1e6)
        best, best_v, curve, t1 = None, np.inf, [], time.time()
        for step in run.tqdm(range(steps), desc=f"train {name}"):
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, step / steps)))
            b = torch.searchsorted(cdf, torch.rand(a.batch, device=dev, generator=gen, dtype=cdf.dtype)).clamp_max(n_tr - 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(V[b] if vision else None, E[b])
            loss = loss_of(o, b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                run.scalar("loss/train", float(loss), step)
            if (step + 1) % ev == 0 or step + 1 == steps:
                net.eval()
                num = den = 0.0
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for c in range(0, len(gval), 512):
                        i = gval[c:c + 512]
                        k = float((ok[i] * Wv[i][:, None, None]).sum())
                        num, den = num + float(loss_of(net(V[i] if vision else None, E[i]), i, Wv[i])) * k, den + k
                net.train()
                v = num / max(den, 1e-6)
                curve.append(dict(step=step + 1, val=v, train=float(loss)))
                run.scalar("loss/val", v, step + 1)
                run.info("step %d: train %.5f val %.5f | %.1f it/s", step + 1, float(loss), v, (step + 1) / (time.time() - t1))
                if v < best_v:
                    best_v, best = v, {k: x.detach().clone() for k, x in net.state_dict().items()}
        its = steps / max(time.time() - t1, 1e-6)
        net.load_state_dict(best)
        idv, ite = torch.as_tensor(i_dv, device=dev), torch.as_tensor(i_te, device=dev)
        P_dv, P_te = predict(net, V, E, idv), predict(net, V, E, ite)
        assert np.isfinite(P_dv).all() and np.isfinite(P_te).all()
        f = run.path("pred.npz")
        np.savez(f.with_suffix(".tmp.npz"), names_dev=names[in_dv], dev=P_dv.astype(np.float32), names_test=tt["names"][:n_te], test=P_te.astype(np.float32), grid=Lb["grid"],
                 arm=a.arm, fold=a.fold, frac=a.frac, seed=a.seed)
        f.with_suffix(".tmp.npz").rename(f)
        torch.save(dict(model=best, config=vars(a), emu=emu, esd=esd, curve=curve), run.path("ckpt.pt"))
        run.path("curve.json").write_text(json.dumps(curve, indent=1))
        (H1 / "pred").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, H1 / "pred" / f".{name}.tmp")
        os.replace(H1 / "pred" / f".{name}.tmp", H1 / "pred" / f"{name}.npz")
        run.summary.update(pred=str(H1 / "pred" / f"{name}.npz"), best_step=int(min(curve, key=lambda c: c["val"])["step"]), best_val=best_v, its=its, n_fit=len(i_fit),
                           logs_fit=len(set(logs[fit].tolist())), n_dev=len(i_dv), vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
        run.info("done: best val %.5f at step %d; %.1f it/s; %.1f GB VRAM", best_v, run.summary["best_step"], its, run.summary["vram_gb"])


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("L", "M", "C", "blind"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0, help="share of the fit logs (learning curve)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    cli_args(ap)
    main(ap.parse_args())
