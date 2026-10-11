#!/usr/bin/env python3
"""LONG-PROBE trainer (GPU): the input-ladder head (lib/lc_head.py) on BODY1's rows, S0's recipe (experiments/body1/scripts/bd1_train.py)
with the longitudinal labels A / B / C (lib/lc.py) as three more logits.

  train --rung blind|vis|nat|priv|blind_nat --seed S [--shards 2 3 4] [--steps 12000] --tag NAME [--auto]
Fit = body1-train-logs minus their validation part (contact_head.is_val), model selection on that validation part only, hold logs never loaded.
Loss = S0's (agent / boundary BCE with `w4` weights, clearance / margin / first-contact regressions, per-step targets) + lam_long x (BCE of A, B, C,
each with its own w4-style weights: 4 user classes x {positive, negative} at equal mass over the fit rows that carry the label; a cell's
per-row weight is capped at the rate of a cell of CELL_FLOOR rows). States are sampled with probability ~ (sum of the four weight sets over
the state's queries) ** alpha, the rest of the weight in the loss (S0's scheme). Selection: per output group the best evaluation step
(lat = S0's rule, mean of the own-plan agent and boundary AUC; A, B, C = the AUC of that label on the own plan and its arc re-timings,
query slots 0, 1, 16-19) -> ckpt.pt holds the four states.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import lc as LC  # noqa: E402
import lc_head as H  # noqa: E402
import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402

FAMS = ("log", "ot1", "yr1")
CELL_FLOOR = 100


def load_extra(R, nat=False, priv=False):
    """Longitudinal labels (+ native / privileged inputs) aligned with the rows of contact_head.load_rows (R['src'])."""
    keys = ["a_lab", "a_sub", "a_t", "b_lab", "b_kind", "b_t", "c_lab", "c_dv", "c_t", "b_ahead", "c_ahead"]
    out = {k: [] for k in keys + (["nat", "hid"] if nat else []) + (["obj", "rte", "lin", "trn"] if priv else [])}
    for d, m, o in R["src"]:
        z = np.load(LC.root() / "rows" / f"{d}.npz")
        assert (z["gi"][m] == R["gi"][o:o + len(m)]).all()
        for k in keys:
            out[k].append(z[k][m])
        if nat:
            y = np.load(LC.root() / "native" / f"{d}.npz")
            out["nat"].append(y["nat"][m]), out["hid"].append(y["hid"][m])
        if priv:
            y = np.load(LC.root() / "priv" / f"{d}.npz")
            assert (y["gi"][m] == z["gi"][m]).all()
            for k in ("obj", "rte", "lin", "trn"):
                out[k].append(y[k][m])
    return {k: np.concatenate(v) for k, v in out.items()}


def load_tokens_host(src, n, chunk: int = 2048, threads: int = 8):
    """The (n, 8, 32, 512) fp16 vision tokens of the selected states in host memory (a torch CPU tensor), gathered from the memory-mapped caches.
    Host mode keeps the card's share to the model (S0 kept all 57 GB on the card; on a shared box no card has that free)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    V = torch.empty((n, 8, 32, 512), dtype=torch.float16)
    jobs = [(d, m[c:c + chunk], o + c) for d, m, o in src for c in range(0, len(m), chunk)]

    def one(j):
        d, rows, o = j
        f = np.load(B.cache_root() / f"{d}@warp" / "front.npy", mmap_mode="r")
        V[o:o + len(rows)] = torch.from_numpy(np.ascontiguousarray(f[rows]))
        return len(rows)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, jobs))
    return V


class HostBatches:
    """Training batches with the tokens gathered on worker threads from the host tensor into page-locked buffers and uploaded ahead of the loop
    (depth batches in flight). Indices are drawn on the caller's thread from a CPU generator, in step order."""

    def __init__(self, V, cdf, batch, seed, step0, dev, depth=3, workers=3):
        import torch
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        self.V, self.cdf, self.batch, self.dev = V, cdf.cpu(), batch, dev
        self.gen = torch.Generator().manual_seed(seed * 100003 + step0)
        self.pool, self.q, self.depth = ThreadPoolExecutor(workers), deque(), depth
        self.n = len(V) if V is not None else len(self.cdf)
        for _ in range(depth):
            self._push()

    def _push(self):
        import torch
        b = torch.searchsorted(self.cdf, torch.rand(self.batch, generator=self.gen, dtype=self.cdf.dtype)).clamp_max(self.n - 1)
        self.q.append(self.pool.submit(self._fetch, b))

    def _fetch(self, b):
        import torch
        if self.V is None:
            return b.to(self.dev), None, None
        buf = torch.empty((len(b),) + tuple(self.V.shape[1:]), dtype=torch.float16, pin_memory=True)
        torch.index_select(self.V, 0, b, out=buf)
        return b.to(self.dev), buf.to(self.dev, non_blocking=True), buf

    def get(self):
        b, v, buf = self.q.popleft().result()
        self._push()
        return b, v, buf


def balance_long(cls_state, lab, qok, fit):
    """w4-style weights of one longitudinal label: (4 user classes x {pos, neg}) cells at equal mass over the fit rows with a label; per-row
    weight = cell mass / max(cell rows, CELL_FLOOR). -> (n, Q) float32 (0 where no label or outside fit + val), mean 1 over the used fit rows; cells."""
    use = (lab >= 0) & qok
    c = np.broadcast_to(cls_state[:, None], lab.shape)
    cell = c * 2 + (lab == 1)
    n = np.bincount(cell[fit][use[fit]], minlength=8).astype(float)
    w = np.where(use, (1.0 / np.maximum(n, CELL_FLOOR))[cell], 0.0)
    return (w / w[fit][use[fit]].mean()).astype(np.float32), n.reshape(4, 2)


SEL = list(C.OWN) + list(range(16, 20))          # A / B / C selection rows: the own plan and its arc re-timings (prereg section 5)


def val_read(Y, out, rows):
    """AUCs on the validation states `rows`: agent / boundary on the own plan (S0's rule); A, B, C on the own plan + its arc re-timings
    (selection, key A ...) and on the own plan alone (key A_own ...)."""
    r = {}
    own = list(C.OWN)
    for k, (y, ok) in Y.items():
        for nm, qs in ((k, own if k in ("a", "b") else SEL), (f"{k}_own", own)):
            yy, oo = y[rows][:, qs], ok[rows][:, qs]
            r[nm] = C.auc(yy[oo], out[:, qs, H.IDX[k]][oo])
            r[f"{nm}_pos"] = int(yy[oo].sum())
    r["lat"] = 0.5 * (r["a"] + r["b"])
    return r


def cmd_train(a):
    import torch
    import torch.nn.functional as F
    from jevdrive.data import splits
    from jevdrive.run import Run
    ptr = LC.root() / "train" / "runs" / f"{a.tag}-s{a.seed}.dir"
    resume = a.resume
    if a.auto and resume is None and ptr.exists():
        d = _pl.Path(ptr.read_text().strip())
        resume = d if (d / "state.pt").exists() and not (d / "DONE").exists() else None
    torch.set_num_threads(4)
    with Run("long_contact", f"train/{a.tag}", seed=a.seed, config=vars(a), resume=resume) as run:
        ptr.parent.mkdir(parents=True, exist_ok=True)
        ptr.write_text(str(run.dir) + "\n")
        dev = torch.device("cuda")
        s_tr, s_ho = splits.load(B.TRAIN), splits.load(B.HOLD)
        run.use_split(s_tr), run.use_split(s_ho)
        splits.check_disjoint(s_tr, s_ho)
        t0 = time.time()
        shards = a.shards if a.shards else list(range(B.NSH))
        logs_all = np.unique(np.concatenate([np.load(B.root() / "rows" / f"{B.cdir('log', k)}.npz")["log"] for k in range(B.NSH)]))
        logs_fit = [l for l in logs_all[s_tr.mask(logs_all)].tolist() if not C.is_val(l)]
        val_of = {l: C.is_val(l) for l in logs_all.tolist()}
        keep_fit = set(logs_fit)
        R = C.load_rows(FAMS, shards, lambda log, hold: ~hold & np.array([val_of[l] or l in keep_fit for l in log.tolist()]), steps=True)
        n = len(R["gi"])
        assert s_tr.mask(R["log"]).all() and not s_ho.mask(R["log"]).any() and not R["hold"].any(), "a hold log reached the trainer"
        nat, priv = a.rung in ("nat", "priv", "blind_nat"), a.rung == "priv"
        X = load_extra(R, nat, priv)
        val = np.array([val_of[l] for l in R["log"].tolist()])
        fit = ~val
        T = dict(np.load(B.root() / "taxonomy" / "tax.npz"))
        cls, _ = C.classes(T)
        cs = cls[R["gi"]]
        Y = C.targets(R)
        W, cells = C.balance(cs, Y, fit)
        WL, CL = {}, {}
        for k in LC.LONG:
            WL[k], CL[k] = balance_long(cs, X[f"{k.lower()}_lab"], R["qok"], fit)
        run.info("states: fit %d, val %d; S0 cells %s; long cells (class x [neg, pos]) %s; tables in %.0f s", fit.sum(), val.sum(), cells.astype(int).tolist(),
                 {k: v.astype(int).tolist() for k, v in CL.items()}, time.time() - t0)
        G = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        vis_ = a.rung not in ("blind", "blind_nat")
        D = dict(V=(load_tokens_host(R["src"], n) if a.host else C.load_tokens(R["src"], n, dev)) if vis_ else None)
        emu, esd = R["ego"][fit].mean(0), R["ego"][fit].std(0) + 1e-6
        D["E"], D["Qp"] = G((R["ego"] - emu) / esd), G(R["q"])
        nmu = nsd = None
        if nat:
            nmu, nsd = X["nat"][fit].reshape(-1, H.NAT).mean(0), X["nat"][fit].reshape(-1, H.NAT).std(0) + 1e-3
            D["nat"], D["hid"] = G((X["nat"] - nmu) / nsd, torch.float16), G(X["hid"], torch.float16)
        if priv:
            for k in ("obj", "rte", "lin", "trn"):
                D[k] = G(X[k], torch.float16)
        run.info("inputs on the card in %.0f s: %s", time.time() - t0, {k: tuple(v.shape) for k, v in D.items() if v is not None})
        clipn = lambda x, c: np.where(x < 90, np.clip(x, *c), np.nan)  # noqa: E731
        clr = np.where(R["a_clr"] < 90, np.clip(R["a_clr"], *C.A_CLIP), C.A_CLIP[1])
        tgt = dict(a=G(Y["a"]), b=G(Y["b"]), b_ok=G(Y["b_ok"]), w=G(W), clr=G(clr), mg=G(clipn(R["b_margin"], C.M_CLIP)),
                   a_s=G(np.where(Y["a"], R["a_s"] / C.S_SCALE, np.nan)), a_t=G(np.where(Y["a"], R["a_t"] / C.T_SCALE, np.nan)), b_s=G(np.where(Y["b"], R["b_s"] / C.S_SCALE, np.nan)),
                   sa=G(np.where(R["sa"].astype(np.float32) < 90, np.clip(R["sa"].astype(np.float32), *C.A_CLIP), C.A_CLIP[1]), torch.float16),
                   sm=G(clipn(R["sm"].astype(np.float32), C.M_CLIP), torch.float16))
        for k in LC.LONG:
            tgt[k] = G(X[f"{k.lower()}_lab"] == 1)
            tgt["w" + k] = G(WL[k])
        Ws = (W + sum(WL.values())).sum(1).astype(np.float64)
        p = np.where(fit, Ws ** a.alpha, 0.0)
        cdf = torch.as_tensor(np.cumsum(p / p.sum()), device=dev)
        inv = G(np.where(Ws > 0, 1.0 / np.maximum(Ws, 1e-12) ** a.alpha, 0.0))
        ival = np.flatnonzero(val)
        gval = torch.as_tensor(ival, device=dev)
        Yv = dict(a=(Y["a"], Y["a_ok"]), b=(Y["b"], Y["b_ok"])) | {k: (X[f"{k.lower()}_lab"] == 1, (X[f"{k.lower()}_lab"] >= 0) & R["qok"]) for k in LC.LONG}
        torch.manual_seed(a.seed)
        net = H.build(a.rung, a.d, a.enc, a.dec, a.drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        lr_at = lambda s: min(1.0, (s + 1) / a.warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))  # noqa: E731
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        step0, best, best_sel, curve, kills = 0, {h: None for h in H.HEADS}, {h: -np.inf for h in H.HEADS}, [], 0
        st_f = run.path("state.pt")
        if st_f.exists():
            st = torch.load(st_f, map_location=dev, weights_only=False)
            net.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), gen.manual_seed(a.seed * 100003 + st["step"])
            step0, best, best_sel, curve, kills = st["step"], st["best"], st["best_sel"], st["curve"], st["kills"] + 1
            run.info("resumed at step %d (restart %d)", step0, kills)
        run.info("%s: %.2f M parameters", a.rung, sum(x.numel() for x in net.parameters()) / 1e6)
        hub = lambda x, y, w: ((F.huber_loss(x, torch.nan_to_num(y.float()), delta=a.delta, reduction="none") * w * torch.isfinite(y)).sum() / (w * torch.isfinite(y)).sum().clamp_min(1e-6))  # noqa: E731
        t2 = time.time()
        src = HostBatches(D["V"] if (a.host and net.vision) else None, cdf, a.batch, a.seed, step0, dev) if a.host else None
        for step in run.tqdm(range(step0, a.steps), desc=f"train {a.tag}", initial=step0, total=a.steps):
            for g in opt.param_groups:
                g["lr"] = a.lr * lr_at(step)
            if src is not None:
                b, Vb, _buf = src.get()
            else:
                b = torch.searchsorted(cdf, torch.rand(a.batch, device=dev, generator=gen, dtype=cdf.dtype)).clamp_max(n - 1)
                Vb = D["V"][b] if net.vision else None
            valid = torch.ones((a.batch, 8), dtype=torch.bool, device=dev)
            if net.vision:
                k = torch.randint(1, 8, (a.batch,), device=dev, generator=gen)
                dr = torch.rand(a.batch, device=dev, generator=gen) < a.slot_drop
                valid = ~dr[:, None] | (torch.arange(8, device=dev)[None] >= (8 - k)[:, None])
            Nb = (D["nat"][b], D["hid"][b]) if net.nat else None
            Pb = tuple(D[x][b] for x in ("obj", "rte", "lin", "trn")) if net.priv else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o, s = net(Vb, valid, D["E"][b], D["Qp"][b], Nb, Pb)
            w = tgt["w"][b] * inv[b][:, None]
            w = w / w.sum()
            la = (F.binary_cross_entropy_with_logits(o[..., 0], tgt["a"][b], reduction="none") * w).sum()
            wb = w * tgt["b_ok"][b]
            lb = (F.binary_cross_entropy_with_logits(o[..., 1], tgt["b"][b], reduction="none") * wb).sum() / wb.sum().clamp_min(1e-6)
            lr_ = hub(o[..., 5], tgt["clr"][b], w) + hub(o[..., 6], tgt["mg"][b], w)
            lf = hub(o[..., 2], tgt["a_s"][b], w) + hub(o[..., 3], tgt["a_t"][b], w) + hub(o[..., 4], tgt["b_s"][b], w)
            ls = (hub(s[..., 0], tgt["sa"][b], w[..., None]) + hub(s[..., 1], tgt["sm"][b], w[..., None])) if s is not None else o.new_zeros(())
            ll = {}
            for k in LC.LONG:
                wk = tgt["w" + k][b] * inv[b][:, None]
                ll[k] = (F.binary_cross_entropy_with_logits(o[..., H.IDX[k]], tgt[k][b].float(), reduction="none") * wk).sum() / wk.sum().clamp_min(1e-9)
            loss = la + lb + a.lam_r * lr_ + a.lam_f * lf + a.lam_s * ls + a.lam_long * sum(ll.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                run.scalars(dict(agent=float(la), boundary=float(lb), reg=float(lr_), first=float(lf), step=float(ls)) | {k: float(v) for k, v in ll.items()}, step, "loss/")
            if (step + 1) % a.eval == 0 or step + 1 == a.steps:
                out = H.predict(net, D, gval)
                r = val_read(Yv, out, ival)
                curve.append(dict(step=step + 1, **r))
                run.scalars({k: v for k, v in r.items() if not k.endswith("_pos")}, step + 1, "val/")
                run.info("step %d: loss a %.4f b %.4f A %.4f B %.4f C %.4f | val own agent %.3f boundary %.3f A %.3f (%d) B %.3f (%d) C %.3f (%d) | %.1f it/s", step + 1,
                         float(la), float(lb), float(ll["A"]), float(ll["B"]), float(ll["C"]), r["a"], r["b"], r["A"], r["A_pos"], r["B"], r["B_pos"], r["C"], r["C_pos"],
                         (step + 1 - step0) / (time.time() - t2))
                for h in H.HEADS:
                    if np.isfinite(r[h]) and r[h] > best_sel[h]:
                        best_sel[h], best[h] = r[h], {kk: v.detach().clone() for kk, v in net.state_dict().items()}
                tmp = st_f.with_suffix(".tmp")
                torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step + 1, best=best, best_sel=best_sel, curve=curve, kills=kills), tmp)
                tmp.rename(st_f)
        its = (a.steps - step0) / max(time.time() - t2, 1e-6)
        last = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        best = {h: (v if v is not None else last) for h, v in best.items()}
        final = {}
        for h in H.HEADS:
            net.load_state_dict(best[h])
            final[h] = val_read(Yv, H.predict(net, D, gval), ival)
        ck = run.path("ckpt.pt")
        torch.save(dict(best=best, config=vars(a) | dict(resume=None), emu=emu, esd=esd, nmu=nmu, nsd=nsd, val=final, cells=cells, cells_long=CL, curve=curve,
                        best_step={h: int(max((c for c in curve if np.isfinite(c[h])), key=lambda c: c[h], default=dict(step=a.steps))["step"]) for h in H.HEADS}),
                   ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").rename(ck)
        run.path("curve.json").write_text(json.dumps(curve, indent=1))
        st_f.unlink(missing_ok=True)
        run.summary.update(ckpt=str(ck), val={h: {k: final[h][k] for k in ("a", "b", "A", "B", "C")} for h in H.HEADS}, its=its, restarts=kills, n_fit=int(fit.sum()),
                           n_val=int(val.sum()), vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
        run.info("done: val (own plan, head selected per group) agent %.3f boundary %.3f A %.3f B %.3f C %.3f; %.1f it/s; %.1f GB VRAM", final["lat"]["a"], final["lat"]["b"],
                 final["A"]["A"], final["B"]["B"], final["C"]["C"], its, torch.cuda.max_memory_reserved() / 2 ** 30)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("train")
    p.add_argument("--rung", required=True, choices=H.RUNGS)
    p.add_argument("--tag", required=True)
    p.add_argument("--shards", type=int, nargs="+", default=None)
    p.add_argument("--slot-drop", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warm", type=int, default=300)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--enc", type=int, default=3)
    p.add_argument("--dec", type=int, default=3)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--lam-r", type=float, default=1.0)
    p.add_argument("--lam-f", type=float, default=1.0)
    p.add_argument("--lam-s", type=float, default=1.0)
    p.add_argument("--lam-long", type=float, default=1.0)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--host", type=int, default=1, help="1: vision tokens in host memory, gathered per batch (default); 0: all on the card (S0)")
    cli_args(p)
    a = ap.parse_args()
    {"train": cmd_train}[a.cmd](a)
