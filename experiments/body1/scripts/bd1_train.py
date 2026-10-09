#!/usr/bin/env python3
"""BODY1 arm S0 trainer (plans/2026-10-10-body1-prereg.md section 3, Amendment 1): the contact predictor on frozen Cinque tokens (lib/contact_head.py).

  steps    (CPU) per-interval clearance / margin labels of every row unit -> $DATA_DIR/runs/body1/s0/steps/<cache dir>.npz (a, m: (n, 24, 9) fp16)
  train    (GPU) --arch step|crit|blind [--shards 2 3 4] [--frac 0.25] [--slots 8] --tag NAME --seed S
           Fits on body1-train-logs minus their validation part (contact_head.is_val); model selection (best step) on that validation part only.
           Hold logs are never loaded here. All tokens of the selected states live on the card (fp16, 0.26 MB per state); the host keeps the
           row tables only. Resumable: state.pt (model, optimiser, step, best; the sampler is reseeded from seed and step) is written atomically at every evaluation and
           `--auto` reopens the unfinished run of the same tag + seed (pointer file s0/runs/<tag>-s<seed>.dir), so a SIGKILL costs minutes:
             scripts/tmux_run.sh bd1-<tag> experiments/body1/scripts/bd1_retry.sh 5 env CUDA_VISIBLE_DEVICES=2 $DATA_DIR/envs/op-train/bin/python \
               experiments/body1/scripts/bd1_train.py train --arch step --tag full --seed 0 --auto
           -> <run dir>/ckpt.pt (best validation step), curve.json; DONE holds the validation reads.
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

import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402

FAMS = ("log", "ot1", "yr1")


# ---------------------------------------------------------------- per-interval labels
def _steps(u):
    f, k = u
    d = B.cdir(f, k)
    z = dict(np.load(B.root() / "rows" / f"{d}.npz"))
    a, m = C.step_labels(z, B.labels())
    da, dm = np.abs(a.astype(np.float32).min(-1) - z["a_clr"].astype(np.float16).astype(np.float32)).max(), np.abs(m.astype(np.float32).min(-1) - z["b_margin"].astype(np.float16).astype(np.float32)).max()
    assert da < 2e-2 and dm < 2e-2, f"{d}: step labels do not reproduce the row labels ({da:.3g}, {dm:.3g})"
    out = C.sroot() / "steps" / f"{d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".tmp.npz"), a=a, m=m)
    out.with_suffix(".tmp.npz").rename(out)
    return d, len(a), float(da), float(dm)


def cmd_steps(a):
    from jevdrive import par
    from jevdrive.run import Run
    with Run("body1", "s0/steps", config=vars(a)) as run:
        units = [(f, k) for f in FAMS for k in range(B.NSH)]
        r = par.pmap(_steps, units, workers=min(a.workers, len(units)), run=run, desc="row units", skip=lambda u: (C.sroot() / "steps" / f"{B.cdir(*u)}.npz").exists() and not a.force)
        r.raise_if_failed()
        run.summary.update(units=len(units))


# ---------------------------------------------------------------- training
def val_read(R, Y, out, rows):
    """AUCs on the validation states `rows` (indices into R): the student's own plan (query slots 0, 1) and all queries."""
    r = {}
    for name, qs in (("own", list(C.OWN)), ("all", slice(None))):
        for k, o in (("a", 0), ("b", 1)):
            ok = Y[f"{k}_ok"][rows][:, qs]
            r[f"{name}_{k}"] = C.auc(Y[k][rows][:, qs][ok], out[:, qs, o][ok])
            r[f"{name}_{k}_pos"] = int(Y[k][rows][:, qs][ok].sum())
    r["sel"] = 0.5 * (r["own_a"] + r["own_b"])
    return r


def cmd_train(a):
    import torch
    import torch.nn.functional as F
    from jevdrive.data import splits
    from jevdrive.run import Run
    ptr = C.sroot() / "runs" / f"{a.tag}-s{a.seed}.dir"
    resume = a.resume
    if a.auto and resume is None and ptr.exists():
        d = _pl.Path(ptr.read_text().strip())
        resume = d if (d / "state.pt").exists() and not (d / "DONE").exists() else None
    torch.set_num_threads(4)
    with Run("body1", f"s0/{a.tag}", seed=a.seed, config=vars(a), resume=resume) as run:
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
        keep_fit = C.frac_logs(logs_fit, a.frac) if a.frac < 1 else set(logs_fit)
        val_of = {l: C.is_val(l) for l in logs_all.tolist()}
        R = C.load_rows(a.fams, shards, lambda log, hold: ~hold & np.array([val_of[l] or l in keep_fit for l in log.tolist()]), steps=True)
        n = len(R["gi"])
        assert s_tr.mask(R["log"]).all() and not s_ho.mask(R["log"]).any() and not R["hold"].any(), "a hold log reached the trainer"
        val = np.array([val_of[l] for l in R["log"].tolist()])
        fit = ~val
        assert not set(R["log"][fit].tolist()) & set(R["log"][val].tolist())
        T = dict(np.load(B.root() / "taxonomy" / "tax.npz"))
        cls, _ = C.classes(T)
        Y = C.targets(R)
        W, cells = C.balance(cls[R["gi"]], Y, fit)
        run.info("states: fit %d (%d logs), val %d (%d logs); cells (class x type) %s; tables in %.0f s", fit.sum(), len(set(R["log"][fit].tolist())), val.sum(),
                 len(set(R["log"][val].tolist())), cells.astype(int).tolist(), time.time() - t0)
        # ---- device tensors
        G = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        vision = a.arch != "blind"
        t1 = time.time()
        V = C.load_tokens(R["src"], n, dev) if vision else None
        run.info("tokens on the card: %.1f GB in %.0f s", (V.numel() * 2 / 2 ** 30) if vision else 0.0, time.time() - t1)
        emu, esd = R["ego"][fit].mean(0), R["ego"][fit].std(0) + 1e-6
        E, Qp = G((R["ego"] - emu) / esd), G(R["q"])
        clipn = lambda x, c: np.where(x < 90, np.clip(x, *c), np.nan)  # noqa: E731  (nan = no target)
        clr = np.where(R["a_clr"] < 90, np.clip(R["a_clr"], *C.A_CLIP), C.A_CLIP[1])           # no valid object: the far clip
        tgt = dict(a=G(Y["a"]), b=G(Y["b"]), b_ok=G(Y["b_ok"]), w=G(W), clr=G(clr), mg=G(clipn(R["b_margin"], C.M_CLIP)),
                   a_s=G(np.where(Y["a"], R["a_s"] / C.S_SCALE, np.nan)), a_t=G(np.where(Y["a"], R["a_t"] / C.T_SCALE, np.nan)), b_s=G(np.where(Y["b"], R["b_s"] / C.S_SCALE, np.nan)),
                   sa=G(np.where(R["sa"].astype(np.float32) < 90, np.clip(R["sa"].astype(np.float32), *C.A_CLIP), C.A_CLIP[1]), torch.float16),
                   sm=G(clipn(R["sm"].astype(np.float32), C.M_CLIP), torch.float16))
        Ws = W.sum(1).astype(np.float64)
        p = np.where(fit, Ws ** a.alpha, 0.0)                                      # tempered state sampling; the loss carries the rest of the weight
        cdf = torch.as_tensor(np.cumsum(p / p.sum()), device=dev)
        inv = G(np.where(Ws > 0, 1.0 / np.maximum(Ws, 1e-12) ** a.alpha, 0.0))
        ival = np.flatnonzero(val)
        gval = torch.as_tensor(ival, device=dev)
        # ---- model
        torch.manual_seed(a.seed)
        net = C.build(a.arch, a.d, a.enc, a.dec, a.drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        lr_at = lambda s: min(1.0, (s + 1) / a.warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))  # noqa: E731
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        step0, best, best_sel, curve, kills = 0, None, -np.inf, [], 0
        st_f = run.path("state.pt")
        if st_f.exists():
            st = torch.load(st_f, map_location=dev, weights_only=False)
            net.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), gen.manual_seed(a.seed * 100003 + st["step"])
            step0, best, best_sel, curve, kills = st["step"], st["best"], st["best_sel"], st["curve"], st["kills"] + 1
            run.info("resumed at step %d (restart %d)", step0, kills)
        run.info("%s: %.2f M parameters", a.arch, sum(x.numel() for x in net.parameters()) / 1e6)
        hub = lambda x, y, w: ((F.huber_loss(x, torch.nan_to_num(y.float()), delta=a.delta, reduction="none") * w * torch.isfinite(y)).sum() / (w * torch.isfinite(y)).sum().clamp_min(1e-6))  # noqa: E731
        t2 = time.time()
        for step in run.tqdm(range(step0, a.steps), desc=f"train {a.tag}", initial=step0, total=a.steps):
            for g in opt.param_groups:
                g["lr"] = a.lr * lr_at(step)
            b = torch.searchsorted(cdf, torch.rand(a.batch, device=dev, generator=gen, dtype=cdf.dtype)).clamp_max(n - 1)
            valid = torch.ones((a.batch, 8), dtype=torch.bool, device=dev)
            if vision:                                                             # slot dropout: only the k newest slots, k ~ U{1..7}, on a share of rows
                k = torch.randint(1, 8, (a.batch,), device=dev, generator=gen)
                dr = torch.rand(a.batch, device=dev, generator=gen) < a.slot_drop
                valid = ~dr[:, None] | (torch.arange(8, device=dev)[None] >= (8 - k)[:, None])
                valid = valid & (torch.arange(8, device=dev)[None] >= 8 - a.slots)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o, s = net(V[b] if vision else None, valid, E[b], Qp[b])
            w = tgt["w"][b] * inv[b][:, None]
            w = w / w.sum()
            la = (F.binary_cross_entropy_with_logits(o[..., 0], tgt["a"][b], reduction="none") * w).sum()
            wb = w * tgt["b_ok"][b]
            lb = (F.binary_cross_entropy_with_logits(o[..., 1], tgt["b"][b], reduction="none") * wb).sum() / wb.sum().clamp_min(1e-6)
            lr_ = hub(o[..., 5], tgt["clr"][b], w) + hub(o[..., 6], tgt["mg"][b], w)
            lf = hub(o[..., 2], tgt["a_s"][b], w) + hub(o[..., 3], tgt["a_t"][b], w) + hub(o[..., 4], tgt["b_s"][b], w)
            ls = (hub(s[..., 0], tgt["sa"][b], w[..., None]) + hub(s[..., 1], tgt["sm"][b], w[..., None])) if s is not None else o.new_zeros(())
            loss = la + lb + a.lam_r * lr_ + a.lam_f * lf + a.lam_s * ls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                run.scalars(dict(agent=float(la), boundary=float(lb), reg=float(lr_), first=float(lf), step=float(ls)), step, "loss/")
            if (step + 1) % a.eval == 0 or step + 1 == a.steps:
                out, _ = C.predict(net, V, E, Qp, gval)
                r = val_read(R, Y, out, ival)
                curve.append(dict(step=step + 1, **r))
                run.scalars({k: v for k, v in r.items() if not k.endswith("_pos")}, step + 1, "val/")
                run.info("step %d: loss a %.4f b %.4f reg %.4f first %.4f step %.4f | val own agent %.3f (%d) boundary %.3f (%d), all %.3f / %.3f | %.1f it/s", step + 1, float(la), float(lb),
                         float(lr_), float(lf), float(ls), r["own_a"], r["own_a_pos"], r["own_b"], r["own_b_pos"], r["all_a"], r["all_b"], (step + 1 - step0) / (time.time() - t2))
                if r["sel"] > best_sel:
                    best_sel, best = r["sel"], {k: v.detach().clone() for k, v in net.state_dict().items()}
                tmp = st_f.with_suffix(".tmp")
                torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step + 1, best=best, best_sel=best_sel, curve=curve, kills=kills), tmp)
                tmp.rename(st_f)
        its = (a.steps - step0) / max(time.time() - t2, 1e-6)
        net.load_state_dict(best)
        # ---- final validation reads of the selected step: slot count (the AlpaSim input standard has few slots on first decisions)
        out, _ = C.predict(net, V, E, Qp, gval)
        final = val_read(R, Y, out, ival)
        slots = {}
        if vision:
            for k in (1, 2, 4):
                slots[k] = val_read(R, Y, C.predict(net, V, E, Qp, gval, slots=k)[0], ival)
                run.info("val with the %d newest slots: own agent %.3f boundary %.3f", k, slots[k]["own_a"], slots[k]["own_b"])
        ck = run.path("ckpt.pt")
        torch.save(dict(model=best, config=vars(a) | dict(resume=None), emu=emu, esd=esd, val=final, val_slots=slots, cells=cells, curve=curve), ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").rename(ck)
        run.path("curve.json").write_text(json.dumps(curve, indent=1))
        st_f.unlink(missing_ok=True)
        run.summary.update(ckpt=str(ck), val=final, val_slots=slots, best_step=int(max(curve, key=lambda c: c["sel"])["step"]), its=its, restarts=kills, n_fit=int(fit.sum()), n_val=int(val.sum()),
                           logs_fit=len(set(R["log"][fit].tolist())), vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
        run.info("done: val own agent %.3f boundary %.3f; %.1f it/s; %.1f GB VRAM; %d restarts", final["own_a"], final["own_b"], its, torch.cuda.max_memory_reserved() / 2 ** 30, kills)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("steps")
    p.add_argument("--workers", type=int, default=16)
    cli_args(p)
    p = sp.add_parser("train")
    p.add_argument("--arch", default="step", choices=("step", "crit", "blind"))
    p.add_argument("--tag", required=True)
    p.add_argument("--fams", nargs="+", default=list(FAMS))
    p.add_argument("--shards", type=int, nargs="+", default=None)
    p.add_argument("--frac", type=float, default=1.0, help="share of the fit logs (learning curve)")
    p.add_argument("--slots", type=int, default=8, help="newest slots the head may see in training (ablation)")
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
    p.add_argument("--alpha", type=float, default=0.5, help="state sampling probability ~ (sum of row weights) ** alpha")
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--lam-r", type=float, default=1.0)
    p.add_argument("--lam-f", type=float, default=1.0)
    p.add_argument("--lam-s", type=float, default=1.0)
    p.add_argument("--auto", action="store_true", help="reopen the unfinished run of this tag + seed if there is one")
    cli_args(p)
    a = ap.parse_args()
    {"steps": cmd_steps, "train": cmd_train}[a.cmd](a)
