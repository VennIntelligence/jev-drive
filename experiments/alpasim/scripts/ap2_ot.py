"""AP2 + off-track rows (lane OT2 piece B, plans/2026-10-09-ot2-dose-prereg.md; decision 210): the AlpaSim input standard of ap2_train.py with a
share of every batch drawn from statically perturbed (re-projected) navtrain rows, the machinery of experiments/op_parity/scripts/ot_rows.py
(decision 198). Dose = perturbation amplitude x batch share.

  prep   --amp a15 --data navtrain_full.s2of12 [--limit N]   ot_rows.py's prep with the amplitude preset (AMP below); a05 is the existing
         `ot1` cache (+-0.5 m, +-2 deg), a15 writes `ot2` (+-1.5 m, +-5 deg: the 1-2 m closed-loop drift of decision 202 and the heading a
         1.5 m drift over 3 s at 8 m/s goes with). Tokens, tab and teacher land in $DATA_DIR/runs/op_parity/cache/<prefix>_<data>[@warp]/.
  train  --tag APO-a15m25-s0 --amp a15 --ot-mass 0.25 ...     ap2_train.py's loop (decision types m = 1..4 with the served backwarp rule,
         AlpaSim ego definitions, AlpaSim route command) for the normal rows; the off-track rows are m = 4 decisions (full real history: the
         state a rollout is in when it has drifted) with the same ego definitions, the route re-expressed in the perturbed frame, the target
         = the logged future in the perturbed frame, the hinge on the plan mapped back to the logged frame (ot_rows.off_hinge). They are
         never anchor rows and never cold-start rows. --ot-mass 0 reproduces ap2_train.py's row stream and losses (trainer check).

Checkpoints land beside op_parity's ($DATA_DIR/runs/op_parity/runs/<tag>/ckpt-final.pt) with the `ap2` record ap2_core.load_model reads,
so `run.sh <dir> ap2` with AP2_TAG=<tag> serves them unchanged.

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/ap2_ot.py train --tag APO-a05m10-s0 --amp a05 --ot-mass 0.1 --data ... --cold backwarp
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts")]
import argparse, time  # noqa: E401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402

import ot_rows as OR  # noqa: E402

# preset -> cache prefix, DY (m), DPSI (rad), YMAX (m: the drift history may start this far off the logged path)
AMP = {"a05": ("ot1", 0.5, np.radians(2.0), 1.0), "a15": ("ot2", 1.5, np.radians(5.0), 3.0)}


def use_amp(amp: str) -> str:
    OR.OT, OR.DY, OR.DPSI, OR.YMAX = AMP[amp]
    return OR.OT


def wp_to_frame(wp, dy, dpsi):
    """Route waypoints (n, 20, 2) in the logged rig frame -> the perturbed rig frame (0, dy, dpsi); NaN padding stays NaN."""
    c, s = np.cos(dpsi)[:, None], np.sin(dpsi)[:, None]
    x, y = wp[..., 0], wp[..., 1] - dy[:, None]
    return np.stack([c * x + s * y, -s * x + c * y], -1).astype(wp.dtype)


def cmd_prep(a):
    use_amp(a.amp)
    OR.cmd_prep(a)


def cmd_train(a):
    import torch
    import ap2_core as AC
    import ap2_train as AT
    import pp_train as T
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    dev = torch.device("cuda")
    pre_ = use_amp(a.amp)
    ot = tuple(f"{pre_}_{d}" for d in a.data) if a.ot_mass > 0 else ()
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data) + ot, split=a.split, frames="warp", host=True, warmup=a.warmup,
                eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train's / ap2_train's row stream
    mrng = np.random.default_rng([cfg.seed, 0, 23])                  # ap2_train's decision types
    orng = np.random.default_rng([cfg.seed, 0, 41])                  # off-track rows: own stream, so --ot-mass 0 is ap2_train bit for bit
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    tr_rows, dv_rows, sp = T.split_rows(dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d, is_wod=S.is_wod), cfg.split)
    nbs = [len(np.load(OR.CR / d / "tab.npz")["names"]) for d in a.data]
    n_base = sum(nbs)
    wp, ok = AT.routes(a.data, S.tab["names"][:n_base])
    off = OR.offsets(cfg.data)
    if ot:                                                           # routes of the off-track rows: their token's m = 4 route seen from the perturbed pose
        src = np.concatenate([np.load(OR.CR / d / "tab.npz")["src_row"] + o for d, o in zip(ot, np.cumsum([0] + nbs))])
        assert (S.tab["names"][:n_base][src] == S.tab["names"][n_base:]).all()
        w4 = wp_to_frame(wp[src, 3], off[n_base:, 0].astype(np.float64), off[n_base:, 1].astype(np.float64))
        wp, ok = np.concatenate([wp, np.repeat(w4[:, None], 4, 1)]), np.concatenate([ok, ok[src]])
    EGO = torch.from_numpy(AC.ego_table(S.tab, wp, False)).to(dev)
    cold = T.Tokens([AT.AROOT / "cache" / d / ("bw.npy" if a.cold == "backwarp" else "cold.npy") for d in a.data], dev, host=True)
    tr_rows, dv_rows = tr_rows[ok[tr_rows, 3]], dv_rows[ok[dv_rows, 3]]
    S.ego = EGO[:, 3]                                                # T.dev_eval reads S.ego: the m = 4 inputs
    is_ot = np.arange(S.n) >= n_base
    tr_o, tr_t, dv_o, dv_t = tr_rows[~is_ot[tr_rows]], tr_rows[is_ot[tr_rows]], dv_rows[~is_ot[dv_rows]], dv_rows[is_ot[dv_rows]]
    nB = cfg.batch
    k = int(round(nB * a.ot_mass))
    assert (k == 0) == (len(ot) == 0) and (k == 0 or len(tr_t) >= k)
    mix = np.asarray(a.mix, float) / np.sum(a.mix)
    model = T.PModel("P2").to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_o, device=dev)].float().std(0).clamp_min(1e-3)        # the normaliser of the reference recipe (normal rows)
    hinge = None
    if cfg.hinge_lam > 0 and k:
        hinge = OR.off_hinge(cfg, S.tab["names"], off, dev)
    elif cfg.hinge_lam > 0:
        from drivable_hinge import Hinge
        hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], S.tab["names"], dev, cfg.hinge_margin, list(cfg.hinge_footprint))
    LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = T.proot("runs", a.tag)
    rec = dict(std="alpasim", cold=a.cold, route=False, mix=list(a.mix))                    # what ap2_core.load_model reads
    with Run("alpasim", f"apo-train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"ap2": rec, "ot": vars(a) | {"amp_values": [float(x) if not isinstance(x, str) else x for x in AMP[a.amp]]}}) as run:
        for x in sp:
            run.use_split(x)
        run.info(f"{a.tag}: train {len(tr_o)} normal + {len(tr_t)} off-track rows ({pre_}), dev {len(dv_o)} + {len(dv_t)}; {k} off-track rows per batch of {nB} "
                 f"({k / nB:.3f}); mix {mix.round(3).tolist()}, cold {a.cold}; routes ok per m {ok[:n_base].mean(0).round(4).tolist()}; hinge coverage "
                 f"{hinge.coverage if hinge else 0:.4f}; off-track |dy| mean {np.abs(off[is_ot, 0]).mean() if k else 0:.3f} m, |dpsi| mean "
                 f"{np.degrees(np.abs(off[is_ot, 1])).mean() if k else 0:.2f} deg")

        def draw():
            r = rng.choice(tr_o, nB - k, replace=len(tr_o) < nB - k)
            an = rng.random(nB) < cfg.d_frac
            m = mrng.choice(4, nB, p=mix) + 1
            if k:
                r = np.concatenate([r, orng.choice(tr_t, k, replace=False)])
                an[nB - k:] = False                                  # off-track rows are imitation rows only
                m[nB - k:] = 4                                       # ... and full-history decisions
            m[an | ~ok[r, m - 1]] = 4
            return r, an, m

        def fetch(dr):
            return dr, AT.slots(S, cold, dr[0], dr[2], a.cold)
        pre = ThreadPoolExecutor(2)
        nxt = pre.submit(fetch, draw())
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            (r, an, m), (front, nv) = nxt.result()
            if step + 1 < cfg.steps:
                nxt = pre.submit(fetch, draw())
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            ego = EGO[rows, torch.as_tensor(m - 1, device=dev)] * (~anchor)[:, None].float()
            total, Ls = LS(model(front, ego, S.tc[rows], nv=nv), S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({q: float(v) for q, v in Ls.items()})
            if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                mean = {q: float(np.mean([h[q] for h in hist if q in h])) for q in hist[-1]}
                hist = []
                el = time.time() - t0
                run.scalars({f"loss/{q}": v for q, v in mean.items()} | {"throughput/steps_per_s": (step + 1) / el}, step + 1)
                run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in mean.items()) + f"; {(step + 1) / el:.2f} it/s, "
                         f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                run.status(f"step {step + 1}/{cfg.steps}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = T.dev_eval(model.eval(), S, dv_o, LS.W) | AT.dev_by_m(model, S, cold, EGO, ok, dv_o, LS.W, a.cold)
                if len(dv_t):
                    ev |= {"ot_" + q: v for q, v in T.dev_eval(model, S, dv_t, LS.W).items()}
                model.train()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "ap2": rec, "ot": vars(a)}, d / "ckpt-final.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_o), n_train_ot=len(tr_t), ot_per_batch=k,
                           vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


if __name__ == "__main__":
    import ap2_inputs as AI
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--amp", required=True, choices=list(AMP))
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--zero", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--amp", default="a05", choices=list(AMP))
    p.add_argument("--ot-mass", type=float, default=0.1, help="share of every batch drawn from the off-track rows (0 = ap2_train.py)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--split", default="navsim/op-parity-s234")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--hinge-lam", type=float, default=30.0)
    p.add_argument("--hinge-margin", type=float, default=0.5)
    p.add_argument("--cold", default="backwarp", choices=["zero", "backwarp"])
    p.add_argument("--mix", type=float, nargs=4, default=list(AI.MIX))
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train}[a.cmd](a)
