"""AP2 trainer: the op_parity P2 / SH30 recipe (experiments/op_parity/scripts/pp_train.py: frozen Cinque vision tokens from the cache, plan
pathway + ego adapter, navtrain imitation + anchor rows + drivable hinge) on training rows built as AlpaSim delivers a decision
(experiments/alpasim/lib/ap2_inputs.py, plans/2026-10-08-alpasim-aligned-prereg.md). pp_train's Store / PModel / Losses are used unchanged.

Per imitation row a decision type m (keyframes available) is drawn with the rollout's own mix (0.1 / 0.1 / 0.1 / 0.7 for m = 1..4):
  slots  m = 4: the cached W tokens; m < 4: scripts/ap2_prep.py's tokens, rule `zero` (only the real slots, the rest invalid) or, arm AB
         (--cold backwarp), rule `backwarp` (8 slots, fabricated history)
  ego    ap2_core.ego_table: history poses filled by the serving rule, velocity / acceleration as DynamicState defines them at that
         decision, command = the shipped route rule on AlpaSim's route rebuilt for the token (scripts/ap2_route.py); --route adds the route
         waypoints themselves (arm R)
Anchor rows keep m = 4 and zeroed inputs (the teacher is shipped Cinque on the full W frames). A row without a rebuilt route for the drawn
m falls back to m = 4; rows without one for m = 4 are not trained on. --std navsim trains the NAVSIM-aligned recipe in this loop (m = 4,
the tab's ego features: the SHP / SH30 recipe; trainer check). Checkpoints land beside op_parity's ($DATA_DIR/runs/op_parity/runs/<tag>/).

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/ap2_train.py --tag AP2P-A-s0 --data navtrain_full.s2of12 ... --split navsim/op-parity-s234
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib")]
import argparse, time  # noqa: E401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ap2_core as AC  # noqa: E402
import ap2_inputs as AI  # noqa: E402
import pp_train as T  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run  # noqa: E402

AROOT = data_dir() / "runs/alpasim/ap2"


def routes(datas, names) -> tuple:
    z = [np.load(AROOT / "route" / f"{d}.npz") for d in datas]
    assert np.concatenate([x["names"] for x in z]).tolist() == list(names), "route files and the op_parity tabs disagree on the token order"
    return np.concatenate([x["wp"] for x in z]), np.concatenate([x["ok"] for x in z])


def slots(S, cold, rows, m, rule: str):
    """Batch slot tokens under a cold-start rule: rows (B,) store rows, m (B,) keyframes -> front (B, 8, 32, 512) fp16 on the device, nv (B,)."""
    front = S.front[rows]
    nv = torch.full((len(rows),), 8, device=front.device)
    sel = np.flatnonzero(m < 4)
    if len(sel):
        c = cold[rows[sel]]
        for mm in (1, 2, 3):
            j = torch.as_tensor(np.flatnonzero(m[sel] == mm), device=front.device)
            if not len(j):
                continue
            i = torch.as_tensor(sel, device=front.device)[j]
            if rule == "backwarp":
                front[i] = c[j, mm - 1]
            else:
                n = AI.N_SLOT[mm]
                front[i, :8 - n] = 0
                front[i, 8 - n:] = c[j][:, AI.COLD_AT[mm]]
                nv[i] = n
    return front, nv


@torch.no_grad()
def dev_by_m(model, S, cold, EGO, ok, dev_rows, W, rule, bs=256) -> dict:
    """ADE of the 8 poses to the log on the dev rows, per decision type m (inputs on)."""
    pi = torch.as_tensor(S.pi, device=S.ego.device)
    out = {}
    for mm in (1, 2, 3, 4):
        rows_m = dev_rows[ok[dev_rows, mm - 1]]
        acc = []
        for i in range(0, len(rows_m), bs):
            r = rows_m[i:i + bs]
            rt = torch.as_tensor(r, device=S.ego.device)
            front, nv = slots(S, cold, r, np.full(len(r), mm), rule)
            p = model(front, EGO[rt, mm - 1], S.tc[rt], nv=nv).float()[:, pi].view(-1, 33, 15)
            x, y, _ = T.rear(p, S.cam_x[rt], W)
            acc.append(torch.hypot(x - S.fut[rt][..., 0], y - S.fut[rt][..., 1]).mean(1)[S.has_fut[rt]])
        out[f"ade_m{mm}"] = float(torch.cat(acc).mean())
    return out


def main(a):
    dev = torch.device("cuda")
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data), split=a.split, frames="warp", host=True, warmup=a.warmup,
                eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train's row stream
    mrng = np.random.default_rng([cfg.seed, 0, 23])                  # decision types: own stream
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    tabs = dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d, is_wod=S.is_wod)
    tr_rows, dv_rows, sp = T.split_rows(tabs, cfg.split)
    ap = a.std == "alpasim"
    if ap:
        wp, ok = routes(cfg.data, S.tab["names"])
        EGO = torch.from_numpy(AC.ego_table(S.tab, wp, a.route)).to(dev)
        cold = T.Tokens([AROOT / "cache" / d / ("bw.npy" if a.cold == "backwarp" else "cold.npy") for d in cfg.data], dev, host=True)
        tr_rows, dv_rows = tr_rows[ok[tr_rows, 3]], dv_rows[ok[dv_rows, 3]]
        S.ego = EGO[:, 3]                                            # T.dev_eval reads S.ego: the m = 4 inputs
    else:
        ok, cold, EGO = np.ones((S.n, 4), bool), None, S.ego[:, None].expand(-1, 4, -1)
    mix = np.asarray(a.mix, float) / np.sum(a.mix) if ap else np.array([0, 0, 0, 1.0])
    model = AC.widen(T.PModel("P2"), ap and a.route).to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    hinge = None
    if cfg.hinge_lam > 0:
        from drivable_hinge import Hinge
        hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], S.tab["names"], dev, cfg.hinge_margin, list(cfg.hinge_footprint))
    LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = T.proot("runs", a.tag)
    nB = cfg.batch
    with Run("alpasim", f"ap2-train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"ap2": vars(a)}) as run:
        for x in sp:
            run.use_split(x)
        run.info(f"{a.tag}: train {len(tr_rows)} dev {len(dv_rows)} rows ({a.std} inputs, cold {a.cold}, route {a.route}, mix {mix.round(3).tolist()}); "
                 f"routes ok per m {ok.mean(0).round(4).tolist()}; hinge coverage {hinge.coverage if hinge else 0:.4f}")

        def draw():
            r = rng.choice(tr_rows, nB, replace=len(tr_rows) < nB)
            an = rng.random(nB) < cfg.d_frac
            m = mrng.choice(4, nB, p=mix) + 1
            m[an | ~ok[r, m - 1]] = 4
            return r, an, m

        def fetch(dr):
            return dr, slots(S, cold, dr[0], dr[2], a.cold) if ap else (S.front[dr[0]], None)
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
                raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({k: float(v) for k, v in Ls.items()})
            if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                mean = {k: float(np.mean([h[k] for h in hist if k in h])) for k in hist[-1]}
                hist = []
                run.scalars({f"loss/{k}": v for k, v in mean.items()} | {"throughput/steps_per_s": (step + 1) / (time.time() - t0)}, step + 1)
                run.info(f"step {step + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in mean.items()) + f"; {(step + 1) / (time.time() - t0):.2f} it/s, "
                         f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                run.status(f"step {step + 1}/{cfg.steps}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = T.dev_eval(model.eval(), S, dv_rows, LS.W)
                if ap:
                    ev |= dev_by_m(model, S, cold, EGO, ok, dv_rows, LS.W, a.cold)
                model.train()
                run.scalars({f"dev/{k}": v for k, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items()))
                run.summary.update({f"dev_{k}": v for k, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "ap2": vars(a)}, d / "ckpt-final.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--split", default="navsim/op-parity-s234")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--hinge-lam", type=float, default=30.0)
    ap.add_argument("--hinge-margin", type=float, default=0.5)
    ap.add_argument("--std", default="alpasim", choices=["alpasim", "navsim"], help="input standard of the training rows")
    ap.add_argument("--cold", default="zero", choices=["zero", "backwarp"], help="slot rule for decisions with fewer than 4 keyframes")
    ap.add_argument("--route", action="store_true", help="arm R: the route waypoints as adapter ego features")
    ap.add_argument("--mix", type=float, nargs=4, default=list(AI.MIX), help="share of training rows with m = 1, 2, 3, 4 keyframes")
    main(ap.parse_args())
