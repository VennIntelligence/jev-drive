"""BODY1 arm 4.3 trainer (prereg Amendment 4): the P2H10 recipe plus the agent hinge (A), hinge-only off-track rows (B) and the scorer-layer
drivable hinge on them (C). pp_train.py is not edited: its Store / PModel / Prefetch / dev_eval are used as they are and its Losses through
lib/loss43.Losses43. With --ho "" and --agent-lam 0 the row stream, the losses and the checkpoint are pp_train's (gate: `ident`).

Batch = (batch - k) rows drawn as pp_train draws them (imitation / anchor) + k hinge-only rows, --ho "ot1:4,yr1:4,bd4:5" per batch, drawn
uniformly from each family's rows on navsim/body1-train-logs (own rng stream: the normal row stream does not depend on the families).

  train  --tag P2H10B-P-s0 --data navtrain_full.s2of12 navtrain_full.s3of12 --steps 3000 --ho ot1:4,yr1:4,bd4:5 --agent-lam 10
  ident  --a <run dir of pp_train.py> --b <run dir of bd4_train.py with the switches off>     losses and weights equal bit for bit -> json

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/bd4_train.py train --tag ... (GPU; through the pool)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

ROAD = "runs/body1/labels_road/navtrain_s01234567891011.npz"


def cmd_train(a):
    import torch
    import ot_rows as OR
    import pp_train as T
    from drivable_hinge import Hinge
    from loss43 import Losses43, OffAgentHinge
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    ho = [(f, int(k)) for f, k in (x.split(":") for x in a.ho.split(","))] if a.ho else []
    fams = [f for f, _ in ho]
    hod = tuple(f"{f}_{d}" for f in fams for d in a.data)
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data) + hod, split=a.split, frames="warp", host=True, warmup=a.warmup,
                eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin, agent_lam=a.agent_lam, agent_margin=a.agent_margin,
                agent_side_margin=a.agent_side_margin, agent_labels=a.agent_labels)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train's row stream
    hrng = np.random.default_rng([cfg.seed, 0, 43])                  # hinge-only rows: own stream
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    names, logs = S.tab["names"], S.tab["log"]
    nd = [len(np.load(B.cache_root() / d / "tab.npz")["names"]) for d in cfg.data]
    n_base = sum(nd[:len(a.data)])
    fam = np.full(S.n, -1)
    o = n_base
    for j, f in enumerate(fams):
        m = sum(nd[len(a.data) * (1 + j):len(a.data) * (2 + j)])
        fam[o:o + m] = j
        o += m
    is_ho = fam >= 0
    tr_rows, dv_rows, sp = T.split_rows(dict(names=np.where(is_ho, "", names), log=logs, is_b2d=S.is_b2d, is_wod=S.is_wod), cfg.split)
    assert not is_ho[tr_rows].any() and not is_ho[dv_rows].any()
    trl = splits.load(B.TRAIN)
    in_tr = trl.mask(logs)
    pools = [np.flatnonzero((fam == j) & in_tr) for j in range(len(fams))]
    kk = [k for _, k in ho]
    k = sum(kk)
    nB = cfg.batch
    model = T.PModel("P2").to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], names, dev, cfg.hinge_margin, list(cfg.hinge_footprint)) if cfg.hinge_lam > 0 else None
    off = OR.offsets(cfg.data)
    agent2 = road = None
    if cfg.agent_lam > 0:                                            # A: imitation rows of the train logs + every hinge-only row
        agent2 = OffAgentHinge(data_dir() / cfg.agent_labels, np.where(in_tr, names, ""), off, dev, cfg.agent_margin,
                               None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
    if k and a.road_lam > 0:                                         # C: the scorer-layer raster (or, --road-labels, another one) on the hinge-only rows
        road = OR.off_hinge(replace(cfg, hinge_labels=(a.road_labels,), hinge_margin=a.road_margin), np.where(is_ho, names, ""), off, dev)
    n_fb = 0
    if road is not None and hinge is not None:                       # note (x): a row that does not start on the scorer-layer road keeps the NAVSIM raster
        hr = torch.nonzero(torch.as_tensor(is_ho, device=dev) & road.ok)[:, 0]
        fb = []
        with torch.no_grad():
            for i in range(0, len(hr), 4096):
                r = hr[i:i + 4096]
                z = torch.zeros(len(r), 8, device=dev)
                fb.append(r[road.margins(z, z, z, r).amin(1) < 0])
        fb = torch.cat(fb)
        road.sdf[fb] = hinge.sdf[fb]
        road.ok[fb] = hinge.ok[fb]
        n_fb = int(len(fb))
        fb_fam = np.bincount(fam[fb.cpu().numpy()], minlength=len(fams)).tolist()
    LS = Losses43(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None,
                  ho=torch.as_tensor(is_ho, device=dev) if k else None, fam=torch.as_tensor(fam, device=dev), fams=fams, agent2=agent2, road=road, road_lam=a.road_lam)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = T.proot("runs", a.tag)
    with Run("op_parity", f"train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"body1": vars(a)}) as run:
        for x in sp + ((trl,) if k or agent2 is not None else ()):
            run.use_split(x)
        run.info(f"{a.tag}: train {len(tr_rows)} normal rows, dev {len(dv_rows)}; hinge-only per batch of {nB}: "
                 + (", ".join(f"{f} {q} of {len(p)}" for (f, q), p in zip(ho, pools)) or "none")
                 + f"; hinge {cfg.hinge_lam} / {cfg.hinge_margin} m coverage {hinge.coverage if hinge else 0:.4f}; agent {cfg.agent_lam} / {cfg.agent_margin} m / side "
                 f"{cfg.agent_side_margin} rows {int(agent2.ok.sum()) if agent2 is not None else 0} (moved {agent2.moved if agent2 is not None else 0}); "
                 f"road {a.road_lam if road is not None else 0} / {a.road_margin} m rows {int(road.ok.sum()) if road is not None else 0} ({a.road_labels}); "
                 f"hinge-only rows not on the scorer-layer road at t0 (NAVSIM raster kept): {n_fb} {dict(zip(fams, fb_fam)) if n_fb else ''}")

        def draw():                                                   # pp_train's draw on the normal rows, then the hinge-only rows
            r = rng.choice(tr_rows, nB - k, replace=len(tr_rows) < nB - k)
            an = rng.random(nB - k) < cfg.d_frac
            if k:
                r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
                an = np.concatenate([an, np.zeros(k, bool)])
            return r, an, None

        def fetch(dr):
            return dr, S.front[dr[0]]
        S.front.pin = a.prefetch > 1
        Sd = T.dev_store(S, dv_rows, a.dev_card_gb)
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            (r, an, _), front = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            ego = S.ego[rows] * (~anchor)[:, None].float()
            total, Ls = LS(model(front, ego, S.tc[rows], None, None, nv=None), S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({q: float(v) for q, v in Ls.items()})
            if (step + 1) % 25 == 0 or step + 1 == cfg.steps:
                keys = sorted({q for h in hist for q in h})
                m = {q: float(np.mean([h[q] for h in hist if q in h])) for q in keys}
                m |= {q.replace("_posmean", "_posn"): float(sum(q in h for h in hist)) for q in keys if q.endswith("_posmean")}   # steps with a positive row
                hist = []
                run.scalars({f"loss/{q}": v for q, v in m.items()}, step + 1)
                el = time.time() - t0
                run.scalars({"throughput/steps_per_s": (step + 1) / el, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in m.items() if "/" not in q) + f"; {(step + 1) / el:.2f} it/s, "
                             f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = T.dev_eval(model.eval(), Sd, dv_rows, LS.W)
                model.train()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "body1": vars(a)}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_rows), ho_per_batch=k,
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


def cmd_ident(a):
    """Two runs (events.jsonl loss scalars + ckpt-final.pt) equal bit for bit?"""
    import torch

    def ev(d):
        out = {}
        for ln in open(_pl.Path(d) / "events.jsonl"):
            e = json.loads(ln)
            if e.get("kind") == "scalar" and e["tag"].split("/")[0] in ("loss", "dev") and e["tag"].count("/") == 1 and not e["tag"].endswith("_posn"):
                out[(e["step"], e["tag"])] = e["value"]
        return out
    ea, eb = ev(a.a), ev(a.b)
    common = sorted(set(ea) & set(eb), key=str)
    dl = max((abs(ea[q] - eb[q]) for q in common), default=float("nan"))
    ca, cb = (torch.load(json.load(open(_pl.Path(x) / "summary.json"))["ckpt"], map_location="cpu", weights_only=False)["model"] for x in (a.a, a.b))
    dw = max(float((ca["net"][q].float() - cb["net"][q].float()).abs().max()) for q in ca["net"])
    dp = max(float((ca["parity"][q].float() - cb["parity"][q].float()).abs().max()) for q in ca["parity"])
    res = dict(a=a.a, b=a.b, scalars_compared=len(common), only_a=len(set(ea) - set(eb)), only_b=len(set(eb) - set(ea)), max_abs_scalar_diff=dl,
               max_abs_net_weight_diff=dw, max_abs_adapter_weight_diff=dp, identical=bool(dl == 0 and dw == 0 and dp == 0 and len(common) > 0))
    print(json.dumps(res, indent=1))
    if a.out:
        _pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if res["identical"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--split", default="navsim/op-parity-full")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--hinge-lam", type=float, default=10.0, help="P2H10's on-log drivable hinge (NAVSIM raster)")
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p.add_argument("--ho", default="", help='hinge-only rows per batch by family, e.g. "ot1:4,yr1:4,bd4:5"; "" = none')
    p.add_argument("--agent-lam", type=float, default=0.0, help="A: agent hinge on imitation rows (train logs) and hinge-only rows; 0 = off")
    p.add_argument("--agent-margin", type=float, default=0.3)
    p.add_argument("--agent-side-margin", type=float, default=0.0)
    p.add_argument("--agent-labels", default="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    p.add_argument("--road-lam", type=float, default=10.0, help="C: drivable hinge on the hinge-only rows (0 = off)")
    p.add_argument("--road-margin", type=float, default=0.3)
    p.add_argument("--road-labels", default=ROAD, help="its raster (default: the scorer-layer raster; the NAVSIM raster for the ablation without C)")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--fetch-workers", type=int, default=2)
    p.add_argument("--dev-card-gb", type=float, default=4.0)
    p = sub.add_parser("ident")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default="")
    a = ap.parse_args()
    {"train": cmd_train, "ident": cmd_ident}[a.cmd](a)
