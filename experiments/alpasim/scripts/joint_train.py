"""JOINT trainer: one trainer that carries every recipe flag of the lambda-10 family (lane JOINT: phase 1, the nuPlan submission #2 candidate,
plans/2026-10-11-joint1-prereg.md; phase 2, EYES-vs-SENSE, experiments/eyes_sense/plans/2026-10-11-eyes-sense-prereg.md).

Nothing it imports is edited: pp_train's Store / PModel / Prefetch / dev_eval / Losses, ap2_train's routes / slots / dev_by_m, ap2_core.ego_table,
ap2_ot.wp_to_frame, ot_rows' off_hinge / offsets, loss43's Losses43 / OffAgentHinge (experiments/body1/lib, whose semantics stay as they are).

Recipe flags (all off = experiments/op_parity/scripts/pp_train.py --arm P2 --frames warp --host --hinge-lam 10, the P2H10 recipe):
  --std alpasim     normal rows built as AlpaSim delivers a decision (ap2_train.py: m = 1..4 keyframes drawn 0.1 / 0.1 / 0.1 / 0.7, missing slots by
                    the served backwarp rule, the simulator's ego definitions, AlpaSim's route generator -> 4-way command); rows without a rebuilt
                    m = 4 route are not trained on. Perturbed rows (yaw-rate and hinge-only) are m = 4 decisions with their source token's route
                    re-expressed in their own frame (ap2_ot.py). The checkpoint carries ap2_ot's `ap2` record, so the AlpaSim image and run.sh serve
                    it through ap2_driver.py (JEV_DRIVER=auto); NAVSIM-standard checkpoints go through sh30_driver.py.
  --yr-mass F       round(batch x F) imitation rows of every batch drawn from the yaw-rate rows (ot3_rows.py, caches yr1_*: target = the logged
                    future re-expressed in the perturbed frame, never anchor rows); the drivable hinge then runs in each row's own frame
                    (ot_rows.off_hinge), as in ot_rows.py / ap2_ot.py
  --ho / --ho-w / --ho-excl / --agent-lam / --agent-* / --road-* / --shape   bd4_train.py's body lesson (loss43.Losses43): the agent hinge on the
                    own plan of every imitation row of the body1 train logs (yaw-rate rows included, in their own frame) and of every hinge-only
                    row; hinge-only off-track rows (ot1 / yr1 / bd4: no imitation target) with the road-and-lane raster; --shape: the new
                    hinges see the plan's shape only. Under --std alpasim a hinge-only row also needs its source token's m = 4 route.
  --mem vtr_<name>  a frozen token bank through the adapter's memory channel (pp_train --mem: arm P2+vtr_<name>, the bank masked on 25 % of the
                    rows from its own rng stream; vtr_W*: three views CAM_F0 / CAM_L0 / CAM_R0). Hinge-only rows have no bank of their own (their
                    side views were never rendered) and always read the memory masked; their bank row is their source token's (finite, unread).
  --label-bank      hinge rasters once per distinct label (lib/drivable_hinge.RowBank; same values bit for bit)

Batch of B rows = (B - k_yr - k_ho) normal rows (imitation, or anchor with p = 0.25) + k_yr yaw-rate rows + k_ho hinge-only rows. Every rng stream
is the one of the trainer its rows come from, so each sub-recipe reproduces its original trainer bit for bit (`ident`): normal rows [seed, 0]
(and the yaw-rate rows under the NAVSIM standard, as ot_rows.py does), decision types [seed, 0, 23] (ap2_train.py), yaw-rate rows under the AlpaSim
standard [seed, 0, 41] (ap2_ot.py), hinge-only rows [seed, 0, 43] (bd4_train.py), memory drop [seed, 0, 11] (pp_train.py). The learning-rate
cosine is written as each original writes it (pp_train: pi * (step / steps); the others: pi * step / steps), which can differ in the last bit.

  train  --tag T [flags]                       checkpoint $DATA_DIR/runs/op_parity/runs/<T>/ckpt-final.pt (+ adapter.pt), run dir runs/<exp>/joint-train-<T>/
  ident  --a TAG --b TAG [--out F.json]        trained net and adapter weights equal bit for bit (exit 0) or not (exit 2), max |difference| per group

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/joint_train.py train --tag JAPS-s0 --std alpasim --yr-mass 0.1 \
      --ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs --agent-lam 10 --shape --label-bank
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts"), str(_R / "experiments/body1/lib")]
import argparse, json, time  # noqa: E401,E402
from dataclasses import asdict, replace  # noqa: E402

import numpy as np  # noqa: E402

FULL = [f"navtrain_full.s{i}of12" for i in range(12)]
ROAD = "runs/body1/labels_road/navtrain_s01234567891011.npz"        # bd4_train.ROAD: the scorer-layer road-and-lane raster


def layout(datas, yr, fams, nd):
    """Store row bookkeeping for data dirs = base + yr1_* (yaw-rate imitation rows) + <fam>_* per hinge-only family, in that order.
    -> is_yr (n,), fam (n,) family index or -1, src (n,) the base row each row was built from (itself for base rows)."""
    from jevdrive.common import data_dir
    cr = data_dir() / "runs" / "op_parity" / "cache"
    nb = len(datas)
    n = sum(nd)
    off = np.cumsum([0] + list(nd))
    base_off = {d: off[i] for i, d in enumerate(datas)}
    is_yr, fam, src = np.zeros(n, bool), np.full(n, -1), np.arange(n)
    groups = [("yr", d) for d in (datas if yr else [])] + [(j, d) for j in range(len(fams)) for d in datas]
    for i, (g, d) in enumerate(groups):
        lo, hi = off[nb + i], off[nb + i + 1]
        if g == "yr":
            is_yr[lo:hi] = True
        else:
            fam[lo:hi] = g
        src[lo:hi] = np.load(cr / f"{'yr1' if g == 'yr' else fams[g]}_{d}" / "tab.npz")["src_row"] + base_off[d]
    return is_yr, fam, src


@__import__("torch").no_grad()
def dev_mem(model, S, rows, W, bs=128) -> dict:
    """Memory arms: dev ADE to the log with the bank on / masked / taken from a row of another log (vis_train vt_r.py check, same rule)."""
    import torch
    import pp_train as T
    pi = torch.as_tensor(S.pi, device=S.ego.device)
    lg, rng = S.tab["log"][rows], np.random.default_rng(0)
    other = np.array([rng.choice(rows[lg != x]) for x in lg])
    acc = {"on": [], "masked": [], "shuf": []}
    for i in range(0, len(rows), bs):
        r = torch.as_tensor(rows[i:i + bs], device=S.ego.device)
        ok = S.has_fut[r]
        for k in acc:
            mem = S.mem[other[i:i + bs] if k == "shuf" else rows[i:i + bs]]
            mask = torch.zeros(len(r), 1, dtype=torch.bool, device=r.device) if k == "masked" else None
            p = model(S.front[r], S.ego[r], S.tc[r], mem, mask).float()[:, pi].view(-1, 33, 15)
            x, y, _ = T.rear(p, S.cam_x[r], W)
            acc[k].append(torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1)[ok])
    return {f"mem_ade_{k}": float(torch.cat(v).mean()) for k, v in acc.items()}


def cmd_train(a):
    import torch
    import ap2_core as AC
    import ap2_ot as AO
    import ap2_train as AT
    import b1 as B1
    import ot_rows as OR
    import pp_train as T
    from drivable_hinge import Hinge
    from loss43 import Losses43, OffAgentHinge
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    ap = a.std == "alpasim"
    ho = [(f, int(k)) for f, k in (x.split(":") for x in a.ho.split(","))] if a.ho else []
    fams, kk = [f for f, _ in ho], [k for _, k in ho]
    nB, k_ho = a.batch, sum(kk)
    k_yr = int(round(nB * a.yr_mass))
    nN = nB - k_yr - k_ho                                            # normal rows per batch
    assert nN > 0 and not (a.mem and (ap or k_yr)), "--mem: NAVSIM input standard, no yaw-rate rows"
    yr = tuple(f"yr1_{d}" for d in a.data) if k_yr else ()
    hod = tuple(f"{f}_{d}" for f in fams for d in a.data)
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=nB, data=tuple(a.data) + yr + hod, split=a.split, frames="warp", host=True, warmup=a.warmup,
                eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin, agent_lam=a.agent_lam, agent_margin=a.agent_margin,
                agent_side_margin=a.agent_side_margin, agent_labels=a.agent_labels)
    if a.mem:
        cfg = replace(cfg, mem=a.mem, arm=f"P2+{a.mem}")             # pp_train --mem
    pp_lr = not (ap or k_yr or k_ho or a.agent_lam > 0)             # pp_train's own recipe: its cosine form
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train / bd4_train / ap2_train / ot_rows' row stream
    mrng = np.random.default_rng([cfg.seed, 0, 23])                  # ap2_train's decision types
    orng = np.random.default_rng([cfg.seed, 0, 41])                  # ap2_ot's yaw-rate rows (AlpaSim standard)
    hrng = np.random.default_rng([cfg.seed, 0, 43])                  # bd4_train's hinge-only rows
    memrng = np.random.default_rng([cfg.seed, 0, 11])                # pp_train's memory drop
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    names, logs = S.tab["names"], S.tab["log"]
    nd = [len(np.load(OR.CR / d / "tab.npz")["names"]) for d in cfg.data]
    n_base = sum(nd[:len(a.data)])
    is_yr, fam, src = layout(list(a.data), bool(k_yr), fams, nd)
    is_ho = fam >= 0
    tr_rows, dv_rows, sp = T.split_rows(dict(names=np.where(is_ho, "", names), log=logs, is_b2d=S.is_b2d, is_wod=S.is_wod), cfg.split)
    assert not is_ho[tr_rows].any() and not is_ho[dv_rows].any()
    off = OR.offsets(cfg.data)
    ok, EGO, cold = np.ones((S.n, 4), bool), None, None
    mix = np.asarray(a.mix, float) / np.sum(a.mix)
    if ap:
        wp, ok = AT.routes(a.data, names[:n_base])
        if S.n > n_base:                                             # perturbed rows: their token's m = 4 route seen from the perturbed pose (ap2_ot.py)
            p = np.arange(n_base, S.n)
            w4 = AO.wp_to_frame(wp[src[p], 3], off[p, 0].astype(np.float64), off[p, 1].astype(np.float64))
            wp, ok = np.concatenate([wp, np.repeat(w4[:, None], 4, 1)]), np.concatenate([ok, ok[src[p]]])
        EGO = torch.from_numpy(AC.ego_table(S.tab, wp, False)).to(dev)
        cold = T.Tokens([AT.AROOT / "cache" / d / "bw.npy" for d in a.data], dev, host=True)
        tr_rows, dv_rows = tr_rows[ok[tr_rows, 3]], dv_rows[ok[dv_rows, 3]]
        S.ego = EGO[:, 3]                                            # T.dev_eval reads S.ego: the m = 4 inputs
    tr_o, tr_t, dv_o, dv_t = tr_rows[~is_yr[tr_rows]], tr_rows[is_yr[tr_rows]], dv_rows[~is_yr[dv_rows]], dv_rows[is_yr[dv_rows]]
    assert k_yr == 0 or len(tr_t) >= k_yr
    trl = splits.load(B1.TRAIN)
    in_tr = trl.mask(logs)
    excl = splits.load(a.ho_excl) if a.ho_excl else None
    in_ho = (in_tr & ~excl.mask(logs) if excl is not None else in_tr) & ok[:, 3]
    pools = [np.flatnonzero((fam == j) & in_ho) for j in range(len(fams))]
    memrow = np.where(np.arange(S.n) < n_base, np.arange(S.n), src) if a.mem else None
    if a.mem:                                                        # the bank of the base dirs only; hinge-only rows read their source row, masked
        S.mem = T.Tokens([T.MEM_ROOT / a.mem / f"{d}.npy" for d in a.data], dev, True)
    model = T.PModel(cfg.arm).to(dev)
    model.set_tr1(dict(noax=False, hist_cv=False))
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_o, device=dev)].float().std(0).clamp_min(1e-3)
    bk = {"bank": True} if a.label_bank else {}
    hinge = None
    if cfg.hinge_lam > 0:
        hinge = (OR.off_hinge(cfg, names, off, dev, **bk) if k_yr else
                 Hinge([data_dir() / f for f in cfg.hinge_labels], names, dev, cfg.hinge_margin, list(cfg.hinge_footprint), **bk))
    agent2 = road = None
    if cfg.agent_lam > 0:                                            # A: imitation rows of the train logs + every hinge-only row, each in its own frame
        agent2 = OffAgentHinge(data_dir() / cfg.agent_labels, np.where(in_tr, names, ""), off, dev, cfg.agent_margin,
                               None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
    if k_ho and a.road_lam > 0:                                      # C: the road-and-lane raster on the hinge-only rows
        road = OR.off_hinge(replace(cfg, hinge_labels=(a.road_labels,), hinge_margin=a.road_margin), np.where(is_ho, names, ""), off, dev, **bk)
    n_fb = 0
    if road is not None and hinge is not None:                       # bd4_train note (x): a row not on the scorer-layer road at t0 keeps the NAVSIM raster
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
    LS = Losses43(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None, ho=torch.as_tensor(is_ho, device=dev) if k_ho else None,
                  fam=torch.as_tensor(fam, device=dev), fams=fams, agent2=agent2, road=road, road_lam=a.road_lam, ho_w=a.ho_w, shape=a.shape)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = T.proot("runs", a.tag)

    def draw():
        if ap:                                                       # ap2_train / ap2_ot
            r = rng.choice(tr_o, nN, replace=len(tr_o) < nN)
            an = rng.random(nB - k_ho) < cfg.d_frac
            m = mrng.choice(4, nB - k_ho, p=mix) + 1
            if k_yr:
                r = np.concatenate([r, orng.choice(tr_t, k_yr, replace=False)])
                an[nN:] = False                                      # yaw-rate rows: imitation rows only, full-history decisions
                m[nN:] = 4
            m[an | ~ok[r, m - 1]] = 4
        else:                                                        # pp_train / bd4_train / ot_rows
            if k_yr:
                r = np.concatenate([rng.choice(tr_o, nN, replace=len(tr_o) < nN), rng.choice(tr_t, k_yr, replace=False)])
            else:
                r = rng.choice(tr_o, nN, replace=len(tr_o) < nN)
            an = rng.random(nB - k_ho) < cfg.d_frac
            an[nN:] = False
            m = None
        if k_ho:                                                     # bd4_train: the hinge-only rows last
            r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
            an = np.concatenate([an, np.zeros(k_ho, bool)])
            m = None if m is None else np.concatenate([m, np.full(k_ho, 4)])
        sm = None
        if a.mem:
            sm = memrng.random((nB, 1)) >= T.MEM_DROP                # True = memory present
            sm[nN + k_yr:] = False                                   # hinge-only rows: masked
        return r, an, m, sm

    def fetch(dr):
        r, _, m, _ = dr
        if ap:
            return dr, AT.slots(S, cold, r, m, "backwarp")
        return dr, (S.front[r], S.mem[memrow[r]] if a.mem else None)
    for tk in (S.front, cold, S.mem if a.mem else None):
        if tk is not None:
            tk.pin = a.prefetch > 1
    if ap:
        import copy
        Sd = copy.copy(S)
        Sd.front = T.card_rows([S.front], dv_rows, a.dev_card_gb)[0]
        cold_d = T.card_rows([cold], dv_o, a.dev_card_gb)[0]           # the cold-start store holds the base rows only
    else:
        Sd, cold_d = T.dev_store(S, dv_rows, a.dev_card_gb), None
    fwd = torch.compile(model, dynamic=False) if a.compile else model
    with Run(a.exp, f"joint-train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"joint": vars(a)}) as run:
        for x in sp + ((trl,) if k_ho or agent2 is not None else ()) + ((excl,) if excl is not None and k_ho else ()):
            run.use_split(x)
        run.info(f"{a.tag}: {a.std} input standard; train {len(tr_o)} normal + {len(tr_t)} yaw-rate rows, dev {len(dv_o)} + {len(dv_t)}; per batch of {nB}: "
                 f"{nN} normal, {k_yr} yaw-rate, {k_ho} hinge-only (" + (", ".join(f"{f} {q} of {len(p)}" for (f, q), p in zip(ho, pools)) or "none") +
                 f"; weight {a.ho_w}, shape-only {a.shape}, excluded logs {a.ho_excl or 'none'}); hinge {cfg.hinge_lam} / {cfg.hinge_margin} m coverage "
                 f"{hinge.coverage if hinge else 0:.4f} ({'own frame' if k_yr else 'logged frame'}); agent {cfg.agent_lam} / {cfg.agent_margin} m rows "
                 f"{int(agent2.ok.sum()) if agent2 is not None else 0}; road {a.road_lam if road is not None else 0} rows {int(road.ok.sum()) if road is not None else 0} "
                 f"(NAVSIM raster kept on {n_fb}); memory {a.mem or 'none'}; routes ok per m {ok[:n_base].mean(0).round(4).tolist()}; label bank {a.label_bank}, "
                 f"compiled step {a.compile}, cosine form {'pp_train' if pp_lr else 'pi * step / steps'}")
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            (r, an, m, sm), fb_ = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                cosf = 0.5 * (1 + np.cos(np.pi * (step / cfg.steps))) if pp_lr else 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * cosf
            if ap:
                front, nv = fb_
                ego = EGO[rows, torch.as_tensor(m - 1, device=dev)] * (~anchor)[:, None].float()
                out = fwd(front, ego, S.tc[rows], nv=nv)
            else:
                front, side = fb_
                ego = S.ego[rows] * (~anchor)[:, None].float()
                out = fwd(front, ego, S.tc[rows], side, torch.as_tensor(sm, device=dev) if a.mem else None, nv=None)
            total, Ls = LS(out, S, rows, anchor)
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
                mm = {q: float(np.mean([h[q] for h in hist if q in h])) for q in keys}
                hist = []
                el = time.time() - t0
                run.scalars({f"loss/{q}": v for q, v in mm.items()} | {"throughput/steps_per_s": (step + 1) / el, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in mm.items() if "/" not in q) + f"; {(step + 1) / el:.2f} it/s, "
                             f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = T.dev_eval(model.eval(), Sd, dv_o, LS.W)
                if ap:
                    ev |= AT.dev_by_m(model, Sd, cold_d, EGO, ok, dv_o, LS.W, "backwarp")
                if len(dv_t):
                    ev |= {"ot_" + q: v for q, v in T.dev_eval(model, S, dv_t, LS.W).items()}
                if a.mem and step + 1 == cfg.steps:
                    ev |= dev_mem(model, Sd, dv_o, LS.W)
                model.train()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        rec = {"joint": vars(a)} | ({"ap2": dict(std="alpasim", cold="backwarp", route=False, mix=list(a.mix))} if ap else {})
        torch.save({"model": model.state(), "cfg": asdict(cfg)} | rec, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_o), n_train_yr=len(tr_t),
                           yr_per_batch=k_yr, ho_per_batch=k_ho, gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


def cmd_ident(a):
    """Two checkpoints (run tags under runs/op_parity/runs/) equal bit for bit in every trained net and adapter weight?"""
    import torch
    import pp_train as T
    ca, cb = (torch.load(T.proot("runs", t) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"] for t in (a.a, a.b))
    res = dict(a=a.a, b=a.b, arm_a=ca["arm"], arm_b=cb["arm"])
    for grp in ("net", "parity"):
        ka, kb = set(ca[grp] or {}), set(cb[grp] or {})
        common = sorted(ka & kb)
        res[f"{grp}_tensors"] = len(common)
        res[f"{grp}_only_a"], res[f"{grp}_only_b"] = sorted(ka - kb), sorted(kb - ka)
        res[f"{grp}_max_abs_diff"] = max((float((ca[grp][k].float() - cb[grp][k].float()).abs().max()) for k in common), default=float("nan"))
        res[f"{grp}_tensors_differ"] = sum(not torch.equal(ca[grp][k], cb[grp][k]) for k in common)
    res["identical"] = bool(ca["arm"] == cb["arm"] and all(res[f"{g}_tensors"] > 0 and not res[f"{g}_only_a"] and not res[f"{g}_only_b"]
                                                           and res[f"{g}_tensors_differ"] == 0 for g in ("net", "parity")))
    print(json.dumps(res, indent=1))
    if a.out:
        _pl.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    raise SystemExit(0 if res["identical"] else 2)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--exp", default="alpasim", help="experiment name of the run dir (runs/<exp>/joint-train-<tag>/)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", nargs="+", default=FULL)
    p.add_argument("--split", default="navsim/op-parity-full")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--hinge-lam", type=float, default=10.0, help="P2H10's drivable hinge on the imitation rows (NAVSIM raster)")
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p.add_argument("--std", default="navsim", choices=["navsim", "alpasim"], help="input standard of the rows (alpasim: ap2_train.py's, backwarp cold start)")
    p.add_argument("--mix", type=float, nargs=4, default=[0.1, 0.1, 0.1, 0.7], help="--std alpasim: share of decisions with m = 1..4 keyframes")
    p.add_argument("--yr-mass", type=float, default=0.0, help="share of every batch drawn from the yaw-rate rows (yr1_*) as imitation rows; 0 = off")
    p.add_argument("--ho", default="", help='hinge-only rows per batch by family, e.g. "ot1:4,yr1:4,bd4:5"; "" = none')
    p.add_argument("--ho-w", type=float, default=1.0)
    p.add_argument("--ho-excl", default="", help="split whose logs give no hinge-only row (navsim/body1-val-logs)")
    p.add_argument("--shape", action="store_true", help="the new hinges get no along-heading pose gradient (body1 Amendment 6)")
    p.add_argument("--agent-lam", type=float, default=0.0)
    p.add_argument("--agent-margin", type=float, default=0.3)
    p.add_argument("--agent-side-margin", type=float, default=0.0)
    p.add_argument("--agent-labels", default="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    p.add_argument("--road-lam", type=float, default=10.0, help="drivable hinge of the hinge-only rows (only with --ho)")
    p.add_argument("--road-margin", type=float, default=0.3)
    p.add_argument("--road-labels", default=ROAD)
    p.add_argument("--mem", default="", help="vtr_<name>: a frozen token bank through the memory channel (pp_train --mem)")
    p.add_argument("--label-bank", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile of the training step: faster, not bit-identical to eager")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--fetch-workers", type=int, default=2)
    p.add_argument("--dev-card-gb", type=float, default=4.0)
    p = sub.add_parser("ident")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default="")
    a = ap_.parse_args()
    {"train": cmd_train, "ident": cmd_ident}[a.cmd](a)
