"""ONPOL-PAI trainer (plans/2026-10-11-onpol-pai-prereg.md): the S recipe (experiments/body1/scripts/bd4_train.py, P2H10S) fine-tuned from an
archived checkpoint with on-policy PAI rows in every batch. Nothing it imports is edited: pp_train's Store / PModel / Prefetch / dev_eval,
loss43's Losses43 / OffAgentHinge and ot_rows' off_hinge are used as they are; the recipe's set-up below is bd4_train.cmd_train's, statement
for statement (without its memory channel, log-consistent margins and route hinge), so with --on-rows 0 and no --init the row stream, the
losses and the checkpoint are bd4_train's (gate: bd4_train.py ident on a pair of runs).

Batch of B = (B - k_ho - k_on) rows drawn as the recipe draws its normal rows (imitation, or anchor with p = 0.25; rng [seed, 0])
           + k_ho hinge-only rows (--ho, rng [seed, 0, 43]; bd4_train's)
           + k_on on-policy rows (--on-rows; rng [seed, 0, 61]): --on-log of them from pass `log` when its cache exists, the rest from pass
             `pol`; train scenes only (--on-train), never a row at or after the rollout's first scorer event, never an anchor row.
One forward for the whole batch; the Store rows go through Losses43 unchanged, the on-policy rows through lib/onp_loss.OnLoss.
--on-arm   imit: the recipe's imitation term on the on-policy rows (target `imit` / `log_fut`);  self: the six terms on the own plan
           (agent, edge, corr shape-only; stop, turn, prog timing-only; lam 10 x ho_w);  both;  none: the rows are in the batch, only
           their non-plan heads are distilled (not an arm). FT-CTRL = --on-rows 0.
--init TAG every arm starts from that checkpoint's adapter and plan pathway; the optimizer, the warm-up and the cosine are the recipe's.

  train  --tag P2H10S-SELF-ON-P-s0 --init P2H10S-F-s0 --on-arm self --steps 3000 --ho ot1:4,yr1:4,bd4:5 --ho-w 3 --ho-excl navsim/body1-val-logs \
         --agent-lam 10 --shape --label-bank --compile
  held   --tags P2H10S-F-s0 P2H10S-SELF-ON-P-s0 ...   the 16 held scenes' pass-pol rows (not after an event): ADE / FDE against `imit`, rate and
         mean of every ON-SELF term, 4 s arc length against the served plan, and navtrain dev ADE / drift -> <out>/held.{md,json}
  check  [--device cpu]   no model: (1) log consistency: every term with the log's own future as the plan; (2) gradient directions: shape-only
         terms have no along-heading pose gradient, timing-only terms no cross-heading one; (3) box_steps against lib/agent_hinge.AgentHinge,
         rear_xy against op_interp.to_rear, the path projection against e0 -> <out>/check.{md,json}

  $DATA_DIR/envs/op-train/bin/python experiments/onpol_pai/scripts/onp_train.py train ...   (one card; tmux through scripts/onp_night.sh)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_E = _pl.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_E / "body1" / "lib"), str(_E / "onpol_pai" / "lib")]
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

ROAD = "runs/body1/labels_road/navtrain_s01234567891011.npz"
FULL = [f"navtrain_full.s{i}of12" for i in range(12)]


def oroot(a) -> _pl.Path:
    from jevdrive.common import data_dir
    return _pl.Path(a.on_root) if a.on_root else data_dir() / "runs" / "onpol_pai"


def scenes(a, which) -> list:
    return [s for f in which for s in (oroot(a) / "split" / f).read_text().split()]


def on_rows(a, dev, passes=("pol", "log")):
    import onp_loss as OL
    c = _pl.Path(a.on_cache) if a.on_cache else oroot(a) / "train" / "cache"
    ps = [p for p in passes if (c / p / "tab.npz").exists()]
    assert "pol" in ps, f"no pass-pol cache under {c} (scripts/onp_cache.py build --pass pol)"
    return OL.OnRows(c, ps, dev, a.ego_box)


def light_ls(R, dev):
    """What OnLoss reads from a pp_train.Losses, without a Store: the 0.5 s grid, the imitation scales, the plan / distillation columns."""
    import torch
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    z = np.load(R.root / R.passes[0] / "tab.npz")
    di, pi = z["di"], z["pi"]
    return SimpleNamespace(W=torch.as_tensor(R2.t_weights(T.T8), device=dev), s=[torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (T.SIG_X, T.SIG_Y, T.SIG_PSI)],
                           pi=torch.as_tensor(pi, device=dev), di=torch.as_tensor(di, device=dev), plan_cols=torch.as_tensor(np.isin(di, pi), device=dev),
                           tstd=torch.ones(len(di), device=dev), dmask=torch.ones(len(di), device=dev), cfg=T.Cfg(arm="P2"))


def cmd_train(a):
    import torch
    import onp_loss as OL
    import ot_rows as OR
    import pp_train as T
    from drivable_hinge import Hinge
    from loss43 import Losses43, OffAgentHinge
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    d = data_dir() / "runs" / "op_parity" / "runs" / a.tag
    assert a.tag != a.init and (a.force or not (d / "ckpt-final.pt").exists()), f"{d}/ckpt-final.pt exists: a checkpoint is never overwritten (--force)"
    assert a.on_rows == 0 or "-ON-" in a.tag, "a checkpoint that sees PAI rollouts carries the family suffix -ON"
    ho = [(f, int(k)) for f, k in (x.split(":") for x in a.ho.split(","))] if a.ho else []
    fams = [f for f, _ in ho]
    hod = tuple(f"{f}_{d_}" for f in fams for d_ in a.data)
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data) + hod, split=a.split, frames="warp", host=True, warmup=a.warmup,
                eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin, agent_lam=a.agent_lam, agent_margin=a.agent_margin,
                agent_side_margin=a.agent_side_margin, agent_labels=a.agent_labels)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train's row stream
    hrng = np.random.default_rng([cfg.seed, 0, 43])                  # hinge-only rows: bd4_train's stream
    orng = np.random.default_rng([cfg.seed, 0, 61])                  # on-policy rows: own stream
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    names, logs = S.tab["names"], S.tab["log"]
    nd = [len(np.load(B.cache_root() / x / "tab.npz")["names"]) for x in cfg.data]
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
    excl = splits.load(a.ho_excl) if a.ho_excl else None
    in_ho = in_tr & ~excl.mask(logs) if excl is not None else in_tr
    pools = [np.flatnonzero((fam == j) & in_ho) for j in range(len(fams))]
    kk = [k for _, k in ho]
    k = sum(kk)
    nB, k_on = cfg.batch, a.on_rows
    nS = nB - k_on                                                   # Store rows per batch (normal + hinge-only)
    model = T.PModel(cfg.arm).to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    bk = {"bank": True} if a.label_bank else {}
    hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], names, dev, cfg.hinge_margin, list(cfg.hinge_footprint), **bk) if cfg.hinge_lam > 0 else None
    off = OR.offsets(cfg.data)
    agent2 = road = None
    if cfg.agent_lam > 0:
        agent2 = OffAgentHinge(data_dir() / cfg.agent_labels, np.where(in_tr, names, ""), off, dev, cfg.agent_margin,
                               None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
    if k and a.road_lam > 0:
        road = OR.off_hinge(replace(cfg, hinge_labels=(a.road_labels,), hinge_margin=a.road_margin), np.where(is_ho, names, ""), off, dev, **bk)
    n_fb = 0
    if road is not None and hinge is not None:                       # bd4_train's note (x): rows not on the scorer-layer road keep the NAVSIM raster
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
    LS = Losses43(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None,
                  ho=torch.as_tensor(is_ho, device=dev) if k else None, fam=torch.as_tensor(fam, device=dev), fams=fams, agent2=agent2, road=road, road_lam=a.road_lam, ho_w=a.ho_w, shape=a.shape)
    if a.init:                                                       # the parent's adapter and plan pathway; no random draw
        ck = torch.load(data_dir() / "runs" / "op_parity" / "runs" / a.init / "ckpt-final.pt", map_location="cpu", weights_only=False)
        assert ck["model"]["arm"] == cfg.arm and set(ck["model"]["net"]) == {q for q, p in model.net.params.items() if p.requires_grad}
        model.load_state(ck["model"])
    ON = OLs = held = None
    kon = []
    if k_on or a.on_eval:
        ON = on_rows(a, dev)
        assert np.array_equal(np.load(ON.root / "pol" / "tab.npz")["di"], np.asarray(S.di)), "the cache's teacher columns are not the Store's"
        OLs = OL.OnLoss(ON, LS, a.on_arm if k_on else "none", a.on_lam, a.ho_w, a.shape)
        held = ON.pool(scenes(a, a.on_held), "pol")
        if k_on:
            pl = {p: ON.pool(scenes(a, a.on_train), p) for p in ON.passes}
            k_log = a.on_log if len(pl.get("log", ())) else 0
            kon = [(p, q) for p, q in (("pol", k_on - k_log), ("log", k_log)) if q]
            assert not set(ON.scene[np.concatenate(list(pl.values()))].tolist()) & set(ON.scene[held].tolist()) and all(len(pl[p]) >= q for p, q in kon)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    with Run("op_parity", f"train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"body1": vars(a)}) as run:
        for x in sp + ((trl,) if k or agent2 is not None else ()) + ((excl,) if excl is not None else ()):
            run.use_split(x)
        run.info(f"{a.tag}: train {len(tr_rows)} normal rows, dev {len(dv_rows)}; per batch of {nB}: {nS - k} normal, hinge-only "
                 + (", ".join(f"{f} {q} of {len(p)}" for (f, q), p in zip(ho, pools)) or "none")
                 + f" (weight {a.ho_w}, shape-only {a.shape}, excluded logs {a.ho_excl or 'none'}); hinge {cfg.hinge_lam} / {cfg.hinge_margin} m coverage "
                 f"{hinge.coverage if hinge else 0:.4f}; agent {cfg.agent_lam} / {cfg.agent_margin} m / side {cfg.agent_side_margin} rows "
                 f"{int(agent2.ok.sum()) if agent2 is not None else 0}; road {a.road_lam if road is not None else 0} / {a.road_margin} m rows "
                 f"{int(road.ok.sum()) if road is not None else 0}; NAVSIM raster kept on {n_fb} hinge-only rows; init {a.init or 'none (the recipe from scratch)'}")
        if ON is not None:
            run.info(f"on-policy rows: arm {a.on_arm if k_on else 'none (not in the batch)'}, per batch "
                     + (", ".join(f"{p} {q} of {len(pl[p])} rows / {len(set(ON.scene[pl[p]].tolist()))} scenes" for p, q in kon) or "0")
                     + f"; passes cached {ON.passes} ({ON.n} rows; after an event or after the scorer's end, masked in every arm: {int(ON.after.sum())}); held read on {len(held)} pass-pol rows of {len(set(ON.scene[held].tolist()))} scenes; ego box `{a.ego_box}` "
                     f"front {float(ON.front.mean()):.3f} rear {float(ON.rear.mean()):.3f} half width {float(ON.hw.mean()):.3f} m; lam {a.on_lam} x ho_w {a.ho_w}; "
                     f"rows with a stop line {float(ON.stop_ok.any(1).float().mean()):.4f}, a turn peak {float(ON.turn_ok.float().mean()):.4f}, open {float(ON.open.float().mean()):.4f}")
            if k_on and not bool(ON.stop_ok.any()):
                run.info("NO STOP LINE in the cached rows (has_stop false or no eligible line): term `stop` (L item B) is skipped, it contributes 0")
            if k_on and "log" not in dict(kon):
                run.info(f"pass `log` is absent or has no train-scene row: all {k_on} on-policy rows come from pass `pol`")

        def draw():                                                   # bd4_train's draw on the Store rows, then the on-policy rows
            r = rng.choice(tr_rows, nS - k, replace=len(tr_rows) < nS - k)
            an = rng.random(nS - k) < cfg.d_frac
            if k:
                r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
                an = np.concatenate([an, np.zeros(k, bool)])
            return r, an, (np.concatenate([orng.choice(pl[p], q, replace=False) for p, q in kon]) if k_on else None)

        def fetch(dr):
            f = S.front[dr[0]]
            return dr, (torch.cat([f, ON.tokens(dr[2]).to(dev, non_blocking=True)]) if k_on else f)
        S.front.pin = a.prefetch > 1
        Sd = T.dev_store(S, dv_rows, a.dev_card_gb)
        fwd = torch.compile(model, dynamic=False) if a.compile else model
        run.info(f"card before the first step: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB allocated; label bank {a.label_bank}, compiled step {a.compile}")
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            (r, an, r_on), front = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            ego, tc = S.ego[rows] * (~anchor)[:, None].float(), S.tc[rows]
            if k_on:
                ro = torch.as_tensor(r_on, device=dev)
                ego, tc = torch.cat([ego, ON.ego[ro]]), torch.cat([tc, ON.tc[ro]])
            out = fwd(front, ego, tc, None, None, nv=None)
            total, Ls = LS(out[:nS] if k_on else out, S, rows, anchor)
            if k_on:
                nrm = ~LS.ho[rows] if LS.ho is not None else torch.ones_like(anchor)
                t_on, L_on = OLs(out[nS:], ro, (nrm & ~anchor & S.has_fut[rows]).sum().clamp_min(1), nrm.sum().clamp_min(1))
                total, Ls = total + t_on, Ls | L_on
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
                m |= {q.replace("_posmean", "_posn"): float(sum(q in h for h in hist)) for q in keys if q.endswith("_posmean")}
                hist = []
                run.scalars({f"loss/{q}": v for q, v in m.items()}, step + 1)
                el = time.time() - t0
                run.scalars({"throughput/steps_per_s": (step + 1) / el, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in m.items() if "/" not in q) + f"; {(step + 1) / el:.2f} it/s, "
                             f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    if k_on:
                        run.info("  on-policy terms, share of rows > 0 (mean): " + "; ".join(
                            f"{p} " + " ".join(f"{q} {m.get(f'on/{p}/{q}_pos', 0):.3f} ({m.get(f'on/{p}/{q}_mean', 0):.4f})" for q in OL.TERMS + ("imit",)) for p, _ in kon))
                    run.status(f"step {step + 1}/{cfg.steps}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = T.dev_eval(model.eval(), Sd, dv_rows, LS.W)
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
                if ON is not None and len(held):
                    hv = OL.held_eval(model, ON, OLs, held)
                    run.scalars({f"onheld/{q}": v for q, v in hv.items()}, step + 1)
                    run.info(f"held on-policy rows @ {step + 1}: " + ", ".join(f"{q} {hv[q]:.4f}" for q in ("ade_1_4", "fde_4", "rate_sum", "active_rate", "arc_ratio_served")
                                                                                 + tuple(f"{t}_rate" for t in OL.TERMS)))
                    run.summary.update({f"onheld_{q}": v for q, v in hv.items()})
                model.train()
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state(), "cfg": asdict(cfg), "body1": vars(a),
                    "onp": dict(init=a.init, arm=a.on_arm if k_on else "", rows=dict(kon), passes=ON.passes if ON is not None else [])}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_rows), ho_per_batch=k, on_per_batch=k_on,
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30, it_s=cfg.steps / (time.time() - t0))


def cmd_held(a):
    import torch
    import onp_loss as OL
    import pp_train as T
    from jevdrive.run import Run
    dev = torch.device("cuda")
    out = _pl.Path(a.out) if a.out else oroot(a) / "train" / "held"
    out.mkdir(parents=True, exist_ok=True)
    with Run("onpol_pai", "held", config=vars(a)) as run:
        ON = on_rows(a, dev, ("pol",))
        ls = light_ls(ON, dev)
        OLs = OL.OnLoss(ON, ls, "none", shape=True)
        rows = ON.pool(scenes(a, a.on_held), "pol")
        S = T.Store(tuple(a.data), dev, need_side=False, frames="warp", host=True)
        _, dv_rows, sp = T.split_rows(dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d, is_wod=S.is_wod), a.split)
        run.use_split(sp[1])
        Sd = T.dev_store(S, dv_rows, 4.0)
        run.info(f"held read: {len(rows)} pass-pol rows of {len(set(ON.scene[rows].tolist()))} held scenes (not after an event); navtrain dev rows {len(dv_rows)}")
        res = {}
        for tag in a.tags:
            m = T.load_pmodel(tag, dev)
            res[tag] = OL.held_eval(m, ON, OLs, rows) | {f"dev_{q}": v for q, v in T.dev_eval(m, Sd, dv_rows, ls.W).items()}
            run.info(f"{tag}: " + ", ".join(f"{q} {v:.4f}" for q, v in res[tag].items()))
        T_ = OL.TERMS
        md = (f"# ONPOL-PAI held read\n\n{len(rows)} pass-1 rows of the {len(set(ON.scene[rows].tolist()))} held scenes ({', '.join(a.on_held)}), rows at or after the rollout's first "
              "scorer event excluded. ADE / FDE: the model's plan against the recovery target `imit` (m; rows whose 8 target poses exist: n target). "
              "Term columns: share of rows with a non-zero term / mean of the term over all rows. Rate sum = agent + edge + corr + stop + turn. "
              "Arc ratio = summed 4 s arc length of the plans / that of the plans served in the rollout (after serve_fix) and / that of the parent's logged model plan. "
              f"Dev = pp_train.dev_eval on the navtrain dev rows ({len(dv_rows)}).\n\n"
              "| Tag | n target | ADE 1-4 s | ADE 0.5-4 s | FDE 4 s | " + " | ".join(T_) + " | corr, plan past the path end | Rate sum | Any active | Arc / served | Arc / parent plan | Dev ADE | Dev drift on | Dev drift off |\n"
              "|---|--:|--:|--:|--:|" + "--:|" * len(T_) + "--:|--:|--:|--:|--:|--:|--:|--:|\n"
              + "".join(f"| `{t}` | {v['n_target']} | {v['ade_1_4']:.3f} | {v['ade_05_4']:.3f} | {v['fde_4']:.3f} | "
                        + " | ".join(f"{v[q + '_rate']:.4f} / {v[q + '_mean']:.4f}" for q in T_)
                        + f" | {v['corr_end_rate']:.4f} | {v['rate_sum']:.4f} | {v['active_rate']:.4f} | {v['arc_ratio_served']:.3f} | {v['arc_ratio_parent']:.3f} | {v['dev_ade']:.3f} | {v['dev_drift_on']:.3f} | {v['dev_drift_off']:.3f} |\n"
                        for t, v in res.items())
              + "\nShare of rows each term is defined on: " + ", ".join(f"{q} {next(iter(res.values()))[q + '_elig']:.3f}" for q in T_) + ".\n")
        (out / f"{a.name}.json").write_text(json.dumps(res, indent=1) + "\n")
        (out / f"{a.name}.md").write_text(md)
        run.info(md)
        run.summary.update(out=str(out / f"{a.name}.md"), tags=len(res))


def cmd_check(a):
    import torch
    import agent_hinge as AH
    import onp_loss as OL
    import pp_train as T
    from jevdrive import op_interp as I
    dev = torch.device(a.device)
    out = _pl.Path(a.out) if a.out else oroot(a) / "train" / "check"
    out.mkdir(parents=True, exist_ok=True)
    R = on_rows(a, dev)
    ls = light_ls(R, dev)
    L = OL.OnLoss(R, ls, "none")
    g = np.random.default_rng(0)
    res = dict(passes=R.passes, rows=R.n, ego_box=a.ego_box)
    ok = np.flatnonzero(R.usable & ~R.after)

    def run_terms(rows, plan_of, bs=256, **kw):
        acc = {}
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs], device=dev)
            with torch.no_grad():
                for q, (v, e) in L.terms(*plan_of(r), r, **kw).items():
                    acc.setdefault(q, []).append(v.cpu().numpy())
        return {q: np.concatenate(v) for q, v in acc.items()}
    # (1) log consistency: the log's own future as the plan
    lf = lambda r: (R.log_fut[r][..., 0], R.log_fut[r][..., 1], R.log_fut[r][..., 2])  # noqa: E731
    lok, e0, ds0 = R.log_ok.cpu().numpy(), R.e0.cpu().numpy(), R.ds0.cpu().numpy()
    pops = {}
    if "log" in R.passes:
        pops["pass log"] = ok[(R.pas[ok] == R.passes.index("log")) & lok[ok]]
    ip = R.passes.index("pol")
    pops["pass pol, |e0| < 0.3 m"] = ok[(R.pas[ok] == ip) & lok[ok] & (np.abs(e0[ok]) < 0.3)]
    pops["pass pol, |e0| < 0.3 m and |ds0| < 0.5 m"] = ok[(R.pas[ok] == ip) & lok[ok] & (np.abs(e0[ok]) < 0.3) & (np.abs(ds0[ok]) < 0.5)]
    res["log_consistency"] = {}
    for name, rows in pops.items():
        if not len(rows):
            continue
        v = run_terms(rows, lf)
        res["log_consistency"][name] = dict(n=int(len(rows)), terms={
            q: dict(share_zero=float((x <= a.eps).mean()), share_exact_zero=float((x == 0).mean()), max=float(x.max()),
                    worst=[dict(scene=str(R.scene[rows[j]]), k=int(R.k[rows[j]]), value=float(x[j])) for j in np.argsort(-x)[:5] if x[j] > a.eps]) for q, x in v.items()})
    # (2) gradient directions on perturbed plans of the parent (lateral offset, speed scale): which pose components each term can move
    rows = g.choice(ok, min(a.n_grad, len(ok)), replace=False)
    gr = {q: dict(n_pos=0, max_grad=0.0, max_along=0.0, max_cross=0.0, max_along_full=0.0) for q in OL.TERMS}
    vd = 0.0
    for i in range(0, len(rows), 128):
        rn = rows[i:i + 128]
        r = torch.as_tensor(rn, device=dev)
        P = R.poses_model[r].clone()
        P[..., 0] *= torch.as_tensor(g.uniform(0.6, 1.6, (len(rn), 1)), dtype=torch.float32, device=dev)
        P[..., 1] += torch.as_tensor(g.uniform(-4, 4, (len(rn), 1)), dtype=torch.float32, device=dev) * torch.linspace(0.2, 1, 8, device=dev)
        for sh in (True, False):
            x, y, psi = (P[..., j].clone().requires_grad_(True) for j in range(3))
            tv = L.terms(x, y, psi, r, shape=sh)
            if sh:
                ref = {q: v.detach() for q, (v, _) in tv.items()}
            else:
                vd = max(vd, max(float((tv[q][0].detach() - ref[q]).abs().max()) for q in OL.TERMS))
            for q, (v, _) in tv.items():
                gx, gy = torch.autograd.grad(v.sum(), (x, y), retain_graph=True, allow_unused=True)
                if gx is None:
                    continue
                c, s = torch.cos(psi.detach()).double(), torch.sin(psi.detach()).double()
                al, cr = (c * gx.double() + s * gy.double()).abs().max(), (-s * gx.double() + c * gy.double()).abs().max()
                if sh:
                    gr[q]["n_pos"] += int((v > 0).sum())
                    gr[q]["max_grad"] = max(gr[q]["max_grad"], float(torch.maximum(gx.abs().max(), gy.abs().max())))
                    gr[q]["max_along"], gr[q]["max_cross"] = max(gr[q]["max_along"], float(al)), max(gr[q]["max_cross"], float(cr))
                else:
                    gr[q]["max_along_full"] = max(gr[q]["max_along_full"], float(al))
    for q, v in gr.items():
        bad = v["max_along"] if q in OL.SHAPE_T else v["max_cross"]
        v["pass"] = bool(v["n_pos"] > 0 and bad <= 1e-6 * max(v["max_grad"], 1e-30))
    res["gradient"] = dict(rows=int(len(rows)), terms=gr, max_value_diff_shape_on_off=vd)
    # (3) the operators against their references
    rows = g.choice(ok, min(2000, len(ok)), replace=False)
    r = torch.as_tensor(rows, device=dev)
    H = AH.AgentHinge.__new__(AH.AgentHinge)                         # the reference class on the same boxes, with its own (Pacifica) ego box
    H.box, H.val, H.M, H.Mv, H.margin, H.side_margin = R.box, R.box_ok, L.M, L.Mv, OL.AG_M, OL.AG_SIDE
    f = AH.FRONT + OL.AG_M
    H.e_off, H.e_hl, H.e_hw, H.c_off = (f + AH.REAR) / 2, (f - AH.REAR) / 2, AH.HALF_W, (AH.FRONT + AH.REAR) / 2
    x, y, psi = (R.poses_model[r][..., j] for j in range(3))
    one = torch.ones(len(r), device=dev)
    bx = torch.einsum("kj,bjoc->bkoc", L.M, R.box[r])
    bv = torch.einsum("kj,bjo->bko", L.Mv, (~R.box_ok[r]).float()) < 0.5
    mine = OL.box_steps(OL.dense(L.M, x, y, psi), bx, bv, AH.FRONT * one, AH.REAR * one, AH.HALF_W * one, OL.AG_M, OL.AG_SIDE)
    theirs = H.per_step(x, y, psi, r)
    res["box_steps_vs_AgentHinge"] = dict(rows=int(len(r)), max_abs_diff=float((mine - theirs).abs().max()), rows_pos=int((theirs.amax(1) > 0).sum()))
    cam = R.cam[r[:64], :2]
    pl = torch.as_tensor(g.normal(0, 1, (len(cam), 33, 15)).cumsum(1) * 0.3, dtype=torch.float32, device=dev)
    pl[..., 11] *= 0.05
    mine = torch.stack(OL.rear_xy(pl, cam, ls.W), -1).cpu().numpy()
    theirs = np.stack([I.to_rear(p[:, 0:3], p[:, 11], I.T_IDXS, c, T.T8, "lever") for p, c in zip(pl.cpu().numpy(), cam.cpu().numpy().astype(np.float64))])
    res["rear_xy_vs_to_rear"] = dict(max_abs_diff=float(np.abs(mine - theirs).max()), cam_xy=cam[0].tolist())
    hp = R.has_path[r]
    s, l, dd = OL.project(R.path[r], R.path_ok[r], torch.zeros(len(r), 1, 2, device=dev))
    hp = hp & (R.rlen[r] > OL.S0 + 1)                                # the ego's own arc is inside the finite path, not at its end
    q = lambda v, f: float(v.quantile(f)) if len(v) else float("nan")  # noqa: E731
    ds, dl = (s[hp, 0] - OL.S0).abs(), (l[hp, 0] - R.e0[r][hp]).abs()
    res["projection"] = dict(rows=int(hp.sum()), abs_s_minus_20=dict(median=q(ds, 0.5), p99=q(ds, 0.99), max=q(ds, 1.0)),
                             abs_l_minus_e0=dict(median=q(dl, 0.5), p99=q(dl, 0.99), max=q(dl, 1.0)), median_abs_e0=q(R.e0[r][hp].abs(), 0.5))
    res["rates_parent_plan"] = {}                                    # activation of every term on the parent's own logged plan
    for p in R.passes:
        rows = ok[R.pas[ok] == R.passes.index(p)]
        v = run_terms(rows, lambda r: (R.poses_model[r][..., 0], R.poses_model[r][..., 1], R.poses_model[r][..., 2]))
        res["rates_parent_plan"][p] = dict(n=int(len(rows)), terms={q: dict(rate=float((x > 0).mean()), mean=float(x.mean()), posmean=float(x[x > 0].mean()) if (x > 0).any() else 0.0)
                                                                    for q, x in v.items()},
                                           elig=dict(stop=float(R.stop_ok[torch.as_tensor(rows, device=dev)].any(1).float().mean()),
                                                     turn=float(R.turn_ok[torch.as_tensor(rows, device=dev)].float().mean()),
                                                     open=float(R.open[torch.as_tensor(rows, device=dev)].float().mean())))
    lc_ok = all(t["share_zero"] >= 0.99 for name, p in res["log_consistency"].items() if "ds0" in name or name == "pass log" for t in p["terms"].values())
    res["pass"] = dict(log_consistency=bool(lc_ok), gradient=all(v["pass"] for v in gr.values()),
                       operators=bool(res["box_steps_vs_AgentHinge"]["max_abs_diff"] < 1e-4 and res["rear_xy_vs_to_rear"]["max_abs_diff"] < 1e-3))
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--on-root", default="", help="lane root (default $DATA_DIR/runs/onpol_pai): split/, train/")
        p.add_argument("--on-cache", default="", help="row cache root (default <on-root>/train/cache)")
        p.add_argument("--on-held", nargs="+", default=["stageA.txt", "stageB.txt"], help="scene lists under <on-root>/split that give no training row")
        p.add_argument("--ego-box", default="veh_sim", choices=["veh_sim", "veh"], help="ego box of the on-policy terms: the simulator's / scorer's, or the rollout spec's")
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--init", default="", help="checkpoint tag to start from (adapter + plan pathway)")
    p.add_argument("--on-arm", default="none", choices=["imit", "self", "both", "none"])
    p.add_argument("--on-rows", type=int, default=16, help="on-policy rows per batch (0 = FT-CTRL: the recipe continued)")
    p.add_argument("--on-log", type=int, default=4, help="of them from pass `log` when it is cached")
    p.add_argument("--on-train", nargs="+", default=["stageC.txt"])
    p.add_argument("--on-lam", type=float, default=10.0)
    p.add_argument("--on-eval", action="store_true", help="with --on-rows 0: still read the held on-policy rows at every dev eval")
    p.add_argument("--force", action="store_true", help="overwrite the tag's checkpoint")
    common(p)
    # ---- bd4_train.py train's flags, unchanged
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", nargs="+", default=FULL)
    p.add_argument("--split", default="navsim/op-parity-full")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--hinge-lam", type=float, default=10.0)
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p.add_argument("--ho", default="")
    p.add_argument("--ho-w", type=float, default=1.0)
    p.add_argument("--ho-excl", default="")
    p.add_argument("--shape", action="store_true")
    p.add_argument("--agent-lam", type=float, default=0.0)
    p.add_argument("--agent-margin", type=float, default=0.3)
    p.add_argument("--agent-side-margin", type=float, default=0.0)
    p.add_argument("--agent-labels", default="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    p.add_argument("--road-lam", type=float, default=10.0)
    p.add_argument("--road-margin", type=float, default=0.3)
    p.add_argument("--road-labels", default=ROAD)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--fetch-workers", type=int, default=2)
    p.add_argument("--dev-card-gb", type=float, default=4.0)
    p.add_argument("--label-bank", action="store_true")
    p.add_argument("--compile", action="store_true")
    p = sub.add_parser("held")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--name", default="held")
    p.add_argument("--out", default="")
    p.add_argument("--data", nargs="+", default=FULL)
    p.add_argument("--split", default="navsim/op-parity-full")
    common(p)
    p = sub.add_parser("check")
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default="check")
    p.add_argument("--out", default="")
    p.add_argument("--eps", type=float, default=1e-4, help="a term below this counts as 0 in the log-consistency read")
    p.add_argument("--n-grad", type=int, default=2048)
    common(p)
    a = ap.parse_args()
    {"train": cmd_train, "held": cmd_held, "check": cmd_check}[a.cmd](a)
