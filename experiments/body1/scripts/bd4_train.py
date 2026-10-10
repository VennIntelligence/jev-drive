"""BODY1 arm 4.3 trainer (prereg Amendment 4): the P2H10 recipe plus the agent hinge (A), hinge-only off-track rows (B) and the scorer-layer
drivable hinge on them (C). pp_train.py is not edited: its Store / PModel / Prefetch / dev_eval are used as they are and its Losses through
lib/loss43.Losses43. With --ho "" and --agent-lam 0 the row stream, the losses and the checkpoint are pp_train's (gate: `ident`).

Batch = (batch - k) rows drawn as pp_train draws them (imitation / anchor) + k hinge-only rows, --ho "ot1:4,yr1:4,bd4:5" per batch, drawn
uniformly from each family's rows on navsim/body1-train-logs (own rng stream: the normal row stream does not depend on the families).
Prereg Amendment 5: --ho-w multiplies both hinge terms of the hinge-only rows; --ho-excl navsim/body1-val-logs keeps the validation part of
the train logs out of the hinge-only rows (the weight is selected there).
Prereg Amendment 6: --shape gives the new hinges the plan's shape only (lib/loss43.shape_only; values unchanged, tags P2H10S-*).
Prereg Amendment 7: --route-band B adds the route hinge (term R: relu(distance of the plan poses to the row's logged path - B), on the
imitation rows of the train logs and on the hinge-only rows; tags P2H10R-*); 0 = off, the code path of P2H10S.
HEAD1b step C (experiments/corridor, amendment of plans/2026-10-10-head1-prereg.md; EXPLORATORY): --mem-e2e qp adds pp_train's memory channel
(arm P2+ge_<tag>, tokenizer of experiments/op_parity/scripts/path_req.py trained jointly, memory masked on pp_train.MEM_DROP of the rows from its
own rng stream) fed with the HEAD1 head's predicted heading profile of every row, hinge-only rows included (the head read on the row's own
tokens and ego). "" = off: the model, the row stream and the optimizer are the ones above.

  train  --tag P2H10B-P-s0 --data navtrain_full.s2of12 navtrain_full.s3of12 --steps 3000 --ho ot1:4,yr1:4,bd4:5 --agent-lam 10
  train  --tag P2H10S-CHECK --shape-check P2H10B-Pw3-s0 ... (the pilot's flags)   Amendment 6 note (a): no training; on --check-batches batches
         of this trainer at that checkpoint's weights: (i) total and every logged scalar with --shape on against off, bit for bit;
         (ii) pose gradient of the new hinges: along-heading component with the switch on, cross-heading component on against off -> json
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
    if a.mem_e2e:                                                    # HEAD1b step C: the names pp_train gives a --mem-e2e run (bench reads the bank mem/ge_<tag>)
        assert a.mem_e2e.startswith("q") and not a.shape_check
        cfg = replace(cfg, mem_e2e=a.mem_e2e, mem_init=a.mem_init, mem=f"ge_{a.tag}", arm=f"P2+ge_{a.tag}")
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
    excl = splits.load(a.ho_excl) if a.ho_excl else None            # Amendment 5: logs kept out of the hinge-only rows
    in_ho = in_tr & ~excl.mask(logs) if excl is not None else in_tr
    pools = [np.flatnonzero((fam == j) & in_ho) for j in range(len(fams))]
    kk = [k for _, k in ho]
    k = sum(kk)
    nB = cfg.batch
    model = T.PModel(cfg.arm).to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    bk = {"bank": True} if a.label_bank else {}                      # speed knob: label rasters once per distinct label (lib/drivable_hinge.RowBank)
    hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], names, dev, cfg.hinge_margin, list(cfg.hinge_footprint), **bk) if cfg.hinge_lam > 0 else None
    off = OR.offsets(cfg.data)
    agent2 = road = None
    if cfg.agent_lam > 0:                                            # A: imitation rows of the train logs + every hinge-only row
        agent2 = OffAgentHinge(data_dir() / cfg.agent_labels, np.where(in_tr, names, ""), off, dev, cfg.agent_margin,
                               None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
    if k and a.road_lam > 0:                                         # C: the scorer-layer raster (or, --road-labels, another one) on the hinge-only rows
        road = OR.off_hinge(replace(cfg, hinge_labels=(a.road_labels,), hinge_margin=a.road_margin), np.where(is_ho, names, ""), off, dev, **bk)
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
                  ho=torch.as_tensor(is_ho, device=dev) if k else None, fam=torch.as_tensor(fam, device=dev), fams=fams, agent2=agent2, road=road, road_lam=a.road_lam, ho_w=a.ho_w, shape=a.shape,
                  **(dict(route_band=a.route_band, route_lam=a.route_lam, route_off=torch.as_tensor(off, dtype=torch.float32, device=dev),
                          route_ok=torch.as_tensor(in_tr, device=dev)) if a.route_band > 0 else {}))
    om = None
    if cfg.mem_e2e:                                                  # built after the model and the losses, as pp_train does
        import path_req as GE
        om = S.mem = GE.attach(cfg, S, hinge, dev)
        mrng = np.random.default_rng([cfg.seed, 0, 11])              # pp_train's memory-drop stream
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}] +
                            ([{"params": om.params, "lr": cfg.lr_new, "base": cfg.lr_new}] if om is not None else []), weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = T.proot("runs", a.tag)
    with Run("op_parity", f"train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"body1": vars(a)}) as run:
        for x in sp + ((trl,) if k or agent2 is not None else ()) + ((excl,) if excl is not None else ()):
            run.use_split(x)
        run.info(f"{a.tag}: train {len(tr_rows)} normal rows, dev {len(dv_rows)}; hinge-only per batch of {nB}: "
                 + (", ".join(f"{f} {q} of {len(p)}" for (f, q), p in zip(ho, pools)) or "none")
                 + f" (weight {a.ho_w}, shape-only hinge gradient {a.shape}, excluded logs {a.ho_excl or 'none'}: {int((is_ho & in_tr & ~in_ho).sum())} rows)"
                 + f"; hinge {cfg.hinge_lam} / {cfg.hinge_margin} m coverage {hinge.coverage if hinge else 0:.4f}; agent {cfg.agent_lam} / {cfg.agent_margin} m / side "
                 f"{cfg.agent_side_margin} rows {int(agent2.ok.sum()) if agent2 is not None else 0} (moved {agent2.moved if agent2 is not None else 0}); "
                 f"road {a.road_lam if road is not None else 0} / {a.road_margin} m rows {int(road.ok.sum()) if road is not None else 0} ({a.road_labels}); "
                 f"hinge-only rows not on the scorer-layer road at t0 (NAVSIM raster kept): {n_fb} {dict(zip(fams, fb_fam)) if n_fb else ''}"
                 + (f"; route hinge band {a.route_band} m, lambda {a.route_lam}" if a.route_band > 0 else ""))

        def draw():                                                   # pp_train's draw on the normal rows, then the hinge-only rows
            r = rng.choice(tr_rows, nB - k, replace=len(tr_rows) < nB - k)
            an = rng.random(nB - k) < cfg.d_frac
            if k:
                r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
                an = np.concatenate([an, np.zeros(k, bool)])
            return r, an, (mrng.random((nB, 1)) >= T.MEM_DROP if om is not None else None)   # True = memory present

        def fetch(dr):
            return dr, S.front[dr[0]]
        if a.shape_check:
            return shape_check(run, a, S, LS, draw, dev, T)
        S.front.pin = a.prefetch > 1
        Sd = T.dev_store(S, dv_rows, a.dev_card_gb)
        fwd = torch.compile(model, dynamic=False) if a.compile else model   # the step only (pp_train's --compile); dev eval stays eager
        run.info(f"card before the first step: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB allocated; label bank {a.label_bank}, compiled step {a.compile}")
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            (r, an, sm), front = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            ego = S.ego[rows] * (~anchor)[:, None].float()
            side, smask = (om[rows], torch.as_tensor(sm, device=dev)) if om is not None else (None, None)
            total, Ls = LS(fwd(front, ego, S.tc[rows], side, smask, nv=None), S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            if om is not None:
                torch.nn.utils.clip_grad_norm_(om.params, 1.0)
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
        if om is not None:                                           # tokenizer weights, dev diagnostics (on / masked / mismatched), the navtest bank
            GE.finish(om, model, S, dv_rows, LS.W, T.rear, hinge, run, a.tag, d)


def shape_check(run, a, S, LS, draw, dev, T):
    """Amendment 6 note (a), items (i) and (ii): see the module docstring. Writes --check-out."""
    import struct
    import torch
    from loss43 import shape_only
    model = T.load_pmodel(a.shape_check, dev)
    bits = lambda v: struct.pack("<d", float(v))  # noqa: E731
    res = dict(weights=a.shape_check, batches=a.check_batches, scalars_compared=0, scalars_differ=[], rows=0, agent_pos_rows=0, road_pos_rows=0,
               max_abs_grad=0.0, max_abs_along_on=0.0, max_abs_along_off=0.0, max_abs_cross_on_minus_off=0.0, max_abs_cross_off=0.0,
               max_abs_dout_on_minus_off=0.0, max_abs_dout_off=0.0, max_abs_pose_value_diff=0.0)
    for b in range(a.check_batches):
        r, an, _ = draw()
        rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
        ego = S.ego[rows] * (~anchor)[:, None].float()
        with torch.no_grad():
            out = model(S.front[rows], ego, S.tc[rows], None, None, nv=None)
        g = {}
        for on in (False, True):                                      # (i) the same batch, switch off / on
            LS.shape = on
            o = out.clone().requires_grad_(True)
            total, Ls = LS(o, S, rows, anchor)
            g[on] = (float(total), {q: float(v) for q, v in Ls.items()}, torch.autograd.grad(total, o)[0])
        keys = sorted(set(g[False][1]) | set(g[True][1]))
        res["scalars_compared"] += len(keys) + 1
        res["scalars_differ"] += [f"batch {b}: {q}" for q in keys if q not in g[False][1] or q not in g[True][1] or bits(g[False][1][q]) != bits(g[True][1][q])]
        if bits(g[False][0]) != bits(g[True][0]):
            res["scalars_differ"].append(f"batch {b}: total")
        res["max_abs_dout_on_minus_off"] = max(res["max_abs_dout_on_minus_off"], float((g[True][2] - g[False][2]).abs().max()))
        res["max_abs_dout_off"] = max(res["max_abs_dout_off"], float(g[False][2].abs().max()))
        # (ii) the new hinges as functions of the poses (the same masks as Losses43)
        h = LS.ho[rows] if LS.ho is not None else torch.zeros_like(anchor)
        imit = ~h & ~anchor & S.has_fut[rows]
        plan = out.float()[:, LS.pi].view(-1, 33, 15)
        x, y, psi = (q.detach().requires_grad_(True) for q in T.rear(plan, S.cam_x[rows], LS.W))

        def H(xs, ys):
            v = xs.sum() * 0.0
            na = nr = 0
            if LS.agent2 is not None:
                m = (imit | h) & LS.agent2.ok[rows]
                if m.any():
                    av = LS.agent2.per_step(xs[m], ys[m], psi[m], rows[m]).mean(1)
                    v, na = v + av.sum(), int((av > 0).sum())
            if LS.road is not None:
                m = h & LS.road.ok[rows]
                if m.any():
                    rv = torch.relu(LS.road.margin - LS.road.margins(xs[m], ys[m], psi[m], rows[m])).mean(1)
                    v, nr = v + rv.sum(), int((rv > 0).sum())
            return v, na, nr
        v0, na, nr = H(x, y)
        g0 = torch.autograd.grad(v0, (x, y))
        xs, ys = shape_only(x, y, psi)
        res["max_abs_pose_value_diff"] = max(res["max_abs_pose_value_diff"], float(torch.maximum((xs - x).abs().max(), (ys - y).abs().max())))
        v1, *_ = H(xs, ys)
        assert bits(v0) == bits(v1)
        g1 = torch.autograd.grad(v1, (x, y))
        c, s = torch.cos(psi.detach()).double(), torch.sin(psi.detach()).double()
        al = lambda q: c * q[0].double() + s * q[1].double()  # noqa: E731
        cr = lambda q: -s * q[0].double() + c * q[1].double()  # noqa: E731
        for q, v in (("max_abs_grad", max(float(g0[0].abs().max()), float(g0[1].abs().max()))), ("max_abs_along_on", float(al(g1).abs().max())),
                     ("max_abs_along_off", float(al(g0).abs().max())), ("max_abs_cross_on_minus_off", float((cr(g1) - cr(g0)).abs().max())),
                     ("max_abs_cross_off", float(cr(g0).abs().max()))):
            res[q] = max(res[q], v)
        res["rows"] += len(rows)
        res["agent_pos_rows"] += na
        res["road_pos_rows"] += nr
    res["values_bit_identical"] = not res["scalars_differ"] and res["max_abs_pose_value_diff"] == 0.0
    res["along_zero"] = res["max_abs_along_on"] <= 1e-6 * res["max_abs_grad"] and res["max_abs_along_off"] > 1e-3 * res["max_abs_grad"]
    res["cross_kept"] = res["max_abs_cross_on_minus_off"] <= 1e-6 * res["max_abs_grad"]
    res["switch_changes_gradient"] = res["max_abs_dout_on_minus_off"] > 0
    res["pass"] = bool(res["values_bit_identical"] and res["along_zero"] and res["cross_kept"] and res["switch_changes_gradient"] and res["agent_pos_rows"] > 0 and res["road_pos_rows"] > 0)
    run.info(json.dumps(res, indent=1))
    run.summary.update({q: v for q, v in res.items() if q != "scalars_differ"})
    if a.check_out:
        _pl.Path(a.check_out).parent.mkdir(parents=True, exist_ok=True)
        _pl.Path(a.check_out).write_text(json.dumps(res, indent=1) + "\n")
    if not res["pass"]:
        raise SystemExit(2)


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
    import pp_train as T
    ck = lambda x: T.proot("runs", _pl.Path(x).parent.name.removeprefix("train-")) / "ckpt-final.pt"  # noqa: E731  run dir = .../train-<tag>/<stamp>
    ca, cb = (torch.load(ck(x), map_location="cpu", weights_only=False)["model"] for x in (a.a, a.b))
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
    p.add_argument("--ho-w", type=float, default=1.0, help="Amendment 5: multiplier on the agent and road hinge of the hinge-only rows")
    p.add_argument("--ho-excl", default="", help='Amendment 5: split whose logs give no hinge-only row, e.g. navsim/body1-val-logs; "" = none')
    p.add_argument("--shape", action="store_true", help="Amendment 6: the new hinges (A, and the road hinge of hinge-only rows) get no along-heading pose gradient")
    p.add_argument("--route-band", type=float, default=0.0, help="Amendment 7: dead band (m) of the route hinge on imitation rows (train logs) and hinge-only rows; 0 = off")
    p.add_argument("--route-lam", type=float, default=10.0)
    p.add_argument("--shape-check", default="", help="Amendment 6 note (a): checkpoint tag; check the switch on batches of this trainer and exit (no training)")
    p.add_argument("--check-batches", type=int, default=40)
    p.add_argument("--check-out", default="")
    p.add_argument("--agent-lam", type=float, default=0.0, help="A: agent hinge on imitation rows (train logs) and hinge-only rows; 0 = off")
    p.add_argument("--agent-margin", type=float, default=0.3)
    p.add_argument("--agent-side-margin", type=float, default=0.0)
    p.add_argument("--agent-labels", default="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    p.add_argument("--road-lam", type=float, default=10.0, help="C: drivable hinge on the hinge-only rows (0 = off)")
    p.add_argument("--road-margin", type=float, default=0.3)
    p.add_argument("--road-labels", default=ROAD, help="its raster (default: the scorer-layer raster; the NAVSIM raster for the ablation without C)")
    p.add_argument("--mem-e2e", default="", help="HEAD1b step C: path_req.py memory kind (qp) through pp_train's memory channel; \"\" = off")
    p.add_argument("--mem-init", default="", help="its tokenizer state dict (path_req.py tok)")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--fetch-workers", type=int, default=2)
    p.add_argument("--dev-card-gb", type=float, default=4.0)
    p.add_argument("--label-bank", action="store_true", help="hinge rasters once per distinct label on the card (same values; 13.8 -> 5.2 GB at full scale): fits a 24 GB card")
    p.add_argument("--compile", action="store_true", help="torch.compile (inductor) of the training step, as pp_train.py --compile: faster, not bit-identical to eager")
    p = sub.add_parser("ident")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default="")
    a = ap.parse_args()
    {"train": cmd_train, "ident": cmd_ident}[a.cmd](a)
