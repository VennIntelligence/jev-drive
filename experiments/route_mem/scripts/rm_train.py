#!/usr/bin/env python3
"""ROUTE-MEM trainer (plans/2026-10-12-route-mem-prereg.md): onpol_nuplan's on_train.py `train` (its setup() imported unedited, its training
loop repeated here) with the route-memory branch of lib/rmem.py on the adapter. Changes against on_train.py cmd_train, nothing else:

  branch    after setup(): the P2 adapter becomes rmem.RouteParityAdapter (every --init tensor kept; the branch's output layer starts at 0, so
            step 0 is the init checkpoint: logged dev ADE at step 0 must equal the init's), the fed row = ego (20) ++ route features (200) from
            $DATA_DIR/runs/route_mem/feat/<data dir>.npz (scripts/rm_build.py: nav / yr / pool; rows without a rebuilt route get an empty memory)
  lr        the branch is a third parameter group at --rm-lr (default 3e-4 = the adapter's original lr_new, NOT scaled by --lr-scale) with its
            own linear warm-up (--rm-warmup 100) under the same cosine; the net and the old adapter keep on_train's groups and schedule
  drop      the route features are zeroed on a share --rm-drop (0.25, pp_train MEM_DROP) of the non-anchor rows of every batch, own rng stream
            [seed, 0, 89] (every row stream of on_train unchanged): "no memory" stays in distribution, which the reliance read needs
  after     ckpt-final.pt (model.state(): the branch is in `parity`) + route_mem.json (sidecar: sh30_core serves the branch iff it exists);
            the reliance read on the dev rows (memory on / masked, per-row ADE; junction rows = the route enters a junction within 40 m,
            straight rows = no junction within 60 m and < 5 deg route turn over 40 m) -> rely.json in the run dir and runs/route_mem/rely/<tag>.json

  $DATA_DIR/envs/op-train/bin/python experiments/route_mem/scripts/rm_train.py train --tag RM-L18-s0 --init YS-L18-s1 --steps 2500 --warmup 1 \
      --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS --on-k 13 \
      --on-split alpasim_nuplan/onpol-train --on-mode self --seed 0
  ... rely --tag RM-L18-s0 [same recipe flags]     the reliance read alone, from a saved checkpoint
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/onpol_nuplan/scripts"), str(_R / "experiments/route_mem/lib")]
import argparse, hashlib, json, time  # noqa: E401,E402
from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402

import on_train as OT  # noqa: E402  (sets the repo paths)
import rmem as RM  # noqa: E402

FEAT = None
HEAD = dict(d=128, layers=2, heads=4)


def feats(cfg_data, S, dev):
    """Route features (S.n, DIM) float32 on dev, and per Store row: junction distance / route turn (dev rows of the NAVSIM dirs, else nan),
    the source dir index. Asserts every file's names equal the Store tab's names of its dir."""
    import torch
    from jevdrive.common import data_dir
    root = data_dir() / "runs/route_mem/feat"
    F, JD, TU, src, okc = [], [], [], [], {}
    o = 0
    for j, d in enumerate(cfg_data):
        z = np.load(root / f"{d}.npz")
        n = len(z["names"])
        assert np.array_equal(z["names"], S.tab["names"][o:o + n]), f"{d}: feature rows differ from the Store's"
        f = z["f"].astype(np.float32)
        f[~z["ok"]] = 0
        F.append(f)
        JD.append(z["jdist"] if "jdist" in z.files else np.full(n, np.nan, np.float32))
        TU.append(z["turn40"] if "turn40" in z.files else np.full(n, np.nan, np.float32))
        src.append(np.full(n, j))
        okc[d] = float(z["ok"].mean())
        o += n
    assert o == S.n, (o, S.n)
    return torch.as_tensor(np.concatenate(F), device=dev), np.concatenate(JD), np.concatenate(TU), np.concatenate(src), okc


def rows_ade(model, S, rows, W, mask, bs=256):
    """Per-row ADE of the 8 rear-axle poses to the log (rows with a future), memory on (mask False) or masked."""
    import torch
    import pp_train as T
    pi = torch.as_tensor(S.pi, device=S.ego.device)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs], device=S.ego.device)
            ego = S.ego[r].clone()
            if mask:
                ego[:, RM.EGO_DIM:] = 0
            p = model(S.front[r], ego, S.tc[r], None, None, nv=None if S.nv is None else S.nv[r]).float()[:, pi].view(-1, 33, 15)
            x, y, _ = T.rear(p, S.cam_x[r], W)
            out.append(torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1).cpu().numpy())
    return np.concatenate(out)


def rely(model, S, dv_o, W, JD, TU, logs):
    """The registered reliance read (prereg 'Gates'): delta = ADE(memory masked) - ADE(on) per dev row; junction rows vs straight rows,
    log-clustered bootstrap (B 10000, rng 0) of mean delta (junction) - mean delta (straight)."""
    import torch
    model.eval()
    has = S.has_fut[torch.as_tensor(dv_o, device=S.ego.device)].cpu().numpy().astype(bool)
    r = dv_o[has]
    on, off = rows_ade(model, S, r, W, False), rows_ade(model, S, r, W, True)
    d, jd, tu, lg = off - on, JD[r], TU[r], np.asarray(logs)[r]
    J = np.isfinite(jd) & (jd <= 40)
    St = ~np.isfinite(jd) & np.isfinite(tu) & (tu < 5)
    ul = np.unique(lg)
    li = np.searchsorted(ul, lg)
    rng = np.random.default_rng(0)
    bj, bs = np.zeros(10000), np.zeros(10000)
    sj, nj = np.bincount(li[J], d[J], len(ul)), np.bincount(li[J], minlength=len(ul))
    ss, ns = np.bincount(li[St], d[St], len(ul)), np.bincount(li[St], minlength=len(ul))
    for b in range(10000):
        c = np.bincount(rng.integers(0, len(ul), len(ul)), minlength=len(ul))
        bj[b], bs[b] = (c * sj).sum() / max((c * nj).sum(), 1), (c * ss).sum() / max((c * ns).sum(), 1)
    diff = bj - bs
    res = dict(rows=int(len(r)), logs=int(len(ul)), ade_on=float(on.mean()), ade_masked=float(off.mean()), junction_rows=int(J.sum()), straight_rows=int(St.sum()),
               d_junction=float(d[J].mean()), d_straight=float(d[St].mean()), d_all=float(d.mean()),
               diff=float(d[J].mean() - d[St].mean()), diff_lo=float(np.quantile(diff, 0.025)), diff_hi=float(np.quantile(diff, 0.975)),
               ade_on_junction=float(on[J].mean()), ade_on_straight=float(on[St].mean()))
    res["read"] = bool(res["d_junction"] >= 0.01 and res["diff_lo"] > 0)
    return res


def build(a, dev):
    """on_train.setup + the branch + the route features. -> (ns, route params, JD, TU, okc)."""
    import torch
    ns = OT.setup(a, dev)
    RM.attach(ns.model, **HEAD)
    if a.rm_from:                                                     # the reliance read of a saved checkpoint
        import pp_train as T
        ck = torch.load(T.proot("runs", a.rm_from) / "ckpt-final.pt", map_location="cpu", weights_only=False)
        ns.model.load_state(ck["model"])
    F, JD, TU, src, okc = feats(ns.cfg.data, ns.S, dev)
    ns.S.ego = torch.cat([ns.S.ego.float(), F], 1)
    rp = list(ns.model.adapter.route.parameters())
    ids = {id(p) for p in rp}
    ns.new = [p for p in ns.model.adapter.parameters() if id(p) not in ids]
    return ns, rp, JD, TU, okc


def cmd_train(a):
    import torch
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    d = T.proot("runs", a.tag)
    assert not (d / "ckpt-final.pt").exists(), f"{d / 'ckpt-final.pt'} exists: a tag is never overwritten"
    a.rm_from = ""
    ns, rp, JD, TU, okc = build(a, dev)
    cfg, S, model, base, new, pools, kk = ns.cfg, ns.S, ns.model, ns.base, ns.new, ns.pools, ns.kk
    nB, nN, k_ho, k_on, tr_o, dv_o = cfg.batch, ns.nN, ns.k_ho, ns.k_on, ns.tr_o, ns.dv_o
    k_yr, tr_t, dv_t = ns.k_yr, ns.tr_t, ns.dv_t
    LS = ns.losses(ns.mode)
    pp_lr = not (k_yr or k_ho or k_on or a.agent_lam > 0)
    rng = np.random.default_rng([cfg.seed, 0])
    hrng = np.random.default_rng([cfg.seed, 0, 43])
    orng = np.random.default_rng([cfg.seed, 0, 53])
    mrng = np.random.default_rng([cfg.seed, 0, 89])                  # route-memory drop
    sha = hashlib.sha1()
    lb, ln = (cfg.lr, cfg.lr_new) if a.lr_scale == 1.0 else (cfg.lr * a.lr_scale, cfg.lr_new * a.lr_scale)
    opt = torch.optim.AdamW([{"params": base, "lr": lb, "base": lb, "wu": cfg.warmup}, {"params": new, "lr": ln, "base": ln, "wu": cfg.warmup},
                             {"params": rp, "lr": a.rm_lr, "base": a.rm_lr, "wu": a.rm_warmup}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()

    def draw():
        r = rng.choice(tr_o, nN, replace=len(tr_o) < nN)
        if k_yr:
            r = np.concatenate([r, rng.choice(tr_t, k_yr, replace=False)])
        an = rng.random(nB - k_ho - k_on) < cfg.d_frac
        an[nN:] = False
        if k_on:
            i = orng.choice(ns.n_on, k_on, replace=ns.n_on < k_on)
            sha.update(i.astype("<i8").tobytes())
            r, an = np.concatenate([r, ns.on0 + i]), np.concatenate([an, np.zeros(k_on, bool)])
        if k_ho:
            r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
            an = np.concatenate([an, np.zeros(k_ho, bool)])
        return r, an, sha.hexdigest()[:16]

    def fetch(dr):
        return dr, S.front[dr[0]]
    S.front.pin = a.prefetch > 1
    Sd = T.dev_store(S, ns.dv_rows, a.dev_card_gb)

    def dev_ev():
        ev = T.dev_eval(model.eval(), Sd, dv_o, LS.W)
        if len(dv_t):
            ev |= {"ot_" + q: v for q, v in T.dev_eval(model, S, dv_t, LS.W).items()}
        has = S.has_fut[torch.as_tensor(dv_o, device=dev)].cpu().numpy()
        ev["ade_nomem"] = float(rows_ade(model, Sd, dv_o[has], LS.W, True).mean())
        model.train()
        return ev
    with Run("route_mem", f"train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"onpol": vars(a), "route_mem": dict(head=HEAD, feat_ok=okc)}) as run:
        for x in ns.sp + ((ns.trl,) if k_ho or ns.agent2 is not None else ()) + ((splits.load(a.on_split),) if a.on and a.on_split else ()):
            run.use_split(x)
        nroute = sum(p.numel() for p in rp)
        run.info(f"{a.tag}: on_train recipe + route branch ({nroute / 1e6:.2f} M, lr {a.rm_lr}, warm-up {a.rm_warmup}, drop {a.rm_drop}); "
                 f"train {len(tr_o)} normal + {len(tr_t)} yaw-rate rows, dev {len(dv_o)} + {len(dv_t)}; per batch {nN} normal, {k_yr} yaw-rate, "
                 f"{k_on} on-policy (mode {ns.mode or 'off'}; {json.dumps(ns.onrec)}); feature rows ok {json.dumps(okc)}")
        if ns.init:
            run.info(f"init {json.dumps(ns.init)}")
            ev = dev_ev()
            run.scalars({f"dev/{q}": v for q, v in ev.items()}, 0)
            run.info("dev @ 0: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items()))
            run.summary.update({f"dev0_{q}": v for q, v in ev.items()} | {"init": ns.init})
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist, hsh = time.time(), [], ""
        for step in range(cfg.steps):
            (r, an, hsh), front = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            cosf = 0.5 * (1 + np.cos(np.pi * (step / cfg.steps))) if pp_lr else 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / g["wu"]) * cosf
            ego = S.ego[rows] * (~anchor)[:, None].float()
            keep = torch.as_tensor(mrng.random(len(r)) >= a.rm_drop, device=dev)
            ego[:, RM.EGO_DIM:] = ego[:, RM.EGO_DIM:] * keep[:, None].float()
            out = model(front, ego, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows])
            total, Ls = LS(out, S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new + rp, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({q: float(v) for q, v in Ls.items()})
            if (step + 1) % 25 == 0 or step + 1 == cfg.steps:
                keys = sorted({q for h in hist for q in h})
                mm = {q: float(np.mean([h[q] for h in hist if q in h])) for q in keys}
                hist = []
                el = time.time() - t0
                rn = float(model.adapter.route.out.weight.norm())
                run.scalars({f"loss/{q}": v for q, v in mm.items()} | {"throughput/steps_per_s": (step + 1) / el, "route/out_norm": rn}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in mm.items() if "/" not in q) + f"; route out |W| {rn:.4f}"
                             + (f"; on_hash {hsh}" if k_on else "") + f"; {(step + 1) / el:.2f} it/s, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if k_on and (step + 1 == 60 or (step + 1) % 1000 == 0 or step + 1 == cfg.steps):
                run.summary.update({f"on_hash_{step + 1}": hsh})
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = dev_ev()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "onpol": vars(a), "route_mem": HEAD}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        RM.write_sidecar(a.tag, HEAD, dict(init=a.init, rm_lr=a.rm_lr, rm_warmup=a.rm_warmup, rm_drop=a.rm_drop))
        rr = rely(model, Sd, dv_o, LS.W, JD, TU, S.tab["log"])
        run.info(f"reliance: {json.dumps(rr)}")
        (_pl.Path(run.dir) / "rely.json").write_text(json.dumps(rr, indent=1))
        from jevdrive.common import data_dir
        out = data_dir() / "runs/route_mem/rely"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.tag}.json").write_text(json.dumps(rr, indent=1))
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), on_hash=hsh, rely=rr, route_params=nroute,
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


def cmd_rely(a):
    import torch
    import pp_train as T
    from jevdrive.common import data_dir
    dev = torch.device("cuda")
    a.rm_from = a.tag
    ns, rp, JD, TU, okc = build(a, dev)
    LS = ns.losses(ns.mode)
    Sd = T.dev_store(ns.S, ns.dv_rows, a.dev_card_gb)
    rr = rely(ns.model, Sd, ns.dv_o, LS.W, JD, TU, ns.S.tab["log"])
    print(json.dumps(rr, indent=1))
    out = data_dir() / "runs/route_mem/rely"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(rr, indent=1))


def cmd_zero(a):
    """--init TAG --tag OUT: TAG's weights + the branch at its zero init (the serving check: plans equal TAG's, the memory is logged)."""
    import torch
    import pp_train as T
    d = T.proot("runs", a.tag)
    assert not (d / "ckpt-final.pt").exists(), f"{d} exists"
    m = T.load_pmodel(a.init, torch.device("cuda"))
    RM.attach(m, **HEAD)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state(), "route_mem": HEAD, "zero_of": a.init}, d / "ckpt-final.pt")
    RM.write_sidecar(a.tag, HEAD, dict(init=a.init, zero=True))
    print("wrote", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "rely"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--exp", default="route_mem")
        p.add_argument("--steps", type=int, default=2500)
        p.add_argument("--batch", type=int, default=128)
        p.add_argument("--warmup", type=int, default=1)
        p.add_argument("--lr-scale", type=float, default=0.1)
        p.add_argument("--eval-every", type=int, default=500)
        OT.recipe_args(p)
        p.add_argument("--on-k", type=int, default=13)
        p.add_argument("--on-mode", default="", choices=("",) + OT.MODES)
        p.add_argument("--on-split", default="")
        p.add_argument("--on-lon", default="")
        p.add_argument("--on-lon-lam", type=float, default=10.0)
        p.add_argument("--on-road", default="")
        p.add_argument("--ego-drop", type=float, default=0.0)
        p.add_argument("--prefetch", type=int, default=4)
        p.add_argument("--fetch-workers", type=int, default=2)
        p.add_argument("--dev-card-gb", type=float, default=4.0)
        p.add_argument("--rm-lr", type=float, default=3e-4)
        p.add_argument("--rm-warmup", type=int, default=100)
        p.add_argument("--rm-drop", type=float, default=0.25)
    p = sub.add_parser("zero")
    p.add_argument("--init", required=True), p.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "zero":
        cmd_zero(a)
        raise SystemExit(0)
    assert a.ego_drop == 0 and not a.on_lon and not a.on_road, "ROUTE-MEM runs on_train's Y recipe only"
    dict(train=cmd_train, rely=cmd_rely)[a.cmd](a)
