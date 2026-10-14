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

Route-night additions (research/next-round/route-night.md 4 C):
  --init    a checkpoint WITH the branch (sidecar route_mem.json beside ckpt-final.pt, e.g. RB-F-s0): on_train.setup runs without --init (its
            load_init demands the plain P2 key set), then the branch is attached and the whole state loaded (cmd_rely's --rm-from path); the
            dev ADE at step 0 is logged against the init sidecar's recorded final dev ADE (`dev_ade`). An init without sidecar: as before
            (on_train's load_init, then the branch at zero). No --init: on_train's fresh path (shipped Cinque weights + a fresh P2 adapter,
            joint_train's start), then the branch; dev metrics at step 0 are logged in every case.
  defaults  on_train's schedule (10 000 steps, warm-up 300, --lr-scale 1.0, eval every 1000) = the YR10m10-F recipe from the shipped
            weights and d264's full continuation schedule; the old ROUTE-MEM continuation passes --steps 2500 --warmup 1 --lr-scale 0.1.
  no pool   --agent-on-only without --on (RB0, RNONE) means no row carries the agent hinge: run as --agent-lam 0 (on_train.setup would
            refuse the flag without a pool), so the RB / RB0 / RBS / RNONE commands share one $REC.
  pool rows --rm-drop never zeroes a pool row's (--on) route input (the route hinge must not train a turn without its reason); the drop rng
            stream still draws one number per batch row. --on-lon / --on-lon-lam / --on-road go to on_train.setup unchanged.

Next-round addition (2026-10-13, experiments/route_follow/plans/2026-10-13-allrows-prereg.md):
  --ego-drop-speed P   the speed history of the fed ego row (EGO_SPEED: vx, vy, ax, ay and the x / y of the 4 history poses, i.e. everything
            that carries the ego's own speed; `present`, the command and the 4 history yaws stay) zeroed on a share P of the non-anchor rows of
            every batch (pool rows included), own rng stream [seed, 0, 71] (on_train --ego-drop's stream; every other stream unchanged).
            Training only: nothing is saved with the weights, the served row is unchanged. The training-side fix for the slowness ratchet
            (research/next-round/steer-long.md 4.4: the plan follows its own speed history). on_train's --ego-drop (the whole EGO_STATE,
            yaws included) stays refused here. 0 = off: no operation is added.

Turning-school addition (2026-10-13, experiments/route_follow/plans/2026-10-13-turning-school-prereg.md):
  --on-weight KEY=VALUE:W,...   pool rows are drawn with probability proportional to the product of W over the matching terms (string keys of
            the pool's tab.npz, e.g. src=synthetic:0.5,topo=straight_exit:2; a row without the key or value has factor 1): the expected
            weight of a row class in the imitation / hinge sums, without touching the loss. Same orng stream [seed, 0, 53] (numpy's
            weighted choice). '' = off: the uniform draw, bit for bit as before.

  --on-cap-scene C [--on-cap-in KEY=VALUE]   per-scene draw cap (2026-10-14, turning school phase 2: a branch-search pool is dominated by the
            scenes with many completing branches): among the pool rows matching --on-cap-in (default: every pool row), the weights after
            --on-weight are rescaled per scene (tab `log`) so that no scene holds more than a share C of the subset's draw mass. Water-filling:
            scenes over C are set to C, the rest scaled up in proportion, repeated until none exceeds; the subset's total mass is kept, so a
            --on-weight W computed for the subset's share of the draws stays exact. Infeasible (scenes x C < 1) is an error. Same orng stream
            and weighted choice as --on-weight; 0 = off. `capcheck` is its known-answer test (CPU: 3 scenes 70 / 20 / 10 % capped at 0.4 ->
            40 / 40 / 20 % within 1e-6, the rest of the pool untouched; exit 2 otherwise); `draws --on POOL [--on-weight ..] [--on-cap-scene ..]`
            prints the draw shares the trainer would use (per `pool` tag, and per scene inside the cap subset before / after), no GPU.

  --on-mode mixed   on_train's per-row mode: pool rows tagged mode == "self" (S's on-policy rows) keep the `self` recipe, the rest (demonstration
            rows) are `both`; `mixcheck` is its known-answer test: on one setup, a batch whose pool rows are all `self` rows gives the
            `self` loss and gradient exactly, a batch of `both` rows the `both` ones (exit 2 otherwise).

Longitudinal safety (2026-10-13, experiments/route_follow/plans/2026-10-13-lon-prereg.md):
  --lon-lead W --lon-curve W --lon-stop W   the timing-only terms of onpol_nuplan lib/on_lonsafe.py on the pool rows (lead envelope, curve speed,
            stop-respect): any > 0 sets --on-lon on_lonsafe and --on-lon-lam 1 (the weights carry the scale: each term enters as W x ho_w x
            sum over the pool rows / n_imit); refused together with another --on-lon module. All 0 (default) = off: no operation is added.

  RB:  rm_train.py train --tag RB-F-s0 --steps 10000 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 \
       --seed 0 --on on1_CF --on-k 12 --on-mode self --on-lon on_route          (RB0: the same without the four --on flags)

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
EGO_SPEED = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]              # ego_features: vx, vy, ax, ay, history poses x / y (not present, cmd, yaw)


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
    import pp_train as T
    init, a.init = a.init, ("" if a.rm_from or (a.init and RM.has_branch(a.init)) else a.init)   # setup's load_init: plain P2 checkpoints only
    ns = OT.setup(a, dev)
    a.init = init
    RM.attach(ns.model, **HEAD)
    src = a.rm_from or (a.init if a.init and RM.has_branch(a.init) else "")
    if src:                                                           # a checkpoint with the branch: the whole state, branch included
        ck = torch.load(T.proot("runs", src) / "ckpt-final.pt", map_location="cpu", weights_only=False)
        sd = json.loads((T.proot("runs", src) / RM.SIDECAR).read_text()) if RM.has_branch(src) else {}
        assert sd.get("head", HEAD) == HEAD, f"{src}: branch {sd.get('head')} != {HEAD}"
        ns.model.load_state(ck["model"])
        st = ns.model.adapter.state_dict()
        assert set(st) == set(ck["model"]["parity"]), "adapter keys differ from the checkpoint's"
        da = max(float((v.detach().cpu().float() - ck["model"]["parity"][k].float()).abs().max()) for k, v in st.items())
        dn = max(float((ns.model.net.params[k].detach().cpu().float() - v.float()).abs().max()) for k, v in ck["model"]["net"].items())
        assert da == 0 and dn == 0, (da, dn)
        if not a.rm_from:
            ns.init = dict(init=src, with_branch=True, net_loaded=len(ck["model"]["net"]), adapter_loaded=len(st), max_abs_diff_after_load=0.0,
                           init_dev_ade=sd.get("dev_ade"))
    F, JD, TU, src, okc = feats(ns.cfg.data, ns.S, dev)
    ns.S.ego = torch.cat([ns.S.ego.float(), F], 1)
    rp = list(ns.model.adapter.route.parameters())
    ids = {id(p) for p in rp}
    ns.new = [p for p in ns.model.adapter.parameters() if id(p) not in ids]
    return ns, rp, JD, TU, okc


def cap_share(w, g, cap):
    """Weights w (n,) of rows in groups g (n,) -> weights with no group above a share `cap` of sum(w), sum kept (--on-cap-scene)."""
    w = np.asarray(w, np.float64)
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    p = np.bincount(inv, w, len(u)) / w.sum()
    assert len(u) * cap >= 1 - 1e-12, f"--on-cap-scene {cap}: infeasible with {len(u)} scenes (needs >= {int(np.ceil(1 / cap))})"
    q, fixed = p.copy(), np.zeros(len(u), bool)
    while ((q > cap + 1e-12) & ~fixed).any():
        fixed |= q > cap + 1e-12
        q[fixed] = cap
        q[~fixed] = p[~fixed] * (1 - cap * fixed.sum()) / p[~fixed].sum()
    return w * (q / p)[inv]


def group_shares(w, g):
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    return {k: float(v) for k, v in zip(u, np.bincount(inv, w, len(u)) / np.sum(w))}


def draw_weights(tb, on_weight="", cap=0.0, cap_in=""):
    """Pool tab -> (per-row draw weights or None for the plain uniform draw, record): --on-weight product, then --on-cap-scene."""
    if not on_weight and not cap:
        return None, {}
    n = len(tb["names"])
    w = np.ones(n)
    for term in on_weight.split(",") if on_weight else []:
        kv, f = term.rsplit(":", 1)
        k, v = kv.split("=", 1)
        if k in tb.files:
            w[tb[k].astype(str) == v] *= float(f)
    rec = {}
    if cap:
        k, v = cap_in.split("=", 1) if cap_in else ("", "")
        sel = tb[k].astype(str) == v if k else np.ones(n, bool)
        assert sel.any() and "log" in tb.files, f"--on-cap-in {cap_in}: no pool row / no `log` tag"
        g = tb["log"].astype(str)[sel]
        before = group_shares(w[sel], g)
        m0 = w[sel].sum()
        w[sel] = cap_share(w[sel], g, cap)
        assert abs(w[sel].sum() - m0) <= 1e-9 * m0
        rec = dict(cap=cap, cap_in=cap_in or "all", rows=int(sel.sum()), scenes=len(before), before=before, after=group_shares(w[sel], g))
    assert (w > 0).all(), "--on-weight / --on-cap-scene: weights"
    if "pool" in tb.files:
        rec["pool_share"] = group_shares(w, tb["pool"])
    return w, rec


def cmd_capcheck(a):
    """Known-answer test of --on-cap-scene (CPU): pool of 100 `s` rows (scenes a / b / c: 70 / 20 / 10 rows) + 30 `r` rows, cap 0.4 on pool=s.
    uniform: shares 0.4 / 0.4 / 0.2 exactly; weighted (--on-weight topo=x:3,pool=s:2.5, unequal weights inside a scene): the shares equal an
    independent solution (q = min(cap, lam p), lam by bisection); both: the subset's mass kept, `r` rows untouched, every row of a scene scaled
    by one factor; an infeasible cap refused."""
    class T(dict):
        files = property(lambda self: list(self))
    rng = np.random.default_rng(0)
    log = np.repeat(["a", "b", "c"], [70, 20, 10])
    tb = T(names=np.arange(130).astype(str), log=np.r_[log, np.full(30, "r")], pool=np.repeat(["s", "r"], [100, 30]), topo=rng.choice(["x", "y"], 130))
    res = {}
    for name, ow in (("uniform", ""), ("weighted", "topo=x:3,pool=s:2.5")):
        w0 = draw_weights(tb, ow)[0] if ow else np.ones(130)
        p0 = group_shares(w0[:100], log)
        p = np.array([p0[k] for k in "abc"])
        lo, hi = 0.0, 1.0 / p.min()
        for _ in range(200):
            lam = (lo + hi) / 2
            lo, hi = (lam, hi) if np.minimum(0.4, lam * p).sum() < 1 else (lo, lam)
        want = dict(zip("abc", (0.4, 0.4, 0.2) if name == "uniform" else np.minimum(0.4, lam * p)))
        w, rec = draw_weights(tb, ow, 0.4, "pool=s")
        f = w[:100] / w0[:100]
        res[name] = dict(before=p0, after=rec["after"], want={k: float(v) for k, v in want.items()},
                         max_err=max(max(abs(rec["after"][k] - want[k]) for k in "abc"), abs(w[:100].sum() / w0[:100].sum() - 1),
                                     float(np.abs(w[100:] - w0[100:]).max())),
                         within_scene_spread=max(float(np.ptp(f[log == k])) for k in "abc"))
    try:
        cap_share(np.ones(10), np.repeat(["a", "b"], 5), 0.4)
        res["infeasible_refused"] = False
    except AssertionError:
        res["infeasible_refused"] = True
    ok = res["infeasible_refused"] and all(res[n]["max_err"] <= 1e-6 and res[n]["within_scene_spread"] <= 1e-9 for n in ("uniform", "weighted"))
    print(json.dumps(dict(res, pass_=ok), indent=1))
    raise SystemExit(0 if ok else 2)


def cmd_draws(a):
    """Draw shares the trainer would use on --on with --on-weight / --on-cap-scene (CPU)."""
    from jevdrive.common import data_dir
    tb = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")
    w, rec = draw_weights(tb, a.on_weight, a.on_cap_scene, a.on_cap_in)
    print(json.dumps(dict(pool=a.on, rows=len(tb["names"]), on_weight=a.on_weight, **rec), indent=1))


def cmd_train(a):
    import torch
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    d = T.proot("runs", a.tag)
    assert not (d / "ckpt-final.pt").exists(), f"{d / 'ckpt-final.pt'} exists: a tag is never overwritten"
    a.rm_from = ""
    assert not a.init or a.init.startswith(("RB-F-", "RB0-F-")) or a.allow_init, f"--init {a.init}: tonight only RB-F / RB0-F (route-night.md 3; --allow-init overrides)"
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
    erng = np.random.default_rng([cfg.seed, 0, 71])                  # --ego-drop-speed's rows
    esp = torch.as_tensor(EGO_SPEED, device=dev)
    sha = hashlib.sha1()
    lb, ln = (cfg.lr, cfg.lr_new) if a.lr_scale == 1.0 else (cfg.lr * a.lr_scale, cfg.lr_new * a.lr_scale)
    opt = torch.optim.AdamW([{"params": base, "lr": lb, "base": lb, "wu": cfg.warmup}, {"params": new, "lr": ln, "base": ln, "wu": cfg.warmup},
                             {"params": rp, "lr": a.rm_lr, "base": a.rm_lr, "wu": a.rm_warmup}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()

    pw, caprec = None, {}
    if a.on_weight or a.on_cap_scene:                                 # --on-weight / --on-cap-scene: per-row draw probabilities of the pool rows
        from jevdrive.common import data_dir
        tb = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")
        assert len(tb["names"]) == ns.n_on, "--on-weight: pool rows"
        w, caprec = draw_weights(tb, a.on_weight, a.on_cap_scene, a.on_cap_in)
        pw = w / w.sum()

    def draw():
        r = rng.choice(tr_o, nN, replace=len(tr_o) < nN)
        if k_yr:
            r = np.concatenate([r, rng.choice(tr_t, k_yr, replace=False)])
        an = rng.random(nB - k_ho - k_on) < cfg.d_frac
        an[nN:] = False
        if k_on:
            i = orng.choice(ns.n_on, k_on, replace=ns.n_on < k_on) if pw is None else orng.choice(ns.n_on, k_on, replace=True, p=pw)
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
        if pw is not None:
            run.info(f"--on-weight {a.on_weight}: pool row draw probability min {pw.min() * ns.n_on:.3f} / max {pw.max() * ns.n_on:.3f} x uniform")
        if caprec:
            run.info(f"draw shares: {json.dumps(caprec)}")
            run.summary.update(draw_shares=caprec)
        if a.ego_drop_speed > 0:
            run.info(f"ego speed history (ego cols {EGO_SPEED}) zeroed on a share {a.ego_drop_speed} of the non-anchor rows, rng [seed, 0, 71]")
        run.info(f"init {json.dumps(ns.init) if ns.init else 'none: shipped weights + fresh adapter (on_train / joint_train start)'}")
        ev = dev_ev()
        run.scalars({f"dev/{q}": v for q, v in ev.items()}, 0)
        run.info("dev @ 0: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items())
                 + (f"; init's recorded dev ade {ns.init['init_dev_ade']:.4f}" if ns.init and ns.init.get("init_dev_ade") is not None else ""))
        run.summary.update({f"dev0_{q}": v for q, v in ev.items()} | {"init": ns.init})
        is_pool = torch.zeros(S.n, dtype=torch.bool, device=dev)
        is_pool[ns.on0:] = bool(k_on)                                 # --rm-drop never applies to pool rows
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist, hsh = time.time(), [], ""
        for step in range(cfg.steps):
            (r, an, hsh), front = pre.get()
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            cosf = 0.5 * (1 + np.cos(np.pi * (step / cfg.steps))) if pp_lr else 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / g["wu"]) * cosf
            ego = S.ego[rows] * (~anchor)[:, None].float()
            keep = torch.as_tensor(mrng.random(len(r)) >= a.rm_drop, device=dev) | is_pool[rows]
            ego[:, RM.EGO_DIM:] = ego[:, RM.EGO_DIM:] * keep[:, None].float()
            if a.ego_drop_speed > 0:
                dm = torch.as_tensor(erng.random(len(r)) < a.ego_drop_speed, device=dev)
                ego[:, esp] = ego[:, esp] * (~dm)[:, None].float()
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
        RM.write_sidecar(a.tag, HEAD, dict(init=a.init, rm_lr=a.rm_lr, rm_warmup=a.rm_warmup, rm_drop=a.rm_drop, rm_drop_pool=False,
                                           on=a.on, on_lon=a.on_lon, steps=cfg.steps, dev_ade=ev.get("ade")))
        rr = rely(model, Sd, dv_o, LS.W, JD, TU, S.tab["log"])
        run.info(f"reliance: {json.dumps(rr)}")
        (_pl.Path(run.dir) / "rely.json").write_text(json.dumps(rr, indent=1))
        from jevdrive.common import data_dir
        out = data_dir() / "runs/route_mem/rely"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.tag}.json").write_text(json.dumps(rr, indent=1))
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), on_hash=hsh, rely=rr, route_params=nroute,
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


def cmd_mixcheck(a):
    """Known-answer test of --on-mode mixed (docstring): same normal rows + k pool rows of one kind, mixed vs the pure mode, per batch."""
    import torch
    dev = torch.device("cuda")
    a.rm_from, a.on_mode = "", "mixed"
    ns, rp, JD, TU, okc = build(a, dev)
    S, model, k = ns.S, ns.model, ns.k_on
    from jevdrive.common import data_dir
    md = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")["mode"].astype(str)
    rng = np.random.default_rng([a.seed, 0, 97])
    res = {}
    for kind, ref in (("self", "self"), ("both", "both")):
        pool = np.flatnonzero(md == kind if kind == "self" else md != "self")
        worst = 0.0
        for b in range(3):
            r = np.concatenate([rng.choice(ns.tr_o, ns.nN, replace=False), ns.on0 + rng.choice(pool, k, replace=False)])
            rows, an = torch.as_tensor(r, device=dev), torch.zeros(len(r), dtype=torch.bool, device=dev)
            with torch.no_grad():
                out = model(S.front[rows], S.ego[rows], S.tc[rows], None, None, nv=None)
            vals = []
            for m in ("mixed", ref):
                o = out.clone().requires_grad_(True)
                tot, _ = ns.losses(m)(o, S, rows, an)
                g = torch.autograd.grad(tot, o)[0]
                vals.append((float(tot), g))
            worst = max(worst, abs(vals[0][0] - vals[1][0]), float((vals[0][1] - vals[1][1]).abs().max()))
        res[kind] = worst
    ok = all(v <= 1e-6 for v in res.values())
    print(json.dumps(dict(pool=a.on, max_abs_diff=res, pass_=ok)))
    raise SystemExit(0 if ok else 2)


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
    for name in ("train", "rely", "mixcheck"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--exp", default="route_mem")
        p.add_argument("--steps", type=int, default=10000)
        p.add_argument("--batch", type=int, default=128)
        p.add_argument("--warmup", type=int, default=300)
        p.add_argument("--lr-scale", type=float, default=1.0)
        p.add_argument("--eval-every", type=int, default=1000)
        p.add_argument("--allow-init", action="store_true", help="permit an --init other than tonight's RB-F / RB0-F")
        OT.recipe_args(p)
        p.add_argument("--on-k", type=int, default=13)
        p.add_argument("--on-mode", default="", choices=("",) + OT.ALL_MODES)
        p.add_argument("--on-split", default="")
        p.add_argument("--on-lon", default="")
        p.add_argument("--on-lon-lam", type=float, default=10.0)
        p.add_argument("--on-road", default="")
        p.add_argument("--ego-drop", type=float, default=0.0)
        p.add_argument("--on-weight", default="", help="KEY=VALUE:W,... draw weights of the pool rows from tab.npz string keys")
        p.add_argument("--on-cap-scene", type=float, default=0.0, help="max share of one scene (tab `log`) in the cap subset's draws (0 = off)")
        p.add_argument("--on-cap-in", default="", help="KEY=VALUE: the pool rows the cap applies to (default: all pool rows)")
        p.add_argument("--ego-drop-speed", type=float, default=0.0, help="share of non-anchor rows with the ego speed history zeroed (training only)")
        for t in ("lead", "curve", "stop"):
            p.add_argument(f"--lon-{t}", type=float, default=0.0, help=f"weight of on_lonsafe's {t} term on the pool rows (0 = off)")
        p.add_argument("--prefetch", type=int, default=4)
        p.add_argument("--fetch-workers", type=int, default=2)
        p.add_argument("--dev-card-gb", type=float, default=4.0)
        p.add_argument("--rm-lr", type=float, default=3e-4)
        p.add_argument("--rm-warmup", type=int, default=100)
        p.add_argument("--rm-drop", type=float, default=0.25)
    p = sub.add_parser("zero")
    p.add_argument("--init", required=True), p.add_argument("--tag", required=True)
    sub.add_parser("capcheck")
    p = sub.add_parser("draws")
    p.add_argument("--on", required=True), p.add_argument("--on-weight", default="")
    p.add_argument("--on-cap-scene", type=float, default=0.0), p.add_argument("--on-cap-in", default="")
    a = ap.parse_args()
    if a.cmd in ("zero", "capcheck", "draws"):
        dict(zero=cmd_zero, capcheck=cmd_capcheck, draws=cmd_draws)[a.cmd](a)
        raise SystemExit(0)
    assert not a.on_cap_scene or (a.on and 0 < a.on_cap_scene < 1), "--on-cap-scene: needs --on, a share in (0, 1)"
    assert a.ego_drop == 0, "ROUTE-MEM: no --ego-drop (--ego-drop-speed drops the speed history only)"
    assert 0 <= a.ego_drop_speed < 1, "--ego-drop-speed is a share in [0, 1)"
    if a.agent_on_only and not a.on:                                 # agent hinge on pool rows only, and no pool: no row carries it (RB0 / RNONE)
        print("rm_train: --agent-on-only without --on -> --agent-lam 0 (no row has an agent label)", flush=True)
        a.agent_lam, a.agent_on_only = 0.0, False
    if max(a.lon_lead, a.lon_curve, a.lon_stop) > 0:                 # on_lonsafe: the weights carry the scale
        assert a.on and a.on_lon in ("", "on_lonsafe"), "--lon-*: needs --on and no other --on-lon module"
        import on_lonsafe
        a.on_lon, a.on_lon_lam = "on_lonsafe", 1.0
        print(f"rm_train: --on-lon on_lonsafe, --on-lon-lam 1, weights {on_lonsafe.configure(a.lon_lead, a.lon_curve, a.lon_stop)}", flush=True)
    dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)
