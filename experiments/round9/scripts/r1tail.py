#!/usr/bin/env python3
"""R1 tail diagnostic (round 9 CB2, after plan amendment 7). Diagnostic only: reads pools, tokens, dumps and checkpoints, writes only
$DATA_DIR/runs/round9/cb2/r1tail/ (no gate, pool, row or lane file is touched). Result page: experiments/round9/results/cb2/r1_tail_diag.md.

Question: why a few rows of P1N''s pools replay (gate R1 (a): trainer forward of the serving checkpoint on the stored row against the served
plan, first 15 plan points x / y) with |diff| ~0.0625 m at points inside 64 m, and whether other lineages' pools have the same tail.

  stat --pools P ...   per pool: the R1 forward exactly as on_rows.cmd_tok diff() (cached eager tokens, batch 128, fed ego / route memory)
                       -> per point |diff|, the point's magnitude, its fp16 step; <out>/stat_<P>.npz + stat_<P>.json
  loc  --pool P        the rows over max(0.03126 m, own step) + --k random rows: the same tokens through (b128) the trainer batch, (b1) one row
                       at a time eager, (c1) one row at a time through Core.compile's policy pass; (j) sensitivity: the plan's change under
                       one-fp16-step jitter of the tokens / 1e-5 jitter of the route memory -> <out>/loc_<P>.json
  full --pool P        a pool whose dumps keep the slot frames (`cur`; PN6): tokens through the eager encoder (pp_prep.encoder, as on_rows)
                       and through Core.compile's encoder pass (8 pairs, as served), crossed with eager b128 / compiled b1 policy, against
                       the served mu -> <out>/full_<P>.json (where the difference enters: encoder vs policy)
GPU, envs/op-train, a pool job (python -m jevdrive.cl submit ... -- $DATA_DIR/envs/op-train/bin/python experiments/round9/scripts/r1tail.py ...).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_R / "experiments/onpol_nuplan/scripts"))
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402
import on_rows as O  # noqa: E402  (path setup, ROWS / CR / FAM)

OUT = O.DATA / "runs/round9/cb2/r1tail"
ULP1 = 0.03126


def step16(x):
    """fp16 rounding step at magnitude x (as on_rows ON_R1_ULP)."""
    x = np.abs(x)
    return np.where(x < 2.0 ** -14, 2.0 ** -24, np.exp2(np.floor(np.log2(np.maximum(x, 2.0 ** -14))) - 10))


class Pool:
    def __init__(self, name, dev):
        import torch
        _sys.path.insert(0, str(_R / "experiments/route_mem/lib"))
        import rmem as RM
        self.torch, self.dev, self.name = torch, dev, name
        self.M = dict(np.load(O.ROWS / name / "meta.npz"))
        tp = O.CR / f"{O.FAM}_{name}" / "tab.npz"
        if tp.exists():
            self.tab = dict(np.load(tp))
        else:                                                            # pool cache deleted (PN6): the fed ego row from the dumps, as geo builds the tab
            self.tab = dict(names=self.M["names"], speed=self.M["speed"],
                            ego=np.stack([np.load(f)["ego"].astype(np.float32) for f in self.M["dump"]]))
        fp = O.CR / f"{O.FAM}_{name}@warp" / "front.npy"
        self.front = np.load(fp, mmap_mode="r") if fp.exists() else None
        tags = sorted(set(self.M["tag"].tolist()))
        assert len(tags) == 1, tags
        self.tag = tags[0]
        self.model = RM.load_pmodel(self.tag, dev)
        self.branch = RM.has_branch(self.tag)
        n = len(self.tab["names"])
        self.ego = torch.from_numpy(self.tab["ego"]).to(dev)
        self.rmem = torch.from_numpy(self.M["rmem"]).to(dev) if "rmem" in self.M else torch.zeros(n, RM.DIM, device=dev)
        self.tc = torch.tensor([[1.0, 0.0]], device=dev)
        s = self.model.net.slices["plan"].start
        self.pi = slice(s, s + 33 * 15)

    def feed(self, r, rm_jit=0.0):
        e = self.ego[r]
        if not self.branch:
            return e
        m = self.rmem[r]
        if rm_jit:
            m = m + (self.torch.rand_like(m) - 0.5) * 2 * rm_jit
        return self.torch.cat([e, m], 1)

    def plan(self, H, r, fn=None, rm_jit=0.0):
        """H (b, 8, 32, 512) tokens on the card -> (b, 33, 15) plan mean (fp16 output as float)."""
        f = fn or self.model
        rt = self.torch.as_tensor(np.atleast_1d(r), device=self.dev)
        return f(H, self.feed(rt, rm_jit), self.tc.repeat(len(rt), 1)).float()[:, self.pi].reshape(-1, 33, 15)

    def tokens(self, r):
        return self.torch.from_numpy(np.ascontiguousarray(self.front[r])).to(self.dev)


def per_point(p, mu):
    """|diff| (n, 15, 2) of the first 15 points x / y, magnitude max(|p|, |mu|), fp16 step there."""
    d = np.abs(p[:, :15, :2] - mu[:, :15, :2])
    mg = np.maximum(np.abs(p[:, :15, :2]), np.abs(mu[:, :15, :2]))
    return d, mg, step16(mg)


def cmd_stat(a):
    import torch
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.pools:
        P = Pool(name, dev)
        n = len(P.tab["names"])
        out = np.zeros((n, 33, 15), np.float32)
        with torch.no_grad():
            for i in range(0, n, 128):                                   # = on_rows diff(): batches of 128 in row order
                r = np.arange(i, min(i + 128, n))
                out[r] = P.plan(P.tokens(r), r).cpu().numpy()
        mu = P.M["mu"]
        d, mg, st = per_point(out, mu)
        dm = d.max((1, 2))
        fl = (d / np.maximum(st, ULP1)).max((1, 2))
        big = mg >= 4.0                                                   # points away from zero: the relative-error view
        rel = (d / np.maximum(mg, 1e-6))[big]
        ust = (d / st)[big]
        fast = np.abs(mu[:, :15, 0]).max(1) >= 32.0
        over = np.flatnonzero(fl > 1.0)
        pts = []
        for i in over:
            j, c = np.unravel_index(np.argmax(d[i] / np.maximum(st[i], ULP1)), d[i].shape)
            pts.append(dict(row=int(i), name=str(P.tab["names"][i]), pt=int(j), axis="xy"[c], diff=float(d[i, j, c]), served=float(mu[i, j, c]),
                            trainer=float(out[i, j, c]), steps_own=float(d[i, j, c] / st[i, j, c]), speed=float(P.tab["speed"][i]),
                            x15=float(mu[i, 14, 0])))
        js = dict(pool=name, tag=P.tag, branch=P.branch, n=n, mu_max=float(dm.max()), over_0p03=int((dm > 0.03).sum()), over_1ulp=int((dm > ULP1).sum()),
                  over_floor=int(len(over)), at_0p0625=int((dm >= 0.0625).sum()), fast_rows=int(fast.sum()),
                  over_floor_fast=int(fast[over].sum()), rate_over_floor_per_1k_fast=float(1e3 * len(over) / max(fast.sum(), 1)),
                  rate_over_floor_per_1k=float(1e3 * len(over) / n), max_steps_own_big=float(ust.max()),
                  steps_own_big_q=[float(np.quantile(ust, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)],
                  rel_big_q=[float(np.quantile(rel, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)], rel_big_max=float(rel.max()),
                  diff_hist={f"{k:.6f}": int((np.isclose(dm, k)).sum()) for k in sorted(set(np.round(dm[dm > ULP1], 7).tolist()))},
                  points=pts)
        np.savez_compressed(OUT / f"stat_{name}.npz", trainer=out.astype(np.float16), dmax=dm, floor=fl)
        (OUT / f"stat_{name}.json").write_text(json.dumps(js, indent=1) + "\n")
        print(name, json.dumps({k: v for k, v in js.items() if k not in ("points",)}), flush=True)
        del P
        torch.cuda.empty_cache()


def compiled(P, enc=False):
    import torch
    import pp_train  # noqa: F401
    from jevdrive import op_adapt as A
    net = P.model.net
    if enc:
        return torch.compile(lambda p, c: net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"])
    return torch.compile(lambda H, ego, tc: P.model(H, ego, tc))     # = sh30_core.Core.compile (graph False, the SH30_COMPILE default)


def maxd(p, mu):
    d, mg, st = per_point(p, mu)
    return d.max((1, 2)), (d / np.maximum(st, ULP1)).max((1, 2))


def cmd_loc(a):
    import torch
    dev = torch.device("cuda")
    P = Pool(a.pool, dev)
    S = json.loads((OUT / f"stat_{a.pool}.json").read_text())
    st = np.load(OUT / f"stat_{a.pool}.npz")
    bad = np.array([q["row"] for q in S["points"]], int)
    rng = np.random.default_rng(0)
    ctl = np.sort(rng.choice(np.setdiff1d(np.arange(S["n"]), bad), a.k, replace=False))
    rows = np.r_[bad, ctl]
    mu = P.M["mu"][rows]
    pol = compiled(P)
    res = dict(pool=a.pool, tag=P.tag, n_bad=len(bad), n_ctl=len(ctl))
    with torch.no_grad():
        b128 = st["trainer"][rows].astype(np.float32)
        b1 = np.concatenate([P.plan(P.tokens([i]), i).cpu().numpy() for i in rows])
        c1 = np.concatenate([P.plan(P.tokens([i]), i, pol).cpu().numpy() for i in rows])
        # ego of the tab against the dump's (the fed row)
        egod = np.array([np.abs(np.load(P.M["dump"][i])["ego"].astype(np.float32) - P.tab["ego"][i]).max() for i in rows])
        jt, jr = [], []
        for i in rows:                                                   # sensitivity: one-step token jitter (each entry -1 / 0 / +1 fp16 step)
            H = P.tokens([i])
            ref = P.plan(H, i).cpu().numpy()
            dj = []
            for s in range(a.jit):
                g = torch.Generator(device=dev).manual_seed(s)
                Hf = H.float()
                stp = torch.exp2(torch.floor(torch.log2(Hf.abs().clamp_min(2.0 ** -14))) - 10)
                sg = torch.randint(-1, 2, H.shape, device=dev, generator=g).float()
                dj.append(np.abs(P.plan((Hf + sg * stp).half(), i).cpu().numpy() - ref)[:, :15, :2].max())
            jt.append(dj)
            jr.append(np.abs(P.plan(H, i, rm_jit=1e-5).cpu().numpy() - ref)[:, :15, :2].max() if P.branch else 0.0)
    jt, jr = np.array(jt), np.array(jr)
    for k, p in (("b128", b128), ("b1_eager", b1), ("c1_compiled", c1)):
        dm, fl = maxd(p, mu)
        res[k] = dict(bad_dmax_median=float(np.median(dm[:len(bad)])), bad_dmax_max=float(dm[:len(bad)].max()), bad_over_floor=int((fl[:len(bad)] > 1).sum()),
                      ctl_dmax_median=float(np.median(dm[len(bad):])), ctl_dmax_max=float(dm[len(bad):].max()), ctl_over_floor=int((fl[len(bad):] > 1).sum()))
    res["b1_vs_b128_identical"] = float((np.abs(b1 - b128).max((1, 2)) == 0).mean())
    res["c1_vs_b1_max"] = dict(bad=float(np.abs(c1 - b1)[:len(bad), :15, :2].max()), ctl=float(np.abs(c1 - b1)[len(bad):, :15, :2].max()),
                               bad_median=float(np.median(np.abs(c1 - b1)[:len(bad), :15, :2].max((1, 2)))),
                               ctl_median=float(np.median(np.abs(c1 - b1)[len(bad):, :15, :2].max((1, 2)))))
    res["ego_tab_vs_dump_max"] = float(egod.max())
    res["token_jitter_plan_change"] = dict(bad_median=float(np.median(jt[:len(bad)].max(1))), bad_max=float(jt[:len(bad)].max()),
                                           ctl_median=float(np.median(jt[len(bad):].max(1))), ctl_max=float(jt[len(bad):].max()),
                                           ctl_q90=float(np.quantile(jt[len(bad):].max(1), 0.9)))
    res["rmem_jitter_plan_change"] = dict(bad_max=float(jr[:len(bad)].max()), ctl_max=float(jr[len(bad):].max()))
    # where the bad rows sit: speed, plan reach, command, stop-like plans
    sp, x15 = P.tab["speed"][rows], P.M["mu"][rows][:, 14, 0]
    res["bad_speed_q"] = [float(np.quantile(sp[:len(bad)], q)) for q in (0.1, 0.5, 0.9)] if len(bad) else []
    res["ctl_speed_q"] = [float(np.quantile(sp[len(bad):], q)) for q in (0.1, 0.5, 0.9)]
    res["bad_x15_q"] = [float(np.quantile(x15[:len(bad)], q)) for q in (0.1, 0.5, 0.9)] if len(bad) else []
    res["ctl_x15_q"] = [float(np.quantile(x15[len(bad):], q)) for q in (0.1, 0.5, 0.9)]
    res["pool_speed_q"] = [float(np.quantile(P.tab["speed"], q)) for q in (0.1, 0.5, 0.9)]
    (OUT / f"loc_{a.pool}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


def cmd_full(a):
    import torch
    import pp_prep as PP
    dev = torch.device("cuda")
    P = Pool(a.pool, dev)
    n = len(P.tab["names"])
    rows = np.arange(n) if a.n <= 0 else np.sort(np.random.default_rng(0).choice(n, min(a.n, n), replace=False))
    _, enc = PP.encoder(dev)
    cenc, pol = compiled(P, enc=True), compiled(P)
    mu = P.M["mu"][rows]
    E, C = np.zeros((len(rows), 8, 32, 512), np.float16), np.zeros((len(rows), 8, 32, 512), np.float16)
    with torch.no_grad():
        for j, i in enumerate(rows):
            cur = np.load(P.M["dump"][i])["cur"]                         # (8, 2, 6, 128, 256) as fed
            prev = np.concatenate([np.zeros_like(cur[:1]), cur[:-1]])
            E[j] = enc(prev, cur)                                         # on_rows make_front: eager encoder (here 8 pairs a call)
            C[j] = cenc(torch.from_numpy(prev).to(dev), torch.from_numpy(cur).to(dev)).reshape(8, 32, 512).cpu().numpy()
        res = dict(pool=a.pool, tag=P.tag, rows=len(rows))
        tk = dict(eager=E, compiled=C)
        if P.front is not None:
            tk["cached"] = np.asarray(P.front[rows])
        res["tokens_compiled_vs_eager"] = dict(max_abs=float(np.abs(C.astype(np.float32) - E).max()),
                                               share_equal=float((C == E).mean()),
                                               rel_q99=float(np.quantile(np.abs(C.astype(np.float32) - E) / (np.abs(E.astype(np.float32)) + 1e-2), 0.99)))
        for tn, T in tk.items():
            for pn in ("b128", "b1_eager", "c1_compiled"):
                ps = []
                if pn == "b128":
                    for i in range(0, len(rows), 128):
                        r = np.arange(i, min(i + 128, len(rows)))
                        ps.append(P.plan(torch.from_numpy(T[r]).to(dev), rows[r]).cpu().numpy())
                else:
                    f = pol if pn == "c1_compiled" else None
                    ps = [P.plan(torch.from_numpy(T[j:j + 1]).to(dev), i, f).cpu().numpy() for j, i in enumerate(rows)]
                p = np.concatenate(ps)
                dm, fl = maxd(p, mu)
                res[f"{tn}+{pn}"] = dict(identical=float((np.abs(p[:, :15, :2] - mu[:, :15, :2]).max((1, 2)) == 0).mean()), dmax_median=float(np.median(dm)),
                                         dmax_max=float(dm.max()), share_1ulp=float((dm <= ULP1).mean()), over_floor=int((fl > 1).sum()))
    (OUT / f"full_{a.pool}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stat", "loc", "full"])
    ap.add_argument("--pools", nargs="*", default=[])
    ap.add_argument("--pool", default="")
    ap.add_argument("--k", type=int, default=64, help="loc: random control rows")
    ap.add_argument("--jit", type=int, default=4, help="loc: token-jitter draws a row")
    ap.add_argument("--n", type=int, default=0, help="full: rows (0 = all)")
    a = ap.parse_args()
    from jevdrive.run import Run
    with Run("round9", f"r1tail-{a.cmd}-{a.pool or '_'.join(a.pools)}", resume=OUT / f"run_{a.cmd}_{a.pool or 'stat'}", config=vars(a)):
        {"stat": cmd_stat, "loc": cmd_loc, "full": cmd_full}[a.cmd](a)
