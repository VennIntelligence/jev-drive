"""junction_eyes step 1 trainer (research/next-round/junction-eyes-plan.md 4.1-4.4): train Cinque's own vision trunk on the
map-labelled rulers R-A (route arc-length points) and R-B (near BEV grid), with the keep term on comma1M. No plan loss; the
policy and every non-vision weight stay at the factory values. Model: experiments/junction_eyes/lib/je_model.py.

  train  --arm L-F0 --seed 0 [--steps N] [--data data.json | --synthetic] [--tag JE-L-F0-s0] [--resume]
  ident  [--n 512] [--data ... | --synthetic]          identity gate: LoRA B = 0 + untouched depthwise kernels vs the frozen trunk

Arms (4.1):  L-F0 / L-F0-nav / U-F0 / L-W90 / L-3V; batch 64 = 24 J_nav + 24 J_on + 16 keep (nav-only arms: 48 J_nav + 16 keep).
Loss:        Huber(delta 1) on standardised R-A (missing masked) + cross-entropy on R-B (-1 ignored) + lam_keep x keep, where
             keep = vt.py's lam_d distillation (lam_d 30; squared error of the factory teacher's distilled columns over their std,
             mean over columns, the plan columns at the same weight as the others) of the frozen factory policy on 9 context
             pairs (8 through the current trunk without grad, the t0 pair with grad) against the 20 Hz-stepped teacher.
Schedule:    AdamW, LoRA 3e-4 wd 0, depthwise 1e-5 wd 0 (U: all vision 1e-5 wd 0), heads 3e-4 wd 0.05; warmup 300, cosine to the
             last step; bf16 compute; gradient clip 1.0 per group (vision / heads); every --eval-every steps the R-A / R-B loss on val
             selects the best checkpoint (sum of the two); navtest and the read sets play no part.
Outputs:     $DATA_DIR/runs/junction_eyes/step1/ckpt/<tag>/{best.pt, last.pt, resume.pt, evals.json}; the run dir (Run) beside it.

--data JSON: {"nav": {"px": npy, "rows": npz}, "on": {...}, "keep": dir, "splits": [registered split names]} (rows schema:
results/trainer.md "Inputs"); --synthetic: plumbing only (real navtrain image pairs from the VT side cache, random labels,
the mini keep store of je_keep_mini.py): smoke and gates, never read.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "experiments/junction_eyes/lib")]
import argparse, json, math, os, time  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import je_data as D  # noqa: E402
import je_model as M  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ARMS = {"L-F0": dict(vis="lora", src=("nav", "on")), "L-F0-nav": dict(vis="lora", src=("nav",)),
        "U-F0": dict(vis="full", src=("nav", "on")), "L-W90": dict(vis="lora", src=("nav",), px="w90"),
        "L-3V": dict(vis="lora", src=("nav",), views=3)}
STEP1 = data_dir() / "runs" / "junction_eyes" / "step1"
CKPT = STEP1 / "ckpt"
SMOKE_PX = data_dir() / "runs" / "vis_train" / "px_side"          # (N, 2 views, 2, 2, 6, 128, 256): navtrain L0 / R0 t0 pairs
SMOKE_KEEP = STEP1 / "keep_mini"
LR = dict(lora=3e-4, dw=1e-5, full=1e-5, head=3e-4)
WD = dict(lora=0.0, dw=0.0, full=0.0, head=0.05)


# ---------------------------------------------------------------- data
STEP0 = data_dir() / "runs" / "junction_eyes" / "step0"
PX = data_dir() / "runs" / "junction_eyes" / "px"
CACHE = data_dir() / "runs" / "op_parity" / "cache"
DEFAULT_DATA = {"nav": {"rows": str(STEP0 / "rows" / "J_nav.parquet"), "px_root": str(PX)},
                "on": {"rows": str(STEP0 / "rows" / "J_on.parquet"), "px_root": str(PX), "per_unit": True},
                "labels": str(STEP0 / "labels"), "keep": str(data_dir() / "runs" / "junction_eyes" / "keep" / "store"), "splits": []}


def sources(a, arm: dict):
    names = ("nav",) if a.nav_only else arm["src"]
    if a.synthetic:
        n = a.synthetic_rows
        src = {"nav": D.synthetic_source("nav", SMOKE_PX / "navtrain_full.s0of12" / "side_t0.npy", 0, n, a.seed),
               "on": D.synthetic_source("on", SMOKE_PX / "navtrain_full.s1of12" / "side_t0.npy", 1, n, a.seed)}
        src["on"].w = D.per_unit_weight(src["on"].unit, src["on"].fit)
        return {k: v for k, v in src.items() if k in names}, D.KeepStore(a.keep or SMOKE_KEEP), []
    spec = DEFAULT_DATA | (json.loads(_pl.Path(a.data).read_text()) if a.data else {})
    src = {k: D.real_source(k, spec[k]["rows"], spec["labels"], spec[k]["px_root"], CACHE, spec[k].get("px_file", "frames-t0.npy"),
                            spec[k].get("per_unit", False)) for k in names}
    return src, D.KeepStore(a.keep or spec["keep"]), spec.get("splits", [])


# ---------------------------------------------------------------- losses
def head_losses(m: M.JE, tok, ego, ra, rb):
    """-> R-A Huber on standardised targets (missing masked), R-B cross-entropy (-1 ignored); fp32 scalars."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pa, pb = m.heads(tok, ego)
    y = (ra - m.ramu) / m.rasd
    ok = torch.isfinite(y)
    la = (F.huber_loss(pa.float(), torch.nan_to_num(y), delta=1.0, reduction="none") * ok).sum() / ok.sum().clamp_min(1)
    lb = F.cross_entropy(pb.float().reshape(-1, 3), rb.long().reshape(-1), ignore_index=-1)
    return la, lb


class Keep:
    """The keep term: lam_d x mean over rows of mean over the distilled columns of ((student - teacher) / std)^2 (logged weighted;
    keep_plan = the unweighted mean over the plan columns)."""

    def __init__(self, m: M.JE, KS: D.KeepStore, rows, dev, lam=30.0):
        di, self.plan = M.distill_cols(m.net)
        self.di = torch.as_tensor(di, device=dev)
        t = torch.as_tensor(KS.teacher[np.asarray(rows)][:, di], device=dev)
        self.std = t.std(0).clamp_min(1e-3)
        self.lam, self.KS, self.dev = lam, KS, dev

    def __call__(self, out, rows):
        t = torch.as_tensor(self.KS.teacher[np.asarray(rows)], device=self.dev)[:, self.di]
        e = ((out.float()[:, self.di] - t) / self.std).pow(2)
        return self.lam * e.mean(1).mean(), e[:, torch.as_tensor(self.plan, device=self.dev)].mean()     # weighted term, raw plan-column error


# ---------------------------------------------------------------- the lane
class Lane:
    def __init__(self, a, dev):
        self.a, self.dev, self.arm = a, dev, ARMS[a.arm]
        torch.manual_seed(a.seed)
        self.src, self.KS, self.splits = sources(a, self.arm)
        self.m = m = M.JE(self.arm["vis"], self.arm.get("views", 1)).to(dev)
        m.compiled = a.compile
        # statistics of the fit rows (P): ego standardisation, R-A target mean / std
        fit = [(s, s.fit_rows()) for s in self.src.values()]
        e = np.concatenate([s.ego[r] for s, r in fit])
        y = np.concatenate([s.ra[r] for s, r in fit])
        with torch.no_grad():
            m.emu.copy_(torch.as_tensor(e.mean(0))), m.esd.copy_(torch.as_tensor(e.std(0) + 1e-6))
            m.ramu.copy_(torch.as_tensor(np.nanmean(y, 0))), m.rasd.copy_(torch.as_tensor(np.nanstd(y, 0) + 1e-6))
        self.keep_fit = np.flatnonzero(~self.KS.held)
        self.keep = Keep(m, self.KS, self.keep_fit, dev, a.lam_keep)
        rng = np.random.default_rng([a.seed, 7])
        self.keep_mon = np.sort(rng.choice(self.keep_fit, min(256, len(self.keep_fit)), replace=False))
        g = m.groups()
        dw = g["vis"] if self.arm["vis"] == "lora" else []
        full = g["vis"] if self.arm["vis"] == "full" else []
        pg = [dict(params=g["lora"], name="lora"), dict(params=dw, name="dw"), dict(params=full, name="full"), dict(params=g["head"], name="head")]
        pg = [dict(p, lr=LR[p["name"]], base=LR[p["name"]], weight_decay=WD[p["name"]]) for p in pg if p["params"]]
        self.opt = torch.optim.AdamW(pg, betas=(0.9, 0.999), fused=True)
        self.clip = {"vision": g["lora"] + g["vis"], "head": g["head"]}
        # row streams (4.2): one stream per source and one for keep rows, seeded by the seed only (same rows for every arm of a seed)
        self.rng = {k: np.random.default_rng([a.seed, i + 1]) for i, k in enumerate(("nav", "on", "keep"))}
        self.nb = {"nav": 24, "on": 24} if "on" in self.src else {"nav": 48}
        self.nkeep = 16
        self.fitp = {k: s.fit_p() for k, s in self.src.items()}
        n_fit = sum(len(s.fit_rows()) for s in self.src.values())
        self.steps = a.steps or int(math.ceil(a.epochs * n_fit / sum(self.nb.values())))

    def draw(self):
        d = {k: self.rng[k].choice(r, self.nb[k], p=p) for k, (r, p) in self.fitp.items()}
        d["keep"] = self.rng["keep"].choice(self.keep_fit, self.nkeep)
        return d

    def fetch(self, d):
        b = {}
        for k in self.src:
            s, r = self.src[k], d[k]
            b[k] = (*s.px.get(r), s.ego[r], s.ra[r], s.rb[r])
        b["keep"] = self.KS.get(d["keep"])
        return d, b

    def upload(self, b):
        dv = lambda x: (x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))).to(self.dev, non_blocking=True)  # noqa: E731
        return {k: tuple(dv(x) for x in v) for k, v in b.items()}

    def forward(self, d, b, grad=True):
        """-> dict of loss terms (R-A, R-B, keep, keep_plan). One grad pass over the J t0 pairs and the keep t0 pairs together."""
        m, ks = self.m, [k for k in self.src]
        kp, kc, kv = b["keep"]
        B = kp.shape[0]
        with torch.no_grad():
            hc = m.encode(kp[:, :-1].flatten(0, 1), kc[:, :-1].flatten(0, 1)).reshape(B, M.A.CONTEXT - 1, 32, 512)
        prev = torch.cat([b[k][0] for k in ks] + [kp[:, -1]])
        cur = torch.cat([b[k][1] for k in ks] + [kc[:, -1]])
        with torch.set_grad_enabled(grad):
            tok = m.encode(prev, cur, grad=grad)
            nj = tok.shape[0] - B
            H = torch.cat([hc, tok[nj:, None]], 1)
            out = m.policy(H, kv, self.KS.action_t)
            L = {}
            L["keep"], L["keep_plan"] = self.keep(out, d["keep"])
            ego, ra, rb = (torch.cat([b[k][i] for k in ks]) for i in (2, 3, 4))
            L["ra"], L["rb"] = head_losses(m, tok[:nj], ego, ra, rb)
        return L

    def total(self, L):
        return L["ra"] + L["rb"] + L["keep"]

    def step(self, item, step, gn_terms=False):
        d, b = item
        b = self.upload(b)
        for g in self.opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (step + 1) / self.a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / self.steps)))
        L = self.forward(d, b)
        tot = self.total(L)
        if not torch.isfinite(tot):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in L.items()} }")
        out = {}
        self.opt.zero_grad(set_to_none=True)
        if gn_terms:                                   # gradient norm of each term on the vision weights (4.3 smoke check)
            vp = [p for p in self.clip["vision"]]
            for name, term in (("heads", L["ra"] + L["rb"]), ("keep", L["keep"])):
                gs = torch.autograd.grad(term, vp, retain_graph=True, allow_unused=True)
                out[f"gn_{name}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        tot.backward()
        for k, ps in self.clip.items():
            out[f"gn_{k}"] = float(torch.nn.utils.clip_grad_norm_(ps, 1.0))
        self.opt.step()
        return {k: float(v.detach()) for k, v in L.items()} | {"total": float(tot.detach())} | out

    @torch.no_grad()
    def evaluate(self, bs=64) -> dict:
        m, res = self.m.eval(), {}
        for k, s in self.src.items():
            r = np.flatnonzero(s.val)
            num = {"ra": 0.0, "rb": 0.0}
            for i in range(0, len(r), bs):
                rr = r[i:i + bs]
                p, c = (x.to(self.dev) for x in s.px.get(rr))
                tok = m.encode(p, c)
                la, lb = head_losses(m, tok, *(torch.from_numpy(np.ascontiguousarray(x)).to(self.dev) for x in (s.ego[rr], s.ra[rr], s.rb[rr])))
                num["ra"] += float(la) * len(rr)
                num["rb"] += float(lb) * len(rr)
            res |= {f"val_{k}_{n}": v / max(len(r), 1) for n, v in num.items()}
        nv = {k: int(s.val.sum()) for k, s in self.src.items()}
        for n in ("ra", "rb"):
            res[f"val_{n}"] = sum(res[f"val_{k}_{n}"] * nv[k] for k in self.src) / max(sum(nv.values()), 1)
        res["val_sel"] = res["val_ra"] + res["val_rb"]
        kk = []
        for i in range(0, len(self.keep_mon), 32):
            rr = self.keep_mon[i:i + 32]
            kp, kc, kv = (x.to(self.dev) for x in self.KS.get(rr))
            B = len(rr)
            H = m.encode(kp.flatten(0, 1), kc.flatten(0, 1)).reshape(B, M.A.CONTEXT, 32, 512)
            kk.append(self.keep(m.policy(H, kv, self.KS.action_t), rr))
        res["mon_keep"] = float(np.mean([float(a) for a, _ in kk]))
        res["mon_keep_plan"] = float(np.mean([float(b) for _, b in kk]))
        m.train()
        return res


def _save(obj, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, f)


def cmd_train(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    if a.compile and a.gn_every:                      # per-term norms retain the graph; compiled backward donates buffers otherwise
        import torch._functorch.config as fc
        fc.donated_buffer = False
    tag = a.tag or f"JE-{a.arm}-s{a.seed}"
    d = CKPT / tag
    if d.exists() and not a.resume and not (a.scratch and tag.startswith("smoke")):
        raise SystemExit(f"{d} exists; --resume continues it")
    if a.scratch and tag.startswith("smoke"):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    with Run("junction_eyes", f"step1/train-{tag}", seed=a.seed, config=vars(a) | {"tag": tag}) as run:
        ln = Lane(a, dev)
        for s in ln.splits:
            run.use_split(splits.load(s))
        m = ln.m
        run.info(f"{tag}: arm {a.arm} ({ARMS[a.arm]}), {ln.steps} steps x batch {sum(ln.nb.values())} + {ln.nkeep} keep; rows "
                 + ", ".join(f"{k} fit {len(s.fit_rows())} val {int(s.val.sum())}" for k, s in ln.src.items())
                 + f"; keep {len(ln.keep_fit)} (held {int(ln.KS.held.sum())}); trainable " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in m.n_params().items()))
        step0, best, evals = 0, math.inf, []
        rs = d / "resume.pt"
        if a.resume and rs.exists():
            st = torch.load(rs, map_location="cpu", weights_only=False)
            m.load_trained(st["model"])
            ln.opt.load_state_dict(st["opt"])
            step0, best, evals = st["step"], st["best"], st["evals"]
            for _ in range(step0):
                ln.draw()
            run.info(f"resumed at step {step0}")
        d.mkdir(parents=True, exist_ok=True)

        def ck(step):
            return {"model": m.trained_state(), "cfg": vars(a) | {"tag": tag, "steps_total": ln.steps}, "step": step, "tag": tag}

        def evaluate(step):
            nonlocal best
            ev = ln.evaluate() | {"step": step}
            evals.append(ev)
            (d / "evals.json").write_text(json.dumps(evals, indent=1))
            run.scalars({f"val/{k}": v for k, v in ev.items() if k != "step"}, step)
            run.info(f"eval @ {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in ev.items() if k != "step"))
            if ev["val_sel"] < best:
                best = ev["val_sel"]
                _save(ck(step) | {"val": ev}, d / "best.pt")
                run.summary.update(best_step=step, best_val_sel=best)

        if step0 == 0:
            evaluate(0)
        pre = D.Prefetch(ln.draw, ln.fetch, ln.steps - step0, a.prefetch, a.fetch_workers)
        t0, tw, hist = time.time(), time.time(), []
        for step in range(step0, ln.steps):
            n = step + 1
            hist.append(ln.step(pre.get(), step, gn_terms=bool(a.gn_every) and (step % a.gn_every == 0)))
            if n % 25 == 0 or n == ln.steps:
                mean = {x: float(np.nanmean([h.get(x, np.nan) for h in hist])) for x in set().union(*hist)}
                its = len(hist) / (time.time() - tw)
                hist, tw = [], time.time()
                run.scalars({f"loss/{x}": v for x, v in mean.items()} | {"throughput/steps_per_s": its, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, n)
                if n % 100 == 0 or n == ln.steps:
                    run.info(f"step {n}: " + ", ".join(f"{x} {v:.4g}" for x, v in sorted(mean.items())) +
                             f"; {its:.2f} it/s ({(n - step0) / (time.time() - t0):.2f} since start), {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {n}/{ln.steps}, {its:.2f} it/s, eta {(ln.steps - n) / max(its, 1e-6) / 3600:.2f} h")
            if n % a.eval_every == 0 or n == ln.steps:
                evaluate(n)
                _save(ck(n) | {"opt": ln.opt.state_dict(), "best": best, "evals": evals}, rs)
                tw = time.time()
        _save(ck(ln.steps), d / "last.pt")
        rs.unlink(missing_ok=True)
        run.summary.update(steps=ln.steps, train_s=time.time() - t0, it_s=(ln.steps - step0) / (time.time() - t0),
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30, ckpt=str(d))


# ---------------------------------------------------------------- identity gate (4.4 / 5.2)
@torch.no_grad()
def cmd_ident(a):
    """The arm's trunk at its start (LoRA B = 0, depthwise / full masters = factory) against the frozen trunk on n pairs, in
    bf16 (training) and fp32: bitwise-equal share and max |difference| of view_39. Plus: B != 0 moves the tokens (control)."""
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("junction_eyes", "step1/ident", seed=0, config=vars(a)) as run:
        if a.synthetic:
            px = D.PairStore(SMOKE_PX / "navtrain_full.s0of12" / "side_t0.npy", 0)
        else:
            px = D.PairStore(PX / "navtrain_full.s0of12" / "frames-t0.npy")
        rows = np.arange(a.n)
        p, c = (x.to(dev) for x in px.get(rows))
        res = {"n": a.n}
        enc = lambda mm, grad=False: torch.cat([mm.encode(p[i:i + 64], c[i:i + 64], grad=grad).detach() for i in range(0, a.n, 64)]).float()  # noqa: E731
        h32 = None
        for dt in (torch.float32, torch.bfloat16):
            ref = M.JE("none", dtype=dt, heads=False).to(dev)
            h0 = enc(ref)
            del ref
            dn = str(dt)[6:]
            if h32 is None:
                h32 = h0
            res[f"frozen_{dn}_vs_fp32"] = dict(max_abs=float((h0 - h32).abs().max()), mean_abs=float((h0 - h32).abs().mean()))
            for vis in ("lora", "full"):
                m = M.JE(vis, dtype=dt, heads=False).to(dev)
                for grad in (False, True):            # the no-grad chunks and the training (grad) pass
                    with torch.set_grad_enabled(grad):
                        h = enc(m, grad)
                    k = f"{vis}_{dn}_{'grad' if grad else 'nograd'}"
                    res[k] = dict(bitwise_equal_share=float((h == h0).float().mean()), max_abs=float((h - h0).abs().max()),
                                  ref_rms=float(h0.pow(2).mean().sqrt()), vs_fp32_mean_abs=float((h - h32).abs().mean()),
                                  vs_fp32_max_abs=float((h - h32).abs().max()))
                    run.info(f"{k}: {res[k]}")
                if vis == "lora":                     # control: a small nonzero B must move the tokens
                    for kk, v in m.net.lora.items():
                        if kk.endswith("__B"):
                            v.normal_(0, 1e-3)
                    h = enc(m)
                    res[f"control_B1e-3_{dn}"] = dict(max_abs=float((h - h0).abs().max()), mean_abs=float((h - h0).abs().mean()))
                del m
                torch.cuda.empty_cache()
        # LoRA (B = 0, depthwise masters) must be bitwise the frozen trunk in both dtypes; U (all masters fp32) bitwise in fp32 and, in
        # bf16, no further from the fp32 trunk than the frozen bf16 trunk is (its fp32 LayerNorm masters skip one bf16 rounding)
        res["pass"] = bool(all(res[f"lora_{d}_{g}"]["bitwise_equal_share"] == 1.0 for d in ("float32", "bfloat16") for g in ("grad", "nograd"))
                           and all(res[f"full_float32_{g}"]["bitwise_equal_share"] == 1.0 for g in ("grad", "nograd"))
                           and all(res[f"full_bfloat16_{g}"]["vs_fp32_mean_abs"] <= 1.1 * res["frozen_bfloat16_vs_fp32"]["mean_abs"] for g in ("grad", "nograd")))
        out = STEP1 / "gates" / "ident.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        run.info(json.dumps(res))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "ident"):
        p = sp.add_parser(name)
        p.add_argument("--data", default="", help="data JSON (see docstring)")
        p.add_argument("--synthetic", action="store_true", help="plumbing only: real navtrain pairs, random labels; never read")
        p.add_argument("--synthetic-rows", type=int, default=4000)
        p.add_argument("--nav-only", action="store_true", help="smoke tags only: J_nav rows alone (48 / step) whatever the arm")
        p.add_argument("--keep", default="", help="keep store dir (default: the mini store with --synthetic)")
        if name == "ident":
            p.add_argument("--n", type=int, default=512)
            continue
        cli_args(p)
        p.add_argument("--arm", required=True, choices=list(ARMS))
        p.add_argument("--tag", default="")
        p.add_argument("--steps", type=int, default=0, help="0: --epochs over the J fit rows at 48 J rows per step")
        p.add_argument("--epochs", type=float, default=12.0)
        p.add_argument("--warmup", type=int, default=300)
        p.add_argument("--eval-every", type=int, default=1000)
        p.add_argument("--lam-keep", type=float, default=30.0, help="lam_d of the keep term (vt.py convention)")
        p.add_argument("--gn-every", type=int, default=0, help="per-term gradient norms every N steps (0 = off; costs two extra backward passes)")
        p.add_argument("--compile", action="store_true", help="torch.compile of the trunk's grad pass")
        p.add_argument("--prefetch", type=int, default=4)
        p.add_argument("--fetch-workers", type=int, default=3)
        p.add_argument("--scratch", action="store_true", help="smoke* tags only: delete the tag's checkpoint dir first")
    a = ap.parse_args()
    if a.cmd == "train":
        assert not (a.nav_only and not (a.tag or "").startswith("smoke")), "--nav-only is a smoke option"
        assert not (a.synthetic and not (a.tag or "").startswith("smoke")), "--synthetic runs are smoke runs: --tag smoke-..."
    {"train": cmd_train, "ident": cmd_ident}[a.cmd](a)
