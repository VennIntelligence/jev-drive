"""Image-command fine-tune (Q2): light fine-tune of Cinque on overlay rows (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft-prereg.md. Port, trainable set, loss scales and schedule are L3's (op_adapt_h.h_train) and
op_adapt_l's, used by import; the batches come from this lane's trunk bank (img_ft_bank.py, pool `train`).

Rows of one batch (48):
  I  24   overlay of a trained family (band / barrier) for command c on a junction frame; target = the route of branch c: for the
          taken class the logged future, else a path along c's centreline (approach + branch) driven with the logged arc length
          vs time (the logged speed profile), the t0 lateral offset to the centreline fading out over max(8 m, 2.5 s x speed)
  D  16   no overlay (10 junction, 6 lane-keeping frames): distilled to the original (plan consistency + every output head), so the
          plan without a command stays the original's and does not learn to guess a branch
  C   8   band along the own lane on lane-keeping frames: plan consistency to the original's `none` plan (paint that agrees with
          the lane changes nothing)

  teacher   the original on every `none` variant of the train pool -> ft/teacher/train.npz (per sample: distilled heads, plan mu)
  train     --arm <name> -> ft/runs/<arm>-s<seed>/ (jevdrive.run.Run; ckpt-final.pt in op_adapt_l's format)

  CUDA_VISIBLE_DEVICES=1 taskset -c 100-149 $DATA_DIR/envs/op-train/bin/python experiments/op_img_cmd/scripts/img_ft_train.py train --arm ft
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
_R = _pl.Path(_os.environ.get("JEV_REPO", _pl.Path(__file__).resolve().parents[3]))
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_R / "experiments" / "op_adapt_h" / "scripts"), str(_pl.Path(__file__).resolve().parent)]
import argparse, json, pickle, time  # noqa: E401,E402
from dataclasses import asdict, dataclass, field, replace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import img_overlay as O  # noqa: E402
import img_ft_bank as FB  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.run import Run  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402

FT = FB.FT
ROLES = {"I": 1, "D": 2, "C": 3}


@dataclass
class FCfg:
    name: str
    seed: int = 0
    steps: int = 2000
    n_i: int = 24
    n_dj: int = 10
    n_ds: int = 6
    n_c: int = 8
    fams: tuple = FB.TRAIN_FAMS
    s4: bool = True
    pol: bool = True
    lr: float = 3e-5
    wd: float = 0.01
    warmup: int = 100
    lam_i: float = 1.0
    lam_d: float = 10.0
    lam_c: float = 1.0
    fade_t: float = 2.5
    fade_min_m: float = 8.0
    ckpt_every: int = 500
    workers: int = 8

    def dump(self):
        return asdict(self)

    def lcfg(self) -> L.LCfg:
        return L.LCfg(name=self.name, seed=self.seed, s4=self.s4, pol=self.pol, intent="none", steps=self.steps)


ARMS = {"smoke": dict(steps=40, ckpt_every=10 ** 9), "ft": dict()}


# ---------------------------------------------------------------- targets
def route_target(s, cmd, fut20, v0, fade_t=2.5, fade_min=8.0):
    """(20, 2) target on the 0.25 s grid in the t0 rear-axle frame for command `cmd` (see the module doc)."""
    if cmd == s["taken"]:
        return np.asarray(fut20, np.float32)
    Q = O.full_centre(s, O.cmd_path(s, cmd))
    t = Q[-1] - Q[-2]
    Q = np.r_[Q, Q[-1] + np.outer(0.5 * np.arange(1, 401), t / np.linalg.norm(t))]       # straight extension past the geometry
    a, nrm = O.arc(Q), O.normals(Q)
    k = int(np.argmin(np.linalg.norm(Q, axis=1)))
    e0 = float(np.dot(-Q[k], nrm[k]))
    P = np.r_[np.zeros((1, 2)), np.asarray(fut20, float)]
    sl = np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    S = max(fade_min, fade_t * max(v0, sl[-1] / 5.0))
    u = np.clip(sl / S, 0, 1)
    e = e0 * (2 * u ** 3 - 3 * u ** 2 + 1)
    q = a[k] + sl
    idx = np.clip(np.searchsorted(a, q), 0, len(Q) - 1)
    return (O.at(Q, q) + e[:, None] * nrm[idx]).astype(np.float32)


class Pool:
    """The train bank + per-variant metadata, targets and teacher arrays."""

    def __init__(self, cfg: "FCfg"):
        import h_prep
        self.G = FB.samples("train")
        self.T = np.load(FT / "bank" / "train" / "trunk.npy", mmap_mode="r")
        with np.load(FT / "bank" / "train" / "var.npz") as z:
            self.v = {k: z[k] for k in z.files}
        G, v = self.G, self.v
        self.fut = h_prep.nav_future([s["token"] for s in G])
        v0 = np.array([s["v"] for s in G])
        sp = np.array([s["split"] for s in G])[v["sample"]]
        kind = np.array([s["kind"] for s in G])[v["sample"]]
        self.hum = np.full((len(v["sample"]), 16, 3), np.nan, np.float32)
        tr = sp == "train"
        self.idx = {"I": np.flatnonzero(tr & (kind == "junction") & np.isin(v["fam"], cfg.fams)),
                    "DJ": np.flatnonzero(tr & (kind == "junction") & (v["fam"] == "none")),
                    "DS": np.flatnonzero(tr & (kind == "straight") & (v["fam"] == "none")),
                    "C": np.flatnonzero(tr & (kind == "straight") & (v["fam"] == "band"))}
        tgt = {}
        for j in self.idx["I"]:
            i, c = int(v["sample"][j]), str(v["cmd"][j])
            if (i, c) not in tgt:
                tgt[(i, c)] = L.human_targets(route_target(G[i], c, self.fut[i], v0[i], cfg.fade_t, cfg.fade_min_m)[None])[0]
            self.hum[j] = tgt[(i, c)]
        tea = FT / "teacher" / "train.npz"
        if tea.exists():
            z = np.load(tea)
            self.tout, self.tmu = z["out"], z["mu"]

    def row(self, j, role):
        i = int(self.v["sample"][j])
        return {"trunk": np.asarray(self.T[j]), "valid": self.v["slot_valid"], "tc": self.v["tc"][j], "role": ROLES[role],
                "hum": self.hum[j], "cam": np.float32(self.v["cam"][j][0]), "tgt": self.tout[i], "tmu": self.tmu[i]}


class Batcher(torch.utils.data.Dataset):
    """Item k = one whole batch drawn with rng (seed, k)."""

    def __init__(self, cfg: FCfg, n=10 ** 9):
        self.cfg, self.n, self.P = cfg, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if self.P is None:
            self.P = Pool(self.cfg)
        c, P = self.cfg, self.P
        rng = np.random.default_rng([c.seed, k])
        rows = []
        for key, role, n in (("I", "I", c.n_i), ("DJ", "D", c.n_dj), ("DS", "D", c.n_ds), ("C", "C", c.n_c)):
            for j in P.idx[key][rng.integers(len(P.idx[key]), size=n)]:
                rows.append(P.row(int(j), role))
        return {k_: torch.from_numpy(np.stack([np.asarray(r[k_]) for r in rows])) for k_ in rows[0]}


class Losses:
    """I rows: imitation of the route target (op_adapt_l's rear-axle 16-point Huber distance); D and C rows: plan consistency to
    the original's `none` plan; D rows also the normalised distillation of every output head (r2 / op_adapt_l scale)."""

    def __init__(self, net, cfg: FCfg, tstd, dev):
        self.cfg, self.base = cfg, L.Losses(net, cfg.lcfg(), tstd, dev)

    def __call__(self, o, b):
        c, base = self.cfg, self.base
        out = o["outputs"].float()
        plan = base.plan(out)
        role = b["role"]
        Ls, tot = {}, 0.0
        m = role == 1
        Ls["imit"] = base.imit_dist(plan[m], None, b["cam"][m], b["hum"][m]).mean()
        tot = tot + c.lam_i * Ls["imit"]
        m = role >= 2
        cons = base.imit_dist(plan[m], b["tmu"][m].float(), b["cam"][m])
        Ls["cons"] = cons.mean()
        Ls["cons_C"] = cons[role[m] == 3].mean().detach()
        tot = tot + c.lam_c * Ls["cons"]
        m = role == 2
        e = ((out[m][:, base.di] - b["tgt"][m].float()) / base.tstd).pow(2)
        Ls["distill"] = e.mean()
        tot = tot + c.lam_d * Ls["distill"]
        return tot, Ls


# ---------------------------------------------------------------- teacher
def cmd_teacher(a):
    dev = torch.device("cuda")
    m = L.load_model(None, dev)
    didx, pi = A.distill_index(m.net.slices), A.plan_index(m.net.slices)
    P = Pool(FCfg("teacher"))
    v = P.v
    nn = np.flatnonzero(v["fam"] == "none")
    assert (v["sample"][nn] == np.arange(len(P.G))).all()
    out = np.zeros((len(nn), len(didx)), np.float16)
    mu = np.zeros((len(nn), 33, 15), np.float32)
    sv = torch.from_numpy(v["slot_valid"]).to(dev)
    with torch.no_grad():
        for i in range(0, len(nn), 64):
            j = nn[i:i + 64]
            tr = torch.from_numpy(np.stack([P.T[x] for x in j])).to(dev)
            o = m(tr, sv[None].expand(len(j), -1), torch.from_numpy(v["tc"][j]).to(dev).half())["outputs"].float()
            out[i:i + len(j)] = o[:, didx].cpu().numpy()
            mu[i:i + len(j)] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
    (FT / "teacher").mkdir(parents=True, exist_ok=True)
    np.savez(FT / "teacher" / "train.npz", out=out, mu=mu)
    # sanity: nav_future vs the geometry's logged future (8 x 0.5 s), and the route targets of the taken class
    g8 = np.stack([np.asarray(s["future"])[:8, :2] for s in P.G if len(s["future"]) >= 8])
    f8 = np.stack([P.fut[i][1::2][:8] for i, s in enumerate(P.G) if len(s["future"]) >= 8])
    print(f"teacher: {len(nn)} samples; |nav_future - geom future| max {np.abs(g8 - f8).max():.3f} m")
    d = []
    for i, s in enumerate(P.G):
        if s["kind"] == "junction":
            c = s["taken"]
            q = route_target(dict(s, taken="__none__"), c, P.fut[i], s["v"])
            d.append(np.linalg.norm(q[15] - P.fut[i][15]))
    print(f"centreline route of the taken class vs the logged future at 4 s: median {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m")


# ---------------------------------------------------------------- train (h_train.train's loop on this lane's batches)
def cmd_train(a):
    cfg = replace(FCfg(name=a.arm, seed=a.seed, **ARMS.get(a.arm, {})), **json.loads(a.override or "{}"))
    if a.steps:
        cfg = replace(cfg, steps=a.steps)
    tag = f"{a.arm}-s{a.seed}"
    d = FT / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    from jevdrive.data import splits
    with Run("op_img_cmd", f"ft-{tag}", resume=d, seed=cfg.seed, config=cfg.dump()) as run:
        run.use_split(splits.load("navsim/img-ft-train"))
        dev = torch.device("cuda")
        model = L.LModel(cfg.lcfg()).to(dev)
        base, _ = model.trainable()
        opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr}], weight_decay=cfg.wd)
        scaler = torch.amp.GradScaler()
        lossf = Losses(model.net, cfg, np.load(L.r2t() / "teacher" / "tstd.npy"), dev)
        step, ck = 0, d / "ckpt.pt"
        if ck.exists() and not a.fresh:
            st = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state(st["model"]), opt.load_state_dict(st["opt"]), scaler.load_state_dict(st["scaler"])
            step = st["step"]
            run.info(f"resumed at step {step}")
        P = Pool(cfg)
        run.info(f"{cfg.name}: {cfg.steps} steps; pools " + json.dumps({k: int(len(v)) for k, v in P.idx.items()}) +
                 f"; trainable {sum(p.numel() for p in base) / 1e6:.1f} M")
        del P
        dl = torch.utils.data.DataLoader(Batcher(cfg), batch_size=None, sampler=range(step, 10 ** 9), num_workers=cfg.workers,
                                         prefetch_factor=4, pin_memory=True)
        it = iter(dl)
        model.train()
        hist, wait, t0, s0 = [], 0.0, time.time(), step
        torch.cuda.reset_peak_memory_stats()
        bar = run.tqdm(total=cfg.steps, initial=step, desc=cfg.name)
        while step < cfg.steps:
            tw = time.time()
            b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
            wait += time.time() - tw
            o = model(b["trunk"], b["valid"], b["tc"].half())
            total, Ls = lossf(o, b)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in Ls.items()} }")
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base, 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            bar.update()
            hist.append({k: float(v) for k, v in Ls.items()} | {"total": float(total)})
            if step % 25 == 0 or step == cfg.steps:
                m = pd.DataFrame(hist[-25:]).mean()
                el = time.time() - t0
                run.scalars({f"loss/{k}": v for k, v in m.items()}, step)
                run.scalars({"throughput/steps_per_s": (step - s0) / el, "throughput/data_wait_frac": wait / el,
                             "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step)
                if step % 100 == 0 or step == cfg.steps:
                    run.info(f"step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) +
                             f"; {(step - s0) / el:.2f} it/s, wait {wait / el:.2f}, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
            if step % cfg.ckpt_every == 0 and step < cfg.steps:
                torch.save({"model": model.state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, d / "ckpt.tmp.pt")
                (d / "ckpt.tmp.pt").replace(ck)
        bar.close()
        del it, dl
        torch.save({"model": model.state(), "cfg": cfg.lcfg().dump(), "fcfg": cfg.dump()}, d / "ckpt-final.pt")
        ck.unlink(missing_ok=True)
        run.summary.update(steps=step, train_s=time.time() - t0, data_wait_frac=wait / (time.time() - t0),
                           peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30, final=hist[-1])
    sys_exit()


def sys_exit():
    _sys.stdout.flush()
    _os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("teacher")
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--override", default="", help="JSON dict of FCfg fields (an iteration arm; log it in the plan first)")
    p.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "train": cmd_train}[a.cmd](a)
