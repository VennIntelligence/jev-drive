"""Image-command fine-tune Q3 (drift-free): trainer on the Q2 bank + img2_bank.py's banks (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md. Port, trainable set, loss and schedule as Q2 (img_ft_train.py, L3 / op_adapt_l).

Rows of one batch (56):
  I 24   trained family (band / barrier) for command c on a junction frame (nav 16, CARLA 8). Target = branch c's centreline
         (approach + branch; t0 lateral offset faded out as in Q2) for every class, timed by the ORIGINAL's own `none` plan on that
         frame (its arc length vs time): the overlay may change the route, not the speed.
  D 22   no overlay, distilled to the original (plan consistency + every output head): Q2 pool junction 4 / straight 2, CARLA
         junction 2, L3's pools nav 6 / WOD 4 / CARLA p6 4.
  N 10   negatives, plan consistency to the original's `none` plan of the same frame: band along the own lane on straight frames 4,
         band_all (every branch painted, no route information) on nav 4 / CARLA 2 junction frames.

  teacher            the original on every `none` row of every bank -> ft/teacher2/<bank>.npz (rows, distilled heads, plan mu)
  train --arm <A|B|C> [--seed s] -> ft/runs/q3<arm>-s<seed>/ckpt-final.pt (op_adapt_l format)
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
_R = _pl.Path(_os.environ.get("JEV_REPO", _pl.Path(__file__).resolve().parents[3]))
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_R / "experiments" / "op_adapt_h" / "scripts"), str(_pl.Path(__file__).resolve().parent)]
import argparse, json, time  # noqa: E401,E402
from dataclasses import asdict, dataclass, replace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import img_ft_bank as FB  # noqa: E402
import img_ft_train as FTR  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.run import Run  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402

FT = FB.FT
ROOT = FT.parent
BANKS = ("train", "nba", "dist", "carla")
T_FUT = 0.25 * np.arange(1, 21)
ROLES = {"I": 1, "D": 2, "N": 3}
MIX = (("IN", "I", 16), ("IC", "I", 8),
       ("QJ", "D", 4), ("QS", "D", 2), ("CJ", "D", 2), ("DN", "D", 6), ("DW", "D", 4), ("DC", "D", 4),
       ("NS", "N", 4), ("NB", "N", 4), ("NC", "N", 2))


@dataclass
class QCfg:
    name: str
    seed: int = 0
    steps: int = 3000
    lr: float = 1e-4
    wd: float = 0.01
    warmup: int = 100
    lam_i: float = 1.0
    lam_d: float = 10.0
    lam_c: float = 1.0
    fade_t: float = 2.5
    fade_min_m: float = 8.0
    ckpt_every: int = 500
    workers: int = 8
    s4: bool = True
    pol: bool = True

    def dump(self):
        return asdict(self)

    def lcfg(self) -> L.LCfg:
        return L.LCfg(name=self.name, seed=self.seed, s4=self.s4, pol=self.pol, intent="none", steps=self.steps)


ARMS = {"smoke": dict(steps=40, ckpt_every=10 ** 9), "A": dict(), "B": dict(lam_d=30.0, lam_c=3.0), "C": dict(lam_d=60.0, lam_c=6.0)}


def bank(name):
    d = FT / "bank" / name
    with np.load(d / "var.npz") as z:
        v = {k: z[k] for k in z.files}
    n = len(v["fam"])
    if v["slot_valid"].ndim == 1:
        v["slot_valid"] = np.tile(v["slot_valid"], (n, 1))
    v["cam0"] = np.asarray(v["cam"], np.float32).reshape(n, -1)[:, 0]
    v["token"] = v["token"].astype(str)
    return np.load(d / "trunk.npy", mmap_mode="r"), v


def geom():
    """token -> sample dict (Q2 train pool + CARLA junctions), with 'split'."""
    import img2_bank as QB
    G = {s["token"]: s for s in FB.samples("train")}
    sp = QB.carla_split()
    for s in QB.carla_samples():
        G[s["token"]] = dict(s, split=sp[s["token"]])
    return G


def o_timing(mu, cam):
    """original's plan mu (33, 15) -> (20, 2) rear-axle positions at 0.25 ... 5 s."""
    import img_run
    from img_report import T_IDXS
    xy, _ = img_run.to_rear(mu[:, 0:3], mu[:, 11], cam)
    return np.stack([np.interp(T_FUT, T_IDXS, xy[:, c]) for c in range(2)], -1).astype(np.float32)


class Pool:
    def __init__(self, cfg: QCfg):
        self.B = {b: bank(b) for b in BANKS}
        self.tea = {b: dict(np.load(FT / "teacher2" / f"{b}.npz")) for b in BANKS}
        none_row = {}
        for b in BANKS:
            for k, r in enumerate(self.tea[b]["rows"]):
                none_row[str(self.B[b][1]["token"][r])] = (b, k)
        self.none_row = none_row
        G = geom()
        tr = self.B["train"][1]
        kind = lambda b: np.array([G[t]["kind"] if t in G else "" for t in self.B[b][1]["token"]])  # noqa: E731
        split = lambda b: np.array([G[t]["split"] if t in G else "" for t in self.B[b][1]["token"]])  # noqa: E731
        kt, st = kind("train"), split("train")
        sc = split("carla")
        vd = self.B["dist"][1]
        tf = np.isin(tr["fam"], FB.TRAIN_FAMS)
        cf = np.isin(self.B["carla"][1]["fam"], FB.TRAIN_FAMS)
        cfam = self.B["carla"][1]["fam"]
        self.idx = {"IN": ("train", np.flatnonzero((st == "train") & (kt == "junction") & tf)),
                    "IC": ("carla", np.flatnonzero((sc == "train") & cf)),
                    "QJ": ("train", np.flatnonzero((st == "train") & (kt == "junction") & (tr["fam"] == "none"))),
                    "QS": ("train", np.flatnonzero((st == "train") & (kt == "straight") & (tr["fam"] == "none"))),
                    "CJ": ("carla", np.flatnonzero((sc == "train") & (cfam == "none"))),
                    "DN": ("dist", np.flatnonzero((vd["split"] == "train") & (vd["dom"] == "nav"))),
                    "DW": ("dist", np.flatnonzero((vd["split"] == "train") & (vd["dom"] == "wod"))),
                    "DC": ("dist", np.flatnonzero((vd["split"] == "train") & (vd["dom"] == "carla"))),
                    "NS": ("train", np.flatnonzero((st == "train") & (kt == "straight") & (tr["fam"] == "band"))),
                    "NB": ("nba", np.flatnonzero(np.isin(self.B["nba"][1]["split"], ["train"]))),
                    "NC": ("carla", np.flatnonzero((sc == "train") & (cfam == "band_all")))}
        for k, (b, ix) in self.idx.items():
            assert len(ix), k
        # imitation targets of the I rows (per (token, cmd))
        self.hum = {}
        for key in ("IN", "IC"):
            b, ix = self.idx[key]
            vb = self.B[b][1]
            H = np.full((len(vb["fam"]), 16, 3), np.nan, np.float32)
            cache = {}
            for j in ix:
                t, c = str(vb["token"][j]), str(vb["cmd"][j])
                if (t, c) not in cache:
                    nb, k = none_row[t]
                    o20 = o_timing(self.tea[nb]["mu"][k], np.asarray(self.B[nb][1]["cam"][self.tea[nb]["rows"][k]], float))
                    s = G[t]
                    cache[(t, c)] = L.human_targets(FTR.route_target(dict(s, taken="__none__"), c, o20, s["v"], cfg.fade_t,
                                                                     cfg.fade_min_m)[None])[0]
                H[j] = cache[(t, c)]
            self.hum[b] = H

    def row(self, b, j, role):
        T, v = self.B[b]
        t = str(v["token"][j])
        nb, k = self.none_row[t]
        hum = self.hum[b][j] if role == "I" else np.full((16, 3), np.nan, np.float32)
        return {"trunk": np.asarray(T[j]), "valid": v["slot_valid"][j], "tc": np.asarray(v["tc"][j], np.float32), "role": ROLES[role],
                "hum": hum, "cam": np.float32(v["cam0"][j]), "tgt": self.tea[nb]["out"][k], "tmu": self.tea[nb]["mu"][k]}


class Batcher(torch.utils.data.Dataset):
    def __init__(self, cfg: QCfg, n=10 ** 9):
        self.cfg, self.n, self.P = cfg, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if self.P is None:
            self.P = Pool(self.cfg)
        P = self.P
        rng = np.random.default_rng([self.cfg.seed, k])
        rows = []
        for key, role, n in MIX:
            b, ix = P.idx[key]
            rows += [P.row(b, int(j), role) for j in ix[rng.integers(len(ix), size=n)]]
        return {k_: torch.from_numpy(np.stack([np.asarray(r[k_]) for r in rows])) for k_ in rows[0]}


# ---------------------------------------------------------------- teacher
def cmd_teacher(a):
    dev = torch.device("cuda")
    m = L.load_model(None, dev)
    didx, pi = A.distill_index(m.net.slices), A.plan_index(m.net.slices)
    (FT / "teacher2").mkdir(parents=True, exist_ok=True)
    for b in BANKS:
        p = FT / "teacher2" / f"{b}.npz"
        if p.exists():
            continue
        T, v = bank(b)
        nn = np.flatnonzero(v["fam"] == "none")
        out = np.zeros((len(nn), len(didx)), np.float16)
        mu = np.zeros((len(nn), 33, 15), np.float32)
        with torch.no_grad():
            for i in range(0, len(nn), 64):
                j = nn[i:i + 64]
                tr = torch.from_numpy(np.stack([T[x] for x in j])).to(dev)
                o = m(tr, torch.from_numpy(v["slot_valid"][j]).to(dev), torch.from_numpy(np.asarray(v["tc"][j], np.float32)).to(dev).half())
                o = o["outputs"].float()
                out[i:i + len(j)] = o[:, didx].cpu().numpy()
                mu[i:i + len(j)] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
        np.savez(p, rows=nn, out=out, mu=mu)
        print(f"teacher {b}: {len(nn)} none rows", flush=True)
    # sanity: the Q2 teacher on the same trunks
    q2 = np.load(FT / "teacher" / "train.npz")["mu"]
    t2 = np.load(FT / "teacher2" / "train.npz")["mu"]
    print(f"teacher2 vs Q2 teacher (train bank): max |mu diff| {np.abs(q2 - t2).max():.4f}", flush=True)
    P = Pool(QCfg("teacher"))
    print("pools", {k: (b, len(ix)) for k, (b, ix) in P.idx.items()}, flush=True)
    for b, H in P.hum.items():
        h = H[~np.isnan(H[:, 0, 0])]
        print(f"I targets {b}: n {len(h)}, 4 s point x median {np.median(h[:, 15, 0]):.2f} m, |y| median {np.median(np.abs(h[:, 15, 1])):.2f} m")


# ---------------------------------------------------------------- train
def cmd_train(a):
    cfg = replace(QCfg(name=f"q3{a.arm}", seed=a.seed, **ARMS[a.arm]), **json.loads(a.override or "{}"))
    tag = f"q3{a.arm}-s{a.seed}"
    d = FT / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    from jevdrive.data import splits
    with Run("op_img_cmd", f"ft-{tag}", resume=d, seed=cfg.seed, config=cfg.dump()) as run:
        for nm in ("navsim/img-ft-train", "b2d/img-carla-train", "navsim/op-adapt-h-nav-train", "b2d/op-adapt-h-carla-train", "wod/r2-train"):
            run.use_split(splits.load(nm))
        dev = torch.device("cuda")
        model = L.LModel(cfg.lcfg()).to(dev)
        base, _ = model.trainable()
        opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr}], weight_decay=cfg.wd)
        scaler = torch.amp.GradScaler()
        lossf = FTR.Losses(model.net, cfg, np.load(L.r2t() / "teacher" / "tstd.npy"), dev)
        step, ck = 0, d / "ckpt.pt"
        if ck.exists() and not a.fresh:
            st = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state(st["model"]), opt.load_state_dict(st["opt"]), scaler.load_state_dict(st["scaler"])
            step = st["step"]
            run.info(f"resumed at step {step}")
        P = Pool(cfg)
        run.info(f"{cfg.name}: {cfg.steps} steps; pools " + json.dumps({k: [b, int(len(ix))] for k, (b, ix) in P.idx.items()}) +
                 f"; trainable {sum(p.numel() for p in base) / 1e6:.1f} M")
        del P
        dl = torch.utils.data.DataLoader(Batcher(cfg), batch_size=None, sampler=range(step, 10 ** 9), num_workers=cfg.workers,
                                         prefetch_factor=4, pin_memory=True)
        it = iter(dl)
        model.train()
        hist, wait, t0, s0 = [], 0.0, time.time(), step
        torch.cuda.reset_peak_memory_stats()
        bar = run.tqdm(total=cfg.steps, initial=step, desc=cfg.name, mininterval=60)
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
                if step % 250 == 0 or step == cfg.steps:
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
    _sys.stdout.flush()
    _os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("teacher")
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", default="")
    p.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "train": cmd_train}[a.cmd](a)
