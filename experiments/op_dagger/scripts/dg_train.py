"""op_dagger fine-tune (op-train env, one leased GPU): the op_route_ft recipe without a route adapter (rft.RModel: stage 4 + plan pathway + action
pathway trainable, stage 1-3 frozen; rft.RLoss: imitation + action target in the shipped head's scale on P rows, plan consistency + every head distilled
to shipped on D rows, every other head distilled on all rows, dw 3), on WOD train clips with the images rendered online from the clip frames.

Rows of a batch (48):
  R 20  DAgger: a visited state of a closed-loop rollout (dg_roll collect; its own history = the rollout's re-projected frames), in the validity cap,
        target = the recovery path from its offset (dg_common.recovery_hum), action target from it (rft.act_target)
        arm `st` (control): the SAME states and targets, history = the logged frames re-projected along a scripted drift that reaches the state's offset
        (op_adapt_h's static O pair); only the history differs
  U 12  logged states (steps 0..K), target = the logged future, action target
  D 16  logged states, plan consistency + every head to shipped
The shipped teacher is evaluated online on the same trunks.

  dg_train.py --tag dg1 --rolls shipped [--steps 400]            DAgger iteration 1
  dg_train.py --tag st1 --rolls shipped --static                 control
  dg_train.py --tag dg2 --rolls shipped,dg1                      iteration 2 (aggregated states, from shipped)
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_common as C  # noqa: E402

ROWS = {"R": 20, "U": 12, "D": 16}


def states_of(rolls: list[str], S: C.Clips):
    """Valid visited states of every collect file: (clip, step j, offsets (K + 1, 3) of the rollout)."""
    out = []
    for r in rolls:
        for f in sorted(C.root("roll", r).glob("train-collect-*.npz")):
            z = np.load(f, allow_pickle=True)
            for c, off, ok in zip(z["c"], z["off"], z["ok"]):
                for j in range(1, C.K + 1):
                    if ok[j]:
                        out.append((int(c), j, off))
    return out


class Batcher(torch.utils.data.Dataset):
    def __init__(self, a, n=10 ** 9):
        self.a, self.n, self.S = a, n, None

    def __len__(self):
        return self.n

    def _open(self):
        self.S = C.Clips("train")
        self.st = states_of(self.a.rolls.split(","), self.S)

    def row(self, c, j, off_hist, hum, role):
        S = self.S
        imgs = np.stack([C.warp(S.imgs[c, j + k], S.t["cam"][c], off_hist[k]) for k in range(C.NH)])
        v = float(S.t["v"][c, C.T0 + j])
        at, aw = C.act_label(hum, v) if role == 1 else (0.0, 0.0)
        return dict(imgs=imgs, tc=S.t["tc"][c], cam=np.float32(S.t["cam"][c][0] - 0.0), hum=hum.astype(np.float32), role=np.int64(role),
                    at=np.float32(at), aw=np.float32(aw), lt=np.zeros((32, 2), np.float32), ltm=np.zeros(32, bool))

    def __getitem__(self, k):
        if self.S is None:
            self._open()
        S, a = self.S, self.a
        rng = np.random.default_rng([a.seed, k])
        rows = []
        for _ in range(ROWS["R"]):
            c, j, off = self.st[rng.integers(len(self.st))]
            dx, dy, dp = off[j]
            v = float(S.t["v"][c, C.T0 + j])
            if a.static:
                oh = C.drift_offsets(dy, dp, v)
            else:
                oh = np.zeros((C.NH, 3))
                for kk in range(C.NH):                       # clip frame j + kk; after t0 (index > T0) the rollout's offset at that step
                    fi = j + kk
                    if fi > C.T0:
                        oh[kk] = off[fi - C.T0]
            rows.append(self.row(c, j, oh, C.recovery_hum(S.t["fut20"][c, j], dy, dp, v), 1))
        for role, n in (("U", ROWS["U"]), ("D", ROWS["D"])):
            for _ in range(n):
                c, j = int(rng.integers(S.n)), int(rng.integers(C.K + 1))
                hum = C.recovery_hum(S.t["fut20"][c, j], 0.0, 0.0, 0.0)
                rows.append(self.row(c, j, np.zeros((C.NH, 3)), hum, 1 if role == "U" else 2))
        return {k: torch.from_numpy(np.stack([np.asarray(r[k]) for r in rows])) for k in rows[0]}


def main():
    import rft
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from experiments.op_adapt_h.lib import op_adapt_h as H
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rolls", default="shipped")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    d = C.root("runs", a.tag)
    if (d / "ckpt-final.pt").exists():
        print("exists", d)
        return
    cfg = rft.RCfg(name=a.tag, enc=None, seed=a.seed, steps=a.steps, lam_p=0.0, workers=a.workers, warmup=max(1, min(100, a.steps // 5)))
    with Run("op_dagger", a.tag, seed=a.seed, config=cfg.dump() | vars(a)) as run:
        run.use_split(splits.load("wod/r2-train"))
        dev = torch.device(a.dev)
        dt = torch.float16 if dev.type == "cuda" else torch.float32
        model = rft.RModel(cfg.lcfg(), None, dtype=dt).to(dev)
        teacher = L.LModel(None, dtype=dt).to(dev).eval()
        base, _ = model.trainable()
        opt = torch.optim.AdamW(base, lr=cfg.lr, weight_decay=cfg.wd)
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
        tstd = np.load(L.r2t() / "teacher" / "tstd.npy")
        lossf = rft.RLoss(model.net, cfg, tstd, dev)
        didx = torch.as_tensor(A.distill_index(model.net.slices), device=dev)
        pidx = torch.as_tensor(A.plan_index(model.net.slices), device=dev)
        ds = Batcher(a)
        ds._open()
        run.info(f"{a.tag}: {len(ds.st)} visited states from {a.rolls}, static={a.static}, {a.steps} steps")
        run.summary["n_states"] = len(ds.st)
        ds.S = None
        dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=range(10 ** 9), num_workers=a.workers, prefetch_factor=2, pin_memory=True)
        it = iter(dl)
        model.train()
        hist, wait, t0 = [], 0.0, time.time()
        bar = run.tqdm(total=a.steps, desc=a.tag)
        for step in range(a.steps):
            tw = time.time()
            b = next(it)
            wait += time.time() - tw
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            valid = torch.ones(len(b["role"]), 9, dtype=torch.bool, device=dev)
            tc = b["tc"].to(dt)
            with torch.no_grad():
                tr = H.trunks(teacher.net, b["imgs"])
                o0 = teacher(tr, valid, tc)["outputs"].float()
            b["tgt"], b["tmu"] = o0[:, didx], o0[:, pidx].view(-1, 33, 15)
            o = model(tr, valid, tc)
            total, Ls = lossf(o, b)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base, 1.0)
            scaler.step(opt)
            scaler.update()
            bar.update()
            hist.append({k: float(v) for k, v in Ls.items()} | {"total": float(total)})
            if (step + 1) % 25 == 0:
                m = {k: float(np.mean([h[k] for h in hist[-25:] if k in h])) for k in hist[-1]}
                el = time.time() - t0
                run.scalars({f"loss/{k}": v for k, v in m.items()} | {"throughput/steps_per_s": (step + 1) / el, "throughput/wait": wait / el}, step + 1)
                if (step + 1) % 100 == 0:
                    run.info(f"step {step + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) + f"; {(step + 1) / el:.2f} it/s, wait {wait / el:.2f}")
        bar.close()
        del it, dl
        torch.save({"model": model.state(), "cfg": cfg.lcfg().dump(), "rcfg": cfg.dump()}, d / "ckpt-final.pt")
        run.summary.update(steps=a.steps, train_s=time.time() - t0, wait_frac=wait / (time.time() - t0))


if __name__ == "__main__":
    main()
