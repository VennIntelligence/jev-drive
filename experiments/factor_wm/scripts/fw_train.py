"""factor_wm G1 fine-tune (op-train env, one GPU): op_dagger's recipe (rft.RModel: stage 4 + plan + action pathways trainable; rft.RLoss:
imitation + action[0] target on P rows, plan consistency + every head distilled to shipped on D rows) plus a supervised action[1]
(acceleration) target, on the factor_wm WOD train clips (50 frames), images re-projected online (plane engine).

Rows of a batch (48):
  R 20  visited states of closed-loop rollouts (fw_roll collect), kept only inside the validity domain up to that step and before the first
        failure event: |dy| <= 1 m, |dpsi| <= 5 deg (turn states beyond it get no label: decision 2026-10-06), |dx| <= 5 m;
        history = the rollout's own re-projected frames
  U 12  logged states (offset 0) of the same clips          (S1: U 32, no R rows)
  D 16  logged states, plan consistency + every head to shipped
Label of a state at offset (dx, dy, dpsi) with ego speed v: the logged 5 s future of the time-synchronous logged frame, lateral recovery
(op_adapt_h.recover_target, t_rec 4 s: the gain restraint against decision 132's over-steer) plus a longitudinal catch-up: the gap to the
logged pose (-dx along the path) closes with a smoothstep over 3 s. action[0] = rft.act_target of that path; action[1] = (v(0.5 s) - v) / 0.5
clipped to [-3.5, 2] m/s2, v(0.5 s) the target path's speed between 0.25 and 0.75 s. The same label function for every arm.

  fw_train.py --tag S3 --rolls s3r1,s3r2,s3r3 --steps 2400 [--init <ckpt>]     DAgger arm
  fw_train.py --tag S1 --rolls none --steps 2400                               off-policy imitation of the same clips
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402

T_REC = 4.0
T_CATCH = 3.0
CAP_Y, CAP_PSI = 1.0, np.radians(5.0)
TF = 0.25 * np.arange(1, 21)


def label(fut20, off, v):
    """(hum (16, 3), action[0] target, weight, action[1] target)."""
    import rft
    from experiments.op_adapt_h.lib import op_adapt_h as H
    from experiments.op_adapt_l.lib import op_adapt_l as L
    dx, dy, dp = (float(x) for x in off)
    f = np.asarray(fut20, float)
    q = H.recover_target(f, dy, dp, v, t_rec=T_REC) if (abs(dy) > 1e-6 or abs(dp) > 1e-9) else f.copy()
    if abs(dx) > 1e-6:
        u = np.clip(TF / T_CATCH, 0, 1)
        q = q.copy()
        q[:, 0] += -dx * (3 * u ** 2 - 2 * u ** 3)
    hum = L.human_targets(np.asarray(q, np.float32)[None])[0]
    at, aw = rft.act_target(hum, v)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.r_[np.zeros((1, 2)), q], axis=0), axis=1))]
    v_mid = (s[3] - s[1]) / 0.5                                                      # t 0.75 - 0.25
    a1 = float(np.clip((v_mid - v) / 0.5, -3.5, 2.0))
    return hum, at, aw, a1


def states_of(rolls):
    out = []
    for r in rolls:
        for f in sorted(glob.glob(str(C.root("roll", r) / "train-collect-[0-9]*of*.npz"))):
            z = np.load(f, allow_pickle=True)
            for c, off, vs, te in zip(z["c"], z["off"], z["vs"], z["t_event"]):
                ok = (np.abs(off[:, 1]) <= CAP_Y) & (np.abs(off[:, 2]) <= CAP_PSI) & (np.abs(off[:, 0]) <= C.DX_CAP)
                ok = np.logical_and.accumulate(ok)
                jmax = len(off) - 1 if not np.isfinite(te) else int(round(te / C.DT)) - 1
                for j in range(1, min(jmax, len(off) - 1) + 1):
                    if ok[j]:
                        out.append((int(c), j, off.astype(np.float32), float(vs[j])))
    return out


class Batcher(torch.utils.data.Dataset):
    def __init__(self, a, n=10 ** 9):
        self.a, self.n, self.S = a, n, None

    def __len__(self):
        return self.n

    def _open(self):
        self.S = C.Clips("train")
        self.st = states_of([r for r in self.a.rolls.split(",") if r and r != "none"])

    def row(self, c, j, oh, off, v, role):
        S = self.S
        imgs = np.stack([C.warp("plane", S.imgs[c, j + k], None, S.t["cam"][c], oh[k]) for k in range(C.NH)])
        hum, at, aw, a1 = label(S.t["fut20"][c, C.T0 + j], off, v)
        if role != 1:
            at, aw, a1 = 0.0, 0.0, 0.0
        return dict(imgs=imgs, tc=S.t["tc"][c], cam=np.float32(S.t["cam"][c][0]), hum=hum.astype(np.float32), role=np.int64(role),
                    at=np.float32(at), aw=np.float32(aw), a1=np.float32(a1), lt=np.zeros((32, 2), np.float32), ltm=np.zeros(32, bool))

    def __getitem__(self, k):
        if self.S is None:
            self._open()
        S, a = self.S, self.a
        rng = np.random.default_rng([a.seed, k])
        rows = []
        nR = 20 if self.st else 0
        for _ in range(nR):
            c, j, off, v = self.st[rng.integers(len(self.st))]
            oh = np.zeros((C.NH, 3))
            for kk in range(C.NH):
                fi = j + kk
                if fi > C.T0:
                    oh[kk] = off[fi - C.T0]
            rows.append(self.row(c, j, oh, off[j], v, 1))
        kl = S.kl
        for role, nrow in ((1, 12 + 20 - nR), (2, 16)):
            for _ in range(nrow):
                c, j = int(rng.integers(S.n)), int(rng.integers(kl + 1))
                rows.append(self.row(c, j, np.zeros((C.NH, 3)), np.zeros(3), float(S.t["v"][c, C.T0 + j]), role))
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
    ap.add_argument("--rolls", default="none")
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lam-acc", type=float, default=1.0)
    a = ap.parse_args()
    d = C.root("runs", a.tag)
    if (d / "ckpt-final.pt").exists():
        print("exists", d)
        return
    cfg = rft.RCfg(name=a.tag, enc=None, seed=a.seed, steps=a.steps, lam_p=0.0, workers=a.workers, warmup=max(1, min(100, a.steps // 5)))
    with Run("factor_wm", a.tag, seed=a.seed, config=cfg.dump() | vars(a)) as run:
        run.use_split(splits.load("wod/r2-train"))
        dev = torch.device("cuda")
        dt = torch.float16
        model = rft.RModel(cfg.lcfg(), None, dtype=dt).to(dev)
        teacher = L.LModel(None, dtype=dt).to(dev).eval()
        base, _ = model.trainable()
        opt = torch.optim.AdamW(base, lr=cfg.lr, weight_decay=cfg.wd)
        scaler = torch.amp.GradScaler()
        tstd = np.load(L.r2t() / "teacher" / "tstd.npy")
        lossf = rft.RLoss(model.net, cfg, tstd, dev)
        a1c = lossf.a0 + 1
        didx = torch.as_tensor(A.distill_index(model.net.slices), device=dev)
        pidx = torch.as_tensor(A.plan_index(model.net.slices), device=dev)
        ds = Batcher(a)
        ds._open()
        run.info(f"{a.tag}: {len(ds.st)} visited states from {a.rolls}, {ds.S.n} clips, {a.steps} steps")
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
            P = b["role"] == 1
            if P.any():
                Ls["acc"] = rft.L.huber((o["outputs"].float()[P][:, a1c] - b["a1"][P]) / rft.SIG_A).mean()
                total = total + a.lam_acc * cfg.lam_a * Ls["acc"]
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
