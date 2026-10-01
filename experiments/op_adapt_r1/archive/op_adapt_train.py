"""op-adapt step 3: the B trial (op-train venv). Cinque stage 4 unfrozen, pedestrian aux heads, distillation to the
original outputs on normal frames; trunk outputs from processed/op_adapt/nusc (experiments/op_adapt_r1/lib/op_adapt_cache.py).
Registered configuration and pass lines: fc65452:todos/2026-09-28-op-adapt.md.

Samples: every cached slot j of a train scene (context slots j, j-2, .., j-16). Keyframe slots carry the GT labels
(processed/op_adapt/nusc_labels.parquet); a slot is "normal" (distilled) when its keyframe (or both neighbouring
keyframes) has no pedestrian / cyclist in the wide corridor. A batch is half keyframes (wide-corridor positives
oversampled to 30 %) and half uniformly drawn slots.

  CUDA_VISIBLE_DEVICES=2 python experiments/op_adapt_r1/archive/op_adapt_train.py --lam-d 1 --minutes 40
Output: runs/op_adapt/train-lam<λ>/<ts>/{ckpt.pt, dev.json, log.txt, events.jsonl, tb/}
"""
import argparse, json, sys, threading, queue, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_data as D  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

AT = (0.275, 0.525)
OFF = np.arange(-2 * (A.CONTEXT - 1), 1, 2)          # context slots relative to the sample slot (stride 2)
DTYPES = {"fp16": torch.float16, "tf32": torch.float32, "bf16": torch.bfloat16}


def split_scenes(lab, n_dev=50, seed=0):
    tr = np.array(sorted(lab[lab.split == "train"].scene.unique()))
    dev = set(np.random.default_rng(seed).permutation(tr)[:n_dev])
    return [s for s in tr if s not in dev], sorted(dev), sorted(lab[lab.split == "val"].scene.unique())


class Store:
    """Trunk slots of a set of scenes in one CPU tensor, with the per-sample metadata."""

    def __init__(self, scenes, lab):
        L = lab.set_index("token")
        blocks, rows, base = [], [], 0
        for s in scenes:
            z = np.load(D.root("nusc") / f"{s}.npz")
            T = z["trunk"]
            n = len(T)
            key = {int(k): str(t) for k, t in zip(z["key_slot"], z["tokens"])}
            kn = {k: not bool(L.at[t, "vru_wide"]) for k, t in key.items()}
            for j in range(n):
                k0 = 5 * (j // 5)
                norm = kn.get(k0, False) and (j % 5 == 0 or kn.get(k0 + 5, False))
                t = key.get(j)
                rows.append((base + j, j, t or "", norm, tuple(z["traffic"])))
            blocks.append(torch.from_numpy(T))
            base += n
        self.T = torch.cat(blocks)                                         # (N, 1024, 8, 16) fp16
        self.slot = np.array([r[0] for r in rows])
        self.local = np.array([r[1] for r in rows])
        self.token = np.array([r[2] for r in rows])
        self.normal = np.array([r[3] for r in rows])
        self.traffic = np.array([r[4] for r in rows], np.float32)
        self.key = self.token != ""
        lk = L.reindex(self.token[self.key])
        self.y = np.zeros((len(rows), 3), np.float32)
        self.y[self.key, 0] = lk.ped_corr.to_numpy(float)
        self.y[self.key, 1] = lk.ped_wide.to_numpy(float)
        self.y[self.key, 2] = np.nan_to_num(lk.ped_dist.to_numpy(float), nan=0.0) / 10
        self.wide_vru = np.zeros(len(rows), bool)
        self.wide_vru[self.key] = lk.vru_wide.to_numpy(bool)

    def batch(self, idx, dev):
        """Context trunks (B, 9, 1024, 8, 16), validity (B, 9), traffic (B, 2) of the samples `idx`."""
        loc = self.local[idx][:, None] + OFF[None]
        valid = loc >= 0
        g = (self.slot[idx][:, None] + OFF[None]).clip(0)
        g = np.where(valid, g, self.slot[idx][:, None])                    # any row; masked to a zero hidden state
        x = self.T[torch.from_numpy(g.ravel())].view(len(idx), A.CONTEXT, *self.T.shape[1:])
        return (x.to(dev, non_blocking=True), torch.from_numpy(valid).to(dev),
                torch.from_numpy(self.traffic[idx]).to(dev))


@torch.no_grad()
def teacher(net, st, dev, didx, bs=256):
    """Original model outputs (distilled positions) and plan means for every sample of a store."""
    out, plan = [], []
    pidx = torch.as_tensor(A.plan_index(net.slices), device=dev)
    di = torch.as_tensor(didx, device=dev)
    for i in range(0, len(st.slot), bs):
        idx = np.arange(i, min(i + bs, len(st.slot)))
        x, v, tc = st.batch(idx, dev)
        o = A.stage4_policy(net, x, AT, tc, v)["outputs"].float()
        out.append(o[:, di].half().cpu())
        plan.append(o[:, pidx].cpu())
    return torch.cat(out), torch.cat(plan).view(-1, 33, 15).numpy()


def loss_aux(pred, y):
    l = F.binary_cross_entropy_with_logits(pred[:, 0], y[:, 0]) + F.binary_cross_entropy_with_logits(pred[:, 1], y[:, 1])
    pos = y[:, 0] > 0.5
    if pos.any():
        l = l + 0.1 * F.smooth_l1_loss(pred[pos, 2], y[pos, 2])
    return l


@torch.no_grad()
def evaluate(net, heads, st, tplan, dev, bs=256):
    """Dev readout: aux-head AUC (corridor, temporal head / token head) on keyframes, plan drift on normal keyframes."""
    from sklearn.metrics import roc_auc_score
    pidx = torch.as_tensor(A.plan_index(net.slices), device=dev)
    kidx = np.flatnonzero(st.key)
    sa, sb, pl = [], [], []
    for i in range(0, len(kidx), bs):
        idx = kidx[i:i + bs]
        x, v, tc = st.batch(idx, dev)
        o = A.stage4_policy(net, x, AT, tc, v)
        a, b = heads(o["select_4"], o["tokens"])
        sa.append(a[:, 0].float().cpu())
        sb.append(b[:, 0].float().cpu())
        pl.append(o["outputs"].float()[:, pidx].cpu())
    y = st.y[kidx, 0]
    pl = torch.cat(pl).view(-1, 33, 15).numpy()
    dr = A.plan_drift(pl, tplan[kidx])
    nm = st.normal[kidx]
    return {"auc_temporal_head": float(roc_auc_score(y, torch.cat(sa))), "auc_token_head": float(roc_auc_score(y, torch.cat(sb))),
            "drift_median": float(np.median(dr[nm])), "drift_p95": float(np.percentile(dr[nm], 95)),
            "n_key": int(len(kidx)), "n_pos": int(y.sum()), "n_normal": int(nm.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-d", type=float, required=True)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-heads", type=float, default=1e-3)
    ap.add_argument("--dtype", default="fp16", choices=DTYPES)
    ap.add_argument("--limit-scenes", type=int, default=0, help="smoke test: first n train scenes")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="aux-head init and batch sampling (the scene split stays seed 0)")
    a = ap.parse_args()
    import pandas as pd
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = a.dtype == "tf32"
    torch.manual_seed(a.seed)
    log = RunLog("op_adapt", f"train-lam{a.lam_d:g}" + (f"-s{a.seed}" if a.seed else ""))
    log.event("start", args=vars(a))
    dev = torch.device("cuda")
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
    tr_sc, dev_sc, _ = split_scenes(lab)
    have = {p.stem for p in D.root("nusc").glob("*.npz") if not p.stem.endswith(".tmp")}
    tr_sc, dev_sc = [s for s in tr_sc if s in have], [s for s in dev_sc if s in have]
    if a.limit_scenes:
        tr_sc, dev_sc = tr_sc[: a.limit_scenes], dev_sc[: max(2, a.limit_scenes // 10)]
    t0 = time.time()
    st, sd = Store(tr_sc, lab), Store(dev_sc, lab)
    log.info(f"train {len(tr_sc)} scenes / {len(st.slot)} slots ({st.key.sum()} keyframes, {int(st.y[:, 0].sum())} corridor "
             f"pedestrians, {st.normal.mean():.2f} normal); dev {len(dev_sc)} scenes / {len(sd.slot)} slots; "
             f"{st.T.numel() * 2 / 2 ** 30:.1f} GB; {time.time() - t0:.0f} s")
    dt = DTYPES[a.dtype]
    net = A.load("cinque", dt, trainable=A.stage4_weights()).to(dev)
    didx = A.distill_index(net.slices)
    t1 = time.time()
    net.eval()
    tgt, tplan = teacher(net, st, dev, didx)
    _, dplan = teacher(net, sd, dev, didx)
    tstd = tgt[torch.from_numpy(st.normal)].float().std(0).clamp_min(1e-3).to(dev)
    log.info(f"teacher outputs for {len(tgt) + len(dplan)} samples in {time.time() - t1:.0f} s; {len(didx)} distilled dims")
    heads = A.AuxHeads().to(dev)
    opt = torch.optim.AdamW([{"params": [p for p in net.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": heads.parameters(), "lr": a.lr_heads}], weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=a.dtype == "fp16")
    base = evaluate(net, heads, sd, dplan, dev)
    log.info(f"dev before training: {base}")
    # sampling pools
    keys = np.flatnonzero(st.key)
    kpos, kneg = keys[st.wide_vru[keys] | (st.y[keys, 1] > 0)], keys[~(st.wide_vru[keys] | (st.y[keys, 1] > 0))]
    half = a.batch // 2

    def draw(rng):
        npos = int(round(0.3 * half))
        k = np.r_[rng.choice(kpos, npos), rng.choice(kneg, half - npos)]
        return np.r_[k, rng.integers(0, len(st.slot), a.batch - half)]

    q = queue.Queue(maxsize=6)
    stop = threading.Event()

    def producer(seed):
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            idx = draw(rng)
            x, v, tc = st.batch(idx, dev)
            q.put((idx, x, v, tc))
    th = [threading.Thread(target=producer, args=(1000 * a.seed + k,), daemon=True) for k in range(3)]
    for t in th:
        t.start()
    di = torch.as_tensor(didx, device=dev)
    deadline, step, hist = time.time() + 60 * a.minutes, 0, []
    net.train()
    tstart = time.time()
    while time.time() < deadline:
        idx, x, v, tc = q.get()
        o = A.stage4_policy(net, x, AT, tc, v)
        ha, hb = heads(o["select_4"], o["tokens"])
        y = torch.from_numpy(st.y[idx[:half]]).to(dev)
        la = loss_aux(ha[:half], y) + loss_aux(hb[:half], y)
        nm = torch.from_numpy(st.normal[idx]).to(dev)
        tt = tgt[torch.from_numpy(idx)].to(dev).float()
        e = ((o["outputs"].float()[:, di] - tt) / tstd).pow(2).mean(1)
        ld = (e * nm).sum() / nm.sum().clamp_min(1)
        loss = la + a.lam_d * ld
        frac = min(1.0, (time.time() - tstart) / (60 * a.minutes))
        for g, lr0 in zip(opt.param_groups, (a.lr, a.lr_heads)):
            g["lr"] = lr0 * (min(1.0, (step + 1) / 200)) * 0.5 * (1 + np.cos(np.pi * frac))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        hist.append((la.item(), ld.item()))
        if step % 50 == 0:
            m = np.mean(hist[-50:], 0)
            sps = step * a.batch / (time.time() - tstart)
            log.scalar("loss/aux", m[0], step)
            log.scalar("loss/distill", m[1], step)
            log.scalar("throughput/samples_per_s", sps, step)
            if step % 500 == 0:
                log.info(f"step {step}: aux {m[0]:.4f} distill {m[1]:.4f}, {sps:.0f} samples/s, "
                         f"{(deadline - time.time()) / 60:.0f} min left")
        if step % a.eval_every == 0:
            net.eval()
            r = evaluate(net, heads, sd, dplan, dev)
            net.train()
            for k, v_ in r.items():
                if not k.startswith("n_"):
                    log.scalar(f"dev/{k}", v_, step)
            log.event("dev", step=step, **r)
    stop.set()
    net.eval()
    r = evaluate(net, heads, sd, dplan, dev)
    gpu_s = time.time() - tstart
    r |= {"steps": step, "samples": step * a.batch, "train_gpu_s": gpu_s, "lam_d": a.lam_d, "before": base}
    log.info(f"dev after training: {r}")
    (log.dir / "dev.json").write_text(json.dumps(r, indent=1))
    torch.save({"stage4": {k: p.detach().cpu() for k, p in net.params.items() if p.requires_grad},
                "heads": heads.state_dict(), "args": vars(a), "dev": r}, log.dir / "ckpt.pt")
    log.event("end", **{k: v_ for k, v_ in r.items() if k != "before"})


if __name__ == "__main__":
    main()
