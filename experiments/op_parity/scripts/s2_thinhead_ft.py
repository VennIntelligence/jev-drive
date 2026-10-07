"""s2-thinhead Q3, the light joint fine-tune pilot (plans/2026-10-08-s2-thinhead-prereg.md): Qwen3-VL-4B on the multi-frame input of the QV stream
(3 cameras x 4 frames at 0.2 s, video path, decoder cut at layer 18), LoRA on the attention q / v projections of decoder layers 15-18 (the last
fusion blocks of the truncated model) + the multi-stream head (vision, ego, plan). Two arms that share everything but the LoRA:
  FZ  head only (the frozen QV feature, same head): pre-trained on supervision a (hindsight class, r2-train), fine-tuned on b by sequence k-fold
  FT  LoRA + head: starts from FZ's pre-trained head and a zero LoRA, pre-trained on a (a subset of r2-train rows, online), then b by the same folds

  cache     (GPU) hidden states after layer 14 of the val rater frames -> $DATA_DIR/runs/op_parity/s2_thinhead/ft/h14_val.npy; checks that the frozen
            layers 15-18 reproduce the cached `L18_mean`
  pretrain  (GPU) FZ head on all train rows (cached features), then FT on N_PRE rows online; hindsight NLL / accuracy of both on r2-dev rows
  kfold     (GPU) b by sequence 5-fold (the folds of repeat 0 of the frozen arms), out-of-fold gain per frame, in-sample gain, one permutation run
  report    (CPU) tables -> results/s2_thinhead/ft*.{csv,md}
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import s2_thinhead as T  # noqa: E402

FT = T.RUNS / "ft"
CUT, RANK, ALPHA, MH, TAU, KEEP = 14, 16, 32, 32, 0.5, T.G.KEEP * 4
N_PRE, N_DEV, BATCH, FOLDS, EPOCHS = 3000, 400, 4, 5, 3


def lora_cls():
    import torch

    class LoRA(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = torch.nn.Linear(base.in_features, RANK, bias=False)
            self.B = torch.nn.Linear(RANK, base.out_features, bias=False)
            torch.nn.init.zeros_(self.B.weight)

        def forward(self, x):
            y = self.base(x)
            return y + (self.B(self.A(x.float())) * (ALPHA / RANK)).to(y.dtype)
    return LoRA


def head_cls():
    import torch

    class Head(torch.nn.Module):
        """vision (standardised) / ego / plan -> one linear encoder each -> GELU -> dropout -> 20 (the M head of the frozen arms)."""

        def __init__(self, mu, sd):
            super().__init__()
            self.register_buffer("mu", torch.as_tensor(mu))
            self.register_buffer("sd", torch.as_tensor(sd))
            self.enc = torch.nn.ModuleList([torch.nn.Linear(d, MH) for d in (len(mu), 20, 15)])
            self.out = torch.nn.Linear(3 * MH, 20)
            torch.nn.init.normal_(self.out.weight, std=0.01)

        def forward(self, v, ego, plan):
            h = torch.cat([self.enc[0]((v - self.mu) / self.sd), self.enc[1](ego), self.enc[2](plan)], -1)
            return self.out(torch.nn.functional.dropout(torch.nn.functional.gelu(h), 0.5, self.training))
    return Head


class Engine:
    """The QV extractor split at layer CUT: front (frozen, no grad) and back (layers CUT + 1 .. 18, LoRA on q / v)."""

    def __init__(self, lora=True):
        import torch
        from jevdrive import waymo_qwenvid as QV
        self.t, self.fx = torch, QV.make_fx()
        for p in self.fx.model.parameters():
            p.requires_grad_(False)
        self.lora = []
        if lora:
            L = lora_cls()
            for layer in self.fx.lm_layers[CUT:QV.LAYER]:
                for nm in ("q_proj", "v_proj"):
                    m = L(getattr(layer.self_attn, nm)).to("cuda")
                    setattr(layer.self_attn, nm, m)
                    self.lora.append(m)
        self.back_layers = self.fx.lm_layers[CUT:QV.LAYER]

    def lora_params(self):
        return [p for m in self.lora for p in (m.A.weight, m.B.weight)]

    def lora_state(self):
        return [p.detach().cpu().clone() for p in self.lora_params()]

    def load_lora(self, st):
        with self.t.no_grad():
            for p, s in zip(self.lora_params(), st):
                p.copy_(s.to(p.device))

    def front(self, batch):
        t, fx = self.t, self.fx
        with t.no_grad():
            key = (*batch[0].shape, *batch[3].shape, *batch[3][0].tolist())
            ids, mm, pv, grid = (x.to("cuda", non_blocking=True) for x in batch)
            pos_embeds, rot, cu, ipos, lm_rot = fx._static(key, ids, mm, grid)
            vis, lm, b = fx.model.visual, fx.model.language_model, len(ids)
            x, deep = vis.patch_embed(pv) + pos_embeds, []
            for i, blk in enumerate(fx.vis_blocks):
                x = blk(x, cu_seqlens=cu, position_embeddings=rot)
                if i in vis.deepstack_visual_indexes:
                    deep.append(vis.deepstack_merger_list[vis.deepstack_visual_indexes.index(i)](x).view(b, -1, lm.config.hidden_size))
            h = lm.embed_tokens(ids)
            h[:, ipos] = vis.merger(x).view(b, -1, lm.config.hidden_size)
            for i, layer in enumerate(fx.lm_layers[:CUT]):
                h = layer(h, position_embeddings=lm_rot)
                if i < len(deep):
                    h[:, ipos] += deep[i]
            self.ipos, self.rot1 = ipos, tuple(r[:1] for r in lm_rot)
        return h

    def back(self, h):
        rot = tuple(r.expand(len(h), *r.shape[1:]) for r in self.rot1)
        for layer in self.back_layers:
            h = layer(h, position_embeddings=rot)
        return h[:, self.ipos].float().mean(1)


def load_data():
    z = np.load(T.RUNS / "data.npz")
    return {k: z[k] for k in ("va_names", "va_seq", "va_ego", "va_plan", "va_J", "tr_names", "tr_dev", "tr_ego", "tr_plan", "tr_lab")}


def clip_items(names):
    from jevdrive import waymo as W
    from jevdrive import waymo_qwenvid as QV
    df = W.load_index()
    key = dict(zip(W.frame_names(df), range(len(df))))
    items, idx, full = W.multicam_clip_items(df, np.array([key[x] for x in names]), QV.FRAMES - 1, QV.STRIDE, W.CAMS)
    return items, full


def raw_qv(names, tag):
    """Cached raw `L18_mean` of the QV set for the frame names (file cache under ft/)."""
    f = FT / f"qv_raw_{tag}.npz"
    if f.exists():
        z = np.load(f)
        if len(z["x"]) == len(names):
            return z["x"], z["ok"]
    x, ok = T.shard_feats(T.QV_SET, ["L18_mean"], names)
    np.savez(f, x=x["L18_mean"], ok=ok)
    return x["L18_mean"], ok


def cmd_cache(a):
    import torch
    from torch.utils.data import DataLoader
    from jevdrive import waymo as W
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-ft-cache", config=vars(a)) as run:
        FT.mkdir(parents=True, exist_ok=True)
        D = load_data()
        names = D["va_names"].astype(str)
        items, full = clip_items(names)
        E = Engine(lora=False)
        dl = DataLoader(W.Shards(items, E.fx.transform), batch_size=2, num_workers=6, collate_fn=E.fx.collate)
        ref, ok = raw_qv(names, "val")
        mm, vs, i = None, [], 0
        for b in run.tqdm(dl, desc="h14"):
            h = E.front(b)
            if mm is None:
                mm = np.lib.format.open_memmap(FT / "h14_val.npy", "w+", np.float16, (int(full.sum()), *h.shape[1:]))
            mm[i:i + len(h)] = h.float().cpu().numpy().astype(np.float16)
            with torch.no_grad():
                vs.append(E.back(h).cpu().numpy())
            i += len(h)
        mm.flush()
        v = np.concatenate(vs)
        r = ref[full]
        rel = np.linalg.norm(v - r, axis=1) / np.linalg.norm(r, axis=1)
        np.savez(FT / "h14_val_meta.npz", full=full, v=v)
        run.info("h14 %s; split forward vs the cached L18_mean: relative L2 median %.4f, max %.4f (frames with a clip %d / %d)", mm.shape, np.median(rel), rel.max(),
                 full.sum(), len(full))
        run.summary.update(rel_median=float(np.median(rel)), rel_max=float(rel.max()), shape=list(mm.shape))


def fz_pretrain(D, x, ok, dev):
    """The FZ head on supervision a: every train row with the cached feature; 600 steps of 4 096 (as the M head of the frozen arms)."""
    import torch
    tr, dv = np.flatnonzero(ok & ~D["tr_dev"]), np.flatnonzero(ok & D["tr_dev"])
    fit = tr[:: max(1, len(tr) // 20000)]
    hd = head_cls()(x[fit].mean(0), x[fit].std(0) + 1e-6).to(dev)
    X, E_, P, y = (torch.from_numpy(np.ascontiguousarray(v)).to(dev) for v in (x, D["tr_ego"], D["tr_plan"], D["tr_lab"].astype(np.int64)))
    opt = torch.optim.AdamW(hd.parameters(), lr=3e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(0)
    tri = torch.from_numpy(tr)
    hd.train()
    for _ in range(600):
        b = tri[torch.randint(len(tri), (4096,), generator=g)].to(dev)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(hd(X[b], E_[b], P[b]), y[b]).backward()
        opt.step()
    hd.eval()
    with torch.no_grad():
        d = torch.from_numpy(dv).to(dev)
        lg = hd(X[d], E_[d], P[d])
        info = {"dev rows": len(dv), "dev_nll": float(torch.nn.functional.cross_entropy(lg, y[d])), "dev_acc": float((lg.argmax(1) == y[d]).float().mean())}
    return hd, info


def cmd_pretrain(a):
    import torch
    from torch.utils.data import DataLoader
    from jevdrive import waymo as W
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", "s2-thinhead-ft-pretrain", seed=0, config=vars(a) | dict(n_pre=a.n_pre, cut=CUT, rank=RANK)) as run:
        for s in ("wod/r2-train", "wod/r2-dev"):
            run.use_split(splits.load(s))
        FT.mkdir(parents=True, exist_ok=True)
        D = load_data()
        tn = D["tr_names"].astype(str)
        x, ok = raw_qv(tn, "train")
        hd, info = fz_pretrain(D, x, ok, dev)
        torch.save(hd.state_dict(), FT / "fz_head.pt")
        run.info("FZ head on a: %s", info)
        rng = np.random.default_rng(0)
        pre = rng.permutation(np.flatnonzero(ok & ~D["tr_dev"]))[: a.n_pre]
        dvr = np.sort(rng.permutation(np.flatnonzero(ok & D["tr_dev"]))[: a.n_dev])
        E = Engine()
        ego, plan, lab = (torch.from_numpy(np.ascontiguousarray(v)).to(dev) for v in (D["tr_ego"], D["tr_plan"], D["tr_lab"].astype(np.int64)))

        def loader(rows):
            items, full = clip_items(tn[rows])
            assert full.all()
            return DataLoader(W.Shards(items, E.fx.transform), batch_size=BATCH, num_workers=8, collate_fn=E.fx.collate)

        def dev_eval(tag):
            hd.eval()
            nll, acc, i, vs = 0.0, 0.0, 0, []
            with torch.no_grad():
                for b in loader(dvr):
                    r = torch.from_numpy(dvr[i:i + len(b[0])]).to(dev)
                    v = E.back(E.front(b))
                    lg = hd(v, ego[r], plan[r])
                    nll += float(torch.nn.functional.cross_entropy(lg, lab[r], reduction="sum"))
                    acc += float((lg.argmax(1) == lab[r]).sum())
                    vs.append(v.cpu().numpy())
                    i += len(r)
            info[tag] = {"dev rows": len(dvr), "dev_nll": nll / len(dvr), "dev_acc": acc / len(dvr)}
            run.info("%s: %s", tag, info[tag])
            return np.concatenate(vs)
        v0 = dev_eval("online, LoRA at zero (= FZ, same rows)")
        rel = np.linalg.norm(v0 - x[dvr], axis=1) / np.linalg.norm(x[dvr], axis=1)
        info["online vs cached feature, relative L2 (median, max)"] = [float(np.median(rel)), float(rel.max())]
        opt = torch.optim.AdamW([{"params": E.lora_params(), "lr": 1e-4, "weight_decay": 1e-2}, {"params": hd.parameters(), "lr": 3e-4, "weight_decay": 1e-2}])
        t0, i, run_loss = time.time(), 0, 0.0
        hd.train()
        for step, b in enumerate(loader(pre)):
            r = torch.from_numpy(pre[i:i + len(b[0])]).to(dev)
            i += len(r)
            loss = torch.nn.functional.cross_entropy(hd(E.back(E.front(b)), ego[r], plan[r]), lab[r])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss)
            if (step + 1) % 100 == 0:
                run.info("step %d / %d, loss %.3f, %.2f s/step", step + 1, len(pre) // BATCH, run_loss / 100, (time.time() - t0) / (step + 1))
                run.scalar("pretrain/loss", run_loss / 100, step + 1)
                run_loss = 0.0
        info["pretrain"] = {"rows": int(i), "s_per_row": (time.time() - t0) / max(i, 1)}
        dev_eval("online, after the LoRA pre-training (FT)")
        torch.save({"lora": E.lora_state(), "head": hd.state_dict()}, FT / f"ft_pre{a.tag}.pt")
        (FT / f"pretrain{a.tag}.json").write_text(json.dumps(info, indent=1))
        run.summary.update(**{k: v for k, v in info.items() if isinstance(v, dict) and "dev_nll" in v})


def cmd_kfold(a):
    import pandas as pd
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", "s2-thinhead-ft-kfold", seed=0, config=vars(a) | dict(epochs=EPOCHS, folds=FOLDS)) as run:
        run.use_split(splits.load("wod/val"))
        D = load_data()
        n = len(D["va_names"])
        meta = np.load(FT / "h14_val_meta.npz")
        full = meta["full"]
        pos = np.full(n, -1)
        pos[full] = np.arange(full.sum())
        h14 = np.load(FT / "h14_val.npy", mmap_mode="r")
        scode = pd.factorize(pd.Series(D["va_seq"].astype(str)))[0]
        fold = (np.random.default_rng(0).permutation(scode.max() + 1) % FOLDS)[scode]          # repeat 0 of the frozen arms
        J = torch.from_numpy(D["va_J"]).float().to(dev)
        ego, plan = torch.from_numpy(D["va_ego"]).float().to(dev), torch.from_numpy(D["va_plan"]).float().to(dev)
        xraw, _ = raw_qv(D["va_names"].astype(str), "val")
        xraw = torch.from_numpy(xraw).to(dev)
        E = Engine()
        from jevdrive import waymo as W
        it, _ = clip_items(D["va_names"].astype(str)[full][:1])
        v1 = E.back(E.front(E.fx.collate([W.Shards(it, E.fx.transform)[0]])))                    # sets the token positions and the rotary table
        run.info("first frame, online vs cached feature: relative L2 %.4f", float((v1[0] - xraw[np.flatnonzero(full)[0]]).norm() / xraw[np.flatnonzero(full)[0]].norm()))
        pre = torch.load(FT / f"ft_pre{a.tag}.pt")
        fz = torch.load(FT / "fz_head.pt")
        Head = head_cls()

        def feats(idx, grad):
            """Vision feature of the frames idx through the back layers (from the cached layer-14 states)."""
            h = torch.from_numpy(np.asarray(h14[np.sort(pos[idx])])).to(dev, torch.bfloat16)
            o = np.argsort(np.argsort(pos[idx]))
            with torch.set_grad_enabled(grad):
                return E.back(h)[torch.as_tensor(o, device=dev)]

        def listwise(hd, v, idx, Jt):
            ii = torch.as_tensor(idx, device=dev)
            loss = 0.0
            for s in range(2):
                tgt = torch.softmax(Jt[s][:, ii].T / TAU, -1)
                loss = loss + (tgt * (torch.log(tgt + 1e-12) - torch.log_softmax(hd(v, ego[ii], plan[s][ii]), -1))).sum(-1).mean() / 2
            return loss

        def realised(hd, v, idx):
            ii = torch.as_tensor(idx, device=dev)
            with torch.no_grad():
                return torch.stack([J[s][hd(v, ego[ii], plan[s][ii]).argmax(1), ii] - J[s][KEEP, ii] for s in range(2)]).mean(0).cpu().numpy()

        def run_arm(arm, Jt, tag):
            d, ins = np.zeros(n), []
            for f in range(1 if a.tag else FOLDS):
                tr, te = np.flatnonzero((fold != f) & full), np.flatnonzero((fold == f) & full)
                if a.tag:
                    tr, te = tr[:24], te[:16]
                hd = Head(np.zeros(xraw.shape[1], np.float32), np.ones(xraw.shape[1], np.float32)).to(dev)
                hd.load_state_dict(fz if arm == "FZ" else pre["head"])
                if arm == "FZ":
                    opt = torch.optim.AdamW(hd.parameters(), lr=3e-4, weight_decay=1e-2)
                else:
                    E.load_lora(pre["lora"])
                    opt = torch.optim.AdamW([{"params": E.lora_params(), "lr": 5e-5, "weight_decay": 1e-2}, {"params": hd.parameters(), "lr": 3e-4, "weight_decay": 1e-2}])
                g = np.random.default_rng(100 + f)
                hd.train()
                for ep in range(1 if a.tag else EPOCHS):
                    o = g.permutation(tr)
                    for i in range(0, len(o), BATCH):
                        b = o[i:i + BATCH]
                        v = xraw[torch.as_tensor(b, device=dev)] if arm == "FZ" else feats(b, True)
                        loss = listwise(hd, v, b, Jt)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                hd.eval()
                ev = lambda idx: np.concatenate([realised(hd, xraw[torch.as_tensor(idx[i:i + 8], device=dev)] if arm == "FZ" else feats(idx[i:i + 8], False), idx[i:i + 8])  # noqa: E731
                                                 for i in range(0, len(idx), 8)])
                d[te] = ev(te)
                sub = tr[:: max(1, len(tr) // 96)]
                ins.append(float(ev(sub).mean()))
                run.info("%s %s fold %d: out-of-fold %+.3f (n %d), train-fold %+.3f", arm, tag, f, d[te].mean(), len(te), ins[-1])
            return d, float(np.mean(ins))
        out, info = {}, {}
        for arm in ("FZ", "FT"):
            out[arm], info[f"{arm} train-fold gain"] = run_arm(arm, J, "")
        pi = torch.as_tensor(np.random.default_rng(5000).permutation(n), device=dev)             # permutation control: targets of another frame
        Jp = J[:, :, pi]
        for arm in ("FZ", "FT")[: 2 if a.perm else 0]:
            out[f"{arm} perm"], info[f"{arm} perm train-fold gain"] = run_arm(arm, Jp, "perm")
        ev0 = {}
        for arm in ("FZ", "FT"):                                                                  # supervision a only (no rater labels): all frames
            hd = Head(np.zeros(xraw.shape[1], np.float32), np.ones(xraw.shape[1], np.float32)).to(dev).eval()
            hd.load_state_dict(fz if arm == "FZ" else pre["head"])
            if arm == "FT":
                E.load_lora(pre["lora"])
            idx = np.flatnonzero(full)[: 16 if a.tag else None]
            d = np.zeros(n)
            d[idx] = np.concatenate([realised(hd, xraw[torch.as_tensor(idx[i:i + 8], device=dev)] if arm == "FZ" else feats(idx[i:i + 8], False), idx[i:i + 8])
                                     for i in range(0, len(idx), 8)])
            ev0[f"{arm} a only"] = d
        np.savez(FT / f"kfold{a.tag}.npz", **out, **ev0, fold=fold, full=full)
        (FT / f"kfold{a.tag}.json").write_text(json.dumps(info, indent=1))
        run.summary.update(**{k: float(v.mean()) for k, v in (out | ev0).items()})


def cmd_report(a):
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "s2-thinhead-ft-report", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        Bd = T.board()
        C, z = Bd.C, dict(np.load(FT / "kfold.npz"))
        info = json.loads((FT / "kfold.json").read_text()) | {"pretrain": json.loads((FT / "pretrain.json").read_text())}
        rows = []
        cells = {k: v.astype(np.float64) for k, v in z.items() if k not in ("fold", "full")}
        cells |= {"FT - FZ (a then b)": cells["FT"] - cells["FZ"], "FT - FZ (a only)": cells["FT a only"] - cells["FZ a only"]}
        if "FT perm" in cells:
            cells["FT - FT perm"] = cells["FT"] - cells["FT perm"]
        for k, d in cells.items():
            for sn in ("all", "stopped", "moving", "turn"):
                c, o = T.ci(C, d, Bd.st[sn]), C.cm(Bd.best - Bd.base, Bd.st[sn])
                rows.append({"arm": k, "stratum": sn, "n": int(Bd.st[sn].sum()), "d": c[0], "lo": c[1], "hi": c[2], "share of oracle": c[0] / o,
                             "train-fold gain (in-sample)": info.get(f"{k} train-fold gain", np.nan)})
        stats.write_table(rows, T.OUT / "ft")
        (T.OUT / "ft_info.json").write_text(json.dumps(info, indent=1))
        import pandas as pd
        run.info("ft:\n%s\n%s", pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:+.3f}"), json.dumps(info["pretrain"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    sp_.add_parser("cache")
    p = sp_.add_parser("pretrain")
    p.add_argument("--n-pre", type=int, default=N_PRE)
    p.add_argument("--n-dev", type=int, default=N_DEV)
    p.add_argument("--tag", default="")
    p = sp_.add_parser("kfold")
    p.add_argument("--perm", type=int, default=1)
    p.add_argument("--tag", default="")
    sp_.add_parser("report")
    a = ap.parse_args()
    {"cache": cmd_cache, "pretrain": cmd_pretrain, "kfold": cmd_kfold, "report": cmd_report}[a.cmd](a)
