"""op_parity representation fix (plans/2026-10-07-representation-design.md section 6): stage 0 offline decoders and the stage 1 memory banks.

  vj21    --datas lb_navtest navtrain_full.s2of12 ...   (envs/wajepa, cwd third_party/wajepa, one GPU)
          front-only V-JEPA 2.1 ViT-L tokens through WA-JEPA's own encoder path (vjepa2_1_vit_large_384, RoPE interpolation, minus_one_to_imagenet,
          256 x 512, the 4 CAM_F0 history frames at 2 Hz, tubelet 2, the 12-frame clip with zero future and the history-token mask of
          predict_trajectory). First the equivalence check: WA's weights + scene projector on --check-n navtest tokens reproduce the cached WA-Cf
          (mean per-token cosine >= 0.999, else exit 3). Then the original vjepa2_1_vitl_dist_vitG_384.pt is loaded into the encoder with WA-JEPA's
          own loader (every encoder tensor must load) and the final layer (encoder norm) of the newest tubelet is 4 x 4 pooled -> 32 x 1024.
          -> $DATA_DIR/runs/op_probe/feats/VJ21/<data>.npz (tokens, raw (N, 32, 1024) fp16)
  x4      --datas ...                                    (envs/op-train, one GPU + CPU render pool)
          Cinque conv2d_36 (stage-4 downsample output, 2048 x 4 x 8) of the t0 policy slot under protocol W (pp_unfreeze's render path); equivalence:
          the frozen stage 4 + head on it = view_39 of the W front cache.  -> runs/op_probe/feats/X4/<data>.npy (N, 2048, 4, 8) fp16 + .tokens.npy
  decode                                                 (envs/op-train, one GPU)
          opb_probe's decoder (2-layer MLP 1024, [source, E], drivable hinge lambda 10, margin 0.3, 4000 steps, batch 512, seed 0, s2-s4 train tokens)
          per stage-0 arm; poses on T20 u S5 u eval_tokens; VJ21 PCA (1024 -> 512, per token, fitted on the train tokens) saved for stage 1;
          report-only junction look-ahead probe (logistic on [X, E], 5-fold log cross-fit on navtest).  -> runs/op_probe/rep/
  mem     --kind wa_cf|vj21 --datas ...                  (CPU) memory banks for pp_train --mem: runs/op_parity/mem/<kind>/<data>.npy (N, 32, 512)
          fp16 in the row order of cache/<data>/tab.npz
  report                                                 (CPU) stage-0 tables and gates from the opb_score.py CSVs -> results/representation/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent),
                 str(_R / "experiments/op_probe/scripts")]
import argparse, json, os, time  # noqa: E401,E402

import numpy as np  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
PROBE = D / "runs" / "op_probe"
FEATS = PROBE / "feats"
OUT = PROBE / "rep"
CR = D / "runs" / "op_parity" / "cache"
MEM = D / "runs" / "op_parity" / "mem"
VJ21_CKPT = D / "models" / "vjepa2_1" / "vjepa2_1_vitl_dist_vitG_384.pt"
TRAIN = ("navtrain_full.s2of12", "navtrain_full.s3of12", "navtrain_full.s4of12")
TEST = "lb_navtest"
RES = _R / "experiments" / "op_parity" / "results" / "representation"
# stage-0 arms: name -> sources concatenated before E (opb_probe stage_matrix convention)
ARMS = {"E": (), "V": ("V",), "WA": ("WA",), "V+WA": ("V", "WA"), "VJ21": ("VJ21",), "V+VJ21": ("V", "VJ21"), "X4": ("X4",),
        "VJ21raw": ("VJ21raw",)}


# ---------------------------------------------------------------- vj21 (wajepa env)
def cmd_vj21(a):
    import torch
    sys_p = [str(_R / "experiments/top10/lib/top10_t2")]
    _sys.path[:0] = sys_p
    import wajepa_run as WR
    import opb_wajepa as OW
    from jevdrive.run import Run
    from models.jepa_common import _load_submodule_state, _select_checkpoint_state
    with Run("op_parity", "rep-vj21", config=vars(a)) as run:
        agent = WR.build_agent()
        model = agent.model.eval()
        dev = next(model.parameters()).device

        class Front(torch.utils.data.Dataset):
            def __init__(self, paths):
                self.p = paths

            def __len__(self):
                return len(self.p)

            def __getitem__(self, i):
                return torch.stack([WR.image(str(p)) for p in self.p[i]])          # (4, 3, H, W), oldest first

        def encode(paths, kind):
            """paths (n, 4) CAM_F0 history frames -> (n, 32, 512) WA-Cf (kind wa) or (n, 32, 1024) final-layer tokens (kind vj21)."""
            dl = torch.utils.data.DataLoader(Front(paths), batch_size=a.batch, num_workers=a.workers, pin_memory=True, prefetch_factor=4)
            out, t0, k = [], time.time(), 0
            with torch.no_grad():
                for h in dl:
                    h = h.to(dev, non_blocking=True)
                    B = h.shape[0]
                    hist = h[:, :, None]                                                            # (B, 4, 1 view, 3, H, W)
                    fut = torch.zeros(B, model.num_future_frames, *hist.shape[2:], device=dev, dtype=hist.dtype)
                    clip = model._normalize_images(model._to_bcthw(torch.cat([hist, fut], 1)[:, :, 0]))   # (B, 3, 12, H, W)
                    mx = model.mask_sampler.full_future_mask(B, device=dev).masks_x[0]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = model.encoder(clip, masks=mx, training=True)                                   # (B, 1024 history tokens, 4 x 1024)
                        z = model.scene_projector(z[:, None])[:, 0] if kind == "wa" else z[..., -model.encoder.embed_dim:]
                    z = z.float()
                    c = z.shape[-1]
                    z = z.reshape(B, 2, 16, 32, c)[:, 1].permute(0, 3, 1, 2)                                # newest tubelet (B, c, 16, 32)
                    out.append(torch.nn.functional.avg_pool2d(z, 4).flatten(2).transpose(1, 2).half().cpu().numpy())
                    k += B
                    if (k // B) % 50 == 0:
                        run.status(f"{kind}: {k}/{len(paths)} ({k / (time.time() - t0):.1f}/s)")
            return np.concatenate(out)

        def front_paths(data, rows):
            z = OW.requests(data, rows)
            return z["keys"], z["img"][:, [1, 5, 9, 13]]                    # time-major x [L0, F0, R0, B0] -> the 4 CAM_F0 frames

        # 1. equivalence: WA weights, front path vs the cached WA-Cf (batched bf16 full-model forward)
        n_all = len(np.load(CR / TEST / "tab.npz")["names"])
        rows = np.arange(n_all)[0::3][: a.check_n]                         # rows of the cached shard lb_navtest.s0of3
        toks, paths = front_paths(TEST, rows)
        cf = encode(paths, "wa").astype(np.float32)
        ref = np.load(FEATS / "WA" / f"{TEST}.s0of3.npz")
        pos = {t: i for i, t in enumerate(ref["tokens"].tolist())}
        rc = ref["Cf"][[pos[t] for t in toks]].astype(np.float32)
        cos = (cf * rc).sum(-1) / (np.linalg.norm(cf, axis=-1) * np.linalg.norm(rc, axis=-1) + 1e-9)
        eq = {"n": int(len(toks)), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_p01": float(np.percentile(cos, 1)),
              "rel_l2": float(np.linalg.norm(cf - rc) / np.linalg.norm(rc)), "pass": bool(cos.mean() >= 0.999)}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "vj21_equivalence.json").write_text(json.dumps(eq, indent=1))
        run.summary["equivalence"] = eq
        run.info(f"equivalence (WA front path vs cached WA-Cf): {eq}")
        if not eq["pass"]:
            raise SystemExit(3)
        # 2. original V-JEPA 2.1 weights into the same encoder (WA-JEPA's own loader); every encoder tensor must come from the checkpoint
        payload = torch.load(VJ21_CKPT, map_location="cpu", weights_only=False)
        st = _select_checkpoint_state(payload, ("target_encoder", "ema_encoder", "encoder"))
        before = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
        ratio = _load_submodule_state(model.encoder, st, log_prefix="vjepa2_1.encoder", min_load_ratio=0.95, allow_partial=False)
        after = model.encoder.state_dict()
        unchanged = [k for k in after if torch.equal(after[k].cpu(), before[k].cpu())]
        load = {"ckpt": str(VJ21_CKPT), "keys_in_payload": sorted(payload)[:20] if isinstance(payload, dict) else None,
                "loaded_numel_ratio": float(ratio), "encoder_tensors": len(after), "unchanged_vs_wa": unchanged}
        (OUT / "vj21_load.json").write_text(json.dumps(load, indent=1))
        run.info(f"V-JEPA 2.1 load: ratio {ratio:.4f}, {len(unchanged)} of {len(after)} encoder tensors equal to WA's: {unchanged[:10]}")
        if ratio < 0.999:
            raise SystemExit(f"V-JEPA 2.1 checkpoint covers only {ratio:.4f} of the encoder")
        del payload, st, before
        # 3. extraction
        d = FEATS / "VJ21"
        d.mkdir(parents=True, exist_ok=True)
        for data in a.datas:
            f = d / f"{data}{'-lim' if a.limit else ''}.npz"
            if f.exists() and not a.force:
                continue
            n = len(np.load(CR / data / "tab.npz")["names"])
            t0 = time.time()
            toks, paths = front_paths(data, np.arange(n)[: a.limit or None])
            x = encode(paths, "vj21")
            np.savez(f, tokens=toks, raw=x)
            run.info(f"{data}: {len(toks)} tokens in {time.time() - t0:.0f} s -> {f}")
            run.summary[f"n_{data}"] = int(len(toks))


# ---------------------------------------------------------------- x4 (op-train env)
def cmd_x4(a):
    import torch
    from collections import deque
    import pp_prep as P
    import op_lb as OL
    import pp_unfreeze as U
    from jevdrive import op_adapt as A
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", "rep-x4", config=vars(a)) as run:
        net = A.load("cinque", torch.float16).to(dev).eval()
        d = FEATS / "X4"
        d.mkdir(parents=True, exist_ok=True)
        for data in a.datas:
            lim = "-lim" if a.limit else ""
            f = d / f"{data}{lim}.npy"
            if f.exists() and not a.force:
                continue
            names = np.asarray(P.full_meta(data)[0]["names"] if data.startswith("navtrain_full") else OL.meta(data)["names"])
            tab = np.load(CR / data / "tab.npz")
            assert (names == tab["names"]).all(), f"{data}: render rows differ from tab.npz"
            N = len(names) if not a.limit else min(a.limit, len(names))
            fr = U.Frames([data], "warp", a.workers or max(1, n_cpus() - 2))
            o = np.lib.format.open_memmap(d / f".{data}{lim}.tmp.npy", "w+", np.float16, (N, *U.X4_SHAPE))
            chunks = [np.arange(i, min(i + 16, N)) for i in range(0, N, 16)]
            q, t0, nxt = deque(), time.time(), 6
            for c in chunks[:6]:
                q.append((c, fr.submit(c)))
            for j in range(len(chunks)):
                rows, futs = q.popleft()
                if nxt < len(chunks):
                    q.append((chunks[nxt], fr.submit(chunks[nxt]))); nxt += 1   # noqa: E702
                prev, cur = U.Frames.gather(futs)                                  # (n, 8 slots, 2, 6, 128, 256)
                p, c = (torch.from_numpy(np.ascontiguousarray(x[:, -1])).to(dev) for x in (prev, cur))   # t0 slot only
                with torch.no_grad():
                    h = net.run_batched(A.vision_feeds(p, c), [U.X4])[U.X4]
                o[rows] = h.reshape(len(rows), *U.X4_SHAPE).cpu().numpy().astype(np.float16)
                if j % 100 == 0:
                    run.status(f"{data}: {rows[-1] + 1}/{N} ({(rows[-1] + 1) / (time.time() - t0):.1f}/s)")
            fr.close()
            o.flush()
            del o
            os.replace(d / f".{data}{lim}.tmp.npy", f)
            np.save(d / f"{data}{lim}.tokens.npy", names[:N])
            x = np.load(f, mmap_mode="r")
            ref = np.load(CR / f"{data}@warp" / "front.npy", mmap_mode="r")
            with torch.no_grad():
                hv = net.run_batched({U.X4: torch.from_numpy(np.ascontiguousarray(x[:64])).to(dev)[:, None]}, ["view_39"])["view_39"]
            hv = hv.reshape(-1, *A.H_SHAPE).float().cpu().numpy()
            r = ref[:64, -1].astype(np.float32)
            eq = {"max_abs": float(np.abs(hv - r).max()), "mean_abs": float(np.abs(hv - r).mean()), "rms_ref": float(np.sqrt((r ** 2).mean())),
                  "tokens_per_s": N / (time.time() - t0)}
            run.summary[f"eq_{data}"] = eq
            run.info(f"{data}: {N} tokens, x4 -> view_39 vs W front cache t0 slot: {eq}")
            if eq["mean_abs"] > 0.02 * eq["rms_ref"]:
                raise SystemExit(f"{data}: X4 t0 slot does not reproduce the W front cache ({eq})")


# ---------------------------------------------------------------- feature access (op-train env)
def _load_tok(kind, data):
    """(tokens, array (N, ...)) of one source for one cache dir (no row selection)."""
    if kind == "V":
        z = np.load(FEATS / "P2-F-s0" / f"{data}.npz")
        return z["tokens"], z["V"]
    if kind == "WA":
        zs = [np.load(f) for f in sorted((FEATS / "WA").glob(f"{data}.s*of3.npz"))]
        return np.concatenate([z["tokens"] for z in zs]), np.concatenate([z["Cf"] for z in zs])
    if kind in ("VJ21", "VJ21raw"):
        z = np.load(FEATS / "VJ21" / f"{data}.npz")
        return z["tokens"], z["raw"]
    if kind == "X4":
        return np.load(FEATS / "X4" / f"{data}.tokens.npy"), np.load(FEATS / "X4" / f"{data}.npy", mmap_mode="r")
    raise ValueError(kind)


def source(kind, toks, datas, pca=None):
    """(n, d) float32 features of `toks` (cache dir per token in `datas`); VJ21 = raw tokens through the PCA (per token, 1024 -> 512)."""
    out = None
    for data in dict.fromkeys(datas):
        m = datas == data
        tk, X = _load_tok(kind, data)
        pos = {t: i for i, t in enumerate(tk.tolist())}
        ix = np.array([pos[t] for t in toks[m]])
        o = np.argsort(ix)
        x = np.empty((len(ix),) + X.shape[1:], np.float32)
        x[o] = X[ix[o]]                                                   # sorted reads (memory maps), back in the token order
        if kind == "X4":
            x = x.reshape(len(x), 2048, 32).transpose(0, 2, 1)             # (n, 32 cells (4 x 8, row-major), 2048)
        if kind == "VJ21":
            x = (x - pca["mean"]) @ pca["comp"].T
        x = x.reshape(len(x), -1)
        if out is None:
            out = np.zeros((len(toks), x.shape[1]), np.float32)
        out[m] = x
    return out


def fit_pca(toks, datas, k=512):
    """Per-token PCA of the raw VJ21 tokens (every token of every train row is a sample) -> mean (1024,), comp (k, 1024), explained."""
    import torch
    X = source("VJ21raw", toks, datas).reshape(-1, 1024)
    Xt = torch.as_tensor(X, device="cuda", dtype=torch.float64)
    mu = Xt.mean(0)
    C = ((Xt - mu).T @ (Xt - mu)) / (len(Xt) - 1)
    ev, Q = torch.linalg.eigh(C)
    ev, Q = ev.flip(0), Q.flip(1)
    return {"mean": mu.float().cpu().numpy(), "comp": Q[:, :k].T.float().cpu().numpy(), "explained": float(ev[:k].sum() / ev.sum())}


def sets():
    """navtest strata: T20 / T45 / S5 masks over tab order, eval-token mask, log ids."""
    tab = np.load(CR / TEST / "tab.npz")
    dy = np.abs(np.degrees(tab["fut"][:, -1, 2]))
    ev = set(t.strip() for t in open(PROBE / "sets" / "eval_tokens.txt") if t.strip())
    return dict(tokens=tab["names"], log=tab["log"], T20=dy > 20, T45=dy > 45, S5=dy < 5, eval=np.array([t in ev for t in tab["names"]]), dyaw=dy)


# ---------------------------------------------------------------- decode (op-train env)
def cmd_decode(a):
    import torch
    import torch.nn as nn
    import opb_probe as P
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "rep-decode", seed=0, config=vars(a)) as run:
        toks, datas, is_dev, dvs = P.train_tokens(False)
        run.use_split(dvs), run.use_split(splits.load("navsim/navtrain")), run.use_split(splits.load("navsim/navtest"))
        lab_pos, LZ = P.labels("navtrain_s23456")
        li = np.array([lab_pos[t] for t in toks])
        tabs = {d: np.load(CR / d / "tab.npz") for d in dict.fromkeys(datas)}
        fut = np.concatenate([tabs[d]["fut"] for d in dict.fromkeys(datas)])
        ego = np.concatenate([tabs[d]["ego"] for d in dict.fromkeys(datas)]).astype(np.float32)
        use = LZ["ok"][li] & ~np.isnan(fut[:, 0, 0]) & ~is_dev
        sdf = torch.as_tensor(LZ["sdf"][li[use]], device=dev)[:, None]
        Y = torch.as_tensor(fut[use], device=dev).float()
        S = sets()
        ev = S["T20"] | S["S5"] | S["eval"]
        tt = S["tokens"][ev]
        ttab = np.load(CR / TEST / "tab.npz")
        e_ego = ttab["ego"][ev].astype(np.float32)
        ttd = np.array([TEST] * len(tt))
        trt, trd = toks[use], datas[use]
        pca = fit_pca(trt, trd)
        np.savez(OUT / "vj21_pca.npz", **pca)
        run.info(f"VJ21 PCA 1024 -> 512 on {use.sum()} train rows x 32 tokens: explained variance {pca['explained']:.4f}")
        run.summary["pca_explained"] = pca["explained"]
        M = torch.as_tensor(P._interp_matrix(), device=dev, dtype=torch.float32)
        C = torch.as_tensor(P.CORNERS, device=dev, dtype=torch.float32)
        res, rows, junc = {"tokens": tt}, [], []
        jt = __import__("pandas").read_parquet(PROBE / "joint" / "navtest_tokens.parquet").set_index("token").loc[S["tokens"]]
        jy = (jt.junction_path & ~jt.junction_t0).to_numpy()
        arms = [x for x in a.arms if x in ARMS]
        for name in arms:
            src = ARMS[name]
            Xtr = np.concatenate([source(k, trt, trd, pca) for k in src] + [ego[use]], 1)
            Xa = torch.as_tensor(Xtr, device=dev)
            del Xtr
            mu, sd = Xa.mean(0), Xa.std(0).clamp_min(1e-6)
            Xa = (Xa - mu) / sd
            Xe = (torch.as_tensor(np.concatenate([source(k, tt, ttd, pca) for k in src] + [e_ego], 1), device=dev) - mu) / sd
            torch.manual_seed(0)
            net = nn.Sequential(nn.Dropout(0.1), nn.Linear(Xa.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 24)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            wu = max(1, a.steps // 20)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / wu) * 0.5 * (1 + np.cos(np.pi * min(k, a.steps) / a.steps)))
            g = torch.Generator(device=dev).manual_seed(0)
            t1 = time.time()
            for _ in range(a.steps):                                    # opb_probe cmd_decode's hinge loop, unchanged
                b = torch.randint(0, len(Xa), (a.batch,), device=dev, generator=g)
                Pp = net(Xa[b]).view(-1, 8, 3)
                li_ = nn.functional.huber_loss(Pp[..., :2], Y[b, :, :2], delta=1.0) + 3.0 * nn.functional.huber_loss(Pp[..., 2], Y[b, :, 2], delta=0.1)
                v = P.sdf_at(sdf[b].float(), P.corners_torch(Pp, M, C))
                loss = li_ + a.lam_hinge * torch.relu(a.hinge_margin - v).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            net.eval()
            with torch.no_grad():
                Pe = torch.cat([net(Xe[i:i + 2048]).view(-1, 8, 3) for i in range(0, len(Xe), 2048)]).cpu().numpy()
            res[name] = Pe.astype(np.float32)
            rows.append(dict(arm=name, dim=int(Xa.shape[1]), final_loss=float(loss), imit=float(li_), train_s=time.time() - t1))
            run.info(json.dumps(rows[-1]))
            del Xa, Xe, net, opt
            torch.cuda.empty_cache()
            if name in a.junction_arms:
                junc.append(dict(arm=name, **junction_probe(name, src, S, ttab, jy, pca, dev)))
                run.info(json.dumps(junc[-1]))
        np.savez(OUT / "decoder_poses.npz", **res)
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT / "fits.csv", index=False)
        pd.DataFrame(junc).to_csv(OUT / "junction_probe.csv", index=False)
        for nm, m in (("t20", S["T20"]), ("rest_eval", S["eval"] & ~S["T20"]), ("rest_s5", S["S5"] & ~S["eval"] & ~S["T20"])):
            (OUT / f"tokens_{nm}.txt").write_text("\n".join(S["tokens"][m]) + "\n")
        run.summary.update(n_eval=int(ev.sum()), arms=arms, out=str(OUT / "decoder_poses.npz"))


def junction_probe(name, src, S, ttab, y, pca, dev, folds=5, steps=400, wd=1e-3):
    """Report-only junction look-ahead probe (prereg 6.1 secondary): logistic on z-scored [X, E] over all navtest tokens, 5-fold cross-fit by
    log, AdamW 400 full-batch steps (lr 1e-2), L2 penalty 1e-3 x |w|^2 (fixed, not tuned); AUC with a log-cluster bootstrap CI."""
    import torch
    import opb_probe as P
    toks = S["tokens"]
    X = np.concatenate([source(k, toks, np.array([TEST] * len(toks)), pca) for k in src] + [ttab["ego"].astype(np.float32)], 1)
    ul = np.unique(S["log"])
    fold = dict(zip(ul, np.random.default_rng(0).permutation(len(ul)) % folds))
    f = np.array([fold[g] for g in S["log"]])
    score = np.zeros(len(toks))
    for k in range(folds):
        tr, te = f != k, f == k
        Xt = torch.as_tensor(X[tr], device=dev)
        mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
        Xt = (Xt - mu) / sd
        yt = torch.as_tensor(y[tr], device=dev, dtype=torch.float32)
        torch.manual_seed(0)
        w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.AdamW([w, b], lr=1e-2, weight_decay=0.0)
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + wd * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            score[te] = (((torch.as_tensor(X[te], device=dev) - mu) / sd) @ w + b).cpu().numpy()
        del Xt
    lo, hi = P.auc_ci(score, y, S["log"], B=2000)
    return dict(auc=P.auc(score, y), auc_lo=lo, auc_hi=hi, n=int(len(y)), pos=int(y.sum()))


# ---------------------------------------------------------------- mem (CPU): stage-1 memory banks
def cmd_mem(a):
    pca = dict(np.load(OUT / "vj21_pca.npz")) if a.kind == "vj21" else None
    d = MEM / a.kind
    d.mkdir(parents=True, exist_ok=True)
    for data in a.datas:
        names = np.load(CR / data / "tab.npz")["names"]
        x = source("VJ21" if a.kind == "vj21" else "WA", names, np.array([data] * len(names)), pca).reshape(len(names), 32, 512)
        np.save(d / f"{data}.npy", x.astype(np.float16))
        print(f"{a.kind} {data}: {x.shape}, rms {np.sqrt((x ** 2).mean()):.3f} -> {d / f'{data}.npy'}", flush=True)


# ---------------------------------------------------------------- report (CPU)
def boot_idx(groups, B, seed=0):
    ug, inv = np.unique(groups, return_inverse=True)
    return inv, len(ug), np.random.default_rng(seed).integers(0, len(ug), (B, len(ug)))


def cl_mean(v, inv, nu, idx):
    """Point mean and the (B,) cluster-bootstrap replicate means (ratio of sums over resampled logs)."""
    s, c = np.bincount(inv, v, nu), np.bincount(inv, None, nu)
    return s.sum() / c.sum(), s[idx].sum(1) / c[idx].sum(1)


def cmd_report(a):
    import pandas as pd
    S = sets()
    df = pd.concat([pd.read_csv(OUT / f"score_{k}.csv") for k in ("t20", "rest_eval", "rest_s5") if (OUT / f"score_{k}.csv").exists()])
    fail = (1 - df.pivot_table(index="token", columns="key", values="drivable_area_compliance"))
    row = {t: i for i, t in enumerate(S["tokens"].tolist())}
    B = 4000
    out = {"B": B, "rates": {}, "diffs": {}, "closure": {}, "repro": {}}

    def rates(mask, keys, tag):
        tk = S["tokens"][mask]
        lg = S["log"][mask]
        inv, nu, idx = boot_idx(lg, B)
        f = fail.reindex(tk)
        res = {}
        for k in keys:
            if k not in f or f[k].isna().any():
                continue
            m, bs = cl_mean(f[k].to_numpy(float), inv, nu, idx)
            res[k] = (m, bs)
        return res, f, inv, nu, idx

    tabrows, gates = [], {}
    keys = [k for k in ARMS if k in fail.columns]
    for st in ("T20", "T45", "S5"):
        R_, f, inv, nu, idx = rates(S[st], keys, st)
        if not R_:
            continue
        V = R_.get("V")
        for k, (m, bs) in R_.items():
            r = {"stratum": st, "arm": k, "n": int(S[st].sum()), "fail_pct": 100 * m, "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)}
            if V is not None and k != "V":
                dlt = V[0] - m, V[1] - bs
                r |= {"V_minus_arm_pp": 100 * dlt[0], "d_lo": 100 * np.percentile(dlt[1], 2.5), "d_hi": 100 * np.percentile(dlt[1], 97.5)}
                if "V+WA" in R_:
                    den = V[0] - R_["V+WA"][0], V[1] - R_["V+WA"][1]
                    cb = dlt[1] / np.where(np.abs(den[1]) > 1e-9, den[1], np.nan)
                    r |= {"closure": dlt[0] / den[0], "c_lo": np.nanpercentile(cb, 2.5), "c_hi": np.nanpercentile(cb, 97.5)}
            tabrows.append(r)
    tb = pd.DataFrame(tabrows)
    RES.mkdir(parents=True, exist_ok=True)
    tb.to_csv(RES / "stage0.csv", index=False)
    g = lambda st, k, c: float(tb[(tb.stratum == st) & (tb.arm == k)][c].iloc[0]) if ((tb.stratum == st) & (tb.arm == k)).any() else np.nan  # noqa: E731
    # reproduction: stratified full-navtest and T20 rates on the eval tokens (opb_report.strat weights: F / R / FF 1, PP n_PP / n_PP scored)
    Z = np.load(PROBE / "sets" / "navtest_sets.npz")
    pos = {t: i for i, t in enumerate(Z["tokens"].tolist())}
    zpp = Z["PP"][[pos[t] for t in S["tokens"]]]
    w_pp = zpp.sum() / (zpp & S["eval"]).sum()                       # 11 494 / 1 500
    for k, ref_all, ref_t20 in (("V", 5.12, 10.9), ("WA", 3.41, 6.0)):
        for nm, msk, ref in (("all", S["eval"], ref_all), ("T20", S["eval"] & S["T20"], ref_t20)):
            tk = S["tokens"][msk]
            w = np.where(zpp[msk], w_pp, 1.0)
            v = fail.reindex(tk)[k].to_numpy(float)
            est = 100 * (w * v).sum() / w.sum()
            out["repro"][f"{k}_{nm}"] = {"est": est, "ref": ref, "diff": est - ref, "tol": 0.3 if nm == "all" else 0.7,
                                         "pass": bool(abs(est - ref) <= (0.3 if nm == "all" else 0.7))}
    # gates (prereg 6.1)
    pc = g("T20", "V+WA", "V_minus_arm_pp"), g("T20", "V+WA", "d_lo")
    gates["positive_control"] = {"V_minus_VWA_pp": pc[0], "lo": pc[1], "pass": bool(pc[0] >= 2.5 and pc[1] > 0)}
    s5 = g("S5", "V+VJ21", "fail_pct") - g("S5", "V", "fail_pct")
    gates["vj21"] = {"closure": g("T20", "V+VJ21", "closure"), "V_minus_VVJ21_pp": g("T20", "V+VJ21", "V_minus_arm_pp"),
                     "lo": g("T20", "V+VJ21", "d_lo"), "S5_VVJ21_minus_V_pp": s5}
    gates["vj21"]["pass"] = bool(gates["vj21"]["closure"] >= 0.5 and gates["vj21"]["lo"] > 0 and s5 <= 0.3)
    cx = g("T20", "X4", "closure")
    gates["x4"] = {"closure": cx, "verdict": "(c2) / stage-4 (a) viable" if cx >= 0.5 else ("(a) needs the whole encoder" if cx < 0.2 else "between: describe only")}
    gates["repro_pass"] = all(v["pass"] for v in out["repro"].values())
    out["gates"] = gates
    (RES / "stage0.json").write_text(json.dumps(out | {"table": tabrows}, indent=1, default=float))
    print(tb.to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps({"repro": out["repro"], "gates": gates}, indent=1, default=float))


# ---------------------------------------------------------------- stage 1 (CPU): early stop on seed 0, gates on the 2-seed means
S1_ARMS = {"H0": "RH0", "MW": "RMW", "MV": "RMV"}
S1 = D / "runs" / "op_parity" / "rep" / "s1"


def s1_units(spec):
    from jevdrive.bench import tables as BT
    u = BT.load("navtest", spec)[0]
    if u is None:
        raise SystemExit(f"navtest: no result for {spec}")
    return u


def s1_frame(specs, S):
    """Per-token metrics (x 100) averaged over the seeds of one arm: EPDMS, DAC / NC+TTC fail, EP, in tab order of navtest."""
    import pandas as pd
    us = [s1_units(x).reindex(S["tokens"]) for x in specs]
    assert all(not u.score.isna().any() for u in us), f"missing navtest tokens in {specs}"
    f = lambda u: pd.DataFrame({"EPDMS": 100 * u.score, "DAC fail %": 100.0 * (u.DAC < 1), "NC+TTC fail %": 100.0 * ((u.NC < 1) | (u.TTC < 1)),  # noqa: E731
                                "EP": 100 * u.EP})
    return sum(f(u) for u in us) / len(us)


def s1_paired(A, Bm, mask, S, col, nb=4000):
    from jevdrive import stats
    r = stats.paired(A[col].to_numpy()[mask], Bm[col].to_numpy()[mask], groups=S["log"][mask], n_boot=nb)
    return {"arm": r["mean_a"], "ref": r["mean_b"], "diff": r["mean"], "lo": r["lo"], "hi": r["hi"], "n": int(mask.sum())}


def speed_ratio(spec):
    from jevdrive.bench.compat import pred_file
    z = np.load(pred_file(spec))
    fz = np.load(D / "runs/navsim_zs/index/navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    F = np.stack([fut[k] for k in z["tokens"]])
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    mv = plen(F) > 2.0
    return float(np.median(plen(z["poses"])[mv] / plen(F)[mv]))


def drift_off(tag):
    import glob
    fs = sorted(glob.glob(str(D / "runs" / "op_parity" / f"train-{tag}" / "*" / "DONE")))
    return json.loads(open(fs[-1]).read()).get("dev_drift_off", np.nan) if fs else np.nan


def cmd_s1gate(a):
    """Seed-0 early stop (prereg 6.2): an arm continues iff its T20 DAC failure rate drops >= 0.4 pp vs H0-s0 and navtest EPDMS diff >= 0.
    Prints the continuing arms; exit 0 if any continues, 2 if the line stops."""
    S = sets()
    H = s1_frame([f"{S1_ARMS['H0']}-F-s0"], S)
    res, keep = {}, []
    for arm in a.arms:
        A = s1_frame([f"{S1_ARMS[arm]}-F-s0"], S)
        t20 = s1_paired(A, H, S["T20"], S, "DAC fail %", a.nb)
        ep = s1_paired(A, H, np.ones(len(S["tokens"]), bool), S, "EPDMS", a.nb)
        ok = (-t20["diff"] >= 0.4) and (ep["diff"] >= 0)
        res[arm] = {"T20 DAC fail %": t20, "EPDMS": ep, "continue": bool(ok)}
        keep += [arm] if ok else []
    res["rule"] = "stop an arm if its seed-0 T20 DAC failure drop vs H0-s0 < 0.4 pp or its navtest EPDMS diff < 0; stop the line if no arm continues"
    res["continue"] = keep
    S1.mkdir(parents=True, exist_ok=True)
    (S1 / "gate-s0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float), file=_sys.stderr)
    print(" ".join(keep))
    raise SystemExit(0 if keep else 2)


def cmd_s1report(a):
    """Stage-1 tables on the seed means (prereg 6.2): EPDMS, T20 / T45 DAC, S5 EPDMS, EP, NC+TTC, speed ratio, drift_off, memory-off reads."""
    import pandas as pd
    S = sets()
    allm = np.ones(len(S["tokens"]), bool)
    seeds = a.seeds
    H = s1_frame([f"{S1_ARMS['H0']}-F-s{k}" for k in seeds], S)
    sr_h = np.mean([speed_ratio(f"{S1_ARMS['H0']}-F-s{k}") for k in seeds])
    rows, gates = [], {}
    for arm in ["H0"] + a.arms:
        for off in ([False] if arm == "H0" else [False, True]):
            specs = [f"{S1_ARMS[arm]}-F-s{k}" + (":noside" if off else "") for k in seeds]
            A = s1_frame(specs, S)
            lab = arm + (" memory off" if off else "")
            r = {"arm": lab, "seeds": len(seeds)}
            for nm, m, col in (("EPDMS", allm, "EPDMS"), ("T20 DAC fail %", S["T20"], "DAC fail %"), ("T45 DAC fail %", S["T45"], "DAC fail %"),
                               ("S5 EPDMS", S["S5"], "EPDMS"), ("EP", allm, "EP"), ("NC+TTC fail %", allm, "NC+TTC fail %"), ("DAC fail %", allm, "DAC fail %")):
                p = s1_paired(A, H, m, S, col, a.nb)
                r[nm] = p["arm"]
                if arm != "H0":
                    r[f"{nm} diff"], r[f"{nm} lo"], r[f"{nm} hi"] = p["diff"], p["lo"], p["hi"]
            sr = np.mean([speed_ratio(x) for x in specs])
            r["speed ratio"], r["speed ratio / H0"] = sr, sr / sr_h
            r["drift_off m"] = np.mean([drift_off(f"{S1_ARMS[arm]}-F-s{k}") for k in seeds]) if not off else np.nan
            rows.append(r)
            if arm != "H0" and not off:
                g = {"EPDMS": r["EPDMS diff"] >= 0.30 and r["EPDMS lo"] > 0, "T20": r["T20 DAC fail % diff"] <= -1.0 and r["T20 DAC fail % hi"] < 0,
                     "EP": r["EP diff"] >= -0.2, "NC+TTC": r["NC+TTC fail % diff"] <= 0.2, "S5": r["S5 EPDMS diff"] >= -0.2,
                     "speed": abs(r["speed ratio / H0"] - 1) <= 0.05, "drift_off": r["drift_off m"] <= 0.10}
                gates[arm] = {k: bool(v) for k, v in g.items()} | {"pass": bool(all(g.values()))}
    if "MW" in a.arms and "MV" in a.arms:
        ew, ev = (next(r["EPDMS diff"] for r in rows if r["arm"] == k) for k in ("MW", "MV"))
        gates["attribution (MV - H0) / (MW - H0), EPDMS"] = ev / ew if abs(ew) > 1e-9 else np.nan
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"{a.out}.csv", index=False)
    (RES / f"{a.out}.json").write_text(json.dumps({"seeds": seeds, "gates": gates, "rows": rows}, indent=1, default=float))
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(gates, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("vj21")
    p.add_argument("--datas", nargs="+", default=[TEST, *TRAIN])
    p.add_argument("--check-n", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("x4")
    p.add_argument("--datas", nargs="+", default=[TEST, *TRAIN])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("decode")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--junction-arms", nargs="*", default=["E", "V", "WA", "VJ21", "X4"])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lam-hinge", type=float, default=10.0)
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p = sp.add_parser("mem")
    p.add_argument("--kind", required=True, choices=["wa_cf", "vj21"])
    p.add_argument("--datas", nargs="+", default=[*TRAIN, TEST])
    p = sp.add_parser("report")
    p = sp.add_parser("s1gate")
    p.add_argument("--arms", nargs="+", default=["MW", "MV"])
    p.add_argument("--nb", type=int, default=4000)
    p = sp.add_parser("s1report")
    p.add_argument("--arms", nargs="+", default=["MW", "MV"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--out", default="stage1")
    p.add_argument("--nb", type=int, default=4000)
    a = ap.parse_args()
    {"vj21": cmd_vj21, "x4": cmd_x4, "decode": cmd_decode, "mem": cmd_mem, "report": cmd_report, "s1gate": cmd_s1gate,
     "s1report": cmd_s1report}[a.cmd](a)
