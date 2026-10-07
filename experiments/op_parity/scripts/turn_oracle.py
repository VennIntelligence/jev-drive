"""op_parity turn-oracle (plans/2026-10-08-turn-oracle-prereg.md): the true drivable-boundary geometry as a privileged input of the P2H plan
pathway. Privileged inputs are an oracle probe only, never a method. Training is pp_train.py --mem <kind>; navtest scores come from jevdrive.bench.

  bank    (op-train, GPU, pool)  memory banks runs/op_parity/mem/<kind>/<data>.npy (N, 32, 512) fp16 for pp_train --mem / bench:
                                 sdf_gt   the decision-148 SDF label raster of the token (128 x 96 cells of 0.5 m), lossless
                                 sdf_shuf the same rows permuted inside the data dir, always to another log (matched control)
                                 sdf_wa / sdf_v  the turn_probe MLP read-out (1 m raster, bilinear to 0.5 m) from WA-Cf / Cinque view_39:
                                          navtrain = 5-fold out-of-fold over logs, navtest = turn_probe's stored full-fit prediction
  replay  (navsim2, CPU, pool)   four_dirs' instrumented devkit replay (fd_navsim._init / work, unchanged) of every listed model on its own
                                 navtest DAC-failure tokens: side / depth / time of the first footprint departure -> .../replay_<name>.parquet
  gate    (op-train, CPU)        seed-0 early stop of the prereg: exit 2 = clear negative
  decode  (op-train, GPU, pool)  step 2 stage A, the fresh-head ceiling: decision 147 / 160's thin decoder (rep.py decode's net and hinge loop,
                                 unchanged) on [V, E], [V, true 1 m SDF, E], [V, shuffled SDF, E], [true SDF, E] -> decode/poses.npz (navtest T20)
  areport (op-train, CPU)        stage A table from `replay --poses` (+ the bench score-poses CSV as the DAC check) -> report-decode/
  report  (op-train, CPU)        tables, verdict, figures -> $DATA_DIR/runs/op_parity/turn_oracle/report[-<name>]/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent), str(_R / "experiments/op_probe/scripts")]
import argparse, json, os, zlib  # noqa: E401,E402

import numpy as np  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs" / "op_parity" / "turn_oracle"
MEM = D / "runs" / "op_parity" / "mem"
CR = D / "runs" / "op_parity" / "cache"
TRAIN = ("navtrain_full.s2of12", "navtrain_full.s3of12", "navtrain_full.s4of12")
TEST = "lb_navtest"
LAB = {**{d: "navtrain_s23456" for d in TRAIN}, TEST: "navtest"}
NTOK, DTOK, ROWS, COLS = 32, 512, 4, 96                    # token k = raster rows 4k .. 4k + 3 (a 2 m strip over the full +-24 m width)
CLIP, SCALE, ONES = 6.0, 3.0, slice(384, 448)              # value = clip(sdf, +-6 m) / 3; dims 384-447 = 1 (scale reference under LayerNorm)
FOLDS = 5
# arm -> checkpoint stem (tag = <stem>-F-s<seed>); H0 / MW are the decision-160 stage-1 runs, reused
STEM = {"H0": "RH0", "OS": "TOS", "OG": "TOG", "PW": "TOW", "PV": "TOV", "MW": "RMW"}
STEM |= {"OS9": "TOS9", "OG9": "TOG9", "OSh": "TOSH", "OGh": "TOGH"}      # step 2 stage B: 9 000 steps / hinge lambda 30, margin 0.5
KIND = {"OS": "sdf_shuf", "OG": "sdf_gt", "PW": "sdf_wa", "PV": "sdf_v"}
LABEL = {"H0": "H0: P2H pilot (no memory)", "OS": "OS: shuffled SDF (matched control)", "OG": "OG: true SDF (oracle)",
         "PW": "PW: SDF read out of WA-Cf", "PV": "PV: SDF read out of Cinque", "MW": "MW: WA-Cf tokens (decision 160)",
         "OG off": "OG, memory masked at test"}
NB = 4000


def spec(arm, seed):
    """'OG' / 'OG off' / '<stem>' (step-2 arms pass their own stem) -> bench model spec."""
    off = arm.endswith((" off", ":off"))                                     # ':off' = the shell-safe spelling
    a = arm[:-4] if off else arm
    return f"{STEM.get(a, a)}-F-s{seed}" + (":noside" if off else "")


# ---------------------------------------------------------------- bank
def encode(sdf05):
    """(N, 128, 96) SDF in m -> (N, 32, 512) fp16 memory tokens."""
    n = len(sdf05)
    out = np.zeros((n, NTOK, DTOK), np.float16)
    out[..., :ROWS * COLS] = (np.clip(sdf05.astype(np.float32), -CLIP, CLIP) / SCALE).reshape(n, NTOK, ROWS * COLS)
    out[..., ONES] = 1.0
    return out


def up2(r1):
    """(N, 64, 48) 1 m raster -> (N, 128, 96) 0.5 m, bilinear (the inverse of opb_probe.raster1m's 2 x 2 mean up to curvature)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(r1, dtype=np.float32))[:, None]
    return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)[:, 0].numpy()


def band_mae(pred05, true05, band=2.0):
    """Per-row |pred - true| over the 0.5 m cells with |true| <= band, x in [0, 32) m, |y| < 16 m (turn_probe's band on the label grid)."""
    t, p = true05[:, 16:80, 16:80].astype(np.float32), pred05[:, 16:80, 16:80].astype(np.float32)
    m = np.abs(t) <= band
    with np.errstate(invalid="ignore"):
        return (np.abs(p - t) * m).sum((1, 2)) / m.sum((1, 2))


def derange(logs, rng):
    """A permutation that sends every row to a row of another log."""
    n = len(logs)
    perm = rng.permutation(n)
    for _ in range(100):
        bad = np.flatnonzero(logs[perm] == logs)
        if not len(bad):
            return perm
        for i, j in zip(bad, rng.integers(0, n, len(bad))):                  # sequential swaps keep it a permutation
            perm[i], perm[j] = perm[j], perm[i]
    raise RuntimeError("no derangement over logs")


def save_bank(kind, data, x):
    d = MEM / kind
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{data}.tmp.npy"
    np.save(tmp, x)
    os.replace(tmp, d / f"{data}.npy")


def cmd_bank(a):
    import torch
    import opb_probe as P
    import rep as REP
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "turn-oracle-bank", seed=0, config=vars(a)) as run:
        s_tr, s_dv, s_te = (splits.load(n) for n in ("navsim/op-parity-s234-train", "navsim/op-parity-s234-dev", "navsim/navtest"))
        for s in (s_tr, s_dv, s_te):
            run.use_split(s)
        stats = {"encoding": dict(tokens=NTOK, dims=DTOK, rows_per_token=ROWS, clip_m=CLIP, scale=SCALE, ones=[ONES.start, ONES.stop])}
        tabs = {d: np.load(CR / d / "tab.npz") for d in (*TRAIN, TEST)}
        gt, lz = {}, {}
        for d in (*TRAIN, TEST):
            if LAB[d] not in lz:
                lz[LAB[d]] = P.labels(LAB[d])
            pos, Z = lz[LAB[d]]
            ix = np.array([pos.get(t, -1) for t in tabs[d]["names"].tolist()])
            ok = (ix >= 0) & Z["ok"][np.maximum(ix, 0)]
            g = Z["sdf"][np.maximum(ix, 0)].astype(np.float16)
            g[~ok] = 0
            gt[d] = g
            stats[f"label_coverage/{d}"] = float(ok.mean())
            assert ok.mean() > 0.99, (d, ok.mean())
        # ---- sdf_gt, sdf_shuf
        rng = np.random.default_rng(0)
        for d in (*TRAIN, TEST):
            bank = encode(gt[d])
            save_bank("sdf_gt", d, bank)
            lg = tabs[d]["log"]
            perm = derange(lg, rng)
            assert not (lg[perm] == lg).any()
            save_bank("sdf_shuf", d, bank[perm])
            np.save(MEM / "sdf_shuf" / f"{d}.perm.npy", perm)
            stats[f"rms/{d}"] = float(np.sqrt((bank[:2000].astype(np.float32) ** 2).mean()))
            stats[f"shuf_band_mae_vs_true/{d}"] = float(np.nanmean(band_mae(gt[d][perm][:4000], gt[d][:4000])))
            run.info(f"sdf_gt / sdf_shuf {d}: {bank.shape}, rms {stats[f'rms/{d}']:.3f}, shuffled-vs-true band MAE {stats[f'shuf_band_mae_vs_true/{d}']:.2f} m")
        # resolution check: the true raster through the predicted arms' path (1 m mean, bilinear back)
        g = gt[TEST].astype(np.float32)
        stats["resolution_band_mae/navtest"] = float(np.nanmean(band_mae(up2(P.raster1m(g)), g)))
        run.info(f"true SDF at 1 m, bilinear back to 0.5 m: band MAE {stats['resolution_band_mae/navtest']:.3f} m on navtest")
        # ---- sdf_wa, sdf_v: turn_probe's MLP, out-of-fold on navtrain
        toks, datas, _, _ = P.train_tokens(False)
        assert tuple(dict.fromkeys(datas)) == TRAIN and (np.concatenate([tabs[d]["names"] for d in TRAIN]) == toks).all()
        cat = lambda k: np.concatenate([tabs[d][k] for d in TRAIN])  # noqa: E731
        fut, ego, log = cat("fut"), cat("ego").astype(np.float32), cat("log")
        sdf = np.concatenate([gt[d] for d in TRAIN])
        import turn_probe as TP
        Yr = P.raster1m(sdf).reshape(len(toks), -1)
        Yc = TP.corridor(sdf, P.RES05, np.nan_to_num(fut))
        use = ~np.isnan(fut[:, 0, 0]) & s_tr.mask(toks)                       # the probe's train rows (turn_probe.cmd_fit)
        fold = np.array([zlib.crc32(g_.encode()) % FOLDS for g_ in log])
        zt = np.load(OUT.parent / "turn_probe" / "pred.npz")
        tpos = {t: i for i, t in enumerate(zt["tokens"].tolist()) if zt["test"][i]}
        ti = np.array([tpos[t] for t in tabs[TEST]["names"].tolist()])
        for kind, arm in (("sdf_wa", "WA"), ("sdf_v", "V")):
            X = np.concatenate([REP.source(arm, toks, datas, None), ego], 1)
            pr = np.zeros((len(toks), P.NH1, P.NW1), np.float32)
            for f in run.tqdm(range(FOLDS), desc=f"{kind} folds"):
                fit, ev = use & (fold != f), fold == f
                qr, _ = P.mlp_fit_predict(X[fit], [Yr[fit], Yc[fit]], X[ev], dev)
                pr[ev] = qr.reshape(-1, P.NH1, P.NW1)
                torch.cuda.empty_cache()
            up = up2(np.clip(pr, -P.CLIP, P.CLIP))
            e = band_mae(up, sdf)
            stats[f"{kind}/navtrain_oof_band_mae"] = float(np.nanmean(e[use]))
            o = 0
            for d in TRAIN:
                n = len(tabs[d]["names"])
                save_bank(kind, d, encode(up[o:o + n]))
                o += n
            upt = up2(zt[f"{arm}/mlp/raster"][ti].astype(np.float32))
            stats[f"{kind}/navtest_band_mae"] = float(np.nanmean(band_mae(upt, gt[TEST])))
            save_bank(kind, TEST, encode(upt))
            run.info(f"{kind}: band MAE navtrain out-of-fold {stats[f'{kind}/navtrain_oof_band_mae']:.3f} m, navtest {stats[f'{kind}/navtest_band_mae']:.3f} m")
            del X
        (OUT / "bank_stats.json").write_text(json.dumps(stats, indent=1))
        run.summary.update(stats)


# ---------------------------------------------------------------- replay (navsim2)
def cmd_replay(a):
    import multiprocessing as mp
    import pickle
    import pandas as pd
    import fd_navsim as FD
    from jevdrive.bench import tables as BT
    from jevdrive.bench.compat import pred_file
    from jevdrive.run import Run
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"replay_{a.name}.parquet"
    P, want = {}, {}
    if a.poses:                                                               # every key of a pose file on all of its tokens (stage A)
        z = np.load(a.poses)
        for k in z.files:
            if k != "tokens":
                P[k] = dict(zip(z["tokens"].tolist(), z[k].astype(np.float64)))
        want = {t: list(P) for t in z["tokens"].tolist()}
    for sp in a.models:
        u = BT.load("navtest", sp)[0]
        assert u is not None, f"no navtest result for {sp}"
        z = np.load(pred_file(sp))
        P[sp] = dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float64)))
        for t in u.index[u.DAC < 1]:
            want.setdefault(t, []).append(sp)
    kf = OUT / f"keys_{a.name}.pkl"
    pickle.dump({"plans": P, "want": want}, open(kf, "wb"))
    todo = sorted(want)
    procs = a.procs or len(os.sched_getaffinity(0))
    with Run("op_parity", f"turn-oracle-replay-{a.name}", config=vars(a) | {"n_tokens": len(todo), "procs": procs}) as run:
        rows = []
        with mp.get_context("fork").Pool(procs, initializer=FD._init, initargs=("navtest", kf)) as pool:
            for i, r in enumerate(pool.imap_unordered(FD.work, todo, chunksize=2)):
                for x in r["rows"]:
                    x.pop("_states")
                    rows.append(x)
                if (i + 1) % 250 == 0:
                    run.status(f"{i + 1}/{len(todo)} tokens")
        df = pd.DataFrame(rows)
        bad = 0 if a.poses else int((df.drivable_area_compliance >= 1).sum())   # every replayed (model, token) is a bench DAC failure
        run.info(f"{len(df)} (model, token) rows on {len(todo)} tokens; replay DAC disagrees with bench on {bad}; no LQR departure found on "
                 f"{int(((df.drivable_area_compliance < 1) & ~df.lqr_out.astype(bool)).sum())} DAC failures")
        df.to_parquet(out)
        run.summary.update(rows=len(df), tokens=len(todo), dac_disagree=bad, out=str(out))


# ---------------------------------------------------------------- step 2 stage A: fresh-head ceiling
DEC = {"V": ("V",), "V+G": ("V", "G"), "V+S": ("V", "S"), "G": ("G",)}


def cmd_decode(a):
    import time
    import torch
    import torch.nn as nn
    import opb_probe as P
    import rep as REP
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    out = OUT / "decode"
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "turn-oracle-decode", seed=0, config=vars(a)) as run:
        toks, datas, is_dev, dvs = P.train_tokens(False)
        run.use_split(dvs), run.use_split(splits.load("navsim/navtrain")), run.use_split(splits.load("navsim/navtest"))
        lab_pos, LZ = P.labels("navtrain_s23456")
        li = np.array([lab_pos[t] for t in toks])
        tabs = {d: np.load(CR / d / "tab.npz") for d in TRAIN}
        fut = np.concatenate([tabs[d]["fut"] for d in TRAIN])
        ego = np.concatenate([tabs[d]["ego"] for d in TRAIN]).astype(np.float32)
        log = np.concatenate([tabs[d]["log"] for d in TRAIN])
        use = LZ["ok"][li] & ~np.isnan(fut[:, 0, 0]) & ~is_dev                 # rep.py decode's train rows
        sdf05 = LZ["sdf"][li[use]]
        sdf = torch.as_tensor(sdf05, device=dev)[:, None]
        Y = torch.as_tensor(fut[use], device=dev).float()
        ttab = np.load(CR / TEST / "tab.npz")
        ev = np.abs(np.degrees(ttab["fut"][:, -1, 2])) > 20                     # navtest T20
        tt, e_ego = ttab["names"][ev], ttab["ego"][ev].astype(np.float32)
        tpos, TZ = P.labels("navtest")
        tsdf = TZ["sdf"][[tpos[t] for t in tt]]
        rng = np.random.default_rng(1)
        G = {"tr": P.raster1m(sdf05).reshape(int(use.sum()), -1), "te": P.raster1m(tsdf).reshape(len(tt), -1)}
        Sh = {"tr": G["tr"][derange(log[use], rng)], "te": G["te"][derange(ttab["log"][ev], rng)]}
        trt, trd, ttd = toks[use], datas[use], np.array([TEST] * len(tt))
        Mi = torch.as_tensor(P._interp_matrix(), device=dev, dtype=torch.float32)
        C = torch.as_tensor(P.CORNERS, device=dev, dtype=torch.float32)
        part = lambda k, w: (REP.source("V", trt, trd) if w == "tr" else REP.source("V", tt, ttd)) if k == "V" else (G if k == "G" else Sh)[w]  # noqa: E731
        res, rows = {"tokens": tt}, []
        for name, src in DEC.items():
            Xa = torch.as_tensor(np.concatenate([part(k, "tr") for k in src] + [ego[use]], 1), device=dev)
            mu, sd = Xa.mean(0), Xa.std(0).clamp_min(1e-6)
            Xa = (Xa - mu) / sd
            Xe = (torch.as_tensor(np.concatenate([part(k, "te") for k in src] + [e_ego], 1), device=dev) - mu) / sd
            torch.manual_seed(0)
            net = nn.Sequential(nn.Dropout(0.1), nn.Linear(Xa.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 24)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            wu = max(1, a.steps // 20)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / wu) * 0.5 * (1 + np.cos(np.pi * min(k, a.steps) / a.steps)))
            g = torch.Generator(device=dev).manual_seed(0)
            t1 = time.time()
            for _ in range(a.steps):                                            # rep.py cmd_decode's loop, unchanged
                b = torch.randint(0, len(Xa), (a.batch,), device=dev, generator=g)
                Pp = net(Xa[b]).view(-1, 8, 3)
                li_ = nn.functional.huber_loss(Pp[..., :2], Y[b, :, :2], delta=1.0) + 3.0 * nn.functional.huber_loss(Pp[..., 2], Y[b, :, 2], delta=0.1)
                v = P.sdf_at(sdf[b].float(), P.corners_torch(Pp, Mi, C))
                hl = torch.relu(a.hinge_margin - v).mean()
                loss = li_ + a.lam_hinge * hl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            net.eval()
            with torch.no_grad():
                Pe = torch.cat([net(Xe[i:i + 2048]).view(-1, 8, 3) for i in range(0, len(Xe), 2048)])
                hv = P.sdf_at(torch.as_tensor(tsdf, device=dev)[:, None].float(), P.corners_torch(Pe, Mi, C))
            res[name] = Pe.cpu().numpy().astype(np.float32)
            rows.append(dict(arm=name, dim=int(Xa.shape[1]), imit=float(li_), hinge=float(hl), train_s=time.time() - t1,
                             navtest_T20_raw_footprint_out=float((hv.min(1).values < 0).float().mean())))
            run.info(json.dumps(rows[-1]))
            del Xa, Xe, net, opt
            torch.cuda.empty_cache()
        np.savez(out / "poses.npz", **res)
        (out / "tokens.txt").write_text("\n".join(tt) + "\n")
        (out / "fits.json").write_text(json.dumps(rows, indent=1))
        run.summary.update(n_eval=len(tt), out=str(out / "poses.npz"))


def cmd_areport(a):
    import pandas as pd
    import fd_navsim as FD
    import turn_probe as TP
    from jevdrive import stats
    out = OUT / "report-decode"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "decode" / "poses.npz")
    tk = z["tokens"]
    tab = np.load(CR / TEST / "tab.npz")
    pos = {t: i for i, t in enumerate(tab["names"].tolist())}
    ix = np.array([pos[t] for t in tk])
    fut, log = tab["fut"][ix].astype(np.float64), tab["log"][ix]
    dpsi = FD.path_geom(fut)["dpsi"]
    sets = {"T20 (> 20 deg)": np.ones(len(tk), bool), "T45 (> 45 deg)": np.abs(dpsi) > 45}
    R = pd.read_parquet(OUT / "replay_decode.parquet")
    sc = pd.read_csv(a.score) if a.score else None
    F, chk = {}, {}
    for k in DEC:
        q = R[R.key == k].set_index("token").reindex(tk)
        assert q.drivable_area_compliance.notna().all(), k
        dac = (q.drivable_area_compliance < 1).to_numpy()
        if sc is not None:
            o = sc[sc.key == k].set_index("token").reindex(tk)
            chk[k] = dict(score_poses_dac_fail=int((o.drivable_area_compliance < 1).sum()), replay_dac_fail=int(dac.sum()),
                          disagree=int(((o.drivable_area_compliance < 1).to_numpy() != dac).sum()))
        inside = dac & (q.lqr_side.to_numpy(float) == np.sign(dpsi))
        with np.errstate(invalid="ignore"):
            under = FD.plan_kin(z[k].astype(np.float64), fut)["gain"] < 0.9
        F[k] = pd.DataFrame({"DAC fail %": dac, "inside-cut %": inside, "cannot-make-turn %": dac & ~inside & under,
                             "raw-plan departure %": dac & q.raw_out.fillna(False).astype(bool).to_numpy(),
                             "EPDMS (no EC)": q.score_noec.to_numpy()}).astype(float) * 100
    rows = []
    for k in DEC:
        for sn, m in sets.items():
            for col in F[k].columns:
                r = dict(arm=k, metric=col, stratum=sn, n=int(m.sum()), value=float(F[k][col].to_numpy()[m].mean()))
                for ref in ("V+S", "V"):
                    if ref != k:
                        p = stats.paired(F[k][col].to_numpy()[m], F[ref][col].to_numpy()[m], groups=log[m], n_boot=NB)
                        r |= {f"diff_vs_{ref}": p["mean"], f"lo_vs_{ref}": p["lo"], f"hi_vs_{ref}": p["hi"]}
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "arms.csv", index=False)
    m = sets["T45 (> 45 deg)"]
    B = TP.Boot(log, m, B=NB)
    cl = {}
    for col in ("inside-cut %", "DAC fail %"):
        (c, rc), (o, ro) = B.mean(F["V+S"][col].to_numpy(), m), B.mean(F["V+G"][col].to_numpy(), m)
        cl[col] = TP.ci(1 - o / c, 1 - ro / rc)
    d = T[(T.arm == "V+G") & (T.metric == "inside-cut %") & (T.stratum == "T45 (> 45 deg)")].iloc[0]
    c = cl["inside-cut %"]["mean"]
    excl = d["hi_vs_V+S"] < 0
    vd = dict(closure_T45=cl, diff_inside_cut_T45=[d["diff_vs_V+S"], d["lo_vs_V+S"], d["hi_vs_V+S"]],
              reading=("usable by a fresh head" if c >= 0.5 and excl else "not usable even by a fresh head" if c < 0.25 else "partly usable"),
              dac_check=chk, fits=json.loads((OUT / "decode" / "fits.json").read_text()))
    (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
    L = ["thin decoder (fresh head), navtest T20 tokens; diff vs V+S with 95% CI (log-cluster paired bootstrap)\n",
         "| metric | stratum | n | " + " | ".join(DEC) + " |", "|:--|:--|--:|" + ":--|" * len(DEC)]
    for sn in sets:
        for col in F["V"].columns:
            q = T[(T.metric == col) & (T.stratum == sn)].set_index("arm")
            L.append(f"| {col} | {sn} | {q.n.iloc[0]} | " + " | ".join(
                f"{q.loc[k, 'value']:.2f}" + ("" if k == "V+S" else f" ({q.loc[k, 'diff_vs_V+S']:+.2f} [{q.loc[k, 'lo_vs_V+S']:+.2f}, {q.loc[k, 'hi_vs_V+S']:+.2f}])")
                for k in DEC) + " |")
    L.append("\n```json\n" + json.dumps(vd, indent=1, default=float) + "\n```")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- frames (report / gate)
class Data:
    def __init__(self, replays):
        import pandas as pd
        import fd_navsim as FD
        self.FD, self.pd = FD, pd
        tab = np.load(CR / TEST / "tab.npz")
        self.tok, self.log, self.fut = tab["names"], tab["log"], tab["fut"].astype(np.float64)
        g = pd.DataFrame(FD.path_geom(self.fut), index=self.tok)
        self.dpsi = g.dpsi.to_numpy()
        a = np.abs(self.dpsi)
        self.sets = {"all": np.ones(len(a), bool), "S5 (< 5 deg)": a < 5, "T20 (> 20 deg)": a > 20, "T45 (> 45 deg)": a > 45,
                     "sharp R < 15 m": FD.buckets(g).sharp.to_numpy()}
        self.rep = pd.concat([pd.read_parquet(OUT / f"replay_{n}.parquet") for n in replays]).drop_duplicates(["key", "token"], keep="last")
        self.check = {}

    def one(self, sp):
        """Per-token read-outs (x 100) of one model spec, navtest tab order."""
        from jevdrive.bench import tables as BT
        from jevdrive.bench.compat import pred_file
        pd = self.pd
        u = BT.load("navtest", sp)[0]
        assert u is not None, f"no navtest result for {sp}"
        u = u.reindex(self.tok)
        assert not u.score.isna().any(), f"missing navtest tokens in {sp}"
        z = np.load(pred_file(sp))
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        Pp = z["poses"][[pos[t] for t in self.tok]].astype(np.float64)
        q = self.rep[self.rep.key == sp].set_index("token").reindex(self.tok)
        dac = (u.DAC < 1).to_numpy()
        side = q.lqr_side.to_numpy(float)
        self.check[sp] = dict(dac_fail=int(dac.sum()), replayed=int(q.key.notna()[dac].sum()), no_side=int(np.isnan(side[dac]).sum()),
                              replay_dac_disagree=int((q.drivable_area_compliance[dac] >= 1).sum()))
        inside = dac & (side == np.sign(self.dpsi))
        with np.errstate(invalid="ignore"):
            under = self.FD.plan_kin(Pp, self.fut)["gain"] < 0.9
        cannot = dac & ~inside & under
        f = pd.DataFrame({"EPDMS": u.score.to_numpy(), "DAC fail %": dac, "inside-cut %": inside, "cannot-make-turn %": cannot,
                          "other DAC fail %": dac & ~inside & ~cannot, "raw-plan departure %": dac & q.raw_out.fillna(False).astype(bool).to_numpy(),
                          "EP": u.EP.to_numpy(), "NC+TTC fail %": ((u.NC < 1) | (u.TTC < 1)).to_numpy()}, index=self.tok).astype(float) * 100
        return f, Pp

    def arm(self, arm, seeds):
        fs = [self.one(spec(arm, s))[0] for s in seeds]
        return sum(fs) / len(fs)


TABLE = (("EPDMS", "all"), ("EPDMS", "S5 (< 5 deg)"), ("EPDMS", "T20 (> 20 deg)"), ("EPDMS", "T45 (> 45 deg)"), ("DAC fail %", "all"),
         ("DAC fail %", "T20 (> 20 deg)"), ("DAC fail %", "T45 (> 45 deg)"), ("inside-cut %", "T45 (> 45 deg)"), ("cannot-make-turn %", "T45 (> 45 deg)"),
         ("other DAC fail %", "T45 (> 45 deg)"), ("raw-plan departure %", "T45 (> 45 deg)"), ("inside-cut %", "T20 (> 20 deg)"),
         ("cannot-make-turn %", "T20 (> 20 deg)"), ("DAC fail %", "sharp R < 15 m"), ("inside-cut %", "sharp R < 15 m"),
         ("cannot-make-turn %", "sharp R < 15 m"), ("EP", "all"), ("NC+TTC fail %", "all"))
PRIMARY = ("inside-cut %", "T45 (> 45 deg)")


def paired(A, B, col, m, log):
    from jevdrive import stats
    r = stats.paired(A[col].to_numpy()[m], B[col].to_numpy()[m], groups=log[m], n_boot=NB)
    return dict(arm=r["mean_a"], ref=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"])


def closure(Dt, ctrl, orc, col, m, other=None):
    """1 - Q(orc) / Q(ctrl) with a log-cluster CI (one resample for both); other: survival (Q(ctrl) - Q(other)) / (Q(ctrl) - Q(orc))."""
    import turn_probe as TP
    B = TP.Boot(Dt.log, m, B=NB)
    (c, rc), (o, ro) = B.mean(ctrl[col].to_numpy(), m), B.mean(orc[col].to_numpy(), m)
    with np.errstate(invalid="ignore", divide="ignore"):
        if other is None:
            return TP.ci(1 - o / c, 1 - ro / rc)
        x, rx = B.mean(other[col].to_numpy(), m)
        return TP.ci((c - x) / (c - o), (rc - rx) / (rc - ro))


def verdict(Dt, F, seeds, ctrl="OS", orc="OG"):
    col, m = PRIMARY[0], Dt.sets[PRIMARY[1]]
    d = paired(F[orc], F[ctrl], col, m, Dt.log)
    c = closure(Dt, F[ctrl], F[orc], col, m)
    n_fail = float(F[ctrl][col].to_numpy()[m].sum() / 100 * len(seeds))
    ep = paired(F[ctrl], F["H0"], "EPDMS", Dt.sets["all"], Dt.log)
    excl = d["hi"] < 0 or d["lo"] > 0
    if n_fail < 30 or abs(ep["diff"]) > 0.5:
        v = "invalid: " + ("control has < 30 inside-cut token-seeds" if n_fail < 30 else "control differs from H0 by > 0.5 EPDMS")
    elif c["mean"] >= 0.5 and excl and d["diff"] < 0:
        v = "vision is the bottleneck"
    elif c["mean"] < 0.25:
        v = "plan head is the bottleneck"
    else:
        v = "both"
    return dict(verdict=v, seeds=list(seeds), Q_control=d["ref"], Q_oracle=d["arm"], diff=d, closure=c, control_fail_token_seeds=n_fail,
                control_minus_H0_EPDMS=ep, ci_excludes_0=bool(excl))


def cmd_gate(a):
    """Seed-0 clear-negative gate (prereg): OG vs OS closure of the T45 inside-cut rate < 0.25 AND T20 DAC failure drop < 0.4 pp -> exit 2."""
    Dt = Data([a.replay])
    F = {k: Dt.arm(k, [0]) for k in ("OS", "OG")}
    c = closure(Dt, F["OS"], F["OG"], PRIMARY[0], Dt.sets[PRIMARY[1]])
    t20 = paired(F["OG"], F["OS"], "DAC fail %", Dt.sets["T20 (> 20 deg)"], Dt.log)
    neg = bool(c["mean"] < 0.25 and -t20["diff"] < 0.4)
    res = dict(rule="clear negative iff seed-0 T45 inside-cut closure (OG vs OS) < 0.25 and T20 DAC failure drop < 0.4 pp", closure=c,
               T20_DAC_fail=t20, Q_OS=float(F["OS"][PRIMARY[0]].to_numpy()[Dt.sets[PRIMARY[1]]].mean()),
               Q_OG=float(F["OG"][PRIMARY[0]].to_numpy()[Dt.sets[PRIMARY[1]]].mean()), clear_negative=neg, checks=Dt.check)
    (OUT / "gate-s0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(2 if neg else 0)


def cmd_bgate(a):
    """Step 2 stage B selection (prereg addendum): the pair with the largest seed-0 T45 inside-cut closure, if it is >= 0.25 and its T20 DAC
    failure drop is >= 0.4 pp. Prints '<control> <oracle>'; exit 2 if no pair qualifies."""
    Dt = Data(a.replays)
    res, best = {}, None
    for ctrl, orc in (("OS9", "OG9"), ("OSh", "OGh")):
        F = {k: Dt.arm(k, [0]) for k in (ctrl, orc)}
        c = closure(Dt, F[ctrl], F[orc], PRIMARY[0], Dt.sets[PRIMARY[1]])
        t20 = paired(F[orc], F[ctrl], "DAC fail %", Dt.sets["T20 (> 20 deg)"], Dt.log)
        ok = bool(c["mean"] >= 0.25 and -t20["diff"] >= 0.4)
        res[f"{orc} vs {ctrl}"] = dict(closure=c, T20_DAC_fail=t20, qualifies=ok)
        if ok and (best is None or c["mean"] > best[0]):
            best = (c["mean"], ctrl, orc)
    res["best"] = list(best[1:]) if best else None
    (OUT / "gate-b.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float), file=_sys.stderr)
    if not best:
        raise SystemExit(2)
    print(best[1], best[2])


def cmd_report(a):
    import pandas as pd
    from jevdrive.run import Run
    out = OUT / ("report" + (f"-{a.name}" if a.name else ""))
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", f"turn-oracle-{out.name}", seed=0, config=vars(a)) as run:
        Dt = Data(a.replays)
        F = {k: Dt.arm(k, a.seeds) for k in a.arms}
        rows = []
        for k in a.arms:
            for col, sn in TABLE:
                m = Dt.sets[sn]
                r = dict(arm=k, metric=col, stratum=sn, n=int(m.sum()), value=float(F[k][col].to_numpy()[m].mean()))
                for ref in a.refs:
                    if ref != k and ref in F:
                        p = paired(F[k], F[ref], col, m, Dt.log)
                        r |= {f"diff_vs_{ref}": p["diff"], f"lo_vs_{ref}": p["lo"], f"hi_vs_{ref}": p["hi"]}
                rows.append(r)
        T = pd.DataFrame(rows)
        T.to_csv(out / "arms.csv", index=False)
        vd = {}
        if "OS" in F and "OG" in F:
            vd = verdict(Dt, F, a.seeds)
            m = Dt.sets[PRIMARY[1]]
            vd["survival"] = {k: closure(Dt, F["OS"], F["OG"], PRIMARY[0], m, F[k]) for k in a.arms if k in ("PW", "PV", "MW", "H0", "OG off", "OG:off")}
            vd["closure_other"] = {f"{col} | {sn}": closure(Dt, F["OS"], F["OG"], col, Dt.sets[sn])
                                   for col, sn in (("DAC fail %", "T45 (> 45 deg)"), ("DAC fail %", "T20 (> 20 deg)"), ("inside-cut %", "sharp R < 15 m"),
                                                   ("inside-cut %", "T20 (> 20 deg)"), ("DAC fail %", "all"))}
        for ctrl, orc in a.pairs:                                                 # step-2 contrasts: closure of the primary of <orc> vs <ctrl>
            vd.setdefault("pairs", {})[f"{orc} vs {ctrl}"] = dict(
                closure=closure(Dt, F[ctrl], F[orc], PRIMARY[0], Dt.sets[PRIMARY[1]]), diff=paired(F[orc], F[ctrl], PRIMARY[0], Dt.sets[PRIMARY[1]], Dt.log))
        vd["checks"] = Dt.check
        if (OUT / "bank_stats.json").exists():
            vd["bank"] = json.loads((OUT / "bank_stats.json").read_text())
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        # per-token table (seed means) of the turning tokens
        tk = pd.DataFrame({"token": Dt.tok, "log": Dt.log, "dpsi_deg": Dt.dpsi})
        for k in a.arms:
            for col in ("EPDMS", "DAC fail %", "inside-cut %", "cannot-make-turn %"):
                tk[f"{k} | {col}"] = F[k][col].to_numpy()
        tk[Dt.sets["T20 (> 20 deg)"]].to_csv(out / "tokens_T20.csv", index=False, float_format="%.4g")
        # markdown
        fm = lambda v, d: f"{v:.{d}f}"  # noqa: E731
        L = [f"seeds {a.seeds}; values are seed means x 100; diffs with 95% CI (log-cluster paired bootstrap, B {NB})\n"]
        for ref in a.refs:
            L.append(f"\n**arm (diff vs {ref})**\n\n| metric | stratum | n | " + " | ".join(a.arms) + " |\n|:--|:--|--:|" + ":--|" * len(a.arms))
            for col, sn in TABLE:
                q = T[(T.metric == col) & (T.stratum == sn)].set_index("arm")
                dg = 2
                cells = [fm(q.loc[k, "value"], dg) + ("" if k == ref or f"diff_vs_{ref}" not in q or pd.isna(q.loc[k, f"diff_vs_{ref}"]) else
                                                     f" ({q.loc[k, f'diff_vs_{ref}']:+.{dg}f} [{q.loc[k, f'lo_vs_{ref}']:+.{dg}f}, {q.loc[k, f'hi_vs_{ref}']:+.{dg}f}])")
                         for k in a.arms]
                L.append(f"| {col} | {sn} | {q.n.iloc[0]} | " + " | ".join(cells) + " |")
        L.append("\n```json\n" + json.dumps({k: v for k, v in vd.items() if k not in ("checks", "bank")}, indent=1, default=float) + "\n```")
        (out / "tables.md").write_text("\n".join(L) + "\n")
        run.info("\n".join(L))
        run.info("checks " + json.dumps(Dt.check))
        figures(out, Dt, F, T, a)
        run.summary.update(verdict=vd.get("verdict", ""), out=str(out))


def figures(out, Dt, F, T, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    import opb_probe as P
    ps.apply()
    pal = [ps.BASELINE, ps.PALETTE["vermillion"], ps.PALETTE["blue"], ps.PALETTE["green"], ps.PALETTE.get("orange", "#E69F00"),
           ps.PALETTE.get("purple", "#CC79A7"), "#56B4E9", "#000000"]
    col = {k: pal[i % len(pal)] for i, k in enumerate(a.arms)}
    # 1. sharp-turn DAC failure anatomy per arm
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.7), constrained_layout=True)
    parts = (("inside-cut %", "#9bbb59", "inside of the turn (corner cut)"), ("cannot-make-turn %", "#4f81bd", "outside, heading gain < 0.9 (cannot make the turn)"),
             ("other DAC fail %", "#bfbfbf", "other"))
    for ax, sn in zip(axs, ("T45 (> 45 deg)", "T20 (> 20 deg)")):
        bot = np.zeros(len(a.arms))
        for c, cc, lb in parts:
            v = np.array([F[k][c].to_numpy()[Dt.sets[sn]].mean() for k in a.arms])
            ax.bar(range(len(a.arms)), v, 0.7, bottom=bot, color=cc, label=lb)
            bot += v
        ax.set_xticks(range(len(a.arms)), a.arms, rotation=30, ha="right")
        ax.set_ylabel(f"DAC failures, % of {int(Dt.sets[sn].sum())} tokens"), ax.set_title(f"navtest {sn}", fontsize=8), ps.bars(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.savefig(out / "anatomy.png", dpi=300)
    plt.close(fig)
    # 2. BEV examples: the same sharp-turn tokens under every plotted arm (seed 0)
    show = [k for k in a.bev if k in a.arms]
    one = {k: Dt.one(spec(k, a.seeds[0])) for k in show}
    m = Dt.sets["T45 (> 45 deg)"] & (one[show[0]][0]["inside-cut %"].to_numpy() > 0) & (one[show[1]][0]["inside-cut %"].to_numpy() > 0)
    idx = np.flatnonzero(m)[np.argsort(Dt.tok[m])]
    idx = idx[[int(q * (len(idx) - 1)) for q in np.linspace(0.08, 0.92, 6)]] if len(idx) >= 6 else idx
    pos, Z = P.labels("navtest")
    C = np.array([[4.049, 1.1485], [4.049, -1.1485], [-1.127, -1.1485], [-1.127, 1.1485], [4.049, 1.1485]])
    xc, yc = P.X0 + (np.arange(128) + 0.5) * 0.5, P.Y0 + (np.arange(96) + 0.5) * 0.5
    fig, axs = plt.subplots(2, 3, figsize=(ps.DOUBLE_COLUMN_IN, 5.0), constrained_layout=True)
    for ax, i in zip(axs.ravel(), idx):
        g = Z["sdf"][pos[Dt.tok[i]]].astype(np.float32)
        ax.contourf(yc, xc, g, levels=[0, 1e3], colors=["#E8E8E8"])
        ax.contour(yc, xc, g, levels=[0], colors="k", linewidths=0.8)
        f = np.vstack([[0, 0, 0], Dt.fut[i]])
        ax.plot(f[:, 1], f[:, 0], color="k", ls=":", lw=0.9)
        tt = []
        for k in show:
            fr, Pp = one[k]
            p = np.vstack([[0, 0, 0], Pp[i]])
            ax.plot(p[:, 1], p[:, 0], color=col[k], lw=1.0, marker=".", ms=2)
            for x, y, h in p[2::2]:
                R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
                q = C @ R.T + [x, y]
                ax.plot(q[:, 1], q[:, 0], color=col[k], lw=0.4, alpha=0.7)
            tt.append(f"{k} {'cut' if fr['inside-cut %'].iloc[i] else 'DAC fail' if fr['DAC fail %'].iloc[i] else 'ok'}")
        xm, ym = f[:, 0].max(), f[:, 1]
        ax.set_xlim(max(ym.max(), 0) + 9, min(ym.min(), 0) - 9), ax.set_ylim(-3, max(xm, 10) + 9), ax.set_aspect("equal"), ax.grid(False)
        ax.set_title(f"{Dt.tok[i][:8]} ({Dt.dpsi[i]:+.0f} deg)\n" + ", ".join(tt), fontsize=6.5)
    h = [plt.Line2D([], [], color="k", lw=0.8), plt.Line2D([], [], color="k", ls=":", lw=0.9)] + [plt.Line2D([], [], color=col[k], lw=1) for k in show]
    fig.legend(h, ["true drivable boundary", "logged future (4 s)"] + [LABEL.get(k, k) for k in show], loc="outside lower center", ncol=3)
    fig.savefig(out / "bev.png", dpi=300)
    plt.close(fig)
    (out / "bev_tokens.txt").write_text("\n".join(Dt.tok[idx]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("bank")
    p = sp.add_parser("replay")
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="*", default=[], help="bench model specs (e.g. TOG-F-s0 TOG-F-s0:noside)")
    p.add_argument("--poses", default="", help="npz with `tokens` and (N, 8, 3) pose arrays: replay every key on every token")
    p.add_argument("--procs", type=int, default=0)
    p = sp.add_parser("decode")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lam-hinge", type=float, default=10.0)
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p = sp.add_parser("areport")
    p.add_argument("--score", default="", help="bench score-poses CSV of decode/poses.npz (DAC check)")
    p = sp.add_parser("gate")
    p.add_argument("--replay", default="s0")
    p = sp.add_parser("bgate")
    p.add_argument("--replays", nargs="+", default=["b0"])
    p = sp.add_parser("report")
    p.add_argument("--name", default="")
    p.add_argument("--arms", nargs="+", default=["H0", "OS", "OG", "PW", "PV", "MW", "OG off"])
    p.add_argument("--refs", nargs="+", default=["OS", "H0"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--replays", nargs="+", default=["s0", "s1"])
    p.add_argument("--bev", nargs="+", default=["H0", "OS", "OG"])
    p.add_argument("--pairs", nargs=2, action="append", default=[], metavar=("CTRL", "ARM"))
    a = ap.parse_args()
    {"bank": cmd_bank, "replay": cmd_replay, "gate": cmd_gate, "report": cmd_report, "decode": cmd_decode, "areport": cmd_areport,
     "bgate": cmd_bgate}[a.cmd](a)
