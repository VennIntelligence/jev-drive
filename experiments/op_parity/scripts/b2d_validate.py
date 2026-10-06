"""Validation of the B2D P2 training cache (b2d_prep.py): one GPU pool job, writes $DATA_DIR/runs/op_parity/b2d_validate/{validate.json, panel_*.png}.

  1. token RMS     per-token RMS of the B2D hidden tokens (ticks.npy, 4000 random ticks) against the navtrain caches (W protocol and the G protocol),
                   same statistic: mean / p5 / p95 of the per-token RMS and the correlation of the per-channel RMS profiles. Expected range: the B2D mean
                   within [0.8, 1.25] x navtrain's (a different domain: CARLA night / fog / rain frames and a 1.86 m camera, not the 1% of decision 144's
                   same-data cache comparison).
  2. decode back   8 random val rows: their 8 context tokens recomputed from frames.mp4 (a fresh full-clip decode, a different batch size) against ticks.npy
                   (max abs diff, relative); 3 of them drawn as road / wide pictures with the logged future (green) and the cached P2 plan (red).
  3. P2 inference  P2-F-s0 (and shipped P0) on cached tokens: (a) 500 val rows with a >20 deg logged heading change and motion (turn rows): sign agreement of the
                   plan's 4 s heading with the log, ADE; (b) every row of the collection check (check/samples.csv, 20 948 rows, stride 10) that the cache also holds:
                   the same agreement from the cache's tokens against the check's own (plan from the replayed video) -- 0.863 in check.json -- and the per-row
                   ADE / yaw difference.

  python experiments/op_parity/scripts/b2d_validate.py [--n-val 500]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/b2d_collect/lib"),
                 str(_R / "experiments/b2d_collect/scripts"), str(_R / "experiments/op_parity/scripts")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

CACHE = data_dir() / "runs" / "op_parity" / "cache"
B2D = CACHE / "b2d_v2"
OUT = data_dir() / "runs" / "op_parity" / "b2d_validate"


def rms_stats(x):
    """x (n, 32, 512) fp16 -> per-token RMS (n,) and per-channel RMS profile (512,)."""
    x = x.astype(np.float32)
    return np.sqrt((x ** 2).mean((1, 2))), np.sqrt((x ** 2).mean((0, 1)))


def sample_tokens(arr, n, rng, slots):
    """n random (row, slot) tokens of an (N, S, 32, 512) mmap -> (n, 32, 512)."""
    rows = np.sort(rng.choice(len(arr), min(n, len(arr)), replace=False))
    return np.stack([arr[r, int(rng.choice(slots))] for r in rows])


def main(a):
    import torch
    import pp_train as T
    import b2dc_check as C
    import b2dc_frames as F
    import pp_prep as PP
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    from jevdrive.run import Run
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    res = {}
    with Run("op_parity", "b2d-validate", config=vars(a)) as run:
        val = splits.load("b2d/b2dc-v2-val")
        run.use_split(val), run.use_split(splits.load("b2d/b2dc-v2-train"))
        ticks = np.load(B2D / "ticks.npy", mmap_mode="r")
        fidx = np.load(B2D / "front_idx.npy")
        tab = dict(np.load(B2D / "tab.npz"))
        ex = dict(np.load(B2D / "extra.npz"))
        N = len(fidx)
        # ---- 1. token RMS
        tk = ticks[np.sort(rng.choice(len(ticks), 4000, replace=False))]
        rb, cb = rms_stats(tk)
        ref = {}
        for name, d in {"navtrain_W": "lb_navtrain@warp", "navtrain_G": "lb_navtrain", "navtrain_full_s0_W": "navtrain_full.s0of12@warp",
                        "navtest_W": "lb_navtest@warp"}.items():
            f = np.load(CACHE / d / "front.npy", mmap_mode="r")
            rr, cr = rms_stats(sample_tokens(f, 4000, rng, range(f.shape[1])))
            ref[name] = dict(mean=float(rr.mean()), p5=float(np.percentile(rr, 5)), p95=float(np.percentile(rr, 95)),
                             ratio_b2d_over_ref=float(rb.mean() / rr.mean()), channel_rms_corr=float(np.corrcoef(cb, cr)[0, 1]))
        res["token_rms"] = dict(b2d=dict(mean=float(rb.mean()), p5=float(np.percentile(rb, 5)), p95=float(np.percentile(rb, 95))), ref=ref,
                                pass_range=all(0.8 <= v["ratio_b2d_over_ref"] <= 1.25 for v in ref.values()))
        run.info("token rms: " + json.dumps(res["token_rms"]))

        # ---- model + helpers
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        pi = None

        def plans(model, rows):
            nonlocal pi
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 128):
                    r = rows[i:i + 128]
                    front = torch.from_numpy(ticks[fidx[r]]).to(dev)
                    ego = torch.from_numpy(tab["ego"][r]).to(dev)
                    tc = torch.tensor([[1.0, 0.0]], device=dev).expand(len(r), 2)
                    o = model(front, ego, tc).float()
                    if pi is None:
                        pi = torch.as_tensor(A.plan_index(model.net.slices), device=dev)
                    p = o[:, pi].view(-1, 33, 15)
                    x, y, psi = T.rear(p, torch.full((len(r),), C.MOUNT[0], device=dev), W)
                    out.append(torch.stack([x, y, psi], -1).cpu().numpy())
            return np.concatenate(out)
        models = {m: T.load_pmodel(m, dev) for m in ("P0", "P2-F-s0")}
        fut = tab["fut"]
        yaw4 = np.degrees(fut[:, -1, 2])
        L = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 1, 2)), fut[:, :, :2]], 1), axis=1), axis=-1).sum(1)
        isval = val.mask(ex["route"])

        # ---- 3a. val turn rows
        cand = np.flatnonzero(isval & (np.abs(yaw4) > 20) & (L > 2.0))
        sel = np.sort(rng.choice(cand, min(a.n_val, len(cand)), replace=False))
        rnd = np.sort(rng.choice(np.flatnonzero(isval), min(a.n_val, int(isval.sum())), replace=False))
        P = {m: plans(models[m], sel) for m in models}
        res["val_turn_rows"] = dict(n=len(sel), candidates=int(len(cand)), val_rows=int(isval.sum()), val_routes=int(len(set(ex["route"][isval]))))
        for m, p in P.items():
            res["val_turn_rows"][m] = dict(turn_sign_agree=float((np.sign(np.degrees(p[:, -1, 2])) == np.sign(yaw4[sel])).mean()),
                                           ade=float(np.linalg.norm(p[..., :2] - fut[sel][..., :2], axis=-1).mean()))
        Pr = plans(models["P2-F-s0"], rnd)
        res["val_random_rows"] = dict(n=len(rnd), P2_ade=float(np.linalg.norm(Pr[..., :2] - fut[rnd][..., :2], axis=-1).mean()))
        run.info("val: " + json.dumps({k: res[k] for k in ("val_turn_rows", "val_random_rows")}))

        # ---- 3b. the collection check's rows
        import pandas as pd
        S = pd.read_csv(data_dir() / "runs/b2d_collect/data/all/check/samples.csv")
        key = {(r, int(t)): i for i, (r, t) in enumerate(zip(ex["route"], ex["tick"]))}
        hit = np.array([key.get((str(r), int(t)), -1) for r, t in zip(S.route_id, S.t0)])
        S, hit = S[hit >= 0].reset_index(drop=True), hit[hit >= 0]
        rows = np.sort(hit)
        order = np.argsort(np.argsort(hit))
        res["check_rows"] = dict(in_check=int(len(pd.read_csv(data_dir() / 'runs/b2d_collect/data/all/check/samples.csv'))), in_cache=int(len(S)))
        for m, cn in (("P0", "P0"), ("P2-F-s0", "P2-F-s0")):
            p = plans(models[m], rows)[order]
            y4 = np.degrees(p[:, -1, 2])
            mv = (S.log_len > 2.0).values
            big = mv & (S.fut_yaw4.abs() > 20).values
            ade = np.linalg.norm(p[..., :2] - fut[hit][..., :2], axis=-1).mean(1)
            res["check_rows"][m] = dict(
                turn_sign_agree_cache=float((np.sign(y4[big]) == np.sign(S.fut_yaw4[big])).mean()),
                turn_sign_agree_check=float((np.sign(S[f"yaw4_{cn}"][big]) == np.sign(S.fut_yaw4[big])).mean()), turn_rows=int(big.sum()),
                ade_cache=float(ade[mv].mean()), ade_check=float(S[f"ade_{cn}"][mv].mean()),
                row_ade_abs_diff_median=float(np.median(np.abs(ade - S[f"ade_{cn}"]))), row_ade_abs_diff_p99=float(np.percentile(np.abs(ade - S[f"ade_{cn}"]), 99)),
                row_yaw4_abs_diff_median_deg=float(np.median(np.abs(y4 - S[f"yaw4_{cn}"]))), row_yaw4_abs_diff_p99_deg=float(np.percentile(np.abs(y4 - S[f"yaw4_{cn}"]), 99)),
                sign_flip_rows=int((np.sign(y4[big]) != np.sign(S[f"yaw4_{cn}"][big])).sum()))
        res["check_rows"]["check_json_P2_turn_sign_agree"] = json.loads((data_dir() / "runs/b2d_collect/data/all/check/check.json").read_text())["turn_sign_agree_P2-F-s0"]
        run.info("check rows: " + json.dumps(res["check_rows"]))

        # ---- 2. decode back: recompute 8 val rows' tokens from the video
        net, enc = PP.encoder(dev)
        pick = rng.choice(np.flatnonzero(isval), 8, replace=False)
        plan_b = json.loads((B2D / "plan.json").read_text())
        clips = {c["route"]: c for c in plan_b["clips"]}
        diffs = []
        for k, r in enumerate(pick):
            c = clips[str(ex["route"][r])]
            t0 = int(ex["tick"][r])
            pairs = F.read_pairs(_pl.Path(c["clip"]) / "frames.mp4")                  # a fresh, full decode
            slot = np.maximum(t0 - 4 * np.arange(7, -1, -1), 4)
            tok = enc(pairs[slot - 4], pairs[slot], bs=3)                              # other batch size than the build
            ref_t = ticks[fidx[r]].astype(np.float32)
            d = np.abs(tok.astype(np.float32) - ref_t)
            diffs.append(dict(row=int(r), route=str(ex["route"][r]), t0=t0, max_abs=float(d.max()), mean_abs=float(d.mean()),
                              rel=float(d.mean() / np.abs(ref_t).mean())))
            if k < 3:
                import cv2
                p = plans(models["P2-F-s0"], np.array([r]))[0]
                paths = [(fut[r][:, :2], (40, 220, 40)), (p[:, :2], (230, 40, 40))]
                road, wide = C.draw(C.yuv_rgb(pairs[t0, 0]), "road", paths), C.draw(C.yuv_rgb(pairs[t0, 1]), "wide", paths)
                img = np.concatenate([road, wide], 1)
                cv2.imwrite(str(OUT / f"panel_{k}_{c['route']}_{t0:05d}.png"), img[:, :, ::-1])
                diffs[-1]["panel"] = f"panel_{k}_{c['route']}_{t0:05d}.png"
        res["decode_back"] = dict(rows=diffs, max_abs=max(d["max_abs"] for d in diffs), rel_max=max(d["rel"] for d in diffs))
        run.info("decode back: " + json.dumps(res["decode_back"]))
        (OUT / "validate.json").write_text(json.dumps(res, indent=1))
        run.summary.update(pass_rms=res["token_rms"]["pass_range"], val_turn_P2=res["val_turn_rows"]["P2-F-s0"]["turn_sign_agree"],
                           check_P2=res["check_rows"]["P2-F-s0"]["turn_sign_agree_cache"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=500)
    from jevdrive.run import cli_args
    cli_args(ap)
    main(ap.parse_args())
