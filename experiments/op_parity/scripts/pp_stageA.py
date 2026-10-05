"""op_parity Stage A (plans/2026-10-06-parity-prereg.md, "补充 1"): which part of the frame protocol matters to shipped Cinque, with no training.

Tokens: runs/op_lb/lb_hq_navtestX (decision 116: 1 499 navtest tokens / 78 logs with the REAL nuPlan 10 Hz CAM_F0 frames at the six 0.2 s lattice
times t0 - {1.4, 1.2, 0.8, 0.6, 0.4, 0.2} s next to the four 2 Hz keys; GIMM frames at the same times). Shipped Cinque (port, fp16, frozen encoder),
zero state, plan at t0. A cell = (queue, gap, source):

  queue  Q8   8 valid policy slots at t = -1.4 .. 0 s every 0.2 s (the op_lb protocol; slot 0 zero)
         Q4f  4 valid slots at -0.6 .. 0 every 0.2 s (same spacing, fewer slots)
         Q4d  4 valid slots at the 2 Hz keys -1.5, -1.0, -0.5, 0 (time-dilated: one slot per 0.5 s), the last 4 policy slots
  gap    g   vision pair (frame t - g, frame t); g = 0 is the static pair (t, t), no motion cue
  source real (lattice frames; any frame before -1.5 s is a zero image in every cell, the op_lb convention for the oldest pair), gimm, warp (CPU ego-motion warp
         of jevdrive.op_interp.synth_cpu at any time)

Cells (see the prereg for which contrast each one serves): R8 = Q8 g0.2 real (reference), G8 = Q8 g0.2 gimm, W8 = Q8 g0.2 warp, R8g4 = Q8 g0.4
real, S8 = Q8 g0, R4f = Q4f g0.2 real, W4f = Q4f g0.2 warp, N4 = Q4d g0.5 real (native 2 Hz), W4d = Q4d g0.2 warp, S4d = Q4d g0.

  run      plans of every cell -> runs/op_lb/lb_hq_navtestX/plans/A<cell>@cinque.npz + $R/stageA/{tokens.npz} (t0-slot hidden token per cell)
  report   per cell: t0-token cosine distance to R8, plan ADE (8 NAVSIM poses) to R8 and to the log, speed ratio (plan / log path length at
           4 s, median) and EPDMS + subscores from the devkit CSVs; paired diffs vs R8 (bootstrap over logs) -> results/stageA.md
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_openloop/lib"),
                 str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir, n_cpus  # noqa: E402

DATA = "lb_hq_navtestX"
FRAME = (2, 6, 128, 256)
Q = {"Q8": np.round(np.arange(-7, 1) * 0.2, 3), "Q4f": np.round(np.arange(-3, 1) * 0.2, 3), "Q4d": np.array([-1.5, -1.0, -0.5, 0.0])}
CELLS = {"R8": ("Q8", 0.2, "real"), "G8": ("Q8", 0.2, "gimm"), "W8": ("Q8", 0.2, "warp"), "R8g4": ("Q8", 0.4, "real"), "S8": ("Q8", 0.0, "real"),
         "R4f": ("Q4f", 0.2, "real"), "W4f": ("Q4f", 0.2, "warp"), "N4": ("Q4d", 0.5, "real"), "W4d": ("Q4d", 0.2, "warp"), "S4d": ("Q4d", 0.0, "real")}
V2 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
      "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def warp_times():
    ts = set()
    for q, g, s in CELLS.values():
        if s == "warp":
            ts |= {round(t, 3) for t in Q[q]} | {round(t - g, 3) for t in Q[q] if t - g >= -1.5 - 1e-6}
    return np.array(sorted(ts))


def _warp_job(args):
    from jevdrive import op_interp as I
    kf, pose, vel, cam, times = args
    return I.synth_cpu(kf, "warp", times, I.track_navsim(pose, vel), cam)


def cmd_run(a):
    import torch
    import op_lb as OL
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    from jevdrive.data import splits
    import pp_prep as PP
    mt = OL.meta(DATA)
    N = len(mt["names"])
    lat = np.r_[OL.I.T_KEY, mt["syn_t"]]                                               # key times, then the 6 lattice times (frame order of real / gimm.npy)
    with Run("op_parity", "stageA", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        keys = OL.Keys(DATA)
        src = {s: np.load(OL.root(DATA) / f"{s}.npy", mmap_mode="r") for s in ("real", "gimm")}
        missing = np.array([not np.asarray(src["real"][i]).any(axis=(1, 2, 3, 4)).all() for i in range(N)])
        run.info(f"{N} tokens, {int(missing.sum())} with a missing real frame (dropped from every cell)")
        wt = warp_times()
        kf_all = [keys[i] for i in range(N)]
        with ProcessPoolExecutor(min(a.workers or n_cpus(), 64)) as ex:
            warped = list(run.tqdm(ex.map(_warp_job, [(kf_all[i], mt["pose"][i], mt["vel"][i], mt["cam"][i], wt) for i in range(N)], chunksize=8),
                                   total=N, desc="warp"))
        dev = torch.device("cuda")
        net, enc = PP.encoder(dev)
        pi = A.plan_index(net.slices)
        pdir = OL.root(DATA, "plans")
        toks = {}
        for cell, (q, g, s) in CELLS.items():
            def img(i, t):
                t = round(float(t), 3)
                if t < -1.5 - 1e-6:                                                 # before the history: zero image in every cell (op_lb convention)
                    return np.zeros(FRAME, np.uint8)
                if s == "warp":
                    return warped[i][int(np.flatnonzero(np.isclose(wt, t))[0])]
                k = np.flatnonzero(np.isclose(lat, t))
                if not len(k):
                    return np.zeros(FRAME, np.uint8)
                return kf_all[i][k[0]] if k[0] < 4 else np.asarray(src[s][i][k[0] - 4])
            T = Q[q]
            n_on = len(T)
            mu = np.zeros((N, 33, 15), np.float32)
            tok = np.zeros((N, 32, 512), np.float16)
            for i0 in range(0, N, 32):
                rows = range(i0, min(i0 + 32, N))
                cur = np.stack([[img(i, t) for t in T] for i in rows]).reshape(-1, *FRAME)
                prev = np.stack([[img(i, t - g) if g > 0 else img(i, t) for t in T] for i in rows]).reshape(-1, *FRAME)
                H = torch.from_numpy(enc(prev, cur)).to(dev).view(len(rows), n_on, *A.H_SHAPE)
                Hc = torch.zeros(len(rows), A.CONTEXT, *A.H_SHAPE, dtype=H.dtype, device=dev)
                Hc[:, A.CONTEXT - n_on:] = H
                valid = torch.zeros(len(rows), A.CONTEXT, dtype=torch.bool, device=dev)
                valid[:, A.CONTEXT - n_on:] = True
                tc = torch.tensor([[0.0, 1.0] if mt["lht"][i] else [1.0, 0.0] for i in rows], device=dev)
                with torch.no_grad():
                    o = A._policy(net, Hc, (0.275, 0.525), tc, valid)["outputs"].float()
                mu[list(rows)] = o[:, pi].cpu().numpy().reshape(-1, 33, 15)
                tok[list(rows)] = H[:, -1].cpu().numpy()
            toks[cell] = tok
            np.savez(pdir / f"A{cell}@cinque.npz", names=np.array(mt["names"]), plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11],
                     plan_mu=mu, steps=31, info=json.dumps({"model": "port shipped", "cell": [q, g, s], "source": "experiments/op_parity/scripts/pp_stageA.py"}))
            run.info(f"cell {cell}: {q} gap {g} {s} done")
        out = data_dir() / "runs" / "op_parity" / "stageA"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "tokens.npz", missing=missing, **toks)


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    import op_lb as OL
    import pp_eval as E
    mt = OL.meta(DATA)
    names = np.array(mt["names"])
    log = dict((e["token"], e["log_name"]) for e in Z.load_index("navtest", slim=True))
    z = np.load(data_dir() / "runs" / "op_parity" / "stageA" / "tokens.npz")
    keep = ~z["missing"]
    fz = np.load(Z.root("index") / "navtest_future.npz")
    fpos = dict(zip(fz["tokens"].tolist(), range(len(fz["tokens"]))))
    fut = np.stack([fz["poses"][fpos[t]] for t in names])
    poses, sc = {}, {}
    for cell in CELLS:
        p = np.load(OL.root(DATA, "preds") / f"A{cell}-cinque__base.npz")
        assert p["tokens"].tolist() == names.tolist()
        poses[cell] = p["poses"]
        fs = sorted((data_dir() / "runs" / "navsim" / "eval").glob(f"v2_navtest_opi_{DATA}_A{cell}-cinque__base/*/*.csv"))
        sc[cell] = E.read_csv(fs[-1])[0] if fs else None
    common = [t for t in names[keep] if all(s is not None and t in s.index for s in sc.values())]
    ci = np.flatnonzero(np.isin(names, common))
    g = np.array([log[t] for t in names[ci]])
    ref = toks_ref = z["R8"].astype(np.float32)
    rows, pr = [], []
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    Lg = plen(fut)
    for cell, (q, gp, s) in CELLS.items():
        t = z[cell].astype(np.float32)
        cos = 1 - (t * toks_ref).sum(-1) / (np.linalg.norm(t, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-6)     # (N, 32)
        P = poses[cell]
        ade_ref = np.linalg.norm(P[:, :, :2] - poses["R8"][:, :, :2], axis=-1).mean(1)
        ade_log = np.linalg.norm(P[:, :, :2] - fut[:, :, :2], axis=-1).mean(1)
        mv = Lg > 2.0
        ratio = plen(P) / np.maximum(Lg, 1e-6)
        S = sc[cell].loc[names[ci]]
        r = {"cell": cell, "queue": q, "gap": gp, "source": s, "n": len(ci), "tok_cos_dist": float(cos[ci].mean()),
             "ade_vs_R8": float(ade_ref[ci].mean()), "ade_vs_log": float(ade_log[ci].mean()),
             "speed_ratio_med": float(np.median(ratio[ci][mv[ci]])), "EPDMS": 100 * float(S.score.mean())}
        r |= {k: 100 * float(S[k].mean()) for k in V2}
        rows.append(r)
        if cell != "R8":
            d = stats.paired(100 * S.score.to_numpy(float), 100 * sc["R8"].loc[names[ci]].score.to_numpy(float), groups=g)
            dr = stats.paired(ratio[ci][mv[ci]], (plen(poses["R8"]) / np.maximum(Lg, 1e-6))[ci][mv[ci]], groups=g[mv[ci]])
            pr.append({"pair": f"{cell} - R8", "EPDMS": stats.fmt(d, ".2f"), "EP": stats.fmt(stats.paired(100 * S.ego_progress.to_numpy(float),
                       100 * sc["R8"].loc[names[ci]].ego_progress.to_numpy(float), groups=g), ".2f"),
                       "DAC": stats.fmt(stats.paired(100 * S.drivable_area_compliance.to_numpy(float),
                                                     100 * sc["R8"].loc[names[ci]].drivable_area_compliance.to_numpy(float), groups=g), ".2f"),
                       "speed_ratio": stats.fmt(dr, ".3f")})
    out = _R / "experiments" / "op_parity" / "results"
    stats.write_table(rows, out / "stageA_cells", floatfmt=".3f", note=f"{len(ci)} tokens of lb_hq_navtestX (real frames complete, scored in every cell); shipped Cinque port")
    stats.write_table(pr, out / "stageA_paired", note="paired vs R8 (real frames, 0.2 s pairs, 8 slots), bootstrap over navtest logs, B 10000; speed ratio over tokens whose logged 4 s path > 2 m")
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=0)
    sp.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)
