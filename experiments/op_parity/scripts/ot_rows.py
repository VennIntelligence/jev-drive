"""op_parity off-track rows (plans/2026-10-09-offtrack-rows-prereg.md): the SH30 recipe with ~10 % of every batch drawn from navtrain rows seen
from a statically perturbed ego pose. No on-policy rollout, no longitudinal perturbation, hinge kept.

An off-track row is a real navtrain token (speed at t0 > 3 m/s, logged future present) whose 10 history frames are re-projected with the
existing plane engine (jevdrive.op_interp.warp_frame, decisions 123 / 141) to the pose `logged pose(t) o (0, y(t), psi(t))`, where (y, psi) is
the scripted drift of experiments/op_adapt_h (heading error ramps from 0 at -1.6 s to dpsi, the lateral error integrates v * psi and ends at dy;
dpsi = 0 is a constant lateral offset). dy ~ U(-0.5, 0.5) m, dpsi ~ U(-2, 2) deg, redrawn while |y(-1.5 s)| > 1 m. One perturbation per token.
Every frame is ONE warp of the nearest real 2 Hz key (keys and lattice frames alike), so a zero offset reproduces the pp_prep W frames.
  ego     history poses re-expressed in the perturbed t0 frame; velocity / acceleration / command as logged (the drift moves the car along its
          own heading to first order)
  target  the logged future (8 poses) re-expressed in the perturbed t0 frame: no smoothed recovery path
  hinge   the same drivable SDF raster of the token (logged frame); the plan is mapped back into the logged frame before it is sampled
  teacher shipped Cinque on the perturbed tokens (non-plan heads are distilled to it; off-track rows are never anchor rows)

  prep    --data navtrain_full.s2of12 [--limit N] [--zero]   -> cache/ot1_<data>/tab.npz, cache/ot1_<data>@warp/{front.npy, teacher.npz}
          (the pp_prep layout, so pp_train.Store reads it unchanged); --zero: offsets 0, checked against the stored W tokens (pipeline gate)
  probe   --tags SHP-F-s0 ... --data ...   dev rows: ADE to the re-expressed future and the slope of the plan's lateral shift on the ideal shift
          (sign / geometry gate before training: a model that reads the image must follow the yaw offset with a positive coefficient)
  train   --tag OTP-F-s0 --ot-mass 0.1 ...   pp_train's Store / PModel / Losses unchanged; --ot-mass 0 reproduces pp_train's row stream
  gate    --new OTP-F-s0 --refs SHP-F-s0 ..  navhard (G frames) paired difference vs the seed mean of the refs; exit 2 = stop (< +1.0)
  report  --arms OT30-F-s0 OT30-F-s1 --refs SH30-F-s0 SH30-F-s1   navhard / navtest verdict lines -> json + md

  $DATA_DIR/envs/op-train/bin/python experiments/op_parity/scripts/ot_rows.py prep --data navtrain_full.s2of12 --limit 64
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir, n_cpus  # noqa: E402

OT = "ot1"                                       # cache prefix = version of the perturbation recipe below
DY, DPSI, YMAX, VMIN = 0.5, np.radians(2.0), 1.0, 3.0
FRAME = (2, 6, 128, 256)
PROFILE, KEY_EXTRA = None, {}                    # hook for another perturbation family (experiments/alpasim/scripts/ot3_rows.py); off = this recipe
CR = data_dir() / "runs" / "op_parity" / "cache"
OUT = data_dir() / "runs" / "op_parity" / "ot_rows"


def t_all() -> np.ndarray:
    """The 10 history frame times: the 4 keys, then the 6 lattice frames (op_lb.SYN_T)."""
    import op_lb as OL
    from jevdrive import op_interp as I
    return np.r_[I.T_KEY, OL.SYN_T]


def sample(speed: np.ndarray, rng) -> tuple:
    """(dy, dpsi) per row: uniform, redrawn while the drift history starts more than YMAX m off the logged path."""
    from experiments.op_adapt_h.lib import op_adapt_h as H
    n = len(speed)
    dy, dp = rng.uniform(-DY, DY, n), rng.uniform(-DPSI, DPSI, n)
    for _ in range(200):
        bad = np.abs(H.drift(-1.5, dy, dp, speed)[0]) > YMAX
        if not bad.any():
            break
        dy[bad], dp[bad] = rng.uniform(-DY, DY, bad.sum()), rng.uniform(-DPSI, DPSI, bad.sum())
    assert not bad.any()
    return dy, dp


def to_frame(P: np.ndarray, dy, dpsi) -> np.ndarray:
    """Poses (..., K, 3) x, y, yaw in the logged t0 frame -> the same poses in the perturbed t0 frame (0, dy, dpsi). dy, dpsi (...,)."""
    dy, dpsi = np.asarray(dy, np.float64)[..., None], np.asarray(dpsi, np.float64)[..., None]
    c, s = np.cos(dpsi), np.sin(dpsi)
    x, y = P[..., 0], P[..., 1] - dy
    return np.stack([c * x + s * y, -s * x + c * y, P[..., 2] - dpsi], -1)


def _job(args):
    """One token -> its 10 history frames (4 keys, 6 lattice) seen from the drifting pose: one warp of the nearest real key each."""
    import cv2
    import navsim_zs_openpilot as NZ
    from jevdrive import op_interp as I
    cv2.setNumThreads(1)
    e, pose, vel, cam, T, ys, ps = args
    kf = NZ.render_token(e)
    tr = I.track_navsim(pose, vel)
    out = np.empty((len(T),) + FRAME, np.uint8)
    for j, t in enumerate(T):
        if j < 4:
            src = j
        else:
            i0, i1, s = I.neighbours(t)
            src = i0 if s <= 0.5 else i1
        if j < 4 and abs(ys[j]) < 1e-9 and abs(ps[j]) < 1e-9:
            out[j] = kf[j]
            continue
        p = tr(t)
        dst = np.r_[p[:2] + I.rot2(p[2]) @ np.array([0.0, ys[j]]), p[2] + ps[j]]
        out[j] = I.warp_frame(kf[src], cam, dst, tr(I.T_KEY[src]))
    return out


# ---------------------------------------------------------------- prep
def cmd_prep(a):
    import torch
    import op_lb as OL
    import parity_adapter as PA
    import pp_prep as PP
    from experiments.op_adapt_h.lib import op_adapt_h as H
    from jevdrive import cache
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    from jevdrive.run import Run
    assert a.data.startswith("navtrain_full"), "off-track rows are built from the full navtrain shards (protocol W)"
    tag = f"{OT}_{a.data}" + ("-zero" if a.zero else "") + (f"-first{a.limit}" if a.limit else "")
    with Run("op_parity", f"otprep-{tag}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        base = dict(np.load(CR / a.data / "tab.npz"))               # per-row metadata from the shard's tab (the navsim_zs navtrain index is gone)
        sel = np.flatnonzero((base["speed"] > VMIN) & ~np.isnan(base["fut"]).any((1, 2)))
        sel = sel[: a.limit] if a.limit else sel
        n, T = len(sel), t_all()
        _sys.path[:0] = [str(_R / "experiments/alpasim/scripts"), str(_R / "experiments/alpasim/lib")]
        import ap2_prep as AP                                        # CAM_F0 paths + calibration of the 4 keys, read from the NAVSIM logs
        ents = AP.entries(base["names"][sel].tolist(), base["log"][sel].tolist(), run)
        mpose, mvel, mcam = (base[q][sel].astype(np.float64) for q in ("pose", "vel", "cam"))
        dcam = np.abs(np.array([e["cams"][-1]["CAM_F0"]["t"] for e in ents[:64]]) - mcam[:64]).max()
        assert dcam < 1e-4, f"camera position of the logs and of the op_parity tab differ by {dcam} m"
        shard = int(a.data.split(".s", 1)[1].split("of")[0])
        extra = {}
        if PROFILE is not None:                                      # another perturbation family: its own (y, psi) history per row
            dy, dp, ys, ps, extra = PROFILE(T, base["speed"][sel].astype(np.float64), np.random.default_rng([1, shard]), a.zero)
        else:
            dy, dp = sample(base["speed"][sel].astype(np.float64), np.random.default_rng([1, shard]))
            if a.zero:
                dy, dp = np.zeros(n), np.zeros(n)
            ys, ps = (np.stack(x) for x in zip(*[H.drift(T, dy[i], dp[i], float(base["speed"][r])) for i, r in enumerate(sel)]))   # (n, 10)
        run.info(f"{tag}: {n} of {len(base['names'])} rows (speed > {VMIN} m/s, logged future); |dy| mean {np.abs(dy).mean():.3f} m, |dpsi| mean "
                 f"{np.degrees(np.abs(dp)).mean():.2f} deg, |y(-1.5 s)| max {np.abs(ys[:, 0]).max():.2f} m")
        kbase = dict(data=a.data, sel=cache.key(params=dict(s=sel.tolist())), ot=OT, dy=DY, dpsi=float(DPSI), ymax=YMAX, vmin=VMIN, zero=a.zero, **KEY_EXTRA)
        root, froot = CR / tag, CR / f"{tag}@warp"
        root.mkdir(parents=True, exist_ok=True), froot.mkdir(parents=True, exist_ok=True)

        def make_tab():
            hist = np.asarray(base["pose"][sel], np.float64)                                    # (n, 4, 3) logged history poses, t0 frame
            c, s = np.cos(hist[..., 2]), np.sin(hist[..., 2])
            world = np.stack([hist[..., 0] - s * ys[:, :4], hist[..., 1] + c * ys[:, :4], hist[..., 2] + ps[:, :4]], -1)   # drifted poses
            pose = to_frame(world, dy, dp).astype(np.float32)
            fut = to_frame(np.asarray(base["fut"][sel], np.float64), dy, dp).astype(np.float32)
            tab = {k: v[sel] for k, v in base.items()}
            tab |= dict(pose=pose, fut=fut, ego=PA.ego_features(pose, tab["vel"], tab["acc"], tab["cmd"][:, -1]),
                        off=np.c_[dy, dp].astype(np.float32), src_row=sel, **extra)
            return tab
        tab = cache.cached(root / "tab.npz", cache.key(params=kbase, code=[to_frame, sample, PA.ego_features] + ([PROFILE] if PROFILE else [])), make_tab,
                           force=a.force)
        assert np.abs(tab["pose"][:, 3]).max() < 1e-4, "the perturbed t0 pose must be the origin of its own frame"

        net, enc = PP.encoder(torch.device("cuda"))
        timing = {}

        def make_front():
            _, src = OL._steps(0.0, False)
            at = lambda s: src[s][1] if src[s][0] == "k" else 4 + src[s][1]  # noqa: E731  index into the 10 frames of _job
            W = a.workers or max(1, n_cpus() - 4)
            pool = ProcessPoolExecutor(W)

            def load(rows):
                fr = np.stack(list(pool.map(_job, [(ents[i], mpose[i], mvel[i], mcam[i], T, ys[i], ps[i])
                                                   for i in rows])))
                img = lambda j, s: fr[j][at(s)] if s >= 0 else np.zeros(FRAME, np.uint8)  # noqa: E731
                cur = np.stack([[img(j, s) for s in PP.STEPS] for j in range(len(rows))])
                prev = np.stack([[img(j, s - 4) for s in PP.STEPS] for j in range(len(rows))])
                if rows[0] == 0:
                    np.savez_compressed(froot / "samples.npz", frames=fr[:4], dy=dy[:4], dpsi=dp[:4], names=tab["names"][:4])
                return rows, prev.reshape(-1, *FRAME), cur.reshape(-1, *FRAME)
            out = np.zeros((n, 8) + A.H_SHAPE, np.float16)
            chunks = [np.arange(i, min(i + 32, n)) for i in range(0, n, 32)]
            t0, tg = time.time(), 0.0
            with ThreadPoolExecutor(8) as ex:
                for rows, prev, cur in run.tqdm(PP._bounded(ex, load, chunks, 16), total=len(chunks), desc="front"):
                    t1 = time.time()
                    out[rows] = enc(prev, cur).reshape(len(rows), 8, *A.H_SHAPE)
                    tg += time.time() - t1
            pool.shutdown()
            el = time.time() - t0
            timing.update(tokens_per_s=n / el, gpu_busy=tg / el, workers=W)
            return out
        front = cache.cached(froot / "front.npy", cache.key(params=kbase | dict(steps=PP.STEPS.tolist()), code=[_job, PP.encoder]), make_front,
                             force=a.force)

        def make_teacher():
            di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
            out, plan = np.zeros((n, len(di)), np.float32), np.zeros((n, 33, 15), np.float32)
            tc = np.where(tab["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)
            with torch.no_grad():
                for i in range(0, n, 256):
                    r = slice(i, min(i + 256, n))
                    Hh = torch.from_numpy(np.ascontiguousarray(front[r])).cuda()
                    Hh = torch.cat([torch.zeros_like(Hh[:, :1]), Hh], 1)
                    valid = torch.ones(Hh.shape[:2], dtype=torch.bool, device=Hh.device)
                    valid[:, 0] = False
                    o = A._policy(net, Hh, (0.275, 0.525), torch.from_numpy(tc[r]).cuda(), valid)["outputs"].float()
                    out[r], plan[r] = o[:, di].cpu().numpy(), o[:, pi].cpu().numpy().reshape(-1, 33, 15)
            return dict(out=out, plan=plan, di=di, pi=pi)
        cache.cached(froot / "teacher.npz", cache.key(params=kbase, inputs=[froot / "front.npy"]), make_teacher, force=a.force)
        run.summary |= {"n": n, "n_base": len(base["names"]), "front_shape": list(front.shape), **timing}
        if a.zero:                                                   # pipeline gate: zero offsets must reproduce the stored W tokens of the same rows
            ref = np.load(CR / f"{a.data}@warp" / "front.npy", mmap_mode="r")[sel].astype(np.float32)
            d = np.abs(front.astype(np.float32) - ref)
            rel = float(d.mean() / np.abs(ref).mean())
            run.summary |= {"zero_max_abs": float(d.max()), "zero_rel_mean": rel, "zero_rows_identical": float((d.reshape(n, -1).max(1) == 0).mean()),
                            "zero_fut_max": float(np.abs(tab["fut"] - base["fut"][sel]).max()), "zero_ego_max": float(np.abs(tab["ego"] - base["ego"][sel]).max())}
            run.info(f"zero-offset check: {json.dumps({k: v for k, v in run.summary.items() if k.startswith('zero')})}")
            assert rel < 1e-3 and run.summary["zero_fut_max"] < 1e-4 and run.summary["zero_ego_max"] < 1e-4, "zero offsets do not reproduce the W cache"
        if timing:
            (froot / "timing.json").write_text(json.dumps(timing, indent=1))


# ---------------------------------------------------------------- trainer pieces
def off_hinge(cfg, names, off, dev):
    """lib/drivable_hinge.Hinge whose plans are given in a row's own (perturbed) frame: off (n, 2) = (dy, dpsi) of the row's t0 pose in the
    frame of its SDF raster (zeros on normal rows, where this is exactly Hinge)."""
    import torch
    from drivable_hinge import Hinge

    class OffHinge(Hinge):
        def margins(self, x, y, psi, rows):
            o = self.off[rows]
            dy, dp = o[:, :1], o[:, 1:]
            c, s = torch.cos(dp), torch.sin(dp)
            P = torch.stack([c * x - s * y, dy + s * x + c * y, psi + dp], -1)
            P0 = torch.cat([torch.stack([torch.zeros_like(dy), dy, dp], -1), P], 1)          # the interpolation starts at the perturbed pose
            d = torch.einsum("kj,bjc->bkc", self.M, P0)
            C = self.C[self.src[rows]]
            c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
            cx = d[..., :1] + c * C[:, None, :, 0] - s * C[:, None, :, 1]
            cy = d[..., 1:2] + s * C[:, None, :, 0] + c * C[:, None, :, 1]
            xy = torch.stack([cx, cy], -1).reshape(len(P), -1, 2)
            from drivable_hinge import X0, Y0
            g = torch.stack([(xy[..., 1] - Y0) / 24.0 - 1.0, (xy[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
            return torch.nn.functional.grid_sample(self.sdf[rows].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
    h = OffHinge([data_dir() / f for f in cfg.hinge_labels], names, dev, cfg.hinge_margin, list(cfg.hinge_footprint))
    h.off = torch.as_tensor(off, dtype=torch.float32, device=dev)
    return h


def offsets(datas) -> np.ndarray:
    """(n, 2) (dy, dpsi) per Store row of `datas`: the ot tabs' `off`, zeros for normal dirs."""
    out = []
    for d in datas:
        z = np.load(CR / d / "tab.npz")
        out.append(z["off"] if "off" in z.files else np.zeros((len(z["names"]), 2), np.float32))
    return np.concatenate(out)


def cmd_train(a):
    import torch
    from dataclasses import asdict
    import pp_train as T
    from jevdrive.run import Run
    dev = torch.device("cuda")
    ot = tuple(f"{a.ot}_{d}" for d in a.data) if a.ot_mass > 0 else ()
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data) + ot, split=a.split, frames="warp", host=True, warmup=a.warmup,
                eval_every=a.eval_every, hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 0])                       # pp_train's row stream (identical draws when --ot-mass 0)
    S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
    tr_rows, dv_rows, sp = T.split_rows(dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d, is_wod=S.is_wod), cfg.split)
    n_base = sum(len(np.load(CR / d / "tab.npz")["names"]) for d in a.data)
    is_ot = np.arange(S.n) >= n_base
    tr_o, tr_t, dv_o, dv_t = tr_rows[~is_ot[tr_rows]], tr_rows[is_ot[tr_rows]], dv_rows[~is_ot[dv_rows]], dv_rows[is_ot[dv_rows]]
    nB = cfg.batch
    k = int(round(nB * a.ot_mass))
    assert (k == 0) == (len(ot) == 0) and (k == 0 or len(tr_t) >= k)
    model = T.PModel("P2").to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_o, device=dev)].float().std(0).clamp_min(1e-3)        # the normaliser of the reference recipe (normal rows)
    off = offsets(cfg.data)
    hinge = off_hinge(cfg, S.tab["names"], off, dev) if cfg.hinge_lam > 0 else None
    LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev, hinge, None)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = T.proot("runs", a.tag)
    with Run("op_parity", f"train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"ot": vars(a)}) as run:
        for x in sp:
            run.use_split(x)
        run.info(f"{a.tag}: train {len(tr_o)} normal + {len(tr_t)} off-track rows, dev {len(dv_o)} + {len(dv_t)}; {k} off-track rows per batch of {nB} "
                 f"({k / nB:.3f}); hinge coverage {hinge.coverage if hinge else 0:.4f}; off-track |dy| mean {np.abs(off[is_ot, 0]).mean() if k else 0:.3f} m")

        def draw():
            if k:
                r = np.concatenate([rng.choice(tr_o, nB - k, replace=len(tr_o) < nB - k), rng.choice(tr_t, k, replace=False)])
            else:
                r = rng.choice(tr_o, nB, replace=len(tr_o) < nB)
            an = rng.random(nB) < cfg.d_frac
            an[nB - k:] = False                                      # off-track rows are imitation rows only
            return r, an

        def fetch(dr):
            return dr, S.front[dr[0]]
        pre = ThreadPoolExecutor(2)
        nxt = pre.submit(fetch, draw())
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            (r, an), front = nxt.result()
            if step + 1 < cfg.steps:
                nxt = pre.submit(fetch, draw())
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            ego = S.ego[rows] * (~anchor)[:, None].float()
            total, Ls = LS(model(front, ego, S.tc[rows]), S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({q: float(v) for q, v in Ls.items()})
            if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                mean = {q: float(np.mean([h[q] for h in hist if q in h])) for q in hist[-1]}
                hist = []
                el = time.time() - t0
                run.scalars({f"loss/{q}": v for q, v in mean.items()} | {"throughput/steps_per_s": (step + 1) / el}, step + 1)
                run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in mean.items()) + f"; {(step + 1) / el:.2f} it/s, "
                         f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                run.status(f"step {step + 1}/{cfg.steps}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = T.dev_eval(model.eval(), S, dv_o, LS.W)
                if len(dv_t):
                    ev |= {"ot_" + q: v for q, v in T.dev_eval(model, S, dv_t, LS.W).items()}
                model.train()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.3f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "ot": vars(a)}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), n_train=len(tr_o), n_train_ot=len(tr_t), ot_per_batch=k)


# ---------------------------------------------------------------- probe: do the plans react to the offset, and with the right sign
def cmd_probe(a):
    import torch
    import pp_train as T
    from jevdrive import stats
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", f"otprobe-{a.name}", config=vars(a)) as run:
        ot = tuple(f"{OT}_{d}{a.suffix}" for d in a.data)
        S = T.Store(ot, dev, need_side=False, frames="warp", host=True)
        B = T.Store(tuple(a.data), dev, need_side=False, frames="warp", host=True)
        nb = np.cumsum([0] + [len(np.load(CR / d / "tab.npz")["names"]) for d in a.data])     # Store row offset of every base dir
        src = np.concatenate([np.load(CR / d / "tab.npz")["src_row"] + o for d, o in zip(ot, nb)])   # the base Store row of every off-track row
        assert (B.tab["names"][src] == S.tab["names"]).all()
        off = offsets(ot)
        rows = np.arange(S.n)
        if a.split:
            from jevdrive.data import splits
            sp = splits.load(f"{a.split}-dev")
            run.use_split(sp)
            rows = rows[sp.mask(S.tab["names"])]
        rows = rows[: a.limit] if a.limit else rows
        W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        pi = torch.as_tensor(S.pi, device=dev)

        def plans(model, St, rr):
            out = []
            with torch.no_grad():
                for i in range(0, len(rr), 256):
                    r = torch.as_tensor(rr[i:i + 256], device=dev)
                    p = model(St.front[r], St.ego[r], St.tc[r]).float()[:, pi].view(-1, 33, 15)
                    out.append(torch.stack(T.rear(p, St.cam_x[r], W), -1).cpu().numpy())
            return np.concatenate(out)
        fut_o, fut_b, log = S.tb["fut"][rows], B.tb["fut"][src[rows]], S.tb["log"][rows]
        # the ideal lateral shift of the plan between the two frames, split into its lateral-offset and its yaw part (to_frame, per pose)
        dy_, dp_ = off[rows, 0:1].astype(np.float64), off[rows, 1:2].astype(np.float64)
        Xd, Xp = -dy_ * np.cos(dp_) * np.ones_like(fut_b[..., 0]), -np.sin(dp_) * fut_b[..., 0] + (np.cos(dp_) - 1) * fut_b[..., 1]
        res, L = {}, ["| model | rows | ADE off-track rows (m) | ADE same tokens, logged pose (m) | lateral error at 4 s off-track (m) | "
                      "response to the lateral offset at 1 / 2 / 4 s | response to the yaw offset at 1 / 2 / 4 s |", "|:--|--:|--:|--:|--:|:--|:--|"]
        for tag in a.tags:
            m = T.load_pmodel(tag, dev)
            po, pb = plans(m, S, rows), plans(m, B, src[rows])
            ade_o = np.linalg.norm(po[..., :2] - fut_o[..., :2], axis=-1).mean(1)
            ade_b = np.linalg.norm(pb[..., :2] - fut_b[..., :2], axis=-1).mean(1)
            sl = {}                                                     # least squares: 1 = the plan moves exactly with the target, 0 = no reaction
            for t, j in (("1s", 1), ("2s", 3), ("4s", 7)):
                c = np.linalg.lstsq(np.c_[Xd[:, j], Xp[:, j]], po[:, j, 1] - pb[:, j, 1], rcond=None)[0]
                sl[f"dy_{t}"], sl[f"yaw_{t}"] = float(c[0]), float(c[1])
            res[tag] = dict(n=len(rows), ade_ot=stats.bootstrap(ade_o, groups=log), ade_base=stats.bootstrap(ade_b, groups=log),
                            lat4_ot=float(np.abs(po[:, 7, 1] - fut_o[:, 7, 1]).mean()), slope=sl)
            L.append(f"| {tag} | {len(rows)} | {ade_o.mean():.3f} | {ade_b.mean():.3f} | {res[tag]['lat4_ot']:.3f} | "
                     f"{sl['dy_1s']:.2f} / {sl['dy_2s']:.2f} / {sl['dy_4s']:.2f} | {sl['yaw_1s']:.2f} / {sl['yaw_2s']:.2f} / {sl['yaw_4s']:.2f} |")
            del m
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"probe_{a.name}.json").write_text(json.dumps(res, indent=1, default=float))
        (OUT / f"probe_{a.name}.md").write_text("\n".join(L) + "\n")
        run.info("\n" + "\n".join(L))
        run.summary |= {t: res[t]["slope"] for t in res}
        if a.min_slope is not None:
            assert all(min(res[t]["slope"]["yaw_4s"], res[t]["slope"]["dy_4s"]) >= a.min_slope for t in res), \
                f"plan response to the yaw or the lateral offset at 4 s below {a.min_slope} (sign / geometry gate)"


# ---------------------------------------------------------------- benchmark reads (through jevdrive.bench's stored units)
def _paired(bench, new, ref, col):
    """Seed-mean per unit of `new` minus `ref` (lists of bench specs), log-cluster bootstrap."""
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    U = {k: [BT.load(bench, s)[0] for s in v] for k, v in (("new", new), ("ref", ref))}
    assert all(u is not None for v in U.values() for u in v), f"missing {bench} units"
    idx = U["ref"][0].index
    for v in U.values():
        for u in v:
            assert len(u) == len(idx) and not u.reindex(idx)[col].isna().any(), f"{bench}: unit sets differ"
    x, y = (np.mean([u.reindex(idx)[col].to_numpy(float) for u in U[k]], 0) for k in ("new", "ref"))
    sc = 100.0 if max(np.abs(x).max(), np.abs(y).max()) <= 1.0 + 1e-9 else 1.0
    r = stats.paired(x * sc, y * sc, groups=U["ref"][0].reindex(idx)["log"].to_numpy())
    return dict(new=r["mean_a"], ref=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"], n=int(r["n"]), units=int(r["units"]),
                per_seed_new=[float(u[col].mean() * sc) for u in U["new"]], per_seed_ref=[float(u[col].mean() * sc) for u in U["ref"]])


def cmd_gate(a):
    """Pilot gate: navhard (G frames) combined, new - ref; exit 2 = stop (point estimate below +1.0)."""
    res = {c: _paired("navhard", [f"{a.new}@gimm"], [f"{r}@gimm" for r in a.refs], c) for c in ("combined", "stage1", "stage2")}
    res |= {f"combined vs {r}": _paired("navhard", [f"{a.new}@gimm"], [f"{r}@gimm"], "combined") for r in a.refs}
    g = res["combined"]
    stop = bool(g["diff"] < 1.0)
    res["gate"] = dict(rule="stop iff the navhard combined point estimate (new - mean of the reference checkpoints) < +1.0", diff=g["diff"], stop=stop, new=a.new, refs=a.refs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(2 if stop else 0)


def cmd_report(a):
    """Verdict lines of the pre-registration: navhard combined >= +2.0 with CI lower bound > 0, and navtest >= -0.2."""
    g = lambda v: [f"{s}@gimm" for s in v]  # noqa: E731
    res = {"navhard": {c: _paired("navhard", g(a.arms), g(a.refs), c) for c in ("combined", "stage1", "stage2")},
           "navtest": {c: _paired("navtest", a.arms, a.refs, c) for c in ("score", "NC", "DAC", "EP", "TTC", "LK")}}
    nh, nt = res["navhard"]["combined"], res["navtest"]["score"]
    res["verdict"] = dict(navhard_pass=bool(nh["diff"] >= 2.0 and nh["lo"] > 0), navtest_pass=bool(nt["diff"] >= -0.2), arms=a.arms, refs=a.refs,
                          rule="navhard combined (arm - ref) >= +2.0 and CI lower bound > 0; navtest EPDMS (arm - ref) >= -0.2")
    res["verdict"]["candidate"] = res["verdict"]["navhard_pass"] and res["verdict"]["navtest_pass"]
    f = lambda r, p=2: f"{r['new']:.{p}f} | {r['ref']:.{p}f} | {r['diff']:+.{p}f} [{r['lo']:+.{p}f}, {r['hi']:+.{p}f}]"  # noqa: E731
    L = [f"Arms {' + '.join(a.arms)} minus refs {' + '.join(a.refs)} (seed means per unit; percentile bootstrap over logs, B 10000).", "",
         "| read-out | arm | ref | arm - ref [95% CI] |", "|:--|--:|--:|:--|"]
    L += [f"| navhard {c} ({res['navhard'][c]['n']} groups, {res['navhard'][c]['units']} logs) | {f(res['navhard'][c])} |" for c in res["navhard"]]
    L += [f"| navtest {'EPDMS' if c == 'score' else c} ({res['navtest'][c]['n']} tokens) | {f(res['navtest'][c])} |" for c in res["navtest"]]
    L += ["", f"Verdict: navhard line {'passed' if res['verdict']['navhard_pass'] else 'not passed'}, navtest line "
              f"{'passed' if res['verdict']['navtest_pass'] else 'not passed'}."]
    out = _pl.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--zero", action="store_true", help="zero offsets: the pipeline must reproduce the stored W tokens (gate)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--split", default="navsim/op-parity-s234")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--hinge-lam", type=float, default=30.0)
    p.add_argument("--hinge-margin", type=float, default=0.5)
    p.add_argument("--ot-mass", type=float, default=0.1, help="share of every batch drawn from the off-track rows (0 = the reference recipe)")
    p.add_argument("--ot", default=OT, help="cache prefix of the off-track rows (ot1: +-0.5 m / 2 deg; ot2: +-1.5 m / 5 deg, ap2_ot.py; yr1: ot3_rows.py)")
    p = sub.add_parser("probe")
    p.add_argument("--name", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--suffix", default="", help="cache suffix of the off-track dirs, e.g. -first64")
    p.add_argument("--split", default="", help="restrict to <split>-dev tokens")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-slope", type=float, default=None, help="gate: least-squares response to the yaw offset at 4 s of every tag")
    p = sub.add_parser("gate")
    p.add_argument("--new", required=True)
    p.add_argument("--refs", nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--refs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "probe": cmd_probe, "gate": cmd_gate, "report": cmd_report}[a.cmd](a)
