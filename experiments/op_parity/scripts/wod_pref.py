"""Rater preference trained into System 1 on WOD-E2E val (plans/2026-10-08-wod-pref-prereg.md, results/wod_pref.md): WLG (WP2 + standstill
gate) fine-tuned on the rated trajectories of the 479 val rater frames by sequence-level 5-fold, read out-of-fold against WLG.

  prep    (op-train env, one GPU, a few render workers) the token path of every rater frame: its own 10-frame stream f - 18 .. f step 2 (the
          renderer and image-pair protocol of the wod_r2 train cache) -> frozen trunk -> stage 4 -> $DATA_DIR/runs/op_parity/cache/wod_rater/
          {front.npy (479, 9, 32, 512) fp16, tab.npz, teacher.npz, frames_t0.npy (479, 2, 6, 128, 256) u8 road / wide model frames at t0};
          gates P0 (new tokens == the wodval cache on the even frames) and P1 (WLG token path == the stored harness run) -> gates.json
  train   (op-train env, one GPU) one objective x seed over the given outer folds: WLG-full-s<seed> fine-tuned (adapter + plan pathway, vision
          frozen) on 16 preference rows + 48 WOD-train rows (the unchanged WLG recipe) per batch; every --eval-every steps the plans of all 479
          frames are stored; early stopping = the step with the best frame-mean RFS on the inner fold (k + 1) % 5
          -> $DATA_DIR/runs/op_parity/wod/pref/<tag>/fold<k>.npz. --fold all: every rater frame fitted, fixed steps, checkpoint saved
          (runs/op_parity/runs/<tag>/ckpt-final.pt, loadable by pp_train.load_pmodel).
  pick    (CPU) the pilot's learning-rate choice from the inner-fold RFS only -> results/wod_pref/pilot.{csv,json}
  report  (jevdrive env, CPU) objective x {out-of-fold, in-sample, permutation control}, strata, clusters, ADE, share of the decision-168
          ceiling, learning curves, verdict -> results/wod_pref/
  figs    (jevdrive env, CPU) figs/wod_pref/{curves.png, frames.png}

Objectives on the preference rows (the plan as 20 WOD rear-axle waypoints at 0.25 .. 5 s; d = normalised Huber distance, sigma 0.3 + 0.2 t / 0.1 + 0.1 t):
  top    d(plan, top-rated trajectory)
  rank   -log sum_k softmax(score / 2)_k exp(-d(plan, traj_k))
  hinge  the metric in the log domain: mean over the 3 s / 5 s checks of -max_k (log10 max(score_k, 0.01) - relu(norm_kh - 1))
  f20    sampled reward: the best-RFS candidate of the F20 family (s2_gohold.factored) of the model's own detached plan, regressed with d
--perm: the (trajectories, scores) of each training frame replaced by another training frame's of the same v0 bin (the control).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT, FIG = _R / "experiments/op_parity/results/wod_pref", _R / "experiments/op_parity/figs/wod_pref"
OBJS = ("top", "rank", "hinge", "f20")
K, BASE = 5, "WLG-full-s%d"
N_PREF, N_ANCHOR = 16, 48
LAM_P = 1.0
T_FUT = 0.25 * np.arange(1, 21)
SIG = np.stack([0.3 + 0.2 * T_FUT, 0.1 + 0.1 * T_FUT], -1)          # (20, 2) x / y scale of the normalised distance
HK = np.array([11, 19])                                             # waypoint indices of the 3 s / 5 s checks
VBINS = (0.5, 2.0, 5.0, 8.0, 12.0)                                  # permutation control: labels move inside these v0 bins
LRS = ((3e-6, 3e-5), (1e-5, 1e-4), (3e-5, 3e-4))                    # pilot grid (plan pathway, adapter)
N_DEV = 512
B = 4000


def cdir():
    return data_dir() / "runs/op_parity/cache/wod_rater"


def pdir(tag=""):
    d = data_dir() / "runs/op_parity/wod/pref" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def folds_of(seq):
    """Outer fold index of each sequence (wod/pref5-f0..4) and the Split objects."""
    from jevdrive.data import splits
    sp = [splits.load(f"wod/pref5-f{k}") for k in range(K)]
    splits.check_disjoint(*sp)
    f = np.full(len(seq), -1)
    for k, s in enumerate(sp):
        f[s.mask(seq)] = k
    assert (f >= 0).all(), "rater sequences outside the folds"
    return f, sp


# ---------------------------------------------------------------- prep
def _render(names):
    import drive_backbones_openpilot as R
    return R.render(names)


def cmd_prep(a):
    import torch
    from concurrent.futures import ProcessPoolExecutor
    import pp_train as T
    import pp_wod as PW
    import wod_parity as WP
    import wod_zeroshot_openpilot as WZ
    from jevdrive import op_adapt as A
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    with Run("op_parity", "wod-pref-prep", config=vars(a)) as run:
        r = Z.load_sets()["rater"]
        names, seq = r["name"].astype(str), r["sequence"].astype(str)
        if a.limit:
            names, seq = names[: a.limit], seq[: a.limit]
        n = len(names)
        fold, sp = folds_of(seq)
        for s in sp:
            run.use_split(s)
        spans, _ = Z.load_spans()
        cal = json.loads((Z.root() / "op_calib.json").read_text())
        streams = [[f"{s}-{int(nm.rsplit('-', 1)[1]) - 2 * j:03d}" for j in range(9, -1, -1)] for nm, s in zip(names, seq)]
        have = np.array([[x in spans for x in st] for st in streams])     # a frame missing from the shards = a stream that starts late (zero image / zero hidden state)
        assert have[:, -1].all(), "a target frame is missing from the slim shards"
        run.info(f"{int((~have.all(1)).sum())} of {n} rater frames have holes in their 1.8 s history: {names[~have.all(1)].tolist()}")
        out = cdir() if not a.limit else cdir().with_name(f"wod_rater-first{a.limit}")
        out.mkdir(parents=True, exist_ok=True)
        front = np.lib.format.open_memmap(out / "front.npy", "w+", np.float16, (n, WP.NCTX, 32, 512))
        f0 = np.lib.format.open_memmap(out / "frames_t0.npy", "w+", np.uint8, (n, 2, 6, 128, 256))
        t0 = time.time()
        with ProcessPoolExecutor(a.workers, initializer=WZ._init, initargs=(spans, cal, str(data_dir() / "datasets/waymo_e2e/front3"))) as ex:
            list(ex.map(int, range(a.workers)))                           # fork the render workers before CUDA exists in this process
            net = A.load("cinque", torch.float16).cuda().eval()
            with torch.no_grad():
                for i, got in enumerate(ex.map(_render, [[x for x, h in zip(st, hv) if h] for st, hv in zip(streams, have)], chunksize=4)):
                    fr = np.zeros((10, 2, 6, 128, 256), np.uint8)
                    fr[have[i]] = got
                    x = torch.as_tensor(fr).cuda()                         # (10, 2, 6, 128, 256): pair (f - 2, f) = (x[j - 1], x[j]), j = 1 .. 9
                    tr = net.run_batched(A.vision_feeds(x[:-1], x[1:]), [A.TRUNK_OUT])[A.TRUNK_OUT][:, 0].to(torch.float16)
                    tk = net.run_batched({A.TRUNK_OUT: tr[:, None].to(net.dtype)}, ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                    tk = tk * torch.as_tensor(have[i, 1:], device=tk.device)[:, None, None].to(tk.dtype)
                    front[i], f0[i] = tk.cpu().numpy(), fr[-1]
                    if (i + 1) % 100 == 0 or i + 1 == n:
                        run.info(f"[{i + 1}/{n}] streams, {time.time() - t0:.0f} s")
        front.flush(), f0.flush()
        del net
        past, intent = r["past"][:n], r["intent"][:n]
        ego, pose = PW.wod_ego(past, intent)
        cam = np.array([np.array(cal[s]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for s in seq], np.float32)
        tab = dict(names=names, log=seq, ego=ego, pose=pose.astype(np.float32), intent=intent, fut=WP.fut_targets(r["future"][:n]), cam=cam,
                   lht=np.zeros(n, bool), n_real=have[:, 1:].sum(1), speed=W.past_kinematics(past)["v"].astype(np.float32), v0=W.init_speed(past).astype(np.float32),
                   future=r["future"][:n, :, :2].astype(np.float32), traj=r["traj"][:n].astype(np.float32), scores=r["scores"][:n],
                   cluster=r["cluster"][:n].astype(str), fold=fold)
        np.savez(out / "tab.npz", **tab)
        dev = torch.device("cuda")
        H = torch.from_numpy(np.ascontiguousarray(front)).to(dev)
        tc = torch.tensor([[1.0, 0.0]], device=dev)
        Wt = torch.as_tensor(wt(), device=dev)

        def fwd(m, Hh, e):
            with torch.no_grad():
                return torch.cat([m(Hh[i:i + 160], e[i:i + 160], tc.expand(len(Hh[i:i + 160]), 2)).float() for i in range(0, len(Hh), 160)])
        p0 = T.load_pmodel("P0", dev)
        di, pi = A.distill_index(p0.net.slices), A.plan_index(p0.net.slices)
        o = fwd(p0, H, torch.zeros(n, ego.shape[1], device=dev))
        np.savez(out / "teacher.npz", out=o[:, di].cpu().numpy(), plan=o[:, pi].view(-1, 33, 15).cpu().numpy(), di=di, pi=pi)
        del p0
        eg = torch.from_numpy(gate(ego)).to(dev)
        dxy = torch.from_numpy(cam[:, :2]).to(dev)
        plans, G = {}, {}
        for s in (0, 1):
            m = T.load_pmodel(BASE % s, dev)
            plans[s] = to_wod_t(fwd(m, H, eg)[:, pi].view(-1, 33, 15), dxy, Wt).double().cpu().numpy()
            if s == 0:                                                     # gate P0: the wodval cache's tokens on the even frames it holds
                sts = json.loads((WP.SRC / "wod_val_plan.json").read_text())["streams"]
                _, _, rows = WP.stream_rows(sts, set(names.tolist()), need_future=False)
                idx = {nm: i for i, nm in enumerate(names)}
                dt, dp = [], []
                for si in sorted({q[0] for q in rows}):
                    z = np.load(WP.SRC / "wodval" / f"{sts[si]['key']}.npz")
                    x = torch.from_numpy(z["trunk"]).to(dev)
                    with torch.no_grad():
                        tk = m.net.run_batched({A.TRUNK_OUT: x[:, None].to(m.net.dtype)}, ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                    for q in [q for q in rows if q[0] == si]:
                        i = idx[q[2]]
                        Hv = tk[q[1] - WP.NCTX + 1: q[1] + 1][None]
                        dt.append(float((Hv[0].float() - H[i].float()).abs().max()))
                        pv = to_wod_t(fwd(m, Hv, eg[i:i + 1])[:, pi].view(-1, 33, 15), dxy[i:i + 1], Wt).double().cpu().numpy()[0]
                        dp.append(float(np.linalg.norm(pv - plans[0][i], axis=-1).mean()))
                G["P0"] = dict(n=len(dp), plan_mean_m=float(np.mean(dp)) if dp else None, plan_max_m=float(np.max(dp)) if dp else None,
                               token_max_abs=float(np.max(dt)) if dt else None, passed=bool(dp and np.mean(dp) < 0.01))
            del m
        v0, cl = tab["v0"].astype(np.float64), tab["cluster"]
        rf = lambda p: np.asarray(W.rater_feedback_score(p, tab["traj"], tab["scores"], v0), float)  # noqa: E731
        hp = {s: np.stack([np.load(Z.root("preds", f"op_cinque_{BASE % s}") / f"{nm}.npz")["wod"] for nm in names]).astype(np.float64)[..., :2] for s in (0, 1)}
        rt, rh = np.mean([rf(plans[s]) for s in (0, 1)], 0), np.mean([rf(hp[s]) for s in (0, 1)], 0)
        dm = float(np.mean([np.linalg.norm(plans[s] - hp[s], axis=-1).mean() for s in (0, 1)]))
        G["P1"] = dict(plan_mean_m=dm, rfs_token=W.rfs_by_cluster(rt, cl)[0], rfs_harness=W.rfs_by_cluster(rh, cl)[0],
                       rfs_frame_abs_diff_mean=float(np.abs(rt - rh).mean()), plan_p99_m=float(np.percentile(np.linalg.norm(plans[0] - hp[0], axis=-1).mean(1), 99)))
        G["P1"]["passed"] = bool(dm < 0.05 and abs(G["P1"]["rfs_token"] - G["P1"]["rfs_harness"]) < 0.05)
        np.savez(out / "wlg_tok.npz", names=names, s0=plans[0], s1=plans[1], h0=hp[0], h1=hp[1])
        (out / "gates.json").write_text(json.dumps(G, indent=1))
        run.summary |= {"n": n, "P0": G["P0"], "P1": G["P1"]}
        run.info(json.dumps(G))
        if not a.limit and not (G["P0"]["passed"] and G["P1"]["passed"]):
            raise SystemExit(3)


# ---------------------------------------------------------------- shared pieces
def wt():
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    return R2.t_weights(T_FUT)


def gate(ego, v=0.5):
    """The WLG serving / training rule: the whole ego row zeroed (adapter off) where the fed speed is below v m/s."""
    return (ego * (ego[:, 4:5] * 10.0 >= v)).astype(np.float32)


def to_wod_t(plan, dxy, Wt):
    """plan (B, 33, 15) camera frame (x fwd, y right, ch 11 yaw), dxy (B, 2) camera x, y on the vehicle -> (B, 20, 2) WOD rear-axle waypoints at
    0.25 .. 5 s (+y left); the torch form of jevdrive.wod_zeroshot.openpilot_to_wod."""
    import torch
    px, py, psi = plan[..., 0], -plan[..., 1], -plan[..., 11]
    dx, dy = dxy[:, 0:1], dxy[:, 1:2]
    c, s = torch.cos(psi), torch.sin(psi)
    return torch.stack([(dx + px - (c * dx - s * dy)) @ Wt.T, (dy + py - (s * dx + c * dy)) @ Wt.T], -1)


def vbin(v0):
    return np.digitize(v0, VBINS)


# ---------------------------------------------------------------- train
def cmd_train(a):
    import torch
    import torch.nn.functional as F
    import pp_train as T
    import s2_gohold as S2
    from jevdrive import waymo as W
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    tag = a.tag or f"{a.obj}-s{a.seed}" + ("-perm" if a.perm else "")
    full = a.folds == ["all"]
    with Run("op_parity", f"wod-pref-{tag}", seed=a.seed, config=vars(a) | dict(n_pref=N_PREF, n_anchor=N_ANCHOR, lam_p=LAM_P, base=BASE % a.seed)) as run:
        ck = torch.load(T.proot("runs", BASE % a.seed) / "ckpt-final.pt", map_location="cpu", weights_only=False)
        cfg = T.Cfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
        assert cfg.stop_gate == 0.5 and cfg.data == ("wod_r2",)
        tr_s, dv_s = splits.load("wod/r2-train"), splits.load("wod/r2-dev")
        tb = dict(np.load(cdir() / "tab.npz"))
        n = len(tb["names"])
        fold, fsp = folds_of(tb["log"])
        assert (fold == tb["fold"]).all()
        splits.check_disjoint(tr_s, dv_s, *fsp)
        for s in (tr_s, dv_s, *fsp):
            run.use_split(s)
        # ---- WOD-train rows (the unchanged WLG recipe)
        S = T.Store(cfg.data, dev, need_side=False, frames=cfg.frames, host=True)
        S.ego = S.ego * (S.ego[:, 4:5] * 10.0 >= cfg.stop_gate).float()
        tr_rows, dv_rows = np.flatnonzero(tr_s.mask(S.tab["log"])), np.flatnonzero(dv_s.mask(S.tab["log"]))
        dv_rows = np.sort(np.random.default_rng(0).choice(dv_rows, N_DEV, replace=False))
        model = T.PModel(ck["model"]["arm"]).to(dev)
        tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
        LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev)
        # ---- preference rows
        tz = np.load(cdir() / "teacher.npz")
        assert (tz["di"] == S.di).all() and (tz["pi"] == S.pi).all()
        t = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        Hr = t(np.load(cdir() / "front.npy"))
        ego_r, dxy, tout_r = t(gate(tb["ego"])), t(tb["cam"][:, :2].astype(np.float32)), t(tz["out"])
        tc_r = torch.tensor([[1.0, 0.0]], device=dev).expand(n, 2)
        v0 = tb["v0"].astype(np.float64)
        traj_true, sc_true = tb["traj"].astype(np.float64), tb["scores"].astype(np.float64)
        Wt, sig, pi = t(wt()), t(SIG.astype(np.float32)), torch.as_tensor(S.pi, device=dev)
        nonplan = ~LS.plan_cols
        cb = json.loads((_R / "experiments/op_parity/results/s2_gohold/codebook.json").read_text())
        scale = np.clip(0.5 + 0.5 * (v0 - 1.4) / (11 - 1.4), 0.5, 1.0)
        thr = t(np.stack([scale[:, None] * np.array(W.RFS_BASE_THRESHOLDS) * m for m in W.RFS_MULTIPLIERS], 1).astype(np.float32))   # (n, [lat, lng], 2 horizons)
        rfs = lambda p, tj, sc, i: np.asarray(W.rater_feedback_score(p, tj[i], sc[i], v0[i]), float)  # noqa: E731

        def dist(p, q):
            """Normalised Huber distance of waypoint sets p (.., 20, 2) and q (broadcast) -> (..,)."""
            z = (p - q) / sig
            return F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0).sum(-1).mean(-1)

        def plans_of(rows, train_mode=False):
            o = model(Hr[rows], ego_r[rows], tc_r[rows])
            return o, to_wod_t(o.float()[:, pi].view(-1, 33, 15), dxy[rows], Wt)

        @torch.no_grad()
        def eval_all():
            model.eval()
            p = torch.cat([plans_of(torch.arange(i, min(i + 128, n), device=dev))[1] for i in range(0, n, 128)]).double().cpu().numpy()
            ade = []
            for i in range(0, len(dv_rows), 128):
                r = torch.as_tensor(dv_rows[i:i + 128], device=dev)
                x, y, _ = T.rear(model(S.front[r], S.ego[r], S.tc[r]).float()[:, pi].view(-1, 33, 15), S.cam_x[r], LS.W)
                ade.append(torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1))
            ade = float(torch.cat(ade).mean())
            model.train()
            return p, ade

        for fk in a.folds:
            f = pdir(tag) / f"fold{fk}.npz"
            if f.exists() and not a.force:
                run.info(f"{tag} fold {fk}: exists, skipped")
                continue
            role = np.zeros(n, np.int8)                                    # 0 fit, 1 inner (early stopping), 2 out-of-fold
            if not full:
                k = int(fk)
                role[fold == (k + 1) % K], role[fold == k] = 1, 2
            fit, inner = np.flatnonzero(role == 0), np.flatnonzero(role == 1)
            rng = np.random.default_rng([a.seed, 0 if full else int(fk)])  # one row stream per (seed, fold): shared by every objective
            lab = np.arange(n)
            if a.perm:                                                     # labels move inside v0 bins, among the training frames (fit + inner) only
                prng = np.random.default_rng([a.seed, 0 if full else int(fk), 99])
                pool = np.flatnonzero(role < 2)
                for b in np.unique(vbin(v0[pool])):
                    i = pool[vbin(v0[pool]) == b]
                    lab[i] = prng.permutation(i)
            traj, sc = traj_true[lab], sc_true[lab]
            lng, lat = W._rater_frames(traj)
            top_t = t(traj[np.arange(n), sc.argmax(1)].astype(np.float32))
            traj_t, sc_t = t(traj.astype(np.float32)), t(sc.astype(np.float32))
            lng_t, lat_t = t(lng[:, :, HK].astype(np.float32)), t(lat[:, :, HK].astype(np.float32))

            def pref_loss(wod, rows, rows_np):
                if a.obj == "top":
                    return dist(wod, top_t[rows]).mean()
                if a.obj == "rank":
                    d = dist(wod[:, None], traj_t[rows])                                       # (B, 3)
                    return -torch.logsumexp(torch.log_softmax(sc_t[rows] / 2.0, -1) - d, -1).mean()
                if a.obj == "hinge":
                    v = wod[:, None, HK] - traj_t[rows][:, :, HK]                              # (B, 3, 2 horizons, 2)
                    nrm = torch.maximum((lng_t[rows] * v).sum(-1).abs() / thr[rows][:, None, 1], (lat_t[rows] * v).sum(-1).abs() / thr[rows][:, None, 0])
                    q = torch.log10(sc_t[rows].clamp_min(0.01))[:, :, None] - F.relu(nrm - 1.0)
                    return -q.max(1).values.mean()
                p = wod.detach().double().cpu().numpy()                                        # f20: sampled reward on the candidates of the own plan
                c = S2.factored(p, v0[rows_np], 3, cb)
                J = np.stack([rfs(np.ascontiguousarray(c[:, m]), traj, sc, rows_np) for m in range(c.shape[1])])
                tgt = c[np.arange(len(p)), S2.first_best(J)]
                return dist(wod, t(tgt.astype(np.float32))).mean()

            model.load_state(ck["model"])
            model.train()
            base, new = model.groups()
            opt = torch.optim.AdamW([{"params": base, "lr": a.lr, "base": a.lr}, {"params": new, "lr": a.lr_new, "base": a.lr_new}], weight_decay=cfg.wd)
            scaler = torch.amp.GradScaler()
            steps, plans, dev_ade, hist, t0 = [], [], [], [], time.time()

            def snap(step):
                p, ade = eval_all()
                steps.append(step), plans.append(p.astype(np.float32)), dev_ade.append(ade)
                r_lab, r_true = rfs(p, traj, sc, np.arange(n)), rfs(p, traj_true, sc_true, np.arange(n))
                run.info(f"{tag} fold {fk} step {step}: RFS fit {r_lab[fit].mean():.3f} inner {r_lab[inner].mean() if len(inner) else float('nan'):.3f} "
                         f"oof {r_true[role == 2].mean() if (role == 2).any() else float('nan'):.3f}; dev ADE {ade:.3f}; "
                         + ", ".join(f"{k_} {np.mean([h[k_] for h in hist[-a.eval_every:]]):.4f}" for k_ in (hist[-1] if hist else {})))
            snap(0)
            for step in range(a.steps):
                rp = rng.choice(fit, N_PREF, replace=False)
                ra = rng.choice(tr_rows, N_ANCHOR, replace=False)
                an = torch.as_tensor(rng.random(N_ANCHOR) < cfg.d_frac, device=dev)
                rows_a, rows_p = torch.as_tensor(ra, device=dev), torch.as_tensor(rp, device=dev)
                for g in opt.param_groups:
                    g["lr"] = g["base"] * min(1.0, (step + 1) / a.warmup)
                out = model(torch.cat([S.front[ra], Hr[rows_p]]), torch.cat([S.ego[rows_a] * (~an)[:, None].float(), ego_r[rows_p]]),
                            torch.cat([S.tc[rows_a], tc_r[rows_p]]))
                total, Ls = LS(out[:N_ANCHOR], S, rows_a, an)
                op = out[N_ANCHOR:].float()
                wod = to_wod_t(op[:, pi].view(-1, 33, 15), dxy[rows_p], Wt)
                Ls["pref"] = pref_loss(wod, rows_p, rp)
                e = ((op[:, LS.di] - tout_r[rows_p]) / tstd).pow(2) * LS.dmask
                Ls["distill_p"] = (e[:, nonplan].sum(1) / e.shape[1]).mean()
                total = total + LAM_P * Ls["pref"] + cfg.lam_d * Ls["distill_p"]
                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite loss at step {step}: { {k_: float(v) for k_, v in Ls.items()} }")
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(base + new, 1.0)
                scaler.step(opt)
                scaler.update()
                hist.append({k_: float(v.detach()) for k_, v in Ls.items()})
                if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
                    snap(step + 1)
                    run.status(f"{tag} fold {fk} step {step + 1}/{a.steps}")
            P = np.stack(plans)
            inner_rfs = np.array([rfs(p.astype(np.float64), traj, sc, np.arange(n))[inner].mean() if len(inner) else np.nan for p in P])
            sel = len(steps) - 1 if full else int(np.argmax(inner_rfs))                      # ties -> the earliest step
            np.savez(f, steps=np.array(steps), plans=P, role=role, lab=lab, sel=sel, inner_rfs=inner_rfs, dev_ade=np.array(dev_ade),
                     loss=np.array([[h.get(k_, np.nan) for k_ in ("imit", "cons", "distill", "pref", "distill_p")] for h in hist], np.float32),
                     names=tb["names"], lr=a.lr, lr_new=a.lr_new, train_s=time.time() - t0)
            run.info(f"{tag} fold {fk}: selected step {steps[sel]} (inner RFS {inner_rfs[sel] if not full else float('nan'):.3f} vs {inner_rfs[0] if not full else float('nan'):.3f} at 0), "
                     f"{a.steps / (time.time() - t0):.2f} it/s incl. evals, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
            run.summary[f"fold{fk}"] = dict(sel_step=int(steps[sel]), train_s=time.time() - t0)
            if full:
                d = T.proot("runs", tag)
                torch.save({"model": model.state(), "cfg": ck["cfg"], "pref": vars(a) | dict(base=BASE % a.seed, n_fit=int(len(fit)))}, d / "ckpt-final.pt")
                torch.save(model.adapter.state_dict(), d / "adapter.pt")
                run.summary["ckpt"] = str(d / "ckpt-final.pt")


# ---------------------------------------------------------------- pick (pilot)
def cmd_pick(a):
    import pandas as pd
    from jevdrive import waymo as W
    tb = dict(np.load(cdir() / "tab.npz"))
    v0 = tb["v0"].astype(np.float64)
    rows = []
    for lr, lrn in LRS:
        z = np.load(pdir(f"pilot-lr{lr:g}") / "fold0.npz")
        r = np.stack([np.asarray(W.rater_feedback_score(p.astype(np.float64), tb["traj"], tb["scores"], v0), float) for p in z["plans"]])
        s, role = int(z["sel"]), z["role"]
        rows.append({"lr": lr, "lr_new": lrn, "sel_step": int(z["steps"][s]), "inner RFS @0": r[0, role == 1].mean(), "inner RFS @sel": r[s, role == 1].mean(),
                     "inner gain": r[s, role == 1].mean() - r[0, role == 1].mean(), "fit gain @sel": r[s, role == 0].mean() - r[0, role == 0].mean(),
                     "fit gain @end": r[-1, role == 0].mean() - r[0, role == 0].mean(), "inner gain @end": r[-1, role == 1].mean() - r[0, role == 1].mean(),
                     "oof gain @sel (not used)": r[s, role == 2].mean() - r[0, role == 2].mean(), "dev ADE @0": float(z["dev_ade"][0]),
                     "dev ADE @sel": float(z["dev_ade"][s]), "it/s": float(len(z["loss"]) / z["train_s"])})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pilot.csv", index=False)
    b = df.iloc[int(df["inner RFS @sel"].to_numpy().argmax())]                               # ties -> the smaller learning rate
    (OUT / "pilot.json").write_text(json.dumps({"lr": float(b.lr), "lr_new": float(b.lr_new), "rule": "max inner-fold frame-mean RFS at the early-stopping step"}))
    pd.set_option("display.width", 250, "display.max_columns", 99)
    print(df.to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"LR {b.lr:g} {b.lr_new:g}")


# ---------------------------------------------------------------- report
def load_arm(tag, tb, rfs):
    """A fold-run set -> per-frame out-of-fold plan / RFS, in-sample RFS, step-0 (WLG) plan, curves, selected steps."""
    n = len(tb["names"])
    oof, base = np.zeros((n, 20, 2)), None
    ins, cnt, sels, curves, dev = np.zeros(n), np.zeros(n), [], [], []
    for k in range(K):
        z = np.load(pdir(tag) / f"fold{k}.npz")
        assert (z["names"] == tb["names"]).all()
        P, role, s = z["plans"].astype(np.float64), z["role"], int(z["sel"])
        base = P[0] if base is None else base
        assert np.abs(P[0] - base).max() < 0.05, f"{tag}: step-0 plans differ between folds"
        oof[role == 2] = P[s][role == 2]
        r = np.stack([rfs(p) for p in P])
        ins[role == 0] += r[s][role == 0]
        cnt[role == 0] += 1
        sels.append(int(z["steps"][s]))
        curves.append(np.stack([r[:, role == q].mean(1) for q in (0, 1, 2)]))                 # true labels, frame mean: fit / inner / oof by step
        dev.append((float(z["dev_ade"][0]), float(z["dev_ade"][s]), float(z["dev_ade"][-1])))
    return dict(oof=oof, base=base, r_oof=rfs(oof), r_base=rfs(base), r_ins=ins / cnt, sels=sels, curves=np.stack(curves), steps=z["steps"], dev=np.array(dev))


def cmd_report(a):
    import pandas as pd
    import s2_gohold as S2
    import wod_launch_report as R
    from jevdrive import stats
    from jevdrive.run import Run
    with Run("op_parity", "wod-pref-report", seed=0, config=vars(a) | dict(B=R.B)) as run:
        OUT.mkdir(parents=True, exist_ok=True)
        C = R.Ctx()
        n = C.n
        tb = dict(np.load(cdir() / "tab.npz"))
        assert (tb["names"] == C.names[:n]).all()
        fold, fsp = folds_of(tb["log"])
        for s in fsp:
            run.use_split(s)
        cl = tb["cluster"].astype(str)
        st = {"all": C.st["all"], "standstill (v0<0.5)": C.st["stopped"], "launch (v0<2, log>5m)": C.st["launch (v<2, log>5m)"], "moving (v0>=0.5)": C.st["moving (v>=0.5)"],
              "turn (intent L/R)": C.intent >= 2, "night": C.st["night"], "day": C.st["day"]} | {f"cluster {c}": cl == c for c in sorted(set(cl))}
        objs = [o for o in OBJS if all((pdir(f"{o}-s{s}") / f"fold{K - 1}.npz").exists() for s in a.seeds)]
        A_ = {(o, s): load_arm(f"{o}-s{s}", tb, C.rfs) for o in objs for s in a.seeds}
        PM = {o: load_arm(f"{o}-s0-perm", tb, C.rfs) for o in objs if (pdir(f"{o}-s0-perm") / f"fold{K - 1}.npz").exists()}
        wlg_p = {s: A_[(objs[0], s)]["base"] for s in a.seeds}
        for (o, s), v in A_.items():
            assert np.abs(v["base"] - wlg_p[s]).max() < 0.05, f"{o}-s{s}: step-0 plans differ from the other objectives"
        wlg = np.mean([C.rfs(wlg_p[s]) for s in a.seeds], 0)
        log, top = C.fut[:n], C.top
        G = json.loads((cdir() / "gates.json").read_text())
        ade = lambda p, k: np.linalg.norm(p - log, axis=-1)[:, :k].mean(1)  # noqa: E731
        f3 = R.f3
        main, strata, per_seed, adr, frames = [], [], [], [], {"name": tb["names"], "cluster": cl, "fold": fold, "v0": C.v0, "rfs_top": C.rfs(top), "rfs_log": C.rfs(log), "rfs_WLG": wlg}
        # the decision-168 ceiling on the same footing: best of F20 U the plan, on the WLG token-path plans (privileged)
        cb = json.loads((_R / "experiments/op_parity/results/s2_gohold/codebook.json").read_text())
        ceil = []
        for s in a.seeds:
            c = S2.factored(wlg_p[s], C.v0, 3, cb)
            ceil.append(np.stack([C.rfs(np.ascontiguousarray(c[:, m])) for m in range(c.shape[1])]).max(0))
        ceil = np.maximum(np.mean(ceil, 0), wlg) - wlg
        cc = C.ci(ceil)
        verdict = {"WLG token path": C.cm(wlg), "WLG harness": G["P1"]["rfs_harness"], "gates": G, "ceiling F20 on WLG (token path)": list(cc), "ceiling decision 168 (on WP2)": 1.068,
                   "expected": 0.08, "objectives": {}}
        for o in objs:
            r_oof = np.mean([A_[(o, s)]["r_oof"] for s in a.seeds], 0)
            r_ins = np.mean([A_[(o, s)]["r_ins"] for s in a.seeds], 0)
            d = r_oof - wlg
            c, ci_ = C.ci(d), C.ci(r_ins - wlg)
            frames |= {f"rfs_{o}_oof": r_oof, f"rfs_{o}_ins": r_ins}
            row = {"objective": o, "RFS oof": C.cm(r_oof), "d oof": c[0], "lo": c[1], "hi": c[2], "d in-sample": ci_[0], "ins lo": ci_[1], "ins hi": ci_[2],
                   "sel steps (10 fold-runs)": " ".join(str(x) for s in a.seeds for x in A_[(o, s)]["sels"]), "share of 1.068": c[0] / 1.068, "share of F20 ceiling on WLG": c[0] / cc[0]}
            if o in PM:
                dp = PM[o]["r_oof"] - C.rfs(wlg_p[0])
                cp, cd = C.ci(dp), C.ci(A_[(o, 0)]["r_oof"] - PM[o]["r_oof"])
                row |= {"perm d oof (s0)": cp[0], "perm lo": cp[1], "perm hi": cp[2], "s0 oof - perm": cd[0], "s0-perm lo": cd[1], "s0-perm hi": cd[2],
                        "perm sel steps": " ".join(str(x) for x in PM[o]["sels"])}
                frames[f"rfs_{o}_perm_oof"] = PM[o]["r_oof"]
            main.append(row)
            neg = []
            for nm, m in st.items():
                cs = C.ci(d, m)
                strata.append({"objective": o, "stratum": nm, "n": int(m.sum()), "RFS WLG": C.cm(wlg, m), "d oof": cs[0], "lo": cs[1], "hi": cs[2],
                               "d in-sample": C.cm(r_ins - wlg, m)} | ({"perm d oof (s0)": C.cm(PM[o]["r_oof"] - C.rfs(wlg_p[0]), m)} if o in PM else {}))
                if cs[2] < 0:
                    neg.append(nm)
            for s in a.seeds:
                v = A_[(o, s)]
                cs = C.ci(v["r_oof"] - v["r_base"])
                per_seed.append({"objective": o, "seed": s, "d oof": cs[0], "lo": cs[1], "hi": cs[2], "d in-sample": C.cm(v["r_ins"] - v["r_base"]),
                                 "sel steps": " ".join(map(str, v["sels"])), "dev ADE @0": v["dev"][:, 0].mean(), "dev ADE @sel": v["dev"][:, 1].mean(), "dev ADE @end": v["dev"][:, 2].mean()})
            row = {"objective": o}
            for nm, k in (("ADE@3s", 12), ("ADE@5s", 20)):
                dd = np.mean([ade(A_[(o, s)]["oof"], k) - ade(wlg_p[s], k) for s in a.seeds], 0)
                cs = C.ci_mean(dd)
                row |= {f"{nm} WLG": float(np.mean([ade(wlg_p[s], k) for s in a.seeds])), f"d {nm}": cs[0], f"{nm} lo": cs[1], f"{nm} hi": cs[2]}
            mv = np.mean([np.linalg.norm(A_[(o, s)]["oof"] - wlg_p[s], axis=-1).mean(1) for s in a.seeds], 0)
            d5 = np.mean([np.linalg.norm(A_[(o, s)]["oof"][:, -1], axis=-1) - np.linalg.norm(wlg_p[s][:, -1], axis=-1) for s in a.seeds], 0)
            row |= {"plan moved vs WLG, mean m": float(mv.mean()), "d 5 s displacement, mean m": float(d5.mean()), "d 5 s displacement standstill": float(d5[C.st["stopped"]].mean()),
                    "dev ADE @0 (r2-dev 512 rows, 8 poses)": float(np.mean([A_[(o, s)]["dev"][:, 0].mean() for s in a.seeds])),
                    "dev ADE @sel": float(np.mean([A_[(o, s)]["dev"][:, 1].mean() for s in a.seeds])), "dev ADE @end": float(np.mean([A_[(o, s)]["dev"][:, 2].mean() for s in a.seeds]))}
            adr.append(row)
            frames |= {f"moved_{o}": mv, f"d5_{o}": d5}
            delivers = bool(c[1] > 0 and not neg)
            verdict["objectives"][o] = {"d oof": list(c), "d in-sample": list(ci_), "delivers": delivers, "strata with CI below 0": neg,
                                        "frame-specific (s0 oof - perm lo > 0)": bool(main[-1].get("s0-perm lo", -1) > 0),
                                        "final steps (median of the selected steps)": int(np.median([x for s in a.seeds for x in A_[(o, s)]["sels"]]))}
        verdict["null on every objective"] = bool(objs and all(v["d oof"][1] <= 0 for v in verdict["objectives"].values()))
        cur = []
        for o in objs:
            for kind, src in (("fit", [A_[(o, s)] for s in a.seeds]), ("perm", [PM[o]] if o in PM else [])):
                if not src:
                    continue
                cv = np.mean([v["curves"] for v in src], (0, 1))                               # (3 roles, E), mean over seeds and folds, true labels
                for j, stp in enumerate(src[0]["steps"]):
                    cur.append({"objective": o, "labels": "true" if kind == "fit" else "permuted", "step": int(stp), "fit": cv[0, j] - cv[0, 0], "inner": cv[1, j] - cv[1, 0],
                                "oof": cv[2, j] - cv[2, 0]})
        for nm, rows in (("main", main), ("strata", strata), ("per_seed", per_seed), ("ade", adr)):
            stats.write_table(rows, OUT / nm)
        pd.DataFrame(cur).to_csv(OUT / "curves.csv", index=False)
        pd.DataFrame(frames).to_csv(OUT / "frames.csv", index=False)
        np.savez(OUT / "plans.npz", names=tb["names"], **{f"wlg_s{s}": wlg_p[s].astype(np.float32) for s in a.seeds},
                 **{f"{o}_s{s}": A_[(o, s)]["oof"].astype(np.float32) for o in objs for s in a.seeds})
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
        pd.set_option("display.width", 250, "display.max_columns", 99)
        run.info("WLG token path %.3f (harness %.3f); F20 ceiling on WLG %s", C.cm(wlg), G["P1"]["rfs_harness"], f3(cc))
        run.info("main:\n%s", pd.DataFrame(main).to_string(float_format=lambda v: f"{v:+.3f}"))
        run.info("per seed:\n%s", pd.DataFrame(per_seed).to_string(float_format=lambda v: f"{v:+.3f}"))
        run.info("ade:\n%s", pd.DataFrame(adr).T.to_string(float_format=lambda v: f"{v:+.3f}"))
        sd = pd.DataFrame(strata)
        run.info("strata:\n%s", sd.to_string(float_format=lambda v: f"{v:+.3f}"))
        run.info("verdict: %s", json.dumps({k: v for k, v in verdict.items() if k != "gates"}))


# ---------------------------------------------------------------- figs
def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from wod_launch_figs import rgb
    FIG.mkdir(parents=True, exist_ok=True)
    cur = pd.read_csv(OUT / "curves.csv")
    objs = [o for o in OBJS if o in set(cur.objective)]
    fig, axs = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 3.6), sharey=True, squeeze=False)
    for ax, o in zip(axs[0], objs):
        c = cur[(cur.objective == o) & (cur["labels"] == "true")]
        for k, col in (("fit", "#c0392b"), ("inner", "#e69f00"), ("oof", "#1f77b4")):
            ax.plot(c.step, c[k], color=col, lw=1.8, label={"fit": "fitted folds (in-sample)", "inner": "inner fold (early stopping)", "oof": "out-of-fold"}[k])
        p = cur[(cur.objective == o) & (cur["labels"] == "permuted")]
        if len(p):
            ax.plot(p.step, p.oof, color="#1f77b4", lw=1.2, ls=(0, (3, 2)), label="out-of-fold, permuted labels (seed 0)")
            ax.plot(p.step, p.fit, color="#c0392b", lw=1.0, ls=(0, (3, 2)), label="fitted folds, permuted labels (true-label RFS)")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(0.08, color="gray", lw=0.6, ls=":")
        ax.set_title(f"objective: {o}", fontsize=10)
        ax.set_xlabel("fine-tuning step")
        ax.grid(alpha=0.25)
    axs[0][0].set_ylabel("frame-mean RFS - WLG (step 0)")
    axs[0][-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("WOD val rater frames: RFS vs fine-tuning step, mean over 5 folds x 2 seeds (dotted grey: the expected +0.08)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "curves.png", dpi=140)
    plt.close(fig)
    # ---- frames: the headline objective's 4 largest out-of-fold gains and 4 largest losses (seed-mean RFS), seed-0 plans drawn
    fr = pd.read_csv(OUT / "frames.csv")
    main = pd.read_csv(OUT / "main.csv")
    o = a.obj or main.sort_values("d oof", ascending=False).objective.iloc[0]
    z = np.load(OUT / "plans.npz")
    tb = dict(np.load(cdir() / "tab.npz"))
    f0 = np.load(cdir() / "frames_t0.npy", mmap_mode="r")
    d = (fr[f"rfs_{o}_oof"] - fr.rfs_WLG).to_numpy()
    order = np.argsort(d)
    sel = list(order[::-1][:4]) + list(order[:4])
    fig, axs = plt.subplots(len(sel), 3, figsize=(13.5, 3.3 * len(sel)), gridspec_kw={"width_ratios": [2, 2, 1.5]})
    rows = []
    for row, i in zip(axs, sel):
        row[0].imshow(rgb(f0[i, 0]))
        row[0].set_title(f"{tb['names'][i][:12]}.. road frame (model input, t0)", fontsize=8)
        row[1].imshow(rgb(f0[i, 1]))
        row[1].set_title("wide frame (model input, t0)", fontsize=8)
        for ax in row[:2]:
            ax.axis("off")
        ax = row[2]
        for k in np.argsort(tb["scores"][i]):
            ax.plot(-tb["traj"][i, k, :, 1], tb["traj"][i, k, :, 0], "o-", ms=2.5, lw=1.0, color=plt.cm.viridis(tb["scores"][i, k] / 10), label=f"rater {tb['scores'][i, k]:.0f}")
        ax.plot(-tb["future"][i, :, 1], tb["future"][i, :, 0], color="gray", lw=1.0, ls=":", label=f"log ({fr.rfs_log[i]:.1f})")
        ax.plot(-z["wlg_s0"][i, :, 1], z["wlg_s0"][i, :, 0], color="k", lw=1.6, label=f"WLG ({fr.rfs_WLG[i]:.1f})")
        ax.plot(-z[f"{o}_s0"][i, :, 1], z[f"{o}_s0"][i, :, 0], color="#d62728", lw=1.6, ls=(0, (4, 1.5)), label=f"fine-tuned {o}, out-of-fold ({fr[f'rfs_{o}_oof'][i]:.1f})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        ax.set_title(f"{fr.cluster[i]}, v0 {fr.v0[i]:.1f} m/s, d RFS {d[i]:+.2f}", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.set_xlabel("left <- y (m) -> right", fontsize=7)
        rows.append({"name": tb["names"][i], "cluster": fr.cluster[i], "v0": fr.v0[i], "scores": " ".join(f"{x:.0f}" for x in tb["scores"][i]), "RFS WLG": fr.rfs_WLG[i],
                     f"RFS {o} oof": fr[f"rfs_{o}_oof"][i], "d": d[i], "RFS log": fr.rfs_log[i]})
    fig.suptitle(f"Objective {o}: the 4 largest out-of-fold RFS gains (top) and the 4 largest losses (bottom) vs WLG; BEV in the rear-axle frame, x forward up", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "frames.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "figure_frames.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    p = sp_.add_parser("prep")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="first n rater frames (smoke; no gates enforced)")
    p = sp_.add_parser("train")
    p.add_argument("--obj", required=True, choices=OBJS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", nargs="+", default=[str(k) for k in range(K)], help="outer folds, or `all` (every rater frame fitted, checkpoint saved)")
    p.add_argument("--perm", action="store_true", help="label-permutation control (inside v0 bins, training frames only)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-new", type=float, default=1e-4)
    p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    sp_.add_parser("pick")
    sp_.add_parser("final-plan", help="print `<objective> <steps>` for every objective whose out-of-fold CI excludes 0 (verdict.json)")
    p = sp_.add_parser("report")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p = sp_.add_parser("figs")
    p.add_argument("--obj", default="", help="objective of the frame figure (default: the largest out-of-fold gain)")
    a = ap.parse_args()
    if a.cmd == "final-plan":
        for o_, v_ in json.loads((OUT / "verdict.json").read_text())["objectives"].items():
            if v_["d oof"][1] > 0 and v_["final steps (median of the selected steps)"] > 0:
                print(o_, v_["final steps (median of the selected steps)"])
        raise SystemExit(0)
    {"prep": cmd_prep, "train": cmd_train, "pick": cmd_pick, "report": cmd_report, "figs": cmd_figs}[a.cmd](a)
