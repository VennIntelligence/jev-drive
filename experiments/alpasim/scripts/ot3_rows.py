"""Yaw-rate rows (lane OT3, plans/2026-10-09-ot3-lambda10-prereg.md, decision 213): training rows that break the undamped yaw-rate
continuation of decision 205 without hiding the ego's rotation from the model.

The model reads its own turning from the synthesised slots (the image pairs between keyframes are one keyframe warped along the ego track) and
continues it (plan yaw at 0.5 s = 0.93 x the yaw turned in the last 0.5 s); on logs that is right, behind a tracker it is an undamped loop.
The off-track rows of decision 198 (ot_rows.py) ramp the heading error linearly over 1.6 s, so the injected yaw rate and the heading offset
are one number (rate = offset / 1.6 s): the offset alone explains the target. Here the two are decoupled:

  psi(t)   heading error against the logged pose, piecewise linear with knots at the keyframes: 0 at t <= -1.5 s, increments a1, a2, a3 over
           [-1.5, -1], [-1, -0.5], [-0.5, 0] s; a_i ~ N(0, 1 deg) clipped at 2.5 deg, independent. With probability 0.4 the row is `recent`:
           a1 = a2 = 0 (the state one decision after the model's own plan turned the car by a3: a yaw rate with almost no offset yet)
  y(t)     lateral error: y(0) = dy0 ~ U(+-0.3 m) (recent) / U(+-1.0 m) (random walk), dy/dt = v * psi (the car moves along its own heading)
  frames   every one of the 10 history frames (4 keys, 6 lattice) is ONE warp of the nearest real key to `logged pose(t) o (0, y(t), psi(t))`
           (ot_rows._job, the plane engine of decision 141): the presented self-rotation = the logged one + the injected one
  ego      history poses re-expressed in the perturbed t0 frame, velocity / acceleration / command as logged (AP2 standard: its own definitions)
  target   the logged future re-expressed in the perturbed t0 frame: the injected rotation is NOT continued, the plan returns to the lane
  hinge    ot_rows.off_hinge (the plan mapped back into the logged frame)
Additive on real rows (speed > 3 m/s), nothing is zeroed, no rollout, no longitudinal change. The rows are served to the trainers of
ot_rows.py (NAVSIM input standard) and ap2_ot.py (AlpaSim input standard, m = 4 decisions) as a share of every batch.

  prep   --data navtrain_full.s2of12 [--limit N] [--zero]   -> cache/yr1_<data>/tab.npz (+ `inj` (n, 3) rad, `rec`), cache/yr1_<data>@warp/
  train  --tag YR10m10-F-s0 --std navsim|alpasim --ot-mass 0.1 --hinge-lam 10 --hinge-margin 0.3 ...
  probe  --name N --tags P2H10-F-s0 ...   held-out rows (<split>-dev): the plan's yaw at 0.5 s on the perturbed row minus on the same token at
         the logged pose, regressed on (a3, total heading offset, lateral offset) -> alpha (continuation of the injected yaw rate at a fixed
         pose offset; 0.93 = the closed-loop amplifier, 0 = the target), beta (response to the heading offset; -1 = the target), the net
         response on recent rows, the spectral radius of e[k+1] = (1 + alpha + beta) e[k] - alpha e[k-1] (heading error behind a tracker that
         executes the plan's yaw), and the legitimate continuation: slope of the plan's yaw at 0.5 s on the logged yaw of the last 0.5 s on
         unperturbed held-out tokens (all moving / turn tokens with |heading change over 4 s| >= 20 deg; the logs' own slope beside it)

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/ot3_rows.py prep --data navtrain_full.s2of12 --limit 64
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

import ot_rows as OR  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

YR = "yr1"                                       # cache prefix = version of the recipe below
SIG, AMAX, P_REC, DY_REC, DY_RW, YMAX = np.radians(1.0), np.radians(2.5), 0.4, 0.3, 1.0, 2.0
KNOT = np.array([-1.5, -1.0, -0.5])
OUT = data_dir() / "runs/alpasim/ot3/results"
AROOT = data_dir() / "runs/alpasim/ap2"


def paths(T, a, dy0, v):
    """Heading and lateral error at times T (m,) for increments a (n, 3) rad, final lateral error dy0 (n,), speed v (n,) -> ys, ps (n, m)."""
    g = np.linspace(-1.6, 0.0, 161)
    ramp = lambda t: np.clip((np.asarray(t)[None, :, None] - KNOT) / 0.5, 0.0, 1.0)  # noqa: E731  (1, m, 3)
    psi_g = (ramp(g) * a[:, None, :]).sum(-1)                                       # (n, 161)
    F = np.concatenate([np.zeros((len(a), 1)), np.cumsum(0.5 * (psi_g[:, 1:] + psi_g[:, :-1]) * np.diff(g), 1)], 1)   # int_{-1.6}^{g} psi
    Ft = np.stack([np.interp(np.clip(T, -1.6, 0.0), g, f) for f in F])
    return dy0[:, None] - v[:, None] * (F[:, -1:] - Ft), (ramp(np.clip(T, -1.6, 0.0)) * a[:, None, :]).sum(-1)


def profile(T, speed, rng, zero):
    """ot_rows.PROFILE: per row (dy, dpsi) at t0, the (y, psi) history at the 10 frame times, and the tab extras (inj, rec)."""
    n = len(speed)

    def draw(k):
        rec = rng.random(k) < P_REC
        a = np.clip(rng.normal(0.0, SIG, (k, 3)), -AMAX, AMAX)
        a[rec, :2] = 0.0
        return a, rng.uniform(-1.0, 1.0, k) * np.where(rec, DY_REC, DY_RW), rec
    a, dy, rec = draw(n)
    for _ in range(200):
        bad = np.abs(paths(T, a, dy, speed)[0]).max(1) > YMAX
        if not bad.any():
            break
        a[bad], dy[bad], rec[bad] = draw(int(bad.sum()))
    assert not bad.any()
    if zero:
        a[:], dy[:] = 0.0, 0.0
    ys, ps = paths(T, a, dy, speed)
    return dy, a.sum(1), ys, ps, dict(inj=a.astype(np.float32), rec=rec)


def use():
    OR.OT, OR.YMAX, OR.PROFILE = YR, YMAX, profile
    OR.KEY_EXTRA = dict(yr=dict(sig=float(SIG), amax=float(AMAX), p_rec=P_REC, dy_rec=DY_REC, dy_rw=DY_RW))


def cmd_prep(a):
    use()
    OR.cmd_prep(a)


def cmd_train(a):
    use()
    if a.std == "navsim":
        a.ot = YR
        OR.cmd_train(a)
    else:
        import ap2_inputs as AI
        import ap2_ot as AO
        a.amp, a.cold, a.mix = "yr1", "backwarp", list(AI.MIX)
        AO.cmd_train(a)
        use()


def boot_ls(X, y, logs, B=10000, seed=0):
    """Least squares y ~ X (n, p) without intercept -> coefficients (p,) and B cluster-bootstrap replicates (B, p), clusters = logs."""
    u, inv = np.unique(logs, return_inverse=True)
    p = X.shape[1]
    G, g = np.zeros((len(u), p, p)), np.zeros((len(u), p))
    np.add.at(G, inv, X[:, :, None] * X[:, None, :])
    np.add.at(g, inv, X * y[:, None])
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    return np.linalg.solve(G.sum(0), g.sum(0)), np.linalg.solve(G[idx].sum(1), g[idx].sum(1)[..., None])[..., 0]


def radius(al, be):
    """Spectral radius of e[k+1] = (1 + al + be) e[k] - al e[k-1]."""
    return np.abs(np.stack([np.roots([1.0, -(1.0 + x + y), x]) for x, y in zip(np.atleast_1d(al), np.atleast_1d(be))])).max(1)


def cmd_probe(a):
    import torch
    import ap2_core as AC
    import ot_ladder as OL
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    use()
    dev = torch.device("cuda")
    with Run("alpasim", f"ot3-probe-{a.name}", config=vars(a)) as run:
        ot = tuple(f"{YR}_{d}" for d in a.data)
        S = T.Store(ot, dev, need_side=False, frames="warp", host=True)
        Bs = T.Store(tuple(a.data), dev, need_side=False, frames="warp", host=True)
        nb = np.cumsum([0] + [len(np.load(OR.CR / d / "tab.npz")["names"]) for d in a.data])
        zt = [np.load(OR.CR / d / "tab.npz") for d in ot]
        src = np.concatenate([z["src_row"] + o for z, o in zip(zt, nb)])
        inj, rec = np.concatenate([z["inj"] for z in zt]).astype(np.float64), np.concatenate([z["rec"] for z in zt])
        assert (Bs.tab["names"][src] == S.tab["names"]).all()
        off = OR.offsets(ot).astype(np.float64)
        sp = splits.load(f"{a.split}-dev")
        run.use_split(sp)
        rz = [np.load(AROOT / "route" / f"{d}.npz") for d in a.data]
        assert np.concatenate([z["names"] for z in rz]).tolist() == Bs.tab["names"].tolist()
        wp4, ok4 = np.concatenate([z["wp"][:, 3] for z in rz]), np.concatenate([z["ok"][:, 3] for z in rz])
        rows = np.flatnonzero(sp.mask(S.tab["names"]) & ok4[src])
        rows = rows[: a.limit] if a.limit else rows
        sb, log = src[rows], S.tb["log"][rows]
        # unperturbed held-out tokens for the legitimate continuation: every dev row with a logged future that moves
        pose_b, fut_b_all, v_b = Bs.tb["pose"].astype(np.float64), Bs.tb["fut"].astype(np.float64), np.asarray(Bs.tb["speed"], np.float64)
        leg = np.flatnonzero(sp.mask(Bs.tab["names"]) & ok4 & (v_b > 2.0) & np.isfinite(fut_b_all).all((1, 2)))
        leg = leg[: 4 * a.limit] if a.limit else leg
        turned = -np.degrees(pose_b[leg, 2, 2])                               # logged yaw turned over the last 0.5 s
        turn = np.abs(np.degrees(fut_b_all[leg, 7, 2])) >= 20.0
        llog = Bs.tb["log"][leg]
        sub = lambda tab, r: {k: tab[k][r] for k in ("pose", "vel", "acc", "fut")}  # noqa: E731
        ego_ap = {"ot": AC.ego_table(sub(S.tb, rows), np.repeat(OL.wp_to_frame(wp4[sb], off[rows, 0], off[rows, 1])[:, None], 4, 1))[:, 3],
                  "base": AC.ego_table(sub(Bs.tb, sb), np.repeat(wp4[sb][:, None], 4, 1))[:, 3],
                  "leg": AC.ego_table(sub(Bs.tb, leg), np.repeat(wp4[leg][:, None], 4, 1))[:, 3]}
        W8 = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        pi = torch.as_tensor(S.pi, device=dev)

        def plans(model, St, rr, ego):
            o = []
            with torch.no_grad():
                for i in range(0, len(rr), 256):
                    r = torch.as_tensor(rr[i:i + 256], device=dev)
                    e = St.ego[r] if ego is None else torch.from_numpy(ego[i:i + 256]).to(dev)
                    p = model(St.front[r], e, St.tc[r]).float()[:, pi].view(-1, 33, 15)
                    o.append(torch.stack(T.rear(p, St.cam_x[r], W8), -1).cpu().numpy())
            return np.concatenate(o).astype(np.float64)
        fut_o, fut_b = S.tb["fut"][rows].astype(np.float64), Bs.tb["fut"][sb].astype(np.float64)
        a3, dps, dy = np.degrees(inj[rows, 2]), np.degrees(off[rows, 1]), off[rows, 0]
        X = np.c_[a3, dps, dy]
        r_ = rec[rows]
        run.info(f"{len(rows)} held-out yaw-rate rows ({int(r_.sum())} recent) of {len(a.data)} shards, {len(np.unique(log))} logs; sd a3 {a3.std():.2f} deg, "
                 f"sd heading offset {dps.std():.2f} deg, corr(a3, offset) {np.corrcoef(a3, dps)[0, 1]:.2f}, |dy| mean {np.abs(dy).mean():.2f} m; "
                 f"legit rows {len(leg)} ({int(turn.sum())} turn)")
        lg_all, lg_turn = np.polyfit(turned, np.degrees(fut_b_all[leg, 0, 2]), 1)[0], np.polyfit(turned[turn], np.degrees(fut_b_all[leg[turn], 0, 2]), 1)[0]
        ci = lambda c, bs: [float(c), *(float(x) for x in np.percentile(bs, [2.5, 97.5]))]  # noqa: E731
        res = {}
        for spec in a.tags:
            tag, _, std = spec.partition("@")
            model, route, _ = AC.load_model(tag, dev)
            ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
            ap = (ck.get("ap2") or {}).get("std", "navsim") == "alpasim" and std != "nav"
            assert not route
            po, pb, pl = plans(model, S, rows, ego_ap["ot"] if ap else None), plans(model, Bs, sb, ego_ap["base"] if ap else None), \
                plans(model, Bs, leg, ego_ap["leg"] if ap else None)
            r = dict(n=len(rows), inputs="alpasim" if ap else "navsim",
                     ade_yr=float(np.linalg.norm(po[..., :2] - fut_o[..., :2], axis=-1).mean()), ade_base=float(np.linalg.norm(pb[..., :2] - fut_b[..., :2], axis=-1).mean()),
                     lat4_yr=float(np.abs(po[:, 7, 1] - fut_o[:, 7, 1]).mean()))
            for t, j in (("05", 0), ("10", 1)):
                c, bs = boot_ls(X, np.degrees(po[:, j, 2] - pb[:, j, 2]), log)
                r[f"alpha_{t}"], r[f"beta_{t}"], r[f"gamma_{t}"] = ci(c[0], bs[:, 0]), ci(c[1], bs[:, 1]), ci(c[2], bs[:, 2])
                if t == "05":
                    r["radius"] = ci(radius(c[0], c[1])[0], radius(bs[:, 0], bs[:, 1]))
                c, bs = boot_ls(X[r_][:, [0, 2]], np.degrees(po[r_, j, 2] - pb[r_, j, 2]), log[r_])
                r[f"net_recent_{t}"] = ci(c[0], bs[:, 0])
            c, bs = boot_ls(np.c_[-dy, -np.sin(off[rows, 1]) * fut_b[:, 7, 0] + (np.cos(off[rows, 1]) - 1) * fut_b[:, 7, 1]], po[:, 7, 1] - pb[:, 7, 1], log)
            r["ret_dy_4s"], r["ret_yaw_4s"] = ci(c[0], bs[:, 0]), ci(c[1], bs[:, 1])     # ot_rows.cmd_probe's 4 s lateral return (1 = back on the logged path)
            for nm, msk in (("all", np.ones(len(leg), bool)), ("turn", turn)):
                yy, xx, gg = np.degrees(pl[msk, 0, 2]), turned[msk], llog[msk]
                u, inv = np.unique(gg, return_inverse=True)
                st = np.zeros((len(u), 5))
                np.add.at(st, inv, np.c_[np.ones_like(xx), xx, yy, xx * xx, xx * yy])
                sl = lambda q: (q[..., 0] * q[..., 4] - q[..., 1] * q[..., 2]) / (q[..., 0] * q[..., 3] - q[..., 1] ** 2)  # noqa: E731  OLS slope with intercept
                idx = np.random.default_rng(0).integers(0, len(u), (10000, len(u)))
                r[f"legit_{nm}"] = ci(sl(st.sum(0)), sl(st[idx].sum(1)))
            r["legit_ade"] = float(np.linalg.norm(pl[..., :2] - fut_b_all[leg][..., :2], axis=-1).mean())
            r["legit_turn_lat4"] = float(np.abs(pl[turn, 7, 1] - fut_b_all[leg[turn], 7, 1]).mean())
            res[spec] = r
            run.info(f"{spec}: alpha {r['alpha_05'][0]:.3f} beta {r['beta_05'][0]:.3f} net recent {r['net_recent_05'][0]:.3f} radius {r['radius'][0]:.3f}; "
                     f"legit all {r['legit_all'][0]:.3f} turn {r['legit_turn'][0]:.3f}; ADE yr {r['ade_yr']:.3f} base {r['ade_base']:.3f}")
            del model
        f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
        L = [f"{len(rows)} held-out yaw-rate rows ({a.split}-dev, cache `{YR}`, {len(a.data)} shards, {len(np.unique(log))} logs; {int(r_.sum())} recent rows); injected yaw "
             f"over the last 0.5 s sd {a3.std():.2f} deg, heading offset sd {dps.std():.2f} deg, correlation {np.corrcoef(a3, dps)[0, 1]:.2f}. Legitimate continuation on "
             f"{len(leg)} unperturbed held-out tokens ({int(turn.sum())} turn tokens); the logs' own slope there: {lg_all:.2f} (all), {lg_turn:.2f} (turn). 95% CIs: cluster "
             "bootstrap over logs, B 10000.", "",
             "| model | inputs | alpha: plan yaw at 0.5 s per deg of injected yaw in the last 0.5 s (fixed pose offset) | beta: per deg of heading offset | net response on recent rows | "
             "loop spectral radius | alpha at 1.0 s | legit slope, all moving | legit slope, turn tokens | 4 s lateral return (lateral / yaw part) | ADE yaw-rate rows / logged pose (m) | "
             "turn tokens: 4 s lateral error (m) |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|"]
        L += [f"| {t} | {r['inputs']} | {f(r['alpha_05'])} | {f(r['beta_05'])} | {f(r['net_recent_05'])} | {f(r['radius'])} | {f(r['alpha_10'])} | {f(r['legit_all'])} | "
              f"{f(r['legit_turn'])} | {r['ret_dy_4s'][0]:.2f} / {r['ret_yaw_4s'][0]:.2f} | {r['ade_yr']:.3f} / {r['ade_base']:.3f} | {r['legit_turn_lat4']:.3f} |" for t, r in res.items()]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"yr_probe_{a.name}.json").write_text(json.dumps(dict(models=res, n=len(rows), n_recent=int(r_.sum()), n_legit=len(leg), n_turn=int(turn.sum()),
                                                                     log_slope_all=float(lg_all), log_slope_turn=float(lg_turn)), indent=1))
        (OUT / f"yr_probe_{a.name}.md").write_text("\n".join(L) + "\n")
        run.info("\n" + "\n".join(L))
        run.summary |= {t: {k: r[k][0] for k in ("alpha_05", "beta_05", "net_recent_05", "radius", "legit_all", "legit_turn")} for t, r in res.items()}
        if a.gate:                                                   # row-construction gate: models trained on logs only must continue the injected yaw
            bad = [t for t, r in res.items() if r["alpha_05"][0] < 0.5]
            assert not bad, f"alpha < 0.5 for {bad}: the rows do not present the injected rotation the way the closed loop does (replay slope 0.93)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--zero", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--std", default="navsim", choices=["navsim", "alpasim"], help="input standard of the training rows (ot_rows.py / ap2_ot.py loop)")
    p.add_argument("--ot-mass", type=float, default=0.1, help="share of every batch drawn from the yaw-rate rows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--split", default="navsim/op-parity-s234")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--hinge-lam", type=float, default=10.0)
    p.add_argument("--hinge-margin", type=float, default=0.3)
    p = sub.add_parser("probe")
    p.add_argument("--name", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--data", nargs="+", default=[f"navtrain_full.s{i}of12" for i in range(12)])
    p.add_argument("--split", default="navsim/op-parity-full")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "probe": cmd_probe}[a.cmd](a)
