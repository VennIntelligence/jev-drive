"""BODY1 arm 4.3 (prereg Amendment 4, item 3 B / C): the `bd4` off-track state family, the scorer-layer drivable raster, and their preview.

bd4 = navtrain tokens seen from a pose with a large heading error, built by experiments/op_parity/scripts/ot_rows.py cmd_prep through its
PROFILE / SELECT hooks (same plane engine, tab and cache layout as ot1 / yr1, so pp_train.Store reads the dirs unchanged):
  heading offset   |dpsi| ~ U(2, 8) deg, random sign;   lateral offset dy ~ U(-0.5, 0.5) m at t0;   no speed cut (ot1 / yr1: > 3 m/s)
  history          v0 >= V_STATIC: op_adapt_h.drift (heading error ramps from 0 at -1.6 s, lateral error integrates v * psi), redrawn while the
                   pose at -1.5 s is more than YMAX off the logged path (so fast rows carry the smaller headings);
                   v0 < V_STATIC: the offset is constant over the history (a standing car has no yaw rate to show: static rows)
  rows             a deterministic subset by token hash (keep()): half of the tokens at <= 3 m/s (the cell no other cache has), a quarter of the
                   faster ones, and every token whose logged 4 s heading change exceeds 45 deg
Hinge-only rows: the tab's `fut` (logged future in the perturbed frame) is stored by cmd_prep but never used as a target by bd4_train.py.

  prep     --data navtrain_full.s2of12 [--limit N]     -> cache/bd4_<data>/tab.npz, cache/bd4_<data>@warp/{front.npy, teacher.npz}
  count                                                 rows per shard / speed bin / turn bin of the subset against all tokens (no build)
  preview  --data navtrain_full.s2of12 --limit 600      BEV + model view of 3-10 states per user class (needs prep --limit of the same size)

Scorer-layer raster (item C; envs/navsim2, CPU; navtrain's nuPlan maps only):
  OPB_LAYERS=ROADBLOCK,ROADBLOCK_CONNECTOR,INTERSECTION,LANE,LANE_CONNECTOR OPB_OUT=$DATA_DIR/runs/body1/labels_road \\
    $DATA_DIR/envs/navsim2/bin/python experiments/op_probe/scripts/opb_labels.py build --split navtrain --shards 0 1 2 3 4 5 6 7 8 9 10 11

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/bd4_prep.py prep --data navtrain_full.s2of12 --limit 600
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import hashlib  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

FAM = "bd4"
PSI_LO, PSI_HI, DY, YMAX, V_STATIC = np.radians(2.0), np.radians(8.0), 0.5, 1.5, 1.0
ROAD_LAYERS = ("ROADBLOCK", "ROADBLOCK_CONNECTOR", "INTERSECTION", "LANE", "LANE_CONNECTOR")
B.FAMS[FAM] = "bd4_"


def road_labels() -> _pl.Path:
    return B.root() / "labels_road" / "navtrain_s01234567891011.npz"


def keep(names, speed, fut) -> np.ndarray:
    """The bd4 row subset (module docstring): bool per token."""
    h = np.array([int(hashlib.sha256(str(t).encode()).hexdigest()[:8], 16) % 4 for t in names])
    turn = np.abs(np.degrees(np.unwrap(np.nan_to_num(fut[:, :, 2].astype(np.float64)), axis=1)[:, -1])) > 45
    return (h < np.where(speed <= 3.0, 2, 1)) | turn


def select(base, rows):
    return rows[keep(base["names"][rows], base["speed"][rows], base["fut"][rows])]


def profile(T, speed, rng, zero):
    """ot_rows.PROFILE: per row (dy, dpsi) at t0, the (y, psi) history at the 10 frame times, and the tab extras."""
    from experiments.op_adapt_h.lib import op_adapt_h as H
    n = len(speed)
    static = speed < V_STATIC

    def draw(k):
        return rng.uniform(-DY, DY, k), rng.choice([-1.0, 1.0], k) * rng.uniform(PSI_LO, PSI_HI, k)
    dy, dp = draw(n)
    for _ in range(2000):
        bad = ~static & (np.abs(H.drift(-1.5, dy, dp, speed)[0]) > YMAX)
        if not bad.any():
            break
        dy[bad], dp[bad] = draw(int(bad.sum()))
    assert not bad.any(), f"{int(bad.sum())} rows without a history inside {YMAX} m"
    if zero:
        dy, dp = np.zeros(n), np.zeros(n)
    ys, ps = (np.stack(x) for x in zip(*[H.drift(T, dy[i], dp[i], float(speed[i])) for i in range(n)]))
    ys[static], ps[static] = dy[static, None], dp[static, None]
    return dy, dp, ys, ps, dict(static=static)


def use():
    import ot_rows as OR
    OR.OT, OR.YMAX, OR.VMIN, OR.PROFILE, OR.SELECT = FAM, YMAX, -1.0, profile, select
    OR.KEY_EXTRA = dict(bd4=dict(psi=[float(PSI_LO), float(PSI_HI)], dy=DY, v_static=V_STATIC, keep="h4<2|1,turn45"))
    return OR


def cmd_prep(a):
    use().cmd_prep(a)


def cmd_count(a):
    from jevdrive.run import Run
    with Run("body1", "bd4-count", config=vars(a)) as run:
        rows = []
        for k in range(B.NSH):
            t = B.tab("log", k)
            ok = ~np.isnan(t["fut"]).any((1, 2))
            kp = ok & keep(t["names"], t["speed"], t["fut"])
            turn = np.abs(np.degrees(np.unwrap(np.nan_to_num(t["fut"][:, :, 2].astype(np.float64)), axis=1)[:, -1]))
            hold = np.array([B.is_hold(x) for x in t["log"]])
            for name, m in (("all", np.ones(len(ok), bool)), ("v < 1", t["speed"] < 1), ("v 1-3", (t["speed"] >= 1) & (t["speed"] <= 3)), ("v > 3", t["speed"] > 3),
                            ("turn > 45", turn > 45), ("turn 20-45", (turn > 20) & (turn <= 45)), ("hold logs", hold)):
                rows.append((k, name, int((ok & m).sum()), int((kp & m).sum())))
        tot = {}
        for _, name, n, m in rows:
            tot[name] = tuple(np.add(tot.get(name, (0, 0)), (n, m)))
        for name, (n, m) in tot.items():
            run.info(f"{name}: {m} of {n} tokens kept ({m / max(n, 1):.3f})")
        run.summary.update({name: [int(n), int(m)] for name, (n, m) in tot.items()} | {"gb": tot["all"][1] * 8 * 32 * 512 * 2 / 2 ** 30})


def user_class(man, haz):
    """prereg 2.1 + Amendment 1 item 1: 1 obstacle ahead, 2 turn, 3 leaving the road, 0 other (b1.MAN / b1.HAZ indices)."""
    c = np.zeros(len(man), int)
    c[np.isin(man, (0, 5)) & (haz != 0)] = 3
    c[(np.isin(man, (2, 3)) & np.isin(haz, (1, 2))) | (man == 3)] = 2
    c[(haz == 0) | (man == 4)] = 1
    return c


def cmd_preview(a):
    """BEV (logged frame: both rasters, agent boxes at 0 / 2 / 4 s, the student's plan sweep from the perturbed pose, labels in the title) next
    to the model's newest road frame at the logged pose and at the perturbed pose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import pp_train as T
    import sweep as SW
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import op_interp as I
    from jevdrive.data import splits
    from jevdrive.run import Run
    OR = use()
    dev = torch.device("cuda")
    d = f"{FAM}_{a.data}-first{a.limit}"
    with Run("body1", "bd4-preview", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        S = T.Store([d], dev, need_side=False, frames="warp")
        t = S.tab
        k = int(a.data.split(".s", 1)[1].split("of")[0])
        grow = B.shard_offsets()[k] + t["src_row"]
        tax = np.load(B.root() / "taxonomy" / "tax.npz")
        man, haz = tax["man"][grow], tax["haz"][grow]
        uc = user_class(man, haz)
        L = B.labels()
        road = np.load(road_labels())
        assert (road["tokens"][grow] == t["names"]).all() and (L["tokens"][grow] == t["names"]).all()
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        model = T.load_pmodel(a.model, dev)
        pi = torch.as_tensor(S.pi, device=dev)
        with torch.no_grad():
            r = torch.arange(S.n, device=dev)
            p = torch.cat([model(S.front[r[i:i + 128]], S.ego[r[i:i + 128]], S.tc[r[i:i + 128]], None, None).float()[:, pi].view(-1, 33, 15) for i in range(0, S.n, 128)])
            own = torch.stack(T.rear(p, S.cam_x, W), -1).cpu().numpy()
        off = t["off"].astype(np.float32)
        lab = SW.labels(own[:, None], off, L["box"][grow], L["valid"][grow], L["cls"][grow], L["sdf"][grow])
        labr = SW.boundary_labels(SW.dense(own[:, None], off[:, None]), road["sdf"][grow])
        run.info(f"{d}: {S.n} states; own-plan agent contact {lab['a_hit'].mean():.3f}, NAVSIM-raster margin < -0.2 m {(lab['b_margin'] < -0.2).mean():.3f}, "
                 f"road-raster margin < -0.2 m {(labr['margin'] < -0.2).mean():.3f}; static rows {t['static'].mean():.3f}; "
                 f"|dpsi| mean {np.degrees(np.abs(off[:, 1])).mean():.2f} deg")
        for name, m in (("v < 1", t["speed"] < 1), ("v 1-3", (t["speed"] >= 1) & (t["speed"] <= 3)), ("v > 3", t["speed"] > 3)):
            run.info(f"  {name}: n {int(m.sum())}, agent {lab['a_hit'][m].mean():.3f}, NAVSIM raster {(lab['b_margin'][m] < -0.2).mean():.3f}, road raster {(labr['margin'][m] < -0.2).mean():.3f}")
        # 3-10 per group: launch / low speed first (the rows to check by eye), then the user classes; positives before negatives
        pos = lab["a_hit"][:, 0] | (labr["margin"][:, 0] < -0.2)
        groups = {"launch_v<1": t["speed"] < 1, "slow_1-3": (t["speed"] >= 1) & (t["speed"] <= 3), "class1": (uc == 1) & (t["speed"] > 3),
                  "class2": (uc == 2) & (t["speed"] > 3), "class3": (uc == 3) & (t["speed"] > 3)}
        _sys.path[:0] = [str(B.REPO / "experiments/alpasim/scripts"), str(B.REPO / "experiments/alpasim/lib")]
        import ap2_prep as AP
        from experiments.op_adapt_h.lib import op_adapt_h as H
        T10 = OR.t_all()
        out = B.REPO / "experiments/body1/figs/loss"
        out.mkdir(parents=True, exist_ok=True)
        dn = SW.dense(own[:, None], off[:, None])[:, 0]                                  # (n, 41, 3) logged frame
        HL, HW, CEN = 5.176 / 2, 2.297 / 2, (4.049 - 1.127) / 2

        def box_xy(x, y, yaw, hl, hw):
            c, s = np.cos(yaw), np.sin(yaw)
            u = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
            return x + c * u[:, 0] - s * u[:, 1], y + s * u[:, 0] + c * u[:, 1]

        def gray(fr):                                                                     # (6, 128, 256) YUV420 planes -> (256, 512) luma
            g = np.zeros((256, 512), np.uint8)
            g[0::2, 0::2], g[1::2, 0::2], g[0::2, 1::2], g[1::2, 1::2] = fr[0], fr[1], fr[2], fr[3]
            return g
        for gname, gm in groups.items():
            idx = np.r_[np.flatnonzero(gm & pos)[: a.per // 2 + 1], np.flatnonzero(gm & ~pos)[: a.per]][: a.per]
            if not len(idx):
                run.info(f"{gname}: no rows")
                continue
            ents = AP.entries(t["names"][idx].tolist(), t["log"][idx].tolist(), run)
            base = B.tab("log", k)
            fig, ax = plt.subplots(len(idx), 3, figsize=(15, 4.2 * len(idx)), squeeze=False, gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
            for j, i in enumerate(idx):
                sr = t["src_row"][i]
                T_ = T10
                ys, ps = (np.full(10, off[i, 0]), np.full(10, off[i, 1])) if t["static"][i] else H.drift(T10, float(off[i, 0]), float(off[i, 1]), float(t["speed"][i]))
                pose, vel, cam = (base[q][sr].astype(np.float64) for q in ("pose", "vel", "cam"))
                fr_p = OR._job((ents[j], pose, vel, cam, T_, ys, ps))
                fr_0 = OR._job((ents[j], pose, vel, cam, T_, np.zeros(10), np.zeros(10)))
                A = ax[j, 0]
                xs, ys_ = -8 + 0.5 * (np.arange(128) + .5), -24 + 0.5 * (np.arange(96) + .5)
                A.contourf(ys_, xs, L["sdf"][grow[i]].astype(np.float32), levels=[0, 99], colors=["#dddddd"])
                A.contour(ys_, xs, road["sdf"][grow[i]].astype(np.float32), levels=[0], colors=["#1f77b4"], linewidths=1.2)
                bx, vd = L["box"][grow[i]], L["valid"][grow[i]]
                for tt, al in ((0, 1.0), (4, 0.5), (8, 0.25)):
                    for o in np.flatnonzero(vd[tt]):
                        px, py = box_xy(bx[tt, o, 0], bx[tt, o, 1], bx[tt, o, 2], bx[tt, o, 3] / 2, bx[tt, o, 4] / 2)
                        A.plot(py, px, color="#d62728", alpha=al, lw=1)
                A.plot(dn[i, :, 1], dn[i, :, 0], color="k", lw=1.5)
                for st in (0, 20, 40):
                    c, s = np.cos(dn[i, st, 2]), np.sin(dn[i, st, 2])
                    px, py = box_xy(dn[i, st, 0] + c * CEN, dn[i, st, 1] + s * CEN, dn[i, st, 2], HL, HW)
                    A.plot(py, px, color="#2ca02c", lw=1.2)
                f = t["fut"][i].astype(np.float64)[None]
                fl = SW.to_log(f, off[i][None])[0]
                A.plot(fl[:, 1], fl[:, 0], "--", color="#7f7f7f", lw=1)
                A.set_xlim(14, -14), A.set_ylim(-6, 36), A.set_aspect("equal")
                A.set_title(f"{t['names'][i][:8]} v0 {t['speed'][i]:.1f} m/s, off {off[i, 0]:+.2f} m / {np.degrees(off[i, 1]):+.1f} deg{' static' if t['static'][i] else ''}\n"
                            f"{B.MAN[man[i]]} / {B.HAZ[haz[i]]}; agent hit {int(lab['a_hit'][i, 0])} (clr {min(lab['a_clr'][i, 0], 9.9):.2f} m), "
                            f"margin raster {lab['b_margin'][i, 0]:+.2f} / road {labr['margin'][i, 0]:+.2f} m", fontsize=8)
                ax[j, 1].imshow(gray(fr_0[3][0]), cmap="gray"), ax[j, 1].set_title("road view, logged pose (newest key)", fontsize=8)
                ax[j, 2].imshow(gray(fr_p[3][0]), cmap="gray"), ax[j, 2].set_title("road view, bd4 pose (plane reprojection)", fontsize=8)
                for q in (1, 2):
                    ax[j, q].axis("off")
            fig.suptitle(f"bd4 preview, {gname}: grey = NAVSIM drivable raster, blue = scorer-layer raster edge, red = agent boxes at 0 / 2 / 4 s, "
                         f"black + green = student plan and ego box at 0 / 2 / 4 s, dashed = logged future", fontsize=9)
            fig.tight_layout()
            fn = out / f"bd4_preview_{gname.replace('<', 'lt').replace(' ', '')}.png"
            fig.savefig(fn, dpi=70)
            plt.close(fig)
            run.info(f"{gname}: {len(idx)} states -> {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--zero", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("count")
    p = sub.add_parser("preview")
    p.add_argument("--data", default="navtrain_full.s2of12")
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--per", type=int, default=6)
    p.add_argument("--model", default="P2H10-F-s0")
    a = ap.parse_args()
    {"prep": cmd_prep, "count": cmd_count, "preview": cmd_preview}[a.cmd](a)
