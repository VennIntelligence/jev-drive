"""BODY1 stage 2, P0-D2 example rows (plans/2026-10-10-stage2-prereg.md): BEV + the model's road frame, the head's two probabilities and the truth.

Rows are picked by rule from the primary subset (speed >= 0.5 m/s), query own0, MKZ truth, two-seed score, each from a different route:
boundary hit / miss / false alarm and agent hit / miss / false alarm (highest-score positives, lowest-score positives, highest-score negatives).
  -> experiments/body1/figs/s2p0/cases.png, experiments/body1/results/s2p0/cases.csv

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/s2p0_fig.py
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import argparse  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import sweep as SW  # noqa: E402
import s2p0_read as R  # noqa: E402

for _p in ("experiments/b2d_collect/scripts", "experiments/b2d_collect/lib"):
    _sys.path.insert(0, str(B.REPO / _p))
PLAN = (("boundary hit", "b", 1, -1, 2), ("boundary miss", "b", 1, 1, 2), ("boundary false alarm", "b", 0, -1, 1),
        ("agent hit", "a", 1, -1, 2), ("agent miss", "a", 1, 1, 2), ("agent false alarm", "a", 0, -1, 1))
BLUE, GREEN, VERM = "#0072B2", "#009E73", "#D55E00"


def main(a, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import b2dc_check as C
    import b2dc_frames as F
    rows, tb, ex = R.subsample()
    z, L = np.load(R.out() / "pred.npz"), np.load(R.out() / "labels.npz")
    s = (z["p_s0"] + z["p_s1"])[:, 0, :2] / 2                                # own0: agent, boundary logits
    p = 1 / (1 + np.exp(-s))
    mg, t0 = L["mkz_b_margin"][:, 0], L["mkz_b_t0"][:, 0]
    yb = (mg < -0.20) & ~t0 & (mg < 90)
    okb = yb | ((mg >= 0) & (mg < 90))
    ya = L["mkz_a_hit"][:, 0]
    prim = tb["speed"] >= 0.5
    used, picks = set(), []
    for name, ct, pos, sgn, k in PLAN:
        y, ok, col = (yb, okb, 1) if ct == "b" else (ya, np.ones(len(ya), bool), 0)
        cand = np.flatnonzero(prim & ok & (y == bool(pos)))
        for i in cand[np.argsort(sgn * s[cand, col], kind="stable")]:
            if ex["route"][i] not in used:
                used.add(ex["route"][i]), picks.append((name, int(i)))
                if sum(n == name for n, _ in picks) == k:
                    break
    fig, ax = plt.subplots(2, len(picks), figsize=(3.6 * len(picks), 7.4), gridspec_kw=dict(height_ratios=[2.2, 1]))
    rec = []
    for j, (name, i) in enumerate(picks):
        cd = R.clip_dir(ex["route"][i])
        tk = int(ex["tick"][i])
        S = np.load(cd / "sdf.npz")
        sdf = S["sdf"][np.searchsorted(S["ticks"], tk)].astype(np.float32)
        box, valid, cls = L["box"][i].astype(np.float32), L["valid"][i], L["cls"][i]
        b = ax[0, j]
        xs, ys = SW.X0 + (np.arange(SW.NH) + 0.5) * SW.RES, SW.Y0 + (np.arange(SW.NW) + 0.5) * SW.RES
        b.contourf(ys, xs, sdf, levels=[-99, 0, 99], colors=["#d9d9d9", "#ffffff"])
        b.contour(ys, xs, sdf, levels=[0], colors="k", linewidths=0.8)
        with R.footprint(**R.MKZ):
            d = SW.dense(z["q"][i, :1])[0]
            ex_, ey_ = SW.ego_centre(d)
            cx, cy = SW.corners(ex_, ey_, d[:, 2], np.full(41, SW.HALF_L, np.float32), np.full(41, SW.HALF_W, np.float32))
        for k in range(0, 41, 5):
            b.fill(cy[k], cx[k], fc="none", ec=BLUE, lw=0.7, alpha=0.35 + 0.6 * k / 40)
        b.plot(d[:, 1], d[:, 0], color=BLUE, lw=1.6, label="student plan (own0)")
        f = np.concatenate([np.zeros((1, 3), np.float32), tb["fut"][i]])
        b.plot(f[:, 1], f[:, 0], color=GREEN, lw=1.2, ls="--", label="expert future")
        for o in np.flatnonzero(cls >= 0):
            for t, al in ((0, 0.9), (8, 0.3)):
                if valid[t, o]:
                    px, py = SW.corners(box[t, o, 0:1], box[t, o, 1:2], box[t, o, 2:3], box[t, o, 3:4] / 2, box[t, o, 4:5] / 2)
                    b.fill(py[0], px[0], fc=VERM, ec=VERM, alpha=al * 0.5, lw=0.6)
        ta, tbd = float(L["mkz_a_t"][i, 0]), float(L["mkz_b_t"][i, 0])
        for tt, mk in ((ta, "x"), (tbd, "+")):
            if tt < 90:
                kk = int(round(tt / 0.1))
                b.plot(d[kk, 1], d[kk, 0], mk, color="k", ms=11, mew=2.2)
        b.set_xlim(14, -14), b.set_ylim(-6, 46), b.set_aspect("equal")
        b.set_title(f"{name}\n{ex['route'][i]} t{tk} {ex['town'][i]} v={tb['speed'][i]:.1f} m/s\np_agent {p[i, 0]:.2f} (truth {int(ya[i])})   p_boundary {p[i, 1]:.2f} (truth {int(yb[i])}, margin {mg[i]:+.2f} m)", fontsize=8)
        if j == 0:
            b.legend(fontsize=7, loc="lower left"), b.set_ylabel("x forward (m); grey = off the drivable raster; red = actor boxes at t0 (dark) and t0 + 4 s (light)")
        pair = F.read_pairs(cd / "frames.mp4", frames=[tk])[0]
        img = C.draw(C.yuv_rgb(pair[0]), "road", [(tb["fut"][i][:, :2], (0, 158, 115)), (z["q"][i, 0][:, :2], (0, 114, 178))])
        ax[1, j].imshow(img), ax[1, j].axis("off")
        rec.append(dict(case=name, route=ex["route"][i], tick=tk, town=ex["town"][i], speed=float(tb["speed"][i]), p_agent=float(p[i, 0]), p_boundary=float(p[i, 1]),
                        truth_agent=int(ya[i]), truth_boundary=int(yb[i]), margin_m=float(mg[i]), t_agent_s=ta, t_boundary_s=tbd))
    fig.suptitle("S0 contact head (navtrain-trained, frozen) zero-shot on CARLA rows: own-plan sweep, MKZ footprint; x = first agent contact, + = first boundary contact", fontsize=10)
    fig.tight_layout()
    o = B.REPO / "experiments/body1/figs/s2p0"
    o.mkdir(parents=True, exist_ok=True)
    fig.savefig(o / "cases.png", dpi=110)
    pd.DataFrame(rec).to_csv(R.RES / "cases.csv", index=False)
    run.info(f"{len(picks)} cases -> {o / 'cases.png'}")


if __name__ == "__main__":
    from jevdrive.data import splits
    from jevdrive.run import Run, cli_args
    ap = argparse.ArgumentParser()
    cli_args(ap)
    a = ap.parse_args()
    with Run("body1", "s2p0-fig", config=vars(a)) as run:
        run.use_split(splits.load("b2d/b2dc-v2-train")), run.use_split(splits.load("b2d/b2dc-v2-val"))
        main(a, run)
