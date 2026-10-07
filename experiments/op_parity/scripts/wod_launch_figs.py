"""Figures of the wod-launch lane (results/wod_launch.md). jevdrive env, CPU, on the box (reads the WOD shards).

Per selected standstill rater frame: the two frames the model is fed at t0 (openpilot road and wide model frames, rendered by the harness'
own renderer, scripts/wod_zeroshot_openpilot.model_frames) next to a BEV of the three rater trajectories (with scores), the log, shipped, WP2
(both seeds) and, when present, the Step 2 arm. Selection (fixed before the frames were looked at): per standstill context class (red, green,
stop_sign, lead, open) the frame where WP2 loses most against the top-rated trajectory; for green and stop_sign also the frame at the median
loss among frames with a loss; plus the log-stays frame with the largest WP2 - shipped deficit.

  python experiments/op_parity/scripts/wod_launch_figs.py [--arm WLG]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = _R / "experiments/op_parity/figs/wod_launch"
OUT = _R / "experiments/op_parity/results/wod_launch"


def rgb(fr):
    """(6, 128, 256) packed YUV420 model frame -> (256, 512, 3) uint8 RGB (BT.601 full range)."""
    Y = np.zeros((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = fr[0], fr[1], fr[2], fr[3]
    U, V = (np.repeat(np.repeat(fr[k].astype(np.float32), 2, 0), 2, 1) - 128 for k in (4, 5))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wod_launch_report as R
    import wod_zeroshot_openpilot as H
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir
    FIG.mkdir(parents=True, exist_ok=True)
    C = R.Ctx()
    n = C.n
    fr = pd.read_csv(OUT / "frames.csv").set_index("name").reindex(C.names[:n])
    P = {"shipped": [C.preds("shipped")], "WP2": [C.preds("WP2-full-s0"), C.preds("WP2-full-s1")]}
    if a.arm and (Z.root("preds", f"op_cinque_{a.arm}-full-s0") / f"{C.names[0]}.npz").exists():
        P[a.arm] = [C.preds(f"{a.arm}-full-s0"), C.preds(f"{a.arm}-full-s1")]
    S = {k: [C.rfs(p) for p in v] for k, v in P.items()}
    s_log = C.rfs(C.fut[:n])
    loss = (fr.rfs_top - fr.rfs_WP2).to_numpy()
    stp = C.st["stopped"]
    sel = []
    for c in ("red", "green", "stop_sign", "lead", "open"):
        i = np.flatnonzero(stp & (fr.context == c).to_numpy())
        if not len(i):
            continue
        sel.append((c, "max loss", int(i[loss[i].argmax()])))
        if c in ("green", "stop_sign"):
            j = i[loss[i] > 0]
            j = j[np.argsort(loss[j])]
            if len(j) > 2:
                sel.append((c, "median loss", int(j[len(j) // 2])))
    ss = np.flatnonzero(C.st["SS (stopped, log stays)"])
    sel.append((str(fr.context.iloc[ss[(fr.rfs_WP2 - fr.rfs_shipped).to_numpy()[ss].argmin()]]), "log stays, largest WP2 - shipped deficit",
                int(ss[(fr.rfs_WP2 - fr.rfs_shipped).to_numpy()[ss].argmin()])))
    spans, _ = Z.load_spans()
    H._init(spans, json.loads((Z.root() / "op_calib.json").read_text()), str(data_dir() / "datasets" / "waymo_e2e" / "front3"))
    col = {"shipped": "#1f77b4", "WP2": "#d62728", a.arm: "#2ca02c"}
    rows = []
    for k, (c, why, i) in enumerate(sel):
        name = C.names[i]
        _, _, frames = H.model_frames(name)
        fig = plt.figure(figsize=(13, 6.2))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.08, wspace=0.12)
        for r_, (lab, m) in enumerate((("road frame fed to the model (t0)", 0), ("wide frame fed to the model (t0)", 1))):
            ax = fig.add_subplot(gs[r_, 0])
            ax.imshow(rgb(frames[-1, m]))
            ax.set_title(lab, fontsize=9, loc="left")
            ax.axis("off")
        ax = fig.add_subplot(gs[:, 1])
        order = np.argsort(-C.sc[i])
        for q, o in enumerate(order):
            t = C.traj[i, o]
            ax.plot(-t[:, 1], t[:, 0], "-" if q == 0 else "--", color="k" if q == 0 else "0.55", lw=2.2 if q == 0 else 1.3,
                    label=f"rater {q + 1}: score {C.sc[i, o]:.0f}, {np.linalg.norm(t[19]):.1f} m at 5 s")
            ax.plot(-t[[11, 19], 1], t[[11, 19], 0], "o", color="k" if q == 0 else "0.55", ms=4)
        lg = C.fut[i]
        ax.plot(-lg[:, 1], lg[:, 0], "-", color="#ff7f0e", lw=1.8, label=f"log: RFS {s_log[i]:.1f}, {np.linalg.norm(lg[19]):.1f} m")
        ax.plot(-lg[[11, 19], 1], lg[[11, 19], 0], "s", color="#ff7f0e", ms=4)
        for arm, ps in P.items():
            for s_, p in enumerate(ps):
                ax.plot(-p[i, :, 1], p[i, :, 0], "-", color=col[arm], lw=1.6, alpha=1.0 if s_ == 0 else 0.55,
                        label=f"{arm}{'' if len(ps) == 1 else f' s{s_}'}: RFS {S[arm][s_][i]:.1f}, {np.linalg.norm(p[i, 19]):.1f} m")
                ax.plot(-p[i, [11, 19], 1], p[i, [11, 19], 0], "^", color=col[arm], ms=4, alpha=1.0 if s_ == 0 else 0.55)
        ax.plot(0, 0, "k*", ms=10)
        ymax = max(3.0, 1.1 * max(np.abs(C.traj[i, :, :, 0]).max(), np.abs(lg[:, 0]).max(), *(np.abs(p[i, :, 0]).max() for ps in P.values() for p in ps)))
        xmax = max(3.0, 1.2 * max(np.abs(C.traj[i, :, :, 1]).max(), np.abs(lg[:, 1]).max(), *(np.abs(p[i, :, 1]).max() for ps in P.values() for p in ps)))
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-0.05 * ymax, ymax)
        ax.set_xlabel("lateral (m, left is left; axis stretched)")
        ax.set_ylabel("forward (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        ax.set_title(f"{c} ({why}); v0 {C.v0[i]:.2f} m/s; markers at 3 s and 5 s", fontsize=9)
        fig.suptitle(f"{k:02d}  {name}", fontsize=9, x=0.02, ha="left")
        f = FIG / f"{k:02d}_{c}_{why.split(',')[0].replace(' ', '_')}_{name[:8]}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        rows.append({"figure": f.name, "frame": name, "context": c, "why": why, "v0": C.v0[i], "light": fr.light.iloc[i], "scores": " ".join(f"{x:.0f}" for x in C.sc[i][order]),
                     "top d5": np.linalg.norm(C.traj[i, order[0], 19]), "log d5": np.linalg.norm(lg[19]), "RFS log": s_log[i],
                     **{f"RFS {arm}": float(np.mean([s[i] for s in S[arm]])) for arm in P}, **{f"d5 {arm}": float(np.mean([np.linalg.norm(p[i, 19]) for p in ps])) for arm, ps in P.items()}})
    pd.DataFrame(rows).to_csv(OUT / "figures.csv", index=False, float_format="%.2f")
    print(pd.DataFrame(rows).to_string())

    # d4 histogram on r2-dev standstill rows (targets_hist.csv): what to look at = WP2 fills the valley between stay and go
    h = pd.read_csv(OUT / "targets_hist.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    arms = ["log", "shipped", "WP1", "WP2"]
    lab = [f"{lo:g}-{hi:g}" if hi < 100 else f">{lo:g}" for lo, hi in h[h.arm == "log"][["bin_lo", "bin_hi"]].to_numpy()]
    for j, (arm, c_) in enumerate(zip(arms, ("#ff7f0e", "#1f77b4", "#7f7f7f", "#d62728"))):
        ax.bar(np.arange(len(lab)) + (j - 1.5) * 0.2, h[h.arm == arm].share.to_numpy(), 0.2, label=arm, color=c_)
    ax.set_xticks(np.arange(len(lab)), lab)
    ax.set_xlabel("distance at 4 s (m), r2-dev standstill rows (v0 < 0.5 m/s, 1 527 rows)")
    ax.set_ylabel("share of rows")
    ax.legend(frameon=False)
    fig.savefig(FIG / "d4_hist_dev.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="WLG")
    main(ap.parse_args())
