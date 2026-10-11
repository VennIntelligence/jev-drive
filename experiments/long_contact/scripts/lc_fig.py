#!/usr/bin/env python3
"""LONG-PROBE figures (CPU).

  labels   BEV panels of label positives drawn at random from the FIT logs (own and arc query slots; labels only, no model output):
           3 per class A / B / C -> $DATA_DIR/runs/long_contact/figs/labels.{png,pdf} (+ labels.csv with the drawn rows)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / "lib"), str(_pl.Path(__file__).resolve().parents[2] / "body1/scripts")]
import argparse  # noqa: E402

import numpy as np  # noqa: E402

import lc as LC  # noqa: E402
import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402
import sweep as SW  # noqa: E402

QN = {0: "own0", 1: "own1", 16: "arc", 17: "arc", 18: "arc", 19: "arc"}


def cmd_labels(a):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    _sys.path.insert(0, str(B.REPO / "research"))
    import plot_style as P
    import bd1_fig as BF
    from jevdrive.run import Run
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    P.apply()
    with Run("long_contact", "figs-labels", config=vars(a)):
        rng = np.random.default_rng(a.seed)
        L, X = B.labels(), LC.load_ctx()
        picks = []
        for lab in ("a", "b", "c"):
            cand = []
            for f in ("log", "ot1", "yr1"):
                for k in range(B.NSH):
                    z = np.load(LC.root() / "rows" / f"{B.cdir(f, k)}.npz")
                    r = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")
                    fit = ~r["hold"] & ~np.array([C.is_val(l) for l in r["log"].tolist()])
                    for q in QN:
                        for i in np.flatnonzero(fit & (z[f"{lab}_lab"][:, q] == 1)):
                            cand.append((f, k, int(i), q))
            used = set()
            for j in rng.permutation(len(cand)):
                f, k, i, q = cand[j]
                r = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")
                if r["log"][i] in used:
                    continue
                used.add(r["log"][i])
                picks.append((lab, f, k, i, q))
                if len([p for p in picks if p[0] == lab]) == a.per:
                    break
        fig, axs = plt.subplots(3, a.per, figsize=(P.DOUBLE_COLUMN_IN, 1.05 * P.DOUBLE_COLUMN_IN))
        rows = []
        for ax, (lab, f, k, i, q) in zip(axs.ravel(), picks):
            r = np.load(B.root() / "rows" / f"{B.cdir(f, k)}.npz")
            z = np.load(LC.root() / "rows" / f"{B.cdir(f, k)}.npz")
            gi, off = int(r["gi"][i]), r["off"][i]
            sdf = np.asarray(L["sdf"][gi]).astype(np.float32)
            xs, ys = SW.X0 + (np.arange(SW.NH) + 0.5) * SW.RES, SW.Y0 + (np.arange(SW.NW) + 0.5) * SW.RES
            ax.contourf(xs, ys, sdf.T, levels=[-1e3, 0], colors=["#D4D4D4"], zorder=0)
            R = X["route"][gi]
            ax.plot(R[:, 0], R[:, 1], color="#888888", lw=0.8, ls="--", zorder=1)
            box, valid, cls = (np.asarray(L[x][gi]) for x in ("box", "valid", "cls"))
            for kk in np.flatnonzero(cls >= 0):
                v = np.flatnonzero(valid[:, kk])
                if not len(v):
                    continue
                col = P.PALETTE[BF.OBJ_COL[cls[kk]]]
                ax.plot(box[v, kk, 0], box[v, kk, 1], color=col, lw=0.5, zorder=2)
                BF.poly(ax, *box[v[0], kk, :3], box[v[0], kk, 3] / 2, box[v[0], kk, 4] / 2, fc=col, ec="none", alpha=0.75, zorder=3)
            d = SW.dense(r["q"][i, q], off)
            ex, ey = SW.ego_centre(d)
            BF.poly(ax, ex[0], ey[0], d[0, 2], SW.HALF_L, SW.HALF_W, fc="none", ec="#222222", lw=0.45, zorder=4)
            ax.plot(d[:, 0], d[:, 1], color=P.PREDICTION, lw=1.0, zorder=6)
            txt = ""
            if lab == "a":
                t = float(z["a_t"][i, q])
                fi = int(round(t / SW.DT))
                BF.poly(ax, ex[fi], ey[fi], d[fi, 2], SW.HALF_L, SW.HALF_W, fc="none", ec=P.PALETTE["vermillion"], lw=0.7, zorder=7)
                txt = f"A{int(z['a_sub'][i, q])} contact at {t:.1f} s"
            if lab == "b":
                for j in range(LC.NL):
                    if X["ln_type"][gi, j] and X["ln_elig"][gi, j]:
                        s_ = X["ln_d"][gi, j]
                        p = LC.pose_at(R[None], X["rhead"][gi][None], np.array([[s_]], np.float32))[0, 0]
                        nx, ny = -np.sin(p[2]), np.cos(p[2])
                        col = P.PALETTE["vermillion"] if (X["ln_st"][gi, j] == 1).any() else P.PALETTE["green"]
                        ax.plot([p[0] - 2 * nx, p[0] + 2 * nx], [p[1] - 2 * ny, p[1] + 2 * ny], color=col, lw=1.6, zorder=5)
                t = float(z["b_t"][i, q])
                fi = int(round(t / SW.DT))
                fx, fy = LC.front_point(d[fi])
                ax.plot(fx, fy, marker="x", ms=4, mew=0.9, color=P.PALETTE["vermillion"], zorder=8)
                st = "".join({1: "R", 0: "g", -1: "?"}[int(s)] for s in X["ln_st"][gi][np.argmax(X["ln_elig"][gi])])
                txt = f"{'red light' if z['b_kind'][i, q] == 2 else 'stop sign'} crossed at {t:.1f} s; light 0-4 s: {st}"
            if lab == "c":
                s_ = X["tn_spk"][gi]
                p = LC.pose_at(R[None], X["rhead"][gi][None], np.array([[s_]], np.float32))[0, 0]
                ax.plot(p[0], p[1], marker="o", ms=4, mfc="none", mew=0.9, color=P.PALETTE["vermillion"], zorder=8)
                txt = f"peak at {s_:.0f} m: plan +{float(z['c_dv'][i, q]):.1f} m/s over log {float(X['tn_vpk'][gi]):.1f} m/s; turn {float(X['tn_dth'][gi]):.0f} deg"
            ax.set_xlim(-8, 52), ax.set_ylim(-22, 22)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{lab.upper()} | {QN[q]} | {f} | v0 {float(B.tab(f, k)['speed'][i]):.1f} m/s\n{txt}", fontsize=5.2, pad=1.5, loc="left")
            rows.append(dict(label=lab.upper(), fam=f, shard=k, row=i, query=q, token=str(L["tokens"][gi]), log=str(r["log"][i])))
        fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.03, hspace=0.25)
        out = LC.root() / "figs"
        out.mkdir(parents=True, exist_ok=True)
        print(P.save(fig, out / "labels"))
        pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("labels")
    p.add_argument("--per", type=int, default=4)
    cli_args(p)
    a = ap.parse_args()
    {"labels": cmd_labels}[a.cmd](a)
