"""BODY1 figures (research/plot_style.py). Run on the box, copy the PNGs to experiments/body1/figs/taxonomy/.

  bev      [--shard 0] 8 representative tokens per user class as BEV with the labels drawn (drivable area, logged agent boxes and their
           4 s tracks, logged future, the student's plan, one perturbed query that makes contact, first-contact marker)
           -> $DATA_DIR/runs/body1/figs/bev_class{1,2,3}.{png,pdf} + bev_tokens.csv
  rates    contact rates per query family and first-contact time distributions from the row summary -> rates.{png,pdf}
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402

import numpy as np  # noqa: E402

import logging  # noqa: E402

import b1 as B  # noqa: E402
import sweep as SW  # noqa: E402

_sys.path.insert(0, str(B.REPO / "research"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
OBJ_COL = ("sky_blue", "purple", "orange", "orange")       # vehicle, generic_object, pedestrian, bicycle
CELLS = {1: [("in-path, static", lambda T: T["f_in"] & T["in_static"]), ("in-path, moving", lambda T: T["f_in"] & ~T["in_static"]), ("go-around", lambda T: T["man"] == 4)],
         2: [("turn 20-45, side", lambda T: (T["man"] == 2) & T["f_side"]), ("turn 20-45, boundary", lambda T: (T["man"] == 2) & T["f_bnd"]),
             ("> 45 deg", lambda T: np.abs(T["dyaw"]) > 45)],
         3: [("straight, boundary", lambda T: (T["man"] == 5) & T["f_bnd"]), ("launch, boundary", lambda T: (T["man"] == 0) & T["f_bnd"]),
             ("straight, no hazard", lambda T: (T["man"] == 5) & (T["haz"] == 3))]}


def pick(T, c, shard, n, rng):
    """n tokens of primary class c in the shard, round-robin over the class's cells (moving tokens only)."""
    base = (T["pc"] == c) & (T["shard"] == shard) & ~T["still"]
    pools = [(name, list(rng.permutation(np.flatnonzero(base & f(T))))) for name, f in CELLS[c]]
    out = []
    while len(out) < n and any(p for _, p in pools):
        for name, p in pools:
            while p and p[0] in [o[1] for o in out]:
                p.pop(0)
            if p and len(out) < n:
                out.append((name, p.pop(0)))
    return out


def poly(ax, cx, cy, yaw, hl, hw, **kw):
    import matplotlib.patches as mp
    x, y = SW.corners(np.float32(cx), np.float32(cy), np.float32(yaw), np.float32(hl), np.float32(hw))
    ax.add_patch(mp.Polygon(np.c_[x, y], closed=True, **kw))


def panel(ax, P, T, i, cell, z, L, r):
    """One token: i = taxonomy row, r = row of the token in the rows unit z."""
    gi = T["gi"][i]
    sdf = np.asarray(L["sdf"][gi]).astype(np.float32)
    xs, ys = SW.X0 + (np.arange(SW.NH) + 0.5) * SW.RES, SW.Y0 + (np.arange(SW.NW) + 0.5) * SW.RES
    ax.contourf(xs, ys, sdf.T, levels=[-1e3, 0], colors=["#D4D4D4"], zorder=0)
    box, valid, cls = (np.asarray(L[x][gi]) for x in ("box", "valid", "cls"))
    for k in np.flatnonzero(cls >= 0):
        v = np.flatnonzero(valid[:, k])
        if not len(v):
            continue
        col = P.PALETTE[OBJ_COL[cls[k]]]
        ax.plot(box[v, k, 0], box[v, k, 1], color=col, lw=0.5, zorder=2)
        poly(ax, *box[v[0], k, :3], box[v[0], k, 3] / 2, box[v[0], k, 4] / 2, fc=col, ec="none", alpha=0.75, zorder=3)
        if np.hypot(*(box[v[-1], k, :2] - box[v[0], k, :2])) > 1.0:
            poly(ax, *box[v[-1], k, :3], box[v[-1], k, 3] / 2, box[v[-1], k, 4] / 2, fc="none", ec=col, lw=0.5, ls=(0, (2, 1)), zorder=3)
    d = SW.dense(z["q"][r])                                                    # (Q, 41, 3), on-log state
    ex, ey = SW.ego_centre(d)
    for t in (0, 20, 40):
        poly(ax, ex[3, t], ey[3, t], d[3, t, 2], SW.HALF_L, SW.HALF_W, fc="none", ec="#222222", lw=0.45, zorder=4)
    ax.plot(d[3, :, 0], d[3, :, 1], color="#222222", lw=0.9, zorder=5)
    ax.plot(d[0, :, 0], d[0, :, 1], color=P.PREDICTION, lw=0.9, zorder=6)
    pert = np.flatnonzero((z["qfam"] >= 4) & (z["a_hit"][r] | z["b_hit"][r]) & ~z["a_t0"][r] & ~z["b_t0"][r])
    if len(pert):
        j = pert[np.argmax(z["a_hit"][r][pert])]                              # prefer an agent contact
        ag = bool(z["a_hit"][r, j])
        f = int(round(float(z["a_t" if ag else "b_t"][r, j]) / SW.DT))
        ax.plot(d[j, :f + 1, 0], d[j, :f + 1, 1], color=P.PALETTE["vermillion"], lw=0.9, zorder=7)
        poly(ax, ex[j, f], ey[j, f], d[j, f, 2], SW.HALF_L, SW.HALF_W, fc="none", ec=P.PALETTE["vermillion"], lw=0.6, zorder=7)
        ax.plot(ex[j, f], ey[j, f], marker="x", ms=3, mew=0.7, color=P.PALETTE["vermillion"], zorder=8)
    ax.set_xlim(-8, 44)
    ax.set_ylim(-19.5, 19.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    flags = "+".join(n for n, q in (("in-path", "f_in"), ("side", "f_side"), ("bnd", "f_bnd"), ("go-around", "f_ga")) if T[q][i]) or "none"
    own = f"own: agent {'%.1f s' % z['a_t'][r, 0] if z['a_hit'][r, 0] else 'no'} (clr {min(z['a_clr'][r, 0], 9.9):.1f}), bnd {'%.1f s' % z['b_t'][r, 0] if z['b_hit'][r, 0] else 'no'} ({min(z['b_margin'][r, 0], 9.9):.1f})"
    ax.set_title(f"{B.MAN[T['man'][i]]} | {flags} | {T['v0'][i]:.0f} m/s\n{own}", fontsize=5.6, pad=1.5, loc="left")
    return dict(token=str(T["token"][i]), cell=cell, man=B.MAN[T["man"][i]], flags=flags, dyaw=float(T["dyaw"][i]), v0=float(T["v0"][i]), city=str(T["city"][i]),
                own_agent=bool(z["a_hit"][r, 0]), own_boundary=bool(z["b_hit"][r, 0]), own_clr=float(z["a_clr"][r, 0]), own_margin=float(z["b_margin"][r, 0]))


def cmd_bev(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import plot_style as P
    from jevdrive.run import Run
    P.apply()
    with Run("body1", "fig-bev", seed=a.seed, config=vars(a)) as run:
        T = dict(np.load(B.root() / "taxonomy" / f"{a.tax}.npz"))
        L = B.labels()
        z = dict(np.load(B.root() / "rows" / f"{B.cdir('log', a.shard)}.npz"))
        at = {g: r for r, g in enumerate(z["gi"].tolist())}
        out = B.root() / "figs"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in (1, 2, 3):
            sel = pick(T, c, a.shard, 8, np.random.default_rng([a.seed, c]))
            fig, axs = plt.subplots(2, 4, figsize=(P.DOUBLE_COLUMN_IN, 3.25))
            for ax, (cell, i) in zip(axs.ravel(), sel):
                rows.append(dict(cls=c, **panel(ax, P, T, i, cell, z, L, at[int(T["gi"][i])])))
            for ax in axs.ravel()[len(sel):]:
                ax.axis("off")
            fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.03, hspace=0.2)
            info = P.save(fig, out / f"bev_class{c}")
            plt.close(fig)
            run.info(f"class {c}: {len(sel)} tokens, png {info['png_bytes'] / 1024:.0f} KiB")
        pd.DataFrame(rows).to_csv(out / "bev_tokens.csv", index=False)
        run.summary.update(out=str(out), tokens=len(rows))


def cmd_rates(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import plot_style as P
    P.apply()
    S = B.root() / "rows" / "summary"
    R, H = pd.read_csv(S / "contact_rates.csv"), pd.read_csv(S / "first_contact_time.csv")
    order = ["log", "ship", "own0", "own1", "lat", "head", "gain", "arc", "stop"]
    fig, axs = plt.subplots(1, 3, figsize=(P.DOUBLE_COLUMN_IN, 2.1))
    for ax, kind, lab in ((axs[0], "agent_after_t0", "agent contact after t = 0 (%)"), (axs[1], "boundary_after_t0", "boundary contact after t = 0 (%)")):
        for j, (sf, col) in enumerate((("log", P.BASELINE), ("ot1", P.PALETTE["blue"]), ("yr1", P.PALETTE["orange"]))):
            x = R[R.sfam == sf].set_index("qfam").reindex(order)
            ax.bar(np.arange(len(order)) + (j - 1) * 0.27, 100 * x[kind], 0.27, color=col, label={"log": "on-log states", "ot1": "off-track ot1", "yr1": "off-track yr1"}[sf])
        ax.set_xticks(np.arange(len(order)), order, rotation=60)
        ax.set_ylabel(lab)
        P.bars(ax)
    axs[0].legend(loc="upper left")
    t = np.r_[0, np.arange(0.5, 4.01, 0.5)]
    for qf, col in (("own0", P.PALETTE["blue"]), ("lat", P.PALETTE["vermillion"]), ("log", P.BASELINE)):
        for kind, ls in (("agent", "-"), ("boundary", "--")):
            h = H[(H.qfam == qf) & (H.kind == kind)][[f"t{b:.1f}" for b in t]].to_numpy()[0].astype(float)
            axs[2].plot(t, 100 * h / max(h.sum(), 1), ls, color=col, label=f"{qf}, {kind}")
    axs[2].set_xlabel("first contact time (s, 0.5 s bins)")
    axs[2].set_ylabel("share of contacts (%)")
    axs[2].legend(fontsize=6.5)
    fig.tight_layout(pad=0.4)
    out = B.root() / "figs"
    out.mkdir(parents=True, exist_ok=True)
    print(P.save(fig, out / "rates"))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bev")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--tax", default="tax")
    cli_args(p)
    sp.add_parser("rates")
    a = ap.parse_args()
    {"bev": cmd_bev, "rates": cmd_rates}[a.cmd](a)
