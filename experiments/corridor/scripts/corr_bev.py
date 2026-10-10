"""CORR0 BEV check sheet: lane graph, driven lane sequence, corridor centreline and the re-target arms on a few tokens (envs/navsim2).

  $DATA_DIR/envs/navsim2/bin/python experiments/corridor/scripts/corr_bev.py --geom geom_tokens_s10.pkl --tag _s10 [--tokens t1 t2 ..]
-> $DATA_DIR/runs/corridor/report/bev<tag>.png. Map, logged path and driven sequence are privileged: analysis only.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[3] / "research")]
import corr_geom as C  # noqa: E402

LG = C.LG


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PSY
    from shapely.geometry import Point
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="geom.pkl"); ap.add_argument("--tag", default=""); ap.add_argument("--tokens", nargs="*")
    ap.add_argument("--labels", nargs="*", default=[])
    a = ap.parse_args()
    PSY.apply()
    G = {r["token"]: r for r in pickle.load(open(C.OUT / a.geom, "rb"))}
    Z = np.load(C.OUT / f"poses{a.tag}.npz")
    tab = np.load(C.TAB)
    names = tab["names"].astype(str)
    pos = {t: i for i, t in enumerate(names)}
    toks = a.tokens or [t for t in G]
    nc = min(5, len(toks))
    nr = int(np.ceil(len(toks) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(PSY.DOUBLE_COLUMN_IN, PSY.DOUBLE_COLUMN_IN / nc * nr * 1.05), squeeze=False)
    col = PSY.PALETTE
    for n, (ax, t) in enumerate(zip(axs.ravel(), toks)):
        r, i = G[t], pos[t]
        g = LG.get(r["loc"])
        o, yaw = r["o"][:2], r["o"][2]
        fut = tab["fut"][i]
        seq = set(r.get("seq", []))
        for k in g.tree.query(Point(*o).buffer(45), predicate="intersects"):
            e = C.to_ego(np.asarray(g.poly[k].exterior.coords)[:, :2], o, yaw)
            on = g.ids[k] in seq
            ax.fill(-e[:, 1], e[:, 0], fc=col["sky_blue"] if on else "#DDDDDD", ec="#888888" if not on else col["blue"], lw=0.25, alpha=0.45 if on else 0.35, zorder=1 + on)
        if "R" in r:
            ax.plot(-r["R"][:, 1], r["R"][:, 0], color=col["blue"], lw=0.8, zorder=4, label="centreline R")
        ax.plot(-np.r_[0, fut[:, 1]], np.r_[0, fut[:, 0]], "k.-", ms=2.5, lw=0.7, zorder=6, label="log")
        for key, c, lab in (("sh0_pp", col["orange"], "plan (PP)"), ("sh0_cp", col["green"], "CP"), ("sh0_ce", col["vermillion"], "CE"), ("sh0_kp", col["purple"], "KP")):
            p = Z[key][i]
            ax.plot(-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], ".-", color=c, ms=2, lw=0.7, zorder=5, label=lab)
        m = max(12.0, float(np.abs(fut[:, :2]).max()) + 6)
        ax.set_xlim(-m, m); ax.set_ylim(-6, 2 * m - 6); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        lab = a.labels[n] if n < len(a.labels) else ""
        ax.text(0.02, 0.98, f"{t[:6]} {lab}\n{r['status']} {r.get('sh0_cls', '')}", transform=ax.transAxes, va="top", fontsize=5.5)
    for ax in axs.ravel()[len(toks):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=5, loc="lower left")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.04, hspace=0.04)
    (C.OUT / "report").mkdir(exist_ok=True)
    print(PSY.save(fig, C.OUT / "report" / f"bev{a.tag}"))


if __name__ == "__main__":
    main()
