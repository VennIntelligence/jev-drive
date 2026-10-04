"""Figures of the route polylines: figs/route_samples.png (12 BEV panels, clean + noised) and figs/route_input_frame.png (model input + route).

  envs/jevdrive on the box (CPU): python experiments/op_route_cmd/scripts/route_figs.py

Frames come from op_adapt_H's sample tables (nav = navtrain tokens, wod = WOD-E2E frames); the route fields are attached with lib/route_poly.attach, i.e.
the way the merged trainer reads them. BEV: x to the right, y forward, ego at the origin; black = clean polyline (vertices every 10 m), colours = 3 draws of the
default navigation noise (lateral 1 m / 30 m correlation, 1.5 m along-track jitter, 5% dropped vertices), orange dashed = one draw of closed-loop option C.
Image: the packed 512 x 256 road view of the t0 frame (YUV420 -> RGB) with the clean polyline projected on the ground plane (pinhole K = 910 / 256 / 47.6 of the
openpilot view frame, camera height and offset from the table).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib")]
import route_poly as RP  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

FIGS = REPO / "experiments/op_route_cmd/figs"
H = data_dir() / "runs/op_adapt_H/samples"
R = data_dir() / "processed/op_route_cmd"
K_ROAD = (910.0, 256.0, 47.6)


def tab(dom):
    with np.load(H / dom / "tab.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def rgb(packed):
    Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
    U = U.repeat(2, 0).repeat(2, 1) - 128
    V = V.repeat(2, 0).repeat(2, 1) - 128
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def project(poly, mask, cam):
    """Ground-plane points (x fwd, y left) -> pixels of the 512 x 256 road view; only points in front of the camera."""
    p = poly[mask]
    xc, yc = p[:, 0] - cam[0], p[:, 1] - cam[1]
    ok = xc > 3.0
    f, cx, cy = K_ROAD
    return np.stack([cx - f * yc[ok] / xc[ok], cy + f * cam[2] / xc[ok]], -1)


def pick(rt):
    """12 diverse rows (dom, row) of the sample tables, by category."""
    cats = [
        ("nav", "left ~90, near", lambda r: (r["turn_deg"] > 75) & (r["turn_deg"] < 105) & (r["turn_s"] < 35)),
        ("nav", "right ~90, near", lambda r: (r["turn_deg"] < -75) & (r["turn_deg"] > -105) & (r["turn_s"] < 35)),
        ("nav", "left 45-75", lambda r: (r["turn_deg"] > 45) & (r["turn_deg"] < 75)),
        ("nav", "right 25-45", lambda r: (r["turn_deg"] < -25) & (r["turn_deg"] > -45)),
        ("nav", "u-turn >= 135", lambda r: np.abs(r["turn_deg"]) >= 135),
        ("nav", "turn far (> 70 m)", lambda r: (np.abs(r["turn_deg"]) >= 60) & (r["turn_s"] > 70)),
        ("nav", "junction straight through", lambda r: np.isnan(r["turn_deg"]) & (r["jct_s"] < 30) & (r["plen"] >= 150) & (r["jct_dist"] < 30)),
        ("nav", "straight road", lambda r: np.isnan(r["turn_deg"]) & np.isnan(r["jct_s"]) & (r["plen"] >= 150) & (r["v0"] > 5)),
        ("nav", "short path (log ends)", lambda r: (r["plen"] < 60) & (r["plen"] > 20)),
        ("wod", "WOD left ~90", lambda r: (r["turn_deg"] > 75) & (r["turn_deg"] < 105) & (r["turn_s"] < 30) & (r["v0"] > 3)),
        ("wod", "WOD right ~90", lambda r: (r["turn_deg"] < -75) & (r["turn_deg"] > -105) & (r["turn_s"] < 30) & (r["v0"] > 3)),
        ("wod", "WOD short (5 s only)", lambda r: (r["plen"] < 40) & (r["v0"] > 2)),
    ]
    rng = np.random.default_rng(3)
    out = []
    for dom, name, fn in cats:
        m = np.flatnonzero(fn(rt[dom]) & rt[dom]["has_route"])
        out.append((dom, name, int(rng.choice(m)) if len(m) else -1))
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    tb = {d: tab(d) for d in ("nav", "wod")}
    rt = {"nav": RP.attach(R / "navtrain/route.npz", tb["nav"]["id"]), "wod": RP.attach(R / "wod/route.npz", tb["wod"]["id"])}
    for d in rt:
        rt[d]["v0"] = tb[d]["v0"]
    sel = pick(rt)
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(3, 4, figsize=(17, 17))
    for ax, (dom, name, i) in zip(axs.ravel(), sel):
        if i < 0:
            ax.axis("off")
            continue
        r = rt[dom]
        poly, mask = r["poly"][i], r["pmask"][i]
        for c in ("#1f77b4", "#2ca02c", "#9467bd"):
            p, m = RP.noise_polyline(poly, mask, rng)
            ax.plot(-p[m, 1], p[m, 0], "-o", color=c, ms=3, lw=1, alpha=0.8)
        p, m = RP.noise_polyline(poly, mask, rng, RP.C_OPTION)
        ax.plot(-p[m, 1], p[m, 0], "--", color="#ff7f0e", lw=1.5)
        ax.plot(-poly[mask, 1], poly[mask, 0], "-o", color="k", ms=5, lw=2)
        ax.plot(0, 0, "k^", ms=11)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        pts = poly[mask]
        lo, hi = np.minimum(pts.min(0), [0, 0]) - 8, np.maximum(pts.max(0), [0, 0]) + 8
        c, half = (lo + hi) / 2, max(hi[0] - lo[0], hi[1] - lo[1], 30) / 2
        ax.set_xlim(-(c[1] + half), -(c[1] - half))
        ax.set_ylim(c[0] - half, c[0] + half)
        td = r["turn_deg"][i]
        ax.set_title(f"{name}\n{tb[dom]['id'][i][:8]} v={tb[dom]['v0'][i]:.1f} m/s  turn={'none' if np.isnan(td) else '%+.0f deg at %.0f m' % (td, r['turn_s'][i])}  path={r['plen'][i]:.0f} m",
                     fontsize=9)
    fig.suptitle("Route polylines (hindsight, vertices every 10 m; panels zoom to each path): black = clean label, colours = navigation noise draws, orange dashed = closed-loop option C", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIGS / "route_samples.png", dpi=75)
    plt.close(fig)
    # one frame: model input next to the BEV
    for dom, name, i in sel:
        if dom == "nav" and name == "left ~90, near" and i >= 0:
            break
    imgs = np.load(H / dom / "imgs.npy", mmap_mode="r")
    cam = tb[dom]["cam"][i]
    r = rt[dom]
    poly, mask = r["poly"][i], r["pmask"][i]
    fig = plt.figure(figsize=(17, 5.6))
    for k, view in enumerate(("road", "wide")):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.imshow(rgb(np.asarray(imgs[i, 9, k])))
        ax.set_title(f"model input, t0 frame, {view} view (512 x 256)" + (" + clean route projected on the ground" if k == 0 else ""), fontsize=9)
        ax.axis("off")
        if k == 0:
            uv = project(poly, mask, cam)
            dense, _ = RP.poly_resample(poly[mask], 0.5)
            ud = project(dense, np.ones(len(dense), bool), cam)
            ax.plot(ud[:, 0], ud[:, 1], "-", color="#ff2d55", lw=2)
            ax.plot(uv[:, 0], uv[:, 1], "o", color="#ff2d55", ms=5, mec="w")
            ax.set_xlim(0, 512)
            ax.set_ylim(256, 0)
    ax = fig.add_subplot(1, 3, 3)
    rng = np.random.default_rng(5)
    for c in ("#1f77b4", "#2ca02c"):
        p, m = RP.noise_polyline(poly, mask, rng)
        ax.plot(-p[m, 1], p[m, 0], "-o", color=c, ms=3, lw=1, alpha=0.8)
    ax.plot(-poly[mask, 1], poly[mask, 0], "-o", color="k", ms=5, lw=2)
    ax.plot(0, 0, "k^", ms=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-10, 110)
    ax.set_title(f"BEV: clean (black), 2 noise draws; turn {r['turn_deg'][i]:+.0f} deg at {r['turn_s'][i]:.0f} m", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "route_input_frame.png", dpi=80)
    print("selected", [(d, n, int(i), str(tb[d]["id"][i])) for d, n, i in sel])


if __name__ == "__main__":
    main()
