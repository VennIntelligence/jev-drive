"""COL1, PAI track: one review GIF per rollout. Left: bird's-eye view, ego heading up (ego box, driven path, logged path, the served
4 s plan and the shipped policy's plan on the same tokens, other actors, the struck one in red). Right: the road and wide model frames
the adapter was fed at that decision (newest slot, from the offline replay). Bottom: ego / log speed, the simulator's gap to the struck
object (controls: nearest object in the corridor) against the lead head's distance, and the lead probability; the cursor is the frame's
time, the red line the first at-fault step.

  col1_pai_clip.py --x <extract dir> --dec <pai_decisions.npz> --cases <csv: scene,x_dir_name,t_from,t_to,name> --out <dir>
                   [--every 4] [--width 760] [--colors 96]        (numpy, matplotlib, PIL; simulator state = labels)
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import col1_lib as L  # noqa: E402
import col1_pai as P  # noqa: E402


def rgb(fr):
    """Model frame (6, 128, 256) uint8 (4 luma phases, U, V) -> (256, 512, 3) RGB."""
    Y = np.zeros((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = fr[0], fr[1], fr[2], fr[3]
    U, V = (np.repeat(np.repeat(fr[i].astype(np.float32) - 128, 2, 0), 2, 1) for i in (4, 5))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def render(o, rp, D, t_from, t_to, every, width, colors, title):
    off, ego, gt = L.center_off(o), o["actors"]["EGO"], o["logged"][0]["traj"]
    Le, We = o["size"]["EGO"][:2]
    T0, t0 = ego[0, 0], rp["t0"].astype(float)
    te = L.first_event(o, "collision_at_fault")
    k_obj = P.struck(o, te)[0] if te is not None else None
    t_evt = None if te is None else (te - T0) * 1e-6
    fk = {int(k): i for i, k in enumerate(rp["frame_k"])}
    v_log = L.speed(gt, t0)
    ks = [k for k in range(len(t0)) if t_from <= D["t"][k] <= t_to and k % every == 0 and k in fk]
    imgs = []
    for k in ks:
        fig = plt.figure(figsize=(width / 100, width * 0.62 / 100), dpi=100)
        gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.25, 1.25], height_ratios=[1, 1, 0.8], left=0.045, right=0.99, top=0.94, bottom=0.08,
                              wspace=0.12, hspace=0.25)
        ax = fig.add_subplot(gs[:2, 0])
        c = L.interp(ego, t0[k])[0]
        fr = np.array([c[0], c[1], c[2] - np.pi / 2])                      # heading up
        T = lambda xy: L.into(fr, xy)                                      # noqa: E731
        ax.plot(*T(gt[:, 1:3]).T, color="0.6", lw=1, ls="--")
        ax.plot(*T(ego[ego[:, 0] <= t0[k], 1:3]).T, color="k", lw=1)
        rig = L.to_rig(c[None], off)[0]
        for n, col, lw in (("p0", "#0072B2", 1.2), ("ft", "#009E73", 1.8)):
            pl = np.c_[rig[0] + rp[n][k][:, 0] * np.cos(rig[2]) - rp[n][k][:, 1] * np.sin(rig[2]),
                       rig[1] + rp[n][k][:, 0] * np.sin(rig[2]) + rp[n][k][:, 1] * np.cos(rig[2])]
            ax.plot(*T(np.r_[rig[None, :2], pl]).T, color=col, lw=lw)
        for a, tr in o["actors"].items():
            if tr[0, 0] - 1e5 <= t0[k] <= tr[-1, 0] + 1e5:
                q = T(L.corners(*L.interp(tr, t0[k])[0], *o["size"][a][:2]))
                ax.fill(*q.T, color="k" if a == "EGO" else "#D55E00" if a == k_obj else "0.75", lw=0)
        ax.set_xlim(-14, 14), ax.set_ylim(-10, 46), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title("plan: served (green), shipped (blue)", fontsize=6.5, pad=2)
        f = rp["frames"][fk[k]]
        for j, name in enumerate(("road frame fed", "wide frame fed")):
            a2 = fig.add_subplot(gs[j, 1:])
            a2.imshow(rgb(f[j])), a2.set_xticks([]), a2.set_yticks([])
            a2.set_ylabel(name, fontsize=6.5)
        tw = (D["t"] >= t_from) & (D["t"] <= t_to)
        for j, (ys, lab, yl) in enumerate(((((D["ve"], "k", "ego"), (v_log, "0.6", "log")), "speed m/s", None),
                                           (((D["g"], "#D55E00", "gap (sim)"), (D["d_p0"], "#0072B2", "lead head")), "gap m", (-2, 60)),
                                           (((D["p_p0"], "#0072B2", "shipped"), (D["p_ft"], "#009E73", "served")), "lead prob", (-0.05, 1.05)))):
            a3 = fig.add_subplot(gs[2, j])
            for y, col, ln in ys:
                a3.plot(D["t"][tw], y[tw], color=col, lw=1, label=ln)
            a3.axvline(D["t"][k], color="k", lw=0.6)
            if t_evt is not None and t_from <= t_evt <= t_to:
                a3.axvline(t_evt, color="r", lw=1)
            if yl:
                a3.set_ylim(*yl)
            a3.set_title(lab, fontsize=6.5, pad=1), a3.tick_params(labelsize=5.5, length=2, pad=1), a3.legend(fontsize=5, frameon=False, loc="upper left")
        hit = "" if t_evt is None else ("  IMPACT" if D["t"][k] >= t_evt else f"  impact in {t_evt - D['t'][k]:.1f} s")
        fig.suptitle(f"{title}   t = {D['t'][k]:.1f} s   v = {D['ve'][k]:.1f} m/s{hit}", fontsize=7.5, y=0.985,
                     color="r" if t_evt is not None and D["t"][k] >= t_evt else "k")
        b = io.BytesIO()
        fig.savefig(b, format="png"), plt.close(fig)
        imgs.append(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    pal = imgs[len(imgs) // 2].quantize(colors, method=Image.Quantize.MEDIANCUT)
    return [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", nargs="+", required=True), ap.add_argument("--dec", required=True), ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True), ap.add_argument("--every", type=int, default=4), ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--colors", type=int, default=96), ap.add_argument("--only", default="")
    a = ap.parse_args()
    z = np.load(a.dec)
    logs = {Path(x).name: (x, L.load(f"{x}/logs.pkl")) for x in a.x}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(a.cases)):
        if a.only and r["name"] not in a.only.split(","):
            continue
        x, lg = logs[r["run"]]
        s = next(k for k in lg if k[7:15] == r["scene"])
        D = {k.split("/")[1]: z[k] for k in z.files if k.startswith(r["scene"] + "/")}
        fr = render(lg[s], dict(np.load(f"{x}/replay/{s}.npz")), D, float(r["t_from"]), float(r["t_to"]), a.every, a.width, a.colors,
                    f"PAI {r['scene']} {r['name']}")
        f = Path(a.out) / f"pai_{r['name']}_{r['scene']}.gif"
        fr[0].save(f, save_all=True, append_images=fr[1:], duration=int(100 * a.every * 1.5), loop=0, optimize=True)
        print(f.name, len(fr), "frames", f.stat().st_size >> 10, "KB", flush=True)


if __name__ == "__main__":
    main()
