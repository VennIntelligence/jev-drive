"""Can the model see the exit it is told to take? Geometry only, no model, no render.

For every (pose, exit) row of the CARLA pair set: the camera is level at the B2D `spec` mount (x 1.59 m ahead of the rear axle, y 0, z 1.86 m;
interface.B2D_SPEC_MOUNT) on the pose at distance d (10 / 20 / 30 m) before the junction (connector start). The exit window is the stretch of
the row's polyline from arc length d to d + 20 m, i.e. the first 20 m after the junction mouth (polyline convention of the sidecars: 10 m vertices,
resampled linearly). A window sample is inside a field of view when it is ahead of the camera and within +-HFOV/2 of the optical axis:
  road  = model road frame, f 910 px, 512 wide: +-15.7 deg (31.4 deg)   wide = model wide frame, f 455 px: +-29.4 deg (58.7 deg)
  120 / 180 = hypothetical horizontal FOVs (+-60 / +-90 deg). For the two model frames the image bottom is also applied (ground point visible
  when the forward distance >= camera height / tan(12.9 deg) = 8.1 m; never binding here since d >= 10 m); the hypothetical FOVs are horizontal only.
Output per group: mean fraction of the window inside (frac), and the share of rows whose whole window is inside (all). Groups: connector turn angle
(|angle|), minimum turn radius of the exit polyline (turn_rmin, m; straight rows have none), and pose distance d for the turning rows.

  python carla_pairs_visibility.py --route <packed>/route.npz --out-md ../results/carla_pairs_visibility.md --out-png ../figs/carla_pairs_visibility.png
"""
import argparse, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive.openpilot import interface as IF  # noqa: E402

CAM_X, CAM_Y, CAM_H = IF.B2D_SPEC_MOUNT
FOVS = {"road 31 deg": np.degrees(np.arctan(256 / 910)), "wide 59 deg": np.degrees(np.arctan(256 / 455)), "120 deg": 60.0, "180 deg": 90.0}
BOTTOM_DEG = np.degrees(np.arctan((256 - 47.6) / 910))        # = (256 - 151.8) / 455: both model frames see 12.9 deg below the horizon
WIN = 20.0


def window(poly, pmask, d):
    """Points of the polyline at arc length d ... d + 20 m, step 0.25 m (linear between the 10 m vertices); fewer when the polyline ends."""
    p = poly[pmask]
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(d, d + WIN + 1e-6, 0.25)
    t = t[t <= s[-1]]
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], -1)


def inside(w, half_deg, bottom):
    xc, yc = w[:, 0] - CAM_X, w[:, 1] - CAM_Y
    ok = (xc > 0) & (np.abs(np.degrees(np.arctan2(yc, np.maximum(xc, 1e-9)))) <= half_deg)
    if bottom:
        ok &= xc >= CAM_H / np.tan(np.radians(BOTTOM_DEG))
    return ok


def per_row(z):
    n = len(z["id"])
    F = {k: np.full(n, np.nan) for k in FOVS}
    for i in range(n):
        w = window(z["poly"][i], z["pmask"][i], float(z["d"][i]))
        if len(w) < 8:
            continue
        for k, h in FOVS.items():
            F[k][i] = inside(w, h, bottom=k.startswith(("road", "wide"))).mean()
    return F


def table(name, groups, F):
    lines = [f"| {name} | rows | " + " | ".join(FOVS) + " |", "|---|--:|" + "--:|" * len(FOVS)]
    out = {}
    for g, m in groups:
        m = m & ~np.isnan(F["road 31 deg"])
        if not m.any():
            continue
        cells = [f"{100 * np.mean(F[k][m]):.0f}% ({100 * np.mean(F[k][m] >= 0.999):.0f}%)" for k in FOVS]
        lines.append(f"| {g} | {int(m.sum())} | " + " | ".join(cells) + " |")
        out[g] = [np.mean(F[k][m]) for k in FOVS]
    return "\n".join(lines), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-png", required=True)
    a = ap.parse_args()
    z = dict(np.load(a.route))
    F = per_row(z)
    ang = np.abs(z["angle"])
    turn = ang >= 25
    r = z["turn_rmin"]
    d = z["d"]
    t_ang, o_ang = table("exit turn angle |angle|", [("straight (< 25 deg)", ang < 25), ("25 - 60 deg", (ang >= 25) & (ang < 60)),
                                                     ("60 - 120 deg (right-angle)", (ang >= 60) & (ang < 120)), (">= 120 deg (u-turn)", ang >= 120)], F)
    t_r, o_r = table("min turn radius R_min (turning rows)", [("tight: R_min < 7 m", turn & (r < 7)), ("mid: 7 - 10 m", turn & (r >= 7) & (r < 10)),
                                                              ("wide: R_min >= 10 m", turn & (r >= 10))], F)
    t_d, o_d = table("pose distance d (rows with |angle| >= 60 deg)", [(f"d = {int(v)} m", (ang >= 60) & (d == v)) for v in (10, 20, 30)], F)
    t_dr, _ = table("d x R_min (|angle| >= 60 deg)", [(f"d = {int(v)} m, " + nm, (ang >= 60) & (d == v) & mm) for v in (10, 20, 30)
                                                      for nm, mm in (("tight (< 7 m)", r < 7), ("wide (>= 10 m)", r >= 10))], F)
    n_ok = int((~np.isnan(F["road 31 deg"])).sum())
    md = f"""# Exit visibility from the open-loop-rig camera (geometry only)

Source: {n_ok} of {len(z['id'])} (pose, exit) rows (the rest have a polyline shorter than the window). Camera level at x {CAM_X} m, y {CAM_Y}, z {CAM_H} m
(`interface.B2D_SPEC_MOUNT`). Each cell: **mean fraction of the exit's first 20 m after the junction mouth that lies inside the FOV** (share of rows whose
whole 20 m window is inside). Window = polyline arc length d ... d + 20 m seen from the pose at distance d before the junction (script docstring has the rest).
FOV half-angles: road {FOVS['road 31 deg']:.1f} deg, wide {FOVS['wide 59 deg']:.1f} deg (the model frames), hypothetical 120 / 180 deg.

{t_ang}

{t_r}

{t_d}

{t_dr}
"""
    Path(a.out_md).write_text(md)
    print(md)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#9aa0a6", "#1a73e8", "#e8710a", "#188038"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (title, o) in zip(axs, (("turn angle", o_ang), ("R_min (turning rows)", o_r), ("d (|angle| >= 60 deg)", o_d))):
        keys, w = list(o), 0.2
        for j, k in enumerate(FOVS):
            ax.bar(np.arange(len(keys)) + (j - 1.5) * w, [100 * o[g][j] for g in keys], w, label=k, color=cols[j])
        ax.set_xticks(range(len(keys)), [g.replace(" (", "\n(") for g in keys], fontsize=7)
        ax.set_title(title, fontsize=10), ax.grid(axis="y", alpha=0.3), ax.set_ylim(0, 105)
    axs[0].set_ylabel("% of the exit's first 20 m inside the FOV")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Is the exit inside the input FOV? camera x 1.59 / z 1.86 m, exit window = 20 m after the junction mouth", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out_png, dpi=110)


if __name__ == "__main__":
    main()
