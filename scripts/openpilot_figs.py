"""Figures and pooled tables for the openpilot smoke run (reads an openpilot_replay run dir).

  python scripts/openpilot_figs.py <replay run dir> --seg <segment id for the timeline> --out <dir>
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import plot_style as ps
from jevdrive.openpilot.frames import MEDMODEL_K, VIEW_FROM_DEVICE, unpack_luma
from jevdrive.openpilot.model import T_IDXS

COLORS = {"small": ps.PALETTE["orange"], "cinque": ps.PALETTE["blue"], "lebowski": ps.PALETTE["vermillion"]}
LABELS = {"small": "small (30M)", "cinque": "Cinque v3 (382M)", "lebowski": "Lebowski (877M)"}
CAM_HEIGHT = 1.22  # m, device above road, as openpilot's UI uses to draw the path
WARMUP = 100


def pooled(rows):
    out = {}
    for r in rows:
        out.setdefault(r["model"], []).append(r)
    keys = [k for k in rows[1] if "@" in k or k.endswith(("_mae", "_r"))]
    table = {}
    for m, rs in out.items():
        n = np.array([r["n"] for r in rs], float)
        table[m] = {"n": int(n.sum()), "segs": len(rs)} | {
            k: float(np.average([r[k] for r in rs], weights=n)) for k in keys if k in rs[0]}
    return table


def timeline(z, models, path):
    t = z["gt/t"]
    fig, axs = plt.subplots(3, 1, figsize=(ps.DOUBLE_COLUMN_IN, 4.2), sharex=True, constrained_layout=True)
    j2 = np.interp(2.0, T_IDXS, np.arange(33))
    lo = int(j2)
    at2 = lambda a: a[:, lo] * (lo + 1 - j2) + a[:, lo + 1] * (j2 - lo)  # noqa: E731
    axs[0].plot(t, at2(z["gt/gt_v"]), color="k", lw=1.3, label="actual speed at $t$+2 s")
    axs[1].plot(t, np.interp(t + 0.275, t, z["gt/gt_curv"]) * 1e3, color="k", lw=1.3, label="actual")
    axs[2].plot(t, np.interp(t + 0.525, t, z["gt/gt_accel"]), color="k", lw=1.3, label="actual")
    for m in models:
        kw = dict(color=COLORS[m], lw=0.9, label=LABELS[m])
        axs[0].plot(t, at2(z[f"{m}/plan_vel"][..., 0]), **kw)
        axs[1].plot(t, z[f"{m}/curvature"] * 1e3, **kw)
        axs[2].plot(t, z[f"{m}/accel_smooth"], **kw)
    for ax in axs:
        ax.axvspan(0, t[WARMUP], color="#DDDDDD", lw=0, zorder=0)
    axs[0].set_ylabel("speed (m/s)")
    axs[1].set_ylabel("curvature (1/km)")
    axs[2].set_ylabel("accel (m/s$^2$)")
    axs[2].set_xlabel("time in segment (s)")
    axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    for ax, l in zip(axs, "abc"):
        ps.panel(ax, f"({l})")
    return ps.save(fig, path)


def horizon(tables_by_h, models, path):
    fig, axs = plt.subplots(1, 2, figsize=(ps.SINGLE_COLUMN_IN * 2, 2.0), constrained_layout=True)
    for ax, key, lab in zip(axs, ("lat", "lon"), ("lateral", "longitudinal")):
        for m in ["const-vel", *models]:
            h, e = tables_by_h[m][key]
            ax.plot(h, e, marker="o", ms=2.5, color=ps.BASELINE if m == "const-vel" else COLORS[m],
                    ls="--" if m == "const-vel" else "-", label="constant velocity" if m == "const-vel" else LABELS[m])
        ax.set_xlabel("horizon (s)")
        ax.set_ylabel(f"mean |{lab} error| (m)")
    axs[0].legend()
    ps.panel(axs[0], "(a)")
    ps.panel(axs[1], "(b)")
    return ps.save(fig, path)


def errors_vs_horizon(zs, models):
    """Pooled mean |error| at every plan time index <= 6 s, frames >= WARMUP with ground truth."""
    idx = np.where(T_IDXS <= 6.0)[0]
    out = {}
    for m in ["const-vel", *models]:
        acc = {"lat": [], "lon": []}
        for z in zs:
            g = z["gt/gt_pos"][WARMUP:, idx]
            p = z["gt/cv_pos"][WARMUP:, idx] if m == "const-vel" else z[f"{m}/plan_pos"][WARMUP:, idx]
            acc["lat"].append(np.abs(p[..., 1] - g[..., 1]))
            acc["lon"].append(np.abs(p[..., 0] - g[..., 0]))
        out[m] = {k: (T_IDXS[idx], np.nanmean(np.concatenate(v), 0)) for k, v in acc.items()}
    return out


def overlay(z, models, path):
    """Road-camera model frame (Y) with actual and planned paths projected at road height."""
    Y = unpack_luma(z["sample_frames"][0])
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, ps.SINGLE_COLUMN_IN / 2 + 0.25), constrained_layout=True)
    ax.imshow(Y, cmap="gray", vmin=0, vmax=255)

    def draw(p, **kw):
        p = p[np.isfinite(p[:, 0]) & (p[:, 0] > 2)] + [0, 0, CAM_HEIGHT]
        uv = (MEDMODEL_K @ VIEW_FROM_DEVICE @ p.T).T
        ax.plot(uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2], **kw)
    draw(z["gt/gt_pos"][600], color="k", lw=2.2, label="actual")
    draw(z["gt/gt_pos"][600], color="w", lw=0.8)
    for m in models:
        draw(z[f"{m}/plan_pos"][600], color=COLORS[m], lw=1.1, label=LABELS[m])
    ax.set_xlim(0, Y.shape[1])
    ax.set_ylim(Y.shape[0], 0)
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.8)
    return ps.save(fig, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["small", "cinque", "lebowski"])
    a = ap.parse_args()
    ps.apply()
    a.out.mkdir(parents=True, exist_ok=True)
    s = json.loads((a.run / "scores.json").read_text())
    zs = {p.stem: dict(np.load(p)) for p in sorted(a.run.glob("*.npz"))}
    table = pooled(s["rows"])
    print(json.dumps(table, indent=1))
    (a.out / "openpilot-smoke-pooled.json").write_text(json.dumps(dict(pooled=table, rows=s["rows"]), indent=1))
    print(timeline(zs[a.seg], a.models, a.out / "openpilot-smoke-timeline"))
    print(horizon(errors_vs_horizon(list(zs.values()), a.models), a.models, a.out / "openpilot-smoke-horizon"))
    print(overlay(zs[a.seg], a.models, a.out / "openpilot-smoke-overlay"))
