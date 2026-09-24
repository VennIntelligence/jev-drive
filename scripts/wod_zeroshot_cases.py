"""Failure-case figure for the WOD-E2E zero-shot exam (runs on the box: it needs the camera images).

Picks, from the latest score run, the rater frames where a zero-shot model loses the most RFS against the
constant-velocity baseline (one frame per model, distinct sequences), and draws for each the FRONT camera image
and a bird's-eye view with the three rated trajectories, the logged future, constant velocity and the models.
  python scripts/wod_zeroshot_cases.py [--frames name ...]  ->  <score run>/wod-zeroshot-cases.{png,pdf}
"""
import argparse, io, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import plot_style as S  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

MODELS = {"alpamayo_nav": ("Alpamayo 1.5 (nav, 6 samples)", S.PALETTE["blue"]),
          "op_cinque": ("openpilot Cinque v3", S.PALETTE["orange"]),
          "op_lebowski": ("openpilot Lebowski", S.PALETTE["vermillion"])}


def front_image(name):
    from PIL import Image
    spans, _ = Z.load_spans()
    sp = spans[name]
    with open(data_dir() / "datasets/waymo_e2e/front3" / sp[0], "rb") as f:
        f.seek(sp[1])
        return np.asarray(Image.open(io.BytesIO(f.read(sp[2]))).convert("RGB"))


def adapter_fig(name, out):
    """What each model is shown: Alpamayo's four re-projected f-theta views and openpilot's two model frames."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import wod_zeroshot as W
    spans, _ = Z.load_spans()
    views, cover = W.alp_views(Z.root("packages") / f"{name}.npz", scale=0.25)
    road, wide = W.op_views(name, spans, json.loads((Z.root() / "op_calib.json").read_text()))
    S.apply()
    fig, ax = plt.subplots(2, 3, figsize=(S.DOUBLE_COLUMN_IN, 2.55))
    titles = [f"Alpamayo {v.replace('_', '-')} (covered {c:.0%})" for v, c in zip(("cross_left", "front_wide",
              "cross_right", "front_tele"), cover)] + ["openpilot road frame", "openpilot wide frame"]
    for a, img, t in zip(ax.flat, views + [road[::2, ::2], wide[::2, ::2]], titles):
        a.imshow(img)
        a.set_title(t, fontsize=7)
        a.axis("off")
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.01, hspace=.18, wspace=.04)
    print(S.save(fig, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--run", default=None)
    ap.add_argument("--adapter", help="frame name: draw the adapter figure for it instead")
    a = ap.parse_args()
    if a.adapter:
        return adapter_fig(a.adapter, data_dir() / "runs/wod_zeroshot/wod-zeroshot-adapter")
    run = Path(a.run) if a.run else sorted((data_dir() / "runs/wod_zeroshot/score").iterdir())[-1]
    z = np.load(run / "per_frame.npz", allow_pickle=False)
    names = [str(n) for n in z["names"]]
    cots = json.loads((run / "cot.json").read_text()) if (run / "cot.json").exists() else {}
    rfs = {k[4:]: z[k] for k in z.files if k.startswith("rfs/")}
    pred = {k[5:]: z[k] for k in z.files if k.startswith("pred/")}
    models = [m for m in MODELS if m in pred]
    if a.frames:
        picks = [(names.index(n), None) for n in a.frames]
    else:
        picks, seen = [], set()
        for m in models:  # the frame where this model loses most against cv (mean over its samples)
            loss = rfs["cv"][:, 0] - rfs[m].mean(1)
            for i in np.argsort(-loss):
                if names[i].rsplit("-", 1)[0] not in seen:
                    picks.append((i, m))
                    seen.add(names[i].rsplit("-", 1)[0])
                    break
    S.apply()
    fig, ax = plt.subplots(2, len(picks), figsize=(S.DOUBLE_COLUMN_IN, 4.4),
                           gridspec_kw={"height_ratios": [1, 1.25]}, squeeze=False)
    info = []
    for j, (i, m) in enumerate(picks):
        img = front_image(names[i])
        ax[0, j].imshow(img[200:900:3, ::3])  # downsampled: keeps the PNG under 500 KB
        ax[0, j].axis("off")
        S.panel(ax[0, j], f"({'abcdef'[j]})")
        b = ax[1, j]
        for k in range(3):
            t = z["traj"][i, k]
            b.plot(-t[:, 1], t[:, 0], color="#BBBBBB", lw=2.2, zorder=1)
            b.text(-t[-1, 1], t[-1, 0], f"{z['scores'][i, k]:.0f}", fontsize=6, color="#777777")
        b.plot(-z["logged"][i, :, 1], z["logged"][i, :, 0], "k--", lw=.9, label="logged future", zorder=3)
        c = pred["cv"][i, 0]
        b.plot(-c[:, 1], c[:, 0], color=S.BASELINE, lw=.9, label="constant velocity", zorder=2)
        for mm in models:
            lab, col = MODELS[mm]
            for s, p in enumerate(pred[mm][i]):
                b.plot(-p[:, 1], p[:, 0], color=col, lw=.8, alpha=.8, label=lab if s == 0 else None, zorder=4)
        b.set_aspect("equal", adjustable="datalim")
        b.set_xlabel("lateral, right + (m)")
        if j == 0:
            b.set_ylabel("longitudinal (m)")
        info.append({"panel": "abcdef"[j], "frame": names[i], "cluster": str(z["cluster"][i]), "picked_for": m,
                     "speed": float(z["speed"][i]), **{f"rfs_{k}": float(rfs[k][i].mean()) for k in ["cv", "logged_future"] + models},
                     "cot_nav": cots.get("nav", {}).get(names[i], [])})
    ax[1, 0].legend(loc="upper left", fontsize=6, bbox_to_anchor=(0, -0.28), ncol=5, frameon=False)
    fig.subplots_adjust(left=.07, right=.99, top=.95, bottom=.2, hspace=.12, wspace=.3)
    print(S.save(fig, run / "wod-zeroshot-cases"))
    (run / "cases.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
