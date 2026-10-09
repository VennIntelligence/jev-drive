"""M1 before / after strips (decision 205, Mac side): the same scene driven by SH30-F-s0 (strong hinge) and by P2H10-F-s0, each row =
bird's-eye view at the time SH30 fails (grey road area, dashed logged path and box, orange driven path and box, blue plans and other
vehicles), two CAM_F0 frames as delivered, and the road / wide frames the model was fed at the last decision before that time. AlpaSim
renders no third-person view; the bird's-eye view stands in for it. Inputs: tmp/c1 (C1 pickles, map), tmp/m1/show (m1_show.sh).

  .venv/bin/python experiments/alpasim/scripts/m1_figs.py
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_lib as L  # noqa: E402
import c1_review as RV  # noqa: E402

SHOW = L.ROOT / "tmp/m1/show"
FIGS = L.ROOT / "experiments/alpasim/figs/m1"
ROWS = (("sh30", "SH30-F-s0 (hinge 30 / 0.5 m)"), ("p2h10", "P2H10-F-s0 (hinge 10 / 0.3 m)"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ld = lambda n: pickle.load(open(SHOW / n, "rb"))                                         # noqa: E731
    logs = {"sh30": L.load("sh30_logs"), "p2h10": ld("p2h10_logs.pkl")}
    frames = {k: ld(f"{k}_frames.pkl") for k, _ in ROWS}
    replay = {k: ld(f"{k}_replay.pkl") for k, _ in ROWS}
    M = L.load("map")
    FIGS.mkdir(parents=True, exist_ok=True)
    for scene in sorted(frames["sh30"]):
        if scene not in logs["p2h10"] or scene not in M:
            continue
        rd, o0 = L.road(M[scene]), logs["sh30"][scene]
        ts = [t for t in (L.first_event(o0, f) for f in L.zero_flags(o0)) if t]
        t_ev = min(ts) if ts else int(L.ego(o0)[-1, 0])
        fig = plt.figure(figsize=(16, 4.3))
        gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.78, 1.78, 2.0, 2.0], wspace=0.02, hspace=0.22)
        for i, (k, lab) in enumerate(ROWS):
            o = logs[k][scene]
            fl = ", ".join(L.SHORT[f] for f in L.zero_flags(o))
            lat = L.signed_lat(L.gt(o)[:, 1:3], L.interp_pose(L.ego(o), t_ev)[:2])[0]
            RV.bev(fig.add_subplot(gs[i, 0]), o, rd, t_ev, f"{lab}\nscore {o['summary']['score']:.2f}{' (' + fl + ')' if fl else ''}, {lat:+.2f} m off the log at {t_ev * 1e-6:.1f} s", w=14.0)
            fs = frames[k][scene]
            for j, d in enumerate((1.0e6, 0.0)):
                t = min(fs, key=lambda q: abs(q - (t_ev - d)))
                ax = fig.add_subplot(gs[i, 1 + j])
                ax.imshow(Image.open(io.BytesIO(fs[t])))
                ax.set_title(f"CAM_F0 as delivered, t = {t * 1e-6:.1f} s", fontsize=6.5, pad=1), ax.axis("off")
            rp = replay[k][scene]
            q = max(x for x in rp["frames"] if o["drive"][x]["now"] < t_ev)
            cur = rp["frames"][q]["cur"][2]
            for j, (nm, im) in enumerate((("road", cur[0]), ("wide", cur[1]))):
                ax = fig.add_subplot(gs[i, 3 + j])
                ax.imshow(RV.rgb(im))
                ax.set_title(f"model input, {nm} frame, decision {q} (t = {o['drive'][q]['now'] * 1e-6:.1f} s)", fontsize=6.5, pad=1), ax.axis("off")
        out = FIGS / f"pair_{scene[-16:]}.jpg"
        fig.savefig(out, dpi=92, bbox_inches="tight", pil_kwargs={"quality": 78})
        plt.close(fig)
        print(out.name, out.stat().st_size >> 10, "KB", flush=True)


if __name__ == "__main__":
    main()
