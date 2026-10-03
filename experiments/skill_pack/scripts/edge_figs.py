"""Lane EDGE figures from the committed tables (results/roadedge/) and two overlay crops (edge_overlay.py --scale 1).

Local (any env with matplotlib + PIL):
  python experiments/skill_pack/scripts/edge_figs.py [--crop0 a.png --crop1 b.png]
-> experiments/skill_pack/figs/roadedge_scale.png, roadedge_overlay.png (when the crops are given).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research"))
import plot_style as PS  # noqa: E402

RES = REPO / "experiments/skill_pack/results/roadedge"
FIG = REPO / "experiments/skill_pack/figs"
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#999999")


def scale_fig():
    import matplotlib.pyplot as plt
    PS.apply()
    S = pd.read_csv(RES / "tables/scale_by_x.csv")
    H = pd.read_csv(RES / "tables/height_by_x.csv")
    hj = json.loads((RES / "height.json").read_text())
    sj = json.loads((RES / "scale.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(6.875, 2.1))
    a = ax[0]
    nt = S[S.board == "navtest"]
    a.plot(nt.x, nt.lane_ratio, "o-", c=OI["grey"], ms=3, label="as shipped")
    a.plot(nt.x, nt.lane_ratio_c, "s--", c=OI["blue"], ms=3, label="own image reading on true ground")
    for arm, c, lab in (("hgt_r1", OI["orange"], "warp frames, camera 1.87 m"), ("hgt_r1.44", OI["verm"], "virtual camera 1.30 m")):
        h = H[H.arm == arm]
        a.plot(h.x, h.lane_ratio, "^-", c=c, ms=3, label=lab)
    a.axhline(1, c="k", lw=.6, ls=":")
    a.set(xlabel="distance ahead (m)", ylabel="model / map ego-lane width", ylim=(0.4, 1.1))
    a.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(0, 0.55), frameon=False)
    PS.panel(a, "a")
    a = ax[1]
    for col, c, lab in (("out_L", OI["sky"], "left, as shipped"), ("out_R", OI["blue"], "right, as shipped"),
                        ("out_L_c", OI["purple"], "left, true ground"), ("out_R_c", OI["verm"], "right, true ground")):
        a.plot(nt.x, nt[col], "o-" if "_c" not in col else "s--", c=c, ms=3, label=lab)
    a.axhline(0, c="k", lw=.6, ls=":")
    a.set(xlabel="distance ahead (m)", ylabel="model edge beyond scorer (m)", ylim=(-2.4, 2.4))
    a.legend(fontsize=5.5, loc="upper center", ncol=2, frameon=False)
    PS.panel(a, "b")
    a = ax[2]
    names = ["navtest\nshipped", "subset\ncamera 1.87 m", "subset\ncamera 1.30 m"]
    vals = [sj["plan_speed"]["navtest_one"]["plan_v0_over_logged"], hj["hgt_r1"]["speed_ratio"], hj["hgt_r1.44"]["speed_ratio"]]
    a.bar(range(3), vals, color=[OI["grey"], OI["orange"], OI["verm"]], width=.6)
    a.axhline(1, c="k", lw=.6, ls=":")
    a.set_xticks(range(3), names, fontsize=6.5)
    a.set(ylabel="plan speed at t0 / logged speed", ylim=(0.5, 1.1))
    PS.panel(a, "c")
    fig.tight_layout()
    PS.save(fig, str(FIG / "roadedge_scale"))


def overlay_fig(c0, c1):
    from PIL import Image, ImageDraw
    a, b = Image.open(c0), Image.open(c1)
    W = 1000
    a, b = (im.resize((W, int(im.height * W / im.width))) for im in (a, b))
    out = Image.new("RGB", (W, a.height + b.height + 4), "white")
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height + 4))
    d = ImageDraw.Draw(out)
    for y, t in ((4, "map projected with the ground at the ego origin (z = 0)"), (a.height + 8, "map projected with the ground 0.35 m below the ego origin")):
        d.rectangle([0, y - 2, 470, y + 14], fill="black")
        d.text((4, y), t, fill="white")
    out.convert("P", palette=Image.ADAPTIVE, colors=128).save(FIG / "roadedge_overlay.png", optimize=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop0")
    ap.add_argument("--crop1")
    a = ap.parse_args()
    scale_fig()
    if a.crop0 and a.crop1:
        overlay_fig(a.crop0, a.crop1)
