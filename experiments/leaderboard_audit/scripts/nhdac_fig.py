"""navhard DAC attribution, step 3 (CPU, navsim2 env): review figure, representative shipped-model failures per primary cause.

Picks per cause (shipped Cinque): stage 2 first, one token per log, the two whose maximum overshoot is nearest the cause's median
(picked without looking at the images or scores). Renders each with the loss-budget renderer (lbx_nav_render.render: nuPlan
CAM_F0 + BEV + openpilot road / wide inputs with the model's road edges) and keeps the frame just after the first departure;
the figure stacks them, two per cause.

  navsim2 env:  nhdac_fig.py [--procs 18]   -> $DATA_DIR/runs/leaderboard_audit/navhard_dac/fig/{*.gif, navhard_dac_review.jpg}
"""
import argparse
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/leaderboard_audit/scripts"), str(REPO / "experiments/skill_pack/scripts")]
import lbx_nav_render as X  # noqa: E402
import offroad_lib as L  # noqa: E402

DD = L.D / "runs/leaderboard_audit/navhard_dac"
CAUSES = ["map_narrow", "tracker_lag", "early_start_offset", "wrong_direction", "other", "over_turn", "under_turn", "too_fast", "lane_change"]
LABEL = dict(map_narrow="M map narrow: off-polygon corners all on nuPlan generic drivable area", tracker_lag="L tracker lag: plan itself inside, LQR replay leaves",
             early_start_offset="E displaced stage-2 start, departure <= 1.5 s", wrong_direction="W wrong direction (decision 88)",
             other="R other (no rule fires)", over_turn="O over-turn / turn on a straight", under_turn="U needed turn not taken",
             too_fast="F too fast: same path at reference speed passes", lane_change="C lane change / merge")
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "ego_progress", "time_to_collision_within_bound"]


def picks():
    d = pd.read_csv(DD / "nhdac_tokens.csv", index_col=0)
    d = d[(d.arm == "native") & (d.primary.fillna("") != "")]
    S = pickle.load(open(DD / "scores.pkl", "rb"))
    idx = {e["token"]: e for e in L.index()}
    out, used = [], set()
    for c in CAUSES:
        g = d[d.primary == c].copy()
        g["log"] = [S["native"].loc[t, "log_name"] for t in g.index]
        g["k"] = (g.d_worst - g.d_worst.median()).abs() + np.where(g.stage == "two", 0, 100)
        n = 0
        for t, r in g.sort_values("k").iterrows():
            if r.log in used:
                continue
            used.add(r.log)
            out.append(dict(cls=c, row=CAUSES.index(c), col=n, token=t, cmd=L.CMDS[int(r.cmd)], v0=float(np.linalg.norm(idx[t]["vel"][-1])), first=int(r["first"]),
                            info=f"first exit {r['first'] / 10:.1f} s, max {r.d_worst:.2f} m, ref DAC {r.ref_dac:.0f}, feasible arc {'yes' if r.feasible else 'no'}, "
                                 f"plan-only inside {'yes' if r.raw_ok else 'no'}, rot0 inside {'yes' if r.rot0_ok else 'no'}",
                            scores={a: {s: float(S[a].loc[t, s]) for s in ["score"] + SUBS} for a in ("native", "best", "ref")}))
            n += 1
            if n == 2:
                break
    return out


def one(p):
    gif, _, _ = X.render(p)
    im = Image.open(gif)
    k = min(10 + int(np.ceil(p["first"] / 2)) + 1, im.n_frames - 2)     # 10 history frames, then plan frames every 0.2 s
    im.seek(k)
    return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    out = DD / "fig"
    out.mkdir(parents=True, exist_ok=True)
    X.OUT = out
    X.NAMES.update({c: LABEL[c].split(":")[0] for c in CAUSES})
    P = picks()
    with mp.get_context("fork").Pool(min(a.procs, len(P)), initializer=X.init, initargs=("navhard",)) as pool:
        ims = pool.map(one, P)
    W, H, HD = 900, 601, 40
    rows = len(CAUSES)
    fig = Image.new("RGB", (2 * W, rows * (H + HD)), (255, 255, 255))
    dr = ImageDraw.Draw(fig)
    for i, (p, im) in enumerate(zip(P, ims)):
        x, y = p["col"] * W, p["row"] * (H + HD)
        dr.text((x + 6, y + 3), LABEL[p["cls"]], fill=(0, 0, 0), font=X.FONT)
        dr.text((x + 6, y + 22), f"{p['token']}: {p['info']}", fill=(60, 60, 60), font=X.FONT_S)
        fig.paste(im.resize((W, H), Image.LANCZOS), (x, y + HD))
    fig.save(out / "navhard_dac_review.jpg", quality=80)
    for p in P:
        print(p["cls"], p["token"], p["info"])


if __name__ == "__main__":
    main()
