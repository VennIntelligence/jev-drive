"""BEV panels of B2D junction turns, shipped vs fine-tuned arms (25-turn set of decision 127; units from experiments/op_guard's b2d_turns).

    .venv/bin/python experiments/op_route_ft/scripts/rft_panels.py --arms shipped rc-ctl-s0 rc-bear-s0 [--pick 10255:0,15102:0,...]
    -> experiments/op_route_ft/figs/b2d_turn_panels.png (2 x 2 panels; default pick: turns the main arm took and shipped did not, then choice turns)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402
import junction_forced_report as F  # noqa: E402
import junction_rig122_report as J  # noqa: E402
import junction_cl_report as R  # noqa: E402

COL = {"shipped": "#888888", "rc-ctl-s0": "#1f77b4", "rc-bear-s0": "#d62728", "rc-poly-s0": "#2ca02c", "rc-all-s0": "#9467bd"}
HALF = 1.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["shipped", "rc-ctl-s0", "rc-bear-s0"])
    ap.add_argument("--pick", default="")
    ap.add_argument("--out", default=str(REPO / "experiments/op_route_ft/figs/b2d_turn_panels.png"))
    a = ap.parse_args()
    lab = F.load_labels()
    traj, rows, geo = {}, {}, {}
    for arm in a.arms:
        dirs = C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED)
        for rid in C.TURN_ROUTES:
            att, _ = C.attempt_of(dirs, rid)
            if att is None or not (att / "route.json").exists():
                continue
            g = geo.setdefault(rid, J.route_geometry(att, rid, lab))
            s = J.score_attempt(rid, arm, att, g, lab)
            if s is None:
                continue
            for r in s["rows"]:
                rows[(rid, r["turn"], arm)] = r
            for mi, tr in s["traj"].items():
                traj[(rid, mi, arm)] = tr
    main_arm = a.arms[-1]
    if a.pick:
        pick = [(r, int(t)) for r, t in (x.split(":") for x in a.pick.split(","))]
    else:
        keys = sorted({(k[0], k[1]) for k in rows})
        won = [k for k in keys if rows.get((*k, main_arm), {}).get("branch") == "yes" and rows.get((*k, "shipped"), {}).get("branch") != "yes"]
        rest = [k for k in keys if k not in won and lab[k]["forced"] == "0" and rows.get((*k, main_arm), {}).get("entered")]
        pick = (won + rest)[:4]
    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    for ax, (rid, mi) in zip(axs.ravel(), pick):
        D, gd, psiD, turns = geo[rid]
        T = next(x for x in turns if x["mi"] == mi)
        a0, a1 = max(T["it0"] - 100, 0), min(T["i1"] + 80, len(D))
        Dp = D[a0:a1]
        tg = np.gradient(Dp, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nr = np.stack([-tg[:, 1], tg[:, 0]], -1)
        ax.fill(*np.vstack([Dp + HALF * nr, (Dp - HALF * nr)[::-1]]).T, color="#dddddd", zorder=0, label="route lane +-1.75 m")
        txt = []
        for arm in a.arms:
            if (rid, mi, arm) not in traj:
                continue
            tr, e = traj[(rid, mi, arm)]
            near, dev = R.project(tr, D)
            sel = (near >= a0) & (near <= a1 - 1) & (dev < 30)
            ax.plot(tr[sel, 0], tr[sel, 1], color=COL.get(arm, "k"), lw=2.0, zorder=3, label=arm)
            z = rows[(rid, mi, arm)]
            txt.append("%-11s %-5s peak %5.1f m  head %.2f / need %.2f 1/m  hits %d" % (arm, e["branch"], e["peak"] if np.isfinite(e["peak"]) else np.nan,
                                                                                    z["head_pk"], z["need"], z["coll"]))
        L = lab[(rid, mi)]
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("%s route %s turn %d: %+.0f deg, R_min %.0f m, %s" % ("FORCED" if L["forced"] == "1" else "choice", rid, mi, T["angle"], T["rmin"], L["kind"]),
                     fontsize=9)
        ax.grid(alpha=.25)
        ax.text(0.0, -0.08, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="top", family="monospace")
    h, lb = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=9)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100)
    print("wrote", a.out, pick)


if __name__ == "__main__":
    main()
