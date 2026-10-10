"""BODY1 arm 4.3, Amendment 6: figure of the pilot gate and of G3 (e) from results/shape/{pilot_gate,g3e}.json (no model).

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/shape_fig.py     -> experiments/body1/figs/shape/shape_gate.png
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RES, FIG = REPO / "experiments/body1/results/shape", REPO / "experiments/body1/figs/shape"
NM = {"P2H10S-P-s0": "shape-only arm", "P2H10B-Pw3-s0": "all terms, full gradient", "P2H10B-Pw3-noA-s0": "B + C without A", "P2H10B-P-Aon-s0": "A on on-log rows only"}
COL = {"P2H10S-P-s0": "tab:blue", "P2H10B-Pw3-s0": "tab:orange", "P2H10B-Pw3-noA-s0": "tab:green", "P2H10B-P-Aon-s0": "tab:purple"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jevdrive.run import Run
    with Run("body1", "shape-fig") as run:
        P, G = json.loads((RES / "pilot_gate.json").read_text())["tags"], json.loads((RES / "g3e.json").read_text())["lines"]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[0.8, 1.5, 1.3]))
        tags = list(NM)
        for i, t in enumerate(tags):                                  # pilot: relative fall of the two own-plan rates
            for j, q in enumerate(("agent", "bnd")):
                r = P[t][q]
                ax[0].bar(j + (i - 1.5) * 0.2, 100 * r["rel_fall"], 0.19, color=COL[t], label=NM[t] if j == 0 else None)
        ax[0].hlines([30, 25], [-0.45, 0.55], [0.45, 1.45], color="k", ls=":", lw=1)
        ax[0].set_xticks([0, 1], ["agent contact\n(line 30 %)", "boundary\n(line 25 %)"], fontsize=8)
        ax[0].set_ylabel("relative fall against the switch-off pilot (%)"), ax[0].set_title("pilot, hold logs: own-plan contact rates", fontsize=10), ax[0].legend(fontsize=7)
        subs = ["all", "open", "lead", "near obj", "near edge", "contact", "fam log", "fam ot1", "fam yr1", "fam bd4"]
        for i, t in enumerate(tags):                                  # pilot: arc ratios
            v = np.array([P[t]["arc_hold"][s][:3] for s in subs])
            x = np.arange(len(subs)) + (i - 1.5) * 0.2
            ax[1].errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, color=COL[t], lw=1)
        ax[1].axhline(1, color="grey", lw=0.6), ax[1].hlines(0.995, -0.5, 1.5, color="k", ls=":", lw=1)
        ax[1].set_xticks(range(len(subs)), [s.replace("fam ", "") for s in subs], fontsize=8), ax[1].set_ylabel("4 s arc of the own plan / switch-off pilot")
        ax[1].set_title("pilot, hold logs: arc ratio by proximity group and state family (line 0.995 on all, open)", fontsize=10)
        labs = list(G)
        keep = [k for k in labs if "lower bound" not in k]
        for i, k in enumerate(keep):                                  # G3 (e): the arm's two seeds against P2H10B-F
            r = G[k]
            thr = float(k.split(">=")[1])
            for s, (v, m) in enumerate(((r["s0"], "o"), (r["s1"], "s"))):
                ax[2].errorbar(i - 0.12 + 0.12 * s, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=m, ms=4, color="tab:blue", lw=1, label="P2H10S-F (seed 0, 1)" if i == 0 and s == 0 else None)
                ax[2].plot(i + 0.18 + 0.1 * s, r["old"][s], m, ms=4, color="tab:orange", label="P2H10B-F (Amendment 5)" if i == 0 and s == 0 else None)
            ax[2].hlines(thr, i - 0.4, i + 0.4, color="k", ls=":", lw=1)
        ax[2].axhline(1, color="grey", lw=0.6)
        ax[2].set_xticks(range(len(keep)), [k.split(">=")[0].replace("on-log ", "").replace("`", "").strip() for k in keep], rotation=25, ha="right", fontsize=8)
        ax[2].set_title("G3 (e), full scale: arc ratio against P2H10-F (dotted = line)", fontsize=10), ax[2].legend(fontsize=7)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "shape_gate.png", dpi=120)
        run.info(f"-> {FIG / 'shape_gate.png'}")


if __name__ == "__main__":
    main()
