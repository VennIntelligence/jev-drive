"""Lane OT3 mechanism readout (Mac side): closed-loop heading error against the log per decision, for every checkpoint, on M1's set (the
log-straight scenes of the 400 diagnosis scenes with start speed > 2 m/s; decision 205's figure, m1_analysis.growth_table unchanged).
Reads tmp/c1 (C1's pickles: the logged paths, labels only) and tmp/ot3/dec_*.npz (m1_decisions.py on the box).

  .venv/bin/python experiments/alpasim/scripts/ot3_heading.py --name a --recipe P2H10=P2H10-F-s0+P2H10-F-s1 --recipe AP2H10=AP2H10-AB-s0+AP2H10-AB-s1 ...
-> experiments/alpasim/results/ot3/heading_<name>.md / .json, experiments/alpasim/figs/ot3/heading_growth_<name>.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_lib as L  # noqa: E402

TMP = L.ROOT / "tmp/ot3"
RES = L.ROOT / "experiments/alpasim/results/ot3"
FIG = L.ROOT / "experiments/alpasim/figs/ot3"


def table(S, files):
    rows = []
    for f in files:
        Z = np.load(f)
        for n in sorted({k.split("|")[0] for k in Z.files}):
            for s, num in zip(Z[f"{n}|scene"], Z[f"{n}|num"]):
                o, k = S.get(s), int(num[0])
                if o is None or abs(L.turn_deg(o)) >= 5 or k >= len(o["drive"]) or 10 * o["rec"][0]["ego"][4] <= 2:
                    continue
                g = L.gt(o)
                rows.append(dict(driver=n, scene=s, k=k, yaw=np.degrees(L.wrap(num[3] - L.interp_pose(g, o["drive"][k]["now"])[2])),
                                 lat=L.signed_lat(g[:, 1:3], num[1:3])[0]))
    return pd.DataFrame(rows).drop_duplicates(["driver", "scene", "k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True), ap.add_argument("--recipe", action="append", default=[]), ap.add_argument("--base", default="P2H10")
    ap.add_argument("--dec", nargs="*", default=[]), ap.add_argument("--plot", nargs="*", default=[])
    a = ap.parse_args()
    S = L.runs()["SH30"]
    D = table(S, [Path(f) for f in a.dec] or sorted(TMP.glob("dec_*.npz")))
    rec = {r.split("=")[0]: r.split("=")[1].split("+") for r in a.recipe}
    rec = {g: [x for x in v if x in set(D.driver)] for g, v in rec.items()}
    rec = {g: v for g, v in rec.items() if v}
    common = set.intersection(*(set(D[D.driver == x].scene) for v in rec.values() for x in v))
    D = D[D.scene.isin(common)]
    sd = D.pivot_table(index="k", columns="driver", values="yaw", aggfunc="std")
    mu = D.pivot_table(index="k", columns="driver", values="yaw", aggfunc="mean")
    la = D.pivot_table(index="k", columns="driver", values="lat", aggfunc=lambda x: x.abs().mean())
    # log-cluster bootstrap of the sd at decision 9 (recipe = mean of the seeds' sd), paired against the base recipe
    d9 = D[D.k == 9].pivot_table(index="scene", columns="driver", values="yaw").dropna()
    logs = np.array(["_".join(s.split("_")[:2]) for s in d9.index])
    u, inv = np.unique(logs, return_inverse=True)
    rng = np.random.default_rng(0)
    draws = [np.concatenate([np.flatnonzero(inv == c) for c in rng.integers(0, len(u), len(u))]) for _ in range(2000)]
    rsd = lambda g, idx=None: float(np.mean([d9[x].to_numpy()[idx if idx is not None else slice(None)].std(ddof=1) for x in rec[g]]))  # noqa: E731
    out = {"scenes": int(len(common)), "logs": int(len(u)), "recipes": {}}
    T = [f"Heading error of the driven ego against the log (deg) on {len(common)} log-straight scenes of the 400 diagnosis scenes (start speed > 2 m/s, {len(u)} logs); "
         "recipe = mean of the two seeds' statistics; CI: bootstrap over logs, 2 000 draws.", "",
         "| recipe | sd at decision 2 | 3 | 5 | 7 | 9 [95% CI] | ratio to " + a.base + " at 9 [95% CI] | mean at 9 (negative = right) | mean abs lateral offset at 9 (m) |",
         "|:--|--:|--:|--:|--:|:--|:--|--:|--:|"]
    for g, v in rec.items():
        b = np.array([rsd(g, i) for i in draws])
        r = dict(sd={int(k): float(sd.loc[k, v].mean()) for k in sd.index}, sd9=[rsd(g), *np.percentile(b, [2.5, 97.5])], mean9=float(mu.loc[9, v].mean()),
                 lat9=float(la.loc[9, v].mean()), per_seed_sd9={x: float(sd.loc[9, x]) for x in v})
        if a.base in rec:
            rb = np.array([rsd(g, i) / rsd(a.base, i) for i in draws])
            r["ratio9"] = [rsd(g) / rsd(a.base), *np.percentile(rb, [2.5, 97.5])]
        out["recipes"][g] = r
        T.append(f"| {g} ({' / '.join(f'{sd.loc[9, x]:.2f}' for x in v)}) | " + " | ".join(f"{r['sd'][k]:.2f}" for k in (2, 3, 5, 7)) + f" | {r['sd9'][0]:.2f} [{r['sd9'][1]:.2f}, {r['sd9'][2]:.2f}] | "
                 + (f"{r['ratio9'][0]:.2f} [{r['ratio9'][1]:.2f}, {r['ratio9'][2]:.2f}]" if "ratio9" in r else "") + f" | {r['mean9']:+.2f} | {r['lat9']:.2f} |")
    RES.mkdir(parents=True, exist_ok=True), FIG.mkdir(parents=True, exist_ok=True)
    (RES / f"heading_{a.name}.md").write_text("\n".join(T) + "\n")
    (RES / f"heading_{a.name}.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(T))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(L.ROOT / "research"))
    import plot_style as PS
    PS.apply()
    fig, ax = plt.subplots(1, 2, figsize=(6.875, 2.6))
    for g in (a.plot or list(rec)):
        v = rec[g]
        col = "0.5" if g == a.base else None
        ax[0].plot(0.5 * sd.index, sd[v].mean(1), marker="o", ms=3, label=g, color=col)
        ax[1].plot(0.5 * mu.index, mu[v].mean(1), marker="o", ms=3, color=col)
    ax[0].set_xlabel("scene time (s)"), ax[0].set_ylabel("heading error vs log, sd (deg)"), ax[0].legend(frameon=False, fontsize=6)
    ax[1].set_xlabel("scene time (s)"), ax[1].set_ylabel("heading error vs log, mean (deg)"), ax[1].axhline(0, color="0.8", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / f"heading_growth_{a.name}.png", dpi=300)


if __name__ == "__main__":
    main()
