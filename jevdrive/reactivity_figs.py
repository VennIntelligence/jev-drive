"""Figures for the reactivity program (todos/2026-09-25-reactivity-program.md), from finished run dirs.

  d0   hazard-probe AUC per scope for openpilot `temporal` / `vision` / `hidden` (route-bootstrap CI), Qwen `L18_last`
       as the grey reference; and the paired AUC delta of each vision-level tap against its own model's `temporal`
  mc   directional flip rate per scope (pedestrian, cut-in, pooled) for every M-C arm, with route-bootstrap CIs

  python -m jevdrive.reactivity_figs d0 --run-dir <p5_pairs/exam-d0/...> --out research/figs
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .p4_carla import _style

SCOPES = ["pedestrian", "DynamicObjectCrossing", "ParkingCrossingPedestrian", "PedestrianCrossing",
          "VehicleTurningRoutePedestrian", "HighwayCutIn", "StaticCutIn", "ParkingCutIn", "hazard pooled"]
SHORT = {"pedestrian": "Pedestrian (4 fam.)", "DynamicObjectCrossing": "DynObjCrossing",
         "ParkingCrossingPedestrian": "ParkingPedestrian", "PedestrianCrossing": "PedCrossing",
         "VehicleTurningRoutePedestrian": "TurnPedestrian", "hazard pooled": "All hazard"}
TAPS = {"op-cinque temporal": ("Cinque temporal", "#0072B2"), "op-cinque vision": ("Cinque vision", "#56B4E9"),
        "op-cinque hidden": ("Cinque hidden", "#009E73"), "op-lebowski temporal": ("Lebowski temporal", "#D55E00"),
        "op-lebowski vision": ("Lebowski vision", "#E69F00"), "op-lebowski hidden": ("Lebowski hidden", "#CC79A7")}


def d0(run: Path, out: Path):
    import matplotlib.pyplot as plt
    ps = _style()
    s = pd.read_csv(run / "probe_auc_paired_scopes.csv")
    scopes = [c for c in SCOPES if c in set(s.scope)]
    x = np.arange(len(scopes))
    fig, axs = plt.subplots(2, 1, figsize=(ps.DOUBLE_COLUMN_IN, 3.9), sharex=True)
    # (a) AUC per tap, CI from the tap's own bootstrap (rows where it is the compared tap)
    ax = axs[0]
    taps = [k for k in TAPS if k in set(s.tap)]
    w = 0.8 / len(taps)
    for j, k in enumerate(taps):
        d = s[s.tap == k].drop_duplicates("scope").set_index("scope").reindex(scopes)
        xi = x + (j - (len(taps) - 1) / 2) * w
        ax.errorbar(xi, d.auc, yerr=[d.auc - d.auc_lo, d.auc_hi - d.auc], fmt="o", color=TAPS[k][1], ms=2.5,
                    elinewidth=0.6, capsize=1.0, label=TAPS[k][0])
    q = s[s.vs == "L18_last"].drop_duplicates("scope").set_index("scope").reindex(scopes)
    ax.scatter(x, q.auc_vs, marker="_", s=120, color=ps.BASELINE, linewidths=1.4, label="Qwen L18_last", zorder=3)
    ax.axhline(0.5, color=ps.BASELINE, linewidth=0.5, linestyle=":")
    ax.axhline(0.6, color=ps.BASELINE, linewidth=0.5, linestyle="--")
    ax.set_ylabel(r"Hazard probe AUC (x$^+$ vs x$^-$)")
    ax.set_ylim(0.42, 1.0)
    ax.legend(ncol=4, fontsize=6.5, loc="upper left")
    ps.panel(ax, "(a)")
    # (b) paired delta against the same model's temporal
    ax = axs[1]
    cmp = [k for k in taps if not k.endswith("temporal")]
    w = 0.8 / max(len(cmp), 1)
    for j, k in enumerate(cmp):
        ref = k.rsplit(" ", 1)[0] + " temporal"
        d = s[(s.tap == k) & (s.vs == ref)].set_index("scope").reindex(scopes)
        xi = x + (j - (len(cmp) - 1) / 2) * w
        ax.errorbar(xi, d.delta, yerr=[d.delta - d.lo, d.hi - d.delta], fmt="o", color=TAPS[k][1], ms=2.5,
                    elinewidth=0.6, capsize=1.0, label=f"{TAPS[k][0]} $-$ temporal")
    ps.zero_line(ax)
    ax.set_ylabel(r"$\Delta$AUC vs own temporal")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(c, c) for c in scopes], rotation=30, ha="right", fontsize=6.5)
    ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    ps.panel(ax, "(b)")
    fig.tight_layout(pad=0.3)
    ps.save(fig, out / "reactivity-d0-vision-probe")
    plt.close(fig)


MC_SCOPES = ["pedestrian", "cut-in", "pooled"]


def mc(run: Path, out: Path, model: str = "cinque"):
    import matplotlib.pyplot as plt
    ps = _style()
    c = pd.read_csv(run / "criteria.csv")
    fl = pd.read_csv(run / "flip_rates.csv")
    arms = [a for a in c.arm if a.endswith(f"[{model}]")]
    names = {f"prior [{model}]": ("prior (ridge_late temporal)", ps.BASELINE),
             f"M-C pair [{model}]": ("pair, dual stream", "#D55E00"), f"M-C pair qwen [{model}]": ("pair, Qwen only", "#E69F00"),
             f"M-C pair op [{model}]": ("pair, openpilot only", "#56B4E9"), f"M-C hard [{model}]": ("hard-example reweight", "#009E73"),
             f"M-C uniform [{model}]": ("uniform imitation", "#0072B2"), f"M-C pair (mu=0) [{model}]": ("pair, $\\mu$=0", "#CC79A7")}
    arms = [a for a in names if a in arms]
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 2.4))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        r = c[c.arm == a].iloc[0]
        p = fl[(fl.examinee == a) & (fl.scope == "pooled")].iloc[0]
        vals = [(r.ped_flip, r.ped_lo, r.ped_hi), (r.cutin_flip, np.nan, np.nan), (p.flip_rate, p.flip_lo, p.flip_hi)]
        xi = np.arange(len(MC_SCOPES)) + (j - (len(arms) - 1) / 2) * w
        v = np.array([u[0] for u in vals])
        ax.bar(xi, v, w, color=names[a][1], label=names[a][0], linewidth=0)
        lo, hi = np.array([u[1] for u in vals]), np.array([u[2] for u in vals])
        ok = ~np.isnan(lo)
        ax.errorbar(xi[ok], v[ok], yerr=[v[ok] - lo[ok], hi[ok] - v[ok]], fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2)
        ax.plot(xi[2] + w * 0.3, r.null_ff_oos, marker="v", color="#333333", ms=2.5)
    ax.set_xticks(np.arange(len(MC_SCOPES)))
    ax.set_xticklabels(["Pedestrian families", "Cut-in families", "Pooled (P5 v0)"])
    ax.set_ylabel("Directional flip rate")
    ax.axhline(0.2, color=ps.BASELINE, linewidth=0.5, linestyle=":")
    ax.set_ylim(0, 1.05)
    ps.bars(ax)
    ax.legend(ncol=4, fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    ps.save(fig, out / f"reactivity-mc-flips-{model}")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fig", choices=("d0", "mc"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="research/figs")
    ap.add_argument("--model", default="cinque")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d0(Path(a.run_dir), out) if a.fig == "d0" else mc(Path(a.run_dir), out, a.model)


if __name__ == "__main__":
    main()
