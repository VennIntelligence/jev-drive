"""Figures of research/alpasim-collisions/index.html, from the committed COL1 tables only.
  python research/alpasim-collisions/make_figs.py            (numpy, matplotlib; run from the repo root)
fig1_taxonomy   struck-object class of every at-fault collision, nuPlan public scenes (P2H10-F, both seeds) and PAI scenes
fig2_handover   PAI: first-0.5 s speed of the served and the shipped plan against the ego's speed just before the hand-over
fig3_lead       PAI collisions with the struck vehicle in the ego's own corridor and the lead head on it: the simulator's gap and the shipped lead head's distance against time to impact
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "research"))
import plot_style as S  # noqa: E402

T = R / "experiments/alpasim/results/collisions"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
S.apply()
rows = lambda f: list(csv.DictReader(open(T / f)))  # noqa: E731
C = S.PALETTE


def fig1():
    nu = [r for r in rows("nuplan_cases.csv") if r.get("driver", r.get("set", "")).startswith("P2H10-F")]
    pai = [r for r in rows("pai_cases.csv") if r["flag"] == "collision_at_fault"]
    kn = next(k for k in ("kind", "cls") if k in nu[0])
    ln = next(k for k in ("lat_off", "lat", "lat_m") if k in nu[0])
    fig, ax = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.5), gridspec_kw=dict(wspace=0.95, left=0.25, right=0.985, bottom=0.17, top=0.9))
    for a, data, key, off, thr, title in ((ax[0], nu, kn, lambda r: abs(float(r[ln])), 0.5, f"nuPlan public scenes, P2H10-F s0 + s1 (n = {len(nu)})"),
                                          (ax[1], pai, "cls", lambda r: abs(float(r["ego_lat_evt"])), 1.0, f"PAI scenes, P2H10-F-s0 (n = {len(pai)})")):
        cnt = Counter(r[key].split(" (VEHICLE)")[0] for r in data)
        offc = Counter(r[key].split(" (VEHICLE)")[0] for r in data if off(r) >= thr)
        names = [k for k, _ in cnt.most_common()][::-1]
        a.barh(names, [cnt[k] for k in names], color=S.BASELINE, label="on the logged path")
        a.barh(names, [offc[k] for k in names], color=C["vermillion"], label=f"off the logged path by >= {thr} m")
        a.set_xlabel("at-fault collision rollouts"), S.panel(a, title), a.grid(axis="y", visible=False)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].legend(loc="lower right", fontsize=7)
    print(S.save(fig, OUT / "fig1_taxonomy"))


def fig2():
    r = [x for x in rows("pai_cases.csv") if x.get("v_handover")]
    v = np.array([float(x["v_handover"]) for x in r])
    spin = np.array([x["scene"] in SPIN for x in r])
    fig, ax = plt.subplots(figsize=(S.SINGLE_COLUMN_IN * 1.35, 2.6), gridspec_kw=dict(left=0.15, right=0.98, bottom=0.17, top=0.95))
    ok = v >= 2.0
    for k, col, lab in (("p0_v05", C["blue"], "shipped policy, same tokens"), ("ft_v05", C["green"], "served checkpoint")):
        y = 100 * (np.array([float(x[k]) for x in r]) / np.maximum(v, 0.5) - 1)
        ax.scatter(v[ok], y[ok], s=14, color=col, label=lab, zorder=3)
        if k == "ft_v05":
            ax.scatter(v[ok & spin], y[ok & spin], s=70, facecolors="none", edgecolors=C["vermillion"], linewidths=1.0, zorder=4, label="spin at the hand-over")
    S.zero_line(ax)
    ax.set_xlabel("ego speed just before the hand-over (m/s)"), ax.set_ylabel("plan speed over the first 0.5 s\nagainst the ego's speed (%)")
    ax.legend(loc="upper right", fontsize=7)
    print(S.save(fig, OUT / "fig2_handover"))


def fig3():
    keep = {x["scene"] for x in rows("pai_cases.csv") if x.get("ahead_s") and float(x["ahead_s"]) >= 2 and float(x["ahead_hit_p0"]) >= 0.7}
    r = [x for x in rows("pai_lead_classA.csv") if x["scene"] in keep]
    sc = sorted(keep)
    fig, ax = plt.subplots(1, len(sc), figsize=(S.DOUBLE_COLUMN_IN, 2.1), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.07, right=0.99, bottom=0.2, top=0.88))
    for a, s in zip(np.atleast_1d(ax), sc):
        q = [x for x in r if x["scene"] == s and float(x["t_to_impact"]) >= -10]
        t, g, d, p = (np.array([float(x[k]) for x in q]) for k in ("t_to_impact", "gap", "d_p0", "p_p0"))
        a.plot(t, g, color=C["vermillion"], label="gap (simulator)")
        a.plot(t[p >= 0.5], d[p >= 0.5], ".", ms=2.5, color=C["blue"], label="lead head (p >= 0.5)")
        a.plot(t[p < 0.5], d[p < 0.5], ".", ms=2.5, color=S.BASELINE, label="lead head (p < 0.5)")
        a.set_ylim(-2, 62), a.set_xlabel("s to impact"), S.panel(a, s)
    np.atleast_1d(ax)[0].set_ylabel("m"), np.atleast_1d(ax)[0].legend(loc="upper left", fontsize=6.5)
    print(S.save(fig, OUT / "fig3_lead"))


SPIN = set(sys.argv[1:]) or {"4f779a92", "05f35348", "dfb0277e", "b45734f4"}     # hand-over spins (pai_arm_v1.md)
if __name__ == "__main__":
    fig1(), fig2(), fig3()
    for f in OUT.glob("*.pdf"):
        f.unlink()                                             # PDFs are not committed (research/README.md)
