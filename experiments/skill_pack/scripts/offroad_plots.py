"""Static figures of the navhard off-road diagnosis (navsim2 env). Writes PNGs to <results>/figs.
  q1_class.png          plan end lateral vs PDM reference end lateral, stage 1 / stage 2, opposite-side points marked
  q2_departure.png      first-departure time, outside distance, class counts (native / N4)
  ex_<class>.png        three failures per class (native blue, N4 orange, PDM reference green dashed) over the map
  q3_road_edge.png      signed road-edge error at the departing corner, edge-right / wrong shares
  q4_submetric.png      combined-score gain if a term were perfect, N4 vs native
  q2_counterfactuals.png  stage-2 DAC compliance and combined EPDMS per counterfactual
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

RES = REPO / "experiments/skill_pack/results/navhard-offroad"
FIG = RES / "figs"
FIG.mkdir(parents=True, exist_ok=True)
C = dict(native="#1f77b4", n4="#e8710a", ref="#1b9e77", grey="#8a949e")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})


def q1():
    d = pd.read_pickle(L.OUT / "token_table.pkl")
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    for i, st in enumerate(("one", "two")):
        for j, m in enumerate(("native", "n4")):
            a = axs[i, j]
            g = d[d.stage == st]
            opp = (g[f"{m}_end_y"] * g.ref_end_y < 0) & (g[f"{m}_end_y"].abs() > 1) & (g.ref_end_y.abs() > 1) & ((g[f"{m}_end_y"] - g.ref_end_y).abs() > 2)
            a.scatter(g.ref_end_y[~opp], g[f"{m}_end_y"][~opp], s=4, c=C["grey"], alpha=.35, label="other")
            a.scatter(g.ref_end_y[opp], g[f"{m}_end_y"][opp], s=9, c=C[m], label=f"opposite side ({opp.sum()} / {len(g)})")
            a.plot([-30, 30], [-30, 30], "k:", lw=.8)
            a.axhline(0, c="k", lw=.5); a.axvline(0, c="k", lw=.5)
            a.set_xlim(-30, 30); a.set_ylim(-30, 30)
            a.set_title(f"{m}, stage {'1' if st == 'one' else '2'}")
            a.set_xlabel("PDM reference lateral y at 4 s (m, left +)"); a.set_ylabel("plan lateral y at 4 s (m)")
            a.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "q1_class.png", dpi=110); plt.close(fig)


def q2():
    d = pd.read_pickle(L.OUT / "stage2_table.pkl")
    cl = pd.read_csv(RES / "tables/q2_classes.csv")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    bins = np.arange(0, 4.1, 0.5)
    for m in ("native", "n4"):
        f = d[(d[f"{m}_dac"] == 0)]
        axs[0].hist(f[f"{m}_first"] * 0.1, bins=bins, histtype="step", lw=2, color=C[m], label=m)
        axs[1].hist(np.log10(f[f"{m}_d_worst"] * 100 + 1), bins=30, histtype="step", lw=2, color=C[m], label=f"{m} worst")
        axs[1].hist(np.log10(f[f"{m}_d_first"] * 100 + 1), bins=30, histtype="step", lw=1.2, ls="--", color=C[m], label=f"{m} at first departure")
    axs[0].set_xlabel("first departure time (s)"); axs[0].set_ylabel("stage-2 DAC failures"); axs[0].legend()
    axs[1].set_xlabel("log10(1 + outside distance in cm)"); axs[1].legend(fontsize=7)
    w = 0.38
    for k, m in enumerate(("native", "n4")):
        x = np.arange(len(L_CLASSES)) + (k - .5) * w
        v = [int(cl[(cl.model == m) & (cl.cls == c)].n.iloc[0]) for c in L_CLASSES]
        axs[2].bar(x, v, w, color=C[m], label=m)
        for xi, vi in zip(x, v):
            axs[2].text(xi, vi + 8, str(vi), ha="center", fontsize=7)
    axs[2].set_xticks(range(len(L_CLASSES))); axs[2].set_xticklabels([c.replace("_", "\n") for c in L_CLASSES], fontsize=8)
    axs[2].legend(); axs[2].set_title("failure classes (plan file order)")
    fig.tight_layout(); fig.savefig(FIG / "q2_departure.png", dpi=110); plt.close(fig)


L_CLASSES = ["start_outside", "wrong_direction", "early_clip", "junction", "late_undershoot", "other"]


def examples():
    F = pickle.load(open(L.OUT / "features.pkl", "rb"))
    d = pd.read_pickle(L.OUT / "stage2_table.pkl")
    cp = L.cache_paths()
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array  # noqa: F401
    for cls in L_CLASSES[1:]:
        g = d[(d.native_class == cls)].sort_values("native_d_worst")
        if len(g) < 3:
            continue
        pick = [g.iloc[int(len(g) * q)] for q in (.25, .5, .75)]
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
        for a, r in zip(axs, pick):
            mc = L.load_cache(cp[r.token])
            am = mc.drivable_area_map
            o = np.array(mc.ego_state.rear_axle.serialize())
            area = set(am.get_indices_of_map_type([S.ROADBLOCK, S.INTERSECTION, S.DRIVABLE_AREA, S.CARPARK_AREA]))
            routeset = set(mc.route_lane_ids)
            for k in range(len(am.tokens)):
                gm = am._geometries[k]
                if gm.distance(mc.ego_state.car_footprint.oriented_box.geometry) > 45:
                    continue
                for p in (list(gm.geoms) if hasattr(gm, "geoms") else [gm]):
                    e = L.to_ego(mc, np.c_[np.asarray(p.exterior.coords), np.zeros(len(p.exterior.coords))])
                    kind = "area" if k in area else "lane"
                    a.fill(e[:, 1], e[:, 0], fc="#dfeee6" if am.tokens[k] in routeset else "#eef1f4", ec="#b0bac4", lw=.4, zorder=1 if kind == "area" else 0)
            ref = F[r.token]["ref"]
            a.plot(ref[:, 1], ref[:, 0], "--", c=C["ref"], lw=2, label="PDM ref", zorder=3)
            for m in ("native", "n4"):
                f = F[r.token]["feat"][m]
                s = f["sim_ego"]
                a.plot(s[:, 1], s[:, 0], c=C[m], lw=2.2, label=f"{m} DAC={f['dac']}", zorder=4)
                if f["first"] is not None:
                    a.plot(s[f["first"], 1], s[f["first"], 0], "x", c=C[m], ms=9, mew=2.5, zorder=5)
            a.plot(0, 0, "ko", zorder=6)
            lim = max(12, np.abs(np.r_[ref[:, :2].ravel(), F[r.token]["feat"]["native"]["sim_ego"][:, :2].ravel()]).max() * 1.05)
            a.set_xlim(lim * .8, -lim * .8); a.set_ylim(-6, lim * 1.4)
            a.set_aspect("equal")
            a.set_title(f"{r.token[:8]} {r.cmd} v={r.speed:.1f} {r['map'].split('-')[-1]}\nnative first dep {r.native_first * .1:.1f}s, outside {r.native_d_worst * 100:.0f} cm", fontsize=9)
            a.legend(fontsize=7, loc="lower right")
        fig.suptitle(f"class {cls}: native stage-2 failures at the 25 / 50 / 75 % of outside distance (x = first departure)")
        fig.tight_layout(); fig.savefig(FIG / f"ex_{cls}.png", dpi=95); plt.close(fig)


def q3():
    tab = RES / "tables/q3_road_edge_split.csv"
    D = pd.read_pickle(L.OUT / "roadedge_table.pkl")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for m in ("native", "n4"):
        v = D.loc[(D[f"{m}_dac"] == 0) & D[f"{m}_dep"].fillna(False).astype(bool), f"{m}_edge_err"].clip(-8, 8)
        axs[0].hist(v, bins=np.arange(-8, 8.01, .5), histtype="step", lw=2, color=C[m], label=f"{m} (n={len(v)})")
    axs[0].axvspan(-1, 1, color=C["ref"], alpha=.12)
    axs[0].set_xlabel("model road edge minus real boundary at the departing corner (m); > 0: edge beyond the boundary")
    axs[0].legend(); axs[0].set_title("failures: road-edge error at departure (green: |err| <= 1 m)")
    t = pd.read_csv(tab)
    cats = ["right_plan_crosses", "right_plan_inside", "too_far_plan_crosses", "too_far_plan_inside", "too_near_plan_crosses", "too_near_plan_inside"]
    w = .38
    for k, m in enumerate(("native", "n4")):
        v = [float(t[(t.model == m) & (t.category == c)].share.iloc[0]) for c in cats]
        x = np.arange(len(cats)) + (k - .5) * w
        axs[1].bar(x, v, w, color=C[m], label=m)
        for xi, vi in zip(x, v):
            axs[1].text(xi, vi + .01, f"{vi:.0%}", ha="center", fontsize=7)
    axs[1].set_xticks(range(len(cats))); axs[1].set_xticklabels([c.replace("_plan_", "\nplan ") for c in cats], fontsize=8)
    axs[1].set_title("share of failures"); axs[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "q3_road_edge.png", dpi=110); plt.close(fig)


def q4():
    t = pd.read_csv(RES / "tables/q4_submetric_loss.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    terms = list(t[t.model == "n4"].term)
    w = .38
    for k, m in enumerate(("n4", "native")):
        g = t[t.model == m].set_index("term").loc[terms]
        ax.bar(np.arange(len(terms)) + (k - .5) * w, g.gain_all, w, color=C[m], label=m)
        for i, v in enumerate(g.gain_all):
            ax.text(i + (k - .5) * w, v + .2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(terms))); ax.set_xticklabels(terms)
    ax.set_ylabel("combined EPDMS points gained if the term were 1 for every token")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "q4_submetric.png", dpi=110); plt.close(fig)


def cf():
    s = json.load(open(L.OUT / "counterfactuals.json"))
    names = ["base", "lag0.3", "lag0.5", "lag1.0", "scale1.25", "scale1.5", "pdm1s"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    for gate, ls in (("all", "-"), ("failures", "--")):
        for m in ("native", "n4"):
            dac = [100 * s[f"{m}/base"]["dac_s2_weighted"]] + [100 * s[f"{m}/{v}/{gate}"]["dac_s2_weighted"] for v in names[1:]]
            sc = [100 * s[f"{m}/base"]["combined"]] + [100 * s[f"{m}/{v}/{gate}"]["combined"] for v in names[1:]]
            axs[0].plot(names, dac, ls, marker="o", c=C[m], label=f"{m}, {gate}")
            axs[1].plot(names, sc, ls, marker="o", c=C[m], label=f"{m}, {gate}")
    axs[0].set_ylabel("stage-2 DAC compliance (%), official weights"); axs[1].set_ylabel("combined EPDMS")
    axs[0].legend(fontsize=8)
    for a in axs:
        a.tick_params(axis="x", labelrotation=30)
    fig.suptitle("privileged counterfactuals (PDM reference / failure label used); solid: all stage-2 tokens modified, dashed: baseline failures only")
    fig.tight_layout(); fig.savefig(FIG / "q2_counterfactuals.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["q1", "q2", "examples", "q3", "q4", "cf"]
    for w in what:
        {"q1": q1, "q2": q2, "examples": examples, "q3": q3, "q4": q4, "cf": cf}[w]()
        print("done", w)
