"""M1 diagnosis tables (decision 205, Mac side): why the "pre-turn shift" of decision 202 happens. Reads C1's pickles (tmp/c1), the M1
pulls in tmp/m1 (replay_full_{sh30,ap2}.pkl from m1_replay.py, dec_*.npz from m1_decisions.py, the navtest plans poses_fullAB.npz and
op_parity's tab.npz) and writes experiments/alpasim/results/m1/tables.md + figs/m1/heading_growth.png. Map and logged paths are privileged:
labels only.

  .venv/bin/python experiments/alpasim/scripts/m1_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_lib as L  # noqa: E402

M1 = L.ROOT / "tmp/m1"
RES = L.ROOT / "experiments/alpasim/results/m1"
FIG = L.ROOT / "experiments/alpasim/figs/m1"
ORDER = ["SH30", "AP2", "OT0", "OT1", "W50", "W0", "W0G", "P2H10", "P2"]
LABEL = {"OT0": "OT30-s0", "OT1": "OT30-s1", "P2": "P2 (no hinge)", "W0": "SH30 + W0", "W50": "SH30 + W50", "W0G": "SH30 + W0G"}


def md(df, fmt="%.2f"):
    return df.to_markdown(floatfmt=fmt.replace("%", ""))


def route_world(o):
    """Lateral position of the route's first waypoint per decision in the frame of the initial logged heading (m)."""
    g = L.gt(o)
    R0 = L.rot(g[0, 3]).T
    return np.array([(R0 @ (np.array(r["anchor"])[:2] + L.rot(r["anchor"][2]) @ np.array(r["route0"]) - g[0, 1:3]))[1] for r in o["rec"]])


def sec_route(R):
    rows = []
    for n, S in R.items():
        for s, o in S.items():
            if abs(L.turn_deg(o)) >= 5:
                continue
            ry, wl = [r["route0"][1] for r in o["rec"]], route_world(o)
            g = L.gt(o)
            yaw = [np.degrees(L.wrap(r["anchor"][2] - L.interp_pose(g, o["drive"][r["k"]]["now"])[2])) for r in o["rec"]]
            rows.append(dict(driver=n, bend="left" if max(ry) > 2 else "right" if min(ry) < -2 else "none", rig=max(ry, key=abs),
                             world=wl.max() - wl.min(), yaw=max(yaw, key=abs), zero=o["summary"]["score"] == 0))
    D = pd.DataFrame(rows)
    T = D.groupby(["driver", "bend"]).agg(scenes=("rig", "size"), rig_shift_median_m=("rig", "median"), world_shift_median_m=("world", "median"),
                                          world_shift_below_1m=("world", lambda x: (x < 1).mean()), ego_yaw_extreme_median_deg=("yaw", "median"),
                                          zeros=("zero", "sum"), zeros_world_below_1m=("world", lambda x: 0)).reset_index()
    T["zeros_world_below_1m"] = [int(((D.driver == a) & (D.bend == b) & D.zero & (D.world < 1)).sum()) for a, b in zip(T.driver, T.bend)]
    return ["## 1. The route does not bend: the ego yaws", "",
            "Log-straight scenes (heading change < 5 deg) of the 400, by decision 202's label (the route's first waypoint moves > 2 m sideways in the rig "
            "frame). `world shift` is the same waypoint's lateral range in the frame of the initial logged heading.", "", md(T.set_index(["driver", "bend"])), ""]


def growth_table(S):
    rows = []
    for f in ("dec_c0b.npz", "dec_s1.npz", "dec_diag.npz"):
        if not (M1 / f).exists():
            continue
        Z = np.load(M1 / f)
        for n in sorted({k.split("|")[0] for k in Z.files}):
            for s, num, pl in zip(Z[f"{n}|scene"], Z[f"{n}|num"], Z[f"{n}|plan"]):
                o, k = S.get(s), int(num[0])
                if o is None or abs(L.turn_deg(o)) >= 5 or k >= len(o["drive"]):
                    continue
                g = L.gt(o)
                gp = L.interp_pose(g, o["drive"][k]["now"])
                rows.append(dict(src=f, driver=n, scene=s, k=k, yaw=np.degrees(L.wrap(num[3] - gp[2])), lat=L.signed_lat(g[:, 1:3], num[1:3])[0],
                                 v0=10 * o["rec"][0]["ego"][4]))
    return pd.DataFrame(rows)


def sec_growth(S):
    D = growth_table(S)
    D = D[D.v0 > 2]
    out = ["## 2. Heading error against the log grows linearly in closed loop", ""]
    for f, title in (("dec_c0b.npz", "all log-straight scenes of the 400 (start speed > 2 m/s)"), ("dec_s1.npz", "log-straight scenes of the S1 set"),
                     ("dec_diag.npz", "all log-straight scenes of the 400, candidates")):
        d = D[D.src == f]
        if d.empty:
            continue
        if f == "dec_s1.npz":
            d = pd.concat([d, D[(D.src == "dec_c0b.npz") & D.scene.isin(set(d.scene))]])
        cols = [c for c in ORDER if c in set(d.driver)]
        for name, fn in (("sd of the heading error, deg", "std"), ("mean heading error, deg (negative = right)", "mean")):
            t = d.pivot_table(index="k", columns="driver", values="yaw", aggfunc=fn)[cols].rename(columns=LABEL)
            out += [f"{title}: {name} ({d.scene.nunique()} scenes)", "", md(t), ""]
        t = d.pivot_table(index="k", columns="driver", values="lat", aggfunc=lambda x: x.abs().mean())[cols].rename(columns=LABEL)
        out += [f"{title}: mean |lateral offset from the logged path|, m", "", md(t), ""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(L.ROOT / "research"))
        import plot_style as PS
        PS.apply()
        d = D[D.src.isin(["dec_c0b.npz", "dec_diag.npz"])].drop_duplicates(["driver", "scene", "k"])
        fig, ax = plt.subplots(1, 2, figsize=(6.875, 2.6))
        for n in [c for c in ("SH30", "AP2", "OT1", "W0G", "P2H10", "P2") if c in set(d.driver)]:
            g = d[d.driver == n].groupby("k").yaw
            ax[0].plot(0.5 * g.std().index, g.std().values, marker="o", ms=3, label=LABEL.get(n, n), color="0.5" if n == "SH30" else None)
            ax[1].plot(0.5 * g.mean().index, g.mean().values, marker="o", ms=3, color="0.5" if n == "SH30" else None)
        ax[0].set_xlabel("scene time (s)"), ax[0].set_ylabel("heading error vs log, sd (deg)"), ax[0].legend(frameon=False, fontsize=7)
        ax[1].set_xlabel("scene time (s)"), ax[1].set_ylabel("heading error vs log, mean (deg)"), ax[1].axhline(0, color="0.8", lw=0.6)
        FIG.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIG / "heading_growth.png", dpi=300)
    except Exception as e:                                              # the tables do not depend on the figure
        print("figure skipped:", repr(e))
    return out


def sec_navsim():
    z, t = np.load(M1 / "poses_fullAB.npz"), np.load(M1 / "tab.npz")
    pose, fut, v = t["pose"], t["fut"], t["speed"]
    turned = -np.degrees(pose[:, 2, 2])
    m = (v > 2) & np.isfinite(fut).all((1, 2))
    st = m & (np.abs(np.degrees(fut[:, 7, 2])) < 5) & (np.abs(np.degrees(pose[:, 0, 2])) < 3)
    rows = []
    for nm, q in (("all moving tokens", m), ("straight tokens", st)):
        for lab, P in (("log future", fut), ("SH30 plan (NAVSIM inputs)", z["SH30_nav"]), ("AP2 plan (NAVSIM inputs)", z["AP2_nav"]), ("AP2 plan (AlpaSim inputs, m = 4)", z["AP2_m4"])):
            d = P[q, 7, 1] - fut[q, 7, 1]
            rows.append({"tokens": f"{nm} ({q.sum()})", "trajectory": lab, "yaw at 0.5 s per deg turned in the last 0.5 s": np.polyfit(turned[q], np.degrees(P[q, 0, 2]), 1)[0],
                         "yaw at 1.0 s per deg": np.polyfit(turned[q], np.degrees(P[q, 1, 2]), 1)[0], "sd of deg turned in the last 0.5 s": turned[q].std(),
                         "4 s lateral error vs log >= 0.7 m": np.nan if P is fut else (np.abs(d) >= 0.7).mean(), "mean 4 s lateral error m": np.nan if P is fut else d.mean()})
    return ["## 3. NAVSIM side: the same yaw continuation, and it is what the logs do", "",
            "navtest, open loop (12 146 tokens; `ap2_offline.py plans`). Straight tokens: |future yaw at 4 s| < 5 deg and |yaw over the past 1.5 s| < 3 deg.", "",
            md(pd.DataFrame(rows).set_index(["tokens", "trajectory"]), "%.3f"), ""]


def sec_replay(R):
    V = ["run", "strF", "arcL", "arcR", "ax0", "S", "L", "R", "strW", "w50", "str", "mir", "mirL"]
    out = ["## 4. Offline replay, one input swapped at a time (log-straight scenes of the 400, start speed > 2 m/s)", ""]
    for n, S in R.items():
        rp = L.load_m1(f"replay_full_{n.lower()}")
        rows, k3 = [], []
        for s, r in rp.items():
            o = S[s]
            if abs(L.turn_deg(o)) >= 5 or 10 * o["rec"][0]["ego"][4] <= 2:
                continue
            for k, P in r["plans"].items():
                if "strF" in P and k >= 2:
                    rows.append(dict(scene=s, k=k, turned=-np.degrees(r["ego"][k][16]), **{f"yaw_{v}": np.degrees(P[v][0, 2]) for v in V if v in P},
                                     **{f"y_{v}": P[v][7, 1] for v in V if v in P}))
                if k == 3 and "navH" in P:
                    k3.append({v: P[v][7, 1] for v in ("run", "navH", "navE", "navHE")} | {f"yaw_{v}": np.degrees(P[v][0, 2]) for v in ("run", "navH", "navE", "navHE")})
        D, K = pd.DataFrame(rows), pd.DataFrame(k3)
        t = []
        for v in V:
            if f"yaw_{v}" not in D:
                continue
            d = D.dropna(subset=[f"yaw_{v}"])
            t.append({"variant": v, "decisions": len(d), "plan yaw at 0.5 s per deg turned in the last 0.5 s": np.polyfit(d.turned, d[f"yaw_{v}"], 1)[0],
                      "sd plan yaw at 0.5 s deg": d[f"yaw_{v}"].std(), "mean plan yaw at 0.5 s deg": d[f"yaw_{v}"].mean(),
                      "mean |change of the 4 s lateral point vs as run| m": (d[f"y_{v}"] - d.y_run).abs().mean(), "4 s lateral >= 0.7 m": (d[f"y_{v}"].abs() >= 0.7).mean()})
        out += [f"{n}: decisions 2-9 of {D.scene.nunique()} scenes that got every variant", "", md(pd.DataFrame(t).set_index("variant"), "%.3f"), ""]
        if len(K):
            t = [{"decision 3 inputs": lab, "scenes": len(K), "4 s lateral >= 0.7 m": (K[v].abs() >= 0.7).mean(), "mean |4 s lateral| m": K[v].abs().mean(),
                  "mean plan yaw at 0.5 s deg": K[f"yaw_{v}"].mean(), "sd plan yaw at 0.5 s deg": K[f"yaw_{v}"].std()}
                 for v, lab in (("run", "rendered frames + simulator ego (as run)"), ("navE", "rendered frames + NAVSIM ego features"),
                                ("navH", "NAVSIM vision tokens + simulator ego"), ("navHE", "NAVSIM tokens + NAVSIM ego"))]
            out += [f"{n}: decision 3 (= the navtest token), vision tokens and ego features swapped", "", md(pd.DataFrame(t).set_index("decision 3 inputs"), "%.3f"), ""]
    return out


def sec_edge(R):
    from shapely.geometry import Point
    M, rows = L.load("map"), []
    for n, S in R.items():
        for s, o in S.items():
            if abs(L.turn_deg(o)) >= 5 or s not in M or len(o["rec"]) < 8 or 10 * o["rec"][2]["ego"][4] <= 2:
                continue
            rd, g = L.road(M[s]), L.gt(o)
            a = np.array(o["rec"][2]["anchor"])
            p, nl = a[:2] + 3.0 * np.array([np.cos(a[2]), np.sin(a[2])]), np.array([-np.sin(a[2]), np.cos(a[2])])

            def reach(sgn):
                d = 0.0
                while d < 15 and rd["area"].covers(Point(p + sgn * (d + 0.25) * nl)):
                    d += 0.25
                return d
            a6 = np.array(o["rec"][6]["anchor"])
            rows.append(dict(driver=n, dl=reach(1), dr=reach(-1), yaw=np.degrees(L.wrap(a6[2] - L.interp_pose(g, o["drive"][6]["now"])[2]))))
    D = pd.DataFrame(rows)
    D["road edge"] = np.where(D.dl < D.dr - 1, "left nearer", np.where(D.dr < D.dl - 1, "right nearer", "within 1 m"))
    T = D.groupby(["driver", "road edge"]).agg(scenes=("yaw", "size"), left_edge_m=("dl", "median"), right_edge_m=("dr", "median"), mean_heading_error_deg=("yaw", "mean"),
                                               share_right_of_1deg=("yaw", lambda x: (x < -1).mean()), share_left_of_1deg=("yaw", lambda x: (x > 1).mean()))
    c = {n: np.corrcoef(g.yaw, np.clip(g.dl - g.dr, -8, 8))[0, 1] for n, g in D.groupby("driver")}
    return ["## 5. Direction: away from the nearer road edge", "",
            "Heading error against the log at decision 6 by which edge of the mapped road area is nearer at decision 2 (rays from 3 m ahead of the rear axle; "
            "the map is privileged, labels only). Correlation of the heading error with (left distance - right distance, clipped at 8 m): "
            + ", ".join(f"{n} {v:+.2f}" for n, v in c.items()) + ".", "", md(T), ""]


def main():
    L.load_m1 = lambda name: __import__("pickle").load(open(M1 / f"{name}.pkl", "rb"))
    R = L.runs()
    T = ["# M1 diagnosis tables (generated by `scripts/m1_analysis.py`; text in ../m1_yaw_instability.md)", ""]
    T += sec_route(R) + sec_growth(R["SH30"]) + sec_navsim() + sec_replay(R) + sec_edge(R)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tables.md").write_text("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
