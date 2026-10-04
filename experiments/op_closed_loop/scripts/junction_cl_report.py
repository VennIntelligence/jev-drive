"""Closed-loop junction report: arms jclA (drive as shipped) / jclD (action head steers everywhere) / jclE (D + x1.95 in junctions).

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/junction_cl_report.py [--seed 2] [--out experiments/op_closed_loop/results]

Reads $DATA_DIR/runs/vlm_arb/arms/v2-jcl<arm>-s<seed>-q*/ (lane: junction_cl_lane.py). Per turn (the 38 turns >= 25 deg of turn_calibration_options.py): the
closed-loop path (ticks.jsonl truth) is projected on the dense centreline (route.json, 0.25 m); window = 5 m before the turn start to the dense exit.
  entered   the car came within 5 m of the dense centreline at the window start
  branch    the car also came within 5 m of the dense exit point (took the intended branch), else "lost" (entered, no exit) or "never" (not entered)
  peak      max cross-track over the window (first entry to first passage of the exit, or the end of the run); leaves = peak > 1.75 m
  head      signed heading error (deg, right-positive) when the car first passes the zone-end sample (dense turn end + 3 m); only for branch=yes
  infractions in the zone: collisions_* / outside_route_lanes whose location projects within 7.5 m (arc) before / after the window and < 8 m from the centreline
Per route: DS, RC (official). CIs: cluster bootstrap over routes (2000); paired differences D-A, E-A are per-route (turn-level rates are route means of turns).
Outputs: junction_closed_loop.md, figs/junction_cl_panels.png, junction_cl_per_turn.csv (small).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[2] / "experiments/vlm_arb/scripts")]
import turn_calibration_lib as L  # noqa: E402
import turn_calibration_options as O  # noqa: E402
import turn_calibration_sparse as S  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ARMS = {"A": "jclA", "D": "jclD", "E": "jclE"}
COL = {"A": "#000000", "D": "#0072B2", "E": "#D55E00", "R": "#56B4E9"}
NAME = {"A": "A drive as shipped (route steers in zones)", "D": "D action head steers everywhere", "E": "E D + curvature x1.95 in turns",
        "R": "open-loop replay of D (kinematic)"}
PANEL = [("tight left", "6999"), ("tight right", "28180"), ("wide turn", "34183"), ("largest replay disagreement", "35243")]
HALF = 1.75
RUN = data_dir() / "runs/vlm_arb"
HARD = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle", "outside_route_lanes")


def attempt(arm, seed, rid):
    for u in sorted(RUN.glob("arms/v2-%s-s%d-q*" % (ARMS[arm], seed))):
        f = u / "done" / (rid + ".json")
        if f.exists():
            return u / "attempts" / rid / str(json.loads(f.read_text()).get("attempt", 1))
    return None


def official(a):
    try:
        rec = json.loads((a / "results.json").read_text())["_checkpoint"]["records"][0]
    except (OSError, ValueError, IndexError, KeyError):
        return None
    return rec


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def turn_geometry(xy):
    P, g = L.resample(xy)
    D, gd = S.poly_resample(xy, 0.25)
    psiD = L.heading(D)
    out = []
    for mi, m in enumerate(x for x in L.maneuvers(P) if abs(x["angle"]) >= 25):
        i0 = int(np.searchsorted(gd, g[m["i0"]]))
        i1 = min(int(np.searchsorted(gd, g[m["i1"]] + 15.0)), len(D) - 1)      # dense exit as in the replay figure (turn end + 15 m)
        iz = min(int(np.searchsorted(gd, g[m["i1"]] + 3.0)), len(D) - 1)
        out.append(dict(mi=mi, it0=max(i0 - 20, 0), i0=i0, i1=i1, iz=iz, angle=m["angle"], rmin=m["rmin"], g0=g[m["i0"]], g1=g[m["i1"]]))
    return D, gd, psiD, out


def project(tr, D):
    d = np.linalg.norm(tr[:, None, :2] - D[None, :, :], axis=2)
    return d.argmin(1), d.min(1)


def eval_turn(tr, yaw, D, psiD, T):
    near, dev = project(tr, D)
    ok_in = np.where((near >= T["it0"]) & (dev < 5.0))[0]
    if not len(ok_in):
        return dict(entered=0, branch="never", peak=np.nan, head=np.nan, span=(0, 0))
    a = ok_in[0]
    out = np.where((near[a:] >= T["i1"]) | (dev[a:] > 25.0))[0]
    b = a + (out[0] if len(out) else len(near) - a - 1)
    span = np.arange(a, b + 1)
    branch = bool(np.min(np.linalg.norm(tr[a:, :2] - D[T["i1"]], axis=1)) < 5.0)
    k = np.where(near[a:] >= T["iz"])[0]
    head = float(np.degrees(wrap(yaw[a + k[0]] - psiD[T["iz"]]))) if branch and len(k) else np.nan
    return dict(entered=1, branch="yes" if branch else "lost", peak=float(dev[span].max()), head=head, span=(a, b))


def boot(groups, stat, rng, B=2000):
    """groups: list of arrays (one per route); returns (estimate, lo, hi) of stat over the concatenation, resampling routes."""
    g = [x for x in groups if len(x)]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    est = stat(np.concatenate(g))
    bs = [stat(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(B)]
    return est, *np.nanpercentile(bs, [2.5, 97.5])


def fmt(t, d=2, pct=False):
    if not np.isfinite(t[0]):
        return "n/a"
    k = 100.0 if pct else 1.0
    return "%.*f [%.*f, %.*f]" % (d, k * t[0], d, k * t[1], d, k * t[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "figs").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows, roads, traj = [], {}, {}
    from junction_cl_lane import ROUTES  # noqa: E402
    for rid in ROUTES:
        att = {k: attempt(k, a.seed, rid) for k in ARMS}
        rec = {k: official(v) if v else None for k, v in att.items()}
        if att["A"] is None:
            continue
        xy = np.array(json.load(open(att["A"] / "route.json"))["xy"])
        D, gd, psiD, turns = turn_geometry(xy)
        roads[rid] = (D, gd, psiD, turns)
        for k in ARMS:
            if att[k] is None or rec[k] is None:
                continue
            tk = [json.loads(x) for x in open(att[k] / "ticks.jsonl")]
            tk = [t for t in tk if "truth" in t]
            tr = np.array([t["truth"][:2] for t in tk])
            yaw = np.array([t["truth"][2] for t in tk])
            infr = []
            for key, vals in rec[k]["infractions"].items():
                for v in vals:
                    mm = re.search(r"x=([-\d.]+), y=([-\d.]+)", v)
                    if mm:
                        infr.append((key, float(mm.group(1)), float(mm.group(2))))
            for T in turns:
                e = eval_turn(tr, yaw, D, psiD, T)
                zi = 0
                for key, x, y in infr:
                    if key in HARD:
                        d = np.linalg.norm(D - [x, y], axis=1)
                        j = int(d.argmin())
                        zi += int(d[j] < 8.0 and T["it0"] - 30 <= j <= T["i1"] + 30)
                traj[(rid, T["mi"], k)] = (tr, e, D, T)
                rows.append(dict(route=rid, turn=T["mi"], arm=k, angle=round(T["angle"], 1), entered=e["entered"], branch=e["branch"], peak=e["peak"], head=e["head"],
                                 leaves=int(e["peak"] > HALF) if e["entered"] else np.nan, zone_hard=zi, route_dev=int(len(rec[k]["infractions"].get("route_dev", [])) > 0)))
    rr = {}
    for rid in ROUTES:
        for k in ARMS:
            at = attempt(k, a.seed, rid)
            r = official(at) if at else None
            if r:
                rr[(rid, k)] = (r["scores"]["score_composed"], r["scores"]["score_route"], r["status"])
    with open(out / "junction_cl_per_turn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # ---------------------------------------------------------------- tables
    done = {r for r in ROUTES if all((r, k) in rr for k in ARMS)}
    keys = sorted({(r["route"], r["turn"]) for r in rows if r["route"] in done})
    L_ = ["# Junction turns of the openpilot action head, closed loop (seed %d)\n" % a.seed,
          "Arms: **A** shipped `drive` agent (dense route steers in the junction zones and on divergence); **D** `zones: false`, `div_m: 1e9`: the action-head curvature "
          "steers everywhere, longitudinal and everything else identical; **E** D plus action curvature x1.95 inside LEFT / RIGHT command runs +-3 m. "
          "No privileged light (`resume` stays `timer`). Routes finished by all three arms: %d of %d (%d turns). Definitions: scripts/junction_cl_report.py docstring. "
          "CIs: cluster bootstrap over routes (2000).\n" % (len(done), len(ROUTES), len(keys))]
    L_.append("\n## Per-turn outcomes\n")
    L_.append("| arm | turns | entered | took intended branch | lost (entered, no exit) | leaves lane (>1.75 m), of entered | median peak cross-track m [CI], entered | median abs exit heading err deg, branch | zone collisions + off-road infractions (n) |")
    L_.append("|---|---|---|---|---|---|---|---|---|")
    S_ = {}
    for k in ARMS:
        rs = [r for r in rows if r["arm"] == k and (r["route"], r["turn"]) in set(keys)]
        by = lambda f, sel=lambda r: True: [np.array([f(r) for r in rs if r["route"] == rid and sel(r)], float) for rid in sorted(done)]  # noqa: E731
        ent = boot(by(lambda r: r["entered"]), np.mean, rng)
        br = boot(by(lambda r: r["branch"] == "yes"), np.mean, rng)
        lost = boot(by(lambda r: r["branch"] == "lost"), np.mean, rng)
        lv = boot(by(lambda r: r["leaves"], lambda r: r["entered"] == 1), np.mean, rng)
        pk = boot(by(lambda r: r["peak"], lambda r: r["entered"] == 1), np.median, rng)
        hd = boot(by(lambda r: abs(r["head"]), lambda r: r["branch"] == "yes"), np.median, rng)
        zh = sum(r["zone_hard"] for r in rs)
        S_[k] = (ent, br, lost, lv, pk, hd)
        L_.append("| %s | %d | %s | %s | %s | %s | %s | %s | %d |" % (k, len(rs), fmt(ent, 0, True), fmt(br, 0, True), fmt(lost, 0, True), fmt(lv, 0, True), fmt(pk), fmt(hd, 1), zh))
    L_.append("\n## Per route (official DS / RC) and paired differences\n")
    L_.append("| arm | routes | DS mean [CI] | RC mean [CI] | routes with route_dev / agent-failed status |")
    L_.append("|---|---|---|---|---|")
    ds, rc = {k: np.array([rr[(r, k)][0] for r in sorted(done)]) for k in ARMS}, {k: np.array([rr[(r, k)][1] for r in sorted(done)]) for k in ARMS}
    for k in ARMS:
        bad = sum(1 for r in done if "deviated" in rr[(r, k)][2] or rr[(r, k)][2].startswith("Failed"))
        L_.append("| %s | %d | %s | %s | %d |" % (k, len(done), fmt(boot([np.array([x]) for x in ds[k]], np.mean, rng), 1), fmt(boot([np.array([x]) for x in rc[k]], np.mean, rng), 1), bad))
    for k in "DE":
        pd = {}
        for nm, v in (("DS", ds), ("RC", rc)):
            pd[nm] = boot([np.array([x]) for x in v[k] - v["A"]], np.mean, rng)
        tk = lambda f: [np.array([f(rd, tn) for rd, tn in keys if rd == rid], float) for rid in sorted(done)]  # noqa: E731
        g = lambda arm, f: {(r["route"], r["turn"]): r for r in rows if r["arm"] == arm}  # noqa: E731
        ra, rk = g("A", 0), g(k, 0)
        d_br = boot(tk(lambda rd, tn: float(rk[(rd, tn)]["branch"] == "yes") - float(ra[(rd, tn)]["branch"] == "yes")), np.mean, rng)
        d_pk = boot(tk(lambda rd, tn: (rk[(rd, tn)]["peak"] - ra[(rd, tn)]["peak"]) if rk[(rd, tn)]["entered"] and ra[(rd, tn)]["entered"] else np.nan), np.nanmedian, rng)
        L_.append("\n**Paired %s - A**: DS %s; RC %s; intended-branch rate (pp) %s; median peak cross-track %s m (turns entered by both)." % (
            k, fmt(pd["DS"], 1), fmt(pd["RC"], 1), fmt((d_br[0], d_br[1], d_br[2]), 1, True), fmt(d_pk)))
    L_.append("\n## Panel turns\n\n![panels](../figs/junction_cl_panels.png)\n")
    # ---------------------------------------------------------------- panels (replay of D from the earlier figure vs closed-loop paths)
    bt = {k: v for k, v in np.load(O.T / "b2d_ticks.npz", allow_pickle=True).items()}
    fig, axs = plt.subplots(2, 2, figsize=(13, 13.5))
    cap = []
    for ax, (nm, rid) in zip(axs.ravel(), PANEL):
        if rid not in roads:
            continue
        D, gd, psiD, turns = roads[rid]
        T = turns[0]
        rm = bt["route"] == rid
        rt = {k: bt[k][rm] for k in ("s_route", "act_k", "plan_k1")}
        zs, ze = T["g0"] - O.ZM, T["g1"] + O.ZM
        fa = O.ff_profile(rt, "act_k", 1.0, zs, ze)
        i0s = int(np.searchsorted(gd, max(T["g0"] - 25.0, 0.0)))
        psi0 = np.arctan2(*(D[min(i0s + 4, len(D) - 1)] - D[i0s])[::-1])
        a0 = max(T["it0"] - 100, 0)
        Dp = D[a0:min(T["i1"] + 80, len(D))]
        tg = np.gradient(Dp, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nr = np.stack([-tg[:, 1], tg[:, 0]], -1)
        poly = np.vstack([Dp + HALF * nr, (Dp - HALF * nr)[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color="#dddddd", zorder=0, label="lane +-1.75 m about the dense centreline")
        if fa is not None:
            trr = O.run(D, gd, i0s, psi0, 5.0, 6.5, xy_of(D), fa, zs, ze, T["i1"])
            ax.plot(trr[:, 0], trr[:, 1], color=COL["R"], lw=2.0, ls="--", zorder=2, label=NAME["R"])
        txt = []
        for k in ARMS:
            if (rid, 0, k) not in traj:
                continue
            tr, e, _, _ = traj[(rid, 0, k)]
            near, dev = project(tr, D)
            sel = (near >= a0) & (near <= min(T["i1"] + 80, len(D) - 1)) & (dev < 30)
            ax.plot(tr[sel, 0], tr[sel, 1], color=COL[k], lw=2.2, zorder=3 + (k == "A"), label=NAME[k])
            if e["entered"] and e["peak"] > HALF:
                s0, s1 = e["span"]
                j = s0 + int(np.argmax(dev[s0:s1 + 1] > HALF))
                ax.plot(tr[j, 0], tr[j, 1], "X", color=COL[k], ms=11, mec="white", mew=1.2, zorder=6)
            txt.append("%s  %s  peak %s m  head %s deg" % (k, e["branch"], "n/a" if not np.isfinite(e["peak"]) else "%.2f" % e["peak"], "n/a" if not np.isfinite(e["head"]) else "%+.1f" % e["head"]))
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("%s: route %s, %+.0f deg, R_min %.1f m" % (nm, rid, T["angle"], T["rmin"]), fontsize=10)
        ax.set_xlabel("x (m, CARLA)")
        ax.set_ylabel("y (m, CARLA, down)")
        ax.grid(alpha=.25)
        ax.text(0.0, -0.13, "\n".join(txt), transform=ax.transAxes, fontsize=8, va="top", family="monospace")
        cap.append("- %s (route %s): %s" % (nm, rid, "; ".join(txt)))
    h, lab = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], marker="X", color="gray", ls="")], lab + ["first sample beyond 1.75 m"], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=9)
    fig.suptitle("Closed-loop paths (A, D, E) vs the open-loop replay of D (5 m/s, same turns as tmp/junction_options/panels.png)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=7)
    fig.savefig(out.parent / "figs/junction_cl_panels.png", dpi=120)
    L_ += ["Per-panel numbers (closed loop): "] + cap
    (out / "junction_closed_loop.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def xy_of(D):
    return D


if __name__ == "__main__":
    main()
