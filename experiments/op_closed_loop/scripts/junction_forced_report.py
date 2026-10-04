"""Report of junction_forced_lane.py: forced vs choice turns, the sky arrow, paired against A (results/junction_forced_and_arrow.md).

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/junction_forced_report.py [--seed 2]

Turn evaluation is junction_cl_report.py's (turn_geometry / eval_turn: entered, took the intended branch = came within 5 m of the dense exit, lost,
peak cross-track, exit heading error), over every turn >= 25 deg of every route the lane ran; labels (forced / choice, kind, exits) from
junction_forced_turns.csv (forced_turns.py). A / D on the 36 earlier routes come from the jcl lane (same seed), the rest from the jfa lane.
Routes with a missing arm are dropped from the tables of that arm set. CIs: cluster bootstrap over routes (2000); paired differences per route.
Outputs: junction_forced_and_arrow.md, figs/junction_forced_panels.png, junction_forced_per_turn.csv.
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
import junction_cl_report as R  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

RES = data_dir() / "runs/vlm_arb/arms"
ARMS = ["A", "D", "N", "Z", "NA", "SA", "NA0"]
PREFIX = {"A": ("jclA", "jfaA"), "D": ("jclD", "jfaD"), "N": ("jfaN",), "Z": ("jfaZ",), "NA": ("jfaNA",), "SA": ("jfaSA",), "NA0": ("jfaNA0",)}
LABEL = {"A": "A drive as shipped (route steers, desire on)", "D": "D action head only (desire on)", "N": "N action head only, desire off",
         "Z": "Z + sky arrow, shipped Cinque", "NA": "NA fine-tune q3NA + arrow", "SA": "SA fine-tune q3SA + arrow", "NA0": "NA0 q3NA, no arrow"}
COL = {"A": "#000000", "D": "#0072B2", "N": "#56B4E9", "Z": "#CC79A7", "NA": "#D55E00", "SA": "#009E73", "NA0": "#E69F00"}
BINS = [("25-60", 25, 60), ("60-120", 60, 120), (">120", 120, 400)]
HALF = R.HALF


def attempt(arm, seed, rid):
    for pre in PREFIX[arm]:
        for u in sorted(RES.glob("v2-%s-s%d-*" % (pre, seed))):
            f = u / "done" / (rid + ".json")
            if f.exists():
                return u / "attempts" / rid / str(json.loads(f.read_text()).get("attempt", 1))
    return None


def load_labels():
    lab, routes = {}, []
    for r in csv.DictReader(open(HERE.parent / "results/junction_forced_turns.csv")):
        lab[(r["route"], int(r["mi"]))] = r
    return lab


def run_routes():
    """Routes with at least one finished arm attempt, in the order: earlier 36, then the rest."""
    from junction_cl_lane import ROUTES as R36
    from junction_forced_lane import route_sets
    allr, _ = route_sets()
    return R36 + [r for k in "vx" for r in allr[k] if r not in R36]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    rng = np.random.default_rng(0)
    lab = load_labels()
    routes = run_routes()
    rows, traj, roads, rr = [], {}, {}, {}
    for rid in routes:
        att = {k: attempt(k, a.seed, rid) for k in ARMS}
        recs = {k: R.official(v) if v else None for k, v in att.items()}
        ref = next((v for v in att.values() if v is not None and (v / "route.json").exists()), None)
        if ref is None:
            continue
        D, gd, psiD, turns = R.turn_geometry(np.array(json.load(open(ref / "route.json"))["xy"]))
        turns = [T for T in turns if (rid, T["mi"]) in lab and abs(float(lab[(rid, T["mi"])]["angle"]) - T["angle"]) < 5.0]
        roads[rid] = (D, gd, psiD, turns)
        for k in ARMS:
            if att[k] is None or recs[k] is None:
                continue
            rr[(rid, k)] = (recs[k]["scores"]["score_composed"], recs[k]["scores"]["score_route"], recs[k]["status"])
            try:
                tk = [t for t in map(json.loads, open(att[k] / "ticks.jsonl")) if "truth" in t]
            except OSError:
                continue
            tr, yaw = np.array([t["truth"][:2] for t in tk]), np.array([t["truth"][2] for t in tk])
            infr = [(key, float(m.group(1)), float(m.group(2))) for key, vals in recs[k]["infractions"].items() for v in vals
                    for m in [re.search(r"x=([-\d.]+), y=([-\d.]+)", v)] if m]
            for T in turns:
                e = R.eval_turn(tr, yaw, D, psiD, T)
                zi = sum(int(np.linalg.norm(D - [x, y], axis=1).min() < 8.0 and T["it0"] - 30 <= int(np.linalg.norm(D - [x, y], axis=1).argmin()) <= T["i1"] + 30)
                         for key, x, y in infr if key in R.HARD)
                traj[(rid, T["mi"], k)] = (tr, e)
                L = lab[(rid, T["mi"])]
                rows.append(dict(route=rid, turn=T["mi"], arm=k, angle=round(T["angle"], 1), forced=int(L["forced"]), kind=L["kind"], n_exits=int(L["n_exits"]),
                                 entered=e["entered"], branch=e["branch"], peak=e["peak"], head=e["head"], leaves=int(e["peak"] > HALF) if e["entered"] else np.nan, zone_hard=zi))
    with open(out / "junction_forced_per_turn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    arms = [k for k in ARMS if any(r["arm"] == k for r in rows)]
    # a turn counts in the tables only if every arm in `arms` produced it (same turns for every arm); A / D exist only on part of the routes otherwise
    have = {k: {(r["route"], r["turn"]) for r in rows if r["arm"] == k} for k in arms}
    common = set.intersection(*have.values())
    keys = sorted(common)
    ix = {(r["route"], r["turn"], r["arm"]): r for r in rows}
    cl = [r for r in rows if (r["route"], r["turn"]) in common]
    L_ = ["# Forced turns and the sky arrow, closed loop (seed %d)\n" % a.seed]
    n_f = sum(ix[(r, t, arms[0])]["forced"] for r, t in keys)
    L_.append("Arms present: %s. Turns scored by every arm: %d on %d routes, of which %d forced and %d choice. Definitions: junction_forced_report.py docstring (turn metrics are "
              "junction_cl_report.py's). CIs: cluster bootstrap over routes (2000).\n" % (", ".join(arms), len(keys), len({r for r, _ in keys}), n_f, len(keys) - n_f))

    def grp(sel):
        return [(r, t) for r, t in keys if sel(ix[(r, t, arms[0])])]

    def by_route(ks, k, f, pred=lambda r: True):
        rs = sorted({r for r, _ in ks})
        return [np.array([f(ix[(r, t, k)]) for rr_, t in ks if rr_ == r and pred(ix[(r, t, k)])], float) for r in rs]

    def table(title, ks):
        L_.append("\n### %s (%d turns, %d routes)\n" % (title, len(ks), len({r for r, _ in ks})))
        L_.append("| arm | entered % | took intended branch % | lost % | leaves lane (>1.75 m) % of entered | median peak cross-track m, entered | median abs exit heading err deg, branch | zone collisions + off-road (n) |")
        L_.append("|---|---|---|---|---|---|---|---|")
        for k in arms:
            e = R.boot(by_route(ks, k, lambda r: r["entered"]), np.mean, rng)
            b = R.boot(by_route(ks, k, lambda r: r["branch"] == "yes"), np.mean, rng)
            lo = R.boot(by_route(ks, k, lambda r: r["branch"] == "lost"), np.mean, rng)
            lv = R.boot(by_route(ks, k, lambda r: r["leaves"], lambda r: r["entered"] == 1), np.mean, rng)
            pk = R.boot(by_route(ks, k, lambda r: r["peak"], lambda r: r["entered"] == 1), np.nanmedian, rng)
            hd = R.boot(by_route(ks, k, lambda r: abs(r["head"]), lambda r: r["branch"] == "yes"), np.nanmedian, rng)
            zh = sum(ix[(r, t, k)]["zone_hard"] for r, t in ks)
            L_.append("| %s | %s | %s | %s | %s | %s | %s | %d |" % (k, R.fmt(e, 0, True), R.fmt(b, 0, True), R.fmt(lo, 0, True), R.fmt(lv, 0, True), R.fmt(pk), R.fmt(hd, 1), zh))

    forced, choice = grp(lambda r: r["forced"] == 1), grp(lambda r: r["forced"] == 0)
    L_.append("\n## Per-turn outcomes\n")
    table("Forced turns (the map offers no exit within 30 deg of straight)", forced)
    table("Choice turns (a straight exit exists)", choice)
    fj = grp(lambda r: r["forced"] == 1 and r["kind"] == "junction")
    fc = grp(lambda r: r["forced"] == 1 and r["kind"] == "curve")
    table("Forced, junction (T-stem, no straight exit)", fj)
    table("Forced, plain road curve / L-bend (single exit)", fc)
    L_.append("\n### By turn angle (took intended branch %, n turns)\n")
    L_.append("| group | angle bin | n | " + " | ".join(arms) + " |\n|---|---|---|" + "---|" * len(arms))
    for gname, ks in (("forced", forced), ("choice", choice)):
        for bn, lo_, hi_ in BINS:
            kk = [(r, t) for r, t in ks if lo_ <= abs(ix[(r, t, arms[0])]["angle"]) < hi_]
            if kk:
                L_.append("| %s | %s | %d | " % (gname, bn, len(kk)) + " | ".join("%d%%" % round(100 * np.mean([ix[(r, t, k)]["branch"] == "yes" for r, t in kk])) for k in arms) + " |")
    # ------------------------------------------------------------------ paired vs A
    L_.append("\n## Paired against A (route-clustered CIs)\n")
    L_.append("Branch rate: per-turn took-branch indicator minus A's on the same turn, averaged within a route, bootstrap over routes. DS / RC: per route, official, all routes finished by both arms.\n")
    L_.append("| arm | forced: branch rate diff pp | choice: branch rate diff pp | DS diff, routes with a forced turn | DS diff, other routes | RC diff, routes with a forced turn | RC diff, other routes |")
    L_.append("|---|---|---|---|---|---|---|")
    froutes = {r for r, _ in forced}
    for k in arms:
        if k == "A":
            continue
        cells = []
        for ks in (forced, choice):
            d = by_route(ks, k, lambda r: 0.0)
            rs = sorted({r for r, _ in ks})
            d = [np.array([float(ix[(r, t, k)]["branch"] == "yes") - float(ix[(r, t, "A")]["branch"] == "yes") for r_, t in ks if r_ == r]) for r in rs]
            cells.append(R.fmt(R.boot(d, np.mean, rng), 1, True))
        for j in (0, 1):
            for sel in (lambda r: r in froutes, lambda r: r not in froutes):
                rs = [r for r in routes if (r, k) in rr and (r, "A") in rr and sel(r)]
                d = [np.array([rr[(r, k)][j] - rr[(r, "A")][j]]) for r in rs]
                cells.append(R.fmt(R.boot(d, np.mean, rng), 1) + " (n=%d)" % len(rs))
        L_.append("| %s | %s | %s | %s | %s | %s | %s |" % (k, *cells[:2], cells[2], cells[3], cells[4], cells[5]))
    L_.append("\n### Route scores (official, mean over routes finished by the arm)\n")
    L_.append("| arm | routes | DS [CI] | RC [CI] |\n|---|---|---|---|")
    for k in arms:
        rs = [r for r in routes if (r, k) in rr]
        L_.append("| %s | %d | %s | %s |" % (k, len(rs), R.fmt(R.boot([np.array([rr[(r, k)][0]]) for r in rs], np.mean, rng), 1), R.fmt(R.boot([np.array([rr[(r, k)][1]]) for r in rs], np.mean, rng), 1)))
    # ------------------------------------------------------------------ per forced turn listing
    L_.append("\n## Every forced turn (took the intended branch: y yes, l lost, n never entered)\n")
    L_.append("| route | turn | angle | kind | exits (deg) | " + " | ".join(arms) + " |\n|---|---|---|---|---|" + "---|" * len(arms))
    lab_by = {k: v for k, v in lab.items()}
    for r, t in forced:
        L = lab_by[(r, t)]
        L_.append("| %s | %d | %+.0f | %s | %s | " % (r, t, ix[(r, t, arms[0])]["angle"], L["kind"], L["exit_angles"]) +
                  " | ".join("%s %.1f" % (ix[(r, t, k)]["branch"][0], ix[(r, t, k)]["peak"]) for k in arms) + " |")
    # ------------------------------------------------------------------ panels
    def spread(r, t):
        return len({ix[(r, t, k)]["branch"] for k in arms})
    pick = []
    for ks, n in ((forced, 3), (choice, 3)):
        # varied outcomes first (an arm differs), then the smallest route id; at most one turn per route; forced: one per kind where possible
        order = sorted(ks, key=lambda x: (-spread(*x), lab_by[x]["kind"] != "junction", x[0]))
        seen, sel = set(), []
        for x in order:
            if x[0] not in seen and len(sel) < n:
                seen.add(x[0])
                sel.append(x)
        pick += sel
    fig, axs = plt.subplots(2, 3, figsize=(18, 12.5))
    cap = []
    for ax, (rid, mi) in zip(axs.ravel(), pick):
        D, gd, psiD, turns = roads[rid]
        T = next(x for x in turns if x["mi"] == mi)
        a0 = max(T["it0"] - 100, 0)
        Dp = D[a0:min(T["i1"] + 80, len(D))]
        tg = np.gradient(Dp, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nr = np.stack([-tg[:, 1], tg[:, 0]], -1)
        ax.fill(np.vstack([Dp + HALF * nr, (Dp - HALF * nr)[::-1]])[:, 0], np.vstack([Dp + HALF * nr, (Dp - HALF * nr)[::-1]])[:, 1], color="#dddddd", zorder=0,
                label="lane +-1.75 m about the dense centreline")
        txt = []
        for k in arms:
            tr, e = traj[(rid, mi, k)]
            near, dev = R.project(tr, D)
            sel = (near >= a0) & (near <= min(T["i1"] + 80, len(D) - 1)) & (dev < 30)
            ax.plot(tr[sel, 0], tr[sel, 1], color=COL[k], lw=2.0, ls="--" if k in ("NA0", "N") else "-", zorder=3 + (k == "A"), label=LABEL[k])
            if e["entered"] and e["peak"] > HALF:
                s0, s1 = e["span"]
                j = s0 + int(np.argmax(dev[s0:s1 + 1] > HALF))
                ax.plot(tr[j, 0], tr[j, 1], "X", color=COL[k], ms=10, mec="white", mew=1.1, zorder=6)
            txt.append("%-3s %-5s peak %5.1f m" % (k, e["branch"], e["peak"] if np.isfinite(e["peak"]) else np.nan))
        L = lab_by[(rid, mi)]
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("%s route %s turn %d: %+.0f deg, R_min %.0f m, %s, exits %s" % ("FORCED" if L["forced"] == "1" else "choice", rid, mi, T["angle"], T["rmin"], L["kind"], L["exit_angles"]), fontsize=9)
        ax.grid(alpha=.25)
        ax.text(0.0, -0.08, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="top", family="monospace")
        cap.append("- %s route %s turn %d (%+.0f deg, %s): %s" % ("forced" if L["forced"] == "1" else "choice", rid, mi, T["angle"], L["kind"], "; ".join(x.replace("  ", " ") for x in txt)))
    h, lb = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], marker="X", color="gray", ls="")], lb + ["first sample beyond 1.75 m"], loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=9)
    (out.parent / "figs").mkdir(exist_ok=True)
    fig.savefig(out.parent / "figs/junction_forced_panels.png", dpi=105)
    L_ += ["\n## Panels\n", "![panels](../figs/junction_forced_panels.png)\n", "Top row forced, bottom row choice; turns picked by outcome spread across arms (then T-stems first, then route id).\n"] + cap
    (out / "junction_forced_tables.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
