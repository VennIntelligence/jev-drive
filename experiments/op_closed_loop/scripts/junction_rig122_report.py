"""Report of junction_rig122_lane.py: junction turns at 1.22 m vs 1.433 m (results/junction_rig122.md).

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_closed_loop/scripts/junction_rig122_report.py [--panel 28180:0,...] [--seed 2]

Arms, all seed 2 (turn metrics are junction_cl_report.py's: entered, took the intended branch = came within 5 m of the dense exit, lost, peak cross-track,
leaves lane = peak > 1.75 m):
  A       shipped `drive`, zones on, 1.433 m          (jcl / jfa lanes)
  D       `drive`, zones off, 1.433 m, raw curvature  (jcl / jfa lanes; the arm of decisions 121 / 122)
  s143nz  `spec` (clip + delay), zones off, 1.433 m   (unified lane for 6 routes, rig122 lane for the rest): height-only control of s122nz
  s122    `spec`, zones on, 1.22 m
  s122nz  `spec`, zones off, 1.22 m
Per turn also: head peak = max |act_k| (plans.jsonl, what the head asks for) in the turn window against the needed curvature 1 / R_min; collisions /
red lights in the window (tick index from the first entry to 3 s after the window end) with what was hit, the ego speed and the ego's distance to the
lane centre at the contact. CIs: cluster bootstrap over routes (2000); rates of 10-20 turns are directions, the discordant pair counts are in the tables.
Outputs: results/junction_rig122_per_turn.csv, results/junction_rig122_collisions.csv, results/junction_rig122_tables.md, figs/junction_rig122_panels.png.
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
import junction_forced_report as F  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DATA = data_dir()
ARMS = ["A", "D", "s143nz", "s122nz", "s122"]
LABEL = {"A": "A drive, zones on, 1.433 m", "D": "D drive, zones off, 1.433 m", "s143nz": "s143nz spec, zones off, 1.433 m", "s122nz": "s122nz spec, zones off, 1.22 m",
         "s122": "s122 spec, zones on, 1.22 m"}
COL = {"A": "#000000", "D": "#0072B2", "s143nz": "#56B4E9", "s122nz": "#D55E00", "s122": "#009E73"}
SHORT = {"A": "A", "D": "D", "s143nz": "143nz", "s122nz": "122nz", "s122": "122"}
HALF = R.HALF
HARD = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")


def attempt(arm, seed, rid):
    if arm in ("A", "D"):
        return F.attempt(arm, seed, rid)
    dirs = list((DATA / "runs/unified/b2d/arms").glob("%s-s%d" % (arm, seed))) + sorted((DATA / "runs/rig122/arms").glob("%s-s%d-k*" % (arm, seed)))
    for u in dirs:
        f = u / "done" / (rid + ".json")
        if f.exists():
            return u / "attempts" / rid / str(json.loads(f.read_text()).get("attempt", 1))
    return None


def rows_of(p):
    try:
        return [json.loads(x) for x in open(p)]
    except OSError:
        return []


def collisions(rec, tk, tr, near, dev, wins):
    """Every collision / red light of the run: time, what, ego speed, ego distance to the lane centre, and the turn window (index) it falls in."""
    out = []
    for key, vals in rec["infractions"].items():
        if key not in HARD and key != "red_light":
            continue
        for v in vals:
            m = re.search(r"x=([-\d.]+), y=([-\d.]+)", v)
            if not m:
                continue
            d = np.linalg.norm(tr - [float(m.group(1)), float(m.group(2))], axis=1)
            i = int(d.argmin())
            what = (re.search(r"type=([\w.]+)", v) or [None, "traffic light"])[1] if key != "red_light" else "traffic light " + (re.search(r"light (\d+)", v) or [0, ""])[1]
            out.append(dict(kind=key.replace("collisions_", "hit ") if key != "red_light" else "red light", what=what, t=round(tk[i]["t"], 1), v=round(tk[i]["v"], 1),
                            dev=round(float(dev[i]), 1), tick=i, turn=next((mi for mi, (a, b) in wins.items() if a <= i <= b + 60), "-")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--panel", default="")
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    lab = F.load_labels()
    from junction_rig122_lane import NEW
    routes = NEW + "28180 24944 27297 9196 6999 34183".split()
    rows, cols, traj, roads, rr, hold = [], [], {}, {}, {}, {}
    for rid in routes:
        att = {k: attempt(k, a.seed, rid) for k in ARMS}
        ref = next((v for v in att.values() if v is not None and (v / "route.json").exists()), None)
        if ref is None:
            continue
        D, gd, psiD, turns = R.turn_geometry(np.array(json.load(open(ref / "route.json"))["xy"]))
        turns = [T for T in turns if (rid, T["mi"]) in lab and abs(float(lab[(rid, T["mi"])]["angle"]) - T["angle"]) < 5.0]
        roads[rid] = (D, gd, psiD, turns)
        for k in ARMS:
            if att[k] is None:
                continue
            rec = R.official(att[k])
            tk = [t for t in rows_of(att[k] / "ticks.jsonl") if "truth" in t]
            if rec is None or not tk:
                continue
            rr[(rid, k)] = (rec["scores"]["score_composed"], rec["scores"]["score_route"], rec["status"])
            tr, yaw = np.array([t["truth"][:2] for t in tk]), np.array([t["truth"][2] for t in tk])
            tt = np.array([t["t"] for t in tk])
            pl = [p for p in rows_of(att[k] / "plans.jsonl") if not p.get("warm") and "act_k" in p]
            pt, pk = np.array([p["t"] for p in pl]), np.array([p["act_k"] for p in pl])
            near, dev = R.project(tr, D)
            ev = {T["mi"]: R.eval_turn(tr, yaw, D, psiD, T) for T in turns}
            wins = {mi: e["span"] for mi, e in ev.items() if e["entered"]}
            cl = collisions(rec, tk, tr, near, dev, wins)
            for c in cl:
                cols.append(dict(route=rid, arm=k, **{x: y for x, y in c.items() if x != "tick"}))
            for T in turns:
                e = ev[T["mi"]]
                L = lab[(rid, T["mi"])]
                if e["entered"]:
                    s0, s1 = e["span"]
                    sel = (pt >= tt[s0]) & (pt <= tt[s1])
                    hp = float(np.abs(pk[sel]).max()) if sel.any() else np.nan
                    vmin = float(np.min([t["v"] for t in tk[s0:s1 + 1]]))
                else:
                    hp, vmin = np.nan, np.nan
                inwin = [c for c in cl if c["turn"] == T["mi"]]
                traj[(rid, T["mi"], k)] = (tr, e)
                rows.append(dict(route=rid, turn=T["mi"], arm=k, angle=round(T["angle"], 1), rmin=round(T["rmin"], 1), need=round(1 / T["rmin"], 3), forced=int(L["forced"]),
                                 kind=L["kind"], n_exits=int(L["n_exits"]), entered=e["entered"], branch=e["branch"], peak=round(e["peak"], 2), head_pk=round(hp, 3),
                                 vmin=round(vmin, 1), coll=sum(c["kind"].startswith("hit") for c in inwin), red=sum(c["kind"] == "red light" for c in inwin),
                                 leaves=int(e["peak"] > HALF) if e["entered"] else np.nan))
    with open(out / "junction_rig122_per_turn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "junction_rig122_collisions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols[0]))
        w.writeheader()
        w.writerows(cols)
    ix = {(r["route"], r["turn"], r["arm"]): r for r in rows}
    keys = sorted({(r["route"], r["turn"]) for r in rows if all((r["route"], r["turn"], k) in ix for k in ARMS)})
    rng = np.random.default_rng(0)
    L_ = ["# Junction turns at 1.22 m vs 1.433 m: raw tables (seed %d)\n" % a.seed]
    forced = [k for k in keys if ix[k + ("A",)]["forced"] == 1]
    choice = [k for k in keys if ix[k + ("A",)]["forced"] == 0]
    L_.append("Turns scored by every arm: %d on %d routes (%d forced, %d choice). Arms: %s.\n" % (len(keys), len({r for r, _ in keys}), len(forced), len(choice),
                                                                                              "; ".join("%s = %s" % (k, LABEL[k][len(k) + 1:]) for k in ARMS)))

    def rate(ks, k, f, pred=lambda r: True):
        g = [np.array([f(ix[(r, t, k)]) for r2, t in ks if r2 == r and pred(ix[(r, t, k)])], float) for r in sorted({r for r, _ in ks})]
        return R.boot(g, np.mean, rng)

    def table(title, ks):
        L_.append("\n### %s (%d turns, %d routes)\n" % (title, len(ks), len({r for r, _ in ks})))
        L_.append("| arm | took branch (n, %) | lost % | leaves lane % of entered | median peak cross-track m | median head peak / needed | collisions in window (n) | red lights in window (n) |")
        L_.append("|---|---|---|---|---|---|---|---|")
        for k in ARMS:
            n = sum(ix[(r, t, k)]["branch"] == "yes" for r, t in ks)
            lo = rate(ks, k, lambda r: r["branch"] == "lost")
            lv = rate(ks, k, lambda r: r["leaves"], lambda r: r["entered"] == 1)
            pk = np.nanmedian([ix[(r, t, k)]["peak"] for r, t in ks if ix[(r, t, k)]["entered"]] or [np.nan])
            hr = np.nanmedian([ix[(r, t, k)]["head_pk"] / ix[(r, t, k)]["need"] for r, t in ks if ix[(r, t, k)]["entered"]] or [np.nan])
            L_.append("| %s | %d / %d (%d%%) | %s | %s | %.1f | %.2f | %d | %d |" % (SHORT[k], n, len(ks), round(100 * n / max(len(ks), 1)), R.fmt(lo, 0, True), R.fmt(lv, 0, True), pk, hr,
                                                                                  sum(ix[(r, t, k)]["coll"] for r, t in ks), sum(ix[(r, t, k)]["red"] for r, t in ks)))

    L_.append("\n## Per-group outcomes\n")
    table("Choice turns", choice)
    table("Forced turns", forced)
    L_.append("\n### By minimum radius (took intended branch n / turns; head peak / needed, median)\n")
    L_.append("| group | R_min | n | " + " | ".join(SHORT[k] for k in ARMS) + " |\n|---|---|---|" + "---|" * len(ARMS))
    for gname, ks in (("choice", choice), ("forced", forced)):
        for bn, lo_, hi_ in (("< 10 m", 0, 10), ("10-20 m", 10, 20), ("> 20 m", 20, 1e9)):
            kk = [(r, t) for r, t in ks if lo_ <= ix[(r, t, "A")]["rmin"] < hi_]
            if kk:
                cells = []
                for k in ARMS:
                    ent = [ix[(r, t, k)]["head_pk"] / ix[(r, t, k)]["need"] for r, t in kk if ix[(r, t, k)]["entered"]]
                    cells.append("%d / %d (%.2f)" % (sum(ix[(r, t, k)]["branch"] == "yes" for r, t in kk), len(kk), np.nanmedian(ent) if ent else np.nan))
                L_.append("| %s | %s | %d | " % (gname, bn, len(kk)) + " | ".join(cells) + " |")
    # paired comparisons against the existing 1.433 m numbers
    L_.append("\n## Paired comparisons (same turn; discordant pairs = turns where only one of the two arms took the branch)\n")
    L_.append("| pair | group | n | both | only first | only second | rate diff pp [route-cluster CI] |\n|---|---|---|---|---|---|---|")
    for p, q in (("s122nz", "D"), ("s122nz", "s143nz"), ("s143nz", "D"), ("s122", "A")):
        for gname, ks in (("choice", choice), ("forced", forced)):
            y = lambda k, r, t: ix[(r, t, k)]["branch"] == "yes"  # noqa: E731
            both = sum(y(p, r, t) and y(q, r, t) for r, t in ks)
            o1 = sum(y(p, r, t) and not y(q, r, t) for r, t in ks)
            o2 = sum(y(q, r, t) and not y(p, r, t) for r, t in ks)
            g = [np.array([float(y(p, r, t)) - float(y(q, r, t)) for r2, t in ks if r2 == r]) for r in sorted({r for r, _ in ks})]
            L_.append("| %s - %s | %s | %d | %d | %d | %d | %s |" % (SHORT[p], SHORT[q], gname, len(ks), both, o1, o2, R.fmt(R.boot(g, np.mean, rng), 0, True)))
    L_.append("\n### Route scores (official; routes run by the arm; DS / RC mean [CI])\n\n| arm | routes | DS | RC | completed | collisions (run total) | red lights |\n|---|---|---|---|---|---|---|")
    for k in ARMS:
        rs = [r for r in routes if (r, k) in rr]
        cc = [c for c in cols if c["arm"] == k]
        L_.append("| %s | %d | %s | %s | %d | %d | %d |" % (SHORT[k], len(rs), R.fmt(R.boot([np.array([rr[(r, k)][0]]) for r in rs], np.mean, rng), 1),
                                                      R.fmt(R.boot([np.array([rr[(r, k)][1]]) for r in rs], np.mean, rng), 1), sum(rr[(r, k)][2] == "Completed" for r in rs),
                                                      sum(c["kind"].startswith("hit") for c in cc), sum(c["kind"] == "red light" for c in cc)))
    # collisions
    L_.append("\n## Every collision / red light of the run, by arm (kind, object, ego speed m/s, ego distance to lane centre m, turn window index or - = outside any turn window)\n")
    L_.append("| arm | route | t s | kind | object | v | off-centre m | turn |\n|---|---|---|---|---|---|---|---|")
    for c in sorted(cols, key=lambda c: (ARMS.index(c["arm"]), c["route"], c["t"])):
        L_.append("| %s | %s | %.1f | %s | %s | %.1f | %.1f | %s |" % (SHORT[c["arm"]], c["route"], c["t"], c["kind"], c["what"], c["v"], c["dev"], c["turn"]))
    L_.append("\n### Collision classes (hit events only)\n\n| arm | n | vehicle, ego within 1.75 m of centre | vehicle, ego outside 1.75 m | static / layout | pedestrian | in a turn window | outside any window |\n|---|---|---|---|---|---|---|---|")
    for k in ARMS:
        h = [c for c in cols if c["arm"] == k and c["kind"].startswith("hit")]
        L_.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (SHORT[k], len(h), sum(c["kind"] == "hit vehicle" and c["dev"] <= HALF for c in h), sum(c["kind"] == "hit vehicle" and c["dev"] > HALF for c in h),
                                                          sum(c["kind"] == "hit layout" for c in h), sum(c["kind"] == "hit pedestrian" for c in h), sum(c["turn"] != "-" for c in h),
                                                          sum(c["turn"] == "-" for c in h)))
    # per-turn listing
    L_.append("\n## Every turn: branch (y / l / n), peak cross-track m, head peak / needed curvature, hits+red in window\n")
    L_.append("| route | turn | group | angle | R_min | kind | " + " | ".join(SHORT[k] for k in ARMS) + " |\n|---|---|---|---|---|---|" + "---|" * len(ARMS))
    for r, t in keys:
        x = ix[(r, t, "A")]
        cells = []
        for k in ARMS:
            z = ix[(r, t, k)]
            cells.append("%s %.1f, %.2f/%.2f, %d+%d" % (z["branch"][0], z["peak"], z["head_pk"], z["need"], z["coll"], z["red"]) if z["entered"] else "n")
        L_.append("| %s | %d | %s | %+.0f | %.1f | %s | " % (r, t, "forced" if x["forced"] else "choice", x["angle"], x["rmin"], x["kind"]) + " | ".join(cells) + " |")
    # panel figure
    if a.panel:
        pick = [(r, int(t)) for r, t in (x.split(":") for x in a.panel.split(","))]
        fig, axs = plt.subplots(2, 2, figsize=(13, 13))
        for ax, (rid, mi) in zip(axs.ravel(), pick):
            D, gd, psiD, turns = roads[rid]
            T = next(x for x in turns if x["mi"] == mi)
            a0, a1 = max(T["it0"] - 100, 0), min(T["i1"] + 80, len(D))
            Dp = D[a0:a1]
            tg = np.gradient(Dp, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            nr = np.stack([-tg[:, 1], tg[:, 0]], -1)
            poly = np.vstack([Dp + HALF * nr, (Dp - HALF * nr)[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], color="#dddddd", zorder=0, label="lane +-1.75 m about the dense centreline")
            txt = []
            for k in ("D", "s143nz", "s122nz", "s122"):
                tr, e = traj[(rid, mi, k)]
                near, dev = R.project(tr, D)
                sel = (near >= a0) & (near <= a1 - 1) & (dev < 30)
                ax.plot(tr[sel, 0], tr[sel, 1], color=COL[k], lw=2.0, ls="-" if "122" in k else "--", zorder=3, label=LABEL[k])
                z = ix[(rid, mi, k)]
                txt.append("%-6s %-5s peak %5.1f m  head %.2f / need %.2f 1/m  hits %d" % (SHORT[k], e["branch"], e["peak"] if np.isfinite(e["peak"]) else np.nan, z["head_pk"], z["need"], z["coll"]))
            L = lab[(rid, mi)]
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title("%s route %s turn %d: %+.0f deg, R_min %.0f m, %s" % ("FORCED" if L["forced"] == "1" else "choice", rid, mi, T["angle"], T["rmin"], L["kind"]), fontsize=9)
            ax.grid(alpha=.25)
            ax.text(0.0, -0.08, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="top", family="monospace")
        h, lb = axs.ravel()[0].get_legend_handles_labels()
        fig.legend(h, lb, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.97))
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=9)
        (out.parent / "figs").mkdir(exist_ok=True)
        fig.savefig(out.parent / "figs/junction_rig122_panels.png", dpi=105)
    (out / "junction_rig122_tables.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
