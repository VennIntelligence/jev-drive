"""Report + BEV panels of the plan-tracking (p7) B2D 25-turn runs (plan_track_lane.py) paired against the action-curvature (curv) runs. Runs on the box.

    .venv/bin/python experiments/op_route_ft/scripts/plan_track_report.py report   -> results/plan_track.body.md, .json
    .venv/bin/python experiments/op_route_ft/scripts/plan_track_report.py panels [--pick 10255:0,...]  -> figs/plan_track_panels.png
Conditions: curv = desire-off action-curvature units ($DATA_DIR/runs/op_route_ft/desire_off/<arm>; rc-poly-s0 has none: its curv reference is the guard's
desire-ON unit), p7 / hyb = $DATA_DIR/runs/op_route_ft/plan_track/<arm>[-hyb] (desire off). Per-turn readouts as desire_off_report.py (rft_split.split_one +
junction_rig122_report.score_attempt).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_route_ft/scripts")]
import rft_split as S  # noqa: E402
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402
import junction_forced_report as F  # noqa: E402
import junction_rig122_report as J  # noqa: E402
import junction_cl_report as R  # noqa: E402

DATA = G.data_dir() / "runs/op_route_ft"
ARMS = ["shipped", "rc-ctl-s0", "rc-bear-s0", "rc-poly-s0"]
CONDS = ["curv", "p7", "hyb"]


def dirs_of(arm, cond):
    if cond == "curv":
        return sorted((DATA / "desire_off" / arm / "b2d").glob("turns-s2-k*")) or C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED)
    a = arm + "-hyb" if cond == "hyb" else arm
    return sorted((DATA / "plan_track" / a / "b2d").glob("turns-s2-k*"))


def collect(lab, arms=ARMS, conds=CONDS):
    keys = set(C.turn_keys())
    res, runs, geo, att_of = {}, {}, {}, {}
    for arm in arms:
        for cond in conds:
            if cond == "hyb" and arm != "shipped":
                continue
            for rid in C.TURN_ROUTES:
                att, _ = C.attempt_of(dirs_of(arm, cond), rid)
                if att is None or not (att / "route.json").exists() or not (att / "ticks.jsonl").exists():
                    continue
                g = geo.setdefault(rid, J.route_geometry(att, rid, lab))
                sc = J.score_attempt(rid, "x", att, g, lab)
                if sc is None:
                    continue
                att_of[(arm, cond, rid)] = att
                runs[(arm, cond, rid)] = dict(ds=sc["rr"][0], rc=sc["rr"][1], status=sc["rr"][2], run_coll=sum(c["kind"].startswith("hit") for c in sc["cols"]))
                pd_ = {r["turn"]: r for r in sc["rows"]}
                for r in S.split_one(rid, arm, att, g, lab):
                    if (rid, r["turn"]) in keys:
                        r.update(leaves=pd_[r["turn"]]["leaves"], coll=pd_[r["turn"]]["coll"], took=float(r["branch"] == "yes"))
                        r["steer"] = float(r["entered"] and r["s_peak"] >= 0.5 * r["need"] and r["s_peak"] > -r["s_neg"])
                        res[(arm, cond, rid, r["turn"])] = r
    return res, runs, geo, att_of


def rows(res, arm, cond, sel=lambda r: True):
    return {k[2:]: r for k, r in res.items() if k[:2] == (arm, cond) and sel(r)}


def pair(res, a, b, f, sel=lambda r: True):
    ra, rb = rows(res, *a, sel), rows(res, *b, sel)
    ks = sorted(set(ra) & set(rb))
    if not ks:
        return None, None, 0
    d, ci = C.paired_ci([f(ra[k]) for k in ks], [f(rb[k]) for k in ks], [k[0] for k in ks])
    return d, ci, len(ks)


def fmt(p):
    d, ci, n = p
    return "-" if d is None else "%+.2f [%+.2f, %+.2f] (n %d)" % (d, ci[0], ci[1], n)


def report(a):
    lab = F.load_labels()
    res, runs, _, _ = collect(lab)
    Path(a.out + ".json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in res.items()}, indent=1, default=float))
    ch, fo = (lambda r: r["forced"] == 0), (lambda r: r["forced"] == 1)
    L = ["## Per arm and executor (25 turns: 13 choice, 12 forced; desire off)\n",
         "took = within 5 m of the dense exit; leaves lane = peak cross-track > 1.75 m, share of entered turns; peak = peak cross-track in the turn window, median over entered turns (m); "
         "collisions = window collisions over the 25 turns; turn-in = arc past the turn start at the first plan step with act_k >= 0.5 / R_min towards the commanded side "
         "(median over turns that steer; the action head's own curvature, also logged under p7); v med = median speed in the 15 s window after entry, stop = share of time < 0.3 m/s; "
         "route DS = mean over the routes finished.\n",
         "| arm | executor | took choice (13) | took forced (12) | took all | entered | leaves lane | peak m, median | window collisions | turn-in m, median (n) | v med m/s | stop share | route DS (n) |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in ARMS:
        for cond in CONDS:
            R_ = rows(res, arm, cond)
            if not R_:
                continue
            v = list(R_.values())
            ent = [r for r in v if r["entered"]]
            tin = [r["tin_m"] for r in ent if r["tin_m"] is not None]
            ds = [x["ds"] for k, x in runs.items() if k[:2] == (arm, cond)]
            L.append("| %s | %s%s | %d / %d | %d / %d | %d / %d | %d | %d / %d | %.2f | %d | %s | %.1f | %.2f | %.1f (%d) |" % (
                arm, cond, " (desire ON)" if (arm == "rc-poly-s0" and cond == "curv") else "", sum(r["took"] for r in v if ch(r)), sum(ch(r) for r in v),
                sum(r["took"] for r in v if fo(r)), sum(fo(r) for r in v), sum(r["took"] for r in v), len(v), len(ent),
                sum(r["leaves"] == 1 for r in ent), len(ent), np.nanmedian([r["peak"] for r in ent]) if ent else np.nan, sum(r["coll"] for r in v),
                ("%.1f (%d)" % (np.median(tin), len(tin))) if tin else "-", np.nanmedian([r["v_med"] for r in ent]) if ent else np.nan,
                np.mean([r["stop_frac"] for r in ent]) if ent else np.nan, np.mean(ds) if ds else np.nan, len(ds)))
    took, ent_ = (lambda r: r["took"]), (lambda r: float(r["entered"]))
    L += ["\n## Paired differences (turns paired, route-cluster bootstrap, 95%)\n", "| contrast | took, all | took, choice | took, forced | entered |", "|---|---|---|---|---|"]
    def line(name, x, y):
        L.append("| %s | %s | %s | %s | %s |" % (name, fmt(pair(res, x, y, took)), fmt(pair(res, x, y, took, ch)), fmt(pair(res, x, y, took, fo)), fmt(pair(res, x, y, ent_))))
    for arm in ARMS:
        line("%s: p7 - curv" % arm, (arm, "p7"), (arm, "curv"))
    line("shipped: hyb - curv", ("shipped", "hyb"), ("shipped", "curv"))
    line("shipped: hyb - p7", ("shipped", "hyb"), ("shipped", "p7"))
    line("p7: rc-bear - rc-ctl (does the command matter?)", ("rc-bear-s0", "p7"), ("rc-ctl-s0", "p7"))
    line("curv: rc-bear - rc-ctl", ("rc-bear-s0", "curv"), ("rc-ctl-s0", "curv"))
    for arm in ARMS[1:]:
        line("p7: %s - shipped" % arm, (arm, "p7"), ("shipped", "p7"))
    ds_rows = ["\n## Route DS paired (p7 - curv, routes with both)\n", "| arm | mean diff DS | n |", "|---|--:|--:|"]
    for arm in ARMS:
        ks = [r for r in C.TURN_ROUTES if (arm, "p7", r) in runs and (arm, "curv", r) in runs]
        if ks:
            d, ci = C.paired_ci([runs[(arm, "p7", r)]["ds"] for r in ks], [runs[(arm, "curv", r)]["ds"] for r in ks], ks)
            ds_rows.append("| %s | %+.1f [%+.1f, %+.1f] | %d |" % (arm, d, ci[0], ci[1], len(ks)))
    L += ds_rows
    conds = [(arm, c) for arm in ARMS for c in CONDS if rows(res, arm, c)]
    L += ["\n## Per turn (Y/n took, peak cross-track m, turn-in m)\n", "| route | turn | forced | kind | side | R_min | " + " | ".join("%s %s" % c for c in conds) + " |",
          "|---|--:|--:|---|---|--:|" + "---|" * len(conds)]
    for k in sorted({k[2:] for k in res}):
        b = next(r for kk, r in res.items() if kk[2:] == k)
        cells = []
        for c in conds:
            r = res.get(c + k)
            cells.append("-" if r is None else ("not entered" if not r["entered"] else "%s %.1f %s" % ("Y" if r["took"] else "n", r["peak"], r["tin_m"])))
        L.append("| %s | %d | %d | %s | %s | %.1f | %s |" % (k[0], k[1], b["forced"], b["kind"], b["side"], b["rmin"], " | ".join(cells)))
    Path(a.out + ".body.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


COL = {"shipped|curv": "#888888", "shipped|p7": "#000000", "rc-bear-s0|curv": "#f4a6a6", "rc-bear-s0|p7": "#d62728"}


def panels(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = F.load_labels()
    series = [("shipped", "curv"), ("shipped", "p7"), ("rc-bear-s0", "curv"), ("rc-bear-s0", "p7")]
    res, runs, geo, att_of = collect(lab, ["shipped", "rc-bear-s0"], ["curv", "p7"])
    traj = {}
    for (arm, cond, rid), att in att_of.items():
        sc = J.score_attempt(rid, "x", att, geo[rid], lab)
        for mi, tr in sc["traj"].items():
            traj[(rid, mi, arm, cond)] = tr
    keys = sorted({k[2:] for k in res})
    if a.pick:
        pick = [(r, int(t)) for r, t in (x.split(":") for x in a.pick.split(","))]
    else:                       # turns where the executors differ most in outcome, choice turns first
        def score(k):
            tk = [res.get(s + k, {}).get("took", 0) for s in series]
            return (-(max(tk) - min(tk)), res.get(series[0] + k, {}).get("forced", 1), -res.get(series[3] + k, {}).get("entered", 0))
        pick = [k for k in sorted(keys, key=score) if all(s + k in res and res[s + k]["entered"] for s in series)][:4]
    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    for ax, (rid, mi) in zip(axs.ravel(), pick):
        D, gd, psiD, turns = geo[rid]
        T = next(x for x in turns if x["mi"] == mi)
        a0, a1 = max(T["it0"] - 100, 0), min(T["i1"] + 80, len(D))
        Dp = D[a0:a1]
        tg = np.gradient(Dp, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nr = np.stack([-tg[:, 1], tg[:, 0]], -1)
        ax.fill(*np.vstack([Dp + 1.75 * nr, (Dp - 1.75 * nr)[::-1]]).T, color="#dddddd", zorder=0, label="route lane +-1.75 m")
        txt = []
        for arm, cond in series:
            if (rid, mi, arm, cond) not in traj:
                continue
            tr, e = traj[(rid, mi, arm, cond)]
            near, dev = R.project(tr, D)
            sel = (near >= a0) & (near <= a1 - 1) & (dev < 30)
            ax.plot(tr[sel, 0], tr[sel, 1], color=COL["%s|%s" % (arm, cond)], lw=2.0, ls="-" if cond == "p7" else "--", zorder=3, label="%s %s" % (arm, cond))
            r = res[(arm, cond, rid, mi)]
            txt.append("%-11s %-4s %-5s peak %5.1f m  v_med %.1f  stop %.2f  hits %d" % (arm, cond, e["branch"], e["peak"] if np.isfinite(e["peak"]) else np.nan, r["v_med"], r["stop_frac"], r["coll"]))
        L = lab[(rid, mi)]
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("%s route %s turn %d: %+.0f deg, R_min %.0f m, %s" % ("FORCED" if L["forced"] == "1" else "choice", rid, mi, T["angle"], T["rmin"], L["kind"]), fontsize=9)
        ax.grid(alpha=.25)
        ax.text(0.0, -0.08, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="top", family="monospace")
    h, lb = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=9)
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=100)
    print("wrote", a.fig, pick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("report", "panels"))
    ap.add_argument("--out", default=str(REPO / "experiments/op_route_ft/results/plan_track"))
    ap.add_argument("--fig", default=str(REPO / "experiments/op_route_ft/figs/plan_track_panels.png"))
    ap.add_argument("--pick", default="")
    a = ap.parse_args()
    report(a) if a.cmd == "report" else panels(a)
