"""Failure split of the 25 B2D junction turns per arm: not chosen / chosen late / chosen but little / crawl-stop / wrong side.

    .venv/bin/python experiments/op_route_ft/scripts/rft_split.py shipped rc-ctl-s0 rc-bear-s0 [...] --out experiments/op_route_ft/results/split
    -> <out>.md (tables) and <out>.json (per turn x arm). Runs on the box (needs the attempt dirs).

Per entered turn (window = entry .. exit of C.eval_turn, <= 15 s): commanded side from the labelled angle (> 0 = right in CARLA's frame,
checked against the logged desire pulse), signed desired curvature s = act_k * (+1 left / -1 right) (convention checked on the turns shipped took),
peak signed s over the approach + window, turn-in time = first plan step with s >= 0.5 / R_min (relative to entry), speed stats in the window.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402
import junction_forced_report as F  # noqa: E402
import junction_rig122_report as J  # noqa: E402
import junction_cl_report as R  # noqa: E402

APPROACH_S = 6.0          # plan steps from this long before entry count as approach
WIN_S = 15.0


def rows_of(p):
    return [json.loads(x) for x in open(p)]


def split_one(rid, arm, att, geom, lab):
    D, gd, psiD, turns = geom
    tk = [t for t in rows_of(att / "ticks.jsonl") if "truth" in t]
    tr = np.array([t["truth"][:2] for t in tk])
    tt = np.array([t["t"] for t in tk])
    v = np.array([t["v"] for t in tk])
    pl = [p for p in rows_of(att / "plans.jsonl") if not p.get("warm") and "act_k" in p]
    pt, pk = np.array([p["t"] for p in pl]), np.array([p["act_k"] for p in pl])
    pdz = np.array([p.get("desire") or 0 for p in pl])
    near, dev = R.project(tr, D)
    out = []
    for T in turns:
        e = R.eval_turn(tr, np.array([t["truth"][2] for t in tk]), D, psiD, T)
        L = lab[(rid, T["mi"])]
        right = T["angle"] > 0
        d = dict(route=rid, turn=T["mi"], arm=arm, forced=int(L["forced"]), kind=L["kind"], angle=round(T["angle"], 0), rmin=round(T["rmin"], 1),
                 need=1 / T["rmin"], entered=e["entered"], branch=e["branch"], side="R" if right else "L")
        if e["entered"]:
            a, b = e["span"]
            b = min(b, a + int(WIN_S * 20)) if len(tt) > 1 else b
            t0, t1 = tt[a], tt[min(b, len(tt) - 1)]
            sg = (-1.0 if right else 1.0)
            sel = (pt >= t0 - APPROACH_S) & (pt <= t1)
            s = pk[sel] * sg
            pw = (pt >= t0) & (pt <= t1)
            vw = v[a:b + 1]
            tin = pt[sel][np.where(s >= 0.5 * d["need"])[0]]
            des = pdz[sel]
            want = 2 if right else 1
            # stopped inside the window: runs of v < 0.3 m/s longer than 1 s
            stop = (vw < 0.3)
            dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.05
            d.update(win_s=round(float(t1 - t0), 1), v_entry=round(float(v[a]), 1), v_med=round(float(np.median(vw)), 1), v_min=round(float(vw.min()), 1),
                     stop_frac=round(float(stop.mean()), 2), stop_s=round(float(stop.sum() * dt), 1),
                     s_peak=round(float(s.max()), 3) if len(s) else np.nan, s_neg=round(float(s.min()), 3) if len(s) else np.nan,
                     s_win_peak=round(float((pk[pw] * sg).max()), 3) if pw.any() else np.nan,
                     tin=round(float(tin[0] - t0), 1) if len(tin) else None, desire_ok=round(float((des == want).mean()), 2) if len(des) else np.nan,
                     peak=round(e["peak"], 2))
        out.append(d)
    return out


def classify(d):
    """First matching cause of a lost turn, in this order (heuristic, thresholds in the md header)."""
    if not d["entered"]:
        return "never entered"
    if d["branch"] == "yes":
        return "took"
    n = d["need"]
    if d["s_peak"] < 0.5 * n and d["s_neg"] < -0.5 * n:
        return "wrong side"
    if d["s_peak"] < 0.5 * n:
        return "not chosen"
    if d["stop_frac"] > 0.25 or d["v_med"] < 1.5:
        return "crawl / stop"
    if d["tin"] is None or d["tin"] > 0.5:
        return "late"
    return "too little"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out", default=str(REPO / "experiments/op_route_ft/results/split"))
    a = ap.parse_args()
    lab = F.load_labels()
    res, geo = [], {}
    for arm in a.arms:
        dirs = C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED)
        for rid in C.TURN_ROUTES:
            att, _ = C.attempt_of(dirs, rid)
            if att is None or not (att / "route.json").exists() or not (att / "ticks.jsonl").exists():
                continue
            g = geo.setdefault(rid, J.route_geometry(att, rid, lab))
            keys = {t["mi"] for t in g[3]}
            res += [r for r in split_one(rid, arm, att, g, lab) if (rid, r["turn"]) in set(C.turn_keys())]
    for r in res:
        r["cause"] = classify(r)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    L = ["# Failure split of the 25 B2D junction turns (rft_split.py)\n",
         "Window = entry .. exit of the turn (<= 15 s). s = desired curvature act_k signed to the commanded side (+ = towards the exit side; L positive, R negative convention). "
         "Causes, first match: not entered; `wrong side` (peak s < 0.5 / R_min and the opposite sign reaches -0.5 / R_min); `not chosen` (peak s over approach 6 s + window < 0.5 / R_min); "
         "`crawl / stop` (steered, but stopped > 25% of the window or median speed < 1.5 m/s); `late` (first s >= 0.5 / R_min later than 0.5 s after entry or never); `too little` (steered in time, still lost).\n"]
    arms = list(dict.fromkeys(r["arm"] for r in res))
    causes = ["took", "not chosen", "wrong side", "crawl / stop", "late", "too little", "never entered"]
    L += ["## Cause counts (turns)\n", "| arm | group | n | " + " | ".join(causes) + " |", "|---|---|--:|" + "--:|" * len(causes)]
    for arm in arms:
        for g, sel in (("choice", lambda r: r["forced"] == 0), ("forced", lambda r: r["forced"] == 1), ("all", lambda r: True)):
            rr = [r for r in res if r["arm"] == arm and sel(r)]
            L.append("| %s | %s | %d | %s |" % (arm, g, len(rr), " | ".join(str(sum(r["cause"] == c for r in rr)) for c in causes)))
    L += ["\n## Entered turns: longitudinal and steering medians\n",
          "| arm | entered | v at entry (m/s) | median v in window | min v | stopped share of window | stopped s | window s | peak s approach+window (1/m) | peak s / needed | turn-in s after entry (median; n steered) | desire pulse matches side |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in arms:
        rr = [r for r in res if r["arm"] == arm and r["entered"]]
        if not rr:
            continue
        m = lambda k: np.nanmedian([r[k] for r in rr])  # noqa: E731
        tins = [r["tin"] for r in rr if r["tin"] is not None]
        L.append("| %s | %d | %.1f | %.1f | %.1f | %.2f | %.1f | %.1f | %.3f | %.2f | %s (%d) | %.2f |" % (
            arm, len(rr), m("v_entry"), m("v_med"), m("v_min"), m("stop_frac"), m("stop_s"), m("win_s"), m("s_peak"),
            np.nanmedian([r["s_peak"] / r["need"] for r in rr]), "%.1f" % np.median(tins) if tins else "-", len(tins), m("desire_ok")))
    L.append("\n## By R_min (entered turns; share lost / steered, median v in window)\n")
    L += ["| arm | R_min bin | n | took | steered (peak s >= 0.5 need) | median v in window | stopped share |", "|---|---|--:|--:|--:|--:|--:|"]
    for arm in arms:
        for nm, lo, hi in (("< 7 m", 0, 7), ("7-10 m", 7, 10), (">= 10 m", 10, 1e9)):
            rr = [r for r in res if r["arm"] == arm and r["entered"] and lo <= r["rmin"] < hi]
            if rr:
                L.append("| %s | %s | %d | %d | %d | %.1f | %.2f |" % (arm, nm, len(rr), sum(r["branch"] == "yes" for r in rr), sum(r["s_peak"] >= 0.5 * r["need"] for r in rr),
                                                                     np.median([r["v_med"] for r in rr]), np.mean([r["stop_frac"] for r in rr])))
    L.append("\n## Per turn\n")
    L += ["| route | turn | forced | kind | side | angle | R_min | " + " | ".join("%s: cause, s peak, v med, stop s, turn-in" % a_ for a_ in arms) + " |", "|---|--:|--:|---|---|--:|--:|" + "---|" * len(arms)]
    ix = {(r["route"], r["turn"], r["arm"]): r for r in res}
    for k in sorted({(r["route"], r["turn"]) for r in res}):
        b = next(ix[k + (a_,)] for a_ in arms if k + (a_,) in ix)
        cells = []
        for a_ in arms:
            r = ix.get(k + (a_,))
            cells.append("-" if r is None else ("%s" % r["cause"] if not r["entered"] else "%s, %.2f, %.1f, %.1f, %s" % (r["cause"], r["s_peak"], r["v_med"], r["stop_s"], r["tin"])))
        L.append("| %s | %d | %d | %s | %s | %d | %.1f | %s |" % (k[0], k[1], b["forced"], b["kind"], b["side"], b["angle"], b["rmin"], " | ".join(cells)))
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print("wrote", a.out + ".md")


if __name__ == "__main__":
    main()
