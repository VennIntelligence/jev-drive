#!/usr/bin/env python
"""op_wide_ft B2D report (plans/2026-10-05-wide-ft-prereg.md). Box, repo root, .venv python.

Arms `<tag>@<fov>` = units of wide_lane.py under $DATA_DIR/runs/op_wide_ft/b2d/<arm>/b2d/<set>-s2-k*; `shipped@doff` and other op_route_ft desire-off
arms (`<arm>@doff`) are read from $DATA_DIR/runs/op_route_ft/desire_off as reference rows. Scored with op_route_ft's code: took / entered / collisions
(cllib.turn_rows -> junction_rig122_report.score_attempt) and the steering split (rft_split.split_one: s_peak = peak desired curvature towards the
commanded side over the turn span, need = 1 / R_min, tin_m = arc of the car past the turn start when it first steers 0.5 / R_min).

  wide_report.py small [--arms ...]   9-turn small set + the pre-registered gate -> experiments/op_wide_ft/results/small.{md,json}
  wide_report.py turns [--arms ...]   25 turns, paired contrasts (route-cluster bootstrap) -> results/turns.{md,json}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_route_ft/scripts"),
                str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402
from near_lane import SMALL  # noqa: E402

W = G.data_dir() / "runs/op_wide_ft"
DOFF = G.data_dir() / "runs/op_route_ft/desire_off"
RES = REPO / "experiments/op_wide_ft/results"
CAPPED = {("28180", 0)}                    # needs peak curvature 0.213 > op-path MAX_CURVATURE 0.2 (decision 135): reported apart
SUB = {"v": "turns"}
_orig = C.b2d_dirs


def b2d_dirs(candidate, mode, kind, seed, force=False):
    if candidate.endswith("@doff"):
        return sorted((DOFF / candidate[:-5] / "b2d").glob("%s-s%d-k*" % (kind, seed)))
    if "@" in candidate:
        return sorted((W / "b2d" / candidate / "b2d").glob("%s-s%d-k*" % (SUB["v"], seed)))
    return _orig(candidate, mode, kind, seed, force)


C.b2d_dirs = b2d_dirs


def collect(arms, keys):
    """Per (arm, turn): took / entered / collisions + the steering split."""
    import junction_forced_report as F
    import junction_rig122_report as J
    import rft_split as S
    lab = F.load_labels()
    out = {}
    for arm in arms:
        rows, _, routes, _ = C.turn_rows(arm, "subset")
        ix = {(r["route"], int(r["turn"])): r for r in rows}
        sp = {}
        dirs = C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED)
        for rid in sorted({k[0] for k in keys}):
            att, _ = C.attempt_of(dirs, rid)
            if att is None or not (att / "route.json").exists() or not (att / "ticks.jsonl").exists():
                continue
            for r in S.split_one(rid, arm, att, J.route_geometry(att, rid, lab), lab):
                r["cause"] = S.classify(r)
                sp[(rid, int(r["turn"]))] = r
        out[arm] = {k: dict(took=float(ix[k]["branch"] == "yes"), entered=bool(ix[k]["entered"]), coll=int(ix[k]["coll"]),
                            forced=int(ix[k]["forced"]), **{f: sp[k].get(f) for f in ("cause", "s_peak", "need", "tin_m", "rmin", "angle", "v_med")}
                            if k in sp else {})
                    for k in keys if k in ix}
    return out


def ratio(r):
    return r["s_peak"] / r["need"] if r.get("entered") and r.get("s_peak") is not None and np.isfinite(r["s_peak"]) else np.nan


def summary(d, keys):
    sc = [k for k in keys if k in d]
    tins = [d[k]["tin_m"] for k in sc if d[k].get("entered") and d[k].get("tin_m") is not None]
    rf = [ratio(d[k]) for k in sc if d[k]["forced"] and k not in CAPPED]
    rc = [ratio(d[k]) for k in sc if not d[k]["forced"]]
    return dict(scored=len(sc), took=int(sum(d[k]["took"] for k in sc)), took_choice=int(sum(d[k]["took"] for k in sc if not d[k]["forced"])),
                took_forced=int(sum(d[k]["took"] for k in sc if d[k]["forced"])), entered=int(sum(d[k]["entered"] for k in sc)),
                coll=int(sum(d[k]["coll"] for k in sc)), late=int(sum(d[k].get("cause") == "late" for k in sc)),
                tin_med=float(np.median(tins)) if tins else None, n_tin=len(tins),
                ratio_forced_med=float(np.nanmedian(rf)) if np.isfinite(rf).any() else None,
                ratio_choice_med=float(np.nanmedian(rc)) if np.isfinite(rc).any() else None,
                ratio_all_med=float(np.nanmedian([ratio(d[k]) for k in sc if k not in CAPPED])) if sc else None)


def paired(a, b, keys, f, sel=lambda k, r: True):
    ks = [k for k in keys if k in a and k in b and sel(k, a[k])]
    x, y = np.array([f(a[k]) for k in ks], float), np.array([f(b[k]) for k in ks], float)
    ok = np.isfinite(x) & np.isfinite(y)
    m, ci = C.paired_ci(x[ok], y[ok], groups=np.array([k[0] for k in ks])[ok])
    return dict(n=int(ok.sum()), mean=m, ci=ci, mean_a=float(np.mean(x[ok])) if ok.any() else None, mean_b=float(np.mean(y[ok])) if ok.any() else None)


def fmt(p):
    return "-" if p["mean"] is None else "%+.3f [%+.3f, %+.3f] (n %d)" % (p["mean"], p["ci"][0], p["ci"][1], p["n"])


def report(name, arms, keys, pairs, title):
    D = collect(arms, keys)
    S = {a: summary(D[a], keys) for a in arms}
    P = {}
    for x, y in pairs:
        if x in D and y in D:
            P[f"{x} - {y}"] = dict(
                took=paired(D[x], D[y], keys, lambda r: r["took"]),
                ratio_forced=paired(D[x], D[y], keys, ratio, lambda k, r: r["forced"] and k not in CAPPED),
                ratio_choice=paired(D[x], D[y], keys, ratio, lambda k, r: not r["forced"]),
                tin_m=paired(D[x], D[y], keys, lambda r: r["tin_m"] if r.get("tin_m") is not None else np.nan))
    L = [f"# {title}", "", "took = the car left through the commanded exit. ratio = peak desired curvature towards the commanded side over the turn span / "
         "needed 1 / R_min (entered turns; rft_split s_peak / need); forced-turn ratios exclude 28180 (needs 0.213 > op-path MAX_CURVATURE 0.2). "
         "turn-in = arc of the car past the turn start when the action first reaches 0.5 / R_min (negative = before). Desire off, zones off, curv "
         "(op-path), spec camera 1.86 m, seed 2, one run per cell.", "",
         "| arm | took | choice | forced | entered | late | turn-in m, median (n) | ratio forced (median) | ratio choice (median) | collisions |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for a in arms:
        s = S[a]
        f2 = lambda v: "-" if v is None else "%.2f" % v  # noqa: E731
        L.append("| %s | %d / %d | %d | %d | %d | %d | %s (%d) | %s | %s | %d |" % (a, s["took"], s["scored"], s["took_choice"], s["took_forced"], s["entered"],
                                                                                 s["late"], f2(s["tin_med"]), s["n_tin"], f2(s["ratio_forced_med"]),
                                                                                 f2(s["ratio_choice_med"]), s["coll"]))
    if P:
        L += ["", "Paired (route-cluster bootstrap, jevdrive.stats.paired):", "", "| contrast | took | ratio forced | ratio choice | turn-in m |", "|---|---|---|---|---|"]
        for c, p in P.items():
            L.append("| %s | %s | %s | %s | %s |" % (c, fmt(p["took"]), fmt(p["ratio_forced"]), fmt(p["ratio_choice"]), fmt(p["tin_m"])))
    L += ["", "Per turn (took / cause / ratio / turn-in m):", "", "| turn | forced | R_min | " + " | ".join(arms) + " |", "|---|--:|--:|" + "---|" * len(arms)]
    for k in keys:
        b = next((D[a][k] for a in arms if k in D[a]), None)
        if b is None:
            continue
        cell = lambda r: "-" if r is None else "%s / %s / %s / %s" % ("yes" if r["took"] else "no", r.get("cause", "-"),  # noqa: E731
                                                                      "%.2f" % ratio(r) if np.isfinite(ratio(r)) else "-",
                                                                      "%.1f" % r["tin_m"] if r.get("tin_m") is not None else "-")
        L.append("| %s:%d%s | %d | %s | %s |" % (k[0], k[1], " (capped)" if k in CAPPED else "", b["forced"],
                                                 "%.1f" % b["rmin"] if b.get("rmin") is not None else "-", " | ".join(cell(D[a].get(k)) for a in arms)))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.md").write_text("\n".join(L) + "\n")
    json.dump(dict(keys=[list(k) for k in keys], summary=S, paired=P, per_turn={a: {"%s:%d" % k: v for k, v in D[a].items()} for a in arms}),
              open(RES / f"{name}.json", "w"), indent=1, default=float)
    print("\n".join(L))
    return S, P


def gate(S, P, ol):
    """Pre-registered gate (plans/2026-10-05-wide-ft-prereg.md, step 3) on the small set + the open-loop 2 x 2."""
    a, b = S["wf-w116-s0@116"], S["wf-w58-s0@58"]
    ga = a["took"] >= b["took"] + 2
    gb = (a["tin_med"] is not None and b["tin_med"] is not None and b["tin_med"] - a["tin_med"] >= 2.0) or \
         (a["ratio_all_med"] is not None and b["ratio_all_med"] is not None and a["ratio_all_med"] - b["ratio_all_med"] >= 0.3)
    gc = ol is not None and ol["diff_vs_w58"]["mean"] >= 0.05 and ol["diff_vs_w58"]["lo"] > 0 and ol["own_116_minus_58"] >= 0.03
    return dict(a=bool(ga), b=bool(gb), c=bool(gc), pass_=bool(ga or gb or gc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("small", "turns"))
    ap.add_argument("--arms", default="")
    ap.add_argument("--ol", default="", help="open-loop gate numbers (json from wide_ol.py) for the small-set gate")
    a = ap.parse_args()
    keys = C.turn_keys()
    if a.what == "small":
        SUB["v"] = "turns-small"
        arms = (a.arms or "shipped@doff,wf-w58-s0@58,wf-w116-s0@116").split(",")
        S, P = report("small", arms, [k for k in keys if k[0] in SMALL], [("wf-w116-s0@116", "wf-w58-s0@58")],
                      "op_wide_ft small set: 9 of the 25 B2D turns (gate read)")
        ol = json.load(open(a.ol)) if a.ol and Path(a.ol).exists() else None
        g = gate(S, P, ol)
        print("GATE", json.dumps(g), "open loop", json.dumps(ol))
        json.dump(dict(gate=g, ol=ol), open(RES / "gate.json", "w"), indent=1)
    else:
        arms = (a.arms or "shipped@doff,wf-w58-s0@58,wf-w116-s0@116,wf-w116-s0@58").split(",")
        report("turns", arms, keys, [("wf-w116-s0@116", "wf-w58-s0@58"), ("wf-w116-s0@116", "wf-w116-s0@58"), ("wf-w116-s0@58", "wf-w58-s0@58"),
                                     ("wf-w58-s0@58", "shipped@doff"), ("wf-w116-s0@116", "shipped@doff")],
               "op_wide_ft: 25 B2D junction turns, wide 116 vs 58.7 deg")


if __name__ == "__main__":
    main()
