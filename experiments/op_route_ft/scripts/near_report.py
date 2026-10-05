#!/usr/bin/env python
"""rc-*-near staged read: the 9 small-set B2D turns (near_lane.SMALL; desire off, zones off, curv, seed 2). Box, repo root, .venv python.

Arms: `<arm>@doff` = 25-turn desire-off units ($DATA_DIR/runs/op_route_ft/desire_off/<arm>/b2d/turns-s2-k*), `<arm>@small` = the small-set units
(.../turns-small-s2-k*). Scored with the same code as pre_report.py (cllib.turn_rows -> junction_rig122_report.score_attempt) and rft_split.py
(turn-in timing), restricted to the 9 turns.
  near_report.py [--arms shipped@doff,rc-ctl-s0@doff,rc-bear-s0@doff,pilot-bear-fix@small,pilot-bear-near@small]
  -> experiments/op_route_ft/results/near_small.{md,json}, near_small_split.{md,json}
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
import near_lane as N  # noqa: E402

DOFF = G.data_dir() / "runs/op_route_ft/desire_off"
RES = REPO / "experiments/op_route_ft/results"
_orig = C.b2d_dirs


def b2d_dirs(candidate, mode, kind, seed, force=False):
    for sfx, k in (("@doff", kind), ("@small", kind + "-small")):
        if candidate.endswith(sfx):
            return sorted((DOFF / candidate[: -len(sfx)] / "b2d").glob("%s-s%d-k*" % (k, seed)))
    return _orig(candidate, mode, kind, seed, force)


C.b2d_dirs = b2d_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="shipped@doff,rc-ctl-s0@doff,rc-bear-s0@doff,pilot-bear-fix@small,pilot-bear-near@small")
    a = ap.parse_args()
    arms = a.arms.split(",")
    keys = [k for k in C.turn_keys() if k[0] in N.SMALL]
    out, rows_all = {}, {}
    for arm in arms:
        rows, _, routes, _ = C.turn_rows(arm, "subset")
        ix = {(r["route"], r["turn"]): r for r in rows}
        rows_all[arm] = ix
        sc = [k for k in keys if k in ix]
        out[arm] = dict(scored=len(sc), took=sum(ix[k]["branch"] == "yes" for k in sc),
                        choice=sum(ix[k]["branch"] == "yes" for k in sc if not int(ix[k]["forced"])),
                        forced=sum(ix[k]["branch"] == "yes" for k in sc if int(ix[k]["forced"])), entered=sum(bool(ix[k]["entered"]) for k in sc),
                        coll=sum(ix[k]["coll"] for k in sc), per_turn={"%s:%d" % k: ix[k]["branch"] for k in sc})
    import rft_split
    sp = RES / "near_small_split"
    sys.argv = ["rft_split.py", *arms, "--out", str(sp)]
    rft_split.main()
    S = [r for r in json.load(open(str(sp) + ".json")) if (r["route"], r["turn"]) in set(keys)]
    for arm in arms:
        ss = [r for r in S if r["arm"] == arm]
        tin = [r["tin_m"] for r in ss if r.get("entered") and r.get("tin_m") is not None and np.isfinite(r["tin_m"])]
        out[arm].update(tin_median_m=float(np.median(tin)) if tin else None, n_tin=len(tin), late=sum(r["cause"] == "late" for r in ss),
                        peak_over_need=float(np.median([r["peak"] / r["need"] for r in ss if r.get("peak") is not None and r.get("need")]))
                        if ss else None, causes={r["route"] + ":" + str(r["turn"]): r["cause"] for r in ss},
                        v_med=float(np.median([r["v_med"] for r in ss if r.get("v_med") is not None])) if ss else None)
    json.dump(dict(keys=[list(k) for k in keys], arms=out), open(RES / "near_small.json", "w"), indent=1, default=float)
    L = ["# rc-*-near staged read: 9 of the 25 B2D turns (desire off, zones off, curv, seed 2)", "",
         "| arm | took (9) | choice (5) | forced (4) | entered | late | turn-in arc vs turn start, median m (n) | peak / needed curvature, median | v median in window | collisions |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in arms:
        r = out[arm]
        L.append("| %s | %d / %d | %d | %d | %d | %d | %s (%d) | %s | %s | %d |" % (
            arm, r["took"], r["scored"], r["choice"], r["forced"], r["entered"], r["late"],
            "%.1f" % r["tin_median_m"] if r["tin_median_m"] is not None else "-", r["n_tin"],
            "%.2f" % r["peak_over_need"] if r["peak_over_need"] is not None else "-", "%.2f" % r["v_med"] if r["v_med"] is not None else "-", r["coll"]))
    L += ["", "Per turn (exit taken / cause):", "", "| turn | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for k in keys:
        kk = "%s:%d" % k
        L.append("| %s | " % kk + " | ".join("%s / %s" % (out[a_]["per_turn"].get(kk, "-"), out[a_]["causes"].get(kk, "-")) for a_ in arms) + " |")
    (RES / "near_small.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
