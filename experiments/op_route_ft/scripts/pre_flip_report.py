#!/usr/bin/env python
"""Command-flip control for rc-bear-pre-s0 (pre_flip_lane.py): took-exit and steered side on the 13 choice turns, flipped vs correct command
(both desire off, B2D, zones off, seed 2). Box, repo root, .venv python.

  .venv/bin/python experiments/op_route_ft/scripts/pre_flip_report.py -> experiments/op_route_ft/results/pre_flip.{md,json}
steered correct = peak desired curvature towards the TRUE exit side >= 0.5 / R_min and above the opposite peak; steered flipped = the mirror
(peak towards the opposite side >= 0.5 / R_min and above the true-side peak). Entered turns only for the steering columns.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_route_ft/scripts"),
                str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402
import junction_forced_report as F  # noqa: E402
import junction_rig122_report as J  # noqa: E402
import rft_split as S  # noqa: E402

RUNS = G.data_dir() / "runs/op_route_ft"
ARM = "rc-bear-pre-s0"
CHOICE_ROUTES = "10255 15102 24758 24944 26872 27297 28008 28147 334 34183 5423 6999 9196".split()   # routes with a choice (forced 0) turn
CONDS = {"correct": RUNS / "desire_off" / ARM / "b2d", "flipped": RUNS / "pre/flip" / ARM / "b2d"}


def collect(lab):
    res = {}
    for cond, base in CONDS.items():
        dirs = sorted(base.glob("turns-s%d-k*" % C.TURN_SEED))
        for rid in CHOICE_ROUTES:
            att, _ = C.attempt_of(dirs, rid)
            if att is None or not (att / "route.json").exists() or not (att / "ticks.jsonl").exists():
                continue
            g = J.route_geometry(att, rid, lab)
            for r in S.split_one(rid, ARM, att, g, lab):
                if r["forced"] == 0:
                    r["steer_true"] = int(r["entered"] and r["s_peak"] >= 0.5 * r["need"] and r["s_peak"] > -r["s_neg"])
                    r["steer_flip"] = int(r["entered"] and -r["s_neg"] >= 0.5 * r["need"] and -r["s_neg"] > r["s_peak"])
                    r["took"] = int(r["branch"] == "yes")
                    res[(cond, rid, r["turn"])] = r
    return res


def main():
    res = collect(F.load_labels())
    (REPO / "experiments/op_route_ft/results/pre_flip.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in res.items()}, indent=1, default=float))
    L = ["# rc-bear-pre-s0 with the command flipped (left <-> right), B2D choice turns, desire off, zones off, seed 2", "",
         "| command | choice turns | entered | took the TRUE exit | steered to the true side | steered to the flipped side | neither | route-level mean peak s/need true, flipped |",
         "|---|--:|--:|--:|--:|--:|--:|---|"]
    for cond in CONDS:
        v = [r for k, r in res.items() if k[0] == cond]
        ent = [r for r in v if r["entered"]]
        tr, fl = sum(r["steer_true"] for r in ent), sum(r["steer_flip"] for r in ent)
        pk = [r["s_peak"] / r["need"] for r in ent]
        ng = [-r["s_neg"] / r["need"] for r in ent]
        L.append("| %s | %d | %d | %d | %d | %d | %d | %.2f, %.2f |" % (cond, len(v), len(ent), sum(r["took"] for r in v), tr, fl, len(ent) - tr - fl,
                                                                         sum(pk) / max(len(pk), 1), sum(ng) / max(len(ng), 1)))
    L += ["", "Per turn (T = took the true exit, s_true / s_flip = peak desired curvature over R_min-need towards the true / the opposite side):", "",
          "| route | turn | kind | true side | R_min | correct: took, s_true, s_flip | flipped: took, s_true, s_flip |", "|---|--:|---|---|--:|---|---|"]
    for key in sorted({k[1:] for k in res}):
        cells = []
        for cond in CONDS:
            r = res.get((cond,) + key)
            cells.append("-" if r is None else ("not entered" if not r["entered"] else "%s %.2f %.2f" % ("T" if r["took"] else "n", r["s_peak"] / r["need"], -r["s_neg"] / r["need"])))
        b = next(r for k, r in res.items() if k[1:] == key)
        L.append("| %s | %d | %s | %s | %.1f | %s | %s |" % (key[0], key[1], b["kind"], b["side"], b["rmin"], cells[0], cells[1]))
    (REPO / "experiments/op_route_ft/results/pre_flip.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
