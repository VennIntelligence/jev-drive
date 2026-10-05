"""Report of the desire-off B2D 25-turn runs (decision 128 suspect b). Runs on the box (needs the attempt dirs).

    .venv/bin/python experiments/op_route_ft/scripts/desire_off_report.py --out experiments/op_route_ft/results/desire_off
Conditions per arm: on = the existing guard b2d_turns units (shipped: the rig122 olnz cache), off = $DATA_DIR/runs/op_route_ft/desire_off/<arm>.
Per-turn readouts: rft_split.split_one (took, steering, turn-in) + junction_rig122_report.score_attempt (leaves lane, window collisions).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_route_ft/scripts")]
import rft_split as S  # noqa: E402  (also puts the guard / closed-loop script dirs on sys.path)
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402
import junction_forced_report as F  # noqa: E402
import junction_rig122_report as J  # noqa: E402

ARMS = ["shipped", "rc-ctl-s0", "rc-bear-s0"]
OFF = G.data_dir() / "runs/op_route_ft/desire_off"


def dirs_of(arm, cond):
    return C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED) if cond == "on" else sorted((OFF / arm / "b2d").glob("turns-s%d-k*" % C.TURN_SEED))


def collect(lab):
    keys = set(C.turn_keys())
    res, geo, runs = {}, {}, {}
    for arm in ARMS:
        for cond in ("on", "off"):
            for rid in C.TURN_ROUTES:
                att, _ = C.attempt_of(dirs_of(arm, cond), rid)
                if att is None or not (att / "route.json").exists() or not (att / "ticks.jsonl").exists():
                    continue
                g = geo.setdefault(rid, J.route_geometry(att, rid, lab))
                sc = J.score_attempt(rid, "x", att, g, lab)
                if sc is None:
                    continue
                runs[(arm, cond, rid)] = dict(ds=sc["rr"][0], status=sc["rr"][2], run_coll=sum(c["kind"].startswith("hit") for c in sc["cols"]))
                sp = {r["turn"]: r for r in S.split_one(rid, arm, att, g, lab)}
                pd_ = {r["turn"]: r for r in sc["rows"]}
                for t, r in sp.items():
                    if (rid, t) in keys:
                        r.update(leaves=pd_[t]["leaves"], coll=pd_[t]["coll"], took=float(r["branch"] == "yes"))
                        r["steer"] = float(r["entered"] and r["s_peak"] >= 0.5 * r["need"] and r["s_peak"] > -r["s_neg"])
                        res[(arm, cond, rid, t)] = r
    return res, runs


def rows(res, arm, cond, sel=lambda r: True):
    return {k[2:]: r for k, r in res.items() if k[:2] == (arm, cond) and sel(r)}


def pair(res, a, b, f, sel=lambda r: True):
    """mean of f over turns, a minus b, paired by turn; route-cluster bootstrap. (a, b) = (arm, cond)."""
    ra, rb = rows(res, *a, sel), rows(res, *b, sel)
    ks = sorted(set(ra) & set(rb))
    if not ks:
        return None, None, 0
    d, ci = C.paired_ci([f(ra[k]) for k in ks], [f(rb[k]) for k in ks], [k[0] for k in ks])
    return d, ci, len(ks)


def fmt(p):
    d, ci, n = p
    return "-" if d is None else "%+.2f [%+.2f, %+.2f] (n %d)" % (d, ci[0], ci[1], n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/op_route_ft/results/desire_off"))
    a = ap.parse_args()
    lab = F.load_labels()
    res, runs = collect(lab)
    Path(a.out + ".json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in res.items()}, indent=1, default=float))
    ch, fo = (lambda r: r["forced"] == 0), (lambda r: r["forced"] == 1)
    L = ["## Per arm and condition (25 turns: 13 choice, 12 forced)\n",
         "took = came within 5 m of the dense exit; steered = peak desired curvature towards the commanded side >= 0.5 / R_min and above the opposite side's peak (entered turns only, whole span); "
         "turn-in = arc of the car past the turn start at the first plan step with curvature >= 0.5 / R_min towards the commanded side (median over turns that steer; < 0 = before the turn start); "
         "leaves lane = peak cross-track > 1.75 m, share of entered turns; collisions = window collisions summed over the 25 turns.\n",
         "| arm | desire | took choice (13) | took forced (12) | took all | entered | steered, choice (entered) | steered, forced | wrong side | turn-in m, median (n) | leaves lane | window collisions | route DS mean |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in ARMS:
        for cond in ("on", "off"):
            R = rows(res, arm, cond)
            if not R:
                continue
            v = list(R.values())
            ent = [r for r in v if r["entered"]]
            tin = [r["tin_m"] for r in ent if r["tin_m"] is not None]
            ds = [x["ds"] for k, x in runs.items() if k[:2] == (arm, cond)]
            L.append("| %s | %s | %d / %d | %d / %d | %d / %d | %d | %d / %d | %d / %d | %d | %s | %d / %d | %d | %.1f (%d routes) |" % (
                arm, cond, sum(r["took"] for r in v if ch(r)), sum(ch(r) for r in v), sum(r["took"] for r in v if fo(r)), sum(fo(r) for r in v),
                sum(r["took"] for r in v), len(v), len(ent), sum(r["steer"] for r in ent if ch(r)), sum(ch(r) for r in ent),
                sum(r["steer"] for r in ent if fo(r)), sum(fo(r) for r in ent), sum(r["cause"] == "wrong side" for r in v) if "cause" in v[0] else sum(1 for r in ent if r["s_peak"] < 0.5 * r["need"] and r["s_neg"] < -0.5 * r["need"]),
                ("%.1f (%d)" % (np.median(tin), len(tin))) if tin else "-", sum(r["leaves"] for r in ent), len(ent), sum(r["coll"] for r in v), np.mean(ds) if ds else float("nan"), len(ds)))
    L += ["\n## Paired differences (turns paired, cluster bootstrap over routes, 95%)\n",
          "| contrast | took, all | took, choice | steered, choice | steered, all (0 if not entered) |", "|---|---|---|---|---|"]
    took, steer = (lambda r: r["took"]), (lambda r: r["steer"])
    for arm in ARMS:
        x, y = (arm, "off"), (arm, "on")
        L.append("| %s: desire off - on | %s | %s | %s | %s |" % (arm, fmt(pair(res, x, y, took)), fmt(pair(res, x, y, took, ch)), fmt(pair(res, x, y, steer, ch)), fmt(pair(res, x, y, steer))))
    for cond in ("off", "on"):
        x, y = ("rc-bear-s0", cond), ("rc-ctl-s0", cond)
        L.append("| rc-bear - rc-ctl, desire %s | %s | %s | %s | %s |" % (cond, fmt(pair(res, x, y, took)), fmt(pair(res, x, y, took, ch)), fmt(pair(res, x, y, steer, ch)), fmt(pair(res, x, y, steer))))
    for arm in ("rc-ctl-s0", "rc-bear-s0"):
        for cond in ("off", "on"):
            L.append("| %s - shipped, desire %s | %s | %s | %s | %s |" % (arm, cond, fmt(pair(res, (arm, cond), ("shipped", cond), took)), fmt(pair(res, (arm, cond), ("shipped", cond), took, ch)),
                                                                        fmt(pair(res, (arm, cond), ("shipped", cond), steer, ch)), fmt(pair(res, (arm, cond), ("shipped", cond), steer))))
    L += ["\n## Per turn (took / peak s over R_min-need ratio / turn-in m)\n", "| route | turn | forced | kind | side | R_min | " + " | ".join("%s %s" % (arm, c) for arm in ARMS for c in ("on", "off")) + " |",
          "|---|--:|--:|---|---|--:|" + "---|" * (2 * len(ARMS))]
    for k in sorted({k[2:] for k in res}):
        b = next(r for kk, r in res.items() if kk[2:] == k)
        cells = []
        for arm in ARMS:
            for cond in ("on", "off"):
                r = res.get((arm, cond) + k)
                cells.append("-" if r is None else ("not entered" if not r["entered"] else "%s %.1f %s" % ("Y" if r["took"] else "n", r["s_peak"] / r["need"], r["tin_m"])))
        L.append("| %s | %d | %d | %s | %s | %.1f | %s |" % (k[0], k[1], b["forced"], b["kind"], b["side"], b["rmin"], " | ".join(cells)))
    Path(a.out + ".body.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
