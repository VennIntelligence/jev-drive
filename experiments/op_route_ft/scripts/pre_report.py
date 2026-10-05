#!/usr/bin/env python
"""rc-*-pre closed-loop report (plans/2026-10-05-route-ft-prereg.md, 2026-10-06 section). Box, repo root, .venv python.

Arm names: a guard candidate (desire on, units under $DATA_DIR/runs/op_guard/<arm>/subset/b2d) or `<arm>@doff` (desire off, units under
$DATA_DIR/runs/op_route_ft/desire_off/<arm>/b2d, desire_off_lane.py / pre_lane.py). `shipped` falls back to decision 127's checked cache.

  pre_report.py table [--desire off,on]       took exit per arm, paired diffs -> experiments/op_route_ft/results/pre_turns.{md,json}
  pre_report.py split ARM ...                  rft_split.py on these arms (turn-in timing, causes) -> results/pre_split[_<tag>].{md,json}
  pre_report.py panels ARM ... [--pick r:t,..] rft_panels.py (BEV) -> figs/pre_b2d_turn_panels[_<tag>].png
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

DOFF = G.data_dir() / "runs/op_route_ft/desire_off"
RES = REPO / "experiments/op_route_ft/results"
_orig_dirs = C.b2d_dirs


def b2d_dirs(candidate, mode, kind, seed, force=False):
    if candidate.endswith("@doff"):
        return sorted((DOFF / candidate[:-5] / "b2d").glob("%s-s%d-k*" % (kind, seed)))
    return _orig_dirs(candidate, mode, kind, seed, force)


C.b2d_dirs = b2d_dirs
ARMS = ("shipped", "rc-ctl-s0", "rc-bear-s0", "rc-ctl-pre-s0", "rc-bear-pre-s0")
PAIRS = (("rc-bear-pre-s0", "rc-ctl-pre-s0"), ("rc-bear-pre-s0", "rc-bear-s0"), ("rc-ctl-pre-s0", "rc-ctl-s0"), ("rc-bear-pre-s0", "shipped"),
         ("rc-bear-s0", "rc-ctl-s0"))


def took(arm):
    rows, cols, routes, _ = C.turn_rows(arm, "subset")
    return {(r["route"], r["turn"]): r for r in rows}, cols, routes


def table(a):
    keys = C.turn_keys()
    out, L = {}, ["# rc-*-pre: 25 B2D junction turns (guard b2d_turns set, zones off, seed 2)", "",
                  "took = the car left through the commanded exit. Paired differences: route-cluster bootstrap (cllib.paired_ci), n 25, one seed, one run per cell.", ""]
    for de in a.desire.split(","):
        sfx = "@doff" if de == "off" else ""
        res = {}
        for arm in ARMS:
            ix, cols, routes = took(arm + sfx)
            if len(ix) < len(keys) or not all(k in ix for k in keys):
                res[arm] = dict(scored=sum(k in ix for k in keys))
                continue
            y = {k: int(ix[k]["branch"] == "yes") for k in keys}
            f = {k: int(ix[k]["forced"]) for k in keys}
            ent = [k for k in keys if ix[k]["entered"]]
            res[arm] = dict(scored=len(keys), took=sum(y.values()), choice=sum(y[k] for k in keys if f[k] == 0), forced=sum(y[k] for k in keys if f[k] == 1),
                            entered=len(ent), leaves=sum(int(ix[k]["leaves"] == 1) for k in ent), coll=sum(ix[k]["coll"] for k in keys),
                            ds=float(np.mean([v["ds"] for v in routes.values()])) if routes else None, y=[y[k] for k in keys])
        diffs = {}
        for x, b in PAIRS:
            if "y" in res.get(x, {}) and "y" in res.get(b, {}):
                m, ci = C.paired_ci(res[x]["y"], res[b]["y"], [k[0] for k in keys])
                gain = sum(p > q for p, q in zip(res[x]["y"], res[b]["y"]))
                lost = sum(p < q for p, q in zip(res[x]["y"], res[b]["y"]))
                diffs[f"{x} - {b}"] = dict(diff=m, lo=ci[0], hi=ci[1], gained=gain, lost=lost)
        out[de] = dict(arms=res, diffs=diffs, keys=[list(k) for k in keys])
        L += [f"## desire {de}", "", "| arm | took (25) | choice (13) | forced (12) | entered | leaves lane (of entered) | collisions in windows | route DS mean |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for arm in ARMS:
            r = res[arm]
            if "took" not in r:
                L.append(f"| {arm} | not run / incomplete ({r['scored']} / 25 scored) | | | | | | |")
                continue
            L.append("| %s | %d | %d | %d | %d | %d / %d | %d | %.1f |" % (arm, r["took"], r["choice"], r["forced"], r["entered"], r["leaves"], r["entered"], r["coll"], r["ds"]))
        L += ["", "| paired difference (took rate) | diff [95% CI] | gained / lost |", "|---|---|--:|"]
        for k, d in diffs.items():
            L.append("| %s | %+.2f [%+.2f, %+.2f] | %d / %d |" % (k, d["diff"], d["lo"], d["hi"], d["gained"], d["lost"]))
        L.append("")
    (RES / "pre_turns.json").write_text(json.dumps(out, indent=1, default=float))
    (RES / "pre_turns.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def split(a):
    import rft_split
    sys.argv = ["rft_split.py", *a.arms, "--out", str(RES / ("pre_split" + (f"_{a.tag}" if a.tag else "")))]
    rft_split.main()


def panels(a):
    import rft_panels
    rft_panels.COL.update({"rc-bear-pre-s0": "#d62728", "rc-ctl-pre-s0": "#1f77b4", "rc-bear-pre-s0@doff": "#d62728", "rc-ctl-pre-s0@doff": "#1f77b4",
                           "rc-bear-s0": "#ff9896", "rc-bear-s0@doff": "#ff9896", "rc-ctl-s0": "#aec7e8", "rc-ctl-s0@doff": "#aec7e8",
                           "shipped": "#7f7f7f", "shipped@doff": "#7f7f7f"})
    sys.argv = ["rft_panels.py", "--arms", *a.arms, "--out", str(REPO / "experiments/op_route_ft/figs" / ("pre_b2d_turn_panels" + (f"_{a.tag}" if a.tag else "") + ".png"))]
    if a.pick:
        sys.argv += ["--pick", a.pick]
    rft_panels.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("table")
    p.add_argument("--desire", default="off,on")
    for n in ("split", "panels"):
        p = sp.add_parser(n)
        p.add_argument("arms", nargs="+")
        p.add_argument("--tag", default="")
        p.add_argument("--pick", default="")
    a = ap.parse_args()
    {"table": table, "split": split, "panels": panels}[a.cmd](a)
