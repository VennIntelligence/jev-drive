"""Spin count and HD-Score of an h_hugsim.sh run over all 64 scenarios, paired against the base Cinque PR #57 run
(arm "base" in experiments/hugsim/results/derot/derot_runs.csv). envs/hugsim python:
    python experiments/op_adapt_h/scripts/h_hugsim64_report.py <out dir> <tag>  -> <out>/spins.csv, <out>/summary.json
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
from derot_report import SPIN10, runs  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

ROUTES = Path(os.environ["DATA_DIR"]) / "tmp_spin" / "routes.json"
BASE = {r["scenario"]: r for r in csv.DictReader(open(REPO / "experiments/hugsim/results/derot/derot_runs.csv")) if r["arm"] == "base"}


def main(out, tag):
    out = Path(out)
    routes = json.load(open(ROUTES))
    rows = []
    for (t, scen), (r, d) in sorted(runs(out / "results.csv", out, {tag}).items()):
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        b = BASE[scen]
        rows.append(dict(scenario=scen, spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), hdscore=float(r["hdscore"]), end=r["end"],
                         steps=int(r["steps"]), base_spin=b["spin"] == "True", base_hd=float(b["hdscore"]), base_end=b["end"]))
    with open(out / "spins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ns = [r for r in rows if r["scenario"] not in SPIN10]
    dl = np.array([r["hdscore"] - r["base_hd"] for r in ns])
    bs = dl[np.random.default_rng(0).integers(0, len(dl), (10000, len(dl)))].mean(1)
    s10 = [r for r in rows if r["scenario"] in SPIN10]
    s = dict(tag=tag, n=len(rows), spins=sum(r["spin"] for r in rows), base_spins=sum(r["base_spin"] for r in rows),
             spins10=sum(r["spin"] for r in s10), n10=len(s10), new_spins=[r["scenario"] for r in rows if r["spin"] and not r["base_spin"]],
             hd_mean=float(np.mean([r["hdscore"] for r in rows])), base_hd_mean=float(np.mean([r["base_hd"] for r in rows if r["scenario"] in {x["scenario"] for x in rows}])),
             n_nonspin10=len(ns), dhd_nonspin=float(dl.mean()), dhd_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
