"""Spin count and HD-Score of one h_hugsim.sh run (controller_spin.md definition via experiments/hugsim/scripts/spin_analysis.py),
next to decision 96's same-code baseline rerun (cinque-fixed-base in $DATA_DIR/runs/hugsim-derot). envs/hugsim python:
    python experiments/op_adapt_h/scripts/h_hugsim_report.py <out dir> <tag>     -> <out>/spins.csv, <out>/summary.json
"""
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
from jevdrive.bench.hugsim import routes as bench_routes
from derot_report import SPIN10, runs  # noqa: E402
from spin_analysis import analyse, load_run  # noqa: E402

DATA = Path(os.environ["DATA_DIR"])
BASE = DATA / "runs" / "hugsim-derot"


def table(root, tag):
    routes = bench_routes()
    out = []
    for (t, scen), (r, d) in sorted(runs(root / "results.csv", root, {tag}).items()):
        pos, th, v, steer, plans = load_run(d, "cinque")
        res, _ = analyse(pos, th, v, steer, plans, routes[r["scene"]])
        out.append(dict(scenario=scen, spin=bool(res["spin"]), max_abs_e=round(res["max_abs_e"], 1), hdscore=float(r["hdscore"]),
                        end=r["end"], steps=int(r["steps"])))
    return out


def main(out, tag):
    out = Path(out)
    rows = table(out, tag)
    base = {r["scenario"]: r for r in table(BASE, "cinque-fixed-base")}
    with open(out / "spins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["scenario", "spin", "max_abs_e", "hdscore", "end", "steps", "base_spin", "base_hd"])
        w.writeheader()
        for r in rows:
            b = base.get(r["scenario"], {})
            w.writerow(r | {"base_spin": b.get("spin"), "base_hd": b.get("hdscore")})
    s = {"tag": tag, "n": len(rows), "spins": sum(r["spin"] for r in rows), "hd_mean": sum(r["hdscore"] for r in rows) / max(len(rows), 1),
         "base_rerun_spins": sum(b["spin"] for b in base.values()), "base_rerun_hd_mean": sum(b["hdscore"] for b in base.values()) / max(len(base), 1),
         "scenarios": [r["scenario"] for r in rows], "spin10": SPIN10}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
