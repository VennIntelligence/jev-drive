"""op_img_cmd closed-loop smoke readout (.venv): per run DS / route completion / turn agreement (op_l_b2d_report definitions:
LEFT / RIGHT command within 15 m, moving, plan-to-route divergence <= 1.0 m) of the imgsky and drive units of img_cl_lane.py,
pooled per arm and paired per (route, seed). Output experiments/op_img_cmd/results/cl_smoke.{md,csv}.

  .venv/bin/python -m experiments.op_img_cmd.scripts.img_cl_report     (or run as a file from the repo root)
"""
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.op_adapt_l.lib.op_l_b2d_report import attempts, run_row  # noqa: E402

RUN = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs" / "op_img_cmd" / "cl" / "arms"
OUT = REPO / "experiments" / "op_img_cmd" / "results"


def main():
    rows = []
    for d in sorted(RUN.glob("*-s[0-9]")):
        arm, seed = d.name.rsplit("-s", 1)
        for rid, adir in attempts(d).items():
            r = run_row(adir)
            rows.append({"arm": arm, "seed": int(seed), "route": rid, **{k: r.get(k) for k in
                         ("DS", "RC", "turn_steps", "turn_agree_steps", "turn_runs", "turn_ok", "coll_veh", "red_light", "blocked", "timeout")}})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "cl_smoke.csv", index=False)
    g = D.groupby("arm").agg(runs=("DS", "size"), DS=("DS", "mean"), RC=("RC", "mean"), turn_steps=("turn_steps", "sum"),
                             agree=("turn_agree_steps", "sum"), turn_ok=("turn_ok", "sum"), turn_runs=("turn_runs", "sum"))
    g["turn_agree"] = g.agree / g.turn_steps
    p = D.pivot_table(index=["route", "seed"], columns="arm", values=["DS", "turn_agree_steps", "turn_steps"])
    L = ["# op_img_cmd closed-loop smoke (sky arrow fine-tune vs drive)", "", "Per arm (pooled turn agreement over turn-window steps):", "",
         g.round(3).to_markdown(), "", "Per route and seed:", "", p.to_markdown(), ""]
    (OUT / "cl_smoke.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
