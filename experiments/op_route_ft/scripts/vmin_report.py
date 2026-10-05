"""Readout of the plan_vmin = 0 confirming run (crawl.md): per arm (shipped baseline vs vmin0) on the same routes, stopped share,
stop-go cycles (stopped intervals >= 1 s) in the 15 s turn windows, stop length, took exit.

    .venv/bin/python experiments/op_route_ft/scripts/crawl_dump.py --arms shipped,vmin0 --routes 24944,10255,26153 --out $DATA_DIR/runs/op_route_ft/vmin0/dump.json
    .venv/bin/python experiments/op_route_ft/scripts/vmin_report.py $DATA_DIR/runs/op_route_ft/vmin0/dump.json [--out results/vmin0.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import crawl as K  # noqa: E402


def main():
    D = json.load(open(sys.argv[1]))
    out = {}
    for arm in dict.fromkeys(r["arm"] for r in D):
        rows = []
        for r in D:
            if r["arm"] != arm:
                continue
            row = dict(route=r["route"], turn=r["turn"], entered=r["entered"], took=r["took"])
            if r["entered"]:
                tk, iv = K.intervals(r)
                nm = [t for t in tk if not K.is_warm(t)]
                row.update(ticks=len(tk), stopped=sum(t["v"] < K.V_STOP for t in tk), cycles=len(iv),
                           cycles_nowarm=sum(K.cause_of(tk[i:j + 1]) != "warm-up" for i, j in iv), stop_len=[round((j - i + 1) * K.DT, 1) for i, j in iv],
                           causes=[K.cause_of(tk[i:j + 1]) for i, j in iv], v_nowarm=[t["v"] for t in nm], longest_run=max([(j - i + 1) * K.DT for i, j in iv] or [0]))
            rows.append(row)
        E = [x for x in rows if x["entered"]]
        L = [l for x in E for l in x["stop_len"]]
        out[arm] = dict(turns=len(rows), entered=len(E), took=sum(x["took"] for x in rows),
                        stopped_share=round(sum(x["stopped"] for x in E) / max(sum(x["ticks"] for x in E), 1), 3),
                        cycles_per_turn=round(sum(x["cycles"] for x in E) / max(len(E), 1), 2),
                        cycles_nowarm_per_turn=round(sum(x["cycles_nowarm"] for x in E) / max(len(E), 1), 2),
                        stop_len_mean=round(float(np.mean(L)), 2) if L else None, stop_len_median=round(float(np.median(L)), 2) if L else None,
                        stop_len_max=max(L) if L else None, v_med_nowarm=round(float(np.median([v for x in E for v in x["v_nowarm"]])), 2) if E else None,
                        per_turn=[{k: v for k, v in x.items() if k != "v_nowarm"} for x in rows])
        print(arm, {k: v for k, v in out[arm].items() if k != "per_turn"})
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
