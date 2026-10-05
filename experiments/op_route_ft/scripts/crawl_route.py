"""Stopped share along whole B2D routes, by CARLA junction flag (decision 128 point 5 control: is the stop-go specific to junctions?).

    .venv/bin/python experiments/op_route_ft/scripts/crawl_route.py --out $DATA_DIR/runs/op_route_ft/crawl_route.json
Per arm over the 20 turn routes' plans.jsonl (warm-up ticks dropped): share of ticks with v < 0.3, median v, split by ctx.junc.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_guard/scripts"), str(REPO / "experiments/op_closed_loop/scripts"), str(REPO / "experiments/vlm_arb/scripts")]
import cllib as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for arm in ("shipped", "zones-on", "rc-ctl-s0", "rc-bear-s0", "rc-poly-s0", "rc-all-s0"):
        dirs = sorted(Path(C.CACHE_TURNS).glob("ol-s%d-k*" % C.TURN_SEED)) if arm == "zones-on" else C.b2d_dirs(arm, "subset", "turns", C.TURN_SEED)
        acc = {"junc": [], "road": [], "t<30s": [], "30-60s": [], "60-120s": [], ">120s": []}
        per_route = {}
        for rid in C.TURN_ROUTES:
            att, _ = C.attempt_of(dirs, rid)
            if att is None or not (att / "plans.jsonl").exists():
                continue
            pl = [json.loads(x) for x in open(att / "plans.jsonl")]
            pl = [p for p in pl if not p.get("warm")]
            for p in pl:
                acc["junc" if (p.get("ctx") or {}).get("junc") else "road"].append(p["v"])
                acc["t<30s" if p["t"] < 35 else "30-60s" if p["t"] < 65 else "60-120s" if p["t"] < 125 else ">120s"].append(p["v"])
            per_route[rid] = dict(n=len(pl), stop=round(float(np.mean([p["v"] < 0.3 for p in pl])), 2), vmed=round(float(np.median([p["v"] for p in pl])), 2))
        out[arm] = {k: dict(n=len(v), stop=round(float(np.mean(np.array(v) < 0.3)), 3), vmed=round(float(np.median(v)), 2), v90=round(float(np.percentile(v, 90)), 1)) for k, v in acc.items() if v}
        out[arm]["routes"] = per_route
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {x: v[x] for x in v if x != "routes"})


if __name__ == "__main__":
    main()
