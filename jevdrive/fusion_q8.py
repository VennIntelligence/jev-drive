"""Fusion diagnostics Q8: the reaction window per P5 family against three latency tiers
(todos/2026-09-25-fusion-diagnostics.md, Q8). Window = (t_div - t_vis) x 0.05 s per pair (pairs.csv, 20 Hz ticks):
from the factor becoming visible to the two egos diverging. Remaining budget = window - latency; a family "must stay
on the fast channel" for a tier when its window p25 < that tier's latency + 0.5 s (execution margin).

Tiers: openpilot Cinque 2.3 ms p50 (smoke, the todo's number); SAM 3.1 image mode on our card (Q4d, 3 cameras,
6 prompts, p50 and p95) + the decision head (a linear head, taken as 0); openjev 470 ms p50 (docs/baselines.md, the
stand-in for a VLM slow channel).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
MARGIN = 0.5
TICK = 0.05


def windows() -> pd.DataFrame:
    p = pd.read_csv(data_dir() / "processed" / "carla_p5" / "pairs.csv", dtype={"base_id": str})
    p = p[p.reason == "ok"].copy() if "reason" in p else p
    p["window_s"] = (p.t_div - p.t_vis) * TICK
    return p[np.isfinite(p.window_s)]


def table(p: pd.DataFrame, tiers: dict) -> pd.DataFrame:
    rows = []
    for fam, g in [*p.groupby("family"), ("all families", p)]:
        w = g.window_s.to_numpy()
        r = {"family": fam, "pairs": len(w), "window_p25_s": np.percentile(w, 25), "window_median_s": np.median(w),
             "window_p75_s": np.percentile(w, 75), "window_min_s": w.min()}
        for name, lat in tiers.items():
            r[f"budget_p25 {name}"] = r["window_p25_s"] - lat
            r[f"fast only {name}"] = bool(r["window_p25_s"] < lat + MARGIN)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latency", required=True, help="Q4d latency.json (sam_detect latency)")
    a = ap.parse_args()
    rl = RunLog("fusion_diag", "q8")
    lat = json.loads(Path(a.latency).read_text())
    s3 = lat["3 cameras"]
    tiers = {"openpilot Cinque (2.3 ms p50)": 0.0023,
             f"SAM 3.1 3 cams p50 ({s3['p50_ms']:.0f} ms)": s3["p50_ms"] / 1000,
             f"SAM 3.1 3 cams p95 ({s3['p95_ms']:.0f} ms)": s3["p95_ms"] / 1000,
             "openjev (470 ms p50)": 0.470}
    p = windows()
    t = table(p, tiers)
    t.to_csv(rl.dir / "q8_windows.csv", index=False)
    rl.log.info("Q8 (pairs with reason ok: %d)\n%s", len(p), t.to_markdown(index=False, floatfmt=".2f"))
    rl.event("end", tiers=tiers)
    rl.close()


if __name__ == "__main__":
    main()
