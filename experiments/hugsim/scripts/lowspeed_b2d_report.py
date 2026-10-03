"""Report of the OD2 B2D lane (plan experiments/op_adapt_h/plans/2026-10-04-od2-prereg.md, section 3).

  $DATA_DIR/envs/jevdrive/bin/python experiments/op_adapt_h/scripts/od2_b2d_report.py [--seeds 2,3]
  -> experiments/op_adapt_h/results/one_driver/b2d/{b2d.md, runs.csv, paired.csv, selector.csv}

Arms: drive (shipped, vmerge2's seed 2 / 3 reruns, v2-drive-s<seed>-q<k>), od2it (it_dw3-s0), od2sel (it_dw3-s0 + server selector).
Paired per (route, seed), route-cluster bootstrap as vmerge2_report.py (vlm_v2_report.pair_rows). Selector statistics from the
plans.jsonl `sel` records: share of non-warm steps the selector ran on (sel=1), share where it swapped the plan, rebuilds, its time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/vlm_arb/scripts"))
import vlm_v2_report as v2  # noqa: E402
from vlm_arb_common import LIGHT_ROUTES, OBS_ROUTES, ROUTES, SIGN_ROUTES, attempt_dir, route_row, unit_dir  # noqa: E402

OUT = REPO / "experiments/hugsim/results/lowspeed_ctrl_b2d"
ARMS = ("drive", "lsc")


def udir(arm, seed, rid):
    return unit_dir(arm, seed, next(k for k in v2.SHARDS if rid in v2.shards()[k]))


def sel_stats(d, rid):
    a = attempt_dir(d, rid)
    if a is None or not (a / "plans.jsonl").exists():
        return None
    n = on = used = reb = 0
    ms = []
    with open(a / "plans.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("warm"):
                continue
            n += 1
            s = r.get("sel") or {}
            if s.get("sel") == 1:
                on += 1
                used += bool(s.get("sel_used"))
                reb += bool(s.get("sel_rebuilt"))
                ms.append(float(s.get("sel_ms", 0.0)))
    return dict(steps=n, sel_on=on, sel_used=used, rebuilds=reb, sel_ms_sum=float(np.sum(ms)), sel_ms_p95=float(np.percentile(ms, 95)) if ms else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2,3")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, srows, missing = [], [], []
    for arm in ARMS:
        for s in seeds:
            for rid in ROUTES:
                d = udir(arm, s, rid)
                r = route_row(d, rid)
                if r is None:
                    missing.append("%s s%d %s" % (arm, s, rid))
                    continue
                rows.append(dict(arm=arm, seed=s, unit=d.name, **r))
                if False:
                    st = sel_stats(d, rid)
                    if st:
                        srows.append(dict(route=rid, seed=s, DS=r["DS"], **st))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)
    other = [r for r in ROUTES if r not in LIGHT_ROUTES + SIGN_ROUTES + OBS_ROUTES]
    sets = [("all", ROUTES), ("obstacle", OBS_ROUTES), ("light", LIGHT_ROUTES), ("stop sign", SIGN_ROUTES), ("other", other)]
    cols = ["DS", "RC", "red_light", "stop_infraction", "collisions", "vehicle_blocked"]
    P = v2.pair_rows(df, [("lsc", "drive")], sets, cols)
    P.to_csv(OUT / "paired.csv", index=False)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "selector.csv", index=False)
    inf = df.groupby("arm")[["red_light", "stop_infraction", "collisions_vehicle", "collisions_layout", "collisions_pedestrian",
                             "outside_route_lanes", "vehicle_blocked", "route_timeout", "scenario_timeouts"]].sum().reindex(list(ARMS))
    noise = []
    for arm in ARMS:
        x = df[df.arm == arm].pivot_table(index="route", columns="seed", values="DS")
        if x.shape[1] >= 2:
            sd = x.std(axis=1, ddof=1).dropna()
            noise.append("%s: per-route DS sd over seeds mean %.1f, median %.1f, max %.1f" % (arm, sd.mean(), sd.median(), sd.max()))
    sel_line = "no selector (the rule is a steering filter)"
    per_route = df.pivot_table(index="route", columns=["arm", "seed"], values="DS", aggfunc="mean").round(1)
    D = ["# Low-speed lateral transfer limit on B2D: `drive` arm, 19 routes, seeds %s" % ", ".join(map(str, seeds)), "",
         "Plan: [../../plans/2026-10-04-lowspeed-ctrl-prereg.md](../../plans/2026-10-04-lowspeed-ctrl-prereg.md), section B. `drive` = vmerge2 reruns of the unchanged shipped arm; `lsc` = + LOWSPEED_CTRL.", "",
         "## Missing runs", "", ("\n".join("- " + m for m in missing) if missing else "none"), "",
         "## Arms", "", v2.arm_table(df, list(ARMS)).to_markdown(), "",
         "## Infractions by type (summed over runs)", "", inf.to_markdown(), "",
         "## Paired differences (mean over routes [95% route-cluster CI])", ""]
    D += v2.pair_md(P, cols, cols)
    D += ["", "Repeat noise: " + "; ".join(noise), "", "## Selector", "", sel_line, "",
          (S.to_markdown(index=False) if len(S) else ""), "", "## DS per route and seed", "", per_route.to_markdown(), ""]
    (OUT / "b2d.md").write_text("\n".join(D))
    print("\n".join(D))


if __name__ == "__main__":
    main()
