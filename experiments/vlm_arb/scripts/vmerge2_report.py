"""Report of the vmerge2 lane (plan 2026-10-03-vmerge.md, section "vmerge2").

  python vmerge2_report.py report     -> $DATA_DIR/runs/vlm_arb/vmerge2/results/{vmerge2.md, vmerge2_runs.csv, vmerge2_paired.csv, vmerge2_components.csv}

Primary read (registered): seeds 0 and 1, paired per route, vmerge2 vs drive and vs vred, sets all / obstacle / light, route bootstrap as
vmerge_report.py (2000 draws, seed 0). Secondary: every finished seed of vmerge2 vs drive (seeds 2, 3 are reruns of the unchanged drive arm).
Runs that never finished are skipped and listed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_v2_report as v2  # noqa: E402
import vmerge_report as vmr  # noqa: E402
from vlm_arb_common import LIGHT_ROUTES, OBS_ROUTES, ROUTES, RUN, SIGN_ROUTES, route_row, unit_dir  # noqa: E402

OUT = RUN / "vmerge2/results"
SEEDS4 = (0, 1, 2, 3)
ARM = "vmerge2"


def run_dir(arm, seed, rid):
    if arm == "drive" and seed >= 2:
        return unit_dir("drive", seed, next(k for k in v2.SHARDS if rid in v2.shards()[k]))
    return v2.run_dir(arm, seed, rid)


def collect(arms):
    rows = []
    for arm in arms:
        for s in SEEDS4:
            for rid in ROUTES:
                d = run_dir(arm, s, rid)
                r = route_row(d, rid)
                if r:
                    rows.append(dict(arm=arm, seed=s, unit=d.name, **r))
    return pd.DataFrame(rows)


def report(_):
    OUT.mkdir(parents=True, exist_ok=True)
    arms = ["drive", "vred", "vmerge", ARM]
    df = collect(arms)
    df.to_csv(OUT / "vmerge2_runs.csv", index=False)
    other = [r for r in ROUTES if r not in LIGHT_ROUTES + SIGN_ROUTES + OBS_ROUTES]
    sets = [("all", ROUTES), ("obstacle", OBS_ROUTES), ("light", LIGHT_ROUTES), ("stop sign", SIGN_ROUTES), ("other", other)]
    cols = ["DS", "RC", "red_light", "stop_infraction", "collisions", "vehicle_blocked"]
    con = [(ARM, "drive"), (ARM, "vred"), (ARM, "vmerge")]
    d01 = df[df.seed.isin((0, 1))]
    P1 = v2.pair_rows(d01, con, sets, cols)
    P1.insert(0, "seeds", "0,1")
    PA = v2.pair_rows(df, [(ARM, "drive")], sets, cols)
    PA.insert(0, "seeds", "all finished")
    pd.concat([P1, PA]).to_csv(OUT / "vmerge2_paired.csv", index=False)
    inf = df.groupby(["arm"])[["red_light", "stop_infraction", "collisions_vehicle", "collisions_layout", "collisions_pedestrian",
                               "outside_route_lanes", "vehicle_blocked", "route_timeout", "scenario_timeouts"]].sum().reindex(arms)
    # repeat noise: per-route DS of drive seeds 0,1 mean vs seeds 2,3 mean, and vmerge2 likewise
    noise = []
    for a in ("drive", ARM):
        x = df[df.arm == a].pivot_table(index="route", columns="seed", values="DS")
        x = x.reindex(columns=[s for s in SEEDS4 if s in x.columns])
        if x.shape[1] >= 2:
            sd = x.std(axis=1, ddof=1).dropna()
            noise.append("%s: %d routes with >= 2 seeds, per-route DS sd mean %.1f, median %.1f, max %.1f; per-seed mean DS %s" % (
                a, len(sd), sd.mean(), sd.median(), sd.max(), ", ".join("s%d %.1f" % (s, x[s].mean()) for s in x.columns)))
    vmr.ARM = ARM                                     # components() reads the arm through this module name
    C = pd.concat([vmr.components(s) for s in SEEDS4], ignore_index=True)
    C.to_csv(OUT / "vmerge2_components.csv", index=False)
    missing = []
    for a in (ARM, "drive"):
        for s in SEEDS4 if a == ARM else (2, 3):
            miss = [r for r in ROUTES if not ((df.arm == a) & (df.seed == s) & (df.route == r)).any()]
            if miss:
                missing.append("%s seed %d: %d of 19 routes missing (%s)" % (a, s, len(miss), " ".join(miss) if len(miss) < 19 else "all"))
    per_route = df.pivot_table(index="route", columns=["arm", "seed"], values="DS", aggfunc="mean").round(1)
    D = ["# vmerge2: vmerge + VLM_R2_TARGET=junction + VM_R5_TMAX=50, all 19 routes (diagnostic)", "",
         "Plan: [../plans/2026-10-03-vmerge.md](../plans/2026-10-03-vmerge.md), section vmerge2. `drive`, `vred`, `vmerge` seeds 0, 1 are the logged runs "
         "of the earlier batches; `drive` seeds 2, 3 are reruns of the unchanged drive arm.", "",
         "## Missing runs", "", ("\n".join("- " + m for m in missing) if missing else "none: every registered run finished"), "",
         "## Arms, seeds 0 and 1 (the registered primary set)", "", v2.arm_table(d01, arms).to_markdown(), "",
         "## Arms, all finished runs (seeds 2, 3 only for vmerge2 and drive)", "", v2.arm_table(df, arms).to_markdown(), "",
         "## Infractions by type (official, summed over runs)", "", inf.to_markdown(), "",
         "## Paired differences, seeds 0 and 1 (registered primary; mean over routes [95% route-cluster CI])", ""]
    D += v2.pair_md(P1, cols, cols)
    D += ["", "## Paired vmerge2 - drive, every seed both arms finished (seeds 0-3)", ""] + v2.pair_md(PA, cols, cols)
    D += ["", v2.NOISE, "", "Repeat noise in this batch: " + "; ".join(noise) if noise else "", "",
          "## Components (vmerge2 logs, summed over runs)", "",
          C.drop(columns=["route", "seed", "byp_first_t", "byp_suppressed"]).sum().to_frame("sum").T.to_markdown(index=False), "", "Per run:", "",
          C.to_markdown(index=False), "", "## DS per route and seed", "", per_route.to_markdown(), ""]
    (OUT / "vmerge2.md").write_text("\n".join(D))
    print("\n".join(D))


if __name__ == "__main__":
    {"report": report}[sys.argv[1]](sys.argv[2:])
