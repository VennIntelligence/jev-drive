"""Report of the gap-rule sweep (plan 2026-10-04-vmerge3.md, addendum).

  python vmerge3gap_report.py report [--out DIR]  -> $DATA_DIR/runs/vlm_arb/vm3gap/results/{vmerge3_gap.md, vmerge3_gap_runs.csv,
                                       vmerge3_gap_collisions.csv, vmerge3_gap_frontier.png}; copy to experiments/vlm_arb/results/.
Arms gap20 / gap25 / gap30 = vm3norel with follower headway T = 2.0 / 2.5 / 3.0 s (gap30 seeds 0, 1 = the vm3norel runs); references drive, vmerge2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_v2_report as v2  # noqa: E402
import vmerge2_report as m2  # noqa: E402
import vmerge_collisions as vc  # noqa: E402
from vlm_arb_common import OBS_ROUTES, ROUTES, RUN, jsonl, route_row, unit_dir, vred_shards  # noqa: E402

OUT = RUN / "vm3gap/results"
ARMS = ("gap20", "gap25", "gap30")
REFS = ("drive", "vmerge2")
LINE_DS, LINE_COL = 20.7, 0.158
GAP_T = {"gap20": "2.0", "gap25": "2.5", "gap30": "3.0"}      # follower headway T (s) of the start rule, per arm
OBS_UNITS = {"oa": ["19324", "2520"], "ob": ["19832", "24497"]}


def rdir(arm, seed, rid):
    if arm not in GAP_T:
        return m2.run_dir(arm, seed, rid)
    if arm == "gap30" and seed < 2:
        return m2.run_dir("vm3norel", seed, rid)
    if rid in OBS_ROUTES:
        return unit_dir(arm, seed, next(k for k, v in OBS_UNITS.items() if rid in v))
    return unit_dir(arm, seed, next(k for k, v in vred_shards().items() if rid in v))


def collect():
    rows = []
    for arm in REFS + ARMS:
        for s in range(4):
            for rid in ROUTES:
                d = rdir(arm, s, rid)
                r = route_row(d, rid)
                if r:
                    rows.append(dict(arm=arm, seed=s, unit=d.name, **r))
    return pd.DataFrame(rows)


def collisions(df):
    rows = []
    for _, r in df[df.collisions_vehicle > 0].iterrows():
        plans = jsonl(Path(r.attempt) / "plans.jsonl")
        for e in vc.describe(r.arm, r.seed, r.route, dict(DS=r.DS, attempt=r.attempt)):
            # pull-out = the bypass path was active (shift_frac >= 0.05) within 3 s before to 1 s after the contact stamp (contacts.jsonl runs ~1 s ahead)
            e["pullout"] = bool(e["pc_bypass_now"]) or any((p.get("pc") or {}).get("bypass") and (p.get("pc") or {}).get("shift_frac", 0) >= 0.05
                                                           for p in plans if e["t"] - 3 <= p.get("t", -1e9) <= e["t"] + 1)
            e["side"] = e["pullout"] and e["rel_lat"] is not None and abs(e["rel_lat"]) >= 1.0
            rows.append(e)
    return pd.DataFrame(rows)


def arm_row(df, C, arm, routes, seeds):
    d = df[(df.arm == arm) & df.route.isin(routes) & df.seed.isin(seeds)]
    c = C[(C.arm == arm) & C.route.isin(routes) & C.seed.isin(seeds)] if len(C) else C
    n = max(len(d), 1)
    return dict(arm=arm, runs=len(d), DS=d.DS.mean(), RC=d.RC.mean(), coll_vehicle=d.collisions_vehicle.sum(), coll_per_run=d.collisions_vehicle.sum() / n,
                pullout_coll=int(c.pullout.sum()) if len(c) else 0, side_pullout=int(c.side.sum()) if len(c) else 0,
                timeouts=int(d.route_timeout.sum()) if "route_timeout" in d else 0, blocked=int(d.vehicle_blocked.sum()), crashes=int(d.crash.sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "vmerge3_gap_runs.csv", index=False)
    C = collisions(df)
    C.to_csv(OUT / "vmerge3_gap_collisions.csv", index=False)
    present = [a for a in ARMS if (df.arm == a).any()]
    D = ["# vmerge3 gap-rule sweep: DS against collisions of the perceived non-privileged bypass", "",
         "Plan: [../plans/2026-10-04-vmerge3.md](../plans/2026-10-04-vmerge3.md) (addendum, pre-registered). Arms are `vm3norel` (perceived bypass, no release "
         "check) with the follower headway T of the start rule d = T + T x v: `gap20` T = 2.0 s, `gap25` 2.5 s, `gap30` 3.0 s (seeds 0, 1 are the earlier "
         "vm3norel runs, seeds 2, 3 new). References: `drive`, `vmerge2` (privileged bypass). Collision counts: `coll_vehicle` = official vehicle collisions; `pullout_coll` / `side_pullout` count first contacts per actor, so they can exceed it (the scorer merges contacts within a cool-down). Pull-out collision = contact while the bypass path is active; side = pull-out "
         "contact with the other vehicle >= 1 m lateral.", ""]
    stats = {}
    for name, routes, seeds in (("obstacle routes, seeds 0-3", OBS_ROUTES, range(4)), ("obstacle routes, seeds 0-1", OBS_ROUTES, range(2)),
                                ("all 19 routes, seeds 0-1 (arms that have them)", ROUTES, range(2))):
        T = pd.DataFrame([arm_row(df, C, a, routes, seeds) for a in REFS + tuple(present)]).round(3)
        T = T[T.runs > 0]
        D += ["## " + name, "", T.to_markdown(index=False), ""]
    # paired DS vs drive on the obstacle routes
    cols = ["DS", "collisions_vehicle"]
    con = [(a, "drive") for a in present] + [("vmerge2", "drive")] + [(a, "vmerge2") for a in present]
    for sname, seeds in (("0-3", range(4)), ("0-1", range(2))):
        dd = df[df.seed.isin(seeds)]
        P = v2.pair_rows(dd, con, [("obstacle", OBS_ROUTES), ("all", ROUTES)], cols)
        P.insert(0, "seeds", sname)
        P.to_csv(OUT / ("vmerge3_gap_paired_s%s.csv" % sname.replace("-", "")), index=False)
        stats[sname] = P
        D += ["## Paired differences, seeds %s (mean over routes [95%% route-cluster CI], 2000 draws)" % sname, ""] + v2.pair_md(P[P.groups > 0], cols, cols) + [""]
    # lines
    P4 = stats["0-3"]
    D += ["## Registered line", "", "Acceptable = obstacle-route DS vs drive >= +%.1f and full-route vehicle collisions per run <= %.3f (drive, seeds 0-1)." % (LINE_DS, LINE_COL), "",
          "| arm | obstacle DS vs drive (seeds 0-3) | >= +20.7 | obstacle coll / run (seeds 0-3) vs drive | full-route coll / run (seeds 0-1) | <= 0.158 | verdict |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for a in present:
        o = P4[(P4.contrast == a + " - drive") & (P4.routes == "obstacle") & (P4.metric == "DS")]
        if o.empty or not o.iloc[0].groups:
            continue
        o = o.iloc[0]
        ro, rd = arm_row(df, C, a, OBS_ROUTES, range(4)), arm_row(df, C, "drive", OBS_ROUTES, range(4))
        full = arm_row(df, C, a, ROUTES, range(2))
        have_full = full["runs"] >= 38
        okd = o.est >= LINE_DS
        okc = bool(full["coll_per_run"] <= LINE_COL) if have_full else None
        verdict = "n/a" if not okd else ("acceptable" if okc else ("fails collisions" if okc is False else "obstacle only: DS line met, full-route collisions not run"))
        if not okd:
            verdict = "fails DS" + ("" if okc is not False else " and collisions")
        D.append("| %s | %s | %s | %.3f vs %.3f | %s | %s | %s |" % (a, v2.fmt(o.to_dict()), "yes" if okd else "no", ro["coll_per_run"], rd["coll_per_run"],
                 "%.3f (%d / %d)" % (full["coll_per_run"], full["coll_vehicle"], full["runs"]) if have_full else "not run", {True: "yes", False: "no", None: "-"}[okc], verdict))
    D += ["", "## Vehicle collisions", "", (C[[c for c in ("arm", "seed", "route", "t", "ego_v", "actor", "actor_v", "rel_long", "rel_lat", "head_diff", "pullout", "side", "byp_state") if c in C]].to_markdown(index=False) if len(C) else "none"), ""]
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    pts = [(a, "obstacle", "o") for a in present] + [("vmerge2", "obstacle", "s")]
    for a, _, mk in pts:
        o = P4[(P4.contrast == a + " - drive") & (P4.routes == "obstacle") & (P4.metric == "DS")]
        if o.empty or not o.iloc[0].groups:
            continue
        o = o.iloc[0]
        x = arm_row(df, C, a, OBS_ROUTES, range(4))["coll_per_run"]
        ax.errorbar(x, o.est, yerr=[[o.est - o.lo], [o.hi - o.est]], marker=mk, capsize=3, ls="none", label=a + (" (T=%ss)" % GAP_T[a] if a in GAP_T else ""))
    xd = arm_row(df, C, "drive", OBS_ROUTES, range(4))["coll_per_run"]
    ax.plot([xd], [0], "k*", label="drive (reference)")
    ax.axhline(LINE_DS, color="gray", ls="--", lw=0.8)
    ax.text(ax.get_xlim()[0], LINE_DS + 1, " DS line +20.7", fontsize=7, color="gray")
    ax.set_xlabel("vehicle collisions per run (4 obstacle routes, seeds 0-3)")
    ax.set_ylabel("obstacle-route DS vs drive [95% route CI]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "vmerge3_gap_frontier.png", dpi=160)
    (OUT / "vmerge3_gap.md").write_text("\n".join(D))
    print("\n".join(D))


if __name__ == "__main__":
    main()
