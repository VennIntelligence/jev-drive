"""Gates and reports of the v2 lane: arms `pbyp2` and `vred2` (plan 2026-10-02-pbyp2-vred2.md).

  python vlm_v2_report.py calibrate   in-loop latency of the shadow units `cal2-s1-q*` -> gates/v2_cal.json (passes only with p95 <= L)
  python vlm_v2_report.py few-pbyp2   checklist over the pbyp2 seed-0 units -> gates/v2_few_pbyp2.json
  python vlm_v2_report.py few-vred2   checklist over the vred2 seed-0 units -> gates/v2_few_vred2.json
  python vlm_v2_report.py report      pbyp2.md, vred2.md, tables and figures -> $DATA_DIR/runs/vlm_arb/v2/results/

Conventions of results/report.md: official infractions; paired differences per (route, seed) averaged over seeds, resampling routes
(2000 draws, seed 0), 95% percentile intervals. Every number is a diagnostic read (19 routes); registered confirmation lines are
marked "not evaluated" and never "passed". V2_TEST=1 maps pbyp2 -> pbyp and vred2 -> vred (code-path test on existing runs).
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_vred_report as vr  # noqa: E402
from vlm_arb_checks import r2_episodes, route_checks, vlm_rows  # noqa: E402
from vlm_arb_common import (LIGHT_ROUTES, OBS_ROUTES, ROUTES, RUN, SEEDS, boot_mean, boot_ratio, drive_dir,  # noqa: E402
                            fmt, route_row, unit_dir, write_json)
from vlm_arb_report import paired  # noqa: E402
from vlm_thin_common import RED, light_masks  # noqa: E402

TEST = os.environ.get("V2_TEST") == "1"
OUT = RUN / ("v2/results_test" if TEST else "v2/results")
SHARDS = ("q0", "q1", "q2")
L_REG = 0.35
ALIAS = {"pbyp2": "pbyp", "vred2": "vred", "cal2": "cal"} if TEST else {}
OTHER = [r for r in ROUTES if r not in OBS_ROUTES]
NOISE = ("13 routes with 2-4 identical `drive` runs: per-route DS standard deviation mean 4.7, median 0.0, max 30.5; 5 of 13 routes changed DS "
         "between repeats (results/report.md)")


def shards():
    return json.loads((RUN / "gates/vred_shards.json").read_text())


def udir(arm, seed, shard):
    return unit_dir(ALIAS.get(arm, arm), seed, shard)


vr.unit_dir = udir                                     # vr.attempts / answers read units through this name


def run_dir(arm, seed, rid):
    arm = ALIAS.get(arm, arm)
    if arm == "drive":
        return drive_dir(rid, seed)
    if arm in ("pred", "pbyp", "jslow"):
        return unit_dir(arm, seed, "tgt" if rid not in ROUTES[:10] else "dev")
    return unit_dir(arm, seed, next(k for k in SHARDS if rid in shards()[k]))


def collect(arms):
    rows = []
    for arm in arms:
        for s in SEEDS:
            for rid in ROUTES:
                d = run_dir(arm, s, rid)
                r = route_row(d, rid)
                if r:
                    rows.append(dict(arm=arm, seed=s, unit=d.name, **r))
    return pd.DataFrame(rows)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def pair_rows(df, contrasts, sets, cols):
    out = []
    for a, b in contrasts:
        for name, routes in sets:
            for col in cols:
                out.append(dict(contrast="%s - %s" % (a, b), routes=name, metric=col, **paired(df, a, b, routes, col)))
    return pd.DataFrame(out)


def pair_md(P, cols, labels):
    D = ["| contrast | routes | n routes | " + " | ".join(labels) + " |", "|:--|:--|--:|" + ":--|" * len(cols)]
    for (c, rs), g in P.groupby(["contrast", "routes"], sort=False):
        f = lambda col: fmt(g[g.metric == col].iloc[0].to_dict())   # noqa: E731
        D.append("| %s | %s | %d | %s |" % (c, rs, g.groups.iloc[0], " | ".join(f(k) for k in cols)))
    return D


def arm_table(df, arms):
    g = df.groupby("arm")
    return g.agg(runs=("route", "count"), crashes=("crash", "sum"), DS=("DS", "mean"), RC=("RC", "mean"), red_light=("red_light", "sum"),
                 stop_sign=("stop_infraction", "sum"), collisions=("collisions", "sum"), blocked=("vehicle_blocked", "sum"),
                 timeouts=("route_timeout", "sum"), v_mean=("v_mean", "mean")).reindex(arms).round(2)


def ci_flag(r):
    return "n/a" if not r["groups"] else ("includes 0" if r["lo"] <= 0 <= r["hi"] else "excludes 0")


# ------------------------------------------------------------------------------------------------------------ gates
def calibrate(_):
    df = vr.answers("cal2", 1)
    if df.empty:
        raise SystemExit("no calibration answers")
    lat = df.lat[df.ok].to_numpy()
    p95 = pct(lat, 95)
    gate = dict(done=True, n=int(len(df)), ok_rate=float(df.ok.mean()), p50_ms=pct(lat, 50), p95_ms=p95, p99_ms=pct(lat, 99), max_ms=float(lat.max()),
                over_L=float(np.mean(lat > 1e3 * L_REG)), over_ttl=float(np.mean(lat > 2500)), queue_p95_ms=pct(df.q_ms[df.ok], 95),
                svc_p50_ms=pct(df.svc[df.ok], 50), L_s=L_REG, load="3 shadow + 3 pbyp2 workers per card on 3 cards",
                passed=bool(p95 <= 1e3 * L_REG and df.ok.mean() >= 0.98 and np.mean(lat > 2500) <= 0.01))
    write_json(RUN / "gates/v2_cal.json", gate)
    txt = ("Calibration (shadow `drive`, Qwen3-VL-4B servers, %s, seed 1, 19 routes): %d requests, %.1f%% answered. In-loop latency p50 %.0f ms, "
           "p95 %.0f ms, p99 %.0f ms, max %.0f ms; server queue wait p95 %.0f ms, service p50 %.0f ms; share > L (%.2f s) %.1f%%, share > TTL 2.5 s %.2f%%. "
           "Registered L = %.2f s holds only if p95 <= %.0f ms: %s.\n" % (gate["load"], len(df), 100 * gate["ok_rate"], gate["p50_ms"], p95, gate["p99_ms"],
                                                                        gate["max_ms"], gate["queue_p95_ms"], gate["svc_p50_ms"], L_REG, 100 * gate["over_L"],
                                                                        100 * gate["over_ttl"], L_REG, 1e3 * L_REG, "yes" if gate["passed"] else "NO"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.md").write_text(txt)
    print(txt)


def few_pbyp2(_):
    """Checklist over the seed-0 pbyp2 units: no crash; every activation on a blocker inside the route; none before the ego first moves
    on a route without a scenario obstacle; a bypass that returns to the lane on at least one obstacle route."""
    rows, crashes, ret = [], 0, 0
    for rid in ROUTES:
        r = route_row(run_dir("pbyp2", 0, rid), rid)
        if not r:
            rows.append(dict(route=rid, finished=False))
            continue
        o, c = route_checks(r["attempt"], "pbyp2")
        crashes += bool(r["crash"])
        pr = route_checks(r["attempt"], "bypass")[1] if rid in OBS_ROUTES else {}
        ret += bool(pr.get("passed_and_returned") and pr.get("path_enabled"))
        rows.append(dict(route=rid, finished=True, crash=bool(r["crash"]), n_act=o["n_activations"], outside_m=o["outside_m"], DS=r["DS"],
                         **{k: bool(v) for k, v in c.items()}, bypass_returned=bool(pr.get("passed_and_returned") and pr.get("path_enabled"))))
    df = pd.DataFrame(rows)
    ok = bool(df.finished.all() and crashes == 0 and df.no_activation_outside_route.fillna(True).all() and
              df.no_activation_before_move.fillna(True).all() and ret >= 1)
    gate = dict(passed=ok, crashes=crashes, obstacle_routes_with_bypass_and_return=ret, routes=rows)
    write_json(RUN / "gates/v2_few_pbyp2.json", gate)
    print(df.to_string())
    print("passed:", ok)


def few_vred2(_):
    """Checklist over the seed-0 vred2 units: a stop short of the stop line, a release after green, no crash, latency line, accounting."""
    rows, stops, short, rolls, crashes = [], 0, 0, 0, 0
    for rid in ROUTES:
        r = route_row(run_dir("vred2", 0, rid), rid)
        if not r:
            rows.append(dict(route=rid, finished=False))
            continue
        o, c = route_checks(r["attempt"], "red_stop2")
        crashes += bool(r["crash"])
        stops += o["n_r2_stops"] > 0
        short += bool(c["stopped_short_of_line"])
        rolls += bool(c["rolls_after_release"])
        rows.append(dict(route=rid, finished=True, n_r2_stops=o["n_r2_stops"], d_stop=o["d_stop"], DS=r["DS"], crash=bool(r["crash"]),
                         **{k: bool(v) for k, v in c.items()}))
    an = vr.answers("vred2", 0)
    lat = an.lat[an.ok].to_numpy()
    lat_ok = bool(len(lat) and pct(lat, 95) <= 1e3 * L_REG and np.mean(lat > 2500) <= 0.01 and an.ok.mean() >= 0.98)
    df = pd.DataFrame(rows)
    bad_short = int(sum(1 for x in rows if x.get("n_r2_stops", 0) and not x.get("stopped_short_of_line", True)))
    gate = dict(passed=bool(df.finished.all() and crashes == 0 and short >= 1 and rolls >= 1 and lat_ok and bad_short == 0), latency_ok=lat_ok,
                p50_ms=pct(lat, 50), p95_ms=pct(lat, 95), over_L=float(np.mean(lat > 1e3 * L_REG)) if len(lat) else None,
                routes_with_stop=int(stops), routes_stopped_short=int(short), routes_stop_not_short=bad_short, routes_with_release=int(rolls),
                crashes=crashes, routes=rows)
    write_json(RUN / "gates/v2_few_vred2.json", gate)
    print(df.to_string())
    print({k: v for k, v in gate.items() if k != "routes"})


# ------------------------------------------------------------------------------------------------------------ pbyp2
def offline(df):
    """bypass_extract + bypass_misfire classes for pbyp and pbyp2 (runs on the compact extracts, as results/bypass_misfire.md)."""
    import bypass_common as bc
    import bypass_extract as bx
    import bypass_misfire as bm
    d = OUT.parent / ("extract_test" if TEST else "extract")
    sel = df[df.arm.isin(["drive", "pbyp", "pbyp2"])][["arm", "seed", "route", "attempt", "unit"]]
    old = sys.stdin
    sys.stdin = io.StringIO(sel.to_csv(index=False))
    try:
        bx.main(str(d))
    finally:
        sys.stdin = old
    runs = bc.load_all(str(d))
    out = {}
    for arm in ("pbyp", "pbyp2"):
        bm.ARM = arm
        rows = bm.build(runs, {})
        out[arm] = dict(rows=rows, coll=bm.target_collisions(runs, rows), runs={k: v for k, v in runs.items() if k[0] == arm})
    return out, bm, bc


def figure_pbyp2(P, df, path, path2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"pbyp2 - drive": "#3b6ea5", "pbyp2 - pbyp": "#2a9d8f", "pbyp - drive": "#b5651d"}
    sets = ["obstacle routes", "other routes", "all routes"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150)
    for ax, metric, title in zip(axs, ("DS", "collisions", "vehicle_blocked"), ("DS", "collisions per run", "blocked per run")):
        k = 0
        for rs in sets:
            for c in col:
                r = P[(P.contrast == c) & (P.routes == rs) & (P.metric == metric)]
                if len(r) and r.iloc[0].groups:
                    r = r.iloc[0]
                    ax.plot([r.lo, r.hi], [-k, -k], color=col[c], lw=2, solid_capstyle="round")
                    ax.plot(r.est, -k, "o", color=col[c], ms=6, label=c if rs == sets[0] else None)
                k += 1
            k += .6
        ax.axvline(0, color="#222", lw=1)
        ax.set_yticks([-1, -4.6, -8.2])
        ax.set_yticklabels(["obstacle (4)", "other (15)", "all (19)"] if ax is axs[0] else [])
        ax.set_xlabel("paired difference (%s)" % title, fontsize=8)
        ax.grid(axis="x", color="#eee", lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    cm = {"drive": "#777", "pbyp": "#b5651d", "pbyp2": "#3b6ea5"}
    for i, rid in enumerate(OBS_ROUTES):
        for j, arm in enumerate(("drive", "pbyp", "pbyp2")):
            x = df[(df.arm == arm) & (df.route == rid)]
            ax.plot([i + (j - 1) * .22] * len(x), x.DS, "o", color=cm[arm], ms=6, label=arm if i == 0 else None)
            ax.plot([i + (j - 1) * .22 - .08, i + (j - 1) * .22 + .08], [x.DS.mean()] * 2, color=cm[arm], lw=2)
    ax.set_xticks(range(len(OBS_ROUTES)))
    ax.set_xticklabels(OBS_ROUTES)
    ax.set_ylabel("DS (one dot per traffic seed, bar = mean)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", color="#eee", lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path2)
    plt.close(fig)


def report_pbyp2():
    df = collect(["drive", "pbyp", "pbyp2"])
    df.to_csv(OUT / "pbyp2_runs.csv", index=False)
    off, bm, bc = offline(df)
    sets = [("obstacle routes", OBS_ROUTES), ("other routes", OTHER), ("all routes", ROUTES)]
    cols = ["DS", "RC", "collisions", "vehicle_blocked", "red_light", "outside_route_lanes"]
    P = pair_rows(df, [("pbyp2", "drive"), ("pbyp2", "pbyp"), ("pbyp", "drive")], sets, cols)
    P.to_csv(OUT / "pbyp2_paired.csv", index=False)
    D = ["# pbyp2: the privileged bypass with the projection fix, static >= 5 s, red-light memory and a same-direction gap check", "",
         "Diagnostic batch, 19 routes x 2 traffic seeds. Every number is a read at this size, not a confirmation; registered confirmation lines are "
         "listed with their value and marked not evaluated. `drive` and `pbyp` are the existing runs of the earlier batch (not rerun); `pbyp2` ran with "
         "the same base configuration. Definitions: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md); classes of activations: "
         "[bypass_misfire.md](bypass_misfire.md).", ""]
    D += ["## Arms", "", arm_table(df, ["drive", "pbyp", "pbyp2"]).to_markdown(), "",
          "Official infraction counts summed over the runs; DS, RC and mean speed averaged over runs; 38 runs expected per arm. `crashes` are program crashes, excluded "
          "from the paired reads.", ""]
    D += ["## Paired differences (mean over routes of the per-route mean over the two seeds; [95% route-cluster CI])", "",
          "Obstacle routes: 24497, 2520, 19324, 19832 (the scenario places a static obstacle); other routes: the remaining 15. Counts are per run, so +0.50 is one "
          "extra event in one of the two runs of a route.", ""]
    D += pair_md(P, cols, ["DS", "RC", "collisions", "blocked", "red light", "outside lanes"]) + [""]
    obs_d = {a: df[(df.arm == a) & df.route.isin(OBS_ROUTES) & ~df.crash].groupby("route").DS.mean() for a in ("drive", "pbyp", "pbyp2")}
    bm_ = {a: boot_mean(obs_d[a].to_numpy()) for a in obs_d}
    D += ["## DS on the four obstacle routes (the estimate of what a correct bypass is worth)", "",
          "| arm | mean DS over the 4 routes [95% route CI] | per route (mean of 2 seeds) |", "|:--|:--|:--|"]
    for a in ("drive", "pbyp", "pbyp2"):
        D.append("| %s | %.1f [%.1f, %.1f] | %s |" % (a, bm_[a]["est"], bm_[a]["lo"], bm_[a]["hi"], ", ".join("%s %.1f" % (r, v) for r, v in obs_d[a].items())))
    pv = lambda a, b, rs, c: paired(df, a, b, rs, c)  # noqa: E731
    D += ["", "pbyp2 - drive on the obstacle routes: %s; pbyp2 - pbyp: %s. %s" % (fmt(pv("pbyp2", "drive", OBS_ROUTES, "DS")), fmt(pv("pbyp2", "pbyp", OBS_ROUTES, "DS")),
                                                                                  ("CI of pbyp2 - drive %s." % ci_flag(pv("pbyp2", "drive", OBS_ROUTES, "DS")))), ""]
    D += ["Repeat noise: " + NOISE + ". A difference whose CI includes 0 is within that noise at this size.", ""]
    per = df[df.arm.isin(["drive", "pbyp", "pbyp2"])].pivot_table(index="route", columns="arm", values=["DS", "collisions", "vehicle_blocked"], aggfunc="mean").round(2)
    per.columns = ["%s_%s" % (m, a) for m, a in per.columns]
    per.insert(0, "obstacle", [("yes" if r in OBS_ROUTES else "") for r in per.index])
    D += ["## Per route (mean over the two seeds)", "", per.reset_index().to_markdown(index=False), ""]
    # activations
    ORDER = bm.CLASS_ORDER
    D += ["## Activations by class (classes of results/bypass_misfire.md)", ""]
    head = ["class", "pbyp non-target", "pbyp2 non-target", "pbyp obstacle routes", "pbyp2 obstacle routes"]
    cnt = lambda arm, tgt, c: sum(1 for f in off[arm]["rows"] if f["target"] == tgt and f["cls"] == c)  # noqa: E731
    rows = [[c] + [cnt(a, t, c) for t in (False, True) for a in ("pbyp", "pbyp2")] for c in ORDER]
    rows = [[r[0], r[1], r[2], r[3], r[4]] for r in rows if any(r[1:])]
    rows.append(["all", *[sum(1 for f in off[a]["rows"] if f["target"] == t) for t in (False, True) for a in ("pbyp", "pbyp2")]])
    D += [bm.mdtable(head, rows), "", "Non-target = 15 routes x 2 seeds, obstacle = 4 routes x 2 seeds. `scenario_obstacle` = the scenario's own static obstacle "
          "(correct); every other class is a misfire.", ""]
    acts = []
    for f in sorted(off["pbyp2"]["rows"], key=lambda f: (f["route"], f["seed"], f["t0"])):
        acts.append([f["route"], f["seed"], "%.1f" % f["t0"], f["cls"], f.get("lead_type", "-"), bm.fmt(f.get("lead_s")), bm.fmt(f.get("ext_m")), bm.fmt(f["ego_s"]), bm.fmt(f["ego_v"]),
                     "oncoming" if f["borrow"] else "same dir", bm.fmt(f.get("static_for")), bm.fmt(f.get("resume_dt")), bm.outcome(f),
                     ";".join("%s@%.1f" % (e["kind"], e["t"]) for e in f["events"])])
    D += ["### Every pbyp2 activation", "", bm.mdtable(["route", "seed", "t0 [s]", "class", "blocker", "blocker s", "ext [m]", "ego s", "ego v", "lane", "static for [s]",
                                                       "resumes after [s]", "outcome within 15 s", "events in episode"], acts, left=(0, 3, 4, 9, 12, 13)) if acts else "none", ""]
    # per-run gap / suppression diagnostics and the checklist items
    diag, early = [], []
    for rid in ROUTES:
        for s in SEEDS:
            r = route_row(run_dir("pbyp2", s, rid), rid)
            if not r:
                continue
            o, c = route_checks(r["attempt"], "pbyp2")
            if o["n_activations"] or o["n_gap_hold"] or o["n_red_memory"]:
                diag.append(dict(route=rid, seed=s, activations=o["n_activations"], gap_hold_snapshots=o["n_gap_hold"], red_memory_snapshots=o["n_red_memory"],
                                 max_outside_m=round(o["outside_m"], 1)))
            if c.get("no_activation_before_move") is False:
                early.append((rid, s))
    D += ["## Suppression and gap-hold diagnostics (runs with at least one of them; snapshots are 0.2 s)", "",
          pd.DataFrame(diag).to_markdown(index=False) if diag else "none", "",
          "Activations with a blocker outside the route: %d. Activations before the ego first moved on a route without a scenario obstacle: %d %s." % (
              sum(1 for d in diag if d["max_outside_m"] > 0), len(early), early or ""), ""]
    # collisions
    coll = off["pbyp2"]["coll"]
    D += ["## Collisions of pbyp2 on the obstacle routes", ""]
    if coll:
        cr = [[x["route"], x["seed"], "%.1f" % x["t"], "%s %s" % (x["type"], x["actor"]), x.get("actor_dir", "-"), bm.fmt(x.get("actor_v")), bm.fmt(x["ego_v"]), x["phase"],
               "yes" if x["bypass"] else "no", bm.fmt(x["ego_lat"]), bm.fmt(x.get("actor_lat")), bm.fmt(x.get("closing")), x["act_cls"]] for x in coll]
        D += [bm.mdtable(["route", "seed", "t [s]", "against", "actor direction", "actor v", "ego v", "phase vs the obstacle", "path shifted", "ego lat [m]", "actor lat [m]",
                          "closing speed (+ = ego approaching)", "activation class"], cr, left=(0, 3, 4, 7, 12)), ""]
        D += ["Phases: before ramp / entering (ramp) = pulling out, abeam obstacle, just past obstacle, returning, after return; `no state yet` and `cleared` are outside an "
              "active bypass state. Old pbyp for comparison: 12 collisions on these routes, 5 while pulling into the adjacent lane against same-direction vehicles "
              "(results/bypass_misfire.md section 5).", ""]
    else:
        D += ["No collision on the obstacle routes.", ""]
    D += ["## Collisions of pbyp2 on the other routes", ""]
    ev = []
    for (arm, seed, rid), d in sorted(off["pbyp2"]["runs"].items()):
        for e in d.get("_events", []):
            if e["kind"].startswith("collisions") and rid not in OBS_ROUTES:
                ct = next((c["type"] for c in d["contacts"] if c["id"] == e["id"]), "?")
                dr = [x for x in df[(df.arm == "drive") & (df.route == rid) & (df.seed == seed)].collisions]
                ev.append([rid, seed, e["kind"], "%.1f" % e["t"] if e["t"] is not None else "-", ct, e.get("att") or "outside any activation", dr[0] if dr else "-"])
    D += [bm.mdtable(["route", "seed", "kind", "t [s]", "against", "in activation episode of class", "drive collisions in the paired run"], ev, left=(0, 2, 4, 5)) if ev else "none", ""]
    D += ["## Registered lines (plan 4.3), pbyp2", "", "| line | read | status |", "|:--|:--|:--|"]
    non = pv("pbyp2", "drive", OTHER, "DS")
    tg = pv("pbyp2", "drive", OBS_ROUTES, "DS")
    D += ["| harmless on non-target routes: DS CI lower bound >= -5, no new blocked, no new collision | DS %s; blocked %+d, collisions %+d | not evaluated at this size (%d routes < 30) |" % (
              fmt(non), non["d_vehicle_blocked"], non["d_collisions"], non["groups"]),
          "| useful on target routes: DS CI lower bound > 0 | DS %s; collisions %+d, blocked %+d | not evaluated at this size (%d routes < 30) |" % (
              fmt(tg), tg["d_collisions"], tg["d_vehicle_blocked"], tg["groups"]), ""]
    figure_pbyp2(P, df, OUT / "pbyp2_paired.png", OUT / "pbyp2_obstacle_ds.png")
    D += ["![paired differences](pbyp2_paired.png)", "", "Figure: paired differences per route set (dot: mean over routes, bar: 95% route-cluster CI) for pbyp2 - drive, pbyp2 - pbyp "
          "and the old pbyp - drive; DS on the left, collisions per run in the middle, blocked per run on the right. Look at the `other routes` rows (did the loss of old pbyp "
          "disappear) and at whether the obstacle-route bars still clear zero.", "",
          "![DS on the obstacle routes](pbyp2_obstacle_ds.png)", "", "Figure: DS of the four obstacle routes, one dot per traffic seed for drive, pbyp and pbyp2 (bar: mean of "
          "the two seeds). Look at how far pbyp2 sits above drive on each route and whether the two seeds agree.", ""]
    (OUT / "pbyp2.md").write_text("\n".join(D) + "\n")
    write_json(OUT / "pbyp2_summary.json", dict(arms=arm_table(df, ["drive", "pbyp", "pbyp2"]).reset_index().to_dict("records")))
    return df


# ------------------------------------------------------------------------------------------------------------ vred2
def hold_flags(ans_df, arm):
    """Add `hold` to the answers: the table was holding the car (rule R2) in the 0.5 s state row of the request."""
    flag = []
    for (rid, s), g in ans_df.groupby(["route", "seed"], sort=False):
        r = route_row(run_dir(arm, s, rid), rid)
        _, st, _ = vlm_rows(r["attempt"])
        t = np.array([x["t"] for x in st])
        h = np.array(["R2" in x.get("rules", []) for x in st])
        i = np.clip(np.searchsorted(t, g.t_q.to_numpy() - 1e-6), 0, len(t) - 1)
        flag += list(zip(g.index, h[i]))
    s = pd.Series(dict(flag))
    return ans_df.assign(hold=s.reindex(ans_df.index).fillna(False).astype(bool))


def stop_rows(df, arm):
    rows = []
    for _, r in df[df.arm == arm].iterrows():
        for e in r2_episodes(r.attempt, "pred" if arm == "pred" else "R2"):
            if e["d_stop"] is not None:
                rows.append(dict(arm=arm, route=r.route, seed=r.seed, t=round(e["t_stop"], 1), d_stop=e["d_stop"], d_min=e["d_min"], hold_s=round(e["t1"] - e["t0"], 1)))
    return pd.DataFrame(rows)


def figure_vred2(P, stops, ans, path, path_stop, path_lat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"vred2 - drive": "#3b6ea5", "vred2 - vred": "#2a9d8f", "vred2 - pred": "#b5651d"}
    sets = ["all routes", "logged-light routes", "no logged light"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)
    for ax, metric, title in ((axs[0], "DS", "DS"), (axs[1], "red_light", "red-light infractions per run")):
        k = 0
        for rs in sets:
            for c in col:
                r = P[(P.contrast == c) & (P.routes == rs) & (P.metric == metric)]
                if len(r) and r.iloc[0].groups:
                    r = r.iloc[0]
                    ax.plot([r.lo, r.hi], [-k, -k], color=col[c], lw=2, solid_capstyle="round")
                    ax.plot(r.est, -k, "o", color=col[c], ms=6, label=c if rs == sets[0] else None)
                k += 1
            k += .6
        ax.axvline(0, color="#222", lw=1)
        ax.set_yticks([-1, -4.6, -8.2])
        ax.set_yticklabels(["all", "logged light", "no light"] if ax is axs[0] else [])
        ax.set_xlabel("paired difference (%s)" % title, fontsize=8)
        ax.grid(axis="x", color="#eee", lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=150)
    cm = {"pred": "#b5651d", "vred": "#777", "vred2": "#3b6ea5"}
    for i, arm in enumerate(("pred", "vred", "vred2")):
        x = stops[stops.arm == arm]
        y = i + (np.random.default_rng(i).random(len(x)) - .5) * .35
        ax.plot(x.d_stop, y, "o", color=cm[arm], ms=5, alpha=.8)
        ax.plot(x.d_min, np.full(len(x), i - .42), "|", color=cm[arm], ms=5, alpha=.5)
    ax.axvline(0, color="#222", lw=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["pred", "vred", "vred2"])
    ax.set_xlabel("distance of the front bumper to the stop line at standstill, m (> 0 = short of the line)", fontsize=8)
    ax.grid(axis="x", color="#eee", lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_stop)
    plt.close(fig)
    lat = np.sort(ans.lat[ans.ok].to_numpy()) / 1e3
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
    ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat), color="#3b6ea5", lw=2)
    ax.axvline(L_REG, color="#222", lw=1)
    ax.axvline(2.5, color="#b5651d", lw=1, ls="--")
    ax.text(L_REG, 0.05, " L = %.2f s" % L_REG, fontsize=7)
    ax.text(2.5, 0.05, " TTL 2.5 s", fontsize=7, color="#b5651d")
    ax.set_xscale("log")
    ax.set_xlabel("answer latency, s (log)", fontsize=8)
    ax.set_ylabel("share of answers <= x", fontsize=8)
    ax.grid(color="#eee", lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_lat)
    plt.close(fig)


def report_vred2():
    arms = ["drive", "pred", "vred", "vred2"]
    df = collect(arms)
    df.to_csv(OUT / "vred2_runs.csv", index=False)
    ans = pd.concat([vr.answers("vred2", s) for s in SEEDS], ignore_index=True)
    ans.to_csv(OUT / "vred2_answers.csv", index=False)
    ans_v = pd.concat([vr.answers("vred", s) for s in SEEDS], ignore_index=True)
    logged = sorted(set(ans.route[(ans.tl != -1) & (ans.tl_dist < 50)]))
    sets = [("all routes", ROUTES), ("logged-light routes", logged), ("no logged light", [r for r in ROUTES if r not in logged])]
    cols = ["DS", "red_light", "vehicle_blocked", "collisions", "v_mean"]
    P = pair_rows(df, [("vred2", "drive"), ("vred2", "vred"), ("vred2", "pred"), ("vred", "drive")], sets, cols)
    P.to_csv(OUT / "vred2_paired.csv", index=False)
    D = ["# vred2: R2 stops short of the traffic light's stop line (zero-shot Qwen3-VL-4B reads the light), closed loop", "",
         "Diagnostic batch, 19 routes x 2 traffic seeds. Every number is a read, not a confirmation; registered confirmation lines are listed with their value and marked not "
         "evaluated. `drive`, `pred` and `vred` are the existing runs (not rerun); `vred2` = `vred` with one change: R2's stop target is the stop line of the light that governs "
         "the ego lane at the next junction on the route (map; the light's state is not read for it), target = bumper-to-line distance - 0.5 m as in `pred`, instead of the "
         "junction entrance. K, release rule, L = 0.35 s, model, resolution and serving are as in `vred`. Plan: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md).", ""]
    D += ["## Arms", "", arm_table(df, arms).to_markdown(), "", "Official infraction counts summed over the runs; DS, RC and mean speed averaged over runs; 38 runs expected per arm.", ""]
    D += ["## Paired differences (mean over routes of the per-route mean over the two seeds; [95% route-cluster CI])", "",
          "`logged-light routes` = routes on which an ego light was seen within 50 m in the vred2 logs (%s); `no logged light` = the other %d. Counts are per run." % (
              ", ".join(logged), len(ROUTES) - len(logged)), ""]
    D += pair_md(P, cols, ["DS", "red light", "blocked", "collisions", "mean speed m/s"]) + [""]
    V, B = df[df.arm == "vred2"].set_index(["route", "seed"]), df[df.arm == "drive"].set_index(["route", "seed"])
    idx = [i for i in V.index.intersection(B.index) if i[0] not in logged]
    D += ["Harm on routes without a logged light (%d runs): runs with more blocked events than their `drive` pair %d; with more collisions %d." % (
        len(idx), int(sum(V.loc[i, "vehicle_blocked"] > B.loc[i, "vehicle_blocked"] for i in idx)), int(sum(V.loc[i, "collisions"] > B.loc[i, "collisions"] for i in idx))), "",
        "Repeat noise: " + NOISE + ". A difference whose CI includes 0 is within that noise at this size.", ""]
    # latency
    lat = ans.lat[ans.ok].to_numpy()
    cal = json.loads((RUN / "gates/v2_cal.json").read_text()) if (RUN / "gates/v2_cal.json").exists() else None
    D += ["## In-batch latency of the light answers", "", "L = %.2f s: an answer is used at t_q + max(L, latency); TTL = 2.5 s." % L_REG, "",
          "| requests | answered ok | p50 ms | p95 ms | p99 ms | max ms | share > L | share > TTL | server queue p95 ms | server service p50 ms |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
          "| %d | %.2f%% | %.0f | %.0f | %.0f | %.0f | %.1f%% | %.2f%% | %.0f | %.0f |" % (len(ans), 100 * ans.ok.mean(), pct(lat, 50), pct(lat, 95), pct(lat, 99), lat.max(),
                                                                                        100 * np.mean(lat > 1e3 * L_REG), 100 * np.mean(lat > 2500), pct(ans.q_ms[ans.ok], 95), pct(ans.svc[ans.ok], 50)), ""]
    if cal:
        D += ["Calibration before the batch (shadow run, %s): p50 %.0f ms, p95 %.0f ms, p99 %.0f ms (%d requests); registered L = 0.35 s %s (`calibration.md`)." % (
            cal["load"], cal["p50_ms"], cal["p95_ms"], cal["p99_ms"], cal["n"], "held" if cal["passed"] else "did NOT hold"), ""]
    # in-loop reading quality, with and without the hold
    D += ["## In-loop reading quality (answers logged during the runs against the simulator's light state)", "",
          "Ego-green recall is shown for all requests and split by whether R2 was holding the car when the request was made (`while holding`): vred holds the car at the junction "
          "entrance, beyond the stop line; vred2 holds it short of the line, with the light in view. Estimate [95% route-cluster CI]; requests and routes behind each.", "",
          "| readout | arm | all requests | while R2 holds | not holding |", "|:--|:--|:--|:--|:--|"]
    for arm, an in (("vred", ans_v), ("vred2", ans)):
        a2 = hold_flags(an[an.ok], arm).reset_index(drop=True)
        m = light_masks(a2)
        for lab, key, hit in (("ego red / yellow answered red, 0-50 m", "red", a2.ans == RED), ("ego green answered green, 0-50 m", "green", a2.ans == "green_for_ego")):
            cells = []
            for sel in (np.ones(len(a2), bool), a2.hold.to_numpy(), ~a2.hold.to_numpy()):
                mk = m[key] & sel
                r = boot_ratio((hit & mk).astype(float), mk.astype(float), a2.route)
                cells.append("%s (n=%d, %d routes)" % (fmt(r, True), r["n"], r["groups"]))
            D.append("| %s | %s | %s |" % (lab, arm, " | ".join(cells)))
    D += [""]
    # stops
    stops = pd.concat([stop_rows(df, a) for a in ("pred", "vred", "vred2")], ignore_index=True)
    stops.to_csv(OUT / "vred2_stops.csv", index=False)
    D += ["## Stop position at every red-light stop", "",
          "Front bumper to the stop line of the governing light (ctx `tl_dist`, simulator truth, evaluation only) at the last plan step of standstill under the arm's hold (where the car finally stood; a car that starts the route standing under a hold is counted where it stood at the end) "
          "(pred: privileged red stop; vred / vred2: R2). Positive = short of the line. `min` = the smallest distance during the hold (negative = the car crept across the line while holding).", "",
          "| arm | stops | median d_stop | min | max | stopped beyond the line | crept across while holding (min < 0) |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for a in ("pred", "vred", "vred2"):
        x = stops[stops.arm == a]
        D.append("| %s | %d | %.2f | %.2f | %.2f | %d | %d |" % (a, len(x), x.d_stop.median() if len(x) else np.nan, x.d_stop.min() if len(x) else np.nan,
                                                               x.d_stop.max() if len(x) else np.nan, int((x.d_stop < 0).sum()), int((x.d_min < 0).sum())))
    D += ["", stops[stops.arm == "vred2"].round(2).to_markdown(index=False), ""]
    # green release, R5
    rel, r5_rows, per_route, infr = [], [], [], []
    for rid in ROUTES:
        row = dict(route=rid, light=("yes" if rid in LIGHT_ROUTES else "") + ("*" if rid in logged else ""))
        for s in SEEDS:
            r = route_row(run_dir("vred2", s, rid), rid)
            if not r:
                continue
            a, st, head = vlm_rows(r["attempt"])
            row["R2_stops_s%d" % s] = len(vr.episodes(st, "R2"))
            for (t0, t1) in vr.episodes(st, "r5"):
                pre = [x for x in st if x["t"] <= t0][-2:]
                r5_rows.append(dict(route=rid, seed=s, t=round(t0, 1), duration_s=round(t1 - t0, 1),
                                    cause="answer not fresh" if any(not x.get("fresh", True) for x in pre) else "held > T_max"))
            for x in vr.green_release(a, st):
                rel.append(dict(route=rid, seed=s, **x))
            for e in vr.infraction_events(rid, s, r):
                T = e["cross"][0] if e["cross"] else None
                lab = vr.classify(r, T) if T is not None else "crossing not found in the log (n_cross=%d)" % e["n_cross"]
                tl, _ = vr.timeline(r, T) if T is not None else ([], None)
                infr.append(dict(route=rid, seed=s, t_line=None if T is None else round(e["cross"][4], 1), state_at_line={1: "yellow", 2: "red"}.get(e["cross"][3]) if T is not None else None,
                                 t_red_rolling=None if T is None else round(T, 1), v_line=None if T is None else round(e["cross"][2], 1), cause=lab, timeline=tl))
        for arm in arms:
            x = df[(df.arm == arm) & (df.route == rid)]
            row.update({"DS_" + arm: round(x.DS.mean(), 1), "red_" + arm: int(x.red_light.sum())})
        row["blocked_vred2"] = int(df[(df.arm == "vred2") & (df.route == rid)].vehicle_blocked.sum())
        row["coll_drive"] = int(df[(df.arm == "drive") & (df.route == rid)].collisions.sum())
        row["coll_vred2"] = int(df[(df.arm == "vred2") & (df.route == rid)].collisions.sum())
        row["v_vred2"] = round(df[(df.arm == "vred2") & (df.route == rid)].v_mean.mean(), 2)
        per_route.append(row)
    pr = pd.DataFrame(per_route)
    pr["dDS_vs_drive"] = (pr.DS_vred2 - pr.DS_drive).round(1)
    rl = pd.DataFrame(rel)
    D += ["## Green after red", ""]
    if len(rl):
        D += ["%d red-to-green changes of the ego light while R2 held the car. Seconds after the light turned green (median [p5, p95]): second consecutive green answer in force %.1f [%.1f, %.1f] "
              "(%d never reached); R2 released %.1f [%.1f, %.1f]; car rolling (> 1 m/s) %.1f [%.1f, %.1f] (%d never)." % (
                  len(rl), np.nanmedian(rl.answer), pct(rl.answer.dropna(), 5), pct(rl.answer.dropna(), 95), int(rl.answer.isna().sum()), np.nanmedian(rl.r2_end),
                  pct(rl.r2_end.dropna(), 5), pct(rl.r2_end.dropna(), 95), np.nanmedian(rl.roll), pct(rl.roll.dropna(), 5), pct(rl.roll.dropna(), 95), int(rl.roll.isna().sum())), "",
              rl.round(1).to_markdown(index=False), ""]
    else:
        D += ["No red-to-green change of the ego light occurred while R2 held the car.", ""]
    D += ["## R5 fallback", "", "%d episodes in the %d vred2 runs." % (len(r5_rows), len(df[df.arm == "vred2"])) + (("\n\n" + pd.DataFrame(r5_rows).to_markdown(index=False)) if r5_rows else ""), ""]
    D += ["## Remaining red-light infractions in vred2 (%d)" % len(infr), "",
          "Crossing time = the tick where the logged ego light, red, passed from distance > 0 to <= 0 from the stop line (ticks.jsonl); the cause label is the machine reading of the logs "
          "(same rules as `vred.md`) and the timelines below are the evidence.", "",
          pd.DataFrame([{k: v for k, v in x.items() if k != "timeline"} for x in infr]).to_markdown(index=False) if infr else "none", ""]
    for x in infr:
        if x["timeline"]:
            D += ["", "Route %s seed %s, stop line at t = %s s (%s), rolling on red at t = %s s: %s" % (x["route"], x["seed"], x["t_line"], x["state_at_line"], x["t_red_rolling"], x["cause"]), "",
                  pd.DataFrame(x["timeline"]).to_markdown(index=False)]
    D += ["", "## Per route (DS and red-light counts: means / sums over the two seeds; light: yes = scenario set, * = ego light seen in the vred2 logs)", "",
          pr[["route", "light", "DS_drive", "DS_vred", "DS_vred2", "dDS_vs_drive", "DS_pred", "red_drive", "red_vred", "red_vred2", "red_pred", "R2_stops_s0", "R2_stops_s1",
              "blocked_vred2", "coll_drive", "coll_vred2", "v_vred2"]].to_markdown(index=False), ""]
    pv = lambda a, b, rs, c: paired(df, a, b, rs, c)  # noqa: E731
    non = [r for r in ROUTES if r not in logged]
    r_ds, r_nl = pv("vred2", "drive", non, "DS"), pv("vred2", "drive", logged, "DS")
    D += ["## Registered lines (plan 4.3), vred2", "", "| line | read | status |", "|:--|:--|:--|",
          "| harmless on routes without a light: DS CI lower bound >= -5, no new blocked, no new collision | DS %s; blocked %+d, collisions %+d | not evaluated at this size (%d routes < 30) |" % (
              fmt(r_ds), r_ds["d_vehicle_blocked"], r_ds["d_collisions"], r_ds["groups"]),
          "| useful on routes with a light: red-light infractions fall, DS CI lower bound > 0 | DS %s; red light %+d runs | not evaluated at this size (%d routes < 30) |" % (
              fmt(r_nl), r_nl["d_red_light"], r_nl["groups"]), ""]
    figure_vred2(P, stops, ans, OUT / "vred2_paired.png", OUT / "vred2_stop_position.png", OUT / "vred2_latency.png")
    D += ["![paired differences](vred2_paired.png)", "", "Figure: paired differences for vred2 - drive, vred2 - vred and vred2 - pred per route set (dot: mean over routes, bar: 95% route-cluster CI), DS on the "
          "left and red-light infractions per run on the right. Look at whether vred2 moved against vred and whether any bar clears zero.", "",
          "![stop position](vred2_stop_position.png)", "", "Figure: distance of the front bumper to the stop line at standstill for every red-light stop of pred, vred and vred2 (dots; ticks below the "
          "dots: the smallest distance during the hold). Look at which side of the zero line the dots sit: vred sat beyond it, vred2 should sit just short of it, like pred.", "",
          "![in-batch latency](vred2_latency.png)", "", "Figure: cumulative distribution of the in-batch answer latency with L = 0.35 s and the TTL. Look at how much of the curve lies right of L."]
    (OUT / "vred2.md").write_text("\n".join(D) + "\n")
    write_json(OUT / "vred2_summary.json", dict(arms=arm_table(df, arms).reset_index().to_dict("records"), logged=logged))


def report(_):
    OUT.mkdir(parents=True, exist_ok=True)
    report_pbyp2()
    report_vred2()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "few-pbyp2", "few-vred2", "report", "pbyp2", "vred2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"calibrate": calibrate, "few-pbyp2": few_pbyp2, "few-vred2": few_vred2, "report": report,
     "pbyp2": lambda _: report_pbyp2(), "vred2": lambda _: report_vred2()}[a.cmd](a)
