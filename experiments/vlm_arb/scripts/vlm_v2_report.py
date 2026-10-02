"""Gates and reports of the v2 lane: arms `pbyp2` and `vred2` (plan 2026-10-02-pbyp2-vred2.md).

  python vlm_v2_report.py calibrate   in-loop latency of the shadow units `cal2-s1-q*` -> gates/v2_cal.json (passes only with p95 <= L)
  python vlm_v2_report.py calibrate3  the same at 3 workers per card, alone on the box -> gates/v2_cal3.json
  python vlm_v2_report.py few-pbyp2   checklist over the pbyp2 seed-0 units -> gates/v2_few_pbyp2.json
  python vlm_v2_report.py few-vred2   checklist over the vred2 seed-0 units -> gates/v2_few_vred2.json
  python vlm_v2_report.py report      pbyp2.md, vred2.md, tables and figures -> $DATA_DIR/runs/vlm_arb/v2/results/

Conventions of results/report.md: official infractions; paired differences per (route, seed) averaged over seeds, resampling routes
(2000 draws, seed 0), 95% percentile intervals. Every number is a diagnostic read (19 routes); registered confirmation lines are
marked "not evaluated" and never "passed". V2_TEST=1 maps pbyp2 -> pbyp and vred2 -> vred (code-path test on existing runs).
"""
import collections
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
                            fmt, jsonl, route_row, unit_dir, write_json)
from vlm_arb_report import paired  # noqa: E402
from vlm_thin_common import RED, light_masks  # noqa: E402

TEST = os.environ.get("V2_TEST") == "1"
OUT = RUN / ("v2/results_test" if TEST else "v2/results")
SHARDS = ("q0", "q1", "q2")
L_REG = 0.35
ALIAS = {"pbyp2": "pbyp", "vred2": "vred", "vred3": "vred", "cal2": "cal"} if TEST else {}
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
def calibrate(a, arm="cal2", gate_name="v2_cal", load="3 shadow + 3 pbyp2 workers per card on 3 cards"):
    df = vr.answers(arm, 1)
    if df.empty:
        raise SystemExit("no calibration answers")
    lat = df.lat[df.ok].to_numpy()
    p95 = pct(lat, 95)
    gate = dict(done=True, n=int(len(df)), ok_rate=float(df.ok.mean()), p50_ms=pct(lat, 50), p95_ms=p95, p99_ms=pct(lat, 99), max_ms=float(lat.max()),
                over_L=float(np.mean(lat > 1e3 * L_REG)), over_ttl=float(np.mean(lat > 2500)), queue_p95_ms=pct(df.q_ms[df.ok], 95),
                svc_p50_ms=pct(df.svc[df.ok], 50), L_s=L_REG, load=load, arm=arm,
                passed=bool(p95 <= 1e3 * L_REG and df.ok.mean() >= 0.98 and np.mean(lat > 2500) <= 0.01))
    write_json(RUN / "gates" / (gate_name + ".json"), gate)
    txt = ("Calibration (shadow `drive`, Qwen3-VL-4B servers, %s, seed 1, 19 routes): %d requests, %.1f%% answered. In-loop latency p50 %.0f ms, "
           "p95 %.0f ms, p99 %.0f ms, max %.0f ms; server queue wait p95 %.0f ms, service p50 %.0f ms; share > L (%.2f s) %.1f%%, share > TTL 2.5 s %.2f%%. "
           "Registered L = %.2f s holds only if p95 <= %.0f ms: %s.\n" % (gate["load"], len(df), 100 * gate["ok_rate"], gate["p50_ms"], p95, gate["p99_ms"],
                                                                        gate["max_ms"], gate["queue_p95_ms"], gate["svc_p50_ms"], L_REG, 100 * gate["over_L"],
                                                                        100 * gate["over_ttl"], L_REG, 1e3 * L_REG, "yes" if gate["passed"] else "NO"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("calibration.md" if arm == "cal2" else "calibration_%s.md" % arm)).write_text(txt)
    print(txt)


def calibrate3(a):
    calibrate(a, "cal3", "v2_cal3", "3 shadow workers per card on 3 cards, nothing else on the box (the `vred` batch's load)")


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
    bad_short = int(sum(1 for x in rows if x.get("n_r2_stops", 0) and x.get("d_stop") is not None and x["d_stop"] < 0))   # a stop beyond the stop line
    gate = dict(passed=bool(df.finished.all() and crashes == 0 and short >= 1 and rolls >= 1 and lat_ok and bad_short == 0), latency_ok=lat_ok,
                p50_ms=pct(lat, 50), p95_ms=pct(lat, 95), over_L=float(np.mean(lat > 1e3 * L_REG)) if len(lat) else None,
                routes_with_stop=int(stops), routes_stopped_short=int(short), routes_stop_not_short=bad_short, routes_with_release=int(rolls),
                crashes=crashes, routes=rows)
    write_json(RUN / "gates/v2_few_vred2.json", gate)
    print(df.to_string())
    print({k: v for k, v in gate.items() if k != "routes"})


def few_vred3(_):
    """Checklist over the seed-0 vred3 units: no crash, R2 stops short of the line, a release after green, R1 on the approach and off past the stop
    line, no R2 start with the front bumper past the line, latency lines, a decision line logged for every yellow encounter."""
    rows, stops, short, rolls, crashes, bad_line, r1_on, bad_r1 = [], 0, 0, 0, 0, 0, 0, 0
    for rid in ROUTES:
        r = route_row(run_dir("vred3", 0, rid), rid)
        if not r:
            rows.append(dict(route=rid, finished=False))
            continue
        o, c = route_checks(r["attempt"], "red_stop3")
        ans, st, _ = vlm_rows(r["attempt"])
        starts = [x for prev, x in zip(st, st[1:]) if "R2" in x.get("rules", []) and "R2" not in prev.get("rules", [])]
        late = [x for x in starts if x.get("d_stop") is not None and x["d_stop"] <= 0.0]
        crashes += bool(r["crash"])
        stops += o["n_r2_stops"] > 0
        short += bool(c["stopped_short_of_line"])
        rolls += bool(c["rolls_after_release"])
        bad_line += len(late)
        r1_on += bool(c["r1_on_approach"])
        bad_r1 += not c["r1_off_past_stop_line"]
        rows.append(dict(route=rid, finished=True, crash=bool(r["crash"]), n_r2_stops=o["n_r2_stops"], d_stop=o["d_stop"], n_yellow=o["n_yellow_events"], n_go=o["n_go"], r2_start_past_line=len(late),
                         r1_on_approach=bool(c["r1_on_approach"]), r1_off_past_line=bool(c["r1_off_past_stop_line"]), DS=r["DS"]))
    an = vr.answers("vred3", 0)
    lat = an.lat[an.ok].to_numpy()
    lat_ok = bool(len(lat) and pct(lat, 95) <= 1e3 * L_REG and np.mean(lat > 2500) <= 0.01 and an.ok.mean() >= 0.98)
    df = pd.DataFrame(rows)
    bad_short = int(sum(1 for x in rows if x.get("n_r2_stops", 0) and x.get("d_stop") is not None and x["d_stop"] < 0))
    gate = dict(passed=bool(df.finished.all() and crashes == 0 and short >= 1 and rolls >= 1 and lat_ok and bad_line == 0 and bad_r1 == 0 and r1_on >= 1), latency_ok=lat_ok,
                p50_ms=pct(lat, 50), p95_ms=pct(lat, 95), over_L=float(np.mean(lat > 1e3 * L_REG)) if len(lat) else None, routes_with_stop=int(stops), routes_stopped_short=int(short),
                routes_stop_beyond_line=bad_short, routes_with_release=int(rolls), r2_starts_past_line=int(bad_line), routes_r1_on=int(r1_on), routes_r1_after_line=int(bad_r1), crashes=crashes, routes=rows)
    write_json(RUN / "gates/v2_few_vred3.json", gate)
    print(df.to_string())
    print({k: v for k, v in gate.items() if k != "routes"})


# ------------------------------------------------------------------------------------------------------------ pbyp2
def offline(df):
    """bypass_extract + bypass_misfire classes for pbyp and pbyp2 (runs on the compact extracts, as results/bypass_misfire.md)."""
    import bypass_common as bc
    import bypass_extract as bx
    import bypass_misfire as bm
    d = OUT.parent / ("extract_test" if TEST else "extract")
    sel = df[df.arm.isin(["drive", "pbyp", "pbyp2", "pbyp2ng"])][["arm", "seed", "route", "attempt", "unit"]]
    old = sys.stdin
    sys.stdin = io.StringIO(sel.to_csv(index=False))
    try:
        bx.main(str(d))
    finally:
        sys.stdin = old
    runs = bc.load_all(str(d))
    out = {}
    for arm in ("pbyp", "pbyp2", "pbyp2ng"):
        bm.ARM = arm
        rows = bm.build(runs, {})
        for f in rows:           # a scenario's cones pushed by the ego's own contact exceed the 0.5 m/s of the `never moves` rule: still the scenario obstacle
            if f["target"] and f.get("lead_type", "").startswith("static.prop") and f["cls"] != "scenario_obstacle":
                f["cls"], f["relabelled"] = "scenario_obstacle", True
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
    cm = {"drive": "#777", "pbyp": "#b5651d", "pbyp2": "#3b6ea5", "pbyp2ng": "#2a9d8f"}
    arms = [a for a in cm if (df.arm == a).any()]
    for i, rid in enumerate(OBS_ROUTES):
        for j, arm in enumerate(arms):
            x = df[(df.arm == arm) & (df.route == rid)]
            xc = i + (j - (len(arms) - 1) / 2) * .2
            ax.plot([xc] * len(x), x.DS, "o", color=cm[arm], ms=6, label=arm if i == 0 else None)
            ax.plot([xc - .07, xc + .07], [x.DS.mean()] * 2, color=cm[arm], lw=2)
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


def ablation_md(df, off, bm):
    """The diagnostic ablation pbyp2ng (pbyp2 without any gap check) on the four obstacle routes; empty while it has not run."""
    ng = df[df.arm == "pbyp2ng"]
    if ng.empty:
        return []
    D = ["## Diagnostic ablation: pbyp2ng = pbyp2 without any gap check (obstacle routes only)", "",
         "Added after the pbyp2 batch showed the frozen same-direction gap rule stalling the car on three of the four obstacle routes (plans section 10). Same projection fix, static >= 5 s and "
         "light memory as pbyp2; no same-direction check and no oncoming-lane check. 4 obstacle routes x 2 seeds = 8 runs; a diagnostic read of what the corrected activation alone is worth.", ""]
    cols = ["DS", "RC", "collisions", "vehicle_blocked"]
    P = pair_rows(df, [("pbyp2ng", "drive"), ("pbyp2ng", "pbyp2"), ("pbyp2ng", "pbyp")], [("obstacle routes", OBS_ROUTES)], cols)
    P.to_csv(OUT / "pbyp2ng_paired.csv", index=False)
    D += pair_md(P, cols, ["DS", "RC", "collisions", "blocked"]) + [""]
    obs = {a: df[(df.arm == a) & df.route.isin(OBS_ROUTES) & ~df.crash].groupby("route").DS.mean() for a in ("drive", "pbyp", "pbyp2", "pbyp2ng")}
    D += ["| arm | mean DS over the 4 routes [95% route CI] | per route (mean of 2 seeds) |", "|:--|:--|:--|"]
    for a, x in obs.items():
        b = boot_mean(x.to_numpy())
        D.append("| %s | %.1f [%.1f, %.1f] | %s |" % (a, b["est"], b["lo"], b["hi"], ", ".join("%s %.1f" % (r, v) for r, v in x.items())))
    D += [""]
    acts = [[f["route"], f["seed"], "%.1f" % f["t0"], f["cls"], f.get("lead_type", "-"), bm.fmt(f["ego_s"]), bm.fmt(f["ego_v"]), bm.outcome(f),
             ";".join("%s@%.1f" % (e["kind"], e["t"]) for e in f["events"])] for f in sorted(off["pbyp2ng"]["rows"], key=lambda f: (f["route"], f["seed"], f["t0"]))]
    D += ["Activations:", "", bm.mdtable(["route", "seed", "t0 [s]", "class", "blocker", "ego s", "ego v", "outcome within 15 s", "events in episode"], acts, left=(0, 3, 4, 7, 8)) if acts else "none", ""]
    coll = off["pbyp2ng"]["coll"]
    if coll:
        cr = [[x["route"], x["seed"], "%.1f" % x["t"], "%s %s" % (x["type"], x["actor"]), x.get("actor_dir", "-"), bm.fmt(x.get("actor_v")), bm.fmt(x["ego_v"]), x["phase"],
               "yes" if x["bypass"] else "no", bm.fmt(x["ego_lat"]), bm.fmt(x.get("actor_lat")), x["act_cls"]] for x in coll]
        D += ["Collisions:", "", bm.mdtable(["route", "seed", "t [s]", "against", "actor direction", "actor v", "ego v", "phase vs the obstacle", "path shifted", "ego lat [m]",
                                              "actor lat [m]", "activation class"], cr, left=(0, 3, 4, 7, 11)), ""]
    else:
        D += ["No collision.", ""]
    return D


def report_pbyp2():
    df = collect(["drive", "pbyp", "pbyp2", "pbyp2ng"])
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
    D += ["## Arms", "", arm_table(df, ["drive", "pbyp", "pbyp2"] + (["pbyp2ng"] if (df.arm == "pbyp2ng").any() else [])).to_markdown(), "",
          "Official infraction counts summed over the runs; DS, RC and mean speed averaged over runs; 38 runs expected per arm (pbyp2ng, the diagnostic ablation of the section "
          "below, ran on the 4 obstacle routes only: 8 runs). `crashes` are program crashes, excluded from the paired reads.", ""]
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
          "(correct); every other class is a misfire. Cones that the ego's own contact pushed above the 0.5 m/s of the `never moves` rule would be labelled `moving_traffic_pause` by the "
          "old classifier; they are counted as `scenario_obstacle` (blocker is a `static.prop` on an obstacle route): %d activations of pbyp2, %d of pbyp relabelled." % (
              sum(bool(f.get("relabelled")) for f in off["pbyp2"]["rows"]), sum(bool(f.get("relabelled")) for f in off["pbyp"]["rows"])), ""]
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
    D += ablation_md(df, off, bm)
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
    write_json(OUT / "pbyp2_summary.json", dict(arms=arm_table(df, ["drive", "pbyp", "pbyp2", "pbyp2ng"]).reset_index().to_dict("records")))
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


def figure_vred2(P, stops, ans, path, path_stop, path_lat, A="vred2", others=("vred", "pred")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#3b6ea5", "#2a9d8f", "#b5651d", "#8e5ea2", "#777777"]
    col = {"%s - %s" % (A, b): palette[i] for i, b in enumerate(("drive",) + tuple(others))}
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
    cm = {"pred": "#b5651d", "vred": "#777", "vred2": "#2a9d8f", "vred3": "#3b6ea5"}
    stop_arms = [a for a in ("pred", "vred", "vred2", "vred3") if a in set(stops.arm)]
    for i, arm in enumerate(stop_arms):
        x = stops[stops.arm == arm]
        y = i + (np.random.default_rng(i).random(len(x)) - .5) * .35
        ax.plot(x.d_stop, y, "o", color=cm[arm], ms=5, alpha=.8)
        ax.plot(x.d_min, np.full(len(x), i - .42), "|", color=cm[arm], ms=5, alpha=.5)
    ax.axvline(0, color="#222", lw=1)
    ax.set_yticks(range(len(stop_arms)))
    ax.set_yticklabels(stop_arms)
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


def collisions_of(r):
    """Official collisions of a run with the actor's position in the ego frame at first contact (along < 0: behind the ego) and the ego speed."""
    a = Path(r["attempt"])
    first = {}
    for c in jsonl(a / "contacts.jsonl"):
        first.setdefault(c["id"], c)
    scene = jsonl(a / "privileged.jsonl")
    out = []
    for cid, c in first.items():
        if not c["type"].startswith("vehicle."):
            continue
        t = c["t"] - 1.05                                    # the contact clock runs about 1.05 s ahead of the plan clock (results/bypass_misfire.md)
        row = min(scene, key=lambda x: abs(x["t"] - t))
        e = row["ego"]
        hd = np.array([np.cos(e["yaw"]), np.sin(e["yaw"])])
        act = next((x for x in row["actors"] if x["id"] == cid), None)
        along = float((np.array(act["xyz"][:2]) - np.array(e["xyz"][:2])) @ hd) if act else np.nan
        out.append(dict(t=round(t, 1), actor=c["type"], along=round(along, 1), ego_v=round(e["v"], 1)))
    return out


def yellow_md(df, A, logged):
    """vred3: every yellow encounter (the decision lines `k = y` of the runs) with its outcome, false stops and the collisions after stops."""
    enc, tent, false_stops, colls = [], collections.Counter(), [], []
    for rid in ROUTES:
        for sd in SEEDS:
            r = route_row(run_dir(A, sd, rid), rid)
            if not r:
                continue
            a = Path(r["attempt"])
            ys = [x for x in jsonl(a / "vlm_decisions.jsonl") if x.get("k") == "y"]
            ans, st, _ = vlm_rows(a)
            infr = [e["cross"][0] for e in vr.infraction_events(rid, sd, r) if e["cross"]]
            eps = r2_episodes(a)
            for y in ys:
                if y["ev"] in ("tentative_cancel", "tentative_confirm"):
                    tent[y["ev"]] += 1
            for i, y in enumerate(ys):
                if y["ev"] != "yellow":
                    continue
                t = y["t"]
                nxt = next((z for z in ys[i + 1:] if z["ev"] in ("tentative_cancel", "tentative_confirm", "go_end") and z["t"] - t < 12), None)
                inf = [x for x in infr if t <= x <= t + 15]
                ep = next((e for e in eps if e["t0"] - 0.6 <= t <= e["t1"] + 0.1), None)
                if inf:
                    outcome = "infraction (red light at t = %.1f)" % inf[0]
                elif y["decision"] in ("go", "stop_infeasible_go"):
                    s_ = [x for x in st if x["t"] >= t]
                    # the front bumper 5.9 m past the junction entrance: from the 's' lines (ego_s, junc_dist at the decision)
                    ent = None
                    for x in st:
                        if x["t"] >= t - 0.3:
                            ent = x["ego_s"] + x["junc_dist"]
                            break
                    tc = next((x["t"] for x in s_ if ent is not None and x["ego_s"] + 3.8394 - ent >= 5.9), None)
                    g = min(ans, key=lambda d: abs(d["t_q"] - tc))["gt"] if tc is not None and ans else {}
                    outcome = "cleared before red (tail clear at t = %.1f, light %s)" % (tc, {0: "green", 1: "yellow", 2: "RED"}.get(g.get("tl"), "?")) if tc is not None else "go, never cleared in the log"
                elif ep is not None and ep["d_stop"] is not None:
                    outcome = "stopped %s the stop line (%.1f m)" % ("before" if ep["d_stop"] >= 0 else "past", ep["d_stop"])
                elif nxt is not None and nxt["ev"] == "tentative_cancel":
                    outcome = "tentative stop cancelled by the second answer"
                else:
                    outcome = "no stop episode found"
                enc.append(dict(route=rid, seed=sd, t=round(t, 1), v=y["v"], d_stop_front=y["d_stop"], d_line=y["d_line"], age=y["age"], remaining=y["remaining"],
                                stop_need=y["stop_need"], truth="tl %s @ %s m" % (y["truth"].get("tl"), y["truth"].get("tl_dist")), decision=y["decision"],
                                second_answer=(nxt["ev"].replace("tentative_", "") if nxt and nxt["ev"].startswith("tent") else "-"), outcome=outcome))
            for e in eps:                                   # R2 stops with no red / yellow truth during the hold: false stops
                rows = [x for x in jsonl(a / "plans.jsonl") if not x["warm"] and e["t0"] <= x["t"] <= e["t1"]]
                states = {x.get("ctx", {}).get("tl") for x in rows}
                if rows and not (states & {1, 2}):
                    false_stops.append(dict(route=rid, seed=sd, t0=round(e["t0"], 1), hold_s=round(e["t1"] - e["t0"], 1), truth_states=sorted(str(x) for x in states)))
            for c in collisions_of(r):
                colls.append(dict(route=rid, seed=sd, **c))
    D = ["## Yellow encounters (every decision of the rule, `k = y` lines)", "",
         "`d_stop_front` = front bumper to the stop line, `d_line` = front bumper to the scorer's line (junction entrance), `age` = time since the frame of the last green answer, `remaining` = yellow time "
         "left after the age and the lag margin, `stop_need` = reaction + comfortable braking distance at the speed; `truth` = the simulator's ego light at the decision (evaluation only). Outcome: "
         "infraction = an official red-light event within 15 s; cleared = the front bumper 5.9 m past the junction entrance (tail clear) and the light then; stopped = the last standstill of the R2 hold "
         "against the stop line (+ = short of it).", "",
         pd.DataFrame(enc).to_markdown(index=False) if enc else "No yellow decision was taken in the batch.", ""]
    if enc:
        e_ = pd.DataFrame(enc)
        D += ["Decisions: %s. Outcomes: %s." % (dict(e_.decision.value_counts()), dict(e_.outcome.str.replace(r" \(.*", "", regex=True).value_counts())), ""]
    D += ["## Tentative starts, false stops and collisions", "",
          "Tentative R2 starts at a first non-green answer: confirmed by the second answer %d, cancelled %d. R2 stops whose hold saw no red or yellow truth state (false stops): %d%s." % (
              tent["tentative_confirm"], tent["tentative_cancel"], len(false_stops), (":\n\n" + pd.DataFrame(false_stops).to_markdown(index=False)) if false_stops else ""), ""]
    w = {a_: df[(df.arm == a_)] for a_ in (A, "vred2", "jslow", "drive")}
    D += ["Blocked events / collisions summed over the 38 runs: " + "; ".join("%s %d / %d" % (a_, int(w[a_].vehicle_blocked.sum()), int(w[a_].collisions.sum())) for a_ in w) + ".", "",
          "Collisions of %s with a vehicle (`along` < 0: the other vehicle was behind the ego at first contact = a rear-end hit on the ego; ego speed in m/s):" % A, "",
          pd.DataFrame(colls).to_markdown(index=False) if colls else "none", ""]
    return D


def report_vred2():
    report_light("vred2")


def report_vred3():
    report_light("vred3")


def report_light(A):
    V3 = A == "vred3"
    arms = ["drive", "pred", "vred", "vred2"] + (["vred3", "jslow"] if V3 else [])
    others = ("vred2", "jslow", "vred", "pred") if V3 else ("vred", "pred")
    df = collect(arms)
    df.to_csv(OUT / ("%s_runs.csv" % A), index=False)
    ans = pd.concat([vr.answers(A, s) for s in SEEDS], ignore_index=True)
    ans.to_csv(OUT / ("%s_answers.csv" % A), index=False)
    ans_v = pd.concat([vr.answers("vred", s) for s in SEEDS], ignore_index=True)
    ans_2 = pd.concat([vr.answers("vred2", s) for s in SEEDS], ignore_index=True) if V3 else None
    logged = sorted(set(ans.route[(ans.tl != -1) & (ans.tl_dist < 50)]))
    sets = [("all routes", ROUTES), ("logged-light routes", logged), ("no logged light", [r for r in ROUTES if r not in logged])]
    cols = ["DS", "red_light", "vehicle_blocked", "collisions", "v_mean"]
    P = pair_rows(df, [(A, "drive")] + [(A, b) for b in others] + ([] if V3 else [("vred", "drive")]), sets, cols)
    P.to_csv(OUT / ("%s_paired.csv" % A), index=False)
    if V3:
        D = ["# vred3: vred2 + junction slow-down on the approach + a yellow / commit rule (zero-shot Qwen3-VL-4B reads the light), closed loop", "",
             "Diagnostic batch, 19 routes x 2 traffic seeds. Every number is a read, not a confirmation; no registered line applies to this arm and the confirmation lines are listed as not evaluated. "
             "`drive`, `pred`, `jslow`, `vred`, `vred2` are existing runs (not rerun in this arm's batch). `vred3` = `vred2` (R2 stops short of the stop line) + R1 on the approach only "
             "(cap 4.5 m/s from 25 m before the junction until the front bumper passes the stop line; lifted entirely afterwards and during a committed go) + the yellow decision at the first non-green answer "
             "after green (comfortable stop, else go if the whole car clears the scorer's line before the red, else hard stop) + the commit rule (no R2 start once the front bumper is past the stop line). "
             "Parameters, rule and offline basis: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md) section 11, [vred3_yellow_offline.md](vred3_yellow_offline.md); inputs by source: section 12 of the plan.", ""]
    else:
        D = ["# vred2: R2 stops short of the traffic light's stop line (zero-shot Qwen3-VL-4B reads the light), closed loop", "",
             "Diagnostic batch, 19 routes x 2 traffic seeds. Every number is a read, not a confirmation; registered confirmation lines are listed with their value and marked not "
             "evaluated. `drive`, `pred` and `vred` are the existing runs (not rerun); `vred2` = `vred` with one change: R2's stop target is the stop line of the light that governs "
             "the ego lane at the next junction on the route (map; the light's state is not read for it), target = bumper-to-line distance - 0.5 m as in `pred`, instead of the "
             "junction entrance. K, release rule, L = 0.35 s, model, resolution and serving are as in `vred`. Plan: [plans/2026-10-02-pbyp2-vred2.md](../plans/2026-10-02-pbyp2-vred2.md).", ""]
    D += ["## Arms", "", arm_table(df, arms).to_markdown(), "", "Official infraction counts summed over the runs; DS, RC and mean speed averaged over runs; 38 runs expected per arm.", ""]
    D += ["## Paired differences (mean over routes of the per-route mean over the two seeds; [95% route-cluster CI])", "",
          "`logged-light routes` = routes on which an ego light was seen within 50 m in the %s logs (%s); `no logged light` = the other %d. Counts are per run." % (
              A, ", ".join(logged), len(ROUTES) - len(logged)), ""]
    D += pair_md(P, cols, ["DS", "red light", "blocked", "collisions", "mean speed m/s"]) + [""]
    V, B = df[df.arm == A].set_index(["route", "seed"]), df[df.arm == "drive"].set_index(["route", "seed"])
    idx = [i for i in V.index.intersection(B.index) if i[0] not in logged]
    D += ["Harm on routes without a logged light (%d runs): runs with more blocked events than their `drive` pair %d; with more collisions %d." % (
        len(idx), int(sum(V.loc[i, "vehicle_blocked"] > B.loc[i, "vehicle_blocked"] for i in idx)), int(sum(V.loc[i, "collisions"] > B.loc[i, "collisions"] for i in idx))), "",
        "Repeat noise: " + NOISE + ". A difference whose CI includes 0 is within that noise at this size.", ""]
    # latency
    lat = ans.lat[ans.ok].to_numpy()
    cal = json.loads((RUN / "gates/v2_cal3.json").read_text()) if (RUN / "gates/v2_cal3.json").exists() else None
    cal2 = json.loads((RUN / "gates/v2_cal.json").read_text()) if (RUN / "gates/v2_cal.json").exists() else None
    D += ["## In-batch latency of the light answers", "", "L = %.2f s: an answer is used at t_q + max(L, latency); TTL = 2.5 s." % L_REG, "",
          "| requests | answered ok | p50 ms | p95 ms | p99 ms | max ms | share > L | share > TTL | server queue p95 ms | server service p50 ms |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
          "| %d | %.2f%% | %.0f | %.0f | %.0f | %.0f | %.1f%% | %.2f%% | %.0f | %.0f |" % (len(ans), 100 * ans.ok.mean(), pct(lat, 50), pct(lat, 95), pct(lat, 99), lat.max(),
                                                                                        100 * np.mean(lat > 1e3 * L_REG), 100 * np.mean(lat > 2500), pct(ans.q_ms[ans.ok], 95), pct(ans.svc[ans.ok], 50)), ""]
    if cal:
        D += ["Calibration before the batch (shadow run, %s): p50 %.0f ms, p95 %.0f ms, p99 %.0f ms (%d requests); registered L = 0.35 s %s (`calibration.md`)." % (
            cal["load"], cal["p50_ms"], cal["p95_ms"], cal["p99_ms"], cal["n"], "held" if cal["passed"] else "did NOT hold"), ""]
    if cal2:
        D += ["A first calibration at 6 workers per card (3 shadow + 3 pbyp2 workers, `calibration.md`) gave p50 %.0f ms, p95 %.0f ms, p99 %.0f ms (%d requests) and failed the registered line "
              "(p95 <= 350 ms); concurrency was reduced to the load above before any vred2 unit ran (vred3 ran at that reduced load as well)." % (cal2["p50_ms"], cal2["p95_ms"], cal2["p99_ms"], cal2["n"]), ""]
    # in-loop reading quality, with and without the hold
    D += ["## In-loop reading quality (answers logged during the runs against the simulator's light state)", "",
          "Ego-green recall is shown for all requests and split by whether R2 was holding the car when the request was made (`while holding`): vred holds the car at the junction "
          "entrance, beyond the stop line; vred2 holds it short of the line, with the light in view. Estimate [95% route-cluster CI]; requests and routes behind each.", "",
          "| readout | arm | all requests | while R2 holds | not holding |", "|:--|:--|:--|:--|:--|"]
    for arm, an in [("vred", ans_v)] + ([("vred2", ans_2)] if V3 else []) + [(A, ans)]:
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
    stop_arms = ["pred", "vred", "vred2"] + (["vred3"] if V3 else [])
    stops = pd.concat([stop_rows(df, a) for a in stop_arms], ignore_index=True)
    stops.to_csv(OUT / ("%s_stops.csv" % A), index=False)
    D += ["## Stop position at every red-light stop", "",
          "Front bumper to the stop line of the governing light (ctx `tl_dist`, simulator truth, evaluation only) at the last plan step of standstill under the arm's hold (where the car finally stood; a car that starts the route standing under a hold is counted where it stood at the end) "
          "(pred: privileged red stop; vred / vred2: R2). Positive = short of the line. `min` = the smallest distance during the hold (negative = the car crept across the line while holding).", "",
          "| arm | stops | median d_stop | min | max | stopped beyond the line | crept across while holding (min < 0) |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for a in stop_arms:
        x = stops[stops.arm == a]
        D.append("| %s | %d | %.2f | %.2f | %.2f | %d | %d |" % (a, len(x), x.d_stop.median() if len(x) else np.nan, x.d_stop.min() if len(x) else np.nan,
                                                               x.d_stop.max() if len(x) else np.nan, int((x.d_stop < 0).sum()), int((x.d_min < 0).sum())))
    D += ["", stops[stops.arm == A].round(2).to_markdown(index=False), ""]
    # green release, R5
    rel, r5_rows, per_route, infr = [], [], [], []
    for rid in ROUTES:
        row = dict(route=rid, light=("yes" if rid in LIGHT_ROUTES else "") + ("*" if rid in logged else ""))
        for s in SEEDS:
            r = route_row(run_dir(A, s, rid), rid)
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
        row["blocked_A"] = int(df[(df.arm == A) & (df.route == rid)].vehicle_blocked.sum())
        row["coll_drive"] = int(df[(df.arm == "drive") & (df.route == rid)].collisions.sum())
        row["coll_A"] = int(df[(df.arm == A) & (df.route == rid)].collisions.sum())
        row["v_A"] = round(df[(df.arm == A) & (df.route == rid)].v_mean.mean(), 2)
        per_route.append(row)
    pr = pd.DataFrame(per_route)
    pr["dDS_vs_drive"] = (pr["DS_" + A] - pr.DS_drive).round(1)
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
    D += ["## R5 fallback", "", "%d episodes in the %d %s runs." % (len(r5_rows), len(df[df.arm == A]), A) + (("\n\n" + pd.DataFrame(r5_rows).to_markdown(index=False)) if r5_rows else ""), ""]
    D += ["## Remaining red-light infractions in %s (%d)" % (A, len(infr)), "",
          "Crossing time = the tick where the logged ego light, red, passed from distance > 0 to <= 0 from the stop line (ticks.jsonl); the cause label is the machine reading of the logs "
          "(same rules as `vred.md`) and the timelines below are the evidence.", "",
          pd.DataFrame([{k: v for k, v in x.items() if k != "timeline"} for x in infr]).to_markdown(index=False) if infr else "none", ""]
    for x in infr:
        if x["timeline"]:
            D += ["", "Route %s seed %s, stop line at t = %s s (%s), rolling on red at t = %s s: %s" % (x["route"], x["seed"], x["t_line"], x["state_at_line"], x["t_red_rolling"], x["cause"]), "",
                  pd.DataFrame(x["timeline"]).to_markdown(index=False)]
    D += ["", "## Per route (DS and red-light counts: means / sums over the two seeds; light: yes = scenario set, * = ego light seen in the %s logs; `_A` columns are %s)" % (A, A), "",
          pr[["route", "light"] + ["DS_" + a for a in arms] + ["dDS_vs_drive"] + ["red_" + a for a in arms] + ["R2_stops_s0", "R2_stops_s1", "blocked_A", "coll_drive", "coll_A", "v_A"]].to_markdown(index=False), ""]
    pv = lambda a, b, rs, c: paired(df, a, b, rs, c)  # noqa: E731
    non = [r for r in ROUTES if r not in logged]
    r_ds, r_nl = pv(A, "drive", non, "DS"), pv(A, "drive", logged, "DS")
    if V3:
        D += yellow_md(df, A, logged)
    D += ["## Registered lines (plan 4.3), %s" % A, "", "| line | read | status |", "|:--|:--|:--|",
          "| harmless on routes without a light: DS CI lower bound >= -5, no new blocked, no new collision | DS %s; blocked %+d, collisions %+d | not evaluated at this size (%d routes < 30) |" % (
              fmt(r_ds), r_ds["d_vehicle_blocked"], r_ds["d_collisions"], r_ds["groups"]),
          "| useful on routes with a light: red-light infractions fall, DS CI lower bound > 0 | DS %s; red light %+d runs | not evaluated at this size (%d routes < 30) |" % (
              fmt(r_nl), r_nl["d_red_light"], r_nl["groups"]), ""]
    figure_vred2(P, stops, ans, OUT / ("%s_paired.png" % A), OUT / ("%s_stop_position.png" % A), OUT / ("%s_latency.png" % A), A, others)
    D += ["![paired differences](%s_paired.png)" % A, "", "Figure: paired differences for %s against %s per route set (dot: mean over routes, bar: 95%% route-cluster CI), DS on the "
          "left and red-light infractions per run on the right. Look at whether %s moved against the previous arms and whether any bar clears zero." % (A, ", ".join(("drive",) + others), A), "",
          "![stop position](%s_stop_position.png)" % A, "", "Figure: distance of the front bumper to the stop line at standstill for every red-light stop (dots; ticks below the "
          "dots: the smallest distance during the hold). Look at which side of the zero line the dots sit: vred sat beyond it, the stop-line arms should sit just short of it, like pred.", "",
          "![in-batch latency](%s_latency.png)" % A, "", "Figure: cumulative distribution of the in-batch answer latency with L = 0.35 s and the TTL. Look at how much of the curve lies right of L."]
    (OUT / ("%s.md" % A)).write_text("\n".join(D) + "\n")
    write_json(OUT / ("%s_summary.json" % A), dict(arms=arm_table(df, arms).reset_index().to_dict("records"), logged=logged))


def report(_):
    OUT.mkdir(parents=True, exist_ok=True)
    report_pbyp2()
    report_vred2()


def report3(_):
    OUT.mkdir(parents=True, exist_ok=True)
    report_vred3()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "calibrate3", "few-pbyp2", "few-vred2", "few-vred3", "report", "report3", "pbyp2", "vred2", "vred3"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"calibrate": calibrate, "calibrate3": calibrate3, "few-pbyp2": few_pbyp2, "few-vred2": few_vred2, "few-vred3": few_vred3, "report": report, "report3": report3,
     "pbyp2": lambda _: report_pbyp2(), "vred2": lambda _: report_vred2(), "vred3": lambda _: report_vred3()}[a.cmd](a)
