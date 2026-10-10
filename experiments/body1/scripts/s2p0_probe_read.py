#!/usr/bin/env python
"""BODY1 stage 2, P0-E: read of the input-port probe arms (experiments/body1/plans/2026-10-10-stage2-prereg.md section 4). CPU, numpy only.

  $DATA_DIR/envs/simlingo/bin/python experiments/body1/scripts/s2p0_probe_read.py [--arms A0 A1]

Per route and pooled, per arm ($DATA_DIR/runs/body1/s2p0/<arm>/arm, the attempt named by done/<route>.json; `probe` = the design's cost probe
runs/body1/s2p/arm, the same arm as A0 without the new logging: the spread of an identical rerun):
  stand        share of ticks with speed < 0.1 m/s (ticks.jsonl `v`, every tick of the attempt, the 5 s warm-up included)
  metres       path length of the simulator rear-axle pose (`truth`)
  plan_v1      the student's own plan speed at 1 s (plans.jsonl `vplan[1]`) on standing ticks after the warm-up: median, p90
  excused      share of standing ticks with (light) the hero's own light red or yellow (`ctx.htl`, CARLA's hero.get_traffic_light_state()),
               (actor) a vehicle / walker / prop box within 12 m ahead of the front bumper and +-1.75 m of the heading line (`ctx.ahead`), (either)
  unexcused    standing ticks that are not excused / ALL ticks. The prereg line, read on arm A1: < 0.20 pooled over the six routes and < 0.30 on
               at least five of the six
  ms           wall ms per tick of the route client (done/<route>.json profile total_ms_mean) and the agent's share (agent_ms_mean)
Fixed before the first read, reported next to the line and never in place of it: (a) the same without the warm-up ticks; (b) a second light
rule from the route's own light, `ctx.tl` in (yellow, red) with the stop line at most 40 m ahead of the bumper (the hero's CARLA light state is
set only inside the light's trigger volume; this rule is also the only one the earlier probe logged), with (c) its lead proxy `ctx.lead_gap`
<= 12 m for the earlier probe. No driving score, route completion or infraction count is read.
Outputs: experiments/body1/results/s2p0/probe.{csv,json}, experiments/body1/figs/s2p0/probe_speed.png.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
REPO = _pl.Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(REPO))
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

IDS = "10255 5423 28008 15102 28147 334".split()
OUT, FIG = REPO / "experiments/body1/results/s2p0", REPO / "experiments/body1/figs/s2p0"
V_STAND, WARM_S, LINE_POOLED, LINE_ROUTE = 0.1, 5.0, 0.20, 0.30
BLUE, VERM, RED, GREY, INK, MUTED = "#0072B2", "#D55E00", "#C0392B", "#6B7280", "#1F2933", "#6B7280"


def load(arm_dir, rid):
    done = json.loads((arm_dir / "done" / f"{rid}.json").read_text())
    att = arm_dir / "attempts" / rid / str(done["attempt"])
    T = [json.loads(ln) for ln in open(att / "ticks.jsonl")]
    P = {r["frame"]: r for r in map(json.loads, open(att / "plans.jsonl"))}
    c = [r.get("ctx") or {} for r in T]
    v = np.array([r["v"] for r in T], float)
    t = np.array([r["t"] for r in T], float)
    xy = np.array([r.get("truth", [np.nan] * 3)[:2] for r in T], float)
    vp = np.array([P[r["frame"]]["vplan"][1] if r["frame"] in P else np.nan for r in T], float)
    warm = np.array([P[r["frame"]]["warm"] if r["frame"] in P else True for r in T], bool)
    pe = np.array([P[r["frame"]].get("pe", [np.nan] * 20) if r["frame"] in P else [np.nan] * 20 for r in T], float)
    return dict(t=t - t[0], v=v, xy=xy, vp=vp, warm=warm, pe=pe, logged=any("htl" in x for x in c), err=np.array(["err" in x for x in c]),
                light=np.array([x.get("htl") in ("Red", "Yellow") for x in c]), actor=np.array(["ahead" in x for x in c]),
                rlight=np.array([x.get("tl") in (1, 2) and x.get("tl_dist", 1e9) <= 40.0 for x in c]),
                lead=np.array([x.get("lead_gap", 1e9) <= 12.0 or x.get("ped_gap", 1e9) <= 12.0 for x in c]),
                ms=done.get("profile", {}).get("total_ms_mean"), ms_agent=done.get("profile", {}).get("agent_ms_mean"), wall_s=done.get("wall_s"))


def stats(parts):
    """One row from one route (or several pooled, tick-weighted)."""
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    v, light, actor, rlight, lead, warm, vp = (cat(k) for k in ("v", "light", "actor", "rlight", "lead", "warm", "vp"))
    logged = all(p["logged"] for p in parts)
    st = v < V_STAND
    n, ns = len(v), max(int(st.sum()), 1)
    w = st & ~warm & np.isfinite(vp)
    nan = float("nan")
    sh = lambda m: float((st & m).sum() / ns)  # noqa: E731
    r = dict(ticks=n, sim_s=round(n * 0.05, 1), stand=float(st.mean()),
             metres=float(sum(np.nansum(np.linalg.norm(np.diff(p["xy"], axis=0), axis=1)) for p in parts)),
             plan_v1_median=float(np.median(vp[w])) if w.any() else nan, plan_v1_p90=float(np.percentile(vp[w], 90)) if w.any() else nan,
             excused_light=sh(light) if logged else nan, excused_actor=sh(actor) if logged else nan, excused_either=sh(light | actor) if logged else nan,
             unexcused=float((st & ~(light | actor)).sum() / n) if logged else nan,
             unexcused_nowarm=float((st & ~warm & ~(light | actor)).sum() / max(int((~warm).sum()), 1)) if logged else nan,
             excused_routelight=sh(rlight), unexcused_routelight_actor=float((st & ~(rlight | actor)).sum() / n) if logged else nan,
             unexcused_routelight_lead=float((st & ~(rlight | lead)).sum() / n),
             ms_tick=float(np.average([p["ms"] for p in parts], weights=[len(p["v"]) for p in parts])) if all(p["ms"] for p in parts) else nan,
             ms_agent=float(np.average([p["ms_agent"] for p in parts], weights=[len(p["v"]) for p in parts])) if all(p["ms_agent"] for p in parts) else nan,
             ctx_err=int(cat("err").sum()))
    pe = cat("pe")
    ok = np.isfinite(pe[:, 0]) & ~warm
    if ok.any():                                              # what was fed (logging arms only)
        q = lambda x: [round(float(z), 3) for z in np.percentile(x, [5, 50, 95])]  # noqa: E731
        r["fed"] = dict(ax_p5_p50_p95=q(pe[ok, 6] * 3), ax_standing_p5_p50_p95=q(pe[ok & st, 6] * 3) if (ok & st).any() else None,
                        ay_p5_p50_p95=q(pe[ok, 7] * 3), vy_p5_p50_p95=q(pe[ok, 5] * 10),
                        cmd_left_straight_right=[round(float(pe[ok, k].mean()), 3) for k in (1, 2, 3)])
    return r


def spans(mask, t):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [(t[a], t[min(b, len(t) - 1)]) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def figure(data, arms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"A0": BLUE, "A1": VERM, "A2": "#009E73", "A3": "#CC79A7"}
    name = {"A0": "A0 as shipped", "A1": "A1 both port fixes", "A2": "A2 fix 1 only", "A3": "A3 fix 2 only"}
    vmax = max(float(np.max(data[a][r]["v"])) for a in arms for r in IDS)
    top = max(vmax * 1.05, 1.0)
    fig, ax = plt.subplots(len(IDS), len(arms), figsize=(5.6 * len(arms), 1.55 * len(IDS) + 1.0), sharex="col", sharey=True, squeeze=False)
    for i, rid in enumerate(IDS):
        for j, a in enumerate(arms):
            d, x = data[a][rid], ax[i][j]
            for s0, s1 in spans(d["light"], d["t"]):
                x.axvspan(s0, s1, ymin=0.90, ymax=1.0, color=RED, lw=0)
            for s0, s1 in spans(d["actor"], d["t"]):
                x.axvspan(s0, s1, ymin=0.78, ymax=0.88, color=GREY, lw=0)
            x.plot(d["t"], d["v"], color=col[a], lw=1.2)
            st = d["v"] < V_STAND
            x.text(0.995, 0.70, "standing %.0f %%, unexcused %.0f %% of ticks" % (100 * st.mean(), 100 * (st & ~(d["light"] | d["actor"])).mean()),
                   transform=x.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
            x.set_ylim(0, top / 0.76)
            x.grid(axis="y", color="#E5E7EB", lw=0.6)
            x.set_axisbelow(True)
            for sp in ("top", "right"):
                x.spines[sp].set_visible(False)
            x.tick_params(labelsize=7.5, colors=MUTED)
            if j == 0:
                x.set_ylabel("route %s\nspeed (m/s)" % rid, fontsize=8, color=INK)
            if i == 0:
                x.set_title(name[a], fontsize=9.5, color=INK, loc="left")
            if i == len(IDS) - 1:
                x.set_xlabel("simulation time (s)", fontsize=8, color=INK)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=RED, label="hero's light red or yellow"), Patch(color=GREY, label="actor box within 12 m ahead, +-1.75 m")],
               loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("P0-E: speed of the student (P2H10-F-s0, spec_plan_smooth) on six CARLA junction routes", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def main(a):
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    with Run("body1", "s2p0-probe-read", config=vars(a)) as run:
        root = data_dir() / "runs/body1"
        dirs = dict({"probe": root / "s2p/arm"}, **{arm: root / "s2p0" / arm / "arm" for arm in a.arms})
        data = {k: {r: load(d, r) for r in IDS} for k, d in dirs.items()}
        rows, res = [], {}
        for k in dirs:
            per = {r: stats([data[k][r]]) for r in IDS}
            pooled = stats(list(data[k].values()))
            res[k] = dict(pooled=pooled, routes=per)
            if k != "probe":
                u = [per[r]["unexcused"] for r in IDS]
                res[k]["line"] = dict(pooled=pooled["unexcused"], pooled_ok=bool(pooled["unexcused"] < LINE_POOLED), routes_below_30=int(sum(x < LINE_ROUTE for x in u)),
                                      passed=bool(pooled["unexcused"] < LINE_POOLED and sum(x < LINE_ROUTE for x in u) >= 5))
            rows += [dict(arm=k, route=r, **{c: v for c, v in s.items() if c != "fed"}) for r, s in list(per.items()) + [("pooled", pooled)]]
        if "A0" in res and "A1" in res:
            drop = res["A0"]["pooled"]["unexcused"] - res["A1"]["pooled"]["unexcused"]
            res["verdict"] = dict(line="unexcused standing / all ticks < 0.20 pooled and < 0.30 on >= 5 of 6 routes, arm A1", passed=res["A1"]["line"]["passed"],
                                  a0_minus_a1_points=round(100 * drop, 1), a2_a3_released=bool(res["A1"]["line"]["passed"] or drop >= 0.10))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
        with open(OUT / "probe.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({c: (f"{v:.4g}" if isinstance(v, float) else v) for c, v in r.items()} for r in rows)
        figure(data, [x for x in a.arms], FIG / "probe_speed.png")
        for r in rows:
            run.info("%-5s %-6s ticks %5d stand %.2f m %6.1f plan_v1 %.2f / %.2f excused l %.2f a %.2f e %.2f unexcused %.3f (nowarm %.3f, routelight+actor %.3f, routelight+lead %.3f) ms %.0f err %d",
                     r["arm"], r["route"], r["ticks"], r["stand"], r["metres"], r["plan_v1_median"], r["plan_v1_p90"], r["excused_light"], r["excused_actor"],
                     r["excused_either"], r["unexcused"], r["unexcused_nowarm"], r["unexcused_routelight_actor"], r["unexcused_routelight_lead"], r["ms_tick"], r["ctx_err"])
        for k in res:
            if k != "verdict":
                run.info("%s line %s fed %s", k, json.dumps(res[k].get("line")), json.dumps(res[k]["pooled"].get("fed")))
        run.info("verdict %s", json.dumps(res.get("verdict")))
        run.summary.update(res.get("verdict") or {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A0", "A1"])
    main(ap.parse_args())
