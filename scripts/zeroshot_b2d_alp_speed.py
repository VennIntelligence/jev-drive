#!/usr/bin/env python3
"""Where the Alpamayo zero-shot Bench2Drive runs lose time (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md,
section 7). Reads b2d_run output dirs (plans.jsonl, ticks.jsonl, results.json per attempt), writes

  routes.csv   one row per route: status, RC, sim time, reference (PDM-Lite expert) duration, first collision, the sim time
               split into moving / standstill before the first collision (by what the plan in force said) / pinned after
               it, throttle response while pinned, execution efficiency while moving, overspeed brake holds, lateral
               overrides, projected finish time at the pre-collision pace
  windows.csv  one row per plan hold window (plan to next plan): speed, plan (Zoo) desired speed, controller action,
               planned vs driven distance, speed at the end of the window
  summary.json the same, pooled per run

Definitions
  plan class      from the plan as the controller reads it (forward-only when the run used it): "stop" when the Zoo desired
                  speed (mean waypoint spacing over 0.5 ... 3 s x 2) is < 0.4 m/s (control_pid brakes), else "go"
  action          throttle | brake_stop (desired < 0.4) | brake_over (desired >= 0.4 but speed > 1.1 desired) | coast
  efficiency      driven distance / the plan's distance over the same window (plan arc length at the window length)
  pinned          v < 0.5 m/s after the first official collision
Python 3.8, NumPy only; runs on the Mac or the box.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

V_STALL = 0.5
WP_T = np.arange(1, 7) * 0.5
CENTRE = 1.388633      # rear axle -> vehicle centre (m), as the collision probe


def jl(p):
    out = []
    with open(p) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def prep(path, forward_only):
    """Plan as b2d_zoo_pid_wrap.ZooPID.prepare reads it: origin prepended, backward segments dropped if forward_only."""
    p = np.vstack([[0.0, 0.0], np.asarray(path, float)])
    t = np.r_[0.0, np.arange(1, len(path) + 1) * 0.25]
    if forward_only:
        d = np.diff(p, axis=0)
        p = np.vstack([[0.0, 0.0], np.cumsum(d * (d[:, :1] > 0), axis=0)])
    return p, t


def desired(p, t):
    wp = np.stack([np.interp(WP_T, t, p[:, k]) for k in range(2)], -1)
    return float(np.linalg.norm(np.diff(wp, axis=0), axis=1).mean() * 2.0)


def arc(p, t, dt):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return float(np.interp(dt, t, s))


def lateral(truth, a, b):
    """Left displacement (m) of the rear axle from tick a to tick b, in the rear-axle frame at a (CARLA: y right)."""
    (x0, y0, h), (x1, y1) = truth[a], truth[b, :2]
    return float(math.sin(h) * (x1 - x0) - math.cos(h) * (y1 - y0))


def collisions(rec, t, v, truth):
    ctr = truth[:, :2] + CENTRE * np.c_[np.cos(truth[:, 2]), np.sin(truth[:, 2])]
    out = []
    for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian"):
        for s in rec["infractions"].get(k, []):
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", s)
            if not m:
                continue
            i = int(np.argmin(np.hypot(*(ctr - [float(m.group(1)), float(m.group(2))]).T)))
            typ = re.search(r"type=(\S+)", s)
            out.append((float(t[i]), float(v[max(i - 1, 0)]), typ.group(1).rstrip(",") if typ else k))
    return sorted(out)


def analyse(att, run, ref):
    ticks, plans = jl(att / "ticks.jsonl"), jl(att / "plans.jsonl")
    rec = json.loads((att / "results.json").read_text())["_checkpoint"]["records"][0]
    summ = att / "agent_summary.json"     # the agent's own config (agent_config.json is the runner's)
    cfg = (json.loads(summ.read_text()).get("config") or {}) if summ.exists() else {}
    fwd = bool(cfg.get("plan_forward_only", False))
    tick_cadence = cfg.get("zoo_cadence", "plan") == "tick"
    rid = rec["route_id"].split("_")[1]
    t = np.array([x["t"] for x in ticks])
    v = np.array([x["v"] for x in ticks])
    thr = np.array([x["throttle"] for x in ticks])
    brk = np.array([x["brake"] for x in ticks])
    steer = np.array([x["steer"] for x in ticks])
    truth = np.array([x["truth"] for x in ticks])
    step = np.r_[0.0, np.hypot(*np.diff(truth[:, :2], axis=0).T)]
    cum = np.cumsum(step)
    cols = collisions(rec, t, v, truth)
    tc = cols[0][0] if cols else math.inf
    # plan in force at every tick
    pt = np.array([p["t"] for p in plans])
    pidx = np.searchsorted(pt, t, side="right") - 1
    pcls, pdes, wins = [], [], []
    for j, p in enumerate(plans):
        pp, tt = prep(p["path"], fwd)
        d = desired(pp, tt)
        pdes.append(d)
        pcls.append("stop" if d < 0.4 else "go")
        t1 = plans[j + 1]["t"] if j + 1 < len(plans) else t[-1]
        a, b = np.searchsorted(t, p["t"]), np.searchsorted(t, t1)
        if b <= a:
            continue
        z = p.get("zoo_pid") or {}
        if tick_cadence:
            act = "throttle" if thr[a:b].mean() > 0.05 else ("brake" if brk[a:b].mean() > 0.5 else "coast")
            if act == "brake":
                act = "brake_stop" if d < 0.4 else "brake_over"
        elif z.get("brake", 0) > 0:
            act = "brake_stop" if d < 0.4 else "brake_over"
        else:
            act = "throttle" if z.get("throttle", 0) > 0 else "coast"
        wins.append({"run": run, "route": rid, "t": round(p["t"], 2), "dt": round(t1 - p["t"], 2),
                     "v0": round(p["speed"], 3), "v1": round(float(v[b - 1]), 3), "desired": round(d, 3),
                     "cls": pcls[-1], "action": act, "plan_m": round(arc(pp, tt, t1 - p["t"]), 3),
                     "drove_m": round(float(cum[b - 1] - cum[a]), 3),
                     "y2": round(float(np.interp(2.0, tt, pp[:, 1])), 3),
                     "y2_drove": round(lateral(truth, a, min(a + 40, len(t) - 1)), 3),
                     "target_steer": int(bool(z) and abs(z.get("angle_final", 0) - z.get("angle_target", 1)) < 1e-9
                                         and abs(z.get("angle", 0) - z.get("angle_target", 0)) > 1e-9),
                     "pre_col": int(p["t"] < tc), "cot": (p.get("cot") or "")[:90]})
    tick_cls = np.array([pcls[i] if i >= 0 else "none" for i in pidx])
    st = v < V_STALL
    pre = t < tc
    dt = 0.05
    moving = ~st
    row = {"run": run, "route": rid, "scenario": rec["scenario_name"].rsplit("_", 1)[0], "status": rec["status"],
           "RC": rec["scores"]["score_route"], "DS": rec["scores"]["score_composed"],
           "length_m": rec["meta"].get("route_length"), "T_s": round(float(t[-1]), 1),
           "T_ref_s": ref.get(rid), "dist_m": round(float(cum[-1]), 1),
           "t_col": round(tc, 1) if cols else None, "v_col": round(cols[0][1], 1) if cols else None,
           "col_type": cols[0][2] if cols else "",
           "moving_s": round(moving.sum() * dt, 1),
           "stall_pre_stop_s": round((st & pre & (tick_cls == "stop")).sum() * dt, 1),
           "stall_pre_go_s": round((st & pre & (tick_cls == "go")).sum() * dt, 1),
           "stall_pre_none_s": round((st & pre & (tick_cls == "none")).sum() * dt, 1),
           "pinned_s": round((st & ~pre).sum() * dt, 1),
           "post_moving_s": round((~st & ~pre).sum() * dt, 1),
           "v_moving": round(float(v[moving].mean()), 2) if moving.any() else 0.0}
    # standstill episodes before the first collision, by how they were entered (route start = never moved yet;
    # brake_over = braked with the controller's desired speed >= 0.4 m/s, i.e. for speed > 1.1 x desired;
    # brake_stop = braked because the desired speed was < 0.4 m/s)
    edges = np.flatnonzero(np.diff(np.r_[0, (st & pre).astype(int), 0]))
    for k in ("start", "brake_over", "brake_stop", "other"):
        row["ep_%s_n" % k] = 0
        row["ep_%s_s" % k] = 0.0
        row["ep_%s_stop_s" % k] = 0.0
    # desired speed the controller used on each tick: the plan's (cadence "plan") or the age-shifted plan's ("tick")
    des_tick = np.array([x.get("zoo_desired", np.nan) for x in ticks], float) if tick_cadence else \
        np.array([pdes[i] if i >= 0 else np.nan for i in pidx])
    for a_, b_ in zip(edges[::2], edges[1::2]):
        if not moving[:a_].any():
            k = "start"
        else:   # the brake that took the speed below V_STALL: the last 0.5 s before the entry tick
            seg = slice(max(a_ - 10, 0), a_ + 1)
            bk = brk[seg] > 0
            k = "other" if not bk.any() else ("brake_over" if np.nanmin(des_tick[seg][bk]) >= 0.4 else "brake_stop")
        row["ep_%s_n" % k] += 1
        row["ep_%s_s" % k] = round(row["ep_%s_s" % k] + (b_ - a_) * dt, 2)
        row["ep_%s_stop_s" % k] = round(row["ep_%s_stop_s" % k] + (tick_cls[a_:b_] == "stop").sum() * dt, 2)
    # pinned: does throttle move the car?
    post = ~pre & (t > tc + 2.0)
    on = np.flatnonzero(post[1:] & (thr[1:] > 0) & (thr[:-1] == 0) & (v[1:] < 0.3)) + 1
    row.update(pinned_throttle_s=round((post & (thr > 0)).sum() * dt, 1), pinned_bursts=len(on),
               pinned_burst_v05=round(float(np.median([v[i:i + 10].max() for i in on])), 3) if len(on) else None,
               pinned_burst_disp1=round(float(np.median([cum[min(i + 20, len(cum) - 1)] - cum[i] for i in on])), 3)
               if len(on) else None,
               pinned_go_plan_share=round(float((tick_cls[post] == "go").mean()), 3) if post.any() else None)
    # moving windows (speed at plan >= 0.5), before the first collision
    mw = [w for w in wins if w["v0"] >= V_STALL and w["pre_col"]]
    pm, dm = sum(w["plan_m"] for w in mw), sum(w["drove_m"] for w in mw)
    over = [w for w in mw if w["action"] == "brake_over"]
    row.update(n_moving_windows=len(mw), efficiency=round(dm / pm, 3) if pm > 0 else None,
               desired_moving_med=round(float(np.median([w["desired"] for w in mw])), 2) if mw else None,
               n_brake_over=len(over), brake_over_share=round(len(over) / len(mw), 3) if mw else None,
               brake_over_v1_over_des=round(float(np.median([w["v1"] / w["desired"] for w in over])), 3) if over else None,
               brake_over_to_stall=sum(w["v1"] < V_STALL for w in over),
               lat_big=sum(abs(w["y2"]) >= 1.0 for w in mw),
               lat_big_target=sum(abs(w["y2"]) >= 1.0 and w["target_steer"] for w in mw))
    row["dist_pre_m"] = round(float(cum[pre][-1]), 1)
    row["pace_pre"] = round(float(cum[pre][-1] / t[pre][-1]), 2)     # m/s over the time before the first collision
    # projected finish at the pre-collision pace (route length / (distance before the collision / time))
    if cols and rec["meta"].get("route_length"):
        pace = cum[pre][-1] / tc if tc > 0 else 0.0
        row["T_proj_s"] = round(rec["meta"]["route_length"] / pace, 1) if pace > 0 else None
    else:
        row["T_proj_s"] = None
    # steer while pinned and throttling (did the controller try to steer around?)
    m = post & (thr > 0)
    row["pinned_abs_steer"] = round(float(np.abs(steer[m]).mean()), 3) if m.any() else None
    return row, wins


def load_ref(path):
    if not path:
        return {}
    recs = json.loads(Path(path).read_text())["_checkpoint"]["records"]
    return {r["route_id"].split("_")[1]: r["meta"]["duration_game"] for r in recs if r["status"] == "Completed"}


def pool(routes, wins):
    out = {}
    for run in sorted({r["run"] for r in routes}):
        R = [r for r in routes if r["run"] == run]
        W = [w for w in wins if w["run"] == run and w["v0"] >= V_STALL and w["pre_col"]]
        tot = sum(r["T_s"] for r in R)
        acts = {}
        for a in ("throttle", "coast", "brake_stop", "brake_over"):
            ws = [w for w in W if w["action"] == a]
            acts[a] = {"n": len(ws), "share": round(len(ws) / max(len(W), 1), 3),
                       "dv_med": round(float(np.median([w["v1"] - w["v0"] for w in ws])), 2) if ws else None}
        over = [w for w in W if w["action"] == "brake_over"]
        Wa = [w for w in wins if w["run"] == run and w["pre_col"]]
        L = [w for w in W if w["v0"] >= 2.0 and abs(w["y2"]) >= 1.0]
        eps = {k: {"n": sum(r["ep_%s_n" % k] for r in R), "s": round(sum(r["ep_%s_s" % k] for r in R), 1),
                   "stop_plan_s": round(sum(r["ep_%s_stop_s" % k] for r in R), 1)}
               for k in ("start", "brake_over", "brake_stop", "other")}
        out[run] = {
            "standstill_episodes_pre_collision": eps,
            "routes": len(R), "sim_s": round(tot, 1),
            "moving_s": round(sum(r["moving_s"] for r in R), 1),
            "stall_pre_stop_s": round(sum(r["stall_pre_stop_s"] for r in R), 1),
            "stall_pre_go_s": round(sum(r["stall_pre_go_s"] for r in R), 1),
            "pinned_s": round(sum(r["pinned_s"] for r in R), 1),
            "post_moving_s": round(sum(r["post_moving_s"] for r in R), 1),
            "efficiency": round(sum(w["drove_m"] for w in W) / max(sum(w["plan_m"] for w in W), 1e-9), 3),
            "actions": acts,
            "brake_over_v1_over_des_med": round(float(np.median([w["v1"] / w["desired"] for w in over])), 3) if over else None,
            "brake_over_to_stall": sum(w["v1"] < V_STALL for w in over),
            "brake_over_des_med": round(float(np.median([w["desired"] for w in over])), 2) if over else None,
            "brake_over_v0_med": round(float(np.median([w["v0"] for w in over])), 2) if over else None,
            # all windows before the first collision: the pace the plans asked for vs the pace driven
            "plan_pace": round(sum(w["plan_m"] for w in Wa) / max(sum(w["dt"] for w in Wa), 1e-9), 2),
            "driven_pace": round(sum(w["drove_m"] for w in Wa) / max(sum(w["dt"] for w in Wa), 1e-9), 2),
            "stop_plan_share": round(sum(w["cls"] == "stop" for w in Wa) / max(len(Wa), 1), 3),
            # moving >= 2 m/s with a plan >= 1 m to the side at 2 s: driven lateral displacement at 2 s / planned
            "lateral_n": len(L), "lateral_driven_over_plan_med": round(float(np.median(
                [w["y2_drove"] / w["y2"] for w in L])), 3) if L else None,
            "lateral_sign_agree": round(float(np.mean([w["y2_drove"] * w["y2"] > 0 for w in L])), 3) if L else None,
            "v_moving_over_desired_med": round(float(np.median([w["v0"] / w["desired"] for w in W if w["desired"] >= 0.4])), 3)
            if W else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="b2d_run output dirs, label=path or path")
    ap.add_argument("--ref", help="reference results (PDM-Lite merged.json): per-route completed duration")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load_ref(a.ref)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    routes, wins = [], []
    for spec in a.runs:
        label, _, path = spec.rpartition("=")
        path = Path(path)
        label = label or path.name
        done = path / "done"
        for rdir in sorted((path / "attempts").iterdir()):
            if done.is_dir() and not (done / (rdir.name + ".json")).exists():
                continue            # unfinished (e.g. cancelled when the run was paused)
            atts = sorted((d for d in rdir.iterdir() if (d / "results.json").exists() and (d / "ticks.jsonl").exists()),
                          key=lambda d: int(d.name))
            if not atts:
                continue
            try:
                row, w = analyse(atts[-1], label, ref)
            except (KeyError, IndexError, ValueError) as exc:
                print("skip", atts[-1], exc)
                continue
            routes.append(row)
            wins += w
    for name, rows in (("routes.csv", routes), ("windows.csv", wins)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    summ = pool(routes, wins)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
