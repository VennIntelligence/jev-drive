"""Export stored logs of the Alpamayo 1.5 zero-shot Bench2Drive runs for the visual review (section `nvcl`).

Reads, under $DATA_DIR/runs/zeroshot-exam/b2d/<run>/attempts/<route>/<n>/: results.json (official record),
plans.jsonl (2 Hz plans: path, nav text, CoC text, frame dump path), ticks.jsonl (20 Hz controls and poses),
route.json (route polyline, newer runs only). Writes one export.json to
$DATA_DIR/runs/openloop_visual_review_20261002/nvcl/. No inference, no simulator, no GPU; json only.
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path

B = Path(os.environ["DATA_DIR"]) / "runs/zeroshot-exam/b2d"
O = Path(os.environ["DATA_DIR"]) / "runs/openloop_visual_review_20261002/nvcl"
RUNS = {  # run dir -> (controller, note)
    "smoke-alpamayo": "fixed controller (pre-registered), 5 routes",
    "smoke-alpamayo-zoopid": "Zoo PID, 5 routes",
    "full220-alpamayo": "fixed controller, full run stopped early (voided)",
    "full220-alpamayo-zoopid": "Zoo PID, full run stopped at 119/220 (voided)",
    "smoke2-alpamayo-f1": "Zoo PID + forward-only plan (F1), re-smoke",
    "smoke2-alpamayo-f1f2b": "Zoo PID + F1 + per-tick PID (F2b), re-smoke",
    "full220-alpamayo-zoopid-f1": "Zoo PID + F1, full run paused at 17/220",
    "lat-pilot-alpamayo-p1": "lateral fix P1 pilot",
    "lat-pilot-alpamayo-p2": "lateral fix P2 pilot",
    "lat-resmoke-alpamayo-p1": "lateral fix P1 re-smoke",
}
CASES = [("smoke-alpamayo", r) for r in ("2373", "2390", "3564", "24211", "1711")] + \
        [("smoke-alpamayo-zoopid", r) for r in ("2373", "2390", "1711", "3564", "24211")] + \
        [("full220-alpamayo-zoopid-f1", "1833"), ("full220-alpamayo", "1833")]
POS = re.compile(r"\(x=([-\d.]+), y=([-\d.]+)")
NAV = re.compile(r"Turn (left|right) in (\d+)m")
OTHER = {"left": "right", "right": "left"}
SCORED = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light", "stop_infraction",
          "outside_route_lanes", "yield_emergency_vehicle_infractions", "scenario_timeouts", "route_dev",
          "vehicle_blocked", "route_timeout")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def latest(run, route):
    """Latest finished attempt of a route with an official record (cancelled / crashed attempts are skipped)."""
    for d in sorted((B / run / "attempts" / route).iterdir(), key=lambda d: -int(d.name)):
        try:
            done = json.load(open(d / "attempt.json"))["status"] == "finished"
            rec = json.load(open(d / "results.json"))["_checkpoint"]["records"]
        except (OSError, ValueError, KeyError):
            continue
        if done and rec:
            return d, rec[0]
    return None, None


def xy_of(row):  # true pose when logged, else the agent's filtered pose (plans only)
    p = row.get("truth") or row.get("pose")
    return p[:2] if p else None


def when(pts, x, y):
    """First time the logged pose is within 0.5 m of its closest approach to an infraction location.
    Collisions report the ego position (match < 1 m); a red light reports the light's own position (match 4-19 m),
    so its time is the closest pass of the pole and only approximate."""
    d = [((p[0] - x) ** 2 + (p[1] - y) ** 2) ** .5 for _, p in pts]
    dmin = min(d)
    return next(t for (t, _), v in zip(pts, d) if v <= dmin + .5), dmin


def load(run, route):
    d, rec = latest(run, route)
    if d is None:
        return None
    plans, ticks = jl(d / "plans.jsonl"), jl(d / "ticks.jsonl")
    pts = [(r["t"], xy_of(r)) for r in (ticks if ticks and "truth" in ticks[-1] else plans) if xy_of(r)]
    events = []
    for k in SCORED:
        for s in rec["infractions"].get(k, []):
            m = POS.search(s)
            t, dist = when(pts, float(m[1]), float(m[2])) if m and pts else (None, None)
            events.append({"kind": k, "text": s, "t": t, "match_m": None if dist is None else round(dist, 2)})
    return {"dir": str(d), "rec": rec, "plans": plans, "ticks": ticks, "events": events}


def responses(plans, ticks):
    """Share of the 0.5 s after each plan in which the throttle is on."""
    out, j = [], 0
    for i, p in enumerate(plans):
        t1 = plans[i + 1]["t"] if i + 1 < len(plans) else p["t"] + .5
        while j < len(ticks) and ticks[j]["t"] <= p["t"]:
            j += 1
        k, n, thr = j, 0, 0
        while k < len(ticks) and ticks[k]["t"] <= t1:
            n += 1; thr += ticks[k]["throttle"] > .05; k += 1
        out.append(thr / n if n else None)
    return out


def stats(run):
    rows, agg = [], Counter()
    routes = sorted(p.name for p in (B / run / "attempts").iterdir())
    for route in routes:
        c = load(run, route)
        if c is None:
            continue
        rec, plans = c["rec"], [p for p in c["plans"] if p.get("accepted", True) and p.get("path")]
        inf = {k: len(rec["infractions"].get(k, [])) for k in SCORED}
        sc = rec["scores"]
        ok = rec["status"] == "Completed" and not any(inf.values())
        col_t = [e["t"] for e in c["events"] if e["kind"].startswith("collisions") and e["t"] is not None]
        t_col = min(col_t) if col_t else 1e9
        thr = responses(plans, c["ticks"])
        r = {"route": route, "scenario": rec["scenario_name"], "status": rec["status"], "DS": sc["score_composed"],
             "RC": sc["score_route"], "penalty": sc["score_penalty"], "success": ok,
             "inf": {k: v for k, v in inf.items() if v}, "n_plans": len(plans), "dir": c["dir"],
             "nav_plans": 0, "nav_contra": 0, "nav_contra_t": [], "rev": 0, "rev_thr": 0, "stay": 0, "stay_thr": 0,
             "go": 0, "go_thr": 0, "red_events": [], "light_plans": 0, "red_plans": 0, "red_nobrake": 0, "flips": 0,
             "double_flips": 0, "double_flip_t": [], "turn_text_contra": 0}
        seq = []  # (t, 'R' | 'G') of plans whose CoC names exactly one light state
        for p, th in zip(plans, thr):
            m = NAV.search(p.get("nav_text") or "")
            if m:
                r["nav_plans"] += 1
                y = p["path"][-1][1]  # rear-axle frame, y left positive; last point is 5.0 s
                if int(m[2]) <= 15 and ((m[1] == "left" and y <= -2) or (m[1] == "right" and y >= 2)):
                    r["nav_contra"] += 1; r["nav_contra_t"].append(round(p["t"], 2))
            cot = (p.get("cot") or "").lower()
            if m and re.search("turn " + OTHER[m[1]] + "|" + OTHER[m[1]] + " turn", cot) and m[1] not in cot:
                r["turn_text_contra"] += 1  # nav names one side, the CoC text announces a turn to the other
            if "light" in cot or "signal" in cot:
                r["light_plans"] += 1
                red, green = "red" in cot, "green" in cot
                if red != green:
                    seq.append((p["t"], "R" if red else "G"))
                if red and not green:
                    r["red_plans"] += 1
                    x5 = p["path"][-1][0]  # distance ahead at 5.0 s
                    r["red_nobrake"] += ("stop" in cot or "slow" in cot) and x5 >= 8 and x5 >= .8 * 5 * p["speed"]
            if p["speed"] < .3 and p["t"] < t_col - 2 and th is not None:  # standstill, before the first collision
                x3 = [q[0] for q in p["path"][:12]]  # 0.25 .. 3.0 s
                k = "rev" if min(x3) < -.3 else "stay" if x3[-1] < 1.2 else "go"
                r[k] += 1; r[k + "_thr"] += th > .5
        for i in range(1, len(seq)):
            near = seq[i][0] - seq[i - 1][0] <= .6
            r["flips"] += near and seq[i][1] != seq[i - 1][1]
            if i >= 2 and near and seq[i - 1][0] - seq[i - 2][0] <= .6 and seq[i][1] == seq[i - 2][1] != seq[i - 1][1]:
                r["double_flips"] += 1; r["double_flip_t"].append(round(seq[i - 1][0], 2))  # R-G-R or G-R-G within 1 s
        for e in c["events"]:
            if e["kind"] == "red_light" and e["t"] is not None:
                cots = [(round(p["t"], 2), p.get("cot") or "", p.get("nav_text"), p["path"][11][0], round(p["speed"], 2))
                        for p in plans if e["t"] - 5 <= p["t"] <= e["t"] + .5]  # path[11] is the 3.0 s point
                red = [x for x in cots if "red" in x[1].lower()]
                read = ("says red, plan keeps going" if any(x[3] >= 1.2 for x in red) else "says red, plan stops") if red else \
                    "says green" if any("green" in x[1].lower() for x in cots) else "light not mentioned"
                r["red_events"].append({"t": e["t"], "match_m": e["match_m"], "read": read, "cots": cots})
        rows.append(r)
        agg["n"] += 1; agg["DS"] += r["DS"]; agg["RC"] += r["RC"]; agg["SR"] += ok
        agg["status:" + r["status"]] += 1
        for k, v in inf.items():
            agg["inf:" + k] += v; agg["routes:" + k] += v > 0
        for k in ("n_plans", "nav_plans", "nav_contra", "rev", "rev_thr", "stay", "stay_thr", "go", "go_thr", "light_plans",
                  "red_plans", "red_nobrake", "flips", "double_flips", "turn_text_contra"):
            agg[k] += r[k]; agg[k + "_routes"] += r[k] > 0
        for e in r["red_events"]:
            agg["red:" + e["read"]] += 1
    n = max(agg["n"], 1)
    return {"note": RUNS[run], "agg": dict(agg), "DS_mean": agg["DS"] / n, "RC_mean": agg["RC"] / n, "routes": rows}


def case(run, route):
    c = load(run, route)
    keep = ("t", "frame", "speed", "path", "nav_text", "cot", "dump", "pose", "truth", "accepted")
    plans = [{**{k: p.get(k) for k in keep}, "zoo": {k: p["zoo_pid"][k] for k in ("throttle", "brake", "steer", "desired_speed")}
              if p.get("zoo_pid") else None} for p in c["plans"]]
    ticks = [[round(r["t"], 2), round(r["v"], 3), round(r["throttle"], 3), round(r["brake"], 3), round(r["steer"], 4)]
             + ([round(v, 3) for v in r["truth"]] if "truth" in r else []) for r in c["ticks"]]
    rj = Path(c["dir"]) / "route.json"
    return {"run": run, "route": route, "dir": c["dir"], "rec": c["rec"], "events": c["events"], "plans": plans, "ticks": ticks,
            "route_xy": json.load(open(rj)).get("xy") if rj.exists() else None,
            "n_frames": len(list((Path(c["dir"]) / "frames").glob("*.jpg"))) if (Path(c["dir"]) / "frames").exists() else 0}


if __name__ == "__main__":
    O.mkdir(parents=True, exist_ok=True)
    out = {"runs": {r: stats(r) for r in RUNS if (B / r / "attempts").exists()},
           "cases": [case(*c) for c in CASES]}
    (O / "export.json").write_text(json.dumps(out))
    for r, s in out["runs"].items():
        print(r, s["agg"]["n"], round(s["DS_mean"], 2), round(s["RC_mean"], 2), flush=True)
    print("DONE", O / "export.json", file=sys.stderr)
