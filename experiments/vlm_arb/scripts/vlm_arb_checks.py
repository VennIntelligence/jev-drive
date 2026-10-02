"""Per-unit readout and checklists of the vlm_arb lane (called by the chain after every unit; also a CLI).

  python vlm_arb_checks.py unit <unit dir> --arm A --seed S --routes a,b [--kind K]

Every unit: all routes finished, no program crash (official status, prefixed forms included), finite speeds, and
for a unit that asks the VLM: >= 98% of requests answered and every answer used no earlier than t_q + L.
Request accounting (plan deviation D14): a request is `answered` (an "a" line), `pending` (still in flight when the
route ended: the queue is FIFO and an answer is logged at t_q + max(L, latency), so the last ceil(latency / query_s)
requests of a route never get a line) or `dropped` (no line although a later request has one). Dropped requests are
a defect and count against the 98% line; pending ones are bounded by the in-flight window of the route's own
latency. Counts and the unanswered rate go to routes.csv and from there to the Phase A table.
Kinds (staged launch on debug routes, plan "staged launch checklist"):
  red_stop   a stop held by R2 before the line and a roll-off after its release
  r5         the R5 fallback fired and the car rolled again
  bypass     the bypass geometry was enabled with a 2.5-4.5 m shift and the car returned to the route
  r1         while R1 is active the speed settles under v_j + 0.7 m/s
  cruise     the per-route set speed is respected (dslow)
  shadow     a wide and a road JPEG per request, truth labels present
  pred       privileged red stop and green go (the privileged arm's own check)
  pbyp2      pbyp2 batch units: no activation on a blocker outside the route; no activation before the ego first moves on a
             route without a scenario obstacle
  pbyp2dbg   `bypass` checks of pbyp2 on a debug obstacle route plus the two of `pbyp2`
  red_stop3  vred3: the red_stop2 checks, R1 active on the approach and off once the front bumper is past the stop line
  red_stop2  vred2: R2 stops with the car still short of the light's stop line (0 <= distance <= 3.5 m at standstill), the stop
             target of the log is the stop line, and a roll-off after the release
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import OBS_ROUTES, REPO, RUN, attempt_dir, jsonl, route_row, write_json  # noqa: E402


def vlm_rows(a):
    p = Path(a) / "vlm_decisions.jsonl"
    rows = jsonl(p) if p.exists() else []
    return [r for r in rows if r.get("k") == "a"], [r for r in rows if r.get("k") == "s"], \
        next((r for r in rows if r.get("k") == "h"), {})


def request_account(a, ans, st, head):
    """Requests of one route: issued, answered, pending at the route end, dropped mid-route; and the saved frames."""
    q = head.get("params", {}).get("query_s", 0.5)
    L = head.get("L", 0.5)
    fdir = Path(a) / "vlm_frames"
    wide = {f.name.split("_")[0] for f in fdir.glob("*_wide.jpg")}
    road = {f.name.split("_")[0] for f in fdir.glob("*_road.jpg")}
    tq = sorted(r["t_q"] for r in ans)
    n_req = len(wide) if wide else len(st)                     # status lines share the request cadence (query_s)
    dropped = int(sum(max(round((b - x) / q) - 1, 0) for x, b in zip(tq, tq[1:])))    # holes between answered requests
    pending = max(n_req - len(ans) - dropped, 0)
    lat = max((r["ans"].get("latency_ms", 0.0) for r in ans), default=0.0) / 1e3
    window = int(np.ceil(max(L, lat) / q - 1e-9)) + 1          # requests that can be in flight at the route end
    failed = sum(not r["ans"]["ok"] for r in ans)
    return dict(n_req=n_req, pending=pending, dropped=dropped, window=window, failed=failed,
                frames=wide == road and bool(wide) and {"%08.2f" % t for t in tq} <= wide)


def bypass_activations(a):
    """Activations of the privileged bypass in one run: first snapshot of each new bypass state (start_s / 5, ids), with the
    states' blockers projected on the route with the extended end segments (positive `outside_m` = beyond the last route
    point or before the first)."""
    sys.path.insert(0, str(REPO / "lib"))
    from b2d_privileged_geometry import project_ext
    a = Path(a)
    xy = np.asarray(json.loads((a / "route.json").read_text())["xy"], float)
    total = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    out, prev = [], None
    for r in jsonl(a / "privileged.jsonl"):
        st = r["pc"].get("bypass_state")
        key = None if st is None else (round(st["start_s"] / 5), tuple(st["ids"]))
        if st is not None and key != prev:
            pos = [x["xyz"][:2] for x in r["actors"] if x["id"] in st["ids"]]
            s_ext = project_ext(pos, xy)[0] if pos else np.array([])
            out.append(dict(t0=r["t"], ids=list(st["ids"]), s=[float(v) for v in s_ext], route_len=total,
                            outside_m=float(max([-v for v in s_ext] + [v - total for v in s_ext] + [0.0])) if len(s_ext) else None,
                            ego_s=r["pc"]["ego_s"], borrow=bool(st.get("borrow"))))
        prev = key
    return out


def r2_episodes(a, hold="R2"):
    """Hold episodes from plans.jsonl (`hold` R2: the table's red-light row; `pred`: the privileged red stop): dicts with start /
    end time, the light's distance to the front bumper at the last plan step of standstill (positive = short of the stop line),
    the smallest distance during the episode (negative = crept across) and the stop target the table used (`r2_src`: stopline |
    junction; None for pred)."""
    rows = [r for r in jsonl(Path(a) / "plans.jsonl") if not r["warm"]]
    eps, cur = [], None
    for r in rows:
        on = ("R2" in r.get("pc", {}).get("vlm", {}).get("rules", [])) if hold == "R2" else ("pred" in r.get("pc", {}).get("controls", {}))
        d = r.get("ctx", {}).get("tl_dist")
        if on and cur is None:
            cur = dict(t0=r["t"], t1=r["t"], d_stop=None, d_min=d, src=None, v_min=r["v"])
        if on:
            cur["t1"] = r["t"]
            if d is not None and (cur["d_min"] is None or d < cur["d_min"]):
                cur["d_min"] = d
            if r["v"] < 0.2 and d is not None:               # the last standstill of the hold = where the car finally stood
                cur["d_stop"], cur["t_stop"] = d, r["t"]
            cur["src"] = r["pc"].get("vlm", {}).get("r2_src", cur["src"])
        elif cur is not None:
            eps.append(cur)
            cur = None
    if cur is not None:
        eps.append(cur)
    return eps


def route_checks(a, kind=""):
    ans, st, head = vlm_rows(a)
    out, c = {}, {}
    acc = request_account(a, ans, st, head) if ans else None
    if ans:
        n = max(acc["n_req"], 1)
        out.update(vlm_req=acc["n_req"], vlm_pending_end=acc["pending"], vlm_dropped=acc["dropped"],
                   vlm_unanswered=(acc["n_req"] - len(ans)) / n, vlm_lost=(acc["dropped"] + acc["failed"]) / n)
        c["requests_accounted"] = out["vlm_lost"] <= 0.02 and acc["pending"] <= acc["window"]
        ok = [r["ans"]["ok"] for r in ans]
        lat = [r["ans"]["latency_ms"] for r in ans if r["ans"]["ok"]]
        out.update(vlm_n=len(ans), vlm_ok=float(np.mean(ok)), lat_p50=float(np.median(lat)) if lat else np.nan,
                   lat_p95=float(np.percentile(lat, 95)) if lat else np.nan,
                   lat_over_L=float(np.mean([x > 1e3 * head.get("L", 0.5) for x in lat])) if lat else np.nan)
        c["vlm_answered"] = out["vlm_ok"] >= 0.98
        c["delay_respected"] = all(r["t"] >= r["t_q"] + head.get("L", 0.5) - 1e-6 and r["t"] >= r["t_eff"] - 1e-6 for r in ans)
    for rule in ("R1", "R2", "R3", "R4", "R5"):
        out["n_" + rule] = sum(rule in r.get("rules", []) for r in st)
    held = [r for r in st if {"R2", "R3"} & set(r.get("rules", []))]
    if kind == "red_stop":
        stops = [r for r in held if r["v"] < 0.2 and r["line"] > -1.0]
        c["stop_before_line"] = bool(stops)
        c["rolls_after_release"] = bool(stops) and any(r["t"] > stops[0]["t"] and not {"R2", "R3"} & set(r["rules"])
                                                       and r["v"] > 1.0 for r in st)
        first = min((r["t"] for r in held), default=None)       # no rule before K answers have arrived
        c["no_rule_before_answers"] = first is not None and len([r for r in ans if r["t_eff"] <= first + 1e-6]) >= 2
    if kind == "r5":
        fired = [r for r in st if r.get("r5") or "R5" in r.get("rules", [])]
        held25 = [r for r in st if "R2" in r.get("rules", []) and r["v"] < 0.2]
        c["held_before_r5"] = bool(fired) and bool(held25) and fired[0]["t"] - held25[0]["t"] >= 25.0 - 0.6
        c["r5_fired"] = bool(fired)
        c["rolls_after_r5"] = bool(fired) and any(r["t"] > fired[0]["t"] and r["v"] > 1.0 for r in st)
    if kind in ("pbyp2", "pbyp2dbg"):
        acts = bypass_activations(a)
        rid = Path(a).parent.name
        moved = next((r["t"] for r in jsonl(Path(a) / "plans.jsonl") if r["v"] > 0.3), np.inf)
        out.update(n_activations=len(acts), outside_m=max([x["outside_m"] or 0.0 for x in acts], default=0.0))
        c["no_activation_outside_route"] = all((x["outside_m"] or 0.0) <= 0.0 for x in acts)
        if rid not in OBS_ROUTES and kind == "pbyp2":
            c["no_activation_before_move"] = all(x["t0"] >= moved for x in acts)
        scene = jsonl(Path(a) / "privileged.jsonl")
        out.update(n_gap_hold=sum(bool(r["pc"].get("gap_hold")) for r in scene),
                   n_red_memory=sum(r["pc"].get("suppressed") == "red_memory" for r in scene))
    if kind in ("red_stop2", "red_stop3"):
        eps = [e for e in r2_episodes(a) if e["d_stop"] is not None]
        c["stopped_short_of_line"] = bool(eps) and all(0.0 <= e["d_stop"] <= 3.5 for e in eps)
        c["target_is_stopline"] = bool(eps) and all(e["src"] == "stopline" for e in eps)
        c["rolls_after_release"] = bool(eps) and any(r["t"] > eps[0]["t1"] and "R2" not in r.get("rules", []) and r["v"] > 1.0 for r in st)
        out.update(n_r2_stops=len(eps), d_stop=eps[0]["d_stop"] if eps else np.nan)
        if kind == "red_stop3":
            c["r1_on_approach"] = any("R1" in r.get("rules", []) for r in st)
            c["r1_off_past_stop_line"] = not any("R1" in r.get("rules", []) and r.get("d_stop") is not None and r["d_stop"] <= 0.0 for r in st)
            ev = [r for r in jsonl(Path(a) / "vlm_decisions.jsonl") if r.get("k") == "y"]
            out.update(n_yellow_events=sum(r.get("ev") == "yellow" for r in ev), n_go=sum(r.get("decision", "").endswith("go") or r.get("decision") == "go" for r in ev))
    if kind in ("bypass", "pbyp", "pbyp2dbg"):
        scene = jsonl(Path(a) / "privileged.jsonl")
        on = [r for r in scene if r["pc"].get("bypass")]
        c["obstacle_detected"] = any(r["pc"].get("obstacles") for r in scene)
        c["path_enabled"] = bool(on)
        c["valid_shift"] = bool(on) and all(2.5 <= abs(r["pc"]["bypass_state"]["offset"]) <= 4.5 for r in on)
        c["passed_and_returned"] = bool(on) and any(r["pc"]["ego_s"] > on[0]["pc"]["bypass_state"]["end_s"] + 23 for r in scene)
        if kind == "bypass":
            c["r4_logged"] = out["n_R4"] > 0
    if kind == "r1":
        on, late, start = [r for r in st if "R1" in r.get("rules", [])], [], None
        for r in st:                                            # speeds 3 s or more into each R1 episode
            start = (r["t"] if start is None else start) if "R1" in r.get("rules", []) else None
            if start is not None and r["t"] - start >= 3.0:
                late.append(r["v"])
        c["r1_active"] = bool(on)
        c["speed_capped"] = bool(late) and max(late) <= 4.5 + 0.7
    if kind == "cruise":
        cfg = json.loads((Path(a) / "agent_config.json").read_text()) if (Path(a) / "agent_config.json").exists() else {}
        v = [r["v"] for r in jsonl(Path(a) / "plans.jsonl") if not r["warm"]]
        c["cruise_respected"] = bool(v) and max(v) <= 5.0 + 0.6
        out["v_max"] = max(v) if v else np.nan
        out["cfg"] = bool(cfg)
    if kind == "shadow":
        c["frames_saved"] = bool(acc) and acc["frames"]         # a wide and a road frame per request; completeness: requests_accounted
        c["labels_present"] = bool(ans) and all("lights" in r["gt"] and "stop_dist" in r["gt"] and "block" in r["gt"] for r in ans)
    if kind == "pred":
        live = [r for r in jsonl(Path(a) / "plans.jsonl") if not r["warm"]]
        stops = [r for r in live if "pred" in r.get("pc", {}).get("controls", {}) and r["v"] < .2]
        c["red_stop"] = bool(stops)
        c["green_go"] = bool(stops) and any(r["pc"].get("light", {}).get("tl") == 0 and r["v"] > 1 and r["t"] > stops[0]["t"] for r in live)
    return out, c


def read_unit(udir, arm, seed, routes, kind="", name=None):
    """Write readouts/<unit>/{routes.csv, checks.json}; return True when every check passes."""
    import pandas as pd
    udir = Path(udir)
    name = name or udir.name
    rows, checks = [], {}
    for rid in routes:
        row = route_row(udir, rid)
        if row is None:
            checks[rid] = {"finished": False}
            continue
        extra, c = route_checks(row["attempt"], kind)
        c.update(finished=True, no_crash=not row["crash"], finite=row["finite"])
        rows.append(dict(unit=name, arm=arm, seed=seed, **row, **extra))
        checks[rid] = c
    out = RUN / "readouts" / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "routes.csv", index=False)
    passed = all(all(bool(v) for v in c.values()) for c in checks.values())
    write_json(out / "checks.json", dict(unit=name, arm=arm, seed=seed, kind=kind, passed=passed, checks=checks,
                                         DS=float(np.mean([r["DS"] for r in rows])) if rows else None,
                                         RC=float(np.mean([r["RC"] for r in rows])) if rows else None))
    return passed


def unit_ok(udir, arm, seed, routes, kind="", name=None):
    """Lane `ok` hook: never raises (a raising hook would look like a lane failure)."""
    try:
        return read_unit(udir, arm, seed, routes, kind, name)
    except Exception:  # noqa: BLE001
        out = RUN / "readouts" / (name or Path(udir).name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ERROR").write_text(traceback.format_exc())
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["unit"])
    ap.add_argument("dir")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--kind", default="")
    a = ap.parse_args()
    ok = read_unit(a.dir, a.arm, a.seed, a.routes.split(","), a.kind)
    print((RUN / "readouts" / Path(a.dir).name / "checks.json").read_text())
    sys.exit(0 if ok else 1)
