"""Per-unit readout and checklists of the vlm_arb lane (called by the chain after every unit; also a CLI).

  python vlm_arb_checks.py unit <unit dir> --arm A --seed S --routes a,b [--kind K]

Every unit: all routes finished, no program crash (official status, prefixed forms included), finite speeds, and
for a unit that asks the VLM: >= 98% of requests answered and every answer used no earlier than t_q + L.
Kinds (staged launch on debug routes, plan "staged launch checklist"):
  red_stop   a stop held by R2 before the line and a roll-off after its release
  r5         the R5 fallback fired and the car rolled again
  bypass     the bypass geometry was enabled with a 2.5-4.5 m shift and the car returned to the route
  r1         while R1 is active the speed settles under v_j + 0.7 m/s
  cruise     the per-route set speed is respected (dslow)
  shadow     a wide and a road JPEG per request, truth labels present
  pred       privileged red stop and green go (the privileged arm's own check)
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import RUN, attempt_dir, jsonl, route_row, write_json  # noqa: E402


def vlm_rows(a):
    p = Path(a) / "vlm_decisions.jsonl"
    rows = jsonl(p) if p.exists() else []
    return [r for r in rows if r.get("k") == "a"], [r for r in rows if r.get("k") == "s"], \
        next((r for r in rows if r.get("k") == "h"), {})


def route_checks(a, kind=""):
    ans, st, head = vlm_rows(a)
    out, c = {}, {}
    if ans:
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
    if kind in ("bypass", "pbyp"):
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
        n_w = len(list((Path(a) / "vlm_frames").glob("*_wide.jpg")))
        n_r = len(list((Path(a) / "vlm_frames").glob("*_road.jpg")))
        c["frames_saved"] = n_w == n_r and len(ans) > 0 and 0 <= n_w - len(ans) <= 2   # requests still pending at the route end have frames, no answer
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
