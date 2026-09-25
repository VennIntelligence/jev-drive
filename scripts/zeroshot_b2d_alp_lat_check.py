#!/usr/bin/env python3
"""Lateral-fix pilot choice (P1 vs P2) and re-smoke acceptance for the Alpamayo B2D Zoo PID adapter.
Rules frozen before the runs: todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md, section 8.

  pilot:   python zeroshot_b2d_alp_lat_check.py pilot p1=<run dir> p2=<run dir> --out <choice.json>
  resmoke: python zeroshot_b2d_alp_lat_check.py resmoke arm=<run dir> --baseline <dir>[:ids] <dir>[:ids] --out <json>

Per arm, over the finished routes (last attempt; scripts/zeroshot_b2d_alp_speed.analyse):
  lat_ratio   median of driven / planned left displacement 2 s after a plan, over plans made before the route's first
              collision at v >= 2 m/s whose plan is >= 1 m to the side at 2 s (section 7.3: f1 0.04, fixed controller 0.64)
  stuck       routes ending in TickRuntime or "Agent got blocked"
  pinned      stall episodes >= 10 s with a collision from 3 s before their start to their end
  collisions, DS, RC, SR (Completed and no infraction other than min-speed)
Pilot choice: fewer stuck, then fewer pinned, then lat_ratio higher by >= 0.1, then fewer collisions, then higher mean DS,
else P2 (the smaller change); a variant with no finished route or with lat_ratio < 0.3 on >= 5 windows is not eligible; none eligible -> exit 1.
Re-smoke pass: A0 infrastructure (as smoke2), A1 reverse-plan throttle share <= 5 %, A2 no reverse-throttle collision
(both from zeroshot_b2d_alp_smoke2_check), L1 lat_ratio >= 0.4, L2 fewer stuck routes than the baseline on the same routes.
NumPy + standard library, Python 3.8.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_b2d_alp_speed import analyse  # noqa: E402
from zeroshot_b2d_alp_smoke2_check import arm_metrics  # noqa: E402

SR_IGNORE = ("min_speed_infractions",)


def routes_of(run, ids=None):
    """(row, windows) of the last attempt of every finished route of a b2d_run dir, optionally only `ids`."""
    run = Path(run)
    out = []
    for rdir in sorted((run / "attempts").iterdir(), key=lambda d: d.name):
        if ids and rdir.name not in ids:
            continue
        if (run / "done").is_dir() and not (run / "done" / (rdir.name + ".json")).exists():
            continue
        atts = sorted((d for d in rdir.iterdir() if (d / "results.json").exists() and (d / "ticks.jsonl").exists()),
                      key=lambda d: int(d.name))
        if atts:
            row, wins = analyse(atts[-1], run.name, {})
            rec = json.loads((atts[-1] / "results.json").read_text())["_checkpoint"]["records"][0]
            row["success"] = rec["status"] == "Completed" and not any(
                v for k, v in rec["infractions"].items() if k not in SR_IGNORE)
            out.append((row, wins))
    return out


def metrics(rr):
    rows = [r for r, _ in rr]
    L = [w for _, ws in rr for w in ws if w["pre_col"] and w["v0"] >= 2.0 and abs(w["y2"]) >= 1.0]
    lat = float(np.median([w["y2_drove"] / w["y2"] for w in L])) if L else None
    return {"routes": sorted(r["route"] for r in rows), "n_routes": len(rows),
            "lat_ratio": None if lat is None else round(lat, 3), "lat_n": len(L),
            "stuck": sum(("TickRuntime" in r["status"]) or ("blocked" in r["status"]) for r in rows),
            "pinned": sum(r["pinned_eps_10s"] for r in rows), "collisions": sum(r["n_collisions"] for r in rows),
            "DS": round(float(np.mean([r["DS"] for r in rows])), 2) if rows else None,
            "RC": round(float(np.mean([r["RC"] for r in rows])), 2) if rows else None,
            "SR": round(float(np.mean([r["success"] for r in rows])), 3) if rows else None,
            "per_route": {r["route"]: {"status": r["status"], "DS": round(r["DS"], 2), "RC": r["RC"],
                                       "collisions": r["n_collisions"], "pinned": r["pinned_eps_10s"]} for r in rows}}


def eligible(m):
    return m["n_routes"] > 0 and not (m["lat_n"] >= 5 and m["lat_ratio"] is not None and m["lat_ratio"] < 0.3)


def pilot(arms):
    res = {k: metrics(routes_of(v)) for k, v in arms.items()}
    for k in res:
        res[k]["eligible"] = eligible(res[k])
    a, b = res["p1"], res["p2"]
    why = None
    if not (a["eligible"] or b["eligible"]):
        return res, None, "neither variant reaches lat_ratio 0.3"
    if not a["eligible"] or not b["eligible"]:
        return res, ("p1" if a["eligible"] else "p2"), "only one eligible"
    for key, sign in (("stuck", -1), ("pinned", -1)):
        if a[key] != b[key]:
            return res, ("p1" if sign * (a[key] - b[key]) > 0 else "p2"), "fewer " + key
    la, lb = a["lat_ratio"] or 0.0, b["lat_ratio"] or 0.0
    if abs(la - lb) >= 0.1:
        return res, ("p1" if la > lb else "p2"), "lat_ratio"
    if a["collisions"] != b["collisions"]:
        return res, ("p1" if a["collisions"] < b["collisions"] else "p2"), "fewer collisions"
    if a["DS"] != b["DS"]:
        return res, ("p1" if a["DS"] > b["DS"] else "p2"), "higher DS"
    return res, "p2", why or "tie: the smaller change"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pilot", "resmoke"))
    ap.add_argument("arms", nargs="+", help="name=run dir")
    ap.add_argument("--baseline", nargs="*", default=[], help="run dir[:id,id,...] (resmoke)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = dict(s.split("=", 1) for s in a.arms)
    if a.mode == "pilot":
        res, choice, why = pilot(arms)
        res.update(choice=choice, reason=why)
        ok = choice is not None
    else:
        (name, run), = arms.items()
        m = metrics(routes_of(run))
        m.update({k: v for k, v in arm_metrics(run).items() if k.startswith(("A0", "A1", "A2"))})
        base = []
        for spec in a.baseline:
            d, _, ids = spec.partition(":")
            base += routes_of(d, set(ids.split(",")) if ids else None)
        base = [x for x in base if x[0]["route"] in m["routes"]]
        bm = metrics(base)
        m["L1_ok"] = m["lat_ratio"] is not None and m["lat_ratio"] >= 0.4
        m["L2_ok"] = m["stuck"] < bm["stuck"]
        m["pass"] = bool(m["A0_ok"] and m["A1_ok"] and m["A2_ok"] and m["L1_ok"] and m["L2_ok"])
        res = {name: m, "baseline": bm}
        ok = m["pass"]
    txt = json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    Path(a.out).write_text(txt)
    print(txt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
