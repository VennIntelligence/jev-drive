"""Vehicle collisions of arms drive and vmerge (seeds 0, 1): per collision, route, time, what the arbitration was doing, the other actor.

  python vmerge_collisions.py [arm ...]     -> markdown table on stdout (CPU only, reads the logged attempts)
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_v2_report as v2  # noqa: E402
from vlm_arb_common import ROUTES, jsonl, route_row  # noqa: E402


def nearest(rows, t, key="t"):
    return min(rows, key=lambda r: abs(r[key] - t)) if rows else None


def events(a):
    """Official vehicle collisions: first contact per actor id (the scorer counts one event per actor and a cool-down)."""
    c = jsonl(a / "contacts.jsonl") if (a / "contacts.jsonl").exists() else []
    first = {}
    for x in c:
        if "vehicle" in x["type"]:
            first.setdefault(x["id"], x)
    return sorted(first.values(), key=lambda x: x["t"])


def describe(arm, seed, rid, r):
    a = Path(r["attempt"])
    priv = jsonl(a / "privileged.jsonl")
    plans = jsonl(a / "plans.jsonl")
    dec = [x for x in jsonl(a / "vlm_decisions.jsonl") if x["k"] == "s"] if (a / "vlm_decisions.jsonl").exists() else []
    out = []
    byp_t = [p["t"] for p in priv if p["pc"].get("bypass")]
    for ev in events(a):
        t = ev["t"]
        pb = nearest([p for p in priv if p["t"] <= t], t)           # last snapshot before the first contact
        ego = pb["ego"]
        act = next((x for x in pb["actors"] if x["id"] == ev["id"]), None)
        yaw = ego["yaw"]
        rel = rv = hd = gap = None
        if act:
            dx, dy = act["xyz"][0] - ego["xyz"][0], act["xyz"][1] - ego["xyz"][1]
            rel = (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))
            sp = math.hypot(*act["velocity"])
            hd = abs((math.degrees(act["yaw"] - yaw) + 180) % 360 - 180)
            gap = act.get("surface_gap")
            rv = sp
        s = nearest([d for d in dec if d["t"] <= t], t)
        pl = nearest([p for p in plans if p["t"] <= t], t) if plans and "t" in plans[0] else None
        pre = [d for d in dec if t - 5 <= d["t"] <= t]
        rules = sorted({x for d in pre for x in d.get("rules", [])})
        slow = [d for d in pre if d.get("cap") is not None and d["cap"] < 8.0]
        b = [d.get("byp") for d in pre if d.get("byp")]
        out.append(dict(arm=arm, seed=seed, route=rid, DS=r["DS"], t=round(t, 1), ego_v=round(ego["v"], 1), actor=ev["type"].replace("vehicle.", ""),
                        actor_v=None if rv is None else round(rv, 1), rel_long=None if rel is None else round(rel[0], 1),
                        rel_lat=None if rel is None else round(rel[1], 1), head_diff=None if hd is None else round(hd),
                        stationary_s=None if act is None else round(act.get("stationary_s", 0), 1), rules_5s=",".join(rules) or "-",
                        cap_last=None if s is None else s["cap"], R1_slow=bool(slow), bypass_snap=bool(byp_t and min(byp_t) <= t),
                        byp_first=round(min(byp_t), 1) if byp_t else None, byp_last=round(max(byp_t), 1) if byp_t else None,
                        byp_state=(b[-1] if b else None), pc_bypass_now=bool(pb["pc"].get("bypass")), borrow_now=bool(pb["pc"].get("borrow")),
                        release_5s=any(d.get("release") for d in pre), r5_5s=any(d.get("r5") for d in pre)))
    return out


def main():
    import pandas as pd
    rows = []
    for arm in (sys.argv[1:] or ["drive", "vmerge"]):
        for s in (0, 1):
            for rid in ROUTES:
                r = route_row(v2.run_dir(arm, s, rid), rid)
                if r and r["collisions_vehicle"]:
                    rows += describe(arm, s, rid, r)
    df = pd.DataFrame(rows)
    df.to_csv(sys.stdout.buffer if False else "/dev/stderr", index=False)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
