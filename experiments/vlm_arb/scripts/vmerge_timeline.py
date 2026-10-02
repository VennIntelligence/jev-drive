"""Timeline around one logged collision: python vmerge_timeline.py ARM SEED ROUTE [seconds_before=5]  (ego speed, bypass / borrow / gap flags, rules, nearest actors)."""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vmerge2_report as v2  # noqa: E402
from vlm_arb_common import jsonl, route_row  # noqa: E402
from vmerge_collisions import events  # noqa: E402

arm, seed, rid = sys.argv[1], int(sys.argv[2]), sys.argv[3]
before = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
a = Path(route_row(v2.run_dir(arm, seed, rid), rid)["attempt"])
ev = events(a)
priv, dec = jsonl(a / "privileged.jsonl"), [x for x in jsonl(a / "vlm_decisions.jsonl") if x.get("k") == "s"]
for e in ev:
    print("contact t=%.2f id=%s %s impulse %.0f" % (e["t"], e["id"], e["type"], e["impulse"]))
t0 = ev[0]["t"]
last = -9
for p in priv:
    if not t0 - before <= p["t"] <= t0 + 0.3 or p["t"] - last < 0.45:
        continue
    last = p["t"]
    e, pc = p["ego"], p["pc"]
    d = min(dec, key=lambda x: abs(x["t"] - p["t"])) if dec else {}
    near = []
    for x in p["actors"]:
        dx, dy = x["xyz"][0] - e["xyz"][0], x["xyz"][1] - e["xyz"][1]
        lo, la = dx * math.cos(e["yaw"]) + dy * math.sin(e["yaw"]), -dx * math.sin(e["yaw"]) + dy * math.cos(e["yaw"])
        near.append((math.hypot(lo, la), x["id"], lo, la, math.hypot(*x["velocity"]), x.get("surface_gap")))
    near.sort()
    print("t=%.1f v=%.1f byp=%s borrow=%s gap=%s rules=%s cap=%s byp_state=%s | %s" % (
        p["t"], e["v"], pc.get("bypass"), pc.get("borrow"), pc.get("gap_open"), d.get("rules"), d.get("cap"), d.get("byp"),
        " ".join("#%s(%.0f,%.0f v%.1f gap%s)" % (n[1], n[2], n[3], n[4], None if n[5] is None else round(n[5], 1)) for n in near[:3])))
