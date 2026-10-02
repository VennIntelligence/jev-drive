"""Timeline of one vlm_arb attempt around a scripted opposite-vehicle conflict (stdlib only, run where the attempt is).

usage: opposite_vehicle_timeline.py <attempt_dir> <other_actor_id> [t0 t1]
Prints ego / other-vehicle state every 0.25 s plus event times (first seen, first moving, contact) and the brake
and acceleration limits the ego showed in this attempt.
"""
import json, math, sys, collections
d = sys.argv[1]; oid = int(sys.argv[2])
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0; t1 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
ego = {}; oth = {}; tls = {}
for l in open(f"{d}/scene.jsonl"):
    r = json.loads(l); t = round(r["t"], 2)
    for a in r["actors"]:
        if a["id"] == r["hero_id"]:
            ego[t] = (a["location"][0], a["location"][1], math.hypot(a["velocity"][0], a["velocity"][1]), a["rotation"][2])
        elif a["id"] == oid:
            oth[t] = (a["location"][0], a["location"][1], math.hypot(a["velocity"][0], a["velocity"][1]), a["rotation"][2])
        elif a["type"] == "traffic.traffic_light":
            pass
ticks = {round(json.loads(l)["t"], 2): json.loads(l) for l in open(f"{d}/ticks.jsonl")}
ts = sorted(ego)
first_seen = min(oth) if oth else None
moving = [t for t in sorted(oth) if oth[t][2] > 0.5]
print("other actor first in scene.jsonl:", first_seen, "first speed>0.5:", moving[:1])
contacts = [json.loads(l) for l in open(f"{d}/contacts.jsonl")]
print("contacts:", [(c["t"], c["type"], round(c["impulse"])) for c in contacts[:3]])
print("t, ego_x, ego_y, ego_v, throttle, brake, other_x, other_y, other_v, dist, light(ctx)")
prev = None
for t in ts:
    if t < t0 or t > t1 or abs(t * 4 - round(t * 4)) > 1e-6: continue
    e = ego[t]; o = oth.get(t); k = ticks.get(t, {})
    dist = math.hypot(e[0] - o[0], e[1] - o[1]) if o else None
    print(f"{t:6.2f} {e[0]:8.2f} {e[1]:8.2f} {e[2]:5.2f} {k.get('throttle',0):4.2f} {k.get('brake',0):4.2f} "
          + (f"{o[0]:8.2f} {o[1]:8.2f} {o[2]:5.2f} {dist:6.2f}" if o else "  -") + f" {k.get('ctx',{}).get('tl')} {k.get('ctx',{}).get('tl_dist')} {k.get('reason')}")
# limits: max decel and accel over 0.5 s windows while moving
acc = []
for i, t in enumerate(ts[:-10]):
    t2 = ts[i + 10]
    acc.append(((ego[t2][2] - ego[t][2]) / (t2 - t), t))
print("ego accel over 0.5 s windows: max", max(acc), "min", min(acc))
