"""Could the ego have stopped short of the scripted vehicle? (stdlib; run where the attempt is)

usage: opposite_vehicle_avoidance.py <attempt_dir> <other_actor_id> <out.csv>
Takes the ego centre path actually driven up to the first contact, and for every 0.25 s asks: with the speed the ego
had then, would a constant deceleration a in {3, 5, 7} m/s^2 started at that instant stop it before the contact
point, keeping the ego's front half-length as clearance. The last t where this holds is the last moment braking
could still avoid contact (speed and path as driven, so a sharper turn or a different speed would move it).
"""
import csv, json, math, sys
d, oid, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ego = {}; oth = {}; frame_t = {}
ext = None
for l in open(f"{d}/scene.jsonl"):
    r = json.loads(l); t = round(r["t"], 2); frame_t[r["frame"]] = t
    for a in r["actors"]:
        if a["id"] == r["hero_id"]: ego[t] = (a["location"][0], a["location"][1], math.hypot(a["velocity"][0], a["velocity"][1])); ext = a["extent"][0]
        elif a["id"] == oid: oth[t] = (a["location"][0], a["location"][1], math.hypot(a["velocity"][0], a["velocity"][1]))
# contacts.jsonl "t" runs on a different clock (0.95 s ahead of scene.jsonl "t"); join on the frame number instead
cf = [json.loads(l)["frame"] for l in open(f"{d}/contacts.jsonl") if json.loads(l)["id"] == oid][0]
ts = sorted(ego); tc = frame_t[min(frame_t, key=lambda x: abs(x - cf))]; S = {}; s = 0.0; prev = None
for t in ts:
    if prev: s += math.hypot(ego[t][0] - ego[prev][0], ego[t][1] - ego[prev][1])
    S[t] = s; prev = t
trig = next(t for t in sorted(oth) if oth[t][2] > 0.5)
rows = []; last = {3: None, 5: None, 7: None}
for t in ts:
    if t < trig - 0.5 or t > tc or abs(t * 4 - round(t * 4)) > 1e-6: continue
    rem = S[tc] - S[t] - ext; v = ego[t][2]
    o = oth.get(t); od = math.hypot(ego[t][0] - o[0], ego[t][1] - o[1]) if o else ""
    row = [t, round(v, 2), round(rem, 2), round(od, 2) if o else ""]
    for a in (3, 5, 7):
        ok = v * v / (2 * a) <= rem
        row += [round(v * v / (2 * a), 2), ok]
        if ok: last[a] = t
    rows.append(row)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "ego_v", "path_to_contact_minus_front", "dist_to_other", "stop3_m", "ok3", "stop5_m", "ok5", "stop7_m", "ok7"])
    w.writerows(rows)
print("trigger", trig, "contact", tc, "last feasible braking start:", last)
for r in rows: print(r)
