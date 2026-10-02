"""Per-attempt encounter summary with the scripted emergency vehicle on a vlm_arb route (stdlib, run on the box).

usage: opposite_vehicle_encounters.py <runs/vlm_arb dir> <route> <out.csv>
For each attempt: when the vehicle first exists, when it starts moving (the benchmark trigger), the ego state at the
trigger, the closest approach, and whether a contact with it was recorded.
"""
import csv, glob, json, math, sys
root, route, out = sys.argv[1:4]
rows = []
for d in sorted(glob.glob(f"{root}/arms/*/attempts/{route}/*")):
    try:
        sc = [json.loads(l) for l in open(f"{d}/scene.jsonl")]
    except Exception:
        continue
    ego = {}; fire = {}
    for r in sc:
        t = round(r["t"], 2)
        for a in r["actors"]:
            if a["id"] == r["hero_id"]: ego[t] = (a["location"][0], a["location"][1], math.hypot(a["velocity"][0], a["velocity"][1]))
            elif "firetruck" in a["type"] or "ambulance" in a["type"] or "police" in a["type"]: fire[t] = (a["location"][0], a["location"][1], math.hypot(a["velocity"][0], a["velocity"][1]), a["id"])
    if not fire: continue
    first = min(fire); mv = [t for t in sorted(fire) if fire[t][2] > 0.5]
    if not mv: continue
    t_trig = mv[0]; e = ego.get(t_trig) or ego[min(ego, key=lambda x: abs(x - t_trig))]
    common = [t for t in fire if t in ego and t >= t_trig]
    tm = min(common, key=lambda t: math.hypot(ego[t][0] - fire[t][0], ego[t][1] - fire[t][1]))
    dmin = math.hypot(ego[tm][0] - fire[tm][0], ego[tm][1] - fire[tm][1])
    try: contacts = [json.loads(l) for l in open(f"{d}/contacts.jsonl")]
    except Exception: contacts = []
    hit = [c["t"] for c in contacts if c["id"] == fire[first][3]]
    arm = d.split("/")[-4]
    rows.append([arm, round(first, 2), round(t_trig, 2), round(e[0], 1), round(e[1], 1), round(e[2], 2), round(tm, 2), round(dmin, 2),
                 round(ego[tm][2], 2), round(hit[0], 2) if hit else ""])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "t_exists", "t_trigger", "ego_x_at_trigger", "ego_y_at_trigger", "ego_v_at_trigger", "t_closest", "d_closest", "ego_v_closest", "t_first_contact"])
    w.writerows(rows)
print(len(rows))
