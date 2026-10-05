#!/usr/bin/env python
"""op_resume report: per-case rows, pre-registered readouts and lines (plans/2026-10-05-op-resume-prereg.md sections 3-4).

  python experiments/op_resume/scripts/or_report.py pilot|full
Reads $DATA_DIR/runs/op_resume/{hugsim,b2d}/<stage>/ (or_submit.py layout); writes experiments/op_resume/results/<stage>/
{hugsim.csv, b2d.csv, summary.json} and prints a markdown block (the result note quotes it).
Ground truth (HUGSIM boxes, B2D light state) is read here only to describe firings, never by the rule.
"""
import csv
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
R = DATA / "runs/op_resume"
OUT = REPO / "experiments/op_resume/results"
STUCK = sorted(r["scenario"] for r in csv.DictReader(open(REPO / "experiments/hugsim/results/op_control_stack/opctrl_runs.csv"))
               if r["arm"] == "opctrl" and r["end"] == "max_steps")
PILOT_STUCK = ["scene-0418-hard-00", "scene-0411-medium-00", "scene-034-easy-00", "scene-095-medium-01",
               "scene-113792265837-easy-00", "scene-3000_3200-medium-00"]
INFR = ["red_light", "stop_infraction", "collisions_vehicle", "collisions_layout", "collisions_pedestrian", "vehicle_blocked",
        "route_timeout", "min_speed_infractions", "outside_route_lanes"]


def hug_rows(stage, arm):
    p = R / "hugsim" / stage / arm / "results.csv"
    rows = {}
    for r in csv.DictReader(open(p)) if p.exists() else []:
        if r["tag"] == "or-" + arm and r["end"] != "crash":
            rows[r["scenario"]] = r
    return rows


def hug_steps(run_dir):
    f = Path(run_dir) / "zs_steps.jsonl"
    return [json.loads(x) for x in open(f) if '"setup"' not in x[:12]] if f.exists() else []


def gt_lane_min(run_dir, k):
    try:
        infos = pickle.load(open(Path(run_dir) / "infos.pkl", "rb"))
        b = [bb for bb in infos[min(k, len(infos) - 1)]["obj_boxes"] if 0 < bb[0] < 30 and abs(bb[1]) < 2.0]
        return round(float(min(bb[0] for bb in b)), 1) if b else None
    except Exception:
        return None


def hugsim(stage):
    spec, rule = hug_rows(stage, "spec"), hug_rows(stage, "rule")
    out = []
    for sc in sorted(set(spec) | set(rule)):
        s, r = spec.get(sc), rule.get(sc)
        row = dict(scenario=sc, set="stuck" if sc in STUCK else "other",
                   hd_spec=float(s["hdscore"]) if s else np.nan, hd_rule=float(r["hdscore"]) if r else np.nan,
                   rc_spec=float(s["rc"]) if s else np.nan, rc_rule=float(r["rc"]) if r else np.nan,
                   end_spec=s["end"] if s else "", end_rule=r["end"] if r else "")
        L = hug_steps(r["run_dir"]) if r else []
        fires = [k for k, x in enumerate(L) if x.get("rr") == "fire"]
        hb = [x.get("rr") for x in L if x.get("rr") in ("speed", "plan", "lead_abort", "timeout")]
        row.update(fires=len(fires), t_fire=L[fires[0]]["t"] if fires else None,
                   launched=bool(fires) and any(x["v"] >= 2.0 for x in L[fires[0]:]),
                   handbacks=";".join(hb), lead_at_fire=(round(L[fires[0]].get("lead_prob") or 0, 2), round(L[fires[0]].get("lead_x") or 0, 1)) if fires else None,
                   gt_lane_at_fire=gt_lane_min(r["run_dir"], fires[0]) if fires else None,
                   coll_after_fire=bool(fires) and row["end_rule"] in ("fg_collision", "bg_collision")
                   and any(0 <= L[-1]["t"] - L[k]["t"] <= 10.0 for k in fires),
                   dist_after_fire=round(float(np.linalg.norm(np.subtract(L[-1]["pos"], L[fires[0]]["pos"]))), 1) if fires else None)
        out.append(row)
    return out


def b2d_record(udir, rid):
    d = Path(udir)
    att = sorted(glob.glob(str(d / "attempts" / rid / "*" / "results.json")), key=lambda p: int(Path(p).parent.name))
    if not att:
        return None
    recs = json.loads(Path(att[-1]).read_text())["_checkpoint"]["records"]
    if not recs:
        return None
    rec = recs[0]
    row = dict(DS=float(rec["scores"]["score_composed"]), RC=float(rec["scores"]["score_route"]), status=rec["status"])
    for k in INFR:
        row[k] = len(rec["infractions"].get(k, []))
    fires, at_light = 0, 0
    p = Path(att[-1]).parent / "plans.jsonl"
    if p.exists():
        for line in open(p):
            if '"rr": "fire"' not in line:
                continue
            x = json.loads(line)
            fires += 1
            c = x.get("ctx") or {}
            at_light += int(c.get("tl") in (1, 2) and c.get("tl_dist", 99) < 40)
    row.update(fires=fires, fires_at_light=at_light)
    return row


def b2d(stage):
    out = []
    for udir in sorted(glob.glob(str(R / "b2d" / stage / "spec-s*-k*"))):
        name = Path(udir).name
        seed = int(name.split("-s")[1].split("-")[0])
        for f in sorted(glob.glob(udir + "/done/*.json")):
            rid = Path(f).stem
            s = b2d_record(udir, rid)
            r = b2d_record(udir.replace("/spec-s", "/rule-s"), rid)
            if s is None or r is None:
                continue
            out.append(dict(route=rid, seed=seed, **{k + "_spec": v for k, v in s.items()}, **{k + "_rule": v for k, v in r.items()}))
    return out


def write(rows, path):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def ci(r, f=".3f"):
    return "%+{0} [%+{0}, %+{0}] (n {1})".format(f, r["n"]) % (r["mean"], r["lo"], r["hi"])


def main():
    stage = sys.argv[1]
    od = OUT / stage
    od.mkdir(parents=True, exist_ok=True)
    H, B = hugsim(stage), b2d(stage)
    write(H, od / "hugsim.csv")
    write(B, od / "b2d.csv")
    S = {}
    print("## HUGSIM (%s)\n" % stage)
    print("| scenario | set | HD spec | HD rule | RC spec | RC rule | end spec | end rule | fires | t first | launched | hand-backs | lead at fire (p, x) | GT lane box at fire (m) | coll. <= 10 s after fire | moved after fire (m) |")
    print("|" + "---|" * 16)
    for h in H:
        print("| %s | %s | %.3f | %.3f | %.3f | %.3f | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            h["scenario"], h["set"], h["hd_spec"], h["hd_rule"], h["rc_spec"], h["rc_rule"], h["end_spec"], h["end_rule"], h["fires"],
            h["t_fire"], h["launched"], h["handbacks"], h["lead_at_fire"], h["gt_lane_at_fire"], h["coll_after_fire"], h["dist_after_fire"]))
    for name, sel in (("stuck", lambda h: h["set"] == "stuck"), ("other", lambda h: h["set"] == "other"),
                      ("pilot_stuck", lambda h: h["scenario"] in PILOT_STUCK)):
        g = [h for h in H if sel(h)]
        if not g:
            continue
        d = stats.paired([h["hd_rule"] for h in g], [h["hd_spec"] for h in g])
        rc = stats.paired([h["rc_rule"] for h in g], [h["rc_spec"] for h in g])
        coll = lambda arm: sum(h["end_" + arm] in ("fg_collision", "bg_collision") for h in g)  # noqa: E731
        ms = lambda arm: sum(h["end_" + arm] == "max_steps" for h in g)  # noqa: E731
        S[name] = dict(n=len(g), launched=sum(h["launched"] for h in g), fired=sum(h["fires"] > 0 for h in g),
                       hd=d, rc=rc, coll_spec=coll("spec"), coll_rule=coll("rule"), coll_after_fire=sum(h["coll_after_fire"] for h in g),
                       max_steps_spec=ms("spec"), max_steps_rule=ms("rule"))
        print("\n**%s** (n %d): fired %d, launched %d; HD rule - spec %s (means %.3f vs %.3f); RC %s; collision endings %d -> %d "
              "(within 10 s of a firing: %d); max_steps %d -> %d" % (
                  name, len(g), S[name]["fired"], S[name]["launched"], ci(d), d["mean_a"], d["mean_b"], ci(rc), S[name]["coll_spec"],
                  S[name]["coll_rule"], S[name]["coll_after_fire"], S[name]["max_steps_spec"], S[name]["max_steps_rule"]))
    if B:
        print("\n## B2D (%s)\n" % stage)
        print("| route | seed | DS spec | DS rule | RC spec | RC rule | red spec / rule | stop spec / rule | veh. coll. spec / rule | fires | fires at light |")
        print("|" + "---|" * 11)
        for b in B:
            print("| %s | %d | %.1f | %.1f | %.1f | %.1f | %d / %d | %d / %d | %d / %d | %d | %d |" % (
                b["route"], b["seed"], b["DS_spec"], b["DS_rule"], b["RC_spec"], b["RC_rule"], b["red_light_spec"], b["red_light_rule"],
                b["stop_infraction_spec"], b["stop_infraction_rule"], b["collisions_vehicle_spec"], b["collisions_vehicle_rule"],
                b["fires_rule"], b["fires_at_light_rule"]))
        d = stats.paired([b["DS_rule"] for b in B], [b["DS_spec"] for b in B])
        fired = [b for b in B if b["fires_rule"] > 0]
        tot = {k: (sum(b[k + "_spec"] for b in B), sum(b[k + "_rule"] for b in B)) for k in ("red_light", "stop_infraction", "collisions_vehicle")}
        totf = {k: (sum(b[k + "_spec"] for b in fired), sum(b[k + "_rule"] for b in fired)) for k in tot}
        S["b2d"] = dict(n=len(B), ds=d, runs_fired=len(fired), fires_at_light=sum(b["fires_at_light_rule"] for b in B), totals=tot,
                        totals_fired=totf)
        print("\nDS rule - spec %s; runs where the rule fired %d; firings at a light %d; totals spec / rule %s; on fired runs %s" % (
            ci(d, ".2f"), len(fired), S["b2d"]["fires_at_light"], tot, totf))
    (od / "summary.json").write_text(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
