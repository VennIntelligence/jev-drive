"""Compact per-attempt extract for the offline release-policy replay and the route-junction analysis (stdlib only, runs on the box).

Read-only on the run directories. Input: CSV on stdin with columns arm,seed,route,attempt (the *_runs.csv files of results/) or, with
--glob, every attempt directory of the units matching a pattern. One gzip JSON per attempt:
  s     : arbitration steps   [t, v, ego_s, junc_dist, rules, release, r5, fresh, jid, line, line_r2]
  a     : VLM light answers   [t_q, t_eff, ok, Q_light, p_green, p_red, p_other, p_nolight, latency_ms, gt_tl, gt_tl_dist, gt_tl_id, gt_junc_dist]
  y     : yellow-rule events (vred3), as logged
  lights: truth state of every light within 40 m at each answer's query time [t_q, [[light_id, state], ...]] (gt.lights of the answer log)
  ctx   : truth light per plan step [t, tl, tl_dist, tl_id] (tl 0 green, 1 yellow, 2 red; tl_dist = front bumper to the stop line)
  ego   : [t, x, y, v] every 0.2 s from privileged.jsonl
  route : {xy, cmd} dense official plan as the agent logged it
  infractions, status, DS

  python3 release_extract.py OUT_DIR < runs.csv
"""
import csv
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

ARMS_GLOB = "/root/autodl-tmp/ujs/runs/vlm_arb/arms"


def jl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def rd(x, n=3):
    return None if x is None else round(x, n)


def one(job):
    out_dir, arm, seed, route, unit, a = job
    name = "%s_s%s_%s_%s.json.gz" % (arm, seed, route, unit)
    try:
        res = json.load(open(a + "/results.json"))["_checkpoint"]["records"][0]
    except Exception:
        return None
    inf = {k: v for k, v in res["infractions"].items() if v and k != "min_speed_infractions"}
    s, ans, y, lights = [], [], [], []
    for d in jl(a + "/vlm_decisions.jsonl"):
        k = d.get("k")
        if k == "s":
            s.append([rd(d["t"]), rd(d["v"]), rd(d["ego_s"]), rd(d["junc_dist"]), "+".join(d.get("rules", [])), int(bool(d.get("release"))),
                      int(bool(d.get("r5"))), int(bool(d.get("fresh"))), d.get("jid"), d.get("line"), d.get("line_r2")])
        elif k == "a":
            an, g = d["ans"], d["gt"]
            p = an.get("Q_light_p") or {}
            ans.append([rd(d["t_q"]), rd(d["t_eff"]), int(bool(an.get("ok"))), an.get("Q_light"), p.get("green_for_ego"), p.get("red_or_yellow_for_ego"),
                        p.get("light_for_other_lane"), p.get("no_light"), rd(an.get("latency_ms"), 0), g.get("tl"), g.get("tl_dist"), g.get("tl_id"),
                        g.get("junc_dist")])
            lights.append([rd(d["t_q"]), [[x[0], x[1]] for x in (g.get("lights") or []) if x[2] <= 40.0]])
        elif k == "y":
            y.append(d)
    ctx = []
    for d in jl(a + "/plans.jsonl"):
        c = d.get("ctx") or {}
        ctx.append([rd(d.get("t")), c.get("tl"), c.get("tl_dist"), c.get("tl_id")])
    ego = []
    for d in jl(a + "/privileged.jsonl"):
        e = d["ego"]
        ego.append([rd(d["t"], 2), rd(e["xyz"][0], 2), rd(e["xyz"][1], 2), rd(e["v"], 2)])
    try:
        rt = json.load(open(a + "/route.json"))
    except Exception:
        rt = None
    doc = dict(arm=arm, seed=seed, route=route, unit=unit, attempt=a, status=res["status"], ds=res["scores"]["score_composed"],
               infractions=inf, s=s, a=ans, y=y, lights=lights, ctx=ctx, ego=ego, route_xy=rt)
    with gzip.open(os.path.join(out_dir, name), "wt") as f:
        json.dump(doc, f, separators=(",", ":"))
    return name


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jobs, seen = [], set()
    for r in csv.DictReader(sys.stdin):
        a = r["attempt"]
        if a in seen:
            continue
        seen.add(a)
        unit = a.split("/arms/")[1].split("/")[0]
        jobs.append((out_dir, r["arm"], r["seed"], r["route"], unit, a))
    with Pool(8) as p:
        done = [x for x in p.imap_unordered(one, jobs) if x]
    print(len(jobs), "attempts,", len(done), "written")


if __name__ == "__main__":
    main(sys.argv[1])
