"""Compact per-run extract of vlm_arb closed-loop logs for the offline bypass analyses (stdlib only, runs on the box).

Reads the finished attempt directories listed in routes.csv (stdin, arms drive and pbyp) and writes one gzip JSON per
attempt: route polyline, 0.2 s privileged snapshots (ego, nearby actors, bypass meta), plan-step signals, contacts,
official infractions and slimmed VLM decision logs. Read-only on the run directories.

  python3 bypass_extract.py OUT_DIR < routes.csv
"""
import csv
import glob
import gzip
import json
import math
import os
import sys

R = 70.0   # keep actors within this distance of the ego


def jl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def r2(x, n=2):
    return None if x is None else round(x, n)


def priv(path):
    rows = []
    for d in jl(path):
        e = d["ego"]
        pc = d["pc"]
        ego = [r2(e["xyz"][0]), r2(e["xyz"][1]), r2(e["yaw"], 3), r2(e["v"], 2)]
        acts = []
        for a in d["actors"]:
            if a["distance"] > R:
                continue
            acts.append([a["id"], a["type"], r2(a["xyz"][0]), r2(a["xyz"][1]), r2(a["velocity"][0]), r2(a["velocity"][1]),
                         r2(a["yaw"], 2), r2(a["extent"][0]), r2(a["extent"][1]), r2(a["stationary_s"], 1), r2(a["surface_gap"])])
        st = pc.get("bypass_state")
        if st:
            st = {k: (r2(v) if isinstance(v, float) else v) for k, v in st.items() if k != "ids"}
            st["ids"] = pc["bypass_state"].get("ids")
        rows.append(dict(t=r2(d["t"], 2), ego=ego, ego_s=pc.get("ego_s"), bypass=pc.get("bypass"), borrow=pc.get("borrow"),
                         obst=[dict(s0=r2(o["start_s"]), s1=r2(o["end_s"]), ids=o["ids"]) for o in pc.get("obstacles", [])],
                         st=st, light=pc.get("light"), junc=[dict(s0=r2(j["start_s"]), s1=r2(j["end_s"]), ids=j["ids"])
                                                             for j in pc.get("junctions", [])], acts=acts))
    return rows


def plans(path):
    rows = []
    for d in jl(path):
        c = d.get("ctx", {})
        lead = d.get("lead") or []
        rows.append([r2(d["t"], 2), r2(d["v"], 3), 1 if d.get("warm") else 0, c.get("tl"), c.get("tl_dist"), c.get("tl_id"),
                     c.get("junc"), c.get("lead_gap"), c.get("lead_v"),
                     lead[0] if lead else None, d.get("lp"), 1 if d.get("pc", {}).get("bypass") else 0,
                     d.get("src"), d.get("lat"), d.get("lat_why")])
    return rows


PLAN_COLS = ["t", "v", "warm", "tl", "tl_dist", "tl_id", "junc", "lead_gap", "lead_v", "lead0", "lp", "bypass", "src", "lat", "lat_why"]


def vlm(path):
    out = []
    for d in jl(path):
        k = d.get("k")
        if k == "h":
            continue
        if k == "s":
            out.append(dict(k="s", t=d["t"], v=d["v"], ego_s=d["ego_s"], junc_dist=d["junc_dist"]))
        elif k == "a":
            a, g = d["ans"], d["gt"]
            out.append(dict(k="a", t=r2(d["t_eff"], 2), tq=r2(d["t_q"], 2), ok=a.get("ok"), Ql=a.get("Q_light"), Qb=a.get("Q_block"),
                            Qs=a.get("Q_side"), lat=r2(a.get("latency_ms"), 0),
                            Pb=((a.get("raw") or {}).get("answers") or {}).get("Q_block", {}).get("probabilities"),
                            gt={x: g.get(x) for x in ("tl", "tl_dist", "junc_dist", "lead_gap", "lead_v", "block", "block_dist", "side")}))
        elif "gt" in d:    # old (phase A style) format
            a, g = d.get("vlm_answer"), d["gt"]
            if a is None:
                continue
            out.append(dict(k="a", t=r2(d["t"], 2), tq=r2(a.get("sim_t_queried"), 2), ok=a.get("ok"), Ql=a.get("Q_light"), Qb=a.get("Q_block"),
                            Qs=a.get("Q_side"), lat=r2(a.get("latency_ms"), 0),
                            Pb=((a.get("raw") or {}).get("answers") or {}).get("Q_block", {}).get("probabilities"),
                            gt=dict(tl=g.get("tl_state"), tl_dist=g.get("tl_dist"), junc_dist=g.get("next_junc_dist"),
                                    block=g.get("gt_block"), side=g.get("gt_side"))))
            out.append(dict(k="s", t=r2(d["t"], 2), v=d["speed"], ego_s=d["ego_s"], junc_dist=g.get("next_junc_dist")))
    return out


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for r in csv.DictReader(sys.stdin):
        if r["arm"] not in ("drive", "pbyp", "pbyp2"):
            continue
        a = r["attempt"]
        name = "%s_s%s_%s.json.gz" % (r["arm"], r["seed"], r["route"])
        rt = json.load(open(a + "/route.json"))
        res = json.load(open(a + "/results.json"))["_checkpoint"]["records"][0]
        inf = {k: v for k, v in res["infractions"].items() if v and k != "min_speed_infractions"}
        doc = dict(arm=r["arm"], seed=int(r["seed"]), route=r["route"], unit=r["unit"], scenario=res["scenario_name"],
                   status=res["status"], infractions=inf, route_xy=rt["xy"], priv=priv(a + "/privileged.jsonl"),
                   plan_cols=PLAN_COLS, plans=plans(a + "/plans.jsonl"), contacts=jl(a + "/contacts.jsonl"),
                   vlm=vlm(a + "/vlm_decisions.jsonl"))
        with gzip.open(os.path.join(out_dir, name), "wt") as f:
            json.dump(doc, f, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1])
