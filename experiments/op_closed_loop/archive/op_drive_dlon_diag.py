#!/usr/bin/env python3
"""op-drive diagnosis: why does `dlon` (route lateral + openpilot longitudinal) stall where `drive` does not?
Reads runs/op_drive/arms/<tag>-<arm>-s<seed>/ (plans.jsonl of the finished attempt) and lists standstill episodes
(v < 0.3 m/s for >= min_s seconds after warm-up) with what held the car: which constraint bound, whether the lead
head's vehicle exists in ground truth (ctx lead_gap within 5 m of the head's distance), red light ahead, plan v(5 s).
    python3 experiments/op_closed_loop/archive/op_drive_dlon_diag.py [--root DIR] [--tags dv d1] [--min-s 8]
Prints a Markdown table (episodes) and one per-run summary table."""
import argparse
import glob
import json
import os
import statistics as st

CAM_TO_BUMPER = 1.3886 + 2.4508 - 1.433


def load(arm_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(arm_dir, "done", "*.json"))):
        rid = os.path.basename(f)[:-5]
        p = os.path.join(arm_dir, "attempts", rid, str(json.load(open(f))["attempt"]), "plans.jsonl")
        if os.path.exists(p):
            out[rid] = [json.loads(x) for x in open(p)]
    return out


def episodes(rs, min_s, vth=0.3):
    eps, cur = [], []
    for r in rs:
        if not r["warm"] and r["v"] < vth:
            cur.append(r)
        else:
            if cur and cur[-1]["t"] - cur[0]["t"] >= min_s:
                eps.append(cur)
            cur = []
    if cur and cur[-1]["t"] - cur[0]["t"] >= min_s:
        eps.append(cur)
    return eps


def describe(ep):
    n = len(ep)
    src = {}
    for r in ep:
        src[r["src"]] = src.get(r["src"], 0) + 1
    lp = [r["lp"][0] for r in ep]
    lead_present = [r["lp"][0] > 0.5 for r in ep]
    phantom = 0
    for r in ep:
        c = r.get("ctx", {})
        if r["lp"][0] > 0.5:
            head = r["lead"][0][0] - CAM_TO_BUMPER
            if "lead_gap" not in c or abs(c["lead_gap"] - head) > 5:
                phantom += 1
    red = sum(1 for r in ep if r.get("ctx", {}).get("tl") in (1, 2) and r["ctx"].get("tl_dist", 99) < 30)
    ped = sum(1 for r in ep if "ped_gap" in r.get("ctx", {}) and r["ctx"]["ped_gap"] < 15)
    real_lead = sum(1 for r in ep if "lead_gap" in r.get("ctx", {}) and r["ctx"]["lead_gap"] < 15)
    return dict(dur=round(ep[-1]["t"] - ep[0]["t"], 1), n=n, src=max(src, key=src.get), lead_on=round(sum(lead_present) / n, 2),
                phantom=round(phantom / n, 2), real_lead=round(real_lead / n, 2), red=round(red / n, 2), ped=round(ped / n, 2),
                junc=round(sum(r.get("ctx", {}).get("junc", 0) for r in ep) / n, 2),
                vplan5=round(st.median(r["vplan"][4] for r in ep), 2), latch=round(sum(r["latch"] for r in ep) / n, 2),
                lat=ep[0]["lat"], route_i=ep[0]["ri"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ["DATA_DIR"], "runs", "op_drive", "arms"))
    ap.add_argument("--tags", nargs="+", default=["dv", "d1"])
    ap.add_argument("--arms", nargs="+", default=["drive", "dlon"])
    ap.add_argument("--min-s", type=float, default=8.0)
    a = ap.parse_args()
    print("| run | route | start t (s) | dur (s) | binding | lead head on | phantom lead | real lead <15 m | red <30 m | ped | junction | plan v5 med | latch |")
    print("|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for tag in a.tags:
        for arm in a.arms:
            for d in sorted(glob.glob(os.path.join(a.root, f"{tag}-{arm}-s*"))):
                for rid, rs in load(d).items():
                    for ep in episodes(rs, a.min_s):
                        x = describe(ep)
                        print(f"| {os.path.basename(d)} | {rid} | {ep[0]['t'] - rs[0]['t']:.0f} | {x['dur']} | {x['src']} | {x['lead_on']} | {x['phantom']} | "
                              f"{x['real_lead']} | {x['red']} | {x['ped']} | {x['junc']} | {x['vplan5']} | {x['latch']} |")


if __name__ == "__main__":
    main()
