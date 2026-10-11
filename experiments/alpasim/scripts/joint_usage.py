#!/usr/bin/env python3
"""Lane JOINT: card-hours actually spent, from the pool's event log ($DATA_DIR/runs/pool/events.jsonl).

  python3 experiments/alpasim/scripts/joint_usage.py [--since "2026-10-11 12:00:00"]       (box; stdlib only)
A lane job = owner alpasim-joint1 / eyes_sense / alpasim-nuplan1491, or a bench job (owner bench) whose name carries one of the lane's tags.
Per job: wall time (end event) x booked VRAM / card VRAM (83.6 GB) = its share of a card; summed by kind (training, closed loop, bench GPU stages,
CPU-only). Also job-hours and the plain sum over GPU jobs of wall time (an upper bound: a job alone on a card).
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

OWNERS = ("alpasim-joint1", "eyes_sense", "alpasim-nuplan1491")
TAGS = re.compile(r"(JAPS|ES[abcd]-|JID-)")
CARD_GB = 83.6


def kind(name, owner):
    if name.startswith("bn-"):
        return "bench (" + name.split("-")[1] + ")"
    if name.startswith(("j1-", "es-", "jid-")):
        return "training"
    if name.startswith("alpasim-"):
        return "closed loop"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-11 12:00:00")
    a = ap.parse_args()
    J = defaultdict(dict)
    for ln in open(Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/pool/events.jsonl"):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("t", "") < a.since or "id" not in e:
            continue
        j = J[e["id"]]
        if e["kind"] == "submit":
            j.update(name=e.get("name", ""), owner=e.get("owner", ""))
        elif e["kind"] == "launch":
            j.update(booked=float(e.get("booked_gb") or 0), gpu=e.get("gpu"))
        elif e["kind"] == "end":
            j.update(wall=float(e.get("wall_s") or 0), state=e.get("state"))
    tot = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for j in J.values():
        if "wall" not in j or not (j.get("owner") in OWNERS or (j.get("owner") == "bench" and TAGS.search(j.get("name", "")))):
            continue
        k = kind(j.get("name", ""), j.get("owner", ""))
        b = j.get("booked", 0.0)
        t = tot[k]
        t[0] += 1
        t[1] += j["wall"] / 3600
        t[2] += j["wall"] / 3600 * min(1.0, b / CARD_GB) if b > 1 else 0.0
        t[3] += j["wall"] / 3600 if b > 1 else 0.0
    print("| kind | jobs | job-hours | card-hours (booked VRAM share) | GPU job-hours (alone-on-card bound) |\n|:--|--:|--:|--:|--:|")
    s = [0, 0.0, 0.0, 0.0]
    for k, t in sorted(tot.items()):
        print(f"| {k} | {t[0]} | {t[1]:.1f} | {t[2]:.1f} | {t[3]:.1f} |")
        s = [x + y for x, y in zip(s, t)]
    print(f"| **total** | {s[0]} | {s[1]:.1f} | {s[2]:.1f} | {s[3]:.1f} |")


if __name__ == "__main__":
    main()
