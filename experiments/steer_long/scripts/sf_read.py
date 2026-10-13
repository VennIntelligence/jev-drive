#!/usr/bin/env python3
"""SPIN-FIX read (Tokyo host, numpy only): per arm root and scene list, spins (sf_detect.py), zeros, mean score against the baseline root on the
same scenes (paired), at-fault / front (ego into something ahead) / rear (hit from behind) collisions, per-scene table optional.
A rollout without metrics (an infrastructure failure, e.g. a renderer CUDA OOM) is not a drive and is skipped; an arm may list several roots (root1,root2: a
later root's rollout of a scene replaces an earlier one, e.g. a rerun of a failed scene).
  sf_read.py --list <scenes.txt> --base <root> --arm name=<root>[,<root>..] ... [--scenes]"""
import argparse, csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_detect import detect


def read(root, keep):
    out = {}
    for sim in glob.glob(os.path.join(root, "runs", "*", "sim")):
        f = os.path.join(sim, "aggregate", "results-summary.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rollouts"]:
            if r["clipgt_id"] not in keep or not r.get("metrics"):
                continue
            c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
            if not os.path.exists(c):
                continue
            rows = list(csv.DictReader(open(c)))
            t = (np.array([float(x["timestamp_us"]) for x in rows]) - float(rows[0]["timestamp_us"])) / 1e6 + 0.1
            g = lambda k: np.array([float(x[k]) for x in rows])  # noqa: E731
            d = detect(t, g("vx"), g("vy"), g("wz"), g("front_steering_angle"))
            m = r.get("metrics") or {}
            d.update(atf=float(m.get("collision_at_fault") or 0) > 0, front=float(m.get("collision_front") or 0) > 0,
                     rear=float(m.get("collision_rear") or 0) > 0)
            out[r["clipgt_id"]] = (float(r.get("score") or 0), d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", nargs="+", required=True)
    ap.add_argument("--scenes", action="store_true")
    a = ap.parse_args()
    S = open(a.list).read().split()
    B = read(a.base, set(S))
    print(f"| arm | scenes | spins | handoff / fast / slow | zeros | at-fault / front / hit from behind | mean | base spins | base zeros | base at-fault / front / behind | base mean (paired) |\n|:--|--:|--:|:--|--:|:--|--:|--:|--:|:--|--:|")
    for name, root in (x.split("=", 1) for x in a.arm):
        A = {k: v for r in root.split(",") for k, v in read(r, set(S)).items()}
        both = [s for s in S if s in A and s in B]
        k = lambda D, n: sum(D[s][1][n] is not None for s in both)  # noqa: E731
        c = lambda D: " / ".join(str(sum(D[s][1][n] for s in both)) for n in ("atf", "front", "rear"))  # noqa: E731
        print(f"| {name} | {len(A)} / {len(S)} | {sum(A[s][1]['spin'] for s in both)} | {k(A, 'handoff')} / {k(A, 'fast')} / {k(A, 'slow')} | "
              f"{sum(A[s][0] == 0 for s in both)} | {c(A)} | {np.mean([A[s][0] for s in both]):.4f} | {sum(B[s][1]['spin'] for s in both)} | "
              f"{sum(B[s][0] == 0 for s in both)} | {c(B)} | {np.mean([B[s][0] for s in both]):.4f} |")
        if a.scenes:
            for s in both:
                print(f"    {s[7:15]} arm {A[s][0]:.3f} {A[s][1]['kind'] or '-'} {A[s][1]['t_first'] or ''} | base {B[s][0]:.3f} {B[s][1]['kind'] or '-'}")


if __name__ == "__main__":
    main()
