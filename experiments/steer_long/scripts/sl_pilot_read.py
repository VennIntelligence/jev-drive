#!/usr/bin/env python3
"""Gains pilot read (results/gains_pilot/README.md): per group, handoff spins (|vx wz| > 6 m/s^2 within 1.5 s after t = 1.7 s, controller CSV,
the sl_mpc_report rule) and zeros (scenario score == 0) of the pilot arm against the stored pai281 baseline on the same scenes and seeds.

  sl_pilot_read.py --arm s0=/data/runs/steer_long/gains20/w20s0 s1=... --base s0=/data/runs/onpol_long/pai281/ys0 s1=... --lists <dir> [--out md]
"""
import argparse, csv, glob, json, os
from pathlib import Path

import numpy as np


def rollouts(root):
    """root of pai_eval.sh -> {scene: (score, spin, handoff speed)}."""
    out = {}
    for sim in glob.glob(os.path.join(root, "runs", "*", "sim")):
        f = os.path.join(sim, "aggregate", "results-summary.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rollouts"]:
            c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
            if not os.path.exists(c):
                continue
            rows = list(csv.DictReader(open(c)))
            t = np.array([float(x["timestamp_us"]) for x in rows])
            vx, wz = (np.array([float(x[k]) for x in rows]) for k in ("vx", "wz"))
            rel = (t - t.min()) / 1e6 + 0.1
            w = (rel > 1.7) & (rel <= 3.2)
            a = float(np.abs(vx * wz)[w].max()) if w.any() else 0.0
            out[r["clipgt_id"]] = (float(r.get("score") or 0), a > 6, float(vx[np.abs(rel - 1.7).argmin()]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    G = {g: Path(a.lists, f"pilot_{g}.txt").read_text().split() for g in ("g1", "g2", "g3")}
    arm = {k: rollouts(v) for k, v in (x.split("=", 1) for x in a.arm)}
    base = {k: rollouts(v) for k, v in (x.split("=", 1) for x in a.base)}
    L = ["| group | seed | scenes (arm / base) | spins arm | spins base | zeros arm | zeros base | mean arm | mean base (paired) |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    tot = {}
    for g, sc in G.items():
        for s in sorted(arm):
            A, B = arm[s], base[s]
            both = [x for x in sc if x in A and x in B]
            r = dict(n=len(both), sa=sum(A[x][1] for x in both), sb=sum(B[x][1] for x in both), za=sum(A[x][0] == 0 for x in both),
                     zb=sum(B[x][0] == 0 for x in both), ma=np.mean([A[x][0] for x in both]) if both else np.nan,
                     mb=np.mean([B[x][0] for x in both]) if both else np.nan)
            tot[(g, s)] = r
            L.append(f"| {g} | {s} | {sum(x in A for x in sc)} / {sum(x in B for x in sc)} | {r['sa']} | {r['sb']} | {r['za']} | {r['zb']} | {r['ma']:.4f} | {r['mb']:.4f} |")
    sp_a = sum(v["sa"] for (g, _), v in tot.items() if g == "g1"); sp_b = sum(v["sb"] for (g, _), v in tot.items() if g == "g1")
    z_a = sum(v["za"] for (g, _), v in tot.items() if g != "g1"); z_b = sum(v["zb"] for (g, _), v in tot.items() if g != "g1")
    n = {g: sum(v["n"] for (h, _), v in tot.items() if h == g) for g in G}
    ok1, ok2 = sp_a <= sp_b / 3, z_a <= z_b + 2
    L += ["", f"rollouts read (paired): {n}", f"Line 1 handoff spins g1: arm {sp_a} vs base {sp_b} (line <= {sp_b / 3:.2f}): {'pass' if ok1 else 'FAIL'}",
          f"Line 2 zeros g2 + g3: arm {z_a} vs base {z_b} (line <= {z_b + 2}): {'pass' if ok2 else 'FAIL'}",
          f"Verdict: {'PASS' if ok1 and ok2 else 'FAIL'}"]
    print("\n".join(L))
    if a.out:
        Path(a.out).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
