#!/usr/bin/env python3
"""Route-night Tokyo: per-rollout facts of PAI 281 reads (pai_eval.sh run roots), plain python3 + numpy. Per label: mean score, zeros, handoff
spins (|vx wz| > 6 m/s^2 within 1.5 s after t = 1.7 s, steer-long's rule), and a turn proxy (no map read): turn scenes = scenes where a
reference label scored > 0 without a handoff spin and its own path turned 60 to 200 deg net (yaw integrated over the whole controller CSV, i.e. also
after the car passed the log's end, where the score no longer looks); a label completes a turn scene when its own path also turns >= 60 deg with the
same sign and the scene scores > 0. Left / right by sign. The `log-window` columns (added 2026-10-16) do the same with the yaw integrated only until
the controller arc reaches the log length (results-summary `gt_dist_traveled_m`), so a turn the car makes after driving past the log's end (the
scored part is over) is not a turn scene; the whole-trace columns are kept for comparison with older reports.

  rf_pai281.py --rows rsg=/data/runs/route_night/r281/rsg ys0=/data/runs/onpol_long/pai281/ys0 ... --ref ybase,ys0,ys1 [--out md]
  --lists hard=<file> control=<file> --vs rsg   per scene list: mean, zeros by kind, spins, plan 2 s deficit (served plan arc through 0.5 .. 2.0 s
                                                minus fed speed x 2 s, inferences with speed >= 1 m/s, whole rollout, per-rollout mean), and the
                                                scene-paired difference of every label against --vs (bootstrap over scenes, 10 000 draws, seed 0)
"""
import argparse, csv, glob, json, os

import numpy as np


def read(root):
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
            t = np.array([float(x["timestamp_us"]) for x in rows]) / 1e6
            vx, wz = (np.array([float(x[k]) for x in rows]) for k in ("vx", "wz"))
            rel = t - t.min() + 0.1
            w = (rel > 1.7) & (rel <= 3.2)
            yaw = np.degrees(np.r_[0, np.cumsum(wz[1:] * np.diff(t))])
            m = r.get("metrics", {})
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(np.array([[float(x[k]) for k in ("x", "y")] for x in rows]), axis=0).T))]
            iw = int(np.searchsorted(arc, float(m.get("gt_dist_traveled_m") or np.inf)))
            kind = "" if r["score"] else ("collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else "corridor" if m.get("left_corridor_laterally") else "other")
            out[r["clipgt_id"]] = dict(score=float(r["score"] or 0), rear=bool(m.get("collision_rear")), spin=bool(w.any() and np.abs(vx * wz)[w].max() > 6), turn=float(yaw[-1]), turn_w=float(yaw[min(iw, len(yaw) - 1)]), kind=kind)
    return out


def deficit(root):
    """{scene: mean plan 2 s deficit (m) of the rollout} from drive.jsonl (PAI: one inference per call)."""
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")):
        acc = {}
        for line in open(f):
            if '"infer": true' not in line:
                continue
            r = json.loads(line)
            v = 10.0 * float(np.hypot(r["ego"][4], r["ego"][5]))
            if v < 1:
                continue
            P = np.array(r["poses"])[:4, :2]
            arc = float(np.hypot(*P[0]) + np.hypot(*np.diff(P, axis=0).T).sum())
            acc.setdefault(r["scene"], []).append(arc - 2.0 * v)
        out.update({k: float(np.mean(v)) for k, v in acc.items()})
    return out


def lists(D, roots, a):
    rng = np.random.default_rng(0)
    Df = {k: deficit(v) for k, v in roots.items()}
    L = []
    for spec in a.lists:
        n, f = spec.split("=", 1)
        sc = [s for s in open(f).read().split() if all(s in d for d in D.values())]
        L += ["", f"## {n}: {len(sc)} scenes", "", "| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs " + a.vs + " (95 % CI) |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
        for k, d in D.items():
            z = [d[s]["kind"] for s in sc if d[s]["kind"]]
            df = np.nanmean([Df[k].get(s, np.nan) for s in sc])
            x = np.array([d[s]["score"] - D[a.vs][s]["score"] for s in sc])
            bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
            L.append(f"| {k} | {np.mean([d[s]['score'] for s in sc]):.4f} | {len(z)} | {z.count('collision')} | {z.count('offroad')} | {z.count('corridor')} | {sum(d[s]['rear'] for s in sc)} | "
                     f"{sum(d[s]['spin'] for s in sc)} | {df:+.2f} | {x.mean():+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}] |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--ref", default="ybase,ys0,ys1")
    ap.add_argument("--out", default="")
    ap.add_argument("--lists", nargs="*", default=[])
    ap.add_argument("--vs", default="")
    a = ap.parse_args()
    roots = dict(x.split("=", 1) for x in a.rows)
    D = {k: read(v) for k, v in roots.items()}
    common = sorted(set.intersection(*[set(d) for d in D.values()]))
    refs = [r for r in a.ref.split(",") if r in D]
    def turn_sets(key):
        ok = lambda r, s: D[r][s]["score"] > 0 and not D[r][s]["spin"] and 60 <= abs(D[r][s][key]) <= 200  # noqa: E731  a scored, non-spinning turn
        return {s: np.sign(next(D[r][s][key] for r in refs if ok(r, s))) for s in common if any(ok(r, s) for r in refs)}
    turns, turns_w = turn_sets("turn"), turn_sets("turn_w")
    lr = lambda T: (sum(v > 0 for v in T.values()), sum(v < 0 for v in T.values()))  # noqa: E731
    L = [f"Scenes scored by every label: {len(common)}; turn scenes (a scoring reference label turned 60-200 deg): whole trace {len(turns)} "
         f"(left {lr(turns)[0]}, right {lr(turns)[1]}), log window {len(turns_w)} (left {lr(turns_w)[0]}, right {lr(turns_w)[1]})", "",
         "| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right | log-window left | log-window right |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, d in D.items():
        z = [d[s]["kind"] for s in common if d[s]["kind"]]
        done = lambda T, key, sg: sum(v == sg and d[s]["score"] > 0 and d[s][key] * sg >= 60 for s, v in T.items())  # noqa: E731
        (nl, nr), (wl, wr) = lr(turns), lr(turns_w)
        L.append(f"| {k} | {np.mean([d[s]['score'] for s in common]):.4f} | {len(z)} | {z.count('collision')} | {z.count('offroad')} | {z.count('corridor')} | "
                 f"{sum(d[s]['spin'] for s in common)} | {done(turns, 'turn', 1)}/{nl} | {done(turns, 'turn', -1)}/{nr} | "
                 f"{done(turns_w, 'turn_w', 1)}/{wl} | {done(turns_w, 'turn_w', -1)}/{wr} |")
    if a.lists:
        L += lists(D, roots, a)
    print("\n".join(L))
    if a.out:
        open(a.out, "w").write("\n".join(L) + "\n")
        json.dump({k: {s: d[s] for s in common} for k, d in D.items()} | {"_turns": {s: float(v) for s, v in turns.items()}, "_turns_w": {s: float(v) for s, v in turns_w.items()}}, open(a.out[:-3] + ".json", "w"))


if __name__ == "__main__":
    main()
