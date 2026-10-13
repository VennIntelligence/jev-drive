#!/usr/bin/env python3
"""SPIN-FIX offline screen of a gate change on a gate read's own closed loop (numpy only, either box): every logged gate call is rebuilt with
track_ref.build(mode="gate") under the reference configuration (--ref, default v7: pp=0) and the candidate (--cfg), from the logged plan, v0
and the controller row's measured acceleration (a0, which the driver log does not keep). The rebuilt reference decision must equal the logged
one (checked); a rollout whose decisions all agree is the same closed loop under the candidate. Also lists the controller rows where the anchor
guard would fire (|(qx, qy)| > TILT). Layouts: PAI (<root>/runs/*/sim + ../driver/drive.jsonl), nuPlan (<run>/aggregate + driver-logs/drive.jsonl).
  sf_screen.py --runs <dir> ... [--list scenes.txt] [--ref pp=0] [--cfg pp=1,...] [--show N]"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../alpasim/lib"))
import track_ref as TR


def sims(d):
    if os.path.exists(os.path.join(d, "aggregate", "results-summary.json")):
        yield d, os.path.join(d, "driver-logs", "drive.jsonl")
    for s in sorted(glob.glob(os.path.join(d, "runs", "*", "sim"))):
        yield s, os.path.join(s, "..", "driver", "drive.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--list")
    ap.add_argument("--ref", default="pp=0")
    ap.add_argument("--cfg", default="pp=1")
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    kv = lambda s: {k: float(v) for k, v in (x.split("=") for x in filter(None, s.split(",")))}  # noqa: E731
    kr, kc = kv(a.ref), kv(a.cfg)
    keep = set(open(a.list).read().split()) if a.list else None
    n_roll = n_call = n_mis = 0
    ch, tilt = [], []
    for d in a.runs:
        for sim, dj in sims(d):
            R = {}
            for line in open(dj):
                r = json.loads(line)
                if r.get("kind") == "drive" and "track" in r and r.get("infer", True):
                    R.setdefault(r["session"], []).append(r)
            for ro in json.load(open(os.path.join(sim, "aggregate", "results-summary.json")))["rollouts"]:
                if keep is not None and ro["clipgt_id"] not in keep:
                    continue
                c = os.path.join(sim, "controller", f"alpasim_controller_{ro['rollout_id']}.csv")
                if not os.path.exists(c) or ro["rollout_id"] not in R:
                    continue
                rows = list(csv.DictReader(open(c)))
                ts = np.array([float(x["timestamp_us"]) for x in rows]); acc = np.array([float(x["acceleration"]) for x in rows])
                q = np.array([[float(x["qx"]), float(x["qy"])] for x in rows])
                n_roll += 1
                tl = np.where(np.hypot(*q.T) > TR.TILT)[0]
                if len(tl):
                    tilt.append(f"{ro['clipgt_id'][7:15]} t {(ts[tl[0]] - ts[0]) / 1e6 + 0.1:.1f} rows {len(tl)}")
                first = None
                for r in R[ro["rollout_id"]]:
                    i = int(np.abs(ts - r["t0"]).argmin())
                    v0 = r["track"]["v0"]
                    _, i0 = TR.build(np.asarray(r["poses"]), v0, 0.0, mode="gate", a0=float(acc[i]), **kr)
                    _, i1 = TR.build(np.asarray(r["poses"]), v0, 0.0, mode="gate", a0=float(acc[i]), **kc)
                    n_call += 1
                    n_mis += int(i0["open"] != r["track"].get("open", 0))
                    if i1["open"] != i0["open"] and first is None:
                        first = (round((r["now"] - ts[0]) / 1e6 + 0.1, 1), v0, i0["gate"], i1["gate"], i1.get("pp"), i1["open"])
                if first:
                    ch.append((ro["clipgt_id"][7:15], first))
    print(f"{n_roll} rollouts, {n_call} gate calls; reference rebuild disagrees with the log on {n_mis} calls")
    print(f"candidate {a.cfg} vs {a.ref}: {len(ch)} rollouts change (first differing call: t, v0, ratio ref, ratio cand, pp, open cand)")
    for s, f in ch[:a.show]:
        print(f"   {s} {f}")
    print(f"anchor guard (|(qx, qy)| > {TR.TILT}): {len(tilt)} rollouts" + "".join(f"\n   {x}" for x in tilt[:a.show]))


if __name__ == "__main__":
    main()
