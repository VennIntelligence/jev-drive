#!/usr/bin/env python3
"""SPIN-FIX per-case look at the spins of one read (Tokyo host, numpy only): for every spin rollout of <root>, the second before the onset
(sf_detect.py) from the controller rows and the driver records: ego speed, steering / side-slip peaks, the plan's 2 s deficit and speed, and
the lateral demand v x dyaw/dt of the reference over 0.5-2 s (the MPC's window) as served and as rebuilt by track_ref.build for each --mode
(the logged plan, ego speed and acceleration of each record), with the rebuilt reference's speed at 2 s.
  sf_mid.py <root> <scenes.txt> [--mode curv time ...] [--cfg k=v,...] [--csv out.csv]"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "../../alpasim/lib"))
from sf_detect import detect
import track_ref as TR


def demand(t, x, y, h, lo=0.5, hi=2.0):
    """-> max |v dyaw/dt| over [lo, hi] s, speed at hi."""
    v = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
    w = np.diff(np.unwrap(h)) / np.diff(t)
    tm = (t[1:] + t[:-1]) / 2
    m = (tm >= lo) & (tm <= hi)
    return float(np.abs(v * w)[m].max()), float(np.interp(hi, tm, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("scenes")
    ap.add_argument("--mode", nargs="*", default=["curv"])
    ap.add_argument("--cfg", default="")
    ap.add_argument("--all", action="store_true", help="every rollout, not only spins (window = the last 6 s)")
    ap.add_argument("--csv")
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (kv.split("=") for kv in filter(None, a.cfg.split(",")))}
    keep = set(open(a.scenes).read().split())
    keep |= {f"clipgt-{s}" if not s.startswith("clipgt-") else s for s in list(keep)}
    out = []
    for sim in sorted(glob.glob(os.path.join(a.root, "runs", "*", "sim"))):
        D = {}
        for line in open(os.path.join(sim, "..", "driver", "drive.jsonl")):
            r = json.loads(line)
            if r["kind"] == "drive" and r.get("infer"):
                D.setdefault(r["session"], []).append(r)
        for ro in json.load(open(os.path.join(sim, "aggregate", "results-summary.json")))["rollouts"]:
            sc = ro["clipgt_id"]
            if sc not in keep:
                continue
            rows = list(csv.DictReader(open(os.path.join(sim, "controller", f"alpasim_controller_{ro['rollout_id']}.csv"))))
            g = lambda k: np.array([float(x[k]) for x in rows])  # noqa: E731
            ts = g("timestamp_us"); t = (ts - ts[0]) / 1e6 + 0.1
            vx, vy, wz, de = g("vx"), g("vy"), g("wz"), g("front_steering_angle")
            d = detect(t, vx, vy, wz, de)
            if not d["spin"] and not a.all:
                continue
            t1 = d["t_first"] if d["spin"] else t[-1]
            w = (t >= t1 - 1.0) & (t <= t1 + 0.3)
            rec = [r for r in D.get(ro["rollout_id"], []) if t1 - 1.0 <= (r["now"] - ts[0]) / 1e6 + 0.1 <= t1 + 0.05]
            info = dict(scene=sc[7:15], score=round(float(ro.get("score") or 0), 3), kind=d["kind"] or "-", t_on=round(t1, 1),
                        v_on=round(float(vx[np.abs(t - t1).argmin()]), 2), d_pk=round(float(np.abs(de[w]).max()), 3),
                        slip_pk=round(float((np.abs(vy) / np.maximum(vx, 1)).max() if not w.any() else (np.abs(vy[w]) / np.maximum(vx[w], 1)).max()), 3),
                        A_pk=round(float(np.abs(vx * wz)[w].max()), 2))
            if rec:
                defs, vpl, A = [], [], {m: [] for m in ["served", *a.mode]}
                v2 = {m: [] for m in a.mode}
                for r in rec:
                    p = np.vstack([[0, 0, 0], np.asarray(r["poses"])])
                    seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
                    v0 = r.get("track", {}).get("v0", r.get("fix", {}).get("v0", seg[0] / 0.5))
                    defs.append(2 * v0 - seg[:4].sum()); vpl.append(seg[2:4].sum())
                    ps = np.vstack([[0, 0, 0], np.asarray(r.get("poses_track", r["poses"]))])
                    A["served"].append(demand(np.arange(9) * 0.5, ps[:, 0], ps[:, 1], ps[:, 2])[0])
                    a0 = 0.0                                          # the measured acceleration is not logged; JLON then starts from 0
                    for m in a.mode:
                        (tt, x, y, h), _ = TR.build(np.asarray(r["poses"]), v0, 0.0, mode=m, a0=a0, **kw)
                        dm, vv = demand(tt, x, y, h)
                        A[m].append(dm); v2[m].append(vv)
                info.update(deficit2=round(max(defs), 2), v_plan12=round(min(vpl), 2),
                            **{f"A_{m}": round(max(v), 2) for m, v in A.items()},
                            **{f"v2_{m}": round(min(v), 2) for m, v in v2.items()})
            out.append(info)
    if not out:
        print("no rows"); return
    keys = list(dict.fromkeys(k for o in out for k in o))
    print(" ".join(f"{k:>9}" for k in keys))
    for o in out:
        print(" ".join(f"{str(o.get(k, '')):>9}" for k in keys))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(out)


if __name__ == "__main__":
    main()
