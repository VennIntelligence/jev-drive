#!/usr/bin/env python3
"""SPIN-FIX collector (Tokyo host, numpy only): every rollout of the given pai_eval.sh roots -> one gzipped CSV of controller rows and one JSON
of rollout facts, pulled to the Mac for sf_cases.py.

  sf_collect.py <out dir> <label>=<pai_eval root> ...
rows.csv.gz: label, session, scene, t (s since the first row + 0.1), x, y, yaw (local), vx, vy, wz, u (steering command), delta (steering state),
acc; rollouts.json: label, session, scene, score, the summary's metrics (flat), the chunk run dir.
"""
import csv, glob, gzip, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor


def one(job):
    label, sim, r = job
    c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
    if not os.path.exists(c):
        return None, []
    rows = list(csv.DictReader(open(c)))
    t0 = float(rows[0]["timestamp_us"])
    out = []
    for x in rows:
        qx, qy, qz, qw = (float(x[k]) for k in ("qx", "qy", "qz", "qw"))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        out.append([label, r["rollout_id"], r["clipgt_id"], round((float(x["timestamp_us"]) - t0) / 1e6 + 0.1, 3), x["x"], x["y"], round(yaw, 6),
                    x["vx"], x["vy"], x["wz"], x["u_steering_angle"], x["front_steering_angle"], x["acceleration"]])
    m = {k: v for k, v in (r.get("metrics") or {}).items() if isinstance(v, (int, float, bool, type(None)))}
    return dict(label=label, session=r["rollout_id"], scene=r["clipgt_id"], score=float(r.get("score") or 0), run=os.path.dirname(sim),
                t_us0=t0, metrics=m), out


def main():
    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    os.makedirs(out, exist_ok=True)
    jobs = []
    for label, root in specs:
        for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
            f = os.path.join(sim, "aggregate", "results-summary.json")
            if os.path.exists(f):
                jobs += [(label, sim, r) for r in json.load(open(f))["rollouts"]]
    meta = []
    with gzip.open(os.path.join(out, "rows.csv.gz"), "wt", newline="") as g, ProcessPoolExecutor(int(os.environ.get("NPROC", "6"))) as ex:
        w = csv.writer(g)
        w.writerow(["label", "session", "scene", "t", "x", "y", "yaw", "vx", "vy", "wz", "u", "delta", "acc"])
        for m, rows in ex.map(one, jobs, chunksize=8):
            if m:
                meta.append(m)
                w.writerows(rows)
    json.dump(meta, open(os.path.join(out, "rollouts.json"), "w"))
    print(len(meta), "rollouts of", len(jobs))


if __name__ == "__main__":
    main()
