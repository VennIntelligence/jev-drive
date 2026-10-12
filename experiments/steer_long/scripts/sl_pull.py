#!/usr/bin/env python3
"""STEER-LONG pull (research/next-round/steer-long.md, section 3.1): compact copies of stored AlpaSim rollouts for the Mac-side read.
Stdlib only, reads run dirs, never writes into them. Works on the GPU box (nuPlan held runs) and on the Tokyo box (PAI chunk runs).

  sl_pull.py <out dir> <label>=<run dir>[,<run dir>...] ...

A run dir is either a nuPlan run (`controller/`, `driver-logs/drive.jsonl`, `rollouts/`, `aggregate/`) or a PAI chunk run (the same under
`sim/`, the driver log in `driver/drive.jsonl`). Per label, <out>/<label>/:
  drive.jsonl.gz   the driver's `drive` records, only the fields the read uses
  ctrl.csv.gz      every controller CSV row with its session uuid in front
  metrics/<scene>__<rollout>.parquet   the evaluator's per-rollout series, copied as they are
  summary.json     the results-summary rollouts of all run dirs
The evaluator writes timestamps_us as a 128-bit integer that pyarrow cannot read; afterwards, with any Python that has polars (AlpaSim venv on the box, or a scratch venv on the Mac),
the per-rollout files of a label are concatenated into <label>/metrics64.parquet (name, timestamps_us cast to Int64, values, rid).
"""
import csv
import glob
import gzip
import json
import os
import shutil
import sys

KEEP = ("session", "scene", "now", "t0", "k", "infer", "anchor", "poses", "poses_model", "fix", "cmd", "n_keys")


def parts(run):
    sim = os.path.join(run, "sim") if os.path.isdir(os.path.join(run, "sim")) else run
    drv = os.path.join(run, "driver", "drive.jsonl")
    if not os.path.exists(drv):
        drv = os.path.join(run, "driver-logs", "drive.jsonl")
    return sim, drv


def main():
    out = sys.argv[1]
    for spec in sys.argv[2:]:
        lab, runs = spec.split("=", 1)
        d = os.path.join(out, lab)
        os.makedirs(os.path.join(d, "metrics"), exist_ok=True)
        summ, n_drv, n_ctl, n_met = [], 0, 0, 0
        with gzip.open(os.path.join(d, "drive.jsonl.gz"), "wt") as fd, gzip.open(os.path.join(d, "ctrl.csv.gz"), "wt", newline="") as fc:
            wc, head = csv.writer(fc), False
            for run in runs.split(","):
                sim, drv = parts(run)
                for line in open(drv):
                    r = json.loads(line)
                    if r.get("kind") == "drive":
                        fd.write(json.dumps({k: r[k] for k in KEEP if k in r}) + "\n")
                        n_drv += 1
                for f in sorted(glob.glob(os.path.join(sim, "controller", "alpasim_controller_*.csv"))):
                    sess = os.path.basename(f)[len("alpasim_controller_"):-4]
                    rows = list(csv.reader(open(f)))
                    if not head:
                        wc.writerow(["session"] + rows[0])
                        head = True
                    wc.writerows([sess] + r for r in rows[1:])
                    n_ctl += 1
                for f in glob.glob(os.path.join(sim, "rollouts", "*", "*", "metrics.parquet")):
                    scene, rid = f.split(os.sep)[-3:-1]
                    shutil.copy(f, os.path.join(d, "metrics", f"{scene}__{rid}.parquet"))
                    n_met += 1
                s = os.path.join(sim, "aggregate", "results-summary.json")
                if os.path.exists(s):
                    summ += json.load(open(s))["rollouts"]
        json.dump(summ, open(os.path.join(d, "summary.json"), "w"))
        print(lab, "drive", n_drv, "controller", n_ctl, "metrics", n_met, "summary", len(summ), flush=True)


if __name__ == "__main__":
    main()
