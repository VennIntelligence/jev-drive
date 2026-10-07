#!/usr/bin/env python
"""Straight / left / right / start-frame split of an op_interp full-benchmark run (research/openpilot-diagnosis/index.html),
same convention as the 2000-token subset's by-command table: driving command from
meta.json's "cmd" (NAVSIM one-hot, argmax), overridden to "start (v0<1)" when meta.json's "speed" < 1 m/s.

    envs/jevdrive/bin/python experiments/op_openloop/archive/op_interp_by_command.py --data navfull --ver v1 --split navtest
Output: runs/op_interp/<data>/by_command.txt (also printed).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402

CMD = {0: "left", 1: "straight", 2: "right", 3: "unknown"}
SUBS = {"v1": ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound"],
        "v2": ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
               "ego_progress", "time_to_collision_within_bound"]}
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
         "ego_progress": "EP", "time_to_collision_within_bound": "TTC"}


def root(*p):
    return data_dir() / "runs" / "op_interp" / Path(*p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nav")
    ap.add_argument("--ver", default="v1", choices=("v1", "v2"))
    ap.add_argument("--split", default="navtest")
    a = ap.parse_args()
    mt = json.loads((root(a.data) / "meta.json").read_text())
    grp = {t: ("start (v0<1)" if s < 1 else CMD[c]) for t, c, s in zip(mt["names"], mt["cmd"], mt["speed"])}
    counts = pd.Series(grp).value_counts().to_dict()
    prefix = f"{a.ver}_{a.split}_opi_" + ("" if a.data == "nav" else f"{a.data}_")  # matches op_interp_score.sh
    rows = []
    for d in sorted(glob.glob(str(data_dir() / "runs/navsim/eval" / f"{prefix}*"))):
        fs = sorted(glob.glob(d + "/*/*.csv"))
        if not fs:
            continue
        name = Path(d).name[len(prefix):]
        df = pd.read_csv(fs[-1])
        df = df[df["token"].isin(grp) & df["valid"].astype(bool)].copy()
        df["grp"] = df["token"].map(grp)
        g = df.groupby("grp")
        rows.append(pd.DataFrame({"PDMS": 100 * g["score"].mean(),
                                   **{SHORT[s]: 100 * g[s].mean() for s in SUBS[a.ver] if s in df.columns}})
                    .T.assign(name=name).set_index("name", append=True).reorder_levels(["name", None]))
    out = pd.concat(rows) if rows else pd.DataFrame()
    (root(a.data) / "by_command.txt").write_text(f"{counts}\n\n{out.round(1).to_string()}\n")
    print(counts)
    with pd.option_context("display.width", 200):
        print(out.round(1).to_string())


if __name__ == "__main__":
    main()
