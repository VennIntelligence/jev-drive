#!/usr/bin/env python3
"""STOPLINE-PROBE nuPlan field check (box, CPU, read-only): do traffic-light states exist on the data disk, how dense are they, and can the
ego lane's light be resolved?

  .venv/bin/python experiments/stopline_probe/scripts/slp_nuplan_fields.py [--workers 8] --out results/nuplan_fields.json

  raw db    $DATA_DIR/datasets/nuplan_raw/**/*.db: table traffic_light_status (lidar_pc_token, lane_connector_id, status); per file the
            status counts and the share of lidar_pc frames with >= 1 status row
  navtrain  $DATA_DIR/datasets/navsim/navsim_logs/trainval/*.pkl: Frame["traffic_lights"] = [(lane_connector_id, is_red)]; frames, frames
            with a non-empty list, entries, red entries
  ego lane  LONG-PROBE's context (experiments/long_contact/scripts/lc_ctx.py -> $DATA_DIR/runs/long_contact/ctx/ctx.npz, one row per
            navtrain token): the first traffic-light stop line (map type 2) the hindsight route crosses, whether its lane connector was
            matched (ln_conn: lane id -> light), and whether the light state at t0 is known (red / non-red)
"""
import argparse
import glob
import json
import os
import pickle
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
FRONT = 4.049


def db_job(f):
    c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
    st = dict(c.execute("select status, count(*) from traffic_light_status group by status").fetchall())
    n_pc = c.execute("select count(*) from lidar_pc").fetchone()[0]
    n_pc_tl = c.execute("select count(distinct lidar_pc_token) from traffic_light_status").fetchone()[0]
    return dict(status=st, lidar_pc=n_pc, lidar_pc_with_tl=n_pc_tl)


def pkl_job(f):
    fr = pickle.load(open(f, "rb"))
    tl = [f.get("traffic_lights") or [] for f in fr]
    return dict(frames=len(fr), has_field=sum("traffic_lights" in f for f in fr), nonempty=sum(bool(t) for t in tl),
                entries=sum(len(t) for t in tl), red=sum(bool(b) for t in tl for _, b in t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dbs = sorted(glob.glob(str(D / "datasets/nuplan_raw/**/*.db"), recursive=True))
    pkls = sorted(glob.glob(str(D / "datasets/navsim/navsim_logs/trainval/*.pkl")))
    with ProcessPoolExecutor(a.workers) as ex:
        R = list(ex.map(db_job, dbs))
        P = list(ex.map(pkl_job, pkls, chunksize=8))
    st = {}
    for r in R:
        for k, v in r["status"].items():
            st[k] = st.get(k, 0) + v
    out = dict(raw_db=dict(files=len(dbs), splits=sorted({Path(f).parent.name for f in dbs}), status_rows=st,
                           lidar_pc=sum(r["lidar_pc"] for r in R), lidar_pc_with_tl=sum(r["lidar_pc_with_tl"] for r in R),
                           files_with_tl=sum(r["lidar_pc_with_tl"] > 0 for r in R)),
               navtrain_logs={k: sum(p[k] for p in P) for k in ("frames", "has_field", "nonempty", "entries", "red")} | {"logs": len(pkls)})
    z = np.load(D / "runs/long_contact/ctx/ctx.npz")
    ty, d, conn, st0 = z["ln_type"], z["ln_d"] - FRONT, z["ln_conn"], z["ln_st"][:, :, 0]
    tl = ty == 2
    q = np.where(tl.any(1), tl.argmax(1), -1)                     # first traffic-light line of each token's route
    i = np.flatnonzero(q >= 0)
    dq, cq, sq = d[i, q[i]], conn[i, q[i]], st0[i, q[i]]
    near = (dq > 0) & (dq <= 40)
    out["ego_lane"] = dict(tokens=int(len(z["ok"])), tokens_ok=int(z["ok"].sum()), with_tl_line_ahead=int(len(i)),
                           tl_line_0_40m=int(near.sum()), connector_matched_0_40m=int(cq[near].sum()),
                           state_known_t0_0_40m=int((cq & near & (sq >= 0)).sum()), red_t0_0_40m=int((cq & near & (sq == 1)).sum()),
                           nonred_t0_0_40m=int((cq & near & (sq == 0)).sum()),
                           stop_sign_line_ahead=int((ty == 1).any(1).sum()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
