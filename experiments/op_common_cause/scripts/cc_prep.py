"""Common-cause factorial: freeze the per-domain sample tables (plan: ../plans/2026-10-03-common-cause-prereg.md).

Main venv. Writes $DATA_DIR/runs/op_common_cause/samples/<domain>.json: one row per sample with the id, cluster, t0 speed,
speed bin, command (-1 left, 0 straight, 1 right), the expert future on the 0.25 s grid (20, 2) rear-axle x fwd / y left
(NaN where the source has none), the camera position on the vehicle and what the runner needs to build the frames.

  .venv/bin/python experiments/op_common_cause/scripts/cc_prep.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run  # noqa: E402

BINS = (0.0, 0.5, 3.0, 8.0, np.inf)
BIN_NAMES = ("stop", "low", "mid", "high")
T_FUT = 0.25 * np.arange(1, 21)
HIST_LONG = 24            # 5 Hz frames before t0 for `long` (4.8 s)
HIST_NORMAL = 8           # 5 Hz frames before t0 for `normal` (1.6 s covers the 1.5 s step window)
OUT = data_dir() / "runs" / "op_common_cause" / "samples"


def speed_bin(v):
    return BIN_NAMES[int(np.searchsorted(BINS, v, side="right") - 1)]


def take(rng, idx, n):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def prep_nav(run, rng):
    from jevdrive import navsim_zs as Z
    mt = json.loads((data_dir() / "runs/op_lb/lb_navtrain/meta.json").read_text())
    sp = splits.load("navsim/navtrain")
    run.use_split(sp)
    idx = {e["token"]: e for e in Z.load_index("navtrain", slim=True)}
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        fut = dict(zip(f["tokens"].tolist(), f["poses"]))
    v = np.asarray(mt["speed"])
    b = np.array([speed_bin(x) for x in v])
    keep = np.sort(np.r_[np.flatnonzero(b != "mid"), take(rng, np.flatnonzero(b == "mid"), 600)])
    rows = []
    for i in keep:
        tok = mt["names"][i]
        e = idx[tok]
        assert tok in sp
        p = np.asarray(fut[tok], float)                                     # (8, 3) at 0.5 s
        t8 = 0.5 * np.arange(1, 9)
        f20 = np.stack([np.interp(T_FUT, t8, p[:, k], right=np.nan) for k in range(2)], 1)
        rows.append(dict(id=tok, row=int(i), cluster=e["log_name"], v=float(v[i]), bin=b[i],
                         cmd={0: -1, 1: 0, 2: 1}.get(mt["cmd"][i], 0), fut=f20.tolist(), cam=mt["cam"][i],
                         lht=bool(mt["lht"][i])))
    return rows


def prep_wod(run, rng):
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as WZ
    run.use_split(splits.load("wod/val"))
    sets = WZ.load_sets()
    spans, _ = WZ.load_spans()
    calib = json.loads((WZ.root() / "op_calib.json").read_text())
    df = W.load_index()
    past, fut = W.load_ego()
    pos = {n: k for k, n in enumerate(W.frame_names(df))}
    rows = []
    for name in sorted({str(n) for s in ("rater", "extra") for n in sets[s]["name"]}):
        k = pos[name]
        seq, fr = name.rsplit("-", 1)
        if seq not in calib or str(df.split.iloc[k]) != "val":
            continue
        hist = [f"{seq}-{int(fr) - 2 * j:03d}" for j in range(HIST_LONG, -1, -1)]      # 5 Hz, oldest first
        avail = [h in spans for h in hist]
        if not all(avail[-(HIST_NORMAL + 1):]):
            continue
        first = int(np.flatnonzero(~np.asarray(avail))[-1] + 1) if not all(avail) else 0
        vv = float(np.linalg.norm(past[k, -1, 2:4]))
        cmd = {2: -1, 1: 0, 3: 1}.get(int(df.intent.iloc[k]), 0)
        dev = np.array(calib[seq]["1"]["extrinsic"]).reshape(4, 4)[:3, 3]
        rows.append(dict(id=name, cluster=seq, v=vv, bin=speed_bin(vv), cmd=cmd, fut=fut[k, :, :2].tolist(),
                         cam=dev.tolist(), hist=hist[first:], long=first == 0))
    return rows


def prep_carla(run, rng):
    from jevdrive import p5_openpilot as P
    D = data_dir() / "processed/carla_p5"
    t = pd.read_parquet(D / "index.parquet")
    past, fut = np.load(D / "past.npy"), np.load(D / "future.npy")
    plan = json.loads((D / "op_plan.json").read_text())
    st = {s["key"]: s for s in plan["streams"]}
    routes = sorted(t.route_id[t.source == "p4"].unique())
    # P4 routes are Bench2Drive routes; no registered split covers these recordings (they are not a benchmark set), so
    # the sample is registered here as its own frozen membership (unit = route).
    sp = splits.define("b2d", "p4-carla-routes", sorted(map(str, routes)), unit="route",
                       origin="processed/carla_p5/index.parquet, source == p4 (P4 BehaviorAgent recordings, 5 Hz)",
                       status="frozen", used_by=["op_common_cause"],
                       notes="Bench2Drive route ids that the P4 generator recorded; no training use")
    run.use_split(sp)
    cand = []
    for k in np.flatnonzero((t.source == "p4").to_numpy()):
        r = t.iloc[k]
        s = st.get(f"p4_{r.route_id}")
        if s is None or r.frame_name not in s["names"]:
            continue
        j = s["names"].index(r.frame_name)
        if j < HIST_NORMAL:
            continue
        fr = [int(n.rsplit("-", 1)[1]) for n in s["names"][max(0, j - HIST_LONG): j + 1]]
        if np.any(np.diff(fr[-(HIST_NORMAL + 1):]) != 4):        # a gap in the 1.6 s window
            continue
        cand.append((k, j, s["key"], len(fr) == HIST_LONG + 1 and bool(np.all(np.diff(fr) == 4))))
    v = np.linalg.norm(past[[c[0] for c in cand], -1, 2:4], axis=1)
    b = np.array([speed_bin(x) for x in v])
    cap = {"stop": 400, "low": 600, "mid": 600, "high": 10 ** 9}
    keep = np.sort(np.concatenate([take(rng, np.flatnonzero(b == n), cap[n]) for n in BIN_NAMES]))
    rows = []
    for c in keep:
        k, j, key, lg = cand[c]
        r = t.iloc[k]
        rows.append(dict(id=r.frame_name, cluster=str(r.route_id), v=float(v[c]), bin=b[c],
                         cmd={2: -1, 1: 0, 3: 1}.get(int(r.intent), 0), fut=fut[k].tolist(),
                         cam=[P.RIG[0][1], P.RIG[0][2], P.RIG[0][3]], stream=key, j=j,
                         files=st[key]["files"][max(0, j - HIST_LONG): j + 1], long=lg))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_common_cause", "prep", seed=0) as run:
        for dom, fn in (("nav", prep_nav), ("wod", prep_wod), ("carla", prep_carla)):
            rows = fn(run, np.random.default_rng(0))
            (OUT / f"{dom}.json").write_text(json.dumps(rows))
            c = pd.DataFrame(rows)
            tab = pd.crosstab(c.bin, c.cmd)
            run.info(f"{dom}: {len(rows)} samples, {c.cluster.nunique()} clusters, long={int(c.get('long', pd.Series([0])).sum())}\n{tab}")
            run.summary[dom] = len(rows)


if __name__ == "__main__":
    main()
