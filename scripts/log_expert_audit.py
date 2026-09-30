"""Log expert audit (todos/2026-10-01-log-expert-audit.md): how many independent expert events per behaviour slice sit in
the real logs (WOD-E2E, NAVSIM navtrain, nuScenes), and how far the frozen Cinque native plan is from the human future
there compared with steady straight driving.

  build  wod|nav|nus   per-frame table (ego kinematics, human future on the 0.5 s grid, cached native plan)
                       -> $DATA_DIR/processed/log_expert_audit/<ds>.npz
  check                implementation checks (decisions 47 rule on WOD val, plan conversion vs the WOD exam preds)
  analyze              slices, independent events, native error + log-clustered bootstrap -> research/results/log-expert-audit/
  figure               research/figs/log_expert_audit.png

Everything is CPU; native plans are read from the op-adapt r2 teacher caches (read only). Run under
`taskset -c <cores>`; the pool size is min(n_cpus(), affinity).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir, n_cpus  # noqa: E402

T_GRID = 0.5 * np.arange(1, 9)                        # human future / plan grid, 0.5 ... 4.0 s
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
CAM_X = {"wod": 1.519, "nav": 1.646, "nus": 1.70}     # camera ahead of the rear axle (op_adapt_r2 index cam_x)
R2 = data_dir() / "runs" / "op_adapt_r2"
OUT = data_dir() / "processed" / "log_expert_audit"
RES = REPO / "research" / "results" / "log-expert-audit"


def pool_size() -> int:
    return max(1, min(n_cpus(), len(os.sched_getaffinity(0))))


# ---------------------------------------------------------------- native plan -> rear-axle grid

def plan_to_rear(mu: np.ndarray, cam_x: float) -> np.ndarray:
    """openpilot plan (n, 33, >=12; x fwd, y right at the camera, channel 11 = yaw) -> rear-axle x fwd / y left at T_GRID,
    (n, 8, 2). Same formula as op_adapt_score.plan_at: rear = (x + c - c cos(psi), -y - c sin(psi)), psi = -yaw."""
    mu = np.asarray(mu, np.float64)
    j = np.clip(np.searchsorted(T_IDXS, T_GRID, side="right") - 1, 0, len(T_IDXS) - 2)
    w = ((T_GRID - T_IDXS[j]) / (T_IDXS[j + 1] - T_IDXS[j]))[None]

    def at(ch):
        return mu[:, j, ch] * (1 - w) + mu[:, j + 1, ch] * w

    x, y, psi = at(0), at(1), -at(11)
    return np.stack([x + cam_x - cam_x * np.cos(psi), -y - cam_x * np.sin(psi)], -1).astype(np.float32)


def teacher_rows(dom: str) -> tuple[np.ndarray, np.ndarray]:
    """(uid (n,), mu (n, 33, 15) memmap) of the r2 teacher `op` plans of a domain."""
    d = R2 / "t" / "teacher" / dom
    return np.load(d / "uid.npy"), np.load(d / "mu.npy", mmap_mode="r")


def save(name: str, tab: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", **tab)
    print(name, {k: v.shape for k, v in tab.items()}, flush=True)


# ---------------------------------------------------------------- WOD-E2E

def _names_of(f: str) -> np.ndarray:
    with np.load(data_dir() / f, allow_pickle=True) as z:
        return z["names"]


def build_wod():
    ix = pd.read_parquet(data_dir() / "processed/waymo_e2e/index.parquet",
                         columns=["sequence", "frame", "split", "intent", "has_future"])
    past = np.load(data_dir() / "processed/waymo_e2e/past.npy", mmap_mode="r")
    fut = np.load(data_dir() / "processed/waymo_e2e/future.npy", mmap_mode="r")
    # native plans: r2 teacher (train, val streams) and the WOD exam preds (val frames, official protocol)
    r2 = pd.read_parquet(R2 / "index/wod.parquet", columns=["uid", "part", "file", "row", "ctx"])
    uid, mu = teacher_rows("wod")
    assert (r2.uid.to_numpy() == uid).all()
    with ThreadPoolExecutor(32) as ex:
        names = dict(zip(files := r2.file.unique(), ex.map(_names_of, files)))
    nm = np.array([names[f][r] for f, r in zip(r2.file, r2.row)])
    ctx_ok = np.array([c.min() >= 0 for c in r2.ctx])
    pos = {n: i for i, n in enumerate(nm)}
    seq_frame = ix.sequence + "-" + ix.frame.map("{:03d}".format)
    # val preds (exam protocol, raw plan: openpilot_to_wod with the camera at dev_xy)
    pdir = data_dir() / "processed/wod_zeroshot/preds/op_cinque"
    preds = {f.stem: f for f in pdir.glob("*.npz")}
    keep = (ix.has_future & (ix.split != "test")).to_numpy() & (((ix.frame % 2) == 0).to_numpy() | seq_frame.isin(preds).to_numpy())
    rows = np.flatnonzero(keep)
    print("wod rows", len(rows), flush=True)
    p, f = np.asarray(past[rows]), np.asarray(fut[rows])
    sp = lambda a: np.linalg.norm(a, axis=-1)  # noqa: E731
    tab = {"log": pd.factorize(ix.sequence.to_numpy()[rows])[0].astype(np.int32),
           "t": (ix.frame.to_numpy()[rows] * 0.1),
           "v0": sp(p[:, 15, 2:4]), "vm05": sp(p[:, 13, 2:4]), "vm1": sp(p[:, 11, 2:4]),
           "pm05": p[:, 13, :2], "pm1": p[:, 11, :2],
           "fut": f[:, 1:16:2, :2], "intent": np.array([-1, 1, 0, 2])[ix.intent.to_numpy()[rows]].astype(np.int8),
           "split": ix.split.astype(str).to_numpy()[rows], "count_ok": (ix.frame.to_numpy()[rows] % 2) == 0}
    plan = np.full((len(rows), 8, 2), np.nan, np.float32)
    src = np.zeros(len(rows), np.int8)                                   # 1 r2 teacher, 2 exam preds
    sf = seq_frame.to_numpy()[rows]
    is_train = (r2.part == "train").to_numpy()
    hit = [(k, pos[n]) for k, n in enumerate(sf) if n in pos and ctx_ok[pos[n]] and is_train[pos[n]]]
    kk, rr = np.array([h[0] for h in hit]), np.array([h[1] for h in hit])
    plan[kk] = plan_to_rear(mu[rr], CAM_X["wod"])
    src[kk] = 1
    for k, n in enumerate(sf):
        if n in preds and tab["split"][k] == "val":
            z = np.load(preds[n])
            m = np.zeros((1, 33, 12))
            m[0, :, 0], m[0, :, 1], m[0, :, 11] = z["plan_pos"][:, 0], z["plan_pos"][:, 1], z["plan_yaw"]
            plan[k] = plan_to_rear(m, float(z["dev_xy"][0]))[0]
            src[k] = 2
    tab.update(plan=plan, plan_src=src)
    save("wod", tab)


# ---------------------------------------------------------------- NAVSIM navtrain

def _log_times(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return [(e["token"], e["timestamp"] * 1e-6) for e in d]


def build_nav():
    idx = pickle.load(open(data_dir() / "runs/navsim_zs/index/navtrain_slim.pkl", "rb"))
    fz = np.load(data_dir() / "runs/navsim_zs/index/navtrain_future.npz")
    tok = np.array([e["token"] for e in idx])
    assert (fz["tokens"] == tok).all()
    logs = sorted({e["log_name"] for e in idx})
    with ProcessPoolExecutor(pool_size()) as ex:
        res = ex.map(_log_times, [str(data_dir() / f"datasets/navsim/navsim_logs/trainval/{l}.pkl") for l in logs])
    ttab = {tk: t for r in res for tk, t in r}
    pose = np.stack([e["pose"][:, :2] for e in idx])                     # (n, 4, 2): -1.5 ... 0 s, current ego frame
    vel = np.linalg.norm(np.stack([e["vel"] for e in idx]), axis=-1)
    cmd = np.stack([e["cmd"][-1] for e in idx])
    intent = np.where(cmd[:, :3].sum(1) > 0, cmd[:, :3].argmax(1), -1).astype(np.int8)   # 0 left, 1 straight, 2 right
    tab = {"log": pd.factorize(np.array([e["log_name"] for e in idx]))[0].astype(np.int32),
           "t": np.array([ttab[t] for t in tok]), "v0": vel[:, 3], "vm05": vel[:, 2], "vm1": vel[:, 1],
           "pm05": pose[:, 2], "pm1": pose[:, 1], "fut": fz["poses"][:, :, :2], "intent": intent,
           "split": np.full(len(idx), "train"), "count_ok": np.ones(len(idx), bool)}
    r2 = pd.read_parquet(R2 / "index/nav.parquet", columns=["uid", "token"])
    uid, mu = teacher_rows("nav")
    assert (r2.uid.to_numpy() == uid).all()
    at = {t: i for i, t in enumerate(r2.token)}
    rr = np.array([at[t] for t in tok])
    tab["plan"] = plan_to_rear(mu[rr], CAM_X["nav"])                    # sample-and-hold 2 Hz protocol, all tokens
    plan2 = np.full((len(tok), 8, 2), np.nan, np.float32)               # GIMM-interpolated protocol (skill-pack N1 / N2 rows)
    pos = {t: i for i, t in enumerate(tok)}
    for f in ("lb_n1train_n19968", "lb_n2train_n40000"):
        z = np.load(data_dir() / "runs/skill_pack/n1/feat" / f"{f}.npz")
        plan2[[pos[t] for t in z["tokens"]]] = z["native"][:, :, :2]
    tab["plan_gimm"] = plan2
    save("nav", tab)


# ---------------------------------------------------------------- nuScenes

def _nus_scene(a):
    """One scene: 5 Hz slots, ego kinematics from the 20 Hz LIDAR_TOP poses, r2 teacher row of each slot."""
    name, file, rows, steps, ts = a
    sc = _IDX["scenes"][name]
    pt, xyz, R = sc["pose_t"].astype(np.float64), sc["pose_xyz"], sc["pose_R"]
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    t = ts * 1.0                                                          # slot times, us
    S = 1e6

    def xy_at(tt):
        return np.stack([np.interp(tt, pt, xyz[:, 0]), np.interp(tt, pt, xyz[:, 1])], -1)

    ok = (t - 1.0 * S >= pt[0] + 0.25 * S) & (t + 4.0 * S <= pt[-1])
    t, rows, steps = t[ok], rows[ok], steps[ok]
    if len(t) == 0:
        return None
    y0 = np.interp(t, pt, yaw)
    c, s = np.cos(y0), np.sin(y0)

    def ego(dt):                                                          # position at t + dt in the ego frame at t
        d = xy_at(t + dt * S) - xy_at(t)
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)

    def speed(dt):
        return np.linalg.norm(xy_at(t + (dt + 0.25) * S) - xy_at(t + (dt - 0.25) * S), axis=-1) / 0.5

    return dict(name=name, rows=rows, t=t / S, v0=speed(0.0), vm05=speed(-0.5), vm1=speed(-1.0), pm05=ego(-0.5), pm1=ego(-1.0),
                fut=np.stack([ego(k) for k in T_GRID], 1), split=sc["split"])


_IDX = None


def _nus_init():
    global _IDX
    _IDX = pickle.load(open(data_dir() / "processed/nusc_zs/trainval_index.pkl", "rb"))


def build_nus():
    import sys as _s
    _s.path.insert(0, str(REPO))
    from jevdrive import op_adapt_data as D
    idx = pickle.load(open(data_dir() / "processed/nusc_zs/trainval_index.pkl", "rb"))
    r2 = pd.read_parquet(R2 / "index/nus.parquet", columns=["uid", "key", "file", "row", "ctx"])
    uid, mu = teacher_rows("nus")
    assert (r2.uid.to_numpy() == uid).all() and (r2.index.to_numpy() == np.arange(len(r2))).all()
    jobs = []
    for name, g in r2.groupby("key"):
        f = g.file.iloc[0]
        with np.load(data_dir() / f) as z:
            steps = z["steps"]
        plan = D.scene_plan(idx, name)
        st = steps[g.row.to_numpy()]
        keep = (st % 4 == 0) & np.array([c.min() >= 0 for c in g.ctx])           # 5 Hz lattice, full 9-slot context
        jobs.append((name, f, g.index.to_numpy()[keep], st[keep], plan["t"][st[keep]]))
    with ProcessPoolExecutor(pool_size(), initializer=_nus_init) as ex:
        out = [o for o in ex.map(_nus_scene, jobs, chunksize=4) if o is not None]
    rows = np.concatenate([o["rows"] for o in out])
    cat = lambda k: np.concatenate([o[k] for o in out])  # noqa: E731
    n = len(rows)
    tab = {"log": np.concatenate([np.full(len(o["rows"]), i, np.int32) for i, o in enumerate(out)]), "t": cat("t"),
           "v0": cat("v0"), "vm05": cat("vm05"), "vm1": cat("vm1"), "pm05": cat("pm05"), "pm1": cat("pm1"),
           "fut": np.concatenate([o["fut"] for o in out]), "intent": np.full(n, -1, np.int8),
           "split": np.concatenate([np.full(len(o["rows"]), o["split"]) for o in out]), "count_ok": np.ones(n, bool)}
    tab["plan"] = plan_to_rear(mu[rows], CAM_X["nus"])
    save("nus", tab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "analyze", "figure"])
    ap.add_argument("ds", nargs="?", choices=["wod", "nav", "nus"])
    a = ap.parse_args()
    if a.cmd == "build":
        {"wod": build_wod, "nav": build_nav, "nus": build_nus}[a.ds]()
