"""Log expert audit (fc65452:todos/2026-10-01-log-expert-audit.md): how many independent expert events per behaviour slice sit in
the real logs (WOD-E2E, NAVSIM navtrain, nuScenes), and how far the frozen Cinque native plan is from the human future
there compared with steady straight driving.

  build  wod|nav|nus   per-frame table (ego kinematics, human future on the 0.5 s grid, cached native plan)
                       -> $DATA_DIR/processed/log_expert_audit/<ds>.npz
  check                implementation checks (decisions 47 rule on WOD val, plan conversion vs the WOD exam preds)
  analyze              slices, independent events, native error + log-clustered bootstrap -> experiments/log_expert_audit/results/
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir, n_cpus  # noqa: E402

T_GRID = 0.5 * np.arange(1, 9)                        # human future / plan grid, 0.5 ... 4.0 s
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
CAM_X = {"wod": 1.519, "nav": 1.646, "nus": 1.70}     # camera ahead of the rear axle (op_adapt_r2 index cam_x)
R2 = data_dir() / "runs" / "op_adapt_r2"
OUT = data_dir() / "processed" / "log_expert_audit"
RES = REPO / "experiments/log_expert_audit/results"


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
                         columns=["sequence", "frame", "split", "intent", "has_future", "n_pref"])
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
           "fut": f[:, 1:16:2, :2], "fut20": f[:, :, :2],
           "rater": (ix.n_pref.to_numpy()[rows] > 0), "intent": np.array([-1, 1, 0, 2])[ix.intent.to_numpy()[rows]].astype(np.int8),
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


# ---------------------------------------------------------------- slices (definitions: todo "预登记" section)

SEG_VALID = 0.8                                       # m/s, a segment heading counts only above this speed
BASE = dict(st_v=0.5, st_d=3.0, st_sp=1.5, stay_d=0.5, stop_v0=3.0, stop_sp=0.5, turn=30.0, onset=10.0, t_lo=0.5, t_hi=3.0,
            past=10.0, nud_v0=3.0, nud_sp=2.0, peak=1.0, lc=2.5, ctl_v0=5.0, ctl_dv=1.5, ctl_psi=5.0, ctl_y=0.75, ctl_past=5.0)
LOOSE = {**BASE, "st_v": 0.8, "st_d": 1.5, "stop_v0": 2.0, "stop_sp": 0.8, "turn": 20.0, "onset": 7.0, "peak": 0.7, "lc": 2.0}
STRICT = {**BASE, "st_v": 0.3, "st_d": 6.0, "stop_v0": 5.0, "stop_sp": 0.3, "turn": 45.0, "onset": 15.0, "peak": 1.5}
VARIANTS = {"base": BASE, "loose": LOOSE, "strict": STRICT}
SLICES = ["start", "stay", "stop", "turn_onset", "in_turn", "nudge", "lane_change", "control"]
# slice -> (component read for the "clearly above control" rule, label)
COMPONENT = {"start": "lon", "stay": "lon", "stop": "lon", "turn_onset": "lat", "in_turn": "lat", "nudge": "lat",
             "lane_change": "lat"}
RATE = {"wod": 5.0, "nav": 2.0, "nus": 5.0}


def slices(tab: dict, th: dict = BASE) -> dict[str, np.ndarray]:
    fut = tab["fut"].astype(np.float64)
    p = np.concatenate([np.zeros_like(fut[:, :1]), fut], 1)
    seg = np.diff(p, axis=1)
    sp = np.linalg.norm(seg, axis=-1) / 0.5
    valid = sp >= SEG_VALID
    apsi = np.where(valid, np.abs(np.degrees(np.arctan2(seg[..., 1], seg[..., 0]))), -1.0)
    dp = (tab["pm05"] - tab["pm1"]).astype(np.float64)
    dpsi_past = np.where(np.linalg.norm(dp, axis=-1) >= 0.4, np.abs(np.degrees(np.arctan2(dp[:, 1], dp[:, 0]))), 0.0)
    v0 = tab["v0"].astype(np.float64)
    vmax1 = np.maximum(v0, np.maximum(tab["vm05"], tab["vm1"]))
    amax, y = apsi.max(1), np.abs(fut[..., 1])
    turn = amax >= th["turn"]
    hit = valid & (apsi >= th["onset"])
    ts = T_GRID[hit.argmax(1)]
    out = {}
    out["start"] = (vmax1 <= th["st_v"]) & (np.linalg.norm(fut[:, 7], axis=-1) >= th["st_d"]) & (sp.max(1) >= th["st_sp"])
    out["stay"] = (vmax1 <= th["st_v"]) & (np.linalg.norm(fut[:, 7], axis=-1) <= th["stay_d"])
    out["stop"] = (v0 >= th["stop_v0"]) & (sp[:, 6] <= th["stop_sp"]) & (sp[:, 7] <= th["stop_sp"])
    out["turn_onset"] = turn & hit.any(1) & (dpsi_past < th["past"]) & (ts >= th["t_lo"]) & (ts <= th["t_hi"])
    out["in_turn"] = turn & (dpsi_past >= th["past"])
    mov = (v0 >= th["nud_v0"]) & valid.all(1) & (sp >= th["nud_sp"]).all(1) & ~turn
    peak, yend, hend, hmax = y.max(1), y[:, 7], apsi[:, 7], amax
    ret = mov & (peak >= th["peak"]) & (yend < 0.5 * peak)
    hold = mov & (peak >= th["peak"]) & (yend >= 1.0) & (yend < th["lc"]) & (hend < 5) & (hmax > 2 * hend)
    out["nudge"] = ret | hold
    out["lane_change"] = mov & (yend >= th["lc"]) & (hend < 10) & (hmax > 2 * hend)
    steady = (np.abs(sp - v0[:, None]) <= th["ctl_dv"]).all(1)
    out["control"] = ((v0 >= th["ctl_v0"]) & valid.all(1) & steady & (amax <= th["ctl_psi"]) & (y.max(1) <= th["ctl_y"])
                      & (dpsi_past < th["ctl_past"]) & ((tab["intent"] == 1) | (tab["intent"] < 0)))
    return out


def mode_47(F: np.ndarray) -> np.ndarray:
    """decisions 47's rule (jevdrive/p6.mode_21, copied): 5 s futures (n, 20, 2) at 0.25 s -> mode name."""
    P0 = np.concatenate([np.zeros_like(F[:, :1]), F], 1)
    st = np.diff(P0, axis=1)
    hd = np.degrees(np.arctan2(st[..., 1], st[..., 0]))
    hd = np.stack([np.convolve(h, np.ones(3) / 3, "same") for h in hd])[:, 1:-1]
    sp = np.linalg.norm(st, axis=-1) / 0.25
    y = F[:, :, 1]
    peak, y_end = np.abs(y).max(1), np.abs(y[:, -1])
    h_end, h_max = np.abs(hd[:, -1]), np.abs(hd).max(1)
    v0, v_end = sp[:, 0], sp[:, -1]
    out = np.full(len(F), "keep", object)
    out[(v_end < 0.5) | ((v0 > 3) & (v_end < 0.3 * v0))] = "stop"
    out[peak >= 1] = "curve_or_other"
    out[(y_end >= 2.5) & (h_end < 10) & (h_max > 2 * h_end)] = "lane_change"
    out[(peak >= 1) & (y_end >= 1) & (y_end < 2.5) & (h_end < 5) & (h_max > 2 * h_end)] = "nudge_hold"
    out[(peak >= 1) & (y_end < 0.5 * peak)] = "nudge_return"
    out[h_end > 25] = "turn"
    return out


def events(log: np.ndarray, t: np.ndarray, mask: np.ndarray, gap: float = 1.0):
    """Independent events: flagged frames of one log whose neighbours in time are <= gap apart form one event.
    Returns (flagged row indices in (log, t) order, event id per flagged row)."""
    idx = np.flatnonzero(mask)
    idx = idx[np.lexsort((t[idx], log[idx]))]
    l, tt = log[idx], t[idx]
    new = np.r_[True, (l[1:] != l[:-1]) | (tt[1:] - tt[:-1] > gap)] if len(idx) else np.zeros(0, bool)
    return idx, np.cumsum(new) - 1


# ---------------------------------------------------------------- errors and the log-clustered bootstrap

NATIVE = [f"{k}{h}" for k in ("ade", "lon", "lat") for h in (1, 2, 3, 4)] + ["bias_lon3"]
CV = ["cv_" + m for m in NATIVE[:12]]                 # constant-velocity straight-ahead baseline, same metrics
CAP = ["cap_disp", "cap_stop", "cap_lat_end", "cap_peak"]
METRICS = NATIVE + CV + CAP
CAP_OF = {"start": "cap_disp", "stop": "cap_stop", "turn_onset": "cap_lat_end", "in_turn": "cap_lat_end",
          "lane_change": "cap_lat_end", "nudge": "cap_peak"}


def _err(fut: np.ndarray, plan: np.ndarray) -> list[np.ndarray]:
    e = plan.astype(np.float64) - fut.astype(np.float64)
    nrm, ax, ay = np.linalg.norm(e, axis=-1), np.abs(e[..., 0]), np.abs(e[..., 1])
    return [nrm[:, :2 * h].mean(1) for h in (1, 2, 3, 4)] + [ax[:, :2 * h].mean(1) for h in (1, 2, 3, 4)] \
        + [ay[:, :2 * h].mean(1) for h in (1, 2, 3, 4)] + [e[:, 5, 0]]


def errors(fut: np.ndarray, plan: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """(n, len(METRICS)): native ADE / lon / lat mean abs error up to 1..4 s and the signed longitudinal error at 3 s
    (plan - human); the same for a constant-velocity straight-ahead plan; and four capture indicators (did the plan
    reproduce at least half of the human manoeuvre: displacement at 4 s / stop by 4 s / signed lateral offset at 4 s /
    signed lateral offset at the human's lateral peak)."""
    fut, plan = fut.astype(np.float64), plan.astype(np.float64)
    cv = np.stack([v0[:, None] * T_GRID[None], np.zeros((len(fut), 8))], -1)
    hp = np.abs(fut[..., 1]).argmax(1)
    ar = np.arange(len(fut))
    cap_disp = np.linalg.norm(plan[:, 7], axis=-1) >= 0.5 * np.linalg.norm(fut[:, 7], axis=-1)
    cap_stop = np.linalg.norm(plan[:, 7] - plan[:, 6], axis=-1) / 0.5 <= 1.0
    cap_lat_end = (np.sign(plan[:, 7, 1]) == np.sign(fut[:, 7, 1])) & (np.abs(plan[:, 7, 1]) >= 0.5 * np.abs(fut[:, 7, 1]))
    cap_peak = (np.sign(plan[ar, hp, 1]) == np.sign(fut[ar, hp, 1])) & (np.abs(plan[ar, hp, 1]) >= 0.5 * np.abs(fut[ar, hp, 1]))
    return np.stack(_err(fut, plan) + _err(fut, cv)[:12] + [cap_disp, cap_stop, cap_lat_end, cap_peak], 1).astype(np.float64)


def boot_stats(log: np.ndarray, X: np.ndarray, masks: dict[str, np.ndarray], ctl: str = "control", B: int = 2000, seed: int = 0):
    """Cluster bootstrap over logs (slice and control share each resample). log (n,) int, X (n, m) metrics of the plan frames,
    masks name -> bool (n,). Returns {name: dict(mean (m,), draws (B, m), ctl_draws (B, m), ctl_mean (m,))}."""
    ul, li = np.unique(log, return_inverse=True)
    L = len(ul)
    W = np.random.default_rng(seed).multinomial(L, np.full(L, 1.0 / L), size=B).astype(np.float64)

    def draw(mask):
        n = np.bincount(li[mask], minlength=L).astype(np.float64)
        s = np.stack([np.bincount(li[mask], weights=X[mask, j], minlength=L) for j in range(X.shape[1])], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (W @ s) / (W @ n)[:, None], s.sum(0) / max(n.sum(), 1)

    bc, mc = draw(masks[ctl])
    out = {}
    for name, m in masks.items():
        if m.any():
            bs, ms = draw(m)
            out[name] = dict(mean=ms, draws=bs, ctl_draws=bc, ctl_mean=mc)
    return out


def ci(a: np.ndarray) -> tuple:
    return tuple(np.nanpercentile(a, [2.5, 97.5], axis=0))


# ---------------------------------------------------------------- analysis

def load(ds: str) -> dict:
    with np.load(OUT / f"{ds}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def sub(tab: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in tab.items()}


def count_rows(name: str, tab: dict, split: str, ds: str) -> list[dict]:
    m = (tab["split"] == split) & tab["count_ok"]
    t = sub(tab, m)
    rows = []
    for vn, th in VARIANTS.items():
        sl = slices(t, th)
        for s in SLICES:
            idx, ev = events(t["log"], t["t"], sl[s])
            n = int(ev.max() + 1) if len(idx) else 0
            plan_ok = ~np.isnan(t["plan"][idx, 0, 0]) if len(idx) else np.zeros(0, bool)
            ne_plan = len(np.unique(ev[plan_ok])) if plan_ok.any() else 0
            extra = {}
            if s == "turn_onset" and vn == "base":
                known = t["intent"][idx] >= 0
                extra["turn_intent_share"] = float(np.isin(t["intent"][idx][known], (0, 2)).mean()) if known.any() else np.nan
            rows.append({"dataset": name, "split": split, "variant": vn, "slice": s, "frames": int(len(idx)),
                         "seconds": round(len(idx) / RATE[ds], 1), "events": n, "logs": int(len(np.unique(t["log"][idx]))),
                         "events_with_plan": ne_plan, "total_frames": int(m.sum()), "total_logs": int(len(np.unique(t["log"]))),
                         **extra})
    return rows


def cmd_analyze():
    RES.mkdir(parents=True, exist_ok=True)
    crow, erow = [], []
    wod, nav, nus = load("wod"), load("nav"), load("nus")
    # counts: full train splits (val / nuScenes val for context)
    crow += count_rows("WOD-E2E", wod, "train", "wod") + count_rows("WOD-E2E", wod, "val", "wod")
    crow += count_rows("NAVSIM navtrain", nav, "train", "nav")
    crow += count_rows("nuScenes", nus, "train", "nus") + count_rows("nuScenes", nus, "val", "nus")
    # decisions-47 rule (5 s, 0.25 s grid) on all WOD frames: reproduces the val rater-frame count, and counts events
    for split in ("train", "val"):
        m = (wod["split"] == split) & wod["count_ok"]
        md = mode_47(wod["fut20"][m].astype(np.float64))
        t = sub(wod, m)
        for nm, sel in (("nudge_47", np.isin(md, ("nudge_return", "nudge_hold"))), ("lane_change_47", md == "lane_change")):
            idx, ev = events(t["log"], t["t"], sel)
            crow.append({"dataset": "WOD-E2E", "split": split, "variant": "rule47_5s", "slice": nm, "frames": int(len(idx)),
                         "seconds": round(len(idx) / 5.0, 1), "events": int(ev.max() + 1) if len(idx) else 0,
                         "logs": int(len(np.unique(t["log"][idx]))), "total_frames": int(m.sum())})
    pd.DataFrame(crow).to_csv(RES / "counts.csv", index=False)
    # errors
    sets = []
    m = (wod["split"] == "train") & (wod["plan_src"] == 1)
    sets += [("WOD-E2E train", "r2 teacher, front-only 5 Hz", "raw", sub(wod, m), 1.0), ("WOD-E2E train", "r2 teacher, front-only 5 Hz", "x1.06", sub(wod, m), 1.06)]
    m = (wod["split"] == "val") & (wod["plan_src"] == 2)
    sets += [("WOD-E2E val", "exam preds, 3 cam 10 s", "raw", sub(wod, m), 1.0), ("WOD-E2E val", "exam preds, 3 cam 10 s", "x1.06", sub(wod, m), 1.06)]
    sets += [("NAVSIM navtrain", "r2 teacher, 2 Hz sample-and-hold", "raw", nav, 1.0)]
    g = ~np.isnan(nav["plan_gimm"][:, 0, 0])
    sets += [("NAVSIM navtrain", "GIMM-interpolated (skill-pack N1/N2 rows)", "raw", {**sub(nav, g), "plan": nav["plan_gimm"][g]}, 1.0)]
    m = nus["split"] == "train"
    sets += [("nuScenes train", "r2 teacher, 20 Hz clock", "raw", sub(nus, m), 1.0)]
    m = nus["split"] == "val"
    sets += [("nuScenes val", "r2 teacher, 20 Hz clock", "raw", sub(nus, m), 1.0)]
    for dsn, src, cal, t, cx in sets:
        has = ~np.isnan(t["plan"][:, 0, 0])
        t = sub(t, has)
        plan = t["plan"].copy()
        plan[..., 0] *= cx
        X = errors(t["fut"], plan, t["v0"])
        sl = slices(t)
        sl["all"] = np.ones(len(X), bool)
        res = boot_stats(t["log"], X, sl)
        for s, r in res.items():
            idx, ev = events(t["log"], t["t"], sl[s])
            row = {"dataset": dsn, "plan_source": src, "calib": cal, "slice": s, "frames_plan": int(sl[s].sum()),
                   "events_plan": int(ev.max() + 1) if len(idx) else 0, "logs_plan": int(len(np.unique(t["log"][sl[s]])))}
            j = {m: i for i, m in enumerate(METRICS)}
            for mn in NATIVE + CV:
                row[mn] = r["mean"][j[mn]]
                row[mn + "_lo"], row[mn + "_hi"] = ci(r["draws"][:, j[mn]])
            for mn in NATIVE[:12]:                                     # pre-registered: ratio to the control slice
                rr = r["draws"][:, j[mn]] / r["ctl_draws"][:, j[mn]]
                row[mn + "_ratio"] = r["mean"][j[mn]] / r["ctl_mean"][j[mn]]
                row[mn + "_rlo"], row[mn + "_rhi"] = ci(rr)
                rc = r["draws"][:, j[mn]] / r["draws"][:, j["cv_" + mn]]   # added after the first numbers: native / constant velocity
                row[mn + "_vs_cv"] = r["mean"][j[mn]] / r["mean"][j["cv_" + mn]]
                row[mn + "_vs_cv_lo"], row[mn + "_vs_cv_hi"] = ci(rc)
            if s in CAP_OF:
                c = CAP_OF[s]
                row["capture_kind"], row["capture"] = c, r["mean"][j[c]]
                row["capture_lo"], row["capture_hi"] = ci(r["draws"][:, j[c]])
            erow.append(row)
        print(dsn, src, cal, "done", flush=True)
    pd.DataFrame(erow).round(4).to_csv(RES / "errors.csv", index=False)


# ---------------------------------------------------------------- checks

def cmd_check():
    out = {}
    wod = load("wod")
    m = (wod["split"] == "val") & wod["rater"]
    md = mode_47(wod["fut20"][m].astype(np.float64))
    out["rule47_val_rater_frames"] = {"n": int(m.sum()), **{k: int((md == k).sum()) for k in np.unique(md)},
                                      "expected_log_nudge_(decisions_47)": 15}
    # conversion: exam preds' own `wod` waypoints (0.25 s grid) vs plan_to_rear on plan_pos / plan_yaw
    pdir = data_dir() / "processed/wod_zeroshot/preds/op_cinque"
    fs = sorted(pdir.glob("*.npz"))[:300]
    d = []
    for f in fs:
        z = np.load(f)
        mm = np.zeros((1, 33, 12))
        mm[0, :, 0], mm[0, :, 1], mm[0, :, 11] = z["plan_pos"][:, 0], z["plan_pos"][:, 1], z["plan_yaw"]
        mine = plan_to_rear(mm, float(z["dev_xy"][0]))[0]
        d.append(np.abs(mine - z["wod"][1:16:2]).max())
    out["conversion_vs_exam_wod_key_max_abs_m"] = {"median": float(np.median(d)), "max": float(np.max(d))}
    # r2 val teacher vs exam preds on the same frames
    r2 = pd.read_parquet(R2 / "index/wod.parquet", columns=["uid", "part", "file", "row"])
    uid, mu = teacher_rows("wod")
    v = r2[r2.part == "val"]
    with ThreadPoolExecutor(16) as ex:
        nm = {f: n for f, n in zip(v.file.unique(), ex.map(_names_of, v.file.unique()))}
    names = np.array([nm[f][r] for f, r in zip(v.file, v.row)])
    pm = {f.stem: f for f in pdir.glob("*.npz")}
    both = [(i, n) for i, n in zip(v.index, names) if n in pm]
    diffs = []
    for i, n in both:
        z = np.load(pm[n])
        mm = np.zeros((1, 33, 12))
        mm[0, :, 0], mm[0, :, 1], mm[0, :, 11] = z["plan_pos"][:, 0], z["plan_pos"][:, 1], z["plan_yaw"]
        a = plan_to_rear(mm, float(z["dev_xy"][0]))[0]
        b = plan_to_rear(mu[i:i + 1], CAM_X["wod"])[0]
        diffs.append([np.linalg.norm(a - b, axis=-1).mean(), (b[-1, 0] - a[-1, 0])])
    dd = np.array(diffs) if diffs else np.zeros((0, 2))
    out["r2_teacher_vs_exam_preds_same_val_frames"] = {"n": len(diffs), "mean_dist_m": float(dd[:, 0].mean()) if len(dd) else None,
                                                       "median_dist_m": float(np.median(dd[:, 0])) if len(dd) else None,
                                                       "mean_dx_at_4s_r2_minus_exam": float(dd[:, 1].mean()) if len(dd) else None}
    for ds in ("wod", "nav", "nus"):
        t = load(ds)
        has = ~np.isnan(t["plan"][:, 0, 0])
        out[f"{ds}_plan_coverage"] = {"rows": int(len(has)), "with_plan": int(has.sum()),
                                     "frame_dt_median_s": float(np.median(np.diff(np.sort(t["t"][t["log"] == t["log"][0]]))))}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "analyze", "figure"])
    ap.add_argument("ds", nargs="?", choices=["wod", "nav", "nus"])
    a = ap.parse_args()
    if a.cmd == "build":
        {"wod": build_wod, "nav": build_nav, "nus": build_nus}[a.ds]()
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "analyze":
        cmd_analyze()
