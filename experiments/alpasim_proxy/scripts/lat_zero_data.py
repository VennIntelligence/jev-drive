#!/usr/bin/env python3
"""LAT-ZERO data census (CPU, read-only): how many training rows existing data holds for two situations.

  T (sharp turn): the human's path turns >= 45 deg of heading within the next 20 m of travel (and within the 4 s future the row
    holds), the human moves >= 1 m/s at the row; split right / left; sub-count human speed in the turn (segment speed at the
    mid-turn point, |dyaw| first >= 22.5 deg) <= 6 m/s.
  S (brake to a stop, no lead): human >= 2 m/s at the row, two consecutive 0.5 s segments < 0.3 m/s starting at <= 3 s; no vehicle
    in the human's corridor within 30 m at t0 (wait_cap wc.log_rows rule: lateral < 1.5 m of the path, gap from the front bumper).
  On-policy pools (S rows, mode self): ego faster than the human at the ego's station (ego speed > 1.05 x human speed there) in a
    sharp turn of the human path ahead; STAND-R domain (supplement 20: human 4 s arc < 0.5 m, 0.5 <= ego v0 < 3, gap finite).

Box, repo root:
  scripts/tmux_run.sh lzd ~/data/envs/jevdrive/bin/python experiments/alpasim_proxy/scripts/lat_zero_data.py
  -> runs/alpasim_proxy/lat_zero_data/{data_counts.json,DONE}
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/wait_cap/scripts")]
import wc  # noqa: E402

DATA = Path("/root/autodl-tmp/ujs")
CR = DATA / "runs/op_parity/cache"
OUT = DATA / "runs/alpasim_proxy/lat_zero_data"
ROWS = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y"} | {f"pn{i}": f"PN{i}" for i in range(7)}
TURN, REACH, V_MOVE, V_TURN, V_STOP_IN, V_STOP, LEAD = 45.0, 20.0, 1.0, 6.0, 2.0, 0.3, 30.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def classify(F, v0, Y=None, yaw0=0.0):
    """F (n, 8, 2) human positions at 0.5 .. 4 s in a frame with the row at the origin; v0 (n,) human speed at the row; Y (n, 8) yaw
    (else from the polyline); yaw0 (n,) path heading at the origin -> dict of (n,) arrays."""
    F = np.asarray(F, np.float64)
    P = np.concatenate([np.zeros((len(F), 1, 2)), F], 1)
    seg = np.diff(P, axis=1)
    sl = np.hypot(seg[..., 0], seg[..., 1])
    arc = np.cumsum(sl, 1)
    if Y is None:                                                     # segment headings, carried over through short segments
        h = np.arctan2(seg[..., 1], seg[..., 0])
        h = np.where(sl > 0.2, h, np.nan)
        h[:, 0] = np.where(np.isnan(h[:, 0]), 0.0, h[:, 0])
        for j in range(1, 8):
            h[:, j] = np.where(np.isnan(h[:, j]), h[:, j - 1], h[:, j])
    else:
        h = np.asarray(Y, np.float64)
    d = np.degrees(np.unwrap(wrap(h - np.asarray(yaw0)[..., None] if np.ndim(yaw0) else h - yaw0), axis=1))
    d = np.where(arc <= REACH, d, 0.0)                                # within the next 20 m of travel (and the 4 s the row holds)
    k = np.abs(d).argmax(1)
    dy = d[np.arange(len(d)), k]
    vs = sl / 0.5
    mid = np.abs(d) >= TURN / 2
    jm = np.where(mid.any(1), mid.argmax(1), 0)
    vturn = np.where(mid.any(1), vs[np.arange(len(d)), jm], np.nan)
    st = vs < V_STOP
    two = st[:, :-1] & st[:, 1:]                                      # stopped over segments k, k + 1 (k <= 6: from <= 3 s, for >= 1 s)
    v0 = np.asarray(v0, np.float64)
    fin = np.isfinite(F).all((1, 2))
    return dict(dyaw=dy, vturn=vturn, sharp=fin & (v0 >= V_MOVE) & (np.abs(dy) >= TURN), stop=fin & (v0 >= V_STOP_IN) & two.any(1),
                arc8=arc[:, -1])


def lead_gap(path, B):
    """path (m, 2) from the origin, B (k, >= 5) x, y, yaw, l, w vehicles at t0 in the same frame -> wc.log_rows' in-path gap (inf: none)."""
    if not len(B):
        return np.inf
    R, _ = wc.resample(np.asarray(path, np.float64), n=91)             # 0 .. 45 m
    dl, sa = wc.lat_s(R, B[:, :2])
    gap = sa - B[:, 3] / 2 - wc.FRONT
    inp = (dl < 1.5) & (gap > -1.0) & (B[:, 0] > 0)
    return float(gap[inp].min()) if inp.any() else np.inf


def counts(C, nolead, unit, extra=None):
    """C = classify output, nolead (n,) bool or None, unit (n,) str (log / scene) -> the census record."""
    unit = np.asarray(unit)
    sh, R = C["sharp"], C["dyaw"] < 0

    def c(m):
        return dict(rows=int(m.sum()), units=int(len(np.unique(unit[m]))))
    r = dict(rows=int(len(sh)), units=int(len(np.unique(unit))),
             T_right=c(sh & R), T_left=c(sh & ~R), T_right_slow6=c(sh & R & (C["vturn"] <= V_TURN)), T_left_slow6=c(sh & ~R & (C["vturn"] <= V_TURN)),
             S_stop_any_lead=c(C["stop"]))
    if nolead is not None:
        r["S_stop_nolead"] = c(C["stop"] & nolead)
    return r | (extra or {})


# ---------------------------------------------------------------- nuPlan trainval (TV-AUDIT census: every eligible 2 Hz frame)
def nuplan_tv():
    z = np.load(DATA / "runs/self_iter/tv_census/frames.npz", allow_pickle=True)
    C = classify(z["fut"], z["v0"])
    nl = z["lead_any"] > LEAD
    tok, log = z["token"].astype(str), z["log"].astype(str)
    tv = {n[3:] for n in np.load(CR / "on1_NR1TV/tab.npz")["names"].astype(str)}
    nav = set()
    for f in sorted(glob.glob(str(CR / "navtrain_full.s*of12/tab.npz"))):
        nav |= set(np.load(f)["names"].astype(str).tolist())
    out = {"census_all": counts(C, nl, log)}
    for k, s in (("NR1TV_rows", tv), ("navtrain_tokens", nav)):
        m = np.isin(tok, list(s))
        out[k] = counts({a: b[m] for a, b in C.items()}, nl[m], log[m], dict(set_size=len(s), matched=int(m.sum())))
    return out


# ---------------------------------------------------------------- NAVSIM navtest (lb_navtest tab + agent_labels navtest-k32)
def navtest():
    z = np.load(CR / "lb_navtest/tab.npz")
    names, fut = z["names"].astype(str), z["fut"]
    C = classify(fut[..., :2], z["speed"], fut[..., 2])
    a = np.load(DATA / "runs/op_parity/agent_labels/navtest-k32.npz")
    pos = {t: i for i, t in enumerate(a["tokens"].astype(str))}
    gaps = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        if n not in pos or not C["stop"][i]:
            continue
        q = pos[n]
        v = a["valid"][q, 0] & (a["cls"][q] == 0)
        gaps[i] = lead_gap(np.r_[np.zeros((1, 2)), fut[i, :, :2]], a["box"][q, 0][v])
    return counts(C, ~(gaps <= LEAD), z["log"].astype(str), dict(lead_src="agent_labels/navtest-k32 cls 0 (vehicle) at t0"))


# ---------------------------------------------------------------- on-policy pools (S rows)
def pools():
    zx = np.load(CR / "on1_JTX/tab.npz")
    names = zx["names"].astype(str)
    nj = len(np.load(CR / "on1_JT/tab.npz")["names"])
    assert np.array_equal(np.load(CR / "on1_JT/tab.npz")["names"].astype(str), names[:nj])
    S = zx["mode"].astype(str) == "self"
    part = np.array([n.split(":")[1] for n in names])
    n = len(names)
    vlog, lead, Dreg, gp = np.full(n, np.nan), np.full(n, np.nan), np.full((n, 8), np.nan), np.full((n, 86, 2), np.nan, np.float32)
    for p, d in ROWS.items():
        rows = np.flatnonzero(S & (part == p))
        if not len(rows):
            continue
        zz = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        m = {k: zz[k] for k in ("names", "v_log", "lead", "D_reg", "gpath", "speed")}
        pos = {q: i for i, q in enumerate(m["names"].astype(str).tolist())}
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in rows])
        assert np.allclose(m["speed"][mi], zx["speed"][rows], atol=1e-3), d
        vlog[rows], lead[rows], Dreg[rows], gp[rows] = m["v_log"][mi], m["lead"][mi], m["D_reg"][mi], m["gpath"][mi]
    g = gp.astype(np.float64)
    yaw0 = np.arctan2(g[:, 16, 1] - g[:, 14, 1], g[:, 16, 0] - g[:, 14, 0])  # logged path heading at the ego's station (index 15)
    fr = zx["fut_reg"]
    C = classify(fr[..., :2], vlog, fr[..., 2], yaw0)
    # human speed at the ego's station: exact (stoppt labels hs8, on1_JT S rows) else from D_reg when the human reaches it within 4 s
    vh = np.full(n, np.nan)
    L = np.load(DATA / "runs/defect_fix/stoppt/labels.npz")
    lpos = {q: i for i, q in enumerate(L["names"].astype(str).tolist())}
    li = np.array([lpos.get(q, -1) for q in names])
    ok = li >= 0
    vh[ok] = L["hs8"][li[ok], 0] / 0.5
    Dp = np.concatenate([np.zeros((n, 1)), Dreg], 1)
    vseg = np.diff(Dp, axis=1) / 0.5
    j = np.array([np.searchsorted(Dp[i], lead[i]) - 1 if np.isfinite(lead[i]) else -1 for i in range(n)])
    good = (lead >= 0) & (j >= 0) & (j < 8)
    vh_d = np.where(good, vseg[np.arange(n), np.clip(j, 0, 7)], np.nan)
    agree = ok & np.isfinite(vh_d) & S
    vh = np.where(np.isfinite(vh), vh, vh_d)
    # lead: pool agent boxes at t0 (every simulator actor is a vehicle), along the logged path ahead of the ego's station
    a = np.load(CR / "on1_JTX/agent.npz")
    box0, val0 = a["box"][:, 0], a["valid"][:, 0]
    gaps = np.full(n, np.nan)
    for i in np.flatnonzero(S & C["stop"]):
        gaps[i] = lead_gap(g[i, 15:], box0[i][val0[i]])
    nolead = ~(gaps <= LEAD)
    v0 = zx["speed"].astype(np.float64)
    fast = v0 > 1.05 * vh
    sh, R = C["sharp"], C["dyaw"] < 0
    # STAND-R domain (nr1_free.rolling): human 4 s arc of fut_reg < 0.5 m, 0.5 <= v0 < 3, stand_gap finite
    hum4 = classify(fr[..., :2], np.zeros(n))["arc8"]                 # = nr1_free.cum_arc(fut_reg)[:, -1]
    fin = np.isfinite(zx["fut"]).all((1, 2)) & np.isfinite(fr).all((1, 2))
    gap = np.load(CR / "on1_JTX/stand_gap.npz")
    assert np.array_equal(gap["names"].astype(str), names)
    standr = S & fin & (hum4 < 0.5) & (v0 >= 0.5) & (v0 < 3.0) & np.isfinite(gap["gap"])
    stand = S & fin & (hum4 < 0.5) & (v0 < 0.5)
    out = {}
    for pool, m0 in (("on1_JT", np.arange(n) < nj), ("on1_JTX", np.ones(n, bool)), ("on1_JTX_pn_only", np.arange(n) >= nj)):
        m = m0 & S
        Cm = {k: v[m] for k, v in C.items()}
        unit = zx["log"].astype(str)[m]

        def c(q):
            return dict(rows=int((q & m).sum()), units=int(len(np.unique(zx["log"].astype(str)[q & m]))))
        ex = dict(ego_faster_sharp_right=c(sh & R & fast), ego_faster_sharp_left=c(sh & ~R & fast),
                  ego_faster_sharp_right_slow6=c(sh & R & fast & (C["vturn"] <= V_TURN)),
                  ego_faster_sharp_right_vh_station_exact=c(sh & R & fast & ok), vh_missing_in_sharp=c(sh & ~np.isfinite(vh)),
                  standr_domain=c(standr), standing_domain=c(stand))
        out[pool] = counts(Cm, nolead[m], unit, ex)
    d = vh_d[agree] - L["hs8"][li[agree], 0] / 0.5
    out["vh_station_check"] = dict(n=int(agree.sum()), med_abs=float(np.median(np.abs(d))) if len(d) else None)
    tvm = ~S                                                           # the pool's trainval imitation rows (mode both): speed is the human's
    Ct = classify(zx["fut"][tvm][..., :2], zx["speed"][tvm], zx["fut"][tvm][..., 2])
    out["pool_tv_rows_no_lead"] = counts(Ct, None, zx["log"].astype(str)[tvm])
    return out


# ---------------------------------------------------------------- PAI 281 (rf_tax extract of the A0 read: the human's log, traffic at 0.5 s)
def pai():
    d = pickle.load(open(DATA / "runs/route_follow/pai_input_loop/tax/A0.pkl", "rb"))["rolls"]
    Fs, vs, Ys, nl, unit, miss = [], [], [], [], [], 0
    for o in d:
        if "log" not in o:
            miss += 1
            continue
        T, P, Yl = np.asarray(o["log"]["T"], np.float64) * 1e-6, np.asarray(o["log"]["P"], np.float64), np.unwrap(np.asarray(o["log"]["Y"], np.float64))
        for i, tg in enumerate(np.asarray(o["tg"], np.float64) * 1e-6):
            if tg - 0.25 < T[0] or tg + 4.0 > T[-1]:
                continue
            tt = tg + 0.5 * np.arange(0, 9)
            Q = np.stack([np.interp(tt, T, P[:, 0]), np.interp(tt, T, P[:, 1])], -1)
            y0 = np.interp(tg, T, Yl)
            F = wc.to_ego(Q[1:], Q[0], y0)
            dx = [np.interp(tg + 0.25, T, P[:, k]) - np.interp(tg - 0.25, T, P[:, k]) for k in (0, 1)]
            Fs.append(F), vs.append(np.hypot(*dx) / 0.5), Ys.append(np.interp(tt[1:], T, Yl) - y0), unit.append(o["scene"])
            tr = o["traffic"][i]
            veh = np.array([not any(w in c.lower() for w in ("pedestrian", "protruding", "animal", "cone")) for c in np.asarray(tr["cls"]).astype(str)], bool)
            if not veh.any():
                nl.append(True)
                continue
            ahead = (T >= tg) & (T <= tg + 30.0)
            path = wc.to_ego(np.r_[Q[:1], P[ahead]], Q[0], y0)
            B = np.c_[wc.to_ego(np.c_[np.asarray(tr["x"])[veh], np.asarray(tr["y"])[veh]], Q[0], y0), np.zeros(veh.sum()),
                      np.asarray(tr["L"])[veh], np.asarray(tr["W"])[veh]]
            nl.append(not (lead_gap(path, B) <= LEAD))
    C = classify(np.array(Fs), np.array(vs), np.array(Ys))
    return counts(C, np.array(nl), np.array(unit), dict(scenes_without_log=miss, rows_def="rollout decision times tg (0.5 s) with 4 s of log ahead"))


# ---------------------------------------------------------------- HUGSIM scenes (front-camera track of meta_data.json)
FRONT_CAM = {"nuscenes": "CAM_FRONT", "kitti360": "cam_0", "waymo": "cam_1", "pandaset": "front_camera"}


def _hug(f):
    ds = f.split("/scenes/")[1].split("/")[0]
    fr = [x for x in json.load(open(f))["frames"] if x["rgb_path"].split("/")[-2] == FRONT_CAM[ds]]
    T = np.array([x["timestamp"] for x in fr], np.float64)
    if np.allclose(np.diff(T), 1.0):
        T = T * 0.1                                                   # frame indices at 10 Hz
    M = np.array([x["camtoworld"] for x in fr], np.float64)
    o = np.argsort(T)
    T, M = T[o], M[o]
    up = -M[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    e1 = M[0, :3, 2] - up * (M[0, :3, 2] @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    P = np.c_[M[:, :3, 3] @ e1, M[:, :3, 3] @ e2]
    Yw = np.unwrap(np.arctan2(M[:, :3, 2] @ e2, M[:, :3, 2] @ e1))
    out = []
    for t0 in np.arange(T[0] + 0.25, T[-1] - 4.0 + 1e-6, 0.5):
        tt = t0 + 0.5 * np.arange(9)
        Q = np.stack([np.interp(tt, T, P[:, 0]), np.interp(tt, T, P[:, 1])], -1)
        y0 = np.interp(t0, T, Yw)
        v = np.hypot(*(np.array([np.interp(t0 + 0.25, T, P[:, k]) - np.interp(t0 - 0.25, T, P[:, k]) for k in (0, 1)]))) / 0.5
        out.append((wc.to_ego(Q[1:], Q[0], y0), v, np.interp(tt[1:], T, Yw) - y0, f.split("/scenes/")[1].rsplit("/", 1)[0]))
    return out


def hugsim():
    fs = sorted(glob.glob(str(DATA / "datasets/hugsim/scenes/*/*/meta_data.json")))
    with ProcessPoolExecutor(16) as ex:
        R = [r for o in ex.map(_hug, fs) for r in o]
    C = classify(np.array([r[0] for r in R]), np.array([r[1] for r in R]), np.array([r[2] for r in R]))
    return counts(C, None, np.array([r[3] for r in R]), dict(scenes=len(fs), lead="none (no boxes read)"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(defs=dict(T=f"|dyaw| >= {TURN} deg within {REACH} m of travel and the 4 s future, human v0 >= {V_MOVE}; right = dyaw < 0; "
                         f"slow6 = segment speed at mid-turn <= {V_TURN}", S=f"v0 >= {V_STOP_IN}, two 0.5 s segments < {V_STOP} m/s from <= 3 s; "
                         f"no lead = no vehicle in the corridor within {LEAD} m at t0", sampling="2 Hz rows; units = logs (nuPlan) / scenes"))
    for k, f in (("nuplan_trainval", nuplan_tv), ("navtest", navtest), ("pools", pools), ("pai281", pai), ("hugsim", hugsim)):
        try:
            res[k] = f()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[k] = dict(error=repr(e))
        print(k, json.dumps(res[k], indent=1), flush=True)
    (OUT / "data_counts.json").write_text(json.dumps(res, indent=1))
    (OUT / "DONE").write_text("ok\n")
