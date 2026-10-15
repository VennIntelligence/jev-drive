#!/usr/bin/env python3
"""APPROACH COURSE rows and labels (experiments/route_follow/plans/2026-10-15-approach-course-prereg.md). Box, envs/op-train (numpy only).

  rf_apppool.py label --src nup [--pool L18RSG] [--workers N]   nuPlan on-policy rows (RSG-L18-s0's own train rollouts, pool on1_L18RSG)
  rf_apppool.py label --src pai [--pool PAIonp]                 PAI train160 rows (pool on1_PAIonp of the RSG-PAI line, RB0-F-s0's states)
      -> $DATA_DIR/runs/route_follow/approach/lab_<src>.npz: one entry per pool row (all rows), class turn / neg / "" and the drop reason
  rf_apppool.py pack [--base L18RB0] [--out APP] [--k-app 7] [--k-s 13]
      the train-log turn + negative rows of both sources -> subset pools on1_APPN / on1_APPP (cache, @warp, feat), then rf_pait merge
      on1_<base>:s:self on1_APPN:a:self on1_APPP:p:self -> on1_<out> (+ approach.npz, + tab `app` turn / neg for --on-weight), and the
      dev-log rows -> $DATA_DIR/runs/route_follow/approach/dev.npz (the gate's held-out rows); counts -> experiments/route_follow/results/
      approach/rows.{json,md}; prints the --on-weight that puts k_app of every k_s + k_app pool draws on the approach rows.

Rows (prereg section 1). Turn = rf_pai_follow.find_turn on the HUMAN's logged path (30 m window, net heading change >= 45 deg, change over
[a - 10, a + 40] m >= 0.75 x 45 deg, entry / exit at 10 / 90 % of the change; rf_shape / rf_entryspeed / decision 293's rule with the
threshold at 45 deg); a turn row = the ego's foot on the human path is 0 < d_e <= 40 m before the first turn entry ahead. Straight
negative = no turn entry in (-10, +80] m of the foot, the human path's heading within 10 deg of the foot's over the next 60 m, and the
route (route.npz, the lane centreline the policy is given) equally straight. Both: ego speed >= 1 m/s (P4's G-crawl), |ego offset to the
human path| <= 3 m, the row's route label ok and the human's entry point (turn rows) <= 2.5 m from the route (the route takes the
human's turn); nuPlan: the pool's kept rows (on_rows excl ""); PAI: rows before any zero event (cache after_event False).
Labels (on_approach.py's approach.npz): turn rows H = the human path at arcs -10 .. +80 m (1 m) from the ego's foot, Hh its heading, Hv the
human's speed when it passed each point (first passage), d_e = entry arc; l3 = the human's speed >= 2 m/s over the whole 26 m before the
entry (rf_entryspeed VH_MIN; a human stopped there for traffic gives no speed target). Negatives: H = the route centreline, Hv NaN, d_e = a
virtual entry drawn from the turn rows' d_e (rng [0, crc32(name)]), l3 False. Entry state of the human relative to the route (reported):
signed lateral offset of the human's entry point to the route (+ toward the turn's inside), heading of the human path minus the route's,
speed. Dev split: logs with crc32(log) % 100 < 15 (nuPlan log unit; PAI scene) are held out of the pool for the pilot gate.
nuPlan check: the human path G is rebuilt from the navsim log exactly as on_rows builds it (frame a = log index of sim time 0 - 10) and the
ego's foot on it must reproduce meta s0 / e0 (max |diff| reported; label refuses > 1e-6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/route_follow/scripts"), str(R / "experiments/onpol_nuplan/scripts"), str(R / "experiments/onpol_nuplan/lib")]
D = Path(os.environ.get("DATA_DIR", "/data"))
CR, FEAT, OUT = D / "runs/op_parity/cache", D / "runs/route_mem/feat", D / "runs/route_follow/approach"
RES = R / "experiments/route_follow/results/approach"
TRAIN160 = R / "experiments/onpol_pai/results/split/train160.tsv"
EXAM = R / "experiments/alpasim/results/pai_exam"
GRID = np.arange(-10.0, 81.0)                                   # on_approach.M points
PRE, TURN_DEG, STRAIGHT_DEG, STRAIGHT_LEN, NEG_CLEAR = 40.0, 45.0, 10.0, 60.0, 80.0
V_MIN, E_MAX, ROUTE_TOL, VH_MIN, V_WIN, DEV_PCT = 1.0, 3.0, 2.5, 2.0, 26.0, 15


def is_dev(log) -> bool:
    return zlib.crc32(str(log).encode()) % 100 < DEV_PCT


def wrapd(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def poly_grid(P, offs):
    """Polyline P (m, 2) whose points sit at arc offsets offs (m,) -> points / heading on GRID (linear in arc; NaN outside the valid span)."""
    ok = np.isfinite(P).all(1)
    out, h = np.full((len(GRID), 2), np.nan), np.full(len(GRID), np.nan)
    if ok.sum() < 2:
        return out, h
    o, Q = offs[ok], P[ok]
    m = (GRID >= o[0]) & (GRID <= o[-1])
    out[m] = np.c_[np.interp(GRID[m], o, Q[:, 0]), np.interp(GRID[m], o, Q[:, 1])]
    d = np.gradient(out, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    h[~np.isfinite(out).all(1)] = np.nan
    return out, h


def route_rel(poly, q, hq):
    """Route polyline (row frame, NaN padded), point q, heading hq -> (signed lateral of q, left +; heading of q minus the route's), nearest segment."""
    import rf_pai_follow as PF
    ok = np.isfinite(poly).all(1)
    if ok.sum() < 2 or not np.isfinite(q).all():
        return np.nan, np.nan, np.nan
    P = PF.dedup(poly[ok].astype(np.float64))
    S = PF.stations(P)
    s, lat = PF.project(P, S, q)
    i = int(np.clip(np.searchsorted(S, s) - 1, 0, len(P) - 2))
    hr = np.arctan2(*(P[i + 1] - P[i])[::-1])
    return lat, float(wrapd(hq - hr)), float(np.hypot(*(q - (P[i] + (s - S[i]) / max(S[i + 1] - S[i], 1e-9) * (P[i + 1] - P[i])))))


def straight_route(poly):
    """The route label (51 points every 2 m from -10 m) turns < STRAIGHT_DEG over 0 .. 60 m (chord headings)."""
    P = poly[5:36]
    if not np.isfinite(P).all():
        return False
    d = np.diff(P, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return bool(np.degrees(np.abs(h - h[0]).max()) < STRAIGHT_DEG)


def classify(s, h, se, s_end):
    """Human path stations s / unwrapped heading h (1 m grid), the ego's foot se, valid end s_end -> (cls, d_e, side, TH deg)."""
    import rf_pai_follow as PF
    TT = PF.all_turns(s, h, 0.0, s_end, deg=TURN_DEG)
    ahead = [(e, x, th) for e, x, th in TT if 0 < e - se <= PRE]
    if ahead:
        e, x, th = ahead[0]
        return "turn", e - se, int(np.sign(th)), float(np.degrees(th))
    if any(-10 < e - se <= NEG_CLEAR for e, _, _ in TT) or se + STRAIGHT_LEN > s_end:
        return "", np.nan, 0, np.nan
    i0, i1 = np.searchsorted(s, [se, se + STRAIGHT_LEN])
    if np.degrees(np.abs(h[i0:i1 + 1] - h[i0]).max()) >= STRAIGHT_DEG:
        return "", np.nan, 0, np.nan
    return "neg", np.nan, 0, np.nan


def first_passage_speed(st, vt, q):
    """Log stations st (nondecreasing over time), speeds vt -> the speed when the log first reached each station q."""
    u, i = np.unique(st, return_index=True)
    t = np.interp(q, u, i.astype(float), left=np.nan, right=np.nan)
    ok = np.isfinite(t)
    out = np.full(len(q), np.nan)
    out[ok] = np.interp(t[ok], np.arange(len(vt)), vt)
    return out


# ---------------------------------------------------------------- nuPlan rows
def nup_scene(job):
    """One scene's pool rows -> list of per-row label dicts (indices into the pool)."""
    import on_rows as ON
    import rf_pai_follow as PF
    idx, M, RT = job
    log = str(M["log"][idx[0]])
    L = ON._log(log)
    T0s = M["t_log_us"][idx] - M["t_us"][idx]
    assert (T0s == T0s[0]).all()
    a = max(int(np.searchsorted(L["T"], T0s[0])) - 10, 0)
    G = ON.Path(L["P"][a:], L["T"][a:])
    s, h = PF.heading(G.xy, G.s)
    tt = np.arange(G.TF[0], G.TF[-1], 1e5)
    st = np.maximum.accumulate(G.s_log(tt))
    vt = np.gradient(st) / 0.1
    out = []
    for i in idx:
        pm = M["pose_map"][i]
        sl = float(G.s_log(M["t_log_us"][i]))
        s0, e0, _ = G.project(pm[:2], sl - 60, sl + 80)
        r = dict(i=int(i), s0_err=abs(s0 - M["s0"][i]), e0_err=abs(e0 - M["e0"][i]), e0=e0, v=float(M["speed"][i]), cls="", why="")
        cls, de, side, th = classify(s, h, s0, G.s_end)
        r |= dict(d_e=de, side=side, th=th)
        if not cls:
            r["why"] = "no_class"
            out.append(r)
            continue
        poly = RT["poly"][i].astype(np.float64)
        if not RT["ok"][i]:
            r["why"] = "route_ok"
        elif r["v"] < V_MIN:
            r["why"] = "crawl"
        elif abs(e0) > E_MAX:
            r["why"] = "far"
        elif cls == "neg" and not straight_route(poly):
            r["why"] = "route_bends"
        if r["why"]:
            out.append(r)
            continue
        q = s0 + GRID
        valid = q <= G.s_end
        P, hd = G.at(q)
        Hr = ON.into(pm, np.c_[P, hd])
        H = np.where(valid[:, None], Hr[:, :2], np.nan)
        Hh = np.where(valid, wrapd(Hr[:, 2]), np.nan)
        Hv = np.where(valid, first_passage_speed(st, vt, q), np.nan)
        if cls == "turn":
            je = int(round(de)) + 10
            if je >= len(GRID) or not np.isfinite(H[je]).all():
                r["why"] = "path_short"
                out.append(r)
                continue
            ve = float(np.interp(de, GRID, Hv))
            lat, dh, dist = route_rel(poly, np.array([np.interp(de, GRID, H[:, 0]), np.interp(de, GRID, H[:, 1])]), float(np.interp(de, GRID, Hh)))
            if not np.isfinite(dist) or dist > ROUTE_TOL:
                r["why"] = "route_mismatch"
                out.append(r)
                continue
            w = (GRID >= de - V_WIN) & (GRID <= de)
            r |= dict(H=H, Hh=Hh, Hv=Hv, l3=bool(np.nanmin(Hv[w]) >= VH_MIN) if np.isfinite(Hv[w]).any() else False, v_h_entry=ve,
                      lat_h=lat * side, dh_h=float(np.degrees(dh)) * side)
        else:
            Hn, Hhn = poly_grid(poly, np.arange(51) * 2.0 - 10.0)
            r |= dict(H=Hn, Hh=Hhn, Hv=np.full(len(GRID), np.nan), l3=False)
        r["cls"] = cls
        out.append(r)
    return out


def label_nup(a):
    from multiprocessing import Pool
    from jevdrive.common import n_cpus
    M = dict(np.load(D / "runs/onpol_nuplan/rows" / a.pool / "meta.npz", allow_pickle=True))
    RT = dict(np.load(CR / f"on1_{a.pool}" / "route.npz"))
    tb = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    assert np.array_equal(M["names"], tb["names"]) and np.array_equal(RT["tokens"], tb["names"])
    Mx = {k: M[k] for k in ("log", "t_log_us", "t_us", "pose_map", "s0", "e0", "speed")}
    jobs = [(np.flatnonzero(M["scene"] == sc), Mx, RT) for sc in np.unique(M["scene"])]
    with Pool(a.workers or n_cpus()) as p:
        res = [r for rr in p.imap_unordered(nup_scene, jobs, chunksize=4) for r in rr]
    save(a, res, tb, M["scene"], M["logu"], "nup")


# ---------------------------------------------------------------- PAI rows
def label_pai(a):
    import rf_pai_follow as PF
    tb = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    RT = dict(np.load(CR / f"on1_{a.pool}" / "route.npz"))
    assert np.array_equal(RT["tokens"], tb["names"])
    Cz = np.load(a.cache)
    C = {k: Cz[k] for k in ("scenes", "sid", "k", "ego_in", "e0", "v0", "path", "path_ok", "path_h", "path_v", "after_event")}   # npz members load per access
    names = tb["names"]
    key = {(sc, int(k)): i for i, (sc, k) in enumerate(zip(C["scenes"][C["sid"]].tolist(), C["k"].tolist()))}
    ci = np.array([key[(n.split(":")[2], int(n.split(":")[3]))] for n in names.tolist()])
    assert np.array_equal(C["ego_in"][ci].astype(np.float32), tb["ego"]), "pool ego != cache ego_in"
    res = []
    offs = np.arange(121) - 20.0
    for i, c in enumerate(ci):
        r = dict(i=i, s0_err=0.0, e0_err=0.0, e0=float(C["e0"][c]), v=float(C["v0"][c]), cls="", why="")
        P, ok = C["path"][c].astype(np.float64), C["path_ok"][c]
        idx = np.flatnonzero(ok)
        if len(idx) < 40 or not ok[20] or C["after_event"][c]:
            r["why"] = "no_path" if not C["after_event"][c] else "after_event"
            res.append(r)
            continue
        X = PF.dedup(P[idx])
        S = PF.stations(X)
        se = PF.project(X, S, P[20])[0]
        s, h = PF.heading(X, S)
        cls, de, side, th = classify(s, h, se, S[-1])
        r |= dict(d_e=de, side=side, th=th)
        poly = RT["poly"][i].astype(np.float64)
        if not cls:
            r["why"] = "no_class"
        elif not RT["ok"][i]:
            r["why"] = "route_ok"
        elif r["v"] < V_MIN:
            r["why"] = "crawl"
        elif abs(r["e0"]) > E_MAX:
            r["why"] = "far"
        elif cls == "neg" and not straight_route(poly):
            r["why"] = "route_bends"
        if r["why"]:
            res.append(r)
            continue
        Pm = np.where(ok[:, None], P, np.nan)
        H, _ = poly_grid(Pm, offs)
        Hh = np.where(np.isfinite(H).all(1), np.interp(GRID, offs, np.where(ok, C["path_h"][c], np.nan)), np.nan)
        Hv = np.where(np.isfinite(H).all(1), np.interp(GRID, offs, np.where(ok, C["path_v"][c], np.nan)), np.nan)
        if cls == "turn":
            je = int(round(de)) + 10
            if je >= len(GRID) or not np.isfinite(H[je]).all():
                r["why"] = "path_short"
                res.append(r)
                continue
            lat, dh, dist = route_rel(poly, np.array([np.interp(de, GRID, H[:, 0]), np.interp(de, GRID, H[:, 1])]), float(np.interp(de, GRID, Hh)))
            if not np.isfinite(dist) or dist > ROUTE_TOL:
                r["why"] = "route_mismatch"
                res.append(r)
                continue
            w = (GRID >= de - V_WIN) & (GRID <= de)
            r |= dict(H=H, Hh=Hh, Hv=Hv, l3=bool(np.nanmin(Hv[w]) >= VH_MIN) if np.isfinite(Hv[w]).any() else False,
                      v_h_entry=float(np.interp(de, GRID, Hv)), lat_h=lat * side, dh_h=float(np.degrees(dh)) * side)
        else:
            Hn, Hhn = poly_grid(poly, np.arange(51) * 2.0 - 10.0)
            r |= dict(H=Hn, Hh=Hhn, Hv=np.full(len(GRID), np.nan), l3=False)
        r["cls"] = cls
        res.append(r)
    sc = np.array([n.split(":")[2] for n in names.tolist()])
    save(a, res, tb, sc, sc, "pai")


def save(a, res, tb, scene, logu, src):
    res = sorted(res, key=lambda r: r["i"])
    n = len(tb["names"])
    assert [r["i"] for r in res] == list(range(n))
    g = lambda k, d=np.nan: np.array([r.get(k, d) for r in res])  # noqa: E731
    H = np.full((n, len(GRID), 2), np.nan, np.float32)
    Hh, Hv = np.full((n, len(GRID)), np.nan, np.float32), np.full((n, len(GRID)), np.nan, np.float32)
    for r in res:
        if "H" in r:
            H[r["i"]], Hh[r["i"]], Hv[r["i"]] = r["H"], r["Hh"], r["Hv"]
    cls = g("cls", "").astype("U4")
    de = g("d_e").astype(np.float32)
    turn = cls == "turn"
    neg = np.flatnonzero(cls == "neg")                         # virtual entry drawn from the turn rows' d_e
    pool_de = np.sort(de[turn])
    for j in neg:
        de[j] = pool_de[np.random.default_rng([0, zlib.crc32(str(tb["names"][j]).encode())]).integers(len(pool_de))] if len(pool_de) else 20.0
    dev = np.array([is_dev(x) for x in logu])
    s0e, e0e = float(np.nanmax(g("s0_err"))), float(np.nanmax(g("e0_err")))
    assert s0e <= 1e-6 and e0e <= 1e-6, f"the rebuilt human path does not reproduce meta s0 / e0: {s0e}, {e0e}"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"lab_{src}.npz", names=tb["names"], scene=scene, log=logu, dev=dev, cls=cls, why=g("why", "").astype("U16"), d_e=de,
             side=g("side", 0).astype(np.int8), th=g("th").astype(np.float32), e0=g("e0").astype(np.float32), v=g("v").astype(np.float32),
             l3=g("l3", False).astype(bool), v_h_entry=g("v_h_entry").astype(np.float32), lat_h=g("lat_h").astype(np.float32),
             dh_h=g("dh_h").astype(np.float32), H=H, Hh=Hh, Hv=Hv)
    rep = dict(src=src, pool=a.pool, rows=n, s0_err_max=s0e, e0_err_max=e0e, cls={c: int((cls == c).sum()) for c in ("turn", "neg")},
               why={w: int((g("why", "") == w).sum()) for w in np.unique(g("why", "")) if w}, turn_train=int((turn & ~dev).sum()),
               turn_dev=int((turn & dev).sum()), neg_train=int(((cls == "neg") & ~dev).sum()), turn_scenes=int(len(np.unique(scene[turn]))),
               turn_l3=int((turn & g("l3", False).astype(bool)).sum()))
    print(json.dumps(rep), flush=True)
    (OUT / f"lab_{src}.json").write_text(json.dumps(rep, indent=1))


# ---------------------------------------------------------------- pack
def subset(src_pool, keep, name, extra):
    """Rows `keep` of cache on1_<src_pool> -> on1_<name> (tab + extra keys, agent / road / drivable / route, @warp front + teacher, feat)."""
    src, dst, dw = CR / f"on1_{src_pool}", CR / f"on1_{name}", CR / f"on1_{name}@warp"
    dst.mkdir(parents=True, exist_ok=True), dw.mkdir(parents=True, exist_ok=True)
    tb = dict(np.load(src / "tab.npz"))
    n0 = len(tb["names"])
    T = {k: (v[keep] if v.ndim and len(v) == n0 else v) for k, v in tb.items()} | extra
    np.savez(dst / "tab.npz", **T)
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        if not (src / f).exists():
            continue
        z = dict(np.load(src / f))
        assert np.array_equal(z["tokens"], tb["names"]), f"{src_pool}/{f}: tokens"
        np.savez(dst / f, **{k: (v[keep] if v.ndim and len(v) == n0 else v) for k, v in z.items()})
    fz = np.load(FEAT / f"on1_{src_pool}.npz")
    assert np.array_equal(fz["names"], tb["names"])
    np.savez(FEAT / f"on1_{name}.npz", names=T["names"], **{k: fz[k][keep] for k in ("f", "k", "ok")})
    th = np.load(CR / f"on1_{src_pool}@warp/teacher.npz")
    np.savez(dw / "teacher.npz", out=th["out"][keep], plan=th["plan"][keep], di=th["di"], pi=th["pi"])
    fr = np.load(CR / f"on1_{src_pool}@warp/front.npy", mmap_mode="r")
    o = np.lib.format.open_memmap(dw / "front.npy", "w+", fr.dtype, (len(keep),) + fr.shape[1:])
    for i in range(0, len(keep), 2048):
        o[i:i + 2048] = fr[keep[i:i + 2048]]
    o.flush()
    rs = np.random.default_rng(0).choice(len(keep), min(len(keep), 128), replace=False)
    assert all(np.array_equal(o[i], fr[keep[i]]) for i in rs)


def cmd_pack(a):
    import subprocess
    srcs = [("nup", "L18RSG", "APPN", "a"), ("pai", "PAIonp", "APPP", "p")]
    L = {s: dict(np.load(OUT / f"lab_{s}.npz")) for s, _, _, _ in srcs}
    tr160 = {l.split()[0] for l in TRAIN160.read_text().splitlines() if l.startswith("clipgt-")}
    bad = set().union(*({l.split()[0] for l in (EXAM / f).read_text().splitlines() if l.startswith("clipgt-")} for f in ("hard_exam.txt", "control.txt", "hard_train.txt")))
    assert not (set(L["pai"]["scene"]) & bad) and set(L["pai"]["scene"]) <= tr160, "PAI rows outside train160"
    rec, dev = dict(sources={}), {}
    lab_rows, app_keys = [], []
    for s, pool, name, pre in srcs:
        Z = L[s]
        use = np.isin(Z["cls"], ["turn", "neg"])
        tr_t = np.flatnonzero(use & ~Z["dev"] & (Z["cls"] == "turn"))
        tr_n_all = np.flatnonzero(use & ~Z["dev"] & (Z["cls"] == "neg"))
        n_neg = int(round(len(tr_t) * a.neg_share / (1 - a.neg_share)))
        tr_n = np.sort(np.random.default_rng(1).choice(tr_n_all, min(n_neg, len(tr_n_all)), replace=False))
        keep = np.r_[tr_t, tr_n]
        keep.sort()
        subset(pool, keep, name, dict(app=Z["cls"][keep].astype("U4")))
        lab_rows.append((pre, Z, keep))
        dv_t = np.flatnonzero(use & Z["dev"] & (Z["cls"] == "turn"))
        dv_n_all = np.flatnonzero(use & Z["dev"] & (Z["cls"] == "neg"))
        dv_n = np.sort(np.random.default_rng(2).choice(dv_n_all, min(len(dv_n_all), max(len(dv_t), 64)), replace=False))
        dev[s] = np.r_[dv_t, dv_n]
        sd = lambda m: dict(rows=int(len(m)), scenes=int(len(np.unique(Z["scene"][m]))), logs=int(len(np.unique(Z["log"][m]))))  # noqa: E731
        q = lambda x: [float(np.nanpercentile(x, p)) for p in (25, 50, 75)] if np.isfinite(x).any() else []  # noqa: E731
        rec["sources"][s] = dict(pool=f"on1_{pool}", rows=int(len(Z["names"])), turn_train=sd(tr_t), neg_train=sd(tr_n), neg_candidates=int(len(tr_n_all)),
                                 turn_dev=sd(dv_t), neg_dev=sd(dv_n), d_e_bins={b: int(((Z["d_e"][tr_t] > lo) & (Z["d_e"][tr_t] <= hi)).sum())
                                                                               for b, lo, hi in (("40-20", 20, 40), ("20-0", 0, 20))},
                                 l3_train=int(Z["l3"][tr_t].sum()), left=int((Z["side"][tr_t] > 0).sum()), right=int((Z["side"][tr_t] < 0).sum()),
                                 human_entry=dict(v_q25_50_75=q(Z["v_h_entry"][tr_t]), lat_inside_q25_50_75=q(Z["lat_h"][tr_t]),
                                                  dh_inside_deg_q25_50_75=q(Z["dh_h"][tr_t])),
                                 ego=dict(v_q25_50_75=q(Z["v"][tr_t]), abs_e0_q25_50_75=q(np.abs(Z["e0"][tr_t]))),
                                 drop={w: int((Z["why"] == w).sum()) for w in np.unique(Z["why"]) if w})
    n_turn = sum(v["turn_train"]["rows"] for v in rec["sources"].values())
    rec["turn_train_total"], rec["supplement_needed"] = n_turn, n_turn < 1500
    p = subprocess.run([sys.executable, str(R / "experiments/route_follow/scripts/rf_pait.py"), "merge", "--pools", f"on1_{a.base}:s:self", "on1_APPN:a:self",
                        "on1_APPP:p:self", "--out", a.out], check=True, capture_output=True, text=True)
    print(p.stdout[-2000:])
    T = dict(np.load(CR / f"on1_{a.out}" / "tab.npz"))
    names = T["names"]
    pos = {str(t): i for i, t in enumerate(names.tolist())}
    N = len(names)
    G = len(GRID)
    A = dict(tokens=names, ok=np.zeros(N, bool), kind=np.zeros(N, np.int8), H=np.full((N, G, 2), np.nan, np.float32), Hh=np.full((N, G), np.nan, np.float32),
             Hv=np.full((N, G), np.nan, np.float32), d_e=np.full(N, np.nan, np.float32), l3=np.zeros(N, bool))
    for pre, Z, keep in lab_rows:
        dst = np.array([pos[f"on:{pre}:{str(Z['names'][i])[3:]}"] for i in keep])
        A["ok"][dst], A["kind"][dst] = True, (Z["cls"][keep] == "turn").astype(np.int8)
        for k in ("H", "Hh", "Hv", "d_e", "l3"):
            A[k][dst] = Z[k][keep]
    np.savez(CR / f"on1_{a.out}" / "approach.npz", **A)
    n_s, n_app = int((T["pool"] == "s").sum()), int(A["ok"].sum())
    w = a.k_app / a.k_s * n_s / n_app
    rec |= dict(pool=f"on1_{a.out}", s_rows=n_s, app_rows=n_app, app_turn=int((A["ok"] & (A["kind"] == 1)).sum()), app_neg=int((A["ok"] & (A["kind"] == 0)).sum()),
                k_s=a.k_s, k_app=a.k_app, on_k=a.k_s + a.k_app, on_weight=f"app=turn:{w:.4f},app=neg:{w:.4f}")
    np.savez(OUT / "dev.npz", **{f"{s}_rows": v for s, v in dev.items()})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rows.json").write_text(json.dumps(rec, indent=1))
    md = ["# Approach course rows (rf_apppool.py pack)", "", f"Pool `{rec['pool']}`: {n_s} S rows (on1_{a.base}, RSG-L18's) + {n_app} approach rows "
          f"({rec['app_turn']} turn, {rec['app_neg']} straight negatives); on-k {rec['on_k']}, --on-weight {rec['on_weight']} (~{a.k_app} approach rows per batch of 128).",
          "", "| source | pool | turn rows (train) | scenes / logs | 40-20 / 20-0 m | left / right | speed term | negatives (train) | dev turn rows (logs) | dev negatives |",
          "|:--|:--|--:|:--|:--|:--|--:|--:|:--|--:|"]
    for s, v in rec["sources"].items():
        md.append(f"| {s} | {v['pool']} | {v['turn_train']['rows']} | {v['turn_train']['scenes']} / {v['turn_train']['logs']} | {v['d_e_bins']['40-20']} / "
                  f"{v['d_e_bins']['20-0']} | {v['left']} / {v['right']} | {v['l3_train']} | {v['neg_train']['rows']} | {v['turn_dev']['rows']} "
                  f"({v['turn_dev']['logs']}) | {v['neg_dev']['rows']} |")
    md += ["", f"Turn rows in total (train): {n_turn} (line 1 500: supplement {'NEEDED' if n_turn < 1500 else 'not needed'}).", "",
           "Human entry state (train turn rows, q25 / median / q75): " + "; ".join(
               f"{s}: v {', '.join(f'{x:.1f}' for x in v['human_entry']['v_q25_50_75'])} m/s, lateral to the route (+ inside) "
               f"{', '.join(f'{x:+.2f}' for x in v['human_entry']['lat_inside_q25_50_75'])} m, heading to the route (+ inside) "
               f"{', '.join(f'{x:+.1f}' for x in v['human_entry']['dh_inside_deg_q25_50_75'])} deg" for s, v in rec["sources"].items()), "",
           "Drops: " + "; ".join(f"{s}: {json.dumps(v['drop'])}" for s, v in rec["sources"].items())]
    (RES / "rows.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label")
    p.add_argument("--src", choices=("nup", "pai"), required=True), p.add_argument("--pool", default="")
    p.add_argument("--cache", default=str(D / "runs/route_follow/paionp/src/cache_pol_tab.npz")), p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("pack")
    p.add_argument("--base", default="L18RB0"), p.add_argument("--out", default="APP"), p.add_argument("--neg-share", type=float, default=0.2)
    p.add_argument("--k-app", type=int, default=7), p.add_argument("--k-s", type=int, default=13)
    a = ap.parse_args()
    if a.cmd == "label":
        a.pool = a.pool or {"nup": "L18RSG", "pai": "PAIonp"}[a.src]
        (label_nup if a.src == "nup" else label_pai)(a)
    else:
        cmd_pack(a)
