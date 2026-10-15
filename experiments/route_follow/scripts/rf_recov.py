#!/usr/bin/env python3
"""RECOVERY-ENVELOPE rows, pools and pilot gates (experiments/route_follow/plans/2026-10-16-recovery-envelope-prereg.md). Box; envs/op-train.

  rf_recov.py label --pool L18RSG [--workers N]           CPU: the longitudinal envelope of every row of an on-policy pool (rows dir
        runs/onpol_nuplan/rows/<P>: meta.npz, lon.npz; tab speed; scene context runs/onpol_nuplan/ctx/ctx.npz) -> $O/lab_<P>.npz (on_envelope's
        arrays for every row + cls, dev, v0, scene, k, log) and $O/lab_<P>.json (counts per class, dev / train)
  rf_recov.py pack --arm L|R|RL --out NAME [--env-pools L18RSG[,RCVfast]] [--rec RCVd316] [--base L18RB0]
        CPU: the train-log rows (dev logs never) -> subset pools on1_RCVE<P> / on1_RCVR<rec>, rf_pait.py merge with the base pool's S rows
        (`self`) -> on1_<NAME> (+ envelope.npz for L / RL); prints and writes $O/pack_<NAME>.json: on-k, --on-mode, --on-weight, --on-cap-*
  rf_recov.py fwd --pool P --rows {dev-env|held01|dev-rec} --tags T1,T2.. [--out DIR]   GPU pool job: the tags' plans (rear axle, row frame) on
        the selected rows of pool on1_<P> (rm_train.build's store, rf_rdom's forward) -> DIR/<P>-<rows>/plans_<tag>.npz (rows, poses); a tag
        whose file exists is skipped; the first run also writes gate.json (the first tag's plans against the pool's stored served plans)
  rf_recov.py gate --tag PILOT --arm L|R|RL [--base RSG-L18-s0] [--rec RCVd316]   CPU: the registered pilot gates from the fwd plans + the
        trainer's dev ADE / rely.json -> $O/gate/<tag>.{json,md}

Envelope (prereg "L"; on_envelope.py applies it). Per row, at its own state (fed speed v0 = tab speed; arcs on the row's route grid of the
on_lon pack, 1 m from the route origin, the same arcs as the scene context):
  stop  first LONG-PROBE-eligible line ahead of the front bumper (0 .. 60 m) that must be stopped at now: a light line red at the keyframe
        nearest the row's log time (<= 0.3 s) and (the human stopped there or every keyframe of [t, t + 4 s] red), or a stop-sign line
        the human stopped at (ln_tstop finite). vuf(s) = sqrt(2 A_DEC (s_stop - s)) before s_stop = line - 1.0 m, 0 from it on.
  turn  the first strict turn (rf_follow.find_turn, >= 60 deg; rf_loninv.strict_turns on the context route) with its entry 0 .. 40 m ahead of
        the rear axle: vur(s) = the expert's profile (rf_loninv.proxy_profile = rule_expert.Scene.profile(V_E 4.6)) + 1.0 on [entry - 26, exit].
  lead  rule_expert's yield on the A-eligible leads of the on_lon pack (vehicles within 1.5 m of the route, ahead, aligned or slow; 4
        nearest): the expert's 1-D speed run from (front arc, v0) on its profile (A_ACC 1.5 / A_MAX 6 at 0.1 s, the A_DEC braking envelope)
        for YIELD_T 3 s; when its front reaches sigma = the leads' least corner arc over [0, 3 s], stop = sigma - STOP_M 1.5 and
        vuf(s) = sqrt(2 A_DEC (stop - s)) + 1.0 before it, 0 from it on.
  feasibility  every upper bound >= sqrt(max(v0^2 - 2 A_MAX (s - s_ego), 0)) (the fastest stop the expert's controller allows is not charged).
  free  rf_lonol's free cell (no A-eligible lead <= 40 m, no stop line of any type in (-0.5, 60] m, route heading change over 60 m < 30 deg,
        v0 >= 2, the human's speed at the station known): vl(s) = 0.85 min(v_ref(s), sqrt(v0^2 + 2 A_ACC (s - s0))) on [s0, s0 + 60], v_ref =
        the expert's free-road profile (proxy profile).
  cls   stop > turn > lead > free (the first that applies; a row may carry several bounds). Only rows on the route (on_lon ok, |l0| <= 3.5 m).
Dev: crc32(nuPlan log file) % 100 < 15 (rf_apppool.is_dev); dev rows never enter a pool.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "experiments/onpol_nuplan/lib"), str(R_ / "experiments/onpol_nuplan/scripts"),
                str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/long_contact/lib"), str(R_ / "experiments/body1/lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, ROWS, FEAT = D / "runs/op_parity/cache", D / "runs/onpol_nuplan/rows", D / "runs/route_mem/feat"
O = D / "runs/route_follow/recovery_envelope"
RES = R_ / "experiments/route_follow/results/recovery_envelope"
BIG, FRONT, LAT = 1e3, 4.049, 3.5
A_DEC, A_MAX, A_ACC, STOP_M, YIELD_T, M_LINE = 3.0, 6.0, 1.5, 1.5, 3.0, 1.0
D_STOP, D_TURN, DECEL_D, MARGIN_UP, LOW_FRAC, D_FREE = 60.0, 40.0, 26.0, 1.0, 0.85, 60.0
CLASSES = ("stop", "turn", "lead", "free")
DEV_PCT = 15


def is_dev(log) -> bool:
    return zlib.crc32(str(log).encode()) % 100 < DEV_PCT


def braking(stop, g, add=0.0):
    """sqrt(2 A_DEC (stop - g)) + add before `stop`, 0 from it on (per row stop (S,), grid g (NR,)) -> (S, NR); stop = inf: BIG."""
    rem = stop[:, None] - g[None]
    v = np.where(rem > 0, np.sqrt(2 * A_DEC * np.maximum(rem, 0)) + add, 0.0)
    return np.where(np.isfinite(stop)[:, None], v, BIG)


def expert_run(vt, s0, v0, n=int(round(YIELD_T / 0.1))):
    """rule_expert.simulate's speed control in 1-D: vt (NR,) allowed speed on the arc grid, from arc s0 at v0 -> arcs (n + 1,)."""
    s, v, out = float(s0), float(v0), [float(s0)]
    for _ in range(n):
        vn = max(0.0, v + float(np.clip((float(np.interp(s, np.arange(len(vt)), vt)) - v) / 0.1, -A_MAX, A_ACC)) * 0.1)
        s += 0.5 * (v + vn) * 0.1
        v = vn
        out.append(s)
    return np.array(out)


def reach_env(vt):
    """rule_expert.envelope without a stop: the A_DEC braking envelope backwards over the 1 m grid."""
    vt = vt.copy()
    for i in range(len(vt) - 2, -1, -1):
        vt[i] = min(vt[i], np.sqrt(vt[i + 1] ** 2 + 2 * A_DEC))
    return vt


def label_scene(job):
    """One scene's rows -> dict of per-row arrays."""
    import rf_loninv as LI
    idx, Z, X, v0, tlog = job
    S = len(idx)
    NR = Z["route"].shape[1]
    g = np.arange(NR, dtype=np.float64)
    s0, sf0, l0 = Z["s0"].astype(np.float64), Z["sf0"].astype(np.float64), Z["l0"].astype(np.float64)
    onr = Z["ok"].astype(bool) & (np.abs(l0) <= LAT)
    out = dict(vuf=np.full((S, NR), BIG, np.float32), vur=np.full((S, NR), BIG, np.float32), vl=np.zeros((S, NR), np.float32))
    flags = {c: np.zeros(S, bool) for c in CLASSES}
    info = dict(d_stop=np.full(S, np.nan), d_entry=np.full(S, np.nan), d_lead=np.full(S, np.nan), v_ref=np.full(S, np.nan))
    # scene context
    Rg, vlog = X["route"].astype(np.float64), X["vlog"].astype(np.float64)
    vprox = LI.proxy_profile(Rg, vlog)
    gx = np.arange(len(vprox), dtype=np.float64)
    vp = np.interp(g, gx, vprox)                                                # the expert's profile on the pack grid
    vreach = reach_env(vp)
    turns = LI.strict_turns(Rg, float(X["hlen"]))
    # ---- stop lines
    ty, ld, st, tstop, elig, kt = X["ln_type"], X["ln_d"].astype(np.float64), X["ln_st"], X["ln_tstop"], X["ln_elig"], X["kf_ts"]
    dF = ld[None] - sf0[:, None]                                                # (S, L) front bumper to each line
    dist = np.where(kt[None] > 0, np.abs(kt[None] - tlog[:, None]), np.int64(1) << 60)
    near = dist.argmin(1)
    known = dist[np.arange(S), near] <= 3e5
    st_now = np.where(known[:, None], st[:, near].T, -1)                        # (S, L)
    win = (kt[None] >= tlog[:, None] - 3e5) & (kt[None] <= tlog[:, None] + 4e6) & (kt[None] > 0)
    stw = np.where(win[:, None, :], st[None], -1)
    red4 = ((stw == 1) | (stw < 0)).all(2) & (stw == 1).any(2)
    hs = np.isfinite(tstop)[None].repeat(S, 0)
    q = LI.first_ahead((elig & (ty > 0))[None].repeat(S, 0), dF)
    d = LI.take(dF, q)
    tq, sq, hq, rq = LI.take(ty[None].repeat(S, 0), q), LI.take(st_now, q), LI.take(hs, q), LI.take(red4, q)
    stop_ok = onr & (q >= 0) & (d <= D_STOP) & (((tq == 1) & hq) | ((tq == 2) & (sq == 1) & (hq | rq)))
    s_stop = np.where(stop_ok, LI.take(ld[None].repeat(S, 0), q) - M_LINE, np.inf)
    vuf = braking(s_stop, g)
    flags["stop"] = stop_ok
    info["d_stop"] = np.where(stop_ok, d, np.nan)
    # ---- leads (rule_expert's yield on the A-eligible leads)
    a_sig, a_val, a_lam, a_g0 = Z["a_sig"].astype(np.float64), Z["a_val"].astype(bool), Z["a_lam"].astype(np.float64), Z["a_g0"].astype(np.float64)
    use = a_val[:, :, :31] & (np.abs(a_lam[:, :, :31]) <= 1.5) & (a_g0 > 0)[:, :, None]
    sig = np.where(use, a_sig[:, :, :31], np.inf).min((1, 2))
    lead_stop = np.full(S, np.inf)
    for r in np.flatnonzero(onr & np.isfinite(sig)):
        arcs = expert_run(vreach, s0[r], v0[r]) + FRONT                           # the expert's front bumper (rear arc + FRONT along the route)
        if arcs.max() >= sig[r]:
            lead_stop[r] = max(sig[r] - STOP_M, sf0[r])
    flags["lead"] = np.isfinite(lead_stop)
    info["d_lead"] = np.where(flags["lead"], sig - sf0, np.nan)
    vuf = np.minimum(vuf, braking(lead_stop, g, MARGIN_UP))
    # ---- turn entry
    vur = np.full((S, NR), BIG)
    for r in np.flatnonzero(onr):
        e = [(en, ex) for en, ex in turns if 0 <= en - s0[r] <= D_TURN]
        if e:
            en, ex = e[0]
            m = (g >= en - DECEL_D) & (g <= ex)
            vur[r, m] = vp[m] + MARGIN_UP
            flags["turn"][r] = True
            info["d_entry"][r] = en - s0[r]
    # ---- feasibility floor on the upper bounds
    fl_f = np.sqrt(np.maximum(v0[:, None] ** 2 - 2 * A_MAX * (g[None] - sf0[:, None]), 0))
    fl_r = np.sqrt(np.maximum(v0[:, None] ** 2 - 2 * A_MAX * (g[None] - s0[:, None]), 0))
    vuf = np.where(vuf < BIG / 2, np.maximum(vuf, fl_f), BIG)
    vur = np.where(vur < BIG / 2, np.maximum(vur, fl_r), BIG)
    # ---- free road (rf_loninv / rf_lonol's cell) -> lower bound
    lead40 = ((a_g0 > -BIG / 2) & (a_g0 <= 40.0)).any(1)
    line60 = ((ty[None] > 0) & np.isfinite(dF) & (dF > -0.5) & (dF <= D_FREE)).any(1)
    rh = Z["rhead"].astype(np.float64)
    i0 = np.clip(np.round(s0).astype(int), 0, NR - 1)
    ii = np.clip(i0[:, None] + np.arange(61)[None], 0, NR - 1)
    turn60 = LI.wrapd(np.take_along_axis(rh, ii, 1) - rh[np.arange(S), i0][:, None]).max(1)
    vh = np.interp(s0, np.arange(len(vlog)), np.nan_to_num(vlog, nan=-1.0))
    vh = np.where((s0 >= 0) & (s0 <= len(vlog) - 1) & (vh >= 0), vh, np.nan)
    free = onr & ~lead40 & ~line60 & (turn60 < 30.0) & (v0 >= 2.0) & np.isfinite(vh)
    vacc = np.sqrt(v0[:, None] ** 2 + 2 * A_ACC * np.maximum(g[None] - s0[:, None], 0))
    vl = np.where(free[:, None] & (g[None] >= s0[:, None]) & (g[None] <= s0[:, None] + D_FREE), LOW_FRAC * np.minimum(vp[None], vacc), 0.0)
    flags["free"] = free
    info["v_ref"] = np.where(free, np.interp(s0, g, vp), np.nan)
    out.update(vuf=vuf.astype(np.float32), vur=vur.astype(np.float32), vl=vl.astype(np.float32))
    cls = np.full(S, "", "U8")
    for c in CLASSES[::-1]:
        cls[flags[c]] = c
    return dict(idx=idx, cls=cls, flags=flags, info=info, **out)


def cmd_label(a):
    from concurrent.futures import ProcessPoolExecutor
    from jevdrive.run import Run
    with Run("route_follow", f"recov-label-{a.pool}", config=vars(a)) as run:
        rd = ROWS / a.pool
        tab = np.load(CR / f"on1_{a.pool}" / "tab.npz")
        m = np.load(rd / "meta.npz")
        Z = dict(np.load(rd / "lon.npz"))
        X = dict(np.load(D / "runs/onpol_nuplan/ctx/ctx.npz"))
        names = tab["names"]
        assert (m["names"] == names).all() and (Z["names"] == names).all(), "tab / meta / lon rows differ"
        N = len(names)
        scene = m["scene"].astype(str)
        sidx = {s: i for i, s in enumerate(X["scene"].tolist())}
        keys = ("route", "rhead", "ok", "s0", "sf0", "l0", "a_sig", "a_val", "a_lam", "a_g0")
        xk = ("route", "vlog", "hlen", "ln_type", "ln_d", "ln_st", "ln_tstop", "ln_elig", "kf_ts")
        jobs = []
        for sc in sorted(set(scene)):
            idx = np.flatnonzero(scene == sc)
            j = sidx[sc]
            jobs.append((idx, {k: Z[k][idx] for k in keys}, {k: X[k][j] for k in xk}, tab["speed"][idx].astype(np.float64),
                         m["t_log_us"][idx].astype(np.int64)))
        NR = Z["route"].shape[1]
        L = dict(names=names, scene=scene, k=m["k"].astype(int), log=m["log"].astype(str), v0=tab["speed"].astype(np.float32),
                 route=Z["route"].astype(np.float32), vuf=np.full((N, NR), BIG, np.float32), vur=np.full((N, NR), BIG, np.float32),
                 vl=np.zeros((N, NR), np.float32), cls=np.full(N, "", "U8"), **{f"f_{c}": np.zeros(N, bool) for c in CLASSES},
                 **{k: np.full(N, np.nan, np.float32) for k in ("d_stop", "d_entry", "d_lead", "v_ref")})
        with ProcessPoolExecutor(a.workers) as ex:
            for res in run.tqdm(ex.map(label_scene, jobs, chunksize=4), total=len(jobs), desc="scenes"):
                i = res["idx"]
                for k in ("vuf", "vur", "vl", "cls"):
                    L[k][i] = res[k]
                for c in CLASSES:
                    L[f"f_{c}"][i] = res["flags"][c]
                for k, v in res["info"].items():
                    L[k][i] = v
        L["dev"] = np.array([is_dev(x) for x in L["log"]])
        L["ok"] = L["cls"] != ""
        O.mkdir(parents=True, exist_ok=True)
        np.savez(O / f"lab_{a.pool}.npz", **L)
        cnt = lambda s: dict(rows=int(s.sum()), scenes=len(set(scene[s].tolist())), logs=len(set(L["log"][s].tolist())))  # noqa: E731
        rec = dict(pool=a.pool, rows=N, labelled=cnt(L["ok"]),
                   cls={c: dict(train=cnt((L["cls"] == c) & ~L["dev"]), dev=cnt((L["cls"] == c) & L["dev"])) for c in CLASSES},
                   flags={c: cnt(L[f"f_{c}"]) for c in CLASSES},
                   stop_speed_q50_q90_max=[float(np.quantile(L["v0"][L["f_stop"]], q)) for q in (0.5, 0.9, 1.0)] if L["f_stop"].any() else None,
                   stop_ge3=cnt(L["f_stop"] & (L["v0"] >= 3)), stop_ge6=cnt(L["f_stop"] & (L["v0"] >= 6)))
        # the stored served plans against the bounds (what the base already does on its own rows)
        import torch
        import on_envelope as OE
        P = dict(route=torch.as_tensor(L["route"]), vuf=torch.as_tensor(L["vuf"]), vur=torch.as_tensor(L["vur"]), vl=torch.as_tensor(L["vl"]),
                 ok=torch.as_tensor(L["ok"]))
        T = OE.terms(torch.as_tensor(m["poses"].astype(np.float32)), torch.arange(N), P)
        for c in CLASSES:
            s = L["cls"] == c
            rec["cls"][c]["served_envU_mean"] = float(T["envU"][torch.as_tensor(s)].mean()) if s.any() else None
            rec["cls"][c]["served_envL_mean"] = float(T["envL"][torch.as_tensor(s)].mean()) if s.any() else None
            rec["cls"][c]["served_violating"] = float(((T["envU"] + T["envL"])[torch.as_tensor(s)] > 0.05).float().mean()) if s.any() else None
        (O / f"lab_{a.pool}.json").write_text(json.dumps(rec, indent=1))
        run.summary.update(rec)
        print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- pools
def subset(src_pool, keep, name, extra):
    import rf_apppool as AP
    AP.subset(src_pool, keep, name, extra)


def merge(parts, out):
    p = subprocess.run([sys.executable, str(HERE / "rf_pait.py"), "merge", "--pools", *parts, "--out", out], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"rf_pait merge failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    print(p.stdout[-1500:])


def cmd_pack(a):
    env_pools = [x for x in a.env_pools.split(",") if x] if a.arm in ("L", "RL") else []
    parts, rec = [f"on1_{a.base}:s:self"], dict(arm=a.arm, out=a.out, base=a.base, parts={})
    labs = {}
    for p in env_pools:
        Lb = dict(np.load(O / f"lab_{p}.npz"))
        keep = np.flatnonzero(Lb["ok"] & ~Lb["dev"])
        fast = p == a.fast
        env = np.full(len(keep), "fast", "U8") if fast else Lb["cls"][keep].astype("U8")
        sub = f"RCVE{p}"
        subset(p, keep, sub, dict(env=env, ecls=Lb["cls"][keep].astype("U8")))
        pre = "f" if fast else "e" + str(len(labs))
        parts.append(f"on1_{sub}:{pre}:self")
        labs[pre] = (Lb, keep)
        rec["parts"][pre] = dict(pool=p, rows=int(len(keep)), env={c: int((env == c).sum()) for c in np.unique(env)})
    if a.arm in ("R", "RL"):
        T = dict(np.load(CR / f"on1_{a.rec}" / "tab.npz"))
        dev = T["dev"].astype(bool) if "dev" in T else np.array([is_dev(x) for x in T["log"]])
        keep = np.flatnonzero(~dev)
        subset(a.rec, keep, f"RCVR{a.rec}", {})
        parts.append(f"on1_RCVR{a.rec}:r:both")
        sc = T.get("unit", T["log"])[keep]
        rec["parts"]["r"] = dict(pool=a.rec, rows=int(len(keep)), dev_rows=int(dev.sum()), units=len(set(map(str, sc))),
                                 kind={k: int((T["kind"][keep] == k).sum()) for k in np.unique(T["kind"])} if "kind" in T else None)
    merge(parts, a.out)
    M = dict(np.load(CR / f"on1_{a.out}" / "tab.npz"))
    names, pool = M["names"], M["pool"]
    n_s = int((pool == "s").sum())
    w, k_on = [], a.k_s
    if labs:                                                          # envelope.npz of the merged pool + the class weights
        N = len(names)
        NR = next(iter(labs.values()))[0]["route"].shape[1]
        E = dict(tokens=np.array([n[3:] for n in names.tolist()]), ok=np.zeros(N, bool), route=np.zeros((N, NR, 2), np.float32),
                 vuf=np.full((N, NR), BIG, np.float32), vur=np.full((N, NR), BIG, np.float32), vl=np.zeros((N, NR), np.float32), cls=np.full(N, "", "U8"))
        for pre, (Lb, keep) in labs.items():
            dst = np.flatnonzero(pool == pre)
            assert len(dst) == len(keep)
            assert Lb["route"].shape[1] == NR, "route grids differ between the envelope pools"
            for k in ("ok", "route", "vuf", "vur", "vl", "cls"):
                E[k][dst] = Lb[k][keep]
        np.savez(CR / f"on1_{a.out}" / "envelope.npz", **E)
        envt = M["env"]
        share = dict(stop=a.k_stop, fast=a.k_fast)
        rest = [c for c in ("turn", "lead", "free") if (envt == c).any()]
        for c in rest:
            share[c] = (a.k_env - a.k_stop - a.k_fast * bool((envt == "fast").any())) / len(rest)
        for c, kc in share.items():
            n_c = int((envt == c).sum())
            if n_c and kc > 0:
                w.append(f"env={c}:{kc / n_c / (a.k_s / n_s):.5f}")
        k_on += a.k_env
        rec["env_draws_per_batch"] = {c: round(v, 3) for c, v in share.items() if (envt == c).any()}
    cap = ""
    if a.arm in ("R", "RL"):
        n_r = int((pool == "r").sum())
        w.append(f"pool=r:{a.k_r / n_r / (a.k_s / n_s):.5f}")
        k_on += a.k_r
        units = M.get("unit", M["log"])[pool == "r"]
        n_u = len(set(map(str, units)))
        c = 0.10 if n_u * 0.10 >= 1 else 0.20 if n_u * 0.20 >= 1 else 0.0
        cap = f"--on-cap-scene {c} --on-cap-in pool=r" if c else ""
        rec |= dict(r_rows=n_r, r_units=n_u, r_cap=c)
    mode = "self" if a.arm == "L" else "mixed"
    flags = f"--on on1_{a.out} --on-k {k_on} --on-mode {mode}" + (f" --on-weight {','.join(w)}" if w else "") + (f" {cap}" if cap else "") \
        + (" --on-lon-all" if a.arm == "RL" else "")
    rec |= dict(s_rows=n_s, rows=int(len(names)), on_k=k_on, flags=flags)
    O.mkdir(parents=True, exist_ok=True)
    (O / f"pack_{a.out}.json").write_text(json.dumps(rec, indent=1))
    (O / f"flags_{a.out}.txt").write_text(flags + "\n")
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- forward reads (GPU)
def rows_of(pool, which):
    """-> (pool row indices, extra dict) of a read set."""
    if which == "dev-env":
        Lb = np.load(O / f"lab_{pool}.npz")
        r = np.flatnonzero(Lb["ok"] & Lb["dev"])
        return r, {}
    if which == "held01":
        m = np.load(ROWS / pool / "meta.npz")
        r = np.flatnonzero(m["k"] <= 1)
        return r, {}
    if which == "dev-rec":
        T = np.load(CR / f"on1_{pool}" / "tab.npz")
        dev = T["dev"].astype(bool) if "dev" in T.files else np.array([is_dev(x) for x in T["log"]])
        return np.flatnonzero(dev), {}
    raise SystemExit(f"rows {which}")


def cmd_fwd(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from types import SimpleNamespace
    sys.path[:0] = [str(R_ / "experiments/route_mem/scripts")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    out = Path(a.out or O / "fwd") / f"{a.pool}-{a.rows}"
    out.mkdir(parents=True, exist_ok=True)
    tags = [t for t in a.tags.split(",") if t]
    todo = [t for t in tags if not (out / f"plans_{t}.npz").exists()]
    if not todo:
        print("all plans exist", out)
        return
    rows, _ = rows_of(a.pool, a.rows)
    dev = torch.device("cuda")
    pool = f"on1_{a.pool}"
    ns_a = SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on=pool, init="", rm_from=todo[0], tag=todo[0], exp="route_follow", lr_scale=1.0,
                           prefetch=1, dev_card_gb=4.0)
    ns, _, _, _, _ = RT.build(ns_a, dev)
    S, model, on0 = ns.S, ns.model, ns.on0
    W = ns.losses(ns.mode).W
    tab = np.load(CR / pool / "tab.npz")
    F = np.load(FEAT / f"{pool}.npz")
    E = S.ego[on0:on0 + len(tab["names"])].cpu().numpy()
    gate = dict(ego_equal=float(np.abs(E[:, :20] - tab["ego"]).max()), feat_equal=float(np.abs(E[:, 20:] - F["f"]).max()), rows=int(len(rows)))
    pi = torch.as_tensor(S.pi, device=dev)

    def fwd(rr, bs=256):
        res = []
        with torch.no_grad():
            for i in range(0, len(rr), bs):
                r = on0 + np.asarray(rr[i:i + bs])
                rt = torch.as_tensor(r, device=dev)
                p = model(S.front[r], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
                x, y, psi = T.rear(p, S.cam_x[rt], W)
                res.append(torch.stack([x, y, psi], -1).cpu())
        return torch.cat(res).numpy() if res else np.zeros((0, 8, 3), np.float32)
    for j, tag in enumerate(todo):
        if j:
            ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
            model.load_state(ck["model"])
        model.eval()
        P = fwd(rows)
        if j == 0 and "served" in tab.files:
            d = np.hypot(*(P[..., :2] - tab["served"][rows][..., :2]).transpose(2, 0, 1)).mean(1)
            gate |= dict(tag=tag, rep_vs_served_median=float(np.median(d)), rep_vs_served_p95=float(np.percentile(d, 95)),
                         det_equal=bool(np.array_equal(fwd(rows[:256]), P[:256])))
            (out / "gate.json").write_text(json.dumps(gate, indent=1))
            print("gate", json.dumps(gate), flush=True)
        np.savez(out / f"plans_{tag}.npz", rows=rows, poses=P.astype(np.float32))
        print(tag, "done", len(rows), flush=True)


# ---------------------------------------------------------------- pilot gates (CPU)
def boot_diff(a_, b_, groups, B=10000, seed=0):
    from jevdrive import stats
    return stats.paired(np.asarray(a_, float), np.asarray(b_, float), groups=np.asarray(groups), n_boot=B, seed=seed)


def plans(pool, rows, tag):
    z = np.load(O / "fwd" / f"{pool}-{rows}" / f"plans_{tag}.npz")
    return z["rows"], z["poses"]


def gate_envelope(tag, base):
    import torch
    import on_envelope as OE
    Lb = dict(np.load(O / "lab_L18RSG.npz"))
    rr, Pt = plans("L18RSG", "dev-env", tag)
    rb, Pb = plans("L18RSG", "dev-env", base)
    assert np.array_equal(rr, rb)
    pk = dict(route=torch.as_tensor(Lb["route"]), vuf=torch.as_tensor(Lb["vuf"]), vur=torch.as_tensor(Lb["vur"]), vl=torch.as_tensor(Lb["vl"]),
              ok=torch.as_tensor(Lb["ok"]))
    Tt = OE.terms(torch.as_tensor(Pt), torch.as_tensor(rr), pk)
    Tb = OE.terms(torch.as_tensor(Pb), torch.as_tensor(rr), pk)
    cls, logs, v0 = Lb["cls"][rr], Lb["log"][rr], Lb["v0"][rr].astype(float)
    out = {}
    for nm, c, key in (("stop_excess", "stop", "envU"), ("A_speed", "turn", "envU"), ("lead_excess", "lead", "envU"), ("free_deficit", "free", "envL")):
        s = cls == c
        if not s.any():
            out[nm] = None
            continue
        t_, b_ = Tt[key].numpy()[s], Tb[key].numpy()[s]
        r = boot_diff(t_, b_, logs[s])
        out[nm] = dict(n=int(s.sum()), logs=len(set(logs[s])), pilot=float(t_.mean()), base=float(b_.mean()),
                       drop=float(1 - t_.mean() / max(b_.mean(), 1e-9)), diff=[r["mean"], r["lo"], r["hi"]])
    s = cls == "free"
    arc4 = lambda P: np.hypot(*np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P[:, :, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)  # noqa: E731
    rt, rbb = arc4(Pt[s]) / 4.0 / np.maximum(v0[s], 0.1), arc4(Pb[s]) / 4.0 / np.maximum(v0[s], 0.1)
    r = boot_diff(rt, rbb, logs[s])
    out["free_ratio"] = dict(n=int(s.sum()), pilot_median=float(np.median(rt)), base_median=float(np.median(rbb)), diff=[r["mean"], r["lo"], r["hi"]])
    L = dict(stop=out["stop_excess"] is not None and out["stop_excess"]["drop"] >= 0.20,
             A_speed=out["A_speed"] is not None and out["A_speed"]["diff"][2] < 0,
             free=out["free_ratio"]["diff"][1] > 0)
    return out, L


def gate_rec(tag, base, rec):
    rr, Pt = plans(rec, "dev-rec", tag)
    _, Pb = plans(rec, "dev-rec", base)
    T = np.load(CR / f"on1_{rec}" / "tab.npz")
    f = T["fut"][rr]
    ade = lambda P: np.hypot(*(P[..., :2] - f[..., :2]).transpose(2, 0, 1)).mean(1)  # noqa: E731
    at, ab = ade(Pt), ade(Pb)
    r = boot_diff(at, ab, T["log"][rr])
    out = dict(n=int(len(rr)), logs=len(set(T["log"][rr].tolist())), pilot=float(at.mean()), base=float(ab.mean()), drop=float(1 - at.mean() / ab.mean()),
               diff=[r["mean"], r["lo"], r["hi"]])
    return out, dict(R_imit=out["drop"] >= 0.20)


def turn_read(tag):
    """rf_appzeros' decisions 0 / 1 instrument on the held pool rows (k <= 1), the tag's offline plan: per row the signed 4 s heading toward the
    nearest >= 45 deg turn whose [entry - 40, exit + 10] holds the station."""
    import pickle
    import pandas as pd
    import rf_appzeros as AZ
    rr, P = plans("L18RSGallheld", "held01", tag)
    m = np.load(ROWS / "L18RSGallheld" / "meta.npz")
    with open(D / "runs/route_follow/route_dominance/zeros/held/rollouts.pkl", "rb") as fh:
        RR = pickle.load(fh)
    rt = {r["scene"]: r for r in RR}
    rows = []
    for j, r in enumerate(rr):
        sc, k = str(m["scene"][r]), int(m["k"][r])
        if sc not in rt:
            continue
        Rl = rt[sc]["route"]
        S = AZ.stations(Rl)
        s_h, h_h = AZ.heading(Rl, S)
        Tn = AZ.turns(s_h, h_h, 45.0)
        x, y, yaw = m["anchor"][r]
        st, lat = AZ.project(Rl, S, np.array([x, y]))
        near = [t for t in Tn if t[0] - 40 <= st <= t[1] + 10]
        if not near:
            continue
        t = min(near, key=lambda t: abs(st - t[0]))
        rel, sg = st - t[0], float(np.sign(t[2]))
        rows.append(dict(scene=sc, k=k, row=int(r), rel=rel, bin=str(pd.cut([rel], [-40, -20, -10, 0, 10, 30], right=False)[0]),
                         h4=float(np.degrees(sg * P[j, 7, 2])), y4=float(sg * P[j, 7, 1]), log=str(m["log"][r])))
    return pd.DataFrame(rows)


def gate_turn(tag, base):
    A, B = turn_read(tag), turn_read(base)
    X = A.merge(B, on=["scene", "k", "row", "rel", "bin", "log"], suffixes=("", "_b"))
    win = (X.rel >= -40) & (X.rel < 10)
    dh = (X.h4 - X.h4_b)[win]
    r = boot_diff(X.h4[win], X.h4_b[win], X.log[win])
    by = {b: dict(n=int(len(g)), dh_mean=float((g.h4 - g.h4_b).mean())) for b, g in X[win].groupby("bin")}
    out = dict(n=int(win.sum()), scenes=int(X[win].scene.nunique()), dh4_mean=float(dh.mean()), dh4_median=float(dh.median()), ci=[r["lo"], r["hi"]], by_bin=by)
    return out, dict(turn=out["dh4_mean"] >= -0.3)


def trainer_gates(tag):
    import pp_train as T  # noqa: F401
    rd = D / "runs/op_parity/runs" / tag
    ade = json.loads((rd / "route_mem.json").read_text()).get("dev_ade")
    rely = None
    for f in (D / "runs/route_mem/rely" / f"{tag}.json", rd / "rely.json"):
        if f.exists():
            rely = json.loads(f.read_text())
            break
    return dict(dev_ade=ade, rely=rely), ade is not None and ade <= 0.538


def cmd_gate(a):
    res, lines = dict(tag=a.tag, arm=a.arm, base=a.base), {}
    sys.path[:0] = [str(R_ / "experiments/op_parity/scripts")]
    tg, lines["dev_ade"] = trainer_gates(a.tag)
    res["trainer"] = tg
    rl = tg["rely"] or {}
    lines["rely"] = bool(rl["read"]) if "read" in rl else None      # rm_train's reliance read (rely.json `read`: G-rely true)
    res["turn"], tl = gate_turn(a.tag, a.base)
    lines |= tl
    if a.arm in ("L", "RL"):
        res["envelope"], el = gate_envelope(a.tag, a.base)
        lines |= el
    if a.arm in ("R", "RL"):
        res["rec"], rl_ = gate_rec(a.tag, a.base, a.rec)
        lines |= rl_
    res["lines"] = lines
    res["pass"] = all(v for v in lines.values() if v is not None) and lines.get("rely") is not False
    (O / "gate").mkdir(parents=True, exist_ok=True)
    (O / "gate" / f"{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"gate_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("PASS" if res["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("label")
    p.add_argument("--pool", default="L18RSG"), p.add_argument("--workers", type=int, default=32)
    p = sp.add_parser("pack")
    p.add_argument("--arm", required=True, choices=("L", "R", "RL")), p.add_argument("--out", required=True)
    p.add_argument("--env-pools", default="L18RSG"), p.add_argument("--fast", default="RCVfast"), p.add_argument("--rec", default="RCVd316")
    p.add_argument("--base", default="L18RB0"), p.add_argument("--k-s", type=int, default=13), p.add_argument("--k-env", type=int, default=7)
    p.add_argument("--k-stop", type=float, default=1.0), p.add_argument("--k-fast", type=float, default=1.0), p.add_argument("--k-r", type=int, default=3)
    p = sp.add_parser("fwd")
    p.add_argument("--pool", required=True), p.add_argument("--rows", required=True), p.add_argument("--tags", required=True), p.add_argument("--out", default="")
    p = sp.add_parser("gate")
    p.add_argument("--tag", required=True), p.add_argument("--arm", required=True, choices=("L", "R", "RL"))
    p.add_argument("--base", default="RSG-L18-s0"), p.add_argument("--rec", default="RCVd316")
    a = ap.parse_args()
    dict(label=cmd_label, pack=cmd_pack, fwd=cmd_fwd, gate=cmd_gate)[a.cmd](a)
