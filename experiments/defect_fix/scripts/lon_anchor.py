#!/usr/bin/env python3
"""LON-ANCHOR (plans/2026-10-20-lon-anchor-prereg.md; lib/nr1_anchor.py, rm_train --on-anchor W): labels, offline gates, draw weights, the
dose calibration and the inertia read-out. Box, repo root.

  $AP lon_anchor.py labels [--pool on1_JTCP]   station-anchored targets of the pool's S rows (nuPlan parts: navsim test logs; cp rows: the
                                              curated PAI row files) -> $CR/<pool>/lon_anchor.npz, $L/labels.npz, results/lon_anchor/labels.json
  $TR lon_anchor.py gate                      G1 (the human's station-matched path as the plan -> the term ~ 0) + reported trigger statistics
                                              on the stored plans -> results/lon_anchor/gate.{json,md}; exit 2 on failure
  $TR lon_anchor.py weights --out NAME        JT00S's draw on on1_JTCP, the nuPlan S share halved and the other half uniform over the cp rows
                                              outside the held 15 % and outside every excluded scene (PAI 281, practice-pai = val160, every
                                              scene id under experiments/pai_fix/scenes/) -> $W/NAME.npz, results/lon_anchor/w_NAME.json
  $TR lon_anchor.py calib --tag T --w W       the calibration run's --diag gradients -> $L/calib.json: W1 = W x RATIO x g_imit / g_anchor
  $TR lon_anchor.py inertia --name N --ctl C --arms A..   inertia.py's report.json -> the registered pass rule per arm -> results/lon_anchor/inertia.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "lib"), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/onpol_nuplan/scripts"),
                str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/wait_cap/scripts")]
import nr1_anchor as NA  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = DATA / "runs/op_parity/cache"
L_ = DATA / "runs/defect_fix/lon_anchor"
WD = DATA / "runs/defect_fix/w"
RES = REPO / "experiments/defect_fix/results/lon_anchor"
NU_ROWS = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y"}     # = stoppt.py: on1_JT's S parts -> onpol_nuplan rows dirs
CP_ROWS = DATA / "runs/practice_pai/curated/rows/CPAI570"
CP_TSV = REPO / "experiments/practice_pai/results/curated/scenes570.tsv"
PAI_FIX = REPO / "experiments/pai_fix/scenes"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
G1_ACTIVE, G1_LINE, ACTIVE_M, RATIO = 0.5, 0.01, 0.5, 0.1      # prereg: G1 row active = max |d| > 0.5 m, line 1 %; the 0.1 gradient rule
TH_US = 150_000                                                  # PAI: a box time within 0.15 s of t_h counts as "at t_h"


# ---------------------------------------------------------------- shared
def clock8(t, T, S):
    """The human's travel from his own position at the same log time t (the clock-aligned profile, for the diagnostics)."""
    S = np.maximum.accumulate(np.asarray(S, np.float64))
    return np.interp(t + NA.DT * np.arange(1, 9), T, S) - np.interp(t, T, S)


def along(P, a, q):
    """Polyline P (m, 2) at arcs a (m,) -> points at arcs q (8,) (straight beyond the ends) and the path headings there."""
    x, y = np.interp(q, a, P[:, 0]), np.interp(q, a, P[:, 1])
    for end, sl in ((q > a[-1], slice(-2, None)), (q < a[0], slice(0, 2))):
        if end.any():
            d = P[sl][1] - P[sl][0]
            u = d / max(np.hypot(*d), 1e-9)
            ref, ar = (P[-1], a[-1]) if sl.start == -2 else (P[0], a[0])
            x[end], y[end] = ref[0] + u[0] * (q[end] - ar), ref[1] + u[1] * (q[end] - ar)
    g = np.gradient(P, a, axis=0)
    h = np.arctan2(np.interp(q, a, g[:, 1]), np.interp(q, a, g[:, 0]))
    return np.c_[x, y, h]


def row_out(lab, s_t, th, why, t, T, S, gap_e, gap_h, hum):
    return dict(s_t=s_t if s_t is not None else np.full(8, np.nan), dt_h=th - t, why=why, gap_e=gap_e, gap_h=gap_h, hc8=clock8(t, T, S),
                hum=hum if hum is not None else np.full((8, 3), np.nan)) | lab


# ---------------------------------------------------------------- nuPlan S rows (one log per job)
def _log_centres(OR, L, t_us):
    """The log's object centres (k, 2, map frame) at log time t_us: tracks of the two neighbouring 2 Hz frames interpolated, a track in one
    frame only taken from it when that frame is within 0.3 s; NaN if no frame is within 0.6 s."""
    T = L["T"]
    f0 = int(np.clip(np.searchsorted(T, t_us, "right") - 1, 0, len(T) - 1))
    f1 = min(f0 + 1, len(T) - 1)
    if min(abs(T[f0] - t_us), abs(T[f1] - t_us)) > 600_000:
        return None
    a, b = OR._ann(L, f0), OR._ann(L, f1)
    w = 0.0 if f1 == f0 else float(np.clip((t_us - T[f0]) / max(T[f1] - T[f0], 1), 0, 1))
    out = []
    for k in set(a) | set(b):
        if k in a and k in b:
            out.append(a[k][:2] + w * (b[k][:2] - a[k][:2]))
        elif k in a and abs(t_us - T[f0]) <= 300_000:
            out.append(a[k][:2])
        elif k in b and abs(t_us - T[f1]) <= 300_000:
            out.append(b[k][:2])
    return np.array(out, np.float64).reshape(-1, 2)


def _nu_job(job):
    import on_rows as OR
    log, R = job
    L = OR._log(log)
    G = OR.Path(L["P"], L["T"])
    T, S = G.TF * 1e-6, G.SF
    ag = np.arange(-15.0, 71.0)                                                  # gpath arcs relative to s_e (on_rows meta)
    ah = np.arange(0.0, NA.LEAD_HI + 2.0)
    out = []
    for r in R:
        t = r["t_log_us"] * 1e-6
        s_e = float(G.s_log(r["t_log_us"])) + r["lead"]
        s_t, th, why = NA.target(t, s_e, T, S)
        gp = r["gpath"]
        gap_e = NA.lead_gap(gp, ag, r["box"])
        gap_h = np.nan
        if why == "ok" and np.isfinite(gap_e):
            B = _log_centres(OR, L, int(round(th * 1e6)))
            if B is not None:
                gap_h = NA.lead_gap(G.at(s_e + ah)[0], ah, B)
        if why == "ok":
            why = NA.lead_skip(gap_e, gap_h)
        hum = along(gp, ag, s_t) if s_t is not None else None
        out.append(dict(i=r["i"], **row_out(dict(e0=r["e0"]), s_t, th, why, t, T, S, gap_e, gap_h, hum)))
    return out


def nu_jobs(tab, agent):
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    pre = np.array([n.split(":")[1] for n in names])
    jobs, poses = defaultdict(list), np.full((len(names), 8, 3), np.nan, np.float32)
    for p, d in NU_ROWS.items():
        zz = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        z = {k: zz[k] for k in ("names", "speed", "poses", "log", "t_log_us", "lead", "gpath", "e0")}
        pos = {n: i for i, n in enumerate(z["names"].astype(str).tolist())}
        rows = np.flatnonzero(S & (pre == p))
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in rows])
        assert np.allclose(z["speed"][mi], tab["speed"][rows], atol=1e-3), f"{d}: meta speed != the pool's"
        poses[rows] = z["poses"][mi]
        for i, m in zip(rows, mi):
            v = agent["valid"][i, 0]
            jobs[str(z["log"][m])].append(dict(i=int(i), t_log_us=int(z["t_log_us"][m]), lead=float(z["lead"][m]), e0=float(z["e0"][m]),
                                               gpath=z["gpath"][m].astype(np.float64), box=agent["box"][i, 0, v, :2].astype(np.float64)))
    return jobs, poses


# ---------------------------------------------------------------- curated PAI rows (one scene per job)
def to_local(B, a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.c_[a[0] + c * B[:, 0] - s * B[:, 1], a[1] + s * B[:, 0] + c * B[:, 1]]


def _pai_job(job):
    f, ks = job
    zz = np.load(f, allow_pickle=True)
    z = {k: zz[k] for k in ("log_t_us", "log_s", "log_xyyaw", "k", "path", "path_ok", "box", "box_ok", "t0_us", "s_ego", "e0", "anchor", "poses_model")}
    T, S = z["log_t_us"] * 1e-6, np.maximum.accumulate(z["log_s"].astype(np.float64))
    u = np.r_[True, np.diff(S) > 1e-6]                                           # the human's path in the local frame by arc (vertices only)
    Pv, Sv = z["log_xyyaw"][u, :2].astype(np.float64), S[u]
    tb = (z["t0_us"][:, None] + np.arange(9)[None] * 500_000)                    # (K, 9) box times
    kk = z["k"].tolist()
    ah = np.arange(0.0, NA.LEAD_HI + 2.0)
    pa = np.arange(-20.0, 101.0)
    out = []
    for k in ks:
        n = kk.index(int(k))
        t, s_e = float(z["t0_us"][n]) * 1e-6, float(z["s_ego"][n])
        s_t, th, why = NA.target(t, s_e, T, S)
        hp = along(Pv, Sv, s_e + ah)[:, :2] if len(Sv) >= 2 else None
        gap_e = NA.lead_gap(hp, ah, to_local(z["box"][n, 0, z["box_ok"][n, 0], :2].astype(np.float64), z["anchor"][n])) if hp is not None else np.inf
        gap_h = np.nan
        if why == "ok" and np.isfinite(gap_e):
            d = np.abs(tb - th * 1e6)
            m, j = np.unravel_index(int(d.argmin()), d.shape)
            if d[m, j] <= TH_US:
                gap_h = NA.lead_gap(hp, ah, to_local(z["box"][m, j, z["box_ok"][m, j], :2].astype(np.float64), z["anchor"][m]))
        if why == "ok":
            why = NA.lead_skip(gap_e, gap_h)
        hum = None
        if s_t is not None:
            P, ok = z["path"][n].astype(np.float64), z["path_ok"][n]
            hum = along(P[ok], pa[ok], s_t) if ok.sum() >= 2 else None
        out.append(row_out(dict(e0=float(z["e0"][n]), poses=z["poses_model"][n].astype(np.float32)), s_t, th, why, t, T, S, gap_e, gap_h, hum))
    return out


def pai_rows(names, rows_dir, workers):
    by = defaultdict(list)
    for i, n in enumerate(names):
        sc, k = n.rsplit(":", 2)[-2:]
        by[sc].append((i, int(k)))
    jobs = [(str(rows_dir / f"{sc}.npz"), [k for _, k in v]) for sc, v in by.items()]
    idx = [[i for i, _ in v] for v in by.values()]
    res = [None] * len(names)
    with ProcessPoolExecutor(workers) as ex:
        for ii, out in zip(idx, ex.map(_pai_job, jobs, chunksize=1)):
            for i, r in zip(ii, out):
                res[i] = r
    return res


# ---------------------------------------------------------------- labels
def cmd_labels(a):
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    workers = a.workers or n_cpus()
    with Run("defect_fix", "lon-anchor-labels", config=vars(a)) as run:
        tab = dict(np.load(CR / a.pool / "tab.npz"))
        names, N = tab["names"].astype(str), len(tab["names"])
        agent = np.load(CR / a.pool / "agent.npz")
        assert np.array_equal(agent["tokens"].astype(str), names), "agent.npz rows"
        agent = dict(box=agent["box"], valid=agent["valid"])
        S = tab["mode"].astype(str) == "self"
        part = tab["pool"].astype(str)
        jobs, poses = nu_jobs(tab, agent)
        res = [None] * N
        run.info(f"{a.pool}: {N} rows, {S.sum()} S; nuPlan {sum(len(v) for v in jobs.values())} rows in {len(jobs)} logs; {workers} workers")
        with ProcessPoolExecutor(workers) as ex:
            for out in ex.map(_nu_job, list(jobs.items()), chunksize=1):
                for r in out:
                    res[r.pop("i")] = r
        ic = np.flatnonzero(S & (part == "cp"))
        for i, r in zip(ic, pai_rows(names[ic], Path(a.rows), workers)):
            poses[i] = r.pop("poses")
            res[i] = r
        miss = [i for i in np.flatnonzero(S) if res[i] is None]
        assert not miss, f"{len(miss)} S rows without a label (e.g. {names[miss[0]]})"
        A = dict(s_t=np.full((N, 8), np.nan, np.float32), hc8=np.full((N, 8), np.nan, np.float32), hum=np.full((N, 8, 3), np.nan, np.float32),
                 dt_h=np.full(N, np.nan, np.float32), gap_e=np.full(N, np.inf, np.float32), gap_h=np.full(N, np.nan, np.float32),
                 e0=np.full(N, np.nan, np.float32), why=np.full(N, "not_s", "U16"))
        for i, r in enumerate(res):
            if r is not None:
                for k in A:
                    A[k][i] = r[k]
        ok = A["why"] == "ok"
        assert np.isfinite(A["s_t"][ok]).all()
        np.savez(CR / a.pool / "lon_anchor.npz", names=tab["names"], s_t=np.where(ok[:, None], A["s_t"], np.nan).astype(np.float32), ok=ok, why=A["why"])
        L_.mkdir(parents=True, exist_ok=True)
        np.savez(L_ / "labels.npz", names=tab["names"], part=part, log=tab["log"], v0=tab["speed"].astype(np.float32), poses=poses, ok=ok, **A)
        rec = dict(pool=a.pool, rows=N, s_rows=int(S.sum()), ok=int(ok.sum()), why=dict(Counter(A["why"][S].tolist()).most_common()),
                   by_part={p: dict(s_rows=int((S & (part == p)).sum()), ok=int((ok & (part == p)).sum()),
                                    why=dict(Counter(A["why"][S & (part == p)].tolist()).most_common())) for p in np.unique(part[S])},
                   dt_h_q=np.nanquantile(A["dt_h"][ok], [0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist(),
                   lag_note="dt_h = t_h - t: > 0 the ego is ahead of the human's schedule at this station, < 0 behind")
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "labels.json").write_text(json.dumps(rec, indent=1))
        run.summary.update(ok=rec["ok"], s_rows=rec["s_rows"])
        print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- gate
def term_np(poses, s_t, ok):
    """The trainer's function (nr1_lon.terms on CPU) -> raw term (n,), d (n, 8)."""
    import torch
    import nr1_lon
    P = torch.as_tensor(np.nan_to_num(poses), dtype=torch.float32)
    st = torch.as_tensor(np.nan_to_num(s_t), dtype=torch.float32)
    okt = torch.as_tensor(ok)
    v = torch.cat([nr1_lon.terms(P[i:i + 8192], st[i:i + 8192], okt[i:i + 8192]) for i in range(0, len(P), 8192)]).numpy()
    xy = np.concatenate([np.zeros((len(P), 1, 2)), np.nan_to_num(poses)[..., :2]], 1)
    sp = np.cumsum(np.hypot(*np.diff(xy, axis=1).transpose(2, 0, 1)), 1)
    return v, sp - np.nan_to_num(s_t)


def cmd_gate(a):
    import csv
    z = dict(np.load(L_ / "labels.npz"))
    ok, part = z["ok"], z["part"].astype(str)
    src = np.where(part == "cp", "pai", "nuplan")
    hum_ok = ok & np.isfinite(z["hum"]).all((1, 2))
    vh, dh = term_np(z["hum"], z["s_t"], hum_ok)
    act_h = hum_ok & (np.abs(dh).max(1) > G1_ACTIVE)
    vp, dp = term_np(z["poses"], z["s_t"], ok & np.isfinite(z["poses"]).all((1, 2)))
    cls = {r["scene_id"]: r["cls"] for r in csv.DictReader(open(CP_TSV), delimiter="\t")}
    C = np.array([cls.get(s, "?") if p == "cp" else "nuplan" for s, p in zip(z["log"].astype(str), part)])
    seg_t = np.diff(np.c_[np.zeros(len(ok)), np.nan_to_num(z["s_t"])], axis=1) / NA.DT
    seg_c = np.diff(np.c_[np.zeros(len(ok)), np.nan_to_num(z["hc8"])], axis=1) / NA.DT
    stop = ok & (seg_t[:, -1] < 0.5)                                         # the anchored target ends standing
    conflict = stop & (seg_c[:, -1] >= 2.0)                                  # ... while the human at this clock moved >= 2 m/s
    goconf = ok & (np.nan_to_num(z["s_t"])[:, -1] >= 2.0) & (np.nan_to_num(z["hc8"])[:, -1] < 0.5)   # target moves >= 2 m, the clock human stands
    act_p = ok & (np.abs(dp).max(1) > ACTIVE_M)
    q = lambda x: np.quantile(x, [0.5, 0.9, 0.99]).round(4).tolist() if len(x) else []  # noqa: E731
    g1 = dict(rows=int(hum_ok.sum()), labelled=int(ok.sum()), no_path_plan=int((ok & ~hum_ok).sum()), active=int(act_h.sum()),
              share=float(act_h.sum() / max(hum_ok.sum(), 1)), line=G1_LINE, term_q=q(vh[hum_ok]),
              by_src={s: dict(rows=int((hum_ok & (src == s)).sum()), share=float((act_h & (src == s)).sum() / max((hum_ok & (src == s)).sum(), 1)))
                      for s in ("nuplan", "pai")})
    g1["pass_"] = bool(g1["share"] <= G1_LINE)
    rep = {}
    for k in ["all", "nuplan", "pai"] + sorted(set(C[part == "cp"])):
        m = ok & (np.ones(len(ok), bool) if k == "all" else (src == k) if k in ("nuplan", "pai") else (C == k))
        d4 = dp[m, -1]
        rep[k] = dict(rows=int(m.sum()), active=float(act_p[m].mean()) if m.any() else 0.0, fast=float((act_p[m] & (dp[m].max(1) > ACTIVE_M)).mean()) if m.any() else 0.0,
                      slow=float((act_p[m] & (dp[m].min(1) < -ACTIVE_M)).mean()) if m.any() else 0.0, d4_q=q(d4), d4_mean=float(d4.mean()) if m.any() else 0.0,
                      term_q=q(vp[m]), target_stop=int(stop[m].sum()), clock_conflict=int(conflict[m].sum()), go_conflict=int(goconf[m].sum()),
                      dt_h_q=q(z["dt_h"][m]))
    S = z["why"] != "not_s"
    res = dict(G1=g1, G4="df.py bitcmp (results/lon_anchor/identity.json)", stored_plans=rep,
               why={s: dict(Counter(z["why"][S & (src == s)].tolist()).most_common()) for s in ("nuplan", "pai")},
               note="stored plans = nuPlan rows' served plans (rows meta `poses`), cp rows' model plans (`poses_model`, NTV0-L18-s0); "
                    "d4 = plan 4 s arc - anchored target (m); active = max |d| > 0.5 m; reported, not gated")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate.json").write_text(json.dumps(res, indent=1))
    Lm = [f"# LON-ANCHOR offline gates (G1 {'PASS' if g1['pass_'] else 'FAIL'})", "",
          f"G1: the human's station-matched path (points on his logged path at s_e + s_t,j, in the row frame) as the plan, the trainer's term: "
          f"{g1['active']} of {g1['rows']} labelled rows active (max |d| > {G1_ACTIVE} m) = {100 * g1['share']:.2f} % (line <= {100 * G1_LINE:.0f} %); "
          f"term quantiles (50 / 90 / 99 %) {g1['term_q']}; nuPlan {100 * g1['by_src']['nuplan']['share']:.2f} %, PAI {100 * g1['by_src']['pai']['share']:.2f} %.",
          "", "Stored plans against the anchored target (reported only):", "",
          "| rows | n | active | plan too fast | plan too slow | d4 mean | d4 50 / 90 / 99 % | target ends standing | of those: human moving at the clock | target moves >= 2 m, clock human stands | dt_h 50 / 90 / 99 % |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, r in rep.items():
        Lm.append(f"| {k} | {r['rows']} | {100 * r['active']:.1f} % | {100 * r['fast']:.1f} % | {100 * r['slow']:.1f} % | {r['d4_mean']:+.2f} | {r['d4_q']} | "
                  f"{r['target_stop']} | {r['clock_conflict']} | {r['go_conflict']} | {r['dt_h_q']} |")
    Lm += ["", f"Label outcomes (S rows): {json.dumps(res['why'])}", "", res["note"]]
    (RES / "gate.md").write_text("\n".join(Lm) + "\n")
    print("\n".join(Lm))
    if not g1["pass_"]:
        raise SystemExit(2)


# ---------------------------------------------------------------- weights
def excluded():
    """Scene ids never drawn: PAI 281, practice-pai (= val160), every id found in experiments/pai_fix/scenes/ (any file, uuid match)."""
    from jevdrive.data import splits
    ex = {m.split("clipgt-")[-1] for nm in ("physical_ai_av/test281", "physical_ai_av/practice-pai") for m in splits.load(nm).members}
    fix, files = set(), []
    if PAI_FIX.is_dir():
        for f in sorted(p for p in PAI_FIX.rglob("*") if p.is_file()):
            files.append(str(f.relative_to(REPO)))
            fix |= set(UUID.findall(f.read_text(errors="ignore")))
    return ex, fix, files


def cmd_weights(a):
    tb = np.load(CR / a.pool / "tab.npz")
    names, part, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["log"].astype(str)
    w0 = np.load(WD / f"{a.base}.npz")
    n0 = len(w0["names"])
    assert np.array_equal(w0["names"].astype(str), names[:n0]) and (part[n0:] == "cp").all(), "merged pool != on1_JT + cp rows"
    ex, fix, files = excluded()
    if a.require_fix and not files:
        raise SystemExit(f"no scene files under {PAI_FIX}: the PAI-FIX exam list is required (prereg) and not there yet")
    P = np.arange(n0, len(names))
    sc = np.array([s.split("clipgt-")[-1] for s in logs[P]])
    held = np.array([zlib.crc32(g.encode()) % 100 < 15 for g in logs[P]])           # = rm_train --on-hold-pct 15
    bad_x, bad_f = np.isin(sc, list(ex)), np.isin(sc, list(fix))
    grp = np.r_[w0["grp"].astype(str), np.full(len(P), "P")].astype("U12")
    grp[P[held]] = "P_held"
    grp[P[bad_f & ~held]] = "P_fix"
    grp[P[bad_x]] = "P_exam"
    w = np.zeros(len(names))
    w[:n0] = w0["w"]
    S0 = np.isin(grp[:n0], ["S", "S_held"])
    fs = float(w[:n0][S0].sum())
    w[:n0][S0] *= 1.0 - a.share
    draw = P[grp[P] == "P"]
    w[draw] = a.share * fs / max(len(draw), 1)
    w /= w.sum()
    assert w[P[bad_x | bad_f]].sum() == 0
    np.savez(WD / f"{a.out}.npz", names=tb["names"], w=w, grp=grp)
    rec = dict(pool=a.pool, base=a.base, share=a.share, rows=len(names), excluded_exam_rows=int(bad_x.sum()), excluded_fix_rows=int(bad_f.sum()),
               excluded_fix_scenes_in_pool=sorted(set(sc[bad_f])), pai_fix_ids=len(fix), pai_fix_files=files,
               share_by_group={g: float(w[grp == g].sum()) for g in np.unique(grp)}, per_batch_32={g: float(w[grp == g].sum() * 32) for g in np.unique(grp)},
               rows_by_group={g: int((grp == g).sum()) for g in np.unique(grp)}, cp_scenes_drawn=int(len(set(logs[draw]))))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"w_{a.out}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- calibration
def cmd_calib(a):
    f = sorted(glob.glob(str(DATA / "runs/route_mem" / f"train-{a.tag}" / "*" / "events.jsonl")))[-1]
    G = defaultdict(dict)
    for line in open(f):
        e = json.loads(line)
        if e.get("kind") == "scalar" and e["tag"] in ("loss/grad/imit", "loss/grad/lon_anchor"):
            G[e["tag"].rsplit("/", 1)[1]][e["step"]] = e["value"]
    st = sorted(set(G["imit"]) & set(G["lon_anchor"]))
    assert len(st) >= 3, f"too few diag steps: {st}"
    r = np.array([G["lon_anchor"][s] / G["imit"][s] for s in st])
    w1 = float(f"{a.w * RATIO / float(np.median(r)):.2g}")
    rec = dict(tag=a.tag, events=f, w_probe=a.w, ratio_target=RATIO, steps=st, g_imit=[G["imit"][s] for s in st], g_anchor=[G["lon_anchor"][s] for s in st],
               ratio_median=float(np.median(r)), w1=w1, doses={"A05": float(f"{0.5 * w1:.2g}"), "A10": w1, "A20": float(f"{2 * w1:.2g}")})
    L_.mkdir(parents=True, exist_ok=True)
    (L_ / "calib.json").write_text(json.dumps(rec, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "calib.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- inertia read-out
STOP_LINE, TURN_LINE, DROP, OPEN_LINE = 0.6, 0.6, 0.15, 0.7


def cmd_inertia(a):
    r = json.load(open(a.report or REPO / f"experiments/defect_fix/results/inertia/report_{a.name}.json"))
    V = r["verdicts"]
    cells = {(c["tag"], c["win"]): c for c in r["cells"]}

    def open_slope(t):
        c = cells.get((t, "OPEN"), {})
        for k in ("v24_B", "slope_v24_B", "v24slope_B"):
            if k in c:
                return c[k][0] if isinstance(c[k], list) else c[k]
        return np.nan
    out, L = {}, ["# LON-ANCHOR inertia read-out (inertia.py unchanged, P160 rows)", "",
                  f"Pass per arm: STOP ratio <= {STOP_LINE} and TURN ratio <= {TURN_LINE}, each at least {DROP} below the control's, and the "
                  f"OPEN v24 slope >= {OPEN_LINE}.", "", "| tag | STOP ratio | TURN ratio | OPEN v24 slope (B) | STOP | TURN | OPEN | pass |", "|:--|--:|--:|--:|:--|:--|:--|:--|"]
    c0 = V[a.ctl]
    for t in [a.ctl] + a.arms + [x for x in V if x not in [a.ctl] + a.arms]:
        v = V[t]
        sr, tr, os_ = v.get("STOP_aux_ratio", np.nan), v.get("TURN_ratio", np.nan), open_slope(t)
        p = None
        if t in a.arms:
            p = bool(sr <= STOP_LINE and tr <= TURN_LINE and sr <= c0["STOP_aux_ratio"] - DROP and tr <= c0["TURN_ratio"] - DROP and os_ >= OPEN_LINE)
        out[t] = dict(stop_ratio=sr, turn_ratio=tr, open_v24=os_, stop=v.get("STOP"), turn=v.get("TURN"), open=v.get("OPEN"), pass_=p)
        L.append(f"| {t} | {sr:.3f} | {tr:.3f} | {os_:.3f} | {v.get('STOP')} | {v.get('TURN')} | {v.get('OPEN')} | {'' if p is None else ('PASS' if p else 'FAIL')} |")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "inertia.json").write_text(json.dumps(dict(report=a.name, ctl=a.ctl, arms=out, lines=dict(stop=STOP_LINE, turn=TURN_LINE, drop=DROP, open=OPEN_LINE)), indent=1, default=float))
    (RES / "inertia.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("labels")
    q.add_argument("--pool", default="on1_JTCP"), q.add_argument("--rows", default=str(CP_ROWS)), q.add_argument("--workers", type=int, default=0)
    sp.add_parser("gate")
    q = sp.add_parser("weights")
    q.add_argument("--pool", default="on1_JTCP"), q.add_argument("--base", default="JT00S"), q.add_argument("--share", type=float, default=0.5)
    q.add_argument("--out", required=True), q.add_argument("--require-fix", action="store_true")
    q = sp.add_parser("calib")
    q.add_argument("--tag", required=True), q.add_argument("--w", type=float, required=True)
    q = sp.add_parser("inertia")
    q.add_argument("--name", default="lon_anchor"), q.add_argument("--report", default=""), q.add_argument("--ctl", required=True), q.add_argument("--arms", nargs="+", required=True)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
