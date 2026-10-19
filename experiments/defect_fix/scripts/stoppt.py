#!/usr/bin/env python3
"""NR1STOPPT (NR1 prereg supplement 18; lib/nr1_free.py stop_label / stoppt_*): "do not pass the human's stop point" on approach states.
Box, repo root.

  $AP stoppt.py labels                 rest-point labels of on1_JT's S rows (navsim test logs) and on1_PPAI160's rows (PAI row files) -> $S/labels.npz,
                                       $CR/on1_JT/stoppt.npz, $CR/on1_PPAI160/stoppt.npz, results/stoppt/labels.json
  $AP stoppt.py assemble --pool P      labels.npz -> $CR/<P>/stoppt.npz for a merged pool (rows "on:<part>:<name>", PAI "on:pai:pai:..")
  $TR stoppt.py weights                JT00 -> JTPAI00 on on1_JTPAI (arm B: nuPlan S half of the S share, PAI rows the other half)
  $AP stoppt.py gate --out DIR         G1-G3 (CPU) -> DIR/gate.json, DIR/a0.parquet; exit 0 pass, 2 fail
  $TR stoppt.py pfeats                 probe features (GPU): newest-slot tokens and NTV0-L18-s0's lead / plan heads per source -> $S/probe/feats/
  $TR stoppt.py probe                  the probe (GPU) -> $S/probe/probe.json, results/stoppt/probe.json; `report` prints the tables
  $AP stoppt.py overrun --tax N=PKL..  d381 overrun zeros (score 0, corridor, cor_route_end) per PAI read -> --out JSON
  $TR stoppt.py pilot --tag T ..       the supplement 18 pilot rule -> MD (+ .json); prints VERDICT

Closed-loop PAI states (A0 = NTV0-L18-s0 + ACC + GAP, d366): rf_tax extract (controller trace and log on the evaluator's clock), the ego's station
= its rear axle projected on the logged polyline, the human's stations = the cumulative logged arc; traffic = the extract's 0.5 s steps.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/route_follow/scripts"),
                str(REPO / "experiments/onpol_nuplan/scripts"), str(REPO / "experiments/wait_cap/scripts")]
import nr1_free as NF  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix"
S_ = O / "stoppt"
CR = DATA / "runs/op_parity/cache"
RES = REPO / "experiments/defect_fix/results/stoppt"
NU_ROWS = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y"}     # on1_JT's S parts -> onpol_nuplan rows dirs
PAI_ROWS = DATA / "runs/practice_pai/rows/PPAI160"
A0 = DATA / "runs/defect_fix/pai_accgap/NTV0-L18-s0"
A0TAX = DATA / "runs/route_follow/pai_input_loop/tax/A0.pkl"
GAP = DATA / "runs/route_follow/pai_gap"
A0R = S_ / "a0r"                                                             # the probe's full A0 rerun (rows / cache)
PCACHE = {"A0R": A0R / "cache/A0R", "P160": DATA / "runs/practice_pai/cache/PPAI160"}
PROWS = {"A0R": A0R / "rows/A0R", "P160": PAI_ROWS}
LEAD_LAT, LEAD_LO, LEAD_HI, HAND = 1.8, -2.0, 20.0, 1.6e6
TAU_LO, TAU_MID, TAU_HI, TAU_NEG, V_PROBE, NNEG_NU = 1.5, 2.0, 4.0, 8.0, 2.0, 30000
PROBE_TAG = "NTV0-L18-s0"


# ---------------------------------------------------------------- labels: one state
def human8(t, th, s_e, T, S):
    """The human's two 4 s paths as arcs from the ego (8 steps): station-matched s_h(t_h + 0.5 j) - s_e and clock max(s_h(t + 0.5 j) - s_e, 0)."""
    S = np.maximum.accumulate(np.asarray(S, np.float64))
    j = 0.5 * np.arange(1, 9)
    hs = np.interp(th + j, T, S) - s_e if np.isfinite(th) else np.full(8, np.nan)
    return hs, np.maximum(np.interp(t + j, T, S) - s_e, 0.0)


def lead_flag(rest, P, i0, B, ok):
    """Any object centre B (m, 2) [ok (m,)] within LEAD_LAT of the human path points P (1 m grid, index i0 = the ego's station) at arcs
    rest + [LEAD_LO, LEAD_HI] (decision 362's rule)."""
    if not np.isfinite(rest) or not ok.any():
        return False
    idx = np.arange(int(np.ceil(rest + LEAD_LO)), int(np.floor(rest + LEAD_HI)) + 1) + i0
    idx = idx[(idx >= 0) & (idx < len(P))]
    if not len(idx):
        return False
    d = np.hypot(B[ok, None, 0] - P[None, idx, 0], B[ok, None, 1] - P[None, idx, 1])
    return bool((d < LEAD_LAT).any())


def label_row(t, s_e, T, S, v0, lead_fn, heading=None):
    lab = NF.stop_label(t, s_e, T, S)
    lead = lead_fn(lab["rest"])
    ok, why = NF.stoppt_ok(lab, v0, lead)
    hs, hc = human8(t, lab["t_h"], s_e, T, S)
    turn = np.nan
    if heading is not None and np.isfinite(lab["t_h"]):
        turn = float(np.degrees(abs(heading(float(np.interp(lab["t_h"] + 8.0, T, np.maximum.accumulate(S)))) - heading(s_e))))
    return dict(rest=lab["rest"], dt_rest=lab["t_rest"] - t, h4c=lab["h4c"], h4s=lab["h4s"], vmin8=lab["vmin8"], dt_h=lab["t_h"] - t,
                turn8=turn, lead=lead, ok=ok, why=why, hs8=hs, hc8=hc, path_ok=lab["path_ok"])


# ---------------------------------------------------------------- nuPlan pool rows (one log per job)
def _nu_job(job):
    import on_rows as OR
    log, R = job
    L = OR._log(log)
    G = OR.Path(L["P"], L["T"])
    T, S = G.TF * 1e-6, G.SF
    hu = np.unwrap(G.h)
    h = lambda s: float(hu[np.clip(np.searchsorted(G.s, s), 0, len(G.s) - 1)])  # noqa: E731
    out = []
    for r in R:
        t = r["t_log_us"] * 1e-6
        s_e = float(G.s_log(r["t_log_us"])) + r["lead"]
        out.append(dict(i=r["i"], **label_row(t, s_e, T, S, r["v0"], lambda rest: lead_flag(rest, r["gpath"], 15, r["box"], r["bval"]), h)))
    return out


def nu_rows(tab, agent):
    """on1_JT S rows -> per-log jobs (meta of the rows dirs, the pool's agent boxes at 0 .. 4 s)."""
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    pre = np.array([n.split(":")[1] for n in names])
    jobs, poses = defaultdict(list), np.full((len(names), 8, 3), np.nan, np.float32)
    for p, d in NU_ROWS.items():
        zz = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        z = {k: zz[k] for k in ("names", "speed", "poses", "log", "t_log_us", "lead", "gpath")}     # once: npz members are re-read per access
        pos = {n: i for i, n in enumerate(z["names"].astype(str).tolist())}
        rows = np.flatnonzero(S & (pre == p))
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in rows])
        assert np.allclose(z["speed"][mi], tab["speed"][rows], atol=1e-3), f"{d}: meta speed != the pool's"
        poses[rows] = z["poses"][mi]
        for i, m in zip(rows, mi):
            jobs[str(z["log"][m])].append(dict(i=int(i), t_log_us=int(z["t_log_us"][m]), lead=float(z["lead"][m]), v0=float(tab["speed"][i]),
                                               gpath=z["gpath"][m].astype(np.float64), box=agent["box"][i, :, :, :2].reshape(-1, 2).astype(np.float64),
                                               bval=agent["valid"][i].reshape(-1)))
    return jobs, poses


# ---------------------------------------------------------------- PAI rows (one scene per job)
def _pai_job(job):
    f, ks, v0s = job
    zz = np.load(f, allow_pickle=True)
    z = {k: zz[k] for k in ("log_t_us", "log_s", "log_xyyaw", "k", "path", "path_ok", "box", "box_ok", "t0_us", "s_ego", "has_stop", "stop_s", "wl_s",
                            "poses_model")}                              # once: an npz member is re-read on every access
    T, S = z["log_t_us"] * 1e-6, z["log_s"].astype(np.float64)
    yaw = np.unwrap(z["log_xyyaw"][:, 2])
    h = lambda s: float(np.interp(s, S, yaw))  # noqa: E731
    kk = z["k"].tolist()
    out = []
    for k, v0 in zip(ks, v0s):
        n = kk.index(int(k))
        P, pok = z["path"][n].astype(np.float64), z["path_ok"][n]
        P = np.where(pok[:, None], P, 1e6)
        B, bok = z["box"][n, :, :, :2].reshape(-1, 2).astype(np.float64), z["box_ok"][n].reshape(-1)
        r = label_row(float(z["t0_us"][n]) * 1e-6, float(z["s_ego"][n]), T, S, float(v0), lambda rest: lead_flag(rest, P, 20, B, bok), h)
        ws = np.concatenate([np.where(z["has_stop"], z["stop_s"][n], np.nan), z["wl_s"][n]]).astype(np.float64)
        r["line"] = bool(np.isfinite(r["rest"]) and (np.abs(ws - r["rest"] + 5) <= 10).any())
        r["poses"] = z["poses_model"][n].astype(np.float32)
        out.append(r)
    return out


def pai_rows(names, speed, rows_dir, workers):
    """Pool / cache rows keyed 'scene:k' -> list of label dicts (same order)."""
    by = defaultdict(list)
    for i, n in enumerate(names):
        sc, k = n.rsplit(":", 2)[-2:]
        by[sc].append((i, int(k)))
    jobs = [(str(rows_dir / f"{sc}.npz"), [k for _, k in v], [float(speed[i]) for i, _ in v]) for sc, v in by.items()]
    idx = [[i for i, _ in v] for v in by.values()]
    res = [None] * len(names)
    with ProcessPoolExecutor(workers) as ex:
        for ii, out in zip(idx, ex.map(_pai_job, jobs, chunksize=1)):
            for i, r in zip(ii, out):
                res[i] = r
    return res


KEYS = ("rest", "dt_rest", "h4c", "h4s", "vmin8", "dt_h", "turn8")


def pack(res, n):
    """list of label dicts (None = not a labelled row) -> column arrays."""
    A = {k: np.full(n, np.nan, np.float32) for k in KEYS}
    A |= dict(lead=np.zeros(n, bool), ok=np.zeros(n, bool), line=np.zeros(n, bool), why=np.full(n, "not_s", "U16"),
              hs8=np.full((n, 8), np.nan, np.float32), hc8=np.full((n, 8), np.nan, np.float32))
    for i, r in enumerate(res):
        if r is None:
            continue
        for k in KEYS:
            A[k][i] = r[k]
        A["lead"][i], A["ok"][i], A["why"][i], A["hs8"][i], A["hc8"][i] = r["lead"], r["ok"], r["why"] or "ok", r["hs8"], r["hc8"]
        A["line"][i] = r.get("line", False)
    return A


def write_stoppt(pool, names, A, v0):
    np.savez(CR / pool / "stoppt.npz", names=names, rest=np.where(np.isfinite(A["rest"]), A["rest"], np.inf).astype(np.float32),
             v0=np.asarray(v0, np.float32), ok=A["ok"], why=A["why"])


def summary(A, m):
    w = Counter(A["why"][m].tolist())
    return dict(rows=int(m.sum()), ok=int((A["ok"] & m).sum()), why=dict(w.most_common()),
                rest_q=np.nanquantile(np.where(np.isfinite(A["rest"]), A["rest"], np.nan)[m & A["ok"]], [0.1, 0.5, 0.9]).round(2).tolist()
                if (A["ok"] & m).any() else [])


def cmd_labels(a):
    tab = dict(np.load(CR / "on1_JT/tab.npz"))
    names = tab["names"].astype(str)
    agent = np.load(CR / "on1_JT/agent.npz")
    assert np.array_equal(agent["tokens"].astype(str), names), "agent.npz rows"
    agent = dict(box=agent["box"], valid=agent["valid"])
    jobs, poses = nu_rows(tab, agent)
    res = [None] * len(names)
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(_nu_job, list(jobs.items()), chunksize=1):
            for r in out:
                res[r.pop("i")] = r
    A = pack(res, len(names))
    write_stoppt("on1_JT", names, A, tab["speed"])
    pt = np.load(CR / "on1_PPAI160/tab.npz")
    pn = pt["names"].astype(str)
    pr = pai_rows(pn, pt["speed"], PAI_ROWS, a.workers)
    B = pack(pr, len(pn))
    write_stoppt("on1_PPAI160", pn, B, pt["speed"])
    ppos = np.stack([r["poses"] for r in pr])
    S_.mkdir(parents=True, exist_ok=True)
    cat = lambda k: np.concatenate([A[k], B[k]])  # noqa: E731
    np.savez(S_ / "labels.npz", names=np.r_[names, pn], src=np.r_[np.full(len(names), "nu"), np.full(len(pn), "pai")],
             log=np.r_[tab["log"].astype(str), pt["log"].astype(str)], v0=np.r_[tab["speed"], pt["speed"]].astype(np.float32),
             poses=np.concatenate([poses, ppos]), fut=np.concatenate([tab["fut"], pt["fut"]]).astype(np.float32),
             fut_reg=np.concatenate([tab["fut_reg"], np.full_like(pt["fut"], np.nan)]).astype(np.float32), **{k: cat(k) for k in A})
    S = tab["mode"].astype(str) == "self"
    rec = dict(on1_JT=summary(A, S), on1_PPAI160=summary(B, np.ones(len(pn), bool)),
               nu_by_part={p: summary(A, S & (np.char.startswith(names, f"on:{p}:"))) for p in NU_ROWS})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "labels.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_assemble(a):
    """labels.npz -> <pool>/stoppt.npz: rows of a merged pool by name (nuPlan parts keep on1_JT's names; PAI 'on:pai:pai:<scene>:<k>')."""
    z = np.load(S_ / "labels.npz")
    pos = {n: i for i, n in enumerate(z["names"].astype(str).tolist())}
    tb = np.load(CR / a.pool / "tab.npz")
    names = tb["names"].astype(str)
    key = [n.replace("on:pai:pai:", "on:pai:", 1) for n in names]
    S = tb["mode"].astype(str) == "self"
    j = np.array([pos.get(k, -1) for k in key])
    assert (j[S] >= 0).all(), f"{a.pool}: S rows without a label"
    hit = S & (j >= 0)
    rest = np.where(hit, z["rest"][np.maximum(j, 0)], np.inf)
    ok = hit & z["ok"][np.maximum(j, 0)]
    assert np.allclose(z["v0"][j[hit]], tb["speed"][hit], atol=1e-3), "speed"
    np.savez(CR / a.pool / "stoppt.npz", names=names, rest=np.where(np.isfinite(rest), rest, np.inf).astype(np.float32),
             v0=tb["speed"].astype(np.float32), ok=ok, why=np.where(hit, z["why"][np.maximum(j, 0)], "not_s"))
    rec = dict(pool=a.pool, rows=len(names), s_rows=int(S.sum()), ok=int(ok.sum()), ok_by_part={p: int((ok & (tb["pool"].astype(str) == p)).sum())
                                                                                              for p in np.unique(tb["pool"].astype(str))})
    (RES / f"labels_{a.pool}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def cmd_weights(a):
    """JTPAI00: JT00's trainval rows unchanged; JT00's S share split half to the nuPlan S rows (relative weights kept) and half uniform over
    the PAI rows outside the held scenes (crc32(scene) % 100 < 15, as the trainer's --on-hold-pct 15)."""
    tb = np.load(CR / "on1_JTPAI/tab.npz")
    names, pool, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["log"].astype(str)
    w0 = np.load(O / "w/JT00.npz")
    n0 = len(w0["names"])
    assert np.array_equal(w0["names"].astype(str), names[:n0]) and (pool[n0:] == "pai").all(), "JTPAI != JT + pai rows"
    w, grp = np.zeros(len(names)), np.r_[w0["grp"].astype(str), np.full(len(names) - n0, "P")].astype("U12")
    w[:n0] = w0["w"]
    S0 = np.isin(grp[:n0], ["S", "S_held"])
    fs = float(w[:n0][S0].sum())
    w[:n0][S0] *= 0.5
    P = np.arange(n0, len(names))
    held = np.array([zlib.crc32(g.encode()) % 100 < 15 for g in logs[P]])           # = jt.held_logs / rm_train --on-hold-pct 15
    grp[P[held]] = "P_held"
    w[P[~held]] = 0.5 * fs / (~held).sum()
    w /= w.sum()
    np.savez(O / "w/JTPAI00.npz", names=tb["names"], w=w, grp=grp)
    rec = dict(rows=len(names), share={g: float(w[grp == g].sum()) for g in np.unique(grp)}, per_batch_70={g: float(w[grp == g].sum() * 70) for g in np.unique(grp)},
               rows_by_group={g: int((grp == g).sum()) for g in np.unique(grp)}, s_share_jt00=fs)
    (RES / "w_JTPAI00.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_weightsf(a):
    """JTPAIF (supplement 18 addendum, arm C): JTPAI00 with the rows where the stored plans fire (F) drawn ~2 times each in 1 000 steps x 70,
    the F share capped at 15 % of the S share, every log / scene at <= 5 % of the F draw; the rest of the S share as arm B, trainval as JT00."""
    z = np.load(S_ / "labels.npz")
    fp, _ = score(z["poses"], z["rest"], z["v0"], z["ok"])
    pos = {n: i for i, n in enumerate(z["names"].astype(str).tolist())}
    tb = np.load(CR / "on1_JTPAI/tab.npz")
    names, pool, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["log"].astype(str)
    S = tb["mode"].astype(str) == "self"
    j = np.array([pos.get(n.replace("on:pai:pai:", "on:pai:", 1), -1) for n in names])
    held = S & np.array([zlib.crc32(g.encode()) % 100 < 15 for g in logs])        # = rm_train --on-hold-pct 15 (S rows)
    F = S & (j >= 0) & fp[np.maximum(j, 0)] & ~held
    w0 = np.load(O / "w/JTPAI00.npz")
    assert np.array_equal(w0["names"].astype(str), names), "JTPAI00 rows"
    w0w, grp = w0["w"].astype(np.float64), w0["grp"].astype(str).astype("U12")
    sS = float(w0w[S].sum())                                                       # the S share (32 / 70)
    nF = int(F.sum())
    fshare = min(2.0 * nF / 70000.0, 0.15 * sS)
    key = np.where(pool == "pai", logs, logs)                                      # nuPlan log / PAI scene (the tab's log = scene for PAI rows)
    u, inv = np.unique(key[F], return_inverse=True)
    q = np.ones(nF) / nF                                                           # per F row share of the F draw
    capped = set()
    for _ in range(100):
        g = np.bincount(inv, q)
        over = g > 0.05 + 1e-12
        if not over.any():
            break
        capped |= set(u[over].tolist())
        q[over[inv]] *= (0.05 / g[inv])[over[inv]]
        rest = ~np.isin(u[inv], list(capped))
        q[rest] *= (1 - np.bincount(inv, q)[np.isin(u, list(capped))].sum()) / q[rest].sum()
    g0 = np.bincount(inv, np.ones(nF) / nF)
    g1 = np.bincount(inv, q)
    w = np.zeros(len(names))
    w[~S] = w0w[~S]
    w[F] = fshare * q
    nuS, paS = S & ~F & (pool != "pai") & ~held, S & ~F & (pool == "pai") & ~held
    rest_s = sS - fshare
    w[nuS] = 0.5 * rest_s * w0w[nuS] / w0w[nuS].sum()
    w[paS] = 0.5 * rest_s / paS.sum()
    w /= w.sum()
    grp = np.where(F, "F", grp).astype("U12")
    np.savez(O / "w/JTPAIF.npz", names=tb["names"], w=w, grp=grp)
    top = lambda g: [(str(u[i]), round(float(g[i]), 4)) for i in np.argsort(-g)[:10]]  # noqa: E731
    passes = w[F] * 70000
    rec = dict(F_rows=nF, F_nu=int((F & (pool != "pai")).sum()), F_pai=int((F & (pool == "pai")).sum()), F_logs_scenes=len(u),
               F_share_total=float(w[F].sum()), F_share_of_S=float(w[F].sum() / w[S].sum()), F_per_batch=float(w[F].sum() * 70),
               passes_median=float(np.median(passes)), passes_min=float(passes.min()), passes_max=float(passes.max()),
               cap=0.05, capped=sorted(capped), top10_before=top(g0), top10_after=top(g1),
               share={k: float(w[grp == k].sum()) for k in np.unique(grp)}, per_batch_70={k: float(w[grp == k].sum() * 70) for k in np.unique(grp)})
    (RES / "w_JTPAIF.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    assert 1.0 <= np.median(passes) <= 3.0, "passes per F row out of the registered range"


# ---------------------------------------------------------------- closed loop (A0, PAI 281)
def _a0_job(job):
    import rf_tax as TX
    s, o, recs = job
    R = TX.Roll(o)
    T = np.asarray(o["log"]["T"], np.float64) * 1e-6
    P = np.asarray(o["log"]["P"], np.float64)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    Y = np.unwrap(np.asarray(o["log"]["Y"], np.float64))
    h = lambda q: float(np.interp(q, S, Y))  # noqa: E731
    tg = np.asarray(R.tg, np.float64)
    out = []
    for r in recs:
        t0, v = float(r["t0"]), float(r["ego"][4]) * 10
        if t0 < R.t[0] + HAND or v < 0.5:
            continue
        a = R.pose(t0)
        s_e = float(TX.poly_proj(R.LP, a[None, :2])[1][0])
        steps = [j for j in range(len(tg)) if t0 <= tg[j] <= t0 + 4e6]
        X = np.concatenate([np.c_[R.tr[j]["x"], R.tr[j]["y"]] for j in steps]) if steps else np.zeros((0, 2))

        def lead(rest):
            if not np.isfinite(rest) or not len(X):
                return False
            q = s_e + rest + np.arange(LEAD_LO, LEAD_HI + 1e-9, 1.0)
            Pp = np.c_[np.interp(q, R.LS, R.LP[:, 0]), np.interp(q, R.LS, R.LP[:, 1])]
            return bool((np.hypot(X[:, None, 0] - Pp[None, :, 0], X[:, None, 1] - Pp[None, :, 1]) < LEAD_LAT).any())
        lab = label_row(t0 * 1e-6, s_e, T, S, v, lead, h)
        out.append(dict(scene=s, t0=t0, k=r.get("k", -1), v0=v, s_e=s_e, s_end=float(S[-1]), poses=np.asarray(r.get("poses_model") or r["poses"], np.float32), **lab))
    return out


def a0_decisions(workers):
    import pai_gap as PG
    rolls = PG.rolls(A0TAX)
    D = PG.drive(A0)
    jobs = [(s, rolls[s], D[s]) for s in sorted(set(rolls) & set(D))]
    with ProcessPoolExecutor(workers) as ex:
        return [x for out in ex.map(_a0_job, jobs, chunksize=1) for x in out]


def score(poses, rest, v0, ok):
    """The trainer's function on stored plans -> fire (n,), s_plan_8 (n,)."""
    import torch
    _, pr = NF.stoppt_terms(torch.as_tensor(np.asarray(poses, np.float32)), torch.as_tensor(NF.stoppt_cap(rest, v0)), torch.as_tensor(np.asarray(ok, bool)), parts=True)
    return pr["fire"].numpy(), pr["s_plan"].numpy()


def line_poses(arcs):
    """(n, 8) arcs -> straight poses with those arcs (the term reads only the plan's arc)."""
    a = np.nan_to_num(np.asarray(arcs, np.float32))
    return np.stack([np.maximum.accumulate(a, 1), np.zeros_like(a), np.zeros_like(a)], -1)


def share(x, m):
    return dict(n=int(m.sum()), fires=int((x & m).sum()), share=float((x & m).sum() / max(m.sum(), 1)))


def cmd_gate(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(S_ / "labels.npz"))
    rest, v0, ok = z["rest"], z["v0"], z["ok"]
    src = z["src"].astype(str)
    G = {}
    # G1 human paths on the pool rows
    g1 = {}
    for nm, arcs in (("station", z["hs8"]), ("clock", z["hc8"])):
        f, _ = score(line_poses(arcs), rest, v0, ok)
        for s in ("nu", "pai"):
            g1[f"{s}_{nm}"] = share(f, (src == s) & ok)
    for nm in ("fut", "fut_reg"):                                         # description: the pool's own time-matched targets
        f, _ = score(z[nm], rest, v0, ok)
        for s in ("nu", "pai"):
            g1[f"{s}_{nm}_describe"] = share(f, (src == s) & ok & np.isfinite(z[nm]).all((1, 2)))
    # stored plans on the pool rows (description) and G3 on the nuPlan S rows
    fp, sp = score(z["poses"], rest, v0, ok)
    lab = src == "nu"
    mS = np.char.startswith(z["names"].astype(str), "on:tv:") == False  # noqa: E712
    no_stop = np.isfinite(z["vmin8"]) & (z["vmin8"] >= NF.V_REST)
    G["pool_stored_plans"] = {s: dict(qualifying=share(ok, (src == s) & mS), fire=share(fp, (src == s) & mS),
                                      fire_ahead_of_human=share(fp, (src == s) & mS & (z["dt_h"] > 0)), fire_behind=share(fp, (src == s) & mS & (z["dt_h"] <= 0)),
                                      cap_raised=int(((src == s) & ok & (NF.stoppt_cap(rest, v0) > np.nan_to_num(rest, posinf=1e6) + NF.M_STOP + 1e-4)).sum()))
                              for s in ("nu", "pai")}
    # closed loop A0
    A = pd.DataFrame(a0_decisions(a.workers))
    sc = pd.read_csv(GAP / "scenes.csv").set_index("scene")
    A["sub"] = sc["sub"].reindex(A.scene).fillna("").values
    A["rest_wl"] = sc["rest_wl"].reindex(A.scene).fillna("").values
    P = np.stack(A.poses.values)
    rA, vA, oA = A.rest.values.astype(np.float64), A.v0.values, A.ok.values.astype(bool)
    A["fire"], A["s_plan8"] = score(P, rA, vA, oA)
    for nm, col in (("station", "hs8"), ("clock", "hc8")):
        f, _ = score(line_poses(np.stack(A[col].values)), rA, vA, oA)
        g1[f"a0_{nm}"] = share(f, oA)
    G["G1"] = dict(pass_=all(v["fires"] == 0 for k, v in g1.items() if not k.endswith("_describe")), **g1)
    re_ = (A["sub"] == "cor_route_end").values
    over = re_ & (A.s_e.values < A.s_end.values) & (A.s_plan8.values > A.s_end.values - A.s_e.values + NF.M_STOP)
    fire = A.fire.values.astype(bool)
    wl = (A.rest_wl.values != "")
    per = []
    for s, x in A[over].groupby("scene"):
        f = x[x.fire.astype(bool)]
        i = f.index[0] if len(f) else None
        per.append(dict(scene=s, decisions=len(x), fires=int(x.fire.sum()), wl=x.rest_wl.iloc[0],
                        first_fire_dist=float(x.s_end[i] - x.s_e[i]) if i is not None else None, first_fire_v=float(x.v0[i]) if i is not None else None,
                        in_time_3=bool(i is not None and x.s_end[i] - x.s_e[i] >= x.v0[i] ** 2 / 6.0),
                        why=dict(Counter(x.why[~x.fire.astype(bool)]).most_common())))
    G["G2"] = dict(**share(fire, over), stop_line=share(fire, over & wl), scenes=len(per), scenes_fired=sum(p["fires"] > 0 for p in per),
                   scenes_in_time_3=sum(p["in_time_3"] for p in per), why_not=dict(Counter(A.why.values[over & ~fire]).most_common()), per_scene=per)
    G["G2"]["pass_"] = G["G2"]["share"] >= 0.5
    noA = np.isfinite(A.vmin8.values) & (A.vmin8.values >= NF.V_REST)
    g3 = dict(a0=share(fire, noA), a0_rolling=share(fire, noA & (A.vmin8.values < 3.0)), a0_turning=share(fire, noA & (A.turn8.values > 30)),
              nu=share(fp, lab & mS & no_stop), nu_rolling=share(fp, lab & mS & no_stop & (z["vmin8"] < 3.0)),
              nu_turning=share(fp, lab & mS & no_stop & (z["turn8"] > 30)), pai_pool=share(fp, (src == "pai") & no_stop))
    G["G3"] = dict(pass_=g3["a0"]["share"] < 0.02 and g3["nu"]["share"] < 0.02, **g3)
    G["a0_all"] = dict(decisions=len(A), qualifying=int(oA.sum()), fire=int(fire.sum()), why=dict(Counter(A.why).most_common()))
    G["gate"] = dict(pass_=G["G1"]["pass_"] and G["G2"]["pass_"] and G["G3"]["pass_"], G4="results/stoppt/identity.json (separate job)")
    A.drop(columns=["poses", "hs8", "hc8"]).to_parquet(out / "a0.parquet")
    (out / "gate.json").write_text(json.dumps(G, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate.json").write_text(json.dumps(G, indent=1, default=float))
    print(json.dumps({k: v for k, v in G.items() if k != "G2"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in G["G2"].items() if k != "per_scene"}, default=float))
    sys.exit(0 if G["gate"]["pass_"] else 2)


# ---------------------------------------------------------------- probe
def probe_rows():
    """-> per dataset: dict of arrays (cluster, src, scene / log, t0, k, kin (n, 20), tau, cue, pos / neg, rest, v0) and the token / feature
    sources (cache tab rows or on1_JT rows)."""
    out = {}
    # PAI: the two caches, labels from the row files
    L = []
    for nm, p in PCACHE.items():
        zz = np.load(p / "tab.npz")
        z = {k: zz[k] for k in ("scenes", "sid", "v0", "usable", "after_event", "after_end", "e0", "k", "t0_us", "ego_in")}
        sc = z["scenes"][z["sid"]].astype(str)
        v0 = z["v0"].astype(np.float64)
        inc = z["usable"] & ~z["after_event"] & ~z["after_end"] & (v0 >= V_PROBE) & (np.abs(np.nan_to_num(z["e0"], nan=99)) <= 3.0)
        r = np.flatnonzero(inc)
        res = pai_rows([f"{sc[i]}:{int(z['k'][i])}" for i in r], v0[r], PROWS[nm], 32)
        L.append(dict(src=np.full(len(r), nm), row=r, cluster=sc[r], t0=z["t0_us"][r], k=z["k"][r], v0=v0[r], kin=z["ego_in"][r].astype(np.float32),
                      rest=np.array([x["rest"] for x in res]), lead=np.array([x["lead"] for x in res]), line=np.array([x["line"] for x in res])))
    out["pai"] = {k: np.concatenate([x[k] for x in L]) for k in L[0]}
    # nuPlan: on1_JT S rows, labels.npz
    z = np.load(S_ / "labels.npz")
    tb = np.load(CR / "on1_JT/tab.npz")
    n = len(tb["names"])
    S = tb["mode"].astype(str) == "self"
    v0 = tb["speed"].astype(np.float64)
    rest = z["rest"][:n].astype(np.float64)
    tau = rest / np.maximum(v0, 1e-3)
    r = np.flatnonzero(S & (v0 >= V_PROBE) & np.isfinite(z["h4c"][:n]))
    pos = (tau[r] >= TAU_LO) & (tau[r] <= TAU_HI)
    neg = ~(rest[r] <= TAU_NEG * v0[r])
    hsh = np.array([zlib.crc32(x.encode()) % 1000003 for x in tb["names"][r].astype(str)])
    keep = pos | (neg & (np.argsort(np.argsort(np.where(neg, hsh, 1 << 30))) < NNEG_NU))
    r = r[keep]
    out["nu"] = dict(src=np.full(len(r), "JT"), row=r, cluster=tb["log"][r].astype(str), t0=np.zeros(len(r)), k=np.zeros(len(r), int), v0=v0[r],
                     kin=tb["ego"][r].astype(np.float32), rest=rest[r], lead=z["lead"][:n][r], line=np.zeros(len(r), bool))
    for d in out.values():
        d["tau"] = d["rest"] / np.maximum(d["v0"], 1e-3)
        d["pos"] = (d["tau"] >= TAU_LO) & (d["tau"] <= TAU_HI)
        d["neg"] = ~(d["rest"] <= TAU_NEG * d["v0"])
        d["cue"] = np.where(d["lead"], "lead", np.where(d["line"], "line", "other"))
    return out


def cmd_pfeats(a):
    import torch
    sys.path[:0] = [str(REPO / "lib"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/route_mem/lib"),
                    str(REPO / "experiments/op_adapt_r2/lib"), str(REPO / "experiments/onpol_pai/lib")]
    import rmem
    from jevdrive.run import Run
    dev = torch.device("cuda")
    F = S_ / "probe/feats"
    F.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "stoppt-pfeats", config=vars(a)) as run:
        rows = probe_rows()
        np.savez(F / "rows.npz", **{f"{d}|{k}": v for d, x in rows.items() for k, v in x.items()})
        m = rmem.load_pmodel(PROBE_TAG, dev)
        sl = m.net.slices
        cols = dict(lead=np.arange(sl["lead"].start, sl["lead"].start + (sl["lead"].stop - sl["lead"].start) // 2),
                    lp=np.arange(sl["lead_prob"].start, sl["lead_prob"].stop), plan=np.arange(sl["plan"].start, sl["plan"].start + 495))
        for d, x in rows.items():
            n = len(x["row"])
            if d == "pai":
                srcs = {nm: (np.load(PCACHE[nm] / "tab.npz")["tix"], np.load(PCACHE[nm] / "tab.npz")["lht"], np.load(PCACHE[nm] / "tok.npy", mmap_mode="r"))
                        for nm in PCACHE}
                get = lambda i: (lambda tx, lh, tok: (tok[tx[x["row"][i]]], bool(lh[x["row"][i]])))(*srcs[str(x["src"][i])])  # noqa: E731
            else:
                fr = np.load(CR / "on1_JT@warp/front.npy", mmap_mode="r")
                get = lambda i: (fr[x["row"][i]], False)  # noqa: E731
            tn = np.lib.format.open_memmap(F / f"{d}_toknew.npy", "w+", np.float16, (n, 32, 512))
            feat = {k: np.empty((n, len(c)), np.float32) for k, c in cols.items()}
            for i in range(0, n, 256):
                q = [get(j) for j in range(i, min(i + 256, n))]
                H = torch.from_numpy(np.stack([np.asarray(h, np.float16) for h, _ in q])).to(dev)
                tn[i:i + len(q)] = H[:, 7].cpu().numpy()
                tc = torch.as_tensor(np.array([[0.0, 1.0] if l else [1.0, 0.0] for _, l in q], np.float32), device=dev)
                with torch.no_grad():
                    o = m(H, torch.from_numpy(x["kin"][i:i + len(q)]).to(dev), tc).float().cpu().numpy()
                for k, c in cols.items():
                    feat[k][i:i + len(q)] = o[:, c]
            tn.flush()
            np.savez(F / f"{d}_{PROBE_TAG}.npz", **feat)
            run.info(f"{d}: {n} rows, pos {int(x['pos'].sum())}, neg {int(x['neg'].sum())}")
        run.summary["rows"] = {d: len(x["row"]) for d, x in rows.items()}


def cmd_probe(a):
    import hashlib
    import rf_stopprobe as SP
    from jevdrive.data import splits
    from jevdrive.run import Run
    F = S_ / "probe/feats"
    out = S_ / "probe"
    with Run("defect_fix", "stoppt-probe", config=vars(a)) as run:
        for sid in ("physical_ai_av/test281", "physical_ai_av/practice-pai", "alpasim_nuplan/onpol-train"):
            run.use_split(splits.load(sid))
        R = np.load(F / "rows.npz")
        J = dict(tables=[], named=[], rows={})
        for d in ("pai", "nu"):
            x = {k.split("|", 1)[1]: R[k] for k in R.files if k.startswith(d + "|")}
            q = np.load(F / f"{d}_{PROBE_TAG}.npz")
            kin = x["kin"]
            Fs = {"KIN": kin, "LEAD": np.c_[kin, q["lead"], q["lp"]], "PLAN": np.c_[kin, q["plan"]]}
            TOK = np.load(F / f"{d}_toknew.npy", mmap_mode="r")
            Fs["TOK-lin"] = np.c_[kin, np.asarray(TOK, np.float32).mean(1)]
            TOK = np.ascontiguousarray(TOK)
            lab = x["pos"] | x["neg"]
            y = x["pos"].astype(int)
            fold = np.array([int(hashlib.sha1(c.encode()).hexdigest(), 16) % SP.FOLDS for c in x["cluster"]])
            Sc, TAU = {}, {}
            for fk in list(Fs) + ["TOK"]:
                sc, tau = np.full(len(y), np.nan), np.zeros(SP.FOLDS)
                for k in range(SP.FOLDS):
                    tr, te = (fold != k) & lab, fold == k
                    if fk == "TOK":
                        s_tr, s_te = SP.fit_tok(TOK[tr], kin[tr], y[tr], TOK[te], kin[te])
                    else:
                        s_tr, s_te = SP.fit_lin(Fs[fk][tr], y[tr], Fs[fk][te])
                    sc[te] = s_te
                    tau[k] = np.quantile(s_tr[y[tr] == 0], 1 - SP.FPR)
                Sc[fk], TAU[fk] = sc, tau
                run.info(f"{d} {fk} done")
            np.savez(out / f"scores_{d}.npz", **Sc, fold=fold)
            J["rows"][d] = dict(rows=len(y), pos=int(x["pos"].sum()), neg=int(x["neg"].sum()), clusters=len(set(x["cluster"])))
            srcs = {"own (A0R + P160)": np.ones(len(y), bool), "A0R": x["src"] == "A0R", "P160": x["src"] == "P160",
                    "own, after handoff (k >= 17)": x["k"] >= 17} if d == "pai" else {"own (on1_JT S)": np.ones(len(y), bool)}
            bins = {"tau [2, 4]": (TAU_MID, TAU_HI), "tau [1.5, 2)": (TAU_LO, TAU_MID - 1e-6), "tau [1.5, 4]": (TAU_LO, TAU_HI)}
            strata = {"all": None, "lead": ["lead"], "no lead": ["line", "other"]} | ({"line": ["line"], "other": ["other"]} if d == "pai" else {})
            for sn, sm in srcs.items():
                for bn, (lo, hi) in bins.items():
                    P_ = x["pos"] & (x["tau"] >= lo) & (x["tau"] <= hi)
                    for tn, tv in strata.items():
                        keep = sm & (x["neg"] | (P_ & (np.ones(len(y), bool) if tv is None else np.isin(x["cue"], tv))))
                        yy = P_[keep]
                        row = dict(data=d, source=sn, bin=bn, stratum=tn, n_pos=int(yy.sum()), n_neg=int((~yy).sum()))
                        if yy.sum() >= 5:
                            res, dr = SP.boot_auc({fk: Sc[fk][keep] for fk in Sc}, yy, x["cluster"][keep])
                            row |= dict(n_pos_clusters=int(len(np.unique(x["cluster"][keep][yy]))), auc=res)
                            for b in ("KIN", "LEAD"):
                                dd = dr["TOK"] - dr[b]
                                row[f"TOK-{b}"] = (res["TOK"][0] - res[b][0], float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975)))
                        J["tables"].append(row)
            if d == "pai":                                                # the 23 A0 overrun scenes: first fire before the human's stop
                import pandas as pd
                sc_ = pd.read_csv(GAP / "scenes.csv").set_index("scene")
                re_ = set(sc_.index[sc_["sub"] == "cor_route_end"])
                for s in sorted(re_):
                    r = np.flatnonzero((x["src"] == "A0R") & (x["cluster"] == s))
                    r = r[np.argsort(x["t0"][r])]
                    r = r[np.isfinite(x["rest"][r])]
                    e = dict(scene=s, rows=int(len(r)), wl=str(sc_.loc[s, "rest_wl"]) if isinstance(sc_.loc[s, "rest_wl"], str) else "",
                             lead=bool(sc_.loc[s, "rest_lead"]))
                    for fk in ("TOK", "LEAD", "KIN"):
                        on = Sc[fk][r] >= TAU[fk][fold[r][0]] if len(r) else np.zeros(0, bool)
                        run_ = np.convolve(on.astype(int), np.ones(SP.K_FIRE, int), "valid") == SP.K_FIRE if len(on) >= SP.K_FIRE else np.zeros(0, bool)
                        if run_.any():
                            i = r[int(np.argmax(run_))]
                            e[fk] = dict(d=float(x["rest"][i]), v=float(x["v0"][i]), ok=bool(x["rest"][i] >= x["v0"][i] ** 2 / (2 * SP.A_COMF)))
                        else:
                            e[fk] = dict(d=None, v=None, ok=False)
                    J["named"].append(e)
        prim = next(t for t in J["tables"] if t["data"] == "pai" and t["source"].startswith("own (") and t["bin"] == "tau [2, 4]" and t["stratum"] == "no lead")
        line = next(t for t in J["tables"] if t["data"] == "pai" and t["source"].startswith("own (") and t["bin"] == "tau [2, 4]" and t["stratum"] == "line")
        rd = lambda t: "auc" in t and t["TOK-LEAD"][0] >= 0.05 and t["TOK-LEAD"][1] > 0  # noqa: E731
        J["verdict"] = ("readable beyond kinematics + lead" if rd(prim) else "readable at stop lines only" if rd(line)
                        else "not readable beyond kinematics + lead: the approach part is an input problem; the term can only learn a prior")
        J["primary"] = prim
        (out / "probe.json").write_text(json.dumps(J, indent=1, default=float))
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "probe.json").write_text(json.dumps(J, indent=1, default=float))
        run.summary.update(verdict=J["verdict"])
        print(J["verdict"])


def cmd_report(a):
    J = json.load(open(a.json or (RES / "probe.json")))
    fks = ("KIN", "LEAD", "PLAN", "TOK-lin", "TOK")
    f = lambda t: f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"  # noqa: E731
    L = ["| data | source | bin | stratum | pos (clusters) / neg | " + " | ".join(fks) + " | TOK - KIN | TOK - LEAD |", "|---|---|---|---|--:|" + "--:|" * (len(fks) + 2)]
    for r in J["tables"]:
        if "auc" not in r:
            L.append(f"| {r['data']} | {r['source']} | {r['bin']} | {r['stratum']} | {r['n_pos']} / {r['n_neg']} | " + " | ".join("-" for _ in fks) + " | - | - |")
            continue
        L.append(f"| {r['data']} | {r['source']} | {r['bin']} | {r['stratum']} | {r['n_pos']} ({r['n_pos_clusters']}) / {r['n_neg']} | " + " | ".join(f(r["auc"][k]) for k in fks)
                 + f" | {r['TOK-KIN'][0]:+.3f} [{r['TOK-KIN'][1]:+.2f}, {r['TOK-KIN'][2]:+.2f}] | {r['TOK-LEAD'][0]:+.3f} [{r['TOK-LEAD'][1]:+.2f}, {r['TOK-LEAD'][2]:+.2f}] |")
    L += ["", "| scene | wait line | lead | rows | " + " | ".join(f"{k} first fire d m @ v" for k in ("TOK", "LEAD", "KIN")) + " |", "|---|---|---|--:|---|---|---|"]
    for e in J["named"]:
        c = [f"{e[k]['d']:.0f} @ {e[k]['v']:.1f}{' ok' if e[k]['ok'] else ' late'}" if e[k]["d"] is not None else "never" for k in ("TOK", "LEAD", "KIN")]
        L.append(f"| {e['scene'][7:15]} | {e['wl'] or '-'} | {'y' if e['lead'] else ''} | {e['rows']} | " + " | ".join(c) + " |")
    L += ["", "in time: " + ", ".join(f"{k} {sum(e[k]['ok'] for e in J['named'])} / {len(J['named'])}" for k in ("TOK", "LEAD", "KIN")), "", f"verdict: {J['verdict']}"]
    print("\n".join(L))


# ---------------------------------------------------------------- pilot
def cmd_smoke(a):
    """The smoke run's events: finite losses, the pool term loaded and firing on some step -> JSON; exit 2 on failure."""
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{a.run}/*/"))[-1]
    sc = defaultdict(list)
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc[e["tag"]].append(e["value"])
    pos, ho = sc.get("loss/on/lon_stoppt_pos", []), sc.get("loss/on/lon_stoppt_ho", [])
    fin = all(np.isfinite(v).all() for k, v in sc.items() if k.startswith("loss/"))
    ok = bool(fin and len(pos) and max(pos) > 0 and len(ho))
    out = dict(run=a.run, dir=d, steps=len(sc.get("loss/total", [])), finite=fin, fire_share_mean=float(np.mean(pos)) if pos else None,
               fire_share_max=float(max(pos)) if pos else None, term_mean=float(np.mean(ho)) if ho else None, pass_=ok)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if ok else 2)


def cmd_repro(a):
    """The probe's A0 rerun against A0: per-scene score equality (description)."""
    import pai_gap as PG
    x, _ = PG.summaries(A0)
    y, _ = PG.summaries(A0R / "roll")
    c = sorted(set(x) & set(y))
    same = [s for s in c if abs(float(x[s]["score"] or 0) - float(y[s]["score"] or 0)) < 1e-9]
    out = dict(scenes_a0=len(x), scenes_rerun=len(y), common=len(c), identical=len(same), differ=sorted(set(c) - set(same)),
               mean_a0=float(np.mean([float(x[s]["score"] or 0) for s in c])), mean_rerun=float(np.mean([float(y[s]["score"] or 0) for s in c])))
    (RES / "a0r_repro.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "differ"}))


def cmd_overrun(a):
    import rf_tax as TX
    from pai281 import ptype, zclass
    out = {}
    for x in a.tax:
        nm, p = x.split("=", 1)
        n, sc = 0, []
        for o in pickle.load(open(p, "rb"))["rolls"]:
            if "ctrl" not in o:
                continue
            s = float(o["score"] or 0.0)
            n += 1
            if s > 0:
                continue
            row, _, _ = TX.classify(o)
            if zclass(ptype(dict(score=s, metrics=o["metrics"], failure_reason=""))) == "corridor" and row.get("primary") == "cor_route_end":
                sc.append(o["scene"])
        out[nm] = dict(rollouts=n, overrun=len(sc), scenes=sorted(sc))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v["rollouts"], v["overrun"]) for k, v in out.items()}))


def cmd_pilot(a):
    import df
    import rtja
    W = {t: rtja.wait_stats(t) for t in (a.tag, a.ctl)}
    H = {t: df.cl_read(t, [a.ctl, "RSG-L18-s0"] if t == a.tag else ["RSG-L18-s0"])["held"] for t in (a.tag, a.ctl)}
    Ar = json.load(open(a.arena))[a.ctl]
    Pg = json.load(open(a.pai))
    Ov = json.load(open(a.overrun))
    P, C = W[a.tag], W[a.ctl]
    LEAD = rtja.LEAD
    pp = Pg["paired_pilot_minus_control"]
    g = dict(held_zeros=H[a.tag]["zeros"] <= H[a.ctl]["zeros"] + 2,
             **{f"delay_{c}": P[c]["delay_med"] <= C[c]["delay_med"] + 0.15 for c in LEAD},
             **{f"launched_{c}": P[c]["launched"] >= 0.95 * C[c]["launched"] for c in LEAD},
             held_front=H[a.tag]["coll"]["front"] <= H[a.ctl]["coll"]["front"] + 2, arena_af_front=Ar["af_front_a"] <= Ar["af_front_b"] + 2,
             pai_half=pp["mean"] >= 0.01, arena=Ar["diff"] >= 0)
    verdict = "PASS" if all(g.values()) else "FAIL"
    hp = H[a.tag]["pair"].get(a.ctl, {})
    yn = lambda x: "ok" if x else "FAIL"  # noqa: E731
    try:
        import stand
        cs = stand.cue_strata(a.ext, a.tag, a.ctl)
        cs = cs[0] if isinstance(cs, tuple) else cs                          # STAND's later version also returns counts
    except Exception as e:  # noqa: BLE001  description only
        cs = {"error": str(e)}
    oc, oa = Ov[a.ctl], Ov[a.tag]
    dr_line, sc_lines = [], []
    if a.drift:
        dr = next(iter(json.load(open(a.drift)).values()))
        dr_line = ["- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): " + ", ".join(
            f"{k} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] ({v['n']})" for k, v in dr.items())]
    if a.pai_dirs:
        import pai_gap as PG
        sc_c, sc_a = (PG.summaries(d)[0] for d in a.pai_dirs.split(","))
        sc_lines = ["", "PAI half overrun zeros (cor_route_end) by scene, score control / arm:", ""] + [
            f"- {x}: {float(sc_c[x]['score'] or 0):.3f} / {float(sc_a[x]['score'] or 0):.3f}"
            f"{' (control overrun)' if x in oc['scenes'] else ''}{' (arm overrun)' if x in oa['scenes'] else ''}"
            for x in sorted(set(oc["scenes"]) | set(oa["scenes"]))]
    L = [f"# NR1STOPPT pilot ({a.tag} against {a.ctl}; NR1 prereg supplement 18; native reads)", "",
         f"| | {a.ctl} (control) | {a.tag} | line |", "|:--|--:|--:|:--|",
         f"| held 460 zeros | {H[a.ctl]['zeros']} | {H[a.tag]['zeros']} | <= control + 2: {yn(g['held_zeros'])} |",
         *[f"| b median launch delay {c} (s) | {C[c]['delay_med']:+.2f} | {P[c]['delay_med']:+.2f} | <= control + 0.15: {yn(g['delay_' + c])} |" for c in LEAD],
         *[f"| b launched {c} | {C[c]['launched']:.3f} | {P[c]['launched']:.3f} | >= 0.95 x control: {yn(g['launched_' + c])} |" for c in LEAD],
         f"| held 460 front collisions | {H[a.ctl]['coll']['front']} | {H[a.tag]['coll']['front']} | <= control + 2: {yn(g['held_front'])} |",
         f"| arena subset AF-front windows | {Ar['af_front_b']} | {Ar['af_front_a']} | <= control + 2: {yn(g['arena_af_front'])} |",
         f"| PAI half (4 chunks) mean / zeros | {Pg['control']['mean']:.4f} / {Pg['control']['zeros']} | {Pg['pilot']['mean']:.4f} / {Pg['pilot']['zeros']} | "
         f"paired {pp['mean']:+.4f} [{pp['lo']:+.4f}, {pp['hi']:+.4f}] >= +0.01: {yn(g['pai_half'])} |",
         f"| PAI half d381 overrun zeros (cor_route_end) | {oc['overrun']} | {oa['overrun']} | description |",
         f"| arena subset ({Ar['windows']} windows) mean / zeros | {Ar['mean_b']:.4f} / {Ar['zeros_b']} | {Ar['mean_a']:.4f} / {Ar['zeros_a']} | "
         f"paired {Ar['diff']:+.4f} [{Ar['lo']:+.4f}, {Ar['hi']:+.4f}] >= 0: {yn(g['arena'])} |",
         "", "Description:", "",
         *[f"- a wait {c}: {C[c]['wait']:.3f} -> {P[c]['wait']:.3f}" for c in LEAD],
         f"- overrun scenes fixed: {sorted(set(oc['scenes']) - set(oa['scenes']))}; new: {sorted(set(oa['scenes']) - set(oc['scenes']))}",
         f"- held 460 mean {H[a.ctl]['mean']:.4f} -> {H[a.tag]['mean']:.4f}, paired {hp.get('mean', float('nan')):+.4f} [{hp.get('lo', float('nan')):+.4f}, "
         f"{hp.get('hi', float('nan')):+.4f}]",
         f"- arena subset k = 9 speed ratio: {Ar['k9']['b']:.4f} -> {Ar['k9']['a']:.4f}, diff {Ar['k9']['diff']:+.4f} [{Ar['k9']['ci'][0]:+.4f}, {Ar['k9']['ci'][1]:+.4f}]",
         "- arena subset strata (control -> arm, zeros): " + "; ".join(f"{s} ({v['windows']}) {v['b']:.3f} -> {v['a']:.3f}, {v['zeros_b']} -> {v['zeros_a']}"
                                                                     for s, v in Ar["strata"].items()),
         "- arena subset by cue (arm - control): " + "; ".join(f"{k} ({v['windows']}) {v['b']:.3f} -> {v['a']:.3f}, {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
                                                              for k, v in cs.items() if isinstance(v, dict)),
         "- arena subset zero types (control / arm): " + json.dumps(Ar["types"]["b"]) + " / " + json.dumps(Ar["types"]["a"]),
         *dr_line, *sc_lines,
         "", "Gates: " + ", ".join(f"{k} {yn(v)}" for k, v in g.items()), "", f"**Verdict: {verdict}**", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L))
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(tag=a.tag, ctl=a.ctl, wait=W, held=H, arena=Ar, pai=Pg, overrun=Ov, cue=cs, gates=g,
                                                                verdict=verdict), indent=1, default=str))
    print("\n".join(L))
    print("VERDICT", verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("labels")
    p.add_argument("--workers", type=int, default=32)
    p = sp.add_parser("assemble")
    p.add_argument("--pool", required=True)
    sp.add_parser("weights")
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=24)
    sp.add_parser("pfeats")
    sp.add_parser("probe")
    p = sp.add_parser("report")
    p.add_argument("--json", default="")
    p = sp.add_parser("smoke")
    p.add_argument("--run", required=True), p.add_argument("--out", required=True)
    sp.add_parser("repro")
    p = sp.add_parser("overrun")
    p.add_argument("--tax", nargs="+", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("pilot")
    for k in ("tag", "ctl", "arena", "pai", "overrun", "ext", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--drift", default=""), p.add_argument("--pai-dirs", default="")
    sp.add_parser("weightsf")
    a = ap.parse_args()
    dict(labels=cmd_labels, assemble=cmd_assemble, weights=cmd_weights, gate=cmd_gate, pfeats=cmd_pfeats, probe=cmd_probe, report=cmd_report,
         overrun=cmd_overrun, pilot=cmd_pilot, smoke=cmd_smoke, repro=cmd_repro, weightsf=cmd_weightsf)[a.cmd](a)
