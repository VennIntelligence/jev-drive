#!/usr/bin/env python3
"""ROUND3 (plans/2026-10-21-round3-prereg.md): where does round 2's launch loss come from? Offline diagnosis, the J-W draw file and the read-outs.
Box, repo root (op-train env for the pool rows, .venv for the read stores; both have numpy / pandas).

  r3.py rows                      state of every PAI human row (plain + both stretches) by its own fed speed v0 and 4 s label: standing
        (v0 < 0.3 m/s and the label path < 0.5 m), launching (v0 < 1 m/s, path >= 0.5 m), stopping (v0 >= 1, the label's last 1 s < 0.5 m),
        moving; for standing rows the human's wait after the label window (plain-row sequence of the scene, 10 Hz); counted over R2JF's
        drawable rows and as expected rows per batch; the same states for the 32 S / trainval pool rows (JT00S) -> results/rows_state.{md,json}
  r3.py launch                    on the launch scenes (handoff < 1 m/s, log moves > 5 m) of the stored reads: the human's launch time t_L
        (log 0.5 m past its handoff pose), the read's model plan (poses_model) at the first decision at or after t_L, the first decision whose
        plan covers >= 2 m, the ego's own launch -> results/launch_plans.{md,json,csv}
  r3.py weights --out R3JW        J-W: R2JF with every PAI human row whose own 4 s label path is < 0.5 m (standing + creeping) at weight 0, each part
        renormalised to its R2JF share -> $DATA_DIR/runs/defect_fix/w/R3JW.npz + results/w_R3JW.json
  r3.py weights --rule ns --out R3NS   round 3b NS: R2JF with both stretch parts at weight 0, the plain part scaled to the PAI human share (32 a batch)
  r3.py weights --rule mv --out R3MV   round 3b MV: R2JF with every PAI human row whose fed speed is < 2 m/s at weight 0, each part renormalised
  r3.py weights --rule mve --out R3MVE  round 3c MVE: R3MV restricted to PAI human rows with an exam event ahead (route turn: heading change
        >= 15 deg along the human's path over 4 s or the next 20 m, whichever reaches further; stop: the human < 0.5 m/s within the label or
        the next 4 s; stretch rows on their own clock, from their base row's stitched log future), each part renormalised to its share
  r3.py weights --rule mvl --out R3MVL  round 3c MVL: R3MV's pool weights unchanged + `nav_names` / `nav_w`: navtrain launch rows (fed < 0.5 m/s,
        the 4 s label leaves 0.5 m) at draw weight 3, every other navtrain row 1 (rm_train renormalises inside the normal rows)
  r3.py classes                   (round 3c, offline) expected rows a batch by class (launch / stand / stop-approach / turn / cruise, label-only
        rules) per arm (R2C, MV, MVL, MVE), the MVE event split of the PAI rows, and the 4 s label distance of moving rows by speed bin, PAI vs
        navtrain -> results/classes3c.{md,json}
  r3.py launchrows                (round 3b, offline) launch rows (fed speed < 0.5 m/s, the label leaves 0.5 m) per PAI part and in navtrain: share,
        4 s label distance, stretch / base-row distance ratio, expected rows per batch under R2JF / R3NS / R3MV -> results/launch_rows.{md,json}
  r3.py steps                     results/steps.json (both arms 9 500 steps = round 2's J and CTRL) -> $O/steps.json
  r3.py smoke --tag T --arm JW|JNF|NS|MV  the 60-step smoke's checks (finite scalars; pool imitation rows 0.75-1.25 x 32; no hinge on them; JW: flattened
        rows 0.6-1.4 x expected and p 0.5 recorded; JNF / NS / MV: no flattening; NS / MV: the drawn part shares) -> results/smoke_<arm>.json; exit 2 on a failure
  r3.py gate --arms TAG.. --ctrl R2C-L18-s0   the registered PAI-FIX native lines per arm against CTRL (stored facts): G-LAUNCH mean >= CTRL - 0.05
        (binding), T281 route-turn zeros <= CTRL - 5, TARGET (val160) mean >= CTRL - 0.02 -> results/<stem>.{json,md} (--stem gate); prints the passing tags
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, W = DATA / "runs/op_parity/cache", DATA / "runs/defect_fix/w"
O, RES = DATA / "runs/round3", REPO / "experiments/round3/results"
PARTS = {"ph": "on1_PAIH", "ph05": "on1_PAIH05", "ph15": "on1_PAIH15"}
V_STAND, L_STAND, V_LAUNCH = 0.3, 0.5, 1.0                     # the registered row-state rule (prereg)
V_LROW, V_MV = 0.5, 2.0                                        # round 3b: launch row = fed speed < 0.5 m/s and the label leaves 0.5 m; MV drops < 2 m/s
SHARES = dict(R2JF=dict(ph=0.25, ph05=0.125, ph15=0.125), R3NS=dict(ph=0.5, ph05=0.0, ph15=0.0), R3MV=dict(ph=0.25, ph05=0.125, ph15=0.125))
SHARES3C = dict(R3MVL=SHARES["R3MV"], R3MVE=SHARES["R3MV"])
TURN_DEG, TURN_M, T_LAB, V_STOP, NAV_UP = 15.0, 20.0, 4.0, 0.5, 3.0  # round 3c: event rule (prereg addendum); MVL's navtrain launch-row weight
K_ON, K_S = 64, 32                                             # J: on-k 64 (w sums to 1 over S + PAI rows); expected S rows per batch 32


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def path_len(fut):
    """fut (n, 8, 3) poses at 0.5 .. 4 s -> (n,) path length from the origin, (n,) length of the last 1 s."""
    P = np.concatenate([np.zeros((len(fut), 1, 2), np.float32), fut[:, :, :2]], 1)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    return seg.sum(1), seg[:, -2:].sum(1)


def state(v0, fut):
    L, L1 = path_len(fut)
    st = np.full(len(v0), "moving", dtype="U9")
    st[(v0 >= V_LAUNCH) & (L1 < 0.5)] = "stopping"
    st[(v0 < V_LAUNCH) & (L >= L_STAND)] = "launching"
    st[(v0 < V_STAND) & (L < L_STAND)] = "standing"
    st[(v0 >= V_STAND) & (v0 < V_LAUNCH) & (L < L_STAND)] = "creeping"
    return st, L


def wait_after(tab):
    """Plain rows: per row the log time (s) from the row's t until the human has moved 0.5 m (inf if not within the clip's rows + the last
    row's label), minus 4 s (the label window)."""
    names, log, v = tab["names"].astype(str), tab["log"].astype(str), tab["speed"].astype(np.float64)
    k = np.array([int(x.rsplit(":", 1)[1]) for x in names])
    fut = tab["fut"]
    out = np.full(len(names), np.nan)
    by = defaultdict(list)
    for i, s in enumerate(log):
        by[s].append(i)
    for s, ix in by.items():
        ix = np.array(ix)[np.argsort(k[ix])]
        kk = k[ix]
        s_ = np.r_[0.0, np.cumsum(0.5 * (v[ix][1:] + v[ix][:-1]) * 0.1 * np.diff(kk))]  # station along the 10 Hz plain rows
        L, _ = path_len(fut[ix[-1:]])
        ext_t = kk[-1] * 0.1 + np.arange(1, 9) * 0.5                                 # past the last row: its label
        Pl = np.concatenate([[[0, 0]], fut[ix[-1], :, :2]])
        ext_s = s_[-1] + np.cumsum(np.linalg.norm(np.diff(Pl, axis=0), axis=1))
        T_ = np.r_[kk * 0.1, ext_t]
        S_ = np.r_[s_, ext_s]
        for j, i in enumerate(ix):
            m = np.nonzero(S_ > S_[j] + 0.5)[0]
            out[i] = (T_[m[0]] - kk[j] * 0.1 - 4.0) if len(m) else np.inf
    return out


def wait_bucket(w):
    return np.select([w <= 2, w <= 6, w <= 10, np.isfinite(w)], ["<=2 s", "2-6 s", "6-10 s", ">10 s"], "not in clip")


def load_part(p):
    t = np.load(CR / PARTS[p] / "tab.npz")
    return {k: t[k] for k in t.files}


def pai_states():
    """-> {part: dict(names, state, L, wait)}; stretch rows take the wait of their plain base row scaled to the stretch clock (/ f)."""
    T = {p: load_part(p) for p in PARTS}
    out = {}
    w_plain = wait_after(T["ph"])
    wp = dict(zip(T["ph"]["names"].astype(str), w_plain))
    for p, t in T.items():
        st, L = state(t["speed"].astype(np.float64), t["fut"])
        if p == "ph":
            w = w_plain
        else:  # base row's wait (log s after its own 4 s window); the stretch label covers 4 f s of log
            w = np.array([wp.get(b, np.nan) for b in t["base"].astype(str)]) + 4.0 - 4.0 * t["f"]
        nm = t["names"].astype(str)
        out[p] = dict(names=np.array([f"on:{p}:{x[3:]}" for x in nm]), log=t["log"].astype(str), state=st, L=L, wait=w,  # merged-pool names
                      speed=t["speed"], fut=t["fut"])
    return out


def _compose(P, d):
    """P (3,) pose (x, y, yaw) ∘ d (n, 3) poses in P's frame -> (n, 3) in the outer frame."""
    c, s = np.cos(P[2]), np.sin(P[2])
    return np.stack([P[0] + c * d[:, 0] - s * d[:, 1], P[1] + s * d[:, 0] + c * d[:, 1], P[2] + d[:, 2]], 1)


def _scene_events(job):
    """One scene's plain rows (k sorted; fut (n, 8, 3), fed speed (n,)) -> per row and clock factor f: route turn ahead (max |heading change|
    >= TURN_DEG over the stitched human path up to 4 f s of log or TURN_M m, whichever reaches further) and stop ahead (the human below
    V_STOP / f m/s within 8 f s of log: label segment speeds and the later rows' fed speeds). The path is stitched from the row's label and the
    labels of the rows 4 s on (two hops, up to 12 s; a missing row: the latest row on the 0.5 s grid before it)."""
    k, fut, v, fs = job
    pos = {int(x): i for i, x in enumerate(k)}
    n = len(k)
    out = {f: (np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)) for f in fs}
    tl = np.arange(1, 9) * 0.5
    for i in range(n):
        T, Pq = [tl], [fut[i]]
        j, Pj, tj = i, np.zeros(3), 0.0
        for _ in range(2):
            m = next((m for m in range(40, 0, -5) if int(k[j]) + m in pos), None)
            if m is None:
                break
            nx = pos[int(k[j]) + m]
            Pj, tj, j = _compose(Pj, fut[j][m // 5 - 1:m // 5])[0], tj + m / 10, nx
            T.append(tj + tl), Pq.append(_compose(Pj, fut[j]))
        T, Pq = np.concatenate(T), np.concatenate(Pq)
        o = np.argsort(T, kind="stable")
        T, Pq = T[o], Pq[o]
        keep = np.r_[True, np.diff(T) > 1e-6]
        T, Pq = T[keep], Pq[keep]
        XY = np.concatenate([[[0.0, 0.0]], Pq[:, :2]])
        seg = np.linalg.norm(np.diff(XY, axis=0), axis=1)
        arc = np.cumsum(seg)
        sv = seg / np.diff(np.r_[0.0, T])                                  # segment speeds, at the segment's end time
        dh = np.abs((Pq[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        later = np.array([pos[x] for x in range(int(k[i]) + 1, int(k[i]) + 121) if x in pos], int)
        tv, vv = (k[later] - k[i]) / 10.0, v[later]
        for f in fs:
            win = (T <= T_LAB * f + 1e-6) | (arc <= TURN_M)
            mx = float(dh[win].max()) if win.any() else 0.0
            stop = bool(((sv < V_STOP / f) & (T <= 2 * T_LAB * f + 1e-6)).any() or ((vv < V_STOP / f) & (tv <= 2 * T_LAB * f + 1e-6)).any())
            out[f][0][i], out[f][1][i], out[f][2][i] = mx >= np.radians(TURN_DEG), stop, np.degrees(mx)
    return out


def pai_events():
    """-> {part: dict(turn, stop, dhead)} for every row of the three PAI parts (merged-pool order of pai_states); stretch rows take their base
    plain row's stitched log future on their own clock (windows x f, stop speed / f)."""
    from concurrent.futures import ProcessPoolExecutor
    from jevdrive.common import n_cpus
    T = {p: load_part(p) for p in PARTS}
    P = T["ph"]
    names, log = P["names"].astype(str), P["log"].astype(str)
    k = np.array([int(x.rsplit(":", 1)[1]) for x in names])
    fs = sorted({1.0} | {round(float(x), 3) for p in ("ph05", "ph15") for x in np.unique(T[p]["f"])})
    by = defaultdict(list)
    for i, s in enumerate(log):
        by[s].append(i)
    idx = [np.array(ix)[np.argsort(k[ix], kind="stable")] for ix in by.values()]
    with ProcessPoolExecutor(max(1, min(n_cpus(), 32))) as px:
        res = list(px.map(_scene_events, [(k[ix], P["fut"][ix].astype(np.float64), P["speed"][ix].astype(np.float64), fs) for ix in idx],
                          chunksize=4))
    ev = {f: (np.zeros(len(names), bool), np.zeros(len(names), bool), np.zeros(len(names))) for f in fs}
    for ix, r in zip(idx, res):
        for f in fs:
            for q in range(3):
                ev[f][q][ix] = r[f][q]
    out = {"ph": dict(turn=ev[1.0][0], stop=ev[1.0][1], dhead=ev[1.0][2])}
    row = {x: i for i, x in enumerate(names)}
    for p in ("ph05", "ph15"):
        b = np.array([row.get(x, -1) for x in T[p]["base"].astype(str)])
        assert (b >= 0).all(), f"{p}: base rows missing"
        f_ = np.array([round(float(x), 3) for x in T[p]["f"]])
        o = {q: np.zeros(len(b), bool) for q in ("turn", "stop")} | dict(dhead=np.zeros(len(b)))
        for f in np.unique(f_):
            m = f_ == f
            o["turn"][m], o["stop"][m], o["dhead"][m] = ev[f][0][b[m]], ev[f][1][b[m]], ev[f][2][b[m]]
        out[p] = o
    return out


def label_class(v0, fut):
    """Label-only row classes (every pool comparable): launch (v0 < 0.5, 4 s path >= 0.5 m), stand (v0 < 0.5, path < 0.5), stop-approach
    (v0 >= 0.5, some 0.5 s label segment below V_STOP m/s), turn (else, |heading| >= TURN_DEG at some label pose), cruise (else)."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[:, :, :2]], 1)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    L = seg.sum(1)
    dh = np.abs((fut[:, :, 2] + np.pi) % (2 * np.pi) - np.pi).max(1)
    c = np.full(len(v0), "cruise", dtype="U13")
    c[dh >= np.radians(TURN_DEG)] = "turn"
    c[(seg / 0.5 < V_STOP).any(1)] = "stop-approach"
    c[(v0 < V_LROW) & (L < L_STAND)] = "stand"
    c[(v0 < V_LROW) & (L >= L_STAND)] = "launch"
    return c, L


def navtrain_rows():
    """navtrain_full, all 12 shards: names, fed speed, 4 s label path, label class, launch-row mask (rows with a full label)."""
    nm, v, Ls, cl, ok = [], [], [], [], []
    for i in range(12):
        t = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        o = ~np.isnan(t["fut"]).any((1, 2))
        c = np.full(len(o), "nolabel", dtype="U13")
        L = np.full(len(o), np.nan)
        c[o], L[o] = label_class(t["speed"][o].astype(np.float64), t["fut"][o].astype(np.float64))
        nm.append(t["names"].astype(str)), v.append(t["speed"].astype(np.float64)), Ls.append(L), cl.append(c), ok.append(o)
    nm, v, L, c, ok = map(np.concatenate, (nm, v, Ls, cl, ok))
    assert len(set(nm.tolist())) == len(nm), "navtrain names not unique"
    return dict(names=nm, v0=v, L=L, cls=c, ok=ok, launch=ok & (v < V_LROW) & (np.nan_to_num(L) >= L_STAND))


STATES = ("standing", "creeping", "launching", "stopping", "moving")
BUCKETS = ("<=2 s", "2-6 s", "6-10 s", ">10 s", "not in clip")


def cmd_rows(a):
    S = pai_states()
    w = np.load(W / "R2JF.npz")
    wn = dict(zip(w["names"].astype(str), w["w"]))
    rec, L = dict(rule=dict(standing=f"v0 < {V_STAND} and label path < {L_STAND} m", creeping=f"{V_STAND} <= v0 < {V_LAUNCH}, path < {L_STAND} m",
                            launching=f"v0 < {V_LAUNCH} and path >= {L_STAND} m", stopping=f"v0 >= {V_LAUNCH} and the label's last 1 s < 0.5 m",
                            moving="else"), parts={}), []
    L += ["# ROUND3: PAI human rows by state (own fed speed and own 4 s label)", "",
          f"Rule: standing v0 < {V_STAND} m/s and label path < {L_STAND} m; creeping {V_STAND} <= v0 < {V_LAUNCH}, path < {L_STAND} m; launching v0 < "
          f"{V_LAUNCH}, path >= {L_STAND} m; stopping v0 >= {V_LAUNCH}, last 1 s of the label < 0.5 m; moving otherwise. `drawable` = R2JF weight > 0 "
          "(outside held and the exam links); `per batch` = expected rows of that state in J's batch (32 PAI human rows).", "",
          "| part | drawable | " + " | ".join(STATES) + " | standing per batch | launching per batch |", "|:--|--:|" + "--:|" * (len(STATES) + 2)]
    tot = defaultdict(float)
    for p, d in S.items():
        wt = np.array([wn.get(x, 0.0) for x in d["names"]])
        dr = wt > 0
        sh = wt.sum()
        c = {s: int(((d["state"] == s) & dr).sum()) for s in STATES}
        pb = {s: float(wt[d["state"] == s].sum() * K_ON) for s in STATES}
        for s in STATES:
            tot[s] += pb[s]
        stand = (d["state"] == "standing") & dr
        wb = wait_bucket(d["wait"][stand])
        rec["parts"][p] = dict(drawable=int(dr.sum()), share=float(sh), counts=c, per_batch=pb,
                               standing_wait={b: int((wb == b).sum()) for b in BUCKETS},
                               standing_scenes=int(len(set(d["log"][stand]))),
                               standing_rows_per_scene_med=float(np.median(np.unique(d["log"][stand], return_counts=True)[1])) if stand.any() else 0.0)
        L.append(f"| {p} | {int(dr.sum())} | " + " | ".join(f"{c[s]} ({c[s] / max(dr.sum(), 1):.1%})" for s in STATES)
                 + f" | {pb['standing']:.2f} | {pb['launching']:.2f} |")
    L.append(f"| all (per batch) | | " + " | ".join(f"{tot[s]:.2f}" for s in STATES) + f" | {tot['standing']:.2f} | {tot['launching']:.2f} |")
    rec["per_batch_all"] = dict(tot)
    L += ["", "Standing rows: the human's wait after the label window (log s until it has moved 0.5 m; stretch rows on their own clock)", "",
          "| part | standing scenes | " + " | ".join(BUCKETS) + " |", "|:--|--:|" + "--:|" * len(BUCKETS)]
    for p, r in rec["parts"].items():
        n = sum(r["standing_wait"].values())
        L.append(f"| {p} | {r['standing_scenes']} | " + " | ".join(f"{r['standing_wait'][b]} ({r['standing_wait'][b] / max(n, 1):.0%})" for b in BUCKETS) + " |")
    # the pool's S / trainval rows (both arms draw 32 of them a batch from JT00S)
    t = np.load(CR / "on1_JT/tab.npz")
    ws = np.load(W / "JT00S.npz")
    assert np.array_equal(ws["names"].astype(str), t["names"].astype(str))
    st, _ = state(t["speed"].astype(np.float64), t["fut"])
    wv = ws["w"] / ws["w"].sum()
    sb = {s: float(wv[st == s].sum() * K_S) for s in STATES}
    rec["s_rows_per_batch"] = sb
    L += ["", "The 32 S / trainval pool rows of a batch (JT00S, both arms): " + ", ".join(f"{s} {sb[s]:.2f}" for s in STATES), ""]
    nc, nn = defaultdict(int), 0                                 # navtrain_full (all 12 shards, rows with a full label): J draws 51 a batch, CTRL 83
    for i in range(12):
        t = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        ok = ~np.isnan(t["fut"]).any((1, 2))
        st, _ = state(t["speed"][ok].astype(np.float64), t["fut"][ok])
        for s_ in STATES:
            nc[s_] += int((st == s_).sum())
        nn += int(ok.sum())
    nv = {s_: nc[s_] / nn for s_ in STATES}
    rec["navtrain"] = dict(rows=nn, counts=dict(nc))
    L += [f"navtrain_full ({nn} rows): " + ", ".join(f"{s_} {nc[s_]} ({nv[s_]:.2%})" for s_ in STATES), ""]
    J_ = {s_: 51 * nv[s_] + sb[s_] + tot[s_] for s_ in STATES}
    C_ = {s_: 83 * nv[s_] + sb[s_] for s_ in STATES}
    rec["per_batch_J"], rec["per_batch_CTRL"] = J_, C_
    L += ["Per batch (navtrain 51 / 83 + 32 S rows + J's 32 PAI human rows; the 13 yaw-rate rows not counted), J vs CTRL: "
          + ", ".join(f"{s_} {J_[s_]:.2f} vs {C_[s_]:.2f}" for s_ in STATES)
          + f"; standing / (standing + launching) {J_['standing'] / (J_['standing'] + J_['launching']):.1%} vs {C_['standing'] / (C_['standing'] + C_['launching']):.1%}."]
    dump(RES / "rows_state.json", rec)
    (RES / "rows_state.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- launch-moment plans of stored reads
HAND = 1.6e6
READS = {  # name -> (read root, tax pkl, subset): the stored reads of round 2 and their references
    "R2J-281ag": ("runs/round2/pai281/R2J-L18-s0-ag", "runs/pai_fix/tax/R2J-L18-s0-281ag.pkl", "281"),
    "NTV0-281ag": ("runs/defect_fix/pai_accgap/NTV0-L18-s0", "runs/pai_fix/tax/NTV0-L18-s0-281ag.pkl", "281"),
    "R2J-vl": ("runs/pai_fix/reads/R2J-L18-s0-vl", "runs/pai_fix/tax/R2J-L18-s0-vl.pkl", "fix"),
    "R2C-vl": ("runs/pai_fix/reads/R2C-L18-s0-vl", "runs/pai_fix/tax/R2C-L18-s0-vl.pkl", "fix"),
    "NTV0-vl": ("runs/pai_fix/reads/NTV0-L18-s0-vl", "runs/pai_fix/tax/NTV0-L18-s0-vl.pkl", "fix"),
    "R2JH-vl": ("runs/pai_fix/reads/R2JH-L18-s0-vl", "runs/pai_fix/tax/R2JH-L18-s0-vl.pkl", "fix"),
}


def plan_arc(P):
    P = np.asarray(P, np.float64)[:, :2]
    return float(np.linalg.norm(np.diff(np.r_[[[0.0, 0.0]], P], axis=0), axis=1).sum())


def _launch_scene(job):
    o, recs = job
    C, Lg = o["ctrl"], o["log"]
    t_c, T, P = np.asarray(C["t"], float), np.asarray(Lg["T"], float), np.asarray(Lg["P"], float)
    th = t_c[0] + HAND
    ph = np.array([np.interp(th, T, P[:, 0]), np.interp(th, T, P[:, 1])])
    d = np.hypot(*(P - ph).T)
    m = np.nonzero((T >= th) & (d > 0.5))[0]
    if not len(m):
        return None
    tL = float(T[m[0]])
    sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    h4 = float(np.interp(tL + 4e6, T, sl) - np.interp(tL, T, sl))
    ex, ey = np.asarray(C["x"], float), np.asarray(C["y"], float)
    eh = np.array([np.interp(th, t_c, ex), np.interp(th, t_c, ey)])
    me = np.nonzero((t_c >= th) & (np.hypot(ex - eh[0], ey - eh[1]) > 0.5))[0]
    tE = float(t_c[me[0]]) if len(me) else np.nan
    # lead at t_L: a vehicle centre 0-25 m ahead, |lat| < 2 m in the human's frame at t_L
    Y = np.unwrap(np.asarray(Lg["Y"], float))
    a = np.array([np.interp(tL, T, P[:, 0]), np.interp(tL, T, P[:, 1]), np.interp(tL, T, Y)])
    j = int(np.argmin(np.abs(np.asarray(o["tg"], float) - tL)))
    tr = o["traffic"][j]
    lead, lead_v = False, np.nan
    if len(tr["id"]):
        c, s = np.cos(a[2]), np.sin(a[2])
        dx, dy = np.asarray(tr["x"]) - a[0], np.asarray(tr["y"]) - a[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        veh = np.array([("bicycle" not in str(k)) and ("pedestrian" not in str(k)) for k in tr["cls"]])
        q = (lx > 0) & (lx < 25) & (np.abs(ly) < 2.0) & veh
        if q.any():
            lead, lead_v = True, float(np.asarray(tr["v"])[q][np.argmin(lx[q])])
    R = sorted((r for r in recs if r["t0"] >= th), key=lambda r: r["t0"])
    at = next((r for r in R if r["t0"] >= tL), None)
    go = next((r for r in R if plan_arc(r.get("poses_model") or r["poses"]) >= 2.0), None)
    win = [r for r in R if tL <= r["t0"] <= tL + 3e6]
    short = [(r["ego"][4] * 10 < 0.5) and plan_arc(r.get("poses_model") or r["poses"]) < 1.0 for r in win]
    pre = [plan_arc(r.get("poses_model") or r["poses"]) for r in R if th <= r["t0"] < tL]
    out = dict(scene=o["scene"], score=float(o["score"] or 0), t_L=(tL - th) / 1e6, h4=h4, lead=lead, lead_v=lead_v,
               ego_move=(tE - tL) / 1e6 if np.isfinite(tE) else np.nan, n_dec=len(R),
               plan_go=(go["t0"] - tL) / 1e6 if go else np.nan,
               short3=float(np.mean(short)) if short else np.nan,
               pre_arc_max=float(max(pre)) if pre else np.nan)
    if at is not None:
        Pm = at.get("poses_model") or at["poses"]
        out |= dict(v0_at=float(at["ego"][4] * 10), arc4_at=plan_arc(Pm), arc2_at=plan_arc(np.asarray(Pm)[:4]), dt_at=(at["t0"] - tL) / 1e6)
    return out


def cmd_launch(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts")]
    import pai_fix as PF
    FA = DATA / "runs/pai_fix/facts"
    facts = {"R2J-281ag": "R2J-L18-s0-281ag", "NTV0-281ag": "NTV0-L18-s0-281ag", "R2J-vl": "R2J-L18-s0-vl", "R2C-vl": "R2C-L18-s0-vl",
             "NTV0-vl": "NTV0-L18-s0-vl", "R2JH-vl": "R2JH-L18-s0-vl"}
    ex = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv")
    sub_col = next(c for c in ("subset", "sub", "group") if c in ex.columns)
    gl = set(ex.scene[ex[sub_col].astype(str).str.lower().str.contains("launch")]) if "scene" in ex.columns else set()
    rows = []
    for name, (root, pkl, subset) in READS.items():
        F = pd.read_csv(FA / f"{facts[name]}.csv").set_index("scene")
        F = F[~F.rpc]
        Lm = PF.launch_mask(F)
        sc = set(F.index[Lm])
        A = {o["scene"]: o for o in pickle.load(open(DATA / pkl, "rb"))["rolls"] if "ctrl" in o and o["scene"] in sc}
        recs = defaultdict(list)
        for f in glob.glob(str(DATA / root / "runs/*/driver/drive.jsonl")):
            for line in open(f):
                if '"infer": true' not in line:
                    continue
                i = line.find('"scene": "')
                s = line[i + 10:i + 53]
                if s not in A:
                    continue
                r = json.loads(line)
                if r["kind"] == "drive":
                    recs[r["scene"]].append(r)
        with ProcessPoolExecutor(16) as px:
            out = list(px.map(_launch_scene, [(A[s], recs[s]) for s in sorted(A)]))
        for r in out:
            if r is None:
                continue
            r |= dict(read=name, subset=subset, launched=bool(PF.launched(F.loc[[r["scene"]]]).iloc[0]), g_launch=r["scene"] in gl,
                      fclass=F.loc[r["scene"], "fclass"])
            rows.append(r)
    D = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES / "launch_plans.csv", index=False)
    md, rec = ["# ROUND3: model plans at the human's launch (stored reads, launch scenes)", "",
               "t_L = the log is 0.5 m past its handoff pose; `arc4@tL` = path length of the model plan (poses_model) at the first decision at or after t_L; "
               "`go` = time from t_L to the first decision whose plan covers >= 2 m (negative: before the human); `short3` = share of decisions in "
               "[t_L, t_L + 3 s] with the ego still (< 0.5 m/s) and a plan under 1 m; `ego` = time from t_L until the ego is 0.5 m past its handoff "
               "pose. Medians; n = launch scenes of the read.", ""], {}

    def block(title, sel, reads):
        md.extend([f"## {title}", "", "| read | n | launched | arc4@tL med | arc4@tL < 1 m | go med (s) | no go | short3 mean | ego move med (s) | lead at t_L |",
                   "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"])
        rec[title] = {}
        for rd in reads:
            g = D[(D.read == rd) & sel(D)]
            if not len(g):
                continue
            x = dict(n=len(g), launched=int(g.launched.sum()), arc4=float(g.arc4_at.median()), arc4_lt1=int((g.arc4_at < 1).sum()),
                     go=float(g.plan_go.median()), nogo=int(g.plan_go.isna().sum()), short3=float(g.short3.mean()), ego=float(g.ego_move.median()),
                     lead=int(g.lead.sum()))
            rec[title][rd] = x
            md.append(f"| {rd} | {x['n']} | {x['launched']} | {x['arc4']:.2f} | {x['arc4_lt1']} | {x['go']:+.2f} | {x['nogo']} | {x['short3']:.2f} | "
                      f"{x['ego']:+.2f} | {x['lead']} |")
        md.append("")

    block("PAI 281, all launch scenes (ACC + GAP)", lambda d: d.subset == "281", ["NTV0-281ag", "R2J-281ag"])
    pj = D[D.read == "R2J-281ag"].set_index("scene")
    pn = D[D.read == "NTV0-281ag"].set_index("scene")
    lost = sorted(set(pn.index[pn.launched]) & set(pj.index[~pj.launched]))
    block(f"PAI 281, the {len(lost)} launches NTV0 makes and R2J loses", lambda d: d.scene.isin(lost), ["NTV0-281ag", "R2J-281ag"])
    block("PAI-FIX G-LAUNCH (VC + LEAD)", lambda d: (d.subset == "fix") & d.g_launch, ["NTV0-vl", "R2C-vl", "R2J-vl", "R2JH-vl"])
    block("PAI-FIX all launch scenes (G-LAUNCH + TARGET, VC + LEAD)", lambda d: d.subset == "fix", ["NTV0-vl", "R2C-vl", "R2J-vl", "R2JH-vl"])
    # per-scene table of the lost launches
    md += [f"## The lost launches, per scene (R2J vs NTV0, ACC + GAP)", "",
           "| scene | lead at t_L (v) | t_L (s) | human 4 s (m) | NTV0 arc4@tL / go / ego | R2J arc4@tL / go / ego | R2J score | R2J class |", "|:--|:--|--:|--:|:--|:--|--:|:--|"]
    f2 = lambda v: "nan" if not np.isfinite(v) else f"{v:.1f}"  # noqa: E731
    for s in lost:
        n_, j_ = pn.loc[s], pj.loc[s]
        md.append(f"| {s[7:15]} | {'yes' if j_.lead else 'no'} ({f2(j_.lead_v)}) | {j_.t_L:.1f} | {j_.h4:.1f} | {f2(n_.arc4_at)} / {f2(n_.plan_go)} / {f2(n_.ego_move)} | "
                  f"{f2(j_.arc4_at)} / {f2(j_.plan_go)} / {f2(j_.ego_move)} | {j_.score:.2f} | {j_.fclass} |")
    rec["lost"] = lost
    dump(RES / "launch_plans.json", rec)
    (RES / "launch_plans.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_weights(a):
    S = pai_states()
    B = np.load(W / f"{a.base}.npz")
    names, w, grp = B["names"].astype(str), B["w"].astype(np.float64).copy(), B["grp"]
    idx = {x: i for i, x in enumerate(names)}
    M = np.load(CR / a.pool / "tab.npz")
    assert np.array_equal(M["names"].astype(str), names), f"{a.base} rows != {a.pool}"
    rule = dict(stand=f"PAI human rows (plain and stretch) whose own 4 s label path is < {L_STAND} m (standing + creeping) -> weight 0; "
                      "each part renormalised to its base share",
                ns="both stretch parts (ph05, ph15) -> weight 0; the plain part scaled to the base's whole PAI human share (rows within it unchanged)",
                mv=f"PAI human rows (plain and stretch) whose fed ego speed is < {V_MV} m/s -> weight 0; each part renormalised to its base share",
                mve=f"MV (fed < {V_MV} m/s -> 0) and every PAI human row without an exam event ahead -> weight 0 (route turn: max heading change >= "
                    f"{TURN_DEG} deg over the human's path to 4 s or {TURN_M} m, whichever reaches further; stop: the human < {V_STOP} m/s within "
                    "the label or the next 4 s; stretch rows on their own clock from the base row's log future); each part renormalised to its base share",
                mvl=f"MV's pool weights (fed < {V_MV} m/s -> 0, renormalised) + navtrain draw weights: launch rows (fed < {V_LROW} m/s, 4 s label "
                    f">= {L_STAND} m) {NAV_UP}, every other navtrain row 1 (rm_train renormalises inside the normal rows)")[a.rule]
    assert a.rule not in ("mve", "mvl") or a.base == "R2JF", "round 3c rules start from R2JF (MV's base)"
    EV = pai_events() if a.rule == "mve" else None
    rec = dict(base=a.base, rule=rule, rule_key=a.rule, parts={})
    pai_share = sum(w[[idx[x] for x in d["names"]]].sum() for d in S.values())
    for p, d in S.items():
        ix = np.array([idx[x] for x in d["names"]])
        assert np.array_equal(M["speed"][ix], d["speed"]) and np.array_equal(M["fut"][ix], d["fut"]), f"{p}: merged rows != source"
        sh = w[ix].sum()
        drop = {"stand": d["L"] < L_STAND,                         # the label stands throughout, whatever the fed speed (standing + creeping)
                "ns": np.full(len(ix), p != "ph"),
                "mv": d["speed"].astype(np.float64) < V_MV, "mvl": d["speed"].astype(np.float64) < V_MV,
                "mve": (d["speed"].astype(np.float64) < V_MV) | ~(EV[p]["turn"] | EV[p]["stop"]) if EV else None}[a.rule] & (w[ix] > 0)
        keep = (w[ix] > 0) & ~drop
        w[ix[drop]] = 0.0
        to = pai_share if (a.rule == "ns" and p == "ph") else (0.0 if a.rule == "ns" else sh)
        if keep.any():
            w[ix[keep]] *= to / w[ix[keep]].sum()
        rec["parts"][p] = dict(share=float(sh), share_after=float(w[ix].sum()), drawable_before=int((drop | keep).sum()), dropped=int(drop.sum()),
                               drawable_after=int(keep.sum()),
                               dropped_by_state={st: int((drop & (d["state"] == st)).sum()) for st in STATES},
                               dropped_share=float(drop.sum() / max((drop | keep).sum(), 1)))
        if EV:
            tu, sp = EV[p]["turn"][keep], EV[p]["stop"][keep]
            rec["parts"][p] |= dict(kept_turn_only=int((tu & ~sp).sum()), kept_stop_only=int((sp & ~tu).sum()), kept_both=int((tu & sp).sum()),
                                    moving_rows=int((((w[ix] > 0) | drop) & (d["speed"] >= V_MV)).sum()))
    assert abs(w.sum() - 1) < 1e-9 and abs(B["w"].sum() - 1) < 1e-9, (w.sum(), B["w"].sum())
    nonpai = ~np.isin(np.arange(len(w)), np.concatenate([[idx[x] for x in d["names"]] for d in S.values()]))
    assert np.array_equal(w[nonpai], B["w"][nonpai]), "non-PAI weights changed"
    rec |= dict(pai_drawable_rows=int(sum(v["drawable_after"] for v in rec["parts"].values())), out=a.out)
    extra = {}
    if a.rule == "mvl":                                            # pool weights must be R3MV's bit for bit; the navtrain weights ride along
        assert np.array_equal(w, np.load(W / "R3MV.npz")["w"]), "mvl: pool weights != R3MV"
        N = navtrain_rows()
        nwt = np.where(N["launch"], NAV_UP, 1.0)
        extra = dict(nav_names=N["names"], nav_w=nwt)
        sh = float(nwt[N["launch"] & N["ok"]].sum() / nwt[N["ok"]].sum())
        rec |= dict(nav_rows=int(len(nwt)), nav_rows_label=int(N["ok"].sum()), nav_launch_rows=int(N["launch"].sum()), nav_up_w=NAV_UP,
                    nav_launch_share_before=float(N["launch"].sum() / N["ok"].sum()), nav_launch_share_after=sh, nav_launch_per_batch_51=51 * sh)
    np.savez(W / f"{a.out}.npz", names=B["names"], w=w, grp=grp, **extra)
    dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_launchrows(a):
    """Round 3b offline table: launch rows (fed v0 < 0.5 m/s, the 4 s label path >= 0.5 m) per PAI part and in navtrain."""
    S = pai_states()
    T = {p: np.load(CR / PARTS[p] / "tab.npz") for p in PARTS}
    Wf = {k: np.load(W / f"{k}.npz") for k in SHARES}
    wn = {k: dict(zip(v["names"].astype(str), v["w"])) for k, v in Wf.items()}
    plainL = dict(zip(S["ph"]["names"], S["ph"]["L"]))
    q = lambda x, p_: float(np.percentile(x, p_)) if len(x) else float("nan")  # noqa: E731
    rec, L = dict(rule=f"launch row: fed speed < {V_LROW} m/s and the row's own 4 s label path >= {L_STAND} m", parts={}), []
    L += ["# ROUND3b: launch rows per pool (offline, before training)", "",
          f"Launch row = fed ego speed v0 < {V_LROW} m/s and the row's own 4 s label path >= {L_STAND} m (the human leaves within the label). "
          "Rows = drawable under R2JF (J / J-NF; outside held and the exam links). `4 s dist` = the label's path length at 4 s (median [p25, p75]); "
          "`/ base` = median ratio of a stretch launch row's 4 s distance to its plain base row's (same human moment; expect about f^2 for a launch "
          "from rest); `per batch` = expected launch rows a batch (64 pool rows) under each draw file; `< 2 m/s` = share of the part's drawable rows "
          "that MV drops.", "",
          "| pool | drawable | launch rows | share | 4 s dist (m) | / base | v0 med (m/s) | per batch R2JF | per batch R3NS | per batch R3MV | < 2 m/s |",
          "|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|"]
    for p, d in S.items():
        dr = np.array([wn["R2JF"].get(x, 0.0) for x in d["names"]]) > 0
        v0 = d["speed"].astype(np.float64)
        lr = (v0 < V_LROW) & (d["L"] >= L_STAND) & dr
        Ld = d["L"][lr]
        if p == "ph":
            rat = np.array([1.0])
        else:
            base = np.array([f"on:ph:{b[3:]}" for b in T[p]["base"].astype(str)])
            bl = np.array([plainL.get(b, np.nan) for b in base[lr]])
            ok = np.isfinite(bl) & (bl > 0.1)
            rat = Ld[ok] / bl[ok]
        pb = {k: float(sum(wn[k].get(x, 0.0) for x in d["names"][lr]) * K_ON) for k in SHARES}
        slow = float(((v0 < V_MV) & dr).sum() / max(dr.sum(), 1))
        r = dict(drawable=int(dr.sum()), launch=int(lr.sum()), share=float(lr.sum() / max(dr.sum(), 1)), dist_med=q(Ld, 50), dist_p25=q(Ld, 25),
                 dist_p75=q(Ld, 75), ratio_to_base_med=q(rat, 50), v0_med=q(v0[lr], 50), per_batch=pb, below_2ms=slow)
        rec["parts"][p] = r
        L.append(f"| PAI {p} | {r['drawable']} | {r['launch']} | {r['share']:.2%} | {r['dist_med']:.2f} [{r['dist_p25']:.2f}, {r['dist_p75']:.2f}] | "
                 f"{r['ratio_to_base_med']:.2f} | {r['v0_med']:.2f} | {pb['R2JF']:.2f} | {pb['R3NS']:.2f} | {pb['R3MV']:.2f} | {slow:.1%} |")
    nl, nn, nd, nv = 0, 0, [], []                                 # navtrain_full, all 12 shards (rows with a full label); J / NS / MV draw 51 a batch
    for i in range(12):
        t = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        ok = ~np.isnan(t["fut"]).any((1, 2))
        v0 = t["speed"][ok].astype(np.float64)
        Lp, _ = path_len(t["fut"][ok])
        m = (v0 < V_LROW) & (Lp >= L_STAND)
        nl, nn = nl + int(m.sum()), nn + int(ok.sum())
        nd.append(Lp[m]), nv.append(v0[m])
    nd, nv = np.concatenate(nd), np.concatenate(nv)
    r = dict(rows=nn, launch=nl, share=nl / nn, dist_med=q(nd, 50), dist_p25=q(nd, 25), dist_p75=q(nd, 75), v0_med=q(nv, 50), per_batch=51 * nl / nn)
    rec["navtrain"] = r
    L.append(f"| navtrain_full | {nn} | {nl} | {r['share']:.2%} | {r['dist_med']:.2f} [{r['dist_p25']:.2f}, {r['dist_p75']:.2f}] | | {r['v0_med']:.2f} | "
             f"{r['per_batch']:.2f} (of 51) | {r['per_batch']:.2f} | {r['per_batch']:.2f} | |")
    tot = {k: sum(x["per_batch"][k] for x in rec["parts"].values()) for k in SHARES}
    dm = {k: sum(x["per_batch"][k] * x["dist_med"] for x in rec["parts"].values()) / max(tot[k], 1e-9) for k in SHARES}
    rec["pai_per_batch"], rec["pai_dist_weighted"] = tot, dm
    L += ["", "PAI launch rows a batch: " + ", ".join(f"{k} {tot[k]:.2f} (row-weighted median 4 s distance {dm[k]:.2f} m)" for k in SHARES)
          + f"; navtrain {r['per_batch']:.2f} at {r['dist_med']:.2f} m."]
    dump(RES / "launch_rows.json", rec)
    (RES / "launch_rows.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


CLASSES = ("launch", "stand", "stop-approach", "turn", "cruise")
VBINS = (0.5, 2, 4, 6, 8, 10, 13, 16, 99)


def cmd_classes(a):
    """Round 3c offline table (before training): expected rows a batch by label-only class per arm, the PAI rows' MVE events, and the 4 s
    label distance of moving rows by fed-speed bin, PAI vs navtrain."""
    S, EV, N = pai_states(), pai_events(), navtrain_rows()
    M = np.load(CR / "on1_JTPH/tab.npz")
    pn = M["names"].astype(str)
    pc, pL = label_class(M["speed"].astype(np.float64), M["fut"].astype(np.float64))
    part = np.full(len(pn), "S", dtype="U5")
    evt = np.full(len(pn), "", dtype="U9")
    idx = {x: i for i, x in enumerate(pn)}
    for p, d in S.items():
        ix = np.array([idx[x] for x in d["names"]])
        part[ix] = p
        e = EV[p]
        evt[ix] = np.select([e["turn"] & e["stop"], e["turn"], e["stop"]], ["turn+stop", "turn", "stop"], "none")
    Wf = {k: np.load(W / f"{k}.npz") for k in ("R3MV", "R3MVL", "R3MVE")}
    for k, z in Wf.items():
        assert np.array_equal(z["names"].astype(str), pn), k
    T0 = np.load(CR / "on1_JT/tab.npz")
    c0, _ = label_class(T0["speed"].astype(np.float64), T0["fut"].astype(np.float64))
    w0 = np.load(W / "JT00S.npz")["w"].astype(np.float64)
    w0 = w0 / w0.sum()
    okn = N["ok"]
    nav_u = {c: float((N["cls"][okn] == c).mean()) for c in CLASSES}
    nwl = np.where(N["launch"], NAV_UP, 1.0)[okn]
    nav_l = {c: float(nwl[N["cls"][okn] == c].sum() / nwl.sum()) for c in CLASSES}
    arms = {"R2C (CTRL)": (83, nav_u, None), "MV": (51, nav_u, "R3MV"), "MVL": (51, nav_l, "R3MVL"), "MVE": (51, nav_u, "R3MVE")}
    rec, md = dict(rules=dict(classes=label_class.__doc__.split("->")[0].strip(), events=_scene_events.__doc__.split("->")[1].strip()), arms={}), []
    md += ["# ROUND3c: rows a batch by class (offline, before training)", "",
           "Label-only classes (each row's fed speed v0 and its own 4 s label, every pool alike): **launch** v0 < 0.5 m/s and the 4 s path >= 0.5 m; "
           "**stand** v0 < 0.5, path < 0.5 m; **stop-approach** v0 >= 0.5 and some 0.5 s label segment below 0.5 m/s; **turn** else and |heading| >= "
           "15 deg at some label pose; **cruise** else. Batch = navtrain rows (83 CTRL / 51 the PAI arms) + pool rows (CTRL: 32 S rows, JT00S; PAI "
           "arms: 64 = 32 S + 32 PAI human rows, the arm's draw file); the 13 yaw-rate rows are not counted. MVL's navtrain draw: launch rows x 3, "
           "renormalised inside navtrain.", "",
           "| arm | " + " | ".join(CLASSES) + " | of which PAI: launch / stand / stop / turn / cruise | PAI rows with MVE event (turn / stop / both / none) |",
           "|:--|" + "--:|" * len(CLASSES) + ":--|:--|"]
    for arm, (nN, nv, wf) in arms.items():
        nav = {c: nN * nv[c] for c in CLASSES}
        if wf is None:
            pool = {c: 32 * float(w0[c0 == c].sum()) for c in CLASSES}
            pai, ev = {c: 0.0 for c in CLASSES}, {}
        else:
            w = Wf[wf]["w"].astype(np.float64)
            w = w / w.sum()
            pool = {c: 64 * float(w[pc == c].sum()) for c in CLASSES}
            pm = part != "S"
            pai = {c: 64 * float(w[pm & (pc == c)].sum()) for c in CLASSES}
            ev = {e: 64 * float(w[pm & (evt == e)].sum()) for e in ("turn", "stop", "turn+stop", "none")}
        tot = {c: nav[c] + pool[c] for c in CLASSES}
        rec["arms"][arm] = dict(nav_rows=nN, draw=wf, total=tot, navtrain=nav, pool=pool, pai=pai, pai_events=ev)
        md.append(f"| {arm} | " + " | ".join(f"{tot[c]:.2f}" for c in CLASSES) + " | " + " / ".join(f"{pai[c]:.2f}" for c in CLASSES) + " | "
                  + (" / ".join(f"{ev[e]:.2f}" for e in ("turn", "stop", "turn+stop", "none")) if ev else "-") + " |")
    md += ["", "Navtrain alone (rows a batch): " + "; ".join(f"{k} " + ", ".join(f"{c} {v['navtrain'][c]:.2f}" for c in CLASSES)
                                                    for k, v in rec["arms"].items() if k in ("R2C (CTRL)", "MV", "MVL")), ""]
    # MVE event split of the moving PAI rows (fed >= 2 m/s; MV's drawable rows)
    md += ["## PAI human rows fed >= 2 m/s (MV's rows): exam events ahead (MVE keeps turn / stop / both)", "",
           "| part | MV rows | turn only | stop only | both | none (dropped by MVE) | kept share |", "|:--|--:|--:|--:|--:|--:|--:|"]
    wmv = Wf["R3MV"]["w"]
    rec["events"] = {}
    for p in PARTS:
        m = (part == p) & (wmv > 0)
        c = {e: int((m & (evt == e)).sum()) for e in ("turn", "stop", "turn+stop", "none")}
        rec["events"][p] = c
        md.append(f"| {p} | {int(m.sum())} | {c['turn']} | {c['stop']} | {c['turn+stop']} | {c['none']} | {1 - c['none'] / max(m.sum(), 1):.1%} |")
    # 4 s label distance of moving rows by fed-speed bin
    w2 = np.load(W / "R2JF.npz")["w"]
    sets = {"navtrain": (N["v0"][okn], N["L"][okn])}
    for p in PARTS:
        m = (part == p) & (w2 > 0)
        sets[f"PAI {p}"] = (M["speed"][m].astype(np.float64), pL[m])
    m = (part == "ph") & (Wf["R3MVE"]["w"] > 0)
    sets["PAI ph, MVE rows"] = (M["speed"][m].astype(np.float64), pL[m])
    md += ["", "## Moving rows: 4 s label distance by fed speed (median m; in brackets median distance / (4 s x v0); n)", "",
           "Drawable rows (navtrain: full labels; PAI: R2JF weight > 0, i.e. before MV's < 2 m/s cut; last row: MVE's plain rows).", "",
           "| v0 bin (m/s) | " + " | ".join(sets) + " |", "|:--|" + ":--|" * len(sets)]
    rec["dist_by_speed"] = {}
    for lo, hi in zip(VBINS[:-1], VBINS[1:]):
        cells = []
        for k_, (v0, L) in sets.items():
            m = (v0 >= lo) & (v0 < hi)
            r = dict(n=int(m.sum()), med=float(np.median(L[m])) if m.any() else float("nan"),
                     ratio=float(np.median(L[m] / (4 * v0[m]))) if m.any() else float("nan"))
            rec["dist_by_speed"].setdefault(f"{lo}-{hi}", {})[k_] = r
            cells.append(f"{r['med']:.1f} ({r['ratio']:.2f}; {r['n']})" if r["n"] else "-")
        md.append(f"| {lo}-{hi if hi < 99 else ''} | " + " | ".join(cells) + " |")
    dump(RES / "classes3c.json", rec)
    (RES / "classes3c.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_steps(a):
    out = dict(rule="both arms take round 2's J / CTRL steps (prereg)", JW=dict(steps=a.n), JNF=dict(steps=a.n), CTRL=dict(steps=a.n, tag="R2C-L18-s0"))
    dump(O / "steps.json", out)
    dump(RES / "steps.json", out)
    print(json.dumps(out))


def run_dir(tag):
    d = sorted(glob.glob(str(DATA / f"runs/route_mem/train-{tag}/*")))
    assert d, f"no run dir for {tag}"
    return Path(d[-1])


def cmd_smoke(a):
    d = run_dir(a.tag)
    S = defaultdict(list)
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S[e["tag"]].append(e["value"])
    done = json.loads((d / "DONE").read_text()) if (d / "DONE").exists() else {}
    fl = done.get("flat_brk", {})
    mean = lambda k: float(np.mean(S[k])) if k in S else float("nan")  # noqa: E731
    ck = dict(done=(bool(done), True, bool(done)), finite=(None, None, all(np.isfinite(v).all() for v in S.values())),
              pool_imit_rows=(mean("loss/on/n_imit"), K_S, 0.75 * K_S <= mean("loss/on/n_imit") <= 1.25 * K_S),
              pool_imit_rows_without_hinge=(max(S.get("loss/on/n_hinge", [np.nan])), 0, max(S.get("loss/on/n_hinge", [1])) == 0),
              agent_ho_zero=(max(S.get("loss/on/agent_ho", [0])), 0, max(S.get("loss/on/agent_ho", [0])) == 0))
    if a.arm == "JW":
        exp = 0.5 * 51 * 0.75 * fl.get("train_rows", 0) / max(fl.get("train_rows_total", 1), 1)
        ck |= dict(flat_rows=(mean("loss/flat/rows"), exp, 0.6 * exp <= mean("loss/flat/rows") <= 1.4 * exp), flat_p=(fl.get("p"), 0.5, fl.get("p") == 0.5))
    else:
        ck |= dict(no_flat=(mean("loss/flat/rows"), 0, not fl and np.nan_to_num(mean("loss/flat/rows")) == 0))
    if a.arm in ("NS", "MV", "MVL", "MVE"):
        gs, want = (done.get("draw_shares") or {}).get("group_share", {}), (SHARES | SHARES3C)[f"R3{a.arm}"]
        ck |= {f"share_{p}": (gs.get(p), v, abs(gs.get(p, -1) - v) < 1e-6) for p, v in want.items()}
        ck |= dict(wfile=((done.get("draw_shares") or {}).get("on_wfile"), f"R3{a.arm}", f"w/R3{a.arm}.npz" in str((done.get("draw_shares") or {}).get("on_wfile"))))
    if a.arm in ("MVL", "MVE"):                                    # round 3c: the navtrain draw weights are on in MVL only
        ds, wj = done.get("draw_shares") or {}, json.loads((RES / "w_R3MVL.json").read_text())
        if a.arm == "MVL":
            ck |= dict(nav_listed=(ds.get("nav_listed"), ds.get("nav_rows"), ds.get("nav_listed", 0) >= 0.99 * ds.get("nav_rows", 1)),
                       nav_up_w=(ds.get("nav_up_w"), [NAV_UP], ds.get("nav_up_w") == [NAV_UP]),
                       nav_up_share=(ds.get("nav_up_share"), wj["nav_launch_share_after"],
                                     abs(ds.get("nav_up_share", -1) - wj["nav_launch_share_after"]) < 0.1 * wj["nav_launch_share_after"]))
        else:
            ck |= dict(no_nav_w=(ds.get("nav_up_share"), None, "nav_rows" not in ds))
    ok = bool(all(v[2] for v in ck.values()))
    res = dict(tag=a.tag, arm=a.arm, run=str(d), pass_=ok, checks=ck, steps=len(S.get("loss/total", [])), draw_shares=done.get("draw_shares"))
    dump(RES / f"smoke_{a.arm}.json", res)
    print(json.dumps(res, indent=1, default=float))
    if not ok:
        raise SystemExit(2)


def cmd_gate(a):
    import pandas as pd
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts")]
    from jevdrive import stats
    E = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv").set_index("scene")
    FA = DATA / "runs/pai_fix/facts"
    rd = lambda t: pd.read_csv(FA / f"{t}-vl.csv").set_index("scene").reindex(E.index)  # noqa: E731
    C = rd(a.ctrl)
    sets = dict(TARGET=["STOP", "STOP-L", "TURN"], G_LAUNCH=["G-LAUNCH"], T281=["T281-STOP", "T281-TURN"])
    out, md, passing = dict(ctrl=a.ctrl, lines=dict(G_LAUNCH="mean >= CTRL - 0.05 (binding)", T281="route-turn zeros <= CTRL - 5",
                                                         TARGET="mean >= CTRL - 0.02"), arms={}), [], []
    md += ["# ROUND3 PAI-FIX native (VC + LEAD) lines against CTRL " + a.ctrl, "", "| arm | G-LAUNCH arm / CTRL (diff, CI) | T281 route-turn zeros arm / CTRL "
           "| TARGET arm / CTRL (diff, CI) | launched G-LAUNCH | pass |", "|:--|:--|:--|:--|:--|:--|"]
    for t in a.arms:
        F = rd(t)
        r = {}
        for k, subs in sets.items():
            m = E.subset.isin(subs) & F.score.notna() & C.score.notna() & ~F.rpc.fillna(False).astype(bool) & ~C.rpc.fillna(False).astype(bool)
            u, v = F.score[m].to_numpy(float), C.score[m].to_numpy(float)
            r[k] = dict(n=int(m.sum()), arm=float(u.mean()), ctrl=float(v.mean()), diff=stats.paired(u, v),
                        rt_arm=int((F.fclass[m] == "route-turn").sum()), rt_ctrl=int((C.fclass[m] == "route-turn").sum()))
        gl = E.subset == "G-LAUNCH"
        import pai_fix as PF
        lau = f"{int((PF.launch_mask(F[gl]) & PF.launched(F[gl])).sum())} / {int((PF.launch_mask(C[gl]) & PF.launched(C[gl])).sum())} of {int(PF.launch_mask(F[gl]).sum())}"
        ok = dict(G_LAUNCH=r["G_LAUNCH"]["arm"] >= r["G_LAUNCH"]["ctrl"] - 0.05, T281=r["T281"]["rt_arm"] <= r["T281"]["rt_ctrl"] - 5,
                  TARGET=r["TARGET"]["arm"] >= r["TARGET"]["ctrl"] - 0.02)
        out["arms"][t] = dict(r, ok=ok, pass_=all(ok.values()), launched_glaunch=lau)
        if all(ok.values()):
            passing.append(t)
        g, q, x = r["G_LAUNCH"], r["T281"], r["TARGET"]
        md.append(f"| {t} | {g['arm']:.3f} / {g['ctrl']:.3f} ({stats.fmt(g['diff'], '+.3f')}) {'ok' if ok['G_LAUNCH'] else 'FAIL'} | {q['rt_arm']} / {q['rt_ctrl']} "
                  f"{'ok' if ok['T281'] else 'FAIL'} | {x['arm']:.3f} / {x['ctrl']:.3f} ({stats.fmt(x['diff'], '+.3f')}) {'ok' if ok['TARGET'] else 'FAIL'} | {lau} | "
                  f"{'PASS' if all(ok.values()) else 'fail'} |")
    out["passing"] = passing
    dump(RES / f"{a.stem}.json", out)
    (RES / f"{a.stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), file=sys.stderr)
    print(" ".join(passing))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows")
    sub.add_parser("launch")
    p = sub.add_parser("weights")
    p.add_argument("--base", default="R2JF"), p.add_argument("--out", default="R3JW"), p.add_argument("--pool", default="on1_JTPH")
    p.add_argument("--rule", choices=("stand", "ns", "mv", "mve", "mvl"), default="stand")
    sub.add_parser("launchrows")
    sub.add_parser("classes")
    p = sub.add_parser("steps")
    p.add_argument("--n", type=int, default=9500)
    p = sub.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--arm", choices=("JW", "JNF", "NS", "MV", "MVL", "MVE"), required=True)
    p = sub.add_parser("gate")
    p.add_argument("--arms", nargs="+", required=True), p.add_argument("--ctrl", default="R2C-L18-s0"), p.add_argument("--stem", default="gate")
    a = ap.parse_args()
    dict(rows=cmd_rows, launch=cmd_launch, launchrows=cmd_launchrows, classes=cmd_classes, weights=cmd_weights, steps=cmd_steps, smoke=cmd_smoke, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
