#!/usr/bin/env python3
"""TURN-CAP (NR1 prereg supplement 21, plans/2026-10-20-nr1-supp21-turncap.md; lib/nr1_free.py turncap_*, rm_train --on-turncap W).

  $TR turn_cap.py label --pool on1_JTX     (box, CPU) the turn label of the pool's S rows: heading range / net change of the logged path
                                           (rows meta `gpath`, 1 m samples) over [s_e - 10, s_e + 20] m -> $CR/<pool>/turn_cap.npz
                                           (names = the tab's); results/turn_cap/label.json (domain counts, JTX00 draw share, stored-plan fires)
  $TR turn_cap.py gate --out DIR           (box, CPU) offline gate G1-G3 on the pool and B2's stored full-length arena reads -> DIR/gate.json,
                                           DIR/decisions.parquet, results/turn_cap/gate.json; exit 0 pass, 2 fail
  $TR turn_cap.py lists                    (box) the turn-set windows outside the six subset chunks, grouped by cache shard -> $S/lists/ct<k>.txt
  $TR turn_cap.py dose --run T --log L --out JSON   100-step frozen eager run at W0: finite, the term fires at some step, the dose rule
  tmp/pcs_local/.venv/bin/python turn_cap.py stage1 [--anchors]   (Mac) the stage 1 rule on the pulled reads -> results/turn_cap/stage1.{json,md}

Closed-loop decisions (gate): free_gate's conventions (scene time t <-> log frame start_idx + round(20 t); the controller trace moved rigidly
onto the log at start_idx; the human's 4 s raw travel from the frame and its reachable version blended from the controller speed over 2 s).
The cap is K_TC x the larger of the two (the trainer's rule); the turn label is the heading range of the log path at 1 m stations over
[s_e - 10, s_e + 20] m, s_e = the ego's projection onto the log polyline (frames f - 200 .. f + 400); the plan = drive.jsonl `poses`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/alpasim_proxy/scripts")]
import nr1_free as NF  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix"
S_ = O / "turn_cap"
CR = DATA / "runs/op_parity/cache"
AR = DATA / "runs/alpasim_proxy/arena"
RES = REPO / "experiments/defect_fix/results/turn_cap"
ROWS = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y", **{f"pn{i}": f"PN{i}" for i in range(7)}}
GI = 15                                            # gpath index of the ego's station (gpath = s0 - 15 .. s0 + 70 m, 1 m)
W0, W_FREE, W_RANGE, DOSE_B = 0.004, 0.001, (0.002, 0.04), 3.0
B2 = {"s1": "STANDFMXH-L18-s1", "s0": "STANDFMXH-L18-s0"}
TARGETS = ("098f5c53", "202f583c", "3f805a18", "476b54f9", "6f27563e", "87705868", "9e7c5a46", "efe356db", "f2305669")
SUBSET = (0, 3, 6, 9, 12, 15)
LAT3 = ("zero: turn window, offroad / corridor", "zero: offroad, non-turn", "zero: corridor, moving non-turn")
CTL, ARMS = "STANDFMXH1K-L18-s0", {"T1": "TCAPA1K-L18-s0", "T3": "TCAPB1K-L18-s0"}


# ---------------------------------------------------------------- pool labels
def cmd_label(a):
    import torch
    tab = np.load(CR / a.pool / "tab.npz")
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    part = np.array([n.split(":")[1] for n in names])
    n = len(names)
    turn, net = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    served = np.full((n, 8, 3), np.nan, np.float32)
    for p, d in ROWS.items():
        rows = np.flatnonzero(S & (part == p))
        if not len(rows):
            continue
        z = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        pos = {q: i for i, q in enumerate(z["names"].astype(str).tolist())}
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in rows])
        assert np.allclose(z["speed"][mi], tab["speed"][rows], atol=1e-3), f"{d}: meta speed != the pool's"
        r, t = NF.turn_range(z["gpath"][mi][:, GI - NF.TC_BACK:GI + NF.TC_AHEAD + 1])
        turn[rows], net[rows], served[rows] = r, t, z["poses"][mi]
    assert np.isfinite(turn[S]).mean() > 0.99, "S rows without a turn label"
    f = CR / a.pool / "turn_cap.npz"
    assert not f.exists(), f"{f} exists: never overwritten"
    np.savez(f, names=names, turn=turn, net=net)
    # description: the domain, its draw share, the stored served plans' fires (what the term would meet at the pool's own plans)
    hum4 = NF.cum_arc(tab["fut_reg"])[:, -1]
    cap, fin = NF.turncap_cap(tab["fut"], tab["fut_reg"])
    dom = S & fin & NF.turncap_rows(tab["speed"], hum4, turn)
    sharp = S & fin & (np.nan_to_num(turn) >= NF.TURN_TC)
    w = np.load(DATA / "runs/defect_fix/w/JTX00.npz")
    assert np.array_equal(w["names"].astype(str), names), "JTX00 rows != the pool's"
    ww = w["w"].astype(np.float64) / w["w"].sum()
    j = np.flatnonzero(dom & np.isfinite(served).all((1, 2)))
    _, pr = NF.stand_terms(torch.as_tensor(served[j]), torch.as_tensor(cap[j]), torch.ones(len(j), dtype=torch.bool), parts=True)
    fire = np.zeros(n, bool)
    fire[j] = pr["fire"].numpy()
    ratio = pr["s_plan"].numpy() / np.maximum(cap[j, -1] / NF.K_TC, 1e-3)
    q = lambda x: np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if len(x) else []  # noqa: E731
    rec = dict(pool=a.pool, rows=n, s_rows=int(S.sum()), labelled=int(np.isfinite(turn[S]).sum()), sharp_s_rows=int(sharp.sum()),
               domain=int(dom.sum()), domain_right=int((dom & (net < 0)).sum()), domain_left=int((dom & (net >= 0)).sum()),
               domain_share_of_s=float(dom.sum() / S.sum()), domain_logs=int(len(np.unique(tab["log"].astype(str)[dom]))),
               blocked=dict(ego_below_v_tc=int((sharp & (tab["speed"] < NF.V_TC)).sum()), human_arc_below_h_tc=int((sharp & (tab["speed"] >= NF.V_TC) & (hum4 < NF.H_TC)).sum())),
               draw_share_jtx00=dict(domain=float(ww[dom].sum()), s_rows=float(ww[S].sum()), per_batch_70=float(70 * ww[dom].sum())),
               served_fire=dict(rows=int(len(j)), fires=int(fire.sum()), share=float(fire[j].mean()) if len(j) else None,
                                draw_share=float(ww[fire].sum()), right=int((fire & (net < 0)).sum()), left=int((fire & (net >= 0)).sum()),
                                plan_over_human_arc_q=q(ratio)),
               by_part=dict(Counter(part[dom].tolist())), turn_q_domain=q(turn[dom]))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "label.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- closed-loop decisions on stored arena reads
class Stations:
    """log polyline stations (moving frames only) for projections and 1 m resampling."""

    def __init__(self, xy):
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        keep = np.r_[True, np.diff(s) > 1e-3]
        self.xy, self.s, self.S, self.fi = xy, s, s[keep], np.flatnonzero(keep)

    def project(self, p, f):
        lo, hi = max(f - 200, 0), min(f + 400, len(self.xy) - 1)
        A, B = self.xy[lo:hi], self.xy[lo + 1:hi + 1]
        ab = B - A
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        u = np.clip(((p - A) * ab).sum(1) / L2, 0, 1)
        d = np.hypot(*(A + u[:, None] * ab - p).T)
        i = int(d.argmin())
        return float(self.s[lo + i] + u[i] * np.sqrt(L2[i])), float(d[i])

    def path(self, s_e):
        g = s_e + np.arange(-NF.TC_BACK, NF.TC_AHEAD + 1, dtype=np.float64)
        X = self.xy[self.fi]
        return np.stack([np.interp(g, self.S, X[:, 0]), np.interp(g, self.S, X[:, 1])], -1)


def state(L, st, s0, C, t):
    """one decision -> dict(raw (8,), reach (8,), turn, net, s_e, lat, v_ego) or None (the log ends before 4 s)."""
    HZ = 20
    f = s0 + int(round(HZ * t))
    if f + 4 * HZ >= len(L.xy):
        return None
    dyaw = L.yaw[s0] - C["yaw"][0]
    R = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
    i = np.clip(np.searchsorted(C["t"], t), 1, len(C["t"]) - 1)
    w = np.clip((t - C["t"][i - 1]) / max(C["t"][i] - C["t"][i - 1], 1e-6), 0, 1)
    exy = R @ (C["xy"][i - 1] + w * (C["xy"][i] - C["xy"][i - 1]) - C["xy"][0]) + L.xy[s0]
    ve = float(C["v"][i - 1] + w * (C["v"][i] - C["v"][i - 1]))
    seg = np.hypot(*np.diff(L.xy[f:f + 4 * HZ + 1], axis=0).T)
    raw = np.cumsum(seg)[HZ // 2 - 1::HZ // 2]
    tau = np.arange(4 * HZ) / HZ
    reach = np.cumsum((ve + (L.v[f:f + 4 * HZ] - ve) * np.minimum(tau / 2, 1)) / HZ)[HZ // 2 - 1::HZ // 2]
    s_e, lat = st.project(exy, f)
    rg, nt = NF.turn_range(st.path(s_e)[None])
    return dict(raw=raw, reach=reach, turn=float(rg[0]), net=float(nt[0]), s_e=s_e, lat=lat, v_ego=ve)


def _read_job(job):
    """one arena run -> decision dicts (k >= 2) with the turn-cap rows / cap / fire on the stored plan."""
    import torch
    import free_gate as FG
    torch.set_num_threads(1)
    read, run, want = job
    R = FG.run_rollouts(run)
    logs, out, P, CAP = {}, [], [], []
    for sc, (C, dec, r) in R.items():
        if sc not in want:
            continue
        log, s0, tok = want[sc]
        if log not in logs:
            L = FG.Log(log)
            logs[log] = (L, Stations(L.xy))
        L, st = logs[log]
        assert L.tok[s0 + 30].hex() == tok, sc
        for k, t, p in dec:
            if k < 2:
                continue
            ve = float(np.interp(t, C["t"], C["v"]))
            d = dict(read=read, id=sc, k=k, t=t, v_ego=ve, score=float(r.get("score") or 0.0), dom=False, fire=False)
            if ve < NF.V_TC:                                      # below the row's speed: 0 by construction, no state needed
                out.append(d)
                continue
            q = state(L, st, s0, C, t)
            if q is None:
                out.append(d)
                continue
            d.update(turn=q["turn"], net=q["net"], lat=q["lat"], raw4=float(q["raw"][-1]), reach4=float(q["reach"][-1]),
                     dom=bool(NF.turncap_rows(np.array([ve]), np.array([q["raw"][-1]]), np.array([q["turn"]]))[0]))
            out.append(d)
            P.append(np.asarray(p, np.float32).reshape(8, 3))
            CAP.append((NF.K_TC * np.maximum(q["raw"], q["reach"])).astype(np.float32))
            d["_j"] = len(P) - 1
    if P:
        _, pr = NF.stand_terms(torch.as_tensor(np.stack(P)), torch.as_tensor(np.stack(CAP)), torch.ones(len(P), dtype=torch.bool), parts=True)
        sp, fire = pr["s_plan"].numpy(), pr["fire"].numpy()
        for d in out:
            j = d.pop("_j", None)
            if j is not None:
                d.update(plan4=float(sp[j]), cap4=float(CAP[j][-1]), over=bool(fire[j]), fire=bool(fire[j]) and d["dom"])
    return out


def cmd_gate(a):
    import pandas as pd
    import torch
    from multiprocessing import Pool
    G = dict(rule=dict(v_tc=NF.V_TC, h_tc=NF.H_TC, turn_tc=NF.TURN_TC, k_tc=NF.K_TC, window=[-NF.TC_BACK, NF.TC_AHEAD]))
    # G1: the human's own paths on the pool's domain rows
    tab = np.load(CR / "on1_JTX/tab.npz")
    pk = NF.turncap_load(CR / "on1_JTX", tab["names"], "cpu")
    ok = pk["ok"].numpy()
    g1 = {}
    for k in ("fut", "fut_reg"):
        _, pr = NF.stand_terms(torch.as_tensor(np.nan_to_num(tab[k][ok]).astype(np.float32)), pk["cap"][torch.as_tensor(ok)],
                               torch.ones(int(ok.sum()), dtype=torch.bool), parts=True)
        g1[k] = int(pr["fire"].sum())
    G["pool"] = pk["rec"]
    G["G1"] = dict(fires=g1, rows=int(ok.sum()), pass_=sum(g1.values()) == 0)
    del tab
    # decisions of B2's full-length arena reads (s1, s0), every held window
    Sx = pd.read_csv(AR / "set/set.csv.gz").drop_duplicates("id").set_index("id")
    held = Sx[Sx.split == "held"]
    want = {i: (w.log, int(w.start_idx), w.token) for i, w in held.iterrows()}
    jobs = [(sd, run, want) for sd, tag in B2.items() for run in sorted(glob.glob(str(O / "jt/arena" / f"{tag}@off" / "c*/run")))
            if (Path(run) / "aggregate/results-summary.json").exists()]
    print(f"{len(jobs)} runs", flush=True)
    with Pool(a.workers) as p:
        out = p.map(_read_job, jobs, chunksize=1)
    D = pd.DataFrame([d for o in out for d in o])
    D["short"] = D.id.str[-8:]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    D.to_parquet(Path(a.out) / "decisions.parquet")
    W = pd.read_csv(REPO / "experiments/alpasim_proxy/results/lat_zero/windows.csv")
    # G2: the 9 target windows' pre-exit decisions (s1: k >= 2, t in [t_ev - 3 s, t_ev)); s0's a2 windows described
    G["per_read"] = {}
    for sd in B2:
        Wt = W[(W.seed == sd) & W["sub"].astype(str).str.startswith("a2") | ((W.seed == sd) & W.short.isin(TARGETS))].drop_duplicates("short")
        rows = []
        for w in Wt.itertuples():
            g = D[(D.read == sd) & (D.short == w.short) & (D.t >= w.t_ev - 3.0) & (D.t < w.t_ev)]
            rows.append(dict(window=w.short, target=w.short in TARGETS, t_ev=w.t_ev, pre=len(g), domain=int(g.dom.sum()), fired=int(g.fire.sum()),
                             over_cap=int(g.over.fillna(False).sum()), turn_med=float(g.turn.median()) if g.turn.notna().any() else None,
                             plan_over_cap_med=float((g.plan4 / g.cap4).median()) if g.plan4.notna().any() else None))
        G["per_read"][sd] = dict(windows=rows)
    t1 = [r for r in G["per_read"]["s1"]["windows"] if r["target"]]
    pre, fired = sum(r["pre"] for r in t1), sum(r["fired"] for r in t1)
    wf = sum(r["fired"] > 0 for r in t1)
    G["G2"] = dict(windows=len(t1), windows_fired=wf, decisions=pre, fired=fired, share=fired / max(pre, 1), lines=dict(share=0.5, windows=6),
                   pass_=len(t1) == 9 and fired / max(pre, 1) >= 0.5 and wf >= 6)
    # G3: B2 s1's held windows scored 1: decisions (k >= 2) that fire, and windows with any fire
    M = D[(D.read == "s1") & (D.score >= 1 - 1e-4)]
    st = Sx.stratum.reindex(M.id)
    fw = M[M.fire].id.unique()
    G["G3"] = dict(windows=int(M.id.nunique()), decisions=len(M), fired=int(M.fire.sum()), share=float(M.fire.mean()), windows_fired=int(len(fw)),
                   windows_fired_share=float(len(fw) / max(M.id.nunique(), 1)), lines=dict(share=0.01, windows=0.05),
                   by_stratum=st[M.fire.values].value_counts().to_dict(), domain_decisions=int(M.dom.sum()),
                   domain_fire_share=float(M[M.dom].fire.mean()) if M.dom.any() else None)
    G["G3"]["pass_"] = G["G3"]["share"] <= 0.01 and G["G3"]["windows_fired_share"] <= 0.05
    # description: every read, every decision; sharp-turn windows (d395's set) pass / fail; low-speed axis (0 by construction)
    T = set(pd.read_csv(RES / "turn_set.csv").id)
    G["describe"] = {sd: dict(decisions=int((D.read == sd).sum()), domain=int((D.dom & (D.read == sd)).sum()), fired=int((D.fire & (D.read == sd)).sum()),
                              fired_right=int((D.fire & (D.read == sd) & (D.net < 0)).sum()),
                              fired_windows=int(D[D.fire & (D.read == sd)].id.nunique()),
                              fired_windows_outside_turn_set=int(D[D.fire & (D.read == sd) & ~D.id.isin(T)].id.nunique()),
                              turn_set_windows_fired_pass=int(D[D.fire & (D.read == sd) & D.id.isin(T) & (D.score >= 1 - 1e-4)].id.nunique()),
                              turn_set_windows_fired_fail=int(D[D.fire & (D.read == sd) & D.id.isin(T) & (D.score < 1 - 1e-4)].id.nunique()),
                              below_v_tc=int(((D.read == sd) & (D.v_ego < NF.V_TC)).sum())) for sd in B2}
    G["pass_"] = bool(G["G1"]["pass_"] and G["G2"]["pass_"] and G["G3"]["pass_"])
    for f in (Path(a.out) / "gate.json", RES / "gate.json"):
        f.write_text(json.dumps(G, indent=1, default=float))
    print(json.dumps(G, indent=1, default=float))
    print("GATE", "PASS" if G["pass_"] else "FAIL")
    raise SystemExit(0 if G["pass_"] else 2)


def cmd_lists(a):
    """the turn-set windows outside the subset chunks, by cache shard -> $S/lists/ct<shard>.txt (+ lists.json)."""
    m = json.load(open(AR / "hold/meta.json"))["chunks"]
    T = [ln.split(",")[0] for ln in open(RES / "turn_set.csv").read().split()[1:]]
    where = {}
    for c in m:
        for i in open(AR / f"hold/lists/c{c['chunk']}.txt").read().split():
            where[i] = (c["chunk"], c["shard"])
    miss = [i for i in T if i not in where]
    assert not miss, f"turn-set windows not in the hold lists: {miss[:3]}"
    by = defaultdict(list)
    for i in T:
        c, k = where[i]
        if c not in SUBSET:
            by[k].append(i)
    (S_ / "lists").mkdir(parents=True, exist_ok=True)
    for k, ids in by.items():
        (S_ / f"lists/ct{k}.txt").write_text("\n".join(ids) + "\n")
    rec = dict(turn_set=len(T), in_subset=sum(where[i][0] in SUBSET for i in T), outside=sum(map(len, by.values())), by_shard={k: len(v) for k, v in by.items()})
    (S_ / "lists.json").write_text(json.dumps(rec, indent=1))
    print(" ".join(str(k) for k in sorted(by)))


def _scalars(run):
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{run}/*/"))[-1]
    sc = {}
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc.setdefault(e["tag"], {})[e["step"]] = e["value"]
    return sc


def cmd_dose(a):
    """the frozen 100-step run (both FREE-M at B2's 0.001 and TURN-CAP at W0 on): finite losses, the term fires at some step; the dose
    W1 = W0 x mean grad/lon_free / mean grad/lon_turncap (2 significant, clipped to W_RANGE), W3 = DOSE_B x W1."""
    sc = _scalars(a.run)
    fin = all(np.isfinite(v) for t, x in sc.items() if t.startswith("loss/") for v in x.values())
    gf, gt = list(sc.get("loss/grad/lon_free", {}).values()), list(sc.get("loss/grad/lon_turncap", {}).values())
    pos = sc.get("loss/on/lon_turncap_pos", {})
    lab = [ln.strip() for ln in open(a.log) if "lon_turncap labels" in ln]
    mf, mt = (float(np.mean(gf)) if gf else np.nan), (float(np.mean(gt)) if gt else np.nan)
    raw = W0 * mf / mt if np.isfinite(mf) and np.isfinite(mt) and mt > 0 else np.nan
    w1 = float(np.clip(float(f"{raw:.2g}"), *W_RANGE)) if np.isfinite(raw) else np.nan
    r = dict(run=a.run, finite=bool(fin), steps=len(pos), pos_max=max(pos.values()) if pos else None, pos_mean=float(np.mean(list(pos.values()))) if pos else None,
             free_pos_mean=float(np.mean(list(sc.get("loss/on/lon_free_pos", {0: np.nan}).values()))),
             grad_free_w0001=mf, grad_turncap_w0004=mt, n_grad=[len(gf), len(gt)], grad_turncap_zero_steps=int(sum(g == 0 for g in gt)),
             W_raw=raw, clipped=bool(np.isfinite(raw) and w1 != float(f"{raw:.2g}")), W=dict(T1=w1, T3=float(f"{DOSE_B * w1:.2g}")), labels=lab[-1] if lab else "")
    r["pass_"] = bool(fin and pos and r["pos_max"] > 0 and np.isfinite(w1))
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


# ---------------------------------------------------------------- Mac: the stage 1 rule
def classify_read(cx, R, tag, dec, keep):
    """-> per-window cause table (easy items only) and the read set's PCS / mean for one tag."""
    import easy_items as EI
    import native_gap_report as NG
    D, _, St, _ = NG.load(Path(dec))
    Dv = D[D.valid & (D.k >= 2) & (D.tag == tag)].sort_values("k")
    rear = St[St.tag == tag].groupby("id").collision_rear.max() > 0 if "collision_rear" in St else None
    g = R[R.tag == tag].drop_duplicates("id").set_index("id")
    ids = [i for i in cx["items"] if i in keep]
    y, typ = g.score.reindex(ids), g.type.reindex(ids)
    C = EI.classify(y, typ, ids, cx["S"], cx["cue"], cx["hoff"], {i: q for i, q in Dv.groupby("id")}, rear)
    E = C[C.id.isin(cx["easy"]) & ~C.cause.str.startswith("unsolvable")] if len(C) else C
    yy = g.score.reindex(cx["z"].items).to_numpy(float)
    yy = np.where(np.isin(cx["z"].items, ids), yy, np.nan)
    sub = [i for i in ids if cx["chunk"].get(i) in SUBSET]
    ys = np.where(np.isin(cx["z"].items, sub), yy, np.nan)
    return E, dict(windows=len(ids), mean=float(y.mean()), zeros=int((y <= 1e-4).sum()), pcs=cx["z"].pcs_of(yy), pcs_6chunks=cx["z"].pcs_of(ys),
                   easy_windows=len(set(ids) & cx["easy"]), scores=y)


def cmd_stage1(a):
    import pandas as pd
    import stand_r as SR
    cx = SR._ctx()
    lists = REPO / "tmp/turn_cap/hold_lists"
    cx["chunk"] = {i: int(f.stem[1:]) for f in lists.glob("c*.txt") for i in f.read_text().split()}
    assert cx["chunk"], f"no hold lists in {lists} (scp them from the box)"
    T = set(pd.read_csv(RES / "turn_set.csv").id)
    R = pd.read_csv(a.rows)
    tags = [CTL] if a.anchors else [CTL, *ARMS.values()]
    keep = set.intersection(*(set(R[(R.tag == t) & R.score.notna()].id) for t in tags))
    E, M = {}, {}
    for t in tags:
        E[t], M[t] = classify_read(cx, R, t, a.dec, keep)
    tgt = lambda e: e[e.cause.isin(LAT3) & e.id.isin(T)]  # noqa: E731
    oth = lambda e: e[~(e.cause.isin(LAT3) & e.id.isin(T))].cause.value_counts().to_dict()  # noqa: E731
    c = M[CTL]
    J = dict(read_set=len(keep & set(cx["items"])), turn_set_read=len(keep & T), control=dict(
        target=len(tgt(E[CTL])), target_ids=sorted(tgt(E[CTL]).id), other=oth(E[CTL]), **{k: v for k, v in c.items() if k != "scores"}))
    if a.anchors:
        print(json.dumps(J, indent=1, default=float))
        return
    need = max(2, int(np.ceil(0.25 * J["control"]["target"])))
    L = ["# TURN-CAP pilot stage 1 (NR1 prereg supplement 21; native reads; control = the stored B2 pilot STANDFMXH1K-L18-s0)", "",
         f"Read set = arena subset (chunks 0 / 3 / 6 / 9 / 12 / 15) + the turn set (d395's sharp-turn held windows + the 9 targets): "
         f"{J['read_set']} frozen board-mix windows every read has ({J['turn_set_read']} of them in the turn set).", ""]
    J["arms"] = {}
    for A, Tg in ARMS.items():
        t = json.load(open(RES / f"s17table_{A}.json"))
        H, Wt, Ar = t["held"], t["wait"], t["arena"]
        e, m = E[Tg], M[Tg]
        oc, oa = J["control"]["other"], oth(e)
        rows = [("main 1: easy lateral zeros in the turn set", J["control"]["target"], len(tgt(e)), len(tgt(e)) <= J["control"]["target"] - need,
                 f"<= control - {need}"),
                ("main 2: read-set PCS (frozen board-mix)", round(c["pcs"]), round(m["pcs"]), m["pcs"] >= c["pcs"] + 10, ">= control + 10"),
                *[(f"guard 1: easy {k}", oc.get(k, 0), oa.get(k, 0), oa.get(k, 0) <= oc.get(k, 0) + 2, "<= control + 2") for k in sorted(set(oc) | set(oa))],
                ("guard 2: held 460 zeros", H[CTL]["zeros"], H[Tg]["zeros"], H[Tg]["zeros"] <= H[CTL]["zeros"] + 2, "<= control + 2"),
                ("guard 2: held 460 front collisions", H[CTL]["coll"]["front"], H[Tg]["coll"]["front"], H[Tg]["coll"]["front"] <= H[CTL]["coll"]["front"] + 2, "<= control + 2"),
                ("guard 2: read-set AF-front windows", Ar["af_front_b"], Ar["af_front_a"], Ar["af_front_a"] <= Ar["af_front_b"] + 2, "<= control + 2"),
                ("guard 3: read-set slow partial windows", t["partials"][CTL]["slow"], t["partials"][Tg]["slow"], t["partials"][Tg]["slow"] <= t["partials"][CTL]["slow"] + 2, "<= control + 2"),
                ("guard 3: read-set k = 9 speed ratio", round(Ar["k9"]["b"], 4), round(Ar["k9"]["a"], 4), Ar["k9"]["a"] >= Ar["k9"]["b"] - 0.03, ">= control - 0.03"),
                *[(f"guard 4: wait a {k}", round(Wt[CTL][k]["wait"], 3), round(Wt[Tg][k]["wait"], 3), Wt[Tg][k]["wait"] >= Wt[CTL][k]["wait"] - 0.03, ">= control - 0.03") for k in Wt[Tg]],
                *[(f"guard 5: wait b median launch delay {k} (s)", round(Wt[CTL][k]["delay_med"], 2), round(Wt[Tg][k]["delay_med"], 2),
                   Wt[Tg][k]["delay_med"] <= Wt[CTL][k]["delay_med"] + 0.15, "<= control + 0.15") for k in Wt[Tg]],
                *[(f"guard 5: wait b launched {k}", round(Wt[CTL][k]["launched"], 3), round(Wt[Tg][k]["launched"], 3),
                   Wt[Tg][k]["launched"] >= 0.95 * Wt[CTL][k]["launched"], ">= 0.95 x control") for k in Wt[Tg]]]
        ok = all(r[3] for r in rows)
        ys, yc = m["scores"], c["scores"]
        tw = {i[-8:]: (float(yc.get(i, np.nan)), float(ys.get(i, np.nan))) for i in ys.index if i[-8:] in TARGETS}
        J["arms"][A] = dict(tag=Tg, rows=[dict(read=r[0], control=r[1], arm=r[2], ok=bool(r[3]), line=r[4]) for r in rows], pass_=ok,
                            mean=[c["mean"], m["mean"]], zeros=[c["zeros"], m["zeros"]], pcs_6chunks=[c["pcs_6chunks"], m["pcs_6chunks"]],
                            held_mean=[H[CTL]["mean"], H[Tg]["mean"]], targets=tw, target_ids=sorted(tgt(e).id),
                            fixed=sorted(set(J["control"]["target_ids"]) - set(tgt(e).id)), new=sorted(set(tgt(e).id) - set(J["control"]["target_ids"])))
        L += [f"## {A} `{Tg}`: {'PASS' if ok else 'FAIL'}", "", f"| read | {CTL} | {Tg} | line |", "|:--|--:|--:|:--|",
              *[f"| {r[0]} | {r[1]} | {r[2]} | {r[4]}: {'ok' if r[3] else 'FAIL'} |" for r in rows], "",
              f"- read-set mean {c['mean']:.4f} -> {m['mean']:.4f}, zeros {c['zeros']} -> {m['zeros']}; PCS on the 6 chunks only "
              f"{c['pcs_6chunks']:.0f} -> {m['pcs_6chunks']:.0f}; held 460 mean {H[CTL]['mean']:.4f} -> {H[Tg]['mean']:.4f}",
              f"- turn-set easy lateral zeros fixed {len(J['arms'][A]['fixed'])}, new {len(J['arms'][A]['new'])}",
              "- the 9 target windows (control -> arm): " + ", ".join(f"{k} {v[0]:.2f} -> {v[1]:.2f}" for k, v in sorted(tw.items())), ""]
    (RES / "stage1.json").write_text(json.dumps(J, indent=1, default=float))
    (RES / "stage1.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("label")
    p.add_argument("--pool", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=18)
    sp.add_parser("lists")
    p = sp.add_parser("dose")
    p.add_argument("--run", required=True), p.add_argument("--log", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("stage1")
    p.add_argument("--rows", default=str(RES / "arena_windows.csv.gz")), p.add_argument("--dec", default=str(REPO / "tmp/turn_cap/ext"))
    p.add_argument("--anchors", action="store_true")
    a = ap.parse_args()
    dict(label=cmd_label, gate=cmd_gate, lists=cmd_lists, dose=cmd_dose, stage1=cmd_stage1)[a.cmd](a)
