#!/usr/bin/env python3
"""TIGHT-IN (NR1 prereg supplement 22, plans/2026-10-20-nr1-supp22-tightin.md; lib/nr1_tight.py, rm_train --on-tightin W).

  $TR tight_in.py label --pool on1_JTX     (box, CPU) the tight-turn labels of the pool's S rows from rows meta `gpath`: the lateral path
                                           (stations -1 .. +25 m), kmax, dh -> $CR/<pool>/tight_in.npz; results/tight_in/label.json
                                           (domain counts, JTX00 draw share, stored served-plan fires by part)
  $TR tight_in.py gate --out DIR           (box, CPU) offline gate G1-G3 on the pool and B2's stored full-length arena reads -> DIR/gate.json,
                                           DIR/decisions.parquet, results/tight_in/gate.json; exit 0 pass, 2 fail
  $TR tight_in.py lists                    (box) the read set's windows (turn set + tight set) outside the six subset chunks, by cache shard
                                           -> $S/lists/ct<k>.txt
  $TR tight_in.py dose --run T --log L --out JSON   100-step frozen eager run at W0: finite, the term fires, the dose rule
  tmp/pcs_local/.venv/bin/python tight_in.py stage1 [--anchors]   (Mac) the stage 1 rule on the pulled reads -> results/tight_in/stage1.{json,md}

Closed-loop decisions (gate): turn_cap's conventions (free_gate: scene time t <-> log frame start_idx + round(20 t), the controller trace
moved rigidly onto the log at start_idx, the human's raw / reachable 4 s travel). The ego pose at t = the interpolated controller trace; the
logged path at 1 m stations over [s_e - 15, s_e + 70] m (s_e = the ego's projection onto the log polyline) in the ego frame plays `gpath`;
the plan = drive.jsonl `poses`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
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
import nr1_tight as NT  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix"
S_ = O / "tight_in"
CR = DATA / "runs/op_parity/cache"
AR = DATA / "runs/alpasim_proxy/arena"
RES = REPO / "experiments/defect_fix/results/tight_in"
ROWS = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y", **{f"pn{i}": f"PN{i}" for i in range(7)}}
W0, W_RANGE, DOSE_B = 0.004, (0.002, 0.04), 3.0
B2 = {"s1": "STANDFMXH-L18-s1", "s0": "STANDFMXH-L18-s0"}
TARGETS = ("098f5c53", "202f583c", "3f805a18", "476b54f9", "6f27563e", "87705868", "9e7c5a46", "efe356db", "f2305669")
SUBSET = (0, 3, 6, 9, 12, 15)
LAT3 = ("zero: turn window, offroad / corridor", "zero: offroad, non-turn", "zero: corridor, moving non-turn")
CTL, ARMS = "STANDFMXH1K-L18-s0", {"T1": "TINA1K-L18-s0", "T3": "TINB1K-L18-s0"}
SPAN = (3.0, 1.0)                                  # G2: decisions 3 s .. 1 s before the scored exit (TURN-SEP's span)


# ---------------------------------------------------------------- pool labels
def cmd_label(a):
    import torch
    tab = np.load(CR / a.pool / "tab.npz")
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    part = np.array([n.split(":")[1] for n in names])
    n = len(names)
    path = np.full((n, NT.NP, 2), np.nan, np.float32)
    km, dh = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    served = np.full((n, 8, 3), np.nan, np.float32)
    for p, d in ROWS.items():
        r = np.flatnonzero(S & (part == p))
        if not len(r):
            continue
        z = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        pos = {q: i for i, q in enumerate(z["names"].astype(str).tolist())}
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in r])
        assert np.allclose(z["speed"][mi], tab["speed"][r], atol=1e-3), f"{d}: meta speed != the pool's"
        g = z["gpath"][mi]
        path[r] = g[:, NT.GI + NT.LO:NT.GI + NT.HI + 1]
        km[r], dh[r] = NT.geom(g)
        served[r] = z["poses"][mi]
    assert np.isfinite(km[S]).mean() > 0.99, "S rows without a label"
    f = CR / a.pool / "tight_in.npz"
    assert not f.exists(), f"{f} exists: never overwritten"
    np.savez(f, names=names, path=path, kmax=km, dh=dh)
    # description: the domain, its draw share, the stored served plans' fires by part
    pk = NT.load(CR / a.pool, names, "cpu")
    ok = pk["ok"].numpy()
    w = np.load(DATA / "runs/defect_fix/w/JTX00.npz")
    assert np.array_equal(w["names"].astype(str), names), "JTX00 rows != the pool's"
    ww = w["w"].astype(np.float64) / w["w"].sum()
    j = np.flatnonzero(ok & np.isfinite(served).all((1, 2)))
    jt = torch.as_tensor(j)
    _, _, pr = NT.terms(torch.as_tensor(served[j]), pk["cap"][jt], pk["path"][jt], pk["sigma"][jt], torch.ones(len(j), dtype=torch.bool), parts=True)
    fire = np.zeros(n, bool)
    fire[j] = pr["fire"].numpy()
    hum4 = NF.cum_arc(tab["fut_reg"])[:, -1]
    turn = S & (tab["speed"] >= NT.V_TI) & (np.abs(np.nan_to_num(dh)) >= NT.TURN_TI)
    q = lambda x: np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if len(x) else []  # noqa: E731
    rec = dict(pool=a.pool, rows=n, s_rows=int(S.sum()), labelled=int(np.isfinite(km[S]).sum()), moving_turn=int(turn.sum()),
               tight_moving_turn=int((turn & (np.nan_to_num(km) >= NT.K_TI)).sum()),
               blocked_human_arc=int((turn & (np.nan_to_num(km) >= NT.K_TI) & (hum4 < NT.H_TI)).sum()),
               domain=int(ok.sum()), domain_right=int((ok & (dh < 0)).sum()), domain_left=int((ok & (dh >= 0)).sum()),
               domain_share_of_s=float(ok.sum() / S.sum()), domain_logs=int(len(np.unique(tab["log"].astype(str)[ok]))),
               draw_share_jtx00=dict(domain=float(ww[ok].sum()), s_rows=float(ww[S].sum()), per_batch_70=float(70 * ww[ok].sum())),
               served_fire=dict(rows=int(len(j)), fires=int(fire.sum()), share=float(fire[j].mean()) if len(j) else None,
                                lat=int(pr["fire_lat"].sum()), arc=int(pr["fire_arc"].sum()), both=int((pr["fire_lat"] & pr["fire_arc"]).sum()),
                                draw_share=float(ww[fire].sum()), outward_q=q(pr["out"].numpy())),
               by_part=dict(Counter(part[ok].tolist())), kmax_q_domain=q(km[ok]), rec=pk["rec"])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "label.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


# ---------------------------------------------------------------- closed-loop decisions on stored arena reads
def state(L, st, s0, C, t):
    """one decision -> dict(raw (8,), reach (8,), gpath (86, 2) ego frame, s_e) or None (the log ends before 4 s)."""
    HZ = 20
    f = s0 + int(round(HZ * t))
    if f + 4 * HZ >= len(L.xy):
        return None
    dyaw = L.yaw[s0] - C["yaw"][0]
    R = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
    i = np.clip(np.searchsorted(C["t"], t), 1, len(C["t"]) - 1)
    w = np.clip((t - C["t"][i - 1]) / max(C["t"][i] - C["t"][i - 1], 1e-6), 0, 1)
    exy = R @ (C["xy"][i - 1] + w * (C["xy"][i] - C["xy"][i - 1]) - C["xy"][0]) + L.xy[s0]
    yw = np.unwrap(np.asarray(C["yaw"], np.float64))
    eyaw = float(yw[i - 1] + w * (yw[i] - yw[i - 1]) + dyaw)
    ve = float(C["v"][i - 1] + w * (C["v"][i] - C["v"][i - 1]))
    seg = np.hypot(*np.diff(L.xy[f:f + 4 * HZ + 1], axis=0).T)
    raw = np.cumsum(seg)[HZ // 2 - 1::HZ // 2]
    tau = np.arange(4 * HZ) / HZ
    reach = np.cumsum((ve + (L.v[f:f + 4 * HZ] - ve) * np.minimum(tau / 2, 1)) / HZ)[HZ // 2 - 1::HZ // 2]
    s_e, _ = st.project(exy, f)
    g = s_e + np.arange(-15.0, 71.0)
    X = st.xy[st.fi]
    P = np.stack([np.interp(g, st.S, X[:, 0]), np.interp(g, st.S, X[:, 1])], -1) - exy
    c, s = np.cos(eyaw), np.sin(eyaw)
    gp = np.stack([c * P[:, 0] + s * P[:, 1], -s * P[:, 0] + c * P[:, 1]], -1)
    return dict(raw=raw, reach=reach, gpath=gp, s_e=s_e, v_ego=ve)


def _read_job(job):
    """one arena run -> decision dicts (k >= 2) with the TIGHT-IN rows / fires (both parts) on the stored plan."""
    import torch
    import free_gate as FG
    import turn_cap as TC
    torch.set_num_threads(1)
    read, run, want = job
    R = FG.run_rollouts(run)
    logs, out, P, CAP, GP, SG = {}, [], [], [], [], []
    for sc, (C, dec, r) in R.items():
        if sc not in want:
            continue
        log, s0, tok = want[sc]
        if log not in logs:
            L = FG.Log(log)
            logs[log] = (L, TC.Stations(L.xy))
        L, st = logs[log]
        assert L.tok[s0 + 30].hex() == tok, sc
        for k, t, p in dec:
            if k < 2:
                continue
            ve = float(np.interp(t, C["t"], C["v"]))
            d = dict(read=read, id=sc, k=k, t=t, v_ego=ve, score=float(r.get("score") or 0.0), dom=False, fire=False, fire_lat=False, fire_arc=False)
            if ve < NT.V_TI:                                      # below the row's speed: 0 by construction, no state needed
                out.append(d)
                continue
            q = state(L, st, s0, C, t)
            if q is None:
                out.append(d)
                continue
            km, dh = NT.geom(q["gpath"][None])
            d.update(kmax=float(km[0]), dh=float(dh[0]), raw4=float(q["raw"][-1]),
                     dom=bool(NT.rows(np.array([ve]), np.array([q["raw"][-1]]), km, dh)[0]))
            out.append(d)
            if d["dom"]:
                P.append(np.asarray(p, np.float32).reshape(8, 3))
                CAP.append((NT.K_ARC * np.maximum(q["raw"], q["reach"])).astype(np.float32))
                GP.append(q["gpath"][NT.GI + NT.LO:NT.GI + NT.HI + 1].astype(np.float32))
                SG.append(1.0 if dh[0] >= 0 else -1.0)
                d["_j"] = len(P) - 1
    if P:
        n = len(P)
        _, _, pr = NT.terms(torch.as_tensor(np.stack(P)), torch.as_tensor(np.stack(CAP)), torch.as_tensor(np.stack(GP)),
                            torch.as_tensor(SG, dtype=torch.float32), torch.ones(n, dtype=torch.bool), parts=True)
        f, fl, fa, ow = (pr[k].numpy() for k in ("fire", "fire_lat", "fire_arc", "out"))
        for d in out:
            j = d.pop("_j", None)
            if j is not None:
                d.update(fire=bool(f[j]), fire_lat=bool(fl[j]), fire_arc=bool(fa[j]), out=float(ow[j]))
    return out


def cmd_gate(a):
    import pandas as pd
    import torch
    from multiprocessing import Pool
    G = dict(rule=dict(v_ti=NT.V_TI, h_ti=NT.H_TI, turn_ti=NT.TURN_TI, k_ti=NT.K_TI, m_ti=NT.M_TI, k_arc=NT.K_ARC, stations=[NT.LO, NT.HI]))
    # G1: the human's own paths on the pool's domain rows (both parts)
    tab = np.load(CR / "on1_JTX/tab.npz")
    pk = NT.load(CR / "on1_JTX", tab["names"], "cpu")
    ok = pk["ok"].numpy()
    okt = torch.as_tensor(ok)
    g1 = {}
    for k in ("fut", "fut_reg"):
        _, _, pr = NT.terms(torch.as_tensor(np.nan_to_num(tab[k][ok]).astype(np.float32)), pk["cap"][okt], pk["path"][okt], pk["sigma"][okt],
                            torch.ones(int(ok.sum()), dtype=torch.bool), parts=True)
        g1[k] = dict(lat=int(pr["fire_lat"].sum()), arc=int(pr["fire_arc"].sum()), out_max=float(pr["out"].max()) if ok.any() else None)
    G["pool"] = pk["rec"]
    G["G1"] = dict(fires=g1, rows=int(ok.sum()), pass_=sum(v["lat"] + v["arc"] for v in g1.values()) == 0)
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
    W = pd.read_csv(RES / "gate_windows.csv")
    # G2: B2's sharp-set lateral zeros: any fire in the decisions 3 s .. 1 s before the scored exit; the 8 sharp-set targets
    G["per_read"] = {}
    for sd in B2:
        rows = []
        for w in W[(W.seed == sd) & W.latzero].itertuples():
            g = D[(D.read == sd) & (D.id == w.id) & (D.t >= w.t_ev - SPAN[0]) & (D.t <= w.t_ev - SPAN[1])] if np.isfinite(w.t_ev) else D.iloc[:0]
            rows.append(dict(window=w.short, target=bool(w.target), tight=bool(w.tight), t_ev=w.t_ev, pre=len(g), domain=int(g.dom.sum()),
                             fired=int(g.fire.sum()), fired_lat=int(g.fire_lat.sum()), fired_arc=int(g.fire_arc.sum())))
        R_ = pd.DataFrame(rows)
        G["per_read"][sd] = dict(windows=rows, fail_windows=len(R_), fail_hit=int((R_.fired > 0).sum()), tight_fail=int(R_.tight.sum()),
                                 tight_fail_hit=int((R_.tight & (R_.fired > 0)).sum()), targets=int(R_.target.sum()),
                                 targets_hit=int((R_.target & (R_.fired > 0)).sum()))
    p1 = G["per_read"]["s1"]
    G["G2"] = dict(fail_hit=p1["fail_hit"], fail_windows=p1["fail_windows"], share=p1["fail_hit"] / max(p1["fail_windows"], 1),
                   targets_hit=p1["targets_hit"], targets=p1["targets"], lines=dict(share=0.5, targets=5),
                   pass_=p1["fail_hit"] / max(p1["fail_windows"], 1) >= 0.5 and p1["targets_hit"] >= 5)
    # G3: B2 s1's held windows scored 1: decisions (k >= 2) that fire, windows with any fire; s0 described the same way
    G["G3"] = {}
    for sd in B2:
        M = D[(D.read == sd) & (D.score >= 1 - 1e-4)]
        fw = M[M.fire].id.unique()
        st = Sx.stratum.reindex(M.id)
        G["G3"][sd] = dict(windows=int(M.id.nunique()), decisions=len(M), fired=int(M.fire.sum()), fired_lat=int(M.fire_lat.sum()),
                           fired_arc=int(M.fire_arc.sum()), share=float(M.fire.mean()), windows_fired=int(len(fw)),
                           windows_fired_share=float(len(fw) / max(M.id.nunique(), 1)), domain_decisions=int(M.dom.sum()),
                           domain_share=float(M.dom.mean()), domain_fire_share=float(M[M.dom].fire.mean()) if M.dom.any() else None,
                           by_stratum=st[M.fire.values].value_counts().to_dict(),
                           tight_set_windows_fired=int(len(set(fw) & set(W[W.tight].id))))
    g3 = G["G3"]["s1"]
    G["G3"]["lines"] = dict(share=0.01, windows=0.05)
    G["G3"]["pass_"] = g3["share"] <= 0.01 and g3["windows_fired_share"] <= 0.05
    G["describe"] = {sd: dict(decisions=int((D.read == sd).sum()), domain=int((D.dom & (D.read == sd)).sum()), fired=int((D.fire & (D.read == sd)).sum()),
                              fired_windows=int(D[D.fire & (D.read == sd)].id.nunique()),
                              fired_right=int((D.fire & (D.read == sd) & (D.dh < 0)).sum()),
                              out_q_domain=np.nanquantile(D[D.dom & (D.read == sd)].out, [0.1, 0.5, 0.9]).round(3).tolist() if (D.dom & (D.read == sd)).any() else [])
                     for sd in B2}
    G["pass_"] = bool(G["G1"]["pass_"] and G["G2"]["pass_"] and G["G3"]["pass_"])
    for f in (Path(a.out) / "gate.json", RES / "gate.json"):
        f.write_text(json.dumps(G, indent=1, default=float))
    print(json.dumps(G, indent=1, default=float))
    print("GATE", "PASS" if G["pass_"] else "FAIL")
    raise SystemExit(0 if G["pass_"] else 2)


def read_set():
    import pandas as pd
    return sorted(set(pd.read_csv(REPO / "experiments/defect_fix/results/turn_cap/turn_set.csv").id) | set(pd.read_csv(RES / "tight_set.csv").id))


def cmd_lists(a):
    """the read set's windows outside the subset chunks, by cache shard -> $S/lists/ct<shard>.txt (+ lists.json)."""
    m = json.load(open(AR / "hold/meta.json"))["chunks"]
    T = read_set()
    where = {}
    for c in m:
        for i in open(AR / f"hold/lists/c{c['chunk']}.txt").read().split():
            where[i] = (c["chunk"], c["shard"])
    miss = [i for i in T if i not in where]
    assert not miss, f"read-set windows not in the hold lists: {miss[:3]}"
    by = defaultdict(list)
    for i in T:
        c, k = where[i]
        if c not in SUBSET:
            by[k].append(i)
    (S_ / "lists").mkdir(parents=True, exist_ok=True)
    for k, ids in by.items():
        (S_ / f"lists/ct{k}.txt").write_text("\n".join(ids) + "\n")
    rec = dict(read_set=len(T), in_subset=sum(where[i][0] in SUBSET for i in T), outside=sum(map(len, by.values())), by_shard={k: len(v) for k, v in by.items()})
    (S_ / "lists.json").write_text(json.dumps(rec, indent=1))
    print(" ".join(str(k) for k in sorted(by)))


def cmd_dose(a):
    """the frozen 100-step run (FREE-M at B2's 0.001 and TIGHT-IN at W0 on): finite losses, the term fires at some step; the dose
    W1 = W0 x mean grad/lon_free / mean grad/lon_tightin (2 significant, clipped to W_RANGE), W3 = DOSE_B x W1."""
    import turn_cap as TC
    sc = TC._scalars(a.run)
    fin = all(np.isfinite(v) for t, x in sc.items() if t.startswith("loss/") for v in x.values())
    gf, gt = list(sc.get("loss/grad/lon_free", {}).values()), list(sc.get("loss/grad/lon_tightin", {}).values())
    pl, pa = sc.get("loss/on/lon_tightin_pos", {}), sc.get("loss/on/lon_tightina_pos", {})
    lab = [ln.strip() for ln in open(a.log) if "lon_tightin labels" in ln]
    mf, mt = (float(np.mean(gf)) if gf else np.nan), (float(np.mean(gt)) if gt else np.nan)
    raw = W0 * mf / mt if np.isfinite(mf) and np.isfinite(mt) and mt > 0 else np.nan
    w1 = float(np.clip(float(f"{raw:.2g}"), *W_RANGE)) if np.isfinite(raw) else np.nan
    pm = lambda x: (max(x.values()), float(np.mean(list(x.values())))) if x else (None, None)  # noqa: E731
    r = dict(run=a.run, finite=bool(fin), steps=len(pl), pos_lat_max_mean=pm(pl), pos_arc_max_mean=pm(pa),
             free_pos_mean=float(np.mean(list(sc.get("loss/on/lon_free_pos", {0: np.nan}).values()))),
             grad_free_w0001=mf, grad_tightin_w0004=mt, n_grad=[len(gf), len(gt)], grad_tightin_zero_steps=int(sum(g == 0 for g in gt)),
             W_raw=raw, clipped=bool(np.isfinite(raw) and w1 != float(f"{raw:.2g}")), W=dict(T1=w1, T3=float(f"{DOSE_B * w1:.2g}")), labels=lab[-1] if lab else "")
    r["pass_"] = bool(fin and pl and (max(pl.values()) > 0 or (pa and max(pa.values()) > 0)) and np.isfinite(w1))
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


# ---------------------------------------------------------------- Mac: the stage 1 rule
def exits(ext, tag, ids):
    """lateral-exit side per window (Mac): the ego at the first offroad / corridor exit (decision anchors interpolated to that time) against the
    human track; side = inside / outside where the human path turns >= 10 deg over [s - 10, s + 10] m around the exit station, else straight."""
    import pandas as pd
    import turn_sep as TS
    H = pickle.load(open(REPO / "tmp/lat_zero/hum.pkl", "rb"))
    d = pd.read_parquet(Path(ext) / "dec.parquet", columns=["tag", "id", "t0", "x", "y"])
    s = pd.read_parquet(Path(ext) / "steps.parquet", columns=["tag", "id", "t_us", "offroad", "left_corridor_laterally", "eval_relevant"])
    d, s = d[(d.tag == tag) & d.id.isin(ids)], s[(s.tag == tag) & s.id.isin(ids)]
    out = {}
    for i, st in s.groupby("id"):
        e = st[(st.eval_relevant > 0) & ((st.offroad > 0) | (st.left_corridor_laterally > 0))].sort_values("t_us")
        q = d[d.id == i].sort_values("t0")
        if not len(e) or len(q) < 2 or i not in H:
            continue
        t = float(e.t_us.iloc[0])
        xy = np.array([np.interp(t, q.t0, q.x), np.interp(t, q.t0, q.y)])
        tr = TS.Track(H[i])
        sx, off = tr.proj(xy[None])
        dh = float(tr.head(sx[0] + 10) - tr.head(sx[0] - 10))
        out[i] = "straight" if abs(dh) < np.radians(10) else ("inside" if np.sign(dh) * off[0] > 0 else "outside")
    return out


def cmd_stage1(a):
    import pandas as pd
    import stand_r as SR
    import turn_cap as TC
    cx = SR._ctx()
    lists = REPO / "tmp/tight_in/hold_lists"
    cx["chunk"] = {i: int(f.stem[1:]) for f in lists.glob("c*.txt") for i in f.read_text().split()}
    assert cx["chunk"], f"no hold lists in {lists} (scp them from the box)"
    TC.SUBSET = SUBSET
    TS_ = set(read_set())
    TI = set(pd.read_csv(RES / "tight_set.csv").id)
    R = pd.read_csv(a.rows)
    tags = [CTL] if a.anchors else [CTL, *ARMS.values()]
    keep = set.intersection(*(set(R[(R.tag == t) & R.score.notna()].id) for t in tags))
    E, M = {}, {}
    for t in tags:
        E[t], M[t] = TC.classify_read(cx, R, t, a.dec, keep)
    tgt = lambda e: e[e.cause.isin(LAT3) & e.id.isin(TI)]  # noqa: E731
    oth = lambda e: e[~(e.cause.isin(LAT3) & e.id.isin(TI))].cause.value_counts().to_dict()  # noqa: E731
    sc = {t: R[R.tag == t].drop_duplicates("id").set_index("id") for t in tags}
    lat = lambda t: [i for i in sorted(keep & TS_) if sc[t].score.get(i, 1) <= 1e-4 and str(sc[t].type.get(i, "")) in ("offroad", "corridor")]  # noqa: E731
    side = {t: exits(a.dec, t, lat(t)) for t in tags}
    ins = {t: sorted(i for i, v in side[t].items() if v == "inside") for t in tags}
    c = M[CTL]
    tpass = sorted(i for i in keep & TI if sc[CTL].score.get(i, 0) >= 1 - 1e-4)
    J = dict(read_set=len(keep & set(cx["items"])), turn_or_tight_read=len(keep & TS_), tight_read=len(keep & TI), control=dict(
        target=len(tgt(E[CTL])), target_ids=sorted(tgt(E[CTL]).id), other=oth(E[CTL]), tight_passed=len(tpass), inside_exits=len(ins[CTL]),
        exit_sides=dict(Counter(side[CTL].values())), **{k: v for k, v in c.items() if k != "scores"}))
    if a.anchors:
        print(json.dumps(J, indent=1, default=float))
        return
    need = max(2, int(np.ceil(0.25 * J["control"]["target"])))
    L = ["# TIGHT-IN pilot stage 1 (NR1 prereg supplement 22; native reads; control = the stored B2 pilot STANDFMXH1K-L18-s0)", "",
         f"Read set = arena subset (chunks 0 / 3 / 6 / 9 / 12 / 15) + the turn set (d395) + the tight set (road max 1 m curvature >= 0.108 /m): "
         f"{J['read_set']} frozen board-mix windows every read has ({J['turn_or_tight_read']} in the turn / tight sets, {J['tight_read']} tight).", ""]
    J["arms"] = {}
    for A, Tg in ARMS.items():
        t = json.load(open(RES / f"s17table_{A}.json"))
        H, Wt, Ar = t["held"], t["wait"], t["arena"]
        e, m = E[Tg], M[Tg]
        oc, oa = J["control"]["other"], oth(e)
        newz = sorted(i for i in tpass if sc[Tg].score.get(i, 1) <= 1e-4)
        rows = [("main 1: easy lateral zeros in the tight set", J["control"]["target"], len(tgt(e)), len(tgt(e)) <= J["control"]["target"] - need,
                 f"<= control - {need}"),
                ("main 2: read-set PCS (frozen board-mix)", round(c["pcs"]), round(m["pcs"]), m["pcs"] >= c["pcs"] + 10, ">= control + 10"),
                ("guard T1: zeros on tight windows the control passes", 0, len(newz), len(newz) <= 2, "<= control + 2"),
                ("guard T2: inside-of-turn lateral exits (turn / tight sets)", len(ins[CTL]), len(ins[Tg]), len(ins[Tg]) <= len(ins[CTL]) + 2, "<= control + 2"),
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
                            held_mean=[H[CTL]["mean"], H[Tg]["mean"]], targets=tw, target_ids=sorted(tgt(e).id), new_tight_zeros=newz,
                            exit_sides=dict(Counter(side[Tg].values())), inside_ids=ins[Tg],
                            fixed=sorted(set(J["control"]["target_ids"]) - set(tgt(e).id)), new=sorted(set(tgt(e).id) - set(J["control"]["target_ids"])))
        L += [f"## {A} `{Tg}`: {'PASS' if ok else 'FAIL'}", "", f"| read | {CTL} | {Tg} | line |", "|:--|--:|--:|:--|",
              *[f"| {r[0]} | {r[1]} | {r[2]} | {r[4]}: {'ok' if r[3] else 'FAIL'} |" for r in rows], "",
              f"- read-set mean {c['mean']:.4f} -> {m['mean']:.4f}, zeros {c['zeros']} -> {m['zeros']}; PCS on the 6 chunks only "
              f"{c['pcs_6chunks']:.0f} -> {m['pcs_6chunks']:.0f}; held 460 mean {H[CTL]['mean']:.4f} -> {H[Tg]['mean']:.4f}",
              f"- tight-set easy lateral zeros fixed {len(J['arms'][A]['fixed'])}, new {len(J['arms'][A]['new'])}; lateral exit sides (control / arm): "
              f"{J['control']['exit_sides']} / {J['arms'][A]['exit_sides']}",
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
    p.add_argument("--rows", default=str(RES / "arena_windows.csv.gz")), p.add_argument("--dec", default=str(REPO / "tmp/tight_in/ext"))
    p.add_argument("--anchors", action="store_true")
    a = ap.parse_args()
    dict(label=cmd_label, gate=cmd_gate, lists=cmd_lists, dose=cmd_dose, stage1=cmd_stage1)[a.cmd](a)
