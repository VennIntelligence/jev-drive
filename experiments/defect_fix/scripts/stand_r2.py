#!/usr/bin/env python3
"""STAND-R2 (NR1 prereg supplement 23, plans/2026-10-20-nr1-supp23-standr2.md; lib/nr1_free.py launch_label / launch_excl,
rm_train --on-stand-launch H): STAND-R (supplement 20) without the rows whose human launches within 4 + H s.

  $TR stand_r2.py launch --pool on1_JTX   (box, CPU) the human's launch after each S row's log time from the whole navsim log
                                          -> $CR/<pool>/stand_launch.npz (names, t_l, cover), results/stand_r2/launch.json
  $AP stand_r2.py wext --tag T --run DIR --out DIR   (box, CPU, AlpaSim venv) the wait-set read's decisions joined with the window's human track
                                          (native_gap's extraction) -> DIR/{dec,win,steps,set_census}.parquet, hum.pkl
  $TR stand_r2.py gate --out DIR          (box, CPU) offline gate G1, G2, G3, G3w -> DIR/gate.json, results/stand_r2/gate.json; exit 0 pass, 2 fail
  $TR stand_r2.py smoke --run T --log L --out JSON   60-step smoke: finite losses, lon_standr_pos > 0 at some step, launch rows dropped
  tmp/pcs_local/.venv/bin/python stand_r2.py stage1   (Mac) supplement 20's stage 1 rule for L1 / L2 -> results/stand_r2/stage1.{json,md}

Decision-level launch (stored reads): the windows come from the raw nuPlan test logs (wait_set.Log, 20 Hz); decision t0 (us from the window
start) -> frame f0 = start_idx + round(20 t0 / 1e6); t_L = the first later frame whose rear-axle position is > L_MOVE from f0's or whose
ego_pose speed is >= L_VGO; cover = the log after f0. The rest (fut / fut_reg arcs, g, caps, fires) is stand_r.py's gate code unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/onpol_nuplan/scripts"),
                str(REPO / "experiments/alpasim_proxy/scripts")]
import nr1_free as NF  # noqa: E402
import stand_r as SR  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix"
CR = DATA / "runs/op_parity/cache"
RES = REPO / "experiments/defect_fix/results/stand_r2"
H_L = 2.0
CTL = SR.CTL
ARMS = {"L1": "STANDRLH1K-L18-s0", "L2": "STANDRL2H1K-L18-s0"}
OLD = dict(SR.ARMS)                                       # STAND-R R1 / R2 (decision 399), compared on the same subset
WS_SET = REPO / "experiments/alpasim_proxy/results/wait_set/set.json"


# ---------------------------------------------------------------- pool labels
def _launch_job(job):
    import on_rows as OR
    log, R = job
    L = OR._log(log)
    G = OR.Path(L["P"], L["T"])
    T, S = G.TF * 1e-6, G.SF
    return [(i, *NF.launch_label(t_us * 1e-6, T, S)) for i, t_us in R]


def cmd_launch(a):
    tab = np.load(CR / a.pool / "tab.npz")
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    part = np.array([n.split(":")[1] for n in names])
    jobs = defaultdict(list)
    for p, d in SR.ROWS.items():
        rows = np.flatnonzero(S & (part == p))
        if not len(rows):
            continue
        zz = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        z = {k: zz[k] for k in ("names", "speed", "log", "t_log_us")}
        pos = {n: i for i, n in enumerate(z["names"].astype(str).tolist())}
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in rows])
        assert np.allclose(z["speed"][mi], tab["speed"][rows], atol=1e-3), f"{d}: meta speed != the pool's"
        for i, m in zip(rows, mi):
            jobs[str(z["log"][m])].append((int(i), int(z["t_log_us"][m])))
    assert sum(map(len, jobs.values())) == int(S.sum()), "S rows without a rows dir"
    t_l, cover = np.full(len(names), np.nan, np.float32), np.full(len(names), np.nan, np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(_launch_job, list(jobs.items()), chunksize=1):
            for i, x, c in out:
                t_l[i], cover[i] = x, c
    f = CR / a.pool / "stand_launch.npz"
    assert not f.exists(), f"{f} exists: never overwritten"
    np.savez(f, names=names, t_l=t_l, cover=cover)
    # description: supplement 20's rows and what supplement 23 drops
    gap = np.load(CR / a.pool / "stand_gap.npz")["gap"]
    hum4 = NF.cum_arc(tab["fut_reg"])[:, -1]
    _, fin = NF.stand_cap(tab["fut"], tab["fut_reg"])
    NF.configure_stand(1.0, SR.V_ROLL, H_L)
    st = S & fin & NF.standing(tab["speed"], hum4)
    roll = S & fin & NF.rolling(tab["speed"], hum4, gap) & ~st
    ex = NF.launch_excl(t_l, cover)
    unk = ex & ~np.isfinite(t_l)
    rq = lambda x: np.nanquantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist() if len(x) else []  # noqa: E731
    rec = dict(pool=a.pool, s_rows=int(S.sum()), s_label_finite_or_inf=int((~np.isnan(t_l[S])).sum()), launch_h=H_L,
               standing=int(st.sum()), standing_dropped=int((st & ex).sum()), standing_unknown=int((st & unk).sum()),
               rolling=int(roll.sum()), rolling_dropped=int((roll & ex).sum()), rolling_unknown=int((roll & unk).sum()),
               standing_kept=int((st & ~ex).sum()), rolling_kept=int((roll & ~ex).sum()),
               t_l_q_standing=rq(t_l[st & np.isfinite(t_l)]), t_l_q_rolling=rq(t_l[roll & np.isfinite(t_l)]),
               rolling_gap_q_kept=rq(gap[roll & ~ex]), rolling_gap_q_dropped=rq(gap[roll & ex]),
               rolling_kept_past_him=int((roll & ~ex & (gap < 0)).sum()), rolling_kept_by_part=dict(Counter(part[roll & ~ex].tolist())))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "launch.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    assert np.isnan(t_l[S]).sum() == 0, "S rows without a launch label"


# ---------------------------------------------------------------- stored reads: decision launch from the raw logs
def _raw_job(job):
    import wait_set as WS
    log, items = job                       # items: [(window id, start_idx, [t0 us ...], hum T, hum P)]
    L = WS.Log(log)
    n, out, chk = len(L.ts), {}, []
    for wid, w0, t0s, hT, hP in items:
        r = []
        for t0 in t0s:
            f0 = min(int(w0 + round(t0 * 20e-6)), n - 1)
            d = np.hypot(*(L.xy[f0 + 1:] - L.xy[f0]).T)
            hit = np.flatnonzero((d > NF.L_MOVE) | (L.v[f0 + 1:] >= NF.L_VGO))
            r.append(((L.ts[f0 + 1 + hit[0]] - L.ts[f0]) / 1e6 if len(hit) else np.inf, (L.ts[-1] - L.ts[f0]) / 1e6))
        out[wid] = r
        if hT is not None:                 # clock check: the window track's displacement from its start against the raw log's
            tt = np.arange(0.5, 5.01, 0.5)
            hx = np.c_[np.interp(tt * 1e6, hT, hP[:, 0]), np.interp(tt * 1e6, hT, hP[:, 1])]
            fr = np.minimum(w0 + np.round(tt * 20).astype(int), n - 1)
            chk.append(float(np.abs(np.hypot(*(hx - hP[0]).T) - np.hypot(*(L.xy[fr] - L.xy[w0]).T)).max()))
    return out, chk


def decision_launch(D, hum, where, workers):
    """D (decisions with id, t0), hum {id: track}, where {id: (log, start_idx)} -> t_l (n,), cover (n,), clock check max (m)."""
    by = defaultdict(list)
    for wid, g in D.groupby("id"):
        h = hum.get(wid)
        by[where[wid][0]].append((wid, int(where[wid][1]), g.t0.astype(float).tolist(),
                                  None if h is None else np.asarray(h["T"], float), None if h is None else np.asarray(h["P"], float)))
    res, chk = {}, []
    with ProcessPoolExecutor(workers) as ex:
        for o, c in ex.map(_raw_job, list(by.items()), chunksize=1):
            res |= o
            chk += c
    t_l, cov = np.zeros(len(D)), np.zeros(len(D))
    pos = Counter()
    for r, wid in enumerate(D.id.values):
        t_l[r], cov[r] = res[wid][pos[wid]]
        pos[wid] += 1
    return t_l, cov, (max(chk) if chk else np.nan)


def caps_new(D, hum, where, workers):
    """supplement 20's kind / fire on the stored plans and supplement 23's: rows with a launch within 4 + H_L s (or unknown) dropped."""
    for c in ("id", "t0"):
        assert c in D
    D = D.sort_values(["id", "t0"], kind="stable").reset_index(drop=True)
    kind, cap, fire = SR.decision_caps(D, hum)
    t_l, cov, chk = decision_launch(D, hum, where, workers)
    ex = NF.launch_excl(t_l, cov)
    k2 = np.where(ex, 0, kind)
    return D, dict(kind=kind, fire=fire, kind2=k2, fire2=fire & (k2 > 0), t_l=t_l, ex=ex, clock=chk)


def cmd_wext(a):
    import pandas as pd
    import native_gap as NG
    runs = sorted(glob.glob(f"{a.run}/*_c*")) or [a.run]
    from multiprocessing import Pool
    with Pool(min(a.workers, len(runs))) as p:
        res = p.map(NG._read, [(a.tag, r) for r in runs])
    D, cfg, W, ST = {}, {}, [], []
    for tag, run, dr, rs, st in res:
        for r in dr:
            D.setdefault(r["id"], {}).setdefault(tag, []).append(r)
            cfg.setdefault(r["id"], f"{run}/generated-user-config-0.yaml")
        W += [dict(tag=tag, id=i, chunk=Path(run).name, **x) for i, x in rs.items()]
        st = st.copy()
        st.insert(0, "tag", tag)
        ST.append(st)
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(NG._window, [(i, cfg[i], D[i]) for i in sorted(D)], chunksize=2))
    errs = [e for _, _, e in out if e]
    dec = pd.DataFrame([r for o, _, _ in out for r in o])
    dec["poses"] = dec.poses.map(lambda x: np.asarray(x, np.float32).tolist())
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    dec.to_parquet(o / "dec.parquet")
    pd.DataFrame(W).to_parquet(o / "win.parquet")
    pd.concat(ST, ignore_index=True).to_parquet(o / "steps.parquet")
    pd.DataFrame(dict(id=sorted(D))).to_parquet(o / "set_census.parquet")
    pickle.dump({i: h for _, (i, h), _ in out if h is not None}, open(o / "hum.pkl", "wb"), protocol=4)
    print(json.dumps(dict(runs=len(runs), windows=len(D), decisions=len(dec), errors=len(errs), first_errors=errs[:3])), flush=True)
    assert not errs and len(D) > 250, "wait-set extraction incomplete"


def cmd_gate(a):
    import pandas as pd
    import torch
    import native_gap_report as NR
    NF.configure_stand(1.0, SR.V_ROLL, H_L)
    # G1: the human's own paths on the pool's kept rows
    tab = np.load(CR / "on1_JTX/tab.npz")
    pk = NF.stand_load(CR / "on1_JTX", tab["names"], "cpu")
    ok = pk["ok"].numpy()
    g1 = {}
    for k in ("fut", "fut_reg"):
        P = torch.as_tensor(np.nan_to_num(tab[k][ok]).astype(np.float32))
        _, pr = NF.stand_terms(P, pk["cap"][torch.as_tensor(ok)], torch.ones(int(ok.sum()), dtype=torch.bool), parts=True)
        g1[k] = int(pr["fire"].sum())
    G = dict(pool=pk["rec"], G1=dict(fires=g1, rows=int(ok.sum()), rolling=int(pk["roll"].sum()), pass_=sum(g1.values()) == 0))
    # G2 / G3 on B2's stored full-length reads
    A = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")
    where = {i: (r.log, r.start_idx) for i, r in A.iterrows()}
    mis = {"STANDFMXH-L18-s0": "misses_STANDFMXH-s0", "STANDFMXH-L18-s1": "misses_STANDFMXH-s1"}
    src = {"STANDFMXH-L18-s0": O / "stand_b/ext_full_H", "STANDFMXH-L18-s1": O / "stand_b/ext_full_H_s1"}
    G["per_read"], tot, clock = {}, Counter(), []
    for tag, d in src.items():
        D, Wn, _, C = NR.load(d)
        D = D[(D.tag == tag) & D.valid & (D.k >= 2)]
        hum = pickle.load(open(d / "hum.pkl", "rb"))
        D, x = caps_new(D, hum, where, a.workers)
        clock.append(x["clock"])
        M = pd.read_csv(REPO / f"experiments/alpasim_proxy/results/easy_items/{mis[tag]}.csv.gz")
        red = set(M.id[(M.cause == SR.RED) & (M.strat == SR.EASY)])
        y = Wn[Wn.tag == tag].drop_duplicates("id").set_index("id").score
        one = set(y.index[y >= 1 - 1e-4])
        mr, m1 = D.id.isin(red).values, D.id.isin(one).values
        k, f, k2, f2 = x["kind"], x["fire"], x["kind2"], x["fire2"]
        r = dict(red_windows=len(red), red_rolling_old=int((mr & (k == 2)).sum()), red_rolling=int((mr & (k2 == 2)).sum()),
                 red_rolling_fire=int((mr & (k2 == 2) & f2).sum()), red_windows_with_rolling=int(D.id[mr & (k2 == 2)].nunique()),
                 red_standing_old=int((mr & (k == 1)).sum()), red_standing=int((mr & (k2 == 1)).sum()), red_standing_fire=int((mr & (k2 == 1) & f2).sum()),
                 red_dropped_t_l=np.round(x["t_l"][mr & (k > 0) & x["ex"]], 1).tolist(),
                 one_windows=len(one), one_decisions=int(m1.sum()),
                 one_rolling_old=int((m1 & (k == 2)).sum()), one_rolling_fire_old=int((m1 & (k == 2) & f).sum()),
                 one_rolling=int((m1 & (k2 == 2)).sum()), one_rolling_fire=int((m1 & (k2 == 2) & f2).sum()),
                 one_rolling_fire_windows=int(D.id[m1 & (k2 == 2) & f2].nunique()),
                 one_standing_fire_old=int((m1 & (k == 1) & f).sum()), one_standing_fire=int((m1 & (k2 == 1) & f2).sum()),
                 one_standing_fire_windows=int(D.id[m1 & (k2 == 1) & f2].nunique()),
                 one_rolling_fire_by_stratum=C.reindex(D.id[m1 & (k2 == 2) & f2]).get("stratum", pd.Series(dtype=str)).value_counts().to_dict())
        G["per_read"][tag] = r
        tot.update({q: v for q, v in r.items() if isinstance(v, int)})
    G["clock_check_max_m"] = float(np.nanmax(clock))
    G["G2"] = dict(decisions=tot["red_rolling"], fired=tot["red_rolling_fire"], share=tot["red_rolling_fire"] / max(tot["red_rolling"], 1), line=0.80,
                   windows_with_rows=tot["red_windows_with_rolling"], windows=tot["red_windows"], windows_line=10)
    G["G2"]["pass_"] = bool(tot["red_rolling"] > 0 and G["G2"]["share"] >= 0.80 and tot["red_windows_with_rolling"] >= 10)
    G["G3"] = dict(decisions=tot["one_decisions"], windows=tot["one_windows"], rolling_fire=tot["one_rolling_fire"], rolling_fire_windows=tot["one_rolling_fire_windows"],
                   share=tot["one_rolling_fire"] / max(tot["one_decisions"], 1), windows_share=tot["one_rolling_fire_windows"] / max(tot["one_windows"], 1),
                   line=0.01, windows_line=0.05, old_rule_rolling_fire=tot["one_rolling_fire_old"],
                   standing_fire=tot["one_standing_fire"], standing_fire_old_rule=tot["one_standing_fire_old"], standing_fire_windows=tot["one_standing_fire_windows"])
    G["G3"]["pass_"] = bool(G["G3"]["share"] <= 0.01 and G["G3"]["windows_share"] <= 0.05)
    # G3w: the control's wait-set read, b variant other_lead / red_queue, decisions within 3 s before the human's launch
    ws = {w["id"]: w for w in json.load(open(WS_SET))["windows"]}
    D, _, _, _ = NR.load(Path(a.wext))
    D = D[D.valid & (D.k >= 2) & D.id.isin(ws)]
    hum = pickle.load(open(Path(a.wext) / "hum.pkl", "rb"))
    D, x = caps_new(D, hum, {i: (w["log"], w["start_idx"]) for i, w in ws.items()}, a.workers)
    clock.append(x["clock"])
    G["clock_check_max_m"] = float(np.nanmax(clock))
    var = D.id.map(lambda i: ws[i]["variant"]).values
    cls = D.id.map(lambda i: ws[i]["cls"]).values
    hl = D.id.map(lambda i: ws[i]["h_launch_s"] if ws[i]["h_launch_s"] is not None else np.nan).values.astype(float)
    t = D.t0.values / 1e6
    b = (var == "b") & np.isin(cls, ["other_lead", "red_queue"]) & np.isfinite(hl)
    pre3, pre = b & (t < hl) & (t >= hl - 3.0), b & (t < hl)
    av = (var == "a") & np.isin(cls, ["other_lead", "red_queue"])
    k, f, k2, f2 = x["kind"], x["fire"], x["kind2"], x["fire2"]
    G["G3w"] = dict(decisions=int(pre3.sum()), windows=int(D.id[pre3].nunique()), fires=int((pre3 & f2).sum()), share=float((pre3 & f2).sum() / max(pre3.sum(), 1)),
                    line=0.01, old_rule_fires=int((pre3 & f).sum()), old_rule_rows=int((pre3 & (k > 0)).sum()),
                    b_pre_all=dict(decisions=int(pre.sum()), fires=int((pre & f2).sum()), old_rule_fires=int((pre & f).sum())),
                    a_all=dict(decisions=int(av.sum()), rows=int((av & (k2 > 0)).sum()), fires=int((av & f2).sum()),
                               old_rule_rows=int((av & (k > 0)).sum()), old_rule_fires=int((av & f).sum())))
    G["G3w"]["pass_"] = bool(pre3.sum() > 0 and G["G3w"]["share"] <= 0.01)
    G["pass_"] = bool(G["G1"]["pass_"] and G["G2"]["pass_"] and G["G3"]["pass_"] and G["G3w"]["pass_"] and G["clock_check_max_m"] < 0.3)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    for fo in (out / "gate.json", RES / "gate.json"):
        fo.write_text(json.dumps(G, indent=1, default=float))
    print(json.dumps(G, indent=1, default=float))
    print("GATE", "PASS" if G["pass_"] else "FAIL")
    raise SystemExit(0 if G["pass_"] else 2)


def cmd_smoke(a):
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{a.run}/*/"))[-1]
    sc = {}
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc.setdefault(e["tag"], {})[e["step"]] = e["value"]
    fin = all(np.isfinite(v) for t, x in sc.items() if t.startswith("loss/") and "grad/" not in t for v in x.values())
    rp = sc.get("loss/on/lon_standr_pos", {})
    lab = [ln for ln in open(a.log) if "lon_stand labels" in ln]
    r = dict(run=a.run, finite=bool(fin), steps=len(rp), roll_pos_max=max(rp.values()) if rp else None, labels=lab[-1].strip() if lab else "")
    r["launch_dropped"] = "launch_excl_rolling" in r["labels"] and "launch_excl_standing" in r["labels"]
    r["pass_"] = bool(fin and rp and r["roll_pos_max"] > 0 and r["launch_dropped"])
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


def cmd_stage1(a):
    import pandas as pd
    cx = SR._ctx()
    R = pd.concat([pd.read_csv(a.rows), pd.read_csv(a.old_rows)], ignore_index=True).drop_duplicates(["tag", "id"])
    tags = [CTL, *ARMS.values()]
    keep = set.intersection(*(set(R[(R.tag == t) & R.score.notna()].id) for t in tags))
    E = {t: SR.classes(cx, R, t, a.dec, keep) for t in tags}
    for t in OLD.values():                                  # STAND-R arms on the same subset (their windows table / decisions)
        k_old = keep & set(R[(R.tag == t) & R.score.notna()].id)
        E[t] = SR.classes(cx, R, t, a.old_dec, k_old) | dict(subset_windows=len(k_old & set(cx["items"])))
    J = dict(subset_windows=len(keep & set(cx["items"])), easy=E, arms={})
    c = E[CTL]
    L = ["# STAND-R2 pilot stage 1 (NR1 prereg supplement 23; native reads; control = the stored B2 pilot STANDFMXH1K-L18-s0)", "",
         f"Arena subset (chunks 0 / 3 / 6 / 9 / 12 / 15) on the frozen board-mix items every read has: {J['subset_windows']} windows. "
         "STAND-R R1 / R2 (decision 399) are shown on the same subset for comparison only.", ""]
    old = {A: json.load(open(SR.RES / f"s17table_{A}.json")) for A in OLD}
    for A, T in ARMS.items():
        t = json.load(open(RES / f"s17table_{A}.json"))
        H, Wt, Ar, Dr = t["held"], t["wait"], t["arena"], next(iter(t["drift"].values()))
        e = E[T]
        cls = sorted(set(c["by_class"]) | set(e["by_class"]) - {SR.RED})
        oo = {OLD[q]: old[q] for q in OLD}

        def ov(fn):
            return " / ".join(str(fn(q, oo[q])) for q in OLD.values())
        rows = [("main 1: easy red-line zeros", c["red"], e["red"], e["red"] <= c["red"] - 6, "<= control - 6", ov(lambda q, _: E[q]["red"])),
                ("main 2: subset PCS (frozen board-mix)", round(c["pcs"]), round(e["pcs"]), e["pcs"] >= c["pcs"] + 15, ">= control + 15", ov(lambda q, _: round(E[q]["pcs"]))),
                ("guard 1: held 460 zeros", H[CTL]["zeros"], H[T]["zeros"], H[T]["zeros"] <= H[CTL]["zeros"] + 2, "<= control + 2", ov(lambda q, o: o["held"][q]["zeros"])),
                ("guard 2: held 460 front collisions", H[CTL]["coll"]["front"], H[T]["coll"]["front"], H[T]["coll"]["front"] <= H[CTL]["coll"]["front"] + 2, "<= control + 2",
                 ov(lambda q, o: o["held"][q]["coll"]["front"])),
                ("guard 2: subset AF-front windows", Ar["af_front_b"], Ar["af_front_a"], Ar["af_front_a"] <= Ar["af_front_b"] + 2, "<= control + 2", ov(lambda q, o: o["arena"]["af_front_a"])),
                ("guard 3: easy misses outside the red line", c["other"], e["other"], e["other"] <= c["other"] + 2, "<= control + 2", ov(lambda q, _: E[q]["other"])),
                *[(f"guard 4: easy {k}", c["by_class"].get(k, 0), e["by_class"].get(k, 0), e["by_class"].get(k, 0) <= c["by_class"].get(k, 0) + 2, "<= control + 2",
                   ov(lambda q, _, k=k: E[q]["by_class"].get(k, 0))) for k in cls if k != SR.RED],
                ("guard 5: subset slow partial windows", t["partials"][CTL]["slow"], t["partials"][T]["slow"], t["partials"][T]["slow"] <= t["partials"][CTL]["slow"] + 3,
                 "<= control + 3", ov(lambda q, o: o["partials"][q]["slow"])),
                ("guard 5: subset k = 9 speed ratio", round(Ar["k9"]["b"], 4), round(Ar["k9"]["a"], 4), Ar["k9"]["a"] >= Ar["k9"]["b"] - 0.03, ">= control - 0.03",
                 ov(lambda q, o: round(o["arena"]["k9"]["a"], 4))),
                *[(f"guard 6: wait a {k}", round(Wt[CTL][k]["wait"], 3), round(Wt[T][k]["wait"], 3), Wt[T][k]["wait"] >= Wt[CTL][k]["wait"] - 0.03, ">= control - 0.03",
                   ov(lambda q, o, k=k: round(o["wait"][q][k]["wait"], 3))) for k in ("other_lead", "red_queue")],
                *[(f"guard 6: wait b median launch delay {k} (s)", round(Wt[CTL][k]["delay_med"], 2), round(Wt[T][k]["delay_med"], 2),
                   Wt[T][k]["delay_med"] <= Wt[CTL][k]["delay_med"] + 0.15, "<= control + 0.15", ov(lambda q, o, k=k: round(o["wait"][q][k]["delay_med"], 2)))
                  for k in ("other_lead", "red_queue")],
                *[(f"guard 6: wait b launched {k}", round(Wt[CTL][k]["launched"], 3), round(Wt[T][k]["launched"], 3),
                   Wt[T][k]["launched"] >= 0.95 * Wt[CTL][k]["launched"], ">= 0.95 x control", ov(lambda q, o, k=k: round(o["wait"][q][k]["launched"], 3)))
                  for k in ("other_lead", "red_queue")],
                ("guard 7: held 460 right drift, non-turn k 11-17 (arm - control, m)", "", round(Dr["k11_17"]["mean"], 3), Dr["k11_17"]["mean"] > -0.05, "> -0.05",
                 ov(lambda q, o: round(next(iter(o["drift"].values()))["k11_17"]["mean"], 3)))]
        ok = all(r[3] for r in rows)
        J["arms"][A] = dict(tag=T, rows=[dict(read=r[0], control=r[1], arm=r[2], ok=bool(r[3]), line=r[4], stand_r=r[5]) for r in rows], pass_=ok,
                            subset_mean=[c["mean"], e["mean"]], held_mean=[H[CTL]["mean"], H[T]["mean"]],
                            red_fixed=sorted(set(c["red_ids"]) - set(e["red_ids"])), red_new=sorted(set(e["red_ids"]) - set(c["red_ids"])))
        L += [f"## {A} `{T}`: {'PASS' if ok else 'FAIL'}", "", f"| read | {CTL} | {T} | line | STAND-R R1 / R2 |", "|:--|--:|--:|:--|--:|",
              *[f"| {r[0]} | {r[1]} | {r[2]} | {r[4]}: {'ok' if r[3] else 'FAIL'} | {r[5]} |" for r in rows], "",
              f"- subset mean {c['mean']:.4f} -> {e['mean']:.4f}, zeros {c['zeros']} -> {e['zeros']}; easy misses {c['easy_miss']} -> {e['easy_miss']} "
              f"(of {e['easy_windows']} easy windows); held 460 mean {H[CTL]['mean']:.4f} -> {H[T]['mean']:.4f}",
              f"- red-line easy zeros fixed {len(J['arms'][A]['red_fixed'])}, new {len(J['arms'][A]['red_new'])}", ""]
    (RES / "stage1.json").write_text(json.dumps(J, indent=1, default=float))
    (RES / "stage1.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("launch")
    p.add_argument("--pool", required=True), p.add_argument("--workers", type=int, default=32)
    p = sp.add_parser("wext")
    p.add_argument("--tag", required=True), p.add_argument("--run", required=True), p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True), p.add_argument("--wext", required=True), p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("smoke")
    p.add_argument("--run", required=True), p.add_argument("--log", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("stage1")
    p.add_argument("--rows", default=str(RES / "arena_windows.csv.gz")), p.add_argument("--dec", default=str(REPO / "tmp/stand_r2/ext"))
    p.add_argument("--old-rows", default=str(SR.RES / "arena_windows.csv.gz")), p.add_argument("--old-dec", default=str(REPO / "tmp/stand_r/ext"))
    a = ap.parse_args()
    dict(launch=cmd_launch, wext=cmd_wext, gate=cmd_gate, smoke=cmd_smoke, stage1=cmd_stage1)[a.cmd](a)
