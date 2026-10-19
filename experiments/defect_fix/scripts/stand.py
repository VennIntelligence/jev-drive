#!/usr/bin/env python3
"""NR1STAND (NR1 prereg supplement 17; lib/nr1_free.py stand_*): the one-sided "do not go further than the standing human" term. Box, repo root.

  $AP stand.py gate --out DIR        offline gate (CPU) -> DIR/gate.json, DIR/decisions.parquet; exit 0 pass, 2 fail
  $TR stand.py dose --run TAG --out JSON     W_S from the frozen-init diag run (grad/lon_free, grad/lon_stand) -> JSON; exit 2 out of range
  $TR stand.py pilot --tag P --ctl C --arena JSON --drift JSON --ext DIR --out MD   the supplement 17 pilot rule for one arm -> MD (+ .json); VERDICT
  $TR stand.py drift --follow DIR --a LABEL --b LABEL --out JSON       lat_diag's closed-loop drift (pnx.drift) A - B on held 460 @18
  (arena extract / speed ratio / strata: free_m.py ext / arena, unchanged)

Every decision is scored with nr1_free.stand_terms(parts=True) as the trainer scores a pool row. Closed-loop states as free_gate.py: scene time t
-> log frame start_idx + round(20 t); ego speed = the controller speed at t; the human's 4 s logged arc (raw) and its reachable version (the log's
speed blended from the controller speed over 2 s); cap = max(raw, reach) + MARGIN per 0.5 s step; standing = ego < 0.5 m/s and raw_8 < 0.5 m.
Human launch within 4 s: the log speed at the frame < 0.5 m/s and the first frame in (f, f + 80] with log speed > 1.0 m/s.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/alpasim_proxy/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix"
AR = DATA / "runs/alpasim_proxy/arena"
CREEP = "JTN-L18-s1@off"
LAUNCH_DUMPS = ("JTN-L18-s0", "RTJA0-L18-s0", "RTB1ARO-L18-s0")
HZ = 20
W_FREE, W_RANGE = 0.004, (0.0004, 0.04)


def hum(L, f, ve):
    """-> raw (8,), reach (8,) cumulative human arcs from frame f; the human's speed at f; his launch delay (s) within 4 s or nan."""
    seg = np.hypot(*np.diff(L.xy[f:f + 4 * HZ + 1], axis=0).T)
    raw = np.cumsum(seg)[HZ // 2 - 1::HZ // 2]
    tau = np.arange(4 * HZ) / HZ
    vb = ve + (L.v[f:f + 4 * HZ] - ve) * np.minimum(tau / 2, 1)
    reach = np.cumsum(vb / HZ)[HZ // 2 - 1::HZ // 2]
    go = np.flatnonzero(L.v[f + 1:f + 4 * HZ + 1] > 1.0)
    return raw, reach, float(L.v[f]), (go[0] + 1) / HZ if (L.v[f] < 0.5 and len(go)) else np.nan


def _run(job):
    """one read / dump run -> decision dicts. job = (set, read label, run dir, {scene: (log, start_idx, token, extra)}, {(scene, k): valid})."""
    import torch
    import free_gate as FG
    import nr1_free as NF
    torch.set_num_threads(1)
    name, label, run, want, valid = job
    R = FG.run_rollouts(run)
    logs, info, P, C_ = {}, [], [], []
    for sc, (C, dec, r) in R.items():
        if sc not in want:
            continue
        log, s0, tok, extra = want[sc]
        L = logs.get(log) or logs.setdefault(log, FG.Log(log))
        assert L.tok[s0 + 30].hex() == tok, sc
        for k, t, p in dec:
            f = s0 + int(round(HZ * t))
            if f + 4 * HZ >= len(L.xy):
                continue
            ve = float(np.interp(t, C["t"], C["v"]))
            raw, reach, vh, tl = hum(L, f, ve)
            P.append(np.asarray(p, np.float32))
            C_.append((np.maximum(raw, reach) + NF.MARGIN).astype(np.float32))
            info.append(dict(set=name, read=label, scene=sc, k=k, t=t, v_ego=ve, raw4=float(raw[-1]), reach4=float(reach[-1]), v_hum=vh, h_launch=tl,
                             valid=bool(valid.get((sc, k), True)) if valid is not None else True, **extra))
    if info:
        ok = torch.as_tensor(NF.standing(np.array([d["v_ego"] for d in info]), np.array([d["raw4"] for d in info])))
        cap = torch.as_tensor(np.stack(C_))
        _, pr = NF.stand_terms(torch.as_tensor(np.stack(P)), cap, torch.ones(len(P), dtype=torch.bool), parts=True)   # mask-free: creep
        for j, d in enumerate(info):
            d.update(plan4=float(pr["s_plan"][j]), cap4=float(cap[j, -1]), standing=bool(ok[j]), creep=bool(pr["fire"][j]),
                     fire=bool(pr["fire"][j]) and bool(ok[j]))
    return info


def cmd_gate(a):
    import pandas as pd
    import torch
    from multiprocessing import Pool
    import native_gap_report as NR
    import nr1_free as NF
    torch.set_num_threads(16)
    res = dict(margin=NF.MARGIN, v_stand=NF.V_STAND, h_stand=NF.H_STAND)
    # 1. the human's own paths on the pool's standing S rows
    pool = DATA / "runs/op_parity/cache/on1_JT"
    z = dict(np.load(pool / "tab.npz"))
    cap, fin = NF.stand_cap(z["fut"], z["fut_reg"])
    S = (z["mode"].astype(str) == "self") & fin
    st = S & NF.standing(z["speed"], NF.cum_arc(z["fut_reg"])[:, -1])
    m = np.flatnonzero(st)
    w = np.load(O / "w/JT00.npz")
    assert np.array_equal(w["names"].astype(str), z["names"].astype(str)), "JT00 rows != on1_JT"
    ww = w["w"] / w["w"].sum()
    hm = dict(s_rows=int(S.sum()), standing_rows=int(len(m)), standing_share_of_s=float(len(m) / max(S.sum(), 1)),
              draw_weight_standing=float(ww[st].sum()), draw_weight_s=float(ww[S].sum()),
              ego_slow_rows=int((S & (z["speed"] < NF.V_STAND)).sum()))
    for plan in ("fut", "fut_reg"):
        _, pr = NF.stand_terms(torch.as_tensor(z[plan][m]), torch.as_tensor(cap[m]), torch.ones(len(m), dtype=torch.bool), parts=True)
        hm[plan] = dict(fires=int(pr["fire"].sum()), rate=float(pr["fire"].float().mean()) if len(m) else 0.0)
        print(f"human plan {plan}: {hm[plan]['fires']} / {len(m)}", flush=True)
    res["human"] = hm
    del z
    # closed-loop sets
    Sx = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    held = Sx[Sx.split == "held"]
    ng = O / "native_gap"
    D, Wn, _, Cn = NR.load(ng)
    D1 = D[D.tag == "JTN1"]
    valid = {(i, int(k)): bool(v) for i, k, v in zip(D1.id, D1.k, D1.valid)}
    W1 = Wn[Wn.tag == "JTN1"].set_index("id")
    cue = NR.cue(held[held.stratum == "static"], Cn)
    jobs = []
    for run in sorted(glob.glob(str(O / "jt/arena" / CREEP / "c*/run"))):
        if (Path(run) / "aggregate/results-summary.json").exists():
            want = {i: (w_.log, int(w_.start_idx), w_.token, dict(stratum=w_.stratum, score=float(W1.score.get(i, np.nan)), type=str(W1.type.get(i, "")),
                                                                    cue=str(cue.get(i, ""))))
                    for i, w_ in held.iterrows()}
            jobs.append(("arena", CREEP, run, want, valid))
    WS = {x["id"]: x for x in json.load(open(REPO / "experiments/alpasim_proxy/results/wait_set/set.json"))["windows"]
          if x["variant"] == "b" and x["cls"] in ("other_lead", "red_queue")}
    for tag in LAUNCH_DUMPS:
        want = {i: (x["log"], int(x["start_idx"]), x["token"], dict(cls=x["cls"])) for i, x in WS.items()}
        jobs.append(("dump", tag, str(O / "launch_diag/dump" / tag / "all"), want, None))
    print(f"{len(jobs)} runs", flush=True)
    with Pool(a.workers) as p:
        out = p.map(_run, jobs, chunksize=1)
    X = pd.DataFrame([d for o in out for d in o])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    X.to_parquet(Path(a.out) / "decisions.parquet")
    Ar = X[X.set == "arena"]
    res["valid_matched"] = dict(decisions=len(Ar), with_native_gap_key=int(sum((s, int(k)) in valid for s, k in zip(Ar.scene, Ar.k))))
    # 2. creep decisions: JTN1 native static zero windows, valid, plan 4 s arc beyond the cap
    st_ = Ar[(Ar.stratum == "static") & (Ar.type != "route-fail")]
    z0 = st_[(st_.score == 0) & st_.valid]
    cr = z0[z0.creep]

    def rate(q):
        return dict(n=len(q), fires=int(q.fire.sum()), rate=float(q.fire.mean()) if len(q) else None)
    res["creep"] = dict(windows=int(z0.scene.nunique()), valid_decisions=len(z0), **rate(cr),
                        blocked_ego_moving=int((cr.v_ego >= NF.V_STAND).sum()), blocked_human_moves=int(((cr.v_ego < NF.V_STAND) & (cr.raw4 >= NF.H_STAND)).sum()),
                        k_le_3=rate(cr[cr.k <= 3]), red_line=rate(cr[cr.cue == "red light at the line, no lead"]),
                        by_cue={c: rate(g) for c, g in cr.groupby("cue")}, windows_fired=int(cr[cr.fire].scene.nunique()),
                        plan4_median=float(cr.plan4.median()), cap4_median=float(cr.cap4.median()))
    ok0 = st_[(st_.score > 0) & st_.valid]
    res["static_success"] = dict(windows=int(ok0.scene.nunique()), decisions=len(ok0), fires=int(ok0.fire.sum()),
                                 rate=float(ok0.fire.mean()) if len(ok0) else None, windows_fired=int(ok0[ok0.fire].scene.nunique()))
    # 3. the human's launch decisions: ego standing, the human stands at the frame and starts within 4 s
    La = X[X.h_launch.notna() & (X.v_ego < NF.V_STAND) & X.valid]
    per = {}
    for (s_, r_), g in La.groupby(["set", "read"]):
        per[f"{s_}:{r_}"] = dict(decisions=len(g), fires=int(g.fire.sum()), rate=float(g.fire.mean()), in_domain=float(g.standing.mean()),
                                 fires_by_delay={f"{lo}-{lo + 1}s": int(g[(g.h_launch > lo) & (g.h_launch <= lo + 1)].fire.sum()) for lo in range(4)})
    res["launch"] = dict(per_source=per, decisions=len(La), fires=int(La.fire.sum()), rate=float(La.fire.mean()) if len(La) else 0.0,
                         in_domain=float(La.standing.mean()) if len(La) else 0.0)
    g = dict(human=all(hm[p]["rate"] <= 0.01 for p in ("fut", "fut_reg")), creep=(res["creep"]["rate"] or 0) >= 0.5, launch=res["launch"]["rate"] < 0.05)
    res["gate"] = g | dict(pass_=all(g.values()))
    (Path(a.out) / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(0 if res["gate"]["pass_"] else 2)


def cmd_dose(a):
    """grad/lon_free and grad/lon_stand, both at W 0.004 (supplement 17 addendum 4), over the frozen-init steps -> W_S = 0.004 x mean free /
    mean stand (2 significant); any non-finite logged value voids the dose (exit 2)."""
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{a.run}/*/"))[-1]
    sc = {}
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc.setdefault(e["tag"], []).append(e["value"])
    gf, gs = sc.get("loss/grad/lon_free", []), sc.get("loss/grad/lon_stand", [])
    pf, ps = sc.get("loss/on/lon_free_pos", []), sc.get("loss/on/lon_stand_pos", [])
    assert len(gf) and len(gs), f"no grad/lon_free or grad/lon_stand in {a.run}: {sorted(k for k in sc if 'grad' in k)}"
    mf, ms = float(np.mean(gf)), float(np.mean(gs))
    fin = bool(np.isfinite(gf).all() and np.isfinite(gs).all())
    w = W_FREE * mf / ms if fin else float("nan")
    w2 = float(f"{w:.2g}")
    ok = fin and W_RANGE[0] <= w2 <= W_RANGE[1]
    out = dict(run=a.run, grad_free_w0004=mf, grad_stand_w0004=ms, finite=fin, per_log=dict(free=gf, stand=gs), n=[len(gf), len(gs)], W_S=w2, W_S_raw=w, in_range=ok,
               fire_share_free=float(np.mean(pf)) if len(pf) else None, fire_share_stand=float(np.mean(ps)) if len(ps) else None,
               grad_imit=float(np.mean(sc["loss/grad/imit"])) if "loss/grad/imit" in sc else None)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if ok else 2)


def cmd_drift(a):
    import pandas as pd
    sys.path[:0] = [str(REPO / "experiments/practice_nuplan/scripts")]
    import pnx
    D, S = pd.read_parquet(Path(a.follow) / "decisions.parquet"), pd.read_csv(Path(a.follow) / "scenes.csv")
    out = {b: pnx.drift(D, S, a.a, b) for b in a.b}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def cue_strata(src, tag, ctl):
    """Description (d380 / coordinator note): the arena-subset static windows by d372's cue (native_gap_report.cue), and the red-line launch
    windows (census red_go_t set: the human goes after red -> green; d380: about half render red after the switch), arm - control paired."""
    import pandas as pd
    import native_gap_report as NR
    from jevdrive import stats
    _, Wn, _, Cn = NR.load(Path(src))
    Sx = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    X = Wn.pivot_table(index="id", columns="tag", values="score")
    T = Wn.pivot_table(index="id", columns="tag", values="type", aggfunc="first")
    ids = X.index[X[[tag, ctl]].notna().all(axis=1) & ~(T[[tag, ctl]] == "route-fail").any(axis=1) & X.index.isin(Sx.index[Sx.split == "held"])]
    S_ = Sx.loc[ids]
    C_ = Cn.reindex(ids)
    groups = {f"static: {c}": S_.index[(S_.stratum == "static") & (NR.cue(S_, Cn) == c)] for c in
              ("red light at the line, no lead", "light of unknown state, no lead", "lead <= 8 m", "lead 8-30 m", "no light, no lead")}
    out_cnt = {t: dict(slow=int((T.loc[ids, t] == "slow").sum()), near1=int(((X.loc[ids, t] >= 0.98) & (X.loc[ids, t] <= 0.999)).sum()))
               for t in (tag, ctl)}
    groups["red-line launch (render-red per d380)"] = ids[C_.red_go_t.notna().values & (S_.stratum != "static").values]
    out = {}
    for k, g in groups.items():
        if not len(g):
            continue
        x, y = X.loc[g, tag].values, X.loc[g, ctl].values
        p = stats.paired(x, y, groups=Sx.log.reindex(g).values) if len(g) > 1 else dict(mean=float(x[0] - y[0]), lo=np.nan, hi=np.nan)
        out[k] = dict(windows=len(g), a=float(x.mean()), b=float(y.mean()), diff=p["mean"], lo=p["lo"], hi=p["hi"], zeros_a=int((x == 0).sum()),
                      zeros_b=int((y == 0).sum()))
    return out, out_cnt


def cmd_pilot(a):
    import df
    import rtja
    W = {t: rtja.wait_stats(t) for t in (a.tag, a.ctl)}
    H = {t: df.cl_read(t, [a.ctl, "RSG-L18-s0"] if t == a.tag else ["RSG-L18-s0"])["held"] for t in (a.tag, a.ctl)}
    Ar = json.load(open(a.arena))[a.ctl]
    Dr = json.load(open(a.drift))
    P, C = W[a.tag], W[a.ctl]
    LEAD = rtja.LEAD
    st = Ar["strata"]["static"]
    g = dict(held_zeros=H[a.tag]["zeros"] <= H[a.ctl]["zeros"] + 2,
             **{f"wait_{c}": P[c]["wait"] >= C[c]["wait"] - 0.03 for c in LEAD},
             **{f"delay_{c}": P[c]["delay_med"] <= C[c]["delay_med"] + 0.15 for c in LEAD},
             **{f"launched_{c}": P[c]["launched"] >= 0.95 * C[c]["launched"] for c in LEAD},
             held_front=H[a.tag]["coll"]["front"] <= H[a.ctl]["coll"]["front"] + 2, arena_af_front=Ar["af_front_a"] <= Ar["af_front_b"] + 2,
             static=st["a"] - st["b"] >= 0, arena_win=Ar["diff"] >= 0.01)
    verdict = "PASS" if all(g.values()) else "FAIL"
    hp = H[a.tag]["pair"].get(a.ctl, {})
    k9 = Ar["k9"]
    dr = next(iter(Dr.values()))
    yn = lambda x: "ok" if x else "FAIL"  # noqa: E731
    cs, cnt = cue_strata(a.ext, a.tag, a.ctl)
    L = [f"# NR1STAND pilot ({a.tag} against {a.ctl}; NR1 prereg supplement 17; native reads)", "",
         f"| | {a.ctl} (control) | {a.tag} | line |", "|:--|--:|--:|:--|",
         f"| held 460 zeros | {H[a.ctl]['zeros']} | {H[a.tag]['zeros']} | <= control + 2: {yn(g['held_zeros'])} |",
         *[f"| a wait {c} | {C[c]['wait']:.3f} | {P[c]['wait']:.3f} | >= control - 0.03: {yn(g['wait_' + c])} |" for c in LEAD],
         *[f"| b median launch delay {c} (s) | {C[c]['delay_med']:+.2f} | {P[c]['delay_med']:+.2f} | <= control + 0.15: {yn(g['delay_' + c])} |" for c in LEAD],
         *[f"| b launched {c} | {C[c]['launched']:.3f} | {P[c]['launched']:.3f} | >= 0.95 x control: {yn(g['launched_' + c])} |" for c in LEAD],
         f"| held 460 front collisions | {H[a.ctl]['coll']['front']} | {H[a.tag]['coll']['front']} | <= control + 2: {yn(g['held_front'])} |",
         f"| arena subset AF-front windows | {Ar['af_front_b']} | {Ar['af_front_a']} | <= control + 2: {yn(g['arena_af_front'])} |",
         f"| arena subset static stratum ({st['windows']} windows) mean / zeros | {st['b']:.4f} / {st['zeros_b']} | {st['a']:.4f} / {st['zeros_a']} | "
         f"paired >= 0: {yn(g['static'])} |",
         f"| arena subset ({Ar['windows']} windows) mean / zeros | {Ar['mean_b']:.4f} / {Ar['zeros_b']} | {Ar['mean_a']:.4f} / {Ar['zeros_a']} | "
         f"paired {Ar['diff']:+.4f} [{Ar['lo']:+.4f}, {Ar['hi']:+.4f}] >= +0.01: {yn(g['arena_win'])} |",
         "", "Description:", "",
         f"- arena subset partials (control -> arm; d382): slow (progress < 0.8) {cnt[a.ctl]['slow']} -> {cnt[a.tag]['slow']}, "
         f"near-1 (0.98-0.999) {cnt[a.ctl]['near1']} -> {cnt[a.tag]['near1']}; frozen-pool PCS: pcs_local.py score --subset on arena_windows.csv.gz (Mac)",
         f"- arena subset k = 9 speed ratio (median v / v_h): {k9['b']:.4f} -> {k9['a']:.4f}, diff {k9['diff']:+.4f} [{k9['ci'][0]:+.4f}, {k9['ci'][1]:+.4f}]",
         "- held 460 closed-loop lateral drift (left +, arm - control, log-clustered CI): " + ", ".join(
             f"{k} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] ({v['n']})" for k, v in dr.items()),
         f"- held 460 mean {H[a.ctl]['mean']:.4f} -> {H[a.tag]['mean']:.4f}, paired {hp.get('mean', float('nan')):+.4f} [{hp.get('lo', float('nan')):+.4f}, "
         f"{hp.get('hi', float('nan')):+.4f}]",
         "- arena subset strata (control -> arm, zeros): " + "; ".join(f"{s} ({v['windows']}) {v['b']:.3f} -> {v['a']:.3f}, {v['zeros_b']} -> {v['zeros_a']}"
                                                                     for s, v in Ar["strata"].items()),
         "- speed ratio by k (control / arm): " + ", ".join(f"k{k} {Ar['ratio_by_k'][a.ctl].get(k, float('nan')):.3f} / {Ar['ratio_by_k'][a.tag].get(k, float('nan')):.3f}"
                                                           for k in sorted(Ar["ratio_by_k"][a.tag], key=int)),
         "- arena subset by cue (arm - control paired, log-clustered; zeros control -> arm): " + "; ".join(
             f"{k} ({v['windows']}) {v['b']:.3f} -> {v['a']:.3f}, {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}], {v['zeros_b']} -> {v['zeros_a']}"
             for k, v in cs.items()),
         "- arena subset zero types (control / arm): " + json.dumps(Ar["types"]["b"]) + " / " + json.dumps(Ar["types"]["a"]),
         "", "Gates: " + ", ".join(f"{k} {yn(v)}" for k, v in g.items()), "", f"**Verdict: {verdict}**", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L))
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(tag=a.tag, ctl=a.ctl, wait=W, held=H, arena=Ar, drift=Dr, cue=cs, partials=cnt, gates=g, verdict=verdict),
                                                           indent=1, default=str))
    print("\n".join(L))
    print("VERDICT", verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("dose")
    p.add_argument("--run", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("drift")
    p.add_argument("--follow", required=True), p.add_argument("--a", required=True), p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("pilot")
    for k in ("tag", "ctl", "arena", "drift", "ext", "out"):
        p.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    dict(gate=cmd_gate, dose=cmd_dose, drift=cmd_drift, pilot=cmd_pilot)[a.cmd](a)
