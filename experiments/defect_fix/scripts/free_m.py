#!/usr/bin/env python3
"""NR1FREE-M (NR1 prereg supplement 16; lib/nr1_free.py with mode "move", FRAC 0.95): the one-sided longitudinal term on moving states only.
Box, repo root.

  $AP free_m.py gate --out DIR            offline gate (CPU) -> DIR/gate.json, DIR/decisions.parquet; exit 0 pass, 2 fail
  $AP free_m.py ext --read TAG=DIR.. --out DIR      native_gap extract of arena reads (dec / win parquet)
  .venv free_m.py arena --src DIR --a TAG --b TAG [--ref TAG..] --out JSON   speed ratio by k, AF-front windows, means, strata (a against b)
  $TR free_m.py pilot --tag P --ctl C --arena JSON --pai JSON --out MD     the supplement 16 pilot rule -> MD (+ .json); prints VERDICT

Every decision is scored with nr1_free.terms(parts=True) as the trainer scores a pool row, the row limit of nr1_free.moving applied to `ok`.
Closed-loop states as free_gate.py (scene time -> log frame start_idx + round(20 t); controller trace on the log by the rigid transform of its
first pose; reference = min(the human's raw 4 s arc, its reachable version from the controller speed)); ego speed = the controller speed at
t; the human's segment speeds = the log's 20 Hz path arc over each 0.5 s.
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
FRAC, MODE = 0.95, "move"
RATCHET = "JTN-L18-s1@off"
FRONT = ("JTN-L18-s1@off", "JTN-L18-s0@off", "JT10-L18-s0@leash3", "JT10-L18-s0@off", "JT00-L18-s0@leash3", "JT00-L18-s0@off")
LATE = ("JTN-L18-s0", "RTJA0-L18-s0")
HZ = 20


def _nf():
    import nr1_free as NF
    NF.configure(FRAC, MODE)
    return NF


def hum_seg(L, f):
    """the human's 0.5 s segment speeds (8,) from log frame f."""
    seg = np.hypot(*np.diff(L.xy[f:f + 4 * HZ + 1], axis=0).T)
    return np.diff(np.cumsum(seg)[HZ // 2 - 1::HZ // 2], prepend=0.0) / 0.5


def _run(job):
    """one read / dump run -> list of decision dicts. job = (set name, read label, run dir, {scene: (log, start_idx, token, t_lo, t_hi, extra)})."""
    import torch
    import free_gate as FG
    torch.set_num_threads(1)
    NF = _nf()
    name, label, run, want = job
    R = FG.run_rollouts(run)
    logs, rows, info = {}, [], []
    for sc, (C, dec, r) in R.items():
        if sc not in want:
            continue
        log, s0, tok, lo, hi, extra = want[sc]
        if callable(lo):
            lo, hi = lo(run, sc, r)
            if lo is None:
                continue
        L = logs.get(log) or logs.setdefault(log, FG.Log(log))
        assert L.tok[s0 + 30].hex() == tok, sc
        for k, t, p in dec:
            if not (lo <= t <= hi):
                continue
            ve = float(np.interp(t, C["t"], C["v"]))
            d = dict(set=name, read=label, scene=sc, k=k, t=t, v_ego=ve, **extra)
            if ve < NF.V_MOVE:                         # the row limit is 0 here by construction: no state needed
                info.append(dict(d, scored=False, moving=False, fire=False))
                continue
            q = FG.state_row(L, s0, C, t, p)
            if q is None:
                continue
            vh = hum_seg(L, q[4]["f"])
            rows.append(q[:4])
            info.append(dict(d, scored=True, moving=bool(NF.moving(np.array([ve]), vh[None])[0]), vh1=float(vh[0]), vh_min=float(vh.min()),
                             raw4=q[4]["raw4"], reach4=q[4]["reach4"]))
    if rows:
        P, Sr, B, V = (torch.as_tensor(np.stack([r[i] for r in rows])) for i in range(4))
        _, pr = NF.terms(P, Sr, torch.ones(len(P), dtype=torch.bool), B, V, parts=True)    # mask-free: why a decision is blocked
        j = 0
        for d in info:
            if d["scored"]:
                d.update(plan4=float(pr["s_plan"][j]), sref4=float(Sr[j, -1]), short=bool(pr["slow"][j]), free=bool(pr["free"][j]),
                         fire=bool(pr["fire"][j]) and d["moving"])
                j += 1
    return info


def _front_window(run, sc, r):
    import free_gate as FG
    tc = FG.first_front(run, sc, r["rollout_id"])
    return (None, None) if tc is None else (tc - 3.0, tc)


def cmd_gate(a):
    import arena as A
    import pandas as pd
    import torch
    from multiprocessing import Pool
    torch.set_num_threads(16)
    NF = _nf()
    res = dict(frac=NF.FRAC, mode=NF.MODE, v_move=NF.V_MOVE, hum_keep=NF.HUM_KEEP, s_min=NF.S_MIN, d0=NF.D0, th=NF.TH)
    # 1. the human's own paths on the pool's S rows
    z = dict(np.load(DATA / "runs/op_parity/cache/on1_JT/tab.npz"))
    ag = dict(np.load(DATA / "runs/op_parity/cache/on1_JT/agent.npz"))
    s_ref, fin = NF.ref_arc(z["fut"], z["fut_reg"])
    S = (z["mode"].astype(str) == "self") & ag["ok"] & fin
    mv = NF.moving(z["speed"], NF.seg_speed(z["fut_reg"]))
    m = np.flatnonzero(S & mv)
    alt = dict(ref_min=s_ref, ref_fut=np.nan_to_num(NF.cum_arc(z["fut"])).astype(np.float32), ref_reg=np.nan_to_num(NF.cum_arc(z["fut_reg"])).astype(np.float32))
    hum = dict(s_rows=int(S.sum()), moving_rows=int(len(m)), moving_share=float(len(m) / max(S.sum(), 1)),
               ego_ge_3=int((S & (z["speed"] >= NF.V_MOVE)).sum()))
    for plan in ("fut", "fut_reg"):
        for rn, Rr in alt.items():
            fire = []
            for i in range(0, len(m), 4096):
                j = m[i:i + 4096]
                _, pr = NF.terms(torch.as_tensor(z[plan][j]), torch.as_tensor(Rr[j]), torch.ones(len(j), dtype=torch.bool),
                                 torch.as_tensor(ag["box"][j]), torch.as_tensor(ag["valid"][j]), parts=True)
                fire.append(pr["fire"].numpy())
            fire = np.concatenate(fire) if fire else np.zeros(0, bool)
            hum[f"{plan}|{rn}"] = dict(fires=int(fire.sum()), rate=float(fire.mean()) if len(fire) else 0.0)
            print(f"human plan {plan} ref {rn}: {fire.sum()} / {len(fire)}", flush=True)
    res["human"] = hum
    del z, ag
    # closed-loop sets
    Sx = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    held = Sx[Sx.split == "held"]
    jobs = []
    for rd in sorted(set((RATCHET,) + FRONT)):
        for run in sorted(glob.glob(str(O / "jt/arena" / rd / "c*/run"))):
            if not (Path(run) / "aggregate/results-summary.json").exists():
                continue
            if rd == RATCHET:                                                     # every held window: ratchet (moving) + static
                want = {i: (w.log, int(w.start_idx), w.token, 0.0, 1e9, dict(stratum=w.stratum)) for i, w in held.iterrows()}
                jobs.append(("ratchet", rd, run, want))
            if rd in FRONT:
                summ = json.load(open(Path(run) / "aggregate/results-summary.json"))["rollouts"]
                af = {r["clipgt_id"] for r in summ if r["clipgt_id"] in held.index and A.failure_type(r).startswith("AF-front")}
                want = {i: (held.loc[i, "log"], int(held.loc[i, "start_idx"]), held.loc[i, "token"], _front_window, None, {}) for i in af}
                if want:
                    jobs.append(("front", rd, run, want))
    WS = {w["id"]: w for w in json.load(open(REPO / "experiments/alpasim_proxy/results/wait_set/set.json"))["windows"]
          if w["variant"] == "b" and w["cls"] in ("other_lead", "red_queue")}
    for tag in LATE:
        want = {i: (w["log"], int(w["start_idx"]), w["token"], 0.0, 1e9, dict(cls=w["cls"])) for i, w in WS.items()}
        jobs.append(("late", tag, str(O / "launch_diag/dump" / tag / "all"), want))
    print(f"{len(jobs)} runs", flush=True)
    with Pool(a.workers) as p:
        out = p.map(_run, jobs, chunksize=1)
    D = pd.DataFrame([d for o in out for d in o])
    for c in ("scored", "moving", "short", "free", "fire"):
        D[c] = D[c].fillna(False).astype(bool)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    D.to_parquet(Path(a.out) / "decisions.parquet")
    # 2. ratchet: non-static held windows of JTN1 native, k >= 2, ego >= 3 m/s, plan 4 s arc < FRAC x s_ref
    R = D[(D.set == "ratchet") & (D.stratum != "static") & (D.k >= 2) & (D.v_ego >= NF.V_MOVE) & D.scored]
    rs = R[R.short]
    res["ratchet"] = dict(moving_decisions=len(R), windows=int(R.scene.nunique()), short=len(rs), fires=int(rs.fire.sum()),
                          rate=float(rs.fire.mean()) if len(rs) else None, fire_share_of_moving=float(R.fire.mean()) if len(R) else None,
                          ratio_median=float((R.plan4 / R.sref4.clip(lower=1e-3))[R.sref4 >= 1].median()),
                          short_share=float(len(rs) / max(len(R), 1)), blocked_human=int((~rs.moving).sum()), blocked_not_free=int((rs.moving & ~rs.free).sum()),
                          windows_with_fire=int(rs[rs.fire].scene.nunique()))
    # 3a. AF-front windows
    F = D[D.set == "front"]
    fr = {}
    for rd, g in F.groupby("read"):
        fr[rd] = dict(windows=int(g.scene.nunique()), decisions=len(g), windows_fired=int(g[g.fire].scene.nunique()), decisions_fired=int(g.fire.sum()),
                      moving_decisions=int((g.v_ego >= NF.V_MOVE).sum()), short_moving=int((g.scored & g.short).sum()))
    nw = sum(v["windows"] for v in fr.values())
    fw = sum(v["windows_fired"] for v in fr.values())
    res["front"] = dict(per_read=fr, windows=nw, windows_fired=fw, rate=fw / max(nw, 1))
    # 3b. static windows, 3c. late-launch lead windows
    St = D[(D.set == "ratchet") & (D.stratum == "static")]
    res["static"] = dict(windows=int(St.scene.nunique()), decisions=len(St), fires=int(St.fire.sum()), rate=float(St.fire.mean()) if len(St) else 0.0,
                         ego_ge_3=int((St.v_ego >= NF.V_MOVE).sum()))
    Lt = D[D.set == "late"]
    res["late"] = dict(per_tag={t: dict(decisions=len(g), fires=int(g.fire.sum())) for t, g in Lt.groupby("read")}, decisions=len(Lt),
                       fires=int(Lt.fire.sum()), rate=float(Lt.fire.mean()) if len(Lt) else 0.0)
    g = dict(human=all(hum[f"{p}|ref_min"]["rate"] <= 0.01 for p in ("fut", "fut_reg")), ratchet=(res["ratchet"]["rate"] or 0) >= 0.5,
             front=res["front"]["rate"] <= 0.05, static=res["static"]["rate"] <= 0.01, late=res["late"]["rate"] <= 0.01)
    res["gate"] = g | dict(pass_=all(g.values()))
    (Path(a.out) / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(0 if res["gate"]["pass_"] else 2)


def cmd_ext(a):
    import native_gap as NG
    NG.READS = dict(x.split("=", 1) for x in a.read)
    NG.READS = {k: Path(v) for k, v in NG.READS.items()}
    a.limit, a.limit_win = 0, 0
    NG.cmd_extract(a)


def cmd_arena(a):
    import pandas as pd
    import native_gap_report as NR
    from jevdrive import stats
    D, Wn, _, _ = NR.load(Path(a.src))
    S = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    X = Wn.pivot_table(index="id", columns="tag", values="score")
    T = Wn.pivot_table(index="id", columns="tag", values="type", aggfunc="first")
    out = dict(a=a.a, b=a.b)
    for other in [a.b, *a.ref]:
        ids = X.index[X[[a.a, other]].notna().all(axis=1) & ~(T[[a.a, other]] == "route-fail").any(axis=1) & X.index.isin(S.index[S.split == "held"])]
        x, y = X.loc[ids, a.a].values, X.loc[ids, other].values
        lg = S.log.reindex(ids).values
        p = stats.paired(x, y, groups=lg)
        r = dict(windows=len(ids), mean_a=float(x.mean()), mean_b=float(y.mean()), diff=p["mean"], lo=p["lo"], hi=p["hi"],
                 zeros_a=int((x == 0).sum()), zeros_b=int((y == 0).sum()),
                 af_front_a=int(T.loc[ids, a.a].fillna("").str.startswith("AF-front").sum()), af_front_b=int(T.loc[ids, other].fillna("").str.startswith("AF-front").sum()))
        M = D[D.id.isin(ids) & D.valid & (D.k >= 2) & (D.vh >= 2) & D.tag.isin([a.a, other])].copy()
        M["r"] = M.v / M.vh
        r["ratio_by_k"] = M.pivot_table(index="k", columns="tag", values="r", aggfunc="median").round(4).to_dict()
        m9 = M[M.k == 9]                                                   # s2report's rule: the median over each tag's valid k = 9 decisions
        ma, mb = m9[m9.tag == a.a], m9[m9.tag == other]
        la, lb = S.log.reindex(ma.id).values, S.log.reindex(mb.id).values
        ra, rb = ma.r.values, mb.r.values
        rng = np.random.default_rng(0)
        u = np.unique(S.log.reindex(ids).values)
        ia, ib = {x: np.flatnonzero(la == x) for x in u}, {x: np.flatnonzero(lb == x) for x in u}
        bs = []
        for _ in range(2000):                                              # one log resample for both tags
            q = rng.choice(u, len(u))
            xa, xb = np.concatenate([ra[ia[x]] for x in q]), np.concatenate([rb[ib[x]] for x in q])
            bs.append(np.median(xa) - np.median(xb) if len(xa) and len(xb) else np.nan)
        r["k9"] = dict(n_a=len(ra), n_b=len(rb), a=float(np.median(ra)), b=float(np.median(rb)), diff=float(np.median(ra) - np.median(rb)),
                       ci=np.nanquantile(bs, [0.025, 0.975]).tolist())
        st = S.stratum.reindex(ids)
        r["strata"] = {s_: dict(windows=int((st == s_).sum()), a=float(x[(st == s_).values].mean()), b=float(y[(st == s_).values].mean()),
                                zeros_a=int(((st == s_).values & (x == 0)).sum()), zeros_b=int(((st == s_).values & (y == 0)).sum()))
                       for s_ in sorted(st.dropna().unique())}
        r["types"] = dict(a=T.loc[ids, a.a].value_counts().to_dict(), b=T.loc[ids, other].value_counts().to_dict())
        out[other] = r
        print(other, json.dumps({k: v for k, v in r.items() if k not in ("ratio_by_k", "types")}, default=float), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))


def cmd_pilot(a):
    import df
    import rtja
    W = {t: rtja.wait_stats(t) for t in (a.tag, a.ctl)}
    H = {t: df.cl_read(t, [a.ctl, "RSG-L18-s0"] if t == a.tag else ["RSG-L18-s0"])["held"] for t in (a.tag, a.ctl)}
    Ar = json.load(open(a.arena))[a.ctl]
    Pg = json.load(open(a.pai))
    P, C = W[a.tag], W[a.ctl]
    LEAD = rtja.LEAD
    g = dict(held_zeros=H[a.tag]["zeros"] <= H[a.ctl]["zeros"] + 2,
             **{f"wait_{c}": P[c]["wait"] >= C[c]["wait"] - 0.05 for c in LEAD},
             held_front=H[a.tag]["coll"]["front"] <= H[a.ctl]["coll"]["front"] + 2, arena_af_front=Ar["af_front_a"] <= Ar["af_front_b"] + 2,
             ratchet_k9=Ar["k9"]["diff"] >= 0.02,
             pai_half=Pg["paired_pilot_minus_control"]["mean"] >= 0 and Pg["pilot"]["zeros"] <= Pg["control"]["zeros"] + 2)
    ok = all(g.values())
    verdict = "PASS" if ok else "FAIL"
    hp = H[a.tag]["pair"].get(a.ctl, {})
    pp = Pg["paired_pilot_minus_control"]
    L = [f"# NR1FREE-M pilot ({a.tag} against {a.ctl}; NR1 prereg supplement 16; native reads)", "",
         "| | " + a.ctl + " (control) | " + a.tag + " | line |", "|:--|--:|--:|:--|",
         f"| held 460 zeros | {H[a.ctl]['zeros']} | {H[a.tag]['zeros']} | <= control + 2: {'ok' if g['held_zeros'] else 'FAIL'} |",
         *[f"| a wait {c} | {C[c]['wait']:.3f} | {P[c]['wait']:.3f} | >= control - 0.05: {'ok' if g['wait_' + c] else 'FAIL'} |" for c in LEAD],
         f"| held 460 front collisions | {H[a.ctl]['coll']['front']} | {H[a.tag]['coll']['front']} | <= control + 2: {'ok' if g['held_front'] else 'FAIL'} |",
         f"| arena subset AF-front windows | {Ar['af_front_b']} | {Ar['af_front_a']} | <= control + 2: {'ok' if g['arena_af_front'] else 'FAIL'} |",
         f"| arena subset k = 9 speed ratio (median v / v_h) | {Ar['k9']['b']:.4f} | {Ar['k9']['a']:.4f} | diff {Ar['k9']['diff']:+.4f} "
         f"[{Ar['k9']['ci'][0]:+.4f}, {Ar['k9']['ci'][1]:+.4f}] >= +0.02: {'ok' if g['ratchet_k9'] else 'FAIL'} |",
         f"| PAI half (4 chunks) mean / zeros | {Pg['control']['mean']:.4f} / {Pg['control']['zeros']} | {Pg['pilot']['mean']:.4f} / {Pg['pilot']['zeros']} | "
         f"paired {pp['mean']:+.4f} [{pp['lo']:+.4f}, {pp['hi']:+.4f}] >= 0, zeros <= control + 2: {'ok' if g['pai_half'] else 'FAIL'} |",
         "", "Description:", "",
         *[f"- b {c}: launched {C[c]['launched']:.3f} -> {P[c]['launched']:.3f}, median delay {C[c]['delay_med']:+.2f} -> {P[c]['delay_med']:+.2f} s" for c in LEAD],
         f"- held 460 mean {H[a.ctl]['mean']:.4f} -> {H[a.tag]['mean']:.4f}, paired {hp.get('mean', float('nan')):+.4f} [{hp.get('lo', float('nan')):+.4f}, {hp.get('hi', float('nan')):+.4f}]",
         f"- arena subset ({Ar['windows']} windows) mean {Ar['mean_b']:.4f} -> {Ar['mean_a']:.4f}, paired {Ar['diff']:+.4f} [{Ar['lo']:+.4f}, {Ar['hi']:+.4f}] "
         f"(log-clustered), zeros {Ar['zeros_b']} -> {Ar['zeros_a']}",
         "- arena subset strata (control -> pilot, zeros): " + "; ".join(f"{s} ({v['windows']}) {v['b']:.3f} -> {v['a']:.3f}, {v['zeros_b']} -> {v['zeros_a']}"
                                                                        for s, v in Ar["strata"].items()),
         "- speed ratio by k (control / pilot): " + ", ".join(f"k{k} {Ar['ratio_by_k'][a.ctl].get(k, float('nan')):.3f} / {Ar['ratio_by_k'][a.tag].get(k, float('nan')):.3f}"
                                                             for k in sorted(Ar["ratio_by_k"][a.tag], key=int)),
         "", "Gates: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in g.items()), "", f"**Verdict: {verdict}**", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L))
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(tag=a.tag, ctl=a.ctl, wait=W, held=H, arena=Ar, pai=Pg, gates=g, verdict=verdict), indent=1, default=str))
    print("\n".join(L))
    print("VERDICT", verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("ext")
    p.add_argument("--read", nargs="+", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=32)
    p = sp.add_parser("arena")
    p.add_argument("--src", required=True), p.add_argument("--a", required=True), p.add_argument("--b", required=True)
    p.add_argument("--ref", nargs="*", default=[]), p.add_argument("--out", required=True)
    p = sp.add_parser("pilot")
    for k in ("tag", "ctl", "arena", "pai", "out"):
        p.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    dict(gate=cmd_gate, ext=cmd_ext, arena=cmd_arena, pilot=cmd_pilot)[a.cmd](a)
