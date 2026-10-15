#!/usr/bin/env python3
"""MODE AVERAGING (plans/2026-10-15-mode-averaging-prereg.md; decision 306): is the plan's turn deficit larger where the junction offers
more exits, smaller where the command already names the turn, and does the plan near the entry look like an average of modes (partial
turns) or a choice between modes (straight or full)? CPU only; stored reads are read, never written.

  thk   (box, repo .venv)  rf_modeavg.py thk --runs TAG=<run>,<run> ... --held <RECEDING held pkl> --out <pkl>
        the flow head's logged samples (drive.jsonl `th_k`, noise rows 0 .. K-1) of every decision of the strict-turn rollouts.
  read  (Mac, .venv)  rf_modeavg.py read --dec <RECEDING decisions.parquet> --pai <pai pkl> --held <held pkl> --rmem NAME=<pkl> ...
        --lanes pai=<pkl> held=<pkl> --out DIR --fig PNG       Reading 1 -> DIR/{mode_avg.md, mode_avg.json, turns.csv, decisions.csv.gz}
  flow  (Mac, .venv)  Reading 2 on the FLOW1 closed-loop reads (see main()).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]


# ---------------------------------------------------------------- thk (box)
def thk(a):
    held = pickle.load(open(a.held, "rb"))
    out = {}
    for spec in a.runs:
        tag, runs = spec.split("=", 1)
        rids = {r["rid"] for r in held if r["set"] == tag}
        for run in runs.split(","):
            for line in open(os.path.join(run, "driver-logs", "drive.jsonl")):
                if '"th_k"' not in line:
                    continue
                r = json.loads(line)
                if r.get("session") in rids:
                    out.setdefault(r["session"], []).append((int(r["t0"]), np.asarray(r["th_k"], np.float32)))
        print(tag, len(rids), "strict rollouts,", sum(k in rids for k in out), "with samples", flush=True)
    res = {}
    for rid, v in out.items():
        v.sort(key=lambda x: x[0])
        t0 = np.array([x[0] for x in v], np.int64)
        keep = np.r_[True, np.diff(t0) > 0]
        res[rid] = dict(t0=t0[keep], th_k=np.stack([x[1] for x in v])[keep])
    pickle.dump(res, open(a.out, "wb"), protocol=4)
    print(len(res), "rollouts ->", a.out)


# ---------------------------------------------------------------- read (Mac): Reading 1
WIN_M, H_MIN, CLUS_DEG, GROUP_D, GROUP_PSI, MATCH_D, PAST_M, N_BOOT = 30.0, 15.0, 20.0, 7.0, 30.0, 2.5, 15.0, 10_000
PSTR, PFULL = 0.25, 0.6
CMD = {0: "L", 1: "S", 2: "R", 3: "U", -1: "-"}


def wrapd(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def lane_geo(lanes):
    """{id: (centre (n, 2), stations, unwrapped heading per point)} for lanes with >= 2 points."""
    import rf_pai_follow as F
    G = {}
    for k, l in lanes.items():
        c = F.dedup(np.asarray(l["c"], float)[:, :2])
        if len(c) < 2:
            continue
        d = np.diff(c, axis=0)
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        G[k] = (c, F.stations(c), np.r_[h, h[-1]])
    return G


def exits(lanes, G, R, S, sh, hh, ent, ext):
    """Outgoing directions at the junction ahead of the turn entry (prereg 'Exits'): -> dict(n_x, n_x_lane, dirs, route_dir, matched, ok)."""
    import rf_pai_follow as F
    s_a = max(ent - 5.0, 0.0)
    p = np.array([np.interp(s_a, S, R[:, 0]), np.interp(s_a, S, R[:, 1])])
    psi = float(np.interp(s_a, sh, hh))
    L = (ext - s_a) + 15.0
    route_dir = float(np.degrees(np.interp(min(s_a + L, sh[-1]), sh, hh) - psi))
    grp, best = [], (np.inf, None)
    for k, (c, sc, hc) in G.items():
        st, lat = F.project(c, sc, p)
        if abs(lat) > GROUP_D:
            continue
        dpsi = abs(wrapd(np.degrees(np.interp(st, sc, hc) - psi)))
        if dpsi > GROUP_PSI:
            continue
        grp.append(k)
        if abs(lat) <= MATCH_D and abs(lat) < best[0]:
            best = (abs(lat), k)

    def dirs_of(ids):
        out = []
        for k in ids:
            for P, _, full in F.chains(lanes, k, p, L, cap=256):
                if len(P) < 2:
                    continue
                Sp = F.stations(P)
                if not full and Sp[-1] < min(25.0, L):
                    continue
                s_, h_ = F.heading(P, Sp)
                out.append(float(wrapd(np.degrees(np.interp(min(L, Sp[-1] - F.HALF), s_, h_)) - np.degrees(psi))))
        return np.array(out)

    def n_clusters(d):
        if not len(d):
            return 0
        d = np.sort(d)
        return int(1 + (np.diff(d) >= CLUS_DEG).sum())
    D = dirs_of(grp)
    D1 = dirs_of([best[1]]) if best[1] is not None else np.array([])
    ok = bool(len(D) and np.abs(wrapd(D - route_dir)).min() <= CLUS_DEG)
    return dict(n_x=n_clusters(D), n_x_lane=n_clusters(D1), dirs=np.round(np.sort(D), 1).tolist(), route_dir=route_dir, matched=best[1] is not None,
                ok=ok, n_group=len(grp))


def cboot(v, g, cl, n_boot=N_BOOT, seed=0):
    """median(v[g]) - median(v[~g]) with a cluster bootstrap (clusters cl) -> dict(d, lo, hi, n1, n0, dropped)."""
    v, g, cl = np.asarray(v, float), np.asarray(g, bool), np.asarray(cl)
    m = np.isfinite(v)
    v, g, cl = v[m], g[m], cl[m]
    res = dict(n1=int(g.sum()), n0=int((~g).sum()), c1=int(len(np.unique(cl[g]))), c0=int(len(np.unique(cl[~g]))))
    if not g.any() or g.all():
        return res | dict(d=np.nan, lo=np.nan, hi=np.nan, dropped=0, m1=np.nan, m0=np.nan)
    u = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in u}
    rng = np.random.default_rng(seed)
    out, drop = [], 0
    for _ in range(n_boot):
        r = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        a, b = v[r][g[r]], v[r][~g[r]]
        if not len(a) or not len(b):
            drop += 1
            continue
        out.append(np.median(a) - np.median(b))
    m1, m0 = float(np.median(v[g])), float(np.median(v[~g]))
    return res | dict(d=m1 - m0, lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)), dropped=drop, m1=m1, m0=m0)


def verdict(b, thr, nmin, cmin=4):
    if b["n1"] < nmin or b["n0"] < nmin or b["c1"] < cmin or b["c0"] < cmin or not np.isfinite(b["d"]):
        return "inconclusive (group too small)"
    if b["d"] >= thr and b["lo"] > 0:
        return "supported"
    if b["hi"] < thr:
        return "refuted"
    return "inconclusive"


def key_of(r):
    return ("x_" if r["src"].endswith("_x") else "") + ("pai_" if r["src"].startswith("pai") else "held_") + r["set"]


def turn_rows(recs, lanes_by, rmem_by, Dx, keys):
    """Per turn (exits) and per window decision (deficit, command, memory) for the records of `keys`."""
    import pandas as pd
    import rf_pai_follow as F
    T, Dd, geo = [], [], {}
    for r in recs:
        k = key_of(r)
        if k not in keys or not r["row"].get("turn") or r.get("plans") is None:
            continue
        pop = "pai" if r["src"].startswith("pai") else "held"
        R = r["R"]
        S = F.stations(R)
        sh, hh = F.heading(R, S)
        ent, ext = float(r["row"]["entry"]), float(r["row"]["exit"])
        lanes = lanes_by[pop].get(r["scene"])
        if lanes is not None and r["scene"] not in geo:
            geo[r["scene"]] = lane_geo(lanes)
        ex = exits(lanes, geo[r["scene"]], R, S, sh, hh, ent, ext) if lanes is not None else dict(n_x=np.nan, n_x_lane=np.nan, dirs=[], route_dir=np.nan,
                                                                                                    matched=False, ok=False, n_group=0)
        cl = r["scene"].rsplit("-", 1)[0] if pop == "held" else r["scene"]
        d = Dx[(Dx.key == k) & (Dx.scene == r["scene"])]
        cls, done = (d.cls.iloc[0], bool(d.completed.iloc[0])) if len(d) else ("-", bool(r["row"].get("completed")))
        w = d[d.approach & (d.d_e <= WIN_M)].copy()
        rm = rmem_by.get(k.removeprefix("x_"), {}).get(r["rid"])
        sg = 1.0 if r["row"]["side"] == "L" else -1.0
        for _, x in w.iterrows():
            o = dict(popn=pop, key=k, scene=r["scene"], cluster=cl, cls=cls, completed=done, side=r["row"]["side"], d_e=float(x.d_e), t0=float(x.t0),
                     dpsi=float(x.dpsi), h_dpsi=float(x.h_dpsi), cmd="-", n_past=np.nan, cov=np.nan, a4=float(x.a4), v=float(x.v),
                     route_arc=float(sg * np.degrees(np.interp(min(ent - x.d_e + x.a4, sh[-1]), sh, hh) - np.interp(ent - x.d_e, sh, hh))))
            if rm is not None:
                j = int(np.searchsorted(rm["t0"], int(x.t0)))
                if j < len(rm["t0"]) and rm["t0"][j] == int(x.t0):
                    o["cmd"] = CMD.get(int(rm["cmd"][j]), "-")
                    V, A = rm["V"][j], rm["anchor"][j]
                    if V.any():
                        c, s_ = np.cos(A[2]), np.sin(A[2])
                        X = rm["XY"][j][V]
                        Wp = np.c_[A[0] + c * X[:, 0] - s_ * X[:, 1], A[1] + s_ * X[:, 0] + c * X[:, 1]]
                        se = ent - x.d_e
                        st = np.array([F.project(R, S, q, se - 30.0, se + 120.0)[0] for q in Wp])
                        q = np.arange(ent, ext + 1e-9, 1.0)
                        o["n_past"], o["cov"] = int((st > ent).sum()), float(((q >= st.min()) & (q <= st.max())).mean())
                    else:
                        o["n_past"], o["cov"] = 0, 0.0
            o["agree"] = o["cmd"] == r["row"]["side"]
            o["opp"] = o["cmd"] in ("L", "R") and o["cmd"] != r["row"]["side"]
            Dd.append(o)
        T.append(dict(popn=pop, key=k, scene=r["scene"], cluster=cl, cls=cls, completed=done, side=r["row"]["side"], turn_deg=float(r["row"]["turn_deg"]),
                      entry=ent, exit=ext, **{kk: (json.dumps(v) if kk == "dirs" else v) for kk, v in ex.items()}))
    D = pd.DataFrame(Dd)
    ok = D.h_dpsi >= H_MIN
    D["delta"] = np.where(ok, np.clip(1.0 - D.dpsi / D.h_dpsi.where(ok, np.nan), -1.0, 2.0), np.nan)
    D["r"] = np.where(ok, D.dpsi / D.h_dpsi.where(ok, np.nan), np.nan)
    ok2 = D.route_arc >= H_MIN                                         # Amendment 3: the route's own turn over the plan's 4 s arc
    D["delta_arc"] = np.where(ok2, np.clip(1.0 - D.dpsi / D.route_arc.where(ok2, np.nan), -1.0, 2.0), np.nan)
    D["r_arc"] = np.where(ok2, D.dpsi / D.route_arc.where(ok2, np.nan), np.nan)
    T = pd.DataFrame(T)
    g = D[np.isfinite(D.delta)].groupby(["key", "scene"])
    T = T.merge(g.delta.median().rename("D").reset_index(), on=["key", "scene"], how="left").merge(
        g.delta.size().rename("n_def").reset_index(), on=["key", "scene"], how="left")
    T["n_win"] = T.merge(D.groupby(["key", "scene"]).size().rename("nw").reset_index(), on=["key", "scene"], how="left").nw.fillna(0).astype(int).values
    return T, D


def m4_shares(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(n=0)
    h, e = np.histogram(np.clip(r, -0.999, 1.999), bins=np.arange(-1.0, 2.0001, 0.1))
    mode = float(e[np.argmax(h)])
    sh = dict(n=int(len(r)), straight=float((r < PSTR).mean()), partial=float(((r >= PSTR) & (r < PFULL)).mean()), full=float((r >= PFULL).mean()),
              median=float(np.median(r)), mode_bin=[round(mode, 2), round(mode + 0.1, 2)])
    if sh["straight"] >= 0.6:
        rd = "straight-dominant"
    elif sh["partial"] >= sh["full"] and 0.1 <= mode < 0.6:
        rd = "averaging"
    elif sh["partial"] < 0.2 and sh["straight"] >= 0.2 and sh["full"] >= 0.2:
        rd = "mode selection"
    else:
        rd = "mixed"
    return sh | dict(reading=rd)


def read(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Dx = pd.read_parquet(a.dec)
    recs = pickle.load(open(a.pai, "rb")) + pickle.load(open(a.held, "rb"))
    rmem_by = {n: pickle.load(open(f, "rb")) for n, f in (x.split("=", 1) for x in a.rmem)}
    lanes_by = {}
    for x in a.lanes:
        n, f = x.split("=", 1)
        L = pickle.load(open(f, "rb"))
        lanes_by[n] = {k: (v["lanes"] if "lanes" in v else v) for k, v in L.get("scenes", L).items()}
    keys = {"pai_RSG-L18-s0", "x_pai_RSG-L18-s0", "held_RSG-L18-s0", "held_RSG-L18-s1", "x_held_RSG-L18-s0", "x_held_RSG-L18-s1"}
    T, D = turn_rows(recs, lanes_by, rmem_by, Dx, keys)
    T.to_csv(out / "turns.csv", index=False)
    D.to_csv(out / "decisions.csv.gz", index=False)
    res = dict(n_turns=int(len(T)), gates={}, M1={}, M2={}, M3={}, M4={}, dist={}, nx_table={})
    pops = {"pai": T.popn == "pai", "held": T.popn == "held", "pooled": T.popn.notna()}
    dpops = {"pai": D.popn == "pai", "held": D.popn == "held", "pooled": D.popn.notna()}
    # G-map
    for p, m in pops.items():
        x = T[m]
        res["gates"][p] = dict(turns=int(len(x)), matched=float(x.matched.mean()), dir_ok=float(x[x.matched].ok.mean()) if x.matched.any() else np.nan,
                               passed=bool(x.matched.mean() >= 0.9 and x[x.matched].ok.mean() >= 0.8),
                               deficit_defined=int(x.D.notna().sum()), window_decisions=int(D[dpops[p]].shape[0]),
                               decisions_with_deficit=int(np.isfinite(D[dpops[p]].delta).sum()), cmd_joined=float((D[dpops[p]].cmd != "-").mean()))
    for p, m in pops.items():
        x = T[m & T.D.notna() & T.n_x.isin([2]) | m & T.D.notna() & (T.n_x >= 3)]
        b = cboot(x.D, x.n_x >= 3, x.cluster)
        v = verdict(b, 0.2, 4)
        if not res["gates"][p]["passed"]:
            v = "inconclusive (G-map failed)"
        res["M1"][p] = b | dict(verdict=v)
        y = T[m]
        res["nx_table"][p] = {str(int(n)) if np.isfinite(n) else "nan": dict(turns=int(len(z)), failed=int((~z.completed).sum()),
                                                                             D_med=float(z.D.median()), D_med_failed=float(z[~z.completed].D.median()),
                                                                             D_med_completed=float(z[z.completed].D.median()))
                              for n, z in y.groupby(y.n_x.clip(upper=4), dropna=False)}
    for p, m in dpops.items():
        x = D[m & np.isfinite(D.delta)]
        z = x[x.agree | (x.cmd == "S")]
        b = cboot(z.delta, z.cmd == "S", z.cluster)
        mt, wsum = 0.0, 0.0
        per = {}
        for lo in (0, 10, 20):
            q = z[(z.d_e >= lo) & (z.d_e < lo + 10)]
            nS, nA = int((q.cmd == "S").sum()), int(q.agree.sum())
            if nS and nA:
                dd = float(q[q.cmd == "S"].delta.median() - q[q.agree].delta.median())
                mt += min(nS, nA) * dd
                wsum += min(nS, nA)
                per[f"{lo}-{lo + 10}"] = dict(nS=nS, nA=nA, d=dd)
            else:
                per[f"{lo}-{lo + 10}"] = dict(nS=nS, nA=nA, d=np.nan)
        res["M2"][p] = b | dict(verdict=verdict(b, 0.2, 20), matched=(mt / wsum if wsum else np.nan), per_bin=per,
                                opp=dict(n=int(x.opp.sum()), med=float(x[x.opp].delta.median()) if x.opp.any() else np.nan),
                                by_cmd={c: dict(n=int((x.cmd == c).sum()), med=float(x[x.cmd == c].delta.median())) for c in ("L", "S", "R", "-") if (x.cmd == c).any()})
    for p, m in dpops.items():
        x = D[m & np.isfinite(D.delta)]
        cb = pd.cut(x["cov"], [-0.01, 0.0, 0.4999, 0.9999, 1.0], labels=["0", "(0,0.5)", "[0.5,1)", "1"])
        nb = pd.cut(x.n_past, [-1, 0, 5, 15, 99], labels=["0", "1-5", "6-15", ">15"])
        res["M3"][p] = {"cov": {str(c): dict(n=int(len(z)), med=float(z.delta.median()), failed=float(z[~z.completed].delta.median()),
                                             completed=float(z[z.completed].delta.median())) for c, z in x.groupby(cb, observed=False)},
                        "n_past": {str(c): dict(n=int(len(z)), med=float(z.delta.median()), failed=float(z[~z.completed].delta.median()),
                                                completed=float(z[z.completed].delta.median())) for c, z in x.groupby(nb, observed=False)}}
        near = D[m & (D.d_e < 10)]
        res["M4"][p] = {"failed": m4_shares(near[~near.completed].r), "completed": m4_shares(near[near.completed].r),
                        "over_turn": m4_shares(near[near.cls == "over_turn"].r)}
        db = pd.cut(x.d_e, [0, 10, 20, 30.001], right=False, labels=["0-10", "10-20", "20-30"])
        res["dist"][p] = {str(c): {k: dict(n=int(len(z[z.cls == k])), med=float(z[z.cls == k].delta.median())) for k in sorted(x.cls.unique())}
                          for c, z in x.groupby(db, observed=False)}
    res["post"] = post(D, Dx, recs, rmem_by)
    (out / "mode_avg.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_md(out, res, T, D)
    if a.fig:
        fig(T, D, a.fig)


def post(D, Dx, recs, rmem_by):
    """Amendment 2 (after the first read): M4's shares at 10-20 m, M3 by distance bin, the served command's onset distance per turn."""
    import pandas as pd
    res = {"m4_10_20": {}, "m3_by_dist": {}, "cmd_onset": {}, "arc": {}}
    for p in ("pai", "held"):
        res["arc"][p] = {}
        for lo in (0, 10, 20):
            x = D[(D.popn == p) & (D.d_e >= lo) & (D.d_e < lo + 10)]
            res["arc"][p][f"{lo}-{lo + 10}"] = {c: dict(m4=m4_shares(z.r_arc), delta_arc=float(np.nanmedian(z.delta_arc)) if np.isfinite(z.delta_arc).any() else np.nan,
                                                        delta=float(np.nanmedian(z.delta)) if np.isfinite(z.delta).any() else np.nan,
                                                        a4=float(z.a4.median()), v=float(z.v.median()), dpsi=float(z.dpsi.median()), h=float(z.h_dpsi.median()),
                                                        route=float(z.route_arc.median()))
                                                for c, z in [("failed", x[~x.completed]), ("completed", x[x.completed])] +
                                                ([(f"cls {k}", x[x.cls == k]) for k in ("late_wide", "straight", "no_zero")] if p == "pai" else []) if len(z)}
        x = D[(D.popn == p) & (D.d_e >= 10) & (D.d_e < 20)]
        res["m4_10_20"][p] = {"failed": m4_shares(x[~x.completed].r), "completed": m4_shares(x[x.completed].r)}
        y = D[(D.popn == p) & np.isfinite(D.delta)]
        db = pd.cut(y.d_e, [0, 10, 20, 30.001], right=False, labels=["0-10", "10-20", "20-30"])
        cb = pd.cut(y["cov"], [-0.01, 0.0, 0.9999, 1.0], labels=["0", "(0,1)", "1"])
        res["m3_by_dist"][p] = {f"{b}|{c}": dict(n=int(len(z)), med=float(z.delta.median())) for (b, c), z in y.groupby([db, cb], observed=False)}
    for r in recs:
        k = key_of(r)
        if r["src"] not in ("pai", "held") or k.removeprefix("x_") not in rmem_by or not r["row"].get("turn"):
            continue
        m = rmem_by[k].get(r["rid"])
        d = Dx[(Dx.key == k) & (Dx.scene == r["scene"]) & Dx.pre & (Dx.d_e >= 0)].sort_values("t0")
        if m is None or not len(d):
            continue
        j = np.minimum(np.searchsorted(m["t0"], d.t0.to_numpy(np.int64)), len(m["t0"]) - 1)
        c = np.where(m["t0"][j] == d.t0.to_numpy(np.int64), m["cmd"][j], -1)
        de, ag = d.d_e.to_numpy(), c == (0 if r["row"]["side"] == "L" else 2)
        bad = np.flatnonzero(~ag)
        i = bad[-1] + 1 if len(bad) else 0
        res["cmd_onset"][f"{k}|{r['scene']}"] = dict(start_de=float(de.max()), onset_de=float(de[i]) if i < len(de) else np.nan, n_S=int((c == 1).sum()),
                                                     cls=str(d.cls.iloc[0]))
    o = [v for v in res["cmd_onset"].values() if v["start_de"] > 45.0 and np.isfinite(v["onset_de"])]
    res["cmd_onset_summary"] = dict(n_turns_starting_beyond_45m=len(o), onset_median=float(np.median([v["onset_de"] for v in o])) if o else np.nan,
                                    onset_min=float(min(v["onset_de"] for v in o)) if o else np.nan, onset_max=float(max(v["onset_de"] for v in o)) if o else np.nan)
    return res


def f_(v, p=2):
    return "-" if v is None or not np.isfinite(v) else f"{v:+.{p}f}"


def write_md(out, res, T, D):
    g = res["gates"]
    md = ["# MODE AVERAGING, Reading 1 (rf_modeavg.py read)", "",
          "Prereg `plans/2026-10-15-mode-averaging-prereg.md`. delta = 1 - plan 4 s heading change / human 4 s heading change at the same route "
          "station (decisions with human >= 15 deg, winsorised to [-1, 2]); window = approach decisions 0-30 m before the entry; D = per-turn median "
          "delta. Populations: pai = PAI 281 RSG-L18-s0 (strict 36 + 2 over-turn zeros), held = held 460 @18 RSG-L18-s0 / -s1 (24 + 24 strict, "
          "5 + 4 over-turn zeros). CIs: scene-clustered bootstrap (held: log), B 10 000.", "",
          "Gates and coverage: " + "; ".join(f"{p}: {v['turns']} turns, lane matched {v['matched']:.2f}, route direction among the map exits {v['dir_ok']:.2f} "
                                             f"(G-map {'pass' if v['passed'] else 'FAIL'}), D defined {v['deficit_defined']}, window decisions "
                                             f"{v['window_decisions']} ({v['decisions_with_deficit']} with a deficit), command joined {v['cmd_joined']:.2f}"
                                             for p, v in g.items()), "",
          "## M1: deficit by the number of exits (turn level, n_x >= 3 minus n_x = 2)", "",
          "| pop | turns 3+ / 2 | median D 3+ / 2 | difference [CI] | verdict |", "|:--|:--|:--|:--|:--|"]
    for p, b in res["M1"].items():
        md.append(f"| {p} | {b['n1']} / {b['n0']} | {f_(b['m1'])} / {f_(b['m0'])} | {f_(b['d'])} [{f_(b['lo'])}, {f_(b['hi'])}] | {b['verdict']} |")
    md += ["", "Per exit count (4 = 4 or more): turns (failed), median D (failed / completed):", "", "| pop | n_x | turns (failed) | D | D failed | D completed |",
           "|:--|:--|:--|:--|:--|:--|"]
    for p, v in res["nx_table"].items():
        for n, z in v.items():
            md.append(f"| {p} | {n} | {z['turns']} ({z['failed']}) | {f_(z['D_med'])} | {f_(z['D_med_failed'])} | {f_(z['D_med_completed'])} |")
    md += ["", "## M2: deficit by the served command (decision level, S minus the turn's own direction)", "",
           "| pop | decisions S / agree (scenes) | median delta S / agree | difference [CI] | verdict | distance-matched | opposite command (n, delta) |",
           "|:--|:--|:--|:--|:--|:--|:--|"]
    for p, b in res["M2"].items():
        md.append(f"| {p} | {b['n1']} / {b['n0']} ({b['c1']} / {b['c0']}) | {f_(b['m1'])} / {f_(b['m0'])} | {f_(b['d'])} [{f_(b['lo'])}, {f_(b['hi'])}] | "
                  f"{b['verdict']} | {f_(b['matched'])} | {b['opp']['n']}, {f_(b['opp']['med'])} |")
    md += ["", "Per distance bin (S - agree; counts S / agree): " + "; ".join(
        f"{p} " + ", ".join(f"{k} {f_(v['d'])} ({v['nS']} / {v['nA']})" for k, v in b["per_bin"].items()) for p, b in res["M2"].items()),
        "", "## M3 (descriptive): deficit by route-memory coverage of the junction", "",
        "| pop | measure | bin | decisions | median delta | failed | completed |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for p, v in res["M3"].items():
        for m_, bins in v.items():
            for k, z in bins.items():
                md.append(f"| {p} | {m_} | {k} | {z['n']} | {f_(z['med'])} | {f_(z['failed'])} | {f_(z['completed'])} |")
    md += ["", "## M4: the plan's heading near the entry (0-10 m, r = plan / human 4 s heading change)", "",
           "| pop | turns | decisions | straight r < 0.25 | partial | full r >= 0.6 | median r | modal bin | reading |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for p, v in res["M4"].items():
        for c, z in v.items():
            if z.get("n"):
                md.append(f"| {p} | {c} | {z['n']} | {z['straight']:.2f} | {z['partial']:.2f} | {z['full']:.2f} | {f_(z['median'])} | {z['mode_bin']} | {z['reading']} |")
    md += ["", "Deficit by distance to the entry and class (median delta, n):", "", "| pop | d_e | " + " | ".join(sorted(D.cls.unique())) + " |",
           "|:--|:--|" + ":--|" * D.cls.nunique()]
    for p, v in res["dist"].items():
        for b, z in v.items():
            md.append(f"| {p} | {b} | " + " | ".join(f"{f_(z[c]['med'])} ({z[c]['n']})" if c in z else "-" for c in sorted(D.cls.unique())) + " |")
    q = res["post"]
    md += ["", "## Amendment 2 (post-hoc, descriptive)", "",
           "Served command: in the 0-30 m window it names the turn's own direction on every decision (no S, no opposite). Turns whose rollout starts "
           f"more than 45 m before the entry ({q['cmd_onset_summary']['n_turns_starting_beyond_45m']}): the command switches from S to the turn "
           f"direction at {f_(q['cmd_onset_summary']['onset_median'], 1)} m (median; {f_(q['cmd_onset_summary']['onset_min'], 1)} to "
           f"{f_(q['cmd_onset_summary']['onset_max'], 1)}) before the entry and stays there.", "",
           "M4 shares at 10-20 m before the entry (where the late-wide deficit sits):", "",
           "| pop | turns | decisions | straight | partial | full | median r | modal bin | reading |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for p, v in q["m4_10_20"].items():
        for c, z in v.items():
            if z.get("n"):
                md.append(f"| {p} | {c} | {z['n']} | {z['straight']:.2f} | {z['partial']:.2f} | {z['full']:.2f} | {f_(z['median'])} | {z['mode_bin']} | {z['reading']} |")
    md += ["", "M3 by distance (median delta (n); memory coverage of the junction 0 / (0, 1) / 1): " + "; ".join(
        f"{p} {b}: " + ", ".join(f"{f_(v[f'{b}|{c}']['med'])} ({v[f'{b}|{c}']['n']})" for c in ("0", "(0,1)", "1"))
        for p, v in q["m3_by_dist"].items() for b in ("0-10", "10-20", "20-30"))]
    md += ["", "## Amendment 3 (post-hoc, descriptive): speed against geometry", "",
           "The human's 4 s covers more road than a slower plan's 4 s. r_arc = plan heading change / the route's own heading change over the "
           "plan's 4 s arc a4 (from the ego's station); decisions with that route change >= 15 deg. Medians.", "",
           "| pop | d_e | turns | plan deg | human 4 s deg | route over plan arc deg | plan arc a4 m | ego v m/s | delta (human) | delta_arc | "
           "straight / partial / full (r_arc) | median r_arc | reading |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for p, v in q["arc"].items():
        for b, w in v.items():
            for c, z in w.items():
                m = z["m4"]
                md.append(f"| {p} | {b} | {c} | {f_(z['dpsi'], 1)} | {f_(z['h'], 1)} | {f_(z['route'], 1)} | {f_(z['a4'], 1)} | {f_(z['v'], 1)} | "
                          f"{f_(z['delta'])} | {f_(z['delta_arc'])} | " + (f"{m['straight']:.2f} / {m['partial']:.2f} / {m['full']:.2f} | {f_(m['median'])} | "
                                                                          f"{m['reading']} |" if m.get("n") else "- | - | - |"))
    md += ["", "Per turn (pai):", "", "| scene | side deg | class | n_x (lane) | exit directions deg | route deg | D | window (deficit) |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for _, r in T[T.popn == "pai"].sort_values(["completed", "cls", "scene"]).iterrows():
        md.append(f"| {r.scene[7:15]} | {r.side} {r.turn_deg:+.0f} | {r.cls} | {r.n_x} ({r.n_x_lane}) | {', '.join(f'{x:+.0f}' for x in json.loads(r.dirs))} | "
                  f"{r.route_dir:+.0f} | {f_(r.D)} | {r.n_win} ({0 if not np.isfinite(r.n_def) else int(r.n_def)}) |")
    (out / "mode_avg.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))


def fig(T, D, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_F, C_C, C_S = "#eb6834", "#2a78d6", "#1baf7a"
    bins = np.arange(-0.5, 1.5001, 0.1)
    P = D[D["popn"] == "pai"]
    fg, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

    def hist(a_, r, col, lab, ls="-"):
        r = np.clip(np.asarray(r, float)[np.isfinite(r)], -0.499, 1.499)
        if len(r):
            a_.hist(r, bins=bins, weights=np.full(len(r), 1 / len(r)), histtype="step", lw=2, ls=ls, color=col, label=f"{lab} ({len(r)})")
    x = P[P.d_e < 10]
    hist(ax[0], x[~x.completed].r, C_F, "failed turns, decisions")
    hist(ax[0], x[x.completed].r, C_C, "completed turns, decisions")
    ax[0].set(xlabel="plan / human heading change over 4 s", ylabel="share of decisions", title="PAI RSG-L18-s0, 0-10 m before entry (M4)")
    x = P[(P.d_e >= 10) & (P.d_e < 20)]
    hist(ax[1], x[x.cls == "late_wide"].r, C_F, "late-wide, plan / human 4 s", ls="--")
    hist(ax[1], x[x.cls == "late_wide"].r_arc, C_F, "late-wide, plan / route over the plan's own arc")
    hist(ax[1], x[x.cls == "straight"].r_arc, C_S, "straight-through, plan / route over the plan's arc")
    ax[1].set(xlabel="heading ratio", ylabel="share of decisions", title="10-20 m before entry: vs human 4 s, vs road")
    x = P[(P.d_e < 20) & np.isfinite(P.r_arc)]
    col = {"completed": C_C, "straight": C_S}
    order = x.groupby("scene").r_arc.median().sort_values().index.tolist()
    cls = x.groupby("scene").cls.first()
    for k, s_ in enumerate(order):
        r = np.clip(x[x.scene == s_].r_arc, -0.499, 1.499)
        ax[2].plot(r, np.full(len(r), k), "o", ms=3, alpha=0.5, color=col.get(cls[s_], C_F))
    for c_, lab in ((C_C, "completed"), (C_S, "straight-through"), (C_F, "other failed")):
        ax[2].plot([], [], "o", color=c_, label=lab)
    ax[2].legend(fontsize=8, frameon=False, loc="lower right")
    ax[2].set(xlabel="plan / route heading over the plan's arc (0-20 m before entry)", ylabel="PAI turn (sorted by median)",
              yticks=[], title="Per turn, each decision")
    for a_ in ax:
        for v in (0.25, 0.6):
            a_.axvline(v, color="#888", lw=0.8, ls=":")
        a_.axvline(1.0, color="#555", lw=1)
        a_.set_xlim(-0.5, 1.5)
        a_.grid(alpha=0.25)
        a_.spines[["top", "right"]].set_visible(False)
    for a_ in ax[:2]:
        a_.legend(fontsize=8, frameon=False, loc="upper left")
    fg.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("thk")
    t.add_argument("--runs", nargs="+", required=True), t.add_argument("--held", required=True), t.add_argument("--out", required=True)
    r = sp.add_parser("read")
    r.add_argument("--dec", required=True), r.add_argument("--pai", required=True), r.add_argument("--held", required=True)
    r.add_argument("--rmem", nargs="+", required=True), r.add_argument("--lanes", nargs="+", required=True)
    r.add_argument("--out", required=True), r.add_argument("--fig", default="")
    a = ap.parse_args()
    {"thk": thk, "read": read}[a.cmd](a)
