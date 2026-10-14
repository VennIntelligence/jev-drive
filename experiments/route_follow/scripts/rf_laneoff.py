#!/usr/bin/env python3
"""LANEOFF reading (plans/2026-10-15-progress-lane-readings-prereg.md, reading B): do RSG-L18's failed turns already sit off the route's lane
before the turn, and does the car ever make the lane changes its route asks for? CPU only (Mac, .venv; DATA_DIR for the Run dir); inputs are
the same sets as rf_progress.py report (rf_tax extract pkl, rf_tax csv, rf_progress rmem pkl, rf_follow / rf_pai_follow scenes.csv + label).

  rf_laneoff.py report --set NAME=<tax pkl>,<tax csv>,<rmem pkl>,<follow csv>,<follow label> ... [--bend NAME=<follow label>]
                       --out DIR --fig PNG [--picks a,b,..]
      -> DIR/laneoff.json, laneoff.md, turns.csv (turn x point), lc.csv (one row per rollout with a lane-change site)

Turns: the stored strict turns (outcome completed vs failed = any other outcome) and, with --bend, the PAI bend turns (45 deg / 60 m).
Points: the first controller sample with route station >= entry - 20 m / >= entry before t_f (undefined: started past it, or not reached).
Measures: lat_route (signed, left +), lat_log (to the log rig path), heading error to the route tangent, speed, lat_mem (to the polyline of the
served memory's valid points in the local frame; only when the ego projects inside it), out-of-lane = |lat_route| > 1.75 m; oncoming =
rf_tax lane class 2 at any 0.5 s step with station in [entry - 20, exit + 10] before t_f.
Claim B (PAI strict, entry - 20 m): D_med = median |lat_route| failed - completed, D_out = out-of-lane share failed - completed (scene
bootstrap). Supported: D_med >= 0.5 and CI lower > 0, or D_out CI lower > 0; unsupported: D_med CI upper < 0.5 and D_out CI upper <= 0.10;
inconclusive otherwise or a group < 3.
Lane changes: LC site = first route station s (ego start .. log end - 40 m) where the route at s + 40 lies >= 2 m to the side of the
tangent at s and the heading changes < 15 deg (strict: < 5 deg); shift = the ego's (and the log's) lateral displacement in that frame from
station s toward the route's side, max over stations [s, s + 60] before t_f; shifted = >= 2 m.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
import rf_progress as RP  # noqa: E402

OUT_LANE, PRE, LC_Y, LC_D = 1.75, 20.0, 2.0, 40.0


def t_fail(e, tax):
    zt = tax.get("zero_type")
    return e.t_fail(zt) if isinstance(zt, str) and zt in ("collision", "offroad", "corridor") else np.inf


def point(e, s, tf):
    t, why = e.cross_t(s, tf)
    if t is None:
        return dict(why=why)
    i = e.i_at(t)
    p, psi = e.P[i], e.yaw[i]
    _, _, ll, _ = e.TX.poly_proj(e.LP, p[None])
    out = dict(why="ok", t=t, lat_route=float(e.lat[i]), lat_log=float(ll[0]), herr=float(np.degrees(RP.wrap(psi - e.route_psi(e.st[i])))),
               v=float(e.v[i]), lat_mem=np.nan)
    X, _ = e.mem_poly(t)
    if len(X) >= 2:
        Sx = e.PF.stations(e.PF.dedup(X))
        if len(Sx) >= 2 and Sx[-1] > 0:
            Xd = e.PF.dedup(X)
            sq, la = e.PF.project(Xd, Sx, p)
            if 0.0 < sq < Sx[-1]:
                out["lat_mem"] = la
    out.update({f"{k}": v for k, v in e.cov(t).items() if k in ("cov_0_20", "cov_20_40")})
    return out


def lanes_extract(a):
    """(box: AlpaSim venv) every scene of the runs -> {scene: lanes near the logged path (rf_pai_follow's lane dict)}."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    jobs = {}
    for run in a.runs:
        for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]:
            jobs.setdefault(r["clipgt_id"], str(Path(run) / "generated-user-config-0.yaml"))
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(lanes_job, sorted(jobs.items())))
    pickle.dump(res, open(a.out, "wb"), protocol=4)
    print(len(res), "scenes ->", a.out)


def lanes_job(job):
    scene, cfg = job
    import logging
    import rf_tax as TX
    logging.disable(logging.WARNING)
    ds = TX.loader(cfg).get_data_source(scene)
    P = np.asarray(ds.rig.trajectory.positions, float)[:, :2]
    out = {}
    for l in ds.map.lanes:
        c = np.asarray(l.center.points, float)
        if len(c) and np.hypot(*(c[:, None, :2] - P[None, ::5]).transpose(2, 0, 1)).min() < 120.0:
            out[str(l.id)] = dict(c=c[:, [0, 1, 3]].astype(np.float32), nx=sorted(map(str, l.next_lanes)), pv=sorted(map(str, l.prev_lanes)),
                                  al=sorted(map(str, l.adj_lanes_left)), ar=sorted(map(str, l.adj_lanes_right)))
    return scene, out


def turn_job(job):
    name, o, tax, rm, frow, kind = job
    PF, TX = RP._imports()
    e = RP.Ego(o, rm, PF, TX)
    tf = t_fail(e, tax)
    ent, ext = float(frow["entry"]), float(frow["exit"])
    rows = []
    for tag, s in (("m20", ent - PRE), ("e0", ent)):
        r = dict(set=name, kind=kind, scene=o["scene"], rid=o["rid"], point=tag, side=frow.get("side"), turn_deg=frow.get("turn_deg"),
                 outcome=frow.get("outcome"), completed=frow.get("outcome") == "completed", entry=ent, start_to_entry=ent - float(e.st[0]))
        r.update(point(e, s, tf))
        rows.append(r)
    stg = np.interp(o["tg"], e.t, e.st)
    m = (stg >= ent - PRE) & (stg <= ext + PF.PAST) & (o["tg"] < tf)
    onc = bool((np.asarray(o["lane"])[m] == 2).any()) if m.any() else False
    for r in rows:
        r["oncoming"] = onc
    return rows


def lc_job(job):
    name, o, tax, rm = job
    PF, TX = RP._imports()
    e = RP.Ego(o, rm, PF, TX)
    tf = t_fail(e, tax)
    s_log_end, _ = PF.project(e.R, e.S, e.LP[-1])
    s_hi = min(s_log_end, e.S[-1]) - LC_D
    out = dict(set=name, scene=o["scene"], rid=o["rid"], score=o["score"], primary=tax.get("primary"))
    res = {}
    for var, dmax in (("lc", 15.0), ("lc5", 5.0)):
        site = None
        for s in np.arange(max(float(e.st[0]), 0.0), s_hi, 1.0):
            psi = e.route_psi(s)
            dpsi = np.degrees(abs(RP.wrap(e.route_psi(s + LC_D) - psi)))
            if dpsi >= dmax:
                continue
            p0, p1 = e.route_xy(np.array([s, s + LC_D]))
            n = np.array([-np.sin(psi), np.cos(psi)])
            y = float((p1 - p0) @ n)
            if abs(y) >= LC_Y:
                site = (float(s), y, float(dpsi), p0, n)
                break
        if site is None:
            res[var] = None
            continue
        s, y, dpsi, p0, n = site
        sg = np.sign(y)
        t0, why = e.cross_t(s, tf)
        r = dict(s=s, y=y, dpsi=dpsi, reached=why == "ok")
        if why == "ok":
            m = (e.st >= s) & (e.st <= s + 60.0) & (e.t < tf) & (e.t >= t0)
            ye = (e.P[m] - p0) @ n
            r["shift"] = float((sg * (ye - ye[0])).max()) if m.any() else np.nan
        else:
            r["shift"] = np.nan
        ls = np.array([PF.project(e.R, e.S, q, s - 30, s + 90)[0] for q in e.LP])
        ml = (ls >= s) & (ls <= s + 60.0)
        yl = (e.LP[ml] - p0) @ n
        r["log_shift"] = float((sg * (yl - yl[0])).max()) if ml.sum() >= 2 else np.nan
        r["shifted"] = bool(r["shift"] >= LC_Y) if np.isfinite(r["shift"]) else False
        res[var] = r
    if res["lc"] is None:
        return None
    for var, r in res.items():
        for k, v in (r or {}).items():
            out[f"{var}_{k}"] = v
        out[f"{var}_site"] = r is not None
    return out


def lc_map_job(job):
    """Amendment 3: lane changes on the map. Every 4 m of route station s from the ego's start to the log's end - 40 m: the lane under the
    route at s (rf_pai_follow lane_at) and all its successor chains to 45 m; a site when the route point at s + 40 m lies >= 2 m from every
    chain and on some lane (centre <= 1 m... lane_at's 2.5 m / 45 deg rule) - the route leaves the lane it is in for another lane, not a
    continuation (turns are continuations). First site per rollout. Shift = ego / log lateral displacement toward the route side in the
    tangent frame at s, max over [s, s + 60] before t_f; in lane after = |lat_route| <= 1.75 m somewhere in [s + 30, s + 60] before t_f;
    stayed = the ego within 1.75 m of the nearest chain at its first crossing of s + 40."""
    name, o, tax, rm, lanes = job
    PF, TX = RP._imports()
    e = RP.Ego(o, rm, PF, TX)
    tf = t_fail(e, tax)
    near = {k: l for k, l in lanes.items() if np.hypot(*(l["c"][:, None, :2].astype(float) - e.R[None, ::4]).transpose(2, 0, 1)).min() < 25.0}
    s_log_end, _ = PF.project(e.R, e.S, e.LP[-1])
    out = dict(set=name, scene=o["scene"], rid=o["rid"], score=o["score"], primary=tax.get("primary"), site=False)
    for s in np.arange(max(float(e.st[0]), 0.0), min(s_log_end, e.S[-1]) - LC_D, 4.0):
        p, psi = e.route_xy(np.array([s]))[0], e.route_psi(s)
        lid = PF.lane_at(near, p, psi)
        if lid is None:
            continue
        q = e.route_xy(np.array([s + LC_D]))[0]
        if PF.lane_at(near, q, e.route_psi(s + LC_D)) is None:
            continue
        ch = PF.chains(near, lid, p, LC_D + 5.0)
        dq = [np.hypot(*(Q - q).T).min() for Q, _, _ in ch if len(Q) >= 2]
        if not dq or min(dq) < LC_Y:
            continue
        Q = min((Q for Q, _, _ in ch if len(Q) >= 2), key=lambda Q: np.hypot(*(Q - q).T).min())
        n = np.array([-np.sin(psi), np.cos(psi)])
        k = int(np.argmin(np.hypot(*(Q - q).T)))
        sg = float(np.sign((q - Q[k]) @ n)) or 1.0
        t0, why = e.cross_t(s, tf)
        out.update(site=True, s=float(s), side="L" if sg > 0 else "R", gap=float(min(dq)), reached=why == "ok", shift=np.nan, in_lane_after=False, stayed=np.nan)
        m = (e.st >= s) & (e.st <= s + 60.0) & (e.t < tf)
        if why == "ok" and m.sum() >= 2:
            ye = (e.P[m] - p) @ n
            out["shift"] = float((sg * (ye - ye[0])).max())
            k2 = m & (e.st >= s + 30.0)
            out["in_lane_after"] = bool((np.abs(e.lat[k2]) <= OUT_LANE).any())
            t40, w40 = e.cross_t(s + LC_D, tf)
            if t40 is not None:
                out["stayed"] = bool(np.hypot(*(Q - e.P[e.i_at(t40)]).T).min() <= OUT_LANE)
        ls = np.array([PF.project(e.R, e.S, x, s - 30, s + 90)[0] for x in e.LP])
        ml = (ls >= s) & (ls <= s + 60.0)
        yl = (e.LP[ml] - p) @ n
        out["log_shift"] = float((sg * (yl - yl[0])).max()) if ml.sum() >= 2 else np.nan
        out["shifted"] = bool(np.isfinite(out["shift"]) and out["shift"] >= LC_Y)
        break
    return out


def med_diff(x, grp, g, n_boot=RP.N_BOOT, seed=0):
    """median(x | grp) - median(x | ~grp), cluster bootstrap over g."""
    x, grp, g = np.asarray(x, float), np.asarray(grp, bool), np.asarray(g)
    ok = np.isfinite(x)
    x, grp, g = x[ok], grp[ok], g[ok]
    if grp.sum() == 0 or (~grp).sum() == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, n1=int(grp.sum()), n0=int((~grp).sum()))
    u, inv = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(inv == j) for j in range(len(u))]
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        r = np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])
        a1, a0 = x[r][grp[r]], x[r][~grp[r]]
        if len(a1) and len(a0):
            b.append(np.median(a1) - np.median(a0))
    return dict(diff=float(np.median(x[grp]) - np.median(x[~grp])), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)),
                n1=int(grp.sum()), n0=int((~grp).sum()), med1=float(np.median(x[grp])), med0=float(np.median(x[~grp])))


def verdict_b(dm, do):
    if min(dm["n1"], dm["n0"]) < 3:
        return "inconclusive (a group < 3)"
    if (dm["diff"] >= 0.5 and dm["lo"] > 0) or do["lo"] > 0:
        return "supported"
    if dm["hi"] < 0.5 and do["hi"] <= 0.10:
        return "unsupported"
    return "inconclusive"


def report(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bend = dict(x.split("=", 1) for x in (a.bend or []))
    with Run("route_follow", "laneoff", config=vars(a)) as run:
        tj, lj, objs = [], [], {}
        for spec in a.set:
            name, rolls, tax, rm, fo = RP.load_set(spec)
            fb = None
            if name in bend:
                import pandas as pd_
                fb = pd_.read_csv(spec.split("=", 1)[1].split(",")[3])
                fb = fb[fb.label == bend[name]].set_index("scene")
            for rid, o in rolls.items():
                if "ctrl" not in o or o["scene"] not in tax.index:
                    continue
                t = tax.loc[o["scene"]].to_dict()
                m = rm.get(rid)
                objs[(name, o["scene"])] = (o, t, m)
                lj.append((name, o, t, m))
                for kind, F in (("strict", fo), ("bend", fb)):
                    if F is not None and o["scene"] in F.index and bool(F.loc[o["scene"]].get("turn")):
                        tj.append((name, o, t, m, F.loc[o["scene"]].to_dict(), kind))
        LN = {}
        for x in a.lanes or []:
            import pickle
            nm, f = x.split("=", 1)
            LN[nm] = pickle.load(open(f, "rb"))
        mj = []
        for (name, sc), (o, t, m) in objs.items():
            ln = LN.get(name) or LN.get(name.split("_")[0])
            if ln is not None and sc in ln:
                mj.append((name, o, t, m, ln[sc]))
        rm_ = par.pmap(lc_map_job, mj, run=run, desc="map lane changes") if mj else None
        M = pd.DataFrame([v for v in (rm_.values if rm_ else []) if v])
        if rm_ is not None and rm_.errors:
            run.info(f"map lc errors: {list(rm_.errors.values())[:2]}")
        M.to_csv(out / "lc_map.csv", index=False)
        rt = par.pmap(turn_job, tj, run=run, desc="turns")
        rl = par.pmap(lc_job, lj, run=run, desc="lane changes")
        if rt.errors or rl.errors:
            run.info(f"errors: {list(rt.errors.values())[:2]} {list(rl.errors.values())[:2]}")
        T = pd.DataFrame([r for v in rt.values if v for r in v])
        L = pd.DataFrame([v for v in rl.values if v])
        T["abs_lat"] = T.lat_route.abs()
        T["out_lane"] = (T.abs_lat > OUT_LANE).astype(float).where(T.why == "ok")
        T["failed"] = ~T.completed.astype(bool)
        T["inward"] = np.where(T.side == "L", 1.0, -1.0) * T.lat_route                  # exploratory: + = toward the turn's inside
        T.to_csv(out / "turns.csv", index=False)
        L.to_csv(out / "lc.csv", index=False)
        res = dict(claim={}, groups={}, defined={}, lc={})
        sel = {"pai_strict": (T.set == "pai") & (T.kind == "strict"), "held_strict": T.set.str.startswith("held") & (T.kind == "strict"),
               "pooled_strict": T.kind == "strict", "pai_bend": (T.set == "pai") & (T.kind == "bend"),
               "pai_strict_no_short": (T.set == "pai") & (T.kind == "strict") & ~T.outcome.astype(str).str.startswith("short"),
               "pooled_strict_no_short": (T.kind == "strict") & ~T.outcome.astype(str).str.startswith("short")}
        for gname, m in sel.items():
            for pt in ("m20", "e0"):
                x = T[m & (T.point == pt)]
                res["defined"][f"{gname}|{pt}"] = {f"{k[0]}|{'failed' if k[1] else 'completed'}": int(v) for k, v in x.groupby(["why", "failed"]).size().items()}
                x = x[x.why == "ok"]
                if not len(x):
                    continue
                g = x.scene.to_numpy()
                r = dict(d_med=med_diff(x.abs_lat, x.failed, g), d_out=RP.diff_boot(x.out_lane, x.failed, g),
                         d_med_log=med_diff(x.lat_log.abs(), x.failed, g), d_herr=med_diff(x.herr.abs(), x.failed, g), d_v=med_diff(x.v, x.failed, g),
                         d_med_mem=med_diff(x.lat_mem.abs(), x.failed, g), mem_defined=int(x.lat_mem.notna().sum()), n=int(len(x)),
                         oncoming_failed=int(x[x.failed].oncoming.sum()), oncoming_completed=int(x[~x.failed].oncoming.sum()),
                         n_failed=int(x.failed.sum()), n_completed=int((~x.failed).sum()),
                         med_cov_0_20=float(x.cov_0_20.median()))
                r["verdict_rule"] = verdict_b(r["d_med"], r["d_out"])
                r["x_inward"] = dict(d_med=med_diff(x.inward, x.failed, g), failed_out_075=int(((x.inward < -0.75) & x.failed).sum()),
                                     completed_out_075=int(((x.inward < -0.75) & ~x.failed).sum()), failed_out_175=int(((x.inward < -1.75) & x.failed).sum()),
                                     completed_out_175=int(((x.inward < -1.75) & ~x.failed).sum()))
                res["groups"][f"{gname}|{pt}"] = r
        p = res["groups"].get("pai_strict|m20")
        res["claim"] = dict(primary=p, verdict=p["verdict_rule"] if p else "inconclusive (no turn with entry - 20 m defined)")
        for gname, m in (("pai", L.set == "pai"), ("held", L.set.str.startswith("held"))):
            for var in ("lc", "lc5"):
                x = L[m & L[f"{var}_site"].fillna(False).astype(bool)]
                rr = x[x[f"{var}_reached"].fillna(False).astype(bool)]
                res["lc"][f"{gname}|{var}"] = dict(sites=int(len(x)), reached=int(len(rr)),
                                                   shifted=int(rr[f"{var}_shifted"].fillna(False).astype(bool).sum()),
                                                   log_shifted=int((rr[f"{var}_log_shift"] >= LC_Y).sum()),
                                                   zero=int((rr.score == 0).sum()), zero_shifted=int(((rr.score == 0) & rr[f"{var}_shifted"].fillna(False).astype(bool)).sum()))
        res["lc_map"] = {}
        if len(M):
            M["validated"] = M.site.astype(bool) & (M.gap <= 6.0) & (M.log_shift >= LC_Y) & (M.log_shift <= 6.0)
            for (gname, m), (vname, vm) in [(g, v) for g in (("pai", M.set == "pai"), ("held", M.set.str.startswith("held")))
                                             for v in (("map site", M.site.astype(bool)), ("validated", M.validated))]:
                y = M[m & vm]
                rr = y[y.reached.astype(bool)]
                res["lc_map"][f"{gname}|{vname}"] = dict(rollouts=int(m.sum()), sites=int(len(y)), scenes=int(y.scene.nunique()), reached=int(len(rr)),
                                            shifted=int(rr.shifted.sum()), in_lane_after=int(rr.in_lane_after.sum()),
                                            stayed=int((rr.stayed == True).sum()), log_shifted=int((rr.log_shift >= LC_Y).sum()),  # noqa: E712
                                            zero=int((rr.score == 0).sum()), zero_shifted=int(((rr.score == 0) & rr.shifted).sum()))
        res["lc_rollouts"] = {"pai": int(sum(1 for k in objs if k[0] == "pai")), "held": int(sum(1 for k in objs if k[0].startswith("held")))}
        (out / "laneoff.json").write_text(json.dumps(res, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
        write_md(out, res, T, L, M)
        if a.fig:
            fig_bev(a, objs, T)
        run.summary.update(verdict=res["claim"]["verdict"])
        print((out / "laneoff.md").read_text())


def write_md(out, res, T, L, M=None):
    fm = lambda r, f=".2f": f"{r['diff']:+{f}} [{r['lo']:+{f}}, {r['hi']:+{f}}]" if np.isfinite(r.get("diff", np.nan)) else "-"  # noqa: E731
    c = res["claim"]
    md = ["# LANEOFF reading B: lateral offset before the turn, and required lane changes (rf_laneoff.py)", "",
          "Prereg `plans/2026-10-15-progress-lane-readings-prereg.md`. Strict turns of RSG-L18 (PAI 281 s0: 36; held 460 s0 / s1: 24 each), completed vs failed "
          "(any other stored outcome). Offsets in m (left +); differences failed - completed with scene-clustered bootstrap 95 % CI (B = 10 000).", ""]
    if c["primary"]:
        p = c["primary"]
        md += [f"**Claim B (PAI strict, entry - 20 m): median |lat_route| failed {p['d_med']['med1']:.2f} vs completed {p['d_med']['med0']:.2f} m, "
               f"D_med {fm(p['d_med'])}; out-of-lane share D_out {fm(p['d_out'])} (n failed {p['n_failed']}, completed {p['n_completed']}) -> {c['verdict']}**", ""]
    else:
        md += [f"**Claim B: {c['verdict']}**", ""]
    md += ["| group / point | failed / completed | median abs lat_route failed / completed | D_med | D_out (share) | D abs lat_log | D abs heading err (deg) | D speed (m/s) | D abs lat_mem (n defined) | oncoming lane failed / completed | rule |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, r in res["groups"].items():
        md.append(f"| {k} | {r['n_failed']} / {r['n_completed']} | {r['d_med']['med1']:.2f} / {r['d_med']['med0']:.2f} | {fm(r['d_med'])} | {fm(r['d_out'])} | {fm(r['d_med_log'])} | "
                  f"{fm(r['d_herr'], '.1f')} | {fm(r['d_v'])} | {fm(r['d_med_mem'])} ({r['mem_defined']}) | {r['oncoming_failed']} / {r['oncoming_completed']} | {r['verdict_rule']} |")
    md += ["", "Exploratory (not registered): the offset signed toward the turn's inside (+ = inside, - = the side away from the turn):", "",
           "| group / point | median inward failed / completed | D (failed - completed) | >= 0.75 m outside failed / completed | >= 1.75 m outside failed / completed |",
           "|:--|:--|:--|:--|:--|"]
    for k, r in res["groups"].items():
        q = r["x_inward"]
        md.append(f"| {k} | {q['d_med']['med1']:+.2f} / {q['d_med']['med0']:+.2f} | {fm(q['d_med'])} | {q['failed_out_075']} / {q['completed_out_075']} | "
                  f"{q['failed_out_175']} / {q['completed_out_175']} |")
    md += ["", "Where the points are undefined (counts by reason x group):", ""]
    for k, v in res["defined"].items():
        md.append(f"- {k}: " + ", ".join(f"{a} {b}" for a, b in sorted(v.items())))
    x = T[(T.set == "pai") & (T.kind == "strict")].pivot_table(index=["scene", "outcome"], columns="point", values=["lat_route", "lat_log", "herr", "v", "lat_mem"], aggfunc="first")
    md += ["", "PAI strict turns (lat_route / lat_log / heading error / speed / lat_mem at entry - 20 m | at entry; blank = undefined):", "",
           "| scene | outcome | entry - 20 m | entry |", "|:--|:--|:--|:--|"]
    for (sc, oc), r in x.iterrows():
        cell = lambda pt: " / ".join("" if not np.isfinite(r.get((m, pt), np.nan)) else f"{r[(m, pt)]:+.1f}" for m in ("lat_route", "lat_log", "herr", "v", "lat_mem"))  # noqa: E731
        md.append(f"| {sc[7:15]} | {oc} | {cell('m20')} | {cell('e0')} |")
    md += ["", "Lane changes required by the route (proxy: route at s + 40 m >= 2 m to the side of the tangent at s, heading change < 15 deg; strict < 5 deg); "
           "shifted = the ego moved >= 2 m toward the route's side within [s, s + 60] before its zero event; log = the same on the human's path:", "",
           "| set | proxy | rollouts with a site | ego reached the site | ego shifted >= 2 m | log shifted >= 2 m | zeros among reached (shifted) |", "|:--|:--|--:|--:|--:|--:|--:|"]
    for k, r in res["lc"].items():
        g, v = k.split("|")
        md.append(f"| {g} | {'< 15 deg' if v == 'lc' else '< 5 deg'} | {r['sites']} | {r['reached']} | {r['shifted']} ({100 * r['shifted'] / max(r['reached'], 1):.0f} %) | "
                  f"{r['log_shifted']} ({100 * r['log_shifted'] / max(r['reached'], 1):.0f} %) | {r['zero']} ({r['zero_shifted']}) |")
    if res.get("lc_map"):
        md += ["", "Amendment 3, map lane changes (the route point 40 m ahead lies >= 2 m from every successor chain of the lane the route is in, and on "
               "another lane; first site per rollout; validated = the other lane is adjacent (gap <= 6 m) and the human's log itself moved 2-6 m to that "
               "side). Shifted = the ego moved >= 2 m toward that side within [s, s + 60] before its zero event; in lane = "
               "|lat_route| <= 1.75 m somewhere in [s + 30, s + 60]; stayed = still within 1.75 m of its own lane's chain at s + 40; log = the human's shift (gate):", "",
               "| set | rollouts scanned | rollouts with a site (scenes) | ego reached the site | ego shifted >= 2 m | ego in the route's lane after | ego stayed in its lane | log shifted >= 2 m | zeros among reached (of them shifted) |",
               "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for k, r in res["lc_map"].items():
            pct = lambda v: f"{v} ({100 * v / max(r['reached'], 1):.0f} %)"  # noqa: E731
            md.append(f"| {k} | {r['rollouts']} | {r['sites']} ({r['scenes']}) | {r['reached']} | {pct(r['shifted'])} | {pct(r['in_lane_after'])} | {pct(r['stayed'])} | "
                      f"{pct(r['log_shifted'])} | {r['zero']} ({r['zero_shifted']}) |")
        md += ["", "Per validated map site:", "", "| set | scene | score | primary | side | gap (m) | reached | ego shift | in lane after | stayed | log shift |",
               "|:--|:--|--:|:--|:--|--:|:--|--:|:--|:--|--:|"]
        for _, r in M[M.validated].sort_values(["set", "scene"]).iterrows():
            md.append(f"| {r.set} | {r.scene[7:15] if r.scene.startswith('clipgt') else r.scene[-16:]} | {r.score:.2f} | {r.primary} | {r.side} | {r.gap:.1f} | {r.reached} | "
                      f"{r['shift']:+.1f} | {r.in_lane_after} | {r.stayed} | {r.log_shift:+.1f} |")
    md += ["", f"Rollouts scanned: PAI {res['lc_rollouts']['pai']}, held {res['lc_rollouts']['held']} (both seeds).", "",
           "Per site (PAI, strict proxy < 5 deg first, then < 15 deg):", "", "| scene | score | primary | y at +40 m | dpsi | reached | ego shift (m) | log shift (m) |", "|:--|--:|:--|--:|--:|:--|--:|--:|"]
    for var in ("lc5", "lc"):
        x = L[(L.set == "pai") & L[f"{var}_site"].fillna(False).astype(bool)]
        for _, r in x.sort_values("scene").iterrows():
            md.append(f"| {r.scene[7:15]} ({var}) | {r.score:.2f} | {r.primary} | {r[f'{var}_y']:+.1f} | {r[f'{var}_dpsi']:.0f} | {r[f'{var}_reached']} | "
                      f"{r[f'{var}_shift']:+.1f} | {r[f'{var}_log_shift']:+.1f} |")
    (out / "laneoff.md").write_text("\n".join(md) + "\n")


def fig_bev(a, objs, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PF, TX = RP._imports()
    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5))
    for ax, p in zip(axs.flat, a.picks.split(",")):
        key = next(k for k in objs if k[0] == "pai" and p in k[1])
        o, tax, rm = objs[key]
        e = RP.Ego(o, rm, PF, TX)
        tf = t_fail(e, tax)
        tt = T[(T.set == "pai") & (T.kind == "strict") & (T.scene == o["scene"])]
        for poly in o.get("fig", {}).get("road", []):
            ax.fill(poly[:, 0], poly[:, 1], color="#e6e6e6", lw=0)
        for poly in o.get("fig", {}).get("holes", []):
            ax.fill(poly[:, 0], poly[:, 1], color="white", lw=0)
        for ln in o.get("fig", {}).get("lanes", []):
            ax.plot(ln[:, 0], ln[:, 1], color="#c8c8c8", lw=0.5)
        ax.plot(e.R[:, 0], e.R[:, 1], "--", color="#4a7fc1", lw=1.2, label="route")
        ax.plot(e.LP[:, 0], e.LP[:, 1], color="#3a9a5b", lw=1.6, label="log path")
        k = e.t <= min(tf, e.t[-1]) + 1e6
        ax.plot(e.P[k, 0], e.P[k, 1], color="#d9622b", lw=2.0, label="driven path")
        title = f"{p} score {o['score']:.2f} {tax.get('primary') or ''}"
        if len(tt):
            r0 = tt[tt.point == "m20"].iloc[0]
            r1 = tt[tt.point == "e0"].iloc[0]
            tmem = r0.t if r0.why == "ok" else (r1.t if r1.why == "ok" else e.t[0])
            for r, mk in ((r0, "^"), (r1, "s")):
                if r.why == "ok":
                    ax.plot(*e.P[e.i_at(r.t)], marker=mk, ms=10, color="black", zorder=6)
            f = lambda r: f"{r.lat_route:+.1f} m / {r.herr:+.0f} deg" if r.why == "ok" else r.why  # noqa: E731
            title += f"\nstrict turn {r1.outcome}: entry-20 {f(r0)}; entry {f(r1)}"
        else:
            tmem = max(min(tf, e.t[-1]) - 3e6, e.t[0])
            ax.plot(*e.P[e.i_at(tmem)], marker="^", ms=10, color="black", zorder=6)
            title += "\nnot a strict turn: memory at t_f - 3 s"
        X, _ = e.mem_poly(tmem)
        gc = e.geo_cov(tmem)
        offs = gc["s_e"] + RP.OFFS
        allp = e.route_xy(offs)
        ok = (offs >= gc.get("g_lo", np.inf)) & (offs <= gc.get("g_hi", -np.inf))
        ax.scatter(allp[~ok, 0], allp[~ok, 1], s=22, facecolors="none", edgecolors="#b03a3a", lw=0.9, label="route -6..72 m not in memory", zorder=4)
        if len(X):
            ax.scatter(X[:, 0], X[:, 1], s=16, color="#1f4e9c", label="memory valid", zorder=5)
        if np.isfinite(tf):
            ax.plot(*e.P[e.i_at(tf)], marker="X", ms=11, color="#b03a3a", zorder=6)
        c = np.r_[e.P[: e.i_at(min(tf, e.t[-1])) + 1], X] if len(X) else e.P[: e.i_at(min(tf, e.t[-1])) + 1]
        lo, hi = c.min(0) - 12, c.max(0) + 12
        m = max(hi - lo) / 2
        ctr = (lo + hi) / 2
        ax.set_xlim(ctr[0] - m, ctr[0] + m)
        ax.set_ylim(ctr[1] - m, ctr[1] + m)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    axs.flat[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle("Reading B: RSG-L18-s0 PAI 281. Triangle = ego at entry - 20 m (or t_f - 3 s), square = at the entry, X = first zero event;\n"
                 "blue dots / red rings = route memory held / missing at the first marker; titles: lateral offset to the route (left +) / heading error",
                 fontsize=10)
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("lanes")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("report")
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--lanes", nargs="*", help="NAME=<lanes pkl> ({scene: lanes}; NAME's prefix before _ also matches, e.g. held)")
    p.add_argument("--bend", nargs="*", help="NAME=<follow label> of the bend turns (PAI: rsg@bend)")
    p.add_argument("--out", required=True)
    p.add_argument("--fig", default="")
    p.add_argument("--picks", default="9f393c8a,2466266e,1a69f757,132ae6cb,403d558a,e5e47fd2")
    a = ap.parse_args()
    {"lanes": lanes_extract, "report": report}[a.cmd](a)
